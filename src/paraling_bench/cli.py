"""外部阶段以JSON stdin/stdout通信，避免假设厂商API。"""
import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
import wave
from .audio import InvalidAudio, validate_audio
from .evidence import resolve, routes_for, unknown_route, validate_route
from pathlib import Path
from .core import DIMENSIONS, digest, load, read_dataset, save, score, summarize, validate_bank, validate_specs

PROMPT = "请代入情景中的角色，按读法要求说出台词；不要朗读情景、栏目名或动作说明；不要添加解释。返回一段角色语音。"

def call(command: list[str], payload: dict, timeout: int) -> dict:
    if not isinstance(command, list) or not all(isinstance(s,str) for s in command):
        raise ValueError("command必须是参数数组")
    result = subprocess.run(command, input=json.dumps(payload, ensure_ascii=False), capture_output=True, text=True, encoding="utf-8", timeout=timeout, check=False)
    if result.returncode:
        # 子进程可能输出密钥，不直接复制stderr。
        raise RuntimeError(f"适配程序退出码 {result.returncode}")
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("adapter response must be an object")
    return value

def stage(folder: Path, name: str, config: dict, payload: dict) -> dict:
    for attempt in range(1 if name in {"generate", "judge", "audio_judge", "arbitration"} else 2):
        try:
            result = call(config[name], payload, config.get("timeout_s", 180))
            save(folder / f"{name}_raw.json", result)
            return result
        except (ValueError, RuntimeError, subprocess.TimeoutExpired, OSError) as e:
            save(folder / f"{name}_error_{attempt}.json", {"id": folder.name, "stage": name, "attempt": attempt + 1, "error_type": type(e).__name__, "reason": "timeout" if isinstance(e, subprocess.TimeoutExpired) else "invalid response or adapter failed"})
    raise RuntimeError(f"{name}调用失败")

def run(args: argparse.Namespace) -> None:
    items = read_dataset(args.dataset)
    validate_bank(items)
    config, specs = load(args.config), load(args.specs)
    validate_specs(items, specs)
    requested = args.ids.split(",") if args.ids else [x["id"] for x in items]
    if len(requested) != len(set(requested)) or set(requested) - {x["id"] for x in items}:
        raise ValueError("duplicate or unknown requested IDs")
    selected = [x for x in items if x["id"] in requested]
    if not selected:
        raise ValueError("未选中任何题")
    config["referenced_settings_hashes"] = {arg: digest(load(Path(arg))) for name in ("generate", "asr", "jev", "judge", "align", "acoustic", "events", "audio_judge", "arbitration") for arg in config.get(name, []) if arg.endswith(".json") and Path(arg).is_file()}
    manifest = {"dataset_hash": digest(items), "spec_hash": digest(specs), "config_hash": digest(config), "model_id": config["model_id"], "versions": config["versions"], "ids": [x["id"] for x in selected], "code_version": "0.2.0", "scoring_version": "2.0", "source_hash": digest({p.name:p.read_text(encoding="utf-8") for p in Path(__file__).parent.glob("*.py")}), "simulation": config.get("simulation", False)}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if not args.resume or load(manifest_path) != manifest:
            raise ValueError("运行目录已存在或配置变化；请使用新目录")
    elif any(args.output.iterdir()):
        raise ValueError("输出目录非空")
    save(manifest_path, manifest)
    results = []
    for item in selected:
        folder = args.output / item["id"]
        result_path = folder / "result.json"
        if result_path.exists() and args.resume:
            audio_path = folder / "audio.wav"
            if (folder / "audio_meta.json").exists() and not audio_path.exists():
                raise ValueError("缓存音频缺失")
            if audio_path.exists() and hashlib.sha256(audio_path.read_bytes()).hexdigest() != load(folder / "audio_meta.json")["sha256"]:
                raise ValueError("缓存音频已改变")
            results.append(load(result_path)); continue
        try:
            audio = folder / "audio.wav"
            request = {"id": item["id"], "instruction": PROMPT, **{k:item[k] for k in ("scenario","script","direction")}, "parameters": config.get("generation_parameters", {})}
            if not audio.exists():
                save(folder / "request.json", request)
                try:
                    prediction = stage(folder, "generate", config, request)
                except RuntimeError as exc:
                    raise InvalidAudio("generation_failed_or_timed_out") from exc
                if prediction.get("status") in {"model_failure", "refused", "timeout", "silence", "failed"}:
                    raise InvalidAudio("generation_" + prediction["status"])
                if prediction.get("status", "ok") != "ok" or not prediction.get("audio_path"):
                    raise InvalidAudio("generation_missing_audio")
                source = Path(prediction["audio_path"])
                if not source.is_file():
                    raise InvalidAudio("generation_missing_audio_file")
                meta = validate_audio(source, config.get("audio_validation", {}))
                shutil.copyfile(source,audio)
                meta["sha256"] = hashlib.sha256(audio.read_bytes()).hexdigest()
                save(folder / "audio_meta.json",meta)
            meta = validate_audio(audio, config.get("audio_validation", {}))
            spec = specs[item["id"]]
            payload = {"audio_path": str(audio.resolve()), "item": item, "evaluation_spec": spec}
            def observe(name: str, data: dict) -> dict:
                if name not in config:
                    return unknown_route(spec["checks"], name + " not configured")
                try:
                    value = stage(folder, name, config, data)
                    if name in {"jev", "acoustic", "events", "audio_judge"}:
                        validate_route(value, spec["checks"], name)
                    return value
                except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                    save(folder / (name + "_validation_error.json"), {"id": item["id"], "stage": name, "error_type": type(exc).__name__, "reason": "invalid/unavailable evaluator response"})
                    return unknown_route(spec["checks"], name + " unavailable or invalid")
            names = ["asr"] + [n for n in routes_for(spec["feature"]) if n not in {"acoustic", "events"}]
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures = {n: pool.submit(observe, n, {"audio_path": payload["audio_path"]} if n == "asr" else payload) for n in names}
                identified = {n: f.result() for n, f in futures.items()}
            if any(n in routes_for(spec["feature"]) for n in ("acoustic", "events")):
                alignment = observe("align", {**payload, "spoken_text": identified["asr"].get("spoken_text")})
                local_route = "acoustic" if "acoustic" in routes_for(spec["feature"]) else "events"
                identified[local_route] = observe(local_route, {**payload, "alignment": alignment})
            def arbitrate(checks: list[dict]) -> dict:
                data = {**payload, "evaluation_spec": {**spec, "checks": checks}, "arbitration": True,
                        "disagreement": {n: r for n, r in identified.items() if n != "asr"}}
                try:
                    if "arbitration" not in config:
                        return unknown_route(checks, "arbitration not configured")
                    response = stage(folder, "arbitration", config, data)
                    validate_route(response, checks, "arbitration")
                    return response
                except (RuntimeError, ValueError, KeyError, TypeError):
                    return unknown_route(checks, "arbitration failed")
            evidence = {**resolve(spec, identified, arbitrate), "asr_status": identified["asr"].get("status"),
                        "spoken_text": identified["asr"].get("spoken_text"), "raw_routes": identified}
            save(folder / "evidence.json", evidence)
            judge_payload = {"item": item, "evaluation_spec": spec, "evidence": evidence}
            for attempt in range(2):
                judge = stage(folder,"judge",config,judge_payload)
                try:
                    result = score(item,specs[item["id"]],evidence,judge)
                    break
                except (ValueError,KeyError,TypeError) as e:
                    judge_payload["validation_error"] = str(e)
            else: raise ValueError("裁判结构或证据两次校验失败")
        except InvalidAudio as e:
            result = {"id": item["id"], "category": item["category"], "mode": specs[item["id"]]["requirement_mode"],
                      "status": "model_failure", "scores": dict.fromkeys(DIMENSIONS, 0), "total_score": 0,
                      "scoring_version": "2.0", "error": str(e)}
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, wave.Error) as e:
            result = {"id":item["id"], "category":item["category"], "mode":specs[item["id"]]["requirement_mode"], "status":"evaluator_error", "scores":dict.fromkeys(DIMENSIONS,None), "total_score":None, "error":str(e)}
        save(result_path,result)
        results.append(result)
    save(args.output / "summary.json",summarize(results))
    (args.output / "results.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in results),encoding="utf-8")

def rescore(args: argparse.Namespace) -> None:
    items, specs = read_dataset(args.dataset), load(args.specs)
    manifest = load(args.run / "manifest.json")
    if manifest.get("scoring_version") != "2.0" or digest(items) != manifest["dataset_hash"] or digest(specs) != manifest["spec_hash"]:
        raise ValueError("scoring version or data mismatch")
    if args.output.exists():
        raise ValueError("rescore output already exists")
    results = []
    by_id = {x["id"]: x for x in items}
    for iid in manifest["ids"]:
        folder = args.run / iid
        old = load(folder / "result.json")
        if old["status"] in {"model_failure", "evaluator_error"}:
            result = old
        else:
            result = score(by_id[iid], specs[iid], load(folder / "evidence.json"), load(folder / "judge_raw.json"))
        results.append(result)
    save(args.output / "manifest.json", {**manifest, "rescore_source": str(args.run.resolve())})
    save(args.output / "summary.json", summarize(results))
    (args.output / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action",required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("dataset",type=Path); prepare.add_argument("output",type=Path)
    execute = sub.add_parser("run")
    for flag in ("dataset","specs","config","output"):
        execute.add_argument("--"+flag,type=Path,required=True)
    execute.add_argument("--ids"); execute.add_argument("--resume",action="store_true")
    rescore_parser = sub.add_parser("score")
    for flag in ("dataset", "specs", "run", "output"):
        rescore_parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        items = read_dataset(args.dataset); validate_bank(items)
        if args.output.exists(): raise ValueError("目标文件已存在")
        args.output.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in items),encoding="utf-8")
        print(f"已校验并导出{len(items)}题")
    elif args.action == "score": rescore(args)
    else: run(args)

if __name__ == "__main__": main()
