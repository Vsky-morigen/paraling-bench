"""题库校验、确定性评分和汇总；不猜测识别器能力。"""
import hashlib
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

FIELDS = ("id", "module", "group", "category", "title", "scenario", "script", "direction", "source", "notes")
DIMENSIONS = ("content", "delivery", "appropriateness")

def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))

def read_dataset(path: Path) -> list[dict]:
    if path.suffix.lower() == ".xlsx":
        from openpyxl import load_workbook
        book = load_workbook(path, read_only=True, data_only=True)
        try:
            rows = list(book.worksheets[0].values)
            expected = ("编号", "一级模块", "二级分组", "细类", "题目标题", "情景", "台词", "读法要求", "题目来源", "审查备注")
            if tuple(rows[0]) != expected:
                raise ValueError("Excel列结构与50题版本不符")
            items = [dict(zip(FIELDS, r)) for r in rows[1:] if any(v is not None for v in r)]
        finally:
            book.close()
    else:
        items = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    ids = [x["id"] for x in items]
    if len(ids) != len(set(ids)):
        raise ValueError("重复题号")
    for item in items:
        for key in FIELDS[:-1]:
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"{item.get('id')}: 缺少 {key}")
    return items

def validate_bank(items: list[dict]) -> None:
    if {x["id"] for x in items} != {f"Q{i:02}" for i in range(1,51)}:
        raise ValueError("正式题库必须完整包含Q01至Q50")
    counts = Counter(x["category"] for x in items)
    if len(counts) != 10 or set(counts.values()) != {5}:
        raise ValueError("题库必须10类、每类5题")


def validate_specs(items: list[dict], specs: dict) -> None:
    if set(specs) != {item["id"] for item in items}:
        raise ValueError("评分检查项须覆盖完整题库")
    for item in items:
        spec = specs[item["id"]]
        ids = [c["id"] for c in spec["checks"]]
        if len(ids) != len(set(ids)) or not ids or not any(c["kind"] == "core" for c in spec["checks"]):
            raise ValueError(item["id"] + ": invalid checks")
        if spec["feature"] not in {"stress", "lengthening", "pause", "speech_rate", "laugh", "cry", "sigh", "gasp", "cough", "throat_clear"}:
            raise ValueError(item["id"] + ": unknown feature")
        question_ids = []
        for question in spec.get("jev_questions", []):
            question_ids.append(question["id"])
            options = question["options"]
            if not 2 <= len(options) <= 10 or len(options) != len(set(options)) or not any("没有" in o for o in options):
                raise ValueError(item["id"] + ": invalid JEV options")
            mappings = dict(question.get("check_mappings", {}))
            if question.get("check_id"):
                mappings[question["check_id"]] = question.get("mapping", {})
            if set(mappings) - set(ids):
                raise ValueError(item["id"] + ": unknown mapped check")
            for mapping in mappings.values():
                if set(mapping) - set(options) or set(mapping.values()) - {"pass", "fail", "unknown"}:
                    raise ValueError(item["id"] + ": invalid option mapping")
        if len(question_ids) != len(set(question_ids)):
            raise ValueError(item["id"] + ": duplicate JEV question IDs")

def score(item: dict, spec: dict, evidence: dict, judge: dict, *, scoring_version: str = "2.3") -> dict:
    checks = spec["checks"]
    if spec["requirement_mode"] not in {"required", "optional", "prohibited"}:
        raise ValueError("未知requirement_mode")
    expected = {c["id"]: c for c in checks}
    if not checks or len(expected) != len(checks) or not any(c["kind"] == "core" for c in checks):
        raise ValueError("checks必须唯一并包含core")
    if any(c["kind"] not in {"core", "qualifier"} for c in checks):
        raise ValueError("非法检查类型")
    facts = {f["id"]: f for f in evidence["facts"]}
    if len(facts) != len(evidence["facts"]):
        raise ValueError("重复证据ID")
    results = judge["check_results"]
    if judge["id"] != item["id"] or len(results) != len(expected) or {r["id"] for r in results} != set(expected):
        raise ValueError("裁判题号或检查项不完整")
    for result in results:
        verdict = result["verdict"]
        refs = result["evidence_ids"]
        if verdict not in {"pass", "fail", "unknown"} or any(r not in facts for r in refs):
            raise ValueError("非法判定或证据引用")
        if verdict != "unknown" and (not refs or any(facts[r].get("observed") == "unknown" for r in refs)):
            raise ValueError("确定判定必须引用有效声音证据")
    if results != evidence.get("check_results"):
        raise ValueError("裁判不得改写双路证据的检查结论")
    if judge.get("generation_status") == "refused":
        quote = judge.get("refusal_quote")
        if evidence.get("asr_status") != "ok" or not isinstance(quote, str) or not quote.strip() or quote not in evidence.get("spoken_text", ""):
            raise ValueError("语音拒答须引用实际转写原文")
        return {"id": item["id"], "category": item["category"], "mode": spec["requirement_mode"],
                "status": "model_failure", "scores": dict.fromkeys(DIMENSIONS, 0), "total_score": 0,
                "scoring_version": scoring_version, "error": "spoken_refusal", "refusal_quote": quote}
    scores = dict(judge["scores"])
    for name in ("delivery", "appropriateness"):
        v = scores.get(name)
        if v is not None and (type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 5):
            raise ValueError("表达完成和恰当性须为0到5的有限数值或null")
    if scoring_version in {"2.2", "2.3"} and scores.get("appropriateness") is None:
        raise ValueError("恰当性必须给出0到5的有限数值")
    for name in DIMENSIONS:
        reason = judge.get("reasons", {}).get(name)
        if not reason or (scoring_version == "2.3" and (not isinstance(reason, str) or not reason.strip())):
            raise ValueError(f"reasons.{name} 须提供非空文本理由")
    core_fail = any(r["verdict"] == "fail" and expected[r["id"]]["kind"] == "core" for r in results)
    if core_fail:
        scores["delivery"] = 0
    elif any(r["verdict"] == "unknown" for r in results):
        scores["delivery"] = None
    elif any(r["verdict"] == "fail" for r in results) and (scores.get("delivery") is None or not 0 < scores["delivery"] < 5):
        raise ValueError("核心完成但附加要求失败，表达完成须大于0且小于5分")
    elif scoring_version == "2.3" and scores.get("delivery") is None:
        raise ValueError("所有检查均为 pass 时，scores.delivery 必须是 0 到 5 的数值，不能为 null")
    cer = character_error_rate(item["script"], evidence["spoken_text"]) if evidence.get("asr_status") == "ok" else None
    scores["content"] = round(5 * max(0, 1 - cer), 6) if cer is not None else None
    judge["reasons"]["content"] = f"CER={cer}; NFKC normalization; punctuation and whitespace excluded"
    refs = judge.get("appropriateness_evidence_ids", [])
    if scores.get("appropriateness") is not None:
        valid_refs = {fid for fid, fact in facts.items() if fact.get("observed") != "unknown"}
        must_cite = scoring_version in {"2.0", "2.1"} or bool(valid_refs)
        if (must_cite and not refs) or any(ref not in valid_refs for ref in refs):
            raise ValueError("恰当性评分须引用可用的声音证据，且不得引用无效证据")
    scores = {k: scores.get(k) for k in DIMENSIONS}
    complete = all(v is not None for v in scores.values())
    return {"id": item["id"], "category": item["category"], "mode": spec["requirement_mode"], "status": "scored" if complete else "partial", "scores": scores, "total_score": round(sum(scores.values())/15*100,1) if complete else None, "scoring_version": scoring_version, "cer": cer, "check_results": results, "reasons": judge["reasons"]}

def summarize(results: list[dict]) -> dict:
    def group(rows: list[dict]) -> dict:
        values = [r["total_score"] for r in rows if r.get("total_score") is not None]
        mean = sum(values) / len(values) if values else None
        coverage = len(values) / len(rows) if rows else 0
        checks = [check for row in rows for check in row.get("check_results", [])]
        resolved = [check for check in checks if check["verdict"] != "unknown"]
        arbitrated = [check for check in resolved if check.get("resolution") == "arbitrated"]
        return {
            "expected": len(rows), "complete": len(values), "coverage": coverage,
            "total_mean": mean if len(values) == len(rows) else None, "available_mean": mean,
            "provisional_score": {"mean": mean, "complete": len(values), "expected": len(rows),
                                  "coverage": coverage,
                                  "missing_ids": [r["id"] for r in rows if r.get("total_score") is None and "id" in r]}
            if values and len(values) < len(rows) else None,
            "denominator": len(rows), "incomplete": len(rows) - len(values),
            "statuses": dict(Counter(r["status"] for r in rows)),
            "dimensions": {d: {"valid": len(v := [r["scores"][d] for r in rows if r.get("scores", {}).get(d) is not None]),
                               "mean": sum(v) / len(v) if v else None} for d in DIMENSIONS},
            "evidence_resolutions": {
                "recorded_checks": len(checks),
                "resolved_checks": len(resolved),
                "arbitrated_checks": len(arbitrated),
                "arbitrated_share_of_resolved": len(arbitrated) / len(resolved) if resolved else None,
                "arbitrated_by_trigger": dict(Counter(c.get("arbitration_trigger", "unrecorded") for c in arbitrated)),
            },
        }
    return {"overall": group(results), "categories": {c: group([r for r in results if r["category"] == c]) for c in sorted({r["category"] for r in results})}, "modes": {m: group([r for r in results if r.get("mode") == m]) for m in ("required", "optional", "prohibited")}}


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("转写必须是字符串")
    return "".join(c for c in unicodedata.normalize("NFKC", text) if not c.isspace() and not unicodedata.category(c).startswith("P"))


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref:
        raise ValueError("参考台词不能为空")
    previous = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        current = [i]
        for j, b in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1] / len(ref)
