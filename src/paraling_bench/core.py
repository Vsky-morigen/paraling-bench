"""题库校验、确定性评分和汇总；不猜测识别器能力。"""
import hashlib
import json
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

def score(item: dict, spec: dict, evidence: dict, judge: dict) -> dict:
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
    core_fail = any(r["verdict"] == "fail" and expected[r["id"]]["kind"] == "core" for r in results)
    delivery = 0 if core_fail else None if any(r["verdict"] == "unknown" for r in results) else 1 if any(r["verdict"] == "fail" for r in results) else 2
    scores = judge["scores"]
    for name in DIMENSIONS:
        v = scores.get(name)
        if v is not None and (type(v) is not int or v not in (0,1,2)):
            raise ValueError("分数必须是0/1/2/null")
        if not judge.get("reasons", {}).get(name):
            raise ValueError("每个维度须有判分或无法判断理由")
    scores = {k: scores.get(k) for k in DIMENSIONS}
    scores["delivery"] = delivery
    if evidence.get("asr_status") != "ok":
        scores["content"] = None
    if scores["appropriateness"] is not None and not judge.get("appropriateness_evidence_ids"):
        raise ValueError("恰当性评分须引用声音证据")
    for ref in judge.get("appropriateness_evidence_ids", []):
        if ref not in facts or facts[ref].get("observed") == "unknown":
            raise ValueError("恰当性引用无效")
    complete = all(v is not None for v in scores.values())
    return {"id": item["id"], "category": item["category"], "mode": spec["requirement_mode"], "status": "scored" if complete else "partial", "scores": scores, "total_score": round(sum(scores.values())/6*100,1) if complete else None, "check_results": results, "reasons": judge["reasons"]}

def summarize(results: list[dict]) -> dict:
    def group(rows: list[dict]) -> dict:
        values = [r["total_score"] for r in rows if r.get("total_score") is not None]
        return {"expected": len(rows), "complete": len(values), "coverage": len(values)/len(rows) if rows else 0, "total_mean": sum(values)/len(values) if values else None, "statuses": dict(Counter(r["status"] for r in rows)), "dimensions": {d: {"valid": len(v := [r["scores"][d] for r in rows if r.get("scores", {}).get(d) is not None]), "mean": 50*sum(v)/len(v) if v else None} for d in DIMENSIONS}}
    return {"overall": group(results), "categories": {c: group([r for r in results if r["category"] == c]) for c in sorted({r["category"] for r in results})}, "modes": {m: group([r for r in results if r.get("mode") == m]) for m in ("required", "optional", "prohibited")}}
