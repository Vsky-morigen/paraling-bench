"""Independent per-check evidence and deterministic consensus."""
from typing import Callable

VERDICTS = {"pass", "fail", "unknown"}


def routes_for(feature: str) -> tuple[str, str]:
    if feature in {"stress", "lengthening"}:
        return "jev", "acoustic"
    if feature in {"speech_rate", "pause"}:
        return "acoustic", "audio_judge"
    return "jev", "events"


def validate_route(response: dict, checks: list[dict], source: str) -> dict:
    rows = response.get("check_results", [])
    ids = [r["id"] for r in rows]
    expected = {c["id"] for c in checks}
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError(f"{source}: missing/duplicate/unknown check IDs")
    for row in rows:
        if row.get("verdict") not in VERDICTS or not row.get("reason"):
            raise ValueError(f"{source}: invalid verdict/reason")
    return {r["id"]: r for r in rows}


def unknown_route(checks: list[dict], reason: str) -> dict:
    return {"status": "unknown", "check_results": [
        {"id": c["id"], "verdict": "unknown", "reason": reason} for c in checks]}


def resolve(spec: dict, responses: dict, arbitrate: Callable[[list[dict]], dict] | None = None) -> dict:
    sources = routes_for(spec["feature"])
    rows = {s: validate_route(responses.get(s, unknown_route(spec["checks"], "route unavailable")), spec["checks"], s) for s in sources}
    facts, checks, conflicts = [], [], []
    for check in spec["checks"]:
        cid = check["id"]
        pair = [rows[s][cid] for s in sources]
        refs = []
        for source, row in zip(sources, pair):
            fid = source + ":" + cid
            refs.append(fid)
            facts.append({"id": fid, "source": source, "check_id": cid,
                          "observed": row["verdict"], "description": row["reason"], "detail": row})
        a, b = (r["verdict"] for r in pair)
        verdict = a if a == b and a != "unknown" else "unknown"
        result = {"id": cid, "verdict": verdict, "evidence_ids": refs if verdict != "unknown" else [],
                  "resolution": "consensus" if verdict != "unknown" else "insufficient"}
        checks.append(result)
        if a != "unknown" and b != "unknown" and a != b:
            conflicts.append(check)
            result["resolution"] = "conflict"
    if conflicts and arbitrate:
        arbitration = validate_route(arbitrate(conflicts), conflicts, "arbitration")
        for result in checks:
            if result["id"] not in arbitration:
                continue
            row = arbitration[result["id"]]
            fid = "arbitration:" + result["id"]
            facts.append({"id": fid, "source": "arbitration", "check_id": result["id"],
                          "observed": row["verdict"], "description": row["reason"], "detail": row})
            result.update(verdict=row["verdict"], evidence_ids=[fid] if row["verdict"] != "unknown" else [], resolution="arbitrated")
    return {"facts": facts, "check_results": checks, "routes": list(sources)}
