"""Independent per-check evidence and deterministic consensus."""
from typing import Callable

VERDICTS = {"pass", "fail", "unknown"}
EVENT_FEATURES = {"laugh", "cry", "sigh", "gasp", "cough", "throat_clear"}


def specs_for_mode(specs: dict, mode: str) -> dict:
    if mode not in {"simple", "complex"}:
        raise ValueError("unknown evaluation mode")
    if mode == "complex":
        return specs
    filtered = {}
    for item_id, spec in specs.items():
        if spec["feature"] not in EVENT_FEATURES:
            filtered[item_id] = spec
            continue
        core = [check for check in spec["checks"] if check["kind"] == "core"]
        ids = {check["id"] for check in core}
        filtered[item_id] = {**spec, "checks": core, "jev_questions": [
            question for question in spec.get("jev_questions", [])
            if question.get("check_id") in ids or ids.intersection(question.get("check_mappings", {}))
        ]}
    return filtered


def routes_for(spec: dict) -> tuple[str, str]:
    if spec.get("audio_judge_first") is True:
        return "audio_judge", "acoustic"
    feature = spec["feature"]
    if feature == "lengthening":
        return "jev", "acoustic"
    if feature in {"stress", "speech_rate", "pause"}:
        return "acoustic", "audio_judge"
    if feature in {"sigh", "cough", "throat_clear"}:
        return "jev", "audio_judge"
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


def resolve(spec: dict, responses: dict, arbitrate: Callable[[list[dict]], dict] | None = None,
            supplement: Callable[[list[dict]], dict] | None = None) -> dict:
    sources = routes_for(spec)
    rows = {s: validate_route(responses.get(s, unknown_route(spec["checks"], "route unavailable")), spec["checks"], s) for s in sources}
    missing = [
        check for check in spec["checks"]
        if sum(rows[source][check["id"]]["verdict"] == "unknown" for source in sources) == 1
    ]
    supplemental = validate_route(supplement(missing), missing, "audio_judge") if (
        missing and supplement and "audio_judge" not in sources
    ) else {}
    facts, checks, conflicts = [], [], []
    for check in spec["checks"]:
        cid = check["id"]
        pair = [(source, rows[source][cid]) for source in sources]
        refs = []
        effective = []
        for source, row in pair:
            fid = source + ":" + cid
            facts.append({"id": fid, "source": source, "check_id": cid,
                          "observed": row["verdict"], "description": row["reason"], "detail": row})
            if row["verdict"] == "unknown" and cid in supplemental:
                replacement = supplemental[cid]
                supplement_id = "audio_judge:" + cid
                facts.append({"id": supplement_id, "source": "audio_judge", "check_id": cid,
                              "observed": replacement["verdict"], "description": replacement["reason"],
                              "detail": replacement, "replaces": fid})
                if replacement["verdict"] != "unknown":
                    fid, row = supplement_id, replacement
            refs.append(fid)
            effective.append(row["verdict"])
        a, b = effective
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
            if row["verdict"] != "unknown":
                result.update(verdict=row["verdict"], evidence_ids=[fid], resolution="arbitrated")
    return {"facts": facts, "check_results": checks, "routes": list(sources)}
