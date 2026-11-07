"""Local alignment, acoustic measurements and windowed AudioSet detection."""
import math
from pathlib import Path
from .audio import read_wav
from .core import normalize_text


def align(settings: dict, request: dict) -> dict:
    provider = settings.get('provider', 'whisperx')
    if provider == 'mfa':
        from .mfa_alignment import align_mfa
        return align_mfa(settings, request)
    if provider != 'whisperx':
        raise ValueError(f'Unknown alignment provider: {provider}')
    import whisperx
    samples, meta = read_wav(Path(request["audio_path"]))
    import numpy as np
    import librosa
    audio = librosa.resample(np.asarray(samples, dtype=np.float32), orig_sr=meta["sample_rate"], target_sr=16000)
    model, metadata = whisperx.load_align_model(language_code="zh", device=settings.get("device", "cpu"),
                                              model_name=settings.get("model"))
    text = normalize_text(request.get("spoken_text") or "")
    if not text:
        raise ValueError("ASR text unavailable for forced alignment")
    raw = whisperx.align([{"start": 0, "end": meta["duration_s"], "text": text}], model, metadata,
                         audio, settings.get("device", "cpu"), return_char_alignments=True)
    chars = [c for segment in raw["segments"] for c in segment.get("chars", []) if normalize_text(c["char"])]
    return {"status": "ok", "chars": chars, "text": text, "raw": raw, "model": settings.get("model") or metadata.get("type")}


def valid_chars(alignment: dict, duration: float) -> list[dict]:
    chars = alignment.get("chars", [])
    if alignment.get("status") != "ok" or not chars:
        raise ValueError("alignment unavailable")
    previous = 0.0
    for c in chars:
        start, end = c.get("start"), c.get("end")
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (start, end)) or not previous <= start < end <= duration:
            raise ValueError("missing/nonmonotonic character alignment")
        previous = end
    if normalize_text("".join(c["char"] for c in chars)) != normalize_text(alignment["text"]):
        raise ValueError("incomplete character alignment")
    return chars


def span(chars: list[dict], target: str, occurrence: int = 1) -> tuple[int, int]:
    text = "".join(c["char"] for c in chars)
    start = -1
    for _ in range(occurrence):
        start = text.find(target, start + 1)
        if start < 0:
            raise ValueError("target absent from aligned ASR text")
    return start, start + len(target)


def acoustic(settings: dict, request: dict) -> dict:
    import numpy as np
    import librosa
    samples, meta = read_wav(Path(request["audio_path"]))
    chars = valid_chars(request["alignment"], meta["duration_s"])
    sr = meta["sample_rate"]
    y = np.asarray(samples, dtype=np.float32)
    measures = []
    for char in chars:
        clip = y[round(char["start"] * sr):round(char["end"] * sr)]
        f0, voiced, _ = librosa.pyin(clip, sr=sr, fmin=settings.get("fmin_hz", 50), fmax=settings.get("fmax_hz", 600))
        pitches = f0[voiced & np.isfinite(f0)]
        measures.append({**char, "duration_s": char["end"] - char["start"],
                         "rms": float(np.sqrt(np.mean(clip ** 2))),
                         "voiced_fraction": float(np.mean(voiced)),
                         "f0_hz": float(np.median(pitches)) if len(pitches) else None})
    total_span = chars[-1]["end"] - chars[0]["start"]
    metrics = {"chars_per_second": len(chars) / total_span,
               "max_char_duration_s": max(m["duration_s"] for m in measures),
               "articulation_chars_per_second": len(chars) / sum(m["duration_s"] for m in measures)}
    for name, target in request["evaluation_spec"].get("targets", {}).items():
        try:
            a, b = span(chars, target["text"], target.get("occurrence", 1))
        except ValueError:
            continue
        selected = measures[a:b]
        rest = measures[:a] + measures[b:]
        metrics[name + ".duration_s"] = sum(x["duration_s"] for x in selected)
        metrics[name + ".voiced_fraction"] = float(np.mean([x["voiced_fraction"] for x in selected]))
        if rest:
            metrics[name + ".rest_mean_duration_s"] = float(np.mean([x["duration_s"] for x in rest]))
        for field in ("duration_s", "rms", "f0_hz"):
            vals = [x[field] for x in selected if x[field] is not None]
            baseline = [x[field] for x in rest if x[field] is not None]
            if vals and baseline and np.median(baseline) > 0:
                metrics[name + "." + field + "_ratio"] = float(np.mean(vals) / np.median(baseline))
        if a:
            metrics[name + ".pause_before_s"] = chars[a]["start"] - chars[a - 1]["end"]
        if b < len(chars):
            metrics[name + ".pause_after_s"] = chars[b]["start"] - chars[b - 1]["end"]
        if a and b < len(chars):
            before = a / (chars[a - 1]["end"] - chars[0]["start"])
            after = (len(chars) - a) / (chars[-1]["end"] - chars[a]["start"])
            metrics[name + ".after_before_rate_ratio"] = after / before
    rows = []
    rules = settings.get("rules", {}).get(request["item"]["id"], {})
    for check in request["evaluation_spec"]["checks"]:
        rule = rules.get(check["id"], {})
        predicates = rule.get("predicates", [])
        for predicate in predicates:
            bounds = [predicate.get(key) for key in ("min", "max")]
            if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v)) for v in bounds):
                raise ValueError("non-finite or nonnumeric acoustic threshold")
            if all(v is not None for v in bounds) and bounds[0] > bounds[1]:
                raise ValueError("acoustic min exceeds max")
        reference_provider = rule.get("reference_alignment_provider")
        reference_matches = reference_provider is None or (
            request["alignment"].get("provider") == reference_provider
            and "".join(char["char"] for char in chars) == normalize_text(request["item"]["script"])
        )
        values = []
        for predicate in predicates:
            value = metrics.get(predicate["metric"])
            if "reference_duration_s" in predicate:
                reference = predicate["reference_duration_s"]
                if type(reference) not in (int, float) or not math.isfinite(reference) or reference <= 0:
                    raise ValueError("invalid acoustic reference duration")
                value = value / reference if value is not None else None
            values.append(value)
        known = rule.get("calibrated") is True and reference_matches and predicates and all(
            v is not None and (p.get("min") is not None or p.get("max") is not None)
            for p, v in zip(predicates, values)
        )
        passed = known and all(
            (p.get("min") is None or v >= p["min"]) and (p.get("max") is None or v <= p["max"])
            for p, v in zip(predicates, values)
        )
        rows.append({"id": check["id"], "verdict": ("pass" if passed else "fail") if known else "unknown",
                     "reason": "calibrated acoustic predicates" if known else "missing target or uncalibrated acoustic thresholds"})
    return {"status": "ok", "check_results": rows, "metrics": metrics, "chars": measures, "rules": rules}


def events(settings: dict, request: dict) -> dict:
    import numpy as np
    import librosa
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    samples, meta = read_wav(Path(request["audio_path"]))
    audio = librosa.resample(np.asarray(samples, dtype=np.float32), orig_sr=meta["sample_rate"], target_sr=16000)
    kwargs = {"revision": settings["revision"]} if settings.get("revision") else {}
    extractor = ASTFeatureExtractor.from_pretrained(settings["model"], **kwargs)
    model = ASTForAudioClassification.from_pretrained(settings["model"], **kwargs).to(settings.get("device", "cpu")).eval()
    feature = request["evaluation_spec"]["feature"]
    rule = settings.get("features", {}).get(feature, {})
    labels = rule.get("labels", [])
    indices = [int(i) for i, label in model.config.id2label.items() if label in labels]
    if not labels or len(indices) != len(labels):
        raise ValueError("event labels absent from configured model")
    window = rule.get("window_s", settings["window_s"])
    hop = rule.get("hop_s", settings["hop_s"])
    if not 0 < hop <= window <= 10:
        raise ValueError("require 0 < hop_s <= window_s <= 10")
    windows = []
    for start in range(0, len(audio), max(1, round(hop * 16000))):
        clip = audio[start:start + round(window * 16000)]
        inputs = extractor(clip, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(settings.get("device", "cpu")) for k, v in inputs.items()}
        with torch.no_grad():
            probs = model(**inputs).logits.sigmoid()[0].cpu().tolist()
        windows.append({"start": start / 16000, "end": (start + len(clip)) / 16000,
                        "probability": max(probs[i] for i in indices),
                        "label_probs": {model.config.id2label[i]: probs[i] for i in indices}})
        if start + len(clip) == len(audio):
            break
    threshold = rule.get("threshold")
    calibrated = rule.get("calibrated") is True and isinstance(threshold, (float, int)) and 0 <= threshold <= 1
    detected = [w for w in windows if calibrated and w["probability"] >= threshold]
    rows = []
    spec = request["evaluation_spec"]
    try:
        chars = valid_chars(request.get("alignment", {}), meta["duration_s"])
    except ValueError:
        chars = []
    location = event_position(detected, chars, spec.get("event_position")) if calibrated else "unknown"
    bursts = merge_windows(detected)
    metrics = {"event_count": len(bursts), "event_duration_s": sum(b["end"] - b["start"] for b in bursts)}
    if bursts and chars:
        event_samples = [x for b in bursts for x in samples[round(b["start"] * meta["sample_rate"]):round(b["end"] * meta["sample_rate"])]]
        speech_samples = [x for c in chars for x in samples[round(c["start"] * meta["sample_rate"]):round(c["end"] * meta["sample_rate"])]]
        if event_samples and speech_samples:
            event_rms = math.sqrt(sum(x*x for x in event_samples) / len(event_samples))
            speech_rms = math.sqrt(sum(x*x for x in speech_samples) / len(speech_samples))
            if speech_rms:
                metrics["event_speech_rms_ratio"] = event_rms / speech_rms
    for check in spec["checks"]:
        verdict = "unknown"
        if calibrated and check["kind"] == "core":
            present = bool(detected)
            passed = not present if spec["requirement_mode"] == "prohibited" else True if spec["requirement_mode"] == "optional" else present
            verdict = "pass" if passed else "fail"
        qualifier_rule = settings.get("check_rules", {}).get(check["id"], {})
        measurement = check.get("measurement", {})
        if calibrated and check["kind"] == "qualifier" and qualifier_rule.get("calibrated") is True:
            metric = measurement.get("metric")
            if metric == "position":
                verdict = location
            elif metric in metrics and (detected or metric == "event_count"):
                # Counts of window clusters and their durations are estimates, requiring
                # separate calibration from clip-level presence classification.
                low = measurement.get("min", qualifier_rule.get("min"))
                high = measurement.get("max", qualifier_rule.get("max"))
                if low is not None or high is not None:
                    verdict = "pass" if (low is None or metrics[metric] >= low) and (high is None or metrics[metric] <= high) else "fail"
        rows.append({"id": check["id"], "verdict": verdict,
                     "reason": "calibrated AudioSet " + (measurement.get("metric", "presence")) if verdict != "unknown" else "uncalibrated detector/temporal rule or insufficient resolution"})
    return {"status": "ok", "check_results": rows, "windows": windows, "detected": detected,
            "position_verdict": location, "bursts": bursts, "metrics": metrics, "time_resolution_s": window, "model": settings["model"], "revision": settings.get("revision"), "rule": rule}


def merge_windows(windows: list[dict]) -> list[dict]:
    bursts = []
    for window in sorted(windows, key=lambda w: w["start"]):
        if bursts and window["start"] <= bursts[-1]["end"]:
            bursts[-1]["end"] = max(bursts[-1]["end"], window["end"])
        else:
            bursts.append({"start": window["start"], "end": window["end"]})
    return bursts


def event_position(events: list[dict], chars: list[dict], rule: dict | None) -> str:
    """Conservative relation between window support and aligned word boundaries."""
    if not chars or not events or not rule:
        return "unknown"
    relation = rule["relation"]
    if relation == "between_words":
        try:
            _, left_end = span(chars, rule["left"])
            right_start, _ = span(chars, rule["right"])
        except ValueError:
            return "unknown"
        if left_end != right_start:
            return "unknown"
        low, high = chars[left_end - 1]["end"], chars[right_start]["start"]
    elif relation == "before_and_during":
        before = [e for e in events if e["end"] <= chars[0]["start"]]
        during = [e for e in events if e["start"] >= chars[0]["start"] and e["end"] <= chars[-1]["end"]]
        return "pass" if before and during else "unknown"
    elif relation == "before_speech":
        low, high = 0, chars[0]["start"]
    elif relation == "after_speech":
        low, high = chars[-1]["end"], math.inf
    elif relation == "during_speech":
        low, high = chars[0]["start"], chars[-1]["end"]
    elif relation in {"before_word", "after_word"}:
        try:
            a, b = span(chars, rule["text"], rule.get("occurrence", 1))
        except ValueError:
            return "unknown"
        low = chars[a - 1]["end"] if a else 0
        high = chars[a]["start"]
        if relation == "after_word":
            low = chars[b - 1]["end"]
            high = chars[b]["start"] if b < len(chars) else math.inf
    else:
        raise ValueError("unknown event position relation")
    if all(e["start"] >= low and e["end"] <= high for e in events):
        return "pass"
    if all(e["end"] <= low or e["start"] >= high for e in events):
        return "fail"
    return "unknown"
