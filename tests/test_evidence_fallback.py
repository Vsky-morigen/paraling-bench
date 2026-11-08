import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from paraling_bench.adapters import execute
from paraling_bench.core import load
from paraling_bench.evidence import resolve, specs_for_mode


def response(*pairs):
    return {"check_results": [{"id": cid, "verdict": verdict, "reason": "test"} for cid, verdict in pairs]}


class FallbackTests(unittest.TestCase):
    def test_one_missing_is_supplemented_and_conflict_or_two_unknowns_are_arbitrated(self):
        checks = [{"id": cid} for cid in ("same", "missing", "different", "both")]
        spec = {"feature": "laugh", "checks": checks}
        jev = response(("same", "pass"), ("missing", "pass"), ("different", "unknown"), ("both", "unknown"))
        ast = response(("same", "pass"), ("missing", "unknown"), ("different", "pass"), ("both", "unknown"))
        supplement = Mock(return_value=response(("missing", "pass"), ("different", "fail")))
        arbitrate = Mock(return_value=response(("different", "fail"), ("both", "pass")))
        result = resolve(spec, {"jev": jev, "events": ast}, arbitrate, supplement)
        supplement.assert_called_once_with(checks[1:3])
        arbitrate.assert_called_once_with(checks[2:])
        self.assertEqual([r["verdict"] for r in result["check_results"]], ["pass", "pass", "fail", "pass"])
        self.assertEqual(result["check_results"][1]["evidence_ids"], ["jev:missing", "audio_judge:missing"])

        self.assertEqual(result["check_results"][3]["evidence_ids"], ["arbitration:both"])

    def test_single_unknown_with_existing_audio_judge_goes_directly_to_arbitration(self):
        spec = {"feature": "stress", "checks": [{"id": "c"}]}
        for acoustic, audio_judge, decision in (
            ("unknown", "pass", "fail"),
            ("pass", "unknown", "pass"),
            ("unknown", "pass", "unknown"),
        ):
            with self.subTest(acoustic=acoustic, audio_judge=audio_judge, decision=decision):
                supplement = Mock()
                arbitrate = Mock(return_value=response(("c", decision)))
                routes = {"acoustic": response(("c", acoustic)), "audio_judge": response(("c", audio_judge))}
                result = resolve(spec, routes, arbitrate, supplement)
                supplement.assert_not_called()
                arbitrate.assert_called_once_with(spec["checks"])
                check = result["check_results"][0]
                self.assertEqual(check["verdict"], decision)
                self.assertEqual(check["evidence_ids"], [] if decision == "unknown" else ["arbitration:c"])

    def test_event_modes_and_cry_count(self):
        complex_specs = load(Path("data/checks.json"))
        simple_specs = specs_for_mode(complex_specs, "simple")
        self.assertEqual([c["id"] for c in simple_specs["Q21"]["checks"]], ["Q21.core"])
        self.assertEqual(simple_specs["Q16"], complex_specs["Q16"])
        self.assertEqual([c["id"] for c in complex_specs["Q27"]["checks"]], ["Q27.core", "Q27.position"])

    def test_audio_providers_use_same_choices_without_answer_mapping(self):
        spec = load(Path("data/checks.json"))["Q42"]
        question = next(q for q in spec["jev_questions"] if q["id"] == "Q42.position")
        spec = {**spec, "checks": [c for c in spec["checks"] if c["id"] == "Q42.position"]}
        chosen = next(o for o, verdict in question["mapping"].items() if verdict == "pass")
        answer = json.dumps({"answers": [{"id": "Q42.position", "answer": chosen, "reason": "听到咳嗽"}]})
        gemini_raw = {"candidates": [{"content": {"parts": [{"text": answer}]}}]}
        gpt_raw = {"choices": [{"message": {"content": answer}}]}
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "sample.wav"
            wav.write_bytes(b"mock audio")
            request = {"item": {"script": "医生，我这两天嗓子一直发痒，晚上会干咳。"},
                       "evaluation_spec": spec, "audio_path": str(wav)}
            with patch.dict("os.environ", {"TEST_GEMINI_KEY": "test"}), patch(
                "urllib.request.urlopen", return_value=io.BytesIO(json.dumps(gemini_raw).encode())
            ) as gemini, patch("paraling_bench.adapters.post", return_value=gpt_raw) as gpt:
                a = execute("audio_judge", {"api_key_env": "TEST_GEMINI_KEY", "model": "gemini"}, request)
                b = execute("arbitration", {"provider": "openai_compatible", "model": "gpt-audio-1.5",
                                            "base_url": "https://example.invalid/v1"}, request)
            gemini_prompt = json.loads(gemini.call_args.args[0].data)["contents"][0]["parts"][0]["text"]
            gpt_parts = gpt.call_args.args[1]["messages"][0]["content"]
            self.assertEqual(gpt_parts[0]["text"], gemini_prompt)
            self.assertEqual(gpt_parts[1]["type"], "input_audio")
            self.assertNotIn('"mapping"', gemini_prompt)
            self.assertNotIn("Q42.count", gemini_prompt)
            self.assertEqual(a["check_results"], b["check_results"])
            self.assertEqual(a["check_results"][0]["verdict"], "pass")
