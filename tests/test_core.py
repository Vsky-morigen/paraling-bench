import copy
import unittest
from paraling_bench.core import score, summarize

class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.item={"id":"Q40","category":"gasp","script":"你好"}
        self.spec={"requirement_mode":"required","checks":[{"id":"c","kind":"core"},{"id":"q","kind":"qualifier"}]}
        self.evidence={"asr_status":"ok","spoken_text":"你好","facts":[{"id":"e","observed":"present"}]}
        self.judge={"id":"Q40","check_results":[{"id":c,"verdict":"pass","evidence_ids":["e"]} for c in ["c","q"]],"scores":{"content":None,"delivery":5,"appropriateness":5},"appropriateness_evidence_ids":["e"],"reasons":dict.fromkeys(["content","delivery","appropriateness"],"test")}
    def result(self):
        self.evidence["check_results"] = copy.deepcopy(self.judge["check_results"])
        return score(self.item,self.spec,self.evidence,self.judge)
    def test_full(self): self.assertEqual(self.result()["total_score"],100)
    def test_all_pass_requires_delivery_score_and_reason(self):
        self.judge["scores"]["delivery"] = None
        with self.assertRaisesRegex(ValueError, "scores.delivery"):
            self.result()
        old_result = score(self.item, self.spec, self.evidence, self.judge, scoring_version="2.2")
        self.assertIsNone(old_result["scores"]["delivery"])
        self.judge["scores"]["delivery"] = 5
        self.judge["reasons"]["delivery"] = " "
        with self.assertRaisesRegex(ValueError, "reasons.delivery"):
            self.result()

    def test_unknown_not_zero(self):
        self.judge["check_results"][1].update(verdict="unknown",evidence_ids=[])
        self.assertIsNone(self.result()["total_score"])
    def test_core_failure_overrides_unknown(self):
        self.judge["check_results"][0]["verdict"]="fail"
        self.judge["check_results"][1].update(verdict="unknown",evidence_ids=[])
        self.assertEqual(self.result()["scores"]["delivery"],0)
    def test_qualifier_failure(self):
        self.judge["scores"]["delivery"]=4
        self.judge["check_results"][1]["verdict"]="fail"
        self.assertEqual(self.result()["scores"]["delivery"],4)
    def test_fractional_scores(self):
        self.judge["scores"].update(delivery=4.5, appropriateness=3.75)
        result = self.result()
        self.assertEqual(result["scores"]["delivery"], 4.5)
        self.assertEqual(result["scores"]["appropriateness"], 3.75)
        self.assertEqual(result["total_score"], 88.3)

    def test_fractional_qualifier_failure(self):
        self.judge["check_results"][1]["verdict"] = "fail"
        for value in (0.5, 2.75, 4.9):
            self.judge["scores"]["delivery"] = value
            self.assertEqual(self.result()["scores"]["delivery"], value)
        for value in (0, 5):
            self.judge["scores"]["delivery"] = value
            with self.assertRaises(ValueError): self.result()

    def test_invalid_numeric_scores(self):
        for name in ("delivery", "appropriateness"):
            for value in (True, False, -0.1, 5.1, float("nan"), float("inf"), "4.5"):
                with self.subTest(name=name, value=value):
                    self.judge["scores"].update(delivery=5, appropriateness=5)
                    self.judge["scores"][name] = value
                    with self.assertRaises(ValueError): self.result()

    def test_appropriateness_requires_numeric_score_without_audio_evidence(self):
        self.evidence["facts"] = []
        for check in self.judge["check_results"]:
            check.update(verdict="unknown", evidence_ids=[])
        self.judge["appropriateness_evidence_ids"] = []
        self.judge["scores"]["appropriateness"] = None
        with self.assertRaisesRegex(ValueError, "恰当性必须"):
            self.result()
        self.judge["scores"]["appropriateness"] = 2.5
        self.assertEqual(self.result()["scores"]["appropriateness"], 2.5)

    def test_fabricated_evidence(self):
        self.judge["check_results"][0]["evidence_ids"]=["fake"]
        with self.assertRaises(ValueError): self.result()
    def test_asr_failure(self):
        self.evidence["asr_status"]="error"
        self.assertIsNone(self.result()["scores"]["content"])
    def test_spoken_refusal(self):
        self.evidence["spoken_text"]="抱歉，我不能完成这个请求。"
        self.judge.update(generation_status="refused", refusal_quote="我不能完成这个请求")
        self.assertEqual(self.result()["status"], "model_failure")
        self.assertEqual(self.result()["total_score"], 0)
    def test_fabricated_refusal_rejected(self):
        self.judge.update(generation_status="refused", refusal_quote="我拒绝")
        with self.assertRaises(ValueError): self.result()
    def test_unknown_fact(self):
        self.evidence["facts"][0]["observed"]="unknown"
        with self.assertRaises(ValueError): self.result()
    def test_coverage(self):
        a=self.result(); b=copy.deepcopy(a);b["total_score"]=None;b["status"]="partial"
        self.assertEqual(summarize([a,b])["overall"]["coverage"],0.5)

if __name__=="__main__":unittest.main()
