import copy
import unittest
from paraling_bench.core import score, summarize

class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.item={"id":"Q40","category":"gasp"}
        self.spec={"requirement_mode":"required","checks":[{"id":"c","kind":"core"},{"id":"q","kind":"qualifier"}]}
        self.evidence={"asr_status":"ok","facts":[{"id":"e","observed":"present"}]}
        self.judge={"id":"Q40","check_results":[{"id":c,"verdict":"pass","evidence_ids":["e"]} for c in ["c","q"]],"scores":{"content":2,"delivery":2,"appropriateness":2},"appropriateness_evidence_ids":["e"],"reasons":dict.fromkeys(["content","delivery","appropriateness"],"test")}
    def result(self): return score(self.item,self.spec,self.evidence,self.judge)
    def test_full(self): self.assertEqual(self.result()["total_score"],100)
    def test_unknown_not_zero(self):
        self.judge["check_results"][1].update(verdict="unknown",evidence_ids=[])
        self.assertIsNone(self.result()["total_score"])
    def test_core_failure_overrides_unknown(self):
        self.judge["check_results"][0]["verdict"]="fail"
        self.judge["check_results"][1].update(verdict="unknown",evidence_ids=[])
        self.assertEqual(self.result()["scores"]["delivery"],0)
    def test_qualifier_failure(self):
        self.judge["check_results"][1]["verdict"]="fail"
        self.assertEqual(self.result()["scores"]["delivery"],1)
    def test_fabricated_evidence(self):
        self.judge["check_results"][0]["evidence_ids"]=["fake"]
        with self.assertRaises(ValueError): self.result()
    def test_asr_failure(self):
        self.evidence["asr_status"]="error"
        self.assertIsNone(self.result()["scores"]["content"])
    def test_unknown_fact(self):
        self.evidence["facts"][0]["observed"]="unknown"
        with self.assertRaises(ValueError): self.result()
    def test_coverage(self):
        a=self.result(); b=copy.deepcopy(a);b["total_score"]=None;b["status"]="partial"
        self.assertEqual(summarize([a,b])["overall"]["coverage"],0.5)

if __name__=="__main__":unittest.main()
