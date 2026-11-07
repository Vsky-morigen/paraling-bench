import unittest
from paraling_bench.mfa_alignment import parse_alignment


class MFAAlignmentTest(unittest.TestCase):
    def raw(self, entries):
        return {'tiers': [{'name': 'words', 'entries': entries}]}

    def test_preserves_pause(self):
        chars = parse_alignment(self.raw([[0, .2, '小'], [.2, .7, '<eps>'], [.7, 1.2, '雨']]), '小雨', 1.3)
        self.assertAlmostEqual(chars[1]['start'] - chars[0]['end'], .5)

    def test_rejects_missing_and_oov(self):
        for entries in ([[0, .2, '小']], [[0, .2, '<unk>']], [[0, .8, '小雨']]):
            with self.assertRaises(ValueError):
                parse_alignment(self.raw(entries), '小雨', 1)

    def test_rejects_overlap(self):
        with self.assertRaises(ValueError):
            parse_alignment(self.raw([[0, .6, '小'], [.5, .8, '雨']]), '小雨', 1)
