import copy
import io
import importlib.util
import json
import math
import os
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch
from paraling_bench.audio import read_wav, validate_audio, InvalidAudio
from paraling_bench.core import character_error_rate, summarize, validate_specs, read_dataset, load
from paraling_bench.evidence import resolve, validate_route
from paraling_bench.adapters import execute
from paraling_bench.cli import run, rescore
from paraling_bench.measurements import acoustic, valid_chars, event_position, merge_windows
from argparse import Namespace


def float_wav(values):
    data = struct.pack('<' + 'f' * len(values), *values)
    fmt = struct.pack('<HHIIHH', 3, 1, 16000, 64000, 4, 32)
    chunks = b'fmt ' + struct.pack('<I', 16) + fmt + b'data' + struct.pack('<I', len(data)) + data
    return b'RIFF' + struct.pack('<I', len(chunks) + 4) + b'WAVE' + chunks


class AudioTests(unittest.TestCase):
    def check(self, data):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'a.wav'; p.write_bytes(data)
            return validate_audio(p)

    def test_float32(self):
        self.assertEqual(self.check(float_wav([0.2, -0.2]))['encoding'], 'float')

    def test_silence_and_nan(self):
        for values in ([0., 0.], [float('nan')], [float('inf')]):
            with self.assertRaises(InvalidAudio): self.check(float_wav(values))

    def test_truncation(self):
        with self.assertRaises(InvalidAudio): self.check(float_wav([0.1])[:-1])

    def test_pcm_unsigned_silence(self):
        out = io.BytesIO()
        with wave.open(out, 'wb') as w:
            w.setparams((1, 1, 16000, 0, 'NONE', '')); w.writeframes(b'\x80' * 16)
        with self.assertRaises(InvalidAudio): self.check(out.getvalue())


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.spec = {'feature': 'stress', 'checks': [{'id': 'c', 'kind': 'core'}]}

    def route(self, verdict):
        return {'check_results': [{'id': 'c', 'verdict': verdict, 'reason': 'test'}]}

    def test_agreement(self):
        result = resolve(self.spec, {'jev': self.route('pass'), 'acoustic': self.route('pass')})
        self.assertEqual(result['check_results'][0]['verdict'], 'pass')
        self.assertEqual(len(result['check_results'][0]['evidence_ids']), 2)

    def test_conflict_calls_arbitration(self):
        calls = []
        def arb(checks):
            calls.append(checks); return self.route('fail')
        result = resolve(self.spec, {'jev': self.route('pass'), 'acoustic': self.route('fail')}, arb)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['check_results'][0]['verdict'], 'fail')

    def test_missing_not_consensus(self):
        result = resolve(self.spec, {'jev': self.route('pass')})
        self.assertEqual(result['check_results'][0]['verdict'], 'unknown')

    def test_duplicate_rejected(self):
        r = self.route('pass'); r['check_results'] *= 2
        with self.assertRaises(ValueError): validate_route(r, self.spec['checks'], 'x')


class MetricTests(unittest.TestCase):
    def test_bank_question_mappings(self):
        validate_specs(read_dataset(Path('data/questions.jsonl')), load(Path('data/checks.json')))

    def test_cer(self):
        self.assertEqual(character_error_rate('你好，世界！', '你好世界'), 0)
        self.assertEqual(character_error_rate('你好世界', '你好'), .5)
        self.assertEqual(character_error_rate('你好', ''), 1)
        self.assertEqual(character_error_rate('Ａ１', 'A1'), 0)

    def test_failures_stay_in_denominator(self):
        rows = [{'status': 'scored', 'category': 'x', 'total_score': 100},
                {'status': 'model_failure', 'category': 'x', 'total_score': 0}]
        self.assertEqual(summarize(rows)['overall']['total_mean'], 50)
        rows.append({'status': 'evaluator_error', 'category': 'x', 'total_score': None})
        summary = summarize(rows)['overall']
        self.assertIsNone(summary['total_mean'])
        self.assertEqual(summary['denominator'], 3)

    def test_alignment_and_position(self):
        chars = [{'char': '你', 'start': 1., 'end': 1.2}, {'char': '好', 'start': 1.6, 'end': 2.}]
        self.assertEqual(event_position([{'start': 1.3, 'end': 1.5}], chars, {'relation': 'after_word', 'text': '你'}), 'pass')
        self.assertEqual(event_position([{'start': .1, 'end': .5}], chars, {'relation': 'after_word', 'text': '你'}), 'fail')
        self.assertEqual(event_position([{'start': 1.1, 'end': 1.7}], chars, {'relation': 'after_word', 'text': '你'}), 'unknown')
        with self.assertRaises(ValueError): valid_chars({'status': 'ok', 'text': '你好啊', 'chars': chars}, 3)
        self.assertEqual(merge_windows([{'start': 0, 'end': 1}, {'start': .5, 'end': 1.5}, {'start': 2, 'end': 3}]),
                         [{'start': 0, 'end': 1.5}, {'start': 2, 'end': 3}])

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'optional numpy not installed')
    def test_acoustic_measurements_with_mock_pitch(self):
        import numpy as np
        import types
        fake = types.SimpleNamespace(pyin=lambda clip, **kw: (np.array([100.]), np.array([True]), np.array([1.])))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'a.wav'; p.write_bytes(float_wav([.2] * 16000))
            request = {'audio_path': str(p), 'item': {'id': 'Q01'}, 'alignment': {'status': 'ok', 'text': '你好', 'chars': [
                {'char': '你', 'start': 0., 'end': .2}, {'char': '好', 'start': .4, 'end': .8}]},
                'evaluation_spec': {'targets': {'target1': {'text': '你'}}, 'checks': [{'id': 'c'}]}}
            settings = {'rules': {'Q01': {'c': {'calibrated': True, 'predicates': [{'metric': 'target1.pause_after_s', 'min': .1}]}}}}
            with patch.dict('sys.modules', {'librosa': fake}): result = acoustic(settings, request)
            self.assertEqual(result['metrics']['chars_per_second'], 2.5)
            self.assertAlmostEqual(result['metrics']['target1.duration_s_ratio'], .5)
            self.assertEqual(result['check_results'][0]['verdict'], 'pass')


class ProtocolTests(unittest.TestCase):
    def test_jev_question_language_and_probabilities(self):
        spec = {'feature': 'stress', 'checks': [{'id': 'c'}], 'jev_questions': [
            {'id': 'q', 'check_id': 'c', 'text': '哪个词重读？', 'options': ['先', '没有重读'], 'mapping': {'先': 'pass'}}]}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'a.wav'; p.write_bytes(float_wav([.2]))
            raw = {'answers': {'q': {'answer': '先', 'confidence': .9, 'probs': {'先': .9, '没有重读': .1}}}}
            with patch('paraling_bench.adapters.post', return_value=raw) as send:
                result = execute('jev', {'url': 'https://example.invalid', 'request_interval_s': 0}, {'audio_path': str(p), 'evaluation_spec': spec})
            self.assertEqual(send.call_args.args[1]['questions'][0]['lang'], 'zh')
            self.assertEqual(result['answers'][0]['probs']['先'], .9)

    def test_jev_position_shared_answer_and_count_calibration(self):
        spec = {'feature': 'cough', 'checks': [{'id': 'core'}, {'id': 'pos'}, {'id': 'count'}], 'jev_questions': [
            {'id': 'q', 'check_id': 'core', 'text': '在哪里？', 'options': ['说话前', '没有咳嗽'], 'mapping': {'说话前': 'pass'}, 'check_mappings': {'pos': {'说话前': 'pass'}}},
            {'id': 'n', 'check_id': 'count', 'text': '几次？', 'options': ['一次', '没有咳嗽'], 'mapping': {'一次': 'pass'}, 'calibration_key': 'cough.count'}]}
        raw = {'answers': {'q': {'answer': '说话前'}, 'n': {'answer': '一次'}}}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'a.wav'; p.write_bytes(float_wav([.2]))
            with patch('paraling_bench.adapters.post', return_value=raw):
                result = execute('jev', {'url': 'https://example.invalid', 'request_interval_s': 0}, {'audio_path': str(p), 'evaluation_spec': spec})
        self.assertEqual([r['verdict'] for r in result['check_results']], ['pass', 'pass', 'unknown'])

    def test_gemini_audio_body_and_blind_first_pass(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'a.wav'; p.write_bytes(float_wav([.2]))
            response = {'candidates': [{'content': {'parts': [{'text': json.dumps({'check_results': [{'id': 'c', 'verdict': 'pass', 'reason': 'audible'}]})}]}}]}
            with patch.dict(os.environ, {'TEST_AUDIO_KEY': 'secret'}), patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as send:
                execute('audio_judge', {'model': 'test', 'api_key_env': 'TEST_AUDIO_KEY'}, {'audio_path': str(p), 'item': {}, 'evaluation_spec': {'checks': [{'id': 'c'}]}, 'disagreement': {'secret_vote': 'fail'}})
            body = json.loads(send.call_args.args[0].data)
            self.assertNotIn('secret_vote', json.dumps(body))
            self.assertIn('inlineData', body['contents'][0]['parts'][1])


class PipelineTests(unittest.TestCase):
    def args(self, out):
        return Namespace(dataset=Path('data/questions.jsonl'), specs=Path('data/checks.json'), config=Path('tests/mock_config.json'), output=out, ids='Q01', resume=False)

    def test_generation_timeout_zero_and_rescore_offline(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.args(Path(d) / 'run')
            with patch('paraling_bench.cli.stage', side_effect=RuntimeError('generate failed')):
                run(args)
            result = json.loads((args.output / 'Q01/result.json').read_text(encoding='utf-8'))
            self.assertEqual(result['status'], 'model_failure')
            self.assertEqual(result['total_score'], 0)
            with patch('paraling_bench.cli.stage', side_effect=AssertionError('network forbidden')):
                rescore(Namespace(dataset=args.dataset, specs=args.specs, run=args.output, output=Path(d) / 'rescored'))

    def test_silent_generation_zero(self):
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d) / 'silent.wav'; audio.write_bytes(float_wav([0.]))
            args = self.args(Path(d) / 'run')
            with patch('paraling_bench.cli.stage', return_value={'audio_path': str(audio)}): run(args)
            result = json.loads((args.output / 'Q01/result.json').read_text(encoding='utf-8'))
            self.assertEqual(result['status'], 'model_failure')


if __name__ == '__main__': unittest.main()
