import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from paraling_bench.adapters import execute, transcribe_api

class ApiTests(unittest.TestCase):
    def test_judge_chat_protocol(self):
        response={"choices":[{"message":{"content":json.dumps({"id":"Q01"})}}]}
        with patch("urllib.request.urlopen",return_value=io.BytesIO(json.dumps(response).encode())) as send, patch.dict(os.environ,{"TEST_KEY":"secret"}):
            value=execute("judge",{"provider":"openai_compatible","base_url":"https://example.invalid/v1/","model":"judge","api_key_env":"TEST_KEY"},{"item":{}})
            req=send.call_args.args[0]
            self.assertEqual(req.full_url,"https://example.invalid/v1/chat/completions")
            body=json.loads(req.data)
            self.assertEqual(body["model"],"judge")
            self.assertEqual(body["response_format"],{"type":"json_object"})
            self.assertEqual(req.get_header("Authorization"),"Bearer secret")
            self.assertEqual(value,{"id":"Q01"})
    def test_unsupported_json_mode_can_be_disabled(self):
        with patch("paraling_bench.adapters.post",return_value={"choices":[{"message":{"content":"{}"}}]}) as send:
            execute("judge",{"provider":"openai_compatible","base_url":"https://example.invalid/v1","model":"x","json_mode":False,"temperature":None},{})
            self.assertNotIn("response_format",send.call_args.args[1])
            self.assertNotIn("temperature",send.call_args.args[1])
    def test_asr_multipart(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"test.wav";p.write_bytes(b"RIFF-test")
            with patch("urllib.request.urlopen",return_value=io.BytesIO(json.dumps({"text":"你好"}).encode())) as send:
                value=transcribe_api({"base_url":"https://example.invalid/v1","model":"asr"},str(p))
                req=send.call_args.args[0]
                self.assertEqual(req.full_url,"https://example.invalid/v1/audio/transcriptions")
                self.assertIn(b"RIFF-test",req.data)
                self.assertIn(b'name="model"',req.data)
                self.assertNotIn(b"script",req.data)
                self.assertEqual(value["spoken_text"],"你好")
    def test_asr_missing_text_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.wav";p.write_bytes(b"x")
            with patch("urllib.request.urlopen",return_value=io.BytesIO(b"{}")):
                with self.assertRaises(ValueError):transcribe_api({"base_url":"https://example.invalid/v1","model":"asr"},str(p))
    def test_missing_key_fails_before_network(self):
        with patch.dict(os.environ,{},clear=True),patch("urllib.request.urlopen") as send:
            with self.assertRaises(ValueError):
                execute("judge",{"provider":"openai_compatible","base_url":"https://example.invalid/v1","model":"x","api_key_env":"MISSING"},{})
            send.assert_not_called()
    def test_funasr_dispatch(self):
        import types
        fake=types.SimpleNamespace(AutoModel=lambda **kw:types.SimpleNamespace(generate=lambda **kw:[{"text":"本地转写"}]))
        with patch.dict("sys.modules",{"funasr":fake}):
            self.assertEqual(execute("asr",{"provider":"funasr","model":"paraformer-zh"},{"audio_path":"x.wav"})["spoken_text"],"本地转写")
