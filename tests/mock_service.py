"""只用于流程测试；不生成可用于评测的语音。"""
import json,sys,wave,tempfile
from pathlib import Path
request=json.load(sys.stdin)
kind=sys.argv[1]
if kind=="generate":
    p=Path(tempfile.gettempdir())/"paraling_mock.wav"
    with wave.open(str(p),"wb") as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(b"\x00\x00"*16000)
    response={"audio_path":str(p)}
elif kind=="asr":response={"status":"ok","spoken_text":"模拟转写"}
elif kind=="jev":response={"status":"ok","facts":[{"id":"mock","observed":"present","source":"simulation"}]}
else:
    response={"id":request["item"]["id"],"check_results":[{"id":c["id"],"verdict":"pass","evidence_ids":["mock"]} for c in request["evaluation_spec"]["checks"]],"scores":{"content":2,"delivery":2,"appropriateness":2},"appropriateness_evidence_ids":["mock"],"reasons":dict.fromkeys(["content","delivery","appropriateness"],"模拟响应，不代表真实质量")}
print(json.dumps(response))
