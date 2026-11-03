"""Offline protocol fixture: synthetic tone, not speech evaluation."""
import json,sys,wave,tempfile,math,struct
from pathlib import Path
request=json.load(sys.stdin)
kind=sys.argv[1]
if kind=="generate":
    p=Path(tempfile.gettempdir())/("paraling_mock_"+request["id"]+".wav")
    with wave.open(str(p),"wb") as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h",int(10000*math.sin(2*math.pi*220*i/16000))) for i in range(16000)))
    response={"status":"ok","audio_path":str(p)}
elif kind=="asr":response={"status":"ok","spoken_text":"模拟转写"}
elif kind=="align":response={"status":"ok","chars":[]}
elif kind=="judge":
    facts=request["evidence"]["facts"]
    refs=[f["id"] for f in facts if f["observed"]!="unknown"]
    response={"id":request["item"]["id"],"check_results":request["evidence"]["check_results"],"scores":{"content":None,"delivery":5,"appropriateness":5 if refs else None},"appropriateness_evidence_ids":refs,"reasons":dict.fromkeys(["content","delivery","appropriateness"],"SIMULATION ONLY")}
else:
    response={"status":"ok","check_results":[{"id":c["id"],"verdict":"pass","reason":"SIMULATION ONLY"} for c in request["evaluation_spec"]["checks"]]}
print(json.dumps(response))
