#!/usr/bin/env python3
"""Long-lane needle test. argv[1] = target token count (default 130000).
Parses both `content` and `reasoning`/`reasoning_content` stream deltas."""
import json, sys, time, urllib.request, urllib.error

URL = "http://localhost:8211/v1/chat/completions"
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 130000

def para(i):
    return (f"Section {i}. The assembly line {i} reports throughput of {100+i%37} units "
            f"on segment {i%13}, calibrated by technician {(i*7)%53} at hour {(i*3)%24}, "
            f"torque spec {(20+i%41)} Nm, dust index {(i%9)/10:.1f}.")  # ~170 chars

def ask(prompt, max_tokens=256):   # reasoning model: room for thinking + answer
    body = json.dumps({"model":"glimmer-long","messages":[{"role":"user","content":prompt}],
                       "max_tokens":max_tokens,"temperature":0,"stream":True}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type":"application/json"})
    t0 = time.time(); ttft = None; out = []
    try:
        with urllib.request.urlopen(req, timeout=1500) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"): continue
                try: ev = json.loads(line[5:].strip())
                except Exception: continue
                d = (ev.get("choices") or [{}])[0].get("delta", {})
                c = d.get("content") or d.get("reasoning") or d.get("reasoning_content")
                if c:
                    if ttft is None: ttft = time.time()-t0
                    out.append(c)
    except Exception as e:
        return ttft, time.time()-t0, f"__ERR__ {type(e).__name__}: {str(e)[:120]}"
    return ttft, time.time()-t0, "".join(out).strip()

n = max(2, TARGET // max(1, len(para(1)) // 4))   # true tokens/paragraph
paras = [para(i) for i in range(n)]
N1 = "NEEDLE ALPHA: the KX-7 module serial number is VRX-8814-Q7."
N2 = "NEEDLE OMEGA: the audit report for order 44-918-XZ was prepared in the city of Thessaloniki."
paras[n//2] = N1 + " " + paras[n//2]
paras[int(n*0.92)] = N2 + " " + paras[int(n*0.92)]
context = "\n".join(paras)
approx = len(context)//4
print(f"target={TARGET} approx_tokens={approx}")

for depth, needle_q, fact, tag in [
    ("50%", f"{context}\n\nQuestion: What is the serial number of the KX-7 module? Answer with just the serial.", "VRX-8814-Q7", "needle@50"),
    ("92%", f"{context}\n\nQuestion: In which city was the audit report for order 44-918-XZ prepared? Answer with just the city.", "Thessaloniki", "needle@92"),
]:
    ttft, tot, ans = ask(needle_q)
    ok = fact in ans
    prefill = int(approx/ttft) if ttft else 0
    print(f"{tag} ttft={ttft} total={tot:.1f} prefill_tps={prefill} pass={ok} answer={ans[:100]!r}")
    if ans.startswith("__ERR__"):
        print("SERVER DIED MID-REQUEST — record crash depth, restart lane, retry lower")
        sys.exit(2)
