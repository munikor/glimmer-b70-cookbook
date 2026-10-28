import json, time, urllib.request, sys
base = "http://127.0.0.1:8210"; model = "glimmer"
def run(n):
    body = json.dumps({"model": model, "max_tokens": n, "temperature": 0, "stream": False,
        "messages": [{"role": "user", "content": "Write a detailed 500-word essay on the history of the Roman aqueducts, be verbose."}]}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=180))
    dt = time.time() - t0
    tok = r["usage"]["completion_tokens"]
    print(f"gen {tok} tokens in {dt:.1f}s = {tok/dt:.1f} tok/s (incl prefill)")
for n in (256, 512, 512):
    run(n)
