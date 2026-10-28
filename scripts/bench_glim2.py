import json, time, urllib.request
base = "http://127.0.0.1:8210"; model = "glimmer"
def run(n, prompt):
    body = json.dumps({"model": model, "max_tokens": n, "temperature": 0, "stream": False,
        "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=300))
    dt = time.time() - t0
    tok = r["usage"]["completion_tokens"]
    print(f"gen {tok} tok in {dt:.1f}s = {tok/dt:.1f} tok/s")
# banked-baseline workload: easy high-entropy-prediction text (counting)
run(512, "Count from 1 to 600, comma separated, nothing else.")
run(512, "Count from 1 to 600, comma separated, nothing else.")
# realistic trading-style prose
run(512, "Write a detailed analysis of momentum investing versus value investing for a portfolio manager.")
