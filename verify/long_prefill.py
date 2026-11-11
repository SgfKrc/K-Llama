#!/usr/bin/env python3
"""Long-prompt prefill cost test.

Measures prompt_ms for a ~1000-1200 token prompt on a running llama-server.
Calibrates text against /tokenize so the chat prompt token count (usage.prompt_tokens)
lands in range. Same chat-completions path as acc_test.py.
"""
import argparse, json, os, sys, time, urllib.request

def post(port, path, body, timeout=1800):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def wait_ok(port, tries=300):
    for _ in range(tries):
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            with urllib.request.urlopen(req, timeout=5) as r:
                d = json.loads(r.read())
            if d.get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--text",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "long_prompt.txt"))
    ap.add_argument("--target-tokens", type=int, default=1100)
    ap.add_argument("--n-predict", type=int, default=16)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    text = open(args.text, encoding="utf-8").read()
    if not wait_ok(args.port):
        print("SERVER NOT OK", flush=True)
        sys.exit(2)

    # calibrate: tokenize, then cut characters proportionally
    n = len(post(args.port, "/tokenize", {"content": text}, timeout=300)["tokens"])
    use = text
    for _ in range(4):
        if n <= args.target_tokens + 40:
            break
        cut = int(len(use) * (args.target_tokens / n) * 0.995)
        use = use[:cut]
        n = len(post(args.port, "/tokenize", {"content": use}, timeout=300)["tokens"])
    print(f"calibrated raw_tokens={n}", flush=True)

    body = {
        "messages": [{"role": "user", "content": use}],
        "max_tokens": args.n_predict,
        "temperature": 0,
        "top_k": 1,
        "cache_prompt": False,
    }
    t0 = time.time()
    d = post(args.port, "/v1/chat/completions", body)
    t = d.get("timings", {})
    u = d.get("usage", {})
    row = {
        "prompt_tokens": t.get("prompt_n", u.get("prompt_tokens")),
        "prompt_ms": t.get("prompt_ms"),
        "prompt_tps": t.get("prompt_per_second"),
        "gen_tokens": t.get("predicted_n", u.get("completion_tokens")),
        "gen_ms": t.get("predicted_ms"),
        "wall_s": round(time.time() - t0, 3),
        "text_head": (d["choices"][0]["message"].get("content") or "")[:60],
    }
    print(json.dumps(row, ensure_ascii=False), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=1)

if __name__ == "__main__":
    main()
