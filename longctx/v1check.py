#!/usr/bin/env python3
"""Same 3 prompts through /v1/chat/completions (server-side template), read spec stats
from /metrics deltas. Used to check alignment with the known n=64 baseline."""
import json, socket, sys, time, urllib.request

PROMPTS = [
    ("prose", "Explain in a few paragraphs how a B-tree maintains balance when keys are inserted, "
              "and why a large branching factor matters for disk I/O."),
    ("code",  "Write a Python function merge_intervals(intervals) that merges overlapping intervals "
              "and returns the merged list. Include a docstring and one worked example."),
    ("zh",    "用通俗的语言解释：为什么混合专家模型（MoE）在推理时比同等总参数量的稠密模型更省算力？"
              "并说明路由器的负载均衡为什么重要。"),
]


def http_text(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def http_json(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def metrics(port):
    out = {}
    for line in http_text("http://127.0.0.1:%d/metrics" % port).splitlines():
        if line.startswith("llamacpp:spec_decode"):
            k, v = line.rsplit(" ", 1)
            out[k] = out.get(k, 0) + float(v)
    return out


def main():
    port = int(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    rows = []
    for name, content in PROMPTS:
        m0 = metrics(port)
        d = http_json("http://127.0.0.1:%d/v1/chat/completions" % port, {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": n, "temperature": 0, "top_k": 1, "cache_prompt": False})
        m1 = metrics(port)
        g = m1.get("llamacpp:spec_decode_num_draft_tokens_total", 0) - m0.get("llamacpp:spec_decode_num_draft_tokens_total", 0)
        a = m1.get("llamacpp:spec_decode_num_accepted_tokens_total", 0) - m0.get("llamacpp:spec_decode_num_accepted_tokens_total", 0)
        s = m1.get("llamacpp:spec_decode_num_drafts_total", 0) - m0.get("llamacpp:spec_decode_num_drafts_total", 0)
        g = int(g); a = int(a); s = int(s)
        row = {"name": name, "n": n, "prompt_n": d.get("timings", {}).get("prompt_n"),
               "gen": d.get("timings", {}).get("predicted_n"),
               "accept_rate": round(a / g, 4) if g else None,
               "mean_len": round(1 + a / s, 3) if s else None,
               "gen_draft": g, "acc": a, "steps": s,
               "tok_s": round(d.get("timings", {}).get("predicted_per_second") or 0, 2)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if rows:
        tg = sum(r["gen_draft"] for r in rows); ta = sum(r["acc"] for r in rows)
        ts = sum(r["steps"] for r in rows)
        print("TOTAL n=%d accept_rate=%.4f mean_len=%.3f gen_draft=%d acc=%d steps=%d" % (
            n, ta / tg, 1 + ta / ts, tg, ta, ts), flush=True)


if __name__ == "__main__":
    main()
