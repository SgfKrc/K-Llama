#!/usr/bin/env python3
# 接受率基线：发 N 个真实 prompt（T=0, n_predict=64），收集 timings。
# 用法: python acc_test.py --port 9402 --out build-split/acc_xxx.json
import argparse, json, sys, urllib.request

PROMPTS = [
    ("prose", "Explain in a few paragraphs how a B-tree maintains balance when keys are inserted, "
              "and why a large branching factor matters for disk I/O."),
    ("code",  "Write a Python function merge_intervals(intervals) that merges overlapping intervals "
              "and returns the merged list. Include a docstring and one worked example."),
    ("zh",    "用通俗的语言解释：为什么混合专家模型（MoE）在推理时比同等总参数量的稠密模型更省算力？"
              "并说明路由器的负载均衡为什么重要。"),
]

def ask(port, content, n_predict=64, ctx=None):
    body = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": n_predict,
        "temperature": 0,
        "top_k": 1,
        "cache_prompt": False,   # 每次都是全新的 prefill，避免前缀缓存影响 tok/s
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = []
    for name, content in PROMPTS:
        d = ask(args.port, content, args.n_predict)
        t = d.get("timings", {})
        usage = d.get("usage", {})
        msg = d["choices"][0]["message"]
        reasoning = (msg.get("reasoning_content") or "")
        content = (msg.get("content") or "")
        txt = reasoning + content
        rows.append({
            "name": name,
            "prompt_tokens": t.get("prompt_n", usage.get("prompt_tokens")),
            "gen_tokens": t.get("predicted_n", usage.get("completion_tokens")),
            "prompt_tps": t.get("prompt_per_second"),
            "gen_tps": t.get("predicted_per_second"),
            "gen_ms": t.get("predicted_ms"),
            "text_head": txt[:80].replace("\n", " "),
            "text": txt,
        })
        print(f"[{name}] prompt_n={rows[-1]['prompt_tokens']} gen_n={rows[-1]['gen_tokens']} "
              f"gen_tps={rows[-1]['gen_tps']:.3f}", flush=True)

    out = {"tag": args.tag, "n_predict": args.n_predict, "rows": rows}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("WROTE", args.out)

if __name__ == "__main__":
    main()
