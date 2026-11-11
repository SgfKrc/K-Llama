#!/usr/bin/env python3
"""Long-context DSpark-6L vs DFlash, k=2, three length tiers.

Tiers: short (~40 tok, 3 acc_test prompts), mid (~1100 tok), long (~3150 tok),
all driven through /completion with the same manual Qwen chat template, T=0,
n_predict=256, cache_prompt=false.

Speculative stats come from /metrics deltas (per-position accepted counts),
timing from response.timings.

Usage:
  python lc6l_bench.py start <cfg> <port>      # export LC_* env; spawns server
  python lc6l_bench.py run <cfg> <port> <tiers> # drive requests, append jsonl
"""
import json, os, re, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = r"H:\llamacpp-c\build-split\bin\llama-server.exe"
TGT = r"H:\qlh_models\qwen3-6-35b-a3b-mtp-official\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
DFLASH = r"H:\qlh_models\dflash\qwen36-35b-a3b-dflash-Q4_K_M.gguf"
DSPARK6 = r"H:\qlh_models\dspark6\Qwen3.6-35B-A3B-DSPARK-6L.gguf"

CFG = {
    # name: (spec-type, k, head)
    "6l-k2":     ("draft-dspark", 2, DSPARK6),
    "dflash-k2": ("draft-dflash", 2, DFLASH),
    "6l-k4":     ("draft-dspark", 4, DSPARK6),
    "dflash-k4": ("draft-dflash", 4, DFLASH),
    "6l-k8":     ("draft-dspark", 8, DSPARK6),
    "dflash-k8": ("draft-dflash", 8, DFLASH),
}

SHORT_PROMPTS = [
    ("prose", "Explain in a few paragraphs how a B-tree maintains balance when keys are inserted, "
              "and why a large branching factor matters for disk I/O."),
    ("code",  "Write a Python function merge_intervals(intervals) that merges overlapping intervals "
              "and returns the merged list. Include a docstring and one worked example."),
    ("zh",    "用通俗的语言解释：为什么混合专家模型（MoE）在推理时比同等总参数量的稠密模型更省算力？"
              "并说明路由器的负载均衡为什么重要。"),
]

TIERS = {
    # tag -> (kind, payload, n_predict)
    "short":   ("inline", SHORT_PROMPTS, 256),
    "short64": ("inline", SHORT_PROMPTS, 64),      # aligns with the known k=2 baseline (n=64)
    "mid":     ("file", os.path.join(HERE, "p1000.txt"), 256),
    "long":    ("file", os.path.join(HERE, "p2500.txt"), 256),
    "hard":    ("file", os.path.join(HERE, "p_hard.txt"), 256),  # same README doc, answer in Chinese -> cannot copy
}


def http_text(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def http_json(url, payload=None, timeout=1800):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def parse_metrics(txt):
    out = {"per_pos": {}}
    for line in txt.splitlines():
        m = re.match(r'^llamacpp:([a-z0-9_]+)(\{position="(\d+)"\})?\s+([-\d.eE+]+)$', line.strip())
        if not m:
            continue
        name, _, pos, val = m.group(1), m.group(2), m.group(3), float(m.group(4))
        if name == "spec_decode_num_accepted_tokens_per_pos_total":
            out["per_pos"][int(pos)] = int(val)
        else:
            out[name] = val
    return out


def wait_ready(port, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if '"ok"' in http_text("http://127.0.0.1:%d/health" % port, timeout=5):
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def cmd_start(name, port):
    if name not in CFG:
        sys.exit("unknown cfg " + name)
    spec, k, head = CFG[name]
    logpath = os.path.join(HERE, "logs", "%s_%d.log" % (name, port))
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    args = [BIN, "-m", TGT, "-md", head,
            "--spec-type", spec, "--spec-draft-n-max", str(k),
            "-c", "8192", "-b", "512", "-np", "1", "-t", "8", "-ngl", "0",
            "--no-warmup", "--metrics",
            "--host", "127.0.0.1", "--port", str(port)]
    logf = open(logpath, "wb")
    proc = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT)
    print(json.dumps({"pid": proc.pid, "log": logpath, "port": port, "cfg": name}), flush=True)


def strip_template(p):
    pre = "<|im_start|>user\n"
    if p.startswith(pre):
        p = p[len(pre):]
    for tail in ("<|im_end|>\n<|im_start|>assistant\n", "<|im_end|>\n<|im_start|>assistant"):
        if p.endswith(tail):
            p = p[:-len(tail)]
            break
    return p


def cmd_run(name, port, tiers, api="completion"):
    spec, k, head = CFG[name]
    outpath = os.path.join(HERE, "lc6l_results.jsonl")
    sel = [t for t in tiers.split(",") if t]
    for tag in sel:
        kind, payload, n_pred_req = TIERS[tag]
        if kind == "inline":
            items = [(c, n) for n, c in payload]
        else:
            # p1000.txt / p2500.txt already are complete chat-templated prompts
            items = [(open(payload, encoding="utf-8").read(), tag)]
        for i, (prompt, pname) in enumerate(items):
            m0 = parse_metrics(http_text("http://127.0.0.1:%d/metrics" % port))
            t0 = time.time()
            if api == "v1":
                resp = http_json("http://127.0.0.1:%d/v1/chat/completions" % port, {
                    "messages": [{"role": "user", "content": strip_template(prompt)}],
                    "max_tokens": n_pred_req,
                    "temperature": 0, "top_k": 1, "cache_prompt": False,
                })
            else:
                resp = http_json("http://127.0.0.1:%d/completion" % port, {
                    "prompt": "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
                              if kind == "inline" else prompt,
                    "n_predict": n_pred_req,
                    "temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": 42,
                    "cache_prompt": False, "stream": False, "n_probs": 0,
                })
            dt = time.time() - t0
            m1 = parse_metrics(http_text("http://127.0.0.1:%d/metrics" % port))
            tm = resp.get("timings", {})

            def dl(k):
                return m1.get(k, 0) - m0.get(k, 0)
            gen_draft = int(dl("spec_decode_num_draft_tokens_total"))
            acc = int(dl("spec_decode_num_accepted_tokens_total"))
            steps = int(dl("spec_decode_num_drafts_total"))
            pos = {i: int(m1["per_pos"].get(i, 0) - m0["per_pos"].get(i, 0))
                   for i in set(list(m0["per_pos"]) + list(m1["per_pos"]))}
            row = {
                "cfg": name, "spec_type": spec, "k": k, "api": api,
                "tier": tag, "prompt_name": pname,
                "prompt_n": tm.get("prompt_n"),
                "n_predict_req": n_pred_req,
                "n_predicted": resp.get("tokens_predicted"),
                "accept_rate": round(acc / gen_draft, 4) if gen_draft else None,
                "mean_len": round(1.0 + acc / steps, 3) if steps else None,
                "gen_draft": gen_draft, "acc_draft": acc, "verif_steps": steps,
                "per_pos_counts": {str(i): pos[i] for i in sorted(pos)},
                "per_pos": {str(i): round(pos[i] / steps, 4) for i in sorted(pos)} if steps else {},
                "tok_s": round(tm.get("predicted_per_second") or 0.0, 2),
                "prompt_tok_s": round(tm.get("prompt_per_second") or 0.0, 2),
                "wall_s": round(dt, 1),
            }
            with open(outpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)
            time.sleep(0.5)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    mode = sys.argv[1]
    if mode == "start":
        cmd_start(sys.argv[2], int(sys.argv[3]))
    elif mode == "run":
        cmd_run(sys.argv[2], int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "short,mid,long",
                sys.argv[5] if len(sys.argv) > 5 else "completion")
