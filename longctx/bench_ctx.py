"""Long-context DFlash vs DSpark speculative-decoding benchmark.

Runs one llama-server per config on its own port (94xx), drives /completion with
long real-document prompts (T=0, n_predict=256), and reads the full speculative
statistics from /metrics before/after each request:

  spec_decode_num_draft_tokens_total          -> gen_draft  (draft tokens generated)
  spec_decode_num_accepted_tokens_total       -> accepted drafts
  spec_decode_num_drafts_total                -> verification steps
  spec_decode_num_accepted_tokens_per_pos_total{position=i} -> accepted at slot i

Derived: accept_rate = accepted/gen_draft, mean_len = 1 + accepted/steps,
per_pos[i] = acc_per_pos[i]/steps.
"""
import json, os, re, subprocess, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = r"H:\llamacpp-c\build-split\bin\llama-server.exe"
TGT = r"H:\qlh_models\qwen3-6-35b-a3b-mtp-official\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
DFLASH = r"H:\qlh_models\dflash\qwen36-35b-a3b-dflash-Q4_K_M.gguf"
DSPARK = r"H:\qlh_models\dspark\Qwen3.6-35B-A3B-DSpark-Q4_K_M.gguf"

CONFIGS = [
    {"name": "dflash-k4",   "type": "draft-dflash", "k": 4, "p": None, "head": DFLASH},
    {"name": "dspark-k4-p0",   "type": "draft-dspark", "k": 4, "p": 0.0, "head": DSPARK},
    {"name": "dspark-k4-p09",  "type": "draft-dspark", "k": 4, "p": 0.9, "head": DSPARK},
    {"name": "dflash-k2",   "type": "draft-dflash", "k": 2, "p": None, "head": DFLASH},
    {"name": "dspark-k2-p09",  "type": "draft-dspark", "k": 2, "p": 0.9, "head": DSPARK},
    {"name": "dflash-k8",   "type": "draft-dflash", "k": 8, "p": None, "head": DFLASH},
    {"name": "dspark-k8-p09",  "type": "draft-dspark", "k": 8, "p": 0.9, "head": DSPARK},
]
DOCS = [("p1000", "p1000.txt"), ("p2500", "p2500.txt")]


def http_json(url, payload=None, timeout=1800):
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def http_text(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def wait_ready(port, logf, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            body = http_text("http://127.0.0.1:%d/health" % port, timeout=5)
            if '"ok"' in body:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def start_server(cfg, port, logpath):
    args = [BIN, "-m", TGT, "-md", cfg["head"],
            "--spec-type", cfg["type"],
            "--spec-draft-n-max", str(cfg["k"]),
            "-c", "8192", "-np", "1",
            "--host", "127.0.0.1", "--port", str(port),
            "--metrics"]
    if cfg["p"] is not None:
        args += ["--spec-draft-p-min", str(cfg["p"])]
    logf = open(logpath, "wb")
    proc = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT)
    return proc, logf


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


def diff_metrics(a, b):
    keys = ["spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total",
            "spec_decode_num_drafts_total", "tokens_predicted_total", "n_decode_total"]
    d = {}
    for k in keys:
        d[k] = b.get(k, 0) - a.get(k, 0)
    per = {}
    for i in set(list(a["per_pos"].keys()) + list(b["per_pos"].keys())):
        per[i] = b["per_pos"].get(i, 0) - a["per_pos"].get(i, 0)
    d["per_pos"] = per
    return d


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 9491
    cfgs = CONFIGS if which == "all" else [c for c in CONFIGS if c["name"] in which.split(",")]
    results_path = os.path.join(HERE, "results.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)

    prompts = {}
    for tag, fn in DOCS:
        prompts[tag] = open(os.path.join(HERE, fn), encoding="utf-8").read()

    for ci, cfg in enumerate(cfgs):
        p = port + ci
        logpath = os.path.join(HERE, "logs", "%s.log" % cfg["name"])
        print("[%s] starting server on :%d" % (cfg["name"], p), flush=True)
        t_start = time.time()
        proc, logf = start_server(cfg, p, logpath)
        try:
            if not wait_ready(p, logf):
                print("[%s] FAILED to become ready" % cfg["name"], flush=True)
                continue
            print("[%s] ready after %.1fs" % (cfg["name"], time.time() - t_start), flush=True)

            # token counts once per doc
            tok = {}
            for tag, fn in DOCS:
                r = http_json("http://127.0.0.1:%d/tokenize" % p, {"content": prompts[tag]}, timeout=120)
                tok[tag] = len(r["tokens"])
                print("[%s] %s prompt tokens = %d" % (cfg["name"], tag, tok[tag]), flush=True)

            for tag, fn in DOCS:
                print("[%s] %s: request n_predict=256 ..." % (cfg["name"], tag), flush=True)
                m0 = parse_metrics(http_text("http://127.0.0.1:%d/metrics" % p))
                t0 = time.time()
                resp = http_json("http://127.0.0.1:%d/completion" % p, {
                    "prompt": prompts[tag],
                    "n_predict": 256,
                    "temperature": 0.0,
                    "top_k": 1,
                    "top_p": 1.0,
                    "seed": 42,
                    "cache_prompt": False,
                    "stream": False,
                    "n_probs": 0,
                })
                dt = time.time() - t0
                m1 = parse_metrics(http_text("http://127.0.0.1:%d/metrics" % p))
                d = diff_metrics(m0, m1)
                tm = resp.get("timings", {})
                gen_draft = int(d["spec_decode_num_draft_tokens_total"])
                acc = int(d["spec_decode_num_accepted_tokens_total"])
                steps = int(d["spec_decode_num_drafts_total"])
                row = {
                    "config": cfg["name"], "spec_type": cfg["type"], "k": cfg["k"], "p": cfg["p"],
                    "doc": tag, "prompt_tokens": tok[tag],
                    "n_predict_req": 256, "n_predicted": resp.get("tokens_predicted"),
                    "accept_rate": round(acc / gen_draft, 4) if gen_draft else None,
                    "mean_len": round(1.0 + acc / steps, 3) if steps else None,
                    "gen_draft": gen_draft, "acc_draft": acc, "verif_steps": steps,
                    "per_pos": {str(i): round(d["per_pos"].get(i, 0) / steps, 4) for i in range(len(d["per_pos"]))} if steps else {},
                    "per_pos_counts": {str(i): d["per_pos"].get(i, 0) for i in sorted(d["per_pos"])},
                    "tok_s": round(tm.get("predicted_per_second", 0.0), 2),
                    "prompt_tok_s": round(tm.get("prompt_per_second", 0.0), 2),
                    "wall_s": round(dt, 1),
                }
                results["%s|%s" % (cfg["name"], tag)] = row
                json.dump(results, open(results_path, "w"), indent=1)
                print(json.dumps(row), flush=True)
                time.sleep(1)
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
            logf.close()
        time.sleep(2)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
