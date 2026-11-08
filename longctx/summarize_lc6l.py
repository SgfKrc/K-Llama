#!/usr/bin/env python3
"""Summarize lc6l_results.jsonl: dedupe, aggregate short tier by gen_draft, print tables."""
import json, os, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "lc6l_results.jsonl")


def load():
    rows = []
    if not os.path.exists(PATH):
        return rows
    for line in open(PATH, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("api") != "v1":
            continue
        rows.append(r)
    # keep last occurrence per (cfg, tier, prompt_name, n_predict_req)
    seen = {}
    for r in rows:
        seen[(r["cfg"], r["tier"], r["prompt_name"], r.get("n_predict_req"))] = r
    return list(seen.values())


def agg(cfg, tier, n):
    rs = [r for r in ROWS if r["cfg"] == cfg and r["tier"] == tier and r.get("n_predict_req") == n]
    if not rs:
        return None
    g = sum(r["gen_draft"] for r in rs)
    a = sum(r["acc_draft"] for r in rs)
    s = sum(r["verif_steps"] for r in rs)
    pos = collections.Counter()
    for r in rs:
        for k, v in r["per_pos_counts"].items():
            pos[int(k)] += v
    tps = sum(r["tok_s"] for r in rs) / len(rs)
    return {"accept": a / g, "mean": 1 + a / s, "gd": g, "acc": a, "steps": s,
            "pos": {i: pos[i] / s for i in sorted(pos)}, "tps": tps,
            "pn": [r["prompt_n"] for r in rs]}


def show(tag, a, b):
    if not a or not b:
        print("| %-6s | (missing) |" % tag)
        return
    print("| %-6s | %.4f | %.3f | %d/%d/%d | %.4f | %.3f | %d/%d/%d | %+6.2f | %+6.2f |" % (
        tag, a["accept"], a["mean"], a["gd"], a["acc"], a["steps"],
        b["accept"], b["mean"], b["gd"], b["acc"], b["steps"],
        100 * (a["accept"] - b["accept"]), 100 * (a["mean"] - b["mean"])))


ROWS = load()
if not ROWS:
    sys.exit("no v1 rows")

print("## k=2 (n_predict=256, /v1)\n")
print("| tier | 6L accept | 6L mean | 6L gd/acc/steps | DF accept | DF mean | DF gd/acc/steps | d_accept(pp) | d_mean |")
print("|---|---|---|---|---|---|---|---|---|")
for t in ["short", "mid", "long", "hard"]:
    show(t, agg("6l-k2", t, 256), agg("dflash-k2", t, 256))

print("\n## k=8 (n_predict=256, /v1)\n")
print("| tier | 6L accept | 6L mean | 6L gd/acc/steps | DF accept | DF mean | DF gd/acc/steps | d_accept(pp) | d_mean |")
print("|---|---|---|---|---|---|---|---|---|")
for t in ["short", "mid", "long", "hard"]:
    show(t, agg("6l-k8", t, 256), agg("dflash-k8", t, 256))

print("\n## n=64 baseline check (k=2 and k=8)\n")
print("| cfg | tier | accept | mean | gd/acc/steps |")
print("|---|---|---|---|---|")
for cfg in ["6l-k2", "dflash-k2", "6l-k8", "dflash-k8"]:
    for t in ["short64"]:
        a = agg(cfg, t, 64)
        if a:
            print("| %s | %s | %.4f | %.3f | %d/%d/%d |" % (cfg, t, a["accept"], a["mean"], a["gd"], a["acc"], a["steps"]))

print("\n## per-position (k=8)\n")
for t in ["short64", "short", "mid", "long", "hard"]:
    a = agg("6l-k8", t, 256) or agg("6l-k8", t, 64)
    b = agg("dflash-k8", t, 256) or agg("dflash-k8", t, 64)
    if a and b:
        mp = 8
        print("### %s\n" % t)
        print("| cfg | " + " | ".join("pos%d" % i for i in range(mp)) + " |")
        print("|---|" + "---|" * mp)
        print("| 6L | " + " | ".join("%.3f" % a["pos"].get(i, float("nan")) for i in range(mp)) + " |")
        print("| DF | " + " | ".join("%.3f" % b["pos"].get(i, float("nan")) for i in range(mp)) + " |")
        print("| d  | " + " | ".join("%+.3f" % (a["pos"].get(i, 0) - b["pos"].get(i, 0)) for i in range(mp)) + " |")

print("\n## prompt_n\n")
for t in ["short", "mid", "long", "hard"]:
    a = agg("6l-k2", t, 256)
    if a:
        print("* %s: prompt_n per request = %s" % (t, a["pn"]))
