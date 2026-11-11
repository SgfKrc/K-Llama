import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, "results.json")))

order = ["dflash-k4", "dspark-k4-p0", "dspark-k4-p09", "dflash-k2", "dspark-k2-p09", "dflash-k8", "dspark-k8-p09"]
tags = ["p1000", "p2500"]

rows = {}
for k, v in res.items():
    rows[(v["config"], v["doc"])] = v

print("## headline table\n")
print("| config | doc | prompt tok | n_pred | accept | mean len | gen_draft | acc/steps | tok/s | wall s |")
print("|---|---|---|---|---|---|---|---|---|---|")
for c in order:
    for t in tags:
        r = rows.get((c, t))
        if not r:
            continue
        print("| %s | %s | %d | %s | %s | %s | %d | %d/%d | %s | %s |" % (
            c, t, r["prompt_tokens"], r.get("n_predicted"),
            r["accept_rate"], r["mean_len"], r["gen_draft"],
            r["acc_draft"], r["verif_steps"], r["tok_s"], r["wall_s"]))

print("\n## per-position acceptance (each column = position index)\n")
for t in tags:
    print("### %s\n" % t)
    names = [c for c in order if (c, t) in rows]
    if not names:
        continue
    maxpos = max(len(rows[(c, t)]["per_pos"]) for c in names)
    print("| config | " + " | ".join("pos%d" % i for i in range(maxpos)) + " |")
    print("|---|" + "---|" * maxpos)
    for c in names:
        pp = rows[(c, t)]["per_pos"]
        print("| %s | " % c + " | ".join(
            (("%.3f" % pp[str(i)]) if str(i) in pp else "-") for i in range(maxpos)) + " |")

print("\n## delta vs short-context baseline (mean_len, from task sheet)\n")
base = {"dflash-k4": 3.657, "dspark-k4-p0": 3.450, "dspark-k4-p09": 3.437, "dflash-k2": 2.653, "dspark-k2-p09": 2.777}
for c in order:
    for t in tags:
        r = rows.get((c, t))
        if not r or c not in base:
            continue
        print("%-16s %-6s mean_len %5.3f -> %5.3f  (%+5.1f%%)   accept %s" % (
            c, t, base[c], r["mean_len"], 100.0 * (r["mean_len"] / base[c] - 1.0), r["accept_rate"]))
