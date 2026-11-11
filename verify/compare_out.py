#!/usr/bin/env python3
"""Compare acc_test JSON outputs byte-for-byte (per prompt text)."""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--test", required=True)
ap.add_argument("--ref-name", default="ref")
ap.add_argument("--test-name", default="test")
a = ap.parse_args()

r = json.load(open(a.ref, encoding="utf-8"))
t = json.load(open(a.test, encoding="utf-8"))
rm = {x["name"]: x for x in r["rows"]}
tm = {x["name"]: x for x in t["rows"]}
all_same = True
for name in rm:
    rt = rm[name]["text"]
    tt = tm.get(name, {}).get("text", "")
    same = rt == tt
    all_same &= same
    print(f"[{name}] {'IDENTICAL' if same else 'DIFF'}  len({a.ref_name})={len(rt)} len({a.test_name})={len(tt)}")
    if not same:
        # first diff position
        for i, (x, y) in enumerate(zip(rt, tt)):
            if x != y:
                print(f"  first diff at char {i}: {a.ref_name}={rt[i:i+40]!r} vs {a.test_name}={tt[i:i+40]!r}")
                break
        else:
            print(f"  prefix identical up to min len; tail diff: {rt[len(tt):][:60]!r} vs {tt[len(rt):][:60]!r}")
print("ALL IDENTICAL" if all_same else "DIFFERENCES FOUND")
