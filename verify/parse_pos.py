#!/usr/bin/env python3
# 从 -lv 4 日志里抓 per-position 接受率（SLT_TRC "acc per pos"）。
import os, re, sys, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _qlh_env import TMP, build_dir   # noqa: E402

def parse(path):
    out = []
    txt = open(path, errors="ignore").read()
    # 每个请求结束会打 "acc per pos = (0.834, 0.79, ...)"（TRC 级别）
    for m in re.finditer(r"task (\d+) \| draft acceptance = ([\d.]+).*?mean len =\s*([\d.]+)", txt):
        task = int(m.group(1))
        tail = txt[m.end():m.end()+4000]
        pm = re.search(r"acc per pos = \(([^)]*)\)", tail)
        pos = [float(x) for x in pm.group(1).split(",")] if pm else []
        out.append({"task": task, "acc": float(m.group(2)), "len": float(m.group(3)), "pos": pos})
    return out

if __name__ == "__main__":
    # 先看本次验收的落点，再看上游构建目录里历史留下的 acc/（旧日志不迁移）。
    pats = [os.path.join(TMP, "acc", "*.log"), os.path.join(build_dir(), "acc", "*.log")]
    found = [p for pat in pats for p in sorted(glob.glob(pat))]
    for f in sys.argv[1:] or found:
        rows = parse(f)
        print(f"== {f}")
        for r in rows:
            if r["task"] == 0:
                continue
            print(f"  task {r['task']:3d} acc={r['acc']:.4f} len={r['len']:.2f} pos={r['pos']}")
