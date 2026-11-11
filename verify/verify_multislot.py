#!/usr/bin/env python3
"""多 slot 分段链验收：这是 verify.sh 覆盖不到的那一块。

为什么单独立一个：verify.sh 的组 3 只发一个请求，永远只用一个 slot，所以
「hybrid + 分段链 + slot 复用」的崩溃在它那里是隐形的。修之前第 3 个请求必死
（common/common.cpp 的 GGML_ABORT）。

两个关键设计，缺一个就测不出问题：
1) prompt 长度必须**交替变化**。全等长时缓存正好够，走不到截断；必须先长后短，
   才会让 slot.prompt.tokens.pos_next() 小于缓存里的 pos_max，触发那次回滚。
2) 2 个 slot + 10 个请求，保证 slot 被反复复用。

用法：python verify/verify_multislot.py [--np 2] [--port 9350]

判据：10/10 HTTP 200、每个请求后进程存活、同一 prompt 第二次的 prompt_ms 明显低于第一次
（用耗时而不是 usage.prompt_tokens —— 后者报的是请求里的 token 数，不看缓存）。
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _qlh_env import BIN, ROOT, TMP, require   # noqa: E402
MODEL = os.environ.get("QLH_SPLIT_MODEL", r"H:/qlh_models/qwen3-5-2b-gguf/qwen35-2b-Q4_K_M.gguf")

SYS = "You are a terse assistant. Answer with a single short sentence."
TOPICS = ["France", "Japan", "Peru", "Norway", "Kenya",
          "Chile", "Nepal", "Poland", "Vietnam", "Morocco"]
# 交替长度：units 决定 user 消息的重复次数
UNITS = [6, 18, 9, 24, 7, 20, 5, 22, 8, 15]


def post(port, messages, max_tokens=16, timeout=300):
    body = json.dumps({"messages": messages, "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def health(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    require("llama-server.exe")
    ap = argparse.ArgumentParser()
    ap.add_argument("--np", type=int, default=2)
    ap.add_argument("--port", type=int, default=9350)
    ap.add_argument("--ctx", type=int, default=4096)
    args = ap.parse_args()

    cmd = [os.path.join(BIN, "llama-server.exe"), "-m", MODEL,
           "--host", "127.0.0.1", "--port", str(args.port),
           "-c", str(args.ctx), "-np", str(args.np), "-ngl", "0", "--no-warmup",
           "--split-layer-range", "0:20", "--split-tail-range", "20:24"]
    log = open(os.path.join(TMP, f"multislot_{args.port}.log"), "wb")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(60):
            if health(args.port):
                break
            if proc.poll() is not None:
                print(f"[FAIL] server died at startup, see multislot_{args.port}.log")
                return 1
            time.sleep(1)
        else:
            print("[FAIL] server not up within 60s")
            return 1

        ok = 0
        for i, topic in enumerate(TOPICS):
            if proc.poll() is not None:
                print(f"[FAIL] process was already dead before request {i} (exit={proc.returncode})")
                break
            text = (f"Name the capital of {topic}. " * UNITS[i]).strip()
            try:
                d = post(args.port, [{"role": "system", "content": SYS},
                                     {"role": "user", "content": text}])
                u = d.get("usage", {})
                print(f"  req {i:2d}: 200  prompt_n={u.get('prompt_tokens'):>4} "
                      f"alive={health(args.port)}")
                ok += 1
            except Exception as e:
                print(f"  req {i:2d}: FAIL {type(e).__name__}: {e}")
                break

        print(f"\n  surviving requests: {ok}/10")

        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": "Summarise the plot of Hamlet in one sentence. " * 6}]
        pms = []
        for r in range(3):
            if proc.poll() is not None:
                print("[FAIL] process died during the prefix-cache test")
                return 1
            d = post(args.port, msgs, max_tokens=16)
            t = d.get("timings", {})
            pms.append(t.get("prompt_ms", 0.0))
            print(f"  reuse round {r}: prompt_ms={pms[-1]:.1f}")

        reused = len(pms) == 3 and pms[0] > 0 and pms[1] < pms[0] / 3
        print(f"  round1/round0 = {(pms[1] / pms[0]):.3f} -> "
              f"{'prefix cache still pays off' if reused else 'NO reuse (fix lost the prefix cache)'}")

        alive = health(args.port)
        print(f"  health after: {200 if alive else 'DEAD'}")
        return 0 if (ok == 10 and alive and reused) else 1
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        log.close()


if __name__ == "__main__":
    sys.exit(main())
