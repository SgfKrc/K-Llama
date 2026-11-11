#!/usr/bin/env python3
"""A 的验收：分段链在 -np > 1 下能不能服务并发请求，且结果与串行逐字节相同。

为什么必须分两个进程跑：串行和并发不能在同一次会话里比。串行跑里第二个请求会命中
prefix cache / 复用 slot，而并发跑里两个请求共享一次 decode，两者的 KV 来源不同，只有在
「同一份代码、同样两个 prompt」下分别取输出再比，才能说明并发路径没有算错。

判据（全部满足才算过）：
  1. 串行两个请求都 200；并发两个请求都 200（以前是 500 + "Compute error"）；
  2. 四次输出（串行 A、串行 B、并发 A、并发 B）里，同一 prompt 的两次逐字节相同；
  3. 并发跑结束后进程还活着。

用法: python verify/accept_np.py --np 2 [--port 9371]
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _qlh_env import BIN, ROOT, TMP, require   # noqa: E402
MODEL = os.environ.get("QLH_SPLIT_MODEL", r"H:/qlh_models/qwen3-5-2b-gguf/qwen35-2b-Q4_K_M.gguf")

# 两个 prompt 必须不同（否则 prefix cache 会让 slot 复用掩盖问题），但都足够长，
# 好让两个 slot 的 prefill 落进同一个 batch（250 + 250 < n_batch 512）。
PROMPTS = [
    "Name the capital of France. Answer with one word.",
    "Name the capital of Japan. Answer with one word.",
    "Name the capital of Peru. Answer with one word.",
    "Name the capital of Norway. Answer with one word.",
]


def health(port):
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def post(port, text, out, key, max_tokens=24):
    body = json.dumps({"messages": [{"role": "user", "content": text}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            raw = r.read()
            out[key] = (r.status, raw)
    except Exception as e:
        out[key] = (getattr(e, "code", "ERR"), str(e).encode())


def content(raw):
    """取 message 的 reasoning_content + content —— 只比采样出来的文字，不比时间戳。"""
    try:
        m = json.loads(raw)["choices"][0]["message"]
    except Exception:
        return None
    return ((m.get("reasoning_content") or "") + "\x00" + (m.get("content") or "")).encode()


def start(port, np, log_path):
    log = open(log_path, "wb")
    cmd = [os.path.join(BIN, "llama-server.exe"), "-m", MODEL,
           "--host", "127.0.0.1", "--port", str(port),
           "-c", str(4096 * max(1, np // 2)), "-np", str(np), "-ngl", "0", "--no-warmup",
           "--split-layer-range", "0:20", "--split-tail-range", "20:24"]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        if health(port):
            return proc, log
        if proc.poll() is not None:
            return None, log
        time.sleep(1)
    return None, log


def stop(proc, log):
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except Exception:
        proc.kill()
    log.close()


def main():
    require("llama-server.exe")
    ap = argparse.ArgumentParser()
    ap.add_argument("--np", type=int, default=2)
    ap.add_argument("--port", type=int, default=9371)
    args = ap.parse_args()
    np = args.np
    texts = PROMPTS[:np]
    n_fail = 0

    def check(what, ok):
        nonlocal n_fail
        print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
        if not ok:
            n_fail += 1

    # ---------------------------------------------------------------- 串行
    serial = {}
    port = args.port
    proc, log = start(port, np, os.path.join(TMP, "accept_serial_%d.log" % port))
    if proc is None:
        print("[FAIL] serial server did not start")
        return 1
    try:
        for i, t in enumerate(texts):
            post(port, t, serial, i)
    finally:
        stop(proc, log)
    check("串行 %d 个请求全部 200" % np,
          all(serial.get(i, (None,))[0] == 200 for i in range(np)))

    # ---------------------------------------------------------------- 并发
    conc = {}
    port2 = port + 1
    proc, log = start(port2, np, os.path.join(TMP, "accept_conc_%d.log" % port2))
    if proc is None:
        print("[FAIL] concurrent server did not start")
        return 1
    alive = False
    try:
        th = [threading.Thread(target=post, args=(port2, t, conc, i)) for i, t in enumerate(texts)]
        t0 = time.time()
        for t in th:
            t.start()
        for t in th:
            t.join()
        dt = time.time() - t0
        alive = health(port2)
        print("    concurrent wall time = %.1fs" % dt)
    finally:
        stop(proc, log)
    check("并发 %d 个请求全部 200" % np,
          all(conc.get(i, (None,))[0] == 200 for i in range(np)))
    check("并发跑之后进程存活", alive)

    # ---------------------------------------------------------------- 逐字节
    for i in range(np):
        a = serial.get(i, (None, b""))[1]
        b = conc.get(i, (None, b""))[1]
        ca, cb = content(a), content(b)
        if ca is None or cb is None:
            print("    prompt %d: 解析失败 serial=%r conc=%r" % (i, a[:80], b[:80]))
            check("prompt %d 串行/并发输出逐字节相同" % i, False)
            continue
        print("    prompt %d: serial=%r conc=%r" % (i, ca[:60], cb[:60]))
        check("prompt %d 串行/并发输出逐字节相同" % i, ca == cb)

    print("\n%s" % ("ALL PASS" if n_fail == 0 else "FAILED"))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
