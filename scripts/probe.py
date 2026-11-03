"""链路自检：连 head 和 tail，跑 N 轮单 token，报每段耗时。

两台机器上各起一个 run.bat 之后，在**任意一台能同时访问两者**的机器上跑：

    runtime\\python.exe probe.py --head 127.0.0.1:9103 --tail 100.x.x.x:9106

它验证三件事：两端都能握手、hidden 尺寸对得上、往返真的能出 token。
"""

import argparse
import socket
import sys
import time
from pathlib import Path

# protocol / wire 在 worker/ 下，包根目录跑的时候要自己补上
sys.path.insert(0, str(Path(__file__).resolve().parent / "worker"))

import numpy as np
import protocol


def split_addr(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    if not host or not port:
        raise argparse.ArgumentTypeError(f"地址要写成 host:port，收到 {text!r}")
    return host, int(port)


def reset(sock: socket.socket) -> bool:
    protocol.send(sock, protocol.RESET)
    r = protocol.read(sock)
    return r.kind == protocol.RESULT and bool(r.meta.get("ok"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=split_addr, required=True, help="第一段 host:port")
    ap.add_argument("--tail", type=split_addr, required=True, help="最后一段 host:port")
    ap.add_argument("-n", type=int, default=24, help="测几轮")
    args = ap.parse_args()

    try:
        hs = socket.create_connection(args.head, timeout=30)
        hm = protocol.read(hs).meta
        ts = socket.create_connection(args.tail, timeout=30)
        tm = protocol.read(ts).meta
    except OSError as exc:
        print(f"连不上：{exc}")
        print("确认两端的 run.bat 都在跑，地址和端口填对了。")
        return 2

    print("head: n_layer=%s range=%s" % (hm.get("n_layer"), hm.get("layer_range")))
    print("tail: n_layer=%s range=%s" % (tm.get("n_layer"), tm.get("layer_range")))

    # 两段必须首尾相接，否则 hidden 语义不对
    hl, tl = int(hm.get("n_layer") or 0), int(tm.get("n_layer") or 0)
    if hl + tl != 40:
        print(f"警告：两段合计 {hl + tl} 层，qwen35moe 整模是 40 层")

    reset(hs)
    reset(ts)

    rows = []
    for i in range(args.n):
        t0 = time.perf_counter()
        protocol.send(hs, protocol.FORWARD, {"tokens": [100], "n_past": i, "dtype": "float32"})
        rh = protocol.read(hs)
        t1 = time.perf_counter()
        if rh.kind != protocol.RESULT:
            print("head 出错：", rh.meta.get("detail"))
            return 3
        if len(rh.blob) != 2048 * 4:
            print(f"警告：hidden 是 {len(rh.blob)} 字节，预期 {2048 * 4}")

        protocol.send(ts, protocol.FORWARD, {"n_tokens": 1, "n_past": i, "dtype": "float32"}, rh.blob)
        rt = protocol.read(ts)
        t2 = time.perf_counter()
        if rt.kind != protocol.RESULT:
            print("tail 出错：", rt.meta.get("detail"))
            return 4
        if i >= 2:
            rows.append(((t1 - t0) * 1000, (t2 - t1) * 1000, (t2 - t0) * 1000))

    rows.sort(key=lambda r: r[2])
    n = len(rows)
    if not n:
        print("没有采到样本")
        return 5
    p50, p90 = rows[n // 2], rows[int(n * 0.9)]
    print("n=%d  p50: head=%.1f tail=%.1f total=%.1f ms -> %.2f tok/s | p90 total=%.1f ms"
          % (n, p50[0], p50[1], p50[2], 1000.0 / p50[2], p90[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
