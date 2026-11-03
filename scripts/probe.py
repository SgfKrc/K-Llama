"""链路自检：连 head 和 tail，跑 N 轮单 token，报每段耗时。

两台机器上各起一个 run.bat 之后，在**任意一台能同时访问两者**的机器上跑：

    runtime\\python.exe probe.py --head 127.0.0.1:9103 --tail 100.x.x.x:9106

它验证三件事：两端都能握手、hidden 尺寸对得上、往返真的能出 token。

退出码：0 通过；2 连不上/等不到 HELLO；3 head 出错；4 tail 出错；5 没有样本；
6 两端身份对不上（层数、层范围或 hidden 宽度），不量 tok/s。
"""

import argparse
import socket
import sys
import time
from pathlib import Path

# protocol / wire / topology 都在仓库根，probe 住在 scripts/ 下；cwd 可能是任意目录，
# 所以按本文件位置补仓库根，而不是靠当前目录。
#
# 打包之后布局变了：probe.py 挪到包根，而 protocol.py 那几个挪进了 worker/。
# 判据用 protocol.py 落在哪，而不是看有没有 worker/ 目录——包里 worker/ 是 protocol.py
# 自己的家，不是它的上一级。
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE.parent, _HERE, _HERE / "worker"):
    if (_cand / "protocol.py").is_file():
        REPO = _cand
        break
else:
    REPO = _HERE.parent
sys.path.insert(0, str(REPO))

import numpy as np
import protocol
from topology import TOTAL_LAYERS


def split_range(text) -> tuple[int, int] | None:
    """把 HELLO 里的 "lo:hi" 拆成 (lo, hi)；没有这个字段或格式不对返回 None。"""
    lo, sep, hi = str(text or "").strip().partition(":")
    if not sep:
        return None
    try:
        return int(lo), int(hi)
    except ValueError:
        return None


def split_addr(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    if not host or not port:
        raise argparse.ArgumentTypeError(f"地址要写成 host:port，收到 {text!r}")
    return host, int(port)


def handshake(sock: socket.socket, addr: tuple[str, int]) -> dict:
    """读一帧 HELLO 并返回它的 meta；等不到就当成连不上（带地址报错）。"""
    try:
        return protocol.read(sock).meta
    except (OSError, protocol.ProtocolError) as exc:
        raise OSError(f"{addr[0]}:{addr[1]} 连上了却没发 HELLO（{exc}）") from None


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
        hm = handshake(hs, args.head)
        ts = socket.create_connection(args.tail, timeout=30)
        tm = handshake(ts, args.tail)
    except OSError as exc:
        print(f"连不上：{exc}")
        print("确认两端的 run.bat 都在跑，地址和端口填对了。")
        # 这是把 --tail 填成 head 的端口时的现场：TCP 连得上，但 worker 一次只服务一个
        # 连接（accept -> 处理到客户端断开），第二个连接只能排队，读 HELLO 必然超时。
        print("注意：一个 worker 同一时刻只服务一个连接。--head 与 --tail 填成同一个端口时，")
        print("第二个连接只是排在队列里，永远等不到 HELLO——tail 必须填 tail 自己的端口。")
        return 2

    print("head: n_layer=%s range=%s" % (hm.get("n_layer"), hm.get("layer_range")))
    print("tail: n_layer=%s range=%s" % (tm.get("n_layer"), tm.get("layer_range")))

    # 两段必须正好拼满整模，并且首尾相接；对不上就直接退出（返回 6），不往下量。
    #
    # 为什么合计层数不够也不行：每层都有人算，模型才等价于整模。少一层就是跳层，
    # 多算一层就是重复，接缝之后的 hidden 都已经不是 qwen35moe 的残差流，最后仍会
    # 输出"看起来像话"的 token，只是那是另一个模型的输出。
    hl, tl = int(hm.get("n_layer") or 0), int(tm.get("n_layer") or 0)
    if hl + tl != TOTAL_LAYERS:
        print(f"层数不对：head {hl} + tail {tl} = {hl + tl} 层，整模是 {TOTAL_LAYERS} 层。")
        print("  这意味着链路没有覆盖整模，tok/s 再好看也是在量一个错的链路。")
        print("  怎么办：核对两端的 --layers 与 LLAMA_LAYER_RANGE，以及拓扑里这两个")
        print(f"  stage 的 layers / layer_range（应分别为一段 {TOTAL_LAYERS} 层的划分）；改完重启 worker。")
        return 6

    # 为什么 hi 必须等于 lo：head 的 layer_range 是半开区间 [lo, hi)，hi 是它**没有算**
    # 的第一个源层，它算的最后一层是 hi-1；tail 必须从这一层开始，才能让 hidden 接着
    # 残差流往下走。hi != lo 时 tail 以为自己接的是第 tr[0] 层，实际拿到的是第 hr[1] 层
    # 的输入，中间缺的层被跳过或多算的层被重复，24 轮全在算错的东西。
    # 这两项判据够用：head.hi == tail.lo 且 hl + tl == TOTAL_LAYERS 时，
    # 总宽 = tail.hi - head.lo = TOTAL_LAYERS，而 0 <= head.lo < tail.hi <= TOTAL_LAYERS，
    # 所以必然 head.lo == 0、tail.hi == TOTAL_LAYERS，即恰好盖住 [0, 40)。
    hr, tr = split_range(hm.get("layer_range")), split_range(tm.get("layer_range"))
    if hr is None or tr is None:
        print("层范围读不出来：HELLO 里没有 \"lo:hi\" 形式的 layer_range，")
        print("probe 只能看到各段几层，看不到是哪几层，首尾接没接上无从判断。")
        print("  怎么办：用整模 + LLAMA_LAYER_RANGE=lo:hi 起两端（拓扑里写 layer_range），")
        print("  再跑本脚本；旧版 worker 不带这个字段。")
        return 6
    if hr[1] != tr[0]:
        print(f"层范围接不上：head 是 {hr[0]}:{hr[1]}，tail 是 {tr[0]}:{tr[1]}，"
              f"期望 tail.lo == head.hi == {hr[1]}。")
        print(f"  这意味着 tail 从第 {tr[0]} 层开始算，可 head 喂过来的 hidden 是"
              f"「前 {hr[1]} 层算完」的残差流；")
        if tr[0] > hr[1]:
            print(f"  中间第 {hr[1]}..{tr[0] - 1} 层没人算，被整段跳过。")
        else:
            print(f"  第 {tr[0]}..{hr[1] - 1} 层被 head 和 tail 各算了一遍。")
        print("  怎么办：改 tail 的 layer_range（或它的启动参数）让它从 "
              f"{hr[1]} 开始；也检查 --head/--tail 的端口有没有填反。")
        return 6

    # hidden 的宽度只能从 HELLO 的 n_embd 推：probe 的 FORWARD 声明 dtype=float32，
    # 所以一帧 hidden 就是 n_embd 个 float32。宽度不对还继续跑的话，后面每一帧都是
    # 把错宽的向量当成残差流塞给另一端，tail 只会报一个含糊的解码错误。
    n_embd = int(hm.get("n_embd") or 0)
    if n_embd <= 0:
        print("HELLO 里没有 n_embd，算不出 hidden 该多长，无法校验接缝数据。")
        print("  怎么办：确认两端跑的是同一份 checkout（旧版 worker 不报 n_embd）。")
        return 6
    want_hidden = n_embd * 4  # float32

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
        if len(rh.blob) != want_hidden:
            print(f"hidden 尺寸不对：收到 {len(rh.blob)} 字节，按 head 的 n_embd={n_embd} "
                  f"和 float32 应该是 {want_hidden} 字节。")
            print("  这意味着 head 和 tail 用的不是同一个模型（或一个是整模、一个是裁层件），"
                  "后面每一轮都会把错宽的向量当残差流。")
            print("  怎么办：核对两端的模型文件与拓扑；probe 固定按 float32 收发，"
                  "期望值就是 n_embd * 4。")
            return 6

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
