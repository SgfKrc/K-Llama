"""统一口径的链路吞吐测量，冷态与热态分开报。

## 口径（写进输出，避免事后对不上）

    模型    Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-I-Quality
    量化    Q6_K，40 层，hidden 2048，整模 23.5 GiB
    切分    head 0:24（24 层）+ tail 24:40（16 层），两端各持完整整模
    序列    单序列
    采样    greedy（这一步只测前向，不采样）
    prefill 按 token id 预填（占位 id），非真实文本

## 冷 / 热的定义

    冷  = 两端 worker 刚加载完的**第一次** prefill 与第一次 decode。
          包含图 reserve、首次 kernel 初始化、页缓存尚未全热。
    热  = 连续跑 hot_rounds 轮之后，去掉前 DROP_ROUNDS 轮的稳态 p50。

这两个数不可互相替代：冷态决定「刚启动时第一句话多久出来」，
热态决定「连续对话时每 token 多久」。

## 用法

    runtime\\python.exe bench_chain.py --head 127.0.0.1:9103 --tail 100.x.x.x:9106

测量期间不要跑别的推理任务，否则数字没有意义。
"""

import argparse
import socket
import sys
import time
from pathlib import Path

# 仓库里本文件在 scripts/ 下、模块在上一级；包里它与 worker/ 同级。
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE.parent, _HERE, _HERE / "worker", _HERE.parent / "worker"):
    if (_cand / "protocol.py").is_file():
        REPO = _cand
        break
else:
    REPO = _HERE.parent
sys.path.insert(0, str(REPO))

import protocol  # noqa: E402

#: 热态统计丢掉前几轮：它们还在补页缓存与调度爬坡
DROP_ROUNDS = 5
#: 占位 token id。只影响数值分布，不影响前向规模——但换模型时它得在词表内。
PROBE_TOKEN = 100


def split_addr(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    if not host or not port:
        raise argparse.ArgumentTypeError(f"地址要写成 host:port，收到 {text!r}")
    return host, int(port)


def connect(addr: tuple[str, int], timeout: float = 180.0):
    sock = socket.create_connection(addr, timeout=timeout)
    hello = protocol.read(sock)
    # 握手帧是 HELLO（kind=1），不是 RESULT（kind=3）——这条曾经写错，报出来的
    # 是「握手失败」但其实 HELLO 内容完全正常。
    if hello.kind != protocol.HELLO:
        raise SystemExit(f"{addr} 握手失败：{hello.meta}")
    return sock, hello.meta


def reset(sock: socket.socket) -> None:
    protocol.send(sock, protocol.RESET)
    protocol.read(sock)


def prefill(head, tail, n: int) -> tuple[float, float, float]:
    """一次 n-token 的 prefill，返回 (head_ms, tail_ms, total_ms)。"""
    t0 = time.perf_counter()
    protocol.send(head, protocol.FORWARD,
                  {"tokens": [PROBE_TOKEN] * n, "n_past": 0, "dtype": "float32"})
    rh = protocol.read(head)
    t1 = time.perf_counter()
    if rh.kind != protocol.RESULT:
        raise SystemExit(f"head prefill 失败：{rh.meta}")
    protocol.send(tail, protocol.FORWARD,
                  {"n_tokens": n, "n_past": 0, "dtype": "float32"}, rh.blob)
    rt = protocol.read(tail)
    t2 = time.perf_counter()
    if rt.kind != protocol.RESULT:
        raise SystemExit(f"tail prefill 失败：{rt.meta}")
    return (t1 - t0) * 1000, (t2 - t1) * 1000, (t2 - t0) * 1000


def decode(head, tail, n_past: int) -> tuple[float, float, float]:
    """单 token 绕链一次，返回 (head_ms, tail_ms, total_ms)。"""
    t0 = time.perf_counter()
    protocol.send(head, protocol.FORWARD,
                  {"tokens": [PROBE_TOKEN], "n_past": n_past, "dtype": "float32"})
    rh = protocol.read(head)
    t1 = time.perf_counter()
    if rh.kind != protocol.RESULT:
        raise SystemExit(f"head decode 失败：{rh.meta}")
    protocol.send(tail, protocol.FORWARD,
                  {"n_tokens": 1, "n_past": n_past, "dtype": "float32"}, rh.blob)
    rt = protocol.read(tail)
    t2 = time.perf_counter()
    if rt.kind != protocol.RESULT:
        raise SystemExit(f"tail decode 失败：{rt.meta}")
    return (t1 - t0) * 1000, (t2 - t1) * 1000, (t2 - t0) * 1000


def pct(rows: list[tuple[float, float, float]], q: float) -> tuple[float, float, float]:
    ordered = sorted(rows, key=lambda r: r[2])
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", required=True, type=split_addr)
    ap.add_argument("--tail", required=True, type=split_addr)
    ap.add_argument("--hot-rounds", type=int, default=24, help="热态采样轮数（不含被丢掉的前几轮）")
    ap.add_argument("--prefill", type=int, default=256, help="prefill 长度（token 数）")
    args = ap.parse_args()

    head, hm = connect(args.head)
    tail, tm = connect(args.tail)

    print(f"head  {hm.get('n_layer')} 层 [{hm.get('layer_range')}]  n_embd={hm.get('n_embd')}")
    print(f"tail  {tm.get('n_layer')} 层 [{tm.get('layer_range')}]")
    print("口径：单序列 / greedy / 占位 token / prefill 按 token id 预填")
    print(f"冷 = 首次调用；热 = 连续 {args.hot_rounds} 轮，去掉前 {DROP_ROUNDS} 轮取 p50")

    # ---- 冷：第一次 prefill，然后第一次 decode ----
    reset(head); reset(tail)
    ph, pt, ptot = prefill(head, tail, args.prefill)
    print(f"\n冷 prefill {args.prefill} token："
          f"head {ph:.1f} + tail {pt:.1f} = {ptot:.1f} ms "
          f"→ {args.prefill / (ptot / 1000):.2f} tok/s")

    reset(head); reset(tail)
    ch, ct, ctot = decode(head, tail, 0)
    print(f"冷 decode 第 1 轮（含图 reserve）："
          f"head {ch:.1f} + tail {ct:.1f} = {ctot:.1f} ms "
          f"→ {1000 / ctot:.2f} tok/s")

    # ---- 热：连续 decode ----
    reset(head); reset(tail)
    rows = []
    for i in range(args.hot_rounds + DROP_ROUNDS):
        h, t, tot = decode(head, tail, i)
        rows.append((h, t, tot))
    warm = rows[DROP_ROUNDS:]
    p50 = pct(warm, 0.5)
    p90 = pct(warm, 0.9)
    print(f"\n热 decode（{len(warm)} 轮）：")
    print(f"  p50  head {p50[0]:.1f} + tail {p50[1]:.1f} = {p50[2]:.1f} ms → {1000 / p50[2]:.2f} tok/s")
    print(f"  p90  total {p90[2]:.1f} ms → {1000 / p90[2]:.2f} tok/s")
    print(f"  首轮（未丢弃）total {rows[0][2]:.1f} ms，"
          f"稳态比它快 {rows[0][2] / p50[2]:.2f}×")

    # ---- 热：连续 prefill ----
    reset(head); reset(tail)
    pfs = [prefill(head, tail, args.prefill) for _ in range(3)]
    pf = pfs[-1]
    print(f"\n热 prefill {args.prefill} token（第 3 次）："
          f"head {pf[0]:.1f} + tail {pf[1]:.1f} = {pf[2]:.1f} ms "
          f"→ {args.prefill / (pf[2] / 1000):.2f} tok/s")
    print(f"  （三次：{', '.join(f'{p[2]:.1f}' for p in pfs)} ms）")

    head.close()
    tail.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
