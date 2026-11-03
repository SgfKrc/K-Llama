#!/usr/bin/env python3
"""量同一位置"单机整模"与"层段接力"的 logits 差。

为什么需要它：非贪心投机（rejection sampling）比的是两个分布的比值，而跨机链路是量化权重 +
两段拼接。如果两边在同一位置的分布差到草稿分布的尾部质量之外，rejection sampling 会退化成
"几乎全拒"——那时这条路不值得做，但只有量过才知道。

三条判据（按严格程度递减）：

  argmax 一致          最弱。argmax 碰撞会掩盖小偏差，但它是"输出正确"的底线。
  top-k 重合度         中等。看的是候选集是否一致，比单点更能反映分布形状。
  cos / max_abs        最强。逐元素比，能看出偏差的量级。

另给两个直接对应 rejection sampling 的数字：

  TV 距离（0.5*sum|p-q|）  两个 softmax 分布的总变差距离。草稿分布 q 对目标 p 的
                            接受率上限就是 1 - TV(p,q)，所以它比 cos 更贴问题本身。
  whole argmax 在链路里的排名  0 表示整模选中的那个 token 也正是链路的 top-1。

用法：

    python checks/probe_logits_delta.py --topology topology-2b.toml
    python checks/probe_logits_delta.py --topology topology-local-32-8.toml \
        --prompts "def quicksort(arr):|The capital of France is" --steps 4 --json out.json

两边的取法必须对称：同一份 GGUF、同一个 prompt、同一批 token id、都在 greedy 路径上、
都取同一个位置（prefill 的末位，或同一段已生成前缀之后）。**增量方式也必须对称**：整模从
头 prefill 整个前缀再取 logits，与"prefill 一次后逐 token 增量"算出的是不同的浮点累加顺序
（KV 重建 vs 增量），2B 上就能造出 cos≈0.997、max_abs≈0.9 的假差异。所以两边都是
prefill 一次 + 每步追加一个 token。

★ 2026-11-03：整模侧不再走 llama-server。2026-11-01 的记录说整模单进程取不到 logits，
理由是 shim 里 `cparams.embeddings = !needs_logits`。带 `QLH_ABI_UPSTREAM_HEADER` 的构建
（`.qlh_tmp/shimout2/qlh_keep_head.dll`，175,104 字节）把 `batch->logits` 钉成 1，但实测
**那一条限制仍然成立**：同一个 ctx 上两者互斥。

    SA_NEEDS_LOGITS=0  → hidden 有（forward rc=0），qlh_kh_logits 返回 -3（该行没有 logits）
    SA_NEEDS_LOGITS=1  → logits 有（rc=248320），forward 返回 -3（没有 hidden 出口）

即使 `qlh_kh_set_skip_hidden_logits(0)` 也一样：`cparams.embeddings=1` 时 llama.cpp 根本不
建 LM head 的那段图，`llama_get_logits_ith` 返回 NULL。钉住 `batch.logits` 只解决了
`llama_get_embeddings_ith` 的行映射，没有解决两者在同一个 ctx 上互斥。

绕过它的办法是**整模侧本来就不需要 hidden**：整模只要 logits，用 `SA_NEEDS_LOGITS=1`
加载全层即可；链路 tail 也是 `SA_NEEDS_LOGITS=1`。因此两边完全对称——比较的是"分段"，不是
"角色"。这也把 llama-server 的 logprobs 路径删掉了：那条路只给 top-n（只能做前两条判据），
还要在 CPU 上再起一个进程、多加载一份权重。单进程整模给的是全词表 logits，三条判据都能做。

★ 内存顺序是硬要求：35B 的 head 16 GiB + tail 5 GiB + 整模 23.5 GiB 同时驻留会超过本机
31.6 GiB，链路 worker 会在整模加载途中被挤死（实测 WinError 10054）。所以先把链路所有位置
取完、关掉 worker，再加载整模重放。

★ 端口：共享工作区里另一个会话可能正占着拓扑里的端口，而 HELLO 只校验模型身份、不校验
"这个 worker 是我起的"，于是会静默连到别人的进程上。`--port-base` 给所有端口加偏移，用于
并行实验。

整模那一侧若改由 CPU 承担（`gpu_layers=0`）：本机 8 GiB 显存装不下 35B 权重，而 spill
到系统内存会让 kernel 选择与链路侧不同，那种差异不是"分段引入的差异"，是另一个变量。
所以 `--chain-gpu` / `--whole-gpu` 是把这条变量单独拧出来的开关：两边 kernel 布局一致时，
剩下的差才是分段引入的。
"""
from __future__ import annotations

import argparse
import ctypes
import dataclasses
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import _shimlib as shimlib  # noqa: E402
import chain_client as cc  # noqa: E402
import config  # noqa: E402
import topology  # noqa: E402

PF = ctypes.POINTER(ctypes.c_float)

#: TV 门限来自 rejection sampling 的含义，不是拍脑袋：草稿分布 q 对目标 p 的接受率上限是
#: 1 - TV(p,q)。TV=0.01 对应理想接受率 ≥ 99%，TV=0.05 对应 ≥ 95%——正好是"无条件下"与
#: "有条件上"两条线。
TV_CLEAN = 0.01
TV_MARGINAL = 0.05


def _softmax(x: np.ndarray) -> np.ndarray:
    """Stable softmax over the full vocabulary, in float64."""
    z = x.astype(np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    x = a.ravel().astype(np.float64)
    y = b.ravel().astype(np.float64)
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def _topk(values: np.ndarray, k: int) -> list[int]:
    # argpartition is enough for the comparison; the order inside the top k is irrelevant.
    idx = np.argpartition(-values, k - 1)[:k]
    return [int(i) for i in idx]


# --------------------------------------------------------------------------- 链路侧
def chain_positions(topo: str, shim: Path | None, prompt: str, n_steps: int,
                    gpu_override: int | None = None, port_base: int = 0
                    ) -> list[tuple[list[int], np.ndarray]]:
    """Preill 后逐 token 贪心前进 ``n_steps`` 步，返回每个位置的 (前缀 token, logits)。

    这是真实路径：stage worker、socket、wire dtype、tail 自己的 LM head。位置取法对称：
    第一个是 prompt 的末位，之后每个位置都是"同一段已生成前缀 + 一个 token"的末位。
    """
    stages = topology.load(topo)
    if port_base:
        stages = tuple(dataclasses.replace(s, port=s.port + int(port_base)) for s in stages)
    if gpu_override is not None:
        # 只改这一段实验：把各段的 n_gpu_layers 压成同一值，用来分离"分段引入的误差"
        # 与"GPU/CPU kernel 差异"。Stage 是 frozen dataclass，只能 replace。
        stages = tuple(dataclasses.replace(s, gpu_layers=int(gpu_override)) for s in stages)
    ctx = config.ctx()
    procs, clients = [], []
    try:
        for stage in stages:
            if stage.is_local:
                procs.append(cc.spawn_worker(stage, shim, ctx)[0])
        for i, stage in enumerate(stages):
            client = cc.StageClient(stage)
            client.connect(proc=procs[i] if stage.is_local else None)
            clients.append(client)
        head = clients[0]
        tail = clients[-1]
        tokens = head.tokenize(prompt)
        for client in clients:
            client.reset()
        stage_ms: list[list[float]] = [[] for _ in clients]
        # Prefill only: one forward through every stage, then the tail's logits.
        # 走 cc.forward 而不是手动 head->tail：中间段（topology-cross 那种三段布局）
        # 也是 relay 的一环，手动拼 head/tail 会把它跳过去。
        cc.forward(clients, tokens, 0, stage_ms)
        positions: list[tuple[list[int], np.ndarray]] = [(list(tokens), tail.logits(-1))]
        prefix = list(tokens)
        for _ in range(int(n_steps)):
            nxt = int(positions[-1][1].argmax())
            prefix.append(nxt)
            # 用链路的 argmax 决定下一个位置；两边看到的前缀因此严格相同。
            cc.forward(clients, [nxt], len(prefix) - 1, stage_ms)
            positions.append((list(prefix), tail.logits(-1)))
        return positions
    finally:
        for client in clients:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
        for proc in procs:
            proc.terminate()
            # Wait before the caller loads the whole checkpoint in-process: stage weights
            # still resident would make the next load refuse to start.
            try:
                proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                proc.kill()


# --------------------------------------------------------------------------- 整模侧
class WholeModel:
    """整份 GGUF 在一个进程里，``SA_NEEDS_LOGITS=1``：全层、token 通道、最后一行的 logits。

    ``prefill`` / ``step`` 分开是为了与链路同构：链路只 prefill 一次，之后每步只推进一个
    token；整模若每步 reset 重算整个前缀，KV 重建与增量前向的浮点累加顺序不同，量出来的
    是"计算方法差"而不是"分段差"（2B 上实测 cos≈0.997、max_abs≈0.9）。

    整模侧只要 logits，不需要 hidden：`SA_NEEDS_LOGITS=1` 时 forward 的 hidden 出口是空的
    （rc=-3），这是预期，不是失败。
    """

    def __init__(self, model: Path, shim: Path, n_layers: int, gpu: int = 0) -> None:
        os.environ["SA_BACKEND_DIR"] = str(shim.resolve().parent)
        self.lib = shimlib.load(shim, range=f"0:{n_layers}", logits=True, gpu=gpu, rs_seq=0)
        self.lib.qlh_kh_logits.argtypes = [ctypes.c_void_p, ctypes.c_int32, PF, ctypes.c_int32]
        self.lib.qlh_kh_logits.restype = ctypes.c_int32
        self.n_embd = 0
        self.n_past = 0
        n_embd, n_layer = ctypes.c_int32(0), ctypes.c_int32(0)
        err = ctypes.create_string_buffer(512)
        self.handle = self.lib.qlh_kh_load(
            str(model).encode("utf-8"), config.ctx(), 4, 512, 1, 0, 0,
            ctypes.byref(n_embd), ctypes.byref(n_layer), err, len(err))
        if not self.handle:
            raise SystemExit(f"整模加载失败: {err.value.decode('utf-8', 'replace')}")
        if int(n_layer.value) != n_layers:
            self.close()
            raise SystemExit(
                f"整模只加载了 {n_layer.value} 层，拓扑声明 {n_layers} 层——layer_range 没生效")
        self.n_embd = int(n_embd.value)
        self.n_vocab = int(self.lib.qlh_kh_n_vocab(self.handle))

    def _forward(self, tokens: list[int], n_past: int) -> int:
        scratch = np.zeros((len(tokens), self.n_embd), dtype=np.float32)
        return int(self.lib.qlh_kh_forward(
            self.handle, (ctypes.c_int32 * len(tokens))(*tokens), len(tokens), n_past,
            scratch.ctypes.data_as(PF)))

    def _last_logits(self) -> np.ndarray:
        out = np.zeros(self.n_vocab, dtype=np.float32)
        rc = int(self.lib.qlh_kh_logits(self.handle, -1, out.ctypes.data_as(PF), self.n_vocab))
        if rc <= 0:
            raise SystemExit(f"整模取 logits 失败 rc={rc}")
        return out

    def prefill(self, tokens: list[int]) -> np.ndarray:
        self.lib.qlh_kh_reset(self.handle)
        self._forward(list(tokens), 0)
        self.n_past = len(tokens)
        return self._last_logits()

    def step(self, token: int) -> np.ndarray:
        self._forward([int(token)], self.n_past)
        self.n_past += 1
        return self._last_logits()

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.qlh_kh_close(self.handle)
            self.handle = None


# --------------------------------------------------------------------------- 判据
def compare(chain: np.ndarray, whole: np.ndarray, k: int) -> dict:
    """三条判据 + 两个直接映射到 rejection sampling 的数字。"""
    a = chain.astype(np.float64)
    b = whole.astype(np.float64)
    top_c = _topk(a, k)
    top_w = _topk(b, k)
    diff = np.abs(a - b)
    pa, pb = _softmax(a), _softmax(b)
    whole_argmax = int(b.argmax())
    return {
        "chain_argmax": int(a.argmax()),
        "whole_argmax": whole_argmax,
        "argmax_equal": bool(a.argmax() == whole_argmax),
        "top_k_overlap": len(set(top_c) & set(top_w)),
        "cos": _cos(a, b),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        # 0 ⇒ 整模选中的 token 就是链路的 top-1；这个数直接读"差多少名"。
        "whole_argmax_rank_in_chain": int((a > a[whole_argmax]).sum()),
        "tv": float(0.5 * np.abs(pa - pb).sum()),
        "chain_top1_prob": float(pa.max()),
        "whole_top1_prob": float(pb.max()),
        "chain_top": [[int(i), float(a[i])] for i in top_c],
        "whole_top": [[int(i), float(b[i])] for i in top_w],
    }


def verdict(rep: dict, k: int) -> tuple[str, str]:
    """PASS / CONDITIONAL / FAIL，带一句为什么。

    不单看 argmax：argmax 相同但 TV 大，说明尾部质量偏了，rejection sampling 的接受率会掉；
    argmax 不同但 TV 很小，说明只是近 tie 翻面，仍可用。
    """
    tv = rep["tv"]
    if tv <= TV_CLEAN and rep["argmax_equal"] and rep["top_k_overlap"] == k:
        return "PASS", f"TV {tv:.5f} ≤ {TV_CLEAN}、argmax 相同、top{k} 全重合"
    if tv <= TV_MARGINAL and rep["argmax_equal"]:
        return "CONDITIONAL", (f"TV {tv:.5f} 在 ({TV_CLEAN}, {TV_MARGINAL}]、argmax 相同，"
                               f"top{k} 差 {k - rep['top_k_overlap']} 个")
    return "FAIL", (f"TV {tv:.5f}、argmax "
                    f"{'same' if rep['argmax_equal'] else 'DIFFERENT'}、"
                    f"top{k} 重合 {rep['top_k_overlap']}/{k}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="topology-2b.toml")
    ap.add_argument("--shim", default="", help="覆盖 SA_SHIM（见 config.find_shim）")
    ap.add_argument("--prompts", default="def quicksort(arr):",
                    help="多个 prompt 用 | 分隔；每个 prompt 都比 prefill 末位与后续 N 步")
    ap.add_argument("--steps", type=int, default=0,
                    help="prefill 后继续贪心前进 N 步，逐步比较同一前缀位置")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--chain-gpu", type=int, default=None,
                    help="覆盖链路各段 gpu_layers（把 kernel 变量单独拧出来）")
    ap.add_argument("--whole-gpu", type=int, default=0,
                    help="整模进程的 gpu_layers；与 --chain-gpu 一致时剩下的差才是分段差")
    ap.add_argument("--port-base", type=int, default=0,
                    help="给拓扑端口加偏移，避免撞上共享工作区里另一个会话的 worker")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    # One rule for finding the DLL: config.find_shim() (--shim > SA_SHIM > this
    # checkout's build). It also does the "does it exist" check that used to be here.
    shim = config.find_shim(args.shim)
    k = int(args.top_k)
    prompts = [p for p in args.prompts.split("|") if p]

    stages = topology.load(args.topology)
    if not all(stage.layer_range for stage in stages):
        raise SystemExit(
            "这个 probe 要求所有 stage 都指向整份 GGUF 并声明 layer_range；"
            "裁段工件没有可加载的整模，量不了同位置对照")
    model = stages[0].model
    if any(stage.model.resolve() != model.resolve() for stage in stages):
        raise SystemExit("各 stage 指向了不同的 GGUF，那本来就不是同一份权重")
    n_layers = max(int(stage.layer_range[1]) for stage in stages)

    print(f"shim    {shim}")
    print(f"model   {model.name}  layers={n_layers}")
    print(f"chain   gpu_layers={args.chain_gpu if args.chain_gpu is not None else '拓扑原值'}"
          f"  whole gpu_layers={args.whole_gpu}  steps={args.steps}", flush=True)

    # 先链路，再整模：内存放不下两者同时驻留（见文件头）。
    captured = []
    for prompt in prompts:
        positions = chain_positions(args.topology, shim, prompt, args.steps,
                                    args.chain_gpu, args.port_base)
        print(f"  chain  {prompt!r}  prompt_tokens={len(positions[0][0])}  "
              f"positions={len(positions)}", flush=True)
        captured.append((prompt, positions))

    results = []
    whole = WholeModel(model, shim, n_layers, args.whole_gpu)
    try:
        for prompt, positions in captured:
            # 整模重放同一串前缀：prefill 第一个位置，之后每个位置只追加链路选出的那个
            # token。两边因此看到完全相同的 token id 与位置。
            whole_l = whole.prefill(positions[0][0])
            for i, (prefix, chain_l) in enumerate(positions):
                if i:
                    whole_l = whole.step(prefix[-1])
                rep = compare(chain_l, whole_l, k)
                # prompt_tokens 是 prompt 本身的长度；prefix_tokens 是当前比较位置之前
                # 的完整前缀长度（position 步之后）。两个数混用会把"第几步"读成"prompt 多长"。
                rep.update({"prompt": prompt,
                            "prompt_tokens": len(positions[0][0]),
                            "prefix_tokens": len(prefix),
                            "position": i})
                name, why = verdict(rep, k)
                rep["verdict"], rep["why"] = name, why
                results.append(rep)
                print(f"\n[{name}] {prompt!r}  pos={i}  tokens={len(prefix)}")
                print(f"  argmax  chain={rep['chain_argmax']} whole={rep['whole_argmax']} "
                      f"{'same' if rep['argmax_equal'] else 'DIFFERENT'}")
                print(f"  top{k}    {rep['top_k_overlap']}/{k}")
                print(f"  cos={rep['cos']:.7f}  max_abs={rep['max_abs']:.4g}  "
                      f"mean_abs={rep['mean_abs']:.4g}")
                print(f"  TV={rep['tv']:.6f}  →  ideal accept rate ≥ {1 - rep['tv']:.4f}")
                print(f"  whole_argmax_rank_in_chain={rep['whole_argmax_rank_in_chain']}  "
                      f"top1 prob chain={rep['chain_top1_prob']:.4f} "
                      f"whole={rep['whole_top1_prob']:.4f}")
                print(f"  {why}")
    finally:
        whole.close()

    worst_tv = max(rep["tv"] for rep in results)
    worst_cos = min(rep["cos"] for rep in results)
    argmax_all = all(rep["argmax_equal"] for rep in results)
    top_overlap = min(rep["top_k_overlap"] for rep in results)
    names = {rep["verdict"] for rep in results}
    if "FAIL" in names:
        overall, tail = "FAIL", "不能上（分布差超出 TV 门限）"
    elif "CONDITIONAL" in names:
        overall, tail = "CONDITIONAL", "有条件上（分布差在门限内但非逐位一致）"
    else:
        overall, tail = "PASS", "可以上（逐位一致，无条件下）"
    print(f"\n汇总  positions={len(results)}  argmax_same={argmax_all}  "
          f"min_top{k}={top_overlap}/{k}  min_cos={worst_cos:.7f}  worst_tv={worst_tv:.6f}")
    if args.json:
        Path(args.json).write_text(
            json.dumps({"shim": str(shim), "model": str(model),
                        "summary": {"positions": len(results), "verdict": overall,
                                    "argmax_same": argmax_all,
                                    "min_top_k": top_overlap, "min_cos": worst_cos,
                                    "worst_tv": worst_tv},
                        "results": results},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"written {args.json}")
    print(f"\n{overall}  判定：{tail}")
    return 0 if overall != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
