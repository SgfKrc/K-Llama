#!/usr/bin/env python3
"""量同一位置"单机整模"与"层段接力"的 logits 差。

为什么需要它：非贪心投机（rejection sampling）比的是两个分布的比值，而跨机链路是量化权重 +
两段拼接。如果两边在同一位置的分布差到草稿分布的尾部质量之外，rejection sampling 会退化成
"几乎全拒"——那时这条路不值得做，但只有量过才知道。

三条判据（按严格程度递减）：

  argmax 一致          最弱。argmax 碰撞会掩盖小偏差，但它是"输出正确"的底线。
  top-k 重合度         中等。看的是候选集是否一致，比单点更能反映分布形状。
  cos / max_abs        最强。逐元素比，能看出偏差的量级。

用法：

    python checks/probe_logits_delta.py --topology topology-local-32-8.toml
    python checks/probe_logits_delta.py --topology topology-local-32-8.toml --json out.json

两边的取法必须对称：同一份 GGUF、同一个 prompt、同一批 token id、都在 greedy 路径上、
都取最后一个位置的 logits。不然量的是两件事。

**已知限制（2026-11-01 实测）**：整模单进程那一侧目前取不到 logits。shim 里
`cparams.embeddings = !needs_logits`，所以"要 hidden"与"要 logits"是互斥的：

- `SA_NEEDS_LOGITS=1` → 有 logits，但 `forward_tokens_to_hidden` 返回 rc=-3
  （`llama_get_embeddings_layer_inp returned nothing`）
- `SA_NEEDS_LOGITS=0` → 有 hidden，但 `batch.logits` 一行都不标，`llama_get_logits_ith`
  返回 NULL

链路侧没有这个问题：tail 段开着 logits 出 token，`logits()` 取的是同一份，不冲突。

要补上整模侧的对比，正确做法是走 `llama-server` 的 logprobs（`checks/verify_cli.py` 已经有
起服务的代码），代价是只能比 top-k 与 argmax，而不是完整分布。在那之前，本脚本的整模侧会
明确失败并说明原因，不返回一个假数字。

整模那一侧若改由 CPU 承担（`gpu_layers=0`）：本机 8 GiB 显装不下 14.79 GiB 权重，而 spill
到系统内存会让 kernel 选择与链路侧不同，那种差异不是"分段引入的差异"，是另一个变量。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

import chain_client as cc  # noqa: E402
import config  # noqa: E402
import topology  # noqa: E402

MODEL = Path(r"H:\qlh_models\qwen3-6-35b-a3b-opus47-distill-apex-mtp"
             r"\Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-I-Quality.gguf")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _topk(values: np.ndarray, k: int) -> list[int]:
    # argpartition is enough for the comparison; the order inside the top k is irrelevant.
    idx = np.argpartition(-values, k - 1)[:k]
    return [int(i) for i in idx]


def chain_logits(topo: str, shim: Path, prompt: str) -> tuple[np.ndarray, list[int]]:
    """Walk the chain once and read the tail's logits for the last prompt position."""
    stages = topology.load(topo)
    ctx = config.ctx()
    procs = []
    clients = []
    try:
        for stage in stages:
            if stage.is_local:
                procs.append(cc.spawn_worker(stage, shim, ctx)[0])
        for stage in stages:
            client = cc.StageClient(stage)
            client.connect(proc=procs[len(clients)] if stage.is_local else None)
            clients.append(client)
        tokens = clients[0].tokenize(prompt)
        for client in clients:
            client.reset()
        # Prefill only: one forward through every stage, then the tail's logits.
        cc.forward(clients, tokens, 0, [[] for _ in clients])
        logits = clients[-1].logits(-1)
        return logits, tokens
    finally:
        for client in clients:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
        for proc in procs:
            proc.terminate()
            # Wait before the caller loads the whole checkpoint in-process: 17.5 GiB of
            # stage weights still resident would make the next load refuse to start.
            try:
                proc.wait(timeout=20)
            except Exception:  # noqa: BLE001
                proc.kill()


BIN = Path(r"H:\llamacpp-c\build-cuda\bin")

#: llama-server's /completion returns logprobs as a top-n list, not a full vocabulary, so
#: this probe compares candidate sets rather than every logit. That is the right granularity
#: for rejection sampling anyway: it accepts or rejects a *draft token*, and what matters is
#: where that token ranks in the target distribution, not the exact tail values.
def server_topk(prompt: str, top_k: int, port: int = 8199) -> list[tuple[int, float]]:
    """The whole model's top-k next-token distribution, via llama-server.

    The shim cannot give this: `cparams.embeddings = !needs_logits` means a single process
    either has hidden states or has logits, never both, and the whole-model reference needs
    the logits. llama-server is a separate process holding the same checkpoint, so it does.
    """
    import subprocess
    import time as _time
    import urllib.request

    exe = BIN / "llama-server.exe"
    if not exe.is_file():
        raise SystemExit(f"llama-server not found: {exe}")
    logdir = Path(".qlh_tmp")
    logdir.mkdir(exist_ok=True)
    handle = open(logdir / "probe-logits-server.log", "w", encoding="utf-8", errors="replace")
    env = dict(os.environ)
    env["PATH"] = str(BIN) + os.pathsep + env.get("PATH", "")
    proc = subprocess.Popen(
        [str(exe), "-m", str(MODEL), "-c", "512", "-ngl", "0", "-t", "8",
         "--host", "127.0.0.1", "--port", str(port), "--no-webui"],
        stdout=handle, stderr=subprocess.STDOUT, cwd=str(BIN))
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = _time.time() + 300
        while _time.time() < deadline:
            if proc.poll() is not None:
                raise SystemExit("llama-server died, see .qlh_tmp/probe-logits-server.log")
            try:
                with urllib.request.urlopen(base + "/health", timeout=3) as r:
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001 - not up yet
                _time.sleep(1.0)
        else:
            raise SystemExit("llama-server never became healthy")

        body = json.dumps({
            "prompt": prompt, "n_predict": 1, "temperature": 0.0,
            "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repeat_last_n": 0,
            "repeat_penalty": 1.0, "cache_prompt": False, "stream": False,
            "n_probs": int(top_k),
        }).encode()
        req = urllib.request.Request(base + "/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            data = json.loads(r.read())
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            proc.kill()
        handle.close()

    entry = (data.get("completion_probabilities") or [{}])[0]
    # The field is `top_logprobs` (verified against this build's /completion), and the value
    # is a natural log of the probability. The relay side hands back raw logits, which are
    # not normalised, so the two cannot be compared as numbers -- only as candidate sets and
    # ranking. That is the comparison this probe makes.
    probs = entry.get("top_logprobs") or []
    if not probs:
        raise SystemExit("llama-server returned no top_logprobs "
                         "(is this build's /completion missing n_probs?)")
    return [(int(p["id"]), float(p.get("logprob", 0.0))) for p in probs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="topology-local-32-8.toml")
    ap.add_argument("--shim", default="", help="覆盖 SA_SHIM（见 config.find_shim）")
    ap.add_argument("--prompt", default="def quicksort(arr):")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    # One rule for finding the DLL: config.find_shim() (--shim > SA_SHIM > this
    # checkout's build). It also does the "does it exist" check that used to be here.
    shim = config.find_shim(args.shim)
    k = int(args.top_k)

    print(f"prompt  {args.prompt!r}")
    print("chain   ... (two-stage relay)", flush=True)
    chained, tokens = chain_logits(args.topology, shim, args.prompt)
    chain_top = _topk(chained, k)
    print(f"        n_vocab={chained.size}  prompt_tokens={len(tokens)}")

    print("whole   ... (llama-server, CPU; this loads the full checkpoint)", flush=True)
    whole = server_topk(args.prompt, k, args.port)
    whole_ids = [tid for tid, _ in whole]
    print(f"        top{k} returned")

    overlap = len(set(chain_top) & set(whole_ids))
    chain_argmax = int(chained.argmax())
    whole_argmax = whole_ids[0]

    # Rank of the whole model's argmax inside the chain's own ordering. Rejection sampling
    # accepts a draft token when the target would have produced it too, so what matters is
    # whether the target's pick is near the top of the relay's distribution -- not the exact
    # tail values.
    rank = int((chained > chained[whole_argmax]).sum())

    out = {
        "prompt": args.prompt,
        "prompt_tokens": len(tokens),
        "n_vocab": int(chained.size),
        "top_k": k,
        "chain_argmax": chain_argmax,
        "whole_argmax": whole_argmax,
        "argmax_equal": bool(chain_argmax == whole_argmax),
        "top_k_overlap": int(overlap),
        "whole_argmax_rank_in_chain": rank,
        "chain_top": [[int(i), float(chained[i])] for i in chain_top],
        "whole_top": [[int(i), float(p)] for i, p in whole],
    }
    print(json.dumps({kk: vv for kk, vv in out.items()
                      if kk not in ("chain_top", "whole_top")}, indent=2, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"written {args.json}")

    # The judgement that decides whether rejection sampling is worth building:
    #   argmax equal  -> the relay would have produced the same token anyway
    #   rank <= 1     -> the target's pick is in the relay's top 2, so a draft has a real
    #                    chance of being accepted even when the two differ
    ok = out["argmax_equal"] or rank <= 1
    print("\n" + ("PASS" if ok else "FAIL")
          + f"  argmax={'same' if out['argmax_equal'] else 'DIFFERENT'}"
          + f"  top{k}_overlap={overlap}/{k}"
          + f"  whole_argmax_rank_in_chain={rank}")
    if not ok:
        print("  -> the two distributions disagree beyond the top 2; rejection sampling "
              "would reject nearly everything. Record this and stop.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
