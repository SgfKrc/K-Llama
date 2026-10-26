"""离线数值门：比对「本机 CPU 段（llama.cpp GGUF）」与「同层段（PyTorch fp16）」的逐层输出。

★ 2026-10-26（LOCAL-EXECUTORS-01）验收判据 4。

## 为什么需要这个脚本

端到端 token 对比做不到"控制变量"：关掉 CPU 段会连带改变层段划分与供给拓扑，
于是"数值差异"和"拓扑差异"混在一起。本脚本把两者拆开 —— 固定同一份 hidden 输入，
只换**执行器**，逐元素比输出。

## 精度前提（必须明确）

CPU 段的工件是 **Q4_K_M 量化的 GGUF**（`q35-9b-local-mid8-24.gguf`），而 PyTorch 侧是
**fp16 safetensors**。两者**不可能逐位一致**。所以本脚本报的是：
  * 最大绝对误差 / 均方根误差
  * 余弦相似度（逐 token、整体）
  * **argmax 一致率**（这才是决定生成 token 的那个量）
判据：argmax 一致率足够高 ⇒ 量化段可以替代 fp16 段承担这 16 层。

## 用法

    python scripts/probe_local_cpu_segment_numerics.py [--start 8] [--end 24]
                                                      [--prompt "2+3=?"]
                                                      [--max-tokens 8]
                                                      [--json out.json]

前置：master 服务**建议停止**（PyTorch 那 16 层约需 9.2 GB 主机内存）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


def _stats(name: str, a, b) -> dict:
    """两批 hidden（同 shape）的差异统计。"""
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return {"name": name, "error": f"shape mismatch {a.shape} != {b.shape}"}
    diff = a - b
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    denom = float(np.linalg.norm(flat_a) * np.linalg.norm(flat_b))
    cosine = float(flat_a @ flat_b / denom) if denom else 0.0
    # 逐 token 的 argmax 一致率：这是"最终会影响生成 token"的量。
    if a.ndim == 2:
        same = int((a.argmax(axis=1) == b.argmax(axis=1)).sum())
        total = int(a.shape[0])
    else:
        same, total = 0, 0
    return {
        "name": name,
        "shape": list(a.shape),
        "max_abs": float(np.abs(diff).max()),
        "mean_abs": float(np.abs(diff).mean()),
        "rmse": float(math.sqrt(float((diff ** 2).mean()))),
        "cosine": cosine,
        "argmax_match": same,
        "argmax_total": total,
        "argmax_match_ratio": (same / total) if total else 0.0,
        "ref_norm": float(np.linalg.norm(flat_a)),
        "rel_l2": float(np.linalg.norm(diff) / np.linalg.norm(flat_a))
        if np.linalg.norm(flat_a) else 0.0,
    }


def _build_hidden(start: int, prompt: str):
    """用真实 prompt 跑 [0, start) 得到本段的输入 hidden（真实分布，非随机）。"""
    from model_module import ModelManager

    manager = ModelManager()
    if start > 0:
        manager.load_layer_range(
            0, start, has_embedding=True, has_lm_head=False,
            model_path=str(_ROOT / "models" / "qwen3-5-9b"),
        )
    else:
        manager.load_layer_range(
            0, start, has_embedding=True, has_lm_head=False,
            model_path=str(_ROOT / "models" / "qwen3-5-9b"),
        )
    tokenizer = manager.tokenizer
    # 用与流水线一致的 chat template 渲染，保证 hidden 分布接近真机。
    try:
        from api_server import _build_chat_prompt

        text = _build_chat_prompt([{"role": "user", "content": prompt}])
    except Exception:
        text = prompt
    encoded = tokenizer(text, return_tensors="pt")
    with __import__("torch").no_grad():
        result = manager.forward_layers(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            use_cache=True,
            apply_lm_head=False,
        )
    hidden = result.get("hidden_states")
    if hidden is None:
        raise RuntimeError("prefix 段没有返回 hidden_states")
    if hasattr(hidden, "detach"):
        hidden = hidden.detach().float().cpu().numpy()
    import numpy as np

    hidden = np.ascontiguousarray(hidden, dtype=np.float32)
    if hidden.ndim == 3 and hidden.shape[0] == 1:
        hidden = hidden[0]
    if hidden.ndim != 2:
        raise RuntimeError(f"hidden 形状异常: {hidden.shape}")
    manager.unload_model()
    return hidden


def _run_pytorch(start: int, end: int, hidden):
    """PyTorch 侧：只物化 [start, end)，喂 hidden，取 hidden。"""
    import numpy as np
    import torch

    from model_module import ModelManager

    manager = ModelManager()
    manager.load_layer_range(
        start, end, has_embedding=False, has_lm_head=False,
        model_path=str(_ROOT / "models" / "qwen3-5-9b"),
    )
    tensor = torch.from_numpy(np.asarray(hidden, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        result = manager.forward_layers(
            hidden_states=tensor, use_cache=True, apply_lm_head=False,
        )
    out = result.get("hidden_states")
    if out is None:
        raise RuntimeError("PyTorch 段没有返回 hidden_states")
    if hasattr(out, "detach"):
        out = out.detach().float().cpu().numpy()
    out = np.ascontiguousarray(out, dtype=np.float32)
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]
    manager.unload_model()
    return out


def _run_llama_cpp(hidden, *, n_ctx: int, n_batch: int, n_threads: int):
    """llama.cpp 侧：keep-head 上游（GGUF Q4_K_M）吃 hidden、吐 hidden。"""
    from keep_head_shim import resolve_keep_head_shim
    from llama_keep_head import KeepHeadUpstream

    shim = resolve_keep_head_shim()
    if not shim:
        raise RuntimeError("找不到 keep-head shim（QLH_KEEP_HEAD_SHIM）")
    model = os.environ.get("QLH_LAYER_GGUF") or ""
    if not model:
        raise RuntimeError("未设置 QLH_LAYER_GGUF")
    path = Path(model)
    if not path.is_absolute():
        path = _ROOT / path
    if not path.is_file():
        raise RuntimeError(f"GGUF 工件不存在: {path}")
    upstream = KeepHeadUpstream(
        shim, str(path), mode="nextn",
        n_ctx=n_ctx, n_batch=max(n_batch, len(hidden)), n_threads=n_threads,
        n_seq_max=1,
    )
    try:
        out = upstream.forward_hidden_to_hidden(
            hidden, n_past=0, positions=list(range(len(hidden))),
        )
    finally:
        upstream.close()
    import numpy as np

    out = np.ascontiguousarray(out, dtype=np.float32)
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=8)
    parser.add_argument("--end", type=int, default=24)
    parser.add_argument("--prompt", default="2+3=?")
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-batch", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    print(f"[1/4] 构造真实 hidden：prompt={args.prompt!r} 跑 [0,{args.start}) …")
    hidden = _build_hidden(args.start, args.prompt)
    print(f"      hidden shape = {hidden.shape}  dtype={hidden.dtype}")

    print(f"[2/4] PyTorch 侧：只物化 [{args.start},{args.end}) …")
    out_torch = _run_pytorch(args.start, args.end, hidden)
    print(f"      out shape = {out_torch.shape}")

    print("[3/4] llama.cpp 侧：keep-head 上游（GGUF Q4_K_M）…")
    out_llama = _run_llama_cpp(
        hidden, n_ctx=args.n_ctx, n_batch=args.n_batch, n_threads=args.n_threads,
    )
    print(f"      out shape = {out_llama.shape}")

    print("[4/4] 对比 …")
    reports = [
        _stats("input_hidden", hidden, hidden),  # 自比，确认统计口径
        _stats(f"layers[{args.start},{args.end}) pytorch_vs_llamacpp", out_torch, out_llama),
    ]
    payload = {
        "start": args.start, "end": args.end,
        "prompt": args.prompt,
        "layers": args.end - args.start,
        "tokens": int(hidden.shape[0]),
        "reports": reports,
    }
    print()
    for item in reports:
        print(f"--- {item['name']} ---")
        if "error" in item:
            print(f"    {item['error']}")
            continue
        for key in ("max_abs", "mean_abs", "rmse", "cosine", "rel_l2",
                    "argmax_match", "argmax_total", "argmax_match_ratio"):
            print(f"    {key:>18} = {item[key]}")
    if args.json:
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"\n已写 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
