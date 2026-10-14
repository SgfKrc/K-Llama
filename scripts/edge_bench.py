#!/usr/bin/env python
"""edge_bench.py — 边缘段标定与工件 metadata 校验（票 19 / EDGE-BENCH-01）。

背景（为什么需要它）：2026-10-14 的四轮 Surface 标定全部是**临时脚本**，跑完即弃；
其中还踩到一个真坑 —— `qwen25-05b-f16-cut-16.gguf` 的 `block_count` 实为 8（而非 16），
按 16 层归一化得出 `3815 ms/token` 的**假异常**。所以本脚本做两件事：

1. **工件 metadata 校验（fail-closed）**：把 GGUF 头里的 `block_count` / `head_count_kv` /
   `key_length` / `full_attention_interval` 与 manifest 声明（或 `--expect` 显式期望）比对，
   不一致即**具名报出差异并返回非零**，不静默继续。
2. **边缘段标定**：给定工件与一组 `n_threads` / `n_ctx`，输出 `decode tok/s`、`ms/token`
   与**每层 ms**，口径与 `docs/已知问题记录.md` #71 的 `c_D` 可比。

报告固定写 `--json-out`（必填），并在记录里写明**测量口径**（prompt 长度、n_ctx、线程数、是否 warmup）。

用法::

    # 校验（fail-closed）
    python scripts/edge_bench.py --model models/x.gguf --verify-manifest m.json

    # 标定
    python scripts/edge_bench.py --model models/x.gguf --threads 4,8 --ctx 512,4096 \
        --layers 4 --gen 16 --json-out build/edge-bench/x.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

#: 需要与 manifest / --expect 对齐的 GGUF 元数据键（按架构前缀尝试）。
_META_KEYS = (
    ("block_count", ("block_count",)),
    ("head_count_kv", ("attention.head_count_kv",)),
    ("key_length", ("attention.key_length",)),
    ("full_attention_interval", ("full_attention_interval",)),
)


def _load_gguf_meta(path: Path) -> dict:
    """从 GGUF 头读关键元数据（架构无关：按 general.architecture 拼前缀）。"""
    from llama_cpp import llama_cpp as lc

    params = lc.llama_model_default_params()
    model = lc.llama_model_load_from_file(str(path).encode("utf-8"), params)
    if not model:
        raise SystemExit(f"FAIL: 无法加载工件 {path}")

    def meta(key: str):
        import ctypes

        buf = ctypes.create_string_buffer(256)
        n = lc.llama_model_meta_val_str(model, key.encode("utf-8"), buf, len(buf))
        return buf.value.decode("utf-8", "replace") if n > 0 else None

    try:
        arch = meta("general.architecture")
        out = {"architecture": arch, "path": str(path), "bytes": path.stat().st_size}
        for name, suffixes in _META_KEYS:
            value = None
            for suffix in suffixes:
                for candidate in (f"{arch}.{suffix}", suffix):
                    got = meta(candidate)
                    if got is not None:
                        value = got
                        break
                if value is not None:
                    break
            if value is not None:
                try:
                    value = int(value)
                except ValueError:
                    pass
            out[name] = value
        return out
    finally:
        lc.llama_model_free(model)


def _verify(meta: dict, expect: dict, source: str) -> list[str]:
    """比对期望值；返回差异列表（空 = 通过）。只比对**双方都提供**的项。"""
    diffs: list[str] = []
    for name, _ in _META_KEYS:
        actual = meta.get(name)
        wanted = expect.get(name)
        if actual is None or wanted is None:
            continue
        if int(actual) != int(wanted):
            diffs.append(f"{name}: 工件={actual} vs {source}={wanted}")
    return diffs


def _bench_one(model_path: Path, *, n_ctx: int, n_threads: int, n_layers: int,
               gen: int, warmup: int, prompt_tokens: int) -> dict:
    from llama_cpp import Llama

    base = "The quick brown fox jumps over the lazy dog. "
    prompt = base * max(4, prompt_tokens // 10)

    llm = Llama(model_path=str(model_path), n_ctx=n_ctx, n_batch=512,
                n_threads=n_threads, seed=1234, verbose=False)
    for _ in range(max(0, warmup)):
        llm(prompt, max_tokens=2, temperature=0.0)

    samples: list[float] = []
    for _ in range(3):
        t0 = time.perf_counter()
        out = llm(prompt, max_tokens=gen, temperature=0.0)
        dt = time.perf_counter() - t0
        ct = out.get("usage", {}).get("completion_tokens", 0)
        if ct:
            samples.append(dt / ct * 1000.0)
    del llm

    ms_per_token = statistics.mean(samples) if samples else float("nan")
    return {
        "n_ctx": n_ctx,
        "n_threads": n_threads,
        "decode_ms_per_token": round(ms_per_token, 3),
        "decode_tokens_per_s": round(1000.0 / ms_per_token, 3) if ms_per_token else None,
        "ms_per_layer": round(ms_per_token / n_layers, 3) if n_layers else None,
        "samples": [round(x, 3) for x in samples],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="边缘段标定与工件 metadata 校验（票 19）")
    ap.add_argument("--model", required=True, help="GGUF 工件")
    ap.add_argument("--verify-manifest", default=None,
                    help="校验模式：对照该 manifest 检查 --model 的元数据（fail-closed）")
    ap.add_argument("--expect", default=None,
                    help="校验模式：显式期望，形如 block_count=8,head_count_kv=2")
    ap.add_argument("--layers", type=int, default=0,
                    help="该工件实际持层的层数（用于换算每层 ms；0 = 不换算）")
    ap.add_argument("--threads", default="8", help="逗号分隔：4,8")
    ap.add_argument("--ctx", default="512", help="逗号分隔：512,2048,4096")
    ap.add_argument("--gen", type=int, default=16, help="每次计时生成的 token 数")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--prompt-tokens", type=int, default=20)
    ap.add_argument("--json-out", required=True, help="记录输出路径（必填，固定落点）")
    args = ap.parse_args(argv)

    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"FAIL: 工件不存在：{model_path}")

    meta = _load_gguf_meta(model_path)
    print("[meta] %s" % {k: v for k, v in meta.items() if k.endswith("_count") or
                         k in ("architecture", "head_count_kv", "key_length",
                               "full_attention_interval")})

    # ---- 模式 1：校验（fail-closed）
    if args.verify_manifest or args.expect:
        expect: dict = {}
        source = ""
        if args.verify_manifest:
            manifest = json.loads(Path(args.verify_manifest).read_text(encoding="utf-8"))
            source = os.path.basename(args.verify_manifest)
            # manifest 的字段名与 GGUF 键不一定同名：把已知别名都收进来
            for name, _ in _META_KEYS:
                if name in manifest:
                    expect[name] = manifest[name]
            # 裁层件的特例：manifest 的 kept_block_count 就是工件的 block_count
            if "kept_block_count" in manifest:
                expect["block_count"] = manifest["kept_block_count"]
        if args.expect:
            source = source or "--expect"
            for item in args.expect.split(","):
                if "=" not in item:
                    continue
                k, v = item.split("=", 1)
                expect[k.strip()] = int(v.strip())

        diffs = _verify(meta, expect, source or "expect")
        record = {
            "mode": "verify",
            "artifact": str(model_path),
            "manifest": args.verify_manifest,
            "expected": expect,
            "actual": meta,
            "differences": diffs,
            # ★ fail-closed 的两层：①有差异 ⇒ 不通过；②压根没有可比对项 ⇒ 也不能算通过，
            #   否则「声明为空」会把这个门退化成永远通过。
            "passed": bool(not diffs and expect),
            "comparable_items": sorted(expect),
        }
        _write(args.json_out, record)
        if diffs:
            for d in diffs:
                print("[FAIL] %s" % d)
            print("[verdict] 工件元数据与声明不一致 —— fail-closed")
            return 1
        if not expect:
            print("[WARN] 没有任何可比对项（manifest 未声明 / 未给 --expect）⇒ 不算通过")
            return 1
        print("[verdict] 工件元数据与声明一致（比对 %d 项）" % len(expect))
        return 0

    # ---- 模式 2：标定
    threads = [int(x) for x in str(args.threads).split(",") if x.strip()]
    ctxs = [int(x) for x in str(args.ctx).split(",") if x.strip()]
    if not threads or not ctxs:
        raise SystemExit("FAIL: --threads / --ctx 不能为空")

    rows = []
    for n_ctx in ctxs:
        for n_threads in threads:
            row = _bench_one(model_path, n_ctx=n_ctx, n_threads=n_threads,
                             n_layers=args.layers, gen=args.gen,
                             warmup=args.warmup, prompt_tokens=args.prompt_tokens)
            rows.append(row)
            print("[bench] n_ctx=%-5d threads=%d  %8.3f ms/token  %7.2f tok/s  每层 %s ms"
                  % (row["n_ctx"], row["n_threads"], row["decode_ms_per_token"],
                     row["decode_tokens_per_s"] or 0,
                     row["ms_per_layer"] if row["ms_per_layer"] is not None else "-"))

    record = {
        "mode": "bench",
        "artifact": str(model_path),
        "meta": meta,
        "layers": args.layers or None,
        # ★ 测量口径必须写进记录，否则数据不可比（票 19 完成定义）
        "protocol": {
            "gen_tokens": args.gen,
            "warmup_runs": args.warmup,
            "prompt_tokens_target": args.prompt_tokens,
            "samples_per_cell": 3,
            "criterion": "decode ms/token（生成 token 的墙钟均值）",
            "comparable_to": "docs/已知问题记录.md #71 的 c_D（每层 ms）",
        },
        "rows": rows,
    }
    _write(args.json_out, record)
    print("[record] %s" % args.json_out)
    return 0


def _write(path: str, record: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
