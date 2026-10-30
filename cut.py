#!/usr/bin/env python
"""cut.py — 离线裁层生成器（自包含，只依赖 gguf）。

--keep-head N 保留 blk.0..N-1（不重编号）；--k K --end E 保留 blk.K..E-1；--k K 保留
blk.K..末尾。后两者重编号为 blk.0..（工件第一层必须叫 blk.0，下游 embd 注入才对齐）。
流式搬运：源 mmap 只读映射，张量按 1 MiB 分块写，不把模型读进 RAM。
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# 每层必需张量（qwen35.cpp 的 create_tensor，已剔除 TENSOR_NOT_REQUIRED 的 wqkv /
# attn_gate 与 output.weight）。只对 qwen35 校验：其它架构清单不同，硬套会假阳性。
_B = "attn_norm.weight post_attention_norm.weight ffn_gate.weight ffn_down.weight ffn_up.weight"
_LIN = (_B + " ssm_conv1d.weight ssm_dt.bias ssm_a ssm_beta.weight ssm_alpha.weight"
        " ssm_norm.weight ssm_out.weight").split()
_FULL = (_B + " attn_q.weight attn_k.weight attn_v.weight attn_output.weight"
         " attn_q_norm.weight attn_k_norm.weight").split()
# llama.cpp 要求长度 == 层数的按层数组型 KV，裁层后必须同口径切片。
_ARR = ("attention.recurrent_layers attention.head_count attention.head_count_kv"
        " attention.key_length full_attention_interval").split()
_FALLBACK = frozenset({"qwen35", "qwen35moe"})  # 非末段可用 embedding 输出回退


def _fail(problems):
    print("\n".join(f"FAIL: {m}" for m in problems))
    return 2


def _kv(r, name):
    f = r.fields.get(name)
    return None if f is None else f.contents()


def _win(d, k, end, head):
    """(mode, lo, hi)：保留区间的源层号半开区间 —— 张量 / KV / manifest 共用同一口径。"""
    if head is not None:
        return "head", 0, int(head)
    return ("middle", int(k), int(end)) if end is not None else ("tail", int(k), d["blocks"])


class _Stream:
    """np.tofile 走 C stdio，大调用会长时间阻塞，故按 1 MiB 分块写；源是 memmap 视图，
    分块只是逐段读盘，不把整张量搬进 RAM。"""

    def __init__(self, t):
        self.t, self.shape, self.dtype, self.nbytes = t, t.shape, t.dtype, t.nbytes

    def byteswap(self, inplace=False):
        return _Stream(self.t.byteswap(inplace=inplace))

    def tofile(self, out):
        v, step = memoryview(self.t).cast("B"), 1 << 20
        for i in range(0, len(v), step):
            b = v[i:i + step]
            if out.write(b) != len(b):
                raise OSError("GGUF 输出未接收完整张量字节")


def _ident(r, src):
    """读源元数据；缺失或自相矛盾一律抛 ValueError（fail-closed）。"""
    arch, names = _kv(r, "general.architecture"), {t.name for t in r.tensors}
    blocks = _kv(r, f"{arch}.block_count") if isinstance(arch, str) and arch else None
    if type(blocks) is not int or blocks <= 0:
        raise ValueError("源缺少 general.architecture 或 block_count 非正整数")
    nextn = int(_kv(r, f"{arch}.nextn_predict_layers") or 0)
    if not 0 <= nextn < blocks:
        raise ValueError(f"源 nextn_predict_layers={nextn} 使解码层数非正")
    return {"source": str(src), "arch": arch, "blocks": blocks, "names": names,
            "nextn": nextn, "n_layer": blocks - nextn, "has_out": "output.weight" in names,
            "n_embd": int(_kv(r, f"{arch}.embedding_length") or 0),
            "iv": _kv(r, f"{arch}.full_attention_interval")}


def _check(r, d, k, end, head):
    """全部 fail-closed 校验，返回 (按层 KV 覆盖, 问题列表)。

    ★ 根因一（切点必须整数倍）：llama.cpp **不读** HF 的 `layer_types`，而是按（裁层重编号
    后的）**本地层号**对 `full_attention_interval` 取模重推层类型（qwen35.cpp:21-28）。
    所以 K、end、end-K 三者都必须是 interval 的整数倍，否则层类型错位、必然 missing
    tensor 而加载失败。
    """
    p, n, iv = [], d["n_layer"], d["iv"]
    _, lo, hi = _win(d, k, end, head)

    def mult(v, label):
        if iv and v % iv:
            p.append(f"{label}={v} 不是 full_attention_interval({iv}) 的整数倍 —— "
                     "hybrid 架构会层类型错位、无法加载")

    if head is not None:
        if not 0 < head < n:
            p.append(f"--keep-head N={head} 不在 (0, {n}) 内")
        mult(head, "--keep-head N")
    elif not 0 < k < n:
        p.append(f"K={k} 不在 (0, {n}) 内")
    if head is None:
        mult(k, "K")
        if end is not None:
            if not k < end <= n:
                p.append(f"--end={end} 必须满足 K < end <= n_layer({n})")
            mult(end, "--end")
            mult(end - k, "段内层数 end-K")

    ov = {}  # 按层数组 KV 必须同口径切片：llama.cpp 要求长度 == 层数，原样复制即失败
    for name, field in r.fields.items():
        sfx = next((s for s in _ARR if name.endswith(s)), None)
        if sfx is None:
            continue
        val, recr = field.contents(), sfx == "attention.recurrent_layers"
        if not isinstance(val, list):
            if recr:
                p.append(f"按层数组 KV {name} 不是数组 —— 源不可信")
            continue  # 标量 head / kv / key_length 每层共用，无需切
        want = n if recr else d["blocks"]
        if len(val) != want:
            p.append(f"按层数组 KV {name} 长度={len(val)} ≠ 源模型层数 {want} —— "
                     "原样复制会加载失败（源不可信）")
        else:
            ov[name] = val[lo:hi]

    if d["arch"] != "qwen35":  # 其它架构张量清单不同，硬套会假阳性
        return ov, p
    names = d["names"]
    # 门控：只在「源看起来是完整模型」时校验（判据是 blk.0 的 linear 张量齐全）；否则任何
    # 最小测试 fixture 都会被判坏源 —— 这道门极易在精简时误删。
    if not all(f"blk.0.{s}" in names for s in _LIN):
        return ov, p
    p += [f"缺少全局必需张量 {s}（qwen35.cpp 无条件 create_tensor）"
          for s in ("token_embd.weight", "output_norm.weight") if s not in names]
    # 判据与 llama.cpp 严格一致：重编号后的本地层号对 interval 取模（缺省 4，同
    # qwen35.cpp:22）；上界取 min(hi, n_layer) —— tail 的 hi 含 MTP 层，其布局是 nextn.*，
    # 套普通层模板必然误报「缺张量」。
    iv = max(1, int(iv or 4))
    miss = [f"blk.{i}.{s}（{'full' if (i - lo + 1) % iv == 0 else 'linear'} 层必需）"
            for i in range(lo, min(hi, n))
            for s in (_FULL if (i - lo + 1) % iv == 0 else _LIN) if f"blk.{i}.{s}" not in names]
    if miss:
        p.append(f"保留区间内缺少必需张量：{'、'.join(miss[:6])}"
                 + (f" 等共 {len(miss)} 个" if len(miss) > 6 else "")
                 + " —— 加载时会报 missing tensor（拒绝产出坏工件）")
    return ov, p


def _plan(r, d, k, end, head):
    """返回 (保留 [(tensor, 新名)], 丢弃名)；不写任何东西，供 --dry-run。"""
    keep, drop, _, lo, hi = [], [], *_win(d, k, end, head)
    omit = d["arch"] in _FALLBACK and hi < d["n_layer"]  # 非末段省掉 output.weight
    for t in r.tensors:
        name = t.name
        if name == "output.weight" and omit:
            drop.append(name)
        elif not name.startswith("blk."):
            keep.append((t, name))
        else:
            parts = name.split(".", 2)
            try:
                i, sfx = int(parts[1]), parts[2]
            except (IndexError, ValueError):
                raise ValueError(f"blk 张量名畸形: {name}") from None
            if not 0 <= i < d["blocks"]:
                raise ValueError(f"{name} 的层号超出 block_count={d['blocks']}")
            if lo <= i < hi:
                keep.append((t, name if head is not None else f"blk.{i - lo}.{sfx}"))
            else:
                drop.append(name)
    return keep, drop


def _write(gguf, r, d, dst, keep, ov):
    """把 KV 与张量流式写进 dst；返回 0 或失败码，失败不留半成品。"""
    w = gguf.GGUFWriter(str(dst), d["arch"])
    align = _kv(r, "general.alignment")
    if align is not None:
        w.data_alignment = int(align)  # 源所有 offset 按它对齐，改掉则整体错位
    failed = None
    try:
        for name, field in r.fields.items():
            # architecture 由 writer 构造时写入；GGUF.version / tensor_count 等由 writer
            # 维护，复制进来会造成 "Duplicate GGUF.version" 这类文件损坏。
            if name == "general.architecture" or name.startswith("GGUF."):
                continue
            kinds = list(getattr(field, "types", []) or [])
            if not kinds:
                raise ValueError(f"无法确定 KV {name} 的类型")
            w.add_key_value(name, ov.get(name, field.contents()), kinds[0],
                            kinds[-1] if kinds[0] == gguf.GGUFValueType.ARRAY else None)
        # Keys that exist only in `ov` are new, not overrides -- provenance goes here.
        # llama.cpp ignores unknown keys (verified: a 5-key artifact loads and runs), and
        # keeping it in the file means the artifact describes itself when it is copied to
        # another machine, with no sidecar to lose.
        for name, val in ov.items():
            if name in r.fields:
                continue
            if isinstance(val, str):
                w.add_string(name, val)
            elif isinstance(val, int):
                w.add_uint32(name, val)
            else:
                raise ValueError(f"新增 KV {name} 的类型不支持: {type(val).__name__}")
        for t, new in keep:
            w.add_tensor(new, _Stream(t.data), raw_dtype=t.tensor_type,
                         tensor_endianess=r.endianess)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
    except (OSError, ValueError, RuntimeError) as exc:
        failed = exc
    finally:
        w.close()
    if failed is None:
        return 0
    dst.unlink(missing_ok=True)  # 不留半成品：坏工件比没有工件更危险
    return _fail([f"写出失败: {failed}"])


def main() -> int:
    ap = argparse.ArgumentParser(description="离线裁层生成器（流式、fail-closed）")
    ap.add_argument("--src", required=True, help="源 GGUF（通常是整模）")
    ap.add_argument("--dst", help="输出裁层 GGUF")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--k", type=int, help="丢弃前 K 层（保留 blk.K.. 起）")
    g.add_argument("--keep-head", type=int, help="保留前 N 层（不重编号）")
    ap.add_argument("--end", type=int, help="与 --k 同用：只保留 blk.K..blk.(end-1)（中段）")
    ap.add_argument("--manifest", help="输出 manifest JSON 的路径")
    ap.add_argument("--dry-run", action="store_true", help="只报计划，不写任何文件")
    a = ap.parse_args()
    k, end, head, src = a.k, a.end, a.keep_head, Path(a.src)
    if not src.is_file():
        return _fail([f"源文件不存在: {src}"])
    if end is not None and k is None:
        return _fail(["--end 只能与 --k 同用（中段工件）"])
    try:
        import gguf  # noqa: PLC0415

        r = gguf.GGUFReader(str(src))
        d = _ident(r, src)
    except ImportError as exc:
        return _fail([f"需要 `gguf` 包: {exc}"])
    except (OSError, ValueError, KeyError) as exc:
        return _fail([f"源 GGUF 不可用: {exc}"])

    ov, problems = _check(r, d, k, end, head)
    try:
        keep, drop = _plan(r, d, k, end, head)
    except ValueError as exc:
        return _fail([str(exc)])
    mode, lo, hi = _win(d, k, end, head)
    detail = {"head": f"保留前 {hi - lo} 层，不重编号",
              "middle": f"保留源 blk.{lo}..blk.{hi - 1} 并重编号为 blk.0..",
              "tail": f"丢弃前 K={lo} 层，保留 blk.{lo}.. 末尾并重编号为 blk.0.."}[mode]
    print(f"[src] {src}\n      architecture={d['arch']} block_count={d['blocks']} "
          f"n_layer={d['n_layer']} nextn={d['nextn']} full_attention_interval={d['iv']}\n"
          f"[plan] mode={mode}：{detail} ⇒ 保留 {len(keep)} 张量、丢弃 {len(drop)} 张量；"
          f"block_count: {d['blocks']} -> {hi - lo}")
    if problems:
        return _fail(problems)
    if a.dry_run:
        print("[dry-run] 不写任何文件。将丢弃的 blk 张量示例（最多 8 个）：")
        print("\n".join(f"    - {n}" for n in drop[:8])
              + (f"\n    … 共 {len(drop)} 个" if len(drop) > 8 else ""))
        return 0
    if not a.dst:
        return _fail(["非 dry-run 模式需要 --dst"])
    dst = Path(a.dst)
    if not dst.parent.is_dir():
        return _fail([f"目标目录不存在: {dst.parent}"])
    if dst.exists():
        return _fail([f"目标已存在，拒绝覆盖: {dst}"])
    need, free = sum(t.n_bytes for t, _ in keep), shutil.disk_usage(dst.parent).free
    if need > free:
        return _fail([f"空间不足：需 {need / 2**20:.0f} MiB，剩 {free / 2**20:.0f} MiB"])

    ov[f"{d['arch']}.block_count"] = hi - lo
    # ★ 根因二（MTP 归零）：MTP（nextn）层是**原模型的最后一层**（block_count-1）。它一旦
    # 不在保留区间，nextn_predict_layers 就必须归 0；否则 llama.cpp 会把最后一个普通层
    # 当成 MTP 层、要求它带 nextn.* tensor ⇒ 报 missing tensor
    # 'blk.N.nextn.eh_proj.weight'，整个模型加载失败（补 hybrid 端到端验证时实测踩到）。
    mtp = d["blocks"] - 1
    if d["nextn"] and not lo <= mtp < hi:
        ov[f"{d['arch']}.nextn_predict_layers"] = 0
        print(f"[kv] MTP 层（blk.{mtp}）不在保留区间 ⇒ nextn_predict_layers: "
              f"{d['nextn']} -> 0")

    # ★ 溯源（qlh.*）：写进工件本体而不是只写 manifest JSON。manifest 是可选参数，
    # 传到别的机器常被落下；而"这段覆盖源的哪几层"是唯一能查出分段缺层/重叠的信息
    # ——少了它，20 层 + 4 层在 24 层的模型上会一路跑通，只是输出全乱。
    # 用源文件名而非完整路径：路径在别的机器上无意义。
    ov["qlh.generator"] = "cut.py/1"
    ov["qlh.source_model"] = Path(d["source"]).name
    ov["qlh.source_layers"] = int(d["n_layer"])
    ov["qlh.range_lo"] = int(lo)
    ov["qlh.range_hi"] = int(hi if hi <= d["n_layer"] else d["n_layer"])
    ov["qlh.role"] = mode

    if _write(gguf, r, d, dst, keep, ov):
        return 2
    # ★ 根因三（manifest 双口径）：顶层 block_count / n_layer / nextn_predict_layers 描述
    # **源模型**（存量 head4 工件顶层写 block_count=33 而 GGUF 内实际是 4）；
    # kept_block_count / source_layer_range / artifact_contains 描述**本段**。混写后下游
    # 再也分不清工件出自哪个模型、覆盖哪一段。
    names = {n for _, n in keep}
    tok, norm, out = ("token_embd.weight" in names, "output_norm.weight" in names,
                      "output.weight" in names)
    m = {"generator": "cut.py", "generator_version": 1, "mode": mode, "source": d["source"],
         "artifact": str(dst), "hidden_size": d["n_embd"], "architecture": d["arch"],
         "block_count": d["blocks"], "n_layer": d["n_layer"],
         "nextn_predict_layers": d["nextn"], "kept_block_count": hi - lo,
         # Size lets the validator notice that an artifact was swapped for a different one
         # while the sidecar stayed behind. Without it a stale sidecar would validate a
         # file it does not describe.
         "artifact_size": dst.stat().st_size,
         "first_local_layer_maps_to": lo, "tensors_kept": len(keep),
         "tensors_dropped": len(drop), "trim_layers": 0 if mode == "head" else lo,
         "source_layer_range": [lo, d["n_layer"] if mode == "tail" else hi],
         # 能承担末段职责：需 final_norm，且末段要么有独立 output.weight，要么 tie
         # embeddings（此时 token_embd 兼作 lm_head）。
         "artifact_contains": {
             "token_embd": tok, "output_norm": norm, "output_weight": out,
             "can_serve_tail": bool(norm and (out or (tok and not d["has_out"])))}, }
    # Sidecar is what the validator reads; the qlh.* keys in the artifact are the fallback
    # for when the sidecar gets lost in transit. Reason for two copies: reading the keys
    # back costs ~7 s per artifact, because GGUF puts tokenizer.ggml.tokens (248k strings)
    # in the header and any reader must scan past it. A sidecar is milliseconds.
    sidecar = Path(a.manifest) if a.manifest else dst.with_suffix(dst.suffix + ".qlh.json")
    sidecar.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {dst}：保留 {len(keep)} 张量、丢弃 {len(drop)}；kept_block_count={hi - lo}"
          f"；溯源 {sidecar.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
