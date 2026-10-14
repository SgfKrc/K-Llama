#!/usr/bin/env python
"""cut_layers_pipeline.py — 切层工件制作**流水线**（`#67` 缺口：把四件事串起来）

背景：此前切层是手工敲 CLI（`cut_layers.py --src … --k … --manifest …`），
"**源 sha256 / 用的命令 / 产物 sha256 / 校验结果**"没有任何一处同时留下；
而且 ③（层类型逐位）/ ⑩（按层数组 KV）/ ④（必需张量集合）三条 fail-closed 校验
虽然都实现进了 `cut_layers.py`，但**分散运行**，没人保证一次制作会把它们全跑过。

本脚本把这条链固定下来，一次调用产出三件套：

    <outdir>/<arch>-cut-<lo>-<hi>.gguf         裁层工件
    <outdir>/<arch>-cut-<lo>-<hi>.manifest.json 可复算 manifest
    <outdir>/cut-report.json                    审计报告（源 sha / 命令 / 产物 sha / 校验结果）

并在**失败时绝不留下 `cut-report.json`**（fail-closed：只有全绿才算一次成功制作）。

用法：
    python scripts/cut_layers_pipeline.py --src models/qwen35-2b-Q4_K_M.gguf --k 4 --outdir out/
    python scripts/cut_layers_pipeline.py --src full.gguf --keep-head 12 --outdir out/
    python scripts/cut_layers_pipeline.py --src full.gguf --k 8 --end 16 --outdir out/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CUT_SCRIPT = REPO_ROOT / "scripts" / "cut_layers.py"
REPORT_SCHEMA = "qlh.cut_layers.report.v1"
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")

#: ★ `#67`-⑨：llama.cpp 判定"加载失败"的标志 —— **取自源码真实的抛出点**
#: （`llama-model-loader.cpp:560` `failed to load model from`、`:1098` `missing tensor`、
#: `llama.cpp:372` `error loading model`、`gguf_init_from_file` `failed to open GGUF file`）。
#: 生成期的 ③⑩④ 只能挡"已知会坏"的源；只有这里才验证"产出的工件真能加载"。
LLAMA_LOAD_FAILURE_MARKERS = (
    "missing tensor",
    "missing tensor info mapping",
    "failed to load model",
    "error loading model",
    "failed to open gguf file",
)


def _validate_model_preflight_metadata(
    source_model_id: str | None,
    tokenizer_sha256: str | None,
) -> tuple[str, str, list[str]]:
    problems: list[str] = []
    if (source_model_id is None) != (tokenizer_sha256 is None):
        problems.append("--source-model-id 与 --tokenizer-sha256 必须成对提供")
        return source_model_id or "", tokenizer_sha256 or "", problems
    if source_model_id is None:
        return "", "", problems
    if not _MODEL_ID_PATTERN.fullmatch(source_model_id):
        problems.append(
            "--source-model-id 必须匹配 ^[A-Za-z0-9_.:-]{1,128}$"
        )
    if not _SHA256_PATTERN.fullmatch(tokenizer_sha256):
        problems.append("--tokenizer-sha256 必须是 64 位十六进制 SHA256")
    return source_model_id, tokenizer_sha256.lower(), problems


def verify_artifact_loads(artifact: Path, llama_bin, *, timeout_s: int = 300,
                          threads: int = 4) -> tuple[bool | None, str]:
    """★ `#67`-⑨：**加载自证** —— 用钉死的 llama.cpp 真的把工件加载一次。

    返回 `(ok, detail)`：
    * `ok is None` —— 没有可用的 llama.cpp 二进制 ⇒ **跳过**（不误判为成功/失败）；
    * `ok is True` —— exit=0 且输出无失败标志；
    * `ok is False` —— 退出码非 0，或输出命中 `LLAMA_LOAD_FAILURE_MARKERS`。
    """
    bin_path = Path(llama_bin) if llama_bin else None
    if not bin_path or not bin_path.is_file():
        return None, "llama.cpp 可执行文件不可用（跳过加载自证）"
    art = Path(artifact)
    if not art.is_file():
        return False, f"工件不存在: {art}"
    cmd = [str(bin_path), "-m", str(art), "-p", "hi", "-n", "1", "-t", str(int(threads))]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, f"llama.cpp 加载超时（>{timeout_s}s）"
    blob = ((proc.stdout or "") + (proc.stderr or "")).lower()
    hits = [marker for marker in LLAMA_LOAD_FAILURE_MARKERS if marker in blob]
    if proc.returncode != 0:
        return False, f"llama.cpp 退出码 {proc.returncode}（命中: {hits or '无已知失败标志'}）"
    if hits:
        return False, f"llama.cpp 输出命中失败标志: {hits}"
    return True, "llama.cpp 加载成功（exit=0 且无失败标志）"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _read_identity(src: Path) -> dict:
    """读源的关键元数据（架构 / 层数 / interval）。缺 gguf 时给出可读错误。"""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        import gguf
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"FAIL: 需要 gguf 包（{exc}）")
    reader = gguf.GGUFReader(str(src))
    fields = reader.fields
    arch = str(fields["general.architecture"].contents())
    block_count = int(fields[f"{arch}.block_count"].contents())
    nextn_field = fields.get(f"{arch}.nextn_predict_layers")
    nextn = int(nextn_field.contents()) if nextn_field is not None else 0
    interval_field = fields.get(f"{arch}.full_attention_interval")
    interval = int(interval_field.contents()) if interval_field is not None else None
    return {
        "architecture": arch,
        "block_count": block_count,
        "n_layer": max(0, block_count - nextn),
        "nextn_predict_layers": nextn,
        "full_attention_interval": interval,
    }


def _legal_cut_check(total_layers: int, cut_multiple: int, k: int | None,
                     end: int | None, keep_head: int | None) -> tuple[bool, str]:
    """切点合法性（与 `relay_cut_objective.legal_cuts` 同一判据）。"""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.relay_cut_objective import legal_cuts

    if keep_head is not None:
        if not (0 < keep_head < total_layers):
            return False, f"--keep-head {keep_head} 不在 (0, {total_layers})"
        if cut_multiple > 1 and keep_head % cut_multiple:
            return False, f"--keep-head {keep_head} 不是 {cut_multiple} 的整数倍"
        return True, ""
    if k is None:
        return False, "需要 --k 或 --keep-head"
    if end is None:
        cuts = legal_cuts(total_layers, cut_multiple=max(1, cut_multiple))
        if k not in cuts:
            return False, f"K={k} 不在合法切点 {cuts}（步长 {cut_multiple}）"
        return True, ""
    cuts = legal_cuts(total_layers, cut_multiple=max(1, cut_multiple))
    if k not in cuts:
        return False, f"K={k} 不在合法切点 {cuts}（步长 {cut_multiple}）"
    if end not in cuts and end != total_layers:
        return False, f"--end={end} 不在合法切点 {cuts}（步长 {cut_multiple}）"
    return True, ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="切层工件制作流水线（产出工件 + manifest + report）")
    ap.add_argument("--src", required=True, help="源 GGUF（通常是整模）")
    ap.add_argument("--k", type=int, default=None, help="丢弃前 K 层（保留 blk.K..）")
    ap.add_argument("--end", type=int, default=None, help="与 --k 同用：中段工件 [K,end)")
    ap.add_argument("--keep-head", type=int, default=None, help="保留前 N 层（上游段工件）")
    ap.add_argument("--outdir", required=True, help="输出目录（工件/manifest/report 都落这里）")
    ap.add_argument("--name", default=None,
                    help="产物基名（默认 `<arch>-cut-<lo>-<hi>`）；用于保持既有引用不失效")
    ap.add_argument("--hf-config", default=None,
                    help="HF config.json（可选）：额外做 #67-③ 层类型逐位校验")
    ap.add_argument("--source-model-id", default=None,
                    help="源模型稳定标识；必须与 --tokenizer-sha256 成对提供")
    ap.add_argument("--tokenizer-sha256", default=None,
                    help="tokenizer 内容 SHA256；必须与 --source-model-id 成对提供")
    ap.add_argument("--llama-bin", default=None,
                    help="★ llama.cpp 可执行文件（如 llama-debug.exe）：做 #67-⑨ **加载自证** "
                         "（真的加载一次工件，抓 missing tensor）；不给则跳过该项")
    args = ap.parse_args(argv)

    source_model_id, tokenizer_sha256, preflight_problems = (
        _validate_model_preflight_metadata(
            args.source_model_id,
            args.tokenizer_sha256,
        )
    )
    if preflight_problems:
        for problem in preflight_problems:
            print(f"FAIL: {problem}")
        return 2

    src = Path(args.src)
    if not src.is_file():
        print(f"FAIL: 源不存在: {src}")
        return 2
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    identity = _read_identity(src)
    total = int(identity["n_layer"])

    # ★ 自动读 cut_multiple（#67）：人工不再需要记住 full_attention_interval。
    from src.relay_cut_objective import cut_multiple_from_gguf  # noqa: PLC0415

    auto_multiple = cut_multiple_from_gguf(src)
    cut_multiple = auto_multiple if auto_multiple is not None else 1
    print(f"[pipeline] 源={src.name} arch={identity['architecture']} n_layer={total} "
          f"interval={identity['full_attention_interval']} => cut_multiple={cut_multiple}")

    ok, why = _legal_cut_check(total, cut_multiple, args.k, args.end, args.keep_head)
    if not ok:
        print(f"FAIL: 切点不合法 — {why}")
        return 2

    # 产物命名（默认 `-cut-<lo>-<hi>`；`--name` 可覆盖，用于保持既有引用不失效，
    # 如 `local_docs/start_tail.ps1` 写死的 `qwen25-05b-f16-mid8-24`）。
    if args.keep_head is not None:
        lo, hi = 0, int(args.keep_head)
    else:
        lo = int(args.k)
        hi = int(args.end) if args.end is not None else total
    stem = args.name or f"{identity['architecture']}-cut-{lo}-{hi}"
    dst = outdir / f"{stem}.gguf"
    manifest_path = outdir / f"{stem}.manifest.json"

    cmd = [sys.executable, str(CUT_SCRIPT), "--src", str(src), "--dst", str(dst),
           "--manifest", str(manifest_path)]
    if args.keep_head is not None:
        cmd += ["--keep-head", str(args.keep_head)]
    else:
        cmd += ["--k", str(args.k)]
        if args.end is not None:
            cmd += ["--end", str(args.end)]
    if args.hf_config:
        cmd += ["--hf-config", str(args.hf_config)]
    if source_model_id and tokenizer_sha256:
        cmd += [
            "--source-model-id", source_model_id,
            "--tokenizer-sha256", tokenizer_sha256,
        ]

    print(f"[pipeline] 执行: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    tail = (proc.stdout or "")[-800:]
    if proc.returncode != 0:
        print(f"FAIL: 切层失败 (exit={proc.returncode})\n{tail}\n{proc.stderr or ''}")
        # fail-closed：绝不在失败时留下 report（产物可能已部分写出，交由人工处理）
        return 2
    if not dst.is_file():
        print(f"FAIL: 切层报告成功但工件不存在: {dst}\n{tail}")
        return 2

    # 复算校验：manifest 对照（#67 要求的"产物 sha"闭合）
    verify = subprocess.run(
        [sys.executable, str(CUT_SCRIPT), "--src", str(dst),
         "--verify-manifest", str(manifest_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    manifest_verified = verify.returncode == 0

    # ★ `#67`-⑨：加载自证（给了 --llama-bin 才做；未给则记为 skipped）
    load_ok, load_detail = verify_artifact_loads(dst, args.llama_bin) if args.llama_bin \
        else (None, "未提供 --llama-bin（跳过加载自证）")
    llama_commit = ""
    if args.llama_bin:
        # 钉死版本（⑨ 要求"钉死 llama.cpp commit"）：从二进制所属仓库读 HEAD
        try:
            probe = subprocess.run(
                ["git", "-C", str(Path(args.llama_bin).parent), "rev-parse", "HEAD"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            if probe.returncode == 0:
                llama_commit = (probe.stdout or "").strip()
        except Exception:  # pragma: no cover
            llama_commit = ""

    report = {
        "schema": REPORT_SCHEMA,
        "source": {
            "path": str(src),
            "sha256": _sha256(src),
            **identity,
            "cut_multiple": cut_multiple,
            "cut_multiple_source": "auto(gguf)" if auto_multiple is not None else "fallback(1)",
            "source_layer_range": [lo, hi],
        },
        "command": cmd,
        "artifact": {
            "path": str(dst),
            "sha256": _sha256(dst),
            "manifest": str(manifest_path),
        },
        "checks": {
            "cut_point_legal": True,
            "manifest_verified": manifest_verified,
            # ⑨：None=跳过（未给 --llama-bin 或二进制不可用），True/False=实测结果
            "load_verified": load_ok,
            "load_detail": load_detail,
            "llama_bin": str(args.llama_bin or ""),
            "llama_commit": llama_commit,
        },
    }
    (outdir / "cut-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[pipeline] 工件={dst.name} sha256={report['artifact']['sha256'][:16]}… "
          f"manifest_verified={manifest_verified}")
    print(f"[pipeline] 加载自证: {load_detail}")
    print(f"[pipeline] report={outdir / 'cut-report.json'}")
    # fail-closed：加载自证**明确失败**（False）时以非零退出；跳过（None）不算失败。
    if load_ok is False:
        return 2
    return 0 if manifest_verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
