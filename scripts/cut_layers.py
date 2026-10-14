#!/usr/bin/env python
"""cut_layers.py — 离线裁层生成器（主仓正式版）

用途：为「上游跑前 K 层、下游跑其余层」的接力造**下游 GGUF 工件**。
主仓此前只有裁层的**合同与校验**（``src/relay_contract.py`` 的 ``RelayTrimPlan`` /
``RelayHandoff``，以及 ``src/pipeline_node_contract.py`` 的节点校验），没有生成器；
本脚本把实验区的 ``cut_layers_generic.py`` 正规化进来，并与合同对齐。

裁层语义（与 ``relay_contract.RelayTrimPlan`` 一致）
  * 保留 ``blk.K..blk.(N-1)``，并**重命名**为 ``blk.0..blk.(N-1-K)``；
  * ``<arch>.block_count`` 由 ``N`` 改写为 ``N-K``；
  * 非 blk 张量（``token_embd`` / ``output_norm`` / ``output`` 等）**原样保留**；
  * 其余 KV 字段全部复制（``general.architecture`` 与 ``GGUF.*`` 由 writer 维护，不复制）。

约束（2026-09-18 实测）
  * ``0 < K < n_layer``；
  * **K 必须是 ``<arch>.full_attention_interval`` 的整数倍** —— 否则 hybrid 架构（如 Qwen3.5）
    的层类型会错位，llama.cpp 报 ``missing tensor 'blk.x.<...>'`` 而无法加载。

用法
  # 预览（不改任何文件）
  python scripts/cut_layers.py --src full.gguf --k 12 --dry-run

  # 生成裁层工件 + manifest
  python scripts/cut_layers.py --src full.gguf --dst cut.gguf --k 12 \
      --manifest cut.json

  # 校验一个已生成的裁层工件（对照其 manifest）
  python scripts/cut_layers.py --src cut.gguf --verify-manifest cut.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


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


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _require_gguf():
    try:
        import gguf  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise SystemExit(
            "需要 `gguf` 包（llama.cpp 的 Python 绑定之一）。\n"
            "  安装示例：pip install gguf\n"
            f"  原始错误：{exc}"
        ) from exc
    return gguf


def _read_identity(reader, source_path: Path) -> dict:
    """读取裁层所需的源工件元数据（与 relay_contract.RelayModelIdentity 对齐）。"""
    fields = reader.fields
    arch = fields["general.architecture"].contents()
    block_count = int(fields[f"{arch}.block_count"].contents())
    interval_field = fields.get(f"{arch}.full_attention_interval")
    interval = int(interval_field.contents()) if interval_field is not None else None
    nextn_field = fields.get(f"{arch}.nextn_predict_layers")
    nextn = int(nextn_field.contents()) if nextn_field is not None else 0
    n_embd_field = fields.get(f"{arch}.embedding_length")
    n_embd = int(n_embd_field.contents()) if n_embd_field is not None else 0
    return {
        "source": str(source_path),
        "architecture": arch,
        "block_count": block_count,
        "n_layer": max(0, block_count - nextn),
        "nextn_predict_layers": nextn,
        "n_embd": n_embd,
        "full_attention_interval": interval,
    }


def _cut_mode(k: int | None, end: int | None, keep_head: int | None) -> str:
    """切分模式：`head`（保留前 N 层）/ `middle`（保留 blk.K..end-1）/ `tail`（保留 K..末尾）。"""
    if keep_head is not None:
        return "head"
    return "middle" if (k is not None and end is not None) else "tail"


def _validate_cut(identity: dict, k: int | None = None, *,
                  end: int | None = None, keep_head: int | None = None) -> list[str]:
    """校验切点；返回问题列表（空 = 通过）。

    三种模式：
      * `tail`   —— 丢弃前 K 层、保留 `blk.K..`（重编号为 `blk.0..`）—— 与
        `relay_contract.RelayTrimPlan` 的语义一致（旧行为）；
      * `middle` —— 再给 `--end K2` ⇒ 保留 `blk.K..blk.(K2-1)`（重编号）；
      * `head`   —— 保留**前 N 层**（`blk.0..N-1`，**不重命名**），供「llama 当上游」使用。

    ⚠️ hybrid（`full_attention_interval`）下，**切点与段内层数都必须是 interval 的整数倍**，
    否则层类型错位、llama.cpp 报 missing tensor（实测）。
    """
    problems: list[str] = []
    n_layer = identity["n_layer"]
    interval = identity["full_attention_interval"]

    def _check_multiple(value: int, label: str) -> None:
        if interval and value % interval:
            problems.append(
                f"{label}={value} 不是 full_attention_interval({interval}) 的整数倍 —— "
                "hybrid 架构会层类型错位、无法加载"
            )

    if keep_head is not None:
        if not (0 < keep_head < n_layer):
            problems.append(f"--keep-head N={keep_head} 不在 (0, {n_layer}) 内")
        _check_multiple(keep_head, "--keep-head N")
        return problems

    if k is None:
        problems.append("需要 --k（或 --keep-head N）")
        return problems
    if not (0 < k < n_layer):
        problems.append(f"K={k} 不在 (0, {n_layer}) 内")
    _check_multiple(k, "K")
    if end is None:
        return problems
    if not (k < end <= n_layer):
        problems.append(f"--end={end} 必须满足 K < end <= n_layer({n_layer})")
    _check_multiple(end, "--end")
    _check_multiple(end - k, "段内层数 end-K")
    return problems


def _validate_layer_types(identity: dict, hf_config: Path) -> list[str]:
    """★ `#67` 缺口 ③：**层类型序列逐位校验**（fail-closed）。

    为什么必须校验：llama.cpp **不读** HF 的 `layer_types`，而是按（裁层重编号后的）
    **本地层号对 `full_attention_interval` 取模**推导层类型
    （`android/.../llama.cpp/src/models/qwen35.cpp:21-27`：
    `is_recr_impl[i] = (i+1) % full_attn_interval != 0`）。

    ⇒ 若源模型的 `layer_types` **本身不遵循该规律**，则无论怎么切，重推序列都对不上，
    加载必报 `missing tensor 'blk.x.<...>'`；而生成器此前**只挡整数倍、不挡这个**
    ⇒ 会**静默产出一个坏工件**，直到设备上加载才炸。

    这里只做"源是否可信"这一半；"工件加载后是否真的对"属 `#67` 缺口 ⑨（需真机加载）。
    """
    problems: list[str] = []
    interval = identity.get("full_attention_interval")
    n_layer = int(identity.get("n_layer") or 0)
    if not interval:
        problems.append(
            "--hf-config 已给出，但源 GGUF 没有 full_attention_interval：无法做层类型校验"
        )
        return problems
    try:
        cfg = json.loads(Path(hf_config).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - 校验路径统一收成 problem
        problems.append(f"--hf-config 读取失败: {exc}")
        return problems
    # Qwen3.5 把文本侧参数放在 text_config 下；兼容直接平铺的写法。
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    layer_types = list(tc.get("layer_types") or [])
    if not layer_types:
        problems.append("--hf-config 里找不到 layer_types（应为 text_config.layer_types）")
        return problems
    if len(layer_types) != n_layer:
        problems.append(
            f"layer_types 长度={len(layer_types)} ≠ 源模型层数 n_layer={n_layer}"
            "（源不可信，拒绝裁层）"
        )
        return problems
    for local, src_idx in enumerate(range(n_layer)):
        want_full = (local + 1) % int(interval) == 0
        got = layer_types[src_idx]
        is_full = got == "full_attention"
        if want_full != is_full:
            problems.append(
                f"layer_types[{src_idx}]={got!r}，但 llama.cpp 按 full_attention_interval="
                f"{interval} 会把**本地第 {local} 层**重推为 "
                f"{'full_attention' if want_full else 'linear_attention'}"
                " —— 层类型会整体错位、加载必报 missing tensor（源不合法，拒绝裁层）"
            )
            break  # 一位不符即足以拒绝；逐位刷屏对现场无益
    return problems


#: `#67` 缺口 ④：每个 block 的**必需**张量（来自 `qwen35.cpp:66-93` 的 `create_tensor` 调用，
#: 已剔除标注 `TENSOR_NOT_REQUIRED` 的 `wqkv`(`attn_qkv`) / `wqkv_gate`(`attn_gate`) / `output.weight`）。
_REQUIRED_LINEAR_SUFFIXES = (
    "attn_norm.weight", "post_attention_norm.weight",
    "ssm_conv1d.weight", "ssm_dt.bias", "ssm_a", "ssm_beta.weight",
    "ssm_alpha.weight", "ssm_norm.weight", "ssm_out.weight",
    "ffn_gate.weight", "ffn_down.weight", "ffn_up.weight",
)
_REQUIRED_FULL_SUFFIXES = (
    "attn_norm.weight", "post_attention_norm.weight",
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
    "attn_q_norm.weight", "attn_k_norm.weight",
    "ffn_gate.weight", "ffn_down.weight", "ffn_up.weight",
)
_REQUIRED_GLOBAL_SUFFIXES = ("token_embd.weight", "output_norm.weight")


#: 哪些架构有"必需张量表"（目前只整理了 `qwen35`；其它架构跳过该校验）。
_REQUIRED_TENSOR_ARCHITECTURES = ("qwen35",)


def _validate_required_tensors(reader, identity: dict, *, k: int | None = None,
                               end: int | None = None,
                               keep_head: int | None = None) -> list[str]:
    """★ `#67` 缺口 ④：**按层类型断言必需张量集合**（fail-closed）。

    为什么必须校验：`qwen35.cpp` 的 `create_tensor` 决定哪些张量"必须有"，
    缺任何一个都只在**设备上加载时**才炸（`missing tensor 'blk.x.<...>'`）。
    现有源件齐全，所以此前从没暴露 —— 但生成器产出的工件若少了必需张量，
    直到真机加载才发现，代价极高。

    ⚠️ **只对已整理必需表的架构生效**（`_REQUIRED_TENSOR_ARCHITECTURES`）：其它架构的
    张量清单与 qwen35 不同，用这张表校验会产生假阳性。

    ⚠️ **层类型判据与 llama.cpp 严格一致**：按**（裁层重编号后的）本地层号**对
    `full_attention_interval` 取模（`qwen35.cpp:25`）；该 KV 缺失时 llama.cpp 默认 4（`:22`），
    这里同样默认 4，避免"我判它合法、llama.cpp 判它非法"。

    ⚠️ 只做**存在性**断言，不校验形状/量化类型（后者属 ⑨ 的"加载自证"）。
    """
    problems: list[str] = []
    arch = str(identity.get("architecture") or "")
    if arch not in _REQUIRED_TENSOR_ARCHITECTURES:
        return problems
    names = {getattr(t, "name", "") for t in reader.tensors}
    # ★ 代表性门控：**校验只在"源看起来是完整模型"时生效**。
    #   判据用 `blk.0`（interval=4 时它是 linear 层）：源若连它的必需张量都不全，
    #   那多半是测试用的最小合成件（只写 attn_norm 之类），不是本校验的对象。
    #   否则每个测试都得造完整模型 fixture，而收益为零（它们测的是别的契约）。
    if not all(f"blk.0.{suffix}" in names for suffix in _REQUIRED_LINEAR_SUFFIXES):
        return problems
    for suffix in _REQUIRED_GLOBAL_SUFFIXES:
        if suffix not in names:
            problems.append(f"缺少全局必需张量 {suffix}（qwen35.cpp 无条件 create_tensor）")
    n_layer = int(identity.get("n_layer") or 0)
    interval = int(identity.get("full_attention_interval") or 4)  # llama.cpp 缺省 4
    if interval <= 1:
        interval = 1
    if keep_head is not None:
        lo, hi = 0, int(keep_head)
    else:
        lo = int(k or 0)
        hi = int(end) if end is not None else n_layer
    missing: list[str] = []
    for src_idx in range(lo, hi):
        local = src_idx - lo
        recr = (local + 1) % interval != 0
        required = _REQUIRED_LINEAR_SUFFIXES if recr else _REQUIRED_FULL_SUFFIXES
        kind = "linear" if recr else "full"
        for suffix in required:
            full_name = f"blk.{src_idx}.{suffix}"
            if full_name not in names:
                missing.append(f"{full_name}（{kind} 层必需）")
    if missing:
        head = "、".join(missing[:6])
        more = f" 等共 {len(missing)} 个" if len(missing) > 6 else ""
        problems.append(
            f"保留区间内缺少**必需**张量：{head}{more}"
            " —— 这些在 qwen35.cpp 里是 create_tensor 无条件要求的，"
            "缺失会在设备加载时报 missing tensor（拒绝产出坏工件）"
        )
    return problems


def _manifest(identity: dict, k: int | None, dst: Path, kept: int, dropped: int, *,
              end: int | None = None, keep_head: int | None = None,
              kept_names: list[str] | None = None,
              source_model_id: str = "", tokenizer_sha256: str = "") -> dict:
    """产出一份可复算的 manifest（三种模式都覆盖）。

    `mode=head` / `mode=middle` 时 `contract` 段**标记为不适用** —— `relay_contract.RelayTrimPlan`
    只描述「丢弃前 K 层、保留到末尾」一种形态，用它描述上游段/中段会误导读者与下游校验。

    ★ `#67`-⑥：`kept_names` 给出时，额外写 `artifact_contains` ——**显式**声明该工件带不带
    `token_embd` / `output_norm` / `output.weight`，并给出 `can_serve_tail`（能否承担末段职责）。
    此前下游只能靠「段类型 + 区间」反推，于是在 `LayerArtifactCatalog.kt` 里写出了与实际产物
    矛盾的断言（"中间段没有 final_norm"，而实测 `tensors_kept=55` 明确含 `output_norm`）。
    """
    mode = _cut_mode(k, end, keep_head)
    if mode == "head":
        assert keep_head is not None
        kept_block_count = keep_head
        source_layer_range = [0, keep_head]
        first_local = 0
        trim = 0
    elif mode == "middle":
        assert k is not None and end is not None
        kept_block_count = end - k
        source_layer_range = [k, end]
        first_local = k
        trim = k
    else:
        assert k is not None
        kept_block_count = max(0, identity["block_count"] - k)
        source_layer_range = [k, identity["n_layer"]]
        first_local = k
        trim = k
    result = {
        "generator": "scripts/cut_layers.py",
        "generator_version": 2,
        "mode": mode,
        "source": identity["source"],
        # Android workers use this logical-model digest to reject a crop from
        # another GGUF family. Keep the artifact digest separate below.
        "source_model_sha256": (
            _sha256(Path(identity["source"]))
            if Path(identity["source"]).is_file() else ""
        ),
        "artifact": str(dst),
        "architecture": identity["architecture"],
        "block_count": identity["block_count"],
        "n_layer": identity["n_layer"],
        "nextn_predict_layers": identity["nextn_predict_layers"],
        "hidden_size": int(identity["n_embd"]),
        # ★ 2026-09-23：段在**源模型**里的层区间（half-open）与本地映射起点。
        "source_layer_range": source_layer_range,
        "trim_layers": trim,
        "kept_block_count": kept_block_count,
        "tensors_kept": kept,
        "tensors_dropped": dropped,
        "first_local_layer_maps_to": first_local,
        "artifact_sha256": _sha256(dst) if dst.exists() else "",
    }
    if source_model_id and tokenizer_sha256:
        result.update({
            "source_model_id": source_model_id,
            "tokenizer_sha256": tokenizer_sha256,
        })
    # ★ `#67`-⑥：显式声明归属（下游不必靠"段类型 + 区间"反推）。
    if kept_names is not None:
        names = set(kept_names)
        has_tok = "token_embd.weight" in names
        has_norm = "output_norm.weight" in names
        has_out = "output.weight" in names
        result["artifact_contains"] = {
            "token_embd": has_tok,
            "output_norm": has_norm,
            "output_weight": has_out,
            # 能否承担**末段**职责：需 final_norm，且末段要么有独立 output.weight，
            # 要么 tie embeddings（此时 token_embd 兼作 lm_head）。
            "can_serve_tail": bool(has_norm and (has_out or has_tok)),
        }
    if mode != "tail":
        # `RelayTrimPlan` 只描述 tail 语义 ⇒ 上游段/中段**不给**可能误导的合同字段。
        result["contract"] = {"skipped": f"mode={mode} 不由 RelayTrimPlan 描述"}
        return result
    try:  # 与主仓合同字段对齐（缺失时不影响生成）
        from relay_contract import RelayModelIdentity, RelayTrimPlan  # noqa: PLC0415

        src = RelayModelIdentity(
            model_sha256=result.get("artifact_sha256", ""),
            architecture=identity["architecture"],
            block_count=identity["block_count"],
            n_embd=identity["n_embd"],
            nextn_predict_layers=identity["nextn_predict_layers"],
        )
        plan = RelayTrimPlan(trim_layers=k, source=src)
        result["contract"] = {
            "trim_plan_valid": plan.is_valid(),
            "kept_block_count": plan.kept_block_count,
            "local_to_source_layer_0": plan.local_to_source_layer(0),
            "last_handoff_layer": src.last_handoff_layer,
        }
    except Exception as exc:  # noqa: BLE001 - 合同属可选校验
        result["contract"] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


#: llama.cpp 要求**长度等于层数**的"按层数组"KV —— 裁层后必须同步裁剪，否则加载失败。
#: （`#67` 缺口 ⑩：`_copy_kv` 对 ARRAY 是原样复制，长度不符会 throw。）
_PER_LAYER_ARRAY_SUFFIXES = ("attention.recurrent_layers",)


def _per_layer_array_overrides(reader, identity: dict, *, k: int | None = None,
                               end: int | None = None,
                               keep_head: int | None = None) -> tuple[dict, list]:
    """★ `#67` 缺口 ⑩：按层数组型 KV 的**重写 + fail-closed 校验**。

    llama.cpp 对这类 KV 要求长度 == 该模型层数；裁层后若仍原样复制，长度就不符。
    这里按**与张量裁层完全相同的口径**取子数组（head ⇒ `[:N]`；tail/middle ⇒ `[k:end)`），
    并在源长度不等于层数时**拒绝**（源不可信，绝不放行坏工件）。
    """
    overrides: dict = {}
    problems: list[str] = []
    n_layer = int(identity.get("n_layer") or 0)
    if keep_head is not None:
        lo, hi = 0, int(keep_head)
    else:
        lo = int(k or 0)
        hi = int(end) if end is not None else n_layer
    for name, field in reader.fields.items():
        if not any(name.endswith(suffix) for suffix in _PER_LAYER_ARRAY_SUFFIXES):
            continue
        try:
            value = list(field.contents())
        except Exception as exc:  # noqa: BLE001 - 统一收成 problem
            problems.append(f"按层数组 KV {name} 读取失败: {exc}")
            continue
        if len(value) != n_layer:
            problems.append(
                f"按层数组 KV {name} 长度={len(value)} ≠ 源模型层数 n_layer={n_layer}"
                " —— llama.cpp 要求两者相等；原样复制到裁层工件会导致加载失败（源不可信）"
            )
            continue
        sliced = value[lo:hi]
        if len(sliced) != hi - lo:
            problems.append(f"按层数组 KV {name} 裁剪异常: 期望 {hi - lo} 项，实得 {len(sliced)}")
            continue
        overrides[name] = sliced
    return overrides, problems


def _copy_kv(gguf, reader, writer, overrides: dict) -> int:
    def infer_type(value):
        if isinstance(value, bool):
            return gguf.GGUFValueType.BOOL
        if isinstance(value, int):
            return gguf.GGUFValueType.INT32
        if isinstance(value, float):
            return gguf.GGUFValueType.FLOAT32
        if isinstance(value, str):
            return gguf.GGUFValueType.STRING
        raise ValueError(f"无法推断类型: {type(value)}")

    count = 0
    for name, field in reader.fields.items():
        # architecture 由 writer 构造时写入；GGUF.version/tensor_count 等由 writer 维护，
        # 复制会造成 "Duplicate GGUF.version" 这类文件损坏。
        if name == "general.architecture" or name.startswith("GGUF."):
            continue
        try:
            value = field.contents()
        except Exception as exc:  # noqa: BLE001
            print(f"  ! KV 跳过 {name}: {exc}")
            continue
        if name in overrides:
            value = overrides[name]
        types = list(getattr(field, "types", []) or [])
        vtype = types[0] if types else None
        if vtype == gguf.GGUFValueType.ARRAY:
            sub = types[1] if len(types) > 1 else infer_type(value[0])
            writer.add_key_value(name, value, gguf.GGUFValueType.ARRAY, sub)
        elif vtype is not None:
            writer.add_key_value(name, value, vtype)
        else:
            raise RuntimeError(f"无法确定 {name} 的类型")
        count += 1
    return count


def _plan_tensors(reader, k: int | None = None, *, end: int | None = None,
                  keep_head: int | None = None) -> tuple[list, list]:
    """返回 (保留张量及新名字, 将丢弃的张量名)。不做写入，便于 --dry-run。

    * `keep_head` 模式：保留 `blk.0..N-1`，**不重命名**（本就连续）；
    * tail / middle 模式：丢掉区间外的层，并把 `blk.K..` 重编号为 `blk.0..`
      （裁层工件的第一层必须叫 `blk.0`，下游 `embd` 注入才对齐）。
    """
    keep: list[tuple] = []
    drop: list[str] = []
    for tensor in reader.tensors:
        name = tensor.name
        if name.startswith("blk."):
            parts = name.split(".", 2)
            index = int(parts[1])
            if keep_head is not None:
                if index >= keep_head:
                    drop.append(name)
                    continue
                keep.append((tensor, name))
                continue
            assert k is not None, "tail/middle 模式必须给 --k"
            if index < k or (end is not None and index >= end):
                drop.append(name)
                continue
            keep.append((tensor, f"blk.{index - k}.{parts[2]}"))
        else:
            keep.append((tensor, name))
    return keep, drop


def main() -> int:
    ap = argparse.ArgumentParser(description="离线裁层生成器（下游工件）")
    ap.add_argument("--src", required=True, help="源 GGUF（通常是整模）")
    ap.add_argument("--dst", help="输出裁层 GGUF")
    ap.add_argument("--k", type=int, help="丢弃前 K 层（保留 blk.K.. 起）")
    ap.add_argument("--end", type=int, default=None,
                    help="★ 与 --k 同用：只保留 blk.K..blk.(end-1)（**中段工件**）；"
                         "缺省 = 保留到末尾（末段工件）")
    ap.add_argument("--keep-head", type=int, default=None,
                    help="★ 保留**前 N 层**（blk.0..N-1，**不重命名**）⇒ 供「llama 当上游」"
                         "（上游段工件）。与 --k 互斥")
    ap.add_argument("--manifest", help="输出 manifest JSON 的路径")
    ap.add_argument("--source-model-id", default=None,
                    help="源模型稳定标识；必须与 --tokenizer-sha256 成对提供")
    ap.add_argument("--tokenizer-sha256", default=None,
                    help="tokenizer 内容 SHA256；必须与 --source-model-id 成对提供")
    ap.add_argument("--verify-manifest", help="校验模式：对照该 manifest 检查 --src 工件")
    ap.add_argument("--dry-run", action="store_true", help="只列出影响，不写文件")
    ap.add_argument("--hf-config", default=None,
                    help="★ HF config.json（可选）：做 `#67`-③ **层类型序列逐位校验** —— "
                         "源的 layer_types 必须与 llama.cpp 按 full_attention_interval 重推的"
                         "序列一致，否则加载必报 missing tensor（fail-closed，拒绝产出坏工件）")
    args = ap.parse_args()

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
    if not src.exists():
        print(f"FAIL: 源文件不存在: {src}")
        return 2

    gguf = _require_gguf()
    reader = gguf.GGUFReader(str(src))
    identity = _read_identity(reader, src)

    # ---------------- 校验模式 ----------------
    if args.verify_manifest:
        manifest = json.loads(Path(args.verify_manifest).read_text(encoding="utf-8"))
        expected = int(manifest.get("kept_block_count", -1))
        ok = True
        if identity["block_count"] != expected:
            print(f"FAIL: block_count={identity['block_count']} ≠ manifest.kept_block_count={expected}")
            ok = False
        digest = _sha256(src)
        if manifest.get("artifact_sha256") and manifest["artifact_sha256"] != digest:
            print("FAIL: artifact_sha256 不一致（工件与 manifest 不匹配）")
            ok = False
        trim = int(manifest.get("trim_layers", 0))
        if identity["block_count"] and trim:
            print(f"OK: block_count={identity['block_count']}，由 {identity['block_count'] + trim} 裁掉前 {trim} 层")
        else:
            print(f"OK: block_count={identity['block_count']}")
        print(f"     artifact_sha256={digest[:16]}…")
        return 0 if ok else 1

    if args.end is not None and args.k is None:
        print("FAIL: --end 只能与 --k 同用（中段工件）")
        return 2
    if args.keep_head is not None and args.k is not None:
        print("FAIL: --keep-head（上游段）与 --k（末段/中段）互斥，二者只能给一个")
        return 2
    if args.keep_head is None and args.k is None:
        print("FAIL: 需要 --k（或 --keep-head N / --verify-manifest）")
        return 2

    problems = _validate_cut(identity, args.k, end=args.end, keep_head=args.keep_head)
    # ★ `#67`-③：层类型序列逐位校验（可选，给了 --hf-config 才做）。
    if args.hf_config:
        problems += _validate_layer_types(identity, Path(args.hf_config))
    # ★ `#67`-⑩：按层数组型 KV（如 attention.recurrent_layers）。
    #   校验放在 dry-run 之前 ⇒ `--dry-run` 也能提前发现这类坏源，而不是等到写出工件。
    per_layer_overrides, per_layer_problems = _per_layer_array_overrides(
        reader, identity, k=args.k, end=args.end, keep_head=args.keep_head,
    )
    problems += per_layer_problems
    # ★ `#67`-④：按层类型断言必需张量集合（同样放在 dry-run 之前）。
    problems += _validate_required_tensors(
        reader, identity, k=args.k, end=args.end, keep_head=args.keep_head,
    )
    keep, drop = _plan_tensors(reader, args.k, end=args.end, keep_head=args.keep_head)
    mode = _cut_mode(args.k, args.end, args.keep_head)
    if mode == "head":
        kept_block_count = int(args.keep_head)
    elif mode == "middle":
        kept_block_count = int(args.end) - int(args.k)
    else:
        kept_block_count = max(0, identity["block_count"] - args.k)
    print(f"[src] {src}")
    print(f"      architecture={identity['architecture']} block_count={identity['block_count']} "
          f"n_layer={identity['n_layer']} nextn={identity['nextn_predict_layers']} "
          f"full_attention_interval={identity['full_attention_interval']}")
    if mode == "head":
        detail = f"保留前 {args.keep_head} 层（blk.0..blk.{int(args.keep_head) - 1}，不重命名）"
    elif mode == "middle":
        detail = (f"保留源模型 blk.{args.k}..blk.{int(args.end) - 1} 并重编号"
                  f"（原 blk.{args.k}.* → blk.0.*）")
    else:
        detail = f"丢弃前 K={args.k} 层（原 blk.{args.k}.* → blk.0.*）"
    print(f"[plan] mode={mode}：{detail} ⇒ 保留 {len(keep)} 张量、丢弃 {len(drop)} 张量；"
          f"block_count: {identity['block_count']} -> {kept_block_count}")
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 2
    if args.dry_run:
        print("[dry-run] 不写任何文件。将丢弃的 blk 张量示例（最多 8 个）：")
        for name in drop[:8]:
            print(f"    - {name}")
        if len(drop) > 8:
            print(f"    … 共 {len(drop)} 个")
        return 0

    if not args.dst:
        print("FAIL: 非 dry-run 模式需要 --dst")
        return 2

    dst = Path(args.dst)
    writer = gguf.GGUFWriter(str(dst), identity["architecture"])
    bc_name = f"{identity['architecture']}.block_count"
    overrides = {bc_name: kept_block_count}
    # ★ `#67`-⑩：按层数组型 KV 已按同一口径裁剪 ⇒ 在此合并覆盖（原样复制会长度不符）。
    overrides.update(per_layer_overrides)
    # ★ 2026-09-27：MTP（nextn）层是**原模型的最后一层**（实测 `qwen35-2b` 的 MTP tensor
    #   名为 `blk.24.nextn.*`，而 `block_count=25`）⇒ **只要它被裁掉/被截断，
    #   `nextn_predict_layers` 就必须归 0**。否则 llama.cpp 会把**最后一个普通层**当成 MTP 层、
    #   要求它带 `nextn.*` tensor ⇒ 报 `missing tensor 'blk.11.nextn.eh_proj.weight'`
    #   而**整个模型加载失败**（补 hybrid 端到端验证时实测踩到；单机/协议级测试测不出来）。
    nextn = int(identity["nextn_predict_layers"])
    if nextn:
        mtp_index = int(identity["block_count"]) - 1        # MTP 所在层号
        kept_last_index = kept_block_count - 1
        # `keep_head`/`tail`/`middle` 三种模式共享同一判据：MTP 层是否还在保留区间里
        mtp_kept = (
            (args.keep_head is not None and mtp_index < int(args.keep_head))
            or (args.keep_head is None and args.end is not None
                and int(args.k) <= mtp_index < int(args.end))
            or (args.keep_head is None and args.end is None and mtp_index >= int(args.k))
        )
        if not mtp_kept:
            nextn_name = f"{identity['architecture']}.nextn_predict_layers"
            overrides[nextn_name] = 0
            print(f"[kv] ★ MTP 层（blk.{mtp_index}）已不在保留区间 ⇒ "
                  f"{nextn_name}: {nextn} -> 0")
        elif kept_last_index != mtp_index:
            # 保留区间含 MTP 但**层号变了**（tail/middle 会重编号）⇒ 记录，便于事后核对
            print(f"[kv] MTP 层保留，重编号后位于 blk.{kept_last_index}"
                  f"（tensor 名随 `_plan_tensors` 一起重写）")
    n_kv = _copy_kv(gguf, reader, writer, overrides)
    print(f"[kv] 复制 {n_kv} 字段，{bc_name}: {identity['block_count']} -> {kept_block_count}")
    for tensor, new_name in keep:
        writer.add_tensor(new_name, tensor.data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[done] {dst}：保留 {len(keep)} 张量、丢弃 {len(drop)}")

    manifest = _manifest(identity, args.k, dst, len(keep), len(drop),
                         end=args.end, keep_head=args.keep_head,
                         kept_names=[new_name for _, new_name in keep],
                         source_model_id=source_model_id,
                         tokenizer_sha256=tokenizer_sha256)
    if args.manifest:
        Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        print(f"[manifest] {args.manifest}")
    print(f"[summary] kept_block_count={manifest['kept_block_count']} "
          f"artifact_sha256={manifest['artifact_sha256'][:16]}… "
          f"contract_valid={manifest.get('contract', {}).get('trim_plan_valid')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
