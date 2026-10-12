"""Metadata-only inspection for distributed PyTorch pipeline models.

The inspector reads ``config.json``, the Safetensors index, and tensor headers.
It never calls ``from_pretrained`` or ``safe_open.get_tensor`` and therefore
does not materialize model weights in host RAM or VRAM.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DESCRIPTOR_SCHEMA_VERSION = 1
# These model types have an in-process loader and forward executor.  Architectures
# may still have a metadata/assignment layout below while remaining fail-closed
# here when they require an isolated runtime.
#
# ★ #31 M1（2026-09-26）：加入 hybrid。**两个名字都要** —— `qwen3-5-2b` 的外层 config 是
#   `model_type=qwen3_5`（多模态包装 `Qwen3_5ForConditionalGeneration`），而**文本子 config**
#   是 `qwen3_5_text`；主仓的层加载路径与 `forward_lm_head` 报出来的正是 **`qwen3_5_text`**
#   （实测 `RuntimeError: 模型架构 qwen3_5_text 缺少最终 Norm`）。两者共用
#   `model.language_model.*` 前缀（实测真工件 318 个层 key 全部匹配）。
PIPELINE_RUNTIME_MODEL_TYPES = frozenset({"qwen", "qwen2", "qwen3_5", "qwen3_5_text"})
_PIPELINE_MODEL_TYPE_ALIASES = {
    # llama.cpp's GGUF architecture name for the Transformers ``qwen3_5``
    # model family.  Pipeline descriptors use the Transformers spelling so
    # descriptors from both engines share one model-type contract.
    "qwen35": "qwen3_5",
}
_MAX_JSON_BYTES = 64 * 1024 * 1024
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CROSS_FORMAT_PROVENANCE = "cross_format.provenance.json"
_CROSS_FORMAT_KIND = "qlh.cross_format_pipeline_identity.v1"
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class PipelineModelDescriptorError(ValueError):
    """The local artifact cannot produce a safe pipeline descriptor."""


def canonical_pipeline_model_type(value: Any) -> str:
    """Return the canonical model type used by pipeline descriptors."""

    normalized = str(value or "").strip().lower()
    return _PIPELINE_MODEL_TYPE_ALIASES.get(normalized, normalized)


_ARCHITECTURE_LAYOUTS = {
    "qwen": {
        "layer_pattern": re.compile(r"^transformer\.h\.(\d+)\."),
        "layer_prefix": "transformer.h.",
        "embedding_prefixes": ("transformer.wte.",),
        "final_norm_prefixes": ("transformer.ln_f.",),
        "lm_head_prefixes": ("lm_head.",),
        "visual_prefixes": (),
        "mtp_prefixes": (),
    },
    "qwen2": {
        "layer_pattern": re.compile(r"^model\.layers\.(\d+)\."),
        "layer_prefix": "model.layers.",
        "embedding_prefixes": ("model.embed_tokens.",),
        "final_norm_prefixes": ("model.norm.",),
        "lm_head_prefixes": ("lm_head.",),
        "visual_prefixes": (),
        "mtp_prefixes": (),
    },
    "qwen3": {
        "layer_pattern": re.compile(r"^model\.layers\.(\d+)\."),
        "layer_prefix": "model.layers.",
        "embedding_prefixes": ("model.embed_tokens.",),
        "final_norm_prefixes": ("model.norm.",),
        "lm_head_prefixes": ("lm_head.",),
        "visual_prefixes": (),
        "mtp_prefixes": (),
    },
    "qwen3_vl": {
        "layer_pattern": re.compile(
            r"^model\.language_model\.layers\.(\d+)\."
        ),
        "layer_prefix": "model.language_model.layers.",
        "embedding_prefixes": ("model.language_model.embed_tokens.",),
        "final_norm_prefixes": ("model.language_model.norm.",),
        "lm_head_prefixes": ("lm_head.",),
        "visual_prefixes": ("model.visual.", "visual."),
        "mtp_prefixes": (),
    },
    "qwen3_5": {
        "layer_pattern": re.compile(
            r"^model\.language_model\.layers\.(\d+)\."
        ),
        "layer_prefix": "model.language_model.layers.",
        "embedding_prefixes": ("model.language_model.embed_tokens.",),
        "final_norm_prefixes": ("model.language_model.norm.",),
        "lm_head_prefixes": ("lm_head.",),
        "visual_prefixes": ("model.visual.", "visual."),
        "mtp_prefixes": ("mtp.",),
        "multimodal_prefixes": (),
        # ★ #31 M3（2026-09-26）：这些分量**参与了权重索引、但不参与层执行** ⇒ 可归零。
        #   依据（均为实测，见 #31 §31.6.1）：`Qwen3_5ForCausalLM` 顶层子模块只有
        #   `['model','lm_head']`（**无 mtp**）、`model` 子模块只有
        #   `['embed_tokens','layers','norm','rotary_emb']`（**无 visual**），且
        #   `_keys_to_ignore_on_load_unexpected = ['^mtp.*','^model.visual.*']`。
        #   ⇒ `pipeline_capacity` 不再为这两类分量要求"运行时计划"（其余仍 fail-closed）。
        "runtime_ignored_components": ("visual", "mtp"),
    },
    # ★ #31 M1：`qwen3-5-2b` 的**文本子 config** 用的就是这个 model_type
    #   （主仓层加载路径与 `forward_lm_head` 报出来的都是它）⇒ 必须单独登记，
    #   否则"只加白名单不生效"（`_ARCHITECTURE_LAYOUTS.get(model_type)` 返回 None）。
    #   前缀与 `qwen3_5` **共用** —— 真工件 318 个层 key 全在 `model.language_model.*` 下（实测）；
    #   作为**纯文本**架构，visual / mtp 前缀为空（那些分量的 key 只在外层多模态口径里）。
    "qwen3_5_text": {
        "layer_pattern": re.compile(
            r"^model\.language_model\.layers\.(\d+)\."
        ),
        "layer_prefix": "model.language_model.layers.",
        "embedding_prefixes": ("model.language_model.embed_tokens.",),
        "final_norm_prefixes": ("model.language_model.norm.",),
        "lm_head_prefixes": ("lm_head.",),
        "visual_prefixes": (),
        "mtp_prefixes": (),
    },
    "gemma4_unified": {
        "layer_pattern": re.compile(
            r"^model\.language_model\.layers\.(\d+)\."
        ),
        "layer_prefix": "model.language_model.layers.",
        "embedding_prefixes": ("model.language_model.embed_tokens.",),
        "final_norm_prefixes": ("model.language_model.norm.",),
        "lm_head_prefixes": ("lm_head.",),
        "visual_prefixes": (),
        "mtp_prefixes": (),
        "multimodal_prefixes": (
            "model.embed_vision.",
            "model.embed_audio.",
        ),
    },
}

for _layout in _ARCHITECTURE_LAYOUTS.values():
    _layout.setdefault("multimodal_prefixes", ())
    # ★ #31 M3：默认"没有被运行时忽略的分量" —— 只有**显式声明**的架构（hybrid）才允许归零。
    _layout.setdefault("runtime_ignored_components", ())

# ★ #31 M3：标记只能取已知的分量名 —— 写错一个字母就等于悄悄放行一个分量，
#   那是这条闸门最不该出的错（顺带校验：被忽略的分量必须**真的**有前缀声明）。
for _name, _layout in _ARCHITECTURE_LAYOUTS.items():
    for _component in _layout["runtime_ignored_components"]:
        if _component not in ("visual", "mtp", "multimodal"):
            raise ValueError(f"{_name}: 未知的 runtime_ignored_components 项 {_component!r}")
        if not _layout.get(f"{_component}_prefixes"):
            raise ValueError(
                f"{_name}: runtime_ignored_components 声明了 {_component!r}，"
                f"但布局里没有 {_component}_prefixes"
            )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise PipelineModelDescriptorError(f"模型元数据不可读: {path.name}") from exc
    if size <= 0 or size > _MAX_JSON_BYTES:
        raise PipelineModelDescriptorError(
            f"模型元数据大小异常: {path.name} ({size} bytes)"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PipelineModelDescriptorError(f"模型元数据不是有效 JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise PipelineModelDescriptorError(f"模型元数据必须是对象: {path.name}")
    return payload


def _decoder_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, dict) else config


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def tokenizer_sha256_from_asset(root: Path) -> str:
    """Return the manifest-declared digest after hashing the live tokenizer."""

    for filename in (".qlh-model-asset.json", "model.manifest.json"):
        path = root / filename
        if not path.is_file():
            continue
        try:
            manifest = _read_json_object(path)
        except PipelineModelDescriptorError:
            continue
        if manifest.get("schema") != 1 and manifest.get("schema_version") != 1:
            continue
        for entry in manifest.get("files", []):
            if not isinstance(entry, dict):
                continue
            relative = Path(str(entry.get("path", "") or ""))
            if relative.name != "tokenizer.json" or relative.is_absolute():
                continue
            declared = str(entry.get("sha256", "") or "").strip().lower()
            if re.fullmatch(r"[0-9a-f]{64}", declared) is None:
                return ""
            tokenizer_path = (root / relative).resolve()
            try:
                tokenizer_path.relative_to(root.resolve())
            except ValueError:
                return ""
            actual = _sha256_file(tokenizer_path)
            return actual if actual == declared else ""
    return ""


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _source_manifest_identity(root: Path) -> dict[str, str]:
    """Validate the lightweight identity fields of a local model manifest."""

    manifest = _read_json_object(root / "model.manifest.json")
    if manifest.get("schema") != 1 and manifest.get("schema_version") != 1:
        raise PipelineModelDescriptorError("cross-format source manifest schema is invalid")

    artifact_sha256 = str(manifest.get("artifact_sha256", "") or "").lower()
    manifest_sha256 = str(manifest.get("manifest_sha256", "") or "").lower()
    if _SHA256_PATTERN.fullmatch(artifact_sha256) is None:
        raise PipelineModelDescriptorError("cross-format source artifact digest is invalid")
    if _SHA256_PATTERN.fullmatch(manifest_sha256) is None:
        raise PipelineModelDescriptorError("cross-format source manifest digest is invalid")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise PipelineModelDescriptorError("cross-format source manifest files are invalid")
    artifact_lines: list[str] = []
    for entry in files:
        if not isinstance(entry, dict):
            raise PipelineModelDescriptorError("cross-format source manifest entry is invalid")
        relative = str(entry.get("path", "") or "")
        digest = str(entry.get("sha256", "") or "").lower()
        size = entry.get("size_bytes")
        if (
            not relative
            or Path(relative).is_absolute()
            or _SHA256_PATTERN.fullmatch(digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise PipelineModelDescriptorError(
                "cross-format source manifest entry is invalid"
            )
        artifact_lines.append(f"{relative}\0{size}\0{digest}")
    actual_artifact_sha256 = hashlib.sha256(
        "\n".join(artifact_lines).encode("utf-8")
    ).hexdigest()
    if actual_artifact_sha256 != artifact_sha256:
        raise PipelineModelDescriptorError(
            "cross-format source artifact digest does not match its manifest"
        )

    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("artifact_sha256", None)
    unsigned_manifest.pop("manifest_sha256", None)
    if _canonical_sha256(unsigned_manifest) != manifest_sha256:
        raise PipelineModelDescriptorError("cross-format source manifest digest mismatch")

    tokenizer_sha256 = tokenizer_sha256_from_asset(root)
    if _SHA256_PATTERN.fullmatch(tokenizer_sha256) is None:
        raise PipelineModelDescriptorError("cross-format source tokenizer is unverified")
    return {
        "artifact_sha256": artifact_sha256,
        "manifest_sha256": manifest_sha256,
        "tokenizer_sha256": tokenizer_sha256,
    }


def cross_format_pipeline_identity(root: Path) -> dict[str, str]:
    """Return a verified GGUF identity explicitly bound to a Safetensors tree.

    Merely placing a GGUF next to a Hugging Face model is not evidence that the
    two contain the same weights. This optional record binds the verified
    source manifest to one immutable converted GGUF. A present but invalid
    record fails closed instead of silently falling back to the source identity.
    """

    provenance_path = root / _CROSS_FORMAT_PROVENANCE
    if not provenance_path.is_file():
        return {}
    provenance = _read_json_object(provenance_path)
    if set(provenance) != {
        "schema_version", "kind", "source", "conversion", "target",
        "provenance_sha256",
    }:
        raise PipelineModelDescriptorError("cross-format provenance fields are invalid")
    if (
        provenance.get("schema_version") != 1
        or provenance.get("kind") != _CROSS_FORMAT_KIND
    ):
        raise PipelineModelDescriptorError("cross-format provenance header is invalid")

    declared_provenance_sha256 = str(
        provenance.get("provenance_sha256", "") or ""
    ).lower()
    unsigned_provenance = dict(provenance)
    unsigned_provenance.pop("provenance_sha256", None)
    if (
        _SHA256_PATTERN.fullmatch(declared_provenance_sha256) is None
        or _canonical_sha256(unsigned_provenance) != declared_provenance_sha256
    ):
        raise PipelineModelDescriptorError("cross-format provenance digest mismatch")

    source = provenance.get("source")
    conversion = provenance.get("conversion")
    target = provenance.get("target")
    if not isinstance(source, dict) or set(source) != {
        "artifact_sha256", "manifest_sha256", "tokenizer_sha256",
    }:
        raise PipelineModelDescriptorError("cross-format source identity is invalid")
    if not isinstance(conversion, dict) or set(conversion) != {
        "output_type", "converter_revision", "converter_sha256",
        "quantizer_revision", "quantizer_sha256",
    }:
        raise PipelineModelDescriptorError("cross-format conversion identity is invalid")
    if not isinstance(target, dict) or set(target) != {
        "path", "artifact_sha256", "tokenizer_sha256",
    }:
        raise PipelineModelDescriptorError("cross-format target identity is invalid")

    source_identity = _source_manifest_identity(root)
    normalized_source = {
        key: str(source.get(key, "") or "").lower() for key in source_identity
    }
    if normalized_source != source_identity:
        raise PipelineModelDescriptorError("cross-format source identity mismatch")

    for field in ("converter_sha256", "quantizer_sha256"):
        if _SHA256_PATTERN.fullmatch(
            str(conversion.get(field, "") or "").lower()
        ) is None:
            raise PipelineModelDescriptorError(
                f"cross-format conversion {field} is invalid"
            )
    for field in ("converter_revision", "quantizer_revision"):
        if re.fullmatch(
            r"[0-9a-f]{40}|[0-9a-f]{64}",
            str(conversion.get(field, "") or "").lower(),
        ) is None:
            raise PipelineModelDescriptorError(
                f"cross-format conversion {field} is invalid"
            )
    output_type = str(conversion.get("output_type", "") or "")
    if re.fullmatch(r"[A-Za-z0-9_]{2,32}", output_type) is None:
        raise PipelineModelDescriptorError("cross-format output type is invalid")

    relative_target = Path(str(target.get("path", "") or ""))
    if not relative_target.parts or relative_target.is_absolute():
        raise PipelineModelDescriptorError("cross-format target path is invalid")
    models_root = root.parent.resolve()
    target_path = (root / relative_target).resolve()
    try:
        target_path.relative_to(models_root)
    except ValueError as exc:
        raise PipelineModelDescriptorError(
            "cross-format target escapes the models root"
        ) from exc
    if (
        not target_path.is_file()
        or target_path.is_symlink()
        or target_path.suffix.lower() != ".gguf"
    ):
        raise PipelineModelDescriptorError("cross-format target GGUF is invalid")

    target_artifact_sha256 = str(target.get("artifact_sha256", "") or "").lower()
    target_tokenizer_sha256 = str(target.get("tokenizer_sha256", "") or "").lower()
    if (
        _SHA256_PATTERN.fullmatch(target_artifact_sha256) is None
        or _sha256_file(target_path) != target_artifact_sha256
    ):
        raise PipelineModelDescriptorError("cross-format target GGUF digest mismatch")
    try:
        from relay_segment_info import read_gguf_tokenizer_sha256
    except ImportError:
        from .relay_segment_info import read_gguf_tokenizer_sha256
    if (
        _SHA256_PATTERN.fullmatch(target_tokenizer_sha256) is None
        or read_gguf_tokenizer_sha256(target_path) != target_tokenizer_sha256
    ):
        raise PipelineModelDescriptorError(
            "cross-format target tokenizer fingerprint mismatch"
        )

    return {
        "source_model_sha256": target_artifact_sha256,
        "tokenizer_sha256": target_tokenizer_sha256,
        "source_tokenizer_sha256": source_identity["tokenizer_sha256"],
        "cross_format_provenance_sha256": declared_provenance_sha256,
    }


def _safe_shard_path(root: Path, filename: str) -> Path:
    if not filename or Path(filename).is_absolute():
        raise PipelineModelDescriptorError("Safetensors 索引包含无效分片路径")
    shard = (root / filename).resolve()
    try:
        shard.relative_to(root.resolve())
    except ValueError as exc:
        raise PipelineModelDescriptorError("Safetensors 分片路径越出模型目录") from exc
    if not shard.is_file() or shard.suffix.lower() != ".safetensors":
        raise PipelineModelDescriptorError(f"Safetensors 分片不存在: {filename}")
    return shard


def _weight_map(root: Path) -> tuple[dict[str, str], int]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = _read_json_object(index_path)
        raw_map = index.get("weight_map")
        if not isinstance(raw_map, dict) or not raw_map:
            raise PipelineModelDescriptorError("Safetensors 索引缺少 weight_map")
        weight_map: dict[str, str] = {}
        for key, filename in raw_map.items():
            if not isinstance(key, str) or not isinstance(filename, str):
                raise PipelineModelDescriptorError("Safetensors weight_map 字段类型无效")
            weight_map[key] = filename
        metadata = index.get("metadata")
        declared_size = 0
        if isinstance(metadata, dict):
            try:
                declared_size = int(metadata.get("total_size", 0) or 0)
            except (TypeError, ValueError):
                declared_size = 0
        return weight_map, declared_size

    from safetensors import safe_open

    weight_map = {}
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in weight_map:
                    raise PipelineModelDescriptorError(
                        f"Safetensors tensor 重名: {key}"
                    )
                weight_map[key] = shard.name
    if not weight_map:
        raise PipelineModelDescriptorError("模型目录中没有 Safetensors 权重")
    return weight_map, 0


def _tensor_nbytes(tensor_slice: Any) -> int:
    dtype = str(tensor_slice.get_dtype()).upper()
    element_size = _DTYPE_BYTES.get(dtype)
    if element_size is None:
        raise PipelineModelDescriptorError(f"不支持的 Safetensors dtype: {dtype}")
    shape = tensor_slice.get_shape()
    try:
        elements = math.prod(int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise PipelineModelDescriptorError("Safetensors tensor shape 无效") from exc
    if elements < 0:
        raise PipelineModelDescriptorError("Safetensors tensor shape 不能为负数")
    return elements * element_size


def _component_for_key(key: str, layout: dict[str, Any]) -> tuple[str, int | None]:
    layer_match = layout["layer_pattern"].match(key)
    if layer_match:
        return "layers", int(layer_match.group(1))
    for component, field in (
        ("embedding", "embedding_prefixes"),
        ("final_norm", "final_norm_prefixes"),
        ("lm_head", "lm_head_prefixes"),
        ("visual", "visual_prefixes"),
        ("mtp", "mtp_prefixes"),
        ("multimodal", "multimodal_prefixes"),
    ):
        if any(key.startswith(prefix) for prefix in layout[field]):
            return component, None
    return "other", None


def inspect_pipeline_model(
    model_path: str | Path,
    *,
    model_id: str = "",
    layer_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build an exact, weight-free pipeline descriptor for a local model."""

    root = Path(model_path).resolve()
    if not root.is_dir():
        raise PipelineModelDescriptorError("模型目录不存在")
    config = _read_json_object(root / "config.json")
    model_type = str(config.get("model_type", "") or "").strip().lower()
    layout = _ARCHITECTURE_LAYOUTS.get(model_type)
    if layout is None:
        raise PipelineModelDescriptorError(
            f"尚未登记流水线元数据布局: {model_type or 'unknown'}"
        )

    decoder_config = _decoder_config(config)
    try:
        total_layers = int(decoder_config.get("num_hidden_layers", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise PipelineModelDescriptorError("config 的 num_hidden_layers 无效") from exc
    if total_layers <= 0:
        raise PipelineModelDescriptorError("config 缺少有效 num_hidden_layers")
    try:
        hidden_size = int(decoder_config.get("hidden_size", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise PipelineModelDescriptorError("config 的 hidden_size 无效") from exc
    if hidden_size <= 0:
        raise PipelineModelDescriptorError("config 缺少有效 hidden_size")

    weight_map, declared_weight_bytes = _weight_map(root)
    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map.items():
        keys_by_shard[filename].append(key)

    from safetensors import safe_open

    component_bytes = {
        "layers": 0,
        "embedding": 0,
        "final_norm": 0,
        "lm_head": 0,
        "visual": 0,
        "mtp": 0,
        "multimodal": 0,
        "other": 0,
    }
    layer_bytes = [0] * total_layers
    observed_weight_bytes = 0
    for filename in sorted(keys_by_shard):
        shard = _safe_shard_path(root, filename)
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            available_keys = set(handle.keys())
            for key in keys_by_shard[filename]:
                if key not in available_keys:
                    raise PipelineModelDescriptorError(
                        f"Safetensors 索引引用不存在的 tensor: {key}"
                    )
                nbytes = _tensor_nbytes(handle.get_slice(key))
                component, layer_index = _component_for_key(key, layout)
                if layer_index is not None:
                    if layer_index >= total_layers:
                        raise PipelineModelDescriptorError(
                            f"权重层号越界: {layer_index} >= {total_layers}"
                        )
                    layer_bytes[layer_index] += nbytes
                component_bytes[component] += nbytes
                observed_weight_bytes += nbytes

    partial_range = None
    if layer_range is not None:
        try:
            partial_range = (int(layer_range[0]), int(layer_range[1]))
        except (TypeError, ValueError, IndexError) as exc:
            raise PipelineModelDescriptorError("layer_range is invalid") from exc
        if (
            partial_range[0] < 0
            or partial_range[1] <= partial_range[0]
            or partial_range[1] > total_layers
        ):
            raise PipelineModelDescriptorError("layer_range is outside model bounds")
    missing_layers = [
        index for index, value in enumerate(layer_bytes)
        if value <= 0 and (
            partial_range is None
            or partial_range[0] <= index < partial_range[1]
        )
    ]
    if missing_layers:
        shown = ", ".join(str(value) for value in missing_layers[:8])
        raise PipelineModelDescriptorError(f"Safetensors 缺少声明层: {shown}")
    if declared_weight_bytes and not partial_range and declared_weight_bytes != observed_weight_bytes:
        raise PipelineModelDescriptorError(
            "Safetensors 索引总字节与文件头不一致: "
            f"declared={declared_weight_bytes}, observed={observed_weight_bytes}"
        )

    runtime_supported = model_type in PIPELINE_RUNTIME_MODEL_TYPES
    architectures = config.get("architectures")
    if not isinstance(architectures, list):
        architectures = []
    descriptor = {
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "inspection_mode": "safetensors_headers_only",
        "model_id": str(model_id or ""),
        "model_type": model_type,
        "architectures": [str(value) for value in architectures],
        "total_layers": total_layers,
        "hidden_size": hidden_size,
        "tokenizer_sha256": tokenizer_sha256_from_asset(root),
        "layer_prefix": layout["layer_prefix"],
        "weight_bytes": observed_weight_bytes,
        "indexed_tensor_count": len(weight_map),
        "weight_file_count": len(keys_by_shard),
        "layer_weight_bytes": layer_bytes,
        "component_weight_bytes": component_bytes,
        # ★ #31 M3：如实记录"这些分量在索引里、但**不参与层执行**" ⇒ 容量账可据此归零。
        #   不静默丢弃：字节数一并给出，便于事后核对（`pipeline_capacity` 只对这两类放行）。
        "runtime_ignored_components": list(layout["runtime_ignored_components"]),
        "runtime_ignored_component_bytes": {
            name: int(component_bytes.get(name, 0))
            for name in layout["runtime_ignored_components"]
        },
        "tie_word_embeddings": bool(decoder_config.get("tie_word_embeddings", False)),
        "pipeline_runtime_supported": runtime_supported,
        "runtime_block_reason": (
            "" if runtime_supported
            else (
                "gemma4_unified 需要隔离 Transformers sidecar；"
                "主运行时禁止复用 Qwen2 执行器"
                if model_type == "gemma4_unified"
                else f"{model_type} 尚未实现按层执行 adapter"
            )
        ),
        "partial_assignment": bool(partial_range),
    }
    if partial_range:
        descriptor["assignment_layer_range"] = list(partial_range)
    descriptor.update(cross_format_pipeline_identity(root))
    return descriptor


def public_pipeline_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
    """Return the non-sensitive projection used by APIs and diagnostics."""

    allowed = {
        "schema_version",
        "inspection_mode",
        "model_id",
        "source_model_id",
        "model_type",
        "architectures",
        "total_layers",
        "hidden_size",
        "tokenizer_sha256",
        "layer_prefix",
        "weight_bytes",
        "indexed_tensor_count",
        "weight_file_count",
        "layer_weight_bytes",
        "component_weight_bytes",
        "tie_word_embeddings",
        "pipeline_runtime_supported",
        "runtime_block_reason",
        "model_sha256",
        # ★ 2026-10-11（Y700 反复重选无效排查）：裁层工件必须能对外声明**源模型**摘要。
        #   客户端（Android LayerArtifactCatalog）的跨段校验是「工件的 source_model_sha256
        #   或 artifact_sha256 等于主节点报的摘要」。每段的 artifact_sha256 天然不同
        #   （head 296fce32… / tail d8167678…），而三段共享同一个 source_model_sha256
        #   （9be22744…）⇒ 只有暴露源摘要，非主节点段才能通过校验；否则远端 worker 恒
        #   source_digest_mismatch、不上报 layer_ranges（实测 ranges=None），
        #   容量求解器看不到该节点（triple 实测 3 节点只 admitted 1 个）。
        "source_model_sha256",
        "source_tokenizer_sha256",
        "cross_format_provenance_sha256",
    }
    return {key: descriptor[key] for key in allowed if key in descriptor}
