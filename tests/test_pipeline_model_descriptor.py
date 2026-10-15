"""Metadata-only Safetensors descriptor tests."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, "src")

from pipeline_model_descriptor import (  # noqa: E402
    _ARCHITECTURE_LAYOUTS,
    PipelineModelDescriptorError,
    canonical_pipeline_model_type,
    inspect_pipeline_model,
    public_pipeline_descriptor,
)


def _write_qwen2_fixture(root):
    (root / "config.json").write_text(
        json.dumps({
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "num_hidden_layers": 2,
            "hidden_size": 2,
            "tie_word_embeddings": True,
        }),
        encoding="utf-8",
    )
    tensors = {
        "model.embed_tokens.weight": torch.zeros(4, 2, dtype=torch.float16),
        "model.layers.0.input_layernorm.weight": torch.zeros(2, dtype=torch.float16),
        "model.layers.1.input_layernorm.weight": torch.zeros(2, dtype=torch.float16),
        "model.norm.weight": torch.zeros(2, dtype=torch.float16),
        "lm_head.weight": torch.zeros(4, 2, dtype=torch.float16),
    }
    save_file(tensors, str(root / "model.safetensors"))
    tokenizer_payload = b'{"version":1}'
    (root / "tokenizer.json").write_bytes(tokenizer_payload)
    entry = {
        "path": "tokenizer.json",
        "sha256": hashlib.sha256(tokenizer_payload).hexdigest(),
        "size_bytes": len(tokenizer_payload),
    }
    manifest = {
        "schema": 1,
        "model_type": "safetensors",
        "revision": "",
        "source": "",
        "files": [entry],
    }
    manifest["artifact_sha256"] = hashlib.sha256(
        f"{entry['path']}\0{entry['size_bytes']}\0{entry['sha256']}".encode()
    ).hexdigest()
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in manifest.items()
             if key not in {"artifact_sha256", "manifest_sha256"}},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    (root / "model.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _write_cross_format_provenance(
    root: Path,
    target: Path,
    *,
    target_tokenizer_sha256: str,
    target_path: str | None = None,
) -> dict:
    manifest = json.loads((root / "model.manifest.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "kind": "qlh.cross_format_pipeline_identity.v1",
        "source": {
            "artifact_sha256": manifest["artifact_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "tokenizer_sha256": next(
                entry["sha256"] for entry in manifest["files"]
                if entry["path"] == "tokenizer.json"
            ),
        },
        "conversion": {
            "output_type": "Q4_K_M",
            "converter_revision": "a" * 40,
            "converter_sha256": "b" * 64,
            "quantizer_revision": "c" * 40,
            "quantizer_sha256": "d" * 64,
        },
        "target": {
            "path": target_path or (
                Path("..") / target.parent.name / target.name
            ).as_posix(),
            "artifact_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "tokenizer_sha256": target_tokenizer_sha256,
        },
    }
    payload["provenance_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    (root / "cross_format.provenance.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return payload


def _write_gemma4_fixture(root):
    (root / "config.json").write_text(
        json.dumps({
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "model_type": "gemma4_unified",
            "text_config": {
                "num_hidden_layers": 2,
                "hidden_size": 2,
                "tie_word_embeddings": True,
            },
            "vision_config": {"mm_embed_dim": 2},
            "audio_config": {"audio_embed_dim": 2},
        }),
        encoding="utf-8",
    )
    save_file({
        "model.language_model.embed_tokens.weight": torch.zeros(4, 2, dtype=torch.float16),
        "model.language_model.layers.0.input_layernorm.weight": torch.zeros(2, dtype=torch.float16),
        "model.language_model.layers.1.input_layernorm.weight": torch.zeros(2, dtype=torch.float16),
        "model.language_model.norm.weight": torch.zeros(2, dtype=torch.float16),
        "lm_head.weight": torch.zeros(4, 2, dtype=torch.float16),
        "model.embed_vision.proj.weight": torch.zeros(2, 2, dtype=torch.float16),
        "model.embed_audio.proj.weight": torch.zeros(2, 2, dtype=torch.float16),
    }, str(root / "model.safetensors"))


def test_pipeline_model_type_canonicalizes_llama_cpp_qwen35_alias():
    assert canonical_pipeline_model_type("qwen35") == "qwen3_5"
    assert canonical_pipeline_model_type(" QWEN2 ") == "qwen2"


def test_descriptor_reads_headers_without_materializing_weights(tmp_path):
    _write_qwen2_fixture(tmp_path)
    descriptor = inspect_pipeline_model(tmp_path, model_id="fixture")

    assert descriptor["inspection_mode"] == "safetensors_headers_only"
    assert descriptor["pipeline_runtime_supported"] is True
    assert descriptor["model_id"] == "fixture"
    assert descriptor["total_layers"] == 2
    assert descriptor["hidden_size"] == 2
    assert descriptor["tokenizer_sha256"] == hashlib.sha256(
        (tmp_path / "tokenizer.json").read_bytes()
    ).hexdigest()
    assert descriptor["indexed_tensor_count"] == 5
    assert descriptor["layer_weight_bytes"] == [4, 4]
    assert descriptor["component_weight_bytes"]["embedding"] == 16
    assert descriptor["component_weight_bytes"]["lm_head"] == 16
    assert descriptor["weight_bytes"] == 44


def test_descriptor_fails_closed_when_tokenizer_no_longer_matches_manifest(tmp_path):
    _write_qwen2_fixture(tmp_path)
    (tmp_path / "tokenizer.json").write_text("tampered", encoding="utf-8")

    descriptor = inspect_pipeline_model(tmp_path, model_id="fixture")

    assert descriptor["tokenizer_sha256"] == ""


def test_cross_format_provenance_maps_safetensors_to_verified_gguf(
    tmp_path, monkeypatch,
):
    source = tmp_path / "models" / "source"
    target = tmp_path / "models" / "gguf" / "converted.gguf"
    source.mkdir(parents=True)
    target.parent.mkdir()
    _write_qwen2_fixture(source)
    target.write_bytes(b"converted-gguf")
    gguf_tokenizer_sha256 = "e" * 64
    provenance = _write_cross_format_provenance(
        source,
        target,
        target_tokenizer_sha256=gguf_tokenizer_sha256,
    )
    monkeypatch.setattr(
        "relay_segment_info.read_gguf_tokenizer_sha256",
        lambda path: gguf_tokenizer_sha256,
    )

    descriptor = inspect_pipeline_model(source, model_id="fixture")

    assert descriptor["source_model_sha256"] == hashlib.sha256(
        target.read_bytes()
    ).hexdigest()
    assert descriptor["tokenizer_sha256"] == gguf_tokenizer_sha256
    assert descriptor["source_tokenizer_sha256"] == hashlib.sha256(
        (source / "tokenizer.json").read_bytes()
    ).hexdigest()
    assert descriptor["cross_format_provenance_sha256"] == provenance[
        "provenance_sha256"
    ]
    public = public_pipeline_descriptor(descriptor)
    assert public["source_model_sha256"] == descriptor["source_model_sha256"]
    assert public["source_tokenizer_sha256"] == descriptor[
        "source_tokenizer_sha256"
    ]
    assert public["cross_format_provenance_sha256"] == provenance[
        "provenance_sha256"
    ]


def test_cross_format_provenance_fails_closed_after_target_tamper(
    tmp_path, monkeypatch,
):
    source = tmp_path / "models" / "source"
    target = tmp_path / "models" / "gguf" / "converted.gguf"
    source.mkdir(parents=True)
    target.parent.mkdir()
    _write_qwen2_fixture(source)
    target.write_bytes(b"converted-gguf")
    _write_cross_format_provenance(
        source,
        target,
        target_tokenizer_sha256="e" * 64,
    )
    target.write_bytes(b"tampered")
    monkeypatch.setattr(
        "relay_segment_info.read_gguf_tokenizer_sha256",
        lambda path: "e" * 64,
    )

    with pytest.raises(PipelineModelDescriptorError, match="GGUF digest mismatch"):
        inspect_pipeline_model(source, model_id="fixture")


def test_cross_format_provenance_rejects_target_outside_models_root(tmp_path):
    source = tmp_path / "models" / "source"
    target = tmp_path / "outside" / "converted.gguf"
    source.mkdir(parents=True)
    target.parent.mkdir()
    _write_qwen2_fixture(source)
    target.write_bytes(b"converted-gguf")
    _write_cross_format_provenance(
        source,
        target,
        target_tokenizer_sha256="e" * 64,
        target_path="../../outside/converted.gguf",
    )

    with pytest.raises(PipelineModelDescriptorError, match="escapes the models root"):
        inspect_pipeline_model(source, model_id="fixture")


def test_descriptor_rejects_missing_layer(tmp_path):
    _write_qwen2_fixture(tmp_path)
    path = tmp_path / "model.safetensors"
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros(4, 2, dtype=torch.float16),
            "model.layers.0.input_layernorm.weight": torch.zeros(2, dtype=torch.float16),
            "model.norm.weight": torch.zeros(2, dtype=torch.float16),
        },
        str(path),
    )
    try:
        inspect_pipeline_model(tmp_path)
    except PipelineModelDescriptorError as exc:
        assert "缺少声明层" in str(exc)
    else:
        raise AssertionError("descriptor accepted an incomplete layer set")


@pytest.mark.skipif(
    not Path("models/qwen3-4b").exists(),
    reason="缺本地工件 models/qwen3-4b（CI 不拉模型）",
)
def test_real_qwen3_artifact_is_described_but_not_admitted():
    descriptor = inspect_pipeline_model("models/qwen3-4b", model_id="qwen3-4b")
    assert descriptor["model_type"] == "qwen3"
    assert descriptor["total_layers"] == 36
    assert descriptor["pipeline_runtime_supported"] is False
    assert "adapter" in descriptor["runtime_block_reason"]


def test_gemma4_unified_uses_nested_text_layout_but_requires_sidecar(tmp_path):
    _write_gemma4_fixture(tmp_path)
    descriptor = inspect_pipeline_model(tmp_path, model_id="gemma4-fixture")

    assert descriptor["model_type"] == "gemma4_unified"
    assert descriptor["total_layers"] == 2
    assert descriptor["layer_prefix"] == "model.language_model.layers."
    assert descriptor["layer_weight_bytes"] == [4, 4]
    assert descriptor["component_weight_bytes"]["embedding"] == 16
    assert descriptor["component_weight_bytes"]["multimodal"] == 16
    assert descriptor["pipeline_runtime_supported"] is False
    assert "隔离 Transformers sidecar" in descriptor["runtime_block_reason"]


# ── ★ #31 M3：descriptor 要如实标出"进了索引但不参与层执行"的分量 ────────────────


def test_real_qwen3_5_artifact_declares_runtime_ignored_components():
    """★ 真 `qwen3-5-2b`：`visual` / `mtp` 在索引里、但**不参与层执行** ⇒ 如实标出并给字节数。

    `#31 M3` 让 `pipeline_capacity` 据此归零；这里守住"**不静默丢弃**"—— 字节数必须还在。
    """
    model_dir = Path("models/qwen3-5-2b")
    if not model_dir.is_dir():
        pytest.skip("本地没有 qwen3-5-2b 工件（该用例依赖真实工件）")

    descriptor = inspect_pipeline_model(model_dir, model_id="qwen3-5-2b")

    assert descriptor["model_type"] == "qwen3_5"
    assert descriptor["pipeline_runtime_supported"] is True
    assert descriptor["runtime_ignored_components"] == ["visual", "mtp"]
    sizes = descriptor["runtime_ignored_component_bytes"]
    assert sizes["visual"] > 0
    assert sizes["mtp"] > 0


def test_every_layout_marker_is_self_consistent():
    """★ 防手抖：任何布局声明的"忽略分量"都必须**真的有**对应前缀声明，且名字合法。

    没有这条，一个笔误（`visual` 写成 `visaul`）就等于让闸门**悄悄**放过一个分量 ——
    这是这条闸门最不该出的错。模块 import 时已跑一次真实校验，这里做正向覆盖。
    """
    for name, layout in _ARCHITECTURE_LAYOUTS.items():
        for component in layout["runtime_ignored_components"]:
            assert component in ("visual", "mtp", "multimodal"), name
            assert layout[f"{component}_prefixes"], f"{name}: {component} 没有前缀声明"


def test_only_hybrid_declares_runtime_ignored_components():
    """★ 保守性：目前**只有** hybrid 声明了这条放行 —— 其它架构仍走原闸门（fail-closed）。"""
    declaring = sorted(
        name for name, layout in _ARCHITECTURE_LAYOUTS.items()
        if layout["runtime_ignored_components"]
    )

    assert declaring == ["qwen3_5"]
