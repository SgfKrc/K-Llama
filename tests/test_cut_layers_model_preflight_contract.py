from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

gguf = pytest.importorskip("gguf")

REPO_ROOT = Path(__file__).resolve().parents[1]
CUT_SCRIPT = REPO_ROOT / "scripts" / "cut_layers.py"
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "cut_layers_pipeline.py"
ARCH = "qwen2"
HIDDEN_SIZE = 12
MODEL_ID = "acme.qwen2:model_v1"
TOKENIZER_SHA256_UPPER = "AB" * 32
TOKENIZER_SHA256 = TOKENIZER_SHA256_UPPER.lower()


def _make_gguf(path: Path) -> None:
    import numpy as np

    writer = gguf.GGUFWriter(str(path), ARCH)
    writer.add_block_count(4)
    writer.add_embedding_length(HIDDEN_SIZE)
    for layer in range(4):
        writer.add_tensor(
            f"blk.{layer}.attn_norm.weight",
            np.ones(HIDDEN_SIZE, dtype=np.float32),
        )
    writer.add_tensor(
        "token_embd.weight",
        np.ones((HIDDEN_SIZE, HIDDEN_SIZE), dtype=np.float32),
    )
    writer.add_tensor(
        "output_norm.weight",
        np.ones(HIDDEN_SIZE, dtype=np.float32),
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


@pytest.fixture()
def src_gguf(tmp_path: Path) -> Path:
    source = tmp_path / "source.gguf"
    _make_gguf(source)
    return source


def test_manifest_writes_model_preflight_fields(src_gguf: Path, tmp_path: Path) -> None:
    destination = tmp_path / "cut.gguf"
    manifest_path = tmp_path / "cut.manifest.json"

    result = _run(
        CUT_SCRIPT,
        "--src", str(src_gguf),
        "--dst", str(destination),
        "--k", "1",
        "--manifest", str(manifest_path),
        "--source-model-id", MODEL_ID,
        "--tokenizer-sha256", TOKENIZER_SHA256_UPPER,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["hidden_size"] == HIDDEN_SIZE
    assert manifest["source_model_id"] == MODEL_ID
    assert manifest["tokenizer_sha256"] == TOKENIZER_SHA256


def test_legacy_invocation_keeps_optional_fields_absent(
    src_gguf: Path,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "legacy.manifest.json"
    result = _run(
        CUT_SCRIPT,
        "--src", str(src_gguf),
        "--dst", str(tmp_path / "legacy.gguf"),
        "--k", "1",
        "--manifest", str(manifest_path),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["hidden_size"] == HIDDEN_SIZE
    assert "source_model_id" not in manifest
    assert "tokenizer_sha256" not in manifest


@pytest.mark.parametrize(
    "metadata_args",
    [
        ("--source-model-id", MODEL_ID),
        ("--tokenizer-sha256", TOKENIZER_SHA256),
    ],
)
def test_cut_cli_rejects_incomplete_metadata_pair(
    src_gguf: Path,
    metadata_args: tuple[str, str],
) -> None:
    result = _run(
        CUT_SCRIPT,
        "--src", str(src_gguf),
        "--k", "1",
        "--dry-run",
        *metadata_args,
    )

    assert result.returncode == 2
    assert "FAIL" in result.stdout


@pytest.mark.parametrize(
    "metadata_args",
    [
        ("--source-model-id", "invalid/model", "--tokenizer-sha256", TOKENIZER_SHA256),
        ("--source-model-id", MODEL_ID, "--tokenizer-sha256", "not-a-sha256"),
    ],
)
def test_cut_cli_rejects_invalid_metadata(
    src_gguf: Path,
    metadata_args: tuple[str, ...],
) -> None:
    result = _run(
        CUT_SCRIPT,
        "--src", str(src_gguf),
        "--k", "1",
        "--dry-run",
        *metadata_args,
    )

    assert result.returncode == 2
    assert "FAIL" in result.stdout


@pytest.mark.parametrize(
    "metadata_args",
    [
        ("--source-model-id", MODEL_ID),
        ("--tokenizer-sha256", TOKENIZER_SHA256),
    ],
)
def test_pipeline_rejects_incomplete_metadata_pair_before_writing(
    src_gguf: Path,
    tmp_path: Path,
    metadata_args: tuple[str, str],
) -> None:
    outdir = tmp_path / "rejected"
    result = _run(
        PIPELINE_SCRIPT,
        "--src", str(src_gguf),
        "--k", "1",
        "--outdir", str(outdir),
        *metadata_args,
    )

    assert result.returncode == 2
    assert "FAIL" in result.stdout
    assert not outdir.exists()


def test_pipeline_forwards_model_preflight_metadata(
    src_gguf: Path,
    tmp_path: Path,
) -> None:
    outdir = tmp_path / "pipeline"
    result = _run(
        PIPELINE_SCRIPT,
        "--src", str(src_gguf),
        "--k", "1",
        "--outdir", str(outdir),
        "--source-model-id", MODEL_ID,
        "--tokenizer-sha256", TOKENIZER_SHA256_UPPER,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    manifest_path = next(outdir.glob("*.manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["hidden_size"] == HIDDEN_SIZE
    assert manifest["source_model_id"] == MODEL_ID
    assert manifest["tokenizer_sha256"] == TOKENIZER_SHA256

    report = json.loads((outdir / "cut-report.json").read_text(encoding="utf-8"))
    command = report["command"]
    model_flag = command.index("--source-model-id")
    tokenizer_flag = command.index("--tokenizer-sha256")
    assert command[model_flag + 1] == MODEL_ID
    assert command[tokenizer_flag + 1] == TOKENIZER_SHA256
