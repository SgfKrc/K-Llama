from __future__ import annotations

# Owned by the external release/offline-asset tooling workspace (module
# ownership decision, 2026-09-14); the local builder is not a main-repo source.
# Legacy local run: QLH_INCLUDE_WORKSPACE_TOOL_TESTS=1 python -m pytest <this file>.
# Opting in requires the real builder; keep missing imports fatal.


import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_offline_bundle as b  # noqa: E402


@pytest.fixture
def fake_assets(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    models = repo / "models"
    (models / "qwen-1_8b-chat").mkdir(parents=True)
    (models / "qwen-1_8b-chat" / "model.safetensors").write_bytes(b"weights")
    (models / "Qwen-1_8B-Chat.Q4_K_M.gguf").write_bytes(b"gguf")
    (models / "qwen3-4b-gguf").mkdir()
    (models / "qwen3-4b-gguf" / "model.gguf").write_bytes(b"gguf")
    (models / "qwen3-4b").mkdir()
    (models / "qwen3-4b" / "model.safetensors").write_bytes(b"weights")
    (models / "qwen3-vl-4b-instruct").mkdir()
    (models / "qwen3-vl-4b-instruct" / "model.safetensors").write_bytes(b"weights")
    (models / "qwen3-vl-4b-instruct-gguf").mkdir()
    (models / "qwen3-vl-4b-instruct-gguf" / "model.gguf").write_bytes(b"gguf")
    for name in ("qwen3-5-2b", "qwen3-5-9b", "gemma4-12b-safetensors", "deepseek-r1-distill-qwen-7b"):
        (models / name).mkdir()
        (models / name / "model.safetensors").write_bytes(b"weights")
    for name in ("qwen3-5-2b-gguf", "qwen3-5-9b-gguf"):
        (models / name).mkdir()
        (models / name / "model.gguf").write_bytes(b"gguf")
    (models / "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf").write_bytes(b"gguf")
    native = models / "gemma4-native"
    native.mkdir()
    (native / "model.gguf").write_bytes(b"gguf")
    (native / "mmproj-f16.gguf").write_bytes(b"mmproj")
    (native / "gemma4-native.lock.json").write_text(json.dumps({
        "artifacts": {"model.gguf": {"size": 4}, "mmproj-f16.gguf": {"size": 6}},
    }), encoding="utf-8")
    monkeypatch.setattr(b, "REPO_ROOT", repo)
    monkeypatch.setattr(b, "OUT_DIR", out)
    return repo, out


def test_missing_lists_fetch_commands(fake_assets):
    repo, _ = fake_assets
    (repo / "models" / "qwen3-4b-gguf" / "model.gguf").unlink()
    missing = b._missing_assets(b.PC_ASSETS)
    assert "qwen3-4b-gguf" in missing
    assert b.ASSETS["qwen3-4b-gguf"]["fetch"]


def test_missing_all_present(fake_assets):
    assert b._missing_assets(b.PC_ASSETS) == []
    assert b._missing_assets(b.ANDROID_ASSETS) == []


def test_pc_bundle_contains_registered_llm_assets_only(fake_assets):
    bundle = b._build_bundle(b.PC_ASSETS, "pc", fmt="zip")
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        assert "MANIFEST.json" in names
        assert "CHECKSUMS.sha256" in names
        assert any("qwen-1_8b-chat/model.safetensors" in name for name in names)
        assert any("gemma4-native/model.gguf" in name for name in names)
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert manifest["variant"] == "pc"
        assert set(manifest["asset_ids"]) == set(b.PC_ASSETS)
        assert manifest["payload_bytes"] >= manifest["stored_bytes"] > 0


def test_android_bundle_only_contains_gguf(fake_assets):
    bundle = b._build_bundle(b.ANDROID_ASSETS, "android", fmt="zip")
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        assert any("Qwen-1_8B-Chat.Q4_K_M.gguf" in name for name in names)
        assert any("model.gguf" in name for name in names)
        assert not any("safetensors" in name or "gemma" in name.lower() for name in names)


def test_verify_roundtrip_and_archive_checksum(fake_assets):
    bundle = b._build_bundle(b.ANDROID_ASSETS, "android", fmt="zip")
    b._verify_bundle(bundle)
    sidecar = Path(str(bundle) + ".sha256")
    assert sidecar.is_file()
    with open(bundle, "ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(b.BundleError):
        b._verify_bundle(bundle)


@pytest.mark.skipif(b._seven_zip() is None, reason="7-Zip not installed")
def test_7z_build_and_verify_roundtrip(fake_assets):
    bundle = b._build_bundle(b.ANDROID_ASSETS, "android", fmt="7z")
    b._verify_bundle(bundle)


def test_capacity_preflight_is_read_only_and_conservative(fake_assets, monkeypatch):
    _, out = fake_assets
    monkeypatch.setattr(b, "TMP_BASE", out / "verify-tmp")
    report = b._capacity_preflight(b.PC_ASSETS, "pc", "zip", verify=True)
    assert report["admitted"] is True
    assert report["payload_bytes"] > 0
    assert report["output_bytes"] > report["payload_bytes"]
    assert not out.exists()


def test_capacity_preflight_rejects_before_writing(fake_assets, monkeypatch):
    _, out = fake_assets
    monkeypatch.setattr(b, "TMP_BASE", out / "verify-tmp")
    monkeypatch.setattr(b.shutil, "disk_usage", lambda _path: type("Usage", (), {"free": 0})())
    with pytest.raises(b.BundleError, match="容量"):
        b._build_bundle(b.ANDROID_ASSETS, "android", fmt="zip")
    assert not out.exists()


def test_verify_rejects_zip_slip_before_extract(fake_assets):
    _, out = fake_assets
    out.mkdir()
    malicious = out / "malicious.zip"
    with zipfile.ZipFile(malicious, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("../outside.txt", "not allowed")
    with pytest.raises(b.BundleError, match="越界"):
        b._verify_bundle(malicious)


def test_volume_split_and_verify(fake_assets):
    if b._seven_zip() is None:
        pytest.skip("7-Zip not installed")
    bundle = b._build_bundle(b.ANDROID_ASSETS, "android", fmt="7z", volume="512k")
    assert not bundle.exists()
    volumes = sorted(b.OUT_DIR.glob(f"qlh-models-android-{b.BUNDLE_VERSION}.7z.*"))
    assert volumes and volumes[0].name.endswith(".001")
    b._verify_bundle(bundle)


def test_parse_volume_size():
    assert b._parse_volume_size("4g") == 4 * (1 << 30)
    assert b._parse_volume_size("512m") == 512 * (1 << 20)
    with pytest.raises(b.BundleError):
        b._parse_volume_size("xyz")
