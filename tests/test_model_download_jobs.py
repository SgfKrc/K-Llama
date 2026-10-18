"""Model one-click download job service tests (P0A, no real network)."""

from __future__ import annotations

import sys
import hashlib
import json
import os
import threading
from pathlib import Path

import pytest

sys.path.insert(0, "src")

import model_download_jobs as mj


def _make_gguf_dir(base: Path, *, real_sig: bool = True) -> Path:
    """构造一个含 config.json + 单个 .gguf 的最小合法模型目录并返回其路径。"""
    d = base / "fake-gguf"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{\"architectures\": [\"LlamaForCausalLM\"]}", encoding="utf-8")
    g = d / "model.Q4_K_M.gguf"
    payload = b"FAKEGGUF" * 2000  # > 0 bytes, non-empty
    g.write_bytes(payload)
    return d


def _run_job_sync(**kw):
    """同步 executor：create_job 时立即把 _run_job 调度完（不启后台线程）。"""
    done = threading.Event()

    def executor(fn):
        fn()
        done.set()

    return done


def test_list_presets_has_expected_fields():
    presets = mj.list_presets()
    # ★ 2026-09-19：`qwen-1_8b-gguf-q4` 预设随模型退役移除 ⇒ 数量 8 → 7。
    assert len(presets) == 7
    assert {
        "qwen2.5-0.5b-instruct",
        "qwen3-0.6b",
        "minicpm4-0.5b",
        "distilqwen25-ds3-0324-7b",
    } <= {p["id"] for p in presets}
    for p in presets:
        assert {"id", "display", "kind", "hf_repo", "installable"} <= set(p)
        assert "blocked_reasons" in p


def test_remote_presets_expose_reviewed_artifact_pins():
    presets = {item["id"]: item for item in mj.list_presets()}
    for preset_id in {
        "qwen2.5-0.5b-instruct",
        "qwen3-0.6b",
        "minicpm4-0.5b",
        "distilqwen25-ds3-0324-7b",
    }:
        item = presets[preset_id]
        assert item["pin_status"] == "pinned"
        assert len(item["revision"]) == 40
        assert item["allow_patterns"]
        assert set(item["required_files"]) <= set(item["allow_patterns"])
        assert item["weight_files"]
        assert all(len(entry["sha256"]) == 64 for entry in item["weight_files"])


def test_invalid_pin_manifest_blocks_remote_presets(tmp_path, monkeypatch):
    pin_path = tmp_path / "invalid-pin.json"
    pin_path.write_text('{"schema": 1}', encoding="utf-8")
    monkeypatch.setattr(mj, "_PIN_MANIFEST_PATH", pin_path)
    presets = {item["id"]: item for item in mj.list_presets()}
    assert presets["qwen3-0.6b"]["installable"] is False
    assert presets["qwen3-0.6b"]["blocked_reasons"]["artifact_pin"] == "PRESET_PIN_INVALID"


def test_pin_paths_reject_parent_escape():
    with pytest.raises(mj.JobError) as exc:
        mj._validate_pin_path("../model.safetensors", field="test")
    assert exc.value.code == "PRESET_PIN_INVALID"


@pytest.mark.skipif(sys.platform != "win32", reason="盘符逃逸是 Windows 语义（POSIX 上 'C:/x' 不是绝对路径）")
def test_pin_paths_reject_drive_escape():
    with pytest.raises(mj.JobError) as exc:
        mj._validate_pin_path("C:/model.safetensors", field="test")
    assert exc.value.code == "PRESET_PIN_INVALID"


def test_download_passes_remote_pin_to_huggingface(tmp_path, monkeypatch):
    calls = {}

    class FakeHub:
        @staticmethod
        def snapshot_download(**kwargs):
            calls.update(kwargs)
            (Path(kwargs["local_dir"]) / "model.safetensors").write_bytes(b"fixture")

    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHub)
    staging = tmp_path / "staging"
    files = mj._download(
        "owner/repo", staging, use_modelscope=False, proxy="",
        revision="a" * 40, allow_patterns=["config.json", "model.safetensors"],
    )
    assert files == [staging / "model.safetensors"]
    assert calls["revision"] == "a" * 40
    assert calls["allow_patterns"] == ["config.json", "model.safetensors"]


def test_verify_pinned_files_checks_exact_set_and_sha256(tmp_path):
    root = tmp_path / "staging"
    root.mkdir()
    payload = b"pinned fixture"
    weight = root / "model.safetensors"
    weight.write_bytes(payload)
    expected = [{
        "path": "model.safetensors",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }]
    mj._verify_pinned_files(root, [weight], expected)

    with pytest.raises(mj.JobError) as exc:
        mj._verify_pinned_files(root, [weight], [{
            "path": "model.safetensors",
            "sha256": "0" * 64,
        }])
    assert exc.value.code == "SHA256_MISMATCH"

    extra = root / "extra.safetensors"
    extra.write_bytes(b"unexpected")
    with pytest.raises(mj.JobError) as exc:
        mj._verify_pinned_files(root, [weight, extra], expected)
    assert exc.value.code == "PINNED_FILE_SET_MISMATCH"


def test_preset_default_targets_match_builtin_asset_directories():
    presets = {item["id"]: item for item in mj.list_presets()}
    assert presets["qwen2.5-0.5b-instruct"]["default_target"] == "qwen2.5-0.5b-instruct"
    assert presets["minicpm4-0.5b"]["default_target"] == "minicpm4-0.5b"


def test_download_target_is_confined_to_models_root(tmp_path):
    root = tmp_path / "models"
    with pytest.raises(mj.JobError) as exc:
        mj._resolve_target("owner/repo", str(tmp_path / "outside"), str(root))
    assert exc.value.code == "TARGET_OUTSIDE_MODELS_ROOT"

    nested = mj._resolve_target("owner/repo", "nested/repo", str(root))
    assert nested == (root / "nested" / "repo").resolve()


def test_download_rejects_symlinked_local_source(tmp_path):
    if os.name == "nt":
        pytest.skip("creating symlinks requires elevated privileges on Windows")
    source = _make_gguf_dir(tmp_path / "source")
    link = tmp_path / "linked-model"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(mj.JobError) as exc:
        mj._download(str(link), tmp_path / "staging", use_modelscope=False, proxy="")
    assert exc.value.code == "SOURCE_SYMLINK"


def test_explicit_gguf_path_cannot_escape_staging(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    with pytest.raises(mj.JobError) as exc:
        mj._resolve_staged_gguf(str(tmp_path / "outside.gguf"), staging)
    assert exc.value.code == "GGUF_PATH_OUTSIDE_TARGET"


def test_create_job_from_local_source_no_network(tmp_path, monkeypatch):
    """本地目录作为 source + sync executor → job 走到 ready，且不触网。"""
    models_root = tmp_path / "models"
    src = _make_gguf_dir(tmp_path / "src")
    target = tmp_path / "models" / "fake-gguf"

    # 本地目录 source：_download 跳过真实下载（source 是目录时 _run_job 直接取权重）
    done = threading.Event()

    def executor(fn):
        fn()
        done.set()

    job = mj.create_job(
        source=str(src), target=str(target), model_id="fake-gguf",
        engine="llama_cpp", quant="Q4_K_M",
        models_root=str(models_root), allow_cpu=True,
        executor=executor,
    )
    assert job["status"] == mj.STATUS_READY, job
    assert job["model_id"] == "fake-gguf"
    assert job["total_bytes"] > 0
    # 目标已发布 + 注册成功
    assert target.is_dir()
    # job 已持久化（SQLite 表存在）
    reloaded = mj.get_job(job["job_id"])
    assert reloaded["status"] == mj.STATUS_READY


def test_create_job_queued_then_executed(tmp_path, monkeypatch):
    """无 executor → 停在 queued；手动跑 _run 后变 ready。"""
    models_root = tmp_path / "models"
    src = _make_gguf_dir(tmp_path / "src2")
    target = tmp_path / "models" / "fake-gguf2"
    # 让 _download 返回 staging 内已造好的权重（不真下载）
    job = mj.create_job(
        source=str(src), target=str(target), model_id="fake-gguf2",
        models_root=str(models_root),
    )
    assert job["status"] == mj.STATUS_QUEUED
    # 手动执行 job（模拟线程池调度）
    mj._run_job(
        job["job_id"], source=str(src), target=str(target), model_id="fake-gguf2",
        preset_id="", engine="llama_cpp", quant="Q4_K_M", use_modelscope=False,
        proxy="", expected_sha256="", gguf_path="", models_root=str(models_root),
        allow_cpu=True,
    )
    reloaded = mj.get_job(job["job_id"])
    assert reloaded["status"] == mj.STATUS_READY


def test_sha_mismatch_fails_job(tmp_path):
    models_root = tmp_path / "models"
    src = _make_gguf_dir(tmp_path / "src3")
    target = tmp_path / "models" / "fake-gguf3"
    job = mj.create_job(
        source=str(src), target=str(target), model_id="fake-gguf3",
        expected_sha256="deadbeef",  # 必然不匹配
        models_root=str(models_root),
    )
    mj._run_job(
        job["job_id"], source=str(src), target=str(target), model_id="fake-gguf3",
        preset_id="", engine="llama_cpp", quant="Q4_K_M", use_modelscope=False,
        proxy="", expected_sha256="deadbeef", gguf_path="",
        models_root=str(models_root), allow_cpu=True,
    )
    reloaded = mj.get_job(job["job_id"])
    assert reloaded["status"] == mj.STATUS_FAILED
    assert reloaded["error_code"] == "SHA256_MISMATCH"
    # staging 已清理，目标未发布
    assert not target.exists()


def test_registration_failure_retains_published_target_for_supply_reconciliation(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    src = _make_gguf_dir(tmp_path / "src-register-failure")
    target = models_root / "register-failure"
    monkeypatch.setattr(mj, "_register", lambda *args, **kwargs: False)

    job = mj.create_job(
        source=str(src), target=str(target), model_id="register-failure",
        models_root=str(models_root), executor=lambda fn: fn(),
    )

    assert job["status"] == mj.STATUS_FAILED
    assert job["error_code"] == "REGISTER_FAILED"
    assert job["error"].startswith("cleanup_pending:")
    assert target.is_dir()
    assert job["target"] == str(target)


def test_unknown_preset_raises():
    with pytest.raises(mj.JobError) as exc:
        mj.create_job(preset_id="nope-not-exist", models_root=str(Path(__file__).parent))
    assert exc.value.code == "PRESET_NOT_FOUND"


def test_preset_without_ms_source_blocks_modelscope():
    # ★ 2026-09-19：原用 `qwen-1_8b-gguf-q4`（该预设已随模型退役移除）。
    #   此处验的是「预设没有 ms_path 时请求 ModelScope 应被拒」，与具体预设无关；
    #   现有预设的 `ms_path` 均为空，改用默认模型 `qwen3-0.6b` 即可。
    with pytest.raises(mj.JobError) as exc:
        mj.create_job(preset_id="qwen3-0.6b", use_modelscope=True,
                      models_root=str(Path(__file__).parent))
    assert exc.value.code == "PRESET_NO_MS_SOURCE"


def test_registered_model_written_to_db(tmp_path):
    models_root = tmp_path / "models"
    src = _make_gguf_dir(tmp_path / "src4")
    target = tmp_path / "models" / "fake-gguf4"

    def executor(fn):
        fn()

    job = mj.create_job(source=str(src), target=str(target), model_id="fake-gguf4",
                        models_root=str(models_root), executor=executor)
    assert job["status"] == mj.STATUS_READY
    import local_store
    registered = local_store.get_local_experimental_models()
    ids = [m.get("model_id") for m in registered if m.get("model_id") == "fake-gguf4"]
    assert ids, "模型应写入 model_registry"


def test_cancel_only_queued(tmp_path):
    models_root = tmp_path / "models"
    src = _make_gguf_dir(tmp_path / "src5")
    target = tmp_path / "models" / "fake-gguf5"
    job = mj.create_job(source=str(src), target=str(target), model_id="fake-gguf5",
                        models_root=str(models_root))
    assert mj.cancel_job(job["job_id"]) is True
    assert mj.get_job(job["job_id"])["status"] == mj.STATUS_CANCELLED
    # 已完成/失败不可取消
    assert mj.cancel_job("no-such-id") is False
