"""Ticket18: operation-bound disk reuse and owner-admitted cache collection."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import config
import model_supply_assets as assets
import model_supply_digest_cache as digests
from tests.test_model_supply_assets import (
    _kwargs, _make_gguf, _safetensors_fixture, _server, _receive,
)
from tests.test_model_supply_worker import (
    setup, segment, command, finish,
)


@pytest.mark.parametrize("fmt", ["gguf", "safetensors"])
def test_builder_new_process_rebinds_operation_without_generating(fmt, tmp_path):
    source = tmp_path / ("full.gguf" if fmt == "gguf" else "hf")
    (_make_gguf(source) if fmt == "gguf" else _safetensors_fixture(source))
    kwargs = _kwargs(source)
    root = tmp_path / "operations"
    builder = assets.build_gguf_segment if fmt == "gguf" else assets.build_safetensors_segment
    first = builder(source, root / kwargs["operation_id"] / kwargs["node_id"], **kwargs, reuse_root=root)
    code = '''import json, sys, model_supply_assets as a
source, root, kwargs, fmt = json.loads(sys.argv[1])
def forbidden(*args, **kw):
    raise AssertionError("restarted cache must not generate tensor bytes")
a._run_cut = a._copy_selected = forbidden
kwargs["operation_id"] = "op-restarted"
builder = a.build_gguf_segment if fmt == "gguf" else a.build_safetensors_segment
from pathlib import Path
result = builder(Path(source), Path(root)/kwargs["operation_id"]/kwargs["node_id"], **kwargs, reuse_root=Path(root))
print(json.dumps(result))
'''
    env = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
    result = subprocess.run([sys.executable, "-c", code, json.dumps([str(source), str(root), kwargs, fmt])],
                            capture_output=True, text=True, env=env, timeout=40)
    assert result.returncode == 0, result.stderr
    second = json.loads(result.stdout)
    public = assets.validate_supply_manifest(second)
    assert public["operation_id"] == "op-restarted"
    assert public["manifest_sha256"] != first["manifest_sha256"]
    assert public["files"] == first["files"]
    assert second["_reused_bytes"] == sum(e["size_bytes"] for e in first["files"])
    for entry in public["files"]:
        assert os.path.samefile(first["local_files"][entry["path"]], second["local_files"][entry["path"]])
    marker = Path(next(iter(second["local_files"].values()))).parent / assets._MARKER
    assert json.loads(marker.read_text())["operation_id"] == "op-restarted"


@pytest.mark.parametrize("change", ["mtime", "size", "inode"])
def test_builder_damaged_cache_rebuilds_when_identity_changes(change, tmp_path, monkeypatch):
    source = tmp_path / "full.gguf"
    _make_gguf(source)
    kwargs = _kwargs(source)
    first = assets.build_gguf_segment(source, tmp_path / "cache", **kwargs)
    weight = Path(first["local_files"]["segment.gguf"])
    stamp = weight.stat()
    identity = assets._file_identity(weight)
    original_bytes = weight.read_bytes()
    damaged = bytearray(original_bytes)
    damaged[-1] ^= 1
    if change == "inode":
        replacement = tmp_path / "replacement.gguf"
        replacement.write_bytes(damaged)
        os.utime(replacement, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        os.replace(replacement, weight)
        assert weight.stat().st_ino != stamp.st_ino
        assert weight.stat().st_size == stamp.st_size
        assert weight.stat().st_mtime_ns == stamp.st_mtime_ns
    else:
        if change == "size":
            damaged.extend(b"extra")
        weight.write_bytes(damaged)
        mtime = stamp.st_mtime_ns + 1_000_000_000 if change == "mtime" else stamp.st_mtime_ns
        os.utime(weight, ns=(stamp.st_atime_ns, mtime))
        assert weight.stat().st_ino == stamp.st_ino
        if change == "mtime":
            assert weight.stat().st_size == stamp.st_size
        else:
            assert weight.stat().st_mtime_ns == stamp.st_mtime_ns
    assert assets._file_identity(weight) != identity
    calls = []
    original = assets._run_cut
    def cut(*args, **kw):
        calls.append(args)
        return original(*args, **kw)
    hashed = []
    original_hash = assets._hash_file
    def hash_file(path, cancelled=None):
        hashed.append(Path(path))
        return original_hash(path, cancelled)
    monkeypatch.setattr(assets, "_run_cut", cut)
    monkeypatch.setattr(assets, "_hash_file", hash_file)
    second = assets.build_gguf_segment(source, tmp_path / "cache", **{**kwargs, "operation_id": "op-repair"})
    assert len(calls) == 1
    assert hashed.count(weight) == int(change != "size")
    assert second["files"] == first["files"]
    assert assets.sha256_file(second["local_files"]["segment.gguf"]) == second["artifact_sha256"]
    assert Path(second["local_files"]["segment.gguf"]).read_bytes() == original_bytes
    assert not os.path.samefile(weight, second["local_files"]["segment.gguf"])
    assert weight.read_bytes() == damaged


def test_builder_rebinds_hot_cache_when_same_length_tamper_restores_mtime(tmp_path, monkeypatch):
    """同长度篡改 + 恢复 mtime 不是同一版本；hot cache 必须失效并从 live 字节重建。

    原断言 `assets._file_identity(weight) == identity` 在 Windows 上物理不成立：
    写回同长度字节会更新 NTFS ChangeTime，而 `os.utime` 只恢复 atime/mtime。
    实测（NTFS，2026-10-27）：同长度写回后 st_dev / st_ino / st_size /
    st_mtime_ns / st_ctime_ns 五项全部保持原值，唯一变化的是 ChangeTime
    （`_file_identity` 的 index 5，来自 `GetFileInformationByHandleEx`）。
    而 `model_supply_digest_cache.file_stamp` 故意包含 ChangeTime
    （docs/已知问题记录.md #127："继续保留包含 ChangeTime 的完整版本边界，
    不以忽略版本变化绕过竞争"），所以"内容被换掉但版本未变"这个前提根本
    构造不出来——它不是实现缺陷，而是版本边界应当具备的敏感度。

    因此把断言改成真正该成立、也更有价值的性质：**差异只落在 ChangeTime**。
    只比较 size+mtime（伪造者能控制的两项）会命中旧摘要并复用被篡改的字节；
    完整版本边界必须让缓存失效、重算 live 字节，并重建出正确内容，而不是
    把污染文件当缓存复用。
    """
    source = tmp_path / "full.gguf"
    _make_gguf(source)
    kwargs = _kwargs(source)
    first = assets.build_gguf_segment(source, tmp_path / "cache", **kwargs)
    weight = Path(first["local_files"]["segment.gguf"])
    original_bytes = weight.read_bytes()
    stamp = weight.stat()
    identity = assets._file_identity(weight)
    damaged = bytearray(original_bytes)
    damaged[-1] ^= 1
    weight.write_bytes(damaged)
    os.utime(weight, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    damaged_identity = assets._file_identity(weight)
    assert damaged_identity != identity
    # 伪造者能回填的字段（size/mtime）连同 inode/device/ctime 全部不变，
    # 唯一改变的是 ChangeTime：若版本边界不含它，这里就会错误复用旧摘要。
    assert damaged_identity[:5] == identity[:5]
    assert damaged_identity[5] != identity[5]
    assert weight.stat().st_size == stamp.st_size
    assert weight.stat().st_mtime_ns == stamp.st_mtime_ns

    # 进程内 `_DIGEST_CACHE` 已由 model_supply_digest_cache 的持久化 store 取代
    # （#127）；store 的 key 是 resolve 后的绝对路径，tmp_path 之间天然隔离，
    # 无需再清空进程内字典。
    hashed = []
    original_hash = assets._hash_file
    def hash_file(path, cancelled=None):
        hashed.append(Path(path))
        return original_hash(path, cancelled)
    monkeypatch.setattr(assets, "_hash_file", hash_file)
    cut = []
    original_cut = assets._run_cut
    def run_cut(*args, **kw):
        cut.append(args)
        return original_cut(*args, **kw)
    monkeypatch.setattr(assets, "_run_cut", run_cut)

    # 版本边界变化 ⇒ 必须重新读盘；陈旧摘要不得返回。
    assert assets._hash_file_cached(weight) == assets.sha256_file(weight)
    assert assets._hash_file_cached(weight) != first["artifact_sha256"]
    assert weight in hashed

    second = assets.build_gguf_segment(source, tmp_path / "cache", **{**kwargs, "operation_id": "op-reuse"})
    # 被污染的缓存段不能复用：必须重新裁切一次并产出与原始段一致的清单。
    assert len(cut) == 1
    assert second["operation_id"] == "op-reuse"
    assert second["manifest_sha256"] != first["manifest_sha256"]
    assert second["files"] == first["files"]
    assert second["artifact_sha256"] == first["artifact_sha256"]
    rebuilt = Path(second["local_files"]["segment.gguf"])
    assert not os.path.samefile(weight, rebuilt)
    assert rebuilt.read_bytes() == original_bytes
    # 已发布的损坏缓存不被就地改写（由后续 GC 按保留策略回收）。
    assert weight.read_bytes() == damaged


@pytest.mark.parametrize("change", ["range", "proof"])
def test_builder_does_not_rebind_another_assignment(change, tmp_path, monkeypatch):
    source = tmp_path / "full.gguf"; _make_gguf(source)
    kwargs = _kwargs(source)
    assets.build_gguf_segment(source, tmp_path / "cache", **kwargs)
    changed = {**kwargs, "operation_id": "op-other"}
    if change == "range": changed.update(start_layer=2, end_layer=4)
    else: changed["conversion_proof"] = {"converter_revision": "other-version"}
    calls = []; original = assets._run_cut
    def cut(*args, **kw):
        calls.append(args); return original(*args, **kw)
    monkeypatch.setattr(assets, "_run_cut", cut)
    assets.build_gguf_segment(source, tmp_path / "cache", **changed)
    assert len(calls) == 1


def _generation(root, operation, node="worker-one"):
    staging = root / ("temp-" + operation)
    manifest = segment(staging, operation=operation, node=node)
    directory = root / ("segment-" + manifest["manifest_sha256"])
    staging.rename(directory)
    return manifest, directory


def test_gc_unknown_native_proof_disables_deletion(tmp_path):
    _, directory = _generation(tmp_path, "unreferenced")
    report = assets.collect_supply_segments(tmp_path, keep=0)
    assert directory.is_dir()
    assert report["skipped"] == {"native_references_unknown": 1}


def test_gc_is_bounded_and_refuses_undeclared_members_and_corrupt_sha(tmp_path):
    _, native = _generation(tmp_path, "native")
    _, current = _generation(tmp_path, "current")
    _, undeclared = _generation(tmp_path, "undeclared")
    (undeclared / "unknown-empty-directory").mkdir()
    manifest, corrupt = _generation(tmp_path, "corrupt")
    bad = corrupt / "segment.gguf"; before = bad.stat()
    identity = assets._file_identity(bad)
    assert assets._hash_file_cached(bad) == manifest["artifact_sha256"]
    data = bytearray(bad.read_bytes()); data[-1] ^= 1; bad.write_bytes(data)
    os.utime(bad, ns=(before.st_atime_ns, before.st_mtime_ns))
    # 原断言 `assets._file_identity(bad) == identity` 在 Windows 上不成立：
    # 实测（NTFS）写回同长度字节只改 ChangeTime，utime 只恢复 atime/mtime，
    # 所以 dev/ino/size/mtime/ctime 五项相同、ChangeTime 必变。它的本意是
    # "摘要缓存仍认为文件是同一版本、因此GC 若信任缓存就会放过损坏段"。
    # 该前提不能再靠伪造时间戳得到，改为直接构造最坏情形：把 store 的
    # 一条记录写成"当前完整版本戳 → 篡改前的摘要"，让缓存确实返回陈旧摘要。
    damaged_identity = assets._file_identity(bad)
    assert damaged_identity != identity
    assert damaged_identity[:5] == identity[:5] and damaged_identity[5] != identity[5]
    assert bad.stat().st_size == before.st_size
    assert bad.stat().st_mtime_ns == before.st_mtime_ns
    digests._save(digests._store_path(), bad.resolve(),
                  {"stamp": list(digests.file_stamp(bad)), "sha256": manifest["artifact_sha256"]})
    # 前置条件断言：缓存此刻确实会把陈旧摘要交给调用方（否则下面
    # file_digest_mismatch 就可能是别的代码路径发现差异，覆盖不到
    # "GC 不信任摘要缓存"这一行为）。
    assert assets._hash_file_cached(bad) == manifest["artifact_sha256"]
    assert assets.sha256_file(bad) != manifest["artifact_sha256"]
    normal = [_generation(tmp_path, "stale-" + str(i))[1] for i in range(5)]
    report = assets.collect_supply_segments(tmp_path, protected_paths=[current],
        native_paths=[native / "segment.gguf"], keep=0, max_remove=2)
    assert report["removed"] == 2
    assert current.is_dir() and native.is_dir() and undeclared.is_dir() and corrupt.is_dir()
    assert sum(p.is_dir() for p in normal) == 3
    # A second bounded pass checks corrupt/undeclared metadata as well.
    report = assets.collect_supply_segments(tmp_path, protected_paths=[current],
        native_paths=[native / "segment.gguf"], keep=0)
    assert report["skipped"]["unexpected_file"] == 1
    assert report["skipped"]["file_digest_mismatch"] == 1


def test_gc_windows_reparse_member_never_touches_outside_bytes(tmp_path):
    _, directory = _generation(tmp_path, "with-reparse")
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir(); sentinel = outside / "sentinel"; sentinel.write_bytes(b"outside")
    junction = directory / "unlisted-junction"
    if sys.platform == "win32":
        import _winapi
        _winapi.CreateJunction(str(outside), str(junction))
    else:
        junction.symlink_to(outside, target_is_directory=True)
    assert assets._link(junction)
    report = assets.collect_supply_segments(tmp_path, native_paths=[], keep=0)
    assert report["removed"] == 0 and directory.is_dir()
    assert sentinel.read_bytes() == b"outside"


def test_receiver_operation_receipts_share_bytes_then_owner_gc_reclaims(tmp_path):
    source = tmp_path / "full.gguf"; _make_gguf(source)
    kwargs = _kwargs(source)
    first = assets.build_gguf_segment(source, tmp_path / "master", **kwargs)
    second = assets.build_gguf_segment(source, tmp_path / "master", **{**kwargs, "operation_id": "op-two"})
    root = tmp_path / "worker"
    with _server(first) as (url, _): old = _receive(first, root, url)
    with _server(second) as (url, state): new = _receive(second, root, url)
    assert not state["requests"] and new["actual_downloaded_bytes"] == 0
    assert new["manifest"]["operation_id"] == "op-two"
    assert os.path.samefile(old["model_path"], new["model_path"])
    report = assets.collect_supply_segments(root, protected_paths=[new["model_path"]], native_paths=[], keep=0)
    assert report["removed"] == 1
    assert len(list(root.glob("segment-*"))) == 1
    assert assets.sha256_file(new["model_path"]) == new["manifest"]["artifact_sha256"]


def test_supply_completion_gc_protects_actual_loaded_and_prepared_host(setup, monkeypatch):
    """owner GC 必须保护真实 loaded engine、prepared candidate 与 selected 指针。

    原测试在 `finish(worker)` 之后断言 `assets.collect_supply_segments` 已被
    调用，即期待 worker 在 available 发布**之后**自行回收。2026-10-26 的有意
    修复把回收前移到发布之前（docs/已知问题记录.md #127：operation dirs →
    segment GC → resident stamp 复核 → check → available），worker 的 `_ensure`
    不再调用 collect_garbage，所以 observed 必然为空——这不是回归，而是新时序
    本身。发布前时机的回归由 coordinator 侧覆盖（tests/test_model_supply.py
    断言 collect_garbage 被调用时 status 仍是 supplying）。

    这里保留 worker 维护入口的强断言：调用时必须持有 full chat 与 candidate
    锁，native_paths 必须包含真实 loaded 与 prepared candidate，current.json
    指向的 selected 必须进入 protected_paths，只有 stale 可删；释放原生引用后
    loaded / candidate 才允许回收，selected 仍受指针保护。
    """
    monkeypatch.setattr(config, "MODEL_SUPPLY_CACHE_KEEP", 0)
    from model_host import ModelHost, _PipelineCandidateTransaction
    from llama_engine import LlamaCppEngine
    worker, scheduler, client = setup
    root = worker.root; root.mkdir(parents=True, exist_ok=True)
    _, loaded = _generation(root, "loaded")
    _, candidate = _generation(root, "candidate")
    manifest, selected = _generation(root, "selected")
    _, stale = _generation(root, "stale")
    host = ModelHost()
    engine = LlamaCppEngine.__new__(LlamaCppEngine)
    engine._model_path = str(loaded / "segment.gguf")
    engine._pipeline_descriptor = {}
    object.__setattr__(host, "_manager", engine)
    transaction = _PipelineCandidateTransaction("tx", "model", str(candidate / "segment.gguf"),
        "gguf", (1, 3), None, engine, {}, phase="prepared")
    object.__setattr__(host, "_pipeline_candidate_transaction", transaction)
    scheduler._host = host
    def receive(*args, **kw):
        return {"manifest": manifest, "model_path": str(selected / "segment.gguf")}
    monkeypatch.setattr(assets, "receive_segment", receive)
    observed = []; original = assets.collect_supply_segments
    def collect(*args, **kw):
        assert host.full_chat_execution_lock._is_owned()
        assert host._pipeline_candidate_lock._is_owned()
        observed.append((kw["native_paths"], kw["protected_paths"]))
        return original(*args, **kw)
    monkeypatch.setattr(assets, "collect_supply_segments", collect)
    worker.handle(command(manifest))
    assert client.wait()["status"] == "available"
    finish(worker)
    # 供给完成不再触发回收：若有人把 GC 搬回 available 之后（#127 修复前的
    # 竞争形态），这里会非空并失败。
    assert observed == []
    assert stale.is_dir()
    assert selected.is_dir() and loaded.is_dir() and candidate.is_dir()

    observed.clear()
    report = worker.collect_garbage()
    assert observed and observed[0] is not None
    native, protected = observed[0]
    native = {Path(path).resolve() for path in native}
    assert (loaded / "segment.gguf").resolve() in native
    assert (candidate / "segment.gguf").resolve() in native
    assert selected.resolve() in {Path(path).resolve() for path in protected}
    # keep=0：三份都只能因"被引用"而保留，并非留存预算或哈希失败。
    assert report["removed"] == 1 and not stale.exists()
    assert report["skipped"]["referenced"] == 3
    assert selected.is_dir() and loaded.is_dir() and candidate.is_dir()
    object.__setattr__(host, "_pipeline_candidate_transaction", None)
    engine._model_path = ""
    report = worker.collect_garbage()
    assert report["removed"] == 2
    assert selected.is_dir() and not loaded.exists() and not candidate.exists()


@pytest.mark.parametrize("cause,reason", [
    ("cut_tool_unavailable", "SUPPLY_LOCAL_CUT_TOOL_UNAVAILABLE"),
    ("insufficient_disk", "SUPPLY_LOCAL_CUT_DISK_INSUFFICIENT"),
    ("cut_timeout", "SUPPLY_LOCAL_CUT_BUDGET_EXCEEDED"),
])
def test_auto_local_cut_falls_back_with_named_progress(setup, tmp_path, monkeypatch, cause, reason):
    worker, scheduler, client = setup
    manifest = segment(tmp_path / "expected")
    calls = []
    def cut(op, supplied_assets):
        calls.append("local_cut"); raise assets.SupplyError(cause, "not feasible")
    def receive(*args, **kw):
        calls.append("distribute")
        destination = worker.root / ("segment-" + manifest["manifest_sha256"])
        import shutil
        shutil.copytree(tmp_path / "expected", destination)
        return {"manifest": manifest, "model_path": str(destination / "segment.gguf"),
                "actual_downloaded_bytes": 7, "downloaded_bytes": 7}
    monkeypatch.setattr(worker, "_local_cut", cut)
    monkeypatch.setattr(assets, "receive_segment", receive)
    worker.handle(command(manifest, strategy="local_cut", allow_fallback=True))
    result = client.wait(); finish(worker)
    assert calls == ["local_cut", "distribute"]
    assert result["status"] == "available" and result["strategy"] == "distribute"
    assert result["strategy_reason_code"] == reason
    assert result["actual_downloaded_bytes"] == 7
    assert any(m.get("strategy_reason_code") == reason for m in client.messages if m["action"] == "progress")


def test_source_corruption_never_triggers_distribution_fallback(setup, tmp_path, monkeypatch):
    from model_supply_worker import SupplyError
    worker, scheduler, client = setup
    manifest = segment(tmp_path / "expected")
    def cut(*args): raise SupplyError("source_model_sha256_mismatch")
    monkeypatch.setattr(worker, "_local_cut", cut)
    monkeypatch.setattr(assets, "receive_segment", lambda *a, **kw: pytest.fail("identity failure cannot fall back"))
    worker.handle(command(manifest, strategy="local_cut", allow_fallback=True))
    result = client.wait(); finish(worker)
    assert result["status"] == "failed" and result["reason_code"] == "source_model_sha256_mismatch"


def test_gc_protects_worker_runtime_previous_orphan_and_queued_prepare(setup, monkeypatch):
    monkeypatch.setattr(config, "MODEL_SUPPLY_CACHE_KEEP", 0)
    from model_host import ModelHost
    from worker_prepare import WorkerPrepareController, LayerRuntime
    from inference_service.engine_host import EngineHost
    worker, scheduler, _ = setup
    scheduler._host = ModelHost()
    root = worker.root; root.mkdir(parents=True)
    directories = {name: _generation(root, name)[1] for name in
                   ("active", "candidate", "previous", "orphan", "queued", "stale")}
    runtime = EngineHost.__new__(EngineHost)
    import threading
    runtime._worker_runtime_lock = threading.RLock()
    def selected(name): return {"model_path": str(directories[name] / "segment.gguf")}
    runtime._worker_runtime_active = LayerRuntime(None, selected("active"))
    runtime._worker_runtime_candidate = LayerRuntime(None, selected("candidate"))
    runtime._worker_runtime_previous = LayerRuntime(None, selected("previous"))
    runtime._worker_runtime_orphans = [LayerRuntime(None, selected("orphan"))]
    controller = WorkerPrepareController(scheduler)
    controller._runtime = runtime
    manifest = json.loads((directories["queued"] / assets._MARKER).read_text())
    controller._queue.append(SimpleNamespace(payload={"identity": {
        "manifest_sha256": manifest["manifest_sha256"]}}))
    scheduler._worker_prepare = controller
    report = worker.collect_garbage()
    assert report["removed"] == 1 and not directories["stale"].exists()
    assert all(directory.is_dir() for name, directory in directories.items() if name != "stale")


def test_owner_gc_fence_interrupts_hash_before_removing_generation(setup, monkeypatch):
    from model_host import ModelHost
    worker, scheduler, _ = setup
    scheduler._host = ModelHost()
    root = worker.root
    root.mkdir(parents=True)
    _, directory = _generation(root, "unreferenced")
    invalidated = False
    original = assets._hash_file
    def hash_then_invalidate(*args, **kwargs):
        nonlocal invalidated
        result = original(*args, **kwargs)
        invalidated = True
        return result
    monkeypatch.setattr(assets, "_hash_file", hash_then_invalidate)
    report = worker.collect_garbage(cancelled=lambda: invalidated)
    assert invalidated and directory.is_dir()
    assert report["removed"] == 0
