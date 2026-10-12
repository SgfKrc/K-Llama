"""Tests for Tailnet PyTorch model synchronization."""

import hashlib
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_missing_deepseek_model_is_downloaded_and_verified(tmp_path, monkeypatch):
    from model_sync import ensure_model_available
    import model_sync

    model_id = "deepseek-r1-distill-qwen-1.5b"
    source_files = {
        "config.json": b'{"num_hidden_layers": 28}',
        "tokenizer_config.json": b'{"model_max_length": 4096}',
        "model.safetensors": b"deepseek-test-weights",
    }
    weight_sha = hashlib.sha256(source_files["model.safetensors"]).hexdigest()
    digest = hashlib.sha256()
    for name, content in sorted(source_files.items()):
        file_sha = hashlib.sha256(content).hexdigest()
        digest.update(f"{name}\0{len(content)}\0{file_sha}\n".encode())
    model_sha = digest.hexdigest()
    manifest = {
        "model_id": model_id,
        "sha256": model_sha,
        "total_layers": 28,
        "files": [
            {
                "path": name,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for name, content in source_files.items()
        ],
    }

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            requests.append(parsed.path)
            if parsed.path == "/api/models/downloadable":
                payload = json.dumps(manifest).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            prefix = f"/api/models/files/{model_id}/"
            if parsed.path.startswith(prefix):
                relative_path = unquote(parsed.path[len(prefix):])
                payload = source_files[relative_path]
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_error(404)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    worker_path = tmp_path / "models" / model_id
    monkeypatch.setattr(
        model_sync,
        "resolve_worker_model_path",
        lambda requested_id: str(worker_path),
    )
    try:
        result = ensure_model_available(
            "127.0.0.1",
            server.server_port,
            model_id,
            model_sha,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result == str(worker_path)
    assert (worker_path / "model.safetensors").read_bytes() == source_files["model.safetensors"]
    assert (worker_path / "config.json").read_bytes() == source_files["config.json"]
    assert (worker_path / "model.sha256").read_text(encoding="utf-8").startswith(model_sha)
    assert len([path for path in requests if "/api/models/files/" in path]) == 3


def test_model_digest_uses_verified_weight_content(tmp_path):
    import model_sync

    root = tmp_path / "deepseek"
    root.mkdir()
    weight = root / "model-00001-of-000001.safetensors"
    weight.write_bytes(b"small fixture standing in for a large shard")
    metadata = root / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    oid = "a" * 64
    (metadata / f"{weight.name}.metadata").write_text(
        f"revision\n{oid}\ntimestamp\n",
        encoding="utf-8",
    )

    value = model_sync.compute_model_sha256(str(root), use_cache=False)

    content_sha = hashlib.sha256(weight.read_bytes()).hexdigest()
    expected = hashlib.sha256(
        f"{weight.name}\0{weight.stat().st_size}\0{content_sha}\n".encode("utf-8")
    ).hexdigest()
    assert value == expected


def test_model_digest_changes_with_config_tokenizer_and_remote_code(tmp_path):
    import model_sync

    root = tmp_path / "qwen"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "config.json").write_text('{"model_type":"qwen"}', encoding="utf-8")
    (root / "qwen.tiktoken").write_bytes(b"tokenizer-v1")
    (root / "modeling_qwen.py").write_text("VERSION = 1\n", encoding="utf-8")

    baseline = model_sync.compute_model_sha256(str(root), use_cache=False)
    for name, content in (
        ("config.json", '{"model_type":"qwen2"}'),
        ("qwen.tiktoken", b"tokenizer-v2"),
        ("modeling_qwen.py", "VERSION = 2\n"),
    ):
        path = root / name
        original = path.read_bytes()
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        assert model_sync.compute_model_sha256(str(root), use_cache=False) != baseline
        path.write_bytes(original)


def _expected_model_digest(artifacts):
    digest = hashlib.sha256()
    for name, content in sorted(artifacts.items()):
        file_sha = hashlib.sha256(content).hexdigest()
        digest.update(f"{name}\0{len(content)}\0{file_sha}\n".encode("utf-8"))
    return digest.hexdigest()


@pytest.mark.parametrize("use_cache", [True, False])
def test_model_digest_supply_marker_does_not_create_cache(tmp_path, use_cache):
    import model_sync

    root = tmp_path / "supplied"
    root.mkdir()
    artifacts = {
        "config.json": b'{"model_type":"qwen2"}',
        "model.safetensors": b"weights-v1",
        "supply.manifest.json": b"not a trusted identity or even valid JSON",
    }
    for name, content in artifacts.items():
        (root / name).write_bytes(content)
    baseline = _expected_model_digest(artifacts)

    for _ in range(3):
        assert model_sync.compute_model_sha256(str(root), use_cache=use_cache) == baseline
        assert {path.name: path.read_bytes() for path in root.iterdir()} == artifacts

    weight = root / "model.safetensors"
    previous_stat = weight.stat()
    artifacts[weight.name] = b"weights-v2"
    weight.write_bytes(artifacts[weight.name])
    os.utime(weight, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns + 1_000_000_000))
    expected = _expected_model_digest(artifacts)
    assert expected != baseline
    assert model_sync.compute_model_sha256(str(root), use_cache=use_cache) == expected
    assert {path.name: path.read_bytes() for path in root.iterdir()} == artifacts


@pytest.mark.parametrize("use_cache", [True, False])
def test_model_digest_supply_marker_ignores_disk_cache(tmp_path, monkeypatch, use_cache):
    import model_sync

    root = tmp_path / "supplied"
    root.mkdir()
    forged_sha = "a" * 64
    artifacts = {
        "model.safetensors": b"actual weights",
        "supply.manifest.json": json.dumps({"model_sha256": forged_sha, "files": []}).encode(),
    }
    for name, content in artifacts.items():
        (root / name).write_bytes(content)
    cache_path = root / "model.sha256"
    metadata_path = root / "model.sha256.meta.json"
    fingerprint = [
        {"path": name, "size": (root / name).stat().st_size,
         "mtime_ns": (root / name).stat().st_mtime_ns}
        for name in sorted(artifacts)
    ]
    cache_path.write_text(f"{forged_sha}  forged\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(fingerprint), encoding="utf-8")
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    original_open = model_sync.Path.open

    def guard_cache_access(path, *args, **kwargs):
        if path in {cache_path, metadata_path}:
            raise AssertionError("supply directory hash caches must not be read or written")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(model_sync.Path, "open", guard_cache_access)
    expected = _expected_model_digest(artifacts)
    assert expected != forged_sha
    for _ in range(3):
        assert model_sync.compute_model_sha256(str(root), use_cache=use_cache) == expected
    monkeypatch.undo()
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


@pytest.mark.parametrize("use_cache", [True, False])
def test_model_digest_normal_directory_preserves_cache_semantics(tmp_path, monkeypatch, use_cache):
    import model_sync

    root = tmp_path / "ordinary"
    root.mkdir()
    artifacts = {"model.safetensors": b"ordinary weights"}
    for name, content in artifacts.items():
        (root / name).write_bytes(content)
    expected = _expected_model_digest(artifacts)
    original_hash = model_sync.compute_file_sha256
    hashed_paths = []

    def record_hash(path):
        hashed_paths.append(path)
        return original_hash(path)

    monkeypatch.setattr(model_sync, "compute_file_sha256", record_hash)
    for _ in range(2):
        assert model_sync.compute_model_sha256(str(root), use_cache=use_cache) == expected
    assert len(hashed_paths) == (1 if use_cache else 2)
    cache_path = root / "model.sha256"
    metadata_path = root / "model.sha256.meta.json"
    assert cache_path.read_text(encoding="utf-8").split()[0] == expected
    fingerprint = json.loads(metadata_path.read_text(encoding="utf-8"))
    weight = root / "model.safetensors"
    assert fingerprint == [{
        "path": weight.name, "size": weight.stat().st_size,
        "mtime_ns": weight.stat().st_mtime_ns,
    }]

    cache_path.write_text(f"{'a' * 64}  forged\n", encoding="utf-8")
    before_count = len(hashed_paths)
    assert model_sync.compute_model_sha256(str(root), use_cache=False) == expected
    assert len(hashed_paths) == before_count + 1
    assert cache_path.read_text(encoding="utf-8").split()[0] == expected
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == fingerprint


def test_same_size_corrupt_weight_is_redownloaded(tmp_path, monkeypatch):
    """大模型不能只凭文件大小和缓存摘要跳过内容校验。"""
    from model_sync import ensure_model_available
    import model_sync

    model_id = "deepseek-corrupt"
    good_weight = b"verified-weight-content"
    corrupt_weight = b"x" * len(good_weight)
    file_sha = hashlib.sha256(good_weight).hexdigest()
    model_sha = hashlib.sha256(
        f"model.safetensors\0{len(good_weight)}\0{file_sha}\n".encode()
    ).hexdigest()
    manifest = {
        "model_id": model_id,
        "sha256": model_sha,
        "files": [{
            "path": "model.safetensors",
            "size_bytes": len(good_weight),
            "sha256": file_sha,
        }],
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/models/downloadable":
                payload = json.dumps(manifest).encode("utf-8")
            else:
                payload = good_weight
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    worker_path = tmp_path / model_id
    worker_path.mkdir()
    (worker_path / "model.safetensors").write_bytes(corrupt_weight)
    (worker_path / "model.sha256").write_text(f"{model_sha}  stale\n", encoding="utf-8")
    monkeypatch.setattr(model_sync, "resolve_worker_model_path", lambda _mid: str(worker_path))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        ensure_model_available("127.0.0.1", server.server_port, model_id, model_sha)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert (worker_path / "model.safetensors").read_bytes() == good_weight


def test_stale_weight_shard_is_removed(tmp_path, monkeypatch):
    """同一模型 ID 更新分片布局后，清单外旧权重不能污染模型摘要。"""
    from model_sync import ensure_model_available
    import model_sync

    model_id = "deepseek-updated"
    weight = b"current-weight"
    file_sha = hashlib.sha256(weight).hexdigest()
    model_sha = hashlib.sha256(
        f"model.safetensors\0{len(weight)}\0{file_sha}\n".encode()
    ).hexdigest()
    manifest = {
        "model_id": model_id,
        "sha256": model_sha,
        "files": [{
            "path": "model.safetensors",
            "size_bytes": len(weight),
            "sha256": file_sha,
        }],
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            payload = (
                json.dumps(manifest).encode("utf-8")
                if parsed.path == "/api/models/downloadable" else weight
            )
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    worker_path = tmp_path / model_id
    worker_path.mkdir()
    (worker_path / "model.safetensors").write_bytes(weight)
    stale = worker_path / "model-00002-of-00002.safetensors"
    stale.write_bytes(b"old-shard")
    monkeypatch.setattr(model_sync, "resolve_worker_model_path", lambda _mid: str(worker_path))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = ensure_model_available(
            "127.0.0.1", server.server_port, model_id, model_sha
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result == str(worker_path)
    assert not stale.exists()


def test_assignment_sync_downloads_only_manifest_files(tmp_path, monkeypatch):
    import model_sync

    model_id = "assignment-model"
    config_id = "cfg-1"
    node_id = "worker-1"
    plan_id = "plan-1"
    source_files = {
        "config.json": b'{"model_type":"qwen2","num_hidden_layers":2}',
        "model.safetensors": b"selected-layer-shard",
        "tokenizer_config.json": b"{}",
    }
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {"model.layers.0.weight": "model.safetensors"},
    }
    index_bytes = json.dumps(index, sort_keys=True, separators=(",", ":")).encode()
    source_files["model.safetensors.index.json"] = index_bytes
    files = [
        {
            "path": path,
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "kind": "weights" if path.endswith(".safetensors") else "support",
        }
        for path, content in source_files.items()
    ]
    manifest = {
        "schema_version": 1,
        "manifest_kind": "pytorch_pipeline_assignment",
        "model_id": model_id,
        "model_sha256": "a" * 64,
        "model_type": "qwen2",
        "total_layers": 2,
        "config_id": config_id,
        "plan_id": plan_id,
        "node_id": node_id,
        "layer_range": [0, 1],
        "has_embedding": False,
        "has_lm_head": False,
        "files": files,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"},
            ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        ).encode()
    ).hexdigest()

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            requests.append(parsed.path)
            if parsed.path.startswith(f"/api/models/pipeline-assignment/{model_id}"):
                payload = json.dumps(manifest).encode()
            else:
                relative = unquote(parsed.path.split(f"/api/models/files/{model_id}/", 1)[1])
                payload = source_files[relative]
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(
        model_sync, "resolve_worker_model_path",
        lambda _model_id: str(tmp_path / "models" / model_id),
    )
    try:
        destination, received = model_sync.ensure_pipeline_assignment_available(
            "127.0.0.1", server.server_port,
            {
                "model_id": model_id, "config_id": config_id, "plan_id": plan_id,
                "node_id": node_id, "model_sha256": "a" * 64,
                "start_layer": 0, "end_layer": 1, "total_layers": 2,
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert received["manifest_sha256"] == manifest["manifest_sha256"]
    root = __import__("pathlib").Path(destination)
    assert (root / "config.json").read_bytes() == source_files["config.json"]
    assert (root / "model.safetensors").read_bytes() == source_files["model.safetensors"]
    assert (root / "assignment.manifest.json").is_file()
    assert all("/api/models/files/" in path for path in requests if "/api/models/files/" in path)


def test_assignment_download_resumes_with_http_range(tmp_path, monkeypatch):
    import model_sync

    payload = b"0123456789"
    model_id = "range-assignment"
    model_path = tmp_path / "model"
    requested_ranges = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path.endswith("/api/models/files/range-assignment/model.safetensors"):
                requested_ranges.append(self.headers.get("Range"))
                value = self.headers.get("Range")
                if value:
                    start = int(value.split("=", 1)[1].split("-", 1)[0])
                    body = payload[start:]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(payload)-1}/{len(payload)}")
                else:
                    body = payload
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def log_message(self, *args):
            pass

    (model_path / "model.safetensors.part").parent.mkdir(parents=True)
    (model_path / "model.safetensors.part").write_bytes(payload[:4])
    monkeypatch.setattr(model_sync, "resolve_worker_model_path", lambda _mid: str(model_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model_sync._download_file(
            f"http://127.0.0.1:{server.server_port}/api/models/files/{model_id}/model.safetensors",
            model_path / "model.safetensors", len(payload),
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert requested_ranges == ["bytes=4-"]
    assert (model_path / "model.safetensors").read_bytes() == payload


def test_assignment_cache_reconcile_removes_stale_and_preserves_active(tmp_path, monkeypatch):
    import model_sync

    root = tmp_path / "model" / ".pipeline_assignments"
    active = root / "cfg-active-worker"
    stale = root / "cfg-stale-worker"
    active.mkdir(parents=True)
    stale.mkdir(parents=True)
    (active / "assignment.manifest.json").write_bytes(b"active")
    (stale / "shard.safetensors.part").write_bytes(b"partial")
    old = 1_000.0
    os.utime(stale, (old, old))
    monkeypatch.setattr(model_sync, "resolve_worker_model_path", lambda _mid: str(tmp_path / "model"))
    monkeypatch.setattr(model_sync, "_assignment_cache_limits", lambda: (1024 * 1024, 0, 60.0))

    result = model_sync.reconcile_pipeline_assignment_cache(
        "fixture", active_config_ids={"cfg-active"}, now=2_000.0,
    )

    assert "cfg-stale-worker" in result["removed"]
    assert (active / "assignment.manifest.json").is_file()
    assert not stale.exists()


def test_assignment_cache_reconcile_applies_lru_budget(tmp_path, monkeypatch):
    import model_sync

    root = tmp_path / "model" / ".pipeline_assignments"
    first = root / "cfg-first-worker"
    second = root / "cfg-second-worker"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "data").write_bytes(b"123456")
    (second / "data").write_bytes(b"abcdef")
    os.utime(first, (100.0, 100.0))
    os.utime(second, (200.0, 200.0))
    monkeypatch.setattr(model_sync, "resolve_worker_model_path", lambda _mid: str(tmp_path / "model"))
    monkeypatch.setattr(model_sync, "_assignment_cache_limits", lambda: (8, 0, 1000.0))

    result = model_sync.reconcile_pipeline_assignment_cache("fixture", now=300.0)

    assert "cfg-first-worker" in result["removed"]
    assert second.exists()
    assert result["bytes_after"] <= 8


def test_assignment_cache_reconcile_protects_inflight_directory_without_marker(
        tmp_path, monkeypatch):
    import model_sync

    root = tmp_path / "model" / ".pipeline_assignments"
    inflight = root / "cfg-inflight-worker"
    inflight.mkdir(parents=True)
    (inflight / "shard.safetensors.part").write_bytes(b"partial")
    os.utime(inflight, (1_000.0, 1_000.0))
    monkeypatch.setattr(model_sync, "resolve_worker_model_path", lambda _mid: str(tmp_path / "model"))
    monkeypatch.setattr(model_sync, "_assignment_cache_limits", lambda: (1024 * 1024, 0, 60.0))

    result = model_sync.reconcile_pipeline_assignment_cache(
        "fixture", active_assignment_dirs={inflight.name}, now=2_000.0,
    )

    assert inflight.name in result["protected"]
    assert inflight.exists()


def test_assignment_sync_rejects_insufficient_disk_budget(tmp_path, monkeypatch):
    import model_sync

    monkeypatch.setattr(model_sync, "resolve_worker_model_path", lambda _mid: str(tmp_path / "model"))
    monkeypatch.setattr(model_sync, "_assignment_cache_limits", lambda: (1024, 4096, 60.0))
    monkeypatch.setattr(
        model_sync.shutil, "disk_usage",
        lambda _path: type("Usage", (), {"free": 100})(),
    )
    destination = tmp_path / "model" / ".pipeline_assignments" / "cfg-node"
    destination.mkdir(parents=True)
    try:
        model_sync._ensure_assignment_disk_budget(destination, 256)
    except RuntimeError as exc:
        assert "insufficient" in str(exc)
    else:
        raise AssertionError("insufficient disk budget was accepted")
