"""Real tiny GGUF/Safetensors and loopback HTTP tests for model supply."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import urllib.parse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

import model_supply_assets as assets
from model_supply_assets import (
    SupplyError, build_gguf_segment, build_safetensors_segment,
    receive_segment, sha256_file, validate_supply_manifest,
)
from relay_segment_info import read_gguf_tokenizer_sha256


def _canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()


def _sign(payload):
    payload = dict(payload)
    payload.pop("local_files", None)
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def _model_sha(root):
    digest = hashlib.sha256()
    suffixes = (".safetensors", ".bin", ".json", ".py", ".tiktoken", ".model", ".txt", ".jinja", ".spm", ".vocab")
    for path in sorted(root.rglob("*")):
        if (not path.is_file() or path.name in {"model.sha256", "model.sha256.meta.json"}
                or path.name.endswith(".part") or not path.name.lower().endswith(suffixes)
                or any(part.startswith(".") for part in path.relative_to(root).parts)):
            continue
        digest.update(f"{path.relative_to(root).as_posix()}\0{path.stat().st_size}\0{sha256_file(path)}\n".encode())
    return digest.hexdigest()


def _make_gguf(path, *, mtp=False, hybrid=False, head_count_kv=None, key_length=None):
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")
    writer = gguf.GGUFWriter(str(path), "qwen2")
    writer.add_block_count(4 + int(mtp))
    writer.add_embedding_length(8)
    for suffix, value in (("head_count_kv", head_count_kv), ("key_length", key_length)):
        if value is not None:
            if isinstance(value, list):
                writer.add_key_value(f"qwen2.attention.{suffix}", value,
                                     gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32)
            else:
                writer.add_key_value(f"qwen2.attention.{suffix}", value, gguf.GGUFValueType.UINT32)
    if mtp:
        writer.add_key_value("qwen2.nextn_predict_layers", 1, gguf.GGUFValueType.UINT32)
    if hybrid:
        writer.add_key_value("qwen2.full_attention_interval", 2, gguf.GGUFValueType.UINT32)
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(["hello", "world", "!"])
    writer.add_tensor("token_embd.weight", np.arange(64, dtype=np.float32).reshape(8, 8))
    writer.add_tensor("output_norm.weight", np.arange(8, dtype=np.float32))
    for layer in range(4):
        writer.add_tensor(f"blk.{layer}.attn_norm.weight", np.full(8, layer + 1, dtype=np.float32))
    if mtp:
        writer.add_tensor("blk.4.nextn.weight", np.ones(8, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture
def gguf_source(tmp_path):
    source = tmp_path / "source.gguf"
    _make_gguf(source)
    return source


def _kwargs(source, start=1, end=3):
    is_gguf = source.is_file()
    return dict(
        source_model_id="tiny.qwen2:v1",
        source_model_sha256=sha256_file(source) if is_gguf else _model_sha(source),
        tokenizer_sha256=read_gguf_tokenizer_sha256(source) if is_gguf else sha256_file(source / "tokenizer.json"),
        start_layer=start, end_layer=end, total_layers=4,
        operation_id="op-1", node_id="worker-1",
        conversion_proof={"converter_revision": "tiny-v1"} if is_gguf else None,
    )


@pytest.fixture
def gguf_segment(gguf_source, tmp_path):
    return build_gguf_segment(gguf_source, tmp_path / "supply", **_kwargs(gguf_source))


def _safetensors_fixture(root, *, sharded=False, tied=False, multimodal=False):
    np = pytest.importorskip("numpy")
    save_file = pytest.importorskip("safetensors.numpy").save_file
    root.mkdir()
    decoder = {"num_hidden_layers": 4, "hidden_size": 8, "tie_word_embeddings": tied,
               "layer_types": ["full_attention"] * 4}
    prefix = "model.language_model." if multimodal else "model."
    config = {"model_type": "qwen3_5", "text_config": decoder} if multimodal else {"model_type": "qwen2", **decoder}
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer.json").write_text('{"vocab":{"hello":0}}', encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    (root / "nested").mkdir()
    (root / "nested" / "vocab.txt").write_text("hello\nworld", encoding="utf-8")
    (root / "empty.txt").write_bytes(b"")
    tensors = {
        prefix + "embed_tokens.weight": np.arange(64, dtype=np.float32).reshape(8, 8),
        prefix + "norm.weight": np.arange(8, dtype=np.float32),
    }
    for layer in range(4):
        tensors[f"{prefix}layers.{layer}.self_attn.q_proj.weight"] = np.full((128, 8), layer + 1, dtype=np.float32)
    if not tied:
        tensors["lm_head.weight"] = np.ones((8, 8), dtype=np.float32)
    if multimodal:
        tensors["model.visual.weight"] = np.ones((128, 8), dtype=np.float32)
        tensors["mtp.weight"] = np.ones((128, 8), dtype=np.float32)
    if sharded:
        items = list(tensors.items())
        weight_map = {}
        for number, partition in enumerate((items[::2], items[1::2]), 1):
            filename = f"model-{number:05d}-of-00002.safetensors"
            save_file(dict(partition), str(root / filename))
            weight_map.update({key: filename for key, _ in partition})
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    else:
        save_file(tensors, str(root / "model.safetensors"))
    # These identities are deliberately invalid for a cropped directory.
    (root / "cross_format.provenance.json").write_text('{"target":{"path":"../missing.gguf"}}', encoding="utf-8")
    (root / "model.manifest.json").write_text('{"files":["full-weights"]}', encoding="utf-8")
    (root / ".qlh-model-asset.json").write_text("{}", encoding="utf-8")
    (root / "model.sha256").write_text("stale", encoding="utf-8")
    (root / "model.sha256.meta.json").write_text("{}", encoding="utf-8")
    return tensors


@pytest.mark.parametrize("span,mode", [((0, 2), "head"), ((1, 3), "middle"), ((2, 4), "tail")])
def test_real_gguf_segment_and_native_pair(gguf_source, tmp_path, span, mode):
    gguf = pytest.importorskip("gguf")
    manifest = build_gguf_segment(gguf_source, tmp_path / "out", **_kwargs(gguf_source, *span))
    public = validate_supply_manifest(manifest)
    assert public["manifest_kind"] == "qlh.segment_supply.v1"
    assert public["segment_mode"] == mode
    assert public["layer_weight_bytes"] == [32 if span[0] <= i < span[1] else 0 for i in range(4)]
    assert public["manifest_sha256"] == _sign(public)["manifest_sha256"]
    assert "local_files" not in public
    assert str(tmp_path) not in json.dumps(public)
    reader = gguf.GGUFReader(manifest["local_files"]["segment.gguf"])
    names = {tensor.name for tensor in reader.tensors}
    assert {f"blk.{i}.attn_norm.weight" for i in range(span[1] - span[0])} <= names
    for tensor in reader.tensors:
        if tensor.name.startswith("blk."):
            layer = int(tensor.name.split(".")[1])
            assert (tensor.data == layer + span[0] + 1).all()
    native = json.loads(Path(manifest["local_files"]["segment.manifest.json"]).read_text())
    assert "source" not in native
    assert native["artifact"] == "segment.gguf"
    assert native["source_layer_range"] == list(span)
    assert native["mode"] == mode
    assert str(tmp_path) not in json.dumps(native)


def test_gguf_tail_drops_mtp(tmp_path):
    source = tmp_path / "mtp.gguf"
    _make_gguf(source, mtp=True)
    result = build_gguf_segment(source, tmp_path / "out", **_kwargs(source, 2, 4))
    metadata, tensors = assets._gguf_table(Path(result["local_files"]["segment.gguf"]))
    assert metadata["qwen2.nextn_predict_layers"] == 0
    assert not any("nextn" in name for name in tensors)
    native = json.loads(Path(result["local_files"]["segment.manifest.json"]).read_text())
    assert native["mode"] == result["segment_mode"] == "tail"
    assert native["cutter_mode"] == "middle"
    assert native["kept_block_count"] == 2
    assert native["source_layer_range"] == [2, 4]
    assert native["artifact_sha256"] == sha256_file(Path(result["local_files"]["segment.gguf"]))
    assets._verify_segment(Path(result["local_files"]["segment.gguf"]).parent, result)


@pytest.mark.parametrize("field,value,code", [
    ("source_model_sha256", "0" * 64, "source_digest_mismatch"),
    ("tokenizer_sha256", "0" * 64, "tokenizer_digest_mismatch"),
    ("total_layers", 5, "layer_range_mismatch"),
    ("end_layer", 9, "invalid_range"),
])
def test_gguf_rejects_source_identity(gguf_source, tmp_path, field, value, code):
    args = _kwargs(gguf_source)
    args[field] = value
    with pytest.raises(SupplyError) as exc:
        build_gguf_segment(gguf_source, tmp_path / "out", **args)
    assert exc.value.code == code
    assert not list((tmp_path / "out").glob("segment-*"))


def test_gguf_reuse_hashes_live_file_even_with_preserved_stat(gguf_source, tmp_path, monkeypatch):
    out = tmp_path / "out"
    first = build_gguf_segment(gguf_source, out, **_kwargs(gguf_source))
    monkeypatch.setattr(assets, "_run_cut", lambda *args: pytest.fail("reuse must not cut again"))
    assert build_gguf_segment(gguf_source, out, **_kwargs(gguf_source)) == first
    artifact = Path(first["local_files"]["segment.gguf"])
    stat = artifact.stat()
    data = bytearray(artifact.read_bytes())
    data[-1] ^= 1
    artifact.write_bytes(data)
    import os
    os.utime(artifact, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(SupplyError, match="digest mismatch"):
        build_gguf_segment(gguf_source, out, **_kwargs(gguf_source))


def _finished_cut(returncode=0, stderr=""):
    return SimpleNamespace(returncode=returncode, communicate=lambda **kw: ("", stderr), poll=lambda: returncode)


def test_cut_failure_does_not_publish(gguf_source, tmp_path, monkeypatch):
    monkeypatch.setattr(assets.subprocess, "Popen", lambda *a, **k: _finished_cut(2))
    with pytest.raises(SupplyError) as exc:
        build_gguf_segment(gguf_source, tmp_path / "out", **_kwargs(gguf_source))
    assert exc.value.code == "cut_failed"
    assert list((tmp_path / "out").iterdir()) == []


def test_frozen_cut_dispatch(gguf_source, tmp_path, monkeypatch):
    command = []
    monkeypatch.setattr(assets.sys, "frozen", True, raising=False)
    def run(argv, **kwargs):
        command.extend(argv)
        return _finished_cut()
    monkeypatch.setattr(assets.subprocess, "Popen", run)
    payload = assets._base_manifest(**_kwargs(gguf_source))
    assets._run_cut(gguf_source, tmp_path / "out.gguf", tmp_path / "native.json", payload, 0)
    assert command[:2] == [assets.sys.executable, "--qlh-cut-layers"]
    assert command[command.index("--end") + 1] == "3"


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("platform_name", ["win32", "linux"])
def test_cut_creationflags_hide_windows_console(gguf_source, tmp_path, monkeypatch, frozen, platform_name):
    payload = assets._base_manifest(**_kwargs(gguf_source))
    calls = []
    def run(argv, **options):
        calls.append((argv, options))
        return _finished_cut()
    monkeypatch.setattr(assets.sys, "frozen", frozen, raising=False)
    monkeypatch.setattr(assets.sys, "platform", platform_name)
    # The constant is unavailable on non-Windows hosts running this regression.
    monkeypatch.setattr(assets.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(assets.subprocess, "Popen", run)
    assets._run_cut(gguf_source, tmp_path / "segment.gguf", tmp_path / "segment.manifest.json", payload, 0)
    command, options = calls[0]
    assert options["creationflags"] == (0x08000000 if platform_name == "win32" else 0)
    assert options["stdout"] == assets.subprocess.PIPE
    assert options["stderr"] == assets.subprocess.PIPE
    assert command[0] == assets.sys.executable
    assert (command[1] == "--qlh-cut-layers") == frozen


@pytest.mark.parametrize("sharded,span,tied,multimodal", [
    (False, (1, 3), False, False), (True, (0, 2), False, False),
    (True, (2, 4), True, False), (False, (1, 3), False, True),
])
def test_safetensors_copies_only_selected_tensor_bytes(tmp_path, monkeypatch, sharded, span, tied, multimodal):
    root = tmp_path / "hf"
    original = _safetensors_fixture(root, sharded=sharded, tied=tied, multimodal=multimodal)
    safetensors = pytest.importorskip("safetensors")
    actual_safe_open = safetensors.safe_open
    monkeypatch.setattr(safetensors, "safe_open", lambda *a, **k: pytest.fail("builder may not materialize tensors"))
    result = build_safetensors_segment(root, tmp_path / "out", **_kwargs(root, *span))
    monkeypatch.setattr(safetensors, "safe_open", actual_safe_open)
    prefix = "model.language_model." if multimodal else "model."
    expected = {key for key in original if key.startswith(tuple(f"{prefix}layers.{i}." for i in range(*span)))}
    expected.add(prefix + "norm.weight")
    if span[0] == 0 or (span[1] == 4 and tied):
        expected.add(prefix + "embed_tokens.weight")
    if span[1] == 4 and not tied:
        expected.add("lm_head.weight")
    found = set()
    output_bytes = 0
    for entry in result["files"]:
        assert Path(result["local_files"][entry["path"]]).is_absolute()
        if entry["kind"] != "weights":
            continue
        assert entry["path"].startswith("assignment-")
        with actual_safe_open(result["local_files"][entry["path"]], framework="np") as handle:
            for key in handle.keys():
                found.add(key)
                tensor = handle.get_tensor(key)
                assert (tensor == original[key]).all()
                output_bytes += tensor.nbytes
    assert found == expected
    assert output_bytes == sum(original[key].nbytes for key in expected)
    assert output_bytes < sum(tensor.nbytes for tensor in original.values())
    assert result["layer_weight_bytes"] == [4096 if span[0] <= i < span[1] else 0 for i in range(4)]
    assert Path(result["local_files"]["config.json"]).read_bytes() == (root / "config.json").read_bytes()
    names = set(result["local_files"])
    assert not names & {"cross_format.provenance.json", "model.manifest.json", ".qlh-model-asset.json", "model.sha256", "model.sha256.meta.json"}
    filtered = json.loads(Path(result["local_files"]["model.safetensors.index.json"]).read_text())
    assert set(filtered["weight_map"]) == expected
    assert filtered["metadata"]["total_size"] == output_bytes
    assets._verify_segment(Path(result["local_files"]["config.json"]).parent,
                           validate_supply_manifest(result))
    assert (root / "model.sha256").read_text() == "stale"
    assert validate_supply_manifest(result)["format"] == "safetensors"


@contextmanager
def _server(manifest, *, failure=None):
    state = {"requests": [], "failure": failure, "failed": False}
    files = {name: Path(path).read_bytes() for name, path in manifest["local_files"].items()}
    prefix = f"/api/models/supply/{manifest['operation_id']}/files/{manifest['node_id']}/"
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *args):
            pass
        def do_GET(self):
            state["requests"].append((self.path, self.headers.get("Range"), self.headers.get("X-QLH-Supply-Token")))
            if not self.path.startswith(prefix) or self.headers.get("X-QLH-Supply-Token") != "secret-token":
                self.send_error(403)
                return
            name = urllib.parse.unquote(self.path[len(prefix):])
            if name not in files:
                self.send_error(404)
                return
            data = files[name]
            offset = int(self.headers["Range"].removeprefix("bytes=").removesuffix("-"))
            fail = state["failure"] if not state["failed"] else None
            if fail:
                state["failed"] = True
            if fail == "redirect":
                self.send_response(302)
                self.send_header("Location", "/elsewhere")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 200 if fail == "status" else 206
            self.send_response(status)
            range_start = offset + 1 if fail == "range" else offset
            self.send_header("Content-Range", f"bytes {range_start}-{len(data)-1}/{len(data)}")
            self.send_header("Content-Length", str(len(data) - offset + (1 if fail == "length" else 0)))
            self.send_header("Connection", "close")
            self.end_headers()
            body = data[offset:]
            if fail == "digest":
                body = bytes([body[0] ^ 1]) + body[1:]
            if fail == "truncate":
                body = body[:max(1, len(body) // 2)]
            try:
                self.wfile.write(body)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            self.close_connection = True
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _receive(manifest, root, url, **kwargs):
    return receive_segment(manifest, root, base_url=url, token="secret-token", **kwargs)


def _assert_transfer_accounting(value):
    assert value["downloaded_bytes"] == (
        value["actual_downloaded_bytes"] + value["reused_bytes"] + value["resumed_bytes"]
    )
    assert value["downloaded_bytes"] == value["completed_bytes"]
    assert 0 <= value["downloaded_bytes"] <= value["total_bytes"]
    assert all(type(value[key]) is int and value[key] >= 0 for key in (
        "downloaded_bytes", "actual_downloaded_bytes", "reused_bytes", "resumed_bytes",
    ))


def _assert_unpublished(root):
    assert not list(root.glob("segment-*"))
    assert not list(root.rglob("supply.manifest.json"))


def test_http_receives_native_pair_and_reuses_with_full_revalidation(gguf_segment, tmp_path):
    root = tmp_path / "received"
    events = []
    with _server(gguf_segment) as (url, state):
        result = _receive(gguf_segment, root, url, progress=events.append)
        _assert_transfer_accounting(result)
        for event in events:
            _assert_transfer_accounting(event)
        assert Path(result["model_path"]).is_file()
        assert result["manifest"] == validate_supply_manifest(gguf_segment)
        assert Path(result["manifest_path"]).is_file()
        assert not result["reused"]
        total = sum(entry["size_bytes"] for entry in gguf_segment["files"])
        assert result["actual_downloaded_bytes"] == total
        assert result["reused_bytes"] == 0
        assert events[-1]["actual_downloaded_bytes"] == events[-1]["total_bytes"]
        requests = list(state["requests"])
        reuse_events = []
        reused = _receive(gguf_segment, root, url, progress=reuse_events.append)
        _assert_transfer_accounting(reused)
        for event in reuse_events:
            _assert_transfer_accounting(event)
        assert reused["reused"]
        assert reused["actual_downloaded_bytes"] == 0
        assert reused["reused_bytes"] == total
        assert reuse_events[-1]["completed_bytes"] == total
        assert reuse_events[-1]["reused_bytes"] == total
        assert state["requests"] == requests
        artifact = Path(result["model_path"])
        raw = bytearray(artifact.read_bytes())
        raw[-1] ^= 1
        artifact.write_bytes(raw)
        with pytest.raises(SupplyError) as exc:
            _receive(gguf_segment, root, url)
        assert exc.value.code == "file_digest_mismatch"
    assert all(path.startswith("/api/models/supply/op-1/files/worker-1/")
               and range_header == "bytes=0-" and token == "secret-token"
               for path, range_header, token in requests)


def test_http_interrupted_download_resumes_real_partial(gguf_segment, tmp_path):
    root = tmp_path / "received"
    with _server(gguf_segment, failure="truncate") as (url, state):
        with pytest.raises(SupplyError) as exc:
            _receive(gguf_segment, root, url)
        assert exc.value.code in {"download_incomplete", "download_failed"}
        _assert_unpublished(root)
        part = next(root.rglob("*.part"))
        partial_size = part.stat().st_size
        assert partial_size > 0
        events = []
        result = _receive(gguf_segment, root, url, progress=events.append)
        _assert_transfer_accounting(result)
        for event in events:
            _assert_transfer_accounting(event)
        assert Path(result["model_path"]).is_file()
        assert state["requests"][1][1] == f"bytes={partial_size}-"
        total = sum(entry["size_bytes"] for entry in gguf_segment["files"])
        assert result["actual_downloaded_bytes"] == total - partial_size
        assert result["reused_bytes"] == 0
        assert result["resumed_bytes"] == partial_size
        assert events[-1]["actual_downloaded_bytes"] == total - partial_size
        assert events[-1]["reused_bytes"] == 0
        assert events[-1]["resumed_bytes"] == partial_size
        assert events[-1]["completed_bytes"] == total
        assert not list(root.rglob("*.part"))


@pytest.mark.parametrize("failure,code", [
    ("status", "http_range_mismatch"), ("range", "http_range_mismatch"),
    ("length", "http_size_mismatch"), ("digest", "file_digest_mismatch"),
    ("redirect", "http_redirect_rejected"),
])
def test_http_protocol_failure_never_publishes(gguf_segment, tmp_path, failure, code):
    root = tmp_path / "received"
    with _server(gguf_segment, failure=failure) as (url, state):
        with pytest.raises(SupplyError) as exc:
            _receive(gguf_segment, root, url)
        assert exc.value.code == code
        assert len(state["requests"]) == 1
        _assert_unpublished(root)
        # A later correct response can recover, including a rejected digest.
        assert Path(_receive(gguf_segment, root, url)["model_path"]).is_file()


def test_http_rejects_range_for_existing_partial_without_overwriting(gguf_segment, tmp_path):
    root = tmp_path / "received"
    stage = root / (".supply-receive-" + gguf_segment["manifest_sha256"])
    stage.mkdir(parents=True)
    part = stage / "segment.gguf.part"
    original = Path(gguf_segment["local_files"]["segment.gguf"]).read_bytes()[:17]
    part.write_bytes(original)
    with _server(gguf_segment, failure="range") as (url, state):
        with pytest.raises(SupplyError) as exc:
            _receive(gguf_segment, root, url)
        assert exc.value.code == "http_range_mismatch"
        assert part.read_bytes() == original
        assert state["requests"][0][1] == "bytes=17-"
        _assert_unpublished(root)


def test_cancellation_during_transfer_preserves_part_and_never_publishes(gguf_segment, tmp_path, monkeypatch):
    root = tmp_path / "received"
    monkeypatch.setattr(assets, "_CHUNK", 64)
    stop = [False]
    def progress(event):
        if event["file_bytes"]:
            stop[0] = True
    with _server(gguf_segment) as (url, state):
        with pytest.raises(SupplyError) as exc:
            _receive(gguf_segment, root, url, cancelled=lambda: stop[0], progress=progress)
        assert exc.value.code == "cancelled"
        _assert_unpublished(root)
        part = next(root.rglob("*.part"))
        partial_size = part.stat().st_size
        assert 0 < partial_size < Path(gguf_segment["local_files"]["segment.gguf"]).stat().st_size
        assert Path(_receive(gguf_segment, root, url)["model_path"]).is_file()
        assert state["requests"][1][1] == f"bytes={partial_size}-"


def test_http_safetensors_receipt_preserves_config_and_filters_full_identity(tmp_path):
    root = tmp_path / "hf"
    _safetensors_fixture(root, sharded=True)
    segment = build_safetensors_segment(root, tmp_path / "supply", **_kwargs(root, 1, 3))
    with _server(segment) as (url, state):
        result = _receive(segment, tmp_path / "received", url)
    model = Path(result["model_path"])
    assert model.is_dir()
    assert (model / "config.json").read_bytes() == (root / "config.json").read_bytes()
    assert not (model / "cross_format.provenance.json").exists()
    assert (model / "empty.txt").read_bytes() == b""
    assert any(path.endswith("/nested/vocab.txt") for path, _, _ in state["requests"])
    assert not any(path.endswith("/empty.txt") for path, _, _ in state["requests"])


@pytest.mark.parametrize("path", [
    "../x", "/x", "C:/x", "C:\\x", "a/../x", "a//x", "a/./x", "\\\\host\\x",
    "a%2fb", "a?b", "a:b", "CON.txt", "a/aux", "supply.manifest.json",
])
def test_manifest_rejects_unsafe_paths(gguf_segment, path):
    manifest = validate_supply_manifest(gguf_segment)
    manifest["files"][0]["path"] = path
    with pytest.raises(SupplyError) as exc:
        validate_supply_manifest(_sign(manifest))
    assert exc.value.code in {"unsafe_path", "unsafe_manifest"}


def test_manifest_private_fields_are_removed_and_public_tampering_rejected(gguf_segment):
    gguf_segment["_private"] = {"source": "C:/private/source.gguf"}
    public = validate_supply_manifest(gguf_segment)
    assert not any(key.startswith("_") for key in public)
    assert "local_files" not in public
    public["operation_id"] = "op-2"
    with pytest.raises(SupplyError) as exc:
        validate_supply_manifest(public)
    assert exc.value.code == "manifest_digest_mismatch"


def test_receive_rechecks_actual_layer_bytes_before_publishing(gguf_segment, tmp_path):
    wrong = validate_supply_manifest(gguf_segment)
    wrong["layer_weight_bytes"][1] += 4
    wrong = _sign(wrong)
    wrong["local_files"] = gguf_segment["local_files"]
    root = tmp_path / "received"
    with _server(wrong) as (url, _):
        with pytest.raises(SupplyError) as exc:
            _receive(wrong, root, url)
    assert exc.value.code == "layer_bytes_mismatch"
    _assert_unpublished(root)


def test_receive_checks_native_manifest_binding(gguf_segment, tmp_path):
    native_path = Path(gguf_segment["local_files"]["segment.manifest.json"])
    native = json.loads(native_path.read_text())
    native["source_layer_range"] = [0, 2]
    native_path.write_bytes(_canonical(native))
    public = validate_supply_manifest(gguf_segment)
    for entry in public["files"]:
        if entry["kind"] == "manifest":
            entry["size_bytes"] = native_path.stat().st_size
            entry["sha256"] = sha256_file(native_path)
    wrong = _sign(public)
    wrong["local_files"] = gguf_segment["local_files"]
    root = tmp_path / "received"
    with _server(wrong) as (url, _):
        with pytest.raises(SupplyError) as exc:
            _receive(wrong, root, url)
    assert exc.value.code == "native_manifest_mismatch"
    _assert_unpublished(root)


def test_manifest_absolute_conversion_path_is_rejected(gguf_segment):
    public = validate_supply_manifest(gguf_segment)
    public["conversion_proof"]["converter"] = "C:/private/tool.py"
    with pytest.raises(SupplyError) as exc:
        validate_supply_manifest(_sign(public))
    assert exc.value.code == "unsafe_manifest"


def test_safetensors_bad_offsets_rejected_without_publish(tmp_path):
    root = tmp_path / "hf"
    _safetensors_fixture(root)
    shard = root / "model.safetensors"
    raw = shard.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + length])
    key = next(key for key in header if key != "__metadata__")
    header[key]["data_offsets"][1] += 1
    encoded = _canonical(header)
    encoded += b" " * (-len(encoded) % 8)
    shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + length:])
    with pytest.raises(SupplyError) as exc:
        build_safetensors_segment(root, tmp_path / "out", **_kwargs(root))
    assert exc.value.code == "invalid_safetensors"
    assert not list((tmp_path / "out").glob("segment-*"))


def test_safetensors_source_aggregate_matches_model_sync(tmp_path):
    root = tmp_path / "hf"
    _safetensors_fixture(root)
    from model_sync import compute_model_sha256
    assert assets._model_digest(root) == compute_model_sha256(str(root), use_cache=False)


@pytest.mark.parametrize("builder_name", ["gguf", "safetensors"])
def test_builders_forbid_full_model_supply(tmp_path, builder_name):
    if builder_name == "gguf":
        source = tmp_path / "source.gguf"
        _make_gguf(source)
        builder = build_gguf_segment
    else:
        source = tmp_path / "hf"
        _safetensors_fixture(source)
        builder = build_safetensors_segment
    with pytest.raises(SupplyError) as exc:
        builder(source, tmp_path / "out", **_kwargs(source, 0, 4))
    assert exc.value.code == "full_model_forbidden"
    assert not (tmp_path / "out").exists()


def test_validator_forbids_full_model_supply(gguf_segment):
    public = validate_supply_manifest(gguf_segment)
    public["layer_range"] = [0, 4]
    public["segment_mode"] = "full"
    public["layer_weight_bytes"] = [32] * 4
    with pytest.raises(SupplyError) as exc:
        validate_supply_manifest(_sign(public))
    assert exc.value.code == "full_model_forbidden"


def _cross_format_fixture(root, target, *, omit_optional=False):
    _safetensors_fixture(root, sharded=True)
    _make_gguf(target)
    names = sorted(path.relative_to(root).as_posix() for path in root.rglob("*")
                   if path.is_file() and path.name not in {
                       "model.manifest.json", "cross_format.provenance.json",
                       ".qlh-model-asset.json", "model.sha256", "model.sha256.meta.json",
                   })
    if omit_optional:
        names = [name for name in names if name in {
            "config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
        } or name.endswith(".safetensors")]
    source_manifest = {"schema": 1, "model_type": "safetensors", "files": [
        {"path": name, "size_bytes": (root / name).stat().st_size,
         "sha256": sha256_file(root / name)} for name in names
    ]}
    source_manifest["manifest_sha256"] = hashlib.sha256(_canonical(source_manifest)).hexdigest()
    source_manifest["artifact_sha256"] = hashlib.sha256("\n".join(
        f"{entry['path']}\0{entry['size_bytes']}\0{entry['sha256']}" for entry in source_manifest["files"]
    ).encode()).hexdigest()
    (root / "model.manifest.json").write_bytes(_canonical(source_manifest))
    proof = {
        "schema_version": 1, "kind": "qlh.cross_format_pipeline_identity.v1",
        "source": {"artifact_sha256": source_manifest["artifact_sha256"],
                   "manifest_sha256": source_manifest["manifest_sha256"],
                   "tokenizer_sha256": sha256_file(root / "tokenizer.json")},
        "conversion": {"output_type": "F32", "converter_revision": "a" * 40,
                       "converter_sha256": "b" * 64, "quantizer_revision": "c" * 40,
                       "quantizer_sha256": "d" * 64},
        "target": {"path": "../" + target.name, "artifact_sha256": sha256_file(target),
                   "tokenizer_sha256": read_gguf_tokenizer_sha256(target)},
    }
    proof["provenance_sha256"] = hashlib.sha256(_canonical(proof)).hexdigest()
    (root / "cross_format.provenance.json").write_bytes(_canonical(proof))
    args = _kwargs(root)
    args.update(source_model_sha256=proof["target"]["artifact_sha256"],
                tokenizer_sha256=proof["target"]["tokenizer_sha256"], conversion_proof=proof)
    return args


def test_cross_format_safetensors_binds_gguf_identity_and_source_tokenizer(tmp_path):
    root, target = tmp_path / "hf", tmp_path / "converted.gguf"
    args = _cross_format_fixture(root, target)
    segment = build_safetensors_segment(root, tmp_path / "supply", **args)
    assert segment["source_model_sha256"] == sha256_file(target)
    assert segment["tokenizer_sha256"] == read_gguf_tokenizer_sha256(target)
    assert segment["conversion_proof"]["source"]["tokenizer_sha256"] == sha256_file(root / "tokenizer.json")
    assert segment["tokenizer_sha256"] != sha256_file(root / "tokenizer.json")
    assert "cross_format.provenance.json" not in segment["local_files"]
    # Receiving a pack must never follow proof.target.path relative to the crop.
    target.unlink()
    with _server(segment) as (url, _):
        result = _receive(segment, tmp_path / "received", url)
    assert Path(result["model_path"]).is_dir()
    assert result["manifest"]["conversion_proof"] == args["conversion_proof"]


@pytest.mark.parametrize("tamper", ["source", "target", "proof", "coverage"])
def test_cross_format_safetensors_reverifies_all_source_files(tmp_path, tamper):
    root, target = tmp_path / "hf", tmp_path / "converted.gguf"
    args = _cross_format_fixture(root, target)
    if tamper == "source":
        shard = next(root.glob("*.safetensors"))
        data = bytearray(shard.read_bytes())
        data[-1] ^= 1
        shard.write_bytes(data)
    elif tamper == "target":
        raw = bytearray(target.read_bytes())
        raw[-1] ^= 1
        target.write_bytes(raw)
    elif tamper == "proof":
        args["conversion_proof"]["conversion"]["converter_sha256"] = "e" * 64
    else:
        manifest_path = root / "model.manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("artifact_sha256")
        manifest.pop("manifest_sha256")
        manifest["files"] = [entry for entry in manifest["files"] if not entry["path"].endswith(".safetensors")]
        manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
        manifest["artifact_sha256"] = hashlib.sha256("\n".join(
            f"{entry['path']}\0{entry['size_bytes']}\0{entry['sha256']}" for entry in manifest["files"]
        ).encode()).hexdigest()
        manifest_path.write_bytes(_canonical(manifest))
        proof = args["conversion_proof"]
        proof["source"].update(artifact_sha256=manifest["artifact_sha256"], manifest_sha256=manifest["manifest_sha256"])
        proof.pop("provenance_sha256")
        proof["provenance_sha256"] = hashlib.sha256(_canonical(proof)).hexdigest()
        (root / "cross_format.provenance.json").write_bytes(_canonical(proof))
    with pytest.raises(SupplyError) as exc:
        build_safetensors_segment(root, tmp_path / "out", **args)
    assert exc.value.code in {"source_digest_mismatch", "conversion_proof_invalid"}
    assert not list((tmp_path / "out").glob("segment-*"))


def test_cross_format_accepts_old_proof_without_optional_support_inventory(tmp_path):
    root, target = tmp_path / "hf", tmp_path / "converted.gguf"
    args = _cross_format_fixture(root, target, omit_optional=True)
    before_proof = (root / "cross_format.provenance.json").read_bytes()
    before_manifest = (root / "model.manifest.json").read_bytes()
    (root / "merges.txt").write_text("hello world", encoding="utf-8")
    (root / "vocab.json").write_text('{"hello":0}', encoding="utf-8")
    result = build_safetensors_segment(root, tmp_path / "supply", **args)
    assert {"chat_template.jinja", "merges.txt", "vocab.json", "nested/vocab.txt"}.issubset(result["local_files"])
    for name in ("chat_template.jinja", "merges.txt", "vocab.json", "nested/vocab.txt"):
        entry = next(item for item in result["files"] if item["path"] == name)
        assert entry["sha256"] == sha256_file(root / name)
        assert entry["size_bytes"] == (root / name).stat().st_size
    assert (root / "cross_format.provenance.json").read_bytes() == before_proof
    assert (root / "model.manifest.json").read_bytes() == before_manifest


def test_receive_publication_failure_has_no_marker_or_selection(gguf_segment, tmp_path, monkeypatch):
    real_replace = assets.os.replace
    def replace(source, target):
        if Path(target).name.startswith("segment-"):
            raise OSError("simulated publication failure")
        return real_replace(source, target)
    monkeypatch.setattr(assets.os, "replace", replace)
    root = tmp_path / "received"
    with _server(gguf_segment) as (url, _):
        with pytest.raises(SupplyError) as exc:
            _receive(gguf_segment, root, url)
    assert exc.value.code == "publication_failed"
    _assert_unpublished(root)


def test_gguf_misaligned_hybrid_cut_does_not_publish(tmp_path):
    source = tmp_path / "source.gguf"
    _make_gguf(source, hybrid=True)
    with pytest.raises(SupplyError) as exc:
        build_gguf_segment(source, tmp_path / "out", **_kwargs(source, 1, 3))
    assert exc.value.code == "cut_failed"
    assert not list((tmp_path / "out").iterdir())


def test_safetensors_reuse_revalidates_live_artifacts(tmp_path, monkeypatch):
    root = tmp_path / "hf"
    _safetensors_fixture(root)
    kwargs = _kwargs(root)
    result = build_safetensors_segment(root, tmp_path / "out", **kwargs)
    monkeypatch.setattr(assets, "_copy_selected", lambda *args: pytest.fail("reuse must not repack"))
    assert build_safetensors_segment(root, tmp_path / "out", **kwargs) == result
    weight = next(Path(path) for name, path in result["local_files"].items() if name.endswith(".safetensors"))
    raw = bytearray(weight.read_bytes())
    raw[-1] ^= 1
    weight.write_bytes(raw)
    with pytest.raises(SupplyError) as exc:
        build_safetensors_segment(root, tmp_path / "out", **kwargs)
    assert exc.value.code == "file_digest_mismatch"


def test_safetensors_source_change_during_pack_is_rejected(tmp_path, monkeypatch):
    root = tmp_path / "hf"
    _safetensors_fixture(root)
    copy = assets._copy_selected
    def changed(*args):
        copy(*args)
        # Core config mutation must not silently acquire the old source SHA.
        (root / "tokenizer_config.json").write_bytes(b'{"changed":true}')
    monkeypatch.setattr(assets, "_copy_selected", changed)
    with pytest.raises(SupplyError) as exc:
        build_safetensors_segment(root, tmp_path / "out", **_kwargs(root))
    assert exc.value.code == "source_digest_mismatch"
    assert not list((tmp_path / "out").iterdir())


def test_receive_symlinked_partial_rejected(gguf_segment, tmp_path):
    root = tmp_path / "received"
    stage = root / (".supply-receive-" + gguf_segment["manifest_sha256"])
    stage.mkdir(parents=True)
    external = tmp_path / "external"
    external.write_bytes(b"unrelated")
    try:
        (stage / "segment.gguf.part").symlink_to(external)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows account")
    with _server(gguf_segment) as (url, state):
        with pytest.raises(SupplyError) as exc:
            _receive(gguf_segment, root, url)
    assert exc.value.code == "unsafe_path"
    assert not state["requests"]
    assert external.read_bytes() == b"unrelated"
    _assert_unpublished(root)


def test_disk_budget_failure_prevents_cut(gguf_source, tmp_path, monkeypatch):
    monkeypatch.setattr(assets.shutil, "disk_usage", lambda root: SimpleNamespace(free=1))
    monkeypatch.setattr(assets, "_run_cut", lambda *args: pytest.fail("cut must not start"))
    with pytest.raises(SupplyError) as exc:
        build_gguf_segment(gguf_source, tmp_path / "out", **_kwargs(gguf_source))
    assert exc.value.code == "insufficient_disk"
    assert not list((tmp_path / "out").iterdir())


def _next_operation(manifest):
    changed = validate_supply_manifest(manifest)
    changed.update(operation_id="op-2", node_id="worker-2")
    return {**_sign(changed), "local_files": dict(manifest["local_files"])}


@pytest.mark.parametrize("format_name", ["gguf", "safetensors"])
def test_cross_operation_receipt_reuses_live_assets_without_http(tmp_path, format_name):
    if format_name == "gguf":
        source = tmp_path / "source.gguf"
        _make_gguf(source)
        builder = build_gguf_segment
    else:
        source = tmp_path / "hf"
        _safetensors_fixture(source, sharded=True)
        builder = build_safetensors_segment
    kwargs = _kwargs(source)
    first = builder(source, tmp_path / "supply", **kwargs)
    second = builder(source, tmp_path / "supply", **{
        **kwargs, "operation_id": "op-2", "node_id": "worker-2",
    })
    assert first["manifest_sha256"] != second["manifest_sha256"]
    # Native GGUF metadata, filtered tensors and support bytes do not bind an
    # operation/node. Only the independently generated supply marker changes.
    assert first["files"] == second["files"]
    root = tmp_path / "received"
    with _server(first) as (url, _):
        old = _receive(first, root, url)
    events = []
    with _server(second) as (url, state):
        received = _receive(second, root, url, progress=events.append)
        _assert_transfer_accounting(received)
        for event in events:
            _assert_transfer_accounting(event)
        assert not state["requests"]
    total = sum(entry["size_bytes"] for entry in second["files"])
    assert received["manifest_path"] != old["manifest_path"]
    assert received["manifest"] == validate_supply_manifest(second)
    assert json.loads(Path(received["manifest_path"]).read_text()) == received["manifest"]
    assert received["actual_downloaded_bytes"] == 0
    assert received["reused_bytes"] == total
    assert received["completed_bytes"] == total
    assert received["reused"]
    assert events[-1]["actual_downloaded_bytes"] == 0
    assert events[-1]["reused_bytes"] == total
    assert events[-1]["completed_bytes"] == total
    for entry in second["files"]:
        assert sha256_file(received["local_files"][entry["path"]]) == entry["sha256"]
    # Immutable publications share an inode where hardlinks are supported;
    # corruption must still be rejected by fresh verification of either receipt.
    weight = next(entry for entry in second["files"] if entry["kind"] == "weights")
    assert os.path.samefile(old["local_files"][weight["path"]], received["local_files"][weight["path"]])
    Path(old["local_files"][weight["path"]]).write_bytes(b"mutated cached asset")
    with pytest.raises(SupplyError):
        assets._reuse(Path(received["manifest_path"]).parent, received["manifest"])


def test_cross_operation_downloads_only_corrupt_cached_file_with_preserved_stat(gguf_segment, tmp_path):
    root = tmp_path / "received"
    with _server(gguf_segment) as (url, _):
        old = _receive(gguf_segment, root, url)
    weight = next(entry for entry in gguf_segment["files"] if entry["kind"] == "weights")
    cached = Path(old["local_files"][weight["path"]])
    before = cached.stat()
    damaged = bytearray(cached.read_bytes())
    damaged[-1] ^= 1
    cached.write_bytes(damaged)
    os.utime(cached, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert cached.stat().st_size == before.st_size
    assert cached.stat().st_mtime_ns == before.st_mtime_ns
    second = _next_operation(gguf_segment)
    events = []
    with _server(second) as (url, state):
        received = _receive(second, root, url, progress=events.append)
        _assert_transfer_accounting(received)
        for event in events:
            _assert_transfer_accounting(event)
        assert [path for path, _, _ in state["requests"]] == [
            "/api/models/supply/op-2/files/worker-2/" + weight["path"],
        ]
    total = sum(entry["size_bytes"] for entry in second["files"])
    assert not received["reused"]
    assert received["actual_downloaded_bytes"] == weight["size_bytes"]
    assert received["reused_bytes"] == total - weight["size_bytes"]
    assert events[-1]["actual_downloaded_bytes"] == weight["size_bytes"]
    assert events[-1]["reused_bytes"] == total - weight["size_bytes"]
    assert events[-1]["completed_bytes"] == total
    assert sha256_file(received["model_path"]) == weight["sha256"]
    assert cached.read_bytes() == damaged
    assert not list(root.rglob(".supply-copy-*.part"))


def test_cross_operation_downloads_only_changed_optional_support(tmp_path):
    source, target = tmp_path / "hf", tmp_path / "converted.gguf"
    kwargs = _cross_format_fixture(source, target, omit_optional=True)
    first = build_safetensors_segment(source, tmp_path / "supply", **kwargs)
    root = tmp_path / "received"
    with _server(first) as (url, _):
        _receive(first, root, url)
    (source / "chat_template.jinja").write_text("{{ messages | reverse }}", encoding="utf-8")
    second = build_safetensors_segment(source, tmp_path / "supply", **{
        **kwargs, "operation_id": "op-2", "node_id": "worker-2",
    })
    assert first["source_model_sha256"] == second["source_model_sha256"]
    assert first["artifact_sha256"] == second["artifact_sha256"]
    changed = next(entry for entry in second["files"] if entry["path"] == "chat_template.jinja")
    events = []
    with _server(second) as (url, state):
        received = _receive(second, root, url, progress=events.append)
        _assert_transfer_accounting(received)
        for event in events:
            _assert_transfer_accounting(event)
        assert [path for path, _, _ in state["requests"]] == [
            "/api/models/supply/op-2/files/worker-2/chat_template.jinja",
        ]
    total = sum(entry["size_bytes"] for entry in second["files"])
    assert received["actual_downloaded_bytes"] == changed["size_bytes"]
    assert received["reused_bytes"] == total - changed["size_bytes"]
    assert events[-1]["completed_bytes"] == total
    assert Path(received["local_files"][changed["path"]]).read_bytes() == (source / changed["path"]).read_bytes()


@pytest.mark.parametrize("cache_state", [
    "no_marker", "bad_marker", "outside_root", "unpublished", "untrusted_wrapper",
])
def test_cross_operation_ignores_untrusted_or_foreign_inventories(gguf_segment, tmp_path, cache_state):
    root = tmp_path / "received"
    old_root = tmp_path / "other-root" if cache_state == "outside_root" else root
    with _server(gguf_segment) as (url, _):
        old = _receive(gguf_segment, old_root, url)
    marker = Path(old["manifest_path"])
    if cache_state == "no_marker":
        marker.unlink()
    elif cache_state == "bad_marker":
        invalid = json.loads(marker.read_text())
        invalid["operation_id"] = "unsigned-tamper"
        marker.write_bytes(_canonical(invalid))
    elif cache_state == "unpublished":
        marker.parent.rename(root / ".supply-unpublished")
    elif cache_state == "untrusted_wrapper":
        wrapper = root / "untrusted-storage"
        wrapper.mkdir()
        marker.parent.rename(wrapper / marker.parent.name)
    second = _next_operation(gguf_segment)
    # Incoming private paths deliberately point at old, potentially foreign
    # bytes. Only destination_root's published markers authorize cache lookup.
    second["local_files"] = old["local_files"]
    # The server serves valid source package bytes rather than cache pointers.
    served = {**second, "local_files": gguf_segment["local_files"]}
    with _server(served) as (url, state):
        received = _receive(second, root, url)
        _assert_transfer_accounting(received)
        assert len(state["requests"]) == len(second["files"])
    assert received["actual_downloaded_bytes"] == sum(entry["size_bytes"] for entry in second["files"])
    assert received["reused_bytes"] == 0


def test_cross_operation_does_not_scan_unlisted_asset_bytes(gguf_segment, tmp_path):
    root = tmp_path / "received"
    with _server(gguf_segment) as (url, _):
        old = _receive(gguf_segment, root, url)
    weight = next(entry for entry in gguf_segment["files"] if entry["kind"] == "weights")
    # Exact same bytes exist locally, but their name is outside the published
    # inventory. Finding them by recursively scanning the root is forbidden.
    Path(old["local_files"][weight["path"]]).rename(Path(old["manifest_path"]).parent / "unlisted.gguf")
    second = _next_operation(gguf_segment)
    with _server(second) as (url, state):
        received = _receive(second, root, url)
        _assert_transfer_accounting(received)
        assert len(state["requests"]) == 1
        assert state["requests"][0][0].endswith("/segment.gguf")
    assert received["actual_downloaded_bytes"] == weight["size_bytes"]
    assert received["reused_bytes"] == sum(
        entry["size_bytes"] for entry in second["files"] if entry["kind"] != "weights"
    )


def test_cross_operation_reuse_still_validates_new_segment_metadata(gguf_segment, tmp_path):
    root = tmp_path / "received"
    with _server(gguf_segment) as (url, _):
        old = _receive(gguf_segment, root, url)
    second = _next_operation(gguf_segment)
    second["layer_weight_bytes"][1] += 4
    second = {**_sign(second), "local_files": gguf_segment["local_files"]}
    with _server(second) as (url, state):
        with pytest.raises(SupplyError) as exc:
            _receive(second, root, url)
        assert exc.value.code == "layer_bytes_mismatch"
        assert not state["requests"]
    assert Path(old["manifest_path"]).is_file()
    assert not (root / ("segment-" + second["manifest_sha256"])).exists()
    assert not list(root.glob(".supply-receive-*/supply.manifest.json"))


def test_cross_operation_cancelled_local_copy_leaves_old_publication_intact(gguf_segment, tmp_path):
    root = tmp_path / "received"
    with _server(gguf_segment) as (url, _):
        old = _receive(gguf_segment, root, url)
    second = _next_operation(gguf_segment)
    stop_during_copy = lambda: bool(list(root.glob(".supply-receive-*/segment.gguf")))
    with _server(second) as (url, state):
        with pytest.raises(SupplyError) as exc:
            _receive(second, root, url, cancelled=stop_during_copy)
        assert exc.value.code == "cancelled"
        assert not state["requests"]
        assert Path(old["manifest_path"]).is_file()
        assert not (root / ("segment-" + second["manifest_sha256"])).exists()
        assert not list(root.rglob(".supply-copy-*.part"))
        assert not list(root.glob(".supply-receive-*/supply.manifest.json"))
        received = _receive(second, root, url)
        _assert_transfer_accounting(received)
        assert received["actual_downloaded_bytes"] == 0
        assert not state["requests"]
    assert received["reused_bytes"] == received["total_bytes"]


def test_cross_operation_reuses_worker_local_cut_published_generation(gguf_source, tmp_path):
    root = tmp_path / "received"
    # Existing worker local_cut currently passes a UUID wrapper to the builder.
    # The receiver must find that published generation within its own safe root.
    local = root / ("local-" + "a" * 32)
    first = build_gguf_segment(gguf_source, local, **_kwargs(gguf_source))
    second = _next_operation(first)
    events = []
    with _server(second) as (url, state):
        received = _receive(second, root, url, progress=events.append)
        _assert_transfer_accounting(received)
        for event in events:
            _assert_transfer_accounting(event)
        assert not state["requests"]
    assert received["actual_downloaded_bytes"] == 0
    assert received["reused_bytes"] == received["total_bytes"]
    assert Path(received["manifest_path"]).parent.parent == root
    assert Path(first["local_files"]["segment.gguf"]).is_file()
    assert events[-1]["completed_bytes"] == received["total_bytes"]

@pytest.mark.parametrize("complete_partial", [False, True])
def test_transfer_counters_separate_verified_files_and_partial_prefix(gguf_segment, tmp_path, complete_partial):
    root = tmp_path / "received"
    stage = root / (".supply-receive-" + gguf_segment["manifest_sha256"])
    stage.mkdir(parents=True)
    weight = next(item for item in gguf_segment["files"] if item["kind"] == "weights")
    native = next(item for item in gguf_segment["files"] if item["kind"] == "manifest")
    # A complete file from a previous attempt counts as reused, while bytes in
    # the separate .part file always count as resumed (even if already full).
    (stage / native["path"]).write_bytes(Path(gguf_segment["local_files"][native["path"]]).read_bytes())
    raw = Path(gguf_segment["local_files"][weight["path"]]).read_bytes()
    prefix_size = len(raw) if complete_partial else 17
    (stage / (weight["path"] + ".part")).write_bytes(raw[:prefix_size])
    events = []
    with _server(gguf_segment) as (url, state):
        result = _receive(gguf_segment, root, url, progress=events.append)
        assert len(state["requests"]) == (0 if complete_partial else 1)
        if not complete_partial:
            assert state["requests"][0][1] == "bytes=17-"
    _assert_transfer_accounting(result)
    for event in events:
        _assert_transfer_accounting(event)
    assert result["downloaded_bytes"] == result["total_bytes"]
    assert result["actual_downloaded_bytes"] == weight["size_bytes"] - prefix_size
    assert result["reused_bytes"] == native["size_bytes"]
    assert result["resumed_bytes"] == prefix_size
    assert events[-1]["downloaded_bytes"] == result["total_bytes"]
    assert events[-1]["actual_downloaded_bytes"] == result["actual_downloaded_bytes"]
    assert events[-1]["reused_bytes"] == result["reused_bytes"]
    assert events[-1]["resumed_bytes"] == prefix_size


def _interrupted_prefix(manifest, root):
    with _server(manifest, failure="truncate") as (url, _):
        with pytest.raises(SupplyError) as exc:
            _receive(manifest, root, url)
        assert exc.value.code in {"download_incomplete", "download_failed"}
    stage = root / (".supply-receive-" + manifest["manifest_sha256"])
    journal = assets._read_resume_journal(stage)
    record = next(item for item in journal["prefixes"] if item["path"].endswith(".gguf"))
    partial = stage / (record["path"] + ".part")
    assert sha256_file(partial) == record["prefix_sha256"]
    assert partial.stat().st_size == record["prefix_bytes"]
    return stage, record, partial


def test_cross_operation_resumes_verified_checkpoint_with_new_range(gguf_segment, tmp_path):
    root = tmp_path / "received"
    old_stage, record, old_partial = _interrupted_prefix(gguf_segment, root)
    old_bytes = old_partial.read_bytes()
    second = _next_operation(gguf_segment)
    events = []
    with _server(second) as (url, state):
        result = _receive(second, root, url, progress=events.append)
        assert state["requests"][0][1] == f"bytes={record['prefix_bytes']}-"
        assert all(path.startswith("/api/models/supply/op-2/files/worker-2/") for path, _, _ in state["requests"])
    _assert_transfer_accounting(result)
    for event in events:
        _assert_transfer_accounting(event)
    assert result["downloaded_bytes"] == result["total_bytes"]
    assert result["actual_downloaded_bytes"] == result["total_bytes"] - record["prefix_bytes"]
    assert result["reused_bytes"] == 0
    assert result["resumed_bytes"] == record["prefix_bytes"]
    assert old_partial.read_bytes() == old_bytes
    assert (old_stage / ".supply-resume.json").is_file()
    assert not (Path(result["manifest_path"]).parent / ".supply-resume.json").exists()
    assert not list(root.rglob(".supply-prefix-*.part"))


@pytest.mark.parametrize("damage", ["bytes", "unsigned_journal", "legacy_no_journal", "foreign_root", "foreign_wrapper"])
def test_cross_operation_rejects_unverified_or_foreign_partial(gguf_segment, tmp_path, damage):
    root = tmp_path / "received"
    old_root = tmp_path / "foreign" if damage == "foreign_root" else root
    stage, record, partial = _interrupted_prefix(gguf_segment, old_root)
    if damage == "bytes":
        before = partial.stat()
        raw = bytearray(partial.read_bytes())
        raw[-1] ^= 1
        partial.write_bytes(raw)
        os.utime(partial, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif damage == "unsigned_journal":
        marker = stage / ".supply-resume.json"
        journal = json.loads(marker.read_text())
        journal["prefixes"][0]["prefix_sha256"] = "0" * 64
        marker.write_bytes(_canonical(journal))
    elif damage == "legacy_no_journal":
        (stage / ".supply-resume.json").unlink()
    elif damage == "foreign_wrapper":
        wrapper = root / "arbitrary-storage"
        wrapper.mkdir()
        stage.rename(wrapper / stage.name)
    second = _next_operation(gguf_segment)
    with _server(second) as (url, state):
        result = _receive(second, root, url)
        assert state["requests"][0][1] == "bytes=0-"
    _assert_transfer_accounting(result)
    assert result["actual_downloaded_bytes"] == result["total_bytes"]
    assert result["resumed_bytes"] == 0
    assert result["reused_bytes"] == 0


def test_cross_operation_prefix_recheck_can_reject_self_consistent_wrong_bytes(gguf_segment, tmp_path):
    root = tmp_path / "received"
    stage, record, partial = _interrupted_prefix(gguf_segment, root)
    raw = bytearray(partial.read_bytes())
    raw[-1] ^= 1
    partial.write_bytes(raw)
    # A locally self-consistent checkpoint cannot override the authoritative
    # complete SHA: even its verified prefix must finish whole-file validation.
    marker = stage / ".supply-resume.json"
    journal = json.loads(marker.read_text())
    journal["prefixes"][0]["prefix_sha256"] = sha256_file(partial)
    journal.pop("journal_sha256")
    journal["journal_sha256"] = assets._digest(journal)
    marker.write_bytes(_canonical(journal))
    second = _next_operation(gguf_segment)
    with _server(second) as (url, state):
        with pytest.raises(SupplyError) as exc:
            _receive(second, root, url)
        assert exc.value.code == "file_digest_mismatch"
        assert state["requests"][0][1] == f"bytes={record['prefix_bytes']}-"
    assert not (root / ("segment-" + second["manifest_sha256"])).exists()
    assert not list(root.glob(".supply-receive-*/supply.manifest.json"))


def test_same_operation_restart_reverifies_checkpoint_and_discards_crash_tail(gguf_segment, tmp_path):
    root = tmp_path / "received"
    stage, record, partial = _interrupted_prefix(gguf_segment, root)
    with partial.open("ab") as handle:
        handle.write(b"uncheckpointed crash tail")
    with _server(gguf_segment) as (url, state):
        result = _receive(gguf_segment, root, url)
        assert state["requests"][0][1] == f"bytes={record['prefix_bytes']}-"
    assert result["resumed_bytes"] == record["prefix_bytes"]
    _assert_transfer_accounting(result)


def test_periodic_checkpoint_keeps_prefix_on_cancellation(gguf_segment, tmp_path, monkeypatch):
    root = tmp_path / "received"
    monkeypatch.setattr(assets, "_CHUNK", 32)
    monkeypatch.setattr(assets, "_RESUME_INTERVAL", 64)
    stop = [False]
    def progress(event):
        if event["actual_downloaded_bytes"] >= 96:
            stop[0] = True
    with _server(gguf_segment) as (url, _):
        with pytest.raises(SupplyError) as exc:
            _receive(gguf_segment, root, url, cancelled=lambda: stop[0], progress=progress)
        assert exc.value.code == "cancelled"
    stage = root / (".supply-receive-" + gguf_segment["manifest_sha256"])
    record = assets._read_resume_journal(stage)["prefixes"][0]
    assert record["prefix_bytes"] == 96
    assert sha256_file(stage / (record["path"] + ".part")) == record["prefix_sha256"]
    second = _next_operation(gguf_segment)
    with _server(second) as (url, state):
        result = _receive(second, root, url)
        assert state["requests"][0][1] == "bytes=96-"
    assert result["resumed_bytes"] == 96
    _assert_transfer_accounting(result)


def test_successful_small_downloads_do_not_write_checkpoint(gguf_segment, tmp_path, monkeypatch):
    monkeypatch.setattr(assets._ResumeJournal, "checkpoint", lambda *args: pytest.fail("complete small files need no prefix checkpoint"))
    with _server(gguf_segment) as (url, _):
        result = _receive(gguf_segment, tmp_path / "received", url)
    _assert_transfer_accounting(result)


def _checkpoint_fixture(manifest, root):
    payload = validate_supply_manifest(manifest)
    stage = root / (".supply-receive-" + payload["manifest_sha256"])
    stage.mkdir(parents=True)
    entry = next(item for item in payload["files"] if item["kind"] == "weights")
    raw = Path(manifest["local_files"][entry["path"]]).read_bytes()[:32]
    (stage / (entry["path"] + ".part")).write_bytes(raw)
    return stage, assets._ResumeJournal(stage, payload), entry, raw


@pytest.mark.skipif(os.name != "nt", reason="native Windows delete-sharing contract")
def test_checkpoint_retries_real_windows_reader_without_delete_sharing(gguf_segment, tmp_path, monkeypatch):
    import ctypes
    from ctypes import wintypes
    stage, journal, entry, raw = _checkpoint_fixture(gguf_segment, tmp_path / "received")
    journal.checkpoint(entry, 17, hashlib.sha256(raw[:17]).hexdigest())
    marker = stage / ".supply-resume.json"
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    held = create(str(marker), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
    assert held != ctypes.c_void_p(-1).value, ctypes.get_last_error()
    release = threading.Event()
    def release_reader():
        release.wait(timeout=2)
        close(held)
    thread = threading.Thread(target=release_reader)
    thread.start()
    real_replace = assets.os.replace
    attempts, denied = [], []
    def replace(source, target):
        if Path(target) != marker:
            return real_replace(source, target)
        attempts.append(Path(source).name)
        try:
            return real_replace(source, target)
        except OSError as exc:
            denied.append(exc.winerror)
            release.set()
            raise
    monkeypatch.setattr(assets.os, "replace", replace)
    try:
        journal.checkpoint(entry, len(raw), hashlib.sha256(raw).hexdigest())
    finally:
        release.set()
        thread.join(timeout=3)
    assert denied and set(denied) <= {5, 32, 33}
    assert 2 <= len(attempts) <= 8
    assert all(name.startswith(".supply-checkpoint-") for name in attempts)
    assert assets._read_resume_journal(stage)["prefixes"][0]["prefix_bytes"] == 32
    assert not list(stage.glob(".supply-checkpoint-*.tmp"))


@pytest.mark.parametrize("winerror,expected_attempts", [(5, 8), (32, 8), (33, 8), (87, 1)])
def test_checkpoint_failure_is_bounded_and_preserves_durable_record(gguf_segment, tmp_path, monkeypatch, winerror, expected_attempts):
    stage, journal, entry, raw = _checkpoint_fixture(gguf_segment, tmp_path / "received")
    journal.checkpoint(entry, 17, hashlib.sha256(raw[:17]).hexdigest())
    marker = stage / ".supply-resume.json"
    before = marker.read_bytes()
    calls, delays = [], []
    real_replace = assets.os.replace
    def replace(source, target):
        if Path(target) != marker:
            return real_replace(source, target)
        calls.append(Path(source).name)
        error = PermissionError("simulated Windows checkpoint conflict")
        error.winerror = winerror
        raise error
    monkeypatch.setattr(assets.sys, "platform", "win32")
    monkeypatch.setattr(assets.os, "replace", replace)
    monkeypatch.setattr(assets.time, "sleep", delays.append)
    with pytest.raises(SupplyError) as exc:
        journal.checkpoint(entry, len(raw), hashlib.sha256(raw).hexdigest())
    assert exc.value.code == "artifact_io_error"
    assert len(calls) == expected_attempts
    assert len(delays) == expected_attempts - 1
    assert sum(delays) < 1.3
    assert marker.read_bytes() == before
    assert journal.records[entry["path"]]["prefix_bytes"] == 17
    assert not list(stage.glob(".supply-checkpoint-*.tmp"))


def test_checkpoint_uses_unique_temporary_for_each_publication(gguf_segment, tmp_path, monkeypatch):
    stage, journal, entry, raw = _checkpoint_fixture(gguf_segment, tmp_path / "received")
    real_replace = assets.os.replace
    names = []
    def replace(source, target):
        if Path(target).name == ".supply-resume.json":
            names.append(Path(source).name)
        return real_replace(source, target)
    monkeypatch.setattr(assets.os, "replace", replace)
    journal.checkpoint(entry, 17, hashlib.sha256(raw[:17]).hexdigest())
    journal.checkpoint(entry, 32, hashlib.sha256(raw).hexdigest())
    journal.checkpoint(entry, 32, hashlib.sha256(raw).hexdigest())
    assert len(names) == len(set(names)) == 2
    assert all(name.startswith(".supply-checkpoint-") for name in names)
    assert not list(stage.glob(".supply-checkpoint-*.tmp"))


@pytest.mark.parametrize("mtp,span,wrong_role", [(False, (0,2), "tail"), (False, (2,4), "middle"), (True, (1,3), "tail"), (True, (2,4), "head")])
def test_wrong_cutter_role_cannot_be_normalized_without_exact_mtp_tail(tmp_path, monkeypatch, mtp, span, wrong_role):
    source = tmp_path / "source.gguf"
    _make_gguf(source, mtp=mtp)
    original = assets._run_cut
    def wrong(*args):
        original(*args)
        native = args[2]
        data = json.loads(native.read_text(encoding="utf-8"))
        data["mode"] = wrong_role
        native.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(assets, "_run_cut", wrong)
    with pytest.raises(SupplyError) as error:
        build_gguf_segment(source, tmp_path / "out", **_kwargs(source, *span))
    assert error.value.code == "cut_manifest_mismatch"
    assert not list((tmp_path / "out").glob("segment-*"))


@pytest.mark.parametrize("field,value", [("mode","head"), ("artifact","another.gguf")])
def test_native_role_and_filename_are_checked_on_pc_reuse(gguf_segment, field, value):
    root = Path(gguf_segment["local_files"]["segment.gguf"]).parent
    path = root / "segment.manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data[field] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SupplyError) as error:
        assets._verify_segment(root, gguf_segment)
    assert error.value.code == "native_manifest_mismatch"


@pytest.mark.parametrize("span", [(0, 2), (1, 3), (2, 4)])
def test_gguf_disk_budget_matches_actual_cropped_tensor_inventory_and_excludes_mtp(tmp_path, span):
    from model_supply import _segment_disk_weight_bytes
    source = tmp_path / "disk-mtp.gguf"
    _make_gguf(source, mtp=True)
    estimate = _segment_disk_weight_bytes({"layer_weight_bytes": [32] * 4,
        "component_weight_bytes": {"layers": 128, "embedding": 256, "final_norm": 32, "other": 32}},
        {"engine": "llama_cpp", "start_layer": span[0], "end_layer": span[1]}, source)
    manifest = build_gguf_segment(source, tmp_path / "disk-out", **_kwargs(source, *span))
    _, actual = assets._gguf_table(Path(manifest["local_files"]["segment.gguf"]))
    assert estimate == sum(entry[1] for entry in actual.values()) == 352
    assert not any(name.startswith("blk.2.nextn") for name in actual)


# Ticket 19: source-authoritative metadata and immutable legacy migration.
def _rewrite_native_declaration(manifest, edit, *, marker=False):
    native_path = Path(manifest["local_files"]["segment.manifest.json"])
    native = json.loads(native_path.read_text(encoding="utf-8"))
    edit(native)
    native_path.write_bytes(_canonical(native))
    public = validate_supply_manifest(manifest)
    for entry in public["files"]:
        if entry["path"] == "segment.manifest.json":
            entry.update(size_bytes=native_path.stat().st_size, sha256=sha256_file(native_path))
    public = _sign(public)
    if marker:
        (native_path.parent / "supply.manifest.json").write_bytes(_canonical(public))
    return {**public, "local_files": dict(manifest["local_files"])}


@pytest.mark.parametrize("span", [(0, 2), (1, 3), (2, 4)])
def test_gguf_metadata_preserves_and_crops_integer_head_arrays(tmp_path, span):
    from gguf_metadata_contract import read_gguf_metadata
    source = tmp_path / "array-heads.gguf"
    _make_gguf(source, head_count_kv=[1, 2, 3, 4], key_length=256)
    result = build_gguf_segment(source, tmp_path / "out", **_kwargs(source, *span))
    raw, _ = assets._gguf_table(Path(result["local_files"]["segment.gguf"]))
    actual = read_gguf_metadata(result["local_files"]["segment.gguf"])
    assert raw["qwen2.attention.head_count_kv"] == [1, 2, 3, 4][span[0]:span[1]]
    native = json.loads(Path(result["local_files"]["segment.manifest.json"]).read_text())
    assert native["gguf_metadata"] == actual
    assert native["gguf_metadata"]["block_count"] == span[1] - span[0]
    assert native["gguf_metadata"]["key_length"] == 256
    assert native["gguf_metadata"]["full_attention_interval"] is None
    assert native["block_count"] == 4
    assert "gguf_metadata" not in validate_supply_manifest(result)


@pytest.mark.parametrize("field", ["block_count", "head_count_kv", "key_length", "full_attention_interval"])
@pytest.mark.parametrize("missing", [False, True])
def test_gguf_receive_and_same_generation_reuse_enforce_all_metadata(gguf_segment, tmp_path, field, missing):
    def edit(native):
        if missing:
            native["gguf_metadata"].pop(field)
        else:
            native["gguf_metadata"][field] = 999
    changed = _rewrite_native_declaration(gguf_segment, edit, marker=True)
    expected_code = "gguf_metadata_missing" if missing else "gguf_metadata_mismatch"
    native_root = Path(changed["local_files"]["segment.gguf"]).parent
    with pytest.raises(SupplyError, match=field) as error:
        assets._reuse(native_root, validate_supply_manifest(changed))
    assert error.value.code == expected_code
    root = tmp_path / "received"
    with _server(changed) as (url, _):
        with pytest.raises(SupplyError, match=field) as error:
            _receive(changed, root, url)
    assert error.value.code == expected_code
    assert not list(root.rglob("supply.manifest.json"))


@pytest.mark.parametrize("same_operation", [True, False])
@pytest.mark.parametrize("mtp", [False, True])
def test_legacy_gguf_cache_upgrades_from_known_source_without_touching_old_ready(tmp_path, monkeypatch, same_operation, mtp):
    from gguf_metadata_contract import read_gguf_metadata
    source = tmp_path / "source.gguf"
    _make_gguf(source, mtp=mtp, head_count_kv=4, key_length=256, hybrid=True)
    out = tmp_path / "out"
    args = _kwargs(source, 2, 4)
    first = build_gguf_segment(source, out, **args)
    legacy = _rewrite_native_declaration(first, lambda native: native.pop("gguf_metadata"), marker=True)
    old_root = Path(legacy["local_files"]["segment.gguf"]).parent
    before = {name: (old_root / name).read_bytes() for name in ("segment.manifest.json", "supply.manifest.json")}
    with pytest.raises(SupplyError) as error:
        assets._reuse(old_root, validate_supply_manifest(legacy))
    assert error.value.code == "gguf_metadata_missing"
    monkeypatch.setattr(assets, "_run_cut", lambda *a, **k: pytest.fail("known-source migration must reuse weights"))
    if not same_operation:
        args["operation_id"] = "op-2"
    upgraded = build_gguf_segment(source, out, **args)
    new_root = Path(upgraded["local_files"]["segment.gguf"]).parent
    assert new_root != old_root
    assert upgraded["manifest_sha256"] != legacy["manifest_sha256"]
    for name, data in before.items():
        assert (old_root / name).read_bytes() == data
    native = json.loads((new_root / "segment.manifest.json").read_text())
    assert native["gguf_metadata"] == read_gguf_metadata(new_root / "segment.gguf")
    assert native["gguf_metadata"]["block_count"] == 2
    assert native["gguf_metadata"]["nextn_predict_layers"] == (0 if mtp else None)
    assert native["gguf_metadata"]["head_count_kv"] == 4
    assert native["gguf_metadata"]["key_length"] == 256
    assert native["gguf_metadata"]["full_attention_interval"] == 2
    assert native["block_count"] == 4 + int(mtp)
    assert native["nextn_predict_layers"] == int(mtp)
    again = build_gguf_segment(source, out, **args)
    assert again["local_files"] == upgraded["local_files"]
    assert again["manifest_sha256"] == upgraded["manifest_sha256"]
    assets._verify_files(new_root, validate_supply_manifest(upgraded))


@pytest.mark.parametrize("field", ["block_count", "head_count_kv", "key_length", "full_attention_interval"])
@pytest.mark.parametrize("same_operation", [False, True])
def test_legacy_declared_mismatch_is_never_overwritten_from_source(gguf_segment, tmp_path, monkeypatch, field, same_operation):
    def edit(native):
        native["gguf_metadata"] = {field: 999}
    changed = _rewrite_native_declaration(gguf_segment, edit, marker=True)
    root = Path(changed["local_files"]["segment.gguf"]).parent
    old_native = (root / "segment.manifest.json").read_bytes()
    monkeypatch.setattr(assets, "_run_cut", lambda *a, **k: pytest.fail("declared mismatch must fail"))
    args = _kwargs(tmp_path / "source.gguf")
    # The fixture and source share tmp_path; cover same-final recovery and rebind.
    if not same_operation:
        args["operation_id"] = "op-2"
    with pytest.raises(SupplyError, match=field) as error:
        build_gguf_segment(tmp_path / "source.gguf", root.parent, **args)
    assert error.value.code == "gguf_metadata_mismatch"
    assert (root / "segment.manifest.json").read_bytes() == old_native


def test_cut_actual_metadata_must_match_source_even_with_legacy_declaration(tmp_path, monkeypatch):
    source = tmp_path / "source.gguf"
    _make_gguf(source, key_length=256)
    original = assets._run_cut
    def wrong_cut(*args, **kwargs):
        original(*args, **kwargs)
        artifact, native_path = args[1:3]
        data = artifact.read_bytes()
        key = b"qwen2.attention.key_length"
        needle = struct.pack("<Q", len(key)) + key + struct.pack("<II", 4, 256)
        assert data.count(needle) == 1
        artifact.write_bytes(data.replace(needle, needle[:-4] + struct.pack("<I", 128)))
        native = json.loads(native_path.read_text(encoding="utf-8"))
        native.pop("gguf_metadata")
        native["artifact_sha256"] = sha256_file(artifact)
        native_path.write_bytes(_canonical(native))
    monkeypatch.setattr(assets, "_run_cut", wrong_cut)
    with pytest.raises(SupplyError, match="key_length") as error:
        build_gguf_segment(source, tmp_path / "out", **_kwargs(source))
    assert error.value.code == "gguf_metadata_mismatch"
    assert not list((tmp_path / "out").glob("segment-*"))


def test_supply_metadata_reuse_and_verification_never_import_native(gguf_segment, gguf_source, monkeypatch):
    import builtins
    root = Path(gguf_segment["local_files"]["segment.gguf"]).parent
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"gguf", "llama_cpp", "numpy", "torch"}
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    reused = build_gguf_segment(gguf_source, root.parent, **_kwargs(gguf_source))
    assert reused == gguf_segment
    assets._verify_segment(root, validate_supply_manifest(reused))


def test_native_kept_block_count_cannot_disagree_with_metadata(gguf_segment):
    def edit(native):
        native["kept_block_count"] = 3
    changed = _rewrite_native_declaration(gguf_segment, edit, marker=True)
    root = Path(changed["local_files"]["segment.gguf"]).parent
    with pytest.raises(SupplyError, match="block_count") as error:
        assets._reuse(root, validate_supply_manifest(changed))
    assert error.value.code == "gguf_metadata_mismatch"


@pytest.mark.parametrize("stop", ["cancel", "deadline", "callback_error"])
def test_cut_reaps_real_child_before_reporting_stop(gguf_source, tmp_path, monkeypatch, stop):
    import subprocess
    import time
    ready = tmp_path / "child-ready"
    children = []
    popen = subprocess.Popen
    def child(argv, **kwargs):
        process = popen([assets.sys.executable, "-c",
            "from pathlib import Path; import time; Path(" + repr(str(ready)) + ").touch(); time.sleep(120)"], **kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(assets.subprocess, "Popen", child)
    def cancelled():
        if stop == "callback_error" and ready.exists():
            raise RuntimeError("owner check failed")
        return stop == "cancel" and ready.exists()
    options = {"cancelled": cancelled, "deadline": time.monotonic() + 1.0}
    payload = assets._base_manifest(**_kwargs(gguf_source))
    expected = RuntimeError if stop == "callback_error" else SupplyError
    with pytest.raises(expected) as exc:
        assets._run_cut(gguf_source, tmp_path / "out.gguf", tmp_path / "native.json", payload, 0, **options)
    if stop != "callback_error":
        assert exc.value.code == ("cancelled" if stop == "cancel" else "cut_timeout")
    assert ready.exists() and len(children) == 1
    assert children[0].poll() is not None, "stop returned before its cutter exited"


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_gguf_build_stopped_before_source_read_or_publish(gguf_source, tmp_path, monkeypatch, stop):
    import time
    kwargs = _kwargs(gguf_source)
    monkeypatch.setattr(assets, "_hash_file", lambda *a, **k: pytest.fail("stopped build read source"))
    monkeypatch.setattr(assets, "_run_cut", lambda *a, **k: pytest.fail("stopped build started child"))
    with pytest.raises(SupplyError) as exc:
        build_gguf_segment(gguf_source, tmp_path / "out", **kwargs,
                           cancelled=lambda: stop == "cancel",
                           deadline=time.monotonic() - 1 if stop == "deadline" else None)
    assert exc.value.code == ("cancelled" if stop == "cancel" else "cut_timeout")
    assert not (tmp_path / "out").exists()
