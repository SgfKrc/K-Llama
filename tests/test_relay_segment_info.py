"""tests/test_relay_segment_info.py — 段构建标识（§10.2 待办）的自包含测试。

覆盖三件事：
1. `collect_local_build` 对小文件（shim / libllama）算 sha256，对 GB 级段工件**默认不算**
   （只记大小/名字），显式 `digest_artifacts=True` 才算；
2. `load_ready_build` 能从服务端 ready 文件读到 `build`，读不到/老格式时**如实标注原因**；
3. `unknown_remote_build` 显式标 `remote_unknown`（记录必须能区分「未知」与「没写」）。
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import relay_segment_info as R  # noqa: E402


def test_collect_local_build_digests_small_files_only(tmp_path: Path) -> None:
    shim = tmp_path / "qlh_keep_head.dll"
    shim.write_bytes(b"shim-bytes")
    (tmp_path / "libllama.dll").write_bytes(b"llama-bytes")
    (tmp_path / "ggml-base.dll").write_bytes(b"ggml-bytes")
    model = tmp_path / "artifact.gguf"
    model.write_bytes(b"x" * 32)

    build = R.collect_local_build(shim=shim, model=model, llama_cpp_module=False)

    assert build["schema_version"] == R.RELAY_SEGMENT_INFO_SCHEMA
    assert build["shim"]["sha256"] == hashlib.sha256(b"shim-bytes").hexdigest()
    # 同目录的 libllama / ggml* 才是真正决定数值的构建产物 ⇒ 一并记摘要
    assert set(build["llama_cpp_build"]) == {"libllama.dll", "ggml-base.dll"}
    # 段工件默认**不算** sha256（GB 级会拖慢每轮实验），但大小/名字必须有
    assert build["model"]["sha256"] is None
    assert build["model"]["size"] == 32
    assert build["model"]["name"] == "artifact.gguf"


def test_collect_local_build_digests_artifact_on_demand(tmp_path: Path) -> None:
    model = tmp_path / "artifact.gguf"
    model.write_bytes(b"abc")
    build = R.collect_local_build(model=model, llama_cpp_module=False, digest_artifacts=True)
    assert build["model"]["sha256"] == hashlib.sha256(b"abc").hexdigest()


def test_collect_local_build_marks_missing_paths(tmp_path: Path) -> None:
    build = R.collect_local_build(shim=tmp_path / "nope.dll", llama_cpp_module=False)
    assert build["shim"]["exists"] is False
    assert build["shim"].get("sha256") is None


def test_load_ready_build_reads_service_build(tmp_path: Path) -> None:
    ready = tmp_path / "tail.ready"
    ready.write_text(json.dumps({
        "role": "tail",
        "n_embd": 4096,
        "build": {"schema_version": R.RELAY_SEGMENT_INFO_SCHEMA,
                  "llama_cpp_version": "0.3.35"},
    }), encoding="utf-8")

    build = R.load_ready_build(ready)

    assert build["source"] == "ready_file"
    assert build["llama_cpp_version"] == "0.3.35"
    assert build["role"] == "tail"
    assert build["schema_version"] == R.RELAY_SEGMENT_INFO_SCHEMA


def test_load_ready_build_marks_unreadable_and_legacy(tmp_path: Path) -> None:
    missing = R.load_ready_build(tmp_path / "nope.ready")
    assert missing["source"] == "ready_file_unreadable"

    # 老版本 ready 文件没有 `build` 字段 ⇒ 显式标注，而不是静默给空 dict
    legacy = tmp_path / "legacy.ready"
    legacy.write_text(json.dumps({"role": "middle", "n_embd": 896}), encoding="utf-8")
    old = R.load_ready_build(legacy)
    assert old["source"] == "ready_file_without_build"
    assert old["role"] == "middle"


def test_unknown_remote_build_is_explicit() -> None:
    info = R.unknown_remote_build("127.0.0.1:50190")
    assert info["source"] == "remote_unknown"
    assert info["endpoint"] == "127.0.0.1:50190"


# ── ★ A15b / #30：零依赖读 GGUF 头（让**工件自己**回答"我覆盖多少层"）──────────


def _kv_str(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _write_gguf(path: Path, kvs, *, magic: bytes = b"GGUF", version: int = 3) -> Path:
    """合成一个**最小 GGUF 头**（只到 KV 区就够 —— `read_gguf_layer_info` 不读张量区）。"""
    body = bytearray(magic)
    body += struct.pack("<I", version)
    body += struct.pack("<Q", 0)           # n_tensors（本用例不需要）
    body += struct.pack("<Q", len(kvs))
    for key, value_type, payload in kvs:
        body += _kv_str(key) + struct.pack("<I", value_type) + payload
    path.write_bytes(bytes(body))
    return path


def _arch_kvs(arch: str, block_count: int, nextn: int = 0, n_embd: int = 2048):
    return [
        ("general.architecture", 8, _kv_str(arch)),
        (f"{arch}.block_count", 4, struct.pack("<I", block_count)),
        (f"{arch}.nextn_predict_layers", 4, struct.pack("<I", nextn)),
        (f"{arch}.embedding_length", 4, struct.pack("<I", n_embd)),
    ]


def test_read_gguf_layer_info_reads_block_count(tmp_path: Path) -> None:
    """★ 工件自证层数：`n_layer = block_count - nextn`（与 `scripts/cut_layers.py:82` 同口径）。"""
    path = _write_gguf(tmp_path / "m.gguf", _arch_kvs("qwen2", 24))

    assert R.read_gguf_layer_info(path) == {
        "architecture": "qwen2", "block_count": 24,
        "nextn_predict_layers": 0, "n_layer": 24, "n_embd": 2048,
    }


def test_read_gguf_layer_info_does_not_assume_kv_order(tmp_path: Path) -> None:
    """★ KV 顺序**不做假设**：`block_count` 出现在 `general.architecture` **之前**也要能读。"""
    path = _write_gguf(tmp_path / "reordered.gguf", [
        ("qwen2.block_count", 4, struct.pack("<I", 8)),
        ("general.architecture", 8, _kv_str("qwen2")),
        ("qwen2.nextn_predict_layers", 4, struct.pack("<I", 0)),
    ])

    info = R.read_gguf_layer_info(path)
    assert info is not None and info["n_layer"] == 8


def test_read_gguf_layer_info_skips_other_value_types(tmp_path: Path) -> None:
    """★ 其余 KV 要按类型**结构化跳过**（字符串 / 布尔 / 数组），否则后面就整体读错位。"""
    kvs = _arch_kvs("qwen2", 8)
    kvs.insert(0, ("general.name", 8, _kv_str("some model name")))
    kvs.append(("general.some_bool", 7, struct.pack("<?", True)))
    # 数组：u32 元素类型 + u64 个数 + 逐个元素
    kvs.append(("tokenizer.ggml.tokens", 9,
                struct.pack("<I", 8) + struct.pack("<Q", 3)
                + _kv_str("a") + _kv_str("bb") + _kv_str("ccc")))

    info = R.read_gguf_layer_info(_write_gguf(tmp_path / "mixed.gguf", kvs))
    assert info is not None and info["n_layer"] == 8


def test_read_gguf_layer_info_subtracts_nextn(tmp_path: Path) -> None:
    """★ MTP（`nextn_predict_layers`）**不计入**层数 —— 上一轮 `mode` 判错正是栽在这里。"""
    path = _write_gguf(tmp_path / "mtp.gguf", _arch_kvs("qwen2", 24, nextn=4))

    info = R.read_gguf_layer_info(path)
    assert info is not None and (info["block_count"], info["n_layer"]) == (24, 20)


def test_read_gguf_layer_info_is_none_on_bad_input(tmp_path: Path) -> None:
    """非 GGUF / 版本过老 / 缺 arch / `block_count` 非正 / 不存在 ⇒ 一律 `None`（**不猜、不抛**）。"""
    not_gguf = _write_gguf(tmp_path / "x.gguf", _arch_kvs("qwen2", 8), magic=b"NOPE")
    old = _write_gguf(tmp_path / "old.gguf", _arch_kvs("qwen2", 8), version=1)
    no_arch = _write_gguf(tmp_path / "noarch.gguf",
                          [("qwen2.block_count", 4, struct.pack("<I", 8))])
    zero = _write_gguf(tmp_path / "zero.gguf", _arch_kvs("qwen2", 0))

    assert R.read_gguf_layer_info(not_gguf) is None
    assert R.read_gguf_layer_info(old) is None
    assert R.read_gguf_layer_info(no_arch) is None
    assert R.read_gguf_layer_info(zero) is None
    assert R.read_gguf_layer_info(tmp_path / "missing.gguf") is None


def test_gguf_tokenizer_fingerprint_is_order_independent_and_header_only(tmp_path: Path) -> None:
    tokenizer = [
        ("tokenizer.ggml.model", 8, _kv_str("gpt2")),
        ("tokenizer.ggml.tokens", 9,
         struct.pack("<IQ", 8, 2) + _kv_str("hello") + _kv_str("world")),
        ("tokenizer.chat_template", 8, _kv_str("{{ messages }}")),
    ]
    first = _write_gguf(tmp_path / "first.gguf", _arch_kvs("qwen35", 8) + tokenizer)
    second = _write_gguf(tmp_path / "second.gguf", list(reversed(tokenizer)) + _arch_kvs("qwen35", 4))
    first.write_bytes(first.read_bytes() + b"unread tensor payload")

    field_hashes = {
        key: hashlib.sha256(struct.pack("<I", kind) + value).digest()
        for key, kind, value in tokenizer
    }
    expected = hashlib.sha256(b"".join(
        key.encode() + b"\0" + field_hashes[key] for key in sorted(field_hashes)
    )).hexdigest()
    assert R.read_gguf_tokenizer_sha256(first) == expected
    assert R.read_gguf_tokenizer_sha256(second) == expected
    changed = _write_gguf(tmp_path / "changed.gguf", tokenizer[:-1] + [
        ("tokenizer.chat_template", 8, _kv_str("different")),
    ])
    assert R.read_gguf_tokenizer_sha256(changed) != expected
    assert R.read_gguf_tokenizer_sha256(_write_gguf(tmp_path / "missing.gguf", tokenizer[:1])) is None


def test_read_artifact_manifest_reads_layer_range(tmp_path: Path) -> None:
    """★ 「来自源模型的哪几层」只能由 manifest 自证（工件头里**没有**这个信息）。"""
    manifest = tmp_path / "mid8-16.gguf.manifest.json"
    manifest.write_text(json.dumps({
        "source_layer_range": [8, 16],
        "n_layer": 8,
        "source_model_sha256": "a" * 64,
        "artifact_sha256": "b" * 64,
        "source_model_id": "qwen2.5-0.5b",
        "hidden_size": 896,
        "tokenizer_sha256": "c" * 64,
    }),
                        encoding="utf-8")

    assert R.read_artifact_manifest(manifest) == {
        "source_layer_range": [8, 16],
        "n_layer": 8,
        "source_model_sha256": "a" * 64,
        "artifact_sha256": "b" * 64,
        "source_model_id": "qwen2.5-0.5b",
        "hidden_size": 896,
        "tokenizer_sha256": "c" * 64,
    }


def test_read_artifact_manifest_is_none_on_bad_input(tmp_path: Path) -> None:
    """坏 JSON / 非对象 / 区间倒置 / 不存在 ⇒ `None`（**不猜**）。"""
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    arr = tmp_path / "arr.json"
    arr.write_text("[1,2,3]", encoding="utf-8")
    inverted = tmp_path / "inv.json"
    inverted.write_text(json.dumps({"source_layer_range": [16, 8]}), encoding="utf-8")

    assert R.read_artifact_manifest(bad) is None
    assert R.read_artifact_manifest(arr) is None
    assert R.read_artifact_manifest(inverted) is None
    assert R.read_artifact_manifest(tmp_path / "nope.json") is None


def test_collect_local_build_attaches_layer_info_and_manifest(tmp_path: Path) -> None:
    """★ 接线：`collect_local_build(model=…)` 的 `model` 段现在带 `layer_info` 与 `manifest`。"""
    artifact = _write_gguf(tmp_path / "tail8.gguf", _arch_kvs("qwen2", 16))
    (tmp_path / "tail8.gguf.manifest.json").write_text(
        json.dumps({"source_layer_range": [8, 24], "n_layer": 24}), encoding="utf-8")

    model = R.collect_local_build(model=artifact)["model"]

    assert model["layer_info"]["n_layer"] == 16
    assert model["manifest"]["source_layer_range"] == [8, 24]


def test_collect_local_build_omits_layer_info_for_non_gguf(tmp_path: Path) -> None:
    """非 GGUF 工件 ⇒ **不写** `layer_info`（如实留空，而不是编一个层数出来）。"""
    plain = tmp_path / "notes.txt"
    plain.write_text("not a model", encoding="utf-8")

    model = R.collect_local_build(model=plain)["model"]

    assert "layer_info" not in model
    assert "manifest" not in model


def test_layer_info_path_does_not_import_llama_cpp(tmp_path: Path) -> None:
    """★ **反污染守卫**：这条路径**绝不能**把 `llama_cpp` / `gguf` 拽进 `sys.modules`
    —— 一旦进了，`KeepHeadUpstream` 会改走 worker 子进程，keep-head 的隔离判据就变了。

    ⚠️ 判据必须是**增量**（`调用后 - 调用前`），不是"绝对不在 `sys.modules` 里"：
    全量跑时别的测试可能早就导入过 `llama_cpp`（实测踩到：绝对判据**全量必红、单文件恒绿**，
    属于那种最容易被误当 flaky 放过的假红）。
    """
    artifact = _write_gguf(tmp_path / "head8.gguf", _arch_kvs("qwen2", 8))
    before = set(sys.modules)

    R.collect_local_build(model=artifact)
    R.read_gguf_layer_info(artifact)

    imported = sorted(
        mod for mod in set(sys.modules) - before
        if mod.split(".")[0] in {"llama_cpp", "gguf", "torch"}
    )
    assert imported == [], f"这条路径隐式导入了不该导入的模块：{imported}"
