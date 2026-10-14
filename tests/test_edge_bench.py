"""票 19 / EDGE-BENCH-01：edge_bench 的工件 metadata 校验（fail-closed）。

只测纯逻辑（_verify 与 main 的校验分支），不加载真模型 —— 标定部分依赖真实工件与时间，
不适合放进单元测试。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("edge_bench", ROOT / "scripts" / "edge_bench.py")
edge_bench = importlib.util.module_from_spec(_SPEC)
sys.modules["edge_bench"] = edge_bench
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(edge_bench)


def test_verify_reports_only_mismatched_known_keys():
    meta = {"block_count": 8, "head_count_kv": 2, "key_length": None}
    expect = {"block_count": 8, "head_count_kv": 4, "full_attention_interval": 4}
    diffs = edge_bench._verify(meta, expect, "expect")
    # block_count 一致、full_attention_interval 工件侧缺失 ⇒ 都不报；只有 head_count_kv 不匹配
    assert diffs == ["head_count_kv: 工件=2 vs expect=4"]


def test_verify_skips_keys_missing_on_either_side():
    assert edge_bench._verify({"block_count": 8}, {}, "expect") == []
    assert edge_bench._verify({}, {"block_count": 8}, "expect") == []


def test_verify_empty_when_all_match():
    assert edge_bench._verify({"block_count": 8}, {"block_count": 8}, "m") == []


def _run_verify(tmp_path, monkeypatch, meta, *, expect=None, manifest=None):
    monkeypatch.setattr(edge_bench, "_load_gguf_meta", lambda p: dict(meta))
    model = tmp_path / "x.gguf"
    model.write_bytes(b"")
    argv = ["--model", str(model), "--json-out", str(tmp_path / "r.json")]
    if expect:
        argv += ["--expect", expect]
    if manifest:
        path = tmp_path / "m.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        argv += ["--verify-manifest", str(path)]
    return edge_bench.main(argv), json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))


def test_main_verify_pass(tmp_path, monkeypatch):
    code, rec = _run_verify(tmp_path, monkeypatch, {"block_count": 8},
                            manifest={"kept_block_count": 8})
    assert code == 0
    assert rec["passed"] is True
    assert rec["differences"] == []
    # manifest 的 kept_block_count 应被映射成 block_count 参与比对
    assert rec["expected"]["block_count"] == 8


def test_main_verify_fail_is_fail_closed(tmp_path, monkeypatch):
    code, rec = _run_verify(tmp_path, monkeypatch, {"block_count": 8},
                            expect="block_count=16,head_count_kv=2")
    assert code == 1
    assert rec["passed"] is False
    assert rec["differences"] == ["block_count: 工件=8 vs --expect=16"]


def test_main_verify_without_comparable_items_does_not_pass(tmp_path, monkeypatch):
    """没有可比对项时不能算「通过」——否则 fail-closed 会退化成永远通过。"""
    code, rec = _run_verify(tmp_path, monkeypatch, {"architecture": "qwen2"},
                            manifest={"source_layer_range": [8, 16]})
    assert code == 1
    assert rec["passed"] is False
    assert rec["differences"] == []
    assert rec["comparable_items"] == []
