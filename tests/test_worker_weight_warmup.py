"""Tests for background weight page warming.

The module is advisory by contract: it must never block the caller, never raise
into the request path, and never change what is read.
"""
from __future__ import annotations

import os
import time

import pytest

from worker_weight_warmup import WeightWarmup, merge_windows


PAGE = 4096


def _write_file(path, size):
    with open(path, "wb") as handle:
        handle.write(b"\0" * size)
    return str(path)


# -- window planning ------------------------------------------------------

def test_merge_windows_aligns_to_pages():
    merged = merge_windows([(1, 10)])
    assert merged == [(0, PAGE)]


def test_merge_windows_joins_adjacent_windows():
    # Adjacent ranges merge before alignment, then the result is page-aligned.
    merged = merge_windows([(0, 100), (100, 200)])
    assert merged == [(0, PAGE)]


def test_merge_windows_joins_adjacent_page_aligned_windows():
    merged = merge_windows([(0, PAGE), (PAGE, 2 * PAGE)])
    assert merged == [(0, 2 * PAGE)]


def test_merge_windows_keeps_large_gaps_separate():
    merged = merge_windows([(0, 100), (10 << 20, (10 << 20) + 100)], min_gap=1 << 20)
    assert len(merged) == 2


def test_merge_windows_joins_small_gaps():
    merged = merge_windows([(0, 100), (4096 * 2, 4096 * 3)], min_gap=1 << 20)
    assert len(merged) == 1


def test_merge_windows_drops_invalid_entries():
    assert merge_windows([None, (10, 10), (-5, 100), "x", (0, 100)]) == [(0, PAGE)]


def test_merge_windows_returns_empty_for_nothing_usable():
    assert merge_windows([]) == []
    assert merge_windows([(5, 1)]) == []


# -- advisory contract ----------------------------------------------------

def test_warm_enqueues_without_blocking(tmp_path):
    path = _write_file(tmp_path / "w.gguf", 1 << 20)
    warmup = WeightWarmup()
    try:
        started = time.monotonic()
        assert warmup.warm(path) is True
        elapsed = time.monotonic() - started
        assert elapsed < 0.5, "warm() must not block on the page-in itself"
        assert warmup.wait_idle(timeout=30) is True
        stats = warmup.stats()
        assert stats["requested"] == 1
        assert stats["advised"] >= 1
        assert stats["bytes_advised"] > 0
        assert stats["failures"] == 0
    finally:
        warmup.close()


def test_warm_missing_file_is_a_no_op(tmp_path):
    warmup = WeightWarmup()
    try:
        assert warmup.warm(str(tmp_path / "absent.gguf")) is False
        assert warmup.stats()["requested"] == 0
    finally:
        warmup.close()


def test_warm_empty_path_is_a_no_op():
    warmup = WeightWarmup()
    try:
        assert warmup.warm("") is False
    finally:
        warmup.close()


def test_disabled_warmup_does_not_queue(tmp_path):
    path = _write_file(tmp_path / "w.gguf", 1 << 16)
    warmup = WeightWarmup(enabled=False)
    try:
        assert warmup.warm(path) is False
        stats = warmup.stats()
        assert stats["skipped_disabled"] == 1
        assert stats["requested"] == 0
    finally:
        warmup.close()


def test_disable_after_construction_is_honoured(tmp_path):
    path = _write_file(tmp_path / "w.gguf", 1 << 16)
    warmup = WeightWarmup()
    try:
        warmup.disable()
        assert warmup.warm(path) is False
        warmup.enable()
        assert warmup.warm(path) is True
        warmup.wait_idle(timeout=30)
    finally:
        warmup.close()


def test_warm_with_explicit_windows_advises_only_those(tmp_path):
    path = _write_file(tmp_path / "w.gguf", 8 << 20)
    warmup = WeightWarmup()
    try:
        # Two windows separated by more than the default min_gap stay separate.
        far = 4 << 19
        assert warmup.warm(path, [(0, PAGE), (far, far + PAGE)]) is True
        warmup.wait_idle(timeout=30)
        stats = warmup.stats()
        assert stats["windows"] == 2
        assert stats["bytes_advised"] == 2 * PAGE
    finally:
        warmup.close()


def test_windows_beyond_file_end_are_clamped(tmp_path):
    path = _write_file(tmp_path / "w.gguf", 8192)
    warmup = WeightWarmup()
    try:
        assert warmup.warm(path, [(0, 1 << 24)]) is True
        warmup.wait_idle(timeout=30)
        assert warmup.stats()["bytes_advised"] <= 8192
    finally:
        warmup.close()


def test_close_is_idempotent(tmp_path):
    path = _write_file(tmp_path / "w.gguf", 4096)
    warmup = WeightWarmup()
    warmup.warm(path)
    warmup.close()
    warmup.close()


def test_stats_snapshot_is_a_copy():
    warmup = WeightWarmup()
    try:
        first = warmup.stats()
        first["requested"] = 999
        assert warmup.stats()["requested"] != 999
    finally:
        warmup.close()
