import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cache_unit_layout import (
    CacheUnitLayout,
    aligned_common_prefix_tokens,
    build_cache_units,
    match_cached_prefix,
)
from paged_kv_cache import PagedKVCache


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "scripts" / "cache_unit_experiment.py"


def test_page_and_unit_boundaries_are_aligned():
    layout = CacheUnitLayout(page_size=8, unit_size=4)
    assert layout.units_per_page == 2
    assert layout.unit_boundaries(10) == (0, 4, 8)
    assert layout.page_boundaries(10) == (0, 8)
    assert layout.complete_token_count(10) == 8


def test_non_divisible_unit_size_is_rejected():
    with pytest.raises(ValueError, match="divide page_size"):
        CacheUnitLayout(page_size=6, unit_size=4)


def test_only_complete_units_match_across_requests():
    cached = [10, 11, 12, 13, 14, 15, 16, 17]
    extended = cached + [20, 21, 22, 23]
    changed_on_boundary = [10, 11, 12, 13, 99, 15, 16, 17]
    changed_inside_unit = [10, 11, 99, 13, 14, 15, 16, 17]
    cached_units = build_cache_units(cached, 4)

    assert aligned_common_prefix_tokens(cached, extended, 4) == 8
    assert match_cached_prefix(cached_units, extended, 4).matched_tokens == 8
    assert match_cached_prefix(cached_units, changed_on_boundary, 4).matched_tokens == 4
    assert match_cached_prefix(cached_units, changed_inside_unit, 4).matched_tokens == 0


def test_paged_cache_exposes_the_same_aligned_unit_contract():
    cache = PagedKVCache(
        page_size=8,
        max_pages=2,
        device="cpu",
        dtype=torch.float32,
        cache_unit_size=4,
    )
    values = torch.zeros(1, 10, 2)
    cache.append_kv(values, values)

    stats = cache.get_stats()
    assert stats["cache_unit_size"] == 4
    assert stats["cache_units_per_page"] == 2
    assert stats["cache_unit_count"] == 2
    assert stats["cache_unit_remainder"] == 2
    assert cache.cache_unit_boundaries() == (0, 4, 8)


@pytest.mark.skipif(
    not EXPERIMENT.exists(),
    reason="缺 scripts/cache_unit_experiment.py（scripts 白名单未放行的实验脚本）",
)
def test_experiment_report_is_deterministic_and_machine_readable():
    result = subprocess.run(
        [sys.executable, str(EXPERIMENT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = __import__("json").loads(result.stdout)
    assert report["schema"] == "qlh.cache_unit_experiment.v1"
    assert report["layout"]["unit_boundaries_12"] == [0, 4, 8, 12]
    assert [item["matched_tokens"] for item in report["measurements"]] == [8, 4, 0]
    assert [item["raw_common_tokens"] for item in report["measurements"]] == [8, 4, 2]
