"""单元测试 — 设备画像到推理开关的显式映射（票 20 `DEVICE-AWARE-INFER-01` 第一步）。

覆盖：引擎／宿主 ISA 事实解析（含「不虚报」）、默认档位与接线前行为逐字一致、
环境变量覆盖与非法值忽略、`apply_to_load_kwargs` 的写入与「显式优先」，
以及设备画像与日志的可观测出口。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

import inference_switches as mod
from inference_switches import (
    ENV_FLASH_ATTN,
    ENV_KV_CACHE,
    ENV_THREADS,
    apply_to_load_kwargs,
    parse_engine_isa,
    resolve_inference_switches,
    server_argv_args,
)

#: 真实绑定输出样例（本机 llama-cpp-python，CPU-only 轮子：有 AVX2/FMA，无 AVX512/VNNI）。
REAL_ENGINE_LINE = (
    "CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | "
    "LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 |"
)


def _clean_env(monkeypatch) -> None:
    for name in (ENV_KV_CACHE, ENV_FLASH_ATTN, ENV_THREADS):
        monkeypatch.delenv(name, raising=False)


class TestEngineIsaFacts:
    def test_only_enabled_flags_are_kept(self):
        isa = parse_engine_isa(REAL_ENGINE_LINE)
        assert isa["AVX2"] == 1
        assert isa["FMA"] == 1
        assert isa["REPACK"] == 1
        # 构建未启用的一律不出现（「有硬件」≠「引擎启用」）
        assert "AVX512" not in isa
        assert "AVX512_VNNI" not in isa

    def test_zero_valued_and_unknown_flags_are_ignored(self):
        isa = parse_engine_isa("CPU : AVX2 = 0 | SOME_FUTURE_FLAG = 1 | AVX512 = 1")
        assert isa == {"AVX512": 1}

    def test_missing_or_empty_output_is_not_guessed(self):
        assert parse_engine_isa("") == {}
        assert parse_engine_isa("no flags here") == {}


class TestDefaultEqualsLegacyBehaviour:
    def test_default_switches_are_inert(self, monkeypatch):
        _clean_env(monkeypatch)
        switches = resolve_inference_switches(None, env={}, engine_isa={})
        assert switches.is_default
        kwargs: dict = {}
        apply_to_load_kwargs(switches, kwargs)
        assert kwargs == {}, "默认档位不得新增任何 load_kwargs 键"

    def test_threads_stay_with_call_site_by_default(self, monkeypatch):
        _clean_env(monkeypatch)
        assert resolve_inference_switches(None, env={}, engine_isa={}).n_threads is None

    def test_cpu_moe_stays_auto(self, monkeypatch):
        _clean_env(monkeypatch)
        assert resolve_inference_switches(None, env={}, engine_isa={}).cpu_moe == "auto"


class TestEnvOverrides:
    def test_kv_cache_enabled_explicitly(self, monkeypatch):
        _clean_env(monkeypatch)
        switches = resolve_inference_switches(None, env={ENV_KV_CACHE: "q8_0"}, engine_isa={})
        assert switches.kv_cache_type_k == "q8_0"
        assert switches.kv_cache_type_v == "q8_0"
        assert not switches.is_default
        assert switches.overrides["kv_cache_type"] == "q8_0"

    def test_kv_cache_off_is_explicit_and_still_inert(self, monkeypatch):
        _clean_env(monkeypatch)
        switches = resolve_inference_switches(None, env={ENV_KV_CACHE: "off"}, engine_isa={})
        assert switches.kv_cache_type_k is None
        assert switches.overrides["kv_cache_type"] == "off"

    def test_kv_cache_invalid_value_is_ignored(self, monkeypatch):
        _clean_env(monkeypatch)
        switches = resolve_inference_switches(None, env={ENV_KV_CACHE: "q3_k"}, engine_isa={})
        assert switches.kv_cache_type_k is None
        assert "kv_cache_type" not in switches.overrides

    def test_flash_attn_toggle(self, monkeypatch):
        _clean_env(monkeypatch)
        on = resolve_inference_switches(None, env={ENV_FLASH_ATTN: "1"}, engine_isa={})
        off = resolve_inference_switches(None, env={ENV_FLASH_ATTN: "0"}, engine_isa={})
        bogus = resolve_inference_switches(None, env={ENV_FLASH_ATTN: "maybe"}, engine_isa={})
        assert on.flash_attn is True
        assert off.flash_attn is False and off.overrides["flash_attn"] == "off"
        assert bogus.flash_attn is False and "flash_attn" not in bogus.overrides

    def test_threads_override_requires_positive_int(self, monkeypatch):
        _clean_env(monkeypatch)
        good = resolve_inference_switches(None, env={ENV_THREADS: "8"}, engine_isa={})
        zero = resolve_inference_switches(None, env={ENV_THREADS: "0"}, engine_isa={})
        text = resolve_inference_switches(None, env={ENV_THREADS: "eight"}, engine_isa={})
        assert good.n_threads == 8
        assert zero.n_threads is None and text.n_threads is None


class TestApplyToLoadKwargs:
    def test_explicit_call_site_values_win(self, monkeypatch):
        _clean_env(monkeypatch)
        switches = resolve_inference_switches(
            None,
            env={ENV_KV_CACHE: "q8_0", ENV_FLASH_ATTN: "1", ENV_THREADS: "8"},
            engine_isa={},
        )
        kwargs = {"type_k": "q4_0", "n_threads": 4}
        apply_to_load_kwargs(switches, kwargs)
        assert kwargs["type_k"] == "q4_0", "调用方显式值不得被画像映射覆盖"
        assert kwargs["n_threads"] == 4
        assert kwargs["type_v"] == "q8_0"
        assert kwargs["flash_attn"] is True

    def test_non_default_switches_reach_the_loader(self, monkeypatch):
        _clean_env(monkeypatch)
        switches = resolve_inference_switches(None, env={ENV_KV_CACHE: "q4_0"}, engine_isa={})
        kwargs: dict = {}
        apply_to_load_kwargs(switches, kwargs)
        assert kwargs == {"type_k": "q4_0", "type_v": "q4_0"}


class TestServerArgTranslation:
    """`llama-server` 命令行翻译：默认档位不产出参数，显式覆盖才产出。

    参数名以本机 `llama-server --help` 实测为准（`-t/--threads`、
    `-ctk/--cache-type-k`、`-ctv/--cache-type-v`、`-fa/--flash-attn [on|off|auto]`）。
    """

    def test_default_emits_no_server_args(self, monkeypatch):
        _clean_env(monkeypatch)
        switches = resolve_inference_switches(None, env={}, engine_isa={})
        assert server_argv_args(switches) == []

    def test_explicit_switches_become_server_args(self, monkeypatch):
        _clean_env(monkeypatch)
        switches = resolve_inference_switches(
            None,
            env={ENV_KV_CACHE: "q8_0", ENV_FLASH_ATTN: "0", ENV_THREADS: "6"},
            engine_isa={},
        )
        assert server_argv_args(switches) == [
            "--threads", "6",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "--flash-attn", "off",
        ]

    def test_flash_attn_absent_until_explicitly_overridden(self, monkeypatch):
        """不传 `--flash-attn` 即沿用 server 自身的 auto；显式关闭才写 `off`。"""
        _clean_env(monkeypatch)
        switches = resolve_inference_switches(None, env={ENV_KV_CACHE: "q4_0"}, engine_isa={})
        args = server_argv_args(switches)
        assert args == ["--cache-type-k", "q4_0", "--cache-type-v", "q4_0"]
        assert "--flash-attn" not in args


class TestObservability:
    def test_to_dict_carries_facts_and_evidence(self, monkeypatch):
        _clean_env(monkeypatch)
        switches = resolve_inference_switches(None, env={}, engine_isa={"AVX2": 1})
        payload = switches.to_dict()
        assert payload["schema_version"] == mod.SCHEMA_VERSION
        assert payload["engine_isa"] == {"AVX2": 1}
        assert payload["is_default"] is True
        assert payload["evidence"]["kv_cache_type"], "每条映射必须带实测依据"

    def test_describe_marks_closed_state(self, monkeypatch):
        _clean_env(monkeypatch)
        text = resolve_inference_switches(None, env={}, engine_isa={"AVX2": 1}).describe()
        assert "kv=off" in text
        assert "flash_attn=off" in text
        assert "threads=auto" in text
        assert "AVX2" in text

    def test_device_profile_exposes_switches(self, monkeypatch):
        _clean_env(monkeypatch)
        from device_profiler import DeviceProfiler

        profile = DeviceProfiler.mock_edge().to_dict()
        assert profile["inference_switches"] is not None
        assert profile["inference_switches"]["is_default"] is True


class TestKvBudgetAndCtx:
    """票 20 完成定义第三条：`n_ctx` 由每段 KV 预算（层数 × 每层字节 × ctx）推导。"""

    def test_bytes_per_token_counts_k_and_v_per_layer(self):
        # 24 层、4 个 KV 头、head_dim 128、f16 ⇒ 24 × 4 × (128×2 + 128×2)
        assert mod.kv_bytes_per_token(24, 4, 128) == 24 * 4 * 512

    def test_bytes_per_token_follows_the_quantised_cache_type(self):
        f16 = mod.kv_bytes_per_token(24, 4, 128)
        q8 = mod.kv_bytes_per_token(24, 4, 128, cache_type_k="q8_0", cache_type_v="q8_0")
        q4 = mod.kv_bytes_per_token(24, 4, 128, cache_type_k="q4_0", cache_type_v="q4_0")
        assert q8 < f16 and q4 < q8
        assert q4 == pytest.approx(f16 * (18 / 32) / 2, rel=1e-6)

    def test_suggest_n_ctx_is_budget_over_bytes_per_token(self):
        per_token = mod.kv_bytes_per_token(24, 4, 128)
        assert mod.suggest_n_ctx(
            budget_bytes=per_token * 4096, layer_count=24, kv_heads=4, key_length=128,
        ) == 4096

    def test_suggest_n_ctx_never_widens_past_max_ctx(self):
        per_token = mod.kv_bytes_per_token(24, 4, 128)
        assert mod.suggest_n_ctx(
            budget_bytes=per_token * 100000, layer_count=24, kv_heads=4, key_length=128,
            max_ctx=4096,
        ) == 4096

    def test_suggest_n_ctx_refuses_to_conclude_below_min_ctx(self):
        per_token = mod.kv_bytes_per_token(24, 4, 128)
        assert mod.suggest_n_ctx(
            budget_bytes=per_token * 10, layer_count=24, kv_heads=4, key_length=128,
        ) is None
        assert mod.suggest_n_ctx(
            budget_bytes=0, layer_count=24, kv_heads=4, key_length=128,
        ) is None

    def test_budget_from_profile_is_available_memory_over_eight(self):
        budget = mod.kv_budget_bytes_from_profile({"ram": {"available_gb": 32}})
        assert budget == int(32 * (1024 ** 3) * mod.KV_BUDGET_MEMORY_RATIO)

    def test_budget_from_profile_is_none_without_usable_ram(self):
        assert mod.kv_budget_bytes_from_profile({}) is None
        assert mod.kv_budget_bytes_from_profile({"ram": {"available_gb": "many"}}) is None
        assert mod.kv_budget_bytes_from_profile({"ram": {"available_gb": 0}}) is None

    def test_from_gguf_uses_header_fields(self, monkeypatch):
        """读 GGUF 头（不加载模型）+ 预算 ⇒ 推导；head_count_kv 为按层数组时取最大。"""
        metadata = {"block_count": 24, "head_count_kv": [4, 4, 8], "key_length": 128}
        monkeypatch.setattr("gguf_metadata_contract.read_gguf_metadata", lambda path: dict(metadata))
        per_token = mod.kv_bytes_per_token(24, 8, 128)
        assert mod.suggest_n_ctx_from_gguf(
            "dummy.gguf", budget_bytes=per_token * 2048, max_ctx=8192,
        ) == 2048

    def test_from_gguf_returns_none_when_header_unreadable(self, monkeypatch):
        def _boom(path):
            raise OSError("not a gguf")

        monkeypatch.setattr("gguf_metadata_contract.read_gguf_metadata", _boom)
        assert mod.suggest_n_ctx_from_gguf("dummy.gguf", budget_bytes=10 ** 9) is None

    def test_from_gguf_returns_none_without_required_fields(self, monkeypatch):
        monkeypatch.setattr("gguf_metadata_contract.read_gguf_metadata",
                            lambda path: {"block_count": 24})
        assert mod.suggest_n_ctx_from_gguf("dummy.gguf", budget_bytes=10 ** 9) is None
