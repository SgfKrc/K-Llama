"""
T9 共享层（src/tui_shared.py）单元测试
====================================
覆盖：interactive 请求体构造、metrics 格式化、路由参数解析、命令注册表。
"""

import sys
import os
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tui_shared import (
    API_PATHS,
    COMMAND_SPECS,
    build_interactive_request,
    format_metrics,
    help_text,
    load_local_chat_image,
    parse_session_line,
    resolve_route_arg,
)


class TestBuildInteractiveRequest:
    def test_minimal_request(self):
        body = build_interactive_request("你好")
        assert body["message"] == "你好"
        assert body["streaming_mode"] == "interactive"
        assert body["routing_preference"] == "auto"
        assert body["show_thinking"] is False
        assert body["generation_id"] is None

    def test_full_request(self):
        body = build_interactive_request(
            "hi",
            session_id="s1",
            generation_id="gen_x",
            routing_preference="distributed_required",
            show_thinking=True,
        )
        assert body["session_id"] == "s1"
        assert body["generation_id"] == "gen_x"
        assert body["routing_preference"] == "distributed_required"
        assert body["show_thinking"] is True

    @pytest.mark.parametrize("mode, expected", [(None, "interactive"), ("interactive", "interactive"), ("full", "full")])
    def test_explicit_streaming_mode_preserves_routing(self, mode, expected):
        body = build_interactive_request("hi", streaming_mode=mode, routing_preference="distributed_required")
        assert body["streaming_mode"] == expected
        assert body["routing_preference"] == "distributed_required"

    @pytest.mark.parametrize("mode", ["fast", "FULL", "", False])
    def test_invalid_streaming_mode_is_rejected(self, mode):
        with pytest.raises(ValueError, match="streaming_mode"):
            build_interactive_request("hi", streaming_mode=mode)

    def test_local_image_rejects_explicit_interactive(self):
        with pytest.raises(ValueError, match="streaming_mode='full'"):
            build_interactive_request(
                "描述图片", routing_preference="local_only", streaming_mode="interactive",
                image_data_urls=["data:image/png;base64,iVBORw0KGgo="],
            )

    def test_invalid_routing_falls_back_to_auto(self):
        body = build_interactive_request("hi", routing_preference="bogus")
        assert body["routing_preference"] == "auto"

    def test_execution_mode_defaults_to_auto_and_is_transmitted(self):
        """Retired task_graph selection resolves to the supported auto mode."""
        assert build_interactive_request("hi")["execution_mode"] == "auto"
        assert (
            build_interactive_request("hi", execution_mode="task_graph")["execution_mode"]
            == "auto"
        )

    def test_invalid_execution_mode_falls_back_to_auto(self):
        assert (
            build_interactive_request("hi", execution_mode="bogus")["execution_mode"]
            == "auto"
        )

    def test_image_request_requires_external_and_preserves_data_url(self):
        image = "data:image/png;base64,iVBORw0KGgo="
        body = build_interactive_request("描述图片", image_data_urls=[image])
        assert body["image_data_urls"] == [image]
        assert body["allow_external"] is True
        assert body["prefer_external"] is True
        assert body["execution_mode"] == "auto"

    def test_image_request_supports_local_only_full_mode(self):
        body = build_interactive_request(
            "描述图片",
            routing_preference="local_only",
            image_data_urls=["data:image/png;base64,iVBORw0KGgo="],
        )
        assert body["streaming_mode"] == "full"
        assert body["allow_external"] is False
        assert body["prefer_external"] is False

    def test_image_request_rejects_multiple_local_images(self):
        with pytest.raises(ValueError, match="一张"):
            build_interactive_request(
                "描述图片",
                routing_preference="local_only",
                image_data_urls=[
                    "data:image/png;base64,iVBORw0KGgo=",
                    "data:image/png;base64,iVBORw0KGgo=",
                ],
            )


class TestLocalImageLoader:
    def test_loads_png_as_valid_data_url(self, tmp_path):
        image_path = Path(tmp_path) / "tiny.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        image = load_local_chat_image(str(image_path))
        assert image["name"] == "tiny.png"
        assert image["data_url"].startswith("data:image/png;base64,")

    def test_rejects_non_image_file(self, tmp_path):
        image_path = Path(tmp_path) / "not-image.txt"
        image_path.write_text("not an image", encoding="utf-8")
        with pytest.raises(ValueError, match="PNG、JPEG 或 WebP"):
            load_local_chat_image(str(image_path))


class TestFormatMetrics:
    def test_engine_and_tokens(self):
        text = format_metrics(
            {"engine": "llama_cpp", "execution_mode": "local",
             "tokens_generated": 42, "tok_per_sec": 12.34},
        )
        assert "llama_cpp" in text and "local" in text
        assert "42 tokens" in text
        assert "12.3 tok/s" in text

    def test_fallback_reason_shown(self):
        text = format_metrics(
            {"engine": "external_api", "execution_mode": "external_api",
             "fallback": True, "fallback_reason": "timeout"},
        )
        assert "回退" in text and "timeout" in text

    def test_distributed_requested_but_not_used(self):
        text = format_metrics(
            {"engine": "llama_cpp", "execution_mode": "local",
             "distributed_requested": True},
        )
        assert "已请求分布式，实际本地" in text

    def test_distributed_used_true_is_shown_with_evidence(self):
        """★ 2026-10-08：真分布式必须**显式**显示 + 承层证据。

        旧规则下 `distributed_used=True` 与「metrics 里根本没这个字段」渲染结果
        完全一样 ⇒ 用户只能靠吞吐速度猜是不是分布式。
        """
        text = format_metrics(
            {"engine": "distributed_pipeline",
             "execution_mode": "route_a_stage_offer_v3",
             "distributed_requested": True, "distributed_used": True,
             "layer_segments": [[8, 15], [16, 23]]},
        )
        assert "分布式 ✓" in text
        assert "2 段·层 8-23" in text

    def test_distributed_used_false_is_explicit_not_silent(self):
        """★ 静默回退对策：后端**明确**给了 `distributed_used=False` 就必须说"本地执行"。

        这一条正是「整模回退伪装成普通本地推理」在 UI 上的对策：`execution_mode`
        可能仍显示得像分布式，只有这个字段能戳破。
        """
        text = format_metrics(
            {"engine": "pytorch", "execution_mode": "local", "distributed_used": False,
             "fallback": True,
             "fallback_reason": "pipeline_failed_then_local_pytorch: model_identity_mismatch"},
        )
        assert "本地执行" in text
        assert "model_identity_mismatch" in text

    def test_claimed_layers_used_when_no_segments(self):
        text = format_metrics({"distributed_used": True, "claimed_layers": [4, 19]})
        assert "层 4-19" in text

    def test_workers_used_as_evidence_fallback(self):
        text = format_metrics({"distributed_used": True, "workers_used": ["y700-1"]})
        assert "1 个 worker" in text

    def test_no_distributed_field_keeps_footer_unchanged(self):
        """没有该字段时**不得**擅自宣称本地（旧行为保持）。"""
        assert format_metrics({"engine": "llama_cpp", "execution_mode": "local"}) == (
            "llama_cpp · local")

    def test_request_side_intent_alone_triggers_the_warning(self):
        """★ 2026-10-08：**只看请求侧意图**也必须提示。

        场景（实测的静默缺口）：用户用 `/route required` 要求分布式，但全局开关
        `distributed_inference` 恰好关着 ⇒ `distributed_requested` 为 False，若只看它
        则界面**什么都不显示**，用户以为在分布式跑。现按 `routing_preference` 兜住。
        """
        text = format_metrics(
            {"engine": "pytorch", "execution_mode": "local",
             "distributed_requested": False, "distributed_used": False,
             "routing_preference": "distributed_required"},
        )
        assert "已请求分布式，实际本地" in text

    def test_preferred_route_also_counts_as_requesting_distributed(self):
        text = format_metrics(
            {"engine": "pytorch", "execution_mode": "local",
             "distributed_used": False, "routing_preference": "distributed_preferred"},
        )
        assert "已请求分布式，实际本地" in text

    def test_auto_route_does_not_claim_distributed_was_requested(self):
        """`auto` 不是分布式意图 ⇒ 不得声称"已请求分布式"。"""
        text = format_metrics(
            {"engine": "pytorch", "execution_mode": "local",
             "distributed_used": False, "routing_preference": "auto"},
        )
        assert "已请求分布式" not in text
        assert "本地执行" in text

    def test_done_model_identity_preserves_two_remote_layer_evidence(self):
        metrics = {
            "engine": "distributed_pipeline", "execution_mode": "route_a_stage_offer_v3",
            "distributed_used": True, "workers_used": ["worker-surface", "worker-android"],
            "layer_segments": [[0, 15], [16, 31]], "generated_tokens": 42,
            "tokens_per_second": 12.34, "config_id": "route_a:request_fixture",
            "generation_id": "gen_request_fixture", "model_operation_id": "supply_response",
            "model_config_id": "config_response", "model_generation": 7,
        }
        text = format_metrics(metrics, history_committed=False)
        assert "分布式 ✓（2 段·层 0-31）" in text
        assert "route_a_stage_offer_v3" in text
        assert "42 tokens" in text and "12.3 tok/s" in text
        assert "模型准备：supply_response" in text
        assert "模型配置：config_response" in text
        assert "模型代次：7" in text
        assert "route_a:request_fixture" not in text
        assert "gen_request_fixture" not in text
        assert "历史未提交" in text

    @pytest.mark.parametrize(("identity", "expected"), [
        ({}, []),
        ({"model_operation_id": "supply_response"}, ["模型准备：supply_response"]),
        ({"model_config_id": "config_response"}, ["模型配置：config_response"]),
        ({"model_generation": 0}, ["模型代次：0"]),
        ({"model_operation_id": "", "model_config_id": None, "model_generation": None}, []),
    ])
    def test_only_returned_model_identity_fields_are_shown(self, identity, expected):
        text = format_metrics({
            "engine": "distributed_pipeline", "execution_mode": "route_a_stage_offer_v3",
            "distributed_used": True, "workers_used": ["worker-surface", "worker-android"],
            "config_id": "route_a:request_fixture", "generation_id": "gen_request_fixture",
            **identity,
        })
        assert "分布式 ✓（2 个 worker）" in text
        for value in expected:
            assert value in text
        for label in ("模型准备：", "模型配置：", "模型代次："):
            if not any(value.startswith(label) for value in expected):
                assert label not in text
        assert "route_a:request_fixture" not in text
        assert "gen_request_fixture" not in text

    def test_history_not_committed(self):
        text = format_metrics({}, history_committed=False)
        assert "历史未提交" in text

    def test_empty_metrics(self):
        assert format_metrics(None) == "unknown · local"


class TestRouteArg:
    def test_full_names(self):
        assert resolve_route_arg("auto") == "auto"
        assert resolve_route_arg("local_only") == "local_only"

    def test_short_aliases(self):
        assert resolve_route_arg("local") == "local_only"
        assert resolve_route_arg("distributed") == "distributed_preferred"
        assert resolve_route_arg("required") == "distributed_required"

    def test_invalid(self):
        assert resolve_route_arg("sideways") is None
        assert resolve_route_arg("") is None


class TestCommandRegistry:
    def test_help_text_mentions_key_commands(self):
        text = help_text()
        for name in ("/new", "/resume", "/route", "/cancel", "/quit"):
            assert name in text

    def test_specs_unique_and_ordered(self):
        names = [spec["name"] for spec in COMMAND_SPECS]
        assert len(names) == len(set(names))
        assert names[0] == "/new" and names[-1] == "/quit"

    def test_session_line(self):
        line = parse_session_line({"session_id": "s1", "title": "调试"})
        assert line == "s1  调试"
        assert parse_session_line({}) == "  (未命名)"


class TestApiPaths:
    def test_cancel_template(self):
        assert API_PATHS["chat_cancel"].format(
            generation_id="gen_x",
        ) == "/chat/generations/gen_x/cancel"

    def test_cluster_resources_path(self):
        assert API_PATHS["cluster_resources"] == "/cluster/resources"

    def test_ha_observation_paths(self):
        assert API_PATHS["cluster_control_plane"] == "/cluster/control-plane"
        assert API_PATHS["cluster_management_score"] == "/cluster/management-score"
        assert API_PATHS["cluster_transfer_logs"] == "/cluster/transfer-logs"

    def test_pipeline_observation_paths(self):
        assert API_PATHS["cluster_layers"] == "/cluster/layers"
        assert API_PATHS["cluster_pipeline_capacity"] == "/cluster/pipeline-capacity"
        assert API_PATHS["cluster_pipeline_reshard"] == "/cluster/pipeline-reshard"


@pytest.mark.parametrize("identity", [
    {"model_operation_id": "supply_response", "model_config_id": "config_response", "model_generation": 7},
    {},
])
def test_textual_done_status_uses_response_metrics_identity(monkeypatch, identity):
    """Exercise the real ChatPane SSE worker and rendered status with a response fixture."""
    import asyncio

    pytest.importorskip("textual")
    from textual.app import App
    from textual.widgets import Static
    import tui_textual as tt

    def stream(api, message, **kwargs):
        yield {"token": "推理完成"}
        yield {"done": True, "response": "推理完成", "metrics": {
            "engine": "distributed_pipeline", "execution_mode": "route_a_stage_offer_v3",
            "distributed_used": True, "workers_used": ["worker-surface", "worker-android"],
            "layer_segments": [[0, 15], [16, 31]], **identity,
        }}

    monkeypatch.setattr(tt, "iter_chat_payloads", stream)

    class ChatStatusApp(App):
        api = object()
        routing_preference = "distributed_required"
        show_thinking = False
        enable_thinking = False
        model_intent = {"operation_id": "supply_ui_current", "config_id": "config_ui_current", "generation": 99}

        def compose(self):
            yield tt.ChatPane()

    async def run():
        app = ChatStatusApp()
        async with app.run_test(size=(140, 25)) as pilot:
            pane = app.query_one(tt.ChatPane)
            await pane.stream_reply("检查本次推理身份").wait()
            await pilot.pause()
            text = str(pane.query_one("#chat-status", Static).content)
            assert "推理完成" in str(pane.query_one("#chat-log", Static).content)
            assert "分布式 ✓（2 段·层 0-31）" in text
            assert "route_a_stage_offer_v3" in text
            if identity:
                assert "模型准备：supply_response" in text
                assert "模型配置：config_response" in text
                assert "模型代次：7" in text
            else:
                assert "模型准备：" not in text
                assert "模型配置：" not in text
                assert "模型代次：" not in text
            assert "ui_current" not in text
            assert "99" not in text

    asyncio.run(run())
