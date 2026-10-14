"""Route-A stopping through the real decode loop, with no models or network."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from llama_engine import CHATML_STOP_SEQUENCES, LlamaCppEngine
from scheduler_pipeline import SchedulerPipelineMixin, _route_a_eos_ids


class _HeadTokenizer:
    """A head GGUF whose ChatML sentinels are ordinary multi-token text."""

    def __init__(self, pieces, eos=999):
        self.eos = eos
        self.prompt = "PROMPT: "
        self.pieces = {1: b"PROMPT: ", 2: b"", 999: b"<eos>"}
        self.pieces.update({10 + i: piece for i, piece in enumerate(pieces)})
        self.single_stops = {}
        self.special_calls = []

    def token_eos(self):
        return self.eos

    def tokenize(self, text, *, add_bos=True, special=False):
        if text == self.prompt.encode():
            return [1, 2]
        if text.decode() in self.single_stops:
            return [self.single_stops[text.decode()]]
        return [10000 + byte for byte in text]

    def detokenize(self, tokens, *, special=False):
        self.special_calls.append(special)
        return b"".join(
            bytes([token - 10000]) if token >= 10000 else self.pieces[token]
            for token in tokens
        )


class _KeepHeadStringTokenizer(_HeadTokenizer):
    """LL facade: no optional keywords and detokenize already returns str."""

    def tokenize(self, text):
        return super().tokenize(text)

    def detokenize(self, tokens):
        return super().detokenize(tokens).decode("utf-8", "replace")


class _NoTorch:
    def __getattr__(self, name):
        raise AssertionError(f"numpy Route-A must not call torch.{name}")


class _RouteAHarness(SchedulerPipelineMixin):
    def __init__(self, engine, backend, generated):
        self._host = engine
        self.backend = backend
        self.generated = iter(generated)
        self.stage_calls = []
        self.forward_calls = []
        self.layer_loads = []
        self._pipeline_lock = threading.Lock()
        self._kv_cache_lock = threading.Lock()
        self._pipeline_context = threading.local()
        self._pipeline_active_tasks = set()
        self._pipeline_task_contracts = {}
        self._pipeline_results = {}
        self._pipeline_events = {}
        self._chain_ack_state = {}
        self._kv_cache = {}
        self.master = dict(node_id="master", start_layer=0, end_layer=4,
                           layers_count=4, has_embedding=True, has_lm_head=False)
        self.stages = [self._stage("middle", 4, 8), self._stage("tail", 8, 12)]

    @staticmethod
    def _stage(node_id, start, end):
        return dict(node_id=node_id, start_layer=start, end_layer=end,
                    layers_count=end - start, execution="stage_offer_v3",
                    stage_model_identity=dict(model_id=f"head-{node_id}",
                                              engine="llama_cpp", format="gguf",
                                              revision="local", sha256="c" * 64))

    def _require_callbacks(self):
        from api_server import _format_model_response

        return SimpleNamespace(
            thinking_system_prompt="",
            active_task_graph_model_identity=lambda: SimpleNamespace(
                sha256="a" * 64, model_id="head4"),
            build_model_chat_prompt=lambda *_args, **_kwargs: self.backend.prompt,
            format_model_response=_format_model_response,
        )

    def get_effective_node_id(self):
        return "master"

    def get_layer_assignments(self):
        return {"assignments": [self.stages[1], self.master, self.stages[0]]}

    def _scheduler_facade_global(self, name):
        assert name == "torch"
        return _NoTorch()

    def _verify_pipeline_readiness(self, _nodes):
        return True, "ok"

    def _execute_layer_stage_offer(self, **kwargs):
        self.stage_calls.append(kwargs)
        if kwargs["want_hidden"]:
            return {"kind": "hidden", "hidden_states": kwargs["hidden_states"]}
        return {"kind": "token", "token_argmax": next(self.generated)}

    def _close_relay_segment_client(self, _task_id):
        pass

    def _record_pipeline_task_accounting(self, **kwargs):
        assert kwargs["success"] is True
        return {"workers_counted": ["middle", "tail"], "counted_nodes": ["master"]}

    def run(self, events=None, **kwargs):
        return self.run_pipeline(
            "hello", max_new_tokens=kwargs.pop("max_new_tokens", 64),
            temperature=0.0, top_p=1.0,
            _stream_callback=events.append if events is not None else None, **kwargs,
        )


@pytest.fixture
def route(monkeypatch):
    def build(pieces, eos=999, backend_cls=_HeadTokenizer):
        pieces = [piece.encode() if isinstance(piece, str) else piece for piece in pieces]
        backend = backend_cls(pieces, eos)
        engine = LlamaCppEngine()
        engine._loaded = True
        engine._model = backend
        scheduler = _RouteAHarness(engine, backend, [10 + i for i in range(len(pieces))] + [999])

        def no_load(*_args, **_kwargs):
            raise AssertionError("stopping tests must never load a model")

        def ensure(*args, **kwargs):
            scheduler.layer_loads.append((args, kwargs))

        def forward(**kwargs):
            scheduler.forward_calls.append(kwargs)
            count = int(kwargs["input_ids"].shape[1])
            return {"hidden_states": np.zeros((1, count, 4), dtype=np.float32),
                    "cache": (kwargs["past_key_values"] or 0) + count}

        monkeypatch.setattr(engine, "load_model", no_load)
        monkeypatch.setattr(engine, "ensure_layer_range", ensure, raising=False)
        monkeypatch.setattr(engine, "forward_layers", forward)
        return scheduler

    return build


def _stream_text(events):
    return "".join(event.get("token", "") for event in events)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("backend_cls", [_HeadTokenizer, _KeepHeadStringTokenizer])
def test_ordinary_chatml_stop_ends_real_route_a_before_next_turn(route, streaming, backend_cls):
    scheduler = route(["2", "<", "|", "im", "_start", "|", ">user\nnext round", "LEAK"],
                      backend_cls=backend_cls)
    events = []
    result = scheduler.run(events if streaming else None)
    assert result["response"] == "2"
    assert result["full_text"] == "PROMPT: 2"
    assert result["finish_reason"] == "stop"
    assert result["metrics"]["stop_reason"] == "text_stop"
    assert result["metrics"]["matched_stop"] == "<|im_start|>"
    assert result["metrics"]["eos_token_ids"] == [999]
    assert len(scheduler.forward_calls) == 7
    assert [call["node_id"] for call in scheduler.stage_calls] == ["middle", "tail"] * 7
    assert [call["positions"] for call in scheduler.stage_calls[::2]] == [[0, 1], [2], [3], [4], [5], [6], [7]]
    assert all(call["middle_channel"] == "keep_head_layer_out" for call in scheduler.stage_calls)
    assert scheduler.layer_loads[0][0] == (0, 4)
    assert not scheduler._kv_cache and not scheduler._pipeline_task_contracts
    assert not scheduler._pipeline_active_tasks and scheduler._pipeline_context.stack == []
    if streaming:
        assert _stream_text(events) == "2"
        assert events[-1]["done"] is True
        assert events[-1]["response"] == "2"


@pytest.mark.parametrize("stop", [["END"], "END"])
def test_explicit_custom_stop_crosses_tokens_and_keeps_text_in_same_token(route, stop):
    scheduler = route(["answer E", "N", "D trailing text", "LEAK"])
    events = []
    result = scheduler.run(events, stop=stop)
    assert result["response"] == "answer"
    assert _stream_text(events) == "answer "
    assert result["metrics"]["matched_stop"] == "END"
    assert len(scheduler.forward_calls) == 3


@pytest.mark.parametrize("eos", [None, -1, -7])
def test_missing_or_negative_eos_uses_text_stop_with_named_diagnostic(route, eos, caplog):
    scheduler = route(["2<|", "im_", "start|>", "LEAK"], eos=eos)
    events = []
    result = scheduler.run(events)
    assert "error" not in result
    assert result["response"] == _stream_text(events) == "2"
    assert result["metrics"]["eos_token_ids"] == []
    assert result["metrics"]["stop_diagnostic"] == "route_a_eos_unavailable_text_stop"
    assert "route_a_eos_unavailable_text_stop" in caplog.text
    assert len(scheduler.forward_calls) == 3


def test_false_stop_prefix_is_released_without_losing_content(route):
    scheduler = route(["literal <|im_", "x and E", "lse", " done"])
    events = []
    result = scheduler.run(events, stop=["END"])
    assert result["response"] == _stream_text(events) == "literal <|im_x and Else done"
    assert result["metrics"]["stop_reason"] == "eos_token"


@pytest.mark.parametrize("pieces,expected", [(["2", "<|im_", "start"], "2"), (["plain answer"], "plain answer")])
def test_length_limit_keeps_stream_and_final_text_consistent(route, pieces, expected):
    scheduler = route(pieces)
    events = []
    result = scheduler.run(events, max_new_tokens=len(pieces))
    assert result["response"] == _stream_text(events) == expected
    assert result["full_text"] == "PROMPT: " + expected
    assert result["finish_reason"] == "length"


def test_cumulative_decode_preserves_utf8_characters_split_across_tokens(route):
    scheduler = route([b"\xe4", b"\xbd", b"\xa0", b"\xe5\xa5", b"\xbd<|im_", b"end|>LEAK"])
    events = []
    result = scheduler.run(events)
    assert result["response"] == _stream_text(events) == "\u4f60\u597d"
    assert "\ufffd" not in _stream_text(events)


def test_stop_preserves_visible_reasoning_and_answer(route):
    scheduler = route(["<think>calculation", "</think>", "2<|", "im_start|>user", "LEAK"])
    events = []
    result = scheduler.run(events, show_thinking=True)
    assert result["response"] == "2"
    assert result["thinking"] == "calculation"
    assert _stream_text(events) == "<think>calculation</think>2"


def test_native_thinking_suppression_and_stop_share_safe_text(route):
    scheduler = route(["private reasoning", "</thi", "nk>2<|", "im_start|>user", "LEAK"])
    scheduler.backend.prompt = "PROMPT: <think>"
    events = []
    result = scheduler.run(events)
    assert result["response"] == _stream_text(events) == "2"


def test_unfinished_native_thinking_is_not_flushed_on_stop(route):
    scheduler = route(["private reasoning", "<|im_", "end|>"])
    scheduler.backend.prompt = "PROMPT: <think>"
    events = []
    result = scheduler.run(events)
    assert result["response"] == _stream_text(events) == ""
    assert "private reasoning" in result["full_text"]


def test_route_a_passes_explicit_thinking_choice_to_native_template(route, monkeypatch):
    scheduler = route(["2", "<|im_end|>"])
    callbacks = scheduler._require_callbacks()
    seen = []

    def build_prompt(*args, **kwargs):
        seen.append(kwargs)
        return scheduler.backend.prompt

    callbacks.build_model_chat_prompt = build_prompt
    monkeypatch.setattr(scheduler, "_require_callbacks", lambda: callbacks)
    assert scheduler.run(enable_thinking=False)["response"] == "2"
    assert seen[0]["enable_thinking"] is False


def test_engine_stop_contract_and_single_token_mapping(route):
    scheduler = route([])
    engine = scheduler._host
    assert engine._merge_stop_sequences(["END", "END", ""]) == ["END"] + CHATML_STOP_SEQUENCES
    scheduler.backend.single_stops = {"<|im_end|>": 50, "END": 51, "missing": 52}
    scheduler.backend.pieces.update({50: b"<|im_end|>", 51: b"END", 52: b"<unk>"})
    assert engine._get_generation_eos_token_ids(["<|im_end|>", "END", "missing", "MULTI"]) == [999, 50, 51]
    assert True in scheduler.backend.special_calls


def test_no_eos_and_no_text_stops_is_rejected_before_generation(route, monkeypatch):
    scheduler = route(["LEAK"], eos=-1)
    monkeypatch.setattr(scheduler._host, "_merge_stop_sequences", lambda _stop: [])
    result = scheduler.run()
    assert result["error"] == "route_a_stop_configuration_invalid:no_eos_or_text_stop"
    assert not scheduler.forward_calls and not scheduler.stage_calls


def test_invalid_custom_stop_is_named_and_never_generates(route):
    scheduler = route(["LEAK"])
    result = scheduler.run(stop=[123])
    assert result["error"].startswith("route_a_stop_configuration_invalid:")
    assert not scheduler.forward_calls


@pytest.mark.parametrize("value,expected", [(None, set()), (-1, set()), (True, set()),
                                          (999, {999}), ([-1, None, 0, 50, False], {0, 50})])
def test_eos_normalization(value, expected):
    assert _route_a_eos_ids(value) == expected


def test_route_a_inference_keeps_user_selection_identity_distinct_from_task_contract(route):
    scheduler = route(["2", "<|im_end|>"])
    selection = {"operation_id": "supply_" + "c" * 32, "config_id": "model-config", "generation": 73}
    for stage in scheduler.stages:
        stage["worker_prepare_identity"] = dict(selection)
    metrics = scheduler.run()["metrics"]
    assert metrics["model_operation_id"] == selection["operation_id"]
    assert metrics["model_config_id"] == "model-config" and metrics["model_generation"] == 73
    assert metrics["config_id"].startswith("route_a:")


def test_route_a_never_claims_one_selection_for_mixed_operation_receipts(route):
    scheduler = route(["2", "<|im_end|>"])
    for index, stage in enumerate(scheduler.stages):
        stage["worker_prepare_identity"] = {"operation_id": "supply_" + str(index) * 32,
                                           "config_id": "model-config", "generation": 73}
    assert "model_operation_id" not in scheduler.run()["metrics"]