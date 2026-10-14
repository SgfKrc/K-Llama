"""PC 侧 v3 层段 Stage 的执行（`EngineHost._execute_layer_forward_stage`）。

补 #33 的第二处接线：`execute_task_worker_stage` 此前只认 `full_inference` /
`aggregate`，`layer_forward` 一律落到「不支持的 Stage 类型」—— 尽管能力声明、
offer 校验、租约与结果回程都早已就位。

测试用假 upstream 顶掉 keep-head 子进程，只验证**契约**：形状校验、中间段的
`hidden_out_f32` + 摘要、尾段的 `token_argmax`、以及 `want_hidden` 的分叉。
"""
import base64
import hashlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference_service.engine_host import EngineHost  # noqa: E402


N_TOKENS = 3
N_EMBD = 4


class _FakeUpstream:
    """顶掉 `KeepHeadUpstream`：记录入参，按固定规则产出。"""

    def __init__(self):
        self.hidden_calls = []
        self.token_calls = []
        self.reset_count = 0
        # ★ `EngineHost._layer_forward_upstream`（engine_host.py:2925）会读
        #   `_layer_upstream.is_available` 决定复用现有上游还是重建，替身需提供该属性。
        self.is_available = True

    def reset(self):
        # 批次从位置 0 开始时（新 prompt 的 prefill）上游必须清 KV，否则上一个请求的
        # 位置会残留（`decode: failed to initialize batch`）。见
        # `EngineHost._execute_layer_forward_stage`。
        self.reset_count += 1

    def forward_hidden_to_hidden(self, hidden, *, n_past=0, seq_ids=None, positions=None):
        self.hidden_calls.append(
            {"n_past": n_past, "seq_ids": seq_ids, "positions": positions,
             "shape": tuple(np.asarray(hidden).shape)},
        )
        return np.asarray(hidden, dtype=np.float32) + 1.0

    def forward_hidden_to_token(self, hidden, *, n_past=0, seq_ids=None, positions=None):
        self.token_calls.append(
            {"n_past": n_past, "seq_ids": seq_ids, "positions": positions,
             "shape": tuple(np.asarray(hidden).shape)},
        )
        return 42


def _hidden_bytes():
    return np.arange(N_TOKENS * N_EMBD, dtype=np.float32).tobytes()


def _request(*, want_hidden, n_tokens=N_TOKENS, n_embd=N_EMBD, positions=None,
             seq_ids=None):
    stage_fields = {
        "layer_range": [16, 20],
        "handoff_at": 16,
        "hidden_sha256": "0" * 64,
        "hidden_spec": {"n_tokens": n_tokens, "n_embd": n_embd, "dtype": "float32"},
    }
    if positions is not None:
        stage_fields["positions"] = positions
    if seq_ids is not None:
        stage_fields["seq_ids"] = seq_ids
    return SimpleNamespace(
        stage_type="layer_forward",
        stage_id="node:step:1",
        workflow_id="wf_test",
        provider_id="remote_x",
        stage_fields=stage_fields,
        root_input={
            "hidden_f32": base64.b64encode(_hidden_bytes()).decode("ascii"),
            "want_hidden": want_hidden,
            "pos_base": 0,
            "context_size": 4096,
        },
    )


def _host(upstream):
    host = EngineHost.__new__(EngineHost)
    host._layer_upstream = upstream
    return host


class TestIntermediateStage:
    def test_resets_kv_when_batch_starts_at_zero(self):
        """新 prompt 的 prefill（positions 从 0 起）必须先清上游 KV。

        否则上一个请求的位置会残留，llama.cpp 报
        `the last position stored in the KV cache for sequence 0 is X = 45` 而 batch
        起始位置是 0 ⇒ `decode: failed to initialize batch`（实测：三段链的第二个请求
        起必失败）。
        """
        upstream = _FakeUpstream()
        _host(upstream)._execute_layer_forward_stage(
            _request(want_hidden=True, positions=[0, 1, 2]),
            SimpleNamespace(is_set=lambda: False),
        )
        assert upstream.reset_count == 1

    def test_does_not_reset_kv_on_incremental_decode(self):
        """decode 步（positions 不从 0 起）**不能**清 KV —— 那会丢掉 prefill 的上下文。"""
        upstream = _FakeUpstream()
        _host(upstream)._execute_layer_forward_stage(
            _request(want_hidden=True, positions=[41, 42, 43]),
            SimpleNamespace(is_set=lambda: False),
        )
        assert upstream.reset_count == 0

    def test_returns_hidden_with_digest(self):
        upstream = _FakeUpstream()
        result = _host(upstream)._execute_layer_forward_stage(
            _request(want_hidden=True), SimpleNamespace(is_set=lambda: False),
        )
        raw = base64.b64decode(out := result["hidden_out_f32"], validate=True)
        assert result["hidden_out_sha256"] == hashlib.sha256(raw).hexdigest()
        assert result["token_argmax"] is None
        arr = np.frombuffer(raw, dtype=np.float32).reshape(N_TOKENS, N_EMBD)
        expected = np.arange(N_TOKENS * N_EMBD, dtype=np.float32).reshape(
            N_TOKENS, N_EMBD,
        ) + 1.0
        assert np.array_equal(arr, expected)
        assert upstream.token_calls == []

    def test_positions_are_passed_through(self):
        upstream = _FakeUpstream()
        _host(upstream)._execute_layer_forward_stage(
            _request(want_hidden=True, positions=[41, 42, 43], seq_ids=[0, 0, 0]),
            SimpleNamespace(is_set=lambda: False),
        )
        call = upstream.hidden_calls[0]
        assert call["positions"] == [41, 42, 43]
        assert call["seq_ids"] == [0, 0, 0]


class TestTailStage:
    def test_returns_token_argmax(self):
        upstream = _FakeUpstream()
        result = _host(upstream)._execute_layer_forward_stage(
            _request(want_hidden=False), SimpleNamespace(is_set=lambda: False),
        )
        assert result == {"token_argmax": 42}
        assert upstream.hidden_calls == []


class TestShapeGuards:
    def test_hidden_length_mismatch_is_rejected(self):
        # hidden_spec 声明 3x4，实际只给 2x4 ⇒ 必须明确失败，不能静默截断
        bad = _request(want_hidden=True)
        bad.root_input["hidden_f32"] = base64.b64encode(
            np.zeros(2 * N_EMBD, dtype=np.float32).tobytes(),
        ).decode("ascii")
        with pytest.raises(Exception, match="长度不符"):
            _host(_FakeUpstream())._execute_layer_forward_stage(
                bad, SimpleNamespace(is_set=lambda: False),
            )

    def test_missing_hidden_is_rejected(self):
        bad = _request(want_hidden=True)
        bad.root_input.pop("hidden_f32")
        with pytest.raises(Exception, match="hidden_f32"):
            _host(_FakeUpstream())._execute_layer_forward_stage(
                bad, SimpleNamespace(is_set=lambda: False),
            )

    def test_cancelled_stage_does_not_run(self):
        upstream = _FakeUpstream()
        with pytest.raises(Exception, match="取消"):
            _host(upstream)._execute_layer_forward_stage(
                _request(want_hidden=True), SimpleNamespace(is_set=lambda: True),
            )
        assert upstream.hidden_calls == []
