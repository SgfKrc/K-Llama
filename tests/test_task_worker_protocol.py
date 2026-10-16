import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task_worker_protocol import (
    MAX_MESSAGE_BYTES,
    MESSAGE_TYPES,
    PROTOCOL_NAME,
    WorkerProtocolError,
    build_message,
    canonical_message_bytes,
    decode_message,
    negotiate_protocol_version,
    restore_inline_hidden,
    split_inline_hidden,
    worker_protocol_status,
)


FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "task_worker_protocol_v1.json",
)


@pytest.fixture(scope="module")
def golden():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def test_v1_golden_messages_round_trip_canonically(golden):
    assert golden["fixture_version"] == 1
    assert golden["protocol"] == PROTOCOL_NAME
    decoded = [decode_message(message) for message in golden["messages"]]

    # v1 golden 覆盖 v1 时代的全部类型；`stage_chunk`（DIST-NEXT-2b）与
    # `stage_result_chunk`（2026-10-11 回程分片）都是 v3 引入的大 payload 分片，
    # 不在 v1 fixture 范围内（其往返由 test_task_worker_chunks 覆盖）。
    assert {message.message_type for message in decoded} == (
        MESSAGE_TYPES - {"stage_chunk", "stage_result_chunk"}
    )
    assert [message.snapshot() for message in decoded] == golden["messages"]
    for message in decoded:
        assert decode_message(canonical_message_bytes(message)) == message


def test_validated_message_payload_cannot_be_mutated_in_place(golden):
    message = decode_message(golden["messages"][2])
    payload = message.payload
    payload["lease_epoch"] = 99

    assert message.payload["lease_epoch"] == 1
    assert message.snapshot() == golden["messages"][2]


def test_negotiation_selects_highest_common_version():
    assert negotiate_protocol_version(1, 1) == 1
    assert negotiate_protocol_version(
        1, 4, local_min_version=1, local_max_version=2,
    ) == 2


def test_negotiation_rejects_non_overlapping_or_invalid_ranges():
    with pytest.raises(WorkerProtocolError) as unsupported:
        negotiate_protocol_version(4, 5)
    assert unsupported.value.code == "unsupported_protocol_version"

    with pytest.raises(WorkerProtocolError) as invalid:
        negotiate_protocol_version(4, 3)
    assert invalid.value.code == "invalid_version_range"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("protocol", "other.protocol", "unsupported_protocol"),
        ("version", 4, "unsupported_protocol_version"),
        ("message_type", "unknown", "unsupported_message_type"),
        ("message_id", "unsafe", "invalid_identifier"),
    ],
)
def test_envelope_identity_is_strict(golden, field, value, code):
    raw = copy.deepcopy(golden["messages"][0])
    raw[field] = value

    with pytest.raises(WorkerProtocolError) as captured:
        decode_message(raw)

    assert captured.value.code == code
    assert captured.value.field == field


def test_unknown_envelope_or_payload_fields_are_rejected(golden):
    envelope = copy.deepcopy(golden["messages"][0])
    envelope["secret"] = "must-not-pass"
    with pytest.raises(WorkerProtocolError) as envelope_error:
        decode_message(envelope)
    assert envelope_error.value.code == "invalid_fields"

    payload = copy.deepcopy(golden["messages"][6])
    payload["payload"]["error_message"] = "raw exception text"
    with pytest.raises(WorkerProtocolError) as payload_error:
        decode_message(payload)
    assert payload_error.value.code == "invalid_fields"


def test_offer_input_digest_detects_tampering(golden):
    offer = copy.deepcopy(golden["messages"][2])
    offer["payload"]["root_input"]["message"] = "tampered"

    with pytest.raises(WorkerProtocolError) as captured:
        decode_message(offer)

    assert captured.value.code == "input_digest_mismatch"


def test_result_output_digest_detects_tampering(golden):
    result = copy.deepcopy(golden["messages"][5])
    result["payload"]["output"]["content"] = "tampered"

    with pytest.raises(WorkerProtocolError) as captured:
        decode_message(result)

    assert captured.value.code == "output_digest_mismatch"


def test_lease_deadline_must_be_after_message_timestamp(golden):
    renewal = copy.deepcopy(golden["messages"][4])
    renewal["payload"]["lease_expires_at_ms"] = renewal["sent_at_ms"]

    with pytest.raises(WorkerProtocolError) as captured:
        decode_message(renewal)

    assert captured.value.code == "invalid_lease_deadline"


def test_worker_capability_lists_and_concurrency_are_bounded(golden):
    duplicate = copy.deepcopy(golden["messages"][0])
    duplicate["payload"]["capabilities"]["engines"].append("pytorch")
    with pytest.raises(WorkerProtocolError) as duplicate_error:
        decode_message(duplicate)
    assert duplicate_error.value.code == "invalid_capabilities"

    unbounded = copy.deepcopy(golden["messages"][0])
    unbounded["payload"]["capabilities"]["max_concurrency"] = 33
    with pytest.raises(WorkerProtocolError) as concurrency_error:
        decode_message(unbounded)
    assert concurrency_error.value.code == "invalid_capabilities"


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("llama.cpp", "llama_cpp"),
        ("llama-cpp", "llama_cpp"),
        ("gguf", "llama_cpp"),
        ("llama", "llama_cpp"),
        ("torch", "pytorch"),
    ],
)
def test_wire_capabilities_normalize_local_engine_aliases(golden, alias, canonical):
    hello = copy.deepcopy(golden["messages"][0])
    capabilities = hello["payload"]["capabilities"]
    capabilities["engines"] = [alias]
    capabilities["models"][0]["engine"] = alias

    decoded = decode_message(hello)

    assert decoded.payload["capabilities"]["engines"] == [canonical]
    assert decoded.payload["capabilities"]["models"][0]["engine"] == canonical


def test_wire_model_identity_normalizes_alias_and_preserves_non_local_engine(golden):
    offer = copy.deepcopy(golden["messages"][2])
    offer["version"] = 2
    offer["payload"]["model_identity"] = {
        "model_id": "model-a",
        "engine": "llama.cpp",
        "format": "gguf",
        "revision": "rev-a",
        "sha256": "a" * 64,
    }
    assert decode_message(offer).payload["model_identity"]["engine"] == "llama_cpp"

    offer["payload"]["model_identity"]["engine"] = "external_api"
    assert decode_message(offer).payload["model_identity"]["engine"] == "external_api"

    hello = copy.deepcopy(golden["messages"][0])
    capabilities = hello["payload"]["capabilities"]
    capabilities["engines"] = ["speculative_assisted"]
    capabilities["models"][0]["engine"] = "speculative_assisted"
    decoded = decode_message(hello)
    assert decoded.payload["capabilities"]["engines"] == ["speculative_assisted"]
    assert decoded.payload["capabilities"]["models"][0]["engine"] == (
        "speculative_assisted"
    )


def test_wire_engine_aliases_fail_closed_and_canonical_duplicates_are_rejected(golden):
    unknown = copy.deepcopy(golden["messages"][0])
    unknown["payload"]["capabilities"]["engines"] = ["llama_magic"]
    unknown["payload"]["capabilities"]["models"][0]["engine"] = "llama_magic"
    with pytest.raises(WorkerProtocolError) as captured:
        decode_message(unknown)
    assert captured.value.code == "invalid_capabilities"

    duplicate = copy.deepcopy(golden["messages"][0])
    duplicate["payload"]["capabilities"]["engines"] = ["llama.cpp", "gguf"]
    duplicate["payload"]["capabilities"]["models"][0]["engine"] = "llama_cpp"
    with pytest.raises(WorkerProtocolError) as captured:
        decode_message(duplicate)
    assert captured.value.code == "invalid_capabilities"


def test_layer_forward_result_metadata_is_accepted():
    """层段结果回记的对账字段必须被接受。

    Android executor 会回记 `stage`/`middle_channel`/`handoff_at`/`tail`；它们此前
    不在允许集合里 ⇒ 回程 result 解码失败、master 只能等到超时（真机表现为
    `remote Stage response timed out`）。
    """
    from task_worker_protocol import _validate_metadata

    _validate_metadata(
        {
            "model": "layer-f6dab6b7",
            "stage": "layer_forward",
            "middle_channel": "extract_hidden",
            "handoff_at": 24,
            "tail": "true",
        },
        version=3,
    )

    with pytest.raises(WorkerProtocolError):
        _validate_metadata({"not_a_known_field": 1}, version=3)


def test_hello_capabilities_may_advertise_layer_budget(golden):
    """★ 2026-10-03：设备自荐层容量（`layer_budget`）是**可选**能力键（向后兼容）。

    它与 `layer_ranges` 分工：后者是"当前已就绪、马上能跑的区间"，前者是
    "本地裁层之后能承载的层数上限"。
    """
    hello = copy.deepcopy(golden["messages"][0])
    hello["payload"]["capabilities"]["layer_budget"] = {
        "available_bytes": 8 * 1024 * 1024 * 1024,
        "per_layer_bytes": 512 * 1024 * 1024,
        "max_layers": 12,
        "local_cut": True,
    }
    decode_message(hello)          # 允许

    missing_key = copy.deepcopy(golden["messages"][0])
    missing_key["payload"]["capabilities"]["layer_budget"] = {
        "available_bytes": 1,
        "per_layer_bytes": 1,
    }
    with pytest.raises(WorkerProtocolError) as missing_error:
        decode_message(missing_key)
    assert missing_error.value.code == "invalid_capabilities"

    bad_layers = copy.deepcopy(golden["messages"][0])
    bad_layers["payload"]["capabilities"]["layer_budget"] = {
        "available_bytes": 1,
        "per_layer_bytes": 1,
        "max_layers": 0,
    }
    with pytest.raises(WorkerProtocolError) as layers_error:
        decode_message(bad_layers)
    assert layers_error.value.code == "invalid_integer"


def test_hello_capabilities_may_advertise_binary_hidden(golden):
    """★ 票 21a（COMM-WIRE-01）：`binary_hidden` 是**可选**能力键（向后兼容）。

    声明它表示该节点能收 `format:"binary"` 的帧（`build_message(binary_body=...)`）；
    未声明者继续走 base64 内联 ⇒ 旧对端永不收到二进制帧。两侧键集必须对称
    （主仓 `_validate_capabilities` 与 android `TaskWorkerProtocol`）。
    """
    # 1) 未声明：向后兼容，允许（旧 hello 逐字节不变）
    decode_message(copy.deepcopy(golden["messages"][0]))

    # 2) 声明 true / false 都允许（值本身不是布尔判据的落点）
    for value in (True, False):
        hello = copy.deepcopy(golden["messages"][0])
        hello["payload"]["capabilities"]["binary_hidden"] = value
        decode_message(hello)

    # 3) 拼错/未知键仍必须被拒（不能因为放行新键而放松精确字段校验）
    typo = copy.deepcopy(golden["messages"][0])
    typo["payload"]["capabilities"]["binary_hiddne"] = True
    with pytest.raises(WorkerProtocolError) as typo_error:
        decode_message(typo)
    assert typo_error.value.code == "invalid_fields"


def test_hello_capabilities_validate_per_artifact_layer_contract(golden):
    hello = copy.deepcopy(golden["messages"][0])
    capabilities = hello["payload"]["capabilities"]
    model = {
        "model_id": "tail20.gguf",
        "engine": "llama_cpp",
        "format": "gguf",
        "revision": "local",
        "sha256": "a" * 64,
    }
    capabilities["engines"] = ["llama_cpp"]
    capabilities["models"] = [model]
    capabilities["layer_ranges"] = [[20, 24]]
    capabilities["layer_artifacts"] = [{
        "layer_range": [20, 24],
        "segment_mode": "tail",
        "model_id": "tail20.gguf",
        "artifact_sha256": "a" * 64,
        "source_model_sha256": "b" * 64,
        "source_model_id": "qwen3-5-2b",
        "hidden_size": 2048,
        "tokenizer_sha256": "d" * 64,
    }]

    decode_message(hello)

    with_bytes = copy.deepcopy(hello)
    with_bytes["payload"]["capabilities"]["layer_artifacts"][0][
        "layer_weight_bytes"
    ] = [1, 2, 3, 4]
    decode_message(with_bytes)

    for invalid in ([1, 2, 3], [1, 0, 3, 4], [1, True, 3, 4]):
        invalid_hello = copy.deepcopy(with_bytes)
        invalid_hello["payload"]["capabilities"]["layer_artifacts"][0][
            "layer_weight_bytes"
        ] = invalid
        with pytest.raises(WorkerProtocolError):
            decode_message(invalid_hello)

    wrong_level = copy.deepcopy(hello)
    wrong_level["payload"]["capabilities"]["layer_weight_bytes"] = [1, 2, 3, 4]
    with pytest.raises(WorkerProtocolError) as wrong_level_error:
        decode_message(wrong_level)
    assert wrong_level_error.value.code == "invalid_fields"

    bad_mode = copy.deepcopy(hello)
    bad_mode["payload"]["capabilities"]["layer_artifacts"][0][
        "segment_mode"
    ] = "unknown"
    with pytest.raises(WorkerProtocolError) as mode_error:
        decode_message(bad_mode)
    assert mode_error.value.code == "invalid_capabilities"

    mismatched_identity = copy.deepcopy(hello)
    mismatched_identity["payload"]["capabilities"]["layer_artifacts"][0][
        "artifact_sha256"
    ] = "c" * 64
    with pytest.raises(WorkerProtocolError) as identity_error:
        decode_message(mismatched_identity)
    assert identity_error.value.code == "invalid_capabilities"

    incomplete = copy.deepcopy(hello)
    incomplete["payload"]["capabilities"]["layer_artifacts"][0].pop(
        "tokenizer_sha256"
    )
    with pytest.raises(WorkerProtocolError) as incomplete_error:
        decode_message(incomplete)
    assert incomplete_error.value.code == "invalid_capabilities"

    pytorch_segment = copy.deepcopy(hello)
    pytorch_segment["payload"]["capabilities"]["engines"] = ["pytorch"]
    pytorch_segment["payload"]["capabilities"]["models"][0]["engine"] = "pytorch"
    pytorch_segment["payload"]["capabilities"]["models"][0]["format"] = "safetensors"
    decode_message(pytorch_segment)
    for engine, format_name in (("pytorch", "gguf"), ("llama_cpp", "safetensors")):
        wrong_runtime = copy.deepcopy(hello)
        wrong_runtime["payload"]["capabilities"]["engines"] = [engine]
        wrong_runtime["payload"]["capabilities"]["models"][0].update(engine=engine, format=format_name)
        with pytest.raises(WorkerProtocolError) as runtime_error:
            decode_message(wrong_runtime)
        assert runtime_error.value.code == "invalid_capabilities"


def test_hello_capabilities_reject_invalid_node_segment_mode(golden):
    hello = copy.deepcopy(golden["messages"][0])
    hello["payload"]["capabilities"]["segment_mode"] = "middle-ish"

    with pytest.raises(WorkerProtocolError) as captured:
        decode_message(hello)

    assert captured.value.code == "invalid_capabilities"


def test_hello_capabilities_may_advertise_middle_channel_and_n_pos_per_embd(golden):
    """★ 2026-09-23：中间段通道与 M-RoPE 位置分量数是**可选**能力键（向后兼容）。"""
    hello = copy.deepcopy(golden["messages"][0])
    hello["payload"]["capabilities"]["middle_channel"] = "keep_head_layer_out"
    hello["payload"]["capabilities"]["n_pos_per_embd"] = 4
    decode_message(hello)          # 允许（值域内）

    bad_channel = copy.deepcopy(golden["messages"][0])
    bad_channel["payload"]["capabilities"]["middle_channel"] = "bogus_channel"
    with pytest.raises(WorkerProtocolError) as channel_error:
        decode_message(bad_channel)
    assert channel_error.value.code == "invalid_capabilities"

    bad_npos = copy.deepcopy(golden["messages"][0])
    bad_npos["payload"]["capabilities"]["n_pos_per_embd"] = 3
    with pytest.raises(WorkerProtocolError) as npos_error:
        decode_message(bad_npos)
    assert npos_error.value.code == "invalid_capabilities"


def test_hello_capabilities_may_advertise_runtime_profile(golden):
    hello = copy.deepcopy(golden["messages"][0])
    hello["payload"]["capabilities"]["runtime_profile"] = "llama_cpp_only"
    decode_message(hello)

    invalid = copy.deepcopy(golden["messages"][0])
    invalid["payload"]["capabilities"]["runtime_profile"] = "torch_edge"
    with pytest.raises(WorkerProtocolError) as captured:
        decode_message(invalid)
    assert captured.value.code == "invalid_capabilities"


def test_hello_ack_fields_cannot_claim_an_invalid_negotiation(golden):
    accepted = copy.deepcopy(golden["messages"][1])
    accepted["payload"]["reason_code"] = "unsupported_protocol_version"
    with pytest.raises(WorkerProtocolError) as accepted_error:
        decode_message(accepted)
    assert accepted_error.value.code == "invalid_negotiation_result"

    rejected = copy.deepcopy(golden["messages"][1])
    rejected["payload"].update({
        "accepted": False,
        "selected_version": 0,
        "reason_code": "unsupported_protocol_version",
    })
    assert decode_message(rejected).payload["accepted"] is False


def test_protocol_status_exposes_n2_transport_without_claiming_stage_readiness():
    status = worker_protocol_status()

    assert status["schema_ready"] is True
    assert status["adapter_connected"] is False
    assert status["transport"] == "existing_tcp_length_prefixed"
    # ★ 2026-09-20：协议升到 v3（层段承载）。本用例验的是「暴露 N2 传输状态、
    #   不谎称 stage 就绪」，与具体版本号无关。
    assert status["preferred_version"] == 3
    assert status["max_version"] == 3
    assert status["admission_state"] == "n2_4_experiment_disabled"


def test_v2_offer_requires_exact_model_identity(golden):
    offer = copy.deepcopy(golden["messages"][2])
    offer["version"] = 2
    offer["payload"]["model_identity"] = copy.deepcopy(
        golden["messages"][0]["payload"]["capabilities"]["models"][0]
    )

    assert decode_message(offer).version == 2

    del offer["payload"]["model_identity"]
    with pytest.raises(WorkerProtocolError) as missing:
        decode_message(offer)
    assert missing.value.code == "invalid_fields"


def test_v2_accepts_android_full_worker_role_without_enabling_transport(golden):
    payload = copy.deepcopy(golden["messages"][0]["payload"])
    payload.update({
        "node_id": "android_worker_01",
        "worker_kind": "android_full_worker",
        "min_version": 2,
        "max_version": 2,
    })
    payload["capabilities"]["engines"] = ["llama_cpp"]
    payload["capabilities"]["models"][0]["engine"] = "llama_cpp"
    hello = build_message(
        "hello",
        payload,
        message_id="msg_android_hello_01",
        sent_at_ms=1700000000000,
        version=2,
    )

    assert hello.payload["worker_kind"] == "android_full_worker"


def test_v2_accept_carries_explicit_retryability(golden):
    accepted = copy.deepcopy(golden["messages"][3])
    accepted["version"] = 2
    accepted["payload"]["retryable"] = False
    assert decode_message(accepted).payload["retryable"] is False

    rejected = copy.deepcopy(accepted)
    rejected["payload"].update({
        "accepted": False,
        "reason_code": "provider_busy",
        "retryable": True,
    })
    assert decode_message(rejected).payload["retryable"] is True

    accepted["payload"]["retryable"] = True
    with pytest.raises(WorkerProtocolError) as invalid:
        decode_message(accepted)
    assert invalid.value.code == "invalid_acceptance_result"


def test_decode_rejects_invalid_utf8_and_oversized_wire_messages():
    with pytest.raises(WorkerProtocolError) as encoding_error:
        decode_message(b"\xff")
    assert encoding_error.value.code == "invalid_encoding"

    with pytest.raises(WorkerProtocolError) as size_error:
        decode_message(b" " * (MAX_MESSAGE_BYTES + 1))
    assert size_error.value.code == "message_too_large"


def test_decode_normalizes_invalid_unicode_to_protocol_error():
    with pytest.raises(WorkerProtocolError) as captured:
        decode_message("\ud800")
    assert captured.value.code == "invalid_encoding"


def test_hello_capabilities_may_advertise_unusable_artifact_reasons():
    """★ 2026-10-07（DIST-NEXT-6）：不可用工件的结构化原因（无路径）。

    fail-closed 不变：这些项**不**进 `layer_ranges`/`layer_artifacts`，只是把
    「为什么没有该区间」以稳定 error code + 身份/区间广告出去。
    """
    identity_fields = {
        "error_code": "artifact_missing",
        "manifest": "mid.manifest.json",
        "architecture": "qwen35",
        "mode": "middle",
        "layer_range": [4, 16],
        "artifact_present": False,
    }

    def capabilities(diagnostics):
        return {
            "stage_types": ["layer_forward"],
            "engines": ["llama_cpp"],
            "models": [{
                "model_id": "qwen35_2b_mid4_16",
                "engine": "llama_cpp",
                "format": "gguf",
                "revision": "local-v1",
                "sha256": "b" * 64,
            }],
            "max_concurrency": 1,
            "layer_artifact_diagnostics": diagnostics,
        }

    def hello(diagnostics):
        return build_message(
            "hello",
            {
                "node_id": "android_worker_01",
                "worker_kind": "android_full_worker",
                "min_version": 1,
                "max_version": 3,
                "capabilities": capabilities(diagnostics),
            },
            message_id="msg_hello_artifactdiag1",
            sent_at_ms=1_700_000_000_000,
            version=3,
        )

    decoded = hello([identity_fields])
    assert decoded.payload["capabilities"]["layer_artifact_diagnostics"] == [
        identity_fields,
    ]

    # `layer_range = null` 合法（「读不到 manifest」还没解析出区间）
    unreadable = {**identity_fields, "error_code": "manifest_unreadable",
                  "layer_range": None}
    assert hello([unreadable]).payload["capabilities"][
        "layer_artifact_diagnostics"
    ] == [unreadable]

    unavailable = {
        **identity_fields,
        "error_code": "fallback_storage_unavailable",
        "manifest": "default-storage",
        "architecture": "",
        "mode": "",
        "layer_range": None,
    }
    assert hello([unavailable]).payload["capabilities"][
        "layer_artifact_diagnostics"
    ] == [unavailable]

    ambiguous = {**identity_fields, "error_code": "artifact_ambiguous"}
    assert hello([ambiguous]).payload["capabilities"][
        "layer_artifact_diagnostics"
    ] == [ambiguous]

    # 空列表没有信息量：要么不给该键，要么给非空列表
    with pytest.raises(WorkerProtocolError) as empty:
        hello([])
    assert empty.value.code == "invalid_capabilities"

    # error code 值域封闭
    with pytest.raises(WorkerProtocolError) as unknown:
        hello([{**identity_fields, "error_code": "whatever"}])
    assert unknown.value.code == "invalid_capabilities"
    assert unknown.value.field == (
        "payload.capabilities.layer_artifact_diagnostics[0].error_code"
    )

    # 键集固定：携带本地路径的额外键会被拒（广告面只带原因，不带路径）
    with pytest.raises(WorkerProtocolError) as extra:
        hello([{**identity_fields, "path": "/sdcard/models/x.gguf"}])
    assert extra.value.code == "invalid_fields"

    # 上限 32 条
    with pytest.raises(WorkerProtocolError) as too_many:
        hello([identity_fields] * 33)
    assert too_many.value.code == "invalid_capabilities"


_CANCEL_IDENTITY = {
    "workflow_id": "wf_cancelremote01",
    "stage_id": "candidate_a",
    "attempt_id": "att_cancelremote01",
    "lease_id": "lease_cancelremote01",
    "lease_epoch": 1,
}


def _cancelled_envelope(**extra):
    return build_message(
        "stage_cancelled",
        {
            **_CANCEL_IDENTITY,
            "provider_id": "remote_worker_01",
            "reason_code": "coordinator_cancelled",
            **extra,
        },
        message_id="msg_cancelled00000001",
        sent_at_ms=1_700_000_000_000,
        version=3,
    ).snapshot()


def test_stage_cancelled_execution_state_is_optional_and_bounded():
    """★ 2026-10-07（DIST-NEXT-1）：取消 ACK 的执行状态可选、值域封闭。

    「请求已取消」与「执行已停止」是两个事实；协议层只接受这两个具名状态，
    旧对端（不带该字段）保持原语义。
    """
    legacy = decode_message(_cancelled_envelope())
    assert legacy.payload["reason_code"] == "coordinator_cancelled"
    assert "execution_state" not in legacy.payload

    for state in ("execution_stopped", "execution_in_flight"):
        decoded = decode_message(_cancelled_envelope(execution_state=state))
        assert decoded.payload["execution_state"] == state

    with pytest.raises(WorkerProtocolError) as captured:
        decode_message(_cancelled_envelope(execution_state="stopped"))
    assert captured.value.code == "unsupported_execution_state"
    assert captured.value.field == "payload.execution_state"


def test_stage_cancel_does_not_accept_execution_state():
    """执行状态只属于**回程** ACK；coordinator 的取消请求不得携带它。"""
    with pytest.raises(WorkerProtocolError) as captured:
        build_message(
            "stage_cancel",
            {**_CANCEL_IDENTITY, "reason_code": "coordinator_cancelled",
             "execution_state": "execution_stopped"},
            message_id="msg_cancelrequest0001",
            sent_at_ms=1_700_000_000_000,
            version=3,
        )
    assert captured.value.code == "invalid_fields"
    assert captured.value.field == "payload"


def test_split_and_restore_inline_hidden_are_bit_identical():
    """★ 票 21a（COMM-WIRE-01）：二进制抽取/放回**逐位无损**，且关能力位零行为变化。"""
    import base64 as _base64

    hidden = bytes(range(32))
    encoded = _base64.b64encode(hidden).decode("ascii")
    offer = {
        "message_type": "stage_offer",
        "payload": {
            "root_input": {"hidden_f32": encoded, "other": 1},
            "input_sha256": "x",
        },
    }

    # 1) 关能力位 ⇒ 逐字段相同、不抽体（旧路径字节不变）
    same, raw = split_inline_hidden(offer, enabled=False)
    assert raw is None
    assert same == offer

    # 2) 开能力位 ⇒ 抽出原始字节，并从 JSON 里删掉该字段（避免同帧两份）
    split, raw = split_inline_hidden(offer, enabled=True)
    assert raw == hidden
    assert "hidden_f32" not in split["payload"]["root_input"]
    assert split["payload"]["root_input"]["other"] == 1
    assert split["payload"]["input_sha256"] == "x"

    # 3) 放回 ⇒ 与原消息逐字段相同（下游不必知道二进制路径存在）
    assert restore_inline_hidden(split, raw) == offer
    # 4) 未带体 ⇒ 逐字段相同
    assert restore_inline_hidden(split, None) == split

    # 5) 回程位置（stage_result.output.hidden_out_f32）同理
    result = {
        "message_type": "stage_result",
        "payload": {"output": {"hidden_out_f32": encoded}, "attempt_id": "a"},
    }
    r_split, r_raw = split_inline_hidden(result, enabled=True)
    assert r_raw == hidden
    assert "hidden_out_f32" not in r_split["payload"]["output"]
    assert restore_inline_hidden(r_split, r_raw) == result


def test_split_inline_hidden_leaves_ref_bad_base64_and_other_types_untouched():
    """已分片（`hidden_ref`）、非法 base64、不承载内联 hidden 的类型都原样返回。"""
    # hidden_ref —— 分片路径尚未接二进制，原样保留
    chunked = {
        "message_type": "stage_offer",
        "payload": {
            "root_input": {"hidden_ref": {"total_bytes": 4, "payload_sha256": "x"}},
        },
    }
    same, raw = split_inline_hidden(chunked, enabled=True)
    assert raw is None
    assert same == chunked

    # 非法 base64 —— 交给既有协议校验报错，helper 不改写
    bad = {
        "message_type": "stage_offer",
        "payload": {"root_input": {"hidden_f32": "not base64!!"}},
    }
    same, raw = split_inline_hidden(bad, enabled=True)
    assert raw is None
    assert same == bad

    # 不承载内联 hidden 的消息类型
    hello = {
        "message_type": "hello",
        "payload": {"root_input": {"hidden_f32": "AAAA"}},
    }
    same, raw = split_inline_hidden(hello, enabled=True)
    assert raw is None
    assert same == hello
