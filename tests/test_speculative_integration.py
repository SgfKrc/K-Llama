"""The production request owner must restore every segment before correction."""
from __future__ import annotations
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from qlh_core.contracts.speculation import resolve_speculative_gamma
from tests.helpers.linear_application import make_application
from tests.test_comm_multipos_bench import _gguf


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("QLH_SPECULATIVE_GAMMA", "1")
    path = tmp_path / "head.gguf"
    _gguf(path)
    states = {node: [] for node in ("master", "middle", "tail")}
    calls = []
    prompt = [1, 2, 1, 2]
    descriptor = {"model_type": "qwen2", "total_layers": 3, "hidden_size": 4, "model_path": str(path)}
    def predict(history):
        # Deliberately break a repeat in the middle of a proposed n-gram.
        if len(history) == 7:
            return 3
        return 2 if history[-1] == 1 else 1
    class Tokenizer:
        eos_token_id = 99
        def __call__(self, *_a, **_kw):
            return {"input_ids": np.array([prompt], dtype=np.int64)}
        def decode(self, tokens, **_kw):
            return "".join(str(t) for t in tokens)
    def reset_master():
        calls.append(("reset", "master"))
        states["master"].clear()
    def forward(**kwargs):
        ids = kwargs["input_ids"].reshape(-1).tolist()
        position = kwargs["past_key_values"] or 0
        if position == 0:
            states["master"].clear()
        assert position == len(states["master"])
        rows = []
        for token in ids:
            states["master"].append(token)
            rows.append([token, len(states["master"]) - 1, sum(states["master"]), 1])
        calls.append(("prefix", position, ids))
        return {"hidden_states": np.array(rows, np.float32), "cache": position + len(ids)}
    manager = SimpleNamespace(tokenizer=Tokenizer(), engine_type="llama_cpp", forward_layers=forward,
        clear_pipeline_kv=reset_master, get_pipeline_descriptor=lambda: deepcopy(descriptor))
    def offer(**kw):
        node = kw["node_id"]
        if kw.get("cleanup_only"):
            calls.append(("reset", node, kw["stage_id"]))
            states[node].clear()
            return {"kv_cleared": True}
        positions = kw["positions"]
        if positions[0] == 0:
            states[node].clear()
        assert positions == list(range(len(states[node]), len(states[node]) + len(positions)))
        rows = kw["hidden_states"]
        tokens = []
        for row, position in zip(rows, positions):
            states[node].append(int(row[0]))
            assert int(row[1]) == position and int(row[2]) == sum(states[node])
            tokens.append(predict(states[node]))
        calls.append(("stage", node, tuple(positions), kw["stage_id"], kw.get("want_all_tokens", False)))
        if kw["want_hidden"]:
            return {"hidden_states": rows}
        result = {"token_argmax": tokens[-1]}
        if kw.get("want_all_tokens"):
            result.update(token_argmax_by_position=tokens, positions=positions, n_tokens=len(tokens), n_vocab=100)
        return result
    callbacks = SimpleNamespace(thinking_system_prompt="", build_model_chat_prompt=lambda *_a, **_k: "prompt",
        format_model_response=lambda text, *_a, **_k: (text, None))
    app, stages, state = make_application(tmp_path / "requests.sqlite3", manager, offer, callbacks, boundaries=(0, 1, 2, 3))
    for assignment in [state["deployments"]["deployment"]["local_assignment"], *state["deployments"]["deployment"]["assignments"].values()]:
        assignment["load_spec"]["batch_size"] = 128
    caps = {"multi_position_argmax": True, "stage_kv_cleanup": True}
    app.scheduler._task_worker_control.worker_snapshot = lambda node: {"capabilities": {**caps, "node_incarnation": node + "-boot"}}
    yield SimpleNamespace(app=app, calls=calls, states=states, caps=caps, descriptor=descriptor,
                          state=state, manager=manager, path=path, prompt=prompt)
    app.requests.close()


@pytest.mark.parametrize("gamma", [2, 4, 8])
def test_production_batch_matches_greedy_after_rejection_and_next_request(pipeline, gamma):
    app = pipeline.app
    baseline = app.run("same", max_new_tokens=16, speculative_gamma=1)
    pipeline.calls.clear()
    emitted = []
    result = app.run("same", max_new_tokens=16, speculative_gamma=gamma, _stream_callback=emitted.append)
    assert "error" not in result, result
    assert result["response"] == baseline["response"]
    assert [s["token"] for s in result["metrics"]["steps"]] == [s["token"] for s in baseline["metrics"]["steps"]]
    assert "".join(e["token"] for e in emitted if "token" in e) == result["response"]
    assert result["metrics"]["request"]["execution_state"] == "stopped"
    assert all(not values for values in pipeline.states.values())
    restores = [c for c in pipeline.calls if c[0] == "reset" and len(c) > 2 and ":restore:" in c[2]]
    assert restores and {c[1] for c in restores} == {"middle", "tail"}
    stage_ids = [(c[1], c[3]) for c in pipeline.calls if c[0] == "stage"]
    assert len(stage_ids) == len(set(stage_ids)), "replay must not collide with verification identities"
    assert any(c[0] == "stage" and c[-1] is True and len(c[2]) > 1 for c in pipeline.calls)
    again = app.run("same", max_new_tokens=16, speculative_gamma=1)
    assert again["response"] == baseline["response"]


@pytest.mark.parametrize("change,code", [
    ("caps", "speculative_multi_position_unsupported"),
    ("hybrid", "speculative_model_unsupported"),
    ("cuda", "speculative_load_policy_unsupported"),
    ("sequence", "speculative_load_policy_unsupported"),
    ("batch", "speculative_replay_batch_limit"),
    ("header", "speculative_attention_state_unsupported"),
    ("dimensions", "speculative_model_metadata_invalid"),
    ("context", "speculative_load_policy_unsupported"),
    ("local_head", "speculative_tail_head_required"),
])
def test_speculation_is_admitted_before_any_native_work(pipeline, change, code):
    deployment = pipeline.state["deployments"]["deployment"]
    if change == "caps": pipeline.caps.pop("multi_position_argmax")
    if change == "hybrid": pipeline.descriptor["model_type"] = "qwen3_5"
    if change == "cuda": deployment["assignments"]["tail"]["load_spec"]["device"] = "cuda:0"
    if change == "sequence": deployment["local_assignment"]["load_spec"]["sequence_count"] = 2
    if change == "batch": deployment["assignments"]["middle"]["load_spec"]["batch_size"] = 2
    if change == "header": _gguf(pipeline.path, extra=(("qwen2.full_attention_interval", 4),))
    if change == "dimensions": pipeline.descriptor["hidden_size"] = 8
    if change == "context": deployment["assignments"]["middle"]["load_spec"]["context_size"] = 256
    if change == "local_head":
        deployment["local_assignment"]["load_spec"]["components"].extend(["lm_head", "final_norm"])
        deployment["assignments"]["tail"]["load_spec"]["components"].remove("lm_head")
    result = pipeline.app.run("same", max_new_tokens=8, speculative_gamma=4)
    assert result.get("reason_code") == code, result
    assert not pipeline.calls
    assert pipeline.app.scheduler._host.runtime_controller.execution_lock.acquire(blocking=False)
    pipeline.app.scheduler._host.runtime_controller.execution_lock.release()


def test_bad_restore_receipt_fails_request_then_confirmed_final_cleanup_allows_reuse(pipeline):
    original = pipeline.app.scheduler._execute_layer_stage_offer
    def invalid(**kw):
        if kw.get("cleanup_only") and ":restore:" in kw["stage_id"]:
            return {"kv_cleared": False}
        return original(**kw)
    pipeline.app.scheduler._execute_layer_stage_offer = invalid
    result = pipeline.app.run("same", max_new_tokens=16, speculative_gamma=4)
    assert result.get("reason_code") == "execution_kv_cleanup_receipt_invalid", result
    # Final cleanup can still positively clear every owner, so the next request
    # remains usable. Only a failed physical final cleanup requires quarantine.
    assert all(not values for values in pipeline.states.values())
    assert "error" not in pipeline.app.run("same", max_new_tokens=4, speculative_gamma=1)


@pytest.mark.parametrize("value", [0, 9, -1, True, False, 2.0, "4", []])
def test_explicit_gamma_is_strict(value):
    with pytest.raises(Exception, match="speculative_gamma_invalid"):
        resolve_speculative_gamma(value)


@pytest.mark.parametrize("value", ["0", "9", "2.0", "", "true", None])
def test_environment_gamma_is_strict(value):
    with pytest.raises(Exception, match="speculative_gamma_invalid"):
        resolve_speculative_gamma(environment={"QLH_SPECULATIVE_GAMMA": value})


def test_request_gamma_one_overrides_environment_and_preserves_scalar_path(pipeline, monkeypatch):
    monkeypatch.setenv("QLH_SPECULATIVE_GAMMA", "8")
    pipeline.caps.pop("multi_position_argmax")
    result = pipeline.app.run("same", max_new_tokens=4, speculative_gamma=1)
    assert "error" not in result
    assert not any(c[0] == "stage" and c[-1] is True for c in pipeline.calls)


@pytest.mark.parametrize("mode,code", [("scalar", "missing_token_output"), ("shifted", "token_output_request_mismatch")])
def test_result_must_bind_to_real_verification_batch(pipeline, mode, code):
    original = pipeline.app.scheduler._execute_layer_stage_offer
    def invalid(**kw):
        result = original(**kw)
        if kw.get("want_all_tokens") and not kw.get("cleanup_only"):
            if mode == "scalar": return {"token_argmax": result["token_argmax"]}
            result["positions"] = [p + 1 for p in result["positions"]]
        return result
    pipeline.app.scheduler._execute_layer_stage_offer = invalid
    emitted = []
    result = pipeline.app.run("same", max_new_tokens=16, speculative_gamma=4, _stream_callback=emitted.append)
    assert result.get("reason_code") == code, result
    assert "".join(e["token"] for e in emitted if "token" in e) == "1"
    assert all(not values for values in pipeline.states.values())


def test_unconfirmed_final_cleanup_quarantines_next_request(pipeline):
    original = pipeline.app.scheduler._execute_layer_stage_offer
    def invalid(**kw):
        if kw.get("cleanup_only"):
            return {"kv_cleared": False}
        return original(**kw)
    pipeline.app.scheduler._execute_layer_stage_offer = invalid
    result = pipeline.app.run("same", max_new_tokens=4, speculative_gamma=4)
    assert result.get("reason_code") == "request_runtime_cleanup_pending", result
    pipeline.calls.clear()
    again = pipeline.app.run("same", max_new_tokens=4, speculative_gamma=1)
    assert again.get("reason_code") == "request_runtime_cleanup_pending"
    assert not pipeline.calls


@pytest.mark.parametrize("mode,code", [("cancel", "generation_cancelled"), ("fence", "execution_deployment_fenced")])
def test_restore_rechecks_request_before_publishing_correction(pipeline, mode, code):
    import threading
    from tests.test_refactor_execution import wait_for
    event = threading.Event()
    original = pipeline.app.scheduler._execute_layer_stage_offer
    changed = []
    def interrupt(**kw):
        result = original(**kw)
        if kw.get("cleanup_only") and ":restore:" in kw["stage_id"] and not changed:
            changed.append(True)
            if mode == "cancel": event.set()
            else: pipeline.state["deployments"]["deployment"]["status"] = "retired"
        return result
    pipeline.app.scheduler._execute_layer_stage_offer = interrupt
    emitted = []
    result = pipeline.app.run("same", max_new_tokens=16, speculative_gamma=4,
                              _cancel_event=event, _stream_callback=emitted.append)
    assert result.get("reason_code") == code, result
    wait_for(lambda: pipeline.app.requests.snapshot()["active"] is None)
    assert changed
    assert "".join(e["token"] for e in emitted if "token" in e) == "1"
    assert all(not values for values in pipeline.states.values())
    pipeline.state["deployments"]["deployment"]["status"] = "active"
    assert "error" not in pipeline.app.run("same", max_new_tokens=4, speculative_gamma=1)


def test_execution_owner_rejects_local_only_even_with_direct_call(pipeline):
    result = pipeline.app.run("same", max_new_tokens=4, speculative_gamma=4, routing_preference="local_only")
    assert result.get("reason_code") == "speculative_route_unsupported"
    assert not pipeline.calls
    assert not pipeline.app.requests.snapshot()["requests"]


def test_admission_reads_loaded_crop_when_descriptor_retains_whole_source(pipeline):
    pipeline.descriptor["loaded_artifact"] = str(pipeline.path)
    pipeline.descriptor["model_path"] = str(pipeline.path.parent / "retired-whole-source.gguf")
    result = pipeline.app.run("same", max_new_tokens=8, speculative_gamma=4)
    assert "error" not in result, result
    assert pipeline.calls


@pytest.mark.parametrize("gamma", [1, 4])
@pytest.mark.parametrize("failure", ["exception", "receipt"])
def test_final_cleanup_continues_after_middle_failure(pipeline, gamma, failure):
    original = pipeline.app.scheduler._execute_layer_stage_offer
    touched = []
    def fail_middle(**kw):
        if kw.get("cleanup_only") and kw["stage_id"].endswith(":cleanup"):
            touched.append(kw["node_id"])
            if kw["node_id"] == "middle":
                if failure == "exception": raise RuntimeError("middle_failure")
                return {"kv_cleared": False}
        return original(**kw)
    pipeline.app.scheduler._execute_layer_stage_offer = fail_middle
    result = pipeline.app.run("same", max_new_tokens=4, speculative_gamma=gamma)
    assert result.get("reason_code") == "request_runtime_cleanup_pending", result
    assert touched == ["middle", "tail"]
    assert pipeline.states["tail"] == pipeline.states["master"] == []


def test_cancelled_restore_tracks_physical_reset_before_final_cleanup(pipeline):
    import threading
    from tests.test_refactor_execution import wait_for
    from qlh_core.contracts.execution import ExecutionError
    cancel, release, entered = threading.Event(), threading.Event(), threading.Event()
    original = pipeline.app.scheduler._execute_layer_stage_offer
    physical = SimpleNamespace(execution_drained=lambda _attempt: release.is_set())
    observed = []
    def delayed(**kw):
        if kw.get("cleanup_only") and ":restore:" in kw["stage_id"] and not entered.is_set():
            kw["execution_attempts"].append((physical, "reset_pending"))
            entered.set()
            cancel.set()
            raise ExecutionError("provider_cancelled")
        if kw.get("cleanup_only"):
            observed.append(release.is_set())
        return original(**kw)
    pipeline.app.scheduler._execute_layer_stage_offer = delayed
    emitted = []
    result = pipeline.app.run("same", max_new_tokens=16, speculative_gamma=4,
                              _cancel_event=cancel, _stream_callback=emitted.append)
    assert entered.is_set() and result.get("reason_code") == "generation_cancelled"
    assert not observed, "final cleanup overlapped the unfinished reset"
    release.set()
    wait_for(lambda: pipeline.app.requests.snapshot()["active"] is None)
    assert observed and all(observed)
    assert all(not values for values in pipeline.states.values())
    assert "error" not in pipeline.app.run("same", max_new_tokens=4, speculative_gamma=1)
