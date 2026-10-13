"""PC command fencing, identity admission, cleanup replay, and shutdown tests."""
from copy import deepcopy
import threading
from types import SimpleNamespace

import pytest

from inference_service.engine_host import EngineHost
import worker_prepare as wp
from worker_prepare_contract import WorkerPrepareError, stage_identity, validate_ack

REAL_SELECTION_RESOLVER = wp.resolve_prepare_selection


def identity(generation=1, *, fmt="safetensors", node_id="pc", operation=None):
    digest = f"{generation:064x}"
    return {
        "operation_id": operation or f"op-{generation}", "config_id": f"cfg-{generation}",
        "generation": generation, "assignment_id": f"assignment-{generation}", "node_id": node_id,
        "model_id": "tiny", "model_sha256": "b" * 64,
        "stage_model_identity": {"model_id": "supply-" + digest[:24] if fmt == "safetensors" else "segment.gguf",
                                 "engine": "pytorch" if fmt == "safetensors" else "llama_cpp",
                                 "format": fmt, "revision": "1", "sha256": digest},
        "artifact_sha256": digest, "manifest_sha256": "a" * 64,
        "source_model_id": "tiny", "source_model_sha256": "b" * 64,
        "tokenizer_sha256": "c" * 64, "layer_range": [1, 3], "total_layers": 4,
        "hidden_size": 16, "segment_mode": "middle",
    }


def command(action, value, **options):
    data = {"action": action, "identity": deepcopy(value)}
    if action == "prepare":
        data.update(context_size=64, middle_channel="keep_head_layer_out")
    if action in {"abort", "finalize"}:
        data["cleanup_id"] = "cleanup-" + action
    data.update(options)
    return data


class Manager:
    def __init__(self):
        self.model = object()
        self.layer_range = (1, 3)
        self.unloads = 0
        self.fail_close = False

    @property
    def is_loaded(self):
        return self.model is not None

    def unload_model(self):
        if self.fail_close:
            raise RuntimeError("native close failed")
        self.unloads += 1
        self.model = None


class Client:
    def __init__(self):
        self.is_registered = True
        self.registration_epoch = 3
        self._connection_generation = 4
        self.acks = []

    def send_data(self, ack, msg_type):
        assert msg_type.value == "model_prepare"
        self.acks.append(validate_ack(ack))


@pytest.fixture
def setup(monkeypatch):
    host = EngineHost.__new__(EngineHost)
    host._layer_upstream = None
    host._supplied_layer_manager = None
    scheduler = SimpleNamespace(_host=SimpleNamespace(full_chat_execution_lock=threading.RLock()),
                                _tcp_client=Client(), _control_fence=None, enabled=True,
                                _effective_role=lambda: "client", get_effective_node_id=lambda: "pc")
    scheduler.get_distributed_inference_enabled = lambda: scheduler.enabled
    selected = lambda value: {"directory": "segment", "model_path": "segment", "manifest_sha256": value["manifest_sha256"],
                              "manifest": {**deepcopy(value), "format": "safetensors"}}
    monkeypatch.setattr(wp, "resolve_prepare_selection", lambda sched, value: selected(value))
    built = []

    def build(data, choice, cancel):
        runtime = wp.LayerRuntime(deepcopy(data["identity"]), choice, manager=Manager())
        built.append(runtime)
        return runtime

    host._build_worker_runtime = build
    controller = wp.WorkerPrepareController(scheduler, lambda: host)
    scheduler._worker_prepare = controller
    yield SimpleNamespace(host=host, scheduler=scheduler, controller=controller, built=built, selected=selected)
    controller.close()


def drain(controller):
    runner = controller._runner
    if runner is not None:
        runner.join(timeout=5)
        assert not runner.is_alive(), "command thread did not finish"


def run(env, action, value, **options):
    assert env.controller.handle(command(action, value, **options))
    drain(env.controller)
    return env.scheduler._tcp_client.acks[-1]


def offer(value):
    return {"prepare_identity": stage_identity(value), "layer_range": value["layer_range"],
            "model_identity": value["stage_model_identity"]}


def test_prepare_materializes_before_ack_commit_only_promotes_and_duplicate_does_not_reload(setup):
    value = identity()
    result = run(setup, "prepare", value)
    assert result["status"] == "prepared" and result["loaded"]
    assert setup.host._supplied_layer_manager is None
    assert len(setup.built) == 1
    with pytest.raises(WorkerPrepareError):
        setup.controller.check_stage(offer(value))
    assert run(setup, "commit", value)["status"] == "ready"
    setup.controller.check_stage(offer(value))
    assert setup.host._supplied_layer_manager is setup.built[0].manager
    run(setup, "prepare", value)
    run(setup, "commit", value)
    assert len(setup.built) == 1
    assert run(setup, "prepare", value, context_size=128)["error"] == "worker_prepare_contract_conflict"


def test_missing_or_other_stage_identity_is_always_rejected(setup):
    with pytest.raises(WorkerPrepareError):
        setup.controller.check_stage({})
    value = identity()
    run(setup, "prepare", value)
    run(setup, "commit", value)
    for data in ({}, offer(identity(2)), {**offer(value), "layer_range": [0, 2]}):
        with pytest.raises(WorkerPrepareError):
            setup.controller.check_stage(data)


def test_abort_commit_restores_exact_previous_manager_and_kv(setup):
    first, second = identity(), identity(2)
    run(setup, "prepare", first)
    run(setup, "commit", first)
    run(setup, "finalize", first)
    manager = setup.host._supplied_layer_manager
    kv = {0: (object(), 7)}
    setup.host._supplied_layer_caches = kv
    setup.host._supplied_layer_workflow = "original-workflow"
    run(setup, "prepare", second)
    assert setup.host._supplied_layer_manager is manager and manager.is_loaded
    run(setup, "commit", second)
    candidate = setup.host._supplied_layer_manager
    assert run(setup, "abort", second)["status"] == "released"
    assert not candidate.is_loaded
    assert setup.host._supplied_layer_manager is manager
    assert setup.host._supplied_layer_caches is kv
    assert setup.host._supplied_layer_workflow == "original-workflow"
    setup.controller.check_stage(offer(first))


def test_finalize_closes_previous_release_closes_current_before_receipt(setup):
    first, second = identity(), identity(2)
    for value in (first, second):
        run(setup, "prepare", value)
        run(setup, "commit", value)
        if value == first:
            run(setup, "finalize", value)
    old = setup.host._worker_runtime_previous.manager
    current = setup.host._supplied_layer_manager
    assert run(setup, "finalize", second)["loaded"]
    assert old.model is None and current.model is not None
    result = run(setup, "release", second, cleanup_id="retire-plan")
    assert result["status"] == "released" and not result["loaded"]
    assert current.model is None and not setup.host.worker_runtime_present()


def test_failed_close_never_acknowledges_released_and_can_retry(setup):
    value = identity()
    run(setup, "prepare", value)
    run(setup, "commit", value)
    manager = setup.host._supplied_layer_manager
    manager.fail_close = True
    result = run(setup, "release", value, cleanup_id="release-one")
    assert result["status"] == "error" and result["loaded"]
    manager.fail_close = False
    assert not run(setup, "release", value, cleanup_id="release-one")["loaded"]


@pytest.mark.parametrize("change", ["generation", "client", "mode", "authority"])
def test_native_completion_rechecks_scope_and_never_ready_after_drift(setup, change):
    entered, complete = threading.Event(), threading.Event()
    original = setup.host._build_worker_runtime

    def block(*args):
        result = original(*args)
        entered.set()
        assert complete.wait(3)
        return result

    setup.host._build_worker_runtime = block
    value = identity()
    client = setup.scheduler._tcp_client
    assert setup.controller.handle(command("prepare", value))
    assert entered.wait(3)
    assert client.acks == []  # TCP dispatch returned while native loading is blocked.
    if change == "generation":
        client._connection_generation += 1
    elif change == "client":
        setup.scheduler._tcp_client = Client()
    elif change == "mode":
        setup.scheduler.enabled = False
    else:
        setup.scheduler._control_fence = object()
    complete.set()
    drain(setup.controller)
    assert not any(ack["status"] == "prepared" for ack in client.acks)
    assert setup.built[0].manager is None or not setup.built[0].manager.is_loaded
    assert not setup.host.worker_runtime_present()


@pytest.mark.parametrize("action", ["release", "abort", "finalize"])
def test_disconnect_retirement_allows_exact_cleanup_replay_without_touching_new_runtime(setup, action):
    old = identity()
    run(setup, "prepare", old)
    run(setup, "commit", old)
    setup.controller.invalidate("disconnect")
    drain(setup.controller)
    setup.scheduler._tcp_client = Client()
    result = run(setup, action, old, cleanup_id="durable-debt")
    assert result["status"] != "error" and not result["loaded"]
    new = identity(2)
    run(setup, "prepare", new)
    run(setup, "commit", new)
    manager = setup.host._supplied_layer_manager
    result = run(setup, action, old, cleanup_id="durable-debt")
    assert result["status"] != "error" and not result["loaded"]
    assert setup.host._supplied_layer_manager is manager and manager.is_loaded
    setup.controller.check_stage(offer(new))


@pytest.mark.parametrize("action", ["release", "abort", "finalize"])
def test_restart_empty_owner_acknowledges_cleanup_as_unloaded(setup, action):
    result = run(setup, action, identity(8), cleanup_id="previous-process")
    assert result["status"] != "error" and not result["loaded"]
    assert setup.built == []


@pytest.mark.parametrize("action", ["release", "abort", "finalize"])
def test_unknown_stale_cleanup_cannot_unload_different_runtime(setup, action):
    value = identity(2)
    run(setup, "prepare", value)
    run(setup, "commit", value)
    manager = setup.host._supplied_layer_manager
    assert run(setup, action, identity(), cleanup_id="unknown-old-debt")["status"] == "error"
    assert setup.host._supplied_layer_manager is manager and manager.is_loaded


def test_failed_selection_can_be_aborted_without_permanent_cleanup_debt(setup, monkeypatch):
    monkeypatch.setattr(wp, "resolve_prepare_selection", lambda *args: (_ for _ in ()).throw(WorkerPrepareError("bad-proof")))
    value = identity()
    assert run(setup, "prepare", value)["status"] == "error"
    assert run(setup, "abort", value)["status"] == "released"
    assert not setup.host.worker_runtime_present()


def test_cancel_native_prepare_then_abort_unloads_and_acknowledges(setup):
    entered, complete = threading.Event(), threading.Event()
    original = setup.host._build_worker_runtime

    def block(*args):
        result = original(*args)
        entered.set()
        assert complete.wait(3)
        return result

    setup.host._build_worker_runtime = block
    value = identity()
    assert setup.controller.handle(command("prepare", value))
    assert entered.wait(3)
    assert setup.controller.handle(command("abort", value))
    complete.set()
    drain(setup.controller)
    assert setup.scheduler._tcp_client.acks[-1]["status"] == "released"
    assert not setup.host.worker_runtime_present()


def test_strict_shape_and_other_node_do_not_enqueue_native_work(setup):
    value = identity()
    assert not setup.controller.handle({**command("prepare", value), "release": True})
    assert not setup.controller.handle(command("prepare", identity(node_id="other")))
    assert setup.built == []


def test_supplied_selection_checks_exact_operation_node_digest_range_and_physical_identity(setup, monkeypatch):
    import model_supply_worker
    value = identity()
    selected = setup.selected(value)
    selected["manifest"].update(format="safetensors")
    monkeypatch.setattr(model_supply_worker, "get_selected_model", lambda *args: deepcopy(selected))
    setup.scheduler._configured_layer_artifact = lambda: value["stage_model_identity"]
    setup.scheduler._task_worker_layer_artifact_identity = lambda artifact: artifact
    resolver = REAL_SELECTION_RESOLVER
    assert resolver(setup.scheduler, value)["manifest"]["operation_id"] == value["operation_id"]
    for key, replacement in (("operation_id", "other"), ("node_id", "other"), ("artifact_sha256", "f"*64), ("layer_range", [0, 2])):
        original = selected["manifest"][key]
        selected["manifest"][key] = replacement
        with pytest.raises(WorkerPrepareError, match="manifest_mismatch"):
            resolver(setup.scheduler, value)
        selected["manifest"][key] = original


def native_offer(value):
    return SimpleNamespace(stage_fields={"prepare_identity": stage_identity(value), "layer_range": value["layer_range"]},
                           model_identity=value["stage_model_identity"])


def assert_fenced(env, value):
    with pytest.raises(WorkerPrepareError):
        env.controller.check_stage(offer(value))
    with pytest.raises(WorkerPrepareError):
        env.host._check_worker_runtime_stage(native_offer(value), required=True)


def test_new_prepare_fences_old_at_admission_and_abort_restores_both_gates(setup):
    old, new = identity(), identity(2)
    for action in ("prepare", "commit", "finalize"):
        run(setup, action, old)
    setup.controller.check_stage(offer(old))
    with setup.scheduler._host.full_chat_execution_lock:
        assert setup.controller.handle(command("prepare", new))
        # Native work cannot start until this lock is released, but both gates
        # must already reject the old stage without waiting on native loading.
        assert_fenced(setup, old)
    drain(setup.controller)
    assert setup.scheduler._tcp_client.acks[-1]["status"] == "prepared"
    assert_fenced(setup, old)
    assert_fenced(setup, new)
    run(setup, "abort", new)
    setup.controller.check_stage(offer(old))
    setup.host._check_worker_runtime_stage(native_offer(old), required=True)


def test_failed_prepare_keeps_old_fenced_until_explicit_abort(setup, monkeypatch):
    old, new = identity(), identity(2)
    for action in ("prepare", "commit", "finalize"):
        run(setup, action, old)
    monkeypatch.setattr(wp, "resolve_prepare_selection", lambda *args: (_ for _ in ()).throw(WorkerPrepareError("bad-proof")))
    assert run(setup, "prepare", new)["status"] == "error"
    assert_fenced(setup, old)
    assert run(setup, "abort", new)["status"] == "released"
    setup.controller.check_stage(offer(old))
    setup.host._check_worker_runtime_stage(native_offer(old), required=True)


@pytest.mark.parametrize("action", ["release", "abort", "finalize"])
def test_matching_previous_cleanup_never_closes_new_committed_runtime(setup, action):
    old, new = identity(), identity(2)
    for value in (old, new):
        run(setup, "prepare", value)
        run(setup, "commit", value)
        if value == old:
            run(setup, "finalize", value)
    previous, current = setup.host._worker_runtime_previous.manager, setup.host._supplied_layer_manager
    result = run(setup, action, old, cleanup_id="old-debt")
    assert result["status"] != "error" and not result["loaded"]
    assert not previous.is_loaded and current.is_loaded
    setup.controller.check_stage(offer(new))
    assert not run(setup, action, old, cleanup_id="old-debt")["loaded"]
    assert current.is_loaded


def test_old_release_closes_only_old_active_and_preserves_prepared_new_candidate(setup):
    old, new = identity(), identity(2)
    for action in ("prepare", "commit", "finalize"):
        run(setup, action, old)
    old_manager = setup.host._supplied_layer_manager
    run(setup, "prepare", new)
    candidate = setup.host._worker_runtime_candidate.manager
    assert not run(setup, "release", old, cleanup_id="retire-old")["loaded"]
    assert not old_manager.is_loaded and candidate.is_loaded
    assert_fenced(setup, new)
    assert run(setup, "commit", new)["status"] == "ready"
    setup.controller.check_stage(offer(new))


def test_finalize_replay_after_release_is_terminal_unloaded_not_ready(setup):
    old = identity()
    for action in ("prepare", "commit", "finalize"):
        run(setup, action, old)
    run(setup, "release", old, cleanup_id="retire")
    ack = run(setup, "finalize", old, cleanup_id="lost-finalize-ack")
    assert ack["status"] == "finalized" and ack["loaded"] is False
    assert_fenced(setup, old)
    assert run(setup, "prepare", old)["error"] == "worker_prepare_retired_identity"


@pytest.mark.parametrize("action", ["release", "abort", "finalize"])
def test_failed_disconnect_close_replayed_cleanup_retries_native_before_ack(setup, action):
    value = identity()
    run(setup, "prepare", value)
    run(setup, "commit", value)
    manager = setup.host._supplied_layer_manager
    manager.fail_close = True
    setup.controller.invalidate("disconnect")
    drain(setup.controller)
    setup.scheduler._tcp_client = Client()
    assert setup.host.worker_runtime_present()
    ack = run(setup, action, value, cleanup_id="retry-close")
    assert ack["status"] == "error"  # No empty-owner/retired proof on failure.
    manager.fail_close = False
    ack = run(setup, action, value, cleanup_id="retry-close")
    assert ack["status"] != "error" and not ack["loaded"]
    assert not setup.host.worker_runtime_present()
    run(setup, "prepare", identity(2))
    assert run(setup, "commit", identity(2))["status"] == "ready"


def test_dead_native_reference_is_not_a_proof_of_successful_retirement(setup):
    value = identity()
    run(setup, "prepare", value)
    run(setup, "commit", value)
    manager = setup.host._supplied_layer_manager
    manager.model = None
    manager.fail_close = True
    assert not setup.host.worker_runtime_loaded(value)
    assert setup.host.worker_runtime_present()
    ack = run(setup, "release", value, cleanup_id="dead-resource")
    assert ack["status"] == "error" and ack["loaded"] is False
    manager.fail_close = False
    assert run(setup, "release", value)["status"] == "released"
    assert not setup.host.worker_runtime_present()


def test_repeated_disconnect_during_close_preserves_old_retirement_proof(setup):
    old = identity()
    run(setup, "prepare", old)
    run(setup, "commit", old)
    entered, complete = threading.Event(), threading.Event()
    manager = setup.host._supplied_layer_manager
    original = manager.unload_model
    def block():
        entered.set()
        assert complete.wait(3)
        original()
    manager.unload_model = block
    setup.controller.invalidate("disconnect-one")
    assert entered.wait(3)
    setup.controller.invalidate("disconnect-two")
    complete.set()
    drain(setup.controller)
    setup.scheduler._tcp_client = Client()
    new = identity(2)
    run(setup, "prepare", new)
    run(setup, "commit", new)
    ack = run(setup, "release", old, cleanup_id="durable-old-debt")
    assert ack["status"] == "released" and not ack["loaded"]
    setup.controller.check_stage(offer(new))


def test_disconnect_during_prepare_cannot_republish_candidate_and_allows_cleanup_replay(setup):
    entered, complete = threading.Event(), threading.Event()
    original = setup.host._build_worker_runtime
    def block(*args):
        runtime = original(*args)
        entered.set()
        assert complete.wait(3)
        return runtime
    setup.host._build_worker_runtime = block
    value = identity()
    client = setup.scheduler._tcp_client
    assert setup.controller.handle(command("prepare", value))
    assert entered.wait(3)
    setup.controller.invalidate("disconnect")
    setup.scheduler._tcp_client = Client()
    complete.set()
    drain(setup.controller)
    assert not any(ack["status"] == "prepared" for ack in client.acks)
    assert setup.controller._candidate is None
    assert not setup.host.worker_runtime_present()
    assert not run(setup, "abort", value, cleanup_id="post-reconnect")["loaded"]


def test_new_prepare_arriving_during_commit_cannot_publish_old_ready(setup):
    old, new = identity(), identity(2)
    run(setup, "prepare", old)
    entered, complete = threading.Event(), threading.Event()
    original = setup.host.commit_worker_runtime
    def block(value):
        original(value)
        entered.set()
        assert complete.wait(3)
    setup.host.commit_worker_runtime = block
    assert setup.controller.handle(command("commit", old))
    assert entered.wait(3)
    assert setup.controller.handle(command("prepare", new))
    assert_fenced(setup, old)
    complete.set()
    drain(setup.controller)
    acks = setup.scheduler._tcp_client.acks
    assert not any(ack["identity"] == old and ack["status"] == "ready" for ack in acks)
    assert acks[-1]["identity"] == new and acks[-1]["status"] == "prepared"
    setup.host.commit_worker_runtime = original
    assert run(setup, "commit", new)["status"] == "ready"


def test_scheduler_stop_start_controller_preserves_proofs_and_can_prepare_new_identity(setup):
    old = identity()
    run(setup, "prepare", old)
    run(setup, "commit", old)
    setup.controller.close()
    assert not setup.controller.handle(command("prepare", identity(2)))
    setup.controller.reopen()
    assert not run(setup, "finalize", old, cleanup_id="restart-debt")["loaded"]
    run(setup, "prepare", identity(2))
    assert run(setup, "commit", identity(2))["status"] == "ready"


def test_superseded_commit_preserves_old_active_for_later_candidate_abort(setup):
    old, candidate, newer = identity(), identity(2), identity(3)
    for action in ("prepare", "commit", "finalize"):
        run(setup, action, old)
    previous = setup.host._supplied_layer_manager
    run(setup, "prepare", candidate)
    entered, complete = threading.Event(), threading.Event()
    original = setup.host.commit_worker_runtime
    def block(value):
        original(value)
        entered.set()
        assert complete.wait(3)
    setup.host.commit_worker_runtime = block
    assert setup.controller.handle(command("commit", candidate))
    assert entered.wait(3)
    assert setup.controller.handle(command("prepare", newer))
    complete.set()
    drain(setup.controller)
    assert previous.is_loaded
    assert setup.host._supplied_layer_manager is previous
    assert_fenced(setup, old)
    run(setup, "abort", newer)
    setup.controller.check_stage(offer(old))
    setup.host._check_worker_runtime_stage(native_offer(old), required=True)
    assert setup.host._supplied_layer_manager is previous


def test_release_committed_candidate_retires_previous_even_before_finalize(setup):
    old, new = identity(), identity(2)
    for value in (old, new):
        run(setup, "prepare", value)
        run(setup, "commit", value)
        if value == old:
            run(setup, "finalize", value)
    previous = setup.host._worker_runtime_previous.manager
    current = setup.host._supplied_layer_manager
    ack = run(setup, "release", new, cleanup_id="retire-unfinalized-plan")
    assert ack["status"] == "released" and not ack["loaded"]
    assert previous.model is None and current.model is None
    assert not setup.host.worker_runtime_present()
    for value in (old, new):
        assert not run(setup, "finalize", value, cleanup_id="old-finalize-debt")["loaded"]
    run(setup, "prepare", identity(3))
    assert run(setup, "commit", identity(3))["status"] == "ready"


def retire_thread(controller, *, timeout=2):
    outcome = {}
    def retire():
        try:
            outcome["result"] = controller.retire_and_join("local_model_load", timeout=timeout)
        except Exception as exc:
            outcome["error"] = exc
    thread = threading.Thread(target=retire)
    thread.start()
    return thread, outcome


def test_retire_and_join_waits_for_native_close_and_blocks_new_prepare(setup):
    value = identity()
    run(setup, "prepare", value)
    run(setup, "commit", value)
    manager = setup.host._supplied_layer_manager
    entered, complete = threading.Event(), threading.Event()
    original = manager.unload_model
    def block():
        entered.set()
        assert complete.wait(3)
        original()
    manager.unload_model = block
    thread, outcome = retire_thread(setup.controller)
    try:
        assert entered.wait(3)
        assert outcome == {} and thread.is_alive()
        assert manager.is_loaded
        assert_fenced(setup, value)
        assert not setup.controller.handle(command("prepare", identity(2)))
    finally:
        complete.set()
        thread.join(timeout=3)
    assert outcome == {"result": True}
    assert manager.model is None and not setup.host.worker_runtime_present()
    assert not run(setup, "finalize", value, cleanup_id="mode-cleanup-debt")["loaded"]


def test_retire_and_join_failed_close_has_named_error_and_successful_retry(setup):
    value = identity()
    run(setup, "prepare", value)
    run(setup, "commit", value)
    manager = setup.host._supplied_layer_manager
    manager.fail_close = True
    with pytest.raises(WorkerPrepareError, match="worker_prepare_retire_failed") as failed:
        setup.controller.retire_and_join("local_model_load", timeout=2)
    assert isinstance(failed.value.__cause__, RuntimeError)
    assert setup.host.worker_runtime_present()
    assert_fenced(setup, value)
    manager.fail_close = False
    assert setup.controller.retire_and_join("local_model_load", timeout=2) is True
    assert not setup.host.worker_runtime_present()


def test_retire_and_join_timeout_is_bounded_and_never_proves_blocked_native_load_unloaded(setup):
    import time
    entered, complete = threading.Event(), threading.Event()
    original = setup.host._build_worker_runtime
    def block(*args):
        runtime = original(*args)
        entered.set()
        assert complete.wait(3)
        return runtime
    setup.host._build_worker_runtime = block
    value = identity()
    assert setup.controller.handle(command("prepare", value))
    assert entered.wait(3)
    started = time.monotonic()
    try:
        with pytest.raises(WorkerPrepareError, match="worker_prepare_retire_timeout"):
            setup.controller.retire_and_join("local_model_load", timeout=0.05)
        assert time.monotonic() - started < 1
        assert setup.built[0].manager.is_loaded
        assert not setup.controller.handle(command("prepare", identity(2)))
        assert_fenced(setup, value)
    finally:
        complete.set()
        drain(setup.controller)
    assert not setup.host.worker_runtime_present()
    assert not any(ack["status"] == "prepared" for ack in setup.scheduler._tcp_client.acks)
    assert setup.controller.retire_and_join("local_model_load", timeout=2) is True


def test_retire_and_join_epoch_replacement_cannot_acknowledge_another_cleanup(setup):
    value = identity()
    run(setup, "prepare", value)
    run(setup, "commit", value)
    entered, complete = threading.Event(), threading.Event()
    manager = setup.host._supplied_layer_manager
    original = manager.unload_model
    def block():
        entered.set()
        assert complete.wait(3)
        original()
    manager.unload_model = block
    thread, outcome = retire_thread(setup.controller)
    try:
        assert entered.wait(3)
        setup.controller.invalidate("another_disconnect")
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert outcome["error"].reason_code == "worker_prepare_retire_epoch_changed"
        assert manager.is_loaded
        assert not setup.controller.handle(command("prepare", identity(2)))
    finally:
        complete.set()
        drain(setup.controller)
        thread.join(timeout=3)
    assert setup.controller.retire_and_join("local_model_load", timeout=2) is True


def test_retire_and_join_rejects_caller_holding_full_chat_lock_without_invalidating(setup):
    value = identity()
    run(setup, "prepare", value)
    run(setup, "commit", value)
    epoch = setup.controller._epoch
    with setup.scheduler._host.full_chat_execution_lock:
        with pytest.raises(WorkerPrepareError, match="worker_prepare_retire_lock_held"):
            setup.controller.retire_and_join("local_model_load", timeout=0.1)
    assert setup.controller._epoch == epoch
    setup.controller.check_stage(offer(value))


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_retire_and_join_rejects_invalid_deadline_without_mutation(setup, timeout):
    epoch = setup.controller._epoch
    with pytest.raises(WorkerPrepareError, match="invalid_worker_prepare_retire_timeout"):
        setup.controller.retire_and_join("local_model_load", timeout=timeout)
    assert setup.controller._epoch == epoch


def test_retire_and_join_checks_resource_ownership_instead_of_trusting_release_return(setup, monkeypatch):
    value = identity()
    run(setup, "prepare", value)
    run(setup, "commit", value)
    monkeypatch.setattr(setup.host, "release_worker_runtime", lambda: None)
    with pytest.raises(WorkerPrepareError, match="worker_prepare_retire_failed") as failed:
        setup.controller.retire_and_join("local_model_load", timeout=2)
    assert failed.value.__cause__.reason_code == "worker_prepare_close_failed"
    assert setup.host.worker_runtime_present()


def test_retire_and_join_binds_factory_to_retire_preexisting_unmanaged_runtime(setup):
    manager = Manager()
    setup.host._supplied_layer_manager = manager
    assert setup.controller._runtime is None
    assert setup.controller.retire_and_join("local_model_load", timeout=2) is True
    assert setup.controller._runtime is setup.host
    assert manager.model is None and not setup.host.worker_runtime_present()


def test_retire_and_join_admission_failure_releases_waiter_barrier(setup, monkeypatch):
    original_scope = setup.controller._scope
    monkeypatch.setattr(setup.controller, "_scope", lambda *args: (_ for _ in ()).throw(RuntimeError("scope unavailable")))
    with pytest.raises(WorkerPrepareError, match="worker_prepare_retire_failed"):
        setup.controller.retire_and_join("local_model_load", timeout=2)
    assert setup.controller._retire_waiters == 0
    monkeypatch.setattr(setup.controller, "_scope", original_scope)
    assert setup.controller.retire_and_join("local_model_load", timeout=2) is True
