"""Pipeline retirement preserves terminal cleanup across unload and restart."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest


class _Manager:
    """Model I/O boundary only; ModelHost owns the candidate state machine."""

    engine_type = "pytorch"

    def __init__(self, *, loaded=False):
        self.is_loaded = loaded
        self.is_pipeline_prepared = False
        self.quant_type = "fp16"
        self.layer_range = None
        self.closed = 0
        self.unload_attempts = 0
        self.fail_unload = False

    def prepare_pipeline_model(self, **kwargs):
        self.is_pipeline_prepared = True
        self.quant_type = kwargs["quant_type"]
        return {
            "model_id": kwargs["model_id"],
            "model_type": "qwen2",
            "model_sha256": kwargs["model_sha256"],
            "total_layers": 8,
            "pipeline_runtime_supported": True,
        }

    def load_layer_range(self, **kwargs):
        if "load_spec" in kwargs:
            import torch
            self.quant_type = kwargs["quant_type"]
            self.model = torch.nn.Linear(2, 2, bias=False).to(dtype=getattr(torch, kwargs["load_spec"]["dtype"]))
        self.layer_range = (kwargs["start_layer"], kwargs["end_layer"])
        self.is_loaded = True
        return {"success": True}

    def unload_model(self):
        self.unload_attempts += 1
        if self.fail_unload:
            raise RuntimeError("previous runtime cleanup unavailable")
        self.closed += 1
        self.is_loaded = False
        self.is_pipeline_prepared = False


def _capacity_plan(descriptor):
    return {
        **descriptor,
        "status": "admitted",
        "admitted": True,
        "plan_id": "retirement-plan",
        "assignments": [
            {
                "node_id": "master", "start_layer": 0, "end_layer": 4,
                "layers_count": 4, "has_embedding": True, "has_lm_head": False,
            },
            {
                "node_id": "worker", "start_layer": 4, "end_layer": 8,
                "layers_count": 4, "has_embedding": False, "has_lm_head": True,
            },
        ],
    }


@pytest.fixture
def retirement_case(monkeypatch, tmp_path):
    import api_server
    import config
    import model_host
    import model_supply_worker
    from scheduler import Scheduler

    # Scheduler construction registers its supply owner; isolate that registry
    # as well as the API facade so no instance escapes into the next test.
    monkeypatch.setattr(config, "STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(model_supply_worker, "_SELECTION_OWNERS", {})
    managers = []

    def candidate_manager(_lazy):
        manager = _Manager()
        managers.append(manager)
        return manager

    monkeypatch.setattr(
        model_host._LazyModelManager, "_get_instance", candidate_manager,
    )

    def new_master(host):
        master = Scheduler(host=host)
        master._distributed_inference_enabled = True
        sent = []
        connected = {"worker"}
        master._tcp_server = SimpleNamespace(
            _running=True,
            get_client_ids=lambda: list(connected),
            send_layer_config=lambda node_id, payload: sent.append(
                (node_id, dict(payload))
            ),
        )
        master.nodes["worker"] = SimpleNamespace(node_type="pc")
        monkeypatch.setattr(master, "_effective_role", lambda: "master")
        monkeypatch.setattr(master, "get_effective_node_id", lambda: "master")
        monkeypatch.setattr(master, "_task_worker_layer_stage_ids", lambda _ids: set())
        monkeypatch.setattr(master, "_task_worker_full_model_ids", lambda: set())
        monkeypatch.setattr(master, "_is_relay_host", lambda _node_id: False)
        monkeypatch.setattr(master, "_relay_segment_for_worker", lambda _node_id: None)
        monkeypatch.setattr(
            master, "get_pipeline_capacity_plan",
            lambda _ids, *, descriptor, require_distributed: _capacity_plan(descriptor),
        )
        # All lifecycle calls run synchronously. No retry thread or automatic
        # planning refresh may outlive a fixture or overwrite its transaction.
        monkeypatch.setattr(master, "_start_layer_config_retry_monitor", lambda: None)
        monkeypatch.setattr(master, "_commit_ready_pipeline_reshard", lambda _plan: True)
        monkeypatch.setattr(master, "push_layer_config_to_clients", lambda: None)
        monkeypatch.setattr(master, "refresh_task_worker_capabilities", lambda: None)
        return SimpleNamespace(master=master, host=host, sent=sent, connected=connected)

    active = _Manager(loaded=True)
    host = model_host.ModelHost(manager=active)
    host.model_loaded = True
    case = new_master(host)
    case.active = active
    case.managers = managers
    case.new_master = new_master
    case.model_path = tmp_path / "model"
    case.model_path.mkdir()
    monkeypatch.setattr(api_server, "model_host", host)
    monkeypatch.setattr(api_server, "model_manager", host)
    monkeypatch.setattr(api_server, "scheduler", case.master)
    monkeypatch.setattr(api_server, "_refresh_pipeline_layer_config", lambda: None)
    monkeypatch.setattr(api_server, "_reset_runtime_conversation_state", lambda **_kw: None)
    case.unload = api_server._unload_model_under_model_lock
    return case


def _start_candidate(case, candidate_id, *, master=None):
    master = case.master if master is None else master
    descriptor = master._host.prepare_pipeline_candidate(
        candidate_id, "retirement-model", str(case.model_path),
        quant_type="fp16", model_sha256="a" * 64,
    )
    return master.start_pipeline_model_transaction(
        candidate_id=candidate_id, descriptor=descriptor,
        model_path=str(case.model_path), quant_type="fp16",
    )


def _worker_ack(master, status):
    expected = master._pipeline_load_transaction["expected"]["worker"]
    master._handle_layer_config_ack("worker", {"data": {
        **expected,
        "status": status,
        "layer_range": [expected["start_layer"], expected["end_layer"]],
        "available_bytes": expected["required_bytes"],
    }})


def _publish(case):
    result = _start_candidate(case, "candidate-first")
    assert result["admitted"] is True
    assert result["transaction_phase"] == "preparing"
    _worker_ack(case.master, "prepared")
    assert case.master._pipeline_load_transaction["phase"] == "committing"
    assert case.host.pipeline_candidate_status()["phase"] == "materialized"
    _worker_ack(case.master, "ready")
    assert case.master._pipeline_load_transaction["phase"] == "ready"
    return case.master._pipeline_load_transaction


def _ack_cleanup(master, record):
    status = "finalized" if record["action"] == "finalize" else "released"
    return master._ack_pipeline_model_cleanup(record["node_id"], {
        **record["payload"], "status": status,
    })


@pytest.mark.parametrize("ack_before_unload", [True, False])
def test_finalized_pipeline_unload_admits_next_prepare(retirement_case, ack_before_unload):
    case = retirement_case
    master = case.master
    published = _publish(case)
    candidate = case.managers[-1]
    assert case.host.pipeline_candidate_status()["phase"] == "finalized"
    assert case.host.peek_manager() is candidate
    assert case.active.closed == 1
    assert master._worker_assignments.state("worker").phase == "ready"
    remote_finalize, = master._pipeline_model_cleanup_ledger.values()
    assert remote_finalize["node_id"] == "worker"
    assert remote_finalize["action"] == "finalize"

    # A scheduler fix must respect the existing irreversible ModelHost contract.
    with pytest.raises(RuntimeError, match="MODEL_TXN_STATE:.*is finalized"):
        case.host.abort_pipeline_candidate("candidate-first")
    assert candidate.closed == 0
    assert case.host.peek_manager() is candidate
    if ack_before_unload:
        assert _ack_cleanup(master, remote_finalize)
    debt_before = deepcopy(master._pipeline_model_cleanup_ledger)
    sent_before = list(case.sent)

    result = case.unload()

    assert result["success"] is True
    assert result["unloaded"] is True
    assert result["loaded"] is False
    assert published["phase"] == "invalidated"
    assert published["reason_code"] == "pipeline_model_changed"
    assert master._pipeline_load_transaction is None
    assert master._active_pipeline_capacity_plan is None
    assert master._worker_assignments.snapshot() == {}
    assert case.host.model_loaded is False
    assert case.host.current_quant is None
    assert case.host.peek_manager() is candidate
    assert candidate.closed == 1
    assert case.active.closed == 1
    assert case.host.pipeline_candidate_status()["phase"] == "finalized"
    assert master._pipeline_model_cleanup_ledger == debt_before
    assert case.sent == sent_before
    if not ack_before_unload:
        assert _ack_cleanup(master, remote_finalize)
    assert master._pipeline_model_cleanup_ledger == {}

    next_prepare = _start_candidate(case, "candidate-next")

    assert next_prepare["admitted"] is True
    assert next_prepare["transaction_phase"] == "preparing"
    assert next_prepare["config_id"] != published["config_id"]
    assert case.host.pipeline_candidate_status()["transaction_id"] == "candidate-next"
    assert case.host.pipeline_candidate_status()["phase"] == "prepared"
    assert case.sent[-1][1]["phase"] == "prepare"
    assert case.sent[-1][1]["config_id"] == next_prepare["config_id"]


def test_unload_keeps_failed_finalize_debt_retryable(retirement_case):
    case = retirement_case
    master = case.master
    case.active.fail_unload = True
    published = _publish(case)
    candidate = case.managers[-1]
    assert case.host.pipeline_candidate_status()["phase"] == "committed"
    assert case.active.unload_attempts == 1
    assert case.active.closed == 0
    debt_before = deepcopy(master._pipeline_model_cleanup_ledger)
    assert {(item["node_id"], item["action"]) for item in debt_before.values()} == {
        ("master", "finalize"), ("worker", "finalize"),
    }
    sent_before = list(case.sent)

    assert case.unload()["unloaded"] is True

    assert published["phase"] == "invalidated"
    assert master._pipeline_load_transaction is None
    assert master._pipeline_model_cleanup_ledger == debt_before
    assert case.sent == sent_before
    assert case.host.peek_manager() is candidate
    assert candidate.closed == 1
    assert case.active.is_loaded is True
    assert case.active.unload_attempts == 1
    assert case.host.pipeline_candidate_status()["phase"] == "committed"
    blocked = master.start_pipeline_model_transaction(
        candidate_id="candidate-next", descriptor=published["plan"],
        model_path=str(case.model_path), quant_type="fp16",
    )
    assert blocked["admitted"] is False
    assert blocked["reason_code"] == "MODEL_TXN_CLEANUP_PENDING"
    assert blocked["transaction_id"] == "candidate-first"

    case.active.fail_unload = False
    retry_at = max(item["next_retry"] for item in debt_before.values()) + 1.0
    assert master._retry_pending_layer_configs(now=retry_at) == 2
    assert case.host.pipeline_candidate_status()["phase"] == "finalized"
    assert case.active.closed == 1
    assert case.active.unload_attempts == 2
    assert candidate.closed == 1
    remote_finalize, = master._pipeline_model_cleanup_ledger.values()
    assert remote_finalize["action"] == "finalize"
    assert remote_finalize["payload"] == debt_before[remote_finalize["cleanup_id"]]["payload"]
    assert _ack_cleanup(master, remote_finalize)
    assert master._pipeline_model_cleanup_ledger == {}
    assert _start_candidate(case, "candidate-next")["admitted"] is True


@pytest.mark.parametrize("phase", [
    "preparing", "committing_local", "committing", "committing_global",
])
def test_unpublished_candidate_invalidation_still_aborts(retirement_case, phase):
    case = retirement_case
    master = case.master
    assert _start_candidate(case, "candidate-first")["admitted"] is True
    candidate = case.managers[-1]
    if phase != "preparing":
        _worker_ack(master, "prepared")
        assert case.host.pipeline_candidate_status()["phase"] == "materialized"
        if phase == "committing_global":
            case.host.commit_pipeline_candidate("candidate-first")
            assert case.host.peek_manager() is candidate
        # Exercise each unpublished control-plane boundary. Committing_local
        # and committing_global are normally transient within the same call.
        master._pipeline_load_transaction["phase"] = phase
    transaction = master._pipeline_load_transaction
    assignment_id = transaction["expected"]["worker"]["assignment_id"]
    sent_before = len(case.sent)

    master._invalidate_pipeline_load_transaction(reason="candidate retired before publication")

    assert transaction["phase"] == "invalidated"
    assert transaction["reason_code"] == "pipeline_model_changed"
    assert case.host.pipeline_candidate_status()["phase"] == "aborted"
    assert case.host.peek_manager() is case.active
    assert case.host.model_loaded is True
    assert case.active.is_loaded is True
    assert case.active.closed == 0
    assert candidate.closed == 1
    remote_abort, = master._pipeline_model_cleanup_ledger.values()
    assert remote_abort["node_id"] == "worker"
    assert remote_abort["action"] == "abort"
    assert remote_abort["assignment_id"] == assignment_id
    assert remote_abort["candidate_config_id"] == transaction["config_id"]
    assert remote_abort["generation"] > transaction["generation"]
    assert remote_abort["payload"]["aborted_config_id"] == transaction["config_id"]
    assert case.sent[sent_before:] == [("worker", remote_abort["payload"])]
    assert _ack_cleanup(master, remote_abort)
    assert master._pipeline_model_cleanup_ledger == {}


@pytest.mark.parametrize("action", ["abort", "finalize"])
def test_restart_retains_master_debt_until_independent_process_end_evidence(retirement_case, action, monkeypatch):
    import local_store
    from model_host import ModelHost
    from scheduler_pipeline import _PIPELINE_LIFECYCLE_STATE_KEY

    case = retirement_case
    master = case.master
    master._record_pipeline_model_cleanup_locked(
        node_id="master", candidate_config_id="candidate-dead-master",
        generation=9, action=action,
    )
    terminal_id_key = "aborted_config_id" if action == "abort" else "finalized_config_id"
    remote = master._record_pipeline_model_cleanup_locked(
        node_id="worker", candidate_config_id="cfg-remote",
        assignment_id="assignment-remote", generation=9, action=action,
        payload={terminal_id_key: "cfg-remote", action: True},
    )
    # Round-trip the real bounded snapshot through JSON and the isolated store.
    # A fresh Scheduler/ModelHost represents the new process, with no candidate
    # or prior runtime from the scheduler that produced this durable record.
    snapshot = json.loads(json.dumps(master._pipeline_lifecycle_snapshot_locked()))
    assert snapshot["schema_version"] == 4
    assert len(snapshot["cleanup_ledger"]) == 2
    local_store.set_local_setting(_PIPELINE_LIFECYCLE_STATE_KEY, snapshot)
    restarted_case = case.new_master(ModelHost(manager=_Manager()))
    restarted = restarted_case.master

    restarted._load_pipeline_recovery_state()

    assert restarted._pipeline_lifecycle_persist_ok is True
    assert restarted._pipeline_recovery_pending is False
    assert restarted._pipeline_load_transaction is None
    assert restarted._host.pipeline_candidate_status()["phase"] == "idle"
    assert len(restarted._pipeline_model_cleanup_ledger) == 2
    local, = [record for record in restarted._pipeline_model_cleanup_ledger.values() if record["node_id"] == "master"]
    assert restarted._dispatch_pipeline_model_cleanup(local) is False
    assert local["process_boot_id"] == master._model_control.boot_id
    restored = restarted._pipeline_model_cleanup_ledger[remote["cleanup_id"]]
    for key in (
        "cleanup_id", "node_id", "candidate_config_id", "assignment_id",
        "generation", "action", "created_at", "payload",
    ):
        assert restored[key] == remote[key]
    assert restored["legacy_identity"] is False
    assert restored["state"] == "await_reconnect"
    assert restored["attempts"] == 0
    assert restored["next_retry"] == 0.0
    blocked = restarted.start_pipeline_model_transaction(
        candidate_id="candidate-after-restart", descriptor={},
        model_path=str(case.model_path), quant_type="fp16",
    )
    assert blocked["reason_code"] == "MODEL_TXN_CLEANUP_PENDING"
    assert blocked["transaction_id"] == "candidate-dead-master"
    assert restarted._replay_pipeline_model_cleanup_for_node("master") is True
    assert restarted._replay_pipeline_model_cleanup_for_node("worker") is True
    assert restarted_case.sent == [("worker", remote["payload"])]
    assert remote["cleanup_id"] in restarted._pipeline_model_cleanup_ledger
    status = "released" if action == "abort" else "finalized"
    assert restarted._ack_pipeline_model_cleanup("worker", {
        **remote["payload"], "status": status, "assignment_id": "stale-assignment",
    }) is False
    assert remote["cleanup_id"] in restarted._pipeline_model_cleanup_ledger
    assert _ack_cleanup(restarted, restored)
    assert set(restarted._pipeline_model_cleanup_ledger) == {local["cleanup_id"]}
    import psutil
    def ended(_pid):
        raise psutil.NoSuchProcess(_pid)
    original_process = psutil.Process
    monkeypatch.setattr(psutil, "Process", ended)
    assert restarted._dispatch_pipeline_model_cleanup(local) is True
    assert restarted._pipeline_model_cleanup_ledger == {}
    monkeypatch.setattr(psutil, "Process", original_process)
    assert _start_candidate(case, "candidate-after-restart", master=restarted)["admitted"] is True
