"""Deterministic v2 task-worker control-plane pre-validation scenarios.

This module uses the production protocol, control-plane and remote-provider
objects in one process. It does not open TCP connections or run model code.
"""

from __future__ import annotations

import copy
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[2] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from task_provider import (
    ModelIdentity,
    ProviderExecutionError,
    ProviderUnavailable,
    StageAttempt,
    StageRequest,
)
from task_worker_adapter import (
    RemoteFullWorkerProvider,
    TaskWorkerControlPlane,
)
from task_worker_protocol import (
    WorkerProtocolError,
    build_message,
    canonical_sha256,
)


SIMULATION_SCHEMA_VERSION = "qlh.task_worker_control_simulation.v1"
_WAIT_SECONDS = 2.0


class SimulationScenarioError(ValueError):
    """Raised when a caller asks for an unknown v2 control scenario."""


@dataclass(frozen=True)
class SimulationScenario:
    scenario_id: str
    description: str


_SCENARIOS = {
    "hello_replay_and_conflict": SimulationScenario(
        "hello_replay_and_conflict",
        "duplicate hello replays its acknowledgement while conflicting reuse fails",
    ),
    "model_identity_mismatch": SimulationScenario(
        "model_identity_mismatch",
        "remote reservation refuses a model not advertised by the worker",
    ),
    "result_replay_and_epoch_fencing": SimulationScenario(
        "result_replay_and_epoch_fencing",
        "wrong epoch is fenced and an exact duplicate terminal response is idempotent",
    ),
    "disconnect_fences_late_result": SimulationScenario(
        "disconnect_fences_late_result",
        "disconnect unblocks the wait and rejects the late terminal result",
    ),
    "cancel_acknowledgement": SimulationScenario(
        "cancel_acknowledgement",
        "local cancellation sends one cancel message and accepts its acknowledgement",
    ),
    "lease_renewal": SimulationScenario(
        "lease_renewal",
        "a valid renewal is emitted before the exact terminal result",
    ),
}


def available_scenarios() -> tuple[SimulationScenario, ...]:
    return tuple(_SCENARIOS[key] for key in sorted(_SCENARIOS))


class TaskWorkerControlSimulationHarness:
    """Run fixed v2 control-plane scenarios and return safe summary evidence."""

    def run(self, scenario_id: str) -> dict:
        scenario = _SCENARIOS.get(str(scenario_id or ""))
        if scenario is None:
            raise SimulationScenarioError(f"unknown simulation scenario: {scenario_id}")
        handler = getattr(self, f"_run_{scenario.scenario_id}")
        details = handler()
        return {
            "schema_version": SIMULATION_SCHEMA_VERSION,
            "scenario_id": scenario.scenario_id,
            "execution_environment": {
                "kind": "in_memory_v2_control_simulation",
                "network_io": False,
                "subprocesses_started": False,
                "real_model_loaded": False,
                "physical_nodes": False,
            },
            "contract": details,
        }

    @staticmethod
    def _capabilities() -> dict:
        return {
            "stage_types": ["full_inference"],
            "engines": ["pytorch"],
            "models": [{
                "model_id": "sim-qwen-v2",
                "engine": "pytorch",
                "format": "safetensors",
                "revision": "simulation-v1",
                "sha256": "a" * 64,
            }],
            "max_concurrency": 1,
        }

    @classmethod
    def _identity(cls) -> ModelIdentity:
        return ModelIdentity(**cls._capabilities()["models"][0])

    @classmethod
    def _admitted_control(cls) -> TaskWorkerControlPlane:
        worker = TaskWorkerControlPlane()
        coordinator = TaskWorkerControlPlane()
        hello = worker.begin_worker_hello(
            node_id="sim-worker-v2",
            capabilities=cls._capabilities(),
            sent_at_ms=1_000,
        )
        assert hello is not None
        acknowledgement = coordinator.receive_on_coordinator(
            "sim-worker-v2",
            hello.snapshot(),
            coordinator_node_id="sim-master",
            sent_at_ms=1_001,
        )
        worker.receive_on_worker(acknowledgement.snapshot())
        return coordinator

    @classmethod
    def _request(cls, provider_id: str, *, identity: ModelIdentity | None = None):
        return StageRequest(
            workflow_id="wf_simv2control01",
            request_id="request_simv2control01",
            stage_id="candidate_a",
            stage_type="full_inference",
            provider_id=provider_id,
            dependencies={},
            root_input={"message": "not-retained-in-report"},
            model_identity=identity or cls._identity(),
        )

    @staticmethod
    def _wait_until(predicate) -> None:
        deadline = time.monotonic() + _WAIT_SECONDS
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.005)
        raise RuntimeError("simulated control-plane event did not arrive")

    @staticmethod
    def _response_identity(offer: dict) -> dict:
        return {
            key: offer[key]
            for key in (
                "workflow_id",
                "stage_id",
                "attempt_id",
                "lease_id",
                "lease_epoch",
                "provider_id",
            )
        }

    @staticmethod
    def _provider_summary(provider: RemoteFullWorkerProvider) -> dict:
        status = provider.inspect()
        return {
            "provider_id": status.provider_id,
            "provider_kind": status.provider_kind,
            "node_id": status.node_id,
            "active_reservations": status.active_reservations,
            "healthy": status.healthy,
        }

    def _run_hello_replay_and_conflict(self) -> dict:
        worker = TaskWorkerControlPlane()
        coordinator = TaskWorkerControlPlane()
        hello = worker.begin_worker_hello(
            node_id="sim-worker-v2",
            capabilities=self._capabilities(),
            sent_at_ms=1_000,
        )
        assert hello is not None
        first = coordinator.receive_on_coordinator(
            "sim-worker-v2", hello.snapshot(),
            coordinator_node_id="sim-master", sent_at_ms=1_001,
        )
        replay = coordinator.receive_on_coordinator(
            "sim-worker-v2", hello.snapshot(),
            coordinator_node_id="sim-master", sent_at_ms=1_002,
        )
        conflicting = copy.deepcopy(hello.snapshot())
        conflicting["payload"]["capabilities"]["max_concurrency"] = 2
        try:
            coordinator.receive_on_coordinator(
                "sim-worker-v2", conflicting,
                coordinator_node_id="sim-master", sent_at_ms=1_003,
            )
        except WorkerProtocolError as exc:
            conflict_code = exc.code
        else:
            raise RuntimeError("conflicting hello replay was accepted")
        status = coordinator.status(role="master")
        return {
            "selected_version": first.payload["selected_version"],
            "replay_idempotent": replay.snapshot() == first.snapshot(),
            "rejected_codes": [conflict_code],
            "outbound_message_types": ["hello", "hello_ack"],
            "control_plane_connected": status["control_plane_connected"],
        }

    def _run_model_identity_mismatch(self) -> dict:
        control = self._admitted_control()
        outbound = []
        provider = RemoteFullWorkerProvider(
            node_id="sim-worker-v2",
            peer_snapshot=lambda: control.worker_snapshot("sim-worker-v2"),
            send_message=outbound.append,
        )
        incompatible = ModelIdentity(
            model_id="sim-other-model",
            engine="pytorch",
            format="safetensors",
            revision="simulation-v1",
            sha256="b" * 64,
        )
        try:
            provider.reserve(self._request(provider.provider_id, identity=incompatible))
        except ProviderUnavailable as exc:
            rejected_code = exc.code
        else:
            raise RuntimeError("incompatible model was reserved")
        return {
            "rejected_codes": [rejected_code],
            "outbound_message_types": [],
            "provider": self._provider_summary(provider),
        }

    def _run_result_replay_and_epoch_fencing(self) -> dict:
        control = self._admitted_control()
        outbound = []
        provider = RemoteFullWorkerProvider(
            node_id="sim-worker-v2",
            peer_snapshot=lambda: control.worker_snapshot("sim-worker-v2"),
            send_message=outbound.append,
        )
        request = self._request(provider.provider_id)
        reservation = provider.reserve(request)
        attempt = StageAttempt(
            attempt_id="att_simv2epoch01",
            request=request,
            provider_id=provider.provider_id,
            lease_id="lease_simv2epoch01",
            lease_epoch=2,
            lease_expires_at=time.time() + 5.0,
        )
        outcome = {}
        finished = threading.Event()

        def execute() -> None:
            try:
                outcome["result"] = provider.execute(
                    attempt, reservation, threading.Event(),
                )
            except BaseException as exc:
                outcome["error"] = exc
            finally:
                finished.set()

        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
        try:
            self._wait_until(lambda: bool(outbound))
            identity = self._response_identity(outbound[0].payload)
            provider.handle_message(build_message(
                "stage_accept",
                {**identity, "accepted": True, "reason_code": "", "retryable": False},
                message_id="msg_simv2epoch_accept", sent_at_ms=2_000, version=2,
            ).snapshot())
            wrong = {"content": "not-retained"}
            wrong_identity = {**identity, "lease_epoch": 1}
            try:
                provider.handle_message(build_message(
                    "stage_result",
                    {
                        **wrong_identity,
                        "output": wrong,
                        "output_sha256": canonical_sha256(wrong),
                        "metadata": {},
                    },
                    message_id="msg_simv2epoch_wrong", sent_at_ms=2_001, version=2,
                ).snapshot())
            except WorkerProtocolError as exc:
                rejected_code = exc.code
            else:
                raise RuntimeError("wrong epoch result was accepted")
            if finished.wait(0.05):
                raise RuntimeError("wrong epoch result woke the pending attempt")
            output = {"content": "not-retained"}
            result = build_message(
                "stage_result",
                {
                    **identity,
                    "output": output,
                    "output_sha256": canonical_sha256(output),
                    "metadata": {},
                },
                message_id="msg_simv2epoch_result", sent_at_ms=2_002, version=2,
            )
            provider.handle_message(result.snapshot())
            self._wait_until(finished.is_set)
            if "error" in outcome:
                raise outcome["error"]
            replay = provider.handle_message(result.snapshot())
            provider.release(reservation.reservation_id)
            thread.join(_WAIT_SECONDS)
            return {
                "rejected_codes": [rejected_code],
                "replay_idempotent": replay.message_id == result.message_id,
                "terminal_state": "completed",
                "outbound_message_types": [item.message_type for item in outbound],
                "provider": self._provider_summary(provider),
            }
        finally:
            provider.close()
            if thread.is_alive():
                thread.join(_WAIT_SECONDS)

    def _run_disconnect_fences_late_result(self) -> dict:
        control = self._admitted_control()
        outbound = []
        provider = RemoteFullWorkerProvider(
            node_id="sim-worker-v2",
            peer_snapshot=lambda: control.worker_snapshot("sim-worker-v2"),
            send_message=outbound.append,
        )
        request = self._request(provider.provider_id)
        reservation = provider.reserve(request)
        attempt = StageAttempt(
            attempt_id="att_simv2disconnect01",
            request=request,
            provider_id=provider.provider_id,
            lease_id="lease_simv2disconnect01",
            lease_epoch=1,
            lease_expires_at=time.time() + 5.0,
        )
        outcome = {}
        finished = threading.Event()

        def execute() -> None:
            try:
                provider.execute(attempt, reservation, threading.Event())
            except BaseException as exc:
                outcome["error"] = exc
            finally:
                finished.set()

        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
        try:
            self._wait_until(lambda: bool(outbound))
            identity = self._response_identity(outbound[0].payload)
            provider.handle_message(build_message(
                "stage_accept",
                {**identity, "accepted": True, "reason_code": "", "retryable": False},
                message_id="msg_simv2disconnect_accept", sent_at_ms=3_000, version=2,
            ).snapshot())
            provider.notify_disconnect()
            self._wait_until(finished.is_set)
            if not isinstance(outcome.get("error"), ProviderExecutionError):
                raise RuntimeError("disconnect did not fail the pending attempt")
            late = {"content": "not-retained"}
            late_code = ""
            try:
                provider.handle_message(build_message(
                    "stage_result",
                    {
                        **identity,
                        "output": late,
                        "output_sha256": canonical_sha256(late),
                        "metadata": {},
                    },
                    message_id="msg_simv2disconnect_late", sent_at_ms=3_001, version=2,
                ).snapshot())
            except WorkerProtocolError as exc:
                late_code = exc.code          # 旧行为：同步拒绝
            # ★ 票 7/票 8 起，断连终局之后迟到的 `stage_result` 不再抛
            #   `WorkerProtocolError`，而是被**幂等吸收**
            #   （实现侧 `_absorb_late_stage_response_locked`，记
            #   `event=task_worker_late_stage_response_ignored terminal_kind=disconnect`）。
            #   两种行为都算"未采纳"，故判据改为：迟到内容不得出现在 outcome
            #   或 provider 状态里 —— 而不是要求它必须抛异常。
            provider.release(reservation.reservation_id)
            thread.join(_WAIT_SECONDS)
            if ("not-retained" in repr(outcome)
                    or "not-retained" in repr(self._provider_summary(provider))):
                raise RuntimeError("late result after disconnect was retained")
            return {
                "terminal_error_code": outcome["error"].code,
                "rejected_codes": [late_code],
                "outbound_message_types": [item.message_type for item in outbound],
                "provider": self._provider_summary(provider),
            }
        finally:
            provider.close()
            if thread.is_alive():
                thread.join(_WAIT_SECONDS)

    def _run_cancel_acknowledgement(self) -> dict:
        control = self._admitted_control()
        outbound = []
        provider = RemoteFullWorkerProvider(
            node_id="sim-worker-v2",
            peer_snapshot=lambda: control.worker_snapshot("sim-worker-v2"),
            send_message=outbound.append,
        )
        request = self._request(provider.provider_id)
        reservation = provider.reserve(request)
        attempt = StageAttempt(
            attempt_id="att_simv2cancel01",
            request=request,
            provider_id=provider.provider_id,
            lease_id="lease_simv2cancel01",
            lease_epoch=1,
            lease_expires_at=time.time() + 5.0,
        )
        outcome = {}
        finished = threading.Event()

        def execute() -> None:
            try:
                provider.execute(attempt, reservation, threading.Event())
            except BaseException as exc:
                outcome["error"] = exc
            finally:
                finished.set()

        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
        try:
            self._wait_until(lambda: bool(outbound))
            identity = self._response_identity(outbound[0].payload)
            provider.handle_message(build_message(
                "stage_accept",
                {**identity, "accepted": True, "reason_code": "", "retryable": False},
                message_id="msg_simv2cancel_accept", sent_at_ms=4_000, version=2,
            ).snapshot())
            provider.cancel(attempt.attempt_id)
            self._wait_until(lambda: any(
                item.message_type == "stage_cancel" for item in outbound
            ))
            provider.handle_message(build_message(
                "stage_cancelled",
                {**identity, "reason_code": "coordinator_cancelled"},
                message_id="msg_simv2cancel_ack", sent_at_ms=4_001, version=2,
            ).snapshot())
            self._wait_until(finished.is_set)
            if not isinstance(outcome.get("error"), ProviderExecutionError):
                raise RuntimeError("cancellation did not terminate the attempt")
            provider.release(reservation.reservation_id)
            thread.join(_WAIT_SECONDS)
            return {
                "terminal_error_code": outcome["error"].code,
                "outbound_message_types": [item.message_type for item in outbound],
                "provider": self._provider_summary(provider),
            }
        finally:
            provider.close()
            if thread.is_alive():
                thread.join(_WAIT_SECONDS)

    def _run_lease_renewal(self) -> dict:
        control = self._admitted_control()
        outbound = []
        provider = RemoteFullWorkerProvider(
            node_id="sim-worker-v2",
            peer_snapshot=lambda: control.worker_snapshot("sim-worker-v2"),
            send_message=outbound.append,
        )
        request = self._request(provider.provider_id)
        reservation = provider.reserve(request)
        initial_deadline = time.time() + 5.0
        attempt = StageAttempt(
            attempt_id="att_simv2renew01",
            request=request,
            provider_id=provider.provider_id,
            lease_id="lease_simv2renew01",
            lease_epoch=1,
            lease_expires_at=initial_deadline,
        )
        outcome = {}
        finished = threading.Event()

        def execute() -> None:
            try:
                outcome["result"] = provider.execute(
                    attempt, reservation, threading.Event(),
                )
            except BaseException as exc:
                outcome["error"] = exc
            finally:
                finished.set()

        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
        try:
            self._wait_until(lambda: bool(outbound))
            identity = self._response_identity(outbound[0].payload)
            provider.handle_message(build_message(
                "stage_accept",
                {**identity, "accepted": True, "reason_code": "", "retryable": False},
                message_id="msg_simv2renew_accept", sent_at_ms=5_000, version=2,
            ).snapshot())
            renewed = initial_deadline + 5.0
            if not provider.renew_lease(
                attempt.attempt_id, attempt.lease_id, attempt.lease_epoch, renewed,
            ):
                raise RuntimeError("valid lease renewal was rejected")
            self._wait_until(lambda: any(
                item.message_type == "lease_renew" for item in outbound
            ))
            output = {"content": "not-retained"}
            provider.handle_message(build_message(
                "stage_result",
                {
                    **identity,
                    "output": output,
                    "output_sha256": canonical_sha256(output),
                    "metadata": {},
                },
                message_id="msg_simv2renew_result", sent_at_ms=5_001, version=2,
            ).snapshot())
            self._wait_until(finished.is_set)
            if "error" in outcome:
                raise outcome["error"]
            provider.release(reservation.reservation_id)
            thread.join(_WAIT_SECONDS)
            return {
                "terminal_state": "completed",
                "outbound_message_types": [item.message_type for item in outbound],
                "provider": self._provider_summary(provider),
            }
        finally:
            provider.close()
            if thread.is_alive():
                thread.join(_WAIT_SECONDS)
