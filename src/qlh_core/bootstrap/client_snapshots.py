"""Read-only client composition with bounded revision consistency checks."""
from copy import deepcopy
import time
from qlh_core.contracts.snapshots import envelope, reason

class ClientSnapshots:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def current(self):
        scheduler = self.scheduler
        owner = scheduler._model_control
        for _ in range(3):
            state = owner.snapshot()
            queue = scheduler._execution_app.requests.queue_snapshot()
            enabled = scheduler.get_distributed_inference_enabled()
            mode = {"desired": "distributed" if enabled else "local", "phase": "stable", "eligible": bool(enabled)}
            participation = getattr(scheduler, "_participation", None)
            if participation is not None:
                mode.update(participation.snapshot())
            deployment = deepcopy(state["deployments"].get(state["active_deployment"]))
            readiness_reason = None
            runnable = False
            if deployment is not None and mode["phase"] == "stable" and enabled:
                try:
                    scheduler._execution_app.freeze()
                    runnable = True
                except Exception as exc:
                    readiness_reason = reason("deployment", getattr(exc, "code", "deployment_observation_unavailable"), "execution_admission",
                                              retryability="retry", cause=getattr(exc, "reason_detail", None))
            if not runnable and readiness_reason is None:
                readiness_reason = reason("deployment", "execution_deployment_unavailable" if enabled else "local_mode", "model_control", retryability="retry")
            end = owner.snapshot()
            if (state["revision"] != end["revision"] or state["boot_id"] != end["boot_id"]
                    or queue["revision"] != scheduler._execution_app.requests.queue_snapshot()["revision"]
                    or enabled != scheduler.get_distributed_inference_enabled()
                    or mode.get("revision", 0) != (participation.snapshot().get("revision", 0) if participation is not None else 0)):
                continue
            now = time.time()
            identity = {"node_id": scheduler.get_effective_node_id(), "boot_id": owner.boot_id, "owner": "model_control"}
            selected = state["selected_operation"]
            operation = None
            if selected and selected in state["operations"]:
                from model_intent import ModelIntentCoordinator
                operation = ModelIntentCoordinator(scheduler, None).get(selected)
                check = owner.snapshot()
                if check["revision"] != state["revision"] or check["boot_id"] != state["boot_id"]:
                    continue
            if deployment is not None:
                deployment = envelope({**deployment, "runnable": runnable}, revision=state["revision"],
                    identity={**identity, "deployment_id": state["active_deployment"]}, observed_at=now,
                    validity="valid" if runnable else "invalid", reason=readiness_reason)
            return envelope({"operation": operation, "deployment": deployment, "runnable": runnable, "queue": queue,
                             "participation": mode, "commit_scope": state["commit_scope"]},
                            revision=state["revision"], identity=identity, observed_at=now,
                            reason=readiness_reason)
        return envelope({"runnable": False}, revision=None, identity={"boot_id": owner.boot_id}, observed_at=time.time(),
                        validity="pending", reason=reason("snapshot", "snapshot_revision_changed", "client_snapshots", retryability="retry"))
