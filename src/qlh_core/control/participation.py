"""Participation intent and transition observations; native retirement stays injected."""
from copy import deepcopy
import threading
from qlh_core.contracts.snapshots import reason

class ParticipationService:
    def __init__(self, *, enabled, persist=None):
        self._lock = threading.RLock()
        self._persist = persist or (lambda mode: None)
        self._value = {"desired": "distributed" if enabled else "local", "phase": "stable",
            "revision": 0, "eligible": bool(enabled), "cleanup_pending": False, "reason_detail": None}

    def bind_persistence(self, persist):
        with self._lock:
            self._persist = persist

    def snapshot(self):
        with self._lock:
            return deepcopy(self._value)

    def change(self, mode, transition):
        if mode not in {"distributed", "local"}:
            raise ValueError("participation_mode_invalid")
        # Commands serialize; snapshots observe the transition without waiting for native cleanup.
        with self._lock:
            if self._value["phase"] in {"joining", "leaving"}:
                raise RuntimeError("participation_transition_busy")
            self._persist(mode)
            self._value.update(desired=mode, phase="joining" if mode == "distributed" else "leaving",
                               revision=self._value["revision"] + 1, eligible=False, cleanup_pending=False, reason_detail=None)
        try:
            result = transition(mode)
        except Exception as exc:
            detail = getattr(exc, "detail", None)
            code = (detail.get("code") or detail.get("reason_code")) if isinstance(detail, dict) else getattr(exc, "code", None)
            code = code or "participation_transition_failed"
            with self._lock:
                self._value.update(phase="failed", revision=self._value["revision"] + 1,
                    cleanup_pending=code in {"WORKER_RUNTIME_RETIRE_FAILED", "request_runtime_cleanup_pending"},
                    reason_detail=reason("participation", code, "mode_transition", retryability="retry",
                        cause=detail.get("reason_detail") if isinstance(detail, dict) else getattr(exc, "reason_detail", None)))
            raise
        with self._lock:
            self._value.update(phase="stable", revision=self._value["revision"] + 1, eligible=mode == "distributed", cleanup_pending=False)
            return {**result, "participation": deepcopy(self._value)}
