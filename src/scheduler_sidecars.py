"""Scheduler mixin for Gemma4/Qwen3 sidecar and dry-run workflows."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

from qwen3_pipeline_transaction import (
    Qwen3PipelineDryRunTransaction,
    Qwen3PipelineProtocolError,
)
from qwen3_pipeline_loopback import (
    Qwen3LoopbackError,
    Qwen3PipelineLoopbackWorker,
    sign_loopback_message,
    validate_loopback_base_url,
    verify_loopback_message,
)
from qwen3_pipeline_sidecar import Qwen3PipelineSidecarSession, Qwen3SidecarError
from qwen3_pipeline_multisidecar import (
    Qwen3PipelineMultiSidecar,
    cleanup_qwen3_local_artifacts,
)
from gemma4_pipeline_multisidecar import (
    Gemma4MultiSidecarError,
    Gemma4PipelineMultiSidecar,
)
from gemma4_pipeline_sidecar import (
    Gemma4PipelineSidecarSession,
    Gemma4SidecarError,
)

logger = logging.getLogger("scheduler")


class SchedulerSidecarMixin:
    """Sidecar methods use state and callbacks supplied by Scheduler."""

    def configure_gemma4_pipeline_sidecar(
        self,
        assignment_paths: dict[str, str],
        *,
        sidecar_python: str | None = None,
        artifact_root: str | None = None,
    ) -> dict:
        """Bind node-local filtered assignments to the experimental route."""
        if not isinstance(assignment_paths, dict) or not assignment_paths:
            raise Gemma4MultiSidecarError(
                "gemma4_scheduler_config_invalid", "assignment path map is empty",
            )
        normalized: dict[str, str] = {}
        resolved_paths: set[str] = set()
        for raw_node_id, raw_path in assignment_paths.items():
            node_id = str(raw_node_id or "")
            path = Path(str(raw_path or "")).expanduser().absolute().resolve(strict=False)
            if not node_id or not path.is_dir():
                raise Gemma4MultiSidecarError(
                    "gemma4_scheduler_config_invalid",
                    "every Gemma 4 node requires a local assignment directory",
                )
            normalized_path = str(path)
            if normalized_path in resolved_paths:
                raise Gemma4MultiSidecarError(
                    "gemma4_scheduler_config_invalid",
                    "each segment requires a distinct filtered assignment directory",
                )
            normalized[node_id] = normalized_path
            resolved_paths.add(normalized_path)
        resolved_python = None
        if sidecar_python:
            resolved_python = str(
                Path(sidecar_python).expanduser().absolute().resolve(strict=False)
            )
        resolved_root = None
        if artifact_root:
            path = Path(artifact_root).expanduser().absolute().resolve(strict=False)
            path.mkdir(parents=True, exist_ok=True)
            if not path.is_dir():
                raise Gemma4MultiSidecarError(
                    "gemma4_scheduler_config_invalid", "artifact root is unavailable",
                )
            resolved_root = str(path)
        with self._gemma4_local_chain_lock:
            if self._gemma4_local_chain is not None:
                raise Gemma4MultiSidecarError(
                    "gemma4_scheduler_chain_active",
                    "cannot reconfigure an active Gemma 4 chain",
                )
            self._gemma4_assignment_paths = normalized
            self._gemma4_sidecar_python_override = resolved_python
            self._gemma4_local_artifact_root_override = resolved_root
        return {
            "configured": True,
            "node_ids": sorted(normalized),
            "assignment_count": len(normalized),
            "runtime_environment": ".venv-gemma4-pipeline",
            "production_admitted": False,
        }

    def _gemma4_local_artifact_root(self) -> Path:
        if self._gemma4_local_artifact_root_override:
            root = Path(self._gemma4_local_artifact_root_override)
        else:
            from config import STATE_DIR

            root = Path(STATE_DIR) / "gemma4-local-chain"
        root = root.expanduser().absolute().resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _gemma4_sidecar_session_from_message(
        self, message: dict,
    ) -> Gemma4PipelineSidecarSession:
        node_id = str(message.get("node_id", "") or "")
        model_path = self._gemma4_assignment_paths.get(node_id)
        if not model_path:
            raise Gemma4SidecarError(
                "gemma4_sidecar_model_missing",
                "node-local filtered assignment path is not configured",
            )
        return Gemma4PipelineSidecarSession(
            model_path=model_path,
            model_id=str(message.get("model_id", "") or ""),
            model_sha256=str(message.get("model_sha256", "") or ""),
            config_id=str(message.get("config_id", "") or ""),
            plan_id=str(message.get("plan_id", "") or ""),
            node_id=node_id,
            layer_range=message.get("layer_range", [0, 0]),
            total_layers=int(message.get("total_layers", 0) or 0),
            has_embedding=bool(message.get("has_embedding", False)),
            has_lm_head=bool(message.get("has_lm_head", False)),
            required_shared_kv_types=message.get("requires_shared_kv_types", []),
            produced_shared_kv_types=message.get("produces_shared_kv_types", []),
            execution_device=str(message.get("execution_device", "cpu") or "cpu"),
            dtype=str(message.get("dtype", "float32") or "float32"),
            generation=int(message.get("generation", 0) or 0),
            assignment_manifest_sha256=str(
                message.get("assignment_manifest_sha256", "") or ""
            ),
            sidecar_python=self._gemma4_sidecar_python_override,
        )

    def begin_gemma4_local_sidecar_chain(
        self,
        contract: dict,
        *,
        session_factory=None,
    ) -> dict:
        """Prepare and commit an explicit local Gemma 4 development chain."""
        factory = session_factory or self._gemma4_sidecar_session_from_message
        with self._gemma4_local_chain_lock:
            if self._gemma4_local_chain is not None:
                raise Gemma4MultiSidecarError(
                    "gemma4_scheduler_chain_active", "another Gemma 4 chain is active",
                )
            chain = Gemma4PipelineMultiSidecar.from_contract(
                contract=contract,
                artifact_root=self._gemma4_local_artifact_root(),
                session_factory=factory,
            )
            self._gemma4_local_chain = chain
            self._gemma4_local_contract = dict(contract)
            try:
                chain.prepare()
                chain.commit()
            except Exception:
                chain.abort()
                self._gemma4_local_chain = None
                self._gemma4_local_contract = None
                raise
            return {
                "status": "committed",
                "state": chain.snapshot,
                "production_admitted": False,
            }

    def _gemma4_require_local_chain(self) -> Gemma4PipelineMultiSidecar:
        chain = self._gemma4_local_chain
        if chain is None:
            raise Gemma4MultiSidecarError(
                "gemma4_scheduler_chain_missing", "no Gemma 4 local chain is active",
            )
        return chain

    def run_gemma4_local_prefill(
        self, *, input_ref: str, batch_size: int, sequence_length: int,
    ) -> dict:
        with self._gemma4_local_chain_lock:
            chain = self._gemma4_require_local_chain()
            result = chain.prefill(
                input_ref=input_ref,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
            return {"status": "prefilled", "result": result, "state": chain.snapshot}

    def run_gemma4_local_decode(
        self, *, input_ref: str, batch_size: int, sequence_length: int,
    ) -> dict:
        with self._gemma4_local_chain_lock:
            chain = self._gemma4_require_local_chain()
            result = chain.decode(
                input_ref=input_ref,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
            return {"status": "decoded", "result": result, "state": chain.snapshot}

    def release_gemma4_local_sidecar_chain(self) -> dict:
        with self._gemma4_local_chain_lock:
            chain = self._gemma4_require_local_chain()
            try:
                result = chain.release()
                return {"status": "released", "result": result, "state": chain.snapshot}
            finally:
                self._gemma4_local_chain = None
                self._gemma4_local_contract = None

    def abort_gemma4_local_sidecar_chain(self) -> dict:
        with self._gemma4_local_chain_lock:
            chain = self._gemma4_local_chain
            if chain is None:
                return {
                    "status": "idle",
                    "active": False,
                    "production_admitted": False,
                }
            try:
                result = chain.abort()
                return {"status": "aborted", "result": result, "state": chain.snapshot}
            finally:
                self._gemma4_local_chain = None
                self._gemma4_local_contract = None

    def get_gemma4_local_sidecar_status(self) -> dict:
        with self._gemma4_local_chain_lock:
            chain = self._gemma4_local_chain
            return {
                "active": chain is not None,
                "state": chain.snapshot if chain is not None else {"phase": "idle"},
                "runtime_environment": ".venv-gemma4-pipeline",
                "production_admitted": False,
            }

    def begin_qwen3_pipeline_dry_run(
        self, contract: dict, *, timeout_seconds: float = 30.0,
    ) -> dict:
        """Register a Qwen3 prepare transaction without sending or loading.

        This is intentionally separate from ``_pipeline_load_transaction``:
        production Qwen3 scheduling remains fail-closed until loopback and
        real multi-node gates are completed.
        """
        transaction = Qwen3PipelineDryRunTransaction(
            contract, timeout_seconds=timeout_seconds,
        )
        with self._layer_config_lock:
            active = self._qwen3_pipeline_dry_run
            if active and active.phase in {"preparing", "committing"}:
                raise Qwen3PipelineProtocolError(
                    "another Qwen3 pipeline dry-run is active"
                )
            self._qwen3_pipeline_dry_run = transaction
        return {
            "transaction": transaction.snapshot(),
            "outbound": transaction.prepare_messages(),
        }

    @staticmethod
    def _qwen3_cluster_secret() -> str:
        from transport_port import get_cluster_secret

        return get_cluster_secret()

    def _dispatch_qwen3_loopback_messages(
        self, messages: list[dict], *, best_effort: bool = False,
    ) -> list[dict]:
        server = self._tcp_server
        secret = self._qwen3_cluster_secret()
        if not server or not getattr(server, "_running", False) or not secret:
            if best_effort:
                return []
            raise Qwen3LoopbackError(
                "qwen3_loopback_auth_unavailable",
                "authenticated TCP server or cluster secret is unavailable",
            )
        dispatched = []
        for message in messages:
            node_id = str(message.get("node_id", "") or "")
            if not server.is_authenticated_loopback_client(node_id):
                if best_effort:
                    logger.info(
                        "Qwen3 loopback 清理等待节点重连: node=%s", node_id,
                    )
                    continue
                raise Qwen3LoopbackError(
                    "qwen3_loopback_peer_rejected",
                    f"worker {node_id} is not an authenticated loopback peer",
                )
        for message in messages:
            node_id = str(message["node_id"])
            if not server.is_authenticated_loopback_client(node_id):
                continue
            outbound = dict(message)
            outbound["assignment_base_url"] = self._qwen3_loopback_base_url
            signed = sign_loopback_message(
                outbound, peer_node_id=node_id, secret=secret,
            )
            try:
                server.send_qwen3_pipeline_dry_run(node_id, signed)
            except (ConnectionError, OSError):
                if best_effort:
                    logger.info(
                        "Qwen3 loopback 消息等待节点重连: node=%s phase=%s",
                        node_id, message.get("phase", ""),
                    )
                    continue
                raise
            dispatched.append(signed)
        return dispatched

    def begin_qwen3_pipeline_loopback(
        self,
        contract: dict,
        *,
        assignment_base_url: str,
        timeout_seconds: float = 30.0,
    ) -> dict:
        """Dispatch one authenticated, header-only loopback dry-run."""
        base_url = validate_loopback_base_url(assignment_base_url)
        transaction = Qwen3PipelineDryRunTransaction(
            contract,
            timeout_seconds=timeout_seconds,
            network_dispatch=True,
        )
        if any(
            "assignment_probe" not in segment
            for segment in transaction.contract["segments"]
        ):
            raise Qwen3LoopbackError(
                "qwen3_range_contract_invalid",
                "every loopback segment requires an assignment probe",
            )
        with self._layer_config_lock:
            active = self._qwen3_pipeline_dry_run
            if active and active.phase in {
                "preparing", "committing", "ready", "releasing",
            }:
                raise Qwen3PipelineProtocolError(
                    "another Qwen3 pipeline dry-run is active"
                )
            self._qwen3_loopback_base_url = base_url
            self._qwen3_loopback_ack_nonces.clear()
            self._qwen3_pipeline_dry_run = transaction
        try:
            outbound = self._dispatch_qwen3_loopback_messages(
                transaction.prepare_messages(),
            )
        except Qwen3LoopbackError:
            with self._layer_config_lock:
                self._qwen3_pipeline_dry_run = None
                self._qwen3_loopback_base_url = ""
            raise
        except Exception as exc:
            with self._layer_config_lock:
                cleanup = transaction.abort(
                    "qwen3_loopback_dispatch_failed", str(exc),
                )
            self._dispatch_qwen3_loopback_messages(
                cleanup.get("outbound", []), best_effort=True,
            )
            raise
        return {
            "transaction": transaction.snapshot(),
            "outbound": outbound,
        }

    def begin_qwen3_pipeline_sidecar(
        self, contract: dict, *, assignment_base_url: str,
    ) -> dict:
        """Explicitly opt into node-local sidecar materialization.

        The normal QW3.5 entry point remains metadata-only.  This method is
        intentionally separate so production admission cannot silently turn
        on segment loading while the cross-node tensor data plane is absent.
        """
        if contract.get("execution_mode") != "node_local_sidecar":
            raise Qwen3PipelineProtocolError(
                "Qwen3 sidecar entry requires execution_mode=node_local_sidecar"
            )
        return self.begin_qwen3_pipeline_loopback(
            contract, assignment_base_url=assignment_base_url,
        )

    def _qwen3_local_artifact_root(self) -> Path:
        if self._qwen3_local_artifact_root_override:
            root = Path(self._qwen3_local_artifact_root_override)
        else:
            from config import STATE_DIR

            root = Path(STATE_DIR) / "qwen3-local-chain"
        root = root.expanduser().absolute().resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def is_qwen3_authenticated_transfer_peer(self, peer_node_id: str) -> bool:
        """Project the live TCP HMAC registration state into HTTP auth."""
        peer = str(peer_node_id or "")
        if not peer:
            return False
        if self._effective_role() == "master":
            server = getattr(self, "_tcp_server", None)
            return bool(
                server
                and getattr(server, "_running", False)
                and server.is_authenticated_loopback_client(peer)
            )
        client = getattr(self, "_tcp_client", None)
        return bool(
            peer == "master"
            and client
            and getattr(client, "_running", False)
            and getattr(client, "_registered", False)
        )

    def is_qwen3_authenticated_transfer_peer_epoch(
        self, peer_node_id: str, peer_epoch: int,
    ) -> bool:
        """Check the exact TCP registration epoch used to sign a request."""
        peer = str(peer_node_id or "")
        try:
            epoch = int(peer_epoch)
        except (TypeError, ValueError):
            return False
        if epoch < 0:
            return False
        if self._effective_role() == "master":
            server = getattr(self, "_tcp_server", None)
            epoch_checker = getattr(server, "is_authenticated_loopback_peer", None)
            if not callable(epoch_checker):
                return bool(
                    server
                    and getattr(server, "_running", False)
                    and int(epoch) == 0
                    and server.is_authenticated_loopback_client(peer)
                )
            return bool(
                server
                and getattr(server, "_running", False)
                and epoch_checker(peer, epoch)
            )
        client = getattr(self, "_tcp_client", None)
        return bool(
            peer == "master"
            and client
            and getattr(client, "_running", False)
            and getattr(client, "_registered", False)
            and int(getattr(client, "registration_epoch", 0)) == epoch
        )

    def configure_qwen3_artifact_transfer(
        self,
        runtime,
        *,
        handoff_transport=None,
        network_coordinator=None,
        peer_verifier=None,
    ) -> dict:
        """Explicit QW3.10 wiring; it does not admit the production route."""
        from qwen3_pipeline_data_plane import Qwen3ArtifactTransferRuntime
        from qwen3_pipeline_network import Qwen3NetworkTransferCoordinator
        from qwen3_pipeline_peer_auth import Qwen3PeerRequestVerifier

        if not isinstance(runtime, Qwen3ArtifactTransferRuntime):
            raise Qwen3PipelineProtocolError("Qwen3 artifact transfer runtime is invalid")
        if network_coordinator is not None and not isinstance(
            network_coordinator, Qwen3NetworkTransferCoordinator,
        ):
            raise Qwen3PipelineProtocolError("Qwen3 network coordinator is invalid")
        if network_coordinator is not None:
            from qwen3_pipeline_state import (
                load_qwen3_network_ledger,
                save_qwen3_network_ledger,
            )

            network_coordinator.configure_persistent_ledger(
                load=lambda: load_qwen3_network_ledger(network_coordinator.local_node_id),
                save=lambda value: save_qwen3_network_ledger(
                    value, network_coordinator.local_node_id,
                ),
            )
        if peer_verifier is None:
            secret = self._qwen3_cluster_secret()
            if not secret:
                raise Qwen3PipelineProtocolError("Qwen3 cluster secret is unavailable")
            peer_verifier = Qwen3PeerRequestVerifier(
                secret,
                is_authenticated_peer=self.is_qwen3_authenticated_transfer_peer,
                is_authenticated_peer_epoch=self.is_qwen3_authenticated_transfer_peer_epoch,
                require_peer_epoch=True,
            )
        self._qwen3_artifact_transfer_runtime = runtime
        self._qwen3_network_transfer_coordinator = network_coordinator
        self._qwen3_peer_request_verifier = peer_verifier
        self._qwen3_network_handoff_transport = handoff_transport
        return {
            "enabled": True,
            "network_handoffs": handoff_transport is not None,
            "network_control": network_coordinator is not None,
            "production_admitted": False,
            "runtime": runtime.snapshot(),
        }

    @staticmethod
    def _qwen3_local_state_load() -> dict:
        from qwen3_pipeline_state import load_qwen3_local_chain_state

        return load_qwen3_local_chain_state()

    @staticmethod
    def _qwen3_local_state_save(value: dict) -> dict:
        from qwen3_pipeline_state import save_qwen3_local_chain_state

        return save_qwen3_local_chain_state(value)

    def _qwen3_local_state_from_chain(
        self, contract: dict, chain: Qwen3PipelineMultiSidecar, *, parity: dict | None = None,
    ) -> dict:
        contract = self._qwen3_local_contract or contract
        snapshot = chain.snapshot
        return {
            "schema_version": 1,
            "contract_sha256": str(contract.get("contract_sha256", "")),
            "config_id": str(contract.get("config_id", "")),
            "plan_id": str(contract.get("plan_id", "")),
            "generation": int(contract.get("generation", 0) or 0),
            "phase": str(snapshot.get("phase", "idle")),
            "segment_count": int(snapshot.get("segment_count", 0) or 0),
            "cleanup_complete": bool(snapshot.get("cleanup_complete", False)),
            "parity": dict(parity if parity is not None else self._qwen3_local_parity),
        }

    def _qwen3_local_reconcile_locked(self) -> dict:
        persisted = self._qwen3_local_state_load()
        active_phases = {
            "starting", "prepared", "committed", "prefilled", "decoded", "parity_passed",
        }
        if self._qwen3_local_chain is not None:
            return self._qwen3_local_state_from_chain(
                {"contract_sha256": self._qwen3_local_chain.chain_id},
                self._qwen3_local_chain,
                parity=self._qwen3_local_parity or persisted.get("parity", {}),
            )
        if persisted.get("phase") not in active_phases:
            return persisted
        cleanup = cleanup_qwen3_local_artifacts(
            self._qwen3_local_artifact_root(), persisted.get("contract_sha256", ""),
        )
        transport = self._qwen3_network_handoff_transport
        if transport is not None:
            try:
                network_cleanup = transport.cleanup()
            except Exception:
                network_cleanup = {"cleanup_complete": False}
            cleanup["cleanup_complete"] = bool(
                cleanup.get("cleanup_complete")
                and network_cleanup.get("cleanup_complete")
            )
        recovered = {
            **persisted,
            "phase": "recovered_aborted",
            "cleanup_complete": bool(cleanup.get("cleanup_complete")),
            "parity": {},
        }
        return self._qwen3_local_state_save(recovered)

    def begin_qwen3_local_sidecar_chain(self, contract: dict) -> dict:
        """Explicit local-only QW3.8 entry; production pipeline remains untouched."""
        if contract.get("execution_mode") != "node_local_sidecar":
            raise Qwen3PipelineProtocolError(
                "Qwen3 local chain requires execution_mode=node_local_sidecar"
            )
        with self._qwen3_local_chain_lock:
            persisted = self._qwen3_local_reconcile_locked()
            contract_sha = str(contract.get("contract_sha256", "") or "")
            generation = int(contract.get("generation", 0) or 0)
            if self._qwen3_local_chain is not None:
                if self._qwen3_local_chain.chain_id == contract_sha:
                    return {"status": "duplicate", "state": self._qwen3_local_state_from_chain(contract, self._qwen3_local_chain)}
                raise Qwen3PipelineProtocolError("another Qwen3 local chain is active")
            if persisted.get("contract_sha256") == contract_sha and contract_sha:
                raise Qwen3PipelineProtocolError("Qwen3 local chain submission is already fenced")
            if generation <= int(persisted.get("generation", 0) or 0) and persisted.get("contract_sha256"):
                raise Qwen3PipelineProtocolError("Qwen3 local chain generation is stale")
            root = self._qwen3_local_artifact_root()
            starting = self._qwen3_local_state_save({
                "contract_sha256": contract_sha,
                "config_id": contract.get("config_id", ""),
                "plan_id": contract.get("plan_id", ""),
                "generation": generation,
                "phase": "starting",
                "segment_count": len(contract.get("segments", [])),
                "cleanup_complete": False,
            })
            chain: Qwen3PipelineMultiSidecar | None = None
            try:
                chain_options = {
                    "contract": contract,
                    "artifact_root": root,
                    "session_factory": self._qwen3_sidecar_session_from_message,
                }
                if self._qwen3_network_handoff_transport is not None:
                    chain_options["handoff_transport"] = self._qwen3_network_handoff_transport
                chain = self._qwen3_multisidecar_factory().from_contract(**chain_options)
                self._qwen3_local_chain = chain
                self._qwen3_local_contract = dict(contract)
                self._qwen3_local_parity = {}
                chain.prepare()
                self._qwen3_local_state_save(self._qwen3_local_state_from_chain(contract, chain))
                chain.commit()
                state = self._qwen3_local_state_save(self._qwen3_local_state_from_chain(contract, chain))
                return {"status": "started", "state": state}
            except Exception:
                if chain is not None:
                    try:
                        chain.abort()
                    except Exception:
                        pass
                    self._qwen3_local_state_save(self._qwen3_local_state_from_chain(contract, chain))
                else:
                    self._qwen3_local_state_save({**starting, "phase": "aborted", "cleanup_complete": True})
                self._qwen3_local_chain = None
                self._qwen3_local_contract = None
                self._qwen3_local_parity = {}
                raise

    def _qwen3_local_require_chain(self) -> Qwen3PipelineMultiSidecar:
        chain = self._qwen3_local_chain
        if chain is None:
            raise Qwen3PipelineProtocolError("no Qwen3 local chain is active")
        return chain

    def run_qwen3_local_prefill(self, *, input_ref: str, batch_size: int, sequence_length: int) -> dict:
        with self._qwen3_local_chain_lock:
            chain = self._qwen3_local_require_chain()
            try:
                chain.prefill(input_ref=input_ref, batch_size=batch_size, sequence_length=sequence_length)
                state = self._qwen3_local_state_save(self._qwen3_local_state_from_chain(
                    {"contract_sha256": chain.chain_id, "generation": chain.generation}, chain,
                ))
                return {"status": "prefilled", "state": state}
            except Exception:
                chain.abort()
                self._qwen3_local_state_save(self._qwen3_local_state_from_chain(
                    {"contract_sha256": chain.chain_id, "generation": chain.generation}, chain,
                ))
                self._qwen3_local_chain = None
                self._qwen3_local_contract = None
                self._qwen3_local_parity = {}
                raise

    def run_qwen3_local_decode(self, *, input_ref: str, batch_size: int, sequence_length: int) -> dict:
        with self._qwen3_local_chain_lock:
            chain = self._qwen3_local_require_chain()
            try:
                chain.decode(input_ref=input_ref, batch_size=batch_size, sequence_length=sequence_length)
                state = self._qwen3_local_state_save(self._qwen3_local_state_from_chain(
                    {"contract_sha256": chain.chain_id, "generation": chain.generation}, chain,
                ))
                return {"status": "decoded", "state": state}
            except Exception:
                chain.abort()
                self._qwen3_local_state_save(self._qwen3_local_state_from_chain(
                    {"contract_sha256": chain.chain_id, "generation": chain.generation}, chain,
                ))
                self._qwen3_local_chain = None
                self._qwen3_local_contract = None
                self._qwen3_local_parity = {}
                raise

    def verify_qwen3_local_cpu_parity(
        self,
        *,
        reference_prefill: str,
        reference_decode: str,
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ) -> dict:
        from qwen3_pipeline_parity import evaluate_qwen3_cpu_parity

        with self._qwen3_local_chain_lock:
            chain = self._qwen3_local_require_chain()
            if chain.phase != "decoded":
                raise Qwen3PipelineProtocolError("Qwen3 CPU parity requires decoded local chain")
            report = evaluate_qwen3_cpu_parity(
                artifact_root=self._qwen3_local_artifact_root(),
                reference_prefill=reference_prefill,
                candidate_prefill=chain.final_output_ref("prefill"),
                reference_decode=reference_decode,
                candidate_decode=chain.final_output_ref("decode"),
                prefill_artifacts=chain.artifact_refs("prefill"),
                prefill_reports=chain.execution_reports("prefill"),
                decode_artifacts=chain.artifact_refs("decode"),
                decode_reports=chain.execution_reports("decode"),
                segment_count=len(chain.sessions),
                generation=chain.generation,
                rtol=rtol,
                atol=atol,
            )
            if report.get("gate_passed") is not True:
                chain.cancel()
                self._qwen3_local_parity = dict(report)
                state = self._qwen3_local_state_save(self._qwen3_local_state_from_chain(
                    {"contract_sha256": chain.chain_id, "generation": chain.generation}, chain,
                    parity=report,
                ))
                self._qwen3_local_chain = None
                self._qwen3_local_contract = None
                self._qwen3_local_parity = {}
                return {"status": "rejected", "state": state, "parity": report}
            chain.phase = "parity_passed"
            self._qwen3_local_parity = dict(report)
            state = self._qwen3_local_state_save(self._qwen3_local_state_from_chain(
                {"contract_sha256": chain.chain_id, "generation": chain.generation}, chain,
                parity=report,
            ))
            return {"status": "passed", "state": state, "parity": report}

    def release_qwen3_local_sidecar_chain(self) -> dict:
        with self._qwen3_local_chain_lock:
            chain = self._qwen3_local_require_chain()
            try:
                chain.release()
                state = self._qwen3_local_state_save(self._qwen3_local_state_from_chain(
                    {"contract_sha256": chain.chain_id, "generation": chain.generation}, chain,
                ))
                self._qwen3_local_chain = None
                self._qwen3_local_contract = None
                self._qwen3_local_parity = {}
                return {"status": "released", "state": state}
            except Exception:
                chain.abort()
                self._qwen3_local_state_save(self._qwen3_local_state_from_chain(
                    {"contract_sha256": chain.chain_id, "generation": chain.generation}, chain,
                ))
                self._qwen3_local_chain = None
                self._qwen3_local_contract = None
                self._qwen3_local_parity = {}
                raise

    def cancel_qwen3_local_sidecar_chain(self) -> dict:
        with self._qwen3_local_chain_lock:
            chain = self._qwen3_local_chain
            if chain is None:
                with_state = self._qwen3_local_reconcile_locked()
                return {"status": "recovered", "state": with_state}
            chain.cancel()
            state = self._qwen3_local_state_save(self._qwen3_local_state_from_chain(
                {"contract_sha256": chain.chain_id, "generation": chain.generation}, chain,
            ))
            self._qwen3_local_chain = None
            self._qwen3_local_contract = None
            self._qwen3_local_parity = {}
            return {"status": "cancelled", "state": state}

    def get_qwen3_local_chain_status(self) -> dict:
        with self._qwen3_local_chain_lock:
            state = self._qwen3_local_reconcile_locked()
            if self._qwen3_local_chain is not None:
                state = self._qwen3_local_state_from_chain(
                    {"contract_sha256": self._qwen3_local_chain.chain_id, "generation": self._qwen3_local_chain.generation},
                    self._qwen3_local_chain,
                    parity=self._qwen3_local_parity or state.get("parity", {}),
                )
            transport = self._qwen3_network_handoff_transport
            return {
                "state": state,
                "active": self._qwen3_local_chain is not None,
                "production_admitted": False,
                "network_transfer": (
                    transport.snapshot()
                    if transport is not None
                    else {"active": False, "mode": "local"}
                ),
            }

    def _resolve_model_runtime_descriptor(self, profile: str, model_id: str) -> tuple[dict, dict]:
        """Inspect a verified local asset without changing the active model."""
        from local_model_assets import resolve_local_model_asset_metadata
        from pipeline_model_descriptor import inspect_pipeline_model

        metadata = resolve_local_model_asset_metadata(model_id)
        if not metadata:
            raise ValueError("受管本地 Safetensors 资产不存在或清单未通过验证")
        expected_type = {
            "qwen3_sidecar": "qwen3",
            "gemma4_pipeline": "gemma4_unified",
        }.get(str(profile))
        if expected_type is None:
            raise ValueError("model runtime Sidecar profile is unsupported")
        descriptor = inspect_pipeline_model(
            metadata["model_path"], model_id=str(model_id),
        )
        if str(descriptor.get("model_type", "")).lower() != expected_type:
            raise ValueError(
                f"模型架构与 Sidecar profile 不匹配: {descriptor.get('model_type', '')}"
            )
        descriptor["config"] = dict(metadata.get("config") or {})
        descriptor["model_sha256"] = str(metadata["model_sha256"])
        # The capacity solver is a metadata solver, while runtime admission is
        # still kept fail-closed by the profile-specific Sidecar contracts.
        descriptor["pipeline_runtime_supported"] = True
        if profile == "gemma4_pipeline":
            components = dict(descriptor.get("component_weight_bytes") or {})
            for key in ("visual", "multimodal", "mtp"):
                components[key] = 0
            descriptor["component_weight_bytes"] = components
        return metadata, descriptor

    @staticmethod
    def _runtime_contract_plan_projection(plan: dict) -> dict:
        """Keep the persisted/UI plan bounded and path-free."""
        allowed = (
            "schema_version", "status", "admitted", "reason_code", "plan_id",
            "model_id", "model_type", "total_layers", "raw_model_bytes",
            "safety_margin", "candidate_node_count", "excluded_nodes",
            "control_only_nodes", "participating_node_count", "aggregate_only",
            "single_node_full_model_candidates", "assignments",
        )
        return {key: plan[key] for key in allowed if key in plan}

    @staticmethod
    def _load_model_runtime_contract_records() -> list[dict]:
        from local_store import get_local_setting

        value = get_local_setting("model_runtime_contracts_v1", {})
        records = value.get("bindings") if isinstance(value, dict) else None
        return [dict(item) for item in records if isinstance(item, dict)] if isinstance(records, list) else []

    @staticmethod
    def _save_model_runtime_contract_records(records: list[dict]) -> None:
        from local_store import set_local_setting

        set_local_setting(
            "model_runtime_contracts_v1",
            {"schema_version": 1, "bindings": records[-64:]},
        )

    @staticmethod
    def _model_runtime_audit_token(value: object, *, fallback: str = "") -> str:
        token = re.sub(r"[^a-z0-9_.-]+", "_", str(value or "").lower()).strip("._-")
        return token[:96] or fallback

    @staticmethod
    def _model_runtime_contract_id_from_state(state: object) -> str:
        if not isinstance(state, dict):
            return ""
        for key in ("contract_sha256", "chain_id"):
            candidate = str(state.get(key, "") or "").lower()
            if len(candidate) == 64 and all(char in "0123456789abcdef" for char in candidate):
                return candidate
        return ""

    def _model_runtime_audit_event(
        self,
        action: str,
        *,
        state: object = None,
        error: Exception | None = None,
    ) -> dict:
        """Create a bounded, path-free lifecycle evidence record."""
        snapshot = state if isinstance(state, dict) else {}
        event = {
            "at": time.time(),
            "action": self._model_runtime_audit_token(
                action, fallback="runtime_operation",
            ),
        }
        phase = self._model_runtime_audit_token(snapshot.get("phase", ""))
        if phase:
            event["phase"] = phase
        for key in ("generation", "segment_count"):
            try:
                value = int(snapshot.get(key, 0) or 0)
            except (TypeError, ValueError):
                continue
            if value >= 0:
                event[key] = value
        if "cleanup_complete" in snapshot:
            event["cleanup_complete"] = bool(snapshot.get("cleanup_complete"))
        if error is not None:
            event["reason_code"] = self._model_runtime_audit_token(
                getattr(error, "reason_code", "") or error.__class__.__name__,
                fallback="runtime_operation_rejected",
            )
        return event

    @staticmethod
    def _model_runtime_execution_projection(record: dict) -> dict:
        audit = record.get("audit") if isinstance(record.get("audit"), list) else []
        events = [dict(item) for item in audit if isinstance(item, dict)][-8:]
        last_event = dict(events[-1]) if events else {}
        action = str(last_event.get("action", "") or "")
        if action in {"prepare_failed", "release_failed", "cancel_failed"}:
            recovery_action = "retry_prepare"
        elif action in {"released", "cancelled", "recovered"}:
            recovery_action = "prepare"
        else:
            recovery_action = "prepare"
        return {
            "event_count": len(audit),
            "last_event": last_event,
            "recent_events": list(reversed(events)),
            "recovery_action": recovery_action,
        }

    def _append_model_runtime_contract_event(
        self,
        profile: str,
        contract_id: str,
        action: str,
        *,
        state: object = None,
        error: Exception | None = None,
    ) -> dict | None:
        """Append bounded lifecycle evidence to a persisted contract only."""
        requested = str(contract_id or "").strip().lower()
        if not requested:
            return None
        event = self._model_runtime_audit_event(action, state=state, error=error)
        with self._model_runtime_contract_lock:
            records = self._load_model_runtime_contract_records()
            for record in reversed(records):
                summary = record.get("summary") if isinstance(record.get("summary"), dict) else {}
                if (
                    record.get("profile") != profile
                    or str(summary.get("contract_id", "") or "").lower() != requested
                ):
                    continue
                audit = record.get("audit") if isinstance(record.get("audit"), list) else []
                record["audit"] = [
                    dict(item) for item in audit if isinstance(item, dict)
                ][-23:] + [event]
                self._save_model_runtime_contract_records(records)
                return event
        return None

    def _model_runtime_active_contract_id(self, profile: str) -> str:
        if profile == "qwen3_sidecar":
            status = self.get_qwen3_local_chain_status()
        elif profile == "gemma4_pipeline":
            status = self.get_gemma4_local_sidecar_status()
        else:
            return ""
        return self._model_runtime_contract_id_from_state(status.get("state"))

    def get_model_runtime_contracts(
        self, profile: Optional[str] = None, model_id: Optional[str] = None,
    ) -> dict:
        with self._model_runtime_contract_lock:
            records = self._load_model_runtime_contract_records()
        filtered = [
            item for item in records
            if (not profile or item.get("profile") == profile)
            and (not model_id or item.get("model_id") == model_id)
        ]
        return {
            "schema_version": 1,
            "contracts": [
                {
                    **dict(item.get("summary") or item),
                    "execution": self._model_runtime_execution_projection(item),
                }
                for item in reversed(filtered)
            ],
        }

    def get_model_runtime_contract(
        self, contract_id: str, *, profile: Optional[str] = None,
    ) -> dict:
        requested = str(contract_id or "").strip().lower()
        if not requested:
            raise ValueError("model runtime contract_id is required")
        with self._model_runtime_contract_lock:
            for record in self._load_model_runtime_contract_records():
                summary = record.get("summary") or {}
                if str(summary.get("contract_id", "")).lower() == requested:
                    if profile and record.get("profile") != profile:
                        raise ValueError("model runtime contract profile does not match the requested Sidecar")
                    contract = record.get("contract")
                    if isinstance(contract, dict):
                        return dict(contract)
        raise ValueError("model runtime task contract was not found")

    def bind_model_runtime_contract(self, profile: str, model_id: str) -> dict:
        """Create an auditable, path-free contract from MODEL-FLEET capacity."""
        if self._effective_role() != "master":
            raise Qwen3PipelineProtocolError(
                "model runtime contract binding is available only on the master node"
            )
        profile = str(profile or "")
        model_id = str(model_id or "").strip()
        if not model_id:
            raise ValueError("model runtime contract model_id is required")
        metadata, descriptor = self._resolve_model_runtime_descriptor(profile, model_id)
        plan = self.get_pipeline_capacity_plan(descriptor=descriptor)
        if not plan.get("admitted"):
            raise ValueError(
                f"MODEL-FLEET capacity plan is not admitted: {plan.get('reason_code', 'unknown')}"
            )
        assignments = list(plan.get("assignments") or [])
        records = self._load_model_runtime_contract_records()
        same_model = [
            item for item in records
            if item.get("profile") == profile and item.get("model_id") == model_id
        ]
        for record in reversed(same_model):
            summary = record.get("summary") or {}
            if summary.get("plan_id") == plan.get("plan_id"):
                return {
                    **dict(summary),
                    "status": "already_bound",
                    "plan": self._runtime_contract_plan_projection(plan),
                }
        generation = max(
            [int((item.get("summary") or {}).get("generation", 0) or 0) for item in same_model]
            or [0]
        ) + 1
        model_token = "".join(
            char if char.isalnum() or char in "._-" else "-"
            for char in model_id
        ).strip(".-") or "model"
        config_id = f"model-runtime-{profile}-{model_token}"
        from model_runtime_contracts import (
            build_model_runtime_contract,
            contract_summary,
            validate_model_runtime_contract,
        )

        contract = build_model_runtime_contract(
            profile,
            config_id=config_id,
            plan_id=str(plan.get("plan_id", "")),
            generation=generation,
            model_id=model_id,
            model_sha256=str(metadata["model_sha256"]),
            descriptor=descriptor,
            assignments=assignments,
        )
        contract = validate_model_runtime_contract(profile, contract)
        summary = contract_summary(profile, contract)
        summary["bound_at"] = time.time()
        record = {
            "profile": profile,
            "model_id": model_id,
            "summary": summary,
            "contract": contract,
            "plan": self._runtime_contract_plan_projection(plan),
            "audit": [self._model_runtime_audit_event(
                "bound",
                state={
                    "phase": "bound",
                    "generation": generation,
                    "segment_count": len(contract.get("segments", [])),
                    "cleanup_complete": False,
                },
            )],
        }
        self._save_model_runtime_contract_records(records + [record])
        return {
            "status": "bound",
            **summary,
            "plan": self._runtime_contract_plan_projection(plan),
        }

    def get_model_runtime_sidecar_status(self) -> dict:
        """Project experimental Sidecar control state without exposing paths.

        Sidecar execution is master-owned and contract-bound.  This summary is
        intentionally metadata-only so a UI can distinguish an unavailable
        control plane from an idle-but-capable one without treating discovery
        of a model directory as permission to load it.
        """
        is_master = self._effective_role() == "master"
        profiles: dict[str, dict] = {
            "qwen3_sidecar": {
                "display_name": "Qwen3 Sidecar",
                "runtime_environment": ".venv-qwen3-sidecar",
                "preflight_supported": True,
                "requires_task_contract": True,
                "production_admitted": False,
                "supported_actions": ["status", "begin", "release", "cancel"],
            },
            "gemma4_pipeline": {
                "display_name": "Gemma 4 Pipeline Sidecar",
                "runtime_environment": ".venv-gemma4-pipeline",
                "preflight_supported": False,
                "requires_task_contract": True,
                "production_admitted": False,
                "supported_actions": ["status", "begin", "release", "cancel"],
            },
        }
        if is_master:
            profiles["qwen3_sidecar"]["session"] = self.get_qwen3_local_chain_status()
            profiles["gemma4_pipeline"]["session"] = self.get_gemma4_local_sidecar_status()
            for profile, capability in profiles.items():
                contract_id = self._model_runtime_contract_id_from_state(
                    capability["session"].get("state"),
                )
                if contract_id:
                    capability["session"]["contract_id"] = contract_id
        else:
            unavailable = {
                "active": False,
                "state": {"phase": "unavailable"},
                "reason_code": "master_only",
            }
            profiles["qwen3_sidecar"]["session"] = dict(unavailable)
            profiles["gemma4_pipeline"]["session"] = dict(unavailable)
        return {
            "schema_version": 1,
            "role": self._effective_role(),
            "control_available": is_master,
            "production_admitted": False,
            "profiles": profiles,
        }

    def begin_model_runtime_sidecar(
        self, profile: str, contract: Optional[dict] = None,
        *, contract_id: Optional[str] = None,
    ) -> dict:
        """Start an explicit experimental chain only from a supplied contract."""
        if self._effective_role() != "master":
            raise Qwen3PipelineProtocolError("model runtime Sidecar control is available only on the master node")
        persisted_contract_id = ""
        if contract_id:
            persisted_contract_id = str(contract_id).strip().lower()
            contract = self.get_model_runtime_contract(contract_id, profile=profile)
        if not isinstance(contract, dict) or not contract:
            raise Qwen3PipelineProtocolError("model runtime Sidecar begin requires a task contract")
        try:
            if profile == "qwen3_sidecar":
                result = self.begin_qwen3_local_sidecar_chain(contract)
            elif profile == "gemma4_pipeline":
                result = self.begin_gemma4_local_sidecar_chain(contract)
            else:
                raise Qwen3PipelineProtocolError("model runtime Sidecar profile is unsupported")
        except Exception as exc:
            if persisted_contract_id:
                self._append_model_runtime_contract_event(
                    profile, persisted_contract_id, "prepare_failed", error=exc,
                )
            raise
        if persisted_contract_id:
            self._append_model_runtime_contract_event(
                profile,
                persisted_contract_id,
                "prepare_duplicate" if result.get("status") == "duplicate" else "prepare_succeeded",
                state=result.get("state"),
            )
        return {"profile": profile, **result, "production_admitted": False}

    def release_model_runtime_sidecar(self, profile: str) -> dict:
        if self._effective_role() != "master":
            raise Qwen3PipelineProtocolError("model runtime Sidecar control is available only on the master node")
        active_contract_id = self._model_runtime_active_contract_id(profile)
        try:
            if profile == "qwen3_sidecar":
                result = self.release_qwen3_local_sidecar_chain()
            elif profile == "gemma4_pipeline":
                result = self.release_gemma4_local_sidecar_chain()
            else:
                raise Qwen3PipelineProtocolError("model runtime Sidecar profile is unsupported")
        except Exception as exc:
            if active_contract_id:
                self._append_model_runtime_contract_event(
                    profile, active_contract_id, "release_failed", error=exc,
                )
            raise
        contract_id = self._model_runtime_contract_id_from_state(result.get("state")) or active_contract_id
        if contract_id:
            self._append_model_runtime_contract_event(
                profile, contract_id, "released", state=result.get("state"),
            )
        return {"profile": profile, **result, "production_admitted": False}

    def cancel_model_runtime_sidecar(self, profile: str) -> dict:
        if self._effective_role() != "master":
            raise Qwen3PipelineProtocolError("model runtime Sidecar control is available only on the master node")
        active_contract_id = self._model_runtime_active_contract_id(profile)
        try:
            if profile == "qwen3_sidecar":
                result = self.cancel_qwen3_local_sidecar_chain()
            elif profile == "gemma4_pipeline":
                result = self.abort_gemma4_local_sidecar_chain()
            else:
                raise Qwen3PipelineProtocolError("model runtime Sidecar profile is unsupported")
        except Exception as exc:
            if active_contract_id:
                self._append_model_runtime_contract_event(
                    profile, active_contract_id, "cancel_failed", error=exc,
                )
            raise
        contract_id = self._model_runtime_contract_id_from_state(result.get("state")) or active_contract_id
        if contract_id:
            action = "recovered" if result.get("status") == "recovered" else "cancelled"
            self._append_model_runtime_contract_event(
                profile, contract_id, action, state=result.get("state"),
            )
        return {"profile": profile, **result, "production_admitted": False}

    def _handle_qwen3_loopback_request(
        self, client_id: str, message: dict,
    ) -> None:
        data = message.get("data", {})
        local_node_id = self.get_effective_node_id()
        secret = self._qwen3_cluster_secret()
        worker = None
        contract_sha256 = str(data.get("contract_sha256", "") or "")
        try:
            if client_id != "master":
                raise Qwen3LoopbackError(
                    "qwen3_loopback_peer_mismatch",
                    "worker accepts loopback dry-run only from its master connection",
                )
            with self._layer_config_lock:
                worker = self._qwen3_loopback_workers.get(contract_sha256)
                if worker is None and data.get("phase") in {"prepare", "release"}:
                    worker = Qwen3PipelineLoopbackWorker(
                        node_id=local_node_id,
                        secret=secret,
                        base_url=str(data.get("assignment_base_url", "") or ""),
                        available_bytes=self._qwen3_loopback_available_bytes,
                        sidecar_session_factory=(
                            self._qwen3_sidecar_session_from_message
                            if data.get("execution_mode") == "node_local_sidecar"
                            else None
                        ),
                    )
                    self._qwen3_loopback_workers[contract_sha256] = worker
                elif worker is None:
                    raise Qwen3LoopbackError(
                        "qwen3_loopback_not_prepared",
                        "loopback contract has no prepared worker state",
                    )
            ack = worker.handle(data)
            if data.get("phase") == "release":
                with self._layer_config_lock:
                    self._qwen3_loopback_workers.pop(contract_sha256, None)
        except Qwen3LoopbackError as exc:
            error_ack = {
                "schema_version": 1,
                "operation": "qwen3_pipeline_dry_run_ack",
                "dry_run": True,
                "phase": str(data.get("phase", "")),
                "node_id": local_node_id,
                "config_id": data.get("config_id"),
                "plan_id": data.get("plan_id"),
                "generation": data.get("generation"),
                "contract_sha256": contract_sha256,
                "status": "error",
                "reason_code": exc.reason_code,
                "reason": exc.reason,
                "full_model_materialized": False,
            }
            ack = sign_loopback_message(
                error_ack, peer_node_id=local_node_id, secret=secret,
            )
        client = getattr(self, "_tcp_client", None)
        if client is not None:
            from transport_port import MessageType

            client.send_data(ack, MessageType.QWEN3_PIPELINE_DRY_RUN_ACK)

    def _qwen3_loopback_available_bytes(self, message: dict) -> int:
        if str(message.get("execution_device", "cpu")) == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    free_bytes, _ = torch.cuda.mem_get_info()
                    return max(0, int(free_bytes))
            except Exception:
                return 0
            return 0
        try:
            import psutil

            return max(0, int(psutil.virtual_memory().available))
        except Exception:
            return 0

    def _qwen3_sidecar_session_from_message(
        self, message: dict,
    ) -> Qwen3PipelineSidecarSession:
        model_path = (
            getattr(self._host, "_full_model_path", None)
            or getattr(self._host, "_model_path", None)
        )
        if not model_path:
            raise Qwen3SidecarError(
                "qwen3_sidecar_model_missing",
                "node-local sidecar has no local model assignment path",
            )
        return Qwen3PipelineSidecarSession(
            model_path=model_path,
            model_id=str(message.get("model_id", "") or ""),
            model_sha256=str(message.get("model_sha256", "") or ""),
            config_id=str(message.get("config_id", "") or ""),
            plan_id=str(message.get("plan_id", "") or ""),
            node_id=str(message.get("node_id", "") or ""),
            layer_range=message.get("layer_range", [0, 0]),
            total_layers=int(message.get("total_layers", 0) or 0),
            has_embedding=bool(message.get("has_embedding", False)),
            has_lm_head=bool(message.get("has_lm_head", False)),
            execution_device=str(message.get("execution_device", "cpu") or "cpu"),
            dtype=str(message.get("dtype", "float32") or "float32"),
            generation=int(message.get("generation", 0) or 0),
            assignment_manifest_sha256=str(
                message.get("assignment_manifest_sha256", "") or ""
            ),
        )

    def _handle_qwen3_loopback_ack(
        self, client_id: str, message: dict,
    ) -> None:
        payload = message.get("data", {})
        try:
            nonce, payload_sha256 = verify_loopback_message(
                payload,
                authenticated_peer_id=client_id,
                secret=self._qwen3_cluster_secret(),
            )
            with self._layer_config_lock:
                transaction = self._qwen3_pipeline_dry_run
                if transaction is None or not transaction.network_dispatch:
                    return
                previous = self._qwen3_loopback_ack_nonces.get(nonce)
                if previous is not None and previous != payload_sha256:
                    raise Qwen3LoopbackError(
                        "qwen3_loopback_replay_mismatch",
                        "ACK nonce was reused with a changed payload",
                    )
                self._qwen3_loopback_ack_nonces[nonce] = payload_sha256
                ack = {
                    key: value for key, value in payload.items()
                    if key != "transport_auth"
                }
                if ack.get("phase") == "release":
                    if not transaction.release_ack(client_id, ack):
                        raise Qwen3LoopbackError(
                            "qwen3_loopback_release_mismatch",
                            "release ACK does not match the active contract",
                        )
                    if transaction.phase == "released":
                        self._qwen3_loopback_base_url = ""
                        self._qwen3_loopback_ack_nonces.clear()
                    return
                if ack.get("status") == "error":
                    result = transaction.abort(
                        str(ack.get("reason_code", "qwen3_loopback_worker_error")),
                        str(ack.get("reason", "loopback worker rejected control frame")),
                    )
                else:
                    result = transaction.handle_ack(client_id, ack)
                outbound = result.get("outbound", [])
            if outbound:
                self._dispatch_qwen3_loopback_messages(
                    outbound,
                    best_effort=transaction.phase in {"aborted", "releasing"},
                )
        except (Qwen3LoopbackError, Qwen3PipelineProtocolError) as exc:
            outbound = []
            with self._layer_config_lock:
                transaction = self._qwen3_pipeline_dry_run
                if transaction is not None and transaction.phase in {
                    "preparing", "committing",
                }:
                    code = getattr(
                        exc, "reason_code", "qwen3_loopback_ack_rejected",
                    )
                    result = transaction.abort(code, str(exc))
                    outbound = result.get("outbound", [])
            if outbound:
                self._dispatch_qwen3_loopback_messages(
                    outbound, best_effort=True,
                )

    def release_qwen3_pipeline_loopback(self) -> dict:
        with self._layer_config_lock:
            transaction = self._qwen3_pipeline_dry_run
            if transaction is None or not transaction.network_dispatch:
                raise Qwen3PipelineProtocolError(
                    "no Qwen3 pipeline loopback is active"
                )
            result = transaction.release()
        result["outbound"] = self._dispatch_qwen3_loopback_messages(
            result["outbound"], best_effort=True,
        )
        result["transaction"] = transaction.snapshot()
        return result

    def handle_qwen3_pipeline_dry_run_ack(
        self, node_id: str, payload: dict,
    ) -> dict:
        """Apply one simulated ACK; never touch the TCP or model host."""
        with self._layer_config_lock:
            transaction = self._qwen3_pipeline_dry_run
            if transaction is None:
                raise Qwen3PipelineProtocolError(
                    "no Qwen3 pipeline dry-run is active"
                )
            result = transaction.handle_ack(node_id, payload)
            result["transaction"] = transaction.snapshot()
            return result

    def retry_qwen3_pipeline_dry_run(self) -> dict:
        with self._layer_config_lock:
            transaction = self._qwen3_pipeline_dry_run
            if transaction is None:
                raise Qwen3PipelineProtocolError(
                    "no Qwen3 pipeline dry-run is active"
                )
            result = {
                "transaction": transaction.snapshot(),
                "outbound": transaction.retry_messages(),
            }
        if transaction.network_dispatch:
            result["outbound"] = self._dispatch_qwen3_loopback_messages(
                result["outbound"], best_effort=True,
            )
        return result

    def retry_qwen3_pipeline_loopback_release(self) -> dict:
        with self._layer_config_lock:
            transaction = self._qwen3_pipeline_dry_run
            if transaction is None or not transaction.network_dispatch:
                raise Qwen3PipelineProtocolError(
                    "no Qwen3 pipeline loopback is active"
                )
            messages = transaction.release_messages()
        return {
            "transaction": transaction.snapshot(),
            "outbound": self._dispatch_qwen3_loopback_messages(
                messages, best_effort=True,
            ),
        }

    def expire_qwen3_pipeline_dry_run(
        self, *, now: float | None = None,
    ) -> dict | None:
        outbound = []
        with self._layer_config_lock:
            transaction = self._qwen3_pipeline_dry_run
            if transaction is None:
                return None
            result = transaction.expire(now=now)
            if result is not None:
                result["transaction"] = transaction.snapshot()
                outbound = result.get("outbound", [])
        if result is not None and transaction.network_dispatch:
            result["outbound"] = self._dispatch_qwen3_loopback_messages(
                outbound, best_effort=True,
            )
        return result

    def abort_qwen3_pipeline_dry_run(
        self, reason_code: str, reason: str = "",
    ) -> dict:
        with self._layer_config_lock:
            transaction = self._qwen3_pipeline_dry_run
            if transaction is None:
                raise Qwen3PipelineProtocolError(
                    "no Qwen3 pipeline dry-run is active"
                )
            result = transaction.abort(reason_code, reason)
            result["transaction"] = transaction.snapshot()
        if transaction.network_dispatch:
            result["outbound"] = self._dispatch_qwen3_loopback_messages(
                result.get("outbound", []), best_effort=True,
            )
        return result

    def release_qwen3_pipeline_dry_run_ack(
        self, node_id: str, payload: dict,
    ) -> bool:
        with self._layer_config_lock:
            transaction = self._qwen3_pipeline_dry_run
            if transaction is None:
                return False
            released = transaction.release_ack(node_id, payload)
            if released and transaction.phase == "released":
                self._qwen3_loopback_base_url = ""
                self._qwen3_loopback_ack_nonces.clear()
            return released

    def get_qwen3_pipeline_dry_run_status(self) -> dict:
        with self._layer_config_lock:
            transaction = self._qwen3_pipeline_dry_run
            return (
                transaction.snapshot()
                if transaction is not None
                else {
                    "schema_version": 1,
                    "dry_run": True,
                    "phase": "idle",
                    "network_dispatch": False,
                    "weight_materialization": False,
                    "full_model_fallback": False,
                }
            )
