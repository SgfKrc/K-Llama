"""Test-only Qwen3 network node process.

The process deliberately exposes only the authenticated loopback control/data
routers. It is not a production service entry point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time

import uvicorn
from fastapi import FastAPI


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from atomic_json import atomic_write_json  # noqa: E402
from qwen3_pipeline_control import router as control_router  # noqa: E402
from qwen3_pipeline_data_plane import Qwen3ArtifactTransferRuntime, router as transfer_router  # noqa: E402
from qwen3_pipeline_network import Qwen3NetworkTransferCoordinator  # noqa: E402
from qwen3_pipeline_peer_auth import Qwen3PeerAuthMiddleware, Qwen3PeerRequestVerifier  # noqa: E402


def _read_ledger_text(path: Path) -> str:
    """Read the ledger while a scanner holds a transient exclusive handle.

    与 save 侧同因（360 主动防御 / SearchIndexer 会短暂独占刚写出的文件）：
    节点重启时要先读旧 ledger，读失败会让整个测试节点启动失败，表现为
    `_spawn_network_node` 抛出 "exited with ..." 这一类不固定失败。这里只
    对 win32 的 `PermissionError` 做 2 秒有界等待，文件缺失或 JSON 损坏仍然
    立即失败。
    """
    deadline = time.monotonic() + 2.0
    while True:
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError:
            if sys.platform != "win32" or time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


class _RegistrationProjection:
    """Small test projection of TCP registration/disconnect events."""

    def __init__(self, peers: list[str]) -> None:
        self._lock = threading.RLock()
        self._epochs = {str(peer): 0 for peer in peers if str(peer)}

    def register(self, peer: str) -> int:
        with self._lock:
            epoch = int(self._epochs.get(str(peer), 0)) + 1
            self._epochs[str(peer)] = epoch
            return epoch

    def disconnect(self, peer: str) -> None:
        with self._lock:
            self._epochs.pop(str(peer), None)

    def authenticated(self, peer: str) -> bool:
        with self._lock:
            return str(peer) in self._epochs

    def authenticated_epoch(self, peer: str, epoch: int) -> bool:
        with self._lock:
            return self._epochs.get(str(peer)) == int(epoch)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._epochs)


class _SyntheticNetworkSidecarExecutor:
    """Test-only target executor used by the QW3.15 process-chain fixture."""

    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root
        self.outputs: list[Path] = []

    def __call__(self, input_path: Path, request: dict) -> dict:
        incoming = input_path.read_bytes()
        segment_index = int(request["segment_index"])
        output = self.artifact_root / (
            f"qwen3-consume-{request['transfer_id']}-{request['phase']}-"
            f"{request['generation']}-{segment_index}.pt"
        )
        output.write_bytes(
            f"node-{segment_index}:{request['phase']}:".encode("ascii") + incoming,
        )
        self.outputs.append(output)
        return {
            "status": "executed",
            "gate_passed": True,
            "output_path": str(output),
            "execution": {
                "full_model_materialized": False,
                "segment_materialized": True,
            },
            "hidden_handoff": {
                "dtype": request["dtype"],
                "device": request["device"],
                "shape": [request["batch_size"], request["sequence_length"], 4],
            },
            "kv_contract": {
                "present": True,
                "shape": [request["batch_size"], request["sequence_length"]],
            },
        }

    def cleanup(self, _request: dict, _reason_code: str) -> None:
        for output in self.outputs:
            output.unlink(missing_ok=True)
        self.outputs.clear()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--secret", required=True)
    parser.add_argument("--allowed-peer", action="append", default=[])
    parser.add_argument("--synthetic-sidecar", action="store_true")
    # QW3.17：真实 sidecar executor 接线（真实 torch artifact 的 CPU 链）
    parser.add_argument("--sidecar-model-path", default="")
    parser.add_argument("--sidecar-python", default="")
    parser.add_argument("--sidecar-layer-range", nargs=2, type=int, default=None)
    parser.add_argument("--sidecar-has-embedding", action="store_true")
    parser.add_argument("--sidecar-has-lm-head", action="store_true")
    parser.add_argument("--sidecar-total-layers", type=int, default=36)
    parser.add_argument("--sidecar-generation", type=int, default=0)
    parser.add_argument("--mm1-manifest", default="")
    parser.add_argument("--mm1-visual-tokens", type=int, default=0)
    args = parser.parse_args()

    runtime = Qwen3ArtifactTransferRuntime.create(
        state_dir=args.state_dir,
        cluster_secret=args.secret,
    )
    coordinator = Qwen3NetworkTransferCoordinator(
        local_node_id=args.node_id,
        runtime=runtime,
    )
    ledger_path = Path(args.state_dir) / "qwen3-network-ledger.json"

    def load_ledger() -> dict:
        if not ledger_path.is_file():
            return {
                "local_node_id": args.node_id,
                "last_generation": -1,
                "active_contract": {},
                "transfers": {},
                "outputs": {},
            }
        return json.loads(_read_ledger_text(ledger_path))

    def save_ledger(value: dict) -> dict:
        # 为什么不用 `Path.with_suffix(".tmp") + replace`：
        # 本机装有 360 主动防御（360sd/360tray）与 Windows SearchIndexer，它们会在
        # `os.replace` 落盘的瞬间以**不带 FILE_SHARE_DELETE** 的方式打开刚写出的
        # ledger（扫描/索引），于是 Windows 直接拒绝替换，抛
        # `PermissionError: [WinError 5]`。这是本项目早已踩过的坑：
        # `src/atomic_json.py` 的存在就是为了它（唯一临时名 + WinError 5/32/33
        # 退避重试）。实测（2026-10-26，这台机器，修复前）：`pytest -n 8` 共 16
        # 次运行里 3 次失败、4 个用例（每次失败的用例不固定：three_process /
        # target_chain[2] / ttl_epoch）；失败瞬间的诊断证明 save **没有并发**
        # （coordinator 的 RLock 正常，active=1、total=26），占用者完全在进程外
        # ——所以靠加锁、换目录或换临时名都治不好，只能像 atomic_json 那样面对
        # 瞬时占用做有界重试，并把临时文件改成唯一名，避免固定 .tmp 在两路保存
        # 之间互相覆盖。
        # 该失败会一路冒泡成 "network ledger could not be persisted"
        # （qwen3_pipeline_network.py:291 → control:529 / multisidecar:649），
        # 出现在三个不固定用例上，因此必须在这里修，而不是在测试里重试。
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(ledger_path, value, sort_keys=True)
        return value

    coordinator.configure_persistent_ledger(load=load_ledger, save=save_ledger)
    registry = _RegistrationProjection(args.allowed_peer)
    verifier = Qwen3PeerRequestVerifier(
        args.secret,
        is_authenticated_peer=registry.authenticated,
        is_authenticated_peer_epoch=registry.authenticated_epoch,
        require_peer_epoch=False,
    )
    app = FastAPI()
    if args.sidecar_model_path:
        # QW3.17：真实 sidecar executor——进程内构造隔离 sidecar session
        # 并接到网络 consume 边界；layer_range 缺省 [0,1]+embedding。
        from qwen3_pipeline_sidecar import (  # noqa: E402
            Qwen3NetworkSidecarExecutor,
            Qwen3PipelineSidecarSession,
        )
        layer_range = args.sidecar_layer_range or [0, 1]
        session = Qwen3PipelineSidecarSession(
            model_path=args.sidecar_model_path,
            model_id="qwen3-4b",
            model_sha256=("2c54d5a09e7e92d4f5126b92a5a457448c9593e6" + "0" * 24),
            config_id="cfg-qw3-chain",
            plan_id="plan-qw3-chain",
            node_id=args.node_id,
            layer_range=layer_range,
            total_layers=args.sidecar_total_layers,
            has_embedding=args.sidecar_has_embedding,
            has_lm_head=args.sidecar_has_lm_head,
            execution_device="cpu",
            dtype="float32",
            generation=args.sidecar_generation,
            sidecar_python=Path(args.sidecar_python) if args.sidecar_python else None,
        )
        session.prepare()
        session.commit()
        sidecar_executor = Qwen3NetworkSidecarExecutor(
            session, artifact_root=runtime.receiver.root,
        )
        if args.mm1_manifest:
            from qwen3_multimodal_runtime import Qwen3MultimodalSidecarAdapter  # noqa: E402
            manifest = json.loads(
                Path(args.mm1_manifest).read_text(encoding="utf-8"),
            )
            sidecar_executor = Qwen3MultimodalSidecarAdapter(
                sidecar_executor,
                manifest=manifest,
                visual_tokens=(args.mm1_visual_tokens or None),
            )
        app.state.qwen3_network_sidecar_executor = sidecar_executor
    else:
        app.state.qwen3_network_sidecar_executor = (
            _SyntheticNetworkSidecarExecutor(runtime.receiver.root)
            if args.synthetic_sidecar else None
        )
    app.state.qwen3_artifact_transfer = runtime
    app.state.qwen3_network_transfer_coordinator = coordinator
    app.add_middleware(Qwen3PeerAuthMiddleware, verifier=verifier)
    app.include_router(transfer_router)
    app.include_router(control_router)

    @app.post("/__fixture/registration")
    async def project_registration(payload: dict):
        peer = str(payload.get("peer", ""))
        action = str(payload.get("action", "register"))
        if action == "disconnect":
            registry.disconnect(peer)
        else:
            registry.register(peer)
        return {"peers": registry.snapshot()}
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        log_level="error",
        access_log=False,
    )


if __name__ == "__main__":
    main()
