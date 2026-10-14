#!/usr/bin/env python3
"""Linux clean-environment gate for PKG-CONTRACT-01.

This gate intentionally uses only the Python standard library. It verifies the
release contract and clean-user startup decisions in a read-only source mount.
It is a code/configuration gate, not a substitute for installing the final
Windows Setup, Android APK, or signed Debian package.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PACKAGING = ROOT / "packaging" / "packaging"
LINUX_RELEASE_DIR = PACKAGING / "linux"
LINUX_TEXT_PAYLOADS = (
    "bjtu",
    "build-deb.sh",
    "control-cpu",
    "control-cuda",
    "launcher.py",
    "postinst",
    "postrm",
    "prerm",
    "qlh-edge-inference.desktop",
    "qlh-edge-inference.service",
    "qlh-env-register",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    inherited_qlh = sorted(name for name in os.environ if name.startswith("QLH_"))
    require(not inherited_qlh, f"host QLH environment leaked into container: {inherited_qlh}")

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "version_contract.py"), "--check"],
        cwd=ROOT,
        check=True,
    )

    for relative_path in LINUX_TEXT_PAYLOADS:
        payload = (LINUX_RELEASE_DIR / relative_path).read_bytes()
        require(
            b"\r" not in payload,
            f"Linux release payload must use LF line endings: {relative_path}",
        )

    sys.path.insert(0, str(SRC))
    import distributed_completion
    import node_config
    import release_contract

    contract = json.loads(
        (ROOT / "release-contract.json").read_text(encoding="utf-8")
    )
    require(
        release_contract.PRODUCT_VERSION == contract["version"]["product"],
        "Python product version diverges from the canonical contract",
    )

    with tempfile.TemporaryDirectory(prefix="qlh-clean-") as tmp:
        clean_root = Path(tmp)
        config_path = clean_root / "config" / "node_config.json"
        os.environ["QLH_RELEASE_PROFILE_ENFORCE"] = "1"
        os.environ["QLH_RELEASE_CONTRACT_PATH"] = str(ROOT / "release-contract.json")
        os.environ["QLH_NODE_CONFIG_PATH"] = str(config_path)
        release_contract.apply_release_profile_to_env()

        require(node_config.resolve_initial_node_role() == "master", "clean role is not deterministic")
        require(os.environ["QLH_ROUTE_A_STAGE_OFFER"] == "1", "Route-A is not fixed on")
        require(
            os.environ["QLH_TASK_WORKER_EXPERIMENTAL_ENABLED"] == "1",
            "task-worker product path is not fixed on",
        )
        require(os.environ["QLH_TASK_GRAPH_ENABLED"] == "0", "task graph is not fixed off")
        require(os.environ["QLH_RELAY_ENABLED"] == "0", "legacy relay product path is enabled")
        require(os.environ["QLH_RELAY_PROBE_ONLY"] == "1", "relay is not probe-only")

        distributed_completion.validate_distributed_completion(
            "distributed_required",
            {
                "distributed_used": True,
                "execution_mode": "route_a_stage_offer_v3",
                "workers_used": ["worker-clean"],
                "layer_segments": [[16, 24]],
            },
        )
        try:
            distributed_completion.validate_distributed_completion(
                "distributed_required",
                {"distributed_used": False, "execution_mode": "local_pytorch"},
            )
        except distributed_completion.DistributedCompletionError:
            pass
        else:
            raise AssertionError("distributed_required accepted a local completion")

        sys.path.insert(0, str(PACKAGING))
        import packaging_release_contract

        child = packaging_release_contract.build_release_child_environment(
            ROOT,
            {
                "HOME": str(clean_root / "home"),
                "XDG_CONFIG_HOME": str(clean_root / "xdg-config"),
                "XDG_STATE_HOME": str(clean_root / "xdg-state"),
                "QLH_NODE_CONFIG_PATH": str(config_path),
            },
        )
        require(child["QLH_NODE_ROLE"] == "master", "launcher child role diverged")
        require(child["QLH_RELEASE_PROFILE_ENFORCE"] == "1", "child profile is not enforced")
        for name, value in release_contract.FIXED_RELEASE_ENV.items():
            require(child.get(name) == value, f"child environment diverged for {name}")

    summary = {
        "ok": True,
        "gate": "linux-clean-source-contract",
        "product_version": release_contract.PRODUCT_VERSION,
        "launcher_version": release_contract.LAUNCHER_VERSION,
        "profile": release_contract.RELEASE_PROFILE_NAME,
        "default_node_role": release_contract.DEFAULT_PACKAGED_NODE_ROLE,
        "topology_invariants": ["star", "master-coordinated", "D-to-L", "L-to-L"],
        "limitations": [
            "not-a-final-deb-install",
            "not-windows-setup",
            "not-android-release-apk",
            "not-real-network-or-model",
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
