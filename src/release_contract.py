"""Canonical version and product profile for release artifacts.

Source checkouts keep their override-friendly development defaults. Frozen
applications and launcher children carrying ``QLH_RELEASE_PROFILE_ENFORCE=1``
apply the product profile before :mod:`config` reads feature switches.
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, MutableMapping


CONTRACT_FILENAME = "release-contract.json"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_NODE_ROLES = {"master", "client", "auto"}


class ReleaseContractError(RuntimeError):
    """The release contract is missing or violates its schema."""


def _candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    override = os.environ.get("QLH_RELEASE_CONTRACT_PATH", "").strip()
    if override:
        candidates.append(Path(override).expanduser())
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / CONTRACT_FILENAME)
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / CONTRACT_FILENAME)
    candidates.extend(
        (
            Path(__file__).resolve().parent.parent / CONTRACT_FILENAME,
            Path.cwd() / CONTRACT_FILENAME,
        )
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate.absolute()
        key = str(resolved).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def _validate_contract(data: Any, *, source: Path) -> dict[str, Any]:
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ReleaseContractError(f"unsupported release contract schema: {source}")
    version = data.get("version")
    profile = data.get("profile")
    if not isinstance(version, dict) or not isinstance(profile, dict):
        raise ReleaseContractError(f"release contract sections are missing: {source}")
    for key in ("product", "launcher", "inference_service_contract"):
        if not str(version.get(key, "")).strip():
            raise ReleaseContractError(f"release version {key!r} is missing: {source}")
    android_code = version.get("android_code")
    if not isinstance(android_code, int) or android_code <= 0:
        raise ReleaseContractError(f"invalid Android version code: {source}")

    profile_name = str(profile.get("name", "")).strip()
    default_role = str(profile.get("default_node_role", "")).strip().lower()
    fixed = profile.get("fixed_environment")
    required = profile.get("distributed_required")
    if not profile_name or default_role not in _NODE_ROLES or not isinstance(fixed, dict):
        raise ReleaseContractError(f"invalid release profile: {source}")
    required_keys = {
        "QLH_ROUTE_A_STAGE_OFFER",
        "QLH_TASK_WORKER_EXPERIMENTAL_ENABLED",
        "QLH_TASK_GRAPH_ENABLED",
        "QLH_RELAY_ENABLED",
        "QLH_RELAY_PROBE_ONLY",
    }
    if set(fixed) != required_keys or any(
        str(value) not in {"0", "1"} for value in fixed.values()
    ):
        raise ReleaseContractError(f"invalid fixed release environment: {source}")
    allowed = required.get("allowed_execution_modes") if isinstance(required, dict) else None
    if (
        not isinstance(allowed, list)
        or not allowed
        or any(not isinstance(value, str) or not value.strip() for value in allowed)
        or required.get("failure_mode") != "named"
    ):
        raise ReleaseContractError(f"invalid distributed_required contract: {source}")
    return data


@lru_cache(maxsize=1)
def load_release_contract() -> dict[str, Any]:
    errors: list[str] = []
    for path in _candidate_paths():
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return _validate_contract(payload, source=path)
        except (OSError, json.JSONDecodeError, ReleaseContractError) as exc:
            errors.append(f"{path}: {exc}")
    detail = "; ".join(errors) if errors else ", ".join(
        str(path) for path in _candidate_paths()
    )
    raise ReleaseContractError(f"release contract unavailable: {detail}")


def release_contract_path() -> Path:
    for path in _candidate_paths():
        if path.is_file():
            return path
    raise ReleaseContractError("release contract file is unavailable")


RELEASE_CONTRACT = load_release_contract()
PRODUCT_VERSION = str(RELEASE_CONTRACT["version"]["product"])
LAUNCHER_VERSION = str(RELEASE_CONTRACT["version"]["launcher"])
ANDROID_VERSION_CODE = int(RELEASE_CONTRACT["version"]["android_code"])
INFERENCE_SERVICE_CONTRACT_VERSION = str(
    RELEASE_CONTRACT["version"]["inference_service_contract"]
)
RELEASE_PROFILE_NAME = str(RELEASE_CONTRACT["profile"]["name"])
DEFAULT_PACKAGED_NODE_ROLE = str(
    RELEASE_CONTRACT["profile"]["default_node_role"]
).lower()
FIXED_RELEASE_ENV = {
    str(name): str(value)
    for name, value in RELEASE_CONTRACT["profile"]["fixed_environment"].items()
}
DISTRIBUTED_REQUIRED_EXECUTION_MODES = frozenset(
    str(value)
    for value in RELEASE_CONTRACT["profile"]["distributed_required"][
        "allowed_execution_modes"
    ]
)


def release_profile_enforced(environment: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environment is None else environment
    explicit = str(env.get("QLH_RELEASE_PROFILE_ENFORCE", "")).strip().lower()
    if explicit:
        return explicit in _TRUE_VALUES
    return bool(getattr(sys, "frozen", False))


def apply_release_profile_to_env(
    environment: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    env = os.environ if environment is None else environment
    if not release_profile_enforced(env):
        return {}
    env["QLH_RELEASE_PROFILE"] = RELEASE_PROFILE_NAME
    for name, value in FIXED_RELEASE_ENV.items():
        env[name] = value
    return dict(FIXED_RELEASE_ENV)


def is_release_environment_locked(
    name: str, environment: Mapping[str, str] | None = None,
) -> bool:
    return release_profile_enforced(environment) and name in FIXED_RELEASE_ENV
