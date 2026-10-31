"""Configuration: the topology file, and the one runtime knob that is not topology.

The layout of N stages across N machines is data, so it lives in topology.toml and
this module only says where that file is. `ctx` stays an environment variable on
purpose: it describes this process's KV budget, not the shape of the chain, and the
same topology is expected to run with different context budgets on different
hardware (a 16 GB edge box cannot afford what this laptop can).

Everything that used to be configured here -- two model paths, the cut point, two
addresses -- is now a `[[stage]]` table. Nothing reads the old SA_* variables; they
are gone rather than left as knobs that no longer do anything.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_TOPOLOGY = "topology.toml"
TOPOLOGY_ENV = "SA_TOPOLOGY"
CTX_ENV = "SA_CTX"
DEFAULT_CTX = 4096


class ConfigError(ValueError):
    pass


def topology_path() -> Path:
    """The topology file to run. Relative paths resolve against the CWD."""
    raw = os.environ.get(TOPOLOGY_ENV, "").strip() or DEFAULT_TOPOLOGY
    p = Path(raw).expanduser()
    if not p.is_file():
        raise ConfigError(
            f"{TOPOLOGY_ENV} does not point to a file: {p} "
            f"(set {TOPOLOGY_ENV} or place topology.toml next to the scripts)")
    return p


def ctx() -> int:
    raw = os.environ.get(CTX_ENV, "").strip()
    if not raw:
        return DEFAULT_CTX
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{CTX_ENV} must be an integer, got {raw!r}") from exc
    if not 512 <= value <= 262144:
        raise ConfigError(f"{CTX_ENV} must be in [512, 262144], got {value}")
    return value


def load():
    """Parse and validate the topology. Imported lazily to keep cycles out."""
    import topology

    return topology.load(topology_path())


#: Which roles need the output projection. Only the tail turns hidden into ids; every
#: other stage otherwise makes the shim run `output.weight` over the whole 248,320-row
#: vocabulary and throw the result away. The per-step cost of that is measured in the
#: stage-timing breakdown; it is not repeated here, because two callers once carried
#: this comment with two different numbers (6.4 ms and 3.6 ms) for the same measurement.
LOGITS_ROLES = frozenset({"tail"})


def needs_logits(role: str) -> str:
    """The ``SA_NEEDS_LOGITS`` value for one stage role.

    Both callers -- the in-process `engine.StageEngine` and the subprocess
    `chain_client.spawn_worker` -- used to spell this rule out separately. One of them
    was going to drift, and the duplicated justification already had.
    """
    return "1" if role in LOGITS_ROLES else "0"
