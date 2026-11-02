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

#: There is deliberately no default topology. `topology.toml` used to sit here; it was the
#: two-process split, which only ever reached 61% of one process, and it was deleted with the
#: scheme. A default that names a file nobody has is not a convenience -- it turns "you forgot
#: --topology" into "the program cannot find a file it never told you about".
DEFAULT_TOPOLOGY = ""
TOPOLOGY_ENV = "SA_TOPOLOGY"
CTX_ENV = "SA_CTX"
DEFAULT_CTX = 4096
SHIM_ENV = "SA_SHIM"
#: Where a checkout-local shim build lands. `shim/bin/` is gitignored: it is build output.
SHIM_RELATIVE = Path("shim") / "bin" / "qlh_keep_head.dll"


class ConfigError(ValueError):
    pass


def topology_path() -> Path:
    """The topology file to run. Relative paths resolve against the CWD."""
    raw = os.environ.get(TOPOLOGY_ENV, "").strip() or DEFAULT_TOPOLOGY
    if not raw:
        raise ConfigError(
            f"no topology selected: pass --topology or set {TOPOLOGY_ENV}. "
            f"There is no default; topology-example.toml shows the file format")
    p = Path(raw).expanduser()
    if not p.is_file():
        raise ConfigError(
            f"{TOPOLOGY_ENV} does not point to a file: {p} (set {TOPOLOGY_ENV} or --topology)")
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


def find_shim(explicit: str | Path | None = None) -> Path:
    """The keep-head DLL to load: ``--shim`` > ``SA_SHIM`` > this checkout's build.

    One rule for all three entry points (``chain_client``, ``koakuma``, ``server``).
    They used to carry three defaults: the CLI had a hardcoded exp-c path and never
    read ``SA_SHIM``, so ``SA_SHIM=<other> python chain_client.py`` loaded a different
    DLL than the TUI on the same box without saying so -- and every measurement in
    this repo is run with ``SA_SHIM`` set, which makes that a measurement bug.

    No fallback to an absolute path that merely exists. The 2026-10-30 build under
    ``H:\\slimarc-proto\\shim\\bin`` still sits on disk and exports 19 of the 34
    symbols in ``shim/qlh_keep_head.def``. It does have ``tokenize`` and ``set_sampling``
    (measured 2026-11-02 -- an earlier version of this note said it did not); what it
    lacks is the newer surface: ``last_hidden``, ``logits``, ``layer_inp``, ``seq_rm``
    and the whole ``mtp_*`` / ``draft_*`` families. So "it happens to be there" is how a
    chain boots into a shim that cannot draft, cannot read hidden states, and cannot roll
    a rejected draft back -- failures that show up deep in the decode loop, not at load.

    An explicit path must exist: a typo'd ``--shim`` that silently falls back is the
    same class of bug as the hardcoded default was.
    """
    if explicit:
        p = Path(str(explicit)).expanduser()
        if not p.is_file():
            raise ConfigError(f"--shim does not point to a file: {p}")
        return p
    env = os.environ.get(SHIM_ENV, "").strip()
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise ConfigError(f"{SHIM_ENV} does not point to a file: {p}")
        return p
    built = Path(__file__).resolve().parent / SHIM_RELATIVE
    if built.is_file():
        return built
    raise ConfigError(
        f"no keep-head shim: neither {SHIM_ENV} nor --shim is set, and this checkout "
        f"has no build at {built} (build one from shim/, or point {SHIM_ENV} at the "
        f"current DLL)")


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


def is_greedy(sampling: dict | None) -> bool:
    """Is this round greedy, i.e. can it speculate?

    Speculative decode takes argmax over the verifier's logits, so it only makes sense when
    nothing is being sampled. Three entry points used to answer this question three
    different ways: the TUI read ``temperature <= 0``, the server read
    ``not sampling or temperature == 0``, and the CLI read its own ``--greedy`` flag while
    deciding whether to speculate purely from ``--spec-k``. That last one meant
    ``--spec-k 2 --temperature 0.7`` ran speculation against a live sampler and printed a
    tok/s number that meant nothing, with no warning.
    """
    if not sampling:
        return True
    try:
        return float(sampling.get("temperature", 1.0)) <= 0.0
    except (TypeError, ValueError):
        return False


def needs_logits(role: str) -> str:
    """The ``SA_NEEDS_LOGITS`` value for one stage role.

    Both callers -- the in-process `engine.StageEngine` and the subprocess
    `chain_client.spawn_worker` -- used to spell this rule out separately. One of them
    was going to drift, and the duplicated justification already had.
    """
    return "1" if role in LOGITS_ROLES else "0"
