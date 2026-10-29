"""Topology: the stage layout, read from TOML before anything spawns.

The prototype hard-coded one GPU stage plus one CPU stage. N stages across N
machines is data, so the layout lives in a file, and every invariant that used to
be a README comment (4-aligned cuts, distinct ports) fails here, before a worker
starts.
"""
from __future__ import annotations

import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path

# The model's block count: the stages must partition it exactly. Read from the
# artifact when possible (see `layers_from_gguf`); this is the fallback for a
# topology that has not been pointed at a real file yet.
TOTAL_LAYERS = 40
# qwen35moe has full_attention_interval=4; a cut that is not a multiple of 4 puts
# the wrong attention type at the seam and the artifact fails to load.
LAYER_ALIGN = 4
KINDS = ("gpu", "cpu", "tail")
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_REQUIRED = object()


class TopologyError(ValueError):
    pass


@dataclass(frozen=True)
class Stage:
    index: int
    model: Path
    host: str
    port: int
    layers: int
    kind: str
    gpu_layers: int
    cpu_moe: int
    cpus: str
    threads: int

    @property
    def is_local(self) -> bool:
        return self.host in LOCAL_HOSTS


def _int(raw: dict, key: str, index: int, lo: int, hi: int | None = None,
         default: object = _REQUIRED) -> int:
    value = raw.get(key, default)
    if value is _REQUIRED:
        raise TopologyError(f"stage[{index}] has no {key!r}")
    # TOML true is a bool and bool subclasses int; a flag is never a layer count.
    if type(value) is not int:
        raise TopologyError(f"stage[{index}].{key} must be an integer, got {value!r}")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        if hi is None:
            raise TopologyError(f"stage[{index}].{key} must be >= {lo}, got {value}")
        raise TopologyError(f"stage[{index}].{key} must be in [{lo}, {hi}], got {value}")
    return value


def _str(raw: dict, key: str, index: int, default: object = _REQUIRED,
         blank: bool = False) -> str:
    value = raw.get(key, default)
    if value is _REQUIRED:
        raise TopologyError(f"stage[{index}] has no {key!r}")
    if not isinstance(value, str):
        raise TopologyError(f"stage[{index}].{key} must be a string, got {value!r}")
    value = value.strip()
    if not value and not blank:
        raise TopologyError(f"stage[{index}].{key} must not be blank")
    return value


def _stage(index: int, raw: dict) -> Stage:
    if not isinstance(raw, dict):
        raise TopologyError(f"stage[{index}] must be a [[stage]] table, got {raw!r}")
    model = Path(_str(raw, "model", index)).expanduser()
    if not model.is_file():
        raise TopologyError(f"stage[{index}].model does not point to a file: {model}")
    kind = _str(raw, "kind", index)
    if kind not in KINDS:
        raise TopologyError(f"stage[{index}].kind must be one of {KINDS}, got {kind!r}")
    return Stage(
        index, model, _str(raw, "host", index),
        _int(raw, "port", index, 1024, 65535),
        _int(raw, "layers", index, 1, TOTAL_LAYERS),
        kind,
        _int(raw, "gpu_layers", index, 0, default=0),
        _int(raw, "cpu_moe", index, 0, default=0),
        _str(raw, "cpus", index, default="", blank=True),
        _int(raw, "threads", index, 1, 64, 3),
    )


def _validate(stages: tuple[Stage, ...], declared_source_layers: int = 0) -> None:
    if len(stages) < 2:
        raise TopologyError(f"a topology needs at least 2 stages, got {len(stages)}")
    kinds = [s.kind for s in stages]
    shape = ["gpu"] + ["cpu"] * (len(stages) - 2) + ["tail"]
    if kinds != shape:
        bad = next((f"stage[{i}] is {k!r} but must be {w!r}"
                    for i, (k, w) in enumerate(zip(kinds, shape)) if k != w), "")
        raise TopologyError(f"kinds must be one 'gpu', then only 'cpu', then one 'tail': {bad} "
                            f"(got {kinds}, expected {shape})")

    for s in stages:
        if s.layers % LAYER_ALIGN:
            raise TopologyError(f"stage[{s.index}].layers={s.layers} is not a multiple of "
                                f"{LAYER_ALIGN}: qwen35moe has full_attention_interval="
                                f"{LAYER_ALIGN}, so this cut misaligns layer types and the "
                                f"artifact fails to load")
        if not 0 <= s.cpu_moe <= s.layers:
            raise TopologyError(
                f"stage[{s.index}].cpu_moe must be in [0, layers={s.layers}], got {s.cpu_moe}")
        if s.kind == "tail" and s.gpu_layers:
            # Offloading on the tail host does not error, it just loses throughput.
            warnings.warn(f"stage[{s.index}] is the tail but gpu_layers={s.gpu_layers}; "
                          f"the tail stage runs entirely on the host", stacklevel=3)

    total = total_layers(stages)
    # A gap is not a lesser mistake than an overlap: the chain would silently skip
    # layers and produce plausible-but-wrong logits, whereas excess layers fail loudly
    # at load. Require an exact cover of the source model's decoder layers.
    #
    # The expected count comes from `source_layers` when the file declares it, because
    # no single stage artifact knows the source total (each reports only what it kept).
    # TOTAL_LAYERS stays as the default so a 35B topology without the key still works.
    expected = declared_source_layers if declared_source_layers else TOTAL_LAYERS
    if total != expected:
        raise TopologyError(
            f"stages sum to {total} layers but the model has {expected}; "
            f"the topology must cover every layer exactly (got "
            f"{[s.layers for s in stages]}). Set `source_layers` at the top of the "
            f"file if the model is not the default {TOTAL_LAYERS}-layer one.")
    ports = [s.port for s in stages]
    dupes = sorted({p for p in ports if ports.count(p) > 1})
    if dupes:
        raise TopologyError(f"ports must be unique, duplicated: {dupes}")


def load(path: str | Path) -> tuple[Stage, ...]:
    p = Path(path).expanduser()
    try:
        with open(p, "rb") as fh:
            doc = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise TopologyError(f"topology file does not exist: {p}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise TopologyError(f"topology file is not valid TOML: {p}: {exc}") from exc
    except OSError as exc:
        raise TopologyError(f"topology file is unreadable: {p}: {exc}") from exc

    raw_stages = doc.get("stage")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise TopologyError(f"topology file needs a non-empty [[stage]] array: {p}")
    # Optional top-level key: how many decoder layers the source model has. Needed
    # because a cut artifact only knows what it kept, so the exact-cover check has no
    # other way to learn the total for a non-default model.
    declared = doc.get("source_layers", 0)
    if declared and (not isinstance(declared, int) or declared <= 0):
        raise TopologyError(f"source_layers must be a positive integer, got {declared!r}")
    stages = tuple(_stage(i, raw) for i, raw in enumerate(raw_stages))
    _validate(stages, declared or 0)
    return stages


def total_layers(stages: tuple[Stage, ...]) -> int:
    return sum(s.layers for s in stages)


def local_stages(stages: tuple[Stage, ...]) -> tuple[Stage, ...]:
    return tuple(s for s in stages if s.is_local)


def sum_by_kind(stages: tuple[Stage, ...]) -> dict[str, int]:
    totals = dict.fromkeys(KINDS, 0)
    for s in stages:
        totals[s.kind] += s.layers
    return totals


def layers_from_gguf(path: str | Path) -> int:
    """Read a stage artifact's own block count.

    A cut artifact reports the layers it kept, not the source model's total, so this
    checks that a stage's declared `layers` matches the file it points at.
    """
    from gguf import GGUFReader

    with open(path, "rb") as fh:
        reader = GGUFReader(fh)
        arch = _arch_of(reader)
        field = reader.get_field(f"{arch}.block_count") if arch else None
    if field is None:
        raise ValueError(f"{path}: no {arch}.block_count")
    return int(field.contents())


def _arch_of(reader) -> str:
    field = reader.get_field("general.architecture")
    return str(field.contents()) if field is not None else ""


def verify_artifacts(stages: tuple[Stage, ...]) -> None:
    """Check每段 declared layer count against the file it points at.

    Catches the failure mode where someone edits the topology and reuses an artifact
    cut for a different range: the chain would still run, on the wrong layers.
    """
    for s in stages:
        actual = layers_from_gguf(s.model)
        if actual != s.layers:
            raise TopologyError(
                f"stage[{s.index}] declares {s.layers} layers but {s.model.name} "
                f"contains {actual}")
