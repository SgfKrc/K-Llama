"""Topology: the stage layout, read from TOML before anything spawns.

The prototype hard-coded one GPU stage plus one CPU stage. N stages across N
machines is data, so the layout lives in a file, and every invariant that used to
be a README comment (4-aligned cuts, distinct ports) fails here, before a worker
starts.
"""
from __future__ import annotations

import json

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


class ProvenanceWarning(UserWarning):
    """An artifact's provenance is missing or the coverage is incomplete.

    Kept separate from TopologyError: the topology itself is valid, it just cannot be
    checked. That distinction matters because the failure this reports is quiet -- a
    segment set covering 20 of 24 layers runs to completion and produces fluent garbage --
    while artifacts predating cut.py's sidecar have no provenance at all and must still
    run.
    """


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
    # Optional [lo, hi) source-layer range when `model` is the FULL checkpoint and the
    # layers are selected at load time instead of pre-cut into an artifact. None keeps the
    # original behaviour (model points at a cut segment). See docs/DEPLOYMENT.md.
    layer_range: tuple[int, int] | None = None

    @property
    def is_local(self) -> bool:
        return is_loopback(self.host)


def is_loopback(host: str) -> bool:
    """True when this host runs the stage itself, i.e. the master spawns it."""
    return host in LOCAL_HOSTS


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


def _layer_range(raw: dict, index: int, layers: int) -> tuple[int, int] | None:
    """Optional source-layer range: `layer_range = "8:24"` or `[8, 24]`.

    When present, `model` is the full checkpoint and llama.cpp selects the layers at load
    time (the C scheme) instead of reading a pre-cut artifact. Accepting both spellings
    because TOML has no tuple type and an array reads better in a config file.
    """
    value = raw.get("layer_range")
    if value is None:
        return None
    if isinstance(value, str):
        parts = value.split(":")
        try:
            lo, hi = int(parts[0]), int(parts[1])
        except (IndexError, ValueError):
            raise TopologyError(
                f"stage[{index}].layer_range must look like \"lo:hi\", got {value!r}") from None
    elif isinstance(value, list) and len(value) == 2 and all(type(v) is int for v in value):
        lo, hi = value
    else:
        raise TopologyError(
            f"stage[{index}].layer_range must be \"lo:hi\" or [lo, hi], got {value!r}")
    if not 0 <= lo < hi <= TOTAL_LAYERS:
        raise TopologyError(
            f"stage[{index}].layer_range [{lo}, {hi}) is out of bounds")
    if hi - lo != layers:
        raise TopologyError(
            f"stage[{index}] declares {layers} layers but layer_range [{lo}, {hi}) covers "
            f"{hi - lo}; they must agree or the chain runs on the wrong layers")
    return (lo, hi)


def _stage(index: int, raw: dict) -> Stage:
    if not isinstance(raw, dict):
        raise TopologyError(f"stage[{index}] must be a [[stage]] table, got {raw!r}")
    model = Path(_str(raw, "model", index)).expanduser()
    host = _str(raw, "host", index)
    # Only local artifacts are checked for existence. A remote stage's path refers to
    # that machine's filesystem, so requiring it here would make every cross-host
    # topology invalid on the master -- and the master never opens it anyway.
    if is_loopback(host) and not model.is_file():
        raise TopologyError(f"stage[{index}].model does not point to a file: {model}")
    kind = _str(raw, "kind", index)
    if kind not in KINDS:
        raise TopologyError(f"stage[{index}].kind must be one of {KINDS}, got {kind!r}")
    layers = _int(raw, "layers", index, 1, TOTAL_LAYERS)
    return Stage(
        index, model, host,
        _int(raw, "port", index, 1024, 65535),
        layers,
        kind,
        _int(raw, "gpu_layers", index, 0, default=0),
        _int(raw, "cpu_moe", index, 0, default=0),
        _str(raw, "cpus", index, default="", blank=True),
        _int(raw, "threads", index, 1, 64, 3),
        _layer_range(raw, index, layers),
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
    # Definitely verify artifacts here rather than exposing a flag: this check existed for
    # months without a single caller, which is exactly how a 20-of-24-layer segment set ran
    # to completion producing garbage. Cheap enough for the local stages.
    for msg in verify_artifacts(stages):
        warnings.warn(msg, ProvenanceWarning, stacklevel=2)
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


def provenance(path: str | Path) -> dict | None:
    """Read the `cut.py` sidecar next to an artifact, or None when there is none.

    The same facts are also stored as qlh.* keys inside the GGUF, which is what makes an
    artifact self-describing on another machine. The sidecar is preferred for validation
    only because reading those keys back costs seconds per file: GGUF put
    tokenizer.ggml.tokens (a 248k-element string array) in the header, so every reader has
    to walk it before reaching anything written later.
    """
    p = Path(path)
    side = p.with_suffix(p.suffix + ".qlh.json")
    try:
        return json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def verify_ranges(stages: tuple[Stage, ...]) -> list[str]:
    """Check that the local stages' source-layer ranges tile the source model.

    Returns warnings rather than raising, because artifacts predating the sidecar cannot
    be checked and refusing to run them would be a regression. What this catches is the
    failure the layer-count check cannot see: `source_layers` was also typed by hand, so a
    topology declaring 20 for a 24-layer source passes every other check while the chain
    quietly runs on five sixths of the model. That produced fluent-looking garbage.
    """
    local = [s for s in stages if s.is_local]
    # Two sources of truth for "which layers is this stage": the topology itself when it
    # declares layer_range (C scheme, whole checkpoint + load-time selection), otherwise
    # the provenance sidecar written by cut.py. The former needs no artifact metadata at
    # all -- the range is right there in the config.
    have: list[tuple] = []
    for s in local:
        if s.layer_range:
            have.append((s, {"source_layer_range": list(s.layer_range)}))
        else:
            p = provenance(s.model)
            if p and "source_layer_range" in p:
                have.append((s, p))
    if not have:
        return ["no artifacts carry provenance (cut.py sidecar) and no stage declares "
                "layer_range; layer coverage is unverified"]

    warnings: list[str] = []
    sources = {p.get("source") for _s, p in have if p.get("source")}
    if len(sources) > 1:
        warnings.append(f"stages come from different source models: {sorted(sources)}")

    spans = []
    for s, p in have:
        lo, hi = p["source_layer_range"]
        if not 0 <= lo < hi:
            warnings.append(f"stage[{s.index}] has an invalid range [{lo}, {hi})")
            continue
        if hi - lo != s.layers:
            warnings.append(f"stage[{s.index}] declares {s.layers} layers but its "
                            f"provenance covers [{lo}, {hi}) = {hi - lo}")
        spans.append((lo, hi, s.index))

    if len(have) < len(local):
        warnings.append(f"{len(local) - len(have)} of {len(local)} local stages have neither "
                        f"layer_range nor provenance; coverage only partially checked")

    spans.sort()
    for (lo1, hi1, i1), (lo2, hi2, i2) in zip(spans, spans[1:]):
        if lo2 < hi1:
            warnings.append(f"stage[{i1}] covers [{lo1}, {hi1}) and stage[{i2}] "
                            f"covers [{lo2}, {hi2}) -- they overlap on "
                            f"[{lo2}, {min(hi1, hi2)})")
        elif lo2 > hi1:
            warnings.append(f"gap between stage[{i1}] ending at {hi1} and stage[{i2}] "
                            f"starting at {lo2}: layers [{hi1}, {lo2}) belong to no stage")

    # Completeness can only be judged when every stage is local: a remote stage keeps its
    # range here, so a local-only view of a chain with a remote tail would always look like
    # it stops early. Overlap is checked either way.
    #
    # The expected total comes from the sidecar's n_layer when there is one, otherwise from
    # the ranges themselves (the C scheme has no artifact metadata to read).
    total = have[0][1].get("n_layer")
    if not isinstance(total, int) and spans and all(s.layer_range for s in local):
        total = max(hi for _lo, hi, _i in spans)
    if isinstance(total, int) and spans and not any(not s.is_local for s in stages):
        if spans[0][0] != 0:
            warnings.append(f"layer 0..{spans[0][0]} belong to no stage")
        if spans[-1][1] != total:
            warnings.append(f"layers [{spans[-1][1]}, {total}) belong to no stage")
    return warnings


def verify_artifacts(stages: tuple[Stage, ...]) -> list[str]:
    """Check each local stage's declared layer count against its provenance, and that the
    stages tile the source model.

    Deliberately reads only the sidecar, never the GGUF: opening a 35B artifact to read its
    header costs ~7 s (the 248k-element tokenizer string array lives there), which would add
    half a minute to every start-up. The cost of that trade is that an artifact with no
    sidecar cannot be verified -- so it gets reported rather than silently trusted.
    """
    extra: list[str] = []
    for s in stages:
        if not s.is_local:
            continue
        p = provenance(s.model)
        if not p:
            # A C-scheme stage points at the full checkpoint, which has no cut.py sidecar
            # by design; its range lives in the topology and is validated while parsing.
            continue
        # Compare against the range width, not kept_block_count: a tail artifact also
        # carries the source model's MTP block, so its block count is one more than the
        # decoder layers the topology declares.
        span = p.get("source_layer_range")
        if isinstance(span, list) and len(span) == 2 and span[1] - span[0] != s.layers:
            raise TopologyError(
                f"stage[{s.index}] declares {s.layers} layers but {s.model.name} "
                f"was cut for [{span[0]}, {span[1]}) = {span[1] - span[0]}")
        size = p.get("artifact_size")
        if isinstance(size, int) and size != s.model.stat().st_size:
            extra.append(
                f"stage[{s.index}] {s.model.name} is {s.model.stat().st_size} bytes but "
                f"its sidecar describes {size}; the artifact was replaced, re-cut it")
    return verify_ranges(stages) + extra
    return verify_ranges(stages)
