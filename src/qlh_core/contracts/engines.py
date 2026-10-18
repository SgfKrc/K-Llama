"""Static engine contracts. No discovery, runtime imports, or model loading.

A family identifies an artifact/engine lineage, never an execution path.
Runtime probes can remove operations from a concrete entry, but cannot add any.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Iterable, Mapping

ENGINE_CAPABILITIES_SCHEMA_VERSION = 1
FULL_REQUEST = "full_request"
LAYER_RANGE = "layer_range"
FORWARD_LAYERS = "forward_layers"
LAYER_RANGE_DYNAMIC = "dynamic"
LAYER_RANGE_PRECUT_ARTIFACT = "precut_artifact"
LAYER_RANGE_NONE = "none"
LLAMA_CPP_BINDING = "llama_cpp_binding"
LLAMA_CPP_KEEP_HEAD = "llama_cpp_keep_head"
LLAMA_CPP_SERVER = "llama_server"
LLAMA_CPP_ANDROID = "llama_cpp_android"
PYTORCH_CPU = "pytorch_cpu"
PYTORCH_CUDA = "pytorch_cuda"
ISLAND_REMOTE = "island_remote"
EXTERNAL_API_REMOTE = "external_api_remote"

COMMON_CAPABILITIES = frozenset({"chat", "chat_stream", "load_model", "unload_model"})


@dataclass(frozen=True)
class EngineSpec:
    family: str
    runtime_kind: str
    artifact_kind: str | None
    layer_range_mode: str
    operations: frozenset[str]
    artifact_formats: tuple[str, ...]
    # Architecture/precision constraints are per operation. An omitted model
    # fact is validated by the artifact/runtime adapter before execution.
    architectures: tuple[tuple[str, tuple[str, ...]], ...]
    precisions: tuple[tuple[str, tuple[str, ...]], ...]
    input_kinds: tuple[str, ...]
    output_kinds: tuple[str, ...]
    native_abi: str
    optimization_constraints: tuple[str, ...]
    features: frozenset[str] = frozenset()

    def supports(self, operation: str, *, architecture: str = "", precision: str = "") -> bool:
        if operation not in self.operations:
            return False
        if architecture and architecture not in dict(self.architectures).get(operation, ()):
            return False
        if precision and precision not in dict(self.precisions).get(operation, ()):
            return False
        return True

    @property
    def capabilities(self) -> frozenset[str]:
        full = COMMON_CAPABILITIES if FULL_REQUEST in self.operations else frozenset()
        layers = frozenset({FORWARD_LAYERS}) if FORWARD_LAYERS in self.operations else frozenset()
        dynamic = (frozenset({"load_layer_range"})
                   if LAYER_RANGE in self.operations and self.layer_range_mode == LAYER_RANGE_DYNAMIC
                   else frozenset())
        return full | layers | dynamic | self.features

    def snapshot(self) -> dict[str, Any]:
        return {"family": self.family, "runtime_kind": self.runtime_kind,
                "operations": sorted(self.operations), "native_abi": self.native_abi}


@dataclass(frozen=True)
class EngineRegistry:
    entries: tuple[EngineSpec, ...]
    # Defaults exist only for legacy table projections and load intent. An
    # actual runtime/planner must supply a concrete runtime_kind.
    legacy_defaults: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        keys = [(entry.family, entry.runtime_kind) for entry in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate engine runtime contract")
        for family, runtime_kind in self.legacy_defaults:
            self.require(family, runtime_kind)

    def get(self, family: str, runtime_kind: str) -> EngineSpec | None:
        return next((entry for entry in self.entries
                     if entry.family == family and entry.runtime_kind == runtime_kind), None)

    def require(self, family: str, runtime_kind: str) -> EngineSpec:
        entry = self.get(family, runtime_kind)
        if entry is None:
            raise ValueError(f"unknown engine runtime: {family}/{runtime_kind}")
        return entry

    def default(self, family: str) -> EngineSpec:
        return self.require(family, dict(self.legacy_defaults)[family])

    def supports(self, family: str, runtime_kind: str, operation: str) -> bool:
        entry = self.get(family, runtime_kind)
        return entry is not None and entry.supports(operation)

    def narrow(self, family: str, runtime_kind: str,
               operations: Iterable[str] | None = None) -> EngineSpec:
        entry = self.require(family, runtime_kind)
        if operations is None:
            return entry
        effective = entry.operations.intersection(operations)
        # Features are unavailable once the corresponding execution operation
        # is removed. This also bounds legacy supports() projections.
        features = entry.features if FULL_REQUEST in effective else frozenset()
        return replace(entry, operations=frozenset(effective), features=features)

    def advertisement(self, entries: Iterable[EngineSpec]) -> dict[str, Any]:
        value = {"schema_version": ENGINE_CAPABILITIES_SCHEMA_VERSION,
                 "runtimes": [entry.snapshot() for entry in entries]}
        self.validate_advertisement(value)
        return value

    def validate_advertisement(self, value: Any) -> tuple[EngineSpec, ...]:
        if not isinstance(value, Mapping) or set(value) != {"schema_version", "runtimes"}:
            raise ValueError("engine_capabilities must contain schema_version and runtimes")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("unsupported engine capability schema version")
        runtimes = value["runtimes"]
        if not isinstance(runtimes, list) or len(runtimes) > len(self.entries):
            raise ValueError("engine runtimes must be a bounded list")
        result = []
        seen = set()
        for item in runtimes:
            if not isinstance(item, Mapping) or set(item) != {"family", "runtime_kind", "operations", "native_abi"}:
                raise ValueError("invalid engine runtime fields")
            family, kind, operations = item["family"], item["runtime_kind"], item["operations"]
            if not isinstance(family, str) or not isinstance(kind, str):
                raise ValueError("engine family and runtime_kind must be strings")
            entry = self.require(family, kind)
            if item["native_abi"] != entry.native_abi:
                raise ValueError("unknown or mismatched engine native ABI")
            if (not isinstance(operations, list)
                    or any(not isinstance(op, str) for op in operations)
                    or len(operations) != len(set(operations))
                    or not set(operations).issubset(entry.operations)):
                raise ValueError("engine runtime operations exceed the static contract")
            if (family, kind) in seen:
                raise ValueError("duplicate engine runtime")
            seen.add((family, kind))
            result.append(self.narrow(family, kind, operations))
        return tuple(result)

    def snapshot(self) -> dict[str, Any]:
        return {"schema_version": ENGINE_CAPABILITIES_SCHEMA_VERSION, "runtimes": [
            {**entry.snapshot(), "artifact_kind": entry.artifact_kind,
             "layer_range_mode": entry.layer_range_mode,
             "capabilities": sorted(entry.capabilities),
             "artifact_formats": list(entry.artifact_formats),
             "architectures": {op: list(values) for op, values in entry.architectures},
             "precisions": {op: list(values) for op, values in entry.precisions},
             "input_kinds": list(entry.input_kinds), "output_kinds": list(entry.output_kinds),
             "optimization_constraints": list(entry.optimization_constraints)}
            for entry in self.entries
        ]}


_LAYER_OPS = frozenset({FULL_REQUEST, LAYER_RANGE, FORWARD_LAYERS})
_FULL_OPS = frozenset({FULL_REQUEST})
_LAYER_ARCHITECTURES = ("qwen", "qwen2", "qwen3_5", "qwen3_5_text")
_FULL_ARCHITECTURES = (*_LAYER_ARCHITECTURES, "llama", "mistral", "gemma", "gemma2", "phi3")


def _spec(family: str, kind: str, artifact: str | None, mode: str,
          operations: frozenset[str], abi: str,
          features: frozenset[str] = frozenset()) -> EngineSpec:
    precisions = ("fp32", "fp16", "int8", "int4") if family == "pytorch" else (
        ("gguf",) if artifact == "gguf" else ("remote",)
    )
    constraints = ["distributed_no_full_model"]
    if mode == LAYER_RANGE_PRECUT_ARTIFACT:
        constraints.append("verified_precut_manifest")
    if kind == LLAMA_CPP_SERVER:
        constraints.append("full_request_only")
    if kind == LLAMA_CPP_KEEP_HEAD:
        constraints.append("keep_head_hidden_extraction")
    if kind == PYTORCH_CPU:
        constraints.append("cpu_fp32_materialization")
    if kind == PYTORCH_CUDA:
        constraints.append("quantized_layers_materialize_fp16")
    return EngineSpec(
        family=family, runtime_kind=kind, artifact_kind=artifact,
        layer_range_mode=mode, operations=operations,
        artifact_formats=(artifact,) if artifact else (),
        architectures=tuple((op, _FULL_ARCHITECTURES if op == FULL_REQUEST else _LAYER_ARCHITECTURES)
                            for op in sorted(operations)),
        precisions=tuple((op, (("fp32",) if kind == PYTORCH_CPU else ("fp32", "fp16"))
                         if family == "pytorch" and op != FULL_REQUEST else precisions)
                         for op in sorted(operations)),
        input_kinds=(("messages",) if FULL_REQUEST in operations else ()) + (
            ("token_ids", "hidden_f32") if FORWARD_LAYERS in operations else ()),
        output_kinds=(("text",) if FULL_REQUEST in operations else ()) + (
            ("hidden_f32", "logits_f32") if FORWARD_LAYERS in operations else ()),
        native_abi=abi, optimization_constraints=tuple(constraints), features=features,
    )


ENGINE_REGISTRY = EngineRegistry(
    entries=(
        _spec("llama_cpp", LLAMA_CPP_BINDING, "gguf", LAYER_RANGE_PRECUT_ARTIFACT,
              _LAYER_OPS, "llama_cpp_python.v1", frozenset({"chat_image"})),
        _spec("llama_cpp", LLAMA_CPP_KEEP_HEAD, "gguf", LAYER_RANGE_PRECUT_ARTIFACT,
              frozenset({LAYER_RANGE, FORWARD_LAYERS}), "qlh_keep_head.v1"),
        _spec("llama_cpp", LLAMA_CPP_SERVER, "gguf", LAYER_RANGE_NONE, _FULL_OPS, "llama_server_http.v1"),
        _spec("llama_cpp", LLAMA_CPP_ANDROID, "gguf", LAYER_RANGE_PRECUT_ARTIFACT,
              _LAYER_OPS, "qlh_llama_jni.v1"),
        _spec("pytorch", PYTORCH_CPU, "safetensors", LAYER_RANGE_DYNAMIC,
              _LAYER_OPS, "torch_module.v1", frozenset({"chat_image", "ensure_full_model"})),
        _spec("pytorch", PYTORCH_CUDA, "safetensors", LAYER_RANGE_DYNAMIC,
              _LAYER_OPS, "torch_module.v1", frozenset({"chat_image", "ensure_full_model"})),
        _spec("island", ISLAND_REMOTE, None, LAYER_RANGE_NONE, _FULL_OPS, "openai_chat.v1"),
        _spec("external_api", EXTERNAL_API_REMOTE, None, LAYER_RANGE_NONE, _FULL_OPS, "openai_chat.v1"),
    ),
    legacy_defaults=(("llama_cpp", LLAMA_CPP_BINDING), ("pytorch", PYTORCH_CPU),
                     ("island", ISLAND_REMOTE)),
)
# Read-only compatibility projections; both legacy tables consume these entries.
LEGACY_ENGINE_SPECS = MappingProxyType({
    family: ENGINE_REGISTRY.require(family, kind)
    for family, kind in ENGINE_REGISTRY.legacy_defaults
})


def advertisement_supports(value: Any, operation: str, *, family: str | None = None) -> bool:
    try:
        entries = ENGINE_REGISTRY.validate_advertisement(value)
    except (ValueError, TypeError):
        return False
    return any(entry.supports(operation) and (family is None or family == entry.family)
               for entry in entries)
