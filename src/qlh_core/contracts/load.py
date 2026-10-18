"""Immutable, content-addressed materialization policy.

These are resolved execution values, never local preferences or ``auto``.
``kv_cache_bytes`` is the KV portion of the node's total reserve_bytes, not
an additional charge. Assignment/artifact identities and runtime receipts
belong to deployment; this contract alone does not upgrade legacy wire peers.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class RuntimeLoadSpec:
    device: str
    dtype: str
    quant_type: str
    context_size: int
    kv_device: str
    kv_dtype: str
    kv_policy: str
    components: tuple[str, ...]
    materialization: str
    kv_cache_bytes: int = 0
    batch_size: int = 1
    sequence_count: int = 1
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported RuntimeLoadSpec schema_version")
        for name in ("device", "kv_device"):
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(r"cpu|mps|cuda:[0-9]+", value):
                raise ValueError(f"{name} must be a resolved device")
        for name in ("dtype", "kv_dtype"):
            if getattr(self, name) not in ("float16", "float32", "bfloat16", "artifact"):
                raise ValueError(f"{name} must be a resolved dtype")
        for name in ("quant_type", "materialization"):
            value = getattr(self, name)
            if (not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", value)
                    or value in ("auto", "unknown", "default")):
                raise ValueError(f"{name} must be an explicit policy")
        if self.kv_policy not in ("dynamic", "preallocated", "disabled"):
            raise ValueError("unsupported kv_policy")
        for name, maximum in (("context_size", 131072), ("batch_size", 131072), ("sequence_count", 131072)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"invalid {name}")
        if self.batch_size > self.context_size or self.sequence_count > self.context_size:
            raise ValueError("batch_size and sequence_count must fit context_size")
        if type(self.kv_cache_bytes) is not int or self.kv_cache_bytes < 0:
            raise ValueError("invalid kv_cache_bytes")
        if self.kv_policy == "disabled" and self.kv_cache_bytes:
            raise ValueError("disabled KV cannot reserve cache bytes")
        components = self.components
        if (not isinstance(components, (list, tuple)) or not components
                or any(not isinstance(x, str) or x not in (
                    "layers", "embedding", "final_norm", "lm_head", "other", "visual", "mtp", "multimodal"
                ) for x in components)
                or len(set(components)) != len(components) or "layers" not in components):
            raise ValueError("invalid components")
        object.__setattr__(self, "components", tuple(sorted(components)))

    def to_dict(self) -> dict[str, Any]:
        value = {field.name: getattr(self, field.name) for field in fields(self)}
        value["components"] = list(self.components)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RuntimeLoadSpec:
        if not isinstance(value, Mapping) or set(value) != {field.name for field in fields(cls)}:
            raise ValueError("invalid RuntimeLoadSpec fields")
        return cls(**dict(value))

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
