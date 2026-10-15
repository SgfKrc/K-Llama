"""Header-only tensor selection shared by GGUF supply and pipeline budgets.

All non-block tensors are retained, including names unknown to the runtime.
Only the MTP blocks declared after the decoder are excluded from every relay
segment. No tensor payload, native loader or optional dependency is used.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

_BLOCK_TENSOR = re.compile(r"blk\.(0|[1-9][0-9]*)\.(.+)", re.ASCII)


@dataclass
class GgufTensorSelection:
    # Output-local tensor name -> original source tensor name.
    tensor_sources: dict[str, str]
    layer_weight_bytes: list[int]
    component_weight_bytes: dict[str, int]
    source_weight_bytes: int
    selected_weight_bytes: int
    excluded_mtp_tensor_bytes: dict[str, int]


def select_gguf_tensors(
    tensor_bytes: Mapping[str, int], *, block_count: int,
    nextn_predict_layers: int = 0, start_layer: int = 0,
    end_layer: int | None = None,
) -> GgufTensorSelection:
    """Select a decoder range using exact payload sizes from a GGUF header.

    The full decoder range is valid for metadata inspection only; callers
    publishing distributed artifacts must additionally require proper segments.
    Invalid names/counts/indices fail closed rather than becoming ``other``.
    Payload bytes exclude alignment padding and the GGUF header itself.
    """
    if type(block_count) is not int or block_count <= 0:
        raise ValueError("GGUF block_count must be a positive integer")
    if (type(nextn_predict_layers) is not int
            or not 0 <= nextn_predict_layers < block_count):
        raise ValueError("GGUF nextn_predict_layers must leave a positive decoder count")
    decoder_layers = block_count - nextn_predict_layers
    if end_layer is None:
        end_layer = decoder_layers
    if (type(start_layer) is not int or type(end_layer) is not int
            or not 0 <= start_layer < end_layer <= decoder_layers):
        raise ValueError("GGUF selection requires a valid decoder range")
    if not isinstance(tensor_bytes, Mapping) or not tensor_bytes:
        raise ValueError("GGUF tensor byte inventory must be a nonempty mapping")
    if decoder_layers > len(tensor_bytes):
        raise ValueError("GGUF decoder count exceeds its tensor inventory")

    tensor_sources = {}
    layer_bytes = [0] * decoder_layers
    components = {"embedding": 0, "final_norm": 0, "lm_head": 0, "other": 0}
    source_bytes = 0
    excluded_mtp = {}
    for name, size in tensor_bytes.items():
        if type(name) is not str or not name or type(size) is not int or size <= 0:
            raise ValueError("GGUF tensor names and payload sizes must be valid")
        source_bytes += size
        if name.startswith("blk."):
            match = _BLOCK_TENSOR.fullmatch(name)
            if not match:
                raise ValueError(f"GGUF has a malformed layer tensor name: {name!r}")
            index = int(match[1])
            if index >= block_count:
                raise ValueError(f"GGUF tensor block index exceeds block_count: {name!r}")
            if index >= decoder_layers:
                excluded_mtp[name] = size
                continue
            if not start_layer <= index < end_layer:
                continue
            target = f"blk.{index - start_layer}.{match[2]}"
            layer_bytes[index] += size
        else:
            target = name
            if name.startswith("token_embd."):
                component = "embedding"
            elif name.startswith("output_norm."):
                component = "final_norm"
            elif name.startswith("output."):
                component = "lm_head"
            else:
                component = "other"
            components[component] += size
        if target in tensor_sources:
            raise ValueError("GGUF selection produces duplicate local tensor names")
        tensor_sources[target] = name
    if any(layer_bytes[index] <= 0 for index in range(start_layer, end_layer)):
        raise ValueError("GGUF segment is missing an assigned decoder layer")
    return GgufTensorSelection(
        tensor_sources=tensor_sources, layer_weight_bytes=layer_bytes,
        component_weight_bytes=components, source_weight_bytes=source_bytes,
        selected_weight_bytes=sum(layer_bytes) + sum(components.values()),
        excluded_mtp_tensor_bytes=excluded_mtp,
    )
