"""Single-tensor benchmark: what does one row of ``lm_head`` actually cost -- and what
does one MoE expert tensor cost at the shape the model really uses?

Two modes, and the ``lm_head`` one is unchanged from before.

**``--mode lm_head``** (default): the cost of ``output.weight`` alone.

Motivation: the project documents disagree about the cost of ``output.weight``
(``docs/SPECULATION.md`` §3 vs ``docs/MOE-SPECULATION-ECONOMICS.md`` §5, which once
claimed 16 rows cost more than the whole 8-layer segment that contains them). The
suspected cause is that earlier measurements advanced ``n_past``, so the attention
cost of a growing KV cache was folded into the per-row figure.

So this measures the matrix multiply **alone**, with no layers, no KV cache and no
positions involved:

    T(M) = t0 + m * M          (linear fit, R^2 reported)

where ``t0`` is the fixed per-call cost (graph compute entry, thread pool wake-up)
and ``m`` is the marginal cost of one more row. ``M`` is the number of hidden rows
fed in -- exactly "how many candidate tokens are verified in one shot".

**The curve is not actually a line, and that is the finding.** For the K-quants
(``Q6_K`` and ``Q4_K``) on the plain CPU buffer, T(M) steps instead of climbing:
8 and 16 rows cost the same, then 17 rows jumps by ~55%. That is ggml-cpu's chunk
split (``ggml-cpu.c``: ``chunk_size = 16``, ``nchunk1 = ceil(ne1/16)``) -- the weight
is streamed once per 16-column band, and rows inside a band ride along nearly for
free. It is also why the old documents could record "16 rows cheaper than 8": both
points sit on the same flat band and only noise separates them. The fit's R^2 says
so, and ``segments_of_median`` in the JSON keeps the steep band and the flat band
apart instead of averaging them into one meaningless slope.

**``--mode experts``**: the same measurement on a *real* MoE expert tensor, because
``docs/MOE-SPECULATION-ECONOMICS.md`` §6.5 says an engine's throughput is set by the
worst combination it serves, and this model's expert tensors are a very bad mixture:

    IQ4_XS  63    <- no repack kernel exists for IQ4_XS at all (repack.cpp has no
                     GGML_TYPE_IQ4_XS branch, and its tensor_traits table has no
                     block_iq4_xs entry)
    Q5_K    30    <- repack.cpp dispatches Q5_K only under NEON
    Q6_K    27    <- same, NEON only
    Q8_0     3

while ``Q4_K`` *does* have an AVX2 8x8 interleaved GEMM (``repack.cpp:5012`` ->
``q4_K_8x8_q8_K``). So the question is whether re-quantising the 63 IQ4_XS expert
tensors to Q4_K turns "no optimised kernel" into "dedicated GEMM" -- and at what
price (4.25 -> 4.5 bpw, i.e. a *smaller* quantisation error, so precision is not the
cost; wall time is the question).

The shapes are read from the GGUF, never assumed: the census prints every
``blk.*.ffn_{gate,up,down}_exps.weight`` with its ggml type and shape, and the
``n_expert`` / ``n_expert_used`` / ``n_embd`` / ``n_ff_exp`` values are taken from
the tensor geometry itself (cross-checked against ``<arch>.expert_*`` metadata).
Nothing is written back to the model file; every re-quantised copy lives in RAM.

Two levels are measured, as in the real graph:

* ``MUL_MAT``: one expert ``[n_embd, n_ff_exp]``, the per-expert multiply that
  ``mul_mat_id`` performs internally. Useful because it isolates the kernel.
* ``MUL_MAT_ID``: the whole expert bank ``[n_embd, n_ff_exp, n_expert]`` with
  ``n_expert_used`` ids per token -- the MoE path, including the gather/scatter and
  per-expert dispatch overhead. Built by hand from ``ggml_mul_mat_id`` with
  ``src1 = [n_embd, 1, n_tokens]`` and ``ids = [n_expert_used, n_tokens]``, which is
  exactly what ``build_moe_ffn`` feeds it (``llama-graph.cpp``: ``cur = ggml_reshape_3d(
  ctx0, cur, n_embd, 1, n_tokens)``), so the broadcast slot is the one the model uses.

**This build has a second kernel path the old numbers never accounted for.** The
checkout on this machine carries a ``tiled`` matmul (``ggml/src/ggml-cpu/tiled/``,
``GGML_CPU_TILED_MM``, on by default) that supports **IQ4_XS as well as Q4_K/Q5_K/
Q6_K**, and engages per expert once the routed row count reaches 8. So "IQ4_XS has no
optimised kernel" is only true for short batches; at M >= 8 the plain-buffer IQ4_XS
weight goes through a real GEMM. The JSON records ``tiled_mm`` and marks every point
with ``tiled_applicable`` so the two regimes are never confused. Run the script a
second time with ``GGML_CPU_TILED_MM=0`` for the pre-tiled comparison.

**How it measures.** Not through the keep-head shim (that would add IPC and layer
cost), and not through ``test-backend-ops`` (no built artifact on this machine, and it
builds its own synthetic tensors). Instead it loads the *real* weights straight out of
the GGUF and builds one graph against the CPU backend, calling the same ``ggml-cpu``
kernels llama.cpp calls.

**Buffer type matters, so both are measured.** llama.cpp *enables* extra buffer types
-- weight repacking -- by default; ``--no-repack`` is the opt-out. Q4_0, Q4_K and
IQ4_NL have AVX2 interleaved kernels; Q5_K/Q6_K only under NEON, and IQ4_XS not at
all, so those can only ever run on a plain buffer. Measuring Q4_0/Q4_K on a plain
buffer too shows what repacking is worth, and measuring only that would have reported
a kernel the program never runs.

For the quantisation comparison, copies are made from the real source weights:
dequantise with ``gguf.quants``, re-quantise with ``ggml_quantize_chunk`` (the C
implementation ``llama-quantize`` itself uses, so the block layout and the rounding
are llama.cpp's, not a Python imitation).

Cross-check (``--verify``): for **every** measured M, the MUL_MAT result is compared
against ``dequantize(weight) @ hidden.T`` in numpy. Agreement means both the ggml call
and the Python decoder are reading the same weights, so the timing really is this
tensor's -- and that the wide M really did compute all its rows, which a flat curve
would otherwise make suspicious. For MUL_MAT_ID the check is the sum of the whole
output block: this build's ``mul_mat_id`` does not lay its result out the way
``ggml.h`` documents (measured: the values are all right but permuted within
``[rows, n_expert_used, n_tokens]``), so a per-element comparison would report a
false failure while a block sum still catches "it skipped an expert".

Usage:
    python checks/bench_lm_head.py --json lm_head.json
    python checks/bench_lm_head.py --types q6_K --ms 1,8,16
    python checks/bench_lm_head.py --buft plain        # what --no-repack would do
    python checks/bench_lm_head.py --mode experts --json experts.json
    GGML_CPU_TILED_MM=0 python checks/bench_lm_head.py --mode experts --json experts-notiled.json
"""
from __future__ import annotations

import argparse
import ctypes
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_GGUF = Path(
    r"H:\qlh_models\qwen3-6-35b-a3b-opus47-distill-apex-mtp")
DEFAULT_MODEL = "Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-I-Quality.gguf"
DEFAULT_BIN = Path(r"H:\llamacpp-c\build-cpu\bin")

# ggml type enum values used here (ggml.h)
GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K = 0, 2, 12, 14
GGML_TYPE_Q5_K, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_I32 = 13, 20, 23, 26
#: lower-case lookup -> (ggml type id, canonical display name)
TYPE_CODES = {"f32": (0, "f32"), "q4_0": (2, "q4_0"), "q8_0": (8, "q8_0"),
              "q4_k": (12, "q4_K"), "q5_k": (13, "q5_K"), "q6_k": (14, "q6_K"),
              "iq4_nl": (20, "iq4_nl"), "iq4_xs": (23, "iq4_xs")}
TENSOR_TYPE_NAMES = {0: "f32", 1: "f16", 2: "q4_0", 8: "q8_0", 12: "q4_K", 13: "q5_K",
                     14: "q6_K", 20: "iq4_nl", 23: "iq4_xs"}


def resolve_type(name: str) -> tuple[int, str] | None:
    return TYPE_CODES.get(name.strip().lower())


#: Types for which this x86 build has an interleaved (repack) kernel, so the weight
#: can live in a CPU_REPACK buffer and the faster GEMM is used. Read off
#: ggml/src/ggml-cpu/repack.cpp's ``ggml_repack_get_optimal_repack_type``: Q4_0 picks
#: q4_0_8x8_q8_0 under AVX2, Q4_K picks q4_K_8x8_q8_K under AVX2, and Q6_K has NEON
#: branches only -- on x86 it falls through to plain row-by-row vec_dot. That
#: asymmetry is the whole reason this comparison exists, so it is stated here rather
#: than discovered at runtime (the selecting function is not exported).
REPACK_TYPES = {"q4_0", "q8_0", "q4_K"}

#: The same fact, stated exactly, for the expert mode: the AVX2 dispatch branches of
#: ``ggml_repack_get_optimal_repack_type`` (repack.cpp:4979-5100) reach Q4_0, Q4_K and
#: IQ4_NL. Q5_K (5070) and Q6_K (5082) only have NEON branches, and IQ4_XS has no
#: branch at all -- there is no ``block_iq4_xs`` tensor_traits instance to reach.
#: A type outside this set must not be put in a CPU_REPACK buffer: the buffer's
#: ``set_tensor`` dereferences the null ``tensor->extra`` and the process dies, which
#: is not a slow path but a crash. So the expert mode skips it and says so.
REPACK_GEMM_TYPES = {"q4_0", "q4_K", "iq4_nl"}

#: Types for which the CPU backend's ``tiled`` matmul (ggml/src/ggml-cpu/tiled/) is
#: eligible. Read off ``ggml_tiled_supported`` (tiled.cpp:616-635). IQ4_NL is absent.
#: Only reachable when GGML_CPU_TILED_MM is unset or non-zero (the default), the
#: weight is on a plain buffer (repacked weights set ``tensor->extra`` and are handed
#: back to the repack kernel), and the routed rows per expert are >= 8.
TILED_MM_TYPES = {"q6_K", "q5_K", "q4_K", "iq4_xs"}
TILED_MM_MIN_ROWS = 8


# --------------------------------------------------------------------------- ggml
class _InitParams(ctypes.Structure):
    # struct ggml_init_params { size_t mem_size; void *mem_buffer; bool no_alloc; }
    _fields_ = [("mem_size", ctypes.c_size_t),
                ("mem_buffer", ctypes.c_void_p),
                ("no_alloc", ctypes.c_bool)]


class Ggml:
    """The handful of ggml entry points this measurement needs, via ctypes.

    Two DLLs: ``ggml-base`` holds the compute-graph API, ``ggml-cpu`` owns the CPU
    backend that supplies the thread pool. ``ggml_backend_cpu_init`` is called
    directly rather than through the backend registry so no backend discovery or
    environment setup is involved.
    """

    def __init__(self, bin_dir: Path) -> None:
        self.base = ctypes.CDLL(str(bin_dir / "ggml-base.dll"))
        self.cpu = ctypes.CDLL(str(bin_dir / "ggml-cpu.dll"))
        b, c = self.base, self.cpu

        b.ggml_init.argtypes = [_InitParams]
        b.ggml_init.restype = ctypes.c_void_p
        b.ggml_free.argtypes = [ctypes.c_void_p]
        b.ggml_new_tensor_2d.argtypes = [ctypes.c_void_p, ctypes.c_int32,
                                         ctypes.c_int64, ctypes.c_int64]
        b.ggml_new_tensor_2d.restype = ctypes.c_void_p
        b.ggml_new_tensor_3d.argtypes = [ctypes.c_void_p, ctypes.c_int32,
                                         ctypes.c_int64, ctypes.c_int64,
                                         ctypes.c_int64]
        b.ggml_new_tensor_3d.restype = ctypes.c_void_p
        b.ggml_mul_mat.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        b.ggml_mul_mat.restype = ctypes.c_void_p
        b.ggml_mul_mat_id.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.c_void_p, ctypes.c_void_p]
        b.ggml_mul_mat_id.restype = ctypes.c_void_p
        b.ggml_new_graph.argtypes = [ctypes.c_void_p]
        b.ggml_new_graph.restype = ctypes.c_void_p
        b.ggml_build_forward_expand.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        b.ggml_backend_alloc_ctx_tensors.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        b.ggml_backend_alloc_ctx_tensors.restype = ctypes.c_void_p
        b.ggml_backend_buffer_free.argtypes = [ctypes.c_void_p]
        b.ggml_backend_tensor_set.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_size_t, ctypes.c_size_t]
        b.ggml_backend_tensor_get.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_size_t, ctypes.c_size_t]
        b.ggml_backend_graph_compute.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        b.ggml_backend_graph_compute.restype = ctypes.c_int32
        b.ggml_backend_free.argtypes = [ctypes.c_void_p]
        b.ggml_row_size.argtypes = [ctypes.c_int32, ctypes.c_int64]
        b.ggml_row_size.restype = ctypes.c_size_t
        b.ggml_nbytes.argtypes = [ctypes.c_void_p]
        b.ggml_nbytes.restype = ctypes.c_size_t
        b.ggml_quantize_chunk.argtypes = [ctypes.c_int32, ctypes.c_void_p,
                                          ctypes.c_void_p, ctypes.c_int64,
                                          ctypes.c_int64, ctypes.c_int64,
                                          ctypes.c_void_p]
        b.ggml_quantize_chunk.restype = ctypes.c_size_t
        c.ggml_backend_cpu_init.argtypes = []
        c.ggml_backend_cpu_init.restype = ctypes.c_void_p
        c.ggml_backend_cpu_set_n_threads.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        b.ggml_backend_alloc_ctx_tensors_from_buft.argtypes = [ctypes.c_void_p,
                                                               ctypes.c_void_p]
        b.ggml_backend_alloc_ctx_tensors_from_buft.restype = ctypes.c_void_p
        b.ggml_backend_buft_name.argtypes = [ctypes.c_void_p]
        b.ggml_backend_buft_name.restype = ctypes.c_char_p
        b.ggml_backend_reg_get_proc_address.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        b.ggml_backend_reg_get_proc_address.restype = ctypes.c_void_p
        b.ggml_backend_reg_dev_get.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        b.ggml_backend_reg_dev_get.restype = ctypes.c_void_p
        c.ggml_backend_cpu_reg.argtypes = []
        c.ggml_backend_cpu_reg.restype = ctypes.c_void_p
        self.backend = None
        self.repack_buft = None

    def find_repack_buft(self) -> str | None:
        """Return the CPU_REPACK buffer type, if this build has one.

        llama.cpp enables extra buffer types (weight repacking) by default -- there is
        a ``--no-repack`` flag precisely because it is on -- so a comparison that skips
        it measures a kernel the program never runs. The device does not export the
        buft symbol directly; it is published through the backend's
        ``ggml_backend_dev_get_extra_bufts`` proc address, which is how llama.cpp
        itself reaches it.
        """
        reg = self.cpu.ggml_backend_cpu_reg()
        if not reg:
            return None
        dev = self.base.ggml_backend_reg_dev_get(reg, 0)
        if not dev:
            return None
        proc = self.base.ggml_backend_reg_get_proc_address(
            reg, b"ggml_backend_dev_get_extra_bufts")
        if not proc:
            return None
        proto = ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)
        array = proto(proc)(dev)
        index = 0
        while array[index]:
            buft = array[index]
            name = self.base.ggml_backend_buft_name(buft)
            if name and name.decode() == "CPU_REPACK":
                self.repack_buft = buft
                return name.decode()
            index += 1
        return None

    def row_size(self, ggml_type: int, n_per_row: int) -> int:
        return int(self.base.ggml_row_size(ggml_type, n_per_row))

    def quantize(self, ggml_type: int, src_f32: np.ndarray, dst_view: np.ndarray,
                 n_per_row: int) -> int:
        """Quantise ``src_f32`` (rows x n_per_row) into ``dst_view``.

        ``start`` stays 0 and both pointers describe the chunk: ggml's ``start`` is an
        element offset that it *adds to src*, so handing it a chunk base together with
        a global offset reads past the chunk (measured: access violation on the first
        attempt at this).
        """
        nrows = src_f32.shape[0]
        written = self.base.ggml_quantize_chunk(
            ggml_type, src_f32.ctypes.data, dst_view.ctypes.data,
            0, nrows, n_per_row, None)
        return int(written)


# ------------------------------------------------------------------ weight source
def read_tensor_bytes(model_path: Path, tensor_name: str) -> tuple[np.ndarray, int, int, int]:
    """Return (raw bytes, ggml type, ne0, ne1) for one tensor, read from the GGUF.

    ``ReaderTensor.data`` is already the right slice of the reader's memmap, so it is
    used instead of arithmetic on ``data_offset``: that field has been both relative
    to the data section and absolute across versions, and getting it wrong yields
    plausible-looking but misaligned bytes.
    """
    import gguf

    reader = gguf.GGUFReader(str(model_path), mode="r")
    target = None
    for tensor in reader.tensors:
        if tensor.name == tensor_name:
            target = tensor
            break
    if target is None:
        raise SystemExit(f"tensor {tensor_name!r} not found in {model_path}")
    # GGUF reports shape with dimension 0 fastest, which is ggml's ne[0].
    ne0, ne1 = int(target.shape[0]), int(target.shape[1])
    dtype = int(target.tensor_type)
    raw = np.ascontiguousarray(target.data).reshape(-1).view(np.uint8)
    if raw.size != int(target.n_bytes):
        raise SystemExit(f"short read on {tensor_name}: {raw.size} != {target.n_bytes}")
    return raw, dtype, ne0, ne1


def convert_weight(g: Ggml, raw: np.ndarray, src_type: int, dst_type: int,
                   ne0: int, ne1: int, chunk_rows: int = 2048) -> np.ndarray:
    """q6_K (or whatever the file has) -> another ggml type, via llama.cpp's quantiser.

    Goes through f32 rather than chasing a direct kernel: the source is q6_K and the
    targets are Q4_0/Q4_K, and ``dequantize`` reaches f32 for 100 MB/s here, which is
    four seconds for the whole tensor.

    Note the extra quantisation error this path accepts: the target is computed from
    the *dequantised* source, not from the f32 master. That is the same path
    ``llama-quantize`` takes when it re-quantises an already-quantised file, and for
    the IQ4_XS -> Q4_K question it is the only path available (the model ships no
    f32 master), but it means "Q4_K" here is Q4_K-of-IQ4_XS, not Q4_K-of-f32.
    """
    from gguf import GGMLQuantizationType as QT
    from gguf import quants

    src_baf = int(g.row_size(src_type, ne0))
    dst_baf = int(g.row_size(dst_type, ne0))
    out = np.empty(ne1 * dst_baf, dtype=np.uint8)
    src_rows = raw.reshape(ne1, src_baf)
    qt_src = QT(src_type)
    for row0 in range(0, ne1, chunk_rows):
        row1 = min(ne1, row0 + chunk_rows)
        block = np.ascontiguousarray(src_rows[row0:row1])
        f32 = quants.dequantize(block, qt_src).astype(np.float32, copy=False)
        g.quantize(dst_type, f32, out[row0 * dst_baf:row1 * dst_baf], ne0)
    return out


def convert_expert_bank(g: Ggml, raw: np.ndarray, src_type: int, dst_type: int,
                        ne0: int, ne1: int, ne2: int,
                        chunk_experts: int = 16) -> np.ndarray:
    """Same as ``convert_weight`` for a 3D expert bank ``[ne0, ne1, ne2]``.

    Chunked over experts, not rows: one expert is ``ne1 * ne0 * 4`` bytes as f32
    (2048*512*4 = 4 MiB), so 16 experts is 64 MiB of f32 in flight rather than the
    1 GiB a whole-bank dequantise would need.
    """
    from gguf import GGMLQuantizationType as QT
    from gguf import quants

    src_baf = int(g.row_size(src_type, ne0))
    dst_baf = int(g.row_size(dst_type, ne0))
    src = raw.reshape(ne2, ne1, src_baf)
    out = np.empty(ne2 * ne1 * dst_baf, dtype=np.uint8)
    qt_src = QT(src_type)
    for e0 in range(0, ne2, chunk_experts):
        e1 = min(ne2, e0 + chunk_experts)
        block = np.ascontiguousarray(src[e0:e1].reshape(-1, src_baf))
        f32 = quants.dequantize(block, qt_src).astype(np.float32, copy=False)
        g.quantize(dst_type, f32, out[e0 * ne1 * dst_baf:e1 * ne1 * dst_baf], ne0)
        del block, f32
    return out


def read_expert_census(model_path: Path) -> dict:
    """Census of the real expert tensors: type histogram per role, plus geometry.

    The point of this function is that *nothing below is assumed*: the shape that
    gets benchmarked is the shape this returns, and the type mixture that motivates
    the whole question is printed from the file rather than quoted from a document.
    """
    import gguf

    pattern = re.compile(r"blk\.(\d+)\.(ffn_(?:gate|up|down)_exps)\.weight$")
    reader = gguf.GGUFReader(str(model_path), mode="r")
    entries = []
    for tensor in reader.tensors:
        hit = pattern.match(tensor.name)
        if not hit:
            continue
        entries.append({
            "name": tensor.name,
            "layer": int(hit.group(1)),
            "role": hit.group(2),
            "type": int(tensor.tensor_type),
            "shape": [int(x) for x in tensor.shape],
        })
    if not entries:
        raise SystemExit(f"no blk.*.ffn_*_exps.weight tensors in {model_path}")

    by_role: dict[str, dict] = {}
    for entry in entries:
        role = entry["role"]
        type_name = TENSOR_TYPE_NAMES.get(entry["type"], f"type{entry['type']}")
        slot = by_role.setdefault(role, {"count": 0, "shapes": set(), "by_type": {}})
        slot["count"] += 1
        slot["shapes"].add(tuple(entry["shape"]))
        slot["by_type"][type_name] = slot["by_type"].get(type_name, 0) + 1

    # Metadata is a cross-check only; the geometry below comes from the tensors.
    meta = {}
    for key in ("expert_count", "expert_used_count", "embedding_length",
                "expert_feed_forward_length", "block_count"):
        for field in reader.fields:
            if field.endswith("." + key):
                try:
                    meta[key] = int(reader.fields[field].contents())
                except (TypeError, ValueError):
                    pass
                break

    return {"entries": entries, "by_role": by_role, "meta": meta,
            "layers": max(e["layer"] for e in entries) + 1}


# ------------------------------------------------------------------- measurement
class Case:
    """One (weight, M) pair: its own context, buffer and graph, reusable across rounds.

    All ``M`` cases stay open for the whole sweep so the rounds can *interleave* them.
    That is not cosmetic: this host has other processes competing for the CPU (the
    harness and its web views were measured at roughly three cores' worth of load
    while writing this), so benchmarking one M in a contiguous block lets a busy
    interval land on a single M and manufacture a step in the curve. That is exactly
    the shape the old numbers show -- 16 rows cheaper than 8.

    Kept open at ~400 MiB per case for lm_head, but an expert slice is only ~0.5 MiB,
    so the expert mode can afford many more cases at once.
    """

    def __init__(self, g: Ggml, weight: np.ndarray, wtype: int,
                 ne0: int, ne1: int, m: int, hidden: np.ndarray,
                 *, repack: bool = False) -> None:
        self.g = g
        self.m = m
        self.ne1 = ne1
        self.buft = "CPU_REPACK" if repack else "CPU"
        self.times: list[float] = []
        # The weight goes in its own context when repacking: its buffer type differs
        # from the activations', and a buffer type is chosen per context.
        self.ctx_w = g.base.ggml_init(_InitParams(64 << 20, None, True))
        self.ctx_a = g.base.ggml_init(_InitParams(64 << 20, None, True))
        if not self.ctx_w or not self.ctx_a:
            raise SystemExit("ggml_init failed")
        self.buf_w = None
        self.buf_a = None
        try:
            self.w = g.base.ggml_new_tensor_2d(self.ctx_w, wtype, ne0, ne1)
            self.a = g.base.ggml_new_tensor_2d(self.ctx_a, GGML_TYPE_F32, ne0, m)
            self.out = g.base.ggml_mul_mat(self.ctx_a, self.w, self.a)
            if repack:
                if g.repack_buft is None:
                    raise SystemExit("repack requested but CPU_REPACK is unavailable")
                self.buf_w = g.base.ggml_backend_alloc_ctx_tensors_from_buft(
                    self.ctx_w, g.repack_buft)
            else:
                self.buf_w = g.base.ggml_backend_alloc_ctx_tensors(self.ctx_w, g.backend)
            self.buf_a = g.base.ggml_backend_alloc_ctx_tensors(self.ctx_a, g.backend)
            if not self.buf_w or not self.buf_a:
                raise SystemExit("backend allocation failed")
            # set on a repack buffer is what performs the repacking, once, exactly like
            # model load does it; the timed region never pays for it.
            g.base.ggml_backend_tensor_set(self.w, weight.ctypes.data, 0, weight.nbytes)
            g.base.ggml_backend_tensor_set(self.a, hidden.ctypes.data, 0, hidden.nbytes)
            self.graph = g.base.ggml_new_graph(self.ctx_a)
            g.base.ggml_build_forward_expand(self.graph, self.out)
        except BaseException:
            self.close()
            raise

    def compute(self) -> float:
        """Run the graph once and return its wall time in milliseconds."""
        t0 = time.perf_counter()
        status = self.g.base.ggml_backend_graph_compute(self.g.backend, self.graph)
        elapsed = (time.perf_counter() - t0) * 1e3
        if status != 0:
            raise SystemExit(f"graph compute failed (status {status})")
        return elapsed

    def warmup(self, times: int) -> None:
        for _ in range(times):
            self.compute()

    def read_logits(self) -> np.ndarray:
        """Copy the logits back to numpy (the cost the real path also pays)."""
        result = np.empty((self.m, self.ne1), dtype=np.float32)
        self.g.base.ggml_backend_tensor_get(self.out, result.ctypes.data, 0,
                                            result.nbytes)
        return result

    def close(self) -> None:
        # Buffers own their tensor data; ggml_free only releases the context's
        # metadata, so leaving them to ggml_free would leak ~400 MiB per case.
        for name in ("buf_w", "buf_a"):
            buf = getattr(self, name, None)
            if buf is not None:
                self.g.base.ggml_backend_buffer_free(buf)
                setattr(self, name, None)
        for name in ("ctx_w", "ctx_a"):
            ctx = getattr(self, name, None)
            if ctx is not None:
                self.g.base.ggml_free(ctx)
                setattr(self, name, None)


class ExpertBank:
    """One expert bank ``[ne0, ne1, ne2]`` resident in a plain or CPU_REPACK buffer.

    The bank is the expensive object (~136 MiB for IQ4_XS), so it is built once per
    (type, buffer type) and shared by every ``MUL_MAT_ID`` row count: each row count
    only allocates its own activations, ids and output.
    """

    def __init__(self, g: Ggml, raw: np.ndarray, wtype: int, ne0: int, ne1: int,
                 ne2: int, *, repack: bool) -> None:
        self.g = g
        self.ne0, self.ne1, self.ne2 = ne0, ne1, ne2
        self.nbytes = int(raw.nbytes)
        self.ctx_w = g.base.ggml_init(_InitParams(64 << 20, None, True))
        if not self.ctx_w:
            raise SystemExit("ggml_init failed")
        self.buf_w = None
        self.w = g.base.ggml_new_tensor_3d(self.ctx_w, wtype, ne0, ne1, ne2)
        if repack:
            if g.repack_buft is None:
                raise SystemExit("repack requested but CPU_REPACK is unavailable")
            self.buf_w = g.base.ggml_backend_alloc_ctx_tensors_from_buft(
                self.ctx_w, g.repack_buft)
        else:
            self.buf_w = g.base.ggml_backend_alloc_ctx_tensors(self.ctx_w, g.backend)
        if not self.buf_w:
            raise SystemExit("backend allocation failed")
        g.base.ggml_backend_tensor_set(self.w, raw.ctypes.data, 0, raw.nbytes)

    def close(self) -> None:
        if self.buf_w is not None:
            self.g.base.ggml_backend_buffer_free(self.buf_w)
            self.buf_w = None
        if self.ctx_w is not None:
            self.g.base.ggml_free(self.ctx_w)
            self.ctx_w = None


class MmidRun:
    """One ``MUL_MAT_ID`` graph over a shared expert bank, for one token count.

    Built the way ``build_moe_ffn`` builds it: ``src1 = [n_embd, 1, n_tokens]`` (the
    hidden row is broadcast to every used expert) and ``ids = [n_expert_used,
    n_tokens]``, giving ``dst = [n_ff_exp, n_expert_used, n_tokens]``.
    """

    def __init__(self, g: Ggml, bank: ExpertBank, m: int, hidden: np.ndarray,
                 ids: np.ndarray) -> None:
        self.g = g
        self.m = m
        self.n_used = int(ids.shape[0])
        self.times: list[float] = []
        self.ctx_a = g.base.ggml_init(_InitParams(64 << 20, None, True))
        self.buf_a = None
        self.a = g.base.ggml_new_tensor_3d(self.ctx_a, GGML_TYPE_F32, bank.ne0, 1, m)
        self.ids = g.base.ggml_new_tensor_2d(self.ctx_a, GGML_TYPE_I32,
                                             self.n_used, m)
        self.out = g.base.ggml_mul_mat_id(self.ctx_a, bank.w, self.a, self.ids)
        self.buf_a = g.base.ggml_backend_alloc_ctx_tensors(self.ctx_a, g.backend)
        if not self.buf_a:
            raise SystemExit("backend allocation failed")
        g.base.ggml_backend_tensor_set(self.a, hidden.ctypes.data, 0, hidden.nbytes)
        g.base.ggml_backend_tensor_set(self.ids, ids.ctypes.data, 0, ids.nbytes)
        self.graph = g.base.ggml_new_graph(self.ctx_a)
        g.base.ggml_build_forward_expand(self.graph, self.out)

    def compute(self) -> float:
        t0 = time.perf_counter()
        status = self.g.base.ggml_backend_graph_compute(self.g.backend, self.graph)
        elapsed = (time.perf_counter() - t0) * 1e3
        if status != 0:
            raise SystemExit(f"graph compute failed (status {status})")
        return elapsed

    def warmup(self, times: int) -> None:
        for _ in range(times):
            self.compute()

    def read_out(self) -> np.ndarray:
        result = np.empty((self.out_ne()), dtype=np.float32)
        self.g.base.ggml_backend_tensor_get(self.out, result.ctypes.data, 0,
                                            result.nbytes)
        return result

    def out_ne(self) -> tuple[int, int, int]:
        return (int(self.g.base.ggml_nbytes(self.out) // 4) // (self.n_used * self.m),
                self.n_used, self.m)

    def close(self) -> None:
        if self.buf_a is not None:
            self.g.base.ggml_backend_buffer_free(self.buf_a)
            self.buf_a = None
        if self.ctx_a is not None:
            self.g.base.ggml_free(self.ctx_a)
            self.ctx_a = None


def fit(ms: list[int], times: list[float]) -> dict:
    """Least-squares T(M) = t0 + m*M, plus R^2 over the linear-only residual."""
    x = np.asarray(ms, dtype=np.float64)
    y = np.asarray(times, dtype=np.float64)
    if x.size < 2:
        return {"slope_ms_per_row": None, "t0_ms": None, "r2": None,
                "points": int(x.size), "residual_max_ms": None}
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"slope_ms_per_row": float(slope),
            "t0_ms": float(intercept),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
            "points": int(x.size),
            "residual_max_ms": float(np.max(np.abs(y - pred)))}


def segment_slopes(ms: list[int], times: list[float]) -> dict:
    """Slope between measured pairs, because the curve is visibly not one line.

    Q6_K (and Q4_K on the plain buffer) flatten out between 8 and 16 rows: the weight
    pass dominates and the extra rows ride along nearly free. A single slope over
    M=1..32 averages a steep region with a flat one and describes neither. These
    segments keep that visible instead of hiding it behind a low R^2.
    """
    index = dict(zip(ms, times))
    out = {}
    for lo, hi in ((1, 8), (8, 16), (16, 32)):
        if lo in index and hi in index:
            out[f"{lo}..{hi}"] = (index[hi] - index[lo]) / (hi - lo)
    return out


def tiled_mm_enabled() -> bool:
    """Whether the CPU backend's tiled matmul is on, matching tiled.cpp's reading."""
    import os
    value = os.environ.get("GGML_CPU_TILED_MM")
    return value is None or int(value) != 0


def buft_modes(args, type_name: str, allowed: set[str]) -> list[str]:
    """Which buffer types to measure for one quantisation type."""
    if args.buft == "auto":
        modes = ["plain"] + (["repack"] if type_name in allowed else [])
    else:
        modes = [args.buft]
    return modes


# ---------------------------------------------------------------- lm_head mode
def run_lm_head(args, g: Ggml) -> dict:
    ms = [int(v) for v in args.ms.split(",") if v.strip()]
    wanted = [v.strip().lower() for v in args.types.split(",") if v.strip()]

    t_load = time.perf_counter()
    raw, src_type, ne0, ne1 = read_tensor_bytes(args.model, args.tensor)
    src_name = TENSOR_TYPE_NAMES.get(src_type, f"type{src_type}")
    print(f"{args.tensor}: ne0={ne0} ne1={ne1} type={src_name} "
          f"({raw.nbytes / 2**20:.1f} MiB, read in {time.perf_counter() - t_load:.1f}s)")
    print(f"threads={args.threads} reps={args.reps} warmup={args.warmup}")

    rng = np.random.RandomState(args.seed)
    hidden = rng.randn(max(ms), ne0).astype(np.float32)

    report: dict = {"model": str(args.model), "tensor": args.tensor,
                    "threads": args.threads, "reps": args.reps,
                    "warmup": args.warmup, "ne0": ne0, "ne1": ne1,
                    "source_type": src_name, "points": [], "fits": {}}

    print()
    header = (f"{'type/buft':>12} {'M':>3} {'median':>9} {'min':>9} "
              f"{'max':>9}  {'GiB/s':>7}")
    print(header)
    print("-" * len(header))

    prepared: dict[str, np.ndarray] = {}
    for name in wanted:
        resolved = resolve_type(name)
        if resolved is None:
            print(f"  skipping unknown type {name!r} (known: {sorted(TYPE_CODES)})",
                  file=sys.stderr)
            continue
        gt, type_name = resolved
        t_conv = time.perf_counter()
        if gt == src_type:
            weight = raw
        elif type_name in prepared:
            weight = prepared[type_name]
        else:
            weight = convert_weight(g, raw, src_type, gt, ne0, ne1)
            prepared[type_name] = weight
        conv_s = time.perf_counter() - t_conv

        modes = buft_modes(args, type_name, REPACK_TYPES)
        if "repack" in modes and g.repack_buft is None:
            print("  CPU_REPACK unavailable in this build; dropping the repack pass",
                  file=sys.stderr)
            modes = [mode for mode in modes if mode != "repack"]

        for mode in modes:
            repack = mode == "repack"
            label = f"{type_name}/{mode}"
            cases = [Case(g, weight, gt, ne0, ne1, m, hidden[:m], repack=repack)
                     for m in ms]
            try:
                for case in cases:
                    case.warmup(args.warmup)
                # Interleave the rounds across all M so a busy interval cannot land on
                # one M and bend the curve (see Case).
                for _ in range(args.reps):
                    for case in cases:
                        case.times.append(case.compute())
                gets = []
                for case in cases:
                    readings = []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        case.read_logits()
                        readings.append((time.perf_counter() - t0) * 1e3)
                    gets.append(statistics.median(readings))

                # Verify every measured M, not just the first: the claim under test is
                # that T(M) grows with M, and an implementation that quietly computed
                # fewer rows would look exactly like a suspiciously flat curve.
                verify_block = None
                if args.verify:
                    from gguf import GGMLQuantizationType as QT
                    from gguf import quants
                    baf = int(g.row_size(gt, ne0))
                    ref = quants.dequantize(weight.reshape(ne1, baf), QT(gt))
                    per_m = []
                    for case in cases:
                        got = case.read_logits()
                        want = hidden[:case.m] @ ref.T
                        err = float(np.abs(got - want).max())
                        scale = float(np.abs(want).max()) or 1.0
                        per_m.append({"m": case.m, "max_abs_err": err,
                                      "rel_err": err / scale})
                    del ref
                    verify_block = {"type": type_name, "buft": mode, "rows": per_m}
                    worst = max(point["rel_err"] for point in per_m)
                    # The tolerance is loose on purpose and must not be tightened into
                    # a correctness gate: ggml quantises the f32 activations to q8_K
                    # before the dot product, so ~1% disagreement with an exact f32
                    # numpy matmul is what the real inference path produces too. Far
                    # above that means the two sides read different bytes.
                    print(f"verify({label}): worst rel err over M="
                          f"{[p['m'] for p in per_m]} = {worst:.3g}")
                    if worst > 0.03:
                        print("  WARNING: ggml and numpy disagree by more than q8_K "
                              "rounding explains; the timing may not be this tensor's",
                              file=sys.stderr)

                medians = []
                for case, get_ms in zip(cases, gets):
                    samples = case.times
                    # effective bandwidth: one pass over the weight bytes per compute
                    gbs = (weight.nbytes / 2**30) / (statistics.median(samples) / 1e3)
                    point = {"type": type_name, "buft": mode, "m": case.m,
                             "ms": samples,
                             "median_ms": statistics.median(samples),
                             "min_ms": min(samples), "max_ms": max(samples),
                             "get_median_ms": get_ms,
                             "weight_mib": weight.nbytes / 2**20,
                             "weight_gib_per_s": gbs}
                    report["points"].append(point)
                    medians.append(point["median_ms"])
                    print(f"{label:>12} {case.m:>3} {point['median_ms']:>9.3f} "
                          f"{point['min_ms']:>9.3f} {point['max_ms']:>9.3f}  "
                          f"{gbs:>7.1f}")
            finally:
                for case in cases:
                    case.close()

            median_fit = fit(ms, medians)
            min_fit = fit(ms, [min(c.times) for c in cases])
            median_fit["conversion_s"] = round(conv_s, 2)
            median_fit["fit_of_min"] = min_fit
            median_fit["segments_of_median"] = segment_slopes(ms, medians)
            median_fit["segments_of_min"] = segment_slopes(
                ms, [min(c.times) for c in cases])
            report["fits"][label] = median_fit
            if verify_block is not None:
                report.setdefault("verify", []).append(verify_block)
            print(f"       fit(median): t0={median_fit['t0_ms']:.3f} ms  "
                  f"m={median_fit['slope_ms_per_row']:.4f} ms/row  "
                  f"R2={median_fit['r2']:.5f}")
            print(f"       fit(min):    t0={min_fit['t0_ms']:.3f} ms  "
                  f"m={min_fit['slope_ms_per_row']:.4f} ms/row  "
                  f"R2={min_fit['r2']:.5f}")
            print()

    # marginal ratio: does a different quantisation actually buy the 2x the docs infer?
    fits = report["fits"]
    baseline_label = f"{src_name}/plain"
    if baseline_label in fits:
        base = fits[baseline_label]["slope_ms_per_row"]
        report["marginal_ratio_vs_source"] = {
            name: (fit_["slope_ms_per_row"] / base if base else None)
            for name, fit_ in fits.items()}
    return report


# ----------------------------------------------------------------- experts mode
def pick_source_entry(census: dict, layer: int | None) -> dict:
    """Choose the expert tensor to measure: a gate bank, IQ4_XS preferred.

    IQ4_XS is preferred because that is the type the question is about (63 of the 120
    expert tensors), and the gate bank because its ``[n_embd, n_ff_exp, n_expert]``
    orientation is the one ``mul_mat_id`` consumes directly; ``ffn_down_exps`` is
    ``[n_ff_exp, n_embd, n_expert]`` and would be a different shape claim.
    """
    candidates = [e for e in census["entries"] if e["role"] == "ffn_gate_exps"]
    if layer is not None:
        exact = [e for e in candidates if e["layer"] == layer]
        if not exact:
            raise SystemExit(f"no ffn_gate_exps.weight in layer {layer}")
        candidates = exact
    iq4 = [e for e in candidates if e["type"] == GGML_TYPE_IQ4_XS]
    chosen = (iq4 or candidates)[0]
    return chosen


def run_experts(args, g: Ggml) -> dict:
    ms = [int(v) for v in args.expert_ms.split(",") if v.strip()]
    id_ms = [int(v) for v in args.expert_mmid_ms.split(",") if v.strip()]
    wanted = [v.strip().lower() for v in args.types.split(",") if v.strip()]

    print("== expert tensor census (read from the GGUF, nothing assumed) ==")
    census = read_expert_census(args.model)
    for role in sorted(census["by_role"]):
        slot = census["by_role"][role]
        shapes = " or ".join(str(list(s)) for s in sorted(slot["shapes"]))
        mix = ", ".join(f"{name} {count}" for name, count in
                        sorted(slot["by_type"].items(), key=lambda kv: -kv[1]))
        print(f"  {role:<18} {shapes:<22} x{slot['count']:<3} {mix}")
    print(f"  metadata cross-check: {census['meta']}")

    source = pick_source_entry(census, args.expert_layer)
    raw3, src_type, ne0, ne1, ne2 = read_expert_bank(args.model, source["name"])
    src_name = TENSOR_TYPE_NAMES.get(src_type, f"type{src_type}")
    n_expert = ne2
    n_used = int(census["meta"].get("expert_used_count") or args.n_used)
    n_embd = int(census["meta"].get("embedding_length") or ne0)
    n_ff = int(census["meta"].get("expert_feed_forward_length") or ne1)
    print(f"\nsource tensor: {source['name']}  shape=[{ne0}, {ne1}, {ne2}]  "
          f"type={src_name}  ({raw3.nbytes / 2**20:.1f} MiB)")
    print(f"derived: n_embd={n_embd} n_ff_exp={n_ff} n_expert={n_expert} "
          f"n_expert_used={n_used}")
    if (ne0, ne1) != (n_embd, n_ff):
        print(f"  NOTE: tensor ne0/ne1 = ({ne0}, {ne1}) differs from metadata "
              f"({n_embd}, {n_ff}); the tensor geometry wins", file=sys.stderr)
    single_exp = np.ascontiguousarray(raw3[: ne1 * int(g.row_size(src_type, ne0))])
    print(f"single expert slice: [{ne0}, {ne1}] = {single_exp.nbytes / 2**10:.1f} KiB")
    print(f"threads={args.threads} reps={args.reps} warmup={args.warmup} "
          f"tiled_mm={tiled_mm_enabled()}")

    rng = np.random.RandomState(args.seed)
    hidden = rng.randn(max(ms), ne0).astype(np.float32)
    # ids: n_expert_used distinct experts per token, drawn without replacement --
    # what a top-k router produces for this architecture (8 of 256 per token).
    ids_bank = {m: np.stack([rng.choice(n_expert, n_used, replace=False)
                             for _ in range(m)], axis=1).astype(np.int32)
                for m in id_ms}
    # b = [n_embd, 1, n_tokens]: one hidden row per token, broadcast to every
    # used expert of that token. C order of [n_tokens, n_embd] is exactly that.
    hidden3 = {m: np.ascontiguousarray(hidden[:m]) for m in id_ms}

    report: dict = {
        "model": str(args.model), "mode": "experts",
        "tensor": source["name"], "threads": args.threads,
        "reps": args.reps, "warmup": args.warmup,
        "ne0": ne0, "ne1": ne1, "ne2": ne2, "source_type": src_name,
        "experts": {
            "census": {"by_role": {role: {"count": slot["count"],
                                          "shapes": [list(s) for s in slot["shapes"]],
                                          "by_type": slot["by_type"]}
                                   for role, slot in census["by_role"].items()},
                       "metadata": census["meta"],
                       "expert_tensor_count": len(census["entries"])},
            "source": {"tensor": source["name"], "layer": source["layer"],
                       "shape": [ne0, ne1, ne2], "type": src_name,
                       "n_expert": n_expert, "n_expert_used": n_used,
                       "n_embd": n_embd, "n_ff_exp": n_ff},
            "tiled_mm": tiled_mm_enabled(),
            "mul_mat": {}, "mul_mat_id": {},
        },
    }
    block = report["experts"]

    prepared3: dict[str, tuple[np.ndarray, int]] = {"iq4_xs": (raw3, src_type)} \
        if src_name == "iq4_xs" else {}
    prepared2: dict[str, tuple[np.ndarray, int]] = {}

    def weight_2d(type_name: str, gt: int) -> np.ndarray:
        if type_name == src_name:
            return single_exp
        if type_name not in prepared2:
            t0 = time.perf_counter()
            prepared2[type_name] = (convert_weight(g, single_exp, src_type, gt,
                                                   ne0, ne1), gt)
            print(f"  converted single expert -> {type_name} in "
                  f"{time.perf_counter() - t0:.2f}s")
        return prepared2[type_name][0]

    def bank_3d(type_name: str, gt: int) -> np.ndarray:
        if type_name == src_name:
            return raw3
        if type_name not in prepared3:
            t0 = time.perf_counter()
            prepared3[type_name] = (convert_expert_bank(g, raw3, src_type, gt,
                                                        ne0, ne1, ne2), gt)
            print(f"  converted expert bank -> {type_name} in "
                  f"{time.perf_counter() - t0:.1f}s "
                  f"({prepared3[type_name][0].nbytes / 2**20:.1f} MiB)")
        return prepared3[type_name][0]

    # ------------------------------------------------------------- MUL_MAT
    print(f"\n== MUL_MAT: one expert [{ne0}, {ne1}], M rows in ==")
    header = (f"{'type/buft':>14} {'M':>3} {'median':>9} {'min':>9} {'GiB/s':>7} "
              f"{'tiled':>5}")
    print(header)
    print("-" * len(header))
    for name in wanted:
        resolved = resolve_type(name)
        if resolved is None:
            print(f"  skipping unknown type {name!r}", file=sys.stderr)
            continue
        gt, type_name = resolved
        weight = weight_2d(type_name, gt)
        modes = buft_modes(args, type_name, REPACK_GEMM_TYPES)
        if "repack" in modes and g.repack_buft is None:
            modes = [mode for mode in modes if mode != "repack"]
        for mode in modes:
            repack = mode == "repack"
            label = f"{type_name}/{mode}"
            cases = []
            try:
                for m in ms:
                    cases.append(Case(g, weight, gt, ne0, ne1, m, hidden[:m],
                                      repack=repack))
                for case in cases:
                    case.warmup(args.warmup)
                for _ in range(args.reps):
                    for case in cases:
                        case.times.append(case.compute())
                points = []
                for case in cases:
                    med = statistics.median(case.times)
                    gbs = (weight.nbytes / 2**30) / (med / 1e3)
                    # tiled engages per expert only when the routed rows reach 8,
                    # the weight is not repacked, and the type is on its list
                    tiled = (tiled_mm_enabled() and not repack
                             and type_name in TILED_MM_TYPES
                             and case.m >= TILED_MM_MIN_ROWS)
                    point = {"type": type_name, "buft": mode, "m": case.m,
                             "ms": case.times, "median_ms": med,
                             "min_ms": min(case.times), "max_ms": max(case.times),
                             "weight_kib": weight.nbytes / 2**10,
                             "weight_gib_per_s": gbs,
                             "tiled_applicable": bool(tiled)}
                    points.append(point)
                    print(f"{label:>14} {case.m:>3} {med:>9.4f} "
                          f"{min(case.times):>9.4f} {gbs:>7.2f} "
                          f"{'yes' if tiled else 'no':>5}")
                if args.verify:
                    check_mul_mat(g, weight, gt, ne0, ne1, hidden[:max(ms)], cases,
                                  type_name, mode)
            finally:
                for case in cases:
                    case.close()
            block["mul_mat"][label] = {
                "points": points,
                "median_ms_by_m": {p["m"]: p["median_ms"] for p in points},
                "min_ms_by_m": {p["m"]: p["min_ms"] for p in points},
            }

    # ---------------------------------------------------------- MUL_MAT_ID
    print(f"\n== MUL_MAT_ID: expert bank [{ne0}, {ne1}, {ne2}], "
          f"n_expert_used={n_used} ==")
    header = (f"{'type/buft':>14} {'M':>3} {'median':>9} {'min':>9} "
              f"{'GiB/s':>7} {'tiled':>5}")
    print(header)
    print("-" * len(header))
    for name in wanted:
        resolved = resolve_type(name)
        if resolved is None:
            continue
        gt, type_name = resolved
        bank_raw = bank_3d(type_name, gt)
        modes = buft_modes(args, type_name, REPACK_GEMM_TYPES)
        if "repack" in modes and g.repack_buft is None:
            modes = [mode for mode in modes if mode != "repack"]
        for mode in modes:
            repack = mode == "repack"
            label = f"{type_name}/{mode}"
            bank = None
            runs = []
            try:
                bank = ExpertBank(g, bank_raw, gt, ne0, ne1, ne2, repack=repack)
                for m in id_ms:
                    runs.append(MmidRun(g, bank, m, hidden3[m], ids_bank[m]))
                for run in runs:
                    run.warmup(args.warmup)
                for _ in range(args.reps):
                    for run in runs:
                        run.times.append(run.compute())
                points = []
                for run in runs:
                    med = statistics.median(run.times)
                    # one pass over the used experts' bytes per compute
                    moved = bank.nbytes * n_used / n_expert
                    gbs = (moved / 2**30) / (med / 1e3)
                    # tiled engages per expert on the number of *routed rows* that
                    # land on it, not on the token count: with 256 experts and 8 used
                    # per token, 8 tokens average 0.25 rows per expert, so M has to
                    # reach n_expert/... before the batch floor of 8 is met.
                    rows_per_expert = run.m * n_used / n_expert
                    tiled = (tiled_mm_enabled() and not repack
                             and type_name in TILED_MM_TYPES
                             and rows_per_expert >= TILED_MM_MIN_ROWS)
                    point = {"type": type_name, "buft": mode, "m": run.m,
                             "ms": run.times, "median_ms": med,
                             "min_ms": min(run.times), "max_ms": max(run.times),
                             "expert_bytes_mib": moved / 2**20,
                             "experts_gib_per_s": gbs,
                             "rows_per_expert_avg": rows_per_expert,
                             "tiled_applicable": bool(tiled)}
                    points.append(point)
                    print(f"{label:>14} {run.m:>3} {med:>9.4f} "
                          f"{min(run.times):>9.4f} {gbs:>7.2f} "
                          f"{'yes' if tiled else 'no':>5}")
                if args.verify:
                    check_mul_mat_id(g, bank_raw, gt, ne0, ne1, ne2, runs,
                                     hidden3, ids_bank, type_name, mode)
            finally:
                for run in runs:
                    run.close()
                if bank is not None:
                    bank.close()
            block["mul_mat_id"][label] = {
                "points": points,
                "median_ms_by_m": {p["m"]: p["median_ms"] for p in points},
                "min_ms_by_m": {p["m"]: p["min_ms"] for p in points},
                "note": ("result layout of ggml_mul_mat_id in this build is not "
                         "the [rows, n_expert_used, n_tokens] order ggml.h documents; "
                         "timing is unaffected, verification is by block sum"),
            }

    block["answers"] = build_answers(block, src_name, ms, id_ms)
    return report


def read_expert_bank(model_path: Path, name: str) -> tuple[np.ndarray, int, int, int, int]:
    """Return (raw bytes, ggml type, ne0, ne1, ne2) for a 3D expert tensor."""
    import gguf

    reader = gguf.GGUFReader(str(model_path), mode="r")
    target = None
    for tensor in reader.tensors:
        if tensor.name == name:
            target = tensor
            break
    if target is None:
        raise SystemExit(f"tensor {name!r} not found in {model_path}")
    if len(target.shape) != 3:
        raise SystemExit(f"{name} is not 3D: {list(target.shape)}")
    ne0, ne1, ne2 = (int(x) for x in target.shape)
    dtype = int(target.tensor_type)
    raw = np.ascontiguousarray(target.data).reshape(-1).view(np.uint8)
    if raw.size != int(target.n_bytes):
        raise SystemExit(f"short read on {name}: {raw.size} != {target.n_bytes}")
    return raw, dtype, ne0, ne1, ne2


def check_mul_mat(g: Ggml, weight: np.ndarray, gt: int, ne0: int, ne1: int,
                  hidden: np.ndarray, cases: list[Case], type_name: str,
                  mode: str) -> float:
    """Compare every MUL_MAT case against an exact f32 reference in numpy."""
    from gguf import GGMLQuantizationType as QT
    from gguf import quants

    baf = int(g.row_size(gt, ne0))
    ref = quants.dequantize(weight.reshape(ne1, baf), QT(gt))
    worst = 0.0
    for case in cases:
        got = case.read_logits()
        want = hidden[:case.m] @ ref.T
        scale = float(np.abs(want).max()) or 1.0
        worst = max(worst, float(np.abs(got - want).max()) / scale)
    del ref
    print(f"       verify: worst rel err vs numpy = {worst:.3g}")
    if worst > 0.03:
        print("  WARNING: ggml and numpy disagree by more than q8_K rounding "
              "explains", file=sys.stderr)
    return worst


def check_mul_mat_id(g: Ggml, bank_raw: np.ndarray, gt: int, ne0: int, ne1: int,
                     ne2: int, runs: list[MmidRun], hidden3: dict,
                     ids_bank: dict, type_name: str, mode: str) -> float:
    """Check MUL_MAT_ID by block sum, not element-wise (see the module docstring).

    ``ggml_mul_mat_id`` in this build returns the right values in a permuted order
    inside ``[rows, n_expert_used, n_tokens]``, so element-wise comparison fails for a
    reason that has nothing to do with the weights. The block sum is invariant to that
    permutation and still catches a dropped expert or a skipped row.
    """
    from gguf import GGMLQuantizationType as QT
    from gguf import quants

    baf = int(g.row_size(gt, ne0))
    banks = bank_raw.reshape(ne2, ne1, baf)
    worst = 0.0
    for run in runs:
        m = run.m
        got = run.read_out()
        ids = ids_bank[m]
        h = hidden3[m]
        want = np.empty((run.out_ne()[0], run.n_used, m), dtype=np.float64)
        cache: dict[int, np.ndarray] = {}
        for t in range(m):
            for e in range(run.n_used):
                x = int(ids[e, t])
                if x not in cache:
                    cache[x] = quants.dequantize(banks[x], QT(gt))
                want[:, e, t] = cache[x] @ h[t]
        # Layout-invariant check: if the only defect is the element order, the two
        # sorted vectors agree. This catches a dropped expert or a skipped row, which
        # a block sum would only blur.
        s_got = np.sort(got.astype(np.float64).ravel())
        s_want = np.sort(want.ravel())
        mag = float(np.abs(s_want).max()) or 1.0
        sorted_rel = float(np.abs(s_got - s_want).max()) / mag
        scale = float(np.abs(want).sum()) or 1.0
        sum_rel = abs(float(got.sum(dtype=np.float64)) - float(want.sum())) / scale
        worst = max(worst, sorted_rel)
        print(f"       verify(M={m}): sorted-value max rel err = {sorted_rel:.3g}, "
              f"block-sum rel err = {sum_rel:.3g} "
              f"(element order not compared: this build permutes it)")
    if worst > 0.2:
        print("  WARNING: MUL_MAT_ID values disagree beyond quantisation rounding; an "
              "expert or row may have been skipped", file=sys.stderr)
    return worst


def build_answers(block: dict, src_name: str, ms: list[int], id_ms: list[int]) -> dict:
    """The four questions this measurement exists to answer, computed from the data.

    Every ratio is ``source_time / candidate_time``, so > 1 means the candidate is
    faster. Both the median-of-medians and the best-observed (min) are reported: on
    this host the two disagree by up to 15% on the same point, and a conclusion that
    only holds for one of them is not a conclusion.
    """
    def get(table: str, label: str, key: str, m: int):
        entry = block[table].get(label)
        if not entry:
            return None
        return entry[key].get(m)

    answers: dict = {"definition": "ratio = source_ms / candidate_ms; >1 = candidate "
                                   "faster"}
    checks = [("mul_mat", f"q4_K/repack", ms),
              ("mul_mat", f"q4_K/plain", ms),
              ("mul_mat_id", f"q4_K/repack", id_ms),
              ("mul_mat_id", f"q4_K/plain", id_ms)]
    for table, candidate, m_list in checks:
        base_label = f"{src_name}/plain"
        for m in m_list:
            med_base = get(table, base_label, "median_ms_by_m", m)
            med_cand = get(table, candidate, "median_ms_by_m", m)
            min_base = get(table, base_label, "min_ms_by_m", m)
            min_cand = get(table, candidate, "min_ms_by_m", m)
            key = f"{table}:{src_name}/plain->{candidate}@M{m}"
            answers[key] = {
                "median_ratio": (med_base / med_cand
                                 if med_base and med_cand else None),
                "min_ratio": (min_base / min_cand if min_base and min_cand else None),
                "base_ms": med_base, "candidate_ms": med_cand,
            }
    return answers


# ------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=DEFAULT_GGUF / DEFAULT_MODEL,
                    help="GGUF holding the real lm_head / expert tensors")
    ap.add_argument("--tensor", default="output.weight")
    ap.add_argument("--bin", type=Path, default=DEFAULT_BIN,
                    help="directory with ggml-base.dll / ggml-cpu.dll")
    ap.add_argument("--threads", type=int, default=6,
                    help="matches topology-local-32-8.toml:26")
    ap.add_argument("--ms", default="1,2,4,8,16,32",
                    help="hidden-row counts to sweep (lm_head mode)")
    ap.add_argument("--types", default=None,
                    help="source type is kept as-is; others are re-quantised from it. "
                         "lm_head default q6_K,q4_0,q4_K; experts default "
                         "iq4_xs,q4_K,q5_K,q6_K,iq4_nl")
    ap.add_argument("--reps", type=int, default=15,
                    help="rounds; every round touches every M once (interleaved)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--verify", action="store_true",
                    help="cross-check every measured M against a numpy reference")
    ap.add_argument("--buft", choices=("auto", "plain", "repack"), default="auto",
                    help="weight buffer type: auto runs both for types that can be "
                         "repacked and plain only for the rest")
    ap.add_argument("--json", type=Path, default=None, help="write raw points here")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("lm_head", "experts"), default="lm_head",
                    help="lm_head: the output projection (unchanged). experts: a real "
                         "MoE expert bank, MUL_MAT and MUL_MAT_ID")
    ap.add_argument("--expert-layer", type=int, default=None,
                    help="which blk.N to take the expert tensor from (default: first "
                         "IQ4_XS ffn_gate_exps)")
    ap.add_argument("--expert-ms", default="1,2,4,8,16",
                    help="M values for the single-expert MUL_MAT sweep")
    ap.add_argument("--expert-mmid-ms", default="1,8",
                    help="M (token count) values for the MUL_MAT_ID sweep")
    ap.add_argument("--n-used", type=int, default=8,
                    help="fallback n_expert_used when the GGUF metadata lacks it")
    args = ap.parse_args(argv)

    if args.types is None:
        args.types = ("q6_K,q4_0,q4_K" if args.mode == "lm_head"
                      else "iq4_xs,q4_K,q5_K,q6_K,iq4_nl")

    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    if not (args.bin / "ggml-base.dll").is_file():
        raise SystemExit(f"ggml-base.dll not found in {args.bin}")

    g = Ggml(args.bin)
    g.backend = g.cpu.ggml_backend_cpu_init()
    if not g.backend:
        raise SystemExit("ggml_backend_cpu_init failed")
    g.cpu.ggml_backend_cpu_set_n_threads(g.backend, args.threads)
    found = g.find_repack_buft()
    if found:
        print(f"weight repacking available: {found} "
              f"(llama.cpp enables extra buffer types by default; --no-repack is the "
              f"opt-out)")
    else:
        print("no CPU_REPACK buffer type in this build", file=sys.stderr)
    print(f"tiled matmul (GGML_CPU_TILED_MM): "
          f"{'on' if tiled_mm_enabled() else 'off'}")

    if args.mode == "experts":
        report = run_experts(args, g)
    else:
        report = run_lm_head(args, g)

    g.base.ggml_backend_free(g.backend)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
