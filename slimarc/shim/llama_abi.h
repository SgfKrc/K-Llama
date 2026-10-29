/* ABI shim for llama_context_params.
 *
 * Why this file exists: the shipped llama.dll is a QLH-patched build whose llama.h
 * is no longer on this machine, and the only llama.h available comes from a
 * llama-cpp-python wheel built from a *newer* experimental branch. That header puts
 * three extra 32-bit fields before n_threads, so a shim compiled against it writes
 * n_threads into the DLL's `ctx_type` slot and context creation fails with
 * "Unsupported ctx type".
 *
 * The offsets below were measured, not guessed (build-headers/probe_layout.py reads
 * llama_context_default_params() out of the real DLL and the known defaults show up
 * exactly here):
 *
 *   offset  0  n_ctx              512
 *   offset  4  n_batch           2048
 *   offset  8  n_ubatch           512
 *   offset 12  n_seq_max            1
 *   offset 16  (unknown)            0
 *   offset 20  (unknown)            0
 *   offset 24  n_threads            4
 *   offset 28  n_threads_batch      4
 *
 * Only these six fields are ever written. Everything else must stay exactly as
 * llama_context_default_params() filled it, so the buffer is deliberately oversized:
 * the true struct size is unknown (somewhere between 96 and 512 bytes) and an
 * over-long buffer is harmless while a short one would truncate the tail that
 * llama_init_from_model reads.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#define QLH_CTX_PARAMS_BYTES 512

typedef struct {
    uint32_t      n_ctx;            /* 0  */
    uint32_t      n_batch;          /* 4  */
    uint32_t      n_ubatch;         /* 8  */
    uint32_t      n_seq_max;        /* 12 */
    uint32_t      unknown_16;       /* 16 -- left untouched */
    uint32_t      unknown_20;       /* 20 -- left untouched */
    int32_t       n_threads;        /* 24 -- 4 bytes earlier than stock llama.h */
    int32_t       n_threads_batch;  /* 28 */
    unsigned char rest[QLH_CTX_PARAMS_BYTES - 32];
} qlh_ctx_params;

/* If any of these move, the dent above is wrong and every context silently
 * misconfigures itself, so fail the build instead. */
_Static_assert(sizeof(qlh_ctx_params) == QLH_CTX_PARAMS_BYTES, "ctx params size drift");
_Static_assert(offsetof(qlh_ctx_params, n_threads) == 24, "n_threads must sit at 24");
_Static_assert(offsetof(qlh_ctx_params, n_threads_batch) == 28, "n_threads_batch must sit at 28");
