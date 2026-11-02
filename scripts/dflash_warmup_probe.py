import ctypes, os, sys, time

BIN = r"H:\llamacpp-c\build-spec\bin"
MODEL = r"H:\qlh_models\dflash\dflash-merged-Q4_K_M.gguf"
os.add_dll_directory(BIN)
lib = ctypes.CDLL(os.path.join(BIN, "llama.dll"))

class ModelParams(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.c_void_p), ("tensor_buft_overrides", ctypes.c_void_p),
        ("n_gpu_layers", ctypes.c_int32), ("split_mode", ctypes.c_int),
        ("load_mode", ctypes.c_int), ("lazy_mode", ctypes.c_int),
        ("main_gpu", ctypes.c_int32), ("tensor_split", ctypes.c_void_p),
        ("progress_callback", ctypes.c_void_p), ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool), ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool), ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool), ("load_mtp", ctypes.c_bool),
    ]

class ContextParams(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32), ("n_batch", ctypes.c_uint32), ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32), ("n_rs_seq", ctypes.c_uint32),
        ("n_outputs_max", ctypes.c_uint32), ("n_outputs_max_per_seq", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32), ("n_threads_batch", ctypes.c_int32),
        ("ctx_type", ctypes.c_int), ("rope_scaling_type", ctypes.c_int),
        ("pooling_type", ctypes.c_int), ("attention_type", ctypes.c_int),
        ("flash_attn_type", ctypes.c_int),
        ("rope_freq_base", ctypes.c_float), ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float), ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float), ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32), ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p), ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int), ("type_v", ctypes.c_int),
        ("moe_cache_size", ctypes.c_size_t),
        ("abort_callback", ctypes.c_void_p), ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool), ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool), ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool), ("kv_unified", ctypes.c_bool),
        ("embeddings_pre_norm", ctypes.c_bool),
        ("samplers", ctypes.c_void_p), ("n_samplers", ctypes.c_size_t),
        ("ctx_other", ctypes.c_void_p),
    ]

class Embd(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_float)), ("n_rows", ctypes.c_size_t), ("n_embd", ctypes.c_size_t)]

lib.llama_backend_init.restype = None
lib.llama_model_default_params.restype = ModelParams
lib.llama_context_default_params.restype = ContextParams
lib.llama_model_load_from_file.restype = ctypes.c_void_p
lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, ModelParams]
lib.llama_init_from_model.restype = ctypes.c_void_p
lib.llama_init_from_model.argtypes = [ctypes.c_void_p, ContextParams]
lib.llama_model_has_encoder.restype = ctypes.c_bool
lib.llama_model_has_encoder.argtypes = [ctypes.c_void_p]
lib.llama_model_has_decoder.restype = ctypes.c_bool
lib.llama_model_has_decoder.argtypes = [ctypes.c_void_p]
lib.llama_model_n_embd.restype = ctypes.c_int32
lib.llama_model_n_embd.argtypes = [ctypes.c_void_p]
lib.llama_model_n_embd_inp_enc.restype = ctypes.c_int32
lib.llama_model_n_embd_inp_enc.argtypes = [ctypes.c_void_p]
lib.llama_batch_ext_init.restype = ctypes.c_void_p
lib.llama_batch_ext_init.argtypes = [ctypes.c_void_p]
lib.llama_batch_ext_add_token.restype = ctypes.c_int32
lib.llama_batch_ext_add_token.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
lib.llama_batch_ext_add_embd.restype = ctypes.c_int32
lib.llama_batch_ext_add_embd.argtypes = [ctypes.c_void_p, ctypes.c_int32, Embd]
lib.llama_batch_ext_set_pos.restype = ctypes.c_bool
lib.llama_batch_ext_set_pos.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
lib.llama_batch_ext_set_output_logits.restype = ctypes.c_bool
lib.llama_batch_ext_set_output_logits.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_bool]
lib.llama_process.restype = ctypes.c_int32
lib.llama_process.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
lib.llama_memory_clear.restype = None
lib.llama_memory_clear.argtypes = [ctypes.c_void_p, ctypes.c_bool]
lib.llama_get_memory.restype = ctypes.c_void_p
lib.llama_get_memory.argtypes = [ctypes.c_void_p]

mode = sys.argv[1] if len(sys.argv) > 1 else "skip_enc"

mp = lib.llama_model_default_params()
model = lib.llama_model_load_from_file(MODEL.encode(), mp)
if not model:
    print("[py] model load failed"); sys.exit(3)
n_embd = lib.llama_model_n_embd(model)
n_enc  = lib.llama_model_n_embd_inp_enc(model)
print("[py] has_encoder=%d has_decoder=%d n_embd=%d n_embd_inp_enc=%d (can_encode_tokens=%s)" % (
    lib.llama_model_has_encoder(model), lib.llama_model_has_decoder(model), n_embd, n_enc,
    n_enc == n_embd), flush=True)

cp = lib.llama_context_default_params()
cp.n_ctx = 256; cp.n_batch = 128; cp.n_ubatch = 128
cp.n_threads = 8; cp.n_threads_batch = 8; cp.no_perf = True
ctx = lib.llama_init_from_model(model, cp)
if not ctx:
    print("[py] ctx failed"); sys.exit(4)
print("[py] ctx ok", flush=True)

BOS, EOS = 248044, 248046

def add_tok(b, tid, pos):
    p = (ctypes.c_int32 * 4)(pos, 0, 0, 0)
    i = lib.llama_batch_ext_add_token(b, 0, tid)
    lib.llama_batch_ext_set_pos(b, i, p)
    return i

if mode == "skip_enc":
    # what warmup does after the fix: no encode, just decode the BOS+EOS pair
    b = lib.llama_batch_ext_init(ctx)
    i = add_tok(b, BOS, 0); lib.llama_batch_ext_set_output_logits(b, i, False)
    i = add_tok(b, EOS, 1); lib.llama_batch_ext_set_output_logits(b, i, True)
    print("[py] DECODE [bos,eos] ...", flush=True)
    print("[py] rc =", lib.llama_process(ctx, 1, b), flush=True)
elif mode == "enc_embd":
    # alternative fix: feed the encoder the 16384-wide feature rows it expects
    b = lib.llama_batch_ext_init(ctx)
    data = (ctypes.c_float * (n_enc * 2))()
    e0 = Embd(data, 1, n_enc)
    e1 = Embd(ctypes.cast(ctypes.byref(data, n_enc * 4), ctypes.POINTER(ctypes.c_float)), 1, n_enc)
    i = lib.llama_batch_ext_add_embd(b, 0, e0)
    lib.llama_batch_ext_set_pos(b, i, (ctypes.c_int32 * 4)(0, 0, 0, 0))
    i = lib.llama_batch_ext_add_embd(b, 0, e1)
    lib.llama_batch_ext_set_pos(b, i, (ctypes.c_int32 * 4)(1, 0, 0, 0))
    print("[py] ENCODE zero features ...", flush=True)
    print("[py] rc =", lib.llama_process(ctx, 0, b), flush=True)
    b2 = lib.llama_batch_ext_init(ctx)
    i = add_tok(b2, BOS, 2); lib.llama_batch_ext_set_output_logits(b2, i, True)
    print("[py] DECODE start token ...", flush=True)
    print("[py] rc =", lib.llama_process(ctx, 1, b2), flush=True)

# post-warmup: a real prompt decode + 2 generated tokens, like llama-cli does
lib.llama_memory_clear(lib.llama_get_memory(ctx), True)
prompt = [248045, 872, 248046, 198, 248045, 74455, 198]  # arbitrary prompt tokens
b3 = lib.llama_batch_ext_init(ctx)
for i, tid in enumerate(prompt):
    j = add_tok(b3, tid, i)
    lib.llama_batch_ext_set_output_logits(b3, j, i == len(prompt) - 1)
print("[py] DECODE prompt (%d toks) ..." % len(prompt), flush=True)
print("[py] rc =", lib.llama_process(ctx, 1, b3), flush=True)

for k in range(2):
    b4 = lib.llama_batch_ext_init(ctx)
    j = add_tok(b4, 872, len(prompt) + k)
    lib.llama_batch_ext_set_output_logits(b4, j, True)
    print("[py] DECODE gen token %d rc =" % k, lib.llama_process(ctx, 1, b4), flush=True)

print("[py] DONE OK (%s)" % mode, flush=True)
