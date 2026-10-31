import ctypes, os, sys

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
lib.llama_model_n_embd.restype = ctypes.c_int32; lib.llama_model_n_embd.argtypes = [ctypes.c_void_p]
lib.llama_model_n_embd_inp_enc.restype = ctypes.c_int32; lib.llama_model_n_embd_inp_enc.argtypes = [ctypes.c_void_p]
lib.llama_model_n_embd_out.restype = ctypes.c_int32; lib.llama_model_n_embd_out.argtypes = [ctypes.c_void_p]
lib.llama_model_decoder_start_token.restype = ctypes.c_int32; lib.llama_model_decoder_start_token.argtypes = [ctypes.c_void_p]
lib.llama_batch_ext_init.restype = ctypes.c_void_p; lib.llama_batch_ext_init.argtypes = [ctypes.c_void_p]
lib.llama_batch_ext_add_token.restype = ctypes.c_int32; lib.llama_batch_ext_add_token.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
lib.llama_batch_ext_add_embd.restype = ctypes.c_int32; lib.llama_batch_ext_add_embd.argtypes = [ctypes.c_void_p, ctypes.c_int32, Embd]
lib.llama_batch_ext_set_embd_token.restype = ctypes.c_bool; lib.llama_batch_ext_set_embd_token.argtypes = [ctypes.c_void_p, ctypes.c_int32, Embd]
lib.llama_batch_ext_set_embd_state.restype = ctypes.c_bool; lib.llama_batch_ext_set_embd_state.argtypes = [ctypes.c_void_p, ctypes.c_int32, Embd]
lib.llama_batch_ext_set_pos.restype = ctypes.c_bool; lib.llama_batch_ext_set_pos.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
lib.llama_batch_ext_set_output_logits.restype = ctypes.c_bool; lib.llama_batch_ext_set_output_logits.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_bool]
lib.llama_process.restype = ctypes.c_int32; lib.llama_process.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

mode = sys.argv[1] if len(sys.argv) > 1 else "decode_start"

mp = lib.llama_model_default_params()
model = lib.llama_model_load_from_file(MODEL.encode(), mp)
if not model:
    sys.exit(3)
n_enc = lib.llama_model_n_embd_inp_enc(model)
print("[py] n_embd=%d n_embd_inp_enc=%d n_embd_out=%d decoder_start_token=%d" % (
    lib.llama_model_n_embd(model), n_enc, lib.llama_model_n_embd_out(model),
    lib.llama_model_decoder_start_token(model)), flush=True)

cp = lib.llama_context_default_params()
cp.n_ctx = 256; cp.n_batch = 128; cp.n_ubatch = 128
cp.n_threads = 8; cp.n_threads_batch = 8; cp.no_perf = True
ctx = lib.llama_init_from_model(model, cp)
if not ctx:
    sys.exit(4)

BOS = 248044
b = lib.llama_batch_ext_init(ctx)
p0 = (ctypes.c_int32 * 4)(0, 0, 0, 0)

if mode == "one_token":
    i = lib.llama_batch_ext_add_token(b, 0, BOS)
    lib.llama_batch_ext_set_pos(b, i, p0)
    lib.llama_batch_ext_set_output_logits(b, i, True)
elif mode == "token_plus_embd":
    # token id present AND a 16384-wide row attached to the same entry
    i = lib.llama_batch_ext_add_token(b, 0, BOS)
    lib.llama_batch_ext_set_pos(b, i, p0)
    data = (ctypes.c_float * n_enc)()
    e = Embd(data, 1, n_enc)
    print("[py] set_embd_token ->", lib.llama_batch_ext_set_embd_token(b, i, e), flush=True)
    lib.llama_batch_ext_set_output_logits(b, i, True)
elif mode == "token_plus_state":
    # what the fork's original warmup did: token + embd_state of encoder width
    i = lib.llama_batch_ext_add_token(b, 0, BOS)
    lib.llama_batch_ext_set_pos(b, i, p0)
    data = (ctypes.c_float * n_enc)()
    e = Embd(data, 1, n_enc)
    print("[py] set_embd_state ->", lib.llama_batch_ext_set_embd_state(b, i, e), flush=True)
    lib.llama_batch_ext_set_output_logits(b, i, True)
else:
    # warmup's decode leg: 1 token, no state, no embd
    i = lib.llama_batch_ext_add_token(b, 0, BOS)
    lib.llama_batch_ext_set_pos(b, i, p0)
    lib.llama_batch_ext_set_output_logits(b, i, True)

print("[py] llama_process ENCODE (%s) ..." % mode, flush=True)
print("[py] rc =", lib.llama_process(ctx, 0, b), flush=True)
print("[py] DONE OK", flush=True)
