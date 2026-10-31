import ctypes, os, sys, time

BIN = r"H:\llamacpp-c\build-spec\bin"
MODEL = r"H:\qlh_models\dflash\dflash-merged-Q4_K_M.gguf"
os.add_dll_directory(BIN)
lib = ctypes.CDLL(os.path.join(BIN, "llama.dll"))

class ModelParams(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.c_void_p),
        ("tensor_buft_overrides", ctypes.c_void_p),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int),
        ("load_mode", ctypes.c_int),
        ("lazy_mode", ctypes.c_int),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.c_void_p),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
        ("load_mtp", ctypes.c_bool),
    ]

class ContextParams(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_rs_seq", ctypes.c_uint32),
        ("n_outputs_max", ctypes.c_uint32),
        ("n_outputs_max_per_seq", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("ctx_type", ctypes.c_int),
        ("rope_scaling_type", ctypes.c_int),
        ("pooling_type", ctypes.c_int),
        ("attention_type", ctypes.c_int),
        ("flash_attn_type", ctypes.c_int),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int),
        ("type_v", ctypes.c_int),
        ("moe_cache_size", ctypes.c_size_t),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("embeddings_pre_norm", ctypes.c_bool),
        ("samplers", ctypes.c_void_p),
        ("n_samplers", ctypes.c_size_t),
        ("ctx_other", ctypes.c_void_p),
    ]

class Embd(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_float)),
                ("n_rows", ctypes.c_size_t),
                ("n_embd", ctypes.c_size_t)]

lib.llama_backend_init.restype = None
lib.llama_backend_init.argtypes = []
lib.llama_model_default_params.restype = ModelParams
lib.llama_model_default_params.argtypes = []
lib.llama_context_default_params.restype = ContextParams
lib.llama_context_default_params.argtypes = []
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
lib.llama_n_ctx.restype = ctypes.c_uint32
lib.llama_n_ctx.argtypes = [ctypes.c_void_p]

mode = sys.argv[1] if len(sys.argv) > 1 else "token"

print("[py] loading model", flush=True)
mp = lib.llama_model_default_params()
t0 = time.time()
model = lib.llama_model_load_from_file(MODEL.encode(), mp)
print("[py] model =", hex(model or 0), "t=%.2fs" % (time.time() - t0), flush=True)
if not model:
    sys.exit(3)

has_enc = lib.llama_model_has_encoder(model)
has_dec = lib.llama_model_has_decoder(model)
n_embd  = lib.llama_model_n_embd(model)
n_enc   = lib.llama_model_n_embd_inp_enc(model)
print("[py] has_encoder=%d has_decoder=%d n_embd=%d n_embd_inp_enc=%d" % (has_enc, has_dec, n_embd, n_enc), flush=True)

cp = lib.llama_context_default_params()
print("[py] cparams sizeof:", ctypes.sizeof(ContextParams), "n_ctx=%d n_batch=%d n_ubatch=%d" % (cp.n_ctx, cp.n_batch, cp.n_ubatch), flush=True)
cp.n_ctx = 256
cp.n_batch = 128
cp.n_ubatch = 128
cp.n_threads = 8
cp.n_threads_batch = 8
cp.no_perf = True
print("[py] creating context", flush=True)
t0 = time.time()
ctx = lib.llama_init_from_model(model, cp)
print("[py] ctx =", hex(ctx or 0), "t=%.2fs" % (time.time() - t0), "n_ctx=", lib.llama_n_ctx(ctx) if ctx else -1, flush=True)
if not ctx:
    sys.exit(4)

batch = lib.llama_batch_ext_init(ctx)
print("[py] batch =", hex(batch or 0), flush=True)

if mode == "token":
    # replicate common_batch_get_one(): token entries, no embedding, last one outputs
    ids = [248044, 248046]  # bos, eos
    for i, tid in enumerate(ids):
        pos = (ctypes.c_int32 * 4)(i, 0, 0, 0)
        idx = lib.llama_batch_ext_add_token(batch, 0, tid)
        print("[py] add_token", tid, "->", idx, flush=True)
        lib.llama_batch_ext_set_pos(batch, idx, pos)
        if i == len(ids) - 1:
            lib.llama_batch_ext_set_output_logits(batch, idx, True)
else:
    # embd row of width n_embd_inp_enc (16384), the encoder's real input
    n = n_enc if n_enc > 0 else n_embd
    data = (ctypes.c_float * n)(*([0.0] * n))
    e = Embd(data, 1, n)
    idx = lib.llama_batch_ext_add_embd(batch, 0, e)
    print("[py] add_embd ->", idx, "width", n, flush=True)
    pos = (ctypes.c_int32 * 4)(0, 0, 0, 0)
    lib.llama_batch_ext_set_pos(batch, idx, pos)

print("[py] calling llama_process ENCODE ...", flush=True)
rc = lib.llama_process(ctx, 0, batch)
print("[py] ENCODE rc =", rc, flush=True)

print("[py] calling llama_process DECODE ...", flush=True)
batch2 = lib.llama_batch_ext_init(ctx)
pos = (ctypes.c_int32 * 4)(2, 0, 0, 0)
if mode == "token":
    idx = lib.llama_batch_ext_add_token(batch2, 0, 248044)
    lib.llama_batch_ext_set_pos(batch2, idx, pos)
    lib.llama_batch_ext_set_output_logits(batch2, idx, True)
rc = lib.llama_process(ctx, 1, batch2)
print("[py] DECODE rc =", rc, flush=True)
print("[py] DONE OK", flush=True)
