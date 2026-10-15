/* Execute the production shim against deterministic llama outputs, without weights. */
#include "../../scripts/model_tools/keep_head_shim/qlh_keep_head.c"

struct llama_model { int unused; };
struct llama_context { int unused; };
struct llama_vocab { int unused; };

static struct llama_model model;
static struct llama_context context;
static struct llama_vocab vocabulary;
static float values[32];
static int rows, columns, decode_rc, missing_row, missing_model, missing_vocab;
static int allocated_batches, freed_batches;

struct llama_model_params llama_model_default_params(void) {
    struct llama_model_params params = {0};
    return params;
}
struct llama_context_params llama_context_default_params(void) {
    struct llama_context_params params = {0};
    return params;
}
struct llama_model * llama_model_load_from_file(const char * path, struct llama_model_params params) {
    abort();
}
struct llama_context * llama_init_from_model(struct llama_model * ptr, struct llama_context_params params) {
    abort();
}
void llama_model_free(struct llama_model * ptr) { abort(); }
void llama_free(struct llama_context * ptr) { abort(); }
uint32_t llama_n_ctx(const struct llama_context * ptr) { return 32; }
uint32_t llama_n_batch(const struct llama_context * ptr) { return 4; }
int32_t llama_model_n_layer(const struct llama_model * ptr) { return 1; }
int32_t llama_model_n_embd(const struct llama_model * ptr) { return 2; }
enum llama_rope_type llama_model_rope_type(const struct llama_model * ptr) { return LLAMA_ROPE_TYPE_NORM; }
void llama_set_embeddings_layer_inp(struct llama_context * ptr, uint32_t layer, bool enabled) { abort(); }
float * llama_get_embeddings_layer_inp(struct llama_context * ptr, uint32_t layer) { abort(); }
const struct llama_model * llama_get_model(const struct llama_context * ptr) {
    return missing_model ? NULL : &model;
}
const struct llama_vocab * llama_model_get_vocab(const struct llama_model * ptr) {
    return missing_vocab ? NULL : &vocabulary;
}
int32_t llama_vocab_n_tokens(const struct llama_vocab * ptr) { return columns; }
llama_memory_t llama_get_memory(const struct llama_context * ptr) { return NULL; }
void llama_memory_clear(llama_memory_t memory, bool data) { abort(); }

struct llama_batch llama_batch_init(int32_t count, int32_t width, int32_t seq_max) {
    struct llama_batch batch = {0};
    batch.n_tokens = count;
    batch.token = calloc((size_t) count, sizeof(*batch.token));
    batch.embd = calloc((size_t) count * (size_t) width, sizeof(*batch.embd));
    batch.pos = calloc((size_t) count, sizeof(*batch.pos));
    batch.n_seq_id = calloc((size_t) count, sizeof(*batch.n_seq_id));
    batch.seq_id = calloc((size_t) count, sizeof(*batch.seq_id));
    batch.logits = calloc((size_t) count, sizeof(*batch.logits));
    for (int32_t i = 0; i < count; ++i) {
        batch.seq_id[i] = calloc((size_t) seq_max, sizeof(**batch.seq_id));
    }
    ++allocated_batches;
    return batch;
}
void llama_batch_free(struct llama_batch batch) {
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        free(batch.seq_id[i]);
    }
    free(batch.token);
    free(batch.embd);
    free(batch.pos);
    free(batch.n_seq_id);
    free(batch.seq_id);
    free(batch.logits);
    ++freed_batches;
}
int32_t llama_decode(struct llama_context * ptr, struct llama_batch batch) { return decode_rc; }
float * llama_get_logits_ith(struct llama_context * ptr, int32_t row) {
    return row == missing_row || row < 0 || row >= rows ? NULL : values + row * columns;
}

int main(int argc, char ** argv) {
    if (argc < 8) { return 2; }
    rows = atoi(argv[1]);
    columns = atoi(argv[2]);
    decode_rc = atoi(argv[3]);
    missing_row = atoi(argv[4]);
    missing_model = atoi(argv[5]);
    missing_vocab = atoi(argv[6]);
    if (rows < 1 || rows > 4 || columns < 0 || columns > 8 || argc != 7 + rows * columns) {
        return 2;
    }
    for (int i = 0; i < rows * columns; ++i) {
        values[i] = strtof(argv[7 + i], NULL);
    }
    qlh_keep_head handle = {0};
    handle.model = missing_model ? NULL : &model;
    handle.ctx = &context;
    handle.n_embd = 2;
    handle.n_layer = 1;
    handle.n_pos_per_embd = 1;
    float hidden[8] = {0};
    int32_t token = 12345;
    const int32_t rc = qlh_kh_forward_embd_token(&handle, hidden, rows, 0, NULL, NULL, NULL, &token);
    printf("{\"rc\":%d,\"token\":%d,\"decode_error\":%d,\"guard\":%d,\"n_vocab\":%d,"
           "\"null_n_vocab\":%d,\"allocated\":%d,\"freed\":%d}\n",
           rc, token, qlh_kh_last_error(&handle), qlh_kh_finite_logits_guard(),
           qlh_kh_n_vocab(&handle), qlh_kh_n_vocab(NULL), allocated_batches, freed_batches);
    return 0;
}
