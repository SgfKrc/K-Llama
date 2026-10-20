/* Multi-position contract against the production C, sharing deterministic llama stubs. */
#define main scalar_argmax_main
#include "keep_head_argmax_harness.c"
#undef main

static int32_t expected_count, expected_sequence, expected_past, expected_rope;

static void inspect_batch(struct llama_batch batch) {
    if (batch.n_tokens != expected_count) { abort(); }
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i][0] != expected_sequence ||
            batch.logits[i] != 1 || batch.embd[2*i] != 0.0f || batch.embd[2*i+1] != 0.0f) {
            abort();
        }
        for (int32_t j = 0; j < expected_rope; ++j) {
            if (batch.pos[j*batch.n_tokens+i] != expected_past + i) { abort(); }
        }
    }
}

int main(int argc, char ** argv) {
    if (argc < 7) { return 2; }
    rows = atoi(argv[1]);
    columns = atoi(argv[2]);
    decode_rc = atoi(argv[3]);
    missing_row = atoi(argv[4]);
    missing_model = atoi(argv[5]);
    missing_vocab = atoi(argv[6]);
    if (rows < 1 || rows > 8 || columns < 0 || columns > 4 || argc < 7 + rows * columns) {
        return 2;
    }
    int offset = 7;
    for (int i = 0; i < rows * columns; ++i) {
        values[i] = strtof(argv[offset++], NULL);
    }
    int32_t past = 0, count = rows, capacity = rows, rope = 1;
    int32_t seq[8], pos[8], nseq[8];
    const int32_t * seq_ptr = NULL, * pos_ptr = NULL, * nseq_ptr = NULL;
    int null_handle = 0, null_embd = 0, null_out = 0;
    while (offset < argc) {
        const char * option = argv[offset++];
        if (strcmp(option, "--null-handle") == 0) { null_handle = 1; }
        else if (strcmp(option, "--null-embd") == 0) { null_embd = 1; }
        else if (strcmp(option, "--null-out") == 0) { null_out = 1; }
        else if (strcmp(option, "--seq") == 0 || strcmp(option, "--pos") == 0 ||
                 strcmp(option, "--nseq") == 0) {
            if (offset + rows > argc) { return 2; }
            int32_t * dest = strcmp(option, "--seq") == 0 ? seq :
                              strcmp(option, "--pos") == 0 ? pos : nseq;
            for (int i = 0; i < rows; ++i) { dest[i] = (int32_t) strtol(argv[offset++], NULL, 10); }
            if (dest == seq) { seq_ptr = seq; }
            else if (dest == pos) { pos_ptr = pos; }
            else { nseq_ptr = nseq; }
        } else {
            if (offset == argc) { return 2; }
            int32_t value = (int32_t) strtol(argv[offset++], NULL, 10);
            if (strcmp(option, "--past") == 0) { past = value; }
            else if (strcmp(option, "--count") == 0) { count = value; }
            else if (strcmp(option, "--capacity") == 0) { capacity = value; }
            else if (strcmp(option, "--rope") == 0) { rope = value; }
            else { return 2; }
        }
    }
    if (count > rows || capacity > 10) { return 2; }
    qlh_keep_head handle = {0};
    handle.model = missing_model ? NULL : &model;
    handle.ctx = &context;
    handle.n_embd = 2;
    handle.n_layer = 1;
    handle.n_pos_per_embd = rope;
    handle.last_error = 99;
    float hidden[16] = {0};
    /* Canaries on both sides and sentinel data prove no partial publication. */
    int32_t output[12];
    for (int i = 0; i < 12; ++i) { output[i] = 12345; }
    expected_count = count;
    expected_sequence = seq_ptr == NULL ? 0 : seq_ptr[0];
    expected_past = pos_ptr == NULL ? past : pos_ptr[0];
    expected_rope = rope;
    batch_observer = inspect_batch;
    int32_t rc = qlh_kh_forward_embd_tokens(
        null_handle ? NULL : &handle, null_embd ? NULL : hidden, count, past,
        nseq_ptr, seq_ptr, pos_ptr, null_out ? NULL : output + 1, capacity);
    printf("{\"rc\":%d,\"decode_error\":%d,\"allocated\":%d,\"freed\":%d,\"output\":[",
           rc, handle.last_error, allocated_batches, freed_batches);
    for (int i = 0; i < 12; ++i) { printf("%s%d", i == 0 ? "" : ",", output[i]); }
    printf("]}\n");
    return 0;
}
