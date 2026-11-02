/* What read bandwidth can this host actually deliver at n threads?
 *
 * `docs/MOE-SPECULATION-ECONOMICS.md` §6.6 quotes 8 threads moving only
 * 15.7 GB/s (~25 % of DDR5 dual-channel) through `mul_mat_id`, and infers a
 * 2--2.7x dispatch headroom from that.  That inference is only valid if the
 * machine can go faster *at all*.  This probe measures the ceiling with the
 * simplest possible reader: n pthreads, each summing bytes out of its own
 * disjoint buffer (one buffer per thread = one expert bank per thread, the way
 * MUL_MAT_ID splits an expert across threads).
 *
 * Two working-set sizes:
 *   --per-thread 64  (MiB)  DRAM-resident, prefetcher-friendly linear stream
 *   --per-thread 0.53       L3-resident, the per-thread slice of one expert
 *
 * Build/run (MSYS2 ucrt64 gcc):
 *   gcc -O2 -fopenmp -o /tmp/bw checks/probe_bw_ceiling.c && /tmp/bw
 */
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <malloc.h>

static double now_s(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

int main(int argc, char ** argv) {
    double per_thread_mib = 64.0;
    double total_gib = 4.0;
    int max_threads = 8;
    for (int i = 1; i < argc - 1; ++i) {
        if (!strcmp(argv[i], "--per-thread")) per_thread_mib = atof(argv[i + 1]);
        if (!strcmp(argv[i], "--total-gib")) total_gib = atof(argv[i + 1]);
        if (!strcmp(argv[i], "--max-threads")) max_threads = atoi(argv[i + 1]);
    }

    const size_t bytes = (size_t) (per_thread_mib * 1024.0 * 1024.0);
    const size_t words = bytes / sizeof(uint64_t);

    printf("per-thread working set %.2f MiB, total %.2f GiB, %d physical cores\n",
           per_thread_mib, total_gib, omp_get_num_procs());
    printf("%6s %10s %10s %10s\n", "threads", "GB/s", "GiB/s", "us/slice");

    for (int nt = 1; nt <= max_threads; ++nt) {
        uint64_t ** buf = calloc((size_t) nt, sizeof(uint64_t *));
        for (int t = 0; t < nt; ++t) {
            buf[t] = (uint64_t *) _aligned_malloc((words + 16) * sizeof(uint64_t), 64);
            if (!buf[t]) { printf("alloc failed\n"); return 1; }
            for (size_t w = 0; w < words; ++w) buf[t][w] = (uint64_t) (w * 2654435761u);
        }
        const int rounds = (int) (total_gib * 1024.0 * 1024.0 * 1024.0 / (bytes * nt));
        double best = 1e9;
        for (int rep = 0; rep < 3; ++rep) {
            const double t0 = now_s();
            #pragma omp parallel num_threads(nt)
            {
                const int tid = omp_get_thread_num();
                const uint64_t * p = buf[tid];
                uint64_t acc = 0;
                for (int r = 0; r < rounds; ++r) {
                    /* 8 independent accumulators: MLP the way 8 vec_dot rows are */
                    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
                    for (size_t w = 0; w + 8 <= words; w += 8) {
                        a0 += p[w + 0]; a1 += p[w + 1]; a2 += p[w + 2]; a3 += p[w + 3];
                        a4 += p[w + 4]; a5 += p[w + 5]; a6 += p[w + 6]; a7 += p[w + 7];
                    }
                    acc += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
                }
                if (acc == 0xdeadbeefULL) printf(" ");
            }
            const double dt = now_s() - t0;
            if (dt < best) best = dt;
        }
        const double moved = (double) bytes * nt * rounds;
        printf("%6d %10.2f %10.2f %10.2f\n", nt, moved / best / 1e9,
               moved / best / (1024.0 * 1024.0 * 1024.0), best / (double) rounds * 1e6);
        for (int t = 0; t < nt; ++t) _aligned_free(buf[t]);
        free(buf);
    }
    return 0;
}
