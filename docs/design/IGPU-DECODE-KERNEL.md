# Intel iGPU decode GEMV：独立 L3 kernel 设计

> 状态：**方案**
> 更新日期：2026-11-14
> 适用范围：Intel Gen12/Gen11 核显上 Vulkan 后端的 decode（n=1）GEMV 路径；不含 prefill/GEMM，不含 CPU 后端。

标注约定：**[实测]** 取自既有 benchmark；**[代码]** 由本仓库源码读出；**[推算]** 由公式算出，假设写在旁边。

---

## 1 结论

1. **瓶颈是 DRAM 并发度不足，不是归约、cache 或指令吞吐。** 五条假设的判定见 §2：只有「占用率不足」与「访存指令冗余」成立，其余三条排除。
2. **L3 目标 kernel 是两个**：`mul_mat_vec_q4_k`（decode 权重占比最大）与 `mul_mat_vec_q6_k`（单位字节最慢，且带 per-superblock barrier）。
3. **L3 的核心不是重写归约，而是重排权重布局**：把 Q4_K 的 144 B superblock 转置成 lane-major，使每个 lane 一次 `uvec4` 读回 4 个 superblock 的数据，把每 lane 每次迭代的 outstanding 字节从 9 B 提到 64 B。
4. **预期 6.31 → 8.5–8.8 t/s（+35~40%）**，算式见 §3.5。
5. **L1 若做对（循环展开 + 解除硬编码 16 线程/block），可拿到其中 70–80%**；L3 的净增量集中在 Q6_K（去掉每 superblock 的 barrier，约 +15~25% 单项）与指令槽余量。增量拆分见 §4.1。

---

## 2 依据：60% 带宽的五条假设逐条判定

基准口径 **[实测]**：9B Q4_K_M，5.24 GiB，`-t 8 -r 5`，`-ngl 99`，tg 6.31 t/s（中位数）；GPU 计算 148.9 ms/token，总 158.5 ms/token，CPU 开销 9.5 ms/token；每 token 必读权重 5.098 GB（blk 4065.6 MiB + output 795.7 MiB，token_embd 不读）。
反推带宽 5.098 GB / 148.9 ms = **34.2 GB/s**，对实测流式上限 58.2 GB/s 为 **59%**。算术强度 2 FLOP/B，故「计算效率 6%」不构成问题。

| 假设 | 判定 | 一句话依据 |
|---|---|---|
| a 归约开销 | **排除** | HYBRID 归约每 wg 仅 1 次 barrier，占总时间 <0.3% |
| b 占用率 / workgroup | **成立（主因）** | SIMD16 下每 EU 只驻留 1 个 warp，in-flight 仅达延迟带宽积的 63% |
| c 访存模式 | **成立（次因）** | 每 superblock 96 条 lane-load 只为 144 B 有效数据，冗余 2.7× |
| d L2/L3 cache | **排除** | 权重每 token 只读一遍，命中率≈0；唯一可复用的 b 向量占比 <2% |
| e 指令效率 | **排除** | 指令槽占 14%，不是当前限制，但构成 L3 的天花板 |

### 2.1 a 归约开销：排除

**[代码]** Intel + Q4_K + m=4096 走 `DMMV_WG_SIZE_LARGE`（`ggml-vulkan.cpp:5917-5929`：Intel 且 `m<=8192 && k>=1024`），`wg=subgroup_size16*4=64`（`:3126`），reduction 取 `SHADER_REDUCTION_MODE_HYBRID`（`:3129`），即 `mul_mat_vec_base.glsl:141-191` 的 `USE_SUBGROUP_ADD` 分支：先 `subgroupAdd`，经 shared 汇总 4 个 subgroup 的部分和，**全 kernel 只有 1 次 `barrier()`（`:157`）**，随后仅 tid==0 写 2 个 float。

**[推算]** 2048 个 wg（m=4096 / NUM_ROWS=2）× 约 100 cycle（barrier 50 + 4 步 shuffle + 串行加）= 204 800 cycle，摊到 96 EU 为 2133 cycle ≈ **1.5 us**，占单 kernel 723.9 us 的 0.2%。纯 shmem 树（`:200-209`，6 次 barrier）在 Intel 上根本不会被选中。

**Q6_K 是例外**，见 §3.4：它的 barrier 不在归约里，而在每 superblock 的解包中。

### 2.2 b workgroup 与占用率：成立（主因）

**[代码]** `ggml-vulkan.cpp:3108` 在 Intel 且支持 subgroup size control 时**强制 subgroup 16**（`subgroup_min_size<=16 && subgroup_max_size>=16`）。Gen12 原生 `subgroupSize=32`，`minSubgroupSize=8` **[实测]**，所以这是主动选择而非硬件限制。后果是 shader 按 **SIMD16** 编译：每个 float 变量占 2 个 GRF。

**[代码]** `mul_mat_vec_q4_k.comp:94-117`：`it_size = gl_WorkGroupSize.x/16 = 4`，每个 superblock 由 16 lane 协作，每 lane 只负责 9 字节；`k=12288` 时 `num_blocks_per_row=48`，每个 16-lane 组要串行迭代 12 个 superblock。

**[推算]** 用 Little 定律定量：iGPU 上 L2 miss 的 DRAM 延迟取 500 ns（400–600 ns 区间），要跑满 58.2 GB/s 需在飞 **29.1 KB**。
实际在飞：每 SIMD16 每次迭代发 6 条 load（qs×2、scales×3、dm×1），其中 scales 三条命中同一 sector、dm 与 scales 同 line，取回约 3 个 64 B line ≈ 192 B；q4_K 的活跃变量（16 个 q + 8 个 sc + 4 个 `vec4` b + dm + 中间值）约 45–60 GRF/线程，SIMD16 下即 90–120 GRF，**每 EU 只能驻留 1 个 SIMD16**（128 GRF/EU），占用率 16/64 lane = 25%。
在飞 = 96 EU × 192 B = **18.4 KB**，为所需 29.1 KB 的 **63%**——与实测 59% 吻合。

**[代码]** 96 EU / 4 EU-per-wg = 24 个 workgroup 同时驻留，每 wg 读 2×6912 B = 13.8 KB，一批 332 KB；批次数 2048/24 = 85。批与批之间没有跨批预取，这正是并发度的天花板。

### 2.3 c 访存模式：成立（次因）

**[代码]** `types.glsl:392-413`：`block_q4_K` = `f16vec2 dm` + `uint8_t scales[12]` + `uint8_t qs[128]` = 144 B。shader 用 `block_q4_K_packed16` / `packed32` 视图访问。

**[代码]** `mul_mat_vec_q4_k.comp:19-21` 每个线程读 3 个 `uint32` 覆盖 12 B 的 scales，但 `v_im = il/2`（`:103`）只有 0/1 两个取值，**16 个 lane 只请求 2 个不同地址**；`:17` 的 `dm` 更极端，16 lane 请求同一地址。于是每 superblock 的有效 144 B 由 16×6 = 96 条 lane-load 拼出，冗余 2.7×（144/4 = 36 个 `uint32` 的理想值）。

对齐方面：superblock 起始偏移 `i*144`，模 64 为 0/16/32/48，单个 superblock 跨 3 个 64 B line 而非 2.25 个；但整行 6912 B 是连续的，相邻 superblock 共享 line，**流式读的 DRAM 字节没有浪费**。所以对齐不是 DRAM 效率问题，只是 load 指令效率问题。

### 2.4 d L2/L3 cache：排除

**[推算]** 5.098 GB 权重每 token 只读一遍，Gen12 iGPU 的 L3 为共享 slice（MB 级，具体容量未在本机测量），对 5 GB 流式工作集命中率≈0，优化空间为零。唯一有复用价值的是 hidden 向量 b（12288×4 B = 48 KB），被 2048 个 wg 各读一遍，逻辑流量 98 MB，全部应在 L2 命中；即使全 miss 也只占 5.098 GB 的 1.9%。
`output`（lm_head，795.7 MiB，Q6_K）与 blk 一样每 token 读一遍，无跨 token 复用；`token_embd` 不参与 decode 的矩阵乘。**结论：cache 维度的任何优化都不值得做，收益必须来自并发度。**

### 2.5 e 指令效率：排除（但它是天花板）

**[代码]** `mul_mat_vec_q4_k.comp:19-82` 每 lane 每 superblock 约 78 条指令：6 条 load、scale 解包 ~17 条（含 3 次 `unpack8`）、qs 解包 ~16 条（4 次 `unpack8`）、b 的 4 条 `vec4` load、35 条 FMA（sx/sy/sz/sw 各 4 + smin 16 + 收尾 3）。

**[推算]** 每 token 35.4M 个 superblock（5.098 GB / 144 B）× 78 条 SIMD16 指令 = 2.76 G 指令；硬件 96 EU × 1.4 GHz = 134 G 指令/s，即 **20.6 ms/token**，占 148.9 ms 的 **14%**。所以指令不是当前瓶颈；但若带宽提到 85%（110 ms），指令仍只占 19%，不构成新瓶颈——**除非**改用更小的 subgroup 或去掉布局重排，那会把这个比例推上去。

---

## 3 方案

### 3.1 目标 kernel 与现状

| kernel | 现状耗时 [实测] | 带宽 [推算] | 结构问题 |
|---|---|---|---|
| `mul_mat_vec_q4_k`，m=4096 n=1 k=12288 | 723.9 us | 28.3 MB/723.9 us = 39.1 GB/s（按 0.5625 B/w；perf logger 口径给 34.8 GB/s，两者同档） | §2.2 占用率 + §2.3 冗余 load |
| `mul_mat_vec_q6_k`，m=4096 n=1 k=12288 | 1148.6 us | 41.3 MB/1148.6 us = 36.0 GB/s | 每 superblock 一次 `barrier()` |

**[代码]** Q6_K 在 `m=4096` 时不满足 `m<4096`（`ggml-vulkan.cpp:5921-5924`），落到 `DMMV_WG_SIZE_SUBGROUP`，`wg=16`（1 个 subgroup），归约走 `USE_SUBGROUP_ADD_NO_SHMEM`（`mul_mat_vec_base.glsl:96-134`，零 barrier）。但 `mul_mat_vec_q6_k.comp:9` 声明 `shared sccache[2][BLOCK_SIZE/16][16]`，`:55-56` 每个 superblock 写一次 shared 并 `barrier()`，`:74` 再从 shared 读 `s_offset..s_offset+6`。`k=12288` 时每 wg 96 次 barrier，**它把 load 的软件流水切断**，这才是 Q6_K 慢 30.6% 的原因，与 MMVQ 无关（MMVQ 在本机 −5.7%，`ggml-vulkan.cpp:7013-7018` 对 Intel Windows 专有驱动直接 `return false`）。

### 3.2 权重布局重排（L3 的支点）

加载时把每个 Q4_K 行按 4 个 superblock 一组转置，写进一个独立 buffer：

```
Q[bg][lane][0..3]    uint32   // Q[bg][lane][j] = qs(block 4*bg+j) 的第 lane 个 uint32
```
32 lane × 16 B = 512 B = 4 × 128 B，一次连续读，零冗余、零填充。
scales 单独重排为每 superblock 8 个 `uint16`（`sc | m<<6`），每块 16 B，只让 lane 0..7 读，再用 `subgroupShuffle` 广播；相对原始 12 B 有 4 B 填充，但每块省下 2 条 load 指令且消除 16 路广播冗余，总字节 4×128 + 4×16 = 576 B，与原 4×144 = 576 B 相等，**不膨胀**。

落地位置 **[代码]**：在 `ggml_vk_create_tensor`（`ggml-vulkan.cpp`）的 buffer 分配处挂一个 `q4k_gemv` 附属 buffer，仅当 `ne0 % 256 == 0 && type == GGML_TYPE_Q4_K && 使用该 tensor 的 MUL_MAT 的 n <= mul_mat_vec_max_cols` 时生成；`ggml_vk_mul_mat_vec_q_f16`（`:7042`）在选 pipeline 前检查附属 buffer 是否存在，存在则走 L3 路径、否则走现有 `ggml_vk_get_dequantize_mul_mat_vec`（`:5853`）。prepack 本身用一次 `vkCmdCopyBuffer` 之后的小 compute shader 完成（一次 5 GB 量级搬运，见 §4.3）。

### 3.3 GLSL 结构

```glsl
layout(local_size_x = 32) in;                 // 1 个 native subgroup，零 barrier
layout(constant_id = 0) const uint NUM_ROWS = 2;
layout(binding = 0) readonly  buffer Q  { uvec4  qs4[]; };   // prepack: 4 个 superblock 交织
layout(binding = 1) readonly  buffer SC { uint16_t scm[]; }; // prepack: sc | m<<6
layout(binding = 2) readonly  buffer B  { vec4    b4[];  };  // hidden，L2 命中
layout(binding = 3) readonly  buffer DM { vec2    dm[];  };  // prepack: d, dmin
layout(binding = 4) writeonly buffer D  { float   out[]; };

void main() {
    const uint lane = gl_SubgroupInvocationID;  // 0..31
    const uint g    = lane / 4;                 // 子块组 0..7
    const uint l0   = 4 * (lane % 4);           // 0,4,...,28
    const uint row0 = NUM_ROWS * (gl_WorkGroupID.x + gl_NumWorkGroups.x * gl_WorkGroupID.z);
    float acc[NUM_ROWS] = {0.0, 0.0};

    for (uint r = 0; r < NUM_ROWS; ++r) {
        for (uint bg = 0; bg < nbg; ++bg) {          // nbg = 48/4 = 12
            const uvec4 q = qs4[(row0+r)*QB4 + bg*32 + lane];  // 1 条 16 B = 4 个 superblock
            const vec2  d = dm[(row0+r)*DB + bg*4];            // block 级，广播 load
            [[unroll]] for (uint j = 0; j < 4; ++j) {
                const uint kb = ((bg*4 + j) * 256 + g*32 + l0) >> 2;  // vec4 索引
                const vec4 blo = b4[kb];                 // k 方向 4 个，L2 命中
                const vec4 bhi = b4[kb + 4];             // +16 个元素
                const vec4 lo4 = vec4(unpack8( q[j]        & 0x0F0F0F0F));
                const vec4 hi4 = vec4(unpack8((q[j] >> 4)  & 0x0F0F0F0F));
                const uint s16 = subgroupShuffle(lane < 8 ? scm[(row0+r)*SB + (bg*4 + j)*8 + lane] : 0u, g);
                const float sq = dot(blo, lo4) + dot(bhi, hi4);
                const float sb = dot(blo, vec4(1.0)) + dot(bhi, vec4(1.0));
                acc[r] = fma(d.x * float(s16 & 63u), sq, acc[r])
                       - d.y * float((s16 >> 6) & 63u) * sb;
            }
        }
    }
    [[unroll]] for (uint r = 0; r < NUM_ROWS; ++r) acc[r] = subgroupAdd(acc[r]);
    if (lane == 0) { for (uint r = 0; r < NUM_ROWS; ++r) out[d0 + r] = acc[r]; }
}
```

要点与理由：

- **wg=32、单 subgroup**：绕开 `:3108` 的强制 16，归约退化为 5 步 `subgroupAdd`，**无 barrier、无 shared**。这同时消除了 §2.2 的 SIMD16 寄存器翻倍。
- **每 lane 一次 `uvec4`**：DRAM 侧每迭代每 lane 一次 16 B load（原实现 9 B 但要 6 条指令），另有 48 B 的 b 落在 L2 不占 DRAM。这是 §2.2 的 Little 定律直接解法：in-flight = 96 EU × 32 lane × 16 B = 49 KB > 29.1 KB，**带宽不再受并发度限制**。
- **`subgroupShuffle` 代替广播 load**：每 superblock 的 3 条 scale load 降为 1 条（8 lane 活跃）+ 1 条 shuffle，消除 §2.3 的 2.7× 冗余。
- **累加顺序**：每个 (group, 4 元素) 内的 FMA 顺序与原 shader 一致，但跨 block 的累加顺序变为 bg-major；浮点结果与现有 kernel 不是 bit-exact，需按 §3.6 用容差校验。
- **可选**：把 `sb = Σb` 预计算成 `bsum[384]`（12288/32），每 lane 省 8 条加法；收益小，首版不做。

**pipeline 注册与回退** **[代码]**：在 `ggml_vk_load_shaders`（`:1880`）里为 Q4_K/Q6_K 各加一个 L3 pipeline（`local_size_x=32`、无 shared、`required_subgroup_size` 留 0 以取 native 32），只注册 2 个实例，不进入 `DMMV_WG_SIZE_COUNT × mul_mat_vec_max_cols` 的组合爆炸；分派条件为 `n==1 && m>=2048 && k>=2048 && prepack 可用`，其余情形全部落回现有路径。这样 L3 是「加一条支路」，而不是替换现有 27 种量化的 16 个 pipeline 槽位。

### 3.4 Q6_K 的差异

**[代码]** `mul_mat_vec_q6_k.comp:30-38` 每 lane 读 `ql` 4 条 `uint16` + `qh` 2 条 + `d` 1 条 = 8 条 load / 210 B，是 Q4_K 的 1.5× 指令密度；`:54-57` 的 `sccache` 写入 + `barrier()` 每 superblock 一次，强制 load 串行。

L3 版本：
1. `sccache` 全部删除，改为每 lane 持有自己的 `scales[itid]`，用 `subgroupShuffle(s, 8*v_im + is + 2*k)`（k=0..3）取 4 个所需值——`s_offset` 的取值只有 `{0,1,8,9}` 四类，目标 lane 编译期可算，**零 barrier**。
2. `ql`/`qh` 按 32 lane 重排：每 superblock `ql=128 B` 由 32 lane 各读 `uvec1`（4 B），`qh=64 B` 由 lane 0..15 各读 4 B，`scales=16 B` 由 lane 0..3 各读 4 B，`d` 广播。
3. 沿用 §3.3 的 4-block 交织，使每 lane 每迭代一次 `uvec4` 拿 4 个 superblock 的 `ql`。

去 barrier 后 in-flight 从 96 × 210 B = 20 KB 提到 96 × 32 × 16 B = 49 KB，与 Q4_K 同档。

### 3.5 预期收益

**[推算]** 设 Q4_K 占 decode 时间的 68%、Q6_K 占 32%（按 §2 的字节分布与 §3.1 的单实例耗时加权；该拆分未逐 tensor 实测，见 §5）。

- Q4_K：34.8 → 49.5 GB/s（58.2 的 85%），时间系数 34.8/49.5 = 0.703
- Q6_K：36.0 → 49.5 GB/s，时间系数 36.0/49.5 = 0.727
- 新 GPU 时间 = 148.9 × (0.68×0.703 + 0.32×0.727) = 148.9 × 0.711 = **105.9 ms/token**
- 加 CPU 9.5 ms → 115.4 ms/token → **8.67 t/s**（对 6.31 为 +37%；对同构建 CPU 7.35 为 +18%）

若只改 Q4_K（Q6_K 不动）：148.9 × (0.68×0.703 + 0.32) = 118.8 ms → 7.8 t/s。
若按「全部权重统一到 85%」的粗口径（即任务给定 +44% 的算法）：148.9 × 34.2/49.5 = 102.9 ms，加 CPU 9.5 ms → **8.9 t/s（+41%）**，与 9.1 t/s 同档。本文的 8.67 是分 Q4_K/Q6_K 各自起点算的保守值，差额来自 Q6_K 只从 36.0 GB/s 起步。

### 3.6 验证判据

1. **单 kernel 带宽**：perf logger 的 `MUL_MAT_VEC q4_K m=4096 n=1 k=12288` 从 723.9 us 降到 **≤ 510 us**（28.3 MB / 510 us = 55.5 GB/s）；q6_K 从 1148.6 us 降到 **≤ 850 us**。
2. **整 token 带宽**：5.098 GB / GPU 时间 ≥ **49.5 GB/s**（85% 流式上限）。
3. **tg**：`-t 8 -r 5` 三轮交替取中位数 ≥ **8.5 t/s**，且必须超过同构建 CPU 基线 7.35。
4. **正确性**：同一 prompt 的 logits 与现 kernel 逐元素比较，相对误差 ≤ 1e-5（累加顺序变了，不要求 bit-exact）；另跑 perplexity 确认无退化。
5. **反例闸门**：若 tg 提升 < 15%，先查 `gl_SubgroupInvocationID` 是否真为 32（驱动可能把 `local_size_x=32` 拆成 2×SIMD16），以及重排 buffer 是否落在 `memoryTypes[0]`（纯 `DEVICE_LOCAL`）而不是 `[1]`/`[2]`（`HOST_VISIBLE` 系，UMA 上可能被映射成非最优路径）。

---

## 4 代价与风险

### 4.1 L1 / L2 / L3 三层对比

| 层次 | 内容 | 工作量 | 预期 tg | 相对 L3 | 上游接受度 |
|---|---|---|---|---|---|
| L1 运行时特化 | 解除 `:3108` 的强制 16、`it_size` 参数化、`for` 循环手动展开 4 次以提高 outstanding load | 50–150 行 | 7.8–8.2 t/s | 90–95% | 高（先例 #29476 / #24877 / #30829 均已合入） |
| L2 按厂商分叉 | Intel 一套 wg/subgroup/展开参数，Adreno 一套 | 200–400 行 | 8.0–8.4 t/s | 92–97% | 中 |
| L3 完全独立 kernel | 新 `.comp` + 新 pipeline + 加载时布局重排 | 500–1500 行 | 8.5–8.8 t/s | 100% | 低（先例 #20451 Slang FA 至今 open） |

**增量要说清楚**：L1 的 `for` 展开本身就能把 in-flight 提到 4 倍（每 lane 同时挂 4 个 superblock 的 load），从而独立达成 §2.2 的修复；它拿不到的是 Q6_K 的 barrier（§3.4，只有 L3 能去）和指令冗余（§2.3，L3 减半）。所以：
- **L3 相对 L1 的净增量 ≈ Q6_K 单项的 15~25% + Q4_K 的 0~5%**，合计整片约 **+6~10%**（8.2 → 8.7）。
- 若目标只是「超过 CPU 7.35」，**L1 就够了**；L3 买的是 Q6_K 的结构性修复、指令槽从 14% 降到约 7%、以及布局重排带来的 in-flight 余量（不再依赖驱动/编译器的循环展开行为）。
- 用户已声明不在意维护成本，则 L3 值这个增量；但**不应把 L3 的收益预期写成 +44%**——那是「带宽到 85%」的总收益，L1 也能拿大头。

### 4.2 三台硬件的取舍

| 设备 | subgroup | 骨架是否复用 | 差异 |
|---|---|---|---|
| Intel Iris Xe（Gen12） | 32，min 8 | 基准实现 | `local_size_x=32` 直接对应 native subgroup |
| Intel Iris Plus（Gen11） | 32，min 8（**[实测]**，与 Xe 同） | **同一份 .comp 与同一份 prepack** | Gen11 的 EU 数与 L3 容量按 SKU 不同，需现场读 `shader_core_count`；`NUM_ROWS` 按 EU 数重调即可，lane 映射不变 |
| Adreno 750（A7xx TBDR） | 未实测，通常 64（或 128） | **设计复用，代码不复用** | 见下 |

Adreno 的具体约束 **[代码+推断]**：
- 上游对 Qualcomm 的 MMVQ 直接 `return false`（`ggml-vulkan.cpp:7033-7034`），说明该路径在 Adreno 上不可用，L3 不能指望复用 MMVQ 的结论。
- TBDR 的 on-chip tile buffer 只服务渲染 pass；compute shader 不经 tile，`VK_QCOM_tile_shading` 对 GEMV 无直接收益，`VK_QCOM_image_processing` 也与 GEMV 无关。**因此 Adreno 的 L3 与 Intel 是同一套设计思想（lane-major prepack + 单 subgroup shuffle 归约 + 每 lane 一条宽 load），但 lane↔元素映射要按 64 lane 重写**（128 B 的 `qs` 由 64 lane 各读 2 B，解包逻辑不同）。
- 结论：**骨架共用，文件分开**。不要试图一份 `.comp` 覆盖两者——64 lane 下每 lane 只有 2 B，`uvec4` 的宽读优势消失，强行统一会退化成两套 `#ifdef` 交织，比分开更难维护。

### 4.3 风险

1. **额外内存**：prepack 后的 Q4_K + Q6_K 约 5.1 GB。UMA 单 heap 上可行，但需在分配失败时回退到原 kernel（不能硬失败）。
2. **加载时间**：一次性 5.1 GB 读 + 5.1 GB 写，理论 175 ms @58.2 GB/s，实际约 1–2 s。对交互式加载可接受，需在日志里说明。
3. **驱动行为不确定**：`local_size_x=32` 是否被拆成 2×SIMD16 无法从源码断定；若被拆，in-flight 减半到 24.5 KB，仍接近 29.1 KB 的需求，预期收益打八折（8.7 → 8.2 t/s）。这是首版必须实测确认的第一件事。
4. **不可上游**：新 buffer 类型 + prepack 步骤会改动 `ggml_vk_buffer` 与模型加载路径，与上游的量化格式抽象冲突，属于本地分叉，不要按可上游的 PR 组织。

---

## 5 未做

- **Q4_K/Q6_K 的逐 tensor 字节占比未实测**。§3.5 的 68/32 拆分由「blk 总量 4065.6 MiB + output 795.7 MiB」与 §3.1 两个单实例耗时加权得到；若实际 Q6_K 占比更高，整片收益会略低于 8.67 t/s（Q6_K 单项收益 0.727 与 Q4_K 的 0.703 接近，故偏差 <2%）。
- **未验证 Gen12 的 L3 容量与 DRAM 实际延迟**（§2.2 取 500 ns 中值，400–600 ns 区间对应 in-flight 需求 23–35 KB，结论「18.4 KB 不足」在区间内均成立）。
- **未验证 `unpack8` 在 IGC 下的实际指令数**（§2.5 按每次 3 条估算；若为 2 条，指令槽从 14% 降到 12%，不改变判定）。
- **未设计 IQ4_XS / Q5_K 的对应变体**。9B Q4_K_M 不涉及；若换模型需按同一 prepack 思路扩展，`qs` 宽度不同但骨架不变。
- **未评估 CPU 侧 9.5 ms/token 的进一步压缩**。它是 6% 的固定开销，L3 落地后占比会升到 8%，届时值得单独看。
