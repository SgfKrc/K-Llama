# Intel iGPU decode GEMV：独立 L3 kernel 设计

> 状态：**方案（L1 已实测，L3 待实现）**
> 更新日期：2026-11-14
> 适用范围：Intel Gen12/Gen11 核显上 Vulkan 后端的 decode（n=1）GEMV 路径；不含 prefill/GEMM，不含 CPU 后端。

标注约定：**[实测]** 取自既有 benchmark；**[代码]** 由本仓库源码读出；**[推算]** 由公式算出，假设写在旁边。

---

## 1 结论

1. **瓶颈是寄存器墙，不是 DRAM 并发度。** 每 EU 128 GRF，SIMD16 下 45–60 个活跃变量占 90–120 GRF，**每 EU 只能驻留 1 个 SIMD16 warp**，没有第二个 warp 能在等 load 时执行计算。五条假设的判定见 §2：b 的判定（每 EU 1 warp）成立，但 L1 实测（§2.2）把 in-flight 提到 2 倍只换来 +1.4%，直接反证了「in-flight 不足是主因」。
2. **L3 目标 kernel 是两个**：`mul_mat_vec_q4_k`（decode 权重占比最大）与 `mul_mat_vec_q6_k`（单位字节最慢，且带 per-superblock barrier）。
3. **L3 的核心不是重写归约，而是重排权重布局以减寄存器**：把 Q4_K 的 144 B superblock 转置成 lane-major，每 lane 一次 `uvec4` 读回多个 superblock、用 `dot` 一次算完，把活跃变量从 45–60 压到 15–20 → 每 warp ≤64 GRF → 每 EU 2 warp → 计算与 load 才能重叠。
4. **预期 7.24 → 7.96–8.33 t/s（+10~15%）**；若 GEMV 部分完全达到纯读上限 48.9 GB/s，理论上限 8.79 t/s（+21%）。算式见 §3.5。
5. **L1 已实现并实测只有 +1.4%**（7.24 → 7.34 t/s，`6e5e95c7b`），因为预取本身就要寄存器（U=2 只预取 qs 已 +16 GRF，全预取 +48~96 GRF 直接退化），拿不到 L3 的减寄存器收益。旧的「L1 拿 70–80%」预期作废，增量拆分见 §4.1。

---

## 2 依据：81% 纯读上限的五条假设逐条判定

基准口径 **[实测]**：9B Q4_K_M，5.24 GiB，`-t 8 -r 5`，`-ngl 99`，tg **7.24 t/s**（中位数，HEAD `a2ef2d731`）；总 138.1 ms/token，CPU 开销 9.5 ms/token，GPU 128.6 ms/token；每 token 必读权重 5.098 GB（blk 4065.6 MiB + output 795.7 MiB，token_embd 不读）。
反推带宽 5.098 GB / 128.6 ms = **39.6 GB/s**，对 GEMV 纯读上限 **48.9 GB/s**（§2.2 实测，非 memcpy 口径）为 **81%**。算术强度 2 FLOP/B，计算开销实测 108.5 us（§2.2），不构成主瓶颈。旧文档全篇的 6.31 t/s / 58.2 GB/s 口径已作废，见 §5。

| 假设 | 判定 | 一句话依据 |
|---|---|---|
| a 归约开销 | **排除** | HYBRID 归约每 wg 仅 1 次 barrier，占总时间 <0.3% |
| b 占用率 / workgroup | **成立，但主因是寄存器** | 每 EU 只驻留 1 个 SIMD16（判定正确）；L1 实测把 in-flight 提到 2 倍只 +1.4%，真正的墙是只剩 16–32 GRF 余量 |
| c 访存模式 | **成立（次因）** | 每 superblock 96 条 lane-load 只为 144 B 有效数据，冗余 2.7× |
| d L2/L3 cache | **排除** | 权重每 token 只读一遍，命中率≈0；唯一可复用的 b 向量占比 <2% |
| e 指令效率 | **排除** | 指令槽占 16%，不是当前限制，但构成 L3 的天花板 |

### 2.1 a 归约开销：排除

**[代码]** Intel + Q4_K + m=4096 走 `DMMV_WG_SIZE_LARGE`（`ggml-vulkan.cpp:5917-5929`：Intel 且 `m<=8192 && k>=1024`），`wg=subgroup_size16*4=64`（`:3126`），reduction 取 `SHADER_REDUCTION_MODE_HYBRID`（`:3129`），即 `mul_mat_vec_base.glsl:141-191` 的 `USE_SUBGROUP_ADD` 分支：先 `subgroupAdd`，经 shared 汇总 4 个 subgroup 的部分和，**全 kernel 只有 1 次 `barrier()`（`:157`）**，随后仅 tid==0 写 2 个 float。

**[推算]** 2048 个 wg（m=4096 / NUM_ROWS=2）× 约 100 cycle（barrier 50 + 4 步 shuffle + 串行加）= 204 800 cycle，摊到 96 EU 为 2133 cycle ≈ **1.5 us**，占单 kernel 687.5 us 的 0.2%。纯 shmem 树（`:200-209`，6 次 barrier）在 Intel 上根本不会被选中。

**Q6_K 是例外**，见 §3.4：它的 barrier 不在归约里，而在每 superblock 的解包中。

### 2.2 b workgroup 与占用率：成立，但主因是寄存器墙

**[代码]** `ggml-vulkan.cpp:3108` 在 Intel 且支持 subgroup size control 时**强制 subgroup 16**（`subgroup_min_size<=16 && subgroup_max_size>=16`）。Gen12 原生 `subgroupSize=32`，`minSubgroupSize=8` **[实测]**，所以这是主动选择而非硬件限制——而 **[实测] 这个选择是对的**：改成 SIMD32 后每标量占 4 GRF，45–60 个变量即 180–240 GRF > 128，实测 `q4_K` −5%、**`q6_K` 慢 210%**（1130 → 3500 us）、tg 3.97 t/s。**`:3108` 不能解除。** SIMD16 下每个 float 变量占 2 个 GRF。

**[代码]** `mul_mat_vec_q4_k.comp:94-117`：`it_size = gl_WorkGroupSize.x/16 = 4`，每个 superblock 由 16 lane 协作，每 lane 只负责 9 字节；`k=12288` 时 `num_blocks_per_row=48`，每个 16-lane 组要串行迭代 12 个 superblock。

**[实测] 纯读上限是 48.9 GB/s，不是 58.2**：把 `q4_K m=4096 n=1 k=12288` 的 dequant+FMA 换成极简累加（load 数不变），687.5 us 降到 **579 us**，即 28.3 MB / 579 us = 48.9 GB/s；同 kernel 在 m=8192 k=4096 上也是 48.1 GB/s。**58.2 是 memcpy 口径，GEMV 达不到**，旧文档用它算 in-flight 需求是错的。

**[推算]** 用 Little 定律定量：iGPU 上 L2 miss 的 DRAM 延迟取 500 ns（400–600 ns 区间），要跑满 48.9 GB/s 需在飞 **24.5 KB**（48.9 × 500 ns）。
实际在飞：每 SIMD16 每次迭代发 6 条 load（qs×2、scales×3、dm×1），其中 scales 三条命中同一 sector、dm 与 scales 同 line，取回约 3 个 64 B line ≈ 192 B；q4_K 的活跃变量（16 个 q + 8 个 sc + 4 个 `vec4` b + dm + 中间值）约 45–60 GRF/线程，SIMD16 下即 90–120 GRF，**每 EU 只能驻留 1 个 SIMD16**（128 GRF/EU），占用率 16/64 lane = 25%。
在飞 = 96 EU × 192 B = **18.4 KB**，为所需 24.5 KB 的 **75%**。

**[实测] 但 in-flight 不是主因。** L1 用双缓冲预取把 in-flight 提到 2 倍（`6e5e95c7b`），tg 只从 7.24 到 **7.34 t/s（+1.4%）**，单 kernel 687.5 → 647.4 us（−5.8%），带宽 41.2 → 43.7 GB/s。**真正的限制是寄存器墙**：L1 试了 5 种预取深度，只有最浅的一种没退化——U=2 全预取 +48 GRF → 700→962 us；U=4 全预取 +96 GRF → 700→2128 us；U=4 只预取 qs +32 GRF → tg 6.58；U=2 qs+dm +32 GRF → tg 7.19；**U=2 只预取 qs +16 GRF → tg 7.34 ✓**。**余量只有 16–32 GRF**：90–120 GRF 的活跃变量之外只剩这么多，再多一个预取寄存器就把每 EU 唯一那个 warp 挤掉。
**计算开销实测 108.5 us**（687.5 − 579，约 17%），且与 load 无法重叠——因为没有第二个 warp。所以瓶颈的准确表述是：**没有第二个 warp 来重叠计算与访存**，不是 in-flight 不足。

**[代码]** 96 EU / 4 EU-per-wg = 24 个 workgroup 同时驻留，每 wg 读 2×6912 B = 13.8 KB，一批 332 KB；批次数 2048/24 = 85。批与批之间没有跨批预取——这是 18.4 KB 的来源，但它只解释了 75%，不是全部缺口。

### 2.3 c 访存模式：成立（次因）

**[代码]** `types.glsl:392-413`：`block_q4_K` = `f16vec2 dm` + `uint8_t scales[12]` + `uint8_t qs[128]` = 144 B。shader 用 `block_q4_K_packed16` / `packed32` 视图访问。

**[代码]** `mul_mat_vec_q4_k.comp:19-21` 每个线程读 3 个 `uint32` 覆盖 12 B 的 scales，但 `v_im = il/2`（`:103`）只有 0/1 两个取值，**16 个 lane 只请求 2 个不同地址**；`:17` 的 `dm` 更极端，16 lane 请求同一地址。于是每 superblock 的有效 144 B 由 16×6 = 96 条 lane-load 拼出，冗余 2.7×（144/4 = 36 个 `uint32` 的理想值）。

对齐方面：superblock 起始偏移 `i*144`，模 64 为 0/16/32/48，单个 superblock 跨 3 个 64 B line 而非 2.25 个；但整行 6912 B 是连续的，相邻 superblock 共享 line，**流式读的 DRAM 字节没有浪费**。所以对齐不是 DRAM 效率问题，只是 load 指令效率问题。

### 2.4 d L2/L3 cache：排除

**[推算]** 5.098 GB 权重每 token 只读一遍，Gen12 iGPU 的 L3 为共享 slice（MB 级，具体容量未在本机测量），对 5 GB 流式工作集命中率≈0，优化空间为零。唯一有复用价值的是 hidden 向量 b（12288×4 B = 48 KB），被 2048 个 wg 各读一遍，逻辑流量 98 MB，全部应在 L2 命中；即使全 miss 也只占 5.098 GB 的 1.9%。
`output`（lm_head，795.7 MiB，Q6_K）与 blk 一样每 token 读一遍，无跨 token 复用；`token_embd` 不参与 decode 的矩阵乘。**结论：cache 维度的任何优化都不值得做，收益必须来自减寄存器、提高每 EU 的 warp 驻留数。**

### 2.5 e 指令效率：排除（但它是天花板）

**[代码]** `mul_mat_vec_q4_k.comp:19-82` 每 lane 每 superblock 约 78 条指令：6 条 load、scale 解包 ~17 条（含 3 次 `unpack8`）、qs 解包 ~16 条（4 次 `unpack8`）、b 的 4 条 `vec4` load、35 条 FMA（sx/sy/sz/sw 各 4 + smin 16 + 收尾 3）。

**[推算]** 每 token 35.4M 个 superblock（5.098 GB / 144 B）× 78 条 SIMD16 指令 = 2.76 G 指令；硬件 96 EU × 1.4 GHz = 134 G 指令/s，即 **20.6 ms/token**，占 128.6 ms 的 **16%**，与 §2.2 实测的计算开销（108.5 us / 687.5 us）同档。所以指令不是当前瓶颈；但若带宽到纯读上限 48.9 GB/s（104.3 ms），指令仍只占 20%，不构成新瓶颈——**除非**改用更宽的 subgroup 或去掉布局重排，那会把这个比例推上去。

---

## 3 方案

### 3.1 目标 kernel 与现状

| kernel | 现状耗时 [实测] | 带宽 [推算] | 结构问题 |
|---|---|---|---|
| `mul_mat_vec_q4_k`，m=4096 n=1 k=12288 | 687.5 us（纯读下限 579 us） | 28.3 MB/687.5 us = 41.2 GB/s（perf logger 口径同档） | §2.2 寄存器墙 + §2.3 冗余 load |
| `mul_mat_vec_q6_k`，m=4096 n=1 k=12288 | 1130 us | 41.3 MB/1130 us = 36.6 GB/s | 每 superblock 一次 `barrier()` |

**[代码]** Q6_K 在 `m=4096` 时不满足 `m<4096`（`ggml-vulkan.cpp:5921-5924`），落到 `DMMV_WG_SIZE_SUBGROUP`，`wg=16`（1 个 subgroup），归约走 `USE_SUBGROUP_ADD_NO_SHMEM`（`mul_mat_vec_base.glsl:96-134`，零 barrier）。但 `mul_mat_vec_q6_k.comp:9` 声明 `shared sccache[2][BLOCK_SIZE/16][16]`，`:55-56` 每个 superblock 写一次 shared 并 `barrier()`，`:74` 再从 shared 读 `s_offset..s_offset+6`。`k=12288` 时每 wg 96 次 barrier，**它把 load 的软件流水切断**，这才是 Q6_K 单位字节带宽比 Q4_K（41.2 GB/s）低 11%、只有 36.6 GB/s 的原因，与 MMVQ 无关（MMVQ 在本机 −5.7%，`ggml-vulkan.cpp:7013-7018` 对 Intel Windows 专有驱动直接 `return false`）。

### 3.2 权重布局重排（L3 的支点）

加载时把每个 Q4_K 行按 2 个 superblock 一组转置，写进一个独立 buffer：

```
Q[bg][lane][0..3]    uint32   // [0..1] = qs(block 2*bg) 的第 2*lane、2*lane+1 个 uint32；[2..3] = block 2*bg+1
```
16 lane × 16 B = 256 B = 2 × 128 B，一次连续读，零冗余、零填充。
scale 与 min 打进一个 16 B 的 `SCDM` 记录：3 个 `uint32` 装 8 个 12-bit 值 `scale | (min<<6)`（sub-block j 在 bit `12*j`），第 4 个 `uint32` 就是原始 `half2(d, dmin)` 原样。于是每 superblock 128 B(qs) + 16 B(SCDM) = **144 B，与原 block 严格相等，prepack 后每 token 的 DRAM 字节数不变**。kernel 侧每个 lane 需要 sub-block `2*(lane/4)` 和 `2*(lane/4)+1`，即 `SCDM` 里连续 24 bit；4 路位域选择（`sci==0 ? x&0xFFFFFF : sci==1 ? (x>>24)|(y<<8) : …`）在 16 lane 上 predicated，约 6 条 ALU。**这个交换是刻意的**：若按 `uint16` 存 `scm`，每块是 128 + 16 + 4 = 148 B（+2.78%），而判据 2 的 600 us 门槛只比 579 us 纯读下限高 3.6%，2.78% 的 DRAM 增量会吃掉几乎全部余量；6 条 ALU 则落在 §2.5 那 16% 的指令槽里。

落地位置 **[代码]**：在 `ggml_vk_create_tensor`（`ggml-vulkan.cpp`）的 buffer 分配处挂一个 `q4k_gemv` 附属 buffer，仅当 `ne0 % 256 == 0 && type == GGML_TYPE_Q4_K && 使用该 tensor 的 MUL_MAT 的 n <= mul_mat_vec_max_cols` 时生成；`ggml_vk_mul_mat_vec_q_f16`（`:7042`）在选 pipeline 前检查附属 buffer 是否存在，存在则走 L3 路径、否则走现有 `ggml_vk_get_dequantize_mul_mat_vec`（`:5853`）。prepack 本身用一次 `vkCmdCopyBuffer` 之后的小 compute shader 完成（一次 5 GB 量级搬运，见 §4.3）。

### 3.3 GLSL 结构

```glsl
layout(local_size_x = 16) in;                 // 1 个 SIMD16 subgroup，与 :3108 一致
layout(constant_id = 0) const uint NUM_ROWS = 2;
layout(binding = 0) readonly  buffer Q  { uvec4  qs4[]; };   // prepack: 2 个 superblock 交织
layout(binding = 1) readonly  buffer SC { uint   scm32[]; }; // prepack: 每 u16 = sc|min<<6，每 u32 两个
layout(binding = 2) readonly  buffer B  { vec4    b4[];  };  // hidden，L2 命中
layout(binding = 3) readonly  buffer DM { vec2    dm[];  };  // prepack: d, dmin
layout(binding = 4) writeonly buffer D  { float   out[]; };

void main() {
    const uint lane = gl_SubgroupInvocationID;   // 0..15
    // Q4_K 真实布局 [代码 ggml-quants.c:1551-1559]：32 B 的 qs 覆盖 64 个元素，
    // 字节 m 的低 4 位 -> 元素 base+m，高 4 位 -> 元素 base+32+m。低/高 4 位相距 32，不是 16。
    const uint base = 64u*(lane/4u) + 8u*(lane%4u);  // 本 lane 在 superblock 内的起始元素
    const uint sc_i = 2u*(lane/4u);                  // 本 lane 用的 (scale,min) 对下标
    const uint row0 = NUM_ROWS * (gl_WorkGroupID.x + gl_NumWorkGroups.x * gl_WorkGroupID.z);
    float acc[NUM_ROWS] = {0.0, 0.0};

    for (uint r = 0; r < NUM_ROWS; ++r) {
        for (uint bg = 0; bg < nbg; ++bg) {          // nbg = 48/2 = 24
            const uvec4 q = qs4[(row0+r)*QB4 + bg*16 + lane];  // 1 条 16 B = 2 个 superblock
            const vec2  d = dm[(row0+r)*DB + bg*2];            // block 级，广播 load
            [[unroll]] for (uint j = 0; j < 2; ++j) {
                const uint qw = (j == 0) ? q.x : q.z;   // 字节 8*lane + 0..3
                const uint qv = (j == 0) ? q.y : q.w;   // 字节 8*lane + 4..7
                const uint e0 = ((bg*2u + j)*256u + base) >> 2;   // 低 4 位段的 vec4 索引
                const uint e1 = e0 + 8u;                          // 高 4 位段 = base+32
                const vec4 blo0 = b4[e0], blo1 = b4[e0+1u];
                const vec4 bhi0 = b4[e1], bhi1 = b4[e1+1u];
                const vec4 lo_a = vec4(unpack8( qw       & 0x0F0F0F0F));  // base+0..3
                const vec4 lo_b = vec4(unpack8( qv       & 0x0F0F0F0F));  // base+4..7
                const vec4 hi_a = vec4(unpack8((qw >> 4) & 0x0F0F0F0F));  // base+32..35
                const vec4 hi_b = vec4(unpack8((qv >> 4) & 0x0F0F0F0F));  // base+36..39
                const uint s32 = scm32[(row0+r)*SB4 + (bg*2u + j)*4u + sc_i];  // 4 lane 同址
                const float sq_lo = dot(blo0, lo_a) + dot(blo1, lo_b);
                const float sq_hi = dot(bhi0, hi_a) + dot(bhi1, hi_b);
                const float sb_lo = dot(blo0 + blo1, vec4(1.0));   // base+0..7 的 b 之和
                const float sb_hi = dot(bhi0 + bhi1, vec4(1.0));   // base+32..39 的 b 之和
                acc[r] = fma(d.x, fma(float( s32        & 63u), sq_lo, float((s32 >> 16) & 63u) * sq_hi), acc[r])
                       - d.y * (float((s32 >>  6) & 63u) * sb_lo + float((s32 >> 22) & 63u) * sb_hi);
            }
        }
    }
    [[unroll]] for (uint r = 0; r < NUM_ROWS; ++r) acc[r] = subgroupAdd(acc[r]);
    if (lane == 0) { for (uint r = 0; r < NUM_ROWS; ++r) out[d0 + r] = acc[r]; }
}
```

要点与理由：

- **wg=16、单 SIMD16 subgroup**：与 `:3108` 的强制 16 一致——**不靠换 subgroup 宽度省寄存器**（实测 SIMD32 更差，§2.2）。归约退化为 4 步 `subgroupAdd`，**无 barrier、无 shared**。
- **qs 的字节↔元素映射是「相距 32」，不是「相距 16」**：`dequantize_row_q4_K`（`ggml-quants.c:1551-1559`）按 32 B 一组解 64 个元素，字节 `m` 的低 4 位→元素 `base+m`、高 4 位→元素 `base+32+m`；`base ∈ {0,64,128,192}`。`mul_mat_vec_q4_k.comp` 的 `y2_idx = y1_idx + 128`、`by132 = b4[y1_idx/4 + 8]`（+8 个 vec4 = +32 元素）与之一致。**这一条错了整个 prepack 布局就失效**，且整模上只表现为「logits 全错」，建议先用单 superblock 对拍确认再上整模。
- **每 lane 一次 `uvec4` + `dot`**：DRAM 侧每迭代每 lane 一次 16 B load（原实现每 superblock 9 B 但要 6 条指令），b 落 L2 不占 DRAM。真正的目的是**减活跃变量**：`q`/`blo0..bhi1`/`lo_a..hi_b` 都是一次性中间量，编译器可复用少数 GRF，把 45–60 压到 15–20 → 每 warp ≤64 GRF → 每 EU 2 warp。in-flight 顺带变成 96 EU × 2 × 16 lane × 32 B = 98 KB ≫ 24.5 KB，但这只是副产品，不是收益来源。
- **scale 用 1 条 u32 load 代替 3 条广播 load**：4 个 lane 同址，硬件 broadcast 合并，消除 §2.3 的 2.7× 冗余，且不需要 `subgroupShuffle`。
- **min 项必须分两段**：低 4 位段用 `min_lo`、高 4 位段用 `min_hi`，各自乘自己那段 8 个 b 之和（`sb_lo`/`sb_hi`）。合成一个 `Σb` 再乘单个 min 是错的——原 kernel 的 `smin` 用 `by10×min0 + by132×min1` 正是两段分开的证据。
- **累加顺序**：每 lane 负责的 16 个元素内的 FMA 顺序与原 shader 一致，但跨 block 的累加顺序变为 bg-major；浮点结果与现有 kernel 不是 bit-exact，需按 §3.6 用容差校验。
- **可选**：把 `sb_lo`/`sb_hi` 预计算成 `bsum[384]`（12288/32），每 lane 省 4 条加法；收益小，首版不做。

**pipeline 注册与回退** **[代码]**：在 `ggml_vk_load_shaders`（`:1880`）里为 Q4_K/Q6_K 各加一个 L3 pipeline（`local_size_x=16`、无 shared、`required_subgroup_size=16` 与 `:3108` 保持一致），只注册 2 个实例，不进入 `DMMV_WG_SIZE_COUNT × mul_mat_vec_max_cols` 的组合爆炸；分派条件为 `n==1 && m>=2048 && k>=2048 && prepack 可用`，其余情形全部落回现有路径。这样 L3 是「加一条支路」，而不是替换现有 27 种量化的 16 个 pipeline 槽位。

### 3.4 Q6_K 的差异

**[代码]** `mul_mat_vec_q6_k.comp:30-38` 每 lane 读 `ql` 4 条 `uint16` + `qh` 2 条 + `d` 1 条 = 8 条 load / 210 B，是 Q4_K 的 1.5× 指令密度；`:54-57` 的 `sccache` 写入 + `barrier()` 每 superblock 一次，强制 load 串行。

L3 版本：
1. `sccache` 全部删除，改为每 lane 持有自己的 `scales[itid]`，用 `subgroupShuffle(s, 8*v_im + is + 2*k)`（k=0..3）取 4 个所需值——`s_offset` 的取值只有 `{0,1,8,9}` 四类，目标 lane 编译期可算，**零 barrier**。
2. `ql`/`qh` 按 16 lane 重排：每 superblock `ql=128 B` 由 16 lane 各读 `uvec2`（8 B），`qh=64 B` 由 16 lane 各读 4 B，`scales=16 B` 由 lane 0..3 各读 4 B，`d` 广播。
3. 沿用 §3.3 的 2-block 交织，使每 lane 每迭代一次 `uvec4` 拿 2 个 superblock 的 `ql`。

去 barrier 后每 lane 每迭代一次 `uvec4`，设计目标下 in-flight 与 Q4_K 同档（98 KB）；但 Q6_K 同样要先把活跃变量压到 ≤64 GRF 才拿得到 2 warp/EU，**去 barrier 只是必要条件**。

### 3.5 预期收益

**[实测] 基线已更新为 7.24 t/s**（HEAD `a2ef2d731`）：总 138.1 ms/token，CPU 9.5 ms，GPU 128.6 ms。旧文档全篇用的 6.31 t/s 来自 `build-vulkan-new` @ `48dfb3ea8`（11-09 快照），**14% 的差异未归因**（见 §5）。

**[推算] 收益机制是减寄存器，不是提 in-flight**：lane-major 重排 + 每 lane 一个 `uvec4` + `dot` 把活跃变量从 45–60 压到 15–20 → 每 warp ≤64 GRF → 每 EU 2 warp → 计算与 load 重叠，才能吃到 579 us 那一档（纯读下限）。L1 已证明靠预取提 in-flight 走不通（+1.4%，§2.2）。

- **理论上限**：5.098 GB / 48.9 GB/s = **104.3 ms** + CPU 9.5 = 113.8 ms → **8.79 t/s**（对 7.24 为 +21%）
- **现实预期**：579 us 只是 load 下限，计算 108.5 us 即使有第二个 warp 也难 100% 重叠。按整片计算量 104.3 × 0.173 ≈ 18.0 ms、重叠率 α = 35–65% 折算：GPU = 104.3 + 18.0 × (1−α) = **116.0–110.6 ms**；加 CPU 9.5 → **125.5–120.1 ms/token → 7.96–8.33 t/s（+10~15%）**
- **旧预期 +35~40% 作废**（按「全部权重统一到 85% 流式上限」粗口径算出的 +44% 同样作废）：两者都建立在 58.2 GB/s（memcpy 口径，GEMV 达不到）与「L1 拿 70–80%」两个被实测推翻的前提上。也不应把理论上限 +21% 当承诺值。

不再按 Q4_K/Q6_K 的 68/32 拆分加权：两者目标都是同一个 48.9 GB/s 纯读上限，拆分只影响结果落在区间哪一端。

### 3.6 验证判据

1. **寄存器与占用率（首要判据）**：IGC 编译报告里 L3 kernel 的 GRF 用量 **≤ 64**、每 EU **≥ 2 个 SIMD16 warp 驻留**。这是收益机制的直接判据——压不下去就说明活跃变量没减掉，后面几条必然不达标。
2. **单 kernel 带宽**：perf logger 的 `MUL_MAT_VEC q4_K m=4096 n=1 k=12288` 从 687.5 us 降到 **≤ 600 us**（579 us 纯读下限 + 3.6% 余量，即 28.3 MB / 600 us = 47.2 GB/s）；q6_K 从 1130 us 降到 **≤ 900 us**（41.3 MB / 900 us = 45.9 GB/s，纯读上限的 94%；850 us 对应 48.6 GB/s，比 q4_K 那条判据更紧，不现实）。
3. **tg**：`-t 8 -r 5` 三轮交替取中位数 ≥ **7.96 t/s（+10%）**，且必须超过同构建 CPU 基线 7.35。
4. **正确性**：同一 prompt 的 logits 与现 kernel 逐元素比较，相对误差 ≤ 1e-5（累加顺序变了，不要求 bit-exact）；另跑 perplexity 确认无退化。
5. **反例闸门**：若 tg 提升 < 8%，先看 GRF 是否真压到 64 以下（>64 说明 lane-major 没减掉活跃变量，直接回到寄存器墙）；再看重排 buffer 是否落在 `memoryTypes[0]`（纯 `DEVICE_LOCAL`）而不是 `[1]`/`[2]`（`HOST_VISIBLE` 系，UMA 上可能被映射成非最优路径）。**不要再查 `gl_SubgroupInvocationID` 是否为 32**：SIMD32 已被实测证伪（§2.2），`local_size_x=32` 只会更差。

---

## 4 代价与风险

### 4.1 L1 / L2 / L3 三层对比

| 层次 | 内容 | 工作量 | 预期 tg | 占 L3 净增量 | 上游接受度 |
|---|---|---|---|---|---|
| L1 运行时特化 | 双缓冲预取（U=2 只预取 qs，+16 GRF）、`it_size` 参数化 | 50–150 行 | **7.34（+1.4%）[实测]** | ≈10% | 高（先例 #29476 / #24877 / #30829 均已合入） |
| L2 按厂商分叉 | Intel 一套 wg/预取参数，Adreno 一套 | 200–400 行 | ≈7.3–7.6（未实测，预计与 L1 同档） | — | 中 |
| L3 完全独立 kernel | 新 `.comp` + 新 pipeline + 加载时布局重排（减寄存器） | 500–1500 行 | **7.96–8.33（+10~15%）[推算]** | 100% | 低（先例 #20451 Slang FA 至今 open） |

**增量要说清楚**：L1 的双缓冲预取把 in-flight 提到 2 倍，实测只 +1.4%（7.24 → 7.34），因为预取寄存器本身就要从 16–32 GRF 的余量里出（全预取 +48~96 GRF 直接退化，§2.2）。L3 的**唯一理由是把活跃变量减下来**：lane-major 布局 + 每 lane 一个 `uvec4` + `dot` → 15–20 个活跃变量 → ≤64 GRF → 每 EU 2 warp → 计算与 load 重叠。这是 L1 做不到的（预取本身就要寄存器），不是 L1 做得不够好。所以：
- **L3 相对 L1 的净增量是 +8.4~13.5%**（7.34 → 7.96~8.33）。旧文档的「L1 拿 70–80%」与「L1 拿 90–95%」两处都作废——实测 1.4% 对 L3 的 10~15%，L1 只占 L3 净增量约 10%。
- L3 的附带收益：Q6_K 去掉每 superblock 的 barrier（§3.4）、指令槽从 16% 降到约 8%、布局重排后不再依赖驱动/编译器的循环展开行为。
- 用户已声明不在意维护成本，则 L3 值这个增量；但**不应把 L3 的收益预期写成 +35~40% 或 +21%**——前者是建立在 58.2 GB/s 口径上的旧数（已作废），后者是「GEMV 全部达到纯读上限」的理论上限，L3 实际只拿到其中一部分。

### 4.2 三台硬件的取舍

| 设备 | subgroup | 骨架是否复用 | 差异 |
|---|---|---|---|
| Intel Iris Xe（Gen12） | 32，min 8 | 基准实现 | `local_size_x=16` + `required_subgroup_size=16`，与 `:3108` 一致（SIMD32 实测更差，§2.2） |
| Intel Iris Plus（Gen11） | 32，min 8（**[实测]**，与 Xe 同） | **同一份 .comp 与同一份 prepack** | Gen11 的 EU 数与 L3 容量按 SKU 不同，需现场读 `shader_core_count`；`NUM_ROWS` 按 EU 数重调即可，lane 映射不变 |
| Adreno 750（A7xx TBDR） | 未实测，通常 64（或 128） | **设计复用，代码不复用** | 见下 |

Adreno 的具体约束 **[代码+推断]**：
- 上游对 Qualcomm 的 MMVQ 直接 `return false`（`ggml-vulkan.cpp:7033-7034`），说明该路径在 Adreno 上不可用，L3 不能指望复用 MMVQ 的结论。
- TBDR 的 on-chip tile buffer 只服务渲染 pass；compute shader 不经 tile，`VK_QCOM_tile_shading` 对 GEMV 无直接收益，`VK_QCOM_image_processing` 也与 GEMV 无关。**因此 Adreno 的 L3 与 Intel 是同一套设计思想（lane-major prepack + 单 subgroup shuffle 归约 + 每 lane 一条宽 load），但 lane↔元素映射要按 64 lane 重写**（128 B 的 `qs` 由 64 lane 各读 2 B，解包逻辑不同）。
- 结论：**骨架共用，文件分开**。不要试图一份 `.comp` 覆盖两者——64 lane 下每 lane 只有 2 B，`uvec4` 的宽读优势消失，强行统一会退化成两套 `#ifdef` 交织，比分开更难维护。

### 4.3 风险

1. **额外内存**：prepack 后的 Q4_K + Q6_K 约 5.1 GB（§3.2 的 144 B/块与原格式严格相等，不膨胀）。UMA 单 heap 上可行，但需在分配失败时回退到原 kernel（不能硬失败）。
2. **加载时间**：一次性 5.1 GB 读 + 5.1 GB 写，理论 209 ms @48.9 GB/s（纯拷贝，memcpy 口径在此适用），实际约 1–2 s。对交互式加载可接受，需在日志里说明。
3. **寄存器能否压到 64 无法从源码断定**：IGC 的活跃变量分配必须实测。若压不下去（>64 GRF），就仍是 1 warp/EU、计算无法与 load 重叠，预期收益打五折（8.33 → 7.5 左右，接近 L1）。这是首版必须实测确认的第一件事。
4. **不可上游**：新 buffer 类型 + prepack 步骤会改动 `ggml_vk_buffer` 与模型加载路径，与上游的量化格式抽象冲突，属于本地分叉，不要按可上游的 PR 组织。

---

## 5 未做

- **Q4_K/Q6_K 的逐 tensor 字节占比未实测**。§3.5 已不再按 68/32 加权（两者目标同为 48.9 GB/s 纯读上限），此项只影响结果落在 7.96–8.33 区间的哪一端，不改变结论。
- **未验证 Gen12 的 L3 容量与 DRAM 实际延迟**（§2.2 取 500 ns 中值，400–600 ns 区间对应 in-flight 需求 19.6–29.4 KB，结论「18.4 KB 不足」在区间内均成立）。
- **未验证 `unpack8` 在 IGC 下的实际指令数**（§2.5 按每次 3 条估算；若为 2 条，指令槽从 16% 降到 14%，不改变判定）。
- **未设计 IQ4_XS / Q5_K 的对应变体**。9B Q4_K_M 不涉及；若换模型需按同一 prepack 思路扩展，`qs` 宽度不同但骨架不变。
- **未评估 CPU 侧 9.5 ms/token 的进一步压缩**。它是约 7% 的固定开销，L3 落地后占比会升到 8%，届时值得单独看。
- **L1 已实现并实测**：双缓冲预取（U=2 只预取 qs）已提交 `6e5e95c7b`，tg 7.24 → 7.34（+1.4%），单 kernel 687.5 → 647.4 us。原计划里的「解除 `:3108` 强制 16」已被实测证伪（SIMD32 下 q6_K 慢 210%），不再列为待做项。
- **基线 6.31 → 7.24 的 14% 差异未归因**：可能是 host buffer 修复的副作用，也可能是 11-09 到现在的其他 shader 改动，未做逐 commit bisect，故不写入任何因果解释。
