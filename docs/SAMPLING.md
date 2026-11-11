# 采样器接入

> 状态：**已实现**（文中引用的 v2 工具链——`chain_client` / `stage_worker` / `shim/` / `koakuma`——已退场，归档在 `slimarc/prototype`，命令与路径只作历史口径）
> 更新日期：2026-11-06
> 适用范围：末段采样链的接入方式、参数映射与验收判据；读者为维护 shim 与 server 的人。
> 相关：`SPECULATION.md` §11（投机与采样）、`TUI.md`（采样开关）

接入前解码是 argmax：`qlh_kh_forward_embd_token` 遍历 logits 取最大值，只把 `int32_t` token id 交给调用方，`temperature` / `top_p` / `seed` 没有可作用的对象， `server.py` 接受后在响应里标注为 ignored。本文记录实现方式与接入点。

---

## 1. 三种做法与取舍

| 方案 | 跨机开销 | 结论 |
| --- | --- | --- |
| 传完整 logits（241,664 × f32） | 每 token **944 KB**，是 hidden 的 118 倍 | 不可行 |
| 只传 top-k 候选 | 每 token 约 320 B | 可行（等价性见下注） |
| **在 shim 内采样** | **0** | **选定** |

> **v2.1 的补充**：早期记的「top-p 失真」后来被协议 v5 推翻了——`top_k` 之后的所有步骤只在
> k 个候选内作用，只要把 `top_k` 之前的步骤（`logit_bias` / `penalties` / `top_n_sigma`）
> 放在有整行的那一侧跑，传候选就是**数学等价**的（见
> [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §2.2 与
> [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §8）。本表仍选「在 shim 内采样」，
> 理由不是等价性，而是 shim 这条链本来就不需要把采样拆到两端——省一次协商，也省一份回退规则。

决定性理由是多段链路的带宽特性：段间只传 hidden（8 KB/token），采样挪到 Python 侧要么传 logits 要么传候选，都会破坏这一点；在末段本地做完，段间协议完全不用改。

---

## 2. 复用上游采样器，不自己写算法

`llama.dll` 导出了全套采样器 API，直接调用即可。签名逐字取自 `build-headers/llama.h`：

```c
struct llama_sampler_chain_params llama_sampler_chain_default_params(void);
struct llama_sampler * llama_sampler_chain_init(struct llama_sampler_chain_params params);
void                   llama_sampler_chain_add(struct llama_sampler * chain, struct llama_sampler * smpl);
struct llama_sampler * llama_sampler_init_penalties(int32_t n_vocab, int32_t penalty_last_n,
                                                    float penalty_repeat, float penalty_freq,
                                                    float penalty_present);
struct llama_sampler * llama_sampler_init_top_k  (int32_t k);
struct llama_sampler * llama_sampler_init_top_p  (float p, size_t min_keep);
struct llama_sampler * llama_sampler_init_min_p  (float p, size_t min_keep);
struct llama_sampler * llama_sampler_init_temp   (float t);
struct llama_sampler * llama_sampler_init_dist   (uint32_t seed);
llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx);
void        llama_sampler_accept(struct llama_sampler * smpl, llama_token token);
void        llama_sampler_reset (struct llama_sampler * smpl);
void        llama_sampler_free  (struct llama_sampler * smpl);
```

`llama_sampler_sample` 内部自己调 `llama_get_logits_ith`，shim 不需要再拿 logits 指针；正确性由上游保证，比自行实现 top-k / top-p / softmax 短得多。同一个函数还自己调 `accept`（§3），所以调用方不能再 accept 一次。 `llama_sampler_init_greedy()` 也在导出表里（`dumpbin /EXPORTS` 已确认，共 36 个 `llama_sampler_*` / `llama_perf_sampler*` 符号），§4 的链在 `temperature <= 0` 时用得到它。

---

## 3. 接入点

`slimarc/prototype:shim/qlh_keep_head_gpu.c` 的 `qlh_kh_forward_embd_token`，argmax 在该函数尾部。保留非有限值检查，然后分两条路：

```c
if (handle->sampler != NULL) {
    const llama_token tok = llama_sampler_sample(handle->sampler, handle->ctx, n_tokens - 1);
    /* 不要再调 llama_sampler_accept：sample 内部已经 accept 过了 */
    *out_token = (int32_t) tok;
} else {
    /* argmax 路径，行为不变 */
}
```

**采样之后不得再 accept。** `dumpbin /DISASM llama.dll` 里 `llama_sampler_sample` （`0x1800F01B0`）先 `mov rax,[r13]` / `mov r8,[rax+10h]` / `call r8` 取 `apply`，再 `mov r8,[rax+8]` / `call r8` 走 `accept`（参数即选中的 token）；行为实验（`probe_sampler.c` Q1）对同一 token accept 两次，其 logit 从 -2.4999 变成 -4.4999 （差 2.0 = 一次额外惩罚），`changed=1/64`。accept 不是幂等空操作：penalties 用环形窗口，重复 accept 同一 token 会占两个槽位、有效历史长度减半（Q1b 验证），用了 penalties 的请求因此得到系统性偏高的重复惩罚，重复修得越狠，长文本越容易退化成重复短词。

**默认必须是 argmax。** 不调 setter 时 `handle->sampler == NULL`，一切照旧——既有测试与吞吐数据都建立在确定性解码上。`idx` 传 `n_tokens - 1` 的前提是本入口对每一行都标了 `batch.logits = 1`，最后一行的 output index 即 `n_tokens - 1`；该取值已实测确认（§6.1）。

---

## 4. 新增 API

`typedef struct qlh_keep_head`（同文件 L95 起）加一个字段 `struct llama_sampler * sampler;`，`NULL` 即 argmax（默认）。参数用 setter 传，不用环境变量：同一进程可能有多个实例需要不同参数，而环境变量是进程级的；风格照 L724 的 `qlh_kh_set_skip_hidden_logits`。签名 8 个参数，三个惩罚量一起传—— `llama_sampler_init_penalties` 本来就一次接受 `penalty_last_n` / `penalty_repeat` / `penalty_freq` / `penalty_present`，拆成两次调用没有意义，还会建成两条 penalties 采样器：

```c
int32_t qlh_kh_set_sampling(void * handle_void,
                            float    temperature,     /* <=0 = greedy；-1.0 = 拆除，回 argmax */
                            float    top_p,           /* >=1.0 = 关闭 */
                            int32_t  top_k,           /* <=0 = 关闭 */
                            float    repeat_penalty,  /* 1.0 = 关闭 */
                            int32_t  repeat_last_n,   /* 0 = 关闭 / 用默认 64 */
                            float    penalty_freq,    /* frequency_penalty */
                            float    penalty_present, /* presence_penalty */
                            uint32_t seed);

/* 只读回显：§5 要求回显实际生效的参数，而 clamp 规则全在 C 侧，Python 复算会漂移。
   返回 1 = 有采样链，0 = argmax 缺省 */
int32_t qlh_kh_get_sampling(void * handle_void,
                            float * temperature, float * top_p, int32_t * top_k,
                            float * repeat_penalty, int32_t * repeat_last_n,
                            float * penalty_freq, float * penalty_present,
                            uint32_t * seed);
```

`temperature == -1.0f`（`QLH_SAMPLING_DISABLE`）是"拆掉采样链、回到 argmax"的显式动作，不是多余的：`temperature <= 0` 走 greedy 链，若用它表达"关闭"，"客户端没传采样参数"与"要求 temperature=0"就无法区分，长驻服务还会把上一个请求的配置留给下一个请求。选 greedy 链而不是把 `sampler` 置回 NULL，是因为 §6.1 的确定性回归必须真的走到新代码：置 NULL 时回归永远走旧 argmax 路径，`idx` 取值、accept 状态机、greedy 链全都测不到，测试会通过但证明不了任何事；代价 0.15 ms/step（§6.3），输出与 argmax 逐位相同（§6.1 实测 20/20，另有 2000 轮纯采样层对照零分歧）。

链的添加顺序（llama.cpp 惯例）：`penalties`（`repeat_penalty != 1.0 && repeat_last_n > 0`，或 freq/present 非零）→ `top_k`（`0 < top_k < n_vocab`）→ `top_p`（`0 < top_p < 1.0`）→ `temp` + `dist` （`temperature > 0`，dist 必须最后）；`temperature <= 0` 时改用 `greedy`，不带 temp/dist。

生命周期：`qlh_kh_reset`（L736）调 `llama_sampler_reset`，否则 penalties 历史跨序列污染；`qlh_kh_close`（L748）调 `llama_sampler_free`；重复调 `set_sampling` 先 free 旧的再建；导出在 `shim/qlh_keep_head.def` 加 `qlh_kh_set_sampling`。

---

## 5. Python 侧接线

`keep_head_worker.py` 声明 argtypes、新增转发方法、请求路由加 `op == "set_sampling"`；`keep_head.py` / `engine.py` 是转发层，风格与 `tokenize` / `piece` 一致；`server.py` 每次请求开始前调到末段，`UNSUPPORTED` 里移除对应项，`slimarc` 段回显**实际生效**的参数。

OpenAI 字段映射：

| OpenAI | shim |
| --- | --- |
| `temperature` | `temperature` |
| `top_p` | `top_p` |
| `seed` | `seed` |
| `frequency_penalty` | `penalty_freq` |
| `presence_penalty` | `penalty_present` |
| `top_k`（非标准） | `top_k` |
| `logit_bias` | 暂不支持，继续标 ignored |

**支持范围**：`logit_bias` 与 `min_p` 不在支持列表。前者原因是从未实现；后者的原因不是 llama.cpp 不支持——`llama.dll` 确实导出 `llama_sampler_init_min_p`——而是 §4 的 setter 签名里没有 `min_p` 参数，且用 `top_p` 近似 `min_p` 是错误的语义。`server.py` 把它留在 `UNSUPPORTED` 并写明是**签名没有该参数**；要加的话是 setter 加第 9 个参数、链里插在 `top_k` 之后。

**回显实际生效的参数**：`LLAMA_DEFAULT_SEED` 的值是 `0xFFFFFFFF`，但它是"请随机化"的哨兵，不是字面种子——实测两条 `llama_sampler_init_dist(0xFFFFFFFF)` 链分别拿到 `0xBEEE9BDC` / `0x2ECD8F8F`，产出的序列也不同（`probe_sampler.c` Q4），实际值随后由 `llama_sampler_get_seed()` 给出。shim 因此在 `sampler != NULL` 时用 `llama_sampler_get_seed(chain)` 回填回显字段：照原样回显 0xFFFFFFFF 会让客户端拿到一个它并没有用过的种子，也就无法复现本次输出。端到端测试（`_test/e2e_server_sampling.py` 第 [6] 项）验证了不传 seed 时回显的是具体数值（实测 `3010234931`），不是哨兵。

---

## 6. 验收判据

三项均已实测通过（2026-10-29，2B 两段，`ctx=1024`，不绑核）。原始数据在 `H:\slimarc-proto\shim\_test\sampling-result.json`，脚本 `accept_sampling.py`。

### 6.1 确定性回归

`temperature <= 0` 时输出必须与 argmax **逐 token 相同**。实测 20 个 token 逐位相同、0 处分歧，argmax（完全不调 setter，`handle->sampler == NULL`）与 greedy 链的前 12 个 token 都是 `15987 73451 148391 162381 220 13424 61951 12 38011 12 12 12 …`。 `idx` 也由此确认：prefill 一次 decode 32 行、只采一个 token，若 `idx` 差一位（取到第 0 行而非第 31 行）首 token 就会不同，而实测首 token 同为 15987，所以 `n_tokens - 1` 成立，不需要试 `0` / `-1`。纯采样层另有对照：`llama_sampler_init_greedy` 与手写 argmax 在 2000 轮随机 logits 上零分歧（其中 500 轮强制制造并列，验证"首个最大值"与"排序后取首"在并列时一致， `probe_sampler.c` Q2）。

### 6.2 可复现

`temperature > 0` 且 `seed` 固定时，两次运行产生相同序列：`t=0.8`、`top_p=0.9`、 `top_k=40`、`seed=42` 两次运行 20 token 完全一致（run A 与 run B 都是 316 558 258 349 519 2586 12 16 15 15 15 12 …）。种子也确实在起作用（否则"可复现"可能只是 "忽略了 seed"）：`seed=43` 得到 32/32 个位置全都不同。

### 6.3 性能

2B 两段、不绑核、`ctx=1024`。基线：HEAD prefill(1 tok) 197 ms、TAIL 首 token 107 ms、STEP `head=33 tail=18 piece=0.2` ms，约 50 ms/step。方法：3 轮交替（采样开 / 关）取中位数并报噪声底，不用单次 A/B 对照——head 段不含采样代码，按"各跑一次"仍会得到 **-5.16%（采样组更快）** 这种不可能的结论，说明漂移大于待测差异本身。

| 配置 | step | head | tail |
| --- | ---: | ---: | ---: |
| greedy（argmax 等价） | 47.97 ms | 30.24 | 17.73 |
| `t=0.8 top_p=0.9 top_k=40` | 47.58 ms | 30.14 | 17.44 |

**Δ = -0.83%，远低于 10% 门槛。** head 段不含任何采样代码，它的 -0.10 ms 波动就是本次测量的噪声底，所以性能差落在噪声里——这不是"更快"，是"测不出来"。50 ms 的 step 上 0.15 ms 的开销测不出来，只能从隔离测量里得到，不能从"端到端跑一遍觉得差不多"里推断。改用纯采样层测（`probe_sampler.c` Q5，`n_vocab=248320`，即 2B 的真实词表， 40 次迭代取均值），两次独立运行的区间：

| 路径 | 单次耗时 | 相对 argmax | 占 50 ms step |
| --- | ---: | ---: | ---: |
| argmax（现有路径） | 0.135 – 0.167 ms | — | 0.27% – 0.33% |
| greedy 采样器 | 0.250 – 0.316 ms | +0.12 – 0.15 ms | +0.23% – 0.30% |
| top_k+top_p+temp+dist | 0.242 – 0.324 ms | +0.11 – 0.16 ms | +0.21% – 0.31% |
| 以上 + penalties | 0.527 – 0.905 ms | +0.39 – 0.74 ms | +0.78% – 1.48% |

结论：默认采样开销约 0.2%–0.3%，最重的惩罚配置 0.8%–1.5%，都远在 10% 以内。"`O(n log k)` 不该显著变慢"的推断成立，但顺序要说对：**主要成本是构造 24.8 万条候选数组并做部分排序，不是 log k 那一项**；penalties 最贵是因为它要遍历整个词表查重复。两次测量相差近一倍说明这台机器上纯 CPU 计时有 20%–40% 的抖动，所以给出区间而不是单点。

### 6.4 状态隔离

三项，全部通过：

- **配置不泄漏**：请求 A 要求 `t=0.9` + 惩罚，请求 B 什么都不传 → B 回到 `mode=argmax`（不是
  `greedy`、不是 `sampling`），`top_p` / `repeat_penalty` 回到中性值（`_test/state_sampling.py`）。
- **penalties 历史不跨序列**：`qlh_kh_reset` 调 `llama_sampler_reset` 后，同 seed 同
  输入的两个序列输出一致，且序列非退化（3 个不同 token）。
- **重复替换不泄漏**：连续 200 次 `set_sampling`，worker 私有内存 0.8 MB → 0.8 MB
  （+0.0 MB）；泄漏一条 24.8 万条目链约 3 MB，200 条会是数百 MB，可分辨。

---

## 7. 约束

- 不改跨机协议：段间仍只传 hidden，采样只在末段本地发生（`protocol.py` / `stage_worker.py` 的帧格式没动，`op` 只是 meta 键，与已有 `tokenize` / `piece` 同一机制）
- 不改 argmax 默认行为；不用环境变量传采样参数；不自己写采样算法；不动 `topology.py` / `test_topology.py` 的校验逻辑
- 头文件不需要改：`llama_sampler_*` 全套规范声明在 `build-headers/llama.h` 里，`llama_sampler_init_greedy` / `llama_sampler_get_seed` 也都在

---

## 8. 环境备忘

- 解释器 `G:\C\PYT\qlh\.venv-test\Scripts\python.exe`，编译 `pwsh -File H:\slimarc-proto\build_shim_msvc.ps1`
- shim 工作副本 `H:\slimarc-proto\shim\qlh_keep_head_gpu.c`（提交走 `slimarc/shim/`）
- 跑之前确认 `available > 12 GB`；2B 两段占 1.44 GB
- 不绑核：实测把单 step 从 50 ms 拖到秒级（`MEMORY-LIMITS.md` §6）
- 测试进程用完清掉：`Get-Process python | Stop-Process -Force`

