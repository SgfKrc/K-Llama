# 采样器接入方案

> 状态：**已实现**（2026-10-29）
> 更新日期：2026-10-29
>
> 实现过程中发现的实测结论与本文档原稿不符的地方，见 §9「与原方案的偏差」。
> 该节是**判断依据**：与本文其余部分冲突时以 §9 为准。

当前解码是 argmax：`qlh_kh_forward_embd_token` 遍历 logits 取最大值，只把
`int32_t` token id 交给调用方。`temperature` / `top_p` / `seed` 没有可作用的对象，
`server.py` 接受这些参数后在响应里标注为 ignored。

本文件记录选定的实现方式与接入点，供后续实现时对照。

---

## 1. 三种做法与取舍

| 方案 | 跨机开销 | 结论 |
| --- | --- | --- |
| 传完整 logits（241,664 × f32） | 每 token **944 KB**，是 hidden 的 118 倍 | 不可行 |
| 只传 top-k 候选 | 每 token 约 320 B | 可行，但 top-p 失真 |
| **在 shim 内采样** | **0** | **选定** |

决定性的理由是多段链路的带宽特性：段间只传 hidden（8 KB/token）。采样挪到
Python 侧，要么传 logits 要么传候选，都会破坏这一点。采样在末段本地做完，
段间协议完全不用改。

---

## 2. 复用上游采样器，不自己写算法

`llama.dll` 导出了全套采样器 API，直接调用即可：

```
llama_sampler_chain_init / chain_add / sample / accept / reset / free
llama_sampler_init_penalties / top_k / top_p / min_p / temp / dist
```

签名（逐字取自 `build-headers/llama.h`）：

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

`llama_sampler_sample` 内部自己调 `llama_get_logits_ith`，shim 不需要再拿 logits
指针。正确性由上游保证，比自行实现 top-k / top-p / softmax 短得多。

**但"内部自己调"还包括 `accept`** —— 见 §9.1。原稿漏了这一点，导致多一次 accept。

`llama_sampler_init_greedy()` 也在导出表里（`dumpbin /EXPORTS` 已确认，共 36 个
`llama_sampler_*` / `llama_perf_sampler*` 符号），§4.2 的链在 `temperature <= 0` 时
用得到它。

---

## 3. 接入点

`shim/qlh_keep_head_gpu.c` 的 `qlh_kh_forward_embd_token`，argmax 在 L555-575。
保留非有限值检查，然后分两条路：

```c
if (handle->sampler != NULL) {
    const llama_token tok = llama_sampler_sample(handle->sampler, handle->ctx, n_tokens - 1);
    /* ⚠️ 不要加 llama_sampler_accept！sample 内部已经 accept 过了，见 §9.1 */
    *out_token = (int32_t) tok;
} else {
    /* 原 argmax 路径，行为不变 */
}
```

> **原稿在这里写了 `llama_sampler_accept(handle->sampler, tok);` 并注释"penalties
> 需要历史"。那行是错的，已删除。** 理由与实测见 §9.1。

**默认必须是 argmax。** 不调 setter 时 `handle->sampler == NULL`，一切照旧——
既有测试与吞吐数据都建立在确定性解码上。

`idx` 传 `n_tokens - 1` 的前提是本入口对每一行都标了 `batch.logits = 1`，
最后一行的 output index 即 `n_tokens - 1`。**已实测确认（§6.1）**：prefill 一次
decode 32 行、只采一个 token 的场景下，argmax 与 greedy 链给出的首 token 相同，
若 idx 错位会在这一步立刻分叉。

---

## 4. 新增 API

### 4.1 结构体字段

`typedef struct qlh_keep_head`（同文件 L95 起）加一个字段：

```c
struct llama_sampler * sampler;   /* NULL = argmax（默认） */
```

### 4.2 setter

参数用 setter 传，不用环境变量：同一进程可能有多个实例需要不同参数，而环境变量
是进程级的。风格照 L724 的 `qlh_kh_set_skip_hidden_logits`。

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
```

> **原稿的签名只有 7 个参数（缺 `penalty_freq` / `penalty_present`）**，而 §5 的
> 字段映射表又要求把 `frequency_penalty` / `presence_penalty` 映射进来 —— 两者矛盾。
> 实现按映射表补齐成 8 个参数。`llama_sampler_init_penalties` 本来就一次接受这三个
> 惩罚量，拆成两次调用既无意义也会建成两条 penalties 采样器。

另加一个只读回显入口（原稿没有，但 §5 要求"回显实际生效的参数"，而 clamp 规则
全在 C 侧，Python 复算会漂移）：

```c
int32_t qlh_kh_get_sampling(void * handle_void,
                            float * temperature, float * top_p, int32_t * top_k,
                            float * repeat_penalty, int32_t * repeat_last_n,
                            float * penalty_freq, float * penalty_present,
                            uint32_t * seed);   /* 返回 1 = 有采样链，0 = argmax 缺省 */
```

`temperature == -1.0f`（`QLH_SAMPLING_DISABLE`）是"拆掉采样链、回到 argmax"的显式
动作。这不是多余的：`temperature <= 0` 走的是 greedy 链（原因见 §9.2），若用它表达
"关闭"，那么"客户端没传采样参数"和"客户端要求 temperature=0"就无法区分，长驻服务
还会把上一个请求的配置留给下一个请求。

链的添加顺序（llama.cpp 惯例）：

1. `penalties`（`repeat_penalty != 1.0 && repeat_last_n > 0`，或 freq/present 非零时）
2. `top_k`（`top_k > 0 && top_k < n_vocab`）
3. `top_p`（`0 < top_p < 1.0`）
4. `temp` + `dist`（`temperature > 0`，dist 必须最后）
   —— 或 `greedy`（`temperature <= 0`，此时不带 temp/dist）

### 4.3 生命周期

- `qlh_kh_reset`（L736）：调 `llama_sampler_reset`，否则 penalties 历史跨序列污染
- `qlh_kh_close`（L748）：调 `llama_sampler_free`
- 重复调 `set_sampling`：先 free 旧的再建，避免泄漏

### 4.4 导出

`shim/qlh_keep_head.def` 加 `qlh_kh_set_sampling`。

---

## 5. Python 侧接线

- `keep_head_worker.py`：声明 argtypes、新增转发方法、请求路由加 `op == "set_sampling"`
- `keep_head.py` / `engine.py`：转发层，风格与 `tokenize` / `piece` 一致
- `server.py`：每次请求开始前调到末段；`UNSUPPORTED` 里移除对应项；
  `slimarc` 段回显**实际生效**的参数

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

---

## 6. 验收判据

三项均已实测通过（2026-10-29，2B 两段，`ctx=1024`，不绑核）。原始数据在
`H:\slimarc-proto\shim\_test\sampling-result.json`，脚本 `accept_sampling.py`。

### 6.1 确定性回归

`temperature <= 0` 时输出必须与现有 argmax **逐 token 相同**。

**实测：20 个 token 逐位相同，0 处分歧。**

```
argmax（完全不调 setter，handle->sampler == NULL）
    15987 73451 148391 162381 220 13424 61951 12 38011 12 12 12 …
temperature=0（走 greedy 链）
    15987 73451 148391 162381 220 13424 61951 12 38011 12 12 12 …
```

`idx` 取值同时得到确认：prefill 一次 decode 32 行、只采一个 token。若 `idx` 差一位
（取到第 0 行而非第 31 行），首 token 就会不同 —— 实测首 token 也是 15987，一致。
所以 `n_tokens - 1` 是对的，不需要试 `0` / `-1`。

补充证据：`llama_sampler_init_greedy` 与 shim 手写 argmax 在 2000 轮随机 logits
上零分歧（其中 500 轮强制制造并列，用来验证"首个最大值"与"排序后取首"在并列时
是否一致）。见 `shim/_test/probe_sampler.c` Q2。

### 6.2 可复现

`temperature > 0` 且 `seed` 固定时，两次运行产生相同序列。

**实测：`t=0.8, top_p=0.9, top_k=40, seed=42`，两次运行 20 token 完全一致。**

```
run A: 316 558 258 349 519 2586 12 16 15 15 15 12 …
run B: 316 558 258 349 519 2586 12 16 15 15 15 12 …
```

同时验证种子**真的在起作用**（否则"可复现"可能只是"忽略了 seed"）：
`seed=43` 得到 32/32 个位置全都不同。

### 6.3 性能

2B 两段、不绑核、`ctx=1024`。基线（文档原记录）：

```
HEAD prefill(1 tok) 197 ms
TAIL 1st token     107 ms
STEP head=33 tail=18 piece=0.2 ms        → 约 50 ms/step
```

**实测（3 轮交替取中位数，避免 A/B 两次之间的机器漂移）：**

| 配置 | step | head | tail |
| --- | ---: | ---: | ---: |
| greedy（argmax 等价） | 47.97 ms | 30.24 | 17.73 |
| `t=0.8 top_p=0.9 top_k=40` | 47.58 ms | 30.14 | 17.44 |

**Δ = -0.83%，远低于 10% 门槛。** head 段不含任何采样代码，它的 -0.10 ms 波动
就是本次测量的噪声底，所以性能差落在噪声里 —— 这不是"更快"，是"测不出来"。

为了给出一个**能分辨**的数字，改用纯采样层测（`probe_sampler.c` Q5，
`n_vocab=248320`，即 2B 的真实词表，40 次迭代取均值）。两次独立运行的区间：

| 路径 | 单次耗时 | 相对 argmax | 占 50 ms step |
| --- | ---: | ---: | ---: |
| argmax（现有路径） | 0.135 – 0.167 ms | — | 0.27% – 0.33% |
| greedy 采样器 | 0.250 – 0.316 ms | +0.12 – 0.15 ms | +0.23% – 0.30% |
| top_k+top_p+temp+dist | 0.242 – 0.324 ms | +0.11 – 0.16 ms | +0.21% – 0.31% |
| 以上 + penalties | 0.527 – 0.905 ms | +0.39 – 0.74 ms | +0.78% – 1.48% |

结论：默认采样开销约 0.2%–0.3%，最重的惩罚配置 0.8%–1.5%，都远在 10% 以内。
原文"O(n log k) 不该显著变慢"的推断成立，但顺序要说对：**主要成本是构造 24.8 万条
候选数组并做部分排序，不是 log k 那一项**；penalties 最贵是因为它要遍历整个词表
查重复。两次测量相差近一倍说明这台机器上纯 CPU 计时有 20%–40% 的抖动，所以给出
区间而不是单点。

### 6.4 状态隔离（原稿未要求，但长驻服务必须验证）

三项，全部通过：

- **配置不泄漏**：请求 A 要求 `t=0.9` + 惩罚，请求 B 什么都不传 → B 回到
  `mode=argmax`（不是 `greedy`、不是 `sampling`），`top_p` / `repeat_penalty` 都回到
  中性值。见 `_test/state_sampling.py`。
- **penalties 历史不跨序列**：`qlh_kh_reset` 调 `llama_sampler_reset` 后，同 seed
  同输入的两个序列输出一致（且序列非退化，3 个不同 token）。
- **重复替换不泄漏**：连续 200 次 `set_sampling`，worker 私有内存 0.8 MB → 0.8 MB
  （+0.0 MB）。泄漏一条 24.8 万条目链约 3 MB，200 条会是数百 MB，可分辨。

---

## 7. 约束

- 不改跨机协议：段间仍只传 hidden，采样只在末段本地发生
- 不改 argmax 默认行为
- 不用环境变量传采样参数
- 不自己写采样算法
- 不动 `topology.py` / `test_topology.py` 的校验逻辑
- 头文件改动只限 `build-headers/llama.h`，并写清原因

---

## 8. 环境备忘

- 解释器 `G:\C\PYT\qlh\.venv-test\Scripts\python.exe`
- shim 工作副本 `H:\slimarc-proto\shim\qlh_keep_head_gpu.c`，提交走 `slimarc/shim/`
- 编译 `pwsh -File H:\slimarc-proto\build_shim_msvc.ps1`
- 跑之前确认 `available > 12 GB`；2B 两段占 1.44 GB
- 不绑核：实测把单 step 从 50 ms 拖到秒级（`MEMORY-LIMITS.md` §3.4）
- 测试进程用完清掉：`Get-Process python | Stop-Process -Force`

---

## 9. 与原方案的偏差

原稿是判断依据，不是权威。以下每条都是实现时**实测或读反汇编**发现的，与原文冲突
时以本节为准。

### 9.1 §3 多了一行 `llama_sampler_accept` —— 会翻倍惩罚，已删除

原稿在 `llama_sampler_sample` 之后又调了一次 `llama_sampler_accept`，注释写
"penalties 需要历史"。**`llama_sampler_sample` 内部已经 accept 过了。**

两条独立证据：

1. **反汇编**（`dumpbin /DISASM llama.dll`，`llama_sampler_sample` @ `0x1800F01B0`）：
   函数体先 `mov rax,[r13]` / `mov r8,[rax+10h]` / `call r8`（`iface->apply`），
   取下 `selected` 的 token id 到 `ebx`，再 `mov r8,[rax+8]` / `call r8`
   （`iface->accept`，参数正是 `edx = ebx`）。`llama_sampler_i` 里 `accept` 在
   偏移 8、`apply` 在偏移 16，与头文件声明顺序一致。
2. **行为实验**（`_test/probe_sampler.c` Q1）：同一个 token accept 两次，该 token
   的 logit 从 -2.4999 变成 -4.4999（差 2.0 = 一次额外惩罚），`changed=1/64`。
   penalties 用的是**环形窗口**，重复 accept 同一 token 会占两个槽位，等于把有效
   历史长度减半（Q1b 单独验证了窗口会被挤动）。

   `accept` 不是幂等空操作 —— 所以原稿那行不是"多余但无害"，而是**改变采样分布**：
   用了 penalties 的请求会得到系统性偏高的重复惩罚，重复修复越狠，长文本越容易
   退化成重复短词。

### 9.2 `temperature <= 0` 建 greedy 链，而不是把 `sampler` 置回 NULL

原稿没规定这件事，但它决定 §6.1 的回归测试**是否真的测到新代码**。选择 greedy 链：

- 输出与 argmax **逐位相同**（§6.1 实测 20/20，另有 2000 轮纯采样层对照零分歧）。
- 若 `temperature=0` 只是把 `sampler` 置 NULL，那么 §6.1 的"确定性回归"永远走的是
  旧代码路径，`llama_sampler_sample` 的 `idx`、accept 状态机、greedy 链全都测不到 ——
  测试会通过，但证明不了任何东西。
- 代价 0.15 ms/step（§6.3）。

由此引出 §4.2 的 `QLH_SAMPLING_DISABLE = -1.0f`：必须有一个**显式**的"回到缺省"
入口，否则"没传参数"和"要求 temperature=0"无法区分，且长驻服务会把上一个请求的
采样配置留给下一个请求（§6.4 第一项就是在测这个）。

### 9.3 `LLAMA_DEFAULT_SEED` 是"随机抽一个"，不是字面值 —— 不能原样回显

`llama.h` 只写 `#define LLAMA_DEFAULT_SEED 0xFFFFFFFF`，没说不传 seed 时会发生什么。
实测（`probe_sampler.c` Q4）：两条 `llama_sampler_init_dist(0xFFFFFFFF)` 链分别拿到
`0xBEEE9BDC` / `0x2ECD8F8F`，产出的序列也不同 —— 即该值是"请随机化"的哨兵。
`llama_sampler_get_seed()` 随后返回**实际**种子。

所以 shim 在 `sampler != NULL` 时用 `llama_sampler_get_seed(chain)` 回填回显字段。
若照原样回显 0xFFFFFFFF，客户端会拿到一个它并没有用的种子，也就无法复现本次输出 ——
这正是 §5 要求"回显实际生效的参数"要避免的事。端到端测试（`_test/e2e_server_sampling.py`
第 [6] 项）验证了不传 seed 时回显的是具体数值（实测 `3010234931`），不是哨兵。

### 9.4 §4.2 的 setter 签名缺两个参数，与 §5 的字段映射表矛盾

原稿签名 7 个参数，是 `penalties(n_vocab, last_n, repeat, freq, present)` 拆开写却
漏了后两个；而 §5 明确要求映射 `frequency_penalty` / `presence_penalty`。
实现补齐为 8 个参数（§4.2 已更新）。

### 9.5 `min_p` 仍未支持，理由变了

原稿把 `min_p` 放在 `UNSUPPORTED` 里，但没有独立理由 —— 它和 `temperature` 一起被
去掉会让人以为现在支持了。`llama.dll` 确实导出了 `llama_sampler_init_min_p`，但
§4.2 定的 setter 签名没有 min_p 参数，而"用 top_p 近似 min_p"是错误的语义。
所以 `min_p` 留在 `UNSUPPORTED`，并在 `server.py` 里写明是**签名没有该参数**，
不是 llama.cpp 不支持。要加的话是 setter 加第 9 个参数 + 链里插在 top_k 之后。

### 9.6 §6.3 的性能验证方法有缺陷，已改

原稿只给了基线数字和"<10%"门槛，没说怎么测。首次按"采样开 / 关各跑一次"对比，
得到 **-5.16%（采样组更快）** 这种不可能的结论 —— head 段完全没有采样代码，却也有
2% 的漂移，说明单次 A/B 的机器漂移大于待测差异本身。

改成 3 轮交替取中位数后 Δ=-0.83%，并把 head 段的波动当作**噪声底**报告。
再加一层能在纯采样层分辨的测量（§6.3 表格），因为 50 ms 的 step 上，
0.15 ms 的开销无论如何都测不出来 —— 那个数字只能从隔离测量里得到，不能从
"端到端跑一遍觉得差不多"里推断。

### 9.7 与原稿一致的部分

- 采样放在 shim 内，跨机协议零改动（`protocol.py` / `stage_worker.py` 的帧格式没动，
  `op` 只是 meta 里的键，与已有 `tokenize` / `piece` 同一机制）。
- 默认 argmax 行为逐 token 不变（§6.1）。
- 参数走 setter，不走环境变量。
- 不自己实现采样算法，全部用 `llama.dll` 导出的采样器。
- `topology.py` / `test_topology.py` / 头文件均未改动，无需在 `llama.h` 写原因。
  头文件**不需要**改：`llama_sampler_*` 全套规范声明在 `build-headers/llama.h` 里，
  `llama_sampler_init_greedy` / `get_seed` 也都在。
