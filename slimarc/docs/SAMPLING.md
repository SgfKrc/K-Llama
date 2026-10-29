# 采样器接入方案

> 状态：方案已定，未实现
> 更新日期：2026-10-29

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

---

## 3. 接入点

`shim/qlh_keep_head_gpu.c` 的 `qlh_kh_forward_embd_token`，argmax 在 L555-575。
保留非有限值检查，然后分两条路：

```c
if (handle->sampler != NULL) {
    const llama_token tok = llama_sampler_sample(handle->sampler, handle->ctx, n_tokens - 1);
    llama_sampler_accept(handle->sampler, tok);   /* penalties 需要历史 */
    *out_token = (int32_t) tok;
} else {
    /* 原 argmax 路径，行为不变 */
}
```

**默认必须是 argmax。** 不调 setter 时 `handle->sampler == NULL`，一切照旧——
既有测试与吞吐数据都建立在确定性解码上。

`idx` 传 `n_tokens - 1` 的前提是本入口对每一行都标了 `batch.logits = 1`，
最后一行的 output index 即 `n_tokens - 1`。这一点必须实测确认（见 §6.1），
不要凭推断。

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
                            float   temperature,     /* <=0 = greedy */
                            float   top_p,           /* >=1.0 = 关闭 */
                            int32_t top_k,           /* <=0 = 关闭 */
                            float   repeat_penalty,  /* 1.0 = 关闭 */
                            int32_t repeat_last_n,   /* 0 = 关闭 */
                            uint32_t seed);
```

链的添加顺序（llama.cpp 惯例）：

1. `penalties`（`repeat_penalty != 1.0 && repeat_last_n > 0` 时）
2. `top_k`（`top_k > 0`）
3. `top_p`（`top_p < 1.0`）
4. `temp`
5. `dist`（必须最后）

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

### 6.1 确定性回归

`temperature <= 0` 时输出必须与现有 argmax **逐 token 相同**。这是回归基线。
同时用它验证 `idx` 取值：若差一位，说明 output index 编号不同，改 `0` 或 `-1`
重试，直到一致。

### 6.2 可复现

`temperature > 0` 且 `seed` 固定时，两次运行产生相同序列。

### 6.3 性能

2B 两段（`topology-2b.toml`）、不绑核、`ctx=1024` 的现有基线：

```
HEAD prefill(1 tok) 197 ms
TAIL 1st token     107 ms
STEP head=33 tail=18 piece=0.2 ms        → 约 50 ms/step
```

采样后单 step 增幅应 **< 10%**。上游实现是 O(n log k)，不该显著变慢。

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
