# 内存预算与硬限制

> 状态：**现行，硬约束**
> 更新日期：2026-10-29
>
> 起因：2026-10-29 凌晨，三段工件同时加载导致整机黑屏两次。

---

## 1. 本机内存账

| 项 | 数值 |
| --- | ---: |
| 物理内存 | 31.6 GiB |
| 典型可用（含 GUI、IDE、Harness） | 20–26 GiB |
| swap | 27.5 GiB（**严重使用时会拖死桌面**） |

**关键**：Windows 不会因超内存而拒绝分配，而是分页到 swap。一旦 swap 打满，
整机失去响应——不是报告 OOM，是黑屏。这是本项目的**最高优先级约束**。

---

## 2. 三段工件的真实成本

| 段 | 文件 | 权重 |
| --- | --- | ---: |
| `[0,8)` | `stage-0-8.gguf` | 4.53 GB |
| `[8,24)` | `stage-8-24.gguf` | 8.36 GB |
| `[24,40)` | `stage-24-40.gguf` | 8.96 GB |
| **合计** | | **21.85 GB** |

但权重只是下限。每段还要：

- KV cache（`n_ctx=4096`，40 层里 16 层的 KV ≈ 数百 MB）
- compute buffer（GPU 段约 1 GB）
- Python + numpy 运行时（每进程约 100 MB）

**实测三段同载 = 黑屏。** 21.85 GB 权重本身已接近可用内存上限。

---

## 3. 硬规则

### 3.1 启动前必须做内存预检

`chain_client.py` 已内置：

```python
MEM_SAFETY = 0.75

def mem_budget_bytes() -> int:
    import psutil
    return int(psutil.virtual_memory().available * MEM_SAFETY)
```

启动时比较 `本地各段权重之和` 与 `budget`，超了直接拒绝：

```
local stages: 3  weights 21.85 GiB  budget 18.66 GiB
refusing to start: local artifacts total 21.85 GiB but only 18.66 GiB is safe
```

**不要绕过这个检查。** 它的存在就是因为绕过会黑屏。

### 3.2 同机最多两段

| 组合 | 权重 | 可行性 |
| --- | ---: | --- |
| 段0 + 段1 | 12.89 GB | ✅ 安全 |
| 段0 + 段2 | 13.49 GB | ✅ 安全 |
| 段1 + 段2 | 17.32 GB | ⚠️ 临界，需可用 > 23 GB |
| **三段全载** | **21.85 GB** | ❌ **禁止** |

### 3.3 三段跑通的条件

必须满足以下之一：

1. **把一段放远端**（ROG，15.15 GB 内存跑 16 层段 8.96 GB，占 59%）
2. 本机可用内存 > 29 GB（21.85 / 0.75）
3. 用更小的模型或更激进的量化

**结论：三段实验必须先配好 ROG，本机单跑三段不可行。**

### 3.4 绑核：给少了会崩 24 倍

**实测数据**（tail 段跑 32 层的单段对照）：

| 绑核 | median | 说明 |
| --- | ---: | --- |
| 全部 20 逻辑核 | 106.7 ms | |
| **3 核（`2,3,4`）** | **1692.1 ms** | **24 倍惩罚** |
| 6 核（`2,3,4,5,6,7`） | **70.3 ms** | 最优 |

**根因**：每个 worker 有 5+ 活动线程（`stage_worker` 主线程、`keep_head` 主线程、
llama 的 3 个计算线程），3 个逻辑核装不下。而且 i9-13900H 是 6P+8E 异构，
相邻逻辑核可能是同一物理 P-core 的超线程对（0/1、2/3……），给 `2,3,4`
实际只有 1.5 个物理核。

**规则：每段至少 6 个逻辑核。** `topology.toml` 曾误配成 3 核，已修正。

### 3.5 测试纪律

- 任何测试前先看 `available`，低于 12 GB 就停
- 测试完检查残留进程：
  ```powershell
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Select-Object ProcessId, @{n='MB';e={[math]::Round($_.WorkingSetSize/1MB,0)}}
  ```
- 残留的 `stage_worker.py` / `keep_head_worker.py` 必须杀掉——它们各自持有整个模型
- **诊断脚本本身也可能泄漏**：写死的 `while` 循环、无界 buffer 解析会吃满内存。
  2026-10-29 凌晨一个拓扑探测脚本死循环吃掉 10.4 GB，差点重演黑屏。
  诊断脚本必须带超时和迭代上限。

---

## 4. 崩溃时怎么恢复

1. **先杀进程，别等系统响应**
   ```powershell
   Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
     Where-Object { $_.CommandLine -match 'slimarc|chain_client' } |
     ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
   ```
   注意：PowerShell 里 `$pid` 是只读变量，循环变量不能叫 `$pid`。

2. 内存通常几秒内回升（实测 16.9 GB → 26.2 GB）

3. swap 不会立即释放，系统会慢慢回收；不必强求

---

## 5. 为什么不用 `--dry-run` 之外的测试

`--dry-run` 只读 TOML 与文件大小，不加载模型，内存占用可忽略：

```powershell
& $py chain_client.py --dry-run
```

它验证：
- 拓扑合法（层数恰好 40、切点 4 倍数、端口不重复）
- 工件存在
- 内存预算足够

**除了真实加载，其余检查都不需要模型。优先用静态检查。**
