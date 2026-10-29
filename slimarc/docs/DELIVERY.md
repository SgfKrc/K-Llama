# 交付清单与 ROG 部署

> 状态：**待提交**
> 更新日期：2026-10-29

---

## 一、首期已完成

### 1.1 代码（`slimarc/`，1406 行 + 裁切工具 269 行）

| 模块 | 行数 | 职责 | 验证状态 |
| --- | ---: | --- | --- |
| `protocol.py` | 66 | 6 种消息，长度前缀帧，hidden 走原始 f32 | ✅ 两段链路跑通 |
| `config.py` | 43 | 环境变量（7 个） | ✅ |
| `topology.py` | 166 | TOML → `tuple[Stage, ...]`，10 条校验 | ✅ 26 单测通过 |
| `keep_head.py` | 227 | shim 客户端（子进程架构） | ✅ 逐位正确 |
| `keep_head_worker.py` | 201 | shim worker（隔离 CRT 与后端注册表） | ✅ |
| `engine.py` | 85 | 单段引擎封装 | ✅ |
| `stage_worker.py` | 132 | 单段 TCP 服务端 | ✅ |
| `chain_client.py` | 333 | N 段链式客户端 + 内存预检 | ✅ 两段 10.6 t/s |
| `cut.py` | 269 | GGUF 裁层（三模式） | ✅ SHA256 与旧工具一致 |
| `check_correct.py` | 74 | 数值对照 | ✅ max abs diff = 0 |
| `runner.py` | 79 | 同进程两段串联（工具用） | ✅ |

### 1.2 已验收的硬指标

| 指标 | 结果 |
| --- | --- |
| 裁层数值无损 | `max abs diff = 0.000000e+00`，`cosine = 1.000000119` |
| 裁切工具字节一致 | 三种模式写出的文件 SHA256 与旧工具**完全相同** |
| 跨进程链路 | 两段 `tg32 = 10.6 t/s`（tail 段 81 ms） |
| fail-closed | 4 种非法切点全部 rc=2，错误含 `full_attention_interval` |
| 单进程基线 | `-ngl 99 -ncmoe 32` = **28.02 t/s** |

### 1.3 文档（14 份）

| 文档 | 内容 |
| --- | --- |
| `ARCHITECTURE.md` | 架构定稿（两路径正交、设备切分、协议、配置） |
| `MEASUREMENTS.md` | 全部实测数据（单进程扫描、两进程、分段方差） |
| `MEMORY-LIMITS.md` | 内存硬约束（黑屏教训、绑核 24 倍惩罚） |
| `CUT-TOOL.md` | 裁切工具验证报告 |
| `DECISIONS.md` | 保留/抛弃决策（基于旧仓库盘点） |
| `INVENTORY.md` | 旧仓库 232 文件 / 129,337 行全量盘点 |
| `BASELINE.md` | 层段接力链路盘点（含 3 处设计文档纠错） |
| `PRIMA-LESSONS.md` / `PRIMA-ANALYSIS.md` | prima.cpp 借鉴结论 |
| `SLIMARC-INTEGRATION.md` | SLIM-ARC 接入障碍（不支持 qwen35moe） |
| `TESTING.md` | 测试策略（反 CI 脆性） |
| `FIXES.md` | 已修缺陷（含 shim 的 C bug） |
| `PROGRESS.md` / `DESIGN.md` | 早期设计 |

---

## 二、提交到 git

### 2.1 应提交

```
slimarc/                    # 1406 行核心 + 269 行裁切 + 测试
  *.py
  topology.toml
  README.md
docs/slimarc/               # 文档（建议放这个前缀下）
  *.md
shim/
  qlh_keep_head_gpu.c       # 改版 shim 源码（+80 行）
  qlh_keep_head.def         # MSVC 导出清单
  qlh_keep_head-gpu.patch   # 可应用补丁
tools/                      # 可选：build_cuda_shim.ps1 / build_shim_msvc.ps1
```

### 2.2 不应提交

| 项 | 原因 |
| --- | --- |
| `shim/bin/`（553 MB） | 二进制，应放 release 或构建产出 |
| `build/` | 构建中间产物 |
| `exp/`（14 MB） | 实测日志，选几个代表性的进 docs 即可 |
| `SLIM-ARC-main/` / `SLIM-ARC-sparse/` | 第三方克隆，用 `.gitignore` |
| `slimarc-trial/` | 临时试用目录 |
| `F:\qlh-models\*` | 模型，不进 git |

### 2.3 建议的 `.gitignore`

```gitignore
slimarc/build/
slimarc/__pycache__/
shim/bin/
shim/_gen/
shim/_test/
exp/
SLIM-ARC-main/
SLIM-ARC-sparse/
slimarc-trial/
*.gguf
```

---

## 三、ROG 需要拷什么

### 3.1 结论：只需要 3 样，共约 9 MB 代码 + 8.96 GB 模型

| # | 内容 | 大小 | 说明 |
| --- | --- | ---: | --- |
| 1 | **CPU 版 shim**（9 个 DLL） | **9.9 MB** | 从 `G:\C\PYT\qlh\build\keephead\build-cpu-skiplogits\bin\` 整个目录 |
| 2 | **`slimarc/` 目录** | 约 250 KB | Python 代码 |
| 3 | **`stage-24-40.gguf`** | **8.96 GB** | tail 段工件 |

**不需要**：CUDA 版 shim（553 MB）、整模（20.8 GB）、其他两段。

### 3.2 为什么用 CPU 版 shim

已实测验证（本机模拟）：

```
用 build-cpu-skiplogits/bin 的 shim 加载 stage-24-40.gguf
→ LOADED n_embd=2048  n_layer=16  ✅
```

它是**旧版 shim**（不含 `SA_N_GPU_LAYERS` / `SA_N_CPU_MOE` / `SA_BACKEND_DIR`），
但硬编码 `n_gpu_layers=0`——**这正是 ROG 纯 CPU 需要的**。

### 3.3 模型：两个方案

#### 方案 A：直接拷切好的 tail 段（推荐，省时）

拷贝 `F:\qlh-models\stage-split-3\stage-24-40.gguf`（8.96 GB）。

#### 方案 B：ROG 上有整模则本地切

```powershell
python cut.py --src <整模路径> --dst stage-24-40.gguf --k 24
```

- 峰值内存仅 **1.5 GB**（实测，与模型规模无关）
- 耗时约 3–5 分钟
- 缺点：需要 ROG 上有 20.8 GB 整模

**若 ROG 要下载模型，方案 A 更快**（8.96 GB vs 20.8 GB）。

### 3.4 ROG 上的目录结构

```
D:\slimarc\                       # 或任意盘
  slimarc\                        # Python 代码
    stage_worker.py
    keep_head.py
    keep_head_worker.py
    engine.py
    protocol.py
    config.py
    topology.py
    cut.py
  shim\bin\                       # 9 个 CPU DLL（9.9 MB）
    qlh_keep_head.dll
    libllama.dll
    ggml.dll
    ggml-base.dll
    ggml-cpu.dll
    libgcc_s_seh-1.dll
    libgomp-1.dll
    libstdc++-6.dll
    libwinpthread-1.dll
  models\
    stage-24-40.gguf              # 8.96 GB
```

### 3.5 ROG 上启动命令

```powershell
cd D:\slimarc\slimarc
python stage_worker.py `
    --role tail `
    --model D:\slimarc\models\stage-24-40.gguf `
    --shim D:\slimarc\shim\bin\qlh_keep_head.dll `
    --host 0.0.0.0 --port 9103 `
    --ctx 4096 --threads 6 --layers 16 `
    --n-gpu-layers 0 --n-cpu-moe 0
```

**注意**：
- `--host 0.0.0.0` 才能接受外部连接（不是 `127.0.0.1`）
- `--threads 6`：ROG 是 6 核 12 线程，给 6
- **不需要 `--cpus`**：ROG 是同构 x86，没有 P/E 核
- 需要 Python 3.10+ 与 `numpy`、`psutil`

### 3.6 ROG 上的 Python 依赖

```powershell
pip install numpy psutil
```

`gguf` 只在裁切时需要，跑 worker 不需要。

---

## 四、master 侧的配置改动

把 `topology.toml` 里 tail 段的 host 改成 ROG 的局域网 IP：

```toml
[[stage]]
model      = "F:/qlh-models/stage-split-3/stage-24-40.gguf"
host       = "192.168.1.XX"      # ← 改成 ROG 的实际 IP
port       = 9103
layers     = 16
kind       = "tail"
gpu_layers = 0
cpu_moe    = 0
cpus       = ""
threads    = 6
```

改完后 master 只加载段 0 与段 1（12.89 GiB），内存预算：
`12.89 / 0.75 = 17.2 GiB` —— 本机常态可用 20–26 GiB，**通过**。

---

## 五、还需要做什么

### 5.1 首期未完成项

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 三段端到端 | ❌ | 必须等 ROG 就位 |
| SLIM-ARC 接入 | ❌ | 不支持 `qwen35moe`，需改补丁 15–25 行 |
| Rust 控制面 | ❌ | 计划在实验之后 |
| 逐 token argmax 全链对照 | ❌ | 两段已验 hidden 逐位一致，全链未验 |
| OpenAI SSE | ⛔ 不做 | 本期范围外 |

### 5.2 建议的实验顺序

1. ROG 配好 → master 改 `topology.toml` 的 host
2. **先 `--dry-run`** 确认内存预算通过
3. 跑 `--gen 32 --repeat 2`，看三段是否串通
4. 若通了，跑 `check_correct.py` 验数值
5. 记录 ROG 每层耗时与真实 RTT
