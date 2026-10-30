# slimarc — 层段分布式推理原型

> 状态：**原型（P1/P2 完成，三段端到端待跨机验证）**
> 更新日期：2026-10-29

Qwen3.6-35B-A3B（MoE，40 层，22 GB）按层切成段，每段作为独立的 llama.cpp 实例，
跨设备串行接力（一个 token 依次经过各段，段间传 hidden state）。

**不追求通用**：不接 OpenAI 接口，不含 PyTorch / SQLite / Android，不做多段调度。
核心 1406 行 + 裁切工具 269 行，够用即止。

---

## 这套东西解决什么

MoE 的权重分布决定了优化方向：**专家占 88%（19.67 GB），但每 token 只用 8/256**。
把专家放进最快的设备、非专家留在主机，比"整层上卡"快 3 倍。

单机实测（详见 `docs/MEASUREMENTS.md`）：

| 方案 | tg128 |
| --- | ---: |
| 纯 CPU | 8.23 |
| **单进程 `-ncmoe 32`** | **28.02** |
| 两进程拆分 | 约 17 |

**两进程比单进程慢**，因为同一块 8 GB 显存要分摊给两段。拆分的价值在跨机——
那时资源是相加而非共享。

**装得下时不该拆。** 本机 31.6 GB 内存装 22 GB 模型有余，拆分是倒退；
它的价值只在"单机装不下"时（如未来 80B）。这一点必须记住，否则会误以为拆分能提速。

---

## 快速开始

### 依赖

```bash
pip install numpy psutil
# 仅裁切需要：
pip install gguf
```

### 1. 准备 shim

shim 是 C 写的 llama.cpp 封装，负责把 hidden 导出/注入。两种取法：

**本机有 CUDA（GPU 段）**：用 `shim/build_shim_msvc.ps1` + `shim/build_cuda_shim.ps1` 自行编译。

**纯 CPU 机器**：整个目录拷贝已编好的 CPU 版（约 10 MB，9 个 DLL）。
注意 CPU 版是旧版 shim，硬编码 `n_gpu_layers=0`——纯 CPU 场景正好需要。

`SA_SHIM` 环境变量可指向 shim 所在目录外的 `qlh_keep_head.dll`。

### 2. 切段

```bash
# head 段（保留前 8 层，不重编号）
python cut.py --src <整模> --dst stage-0-8.gguf   --keep-head 8 --manifest stage-0-8.json

# middle 段（保留 8..23，重编号为 blk.0..）
python cut.py --src <整模> --dst stage-8-24.gguf  --k 8 --end 24 --manifest stage-8-24.json

# tail 段（保留 24..末尾）
python cut.py --src <整模> --dst stage-24-40.gguf --k 24 --manifest stage-24-40.json
```

切点必须是 `full_attention_interval`（qwen35moe 是 4）的整数倍，否则层类型错位、
加载报 `missing tensor`。`cut.py` 会 fail-closed 拒绝。

峰值内存约 1.5 GB，与模型规模无关（memmap + 1 MiB 分块写）。

### 3. 配拓扑

改 `topology.toml`。`host` 是 loopback 时本进程拉起 worker，其他地址只连接。

```toml
[[stage]]
model      = "/path/stage-0-8.gguf"
host       = "127.0.0.1"
port       = 9101
layers     = 8
kind       = "gpu"
gpu_layers = 99
cpu_moe    = 0
cpus       = "0,1"
threads    = 3
```

⚠️ **`cpus` 别给太少**：每个 worker 有 5+ 活动线程，3 个逻辑核跑 32 层要 1692 ms，
6 个只要 70 ms（**24 倍差**）。详见 `docs/MEMORY-LIMITS.md`。

### 4. 跑

```bash
python chain_client.py --dry-run                  # 只校验拓扑与内存预算
python chain_client.py --gen 32 --repeat 2        # 真实推理
```

`chain_client.py` 有内存预检：本地各段权重之和超过 `available × 0.75` 会拒绝启动。
这是硬规则——Windows 超内存不报错，它会分页到 swap 把整机拖死。

### 5. 跨机

远端机器先起 worker：

```powershell
python stage_worker.py --role tail `
    --model <stage-24-40.gguf> --shim <shim 路径> `
    --host 0.0.0.0 --port 9103 `
    --ctx 4096 --threads 6 --layers 16 --n-gpu-layers 0 --n-cpu-moe 0
```

然后把 `topology.toml` 里该段的 `host` 改成远端 IP。

### 6. 验数值

```bash
python check_correct.py --full-model <整模> --prompt-tokens 16
```

对照 head 段产出的 hidden 与整模对应层的输入，要求逐位相同。
实测 `max abs diff = 0.000000e+00`。

---

## 组成

| 文件 | 行数 | 职责 |
| --- | ---: | --- |
| `config.py` | 70 | 7 个环境变量 |
| `protocol.py` | 66 | 6 种消息，长度前缀帧，hidden 走原始 float32 |
| `engine.py` | 83 | keep-head shim 封装 |
| `stage_worker.py` | 112 | 单个阶段进程（TCP 服务端） |
| `chain_client.py` | 144 | 拉起两个 worker，驱动贪心解码 |
| `runner.py` | 79 | 同进程两段串联 |
| `breakdown.py` | 62 | 分段耗时 |
| `check_correct.py` | 85 | 数值正确性对照 |
| `bench.py` / `smoke.py` / `probe_hops.py` | 171 | 测量与调试 |

---

## 前置：CUDA 版 keep-head shim

上游 QLH 的 shim 全是纯 CPU 构建（bin 目录里只有 `ggml-cpu.dll`，没有
`ggml-cuda.dll`，且 `mparams.n_gpu_layers` 硬编码为 0）。原型需要 GPU offload，所以：

1. **打补丁**：`git apply --ignore-whitespace qlh_keep_head-gpu.patch`
   相对 `scripts/model_tools/keep_head_shim/qlh_keep_head.c`，新增 80 行：
   - `n_gpu_layers` 与 `n_cpu_moe` 改为从 `SA_N_GPU_LAYERS` / `SA_N_CPU_MOE` 读取
   - 调 `ggml_backend_load_all_from_path()` 加载后端（需 `SA_BACKEND_DIR`）
   - 实现 `n_cpu_moe`：按 llama-bench 的方式给前 N 层专家张量钉 CPU buffer

2. **构建 llama.cpp（CUDA + 动态后端）**：

   ```
   -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_CUDA_ARCHITECTURES=89
   -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF
   ```

   `GGML_CPU_ALL_VARIANTS` 需要 `GGML_BACKEND_DL`（见 `ggml/src/CMakeLists.txt:373`）。
   脚本：`../build_cuda_shim.ps1`

3. **编译 shim（MSVC）**：`../build_shim_msvc.ps1`

   必须用 MSVC 而非 MinGW——CUDA 构建是 MSVC，跨 ABI 会在第一个
   struct-by-value 调用处崩。需要 `.def` 文件导出 `qlh_kh_*`（MinGW 默认导出全部，
   MSVC 一个都不导）。

4. **拷运行时**：`llama.dll` / `ggml*.dll` / `ggml-cpu-*.dll` / `cublas*` /
   `cudart64_*` / `MSVCP140.dll` / `VCRUNTIME140*.dll`

   ⚠️ **必须删除 `ggml-cpu.dll`**（非 DL 版本的旧文件）：它的 `ggml_backend_score`
   缺失，会让加载器选错 ISA 变体。

### 一个实测的坑

各 ISA 变体的 `ggml_backend_score` 在本机是：

```
x64=1  sse42=5  sandybridge=17  haswell=44
alderlake=0  icelake=0  cascadelake=0  cannonlake=0  skylakex=0
```

i9-13900H 上 `alderlake`/`icelake` **得 0 分**（视为不支持），`haswell` 就是本机可用的
最优变体。这不是 bug，是 CPUID 检测的正常结果。

---

## 裁层

切分点必须是 `full_attention_interval`（=4）的整数倍，否则层类型错位、加载失败：

```bash
python scripts/cut_layers.py --src <full.gguf> --dst gpu-0-8.gguf --keep-head 8
python scripts/cut_layers.py --src <full.gguf> --dst cpu-8-40.gguf --k 8
```

---

## 运行

```bash
export SA_GPU_MODEL=.../gpu-0-8.gguf
export SA_CPU_MODEL=.../cpu-8-40.gguf
export SA_GPU_LAYERS=8
export SA_CTX=4096

python check_correct.py --prompt-tokens 16     # 数值对照
python chain_client.py --gen 128 --repeat 3 \
       --threads 3 --gpu-cpus 0,1 --cpu-cpus 2,3,4,5,6,7
```

**必须绑核**：本机同时跑着 GUI 与工具链，背景占用约 2.8 核；测试负载只有 3 线程，
不绑核时波动可达 ±4 t/s。

`--repeat` 的第 1 轮永远是冷启动，稳态取第 2 轮起。

---

## 两条实测结论

**① 两进程最优线程数是 3，单进程是 8。** 两进程时两个 Python 主循环都要占核，
CPU 段本身又受内存带宽约束，线程多了只争用。

**② 显存溢出是静默的。** Windows WDDM 在超显存时不报错，把系统内存当显存用，只表现
为掉速。CPU 段一旦让专家上卡（`n_cpu_moe` 32→30），吞吐从 18.48 崩到 0.45。
所以切分点要留余量，不能贴边。

---

## 未做

- OpenAI SSE 接口
- 跨机（Surface）
- SLIM-ARC 运行时集成（其收益前提是"模型远大于内存"，本场景 22 GB / 15 GB
  预算下机制会激活，但单机拆分的瓶颈在显存分摊，不在页管理）
- 4 段以上切分
