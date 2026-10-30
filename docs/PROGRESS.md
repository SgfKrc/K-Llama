# 原型进展与一个关键发现

> 状态：**进行中（P1 已通过，吞吐待 CUDA shim）**
> 更新日期：2026-10-28

---

## 1. P1 已通过：两段拆分数值完全一致

`gpu-0-8.gguf`（前 8 层，`nextn` 模式）与整模（`layer_inp cut_layer=8`）的 hidden 输出：

```
reference shape=(8, 2048)  candidate shape=(8, 2048)
max abs diff : 0.000000e+00
cosine       : 1.000000000
result: PASS
```

**逐位相同**，说明裁层工具无损、两段串联的数值基础可靠。

两段关键工件的实测能力（shim 加载）：

| 段 | 工件 | 大小 | 张量数 |
| --- | --- | ---: | ---: |
| GPU | `gpu-0-8.gguf` | 5.03 GB | 149 |
| CPU | `cpu-8-40.gguf` | 16.81 GB | 587 |

## 2. 一个必须记录的架构发现

**QLH 现存的 keep-head shim 全部是纯 CPU 构建，没有任何 CUDA 版本。**

核对过项目里所有 `qlh_keep_head.dll`：

- `build/keephead/build-cpu-skiplogits/bin/` 只有 `ggml-cpu.dll`，无 `ggml-cuda.dll`
- `build/r6-unified-20261021/cuda-dist/.../keep_head/` 名为 CUDA dist，其 `keep_head/`
  子目录里同样只有 `ggml-cpu.dll`
- CUDA 构建（`build/moe-cache-20261026/b11889/`）把 llama 静态链进 exe，
  **没有 `libllama.dll`**，而 shim 依赖这个 DLL

原因合理：QLH 的 GPU 段用 PyTorch，CPU 段用 llama.cpp，shim 从未被要求支持 CUDA。

**对本原型的直接影响**：两段串联实测只有 **3.71 t/s**，因为 GPU 段那 8 层实际跑在
CPU 上。这与单进程 `-ncmoe 32` 的 28.47 t/s 差 7.7 倍。

**因此必须为原型单独构建 CUDA 版 shim。** 构建脚本 `H:\slimarc-proto\build_cuda_shim.ps1`：

```
-DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_CUDA_ARCHITECTURES=89
-DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DLLAMA_BUILD_EXAMPLES=OFF
```

工具链：MSVC 14.51（VS 18 BuildTools）+ CUDA 13.3 + Ninja。构建中。

## 3. 单进程基线（已完成，作为对照）

| 方案 | 配置 | pp512 | tg256 |
| --- | --- | ---: | ---: |
| 纯 CPU | `-ngl 0 -t 8` | 91.5 | 8.23 |
| 专家上 GPU，11 层 | `-ncmoe 29` | 263.2 | 30.45（违反 4 倍数，不可用于裁层） |
| **专家上 GPU，8 层** | `-ncmoe 32` | 237.2 | **28.47 ± 0.34** |
| 整层上 GPU，11 层 | `-ngl 11` | 201.2 | 10.11 |

**关键机理**：只把专家放 GPU 比整层放 GPU 快 3 倍。专家占 88% 权重但每 token 只用
8/256，把「体积最大、访问最稀疏」的部分放进最快设备，收益最大。

**显存硬边界**：11 层装得下（约 4.84 GiB 专家），12 层溢出（WDDM 静默用系统内存
当显存，tg 从 30 掉到 16，不报错）。

## 4. 二期设计要点（待 CUDA shim 就绪后验证）

### 4.1 两段的显存占用不同，需实测

整模 `-ncmoe 32` 只把 8 层的**专家**放 GPU（约 3.5 GiB）；
拆分方案的 GPU 段含 8 层**全部张量**（5.03 GiB）。拆分反而更占显存。

这带来一个待验证的问题：**拆分方案的 GPU 段能否装下**（5.03 GiB + KV + compute
在 8 GiB 内），以及吞吐能否保住 28 t/s 量级。

### 4.2 若能装下但更慢，属预期

拆分的价值不在单机提速，而在**能把 CPU 段挪到另一台设备**。单机两段若慢于整模，
应记录为「分布式入场成本」，而非失败。

### 4.3 阶段划分

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| P0 | 单进程基线 `-ncmoe 32` | **完成**（28.47 t/s） |
| P1 | 两段拆分正确性 | **完成**（bit-exact） |
| P1.5 | CUDA 版 shim 构建 | **进行中** |
| P2 | 吞吐验证（目标 ≥28 t/s 量级） | 待 P1.5 |
| P2.5 | 两进程 + socket 传输 | 待 P2 |
| P3 | SLIM-ARC 可选开关 A/B | 待 P2.5 |

## 5. 代码现状

`H:\slimarc-proto\slimarc\`：

| 文件 | 行数 | 职责 |
| --- | ---: | --- |
| `config.py` | 118 | 14 个环境变量，切分点强制 4 的倍数 |
| `protocol.py` | 82 | 6 种消息，长度前缀帧，hidden 走原始 float32 |
| `engine.py` | 88 | keep-head shim 封装（复用 QLH 的 `llama_keep_head`） |
| `runner.py` | 76 | 两段串行、贪心解码 |
| `bench.py` | 52 | 吞吐测量（pp/tg 口径） |
| `check_correct.py` | 79 | hidden 数值对照 |
| `smoke.py` | 74 | 冒烟 |
| **合计** | **569** | 预算 1220，余量充足 |

## 6. 复用而非重写

`engine.py` 直接 import QLH 的 `src/llama_keep_head.py`，不重写 C 绑定。理由：

- 该模块已有 `forward_tokens_to_hidden` / `forward_hidden_to_hidden` /
  `forward_hidden_to_token`，正好对应首段/中段/末段
- 已处理 worker 隔离、KV、多位置、错误码
- 重写等于复制 900 行已验证逻辑

## 7. 纠正一个此前的判断

先前认为「P1 的主要障碍是没有 hidden 注入 API」。实际 `llama.h` 只有导出
（`llama_set_embeddings_layer_inp`），但 **QLH 的 keep-head shim 已实现完整通道**，
且 `mode 0 = nextn` 配 head 裁层工件时**只跑文件内的层**，正是首段需要的。

`mode 1 = layer_inp` 会跑满全部层，只适合配整模做验证——这一点在 `engine.py` 的
注释里已写明。
