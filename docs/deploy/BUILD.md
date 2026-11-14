# 构建

> 状态：**现行**
> 更新日期：2026-11-11
> 适用范围：用 `docs/llamacpp-split.patch`（叠加可选的 `llamacpp-slimarc.patch`）在
> Windows/MSVC 上编出可验收的二进制并确认它不是坏的；不覆盖 CI、交叉编译与 Android。

---

## 1. 构建为什么会改变结论

构建开关不是偏好，是结论的一部分。核显实验的基线是 `build-cpu`（通用 CPU 变体，`GGML_CPU_ALL_VARIANTS=OFF`
且 `GGML_NATIVE=OFF`），测试构建却是 `build-vulkan`（`ALL_VARIANTS=ON` + `NATIVE=OFF`）；两者不同级、基线被低估，
于是「核显 prefill 3.1×、decode +5.4%」进了结论。同源基线复测是 **1.90× / −4.8%**
（[IGPU-SLOWDOWN.md](../measure/IGPU-SLOWDOWN.md) §①、§⑥）。下面每个开关都要能复现。

## 2. 从零到可编译

### 2.1 前置

- Windows 10/11 x64，Visual Studio 2022 或 2026 BuildTools（MSVC 14.5x，本机实测 14.51）。
- CMake ≥ 3.20（实测 4.4.2）与 Ninja（用 VS 自带的那份即可，见 §2.3）。
- Git Bash。
- 纯 CPU 构建不需要 CUDA 与 Vulkan SDK；`build-cuda` 要 CUDA Toolkit（实测 13.3），
  `build-vulkan` 要 Vulkan SDK（实测 1.4.363.0）。

### 2.2 取源码并叠补丁

基底是上游 `ggml-org/llama.cpp` 的 `5676dfbe0`（v2.2 基底，fork HEAD `6d0fa16c9` 之前 7 个提交；tag `b12342`）。

本机 fork 的等价取法（不碰 fork 自己的 checkout）：

```bash
git -C H:/llamacpp-c worktree add --detach H:/llamacpp-src 5676dfbe0
```

陌生人从上游取：

```bash
git clone https://github.com/ggml-org/llama.cpp.git llamacpp
cd llamacpp
git checkout 5676dfbe0
```

叠补丁（顺序不能反；实测命令，`--whitespace=nowarn` 只是压掉行尾空白告警）：

```bash
git apply --whitespace=nowarn /path/to/K-Llama/docs/llamacpp-split.patch
git apply --whitespace=nowarn /path/to/K-Llama/docs/llamacpp-slimarc.patch
```

校验叠加结果（overlay 判据）：叠完的树必须等于 fork HEAD 的树 `61781c712`，不等就说明补丁
被改过或顺序错了：

```bash
git add -A -f && git write-tree      # 期望 61781c7126be39f54d1499f83f782f2406612615
```

`scripts/export_patches.sh` 把这条判据脚本化：`5676dfbe0` + 主补丁 = 树 A（`2541c9d29`），
再叠增量 = 树 B（`61781c712`）；B 与 HEAD 树不符时脚本 exit 1 且不写任何文件。
只叠主补丁会得到树 A（不含 SLIM-ARC）；本文的编译验证只对两个补丁都叠的树做过。

### 2.3 配置与编译

Ninja 构建必须在 `vcvars64` 环境里跑；在 Git Bash 里直接跑 `ninja` 会因为缺 `INCLUDE`/`LIB`
报 `C1083: 无法打开包括文件: "stdbool.h"`。把下面存成源码根的 `build-qlh.bat`：

```bat
@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "NINJA=C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe"
cmake -S . -B build-split -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_NATIVE=ON -DGGML_CPU_ALL_VARIANTS=OFF ^
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_RPC=OFF ^
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-split --target split-stage llama-server llama-cli llama-bench ^
  split-smoke split-rows dflash-taps -- -j 8
```

执行：`cmd //c build-qlh.bat`。本机实测 configure 16 秒；从零编 `split-stage` + `llama-server`
1 分 52 秒（187 步），再编其余 5 个 target 10 秒。产物在 `build-split/bin/`。

这组开关与开发机 `build-split` 的 `CMakeCache.txt` 逐项一致，是 v2.1 纯 CPU 验收构建。
`-DCMAKE_MAKE_PROGRAM` 可以省略让 CMake 自己找，但本机 VS 的 ninja 不在 `PATH` 上。

### 2.4 编完确认不是坏的

```bash
cd build-split/bin
./llama-cli.exe --version      # version: 0.9.0-dev (build N, commit 6d0fa16c9)
./split-stage.exe              # 打印 usage，退出码 0
```

两条都过 = 动态库齐全、链接自洽。`llama-server.exe --version` 同理。

**只拷 exe 不算编好**：`llama-server.exe` / `llama-cli.exe` / `llama-bench.exe` 只有 10 KB，
实体在 `llama-server-impl.dll` 等；必须连 `llama.dll`、`llama-common.dll`、`ggml*.dll`、
`mtmd.dll` 一起拷，缺一个都会在启动时报「找不到指定的模块」。

## 3. 开关矩阵

本机实际是 **8 个 `build-*` 目录 + 1 个 dll 池目录 `_bin_cpuonly`**。数值逐目录读
`CMakeCache.txt` 得到（2026-11-10）：

| 目录 | ALL_VARIANTS | NATIVE | CUDA | VULKAN | RPC | BACKEND_DL | OPENMP | TESTS | 用途 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `build-split` | OFF | ON | OFF | OFF | OFF | OFF | ON | ON | v2.1 CPU 验收（52 项；组 12 需 CUDA 构建，合计 53；组 13 需 KVMem 构建 `build-kvmem2`，合计 57；组 14 只需 2B MTP 权重，合计 77） |
| `build-cuda` | OFF | ON | ON | OFF | OFF | OFF | **OFF** | OFF | CUDA 实验；`CMAKE_CUDA_ARCHITECTURES=89` |
| `build-vulkan` | ON | OFF | OFF | ON | OFF | **ON** | ON | ON | 核显实验：多 CPU 变体 + 动态后端 |
| `build-vulkan-new` | ON | OFF | OFF | ON | OFF | ON | ON | ON | 与 `build-vulkan` 开关完全相同（cache 逐行 diff 只差目录名），重建副本 |
| `build-cpu` | OFF | OFF | OFF | OFF | OFF | OFF | OFF | OFF | 通用 x64 变体；**不是公平基线**（§4） |
| `build-spec` | OFF | ON | OFF | OFF | OFF | OFF | ON | OFF | 与 `build-split` 只差 TESTS；用途无书面出处 |
| `build-layer-inp` | OFF | ON | OFF | OFF | OFF | OFF | ON | ON | 与 `build-split` 开关相同；用途无书面出处 |
| `build-rpc` | — | — | OFF | — | ON | — | — | — | **配置失败**：cache 仅 112 行、`CMAKE_MAKE_PROGRAM-NOTFOUND`、无 `build.ninja`；无存在理由 |
| `_bin_cpuonly` | 拷自 `build-vulkan-new` | | | | | | | | 公平基线：从 `build-vulkan-new/bin` 拷出的 CPU-only 子集（20 个文件、无 `ggml-vulkan.dll`、保留 9 个 `ggml-cpu-*.dll` 与 bench/cli/server） |

除表内这些列外，其余一切（`GGML_AVX*` 等）由 `GGML_NATIVE` 决定，不是独立开关：
`NATIVE=ON` 时 cache 里的 `GGML_AVX/AVX2=OFF` 只是「没单独指定」，实际编译按本机 ISA
（本机 configure 日志：`Adding CPU backend variant ggml-cpu: /arch:AVX2`）。

## 4. CPU 侧：ALL_VARIANTS 与 NATIVE

| 组合 | 产物 | 运行时行为 | 可移植性 |
| --- | --- | --- | --- |
| `ALL_VARIANTS=ON` | 一组 `ggml-cpu-<arch>.dll`（haswell / skylakex / alderlake / …） | 启动时按 CPU 选一份加载 | 跨 x86-64 机器可跑 |
| `NATIVE=ON` | 单份 `ggml-cpu.dll`，按本机 ISA 编（本机 `/arch:AVX2`） | 直接加载 | 搬到不支持该 ISA 的机器会在启动后 **SIGILL** |
| 两者 OFF | 单份通用 `ggml-cpu.dll` | 直接加载 | 最广，最慢 |

实测：`build-vulkan-new` 移走 `ggml-vulkan.dll` 后按 haswell 变体跑，pp32 45.18 / pp256 42.70；
`build-cpu` 通用变体 30.40 / 33.53，慢 **33% / 21%**（[IGPU-SLOWDOWN.md](../measure/IGPU-SLOWDOWN.md) §⑥）。

做任何 CPU 侧 A/B 之前先过三条判据：

1. **同源**：基线从测试构建的 bin 整份复制、只移走 GPU 后端 dll（`ggml-vulkan.dll` 或
   `ggml-cuda.dll`），其余文件一个不动；或者两份构建用同一组 CPU 开关。不要用另一个
   `build-*` 目录的产物当基线。
2. **看实际加载的变体**：启动日志里 `load_backend: loaded CPU backend from ...ggml-cpu-haswell.dll`
   （[FIXES.md](../issues/FIXES.md) 记录了这条日志）。两侧这一行必须指向同一类变体。
3. **核对 dll 清单**：两边 bin 里 `ggml-cpu*` 的文件名集合必须一致。`build-cpu` 只有
   `ggml-cpu.dll` 一个，`build-vulkan` 有 9 个 `ggml-cpu-*.dll`——这本身就是「不同级」的证据。

### 4.1 `NATIVE` 与 `BACKEND_DL` 互斥，「多变体 + 原生优化」不存在

`ggml-cpu/CMakeLists.txt:415` 在 `NATIVE=ON` 且 `BACKEND_DL=ON` 时直接
`FATAL_ERROR`（"GGML_NATIVE is not compatible with GGML_BACKEND_DL, consider using
GGML_CPU_ALL_VARIANTS"），而 `ALL_VARIANTS` 又要求 `BACKEND_DL=ON`
（`ggml/src/CMakeLists.txt:491`）。所以只能这么组合：

| 想开 | 必须同时 | 代价 |
| --- | --- | --- |
| `NATIVE=ON` | `BACKEND_DL=OFF` | 二进制不可移植（换 ISA 就 SIGILL） |
| `ALL_VARIANTS=ON` | `BACKEND_DL=ON` | `NATIVE` 被挤掉，走固定特性变体 |

**`NATIVE` 被挤掉不是无害的。** 2026-11-14 实测：`build-vulkan-new` 在
`NATIVE=OFF + ALL_VARIANTS=ON` 下，组 11（DFlash 整模 vs 远程链）**整模 137 字符 /
链 114 字符**，判据「逐字节相同」红；改成 `NATIVE=ON + BACKEND_DL=OFF` 后
**137 / 137 一致**，组 11 恢复 8/0/0。变的不是整模（一直是 137），是链——多变体构建让
「head 32 层 + stage 8 层」的双进程路径与单进程整模走了不同的向量化路径，浮点累加顺序
不同，T=0 下 argmax 在临界 token 上翻转。

**推论（2026-11-14 用户确认）**：**数值一致性在异构场景下本来就会被破坏**——不同设备硬件不同、
数据流完全不一样，这是异构集群的本质，不是缺陷。用户判断这是**值得的取舍**，大多数情况下不会
引起质量退化。**所以测试与分发各自维护一套完整构建**，每套都含纯 CPU + Vulkan + 可选 CUDA：

| 用途 | CPU 侧 | 理由 |
| --- | --- | --- |
| **测试** | `NATIVE=ON` + `BACKEND_DL=OFF` | 分段/多设备路径在同一 ISA 下可比，「逐字节相同」在这里才有效 |
| **分发** | `ALL_VARIANTS=ON` + `BACKEND_DL=ON` | 一份二进制跨 x86-64 机器，Surface（Gen11）/ Y700 都能跑 |

**「逐字节相同」只在「同一构建 + 同一 ISA」下是有效判据**；跨设备、跨构建、跨后端都不该用它，
那时的正确性判据是「质量不退化」（token 序列一致或容差），不是位级相等。这条不是妥协——
异构集群里两条数据流本来就不同，要求位级相等等于要求它们走同一条路径。

v2.1 从没暴露这个问题，因为它不碰 Vulkan、没必要开 `BACKEND_DL`，等于一直待在
`NATIVE=ON` 的配置上。

## 5. Windows / MSVC 的构建边界

### 5.1 改过 `.h` 一律 `--clean-first`

后果：增量构建把新旧目标文件混链进同一个 dll，ABI 不一致；症状是随机崩溃或数值错误，
不是编译报错。

证据：
- `ninja -C build-split -t deps` 共 282 个 obj，其中 14 个 `#deps 0`；`build-vulkan` 838 个里
  **528 个**为 0，11 个 unity 单元全为 0。`#deps 0` 意味着 ninja 认为该对象不依赖任何头文件。
- 改 `include/llama.h` 后增量重编 43 个 obj——因为它被 `src/CMakeLists.txt` 列为 `llama` 的
  显式源，这条依赖可靠；不在源列表里的内部头没有这个保护。
- unity 批（`src/CMakeLists.txt:74-75`，`UNITY_BUILD_BATCH_SIZE 16`）里的 `.cpp` 改动不会唤醒
  同批对象；`.qlh_tmp/build_llama.ps1` 记录过 qwen35moe.cpp 必须手动删 `unity_*.obj`。
- `build-split/verify.sh` 头部写明：「build-split 的 ninja 依赖记录不可靠……改过 `.h` 之后一律 `--clean`」。

可操作判据（改头之前、或怀疑构建目录被增量污染时）：

```bash
ninja -C build-split -t deps | awk '/#deps /{if($3+0==0)c++} END{print c}'   # 非 0 → 不可信
cmake --build build-split --clean-first -- -j 8                               # 全量重编
```

全量重编约 2 分钟，比一次随机崩溃便宜。

### 5.2 `build-cuda --clean-first` 只清 `build-cuda/bin`

`build-cuda --clean-first` 只清 `build-cuda/bin`（12 个文件）。依据：

- `CMakeLists.txt:40-41` 把运行时/库/归档输出全绑到 `${CMAKE_BINARY_DIR}/bin`，每个构建目录
  各有自己的 `bin`，没有任何共享输出目录。
- `build-cuda/build.ninja` 里 grep `build-split` 零命中；`clean` 规则只引用 build-cuda 记录的产物。
- 两个 bin 之间没有 junction，也没有硬链接（`fsutil hardlink list` 只列出自身）；文件集也不同
  （`build-split/bin` 110 个、`build-cuda/bin` 12 个，后者有 `ggml-cuda.dll`、前者没有）。
- 两者唯一共享的是源码树 `H:/llamacpp-c`，`clean` 不碰源码。

真正该守的纪律是：不要把不同构建的 bin 互相拷 dll，也不要拿 `build-cuda/bin` 的 dll 补
`build-split/bin`——那才会造成 ABI 混用。`_bin_cpuonly` 那种「整份复制再删一个 dll」是安全的，
因为来源是同一构建。

### 5.3 `.qlh_tmp\vrambuild.bat` 默认增量

187 字节：`cmake --build %1 --target %2 %3 %4 -- -j 8`。前两个位置参数是构建目录和 target，
后两个透传。默认**不带** `--clean-first`，所以 `vrambuild.bat build-split all` 是增量构建，
改过 `.h` 之后不能用；必须显式写 `vrambuild.bat build-split all --clean-first`
（[PATCHES.md](PATCHES.md) 的纯分段独立验收就是这么调的）。

## 6. 产物清单

| 产物（`bin/`） | CMake target | 源码位置 | 用途 |
| --- | --- | --- | --- |
| `split-stage.exe` | `split-stage` | `tools/split-stage/`（`tools/CMakeLists.txt:32`） | 链上的 stage 服务；验收核心 |
| `llama-server.exe` + `llama-server-impl.dll` | `llama-server` | `tools/server/` | head / 端到端；exe 是 10 KB 壳 |
| `llama-cli.exe` + `llama-cli-impl.dll` | `llama-cli` | `tools/cli/` | 手动推理、MTP 验证 |
| `llama-bench.exe` + `llama-bench-impl.dll` | `llama-bench` | `tools/llama-bench/` | pp / tg 吞吐 |
| `split-smoke.exe` | `split-smoke` | `tests/CMakeLists.txt:431` | 整模 vs 0:N + N:M 逐位（手动） |
| `split-rows.exe` | `split-rows` | `tests/CMakeLists.txt:437` | 投机前置条件（手动） |
| `dflash-taps.exe` | `dflash-taps` | `tests/CMakeLists.txt:441` | DFlash 8 个 tap 逐位（手动） |
| `llama.exe` | `llama-app` | 顶层 `app/`（根 `CMakeLists.txt:260`） | 多前端入口（把 server/cli/bench 收进一个 exe） |

- `tests/` 下三个是**手动目标**：不注册 ctest，需要真实 GGUF（`argv[1]` 或 `QLH_SPLIT_MODEL`），
  没有模型时打印一行并以 0 退出。
- `build-split/bin` 里的 `argmax_bench`、`bvs_mini`、`chain_probe`、`seg_probe`、`seq_rm_split`、
  `batch_vs_single` 源码在 `H:\llamacpp-c\.qlh_tmp\*.cpp`，**不在 CMake 图里**，是手工编译遗留；
  用 cmake 重建不出来，不要依赖它们。
- `verify.sh` 里出现的 `ollama.exe` / `app.exe` 是机器上别人的进程名（防误杀），不是本仓库产物。

## 7. 验收

验收脚本是 `bash verify/verify.sh`（十五组共 95 项），用 `QLH_BUILD_DIR` 指向构建目录：默认增量
编译后全跑，`--clean` 全量重编，`--skip-build` 只跑验证。组 9/10 需要 35B 与 DFlash 权重及显存/内存
闸门；组 12 需要 CUDA 构建（`QLH_CUDA_BUILD_DIR`），组 13 需要 KVMem 构建（`QLH_KVMEM_BUILD_DIR`，
默认 `H:/llamacpp-c/build-kvmem2`）与 KVMem 源码根（`LLAMA_KVMEM_ROOT`），组 14 需要 2B 的 MTP
权重与 `split-stage.exe`，组 15（判据 2–6 的多序列基础语义）只需要 2B 与 `split-stage.exe`，
缺任一条该组记 SKIP。脚本随仓库分发（`verify/README.md`），新编一个目录也能跑。

项数与端口：`EXPECT_PASS=95` = 组 1–8 的 22 + 组 9/10/11 的 30 + 组 12 的 1 + 组 13 的 4 +
组 14 的 20 + 组 15 的 18；每组整组 SKIP 时按 `SKIP_ITEMS_*` 从期望里扣掉再断言。
端口基准：组 14 用 `QLH_VERIFY_PORT_PESPEC_BASE`（默认 9380），组 15 用
`QLH_VERIFY_PORT_MULTISEQ_BASE`（默认 9370），其余组用 `QLH_VERIFY_PORT_*`。

## 8. 这份说明还缺什么

- `build-spec`、`build-layer-inp` 的用途没有书面出处，开关又与 `build-split` 相同，疑似实验残留；
  `build-rpc` 是配置未完成的坏目录。三者都可以删。
- CUDA / Vulkan 的配置只在本机的 VS 路径、CUDA 13.3、Vulkan SDK 1.4.363.0 下成立，没有写成
  可移植命令（`build-vulkan.bat` 是现成入口，但硬编码了 SDK 路径）。
- 没有模型/权重的获取与放置说明；`split-smoke`、`split-rows`、`dflash-taps`、组 9/10 都需要真实
  GGUF 才能跑。
