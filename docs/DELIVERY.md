# 交付状态与部署

> 状态：**现行**
> 更新日期：2026-11-03
> 适用范围：现在能交付什么、拿到手怎么跑起来；读者为交付与部署执行者。
> 相关：`ARCHITECTURE.md`（机制）、`MEASUREMENTS.md`（数字口径）、
> `ANDROID-SURFACE-段接入评估.md`（弱设备链的拆解）

---

## 1. 能交付什么

### 1.1 已验证的能力

| 能力 | 结果 | 证据 |
| --- | --- | --- |
| 单机两段跨进程 + DFlash | 20.9 tok/s（64 token 口径） | [`SPECULATION.md`](SPECULATION.md) §11 |
| 跨机两段，无投机 | 12.2 tok/s | [`MEASUREMENTS.md`](MEASUREMENTS.md) |
| 两台弱设备各持整模 | decode 6.50 tok/s，prefill 22 tok/s | [`PROGRESS.md`](PROGRESS.md) §1.1 |
| 多轮对话 | 35B 三轮 PASS（本机两段与 **Surface+Y700 弱设备链**各一次）；增量 KV 与从头重放逐位相同（23/23 token） | `checks/verify_multiturn.py` |
| 非贪心采样 | 分段不引入误差（TV=0）；输出连贯 | `checks/probe_logits_delta.py` |
| 常驻与自愈 | kill 掉任一段 worker 后 2 秒重启，链路恢复 | `stage_supervisor.py` |
| 绿色目录 | 96 MB（解压后），清空工作区后独立跑通 | `scripts/pack-portable.py` |

**分段本身不引入误差**（2026-11-03）：同 kernel 布局下整模与链路在同一位置的 logits
逐位相同（`max_abs=0`、`cos=1.0000000`、top-8 全重合、TV=0），2B 测 16 个位置、
35B 测 12 个位置无例外。段间传 float32 hidden 不是「误差落在量化噪声底下」，是没有误差。

### 1.2 没验证的

| 项 | 为什么算未验证 |
| --- | --- |
| 真正干净的 Windows | 只清空了工作区，VC++ 运行库等系统依赖还在；`bin/` 里那 4 个 MSVC dll 就是为此准备的，但没在全新系统上确认过 |
| 四台及以上机器 | 拓扑与 supervisor 支持 N 段，三段（本机 + 两台弱设备）已跑通，但没试过更多 |
| 除 `qwen35` / `qwen35moe` 外的架构 | 见 [`ARCHITECTURE.md`](ARCHITECTURE.md) 的架构适配说明 |

---

## 2. 部署

### 2.1 每台机器放什么

一台机器跑一段。每段都要**同一份整模**（`LLAMA_LAYER_RANGE` 只选层，不建裁层工件）。

```bash
python scripts/pack-portable.py \
    --python  <装好 numpy 的 CPython 3.12> \
    --llama-bin <llama.dll 与 ggml*.dll 所在目录> \
    --shim    <带 QLH_ABI_UPSTREAM_HEADER 的 qlh_keep_head.dll>

cd .qlh_tmp/dist && tar czf qlh-portable.tgz qlh-portable
```

产物解压到目标机器，改 `config.toml` 后跑 `run.bat`。安卓侧用
`scripts/termux-install.sh`，见 [`PACKAGING.md`](PACKAGING.md)。

### 2.2 拓扑怎么写

```toml
source_layers = 40

[[stage]]
model       = "<本机路径>/model.gguf"
host        = "0.0.0.0"        # 本机段写 0.0.0.0，理由见 2.2 的约束 1
port        = 9103
layers      = 24
layer_range = "0:24"
kind        = "gpu"
threads     = 6

[[stage]]
model       = "<那台机器上的路径>/model.gguf"
host        = "100.x.x.x"      # 远端段写真实地址
port        = 9106
layers      = 16
layer_range = "24:40"
kind        = "tail"
threads     = 5
```

三条硬约束：

1. **`host` 有两个含义**：既是连接地址，也是 worker 的 bind 地址。本机段写
   `127.0.0.1` 会让 worker 只绑回环、远端 connection refused；写 `0.0.0.0` 才既绑全接口
   又被 `topology.is_loopback` 认成本机段。
2. **切点与段内层数都是 4 的倍数**：`qwen35moe` 是 hybrid（每 4 层一个 full attention），
   错位会层类型不匹配。`32:38` 这种 6 层划分在解析阶段就被拒绝。
3. **两段的 `model` 必须是同一个文件**。要机器可查就填 `sha256`（可选，代价约 30 秒）。

### 2.3 起服务

```bash
# 单段（调试）
python stage_worker.py --role gpu --model <gguf> --shim <dll> \
    --host 0.0.0.0 --port 9103 --threads 6 --layers 24 --ctx 2048

# 常驻（推荐）：读拓扑起本机段，崩了按退避重启
SA_TOPOLOGY=<topology.toml> python stage_supervisor.py
```

两端都起来后自检：

```bash
python scripts/probe.py --head 127.0.0.1:9103 --tail <另一台>:9106
```

它会校验层数合计、层范围首尾相接、hidden 宽度三项，任一不符直接非零退出——
这三项都是「不报错但输出错误」的那类故障。

---

## 3. 明确不做的

| 想法 | 理由 |
| --- | --- |
| 服务发现（mDNS / 注册中心） | 2–3 台机器，Tailscale 加手填 IP 够用；出问题时 `probe.py` 能立刻分出「连不上」与「连上了没发 HELLO」 |
| Android APK | 要 NDK 重编（Termux 的 libc++ 与 bionic 不是一回事）、python 栈要么塞解释器要么重写、21.87 GiB（23.48 GB）的模型还得让用户 SAF 选一次。一键脚本成本低一个数量级 |
| 单独的健康探测**服务** | 心跳已经内建在 worker 里（`_session` 每圈 touch 一个文件），supervisor 看 mtime 就够；再起一个独立探测进程就是给同一事实造第二份判断 |
| 真并发混批（多序列换吞吐） | 8–12 人天，且一个 worker 一个 context、decode 仍串行。多序列的收益是会话共存与前缀复用，见 [`ISSUE-multi-sequence.md`](ISSUE-multi-sequence.md) |
| 非贪心的 GPU 逐位等价 | head 段 6 层在 GPU 时 TV 0.0014–0.0299；要逐位就把 `gpu_layers` 设 0，代价是 32 层全走 CPU |

---

## 4. 已知边界

- **hybrid 不允许删最后位置**：`seq_rm` 收不回投机多算的位置（实测 `removed=0`，
  随后 decode 报 `pos_max X must be < n_past Y`），只能记账。
- **spec_k>0 时多轮不逐位可复现**：投机回滚的 ~2e-2 跨形状偏差在 100+ token 上下文里
  累积到翻转一个 argmax（答案仍对）。`spec_k=0` 下逐位相同。
- **supervisor 的判据有两道，但不是全知**：`poll()` 看进程退出，心跳看「还在走」
  （`stage_worker` 卡在 engine 调用里时心跳会停）。覆盖不到的是「段还在正常服务、但算错了」
  ——那一类只能靠验收脚本发现。
- **一条长连接会占满一个段的 session**：`stage_worker` 单线程 `accept → _session`，期间新
  连接只在 `listen(4)` 的 backlog 里排队（表现为「端口 LISTENING、连得上、不回 HELLO」）。
  空闲 60 秒的 session 会被主动断开，所以死连接最多堵这么久；但**同一段不要同时挂两个
  客户端**。见 [`ISSUE-tail-worker-hang.md`](ISSUE-tail-worker-hang.md)。
- **端口被占会等 60 秒后放弃该段**：不需要自愈，需要人去看是谁占着；`*.prev.log`
  留着崩溃现场。
- **`n_ctx / n_seq_max` 在两段各算一次**：多序列时两段的 `n_seq_max` 与 `n_ctx`
  必须同值，否则位置边界错开且不报错。

---

## 5. 验收命令

```bash
python -m unittest test_contract test_topology    # 39 项
python checks/verify_multiturn.py                 # 多轮 R1–R5
python checks/probe_logits_delta.py --topology topology-2b.toml --steps 7
```

补丁或 shim 重编后跑等价性对照：`checks/verify_equiv.py`、`checks/verify_cli.py`。
