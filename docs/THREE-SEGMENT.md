# 三段链路：本机 + Surface + Y700

> 状态：**已验证**（文中引用的 v2 工具链——`chain_client` / `stage_worker` / `shim/` / `koakuma`——已退场，归档在 `slimarc/prototype`，命令与路径只作历史口径）
> 更新日期：2026-11-03
> 适用范围：三台设备各持整模、按层段串成一条链的部署与实测；读者为运维与后续维护者。
> 相关：`ANDROID-SURFACE-段接入评估.md`（两段基线）、`MEASUREMENT-PITFALLS.md`

把 41 层切成 **24 / 8 / 8** 三段，分别落在台式机、Surface Pro 7、Y700 上，跨公网 Tailscale 组链。本文记录切点怎么定、实测多少、以及为什么「每台 8 层」救不了尾段——瓶颈不在层数，而在一次跨网往返加 `lm_head`。

---

## 1. 结论

同一份 35B 模型、greedy、`n_ctx=1024`、`gen 24`，三个拓扑：

| 拓扑 | 段划分 | 每 token | 吞吐 | 备注 |
| --- | --- | ---: | ---: | --- |
| 两段（局域网基线） | Surface `0:24` + Y700 `24:40` | 153.4 ms | 6.50 t/s | 网络 17.5 ms；口径见 `ANDROID-SURFACE-段接入评估.md` §6 |
| **三段（本文）** | 本机 `0:24` + Surface `24:32` + Y700 `32:40` | **201.7 ms** | **5.00 t/s** | Tailscale，5 GHz 校园网，6 次运行稳态中位 |
| 三段（2.4 GHz 局域网） | 同上 | 290 ms | 3.4 t/s | 同一台 2.4G AP，三机同 BSSID |

本机另做 2 次对照（`--repeat 2`，均值含首步冷启动）为 228.0 / 241.4 ms——差在冷启动那几步，不是拓扑差异（§7.2）。

三次都是 41 层全切、模型完整，**三段没有增加容量，只改变每台的内存压力**（层权重，不含 `token_embd` / `output` / MTP，口径见 §2 累积表）：

- Surface：24 层 **11.9 GiB** → 8 层 **3.6 GiB**。只有它真的贴墙——可用 8–9.5 GiB，24 层段连 `token_embd` 要 12.3 GiB，靠 mmap 选择性加载才跑得起来。
- Y700：16 层 **8.3 GiB** → 8 层 **4.8 GiB**（层权重）。实测进程 RSS 要 **8.4–10.3 GB**：加上 `output`、MTP 与计算缓冲，8 层在这台 15.6 GB 的机器上余量很小。
- 本机：两段时不参与；三段里接走 24 层（11.9 GiB），而本机空闲 22.5 GiB，余量最松。

代价是 **+31% 延迟**（153.4 → 201.7 ms），换来的是两台弱设备都从紧余量里退出来。 **只求快则两段更好；要长期挂着跑、不撞内存墙，三段更好。**

三段的两条结论值得单独记住：

1. **远端段的耗时主要是网络，不是层数。** 2.4 GHz → 5 GHz 时本机段 74 → 73.4 不动，
   Surface 100 → 64.6、Y700 100 → 55.7——同批二进制、同层数，唯一变量是网络。
2. **层数减半救不了尾段。** 尾段每步要读 `output` 的 397.9 MiB 权重，再付一次跨网
   往返，这两项盖过那 8 层本身的差别（§4）；切点 36:40 只快 4% 是同一现象。

---

## 2. 切点：按字节分，不按层数分

切点必须同时满足两个约束：

- `LLAMA_LAYER_RANGE=lo:hi` 把源 `blk.lo+i` 映射成本地 `blk.i`；层类型由**本地层号**模
  `full_attention_interval=4` 现算，所以切点必须与 4 对齐（机制与解除代价见
  `ARCHITECTURE.md` §6）。这条是 `topology.py` 在解析阶段的检查，不是 llama.cpp 的
- 各段权重之和要落在各自的可用内存里

**这个模型不是均匀量化**，「平均每层多大」的估算不成立。逐张量清点（`scripts/inspect_gguf_quant.py`）：

| 量化 | 字节 | 占比 |
| --- | ---: | ---: |
| IQ4_XS | 8.367 GiB | 38.3% |
| Q6_K | 7.294 | 33.4% |
| Q5_K | 5.156 | 23.6% |
| Q8_0 | 0.960 | 4.4% |
| F32 | 0.083 | 0.4% |

每层的真实重量：

| 层 | 量化 | 每层 |
| --- | --- | ---: |
| `blk.0–4` | Q6_K | 0.646 GiB |
| `blk.5–9` | Q5_K | 0.547 |
| `blk.10–30` | IQ4_XS | **0.429** |
| `blk.31–35` | Q5_K | 0.547 |
| `blk.36–39` | Q6_K | 0.646 |
| `blk.40` | Q8_0 | **0.836**（MTP 层） |

非层张量 0.777 GiB（`token_embd` 0.3885 + `output` 0.3885），跟着头段和尾段各背一份。文件总长 23,484,576,128 B = 21.87 GiB，三台机器上的副本逐字节相同。

累积到第 k 层的字节（GiB）：

| k | 8 | 16 | 24 | 32 | 40 | 41 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0:k` | 4.86 | 8.52 | 11.94 | 15.49 | 20.25 | 21.08 |

**`blk.40` 是 MTP 层（Q8_0，0.836 GiB），必须留在尾段**，尾段还独占 `output`。按层数均分（14/14/13）会让本机背到 17 GiB、Y700 背到 14 GiB，两台弱设备都爆。按上表切 24/8/8 是唯一同时满足"4 的倍数"和"都留 2 GiB 以上余量"的方案。

---

## 3. 拓扑与接线约束

| 段 | 层 | 监听 | 后端 |
| --- | --- | --- | --- |
| stage0（本机） | `0:24` | `127.0.0.1:9103` | gpu（Vulkan 加载但 `gpu_layers=0`），`-t 12` |
| stage1（Surface） | `24:32` | `0.0.0.0:9104` | cpu，`-t 4` |
| stage2（Y700） | `32:40` | `0.0.0.0:9106` | tail，`-t 5` |

三条接线约束，错了不会报错，只会算错：

1. **`n_ctx` 必须三台一致。** `llama-context.cpp:391-405` 里 seq 边界由各自的
   `n_ctx` 决定，不一致时位置边界会静默错位，输出仍然"像正常"。全部 `SA_CTX=1024`；
   `client` 侧也要传，因为 `config.DEFAULT_CTX = 4096`。
2. **shim 必须和 backend DLL 同目录。** `engine.py:61` 用 shim 所在目录覆盖
   `SA_BACKEND_DIR`，外层环境变量会被无视。正确做法是直接用
   `H:/llamacpp-c/build-vulkan/bin/qlh_keep_head.dll`（它旁边就是 `llama.dll` 和
   `ggml*.dll`），`.qlh_tmp/shimout/` 里那份只是构建产物。
3. **`0.0.0.0` 必须算本地。** `topology.py` 的 `LOCAL_HOSTS` 不含它时，worker 会把
   自己判成"本机没有我的段"然后退出，日志是一行 `has no stage on this machine`。

拓扑文件带设备地址，被 `.gitignore` 的 `topology-*.toml` 排除，不入库。

---

## 4. 段计时含往返，不能当纯计算

**每段耗时不能按层数摊。** `chain_client.py:1019/1039` 的 `t0`/`t1` 打在调用前后， `stage_ms[i]` 含客户端到该段的**完整 TCP 往返**：

    stage_ms[i] = RTT(客户端↔段 i) + 段内计算 + 子进程 IPC/唤醒 + (仅尾段) lm_head

而且三段是**星形**：客户端在本机，三个 client 各自连 worker，hidden 每段都回本机再发下一段——不是段间转发，是每段各付一次跨网往返。

能定的两个量：

- **网络变、本机段不动**：2.4 GHz → 5 GHz，stage0（回环）74 → 73.4，stage1
  100 → 64.6，stage2 100 → 55.7。同批二进制、同层数，唯一变量是网络。
- **尾段多一个 `lm_head`**：`LOGITS_ROLES={"tail"}`（`config.py:115`），`output` 是
  248320×2048 的 q6_K（397.9 MiB），每步必读。M=1 实测 **12.0 ms**
  （`slimarc/prototype:checks/bench_lm_head.py`（v2，已退场），本机 6 线程）；`config.py:110-114` 的两处调用点对这个
  量各记过 6.4 与 3.6 ms，测量口径不稳，Y700 上未测。

**不同拓扑的段耗时不具可比性**：两段基线的客户端在 Y700（tail 走本地回环），三段在本机（每段跨网一次），45.5 与 55.7 不是同一口径。按同一口径拆，今天的 Y700 段是 8 层约 23 ms ＋ `lm_head` 10–20 ＋ 往返 25–30。**层数减半救不了尾段，大头不是层。** 这与切点 36:40 只快 4%（`ANDROID-SURFACE-段接入评估.md` §5）是同一现象。

**48 ms 的差距无法归因**：两段基线的客户端在 Y700、三段在本机，位置不同；缺同位置对照，任何分解都是拼凑。§9 给出可把它劈开的实验。

---

## 5. 2.4 GHz 局域网输给公网 NAT

同一套三段，在 2.4 GHz 局域网里是 290 ms / 3.4 t/s，比跨公网 Tailscale 慢 44%。当时的每段耗时是 74 / 100 / 100 ms，两台弱设备都比现在多 35–45 ms。

原因不是带宽，是**同一个 AP 上的自干扰**：三台设备都在 `Siegfried Kkm.` （BSSID `ac:45:b0:34:1e:1a`，channel 6，24 MHz 宽），而 2432–2472 MHz 上有 20 多个邻居 AP。三台同 AP 的机器之间虽然是 hairpin、不过空口两次，信道忙的时候仍然要排队。

5 GHz 那边完全相反：三台切回 `<campus-ssid>` 后落在同一个 BSSID （`<bssid>`），Y700 侧 `Frequency: 5745`、ch149、802.11ax 516 Mbps、 RSSI −31…−42 dBm，本机↔Surface 的 hairpin 只有 **9 ms**。

但 `<campus-ssid>` **开了客户端隔离**：`10.61.x.x` 之间 ARP 直接 FAILED，局域网 IP 互相 ping 不通。所以这套拓扑上只能用 Tailscale 地址，不能用 LAN 地址—— `topology-three-local.toml` 里那版 LAN IP 在这张网上是死的。

**一个反直觉的结论**：这所校园网的 NAT 是 `MappingVariesByDestIP: false`（好 NAT， UDP 可打洞），最近 DERP 在东京 59.4 ms。打洞成功后 point-to-point 走 AP hairpin， **比同一张 2.4 GHz 网里的直连更快**。局域网不等于快，链路质量才是。

---

## 6. Tailscale 打洞与保活

判读规则（这几个坑踩过）：

| 命令 | 看什么 | 含义 |
| --- | --- | --- |
| `tailscale status` | `CurAddr` 非空 | 直连 |
| `tailscale status` | `CurAddr` 空 + `Relay: tok` | 空闲态，**不是**降级 |
| `tailscale ping <ip>` | 第一行就是 `via DERP` 还是 `direct` | 只打一行就停，属于设计 |
| `tailscale netcheck` | `MappingVariesByDestIP` | `false` 才好打洞 |

`tailscale ping --c N` 的 `--c` 是上限，**一旦建立直连就立刻停**，所以只回一行不是失败。真正的降级要看 `netcheck` 的 UDP 可达性和 `Nearest DERP` 的延迟。

保活实测（180 秒连续流量，每 5 秒采样）：**36/36 全部直连，从未回落到 DERP**。静默 60 秒后 `CurAddr` 变空（空闲态），下一个包仍走直连——链式推理每步本身就是打洞流量，不需要额外心跳。

Y700 侧两个状态要盯：

- 偶尔落到 `DERP(hel)`（赫尔辛基），延迟不可用。切 5 GHz 并制造流量后恢复直连。
- `adb shell ping` 对 Tailscale 地址**永远 100% 丢包**，因为 shell 进程不在
  VpnService 的 per-app 路由里，它走的是 wlan0。这不是链路故障，见 §7。

---

## 7. 两个测量坑

### 7.1 `adb shell ping` 的假象

Y700 上的 Tailscale 是 VpnService。`adb shell` 起的进程没有继承那个路由， `ping 100.x` 必然 100% 丢包，即使整条链这一步刚跑完 200 ms/token。 **判链路只能从主机侧看**：`tailscale ping`、`adb logcat | grep magicsock` （出现 `now using ... via=direct`）、以及 stage2 的实际墙钟。

### 7.2 段计时相加 ≠ 整链延迟

`forward_hidden` 是同步请求-响应，但计算是异步提交的。两段之间互相夹住的时候， "等上一轮自己结果"的时间会记到下一步的下一段头上。所以**各段之和大于整链时间是正常的**，流水线本来就在重叠（详见 `MEASUREMENT-PITFALLS.md` §2）。判断优化是否有效只看整链吞吐。

本次三段的段和 73.4 + 64.6 + 55.7 = 193.7 ms，整链中位 201.7 ms，差 8 ms 量级正常——这个拓扑上段与段严格串行（每段都等上一段的 hidden），没有重叠空间。

---

## 8. 复现

```bash
# 三台各自起 worker（端口见 §3），等它们都 LISTENING
# 本机
SA_CTX=1024 python -m worker.stage_supervisor -t topology-three-ts.toml

# 链式推理
SA_CTX=1024 python client/chain_client.py --topology topology-three-ts.toml \
    --greedy --n-predict 24 --repeat 2
```

必须带 `--greedy`（或 `--spec-k 0`）：多轮 `spec_k>0` 可用但不是逐位可复现（`first_diff` 在 9–11），验证路径只暴露 argmax。要确定性就 `spec_k=0`。

---

## 9. 未解决

1. **长跑可用，但有个已知的可用性缺口**（2026-11-04）。三段 `0:32 / 32:36 / 36:40`
   （弱段各 4 层）在**不加任何诊断代码**的情况下连跑 **60 轮 `gen 256` 零异常**；`gen 900`
   （932/1024 位置）也干净。**当前实现具备长跑能力**——早先"不具备"的结论是错的，它来自
   调查期挂的 `faulthandler` 周期 dump：那东西在 Windows 上跨线程遍历栈会自己触发
   `0xc0000005`（有 faulthandler 77 轮 2 次 AV，无诊断 60 轮 0 次）。
   真正的缺口是另一件事：worker 是单线程 `accept → _session`，**一条长连接占满 session 时
   新连接只在 backlog 排队、拿不到 HELLO**。进程正常、端口 LISTENING，但整段对新人不可用。
   见 `ISSUE-tail-worker-hang.md`。
2. **本机↔Y700 的数据面 RTT 没测准**：`tailscale ping` 报 124 ms，UDP 探测 25–30 ms，差 4 倍。
   要的是 `ss -ti` 的 `rtt` 字段。
3. **`lm_head` 在 Y700 上没测过**：12.3 ms 是本机 6 线程的数（§4），Termux 上只会更慢。
4. **48 ms 的差距归不了因**：两段基线的客户端在 Y700、三段在本机，客户端位置不同。
