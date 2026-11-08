# 分段链实测数据

> 状态：**现行**
> 更新日期：2026-11-08
> 适用范围：`H:\llamacpp-c`（HEAD `c5bd787b`）的 `build-split`，2B 与 35B 两条分段链上这一周的
> 全部数字。读者是调参、写文档、以及下一次要判断「这个数还成不成立」的人。
> 相关：[`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md)（这些边界）、[`SPECULATION.md`](SPECULATION.md)
> （投机判定）、[`PATCHES.md`](PATCHES.md)（第四代实现）。

这份文档存在的理由：本周的性能数字一度只存在于某次对话里，`SEGMENTED-LIMITS.md` 引用了它们
却没有出处。下面每一条都注明日志文件与时间戳；**日志里找不到的，一律进 §7 标成未复现，不在这里
编一个。**

**结论先行**：

- checkpoint 覆盖全链之后，3103 token 的历史第二轮只算 38 个新 token：**714 ms**，而整段重算是
  **19847 ms**。
- tail 的 logits 缓冲从 **1710.76 MiB** 降到 **3.79 MiB**，同一 ubatch 的 prefill 顺带从
  172.6 涨到 219.4 t/s。
- MTP 的收益完全由接受率决定：本仓 2B 上只有退化 prompt 能到 1.3×，真实代码/中文 prompt 上
  **k=2 反而慢 13%–37%**（§2）。
- DFlash k=3 在 35B 链上是唯一稳定的正收益：**1.12×–1.24×**（§3）；k=15 亏（0.46×）。
- 远程链的投机 2026-11-08 起是终态（协议 v7 验证通道 + v8 远程 MTP draft + grammar/rbudget
  兼容）：ROG 五类 prompt 上 **MTP k=2 中位 +6%（+8.4% ~ −1.2%）**，Surface 上 **−40%**，
  单帧回复从 **993,312 B 降到 48 B**（§9）。判据不是「与不投机逐字节相同」（§9.4）。
- `seq_rm` 的「删尾部」在上游 hybrid 的默认配置（`n_rs_seq = 0`）下**永远失败**，而且 attention
  那半一行都不删（上游 open PR #28007 / issue #31656）。本仓把它从静默改成可见，五个调用点
  按场景降级；两种形态的精度实测见 §10（FULL 两红、RS 全绿）。
- **2026-11-08 换到官方 `UD-Q4_K_M`**：形状全部对得上、**零改代码**；整模 MTP k=2 的接受率从
  蒸馏版的 0.699 回到 **0.910**（mean len 2.39 → 2.81），k 甜点仍是 2。而**本地双段链上只有
  0.688**，比整模低 24%——根因是本仓新加的 `!is_chain` 跳 catch-up + 草稿 KV 每轮清空，这是
  自洽的设计选择，不是 bug（§11）。

---

## 0. 口径

| 项 | 口径 |
| --- | --- |
| 硬件 | 本机 i9-13900H + RTX 4060 Laptop；`build-split` 是**纯 CPU** 构建，启动即打印 `warning: no usable GPU found`，全部数字 `-ngl 0` |
| 2B 链 | `qwen35-2b-Q4_K_M`，head `0:20` + tail `20:24`（24 层），本地双段（进程内 tail） |
| 35B 链 | head `0:32` + tail `32:40`（qwen35moe，40 层 trunk + 1 个 NextN 层）。**2026-11-08 起是官方的 `Qwen3.6-35B-A3B-UD-Q4_K_M`**（`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`，21.11 GiB，§11）。**§3 与 §9 的 35B 数字是在旧的蒸馏版上测的**（`Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-I-Quality`，21.87 GiB），两份权重不能混用 |
| 采样 | 贪心（`temperature 0`、`top_k 1`） |
| `-np` | 除注明外都是 1 |
| 逐位对比 | 只在**单 ubatch 内**成立；跨 ubatch 的差异是浮点归约序，见 [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §3.1 |
| 日志根 | `H:\llamacpp-c\build-split\` 与 `H:\llamacpp-c\.qlh_tmp\`（下表只写相对路径） |

### 0.1 最要紧的一条口径：**本机双进程测不出投机收益**

§2/§3 所有投机数字都是「head 与 tail 跑在同一台机器上」。这个配置对投机是**退化**的：

- **它拿不到投机唯一该拿的好处**。投机减少的是**每轮往返次数**（一轮前进 3.2 token
  而不是 1），而在本机双进程里 head↔tail 走内存与管道，**没有 25–30 ms 那个跨机往返**。
  付出 draft 成本、却省下一个不存在的量。
- **它放大了投机的代价**。两个进程抢同一套 CPU 核心、L3 与内存带宽，而 draft（MTP 的
  nextn 层 / DFlash 的 6 层 dense）是纯额外算力，全压在已经很挤的 tail 那台「机器」上。

所以本机测出的「亏」是**对一个不会部署的配置**的测量，但它**不能反推出「多机一定赚」**。

**跨机已经测了（2026-11-08，协议 v7/v8）**：ROG（head `0:32` + tail `32:40`，LAN）五类真实
prompt 上 MTP k=2 相对不投机是 +3.6% / +8.4% / −1.2% / +1.5% / +10.5%，**中位约 +6%**，§9 有
全表。所以本节要补一句：**「本机测不出来」只说明本机的负数不能外推，跨机的正确答案是「小
幅正收益，随负载翻号」**——弱机器（Surface，−40%）与带 grammar 的回环（0.67×）都是负的。
§2/§3 那些本机数字说明的「k 与源要按负载和拓扑选」依然成立。

反过来，checkpoint 复用（§1）、并发（§4）、内存（§5）这些不受这条影响：它们的收益不来自
跨机往返。

---

## 1. checkpoint 复用（远程链）

远程链 = 本机 head + 两个 `split-stage` 进程（`:9331` relay、`:9332` tail），`-c 8192`、`-np 1`。

| 项 | 值 | 出处 |
| --- | --- | --- |
| 3103 token 历史的**整段** prompt eval | **19847.34 ms**（19.85 s，156.3 t/s） | `build-split/reuselong_9412.log` 0.21.859 |
| 同一 fit 的**第二轮**（历史 3119 + 新 38） | prompt eval **714.02 ms**，total 1689.66 ms | 同上 0.23.563 |
| 另一轮同配置 | 整段 15890.42 ms（195.3 t/s）→ 第二轮 **603.73 ms** | `build-split/reuselong_9404.log` 0.17.781 / 0.19.228 |
| 链路失效时的样子（修复前） | `seq_rm: remote stage 1 failed … send failed`；`state_seq_set_data: … failed the state load` | `build-split/kill_ckpt_9415.log` 0.03.615、`kill_ckpt_9416.log` 0.11.175 |
| 本地链上下文 checkpoint 大小 | **19.266 MiB** @1806 token（`checkpoint 1 of 32`） | `.qlh_tmp/issue1_after_final.log`、`issue1_before_ub2048.log` |
| 远程段 state blob（2B 的 8 层段、`--rs-seq 8`） | **6,733,996 B = 6.42 MiB**，与历史长度无关（`pos_max` 17 与 59 同值） | `build-split/stage{816,1624}.log` 的 `OP_STATE_SAVE` |

**19.85 s 与 0.71 s 是同一份日志里的两件事，不是一次 A/B**：前者是「3103 token 整段重算」的成本，
后者是「只算 38 个新 token」的成本。修复前的失败形态在 `kill_ckpt_9415/9416` 里是**报错**，不是
「慢 19.85 s」——链首恢复了、尾段停在旧位置，下一次截断式 `seq_rm` 在尾段失败。日志里没有
「修复前第二轮 19.85 s」这一行（§7）。

---

## 2. MTP（2B 链，`-np 1`，`n_predict 192`，三次取中位）

### 2.1 退化 prompt（5 token，输出是复读）

| 配置 | ms/token（3 次） | 中位 | 相对 off | 接受率 |
| --- | --- | ---: | ---: | ---: |
| off | 36.84 / 37.36 / 37.32 | **37.32** | 1.00× | — |
| k=1 | 29.54 / 30.96 / 31.24 | **30.96** | 1.21× | 98.96%（95/96） |
| k=2 | 26.26 / 28.32 / 29.44 | **28.32** | **1.32×** | 97.67%（126/129） |
| k=3 | 23.18 / 25.09 / 25.50 | **25.09** | **1.49×** | 96.60%（142/147） |

出处：`build-split/mtp_time/np1_chain_{off,k1,k2,k3}_*.json`（01:50–01:51）、
`mtp_time/srv2_chain_{k1,k2,k3}.log` 的 `draft acceptance`。prompt 是 `The capital of France is`，
模型只会复读 `The capital of France is Paris.`，**接受率 97% 是这句话的性质，不是 MTP 的性质**。

### 2.2 三个真实负载（`-np 1`，三个 prompt 各三次，今日 06:44 重测）

| 负载 | prompt token | off | k=2 | 比值 | 接受率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 代码续写（`def quicksort` 上下文） | 131 | 35.76 | **41.30** | **0.87×** | 46.9%（92/196） |
| 中英混合（Python + 中文提问） | 95 | 35.74 | **51.69** | **0.69×** | 27.0%（67/248） |
| 纯中文（续写北京介绍） | 23 | 35.78 | **56.86** | **0.63×** | 19.4%（53/273） |

出处：`.qlh_tmp/mtp_remeasure_out.log`（脚本 `.qlh_tmp/mtp_remeasure.sh`）、服务日志
`.qlh_tmp/mtp_re_{off,k2}_064306.srv.log`。复现命令：`bash .qlh_tmp/mtp_remeasure.sh`。

**这张表是本节最重要的一条。** 把 2.1 与 2.2 两组按「每轮成本」折算，同一副链上 k=2 的一轮
实际花 **2.21–2.24 个 plain step**（用 `mean len × ms/token ÷ off 的 ms/token` 算：83.3/37.3、
80.1/35.8、79.6/35.7、79.0/35.8），这与 [`SPECULATION.md`](SPECULATION.md) §0 的成本模型
（k=2 是 4.36 ÷ 2 = 2.18 个单位）吻合。于是回本线是 `mean len > 2.22`，即**每轮接受 1.22 个草稿
（接受率 61%）以上**；三个真实负载的 `mean len` 是 1.94 / 1.54 / 1.39，全在亏的那一侧，
而 2.1 的复读 prompt 是 2.94。**1.3× 不是 MTP 的性质，是接受率的性质**——2B 上别按 1.3× 规划。

### 2.3 `-np > 1` 与 MTP

| 配置 | np=1 | np=4 |
| --- | ---: | ---: |
| 链 + MTP k=2 | **28.33** | **34.55**（1.22× 慢） |
| 整模 + MTP k=2 | **22.56** | **28.37**（1.26× 慢） |
| 整模不投机（对照） | 32.09 | — |

出处：`build-split/mtp_time/srv_np1_spec.log`、`srv_np4_spec.log`（链）；`build-split/srv_w1_m.log`、
`srv_w4_m.log`（整模，`n_slots` 1→4）；`mtp_time/whole_off_*.json`（对照行，**`predicted_n=128`**，
而 MTP 各行是 192 token，两次长度不同，只能当量级参考）。**`-np 4` 的三次请求仍是逐个发的**
（日志里没有重叠的 batch），所以变慢不是排队，而是 `n_seq_max` 带大的缓冲——与
[`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §3.2 的猜测一致，但机理仍未定论。

---

## 3. DFlash（35B 链，CPU）

四个 prompt（两段代码、一段解释、一句中文），`n_predict 160`，`-np 1`，`-c 4096`，
每个配置各跑一遍（`.qlh_tmp/casc_probe.sh`、`casc_base.sh`）：

| 配置 | ms/token（4 prompt） | 均值 | 相对不投机 | 整块接受率 |
| --- | --- | ---: | ---: | ---: |
| 不投机 | 74.79 / 68.35 / 71.43 / 74.57 | 72.28 | 1.00× | — |
| MTP k=2 | 74.97 / 74.19 / 61.50 / 74.59 | 71.31 | 1.01× | 56.4%–80.3% |
| DFlash k=3 | 59.43 / 67.38 / 52.25 / 78.12 | **64.30** | **1.12×** | 49.7%–76.4% |
| DFlash k=15 | 150.90 / 197.41 / 118.36 / 155.40 | 155.52 | 0.46× | 12.6%–26.2% |

出处：`.qlh_tmp/casc/{run,run_base}.log`（逐请求 tok/s 与 `draft_n/accepted`），服务日志
`.qlh_tmp/casc/srv_{base,mtp2,k3,k15}.log` 的 `eval time`。

单 prompt 对照（`n_max=3`，两条请求的 `predicted_n` 不同——402 vs 512，投机那条提前停了，
所以比的是 ms/token 而不是总时长）：链上 **59.96 / 61.66** vs 不投机 **76.72 / 74.18**
→ **1.24×**；整模 + DFlash k=3 是 63.96 / 61.93（`build-split/dflash/speed_{chain,chain_nospec,whole}.log`，
03:19–03:20）。`draft acceptance = 0.71916 (274/381)`（`dflash/speed_chain.log`），链与整模逐 token
相同 3/3（`dflash/accept.log`）。

8 路 tap 的逐位结论（`dflash/taps_final.log`，单 ubatch，15 token）：8/8 与整模相同；
tail 的 local 1 = 整模 33、local 6 = 整模 38；logits 15/15 行逐位相同。

---

## 4. 并发与多 slot

| 项 | np=2 | np=4 | 出处 |
| --- | ---: | ---: | --- |
| 10 个不同长度请求 | 10/10 200，存活 | 10/10 200，存活 | `build-split/multislot_{np2,np4}.out` |
| prefix cache 复用（round1/round0） | 0.115（408.1→47.0 ms） | 0.145（406.6→59.0 ms） | 同上（15:12 重跑覆盖；早先两轮是 0.100 / 0.117） |
| 串行/并发输出逐字节一致 | PASS | — | `.qlh_tmp/accept_np.py --np 2` |
| tail 的 logits 缓冲按「实际要读的行」算 | 1 行 → 0.95 MiB（`n_outputs_max=1`，`build-split/stage_tail.log`） | 4 行 → 3.79 MiB（`n_outputs_max=4`，`.qlh_tmp/issue1_after_final.log`） | 缓冲 = `4 B × n_vocab × 行数`，行数 = `n_seq_max × (1+n_max)` |

链上 `-np` 的代价见 §2.3。`verify_multislot.py` 的两项（np 2 / np 4）在 `verify.sh` 组 5。

---

## 5. 内存

同一个 1806 token 的 ubatch（`-c 8192 -np 4`，2B 链，`-b 2048`）：

| 项 | 修前 | 修后 | 出处 |
| --- | ---: | ---: | --- |
| tail 的 logits 缓冲 | **1710.76 MiB**（`n_outputs_max=1806`） | **3.79 MiB**（`n_outputs_max=4`） | `issue1_before_ub2048.log:533` / `issue1_after_final.log:459` |
| head 的 hidden 缓冲（`logits=no, embd=yes`） | 14.11 MiB | 14.11 MiB | 两 log 的 532 行 |
| 1806 token 的那次 prefill（1810 含采样行） | 172.61 t/s（10485.85 ms） | **219.39 t/s**（8250.22 ms） | 同上 |
| 上下文 checkpoint（1806 token） | — | 19.266 MiB | `create_check` |
| 远程段 state blob（§1 的三段远程链，2B 的 8 层段、`--rs-seq 8`） | — | 6.42 MiB | §1 |

**读法**：修前 tail 按 `n_batch` 行分配 logits，修后只为「实际要被读的行」分配
（`n_seq_max × (1+n_max)`）。省下来的不只是内存——同一 ubatch 的 prefill 顺带快了 27%，
因为不再申请/清零那 1.7 GiB。

远程 stage 那边同一笔账更大：`split-stage` 只做 tail、`-b 2048`、248320 词表时，`sched_reserve`
的 pp 图缓冲（`graph (pp bs=2048, tg bs=1)`）从 **2052.10 MiB** 降到 **240.09 MiB**。口径：
`--role tail`、2B 的 20:24 段、`-c 4096 -b 2048`、`n_seq_max=1`；两次运行其余参数相同，只差
logits 行数 `n_outputs_max`（2048 → 1）。出处 `build-split/i2_tail_before.log:553`（2048 行）与
`i2_tail_after.log:553`（1 行）。

---

## 6. 验收覆盖

| 入口 | 覆盖 | 最近一次结果（2026-11-08，纯分段树 `b1646d6d`） |
| --- | --- | --- |
| `bash build-split/verify.sh` | 九组 **25 项**（含一次三目标构建检查；只用 `--skip-build` 跑时是 24 项）：本地链逐位、N=1 直通、server 双段、跨进程、多 slot、三段链、远程 server、链上 MTP、DFlash 8 tap | **PASS=25 FAIL=0** |
| `python build-split/verify_multislot.py --np 2` | 组 5 的一半 | 10/10 200，prefix cache 生效 |
| `python build-split/verify_multislot.py --np 4` | 组 5 的另一半 | 10/10 200，prefix cache 生效 |
| `.qlh_tmp/accept_np.py --np 2` | 串行 vs 并发四个输出的逐字节一致 | **ALL PASS**（并发墙钟 1.6 s） |
| `bash .qlh_tmp/mtp_remeasure.sh` | §2.2 | 三个负载 ×（off / k=2）× 3 次，见 §2.2 |
| `python build-split/verify_seq_rm_exact.py --expect-known-fail` | `seq_rm` 删尾部的精度（A/B/C 三形态，无 split 无 server） | **PASS=4 FAIL=2**，红的两条是 FULL 形态，退出 0；见 §10 |
| `python build-split/verify_seq_rm_shared.py` | shared-prefix 下 `prompt_clear` 是否误伤另一个 slot | 两次各 2 条 completion 逐字节相同、`health` 均 200；`[INFO] injection never fired`（该形状不走 partial `seq_rm`） |

`verify.sh` 的判据全部是**逐位/逐字节**（`split-smoke`、`split-rows`、`dflash-taps` 三个目标），
不看生成文字；组 9 需要 35B + DFlash 权重。上表第一行的 `PASS=25 FAIL=0` 有两份 11-06 的完整
记录：`.qlh_tmp/verify_after_comments.log`（06:52）与 `build-split/verify_run1.log`（10:37）；
`--skip-build` 那 24 项的同一条结论见 `build-split/verify_final_hello.log`（15:15）。11-08 的
25 项记录在 `.qlh_tmp/pure-split-acceptance/verify_pure_2026-11-08.log`（纯分段树，带构建）。

**纯分段树上的独立验收（2026-11-08）**：`e0f142d` + 主补丁 = 树 `b1646d6d`（45 文件，不含任何
SLIM-ARC），`--clean-first` 全量构建后 `verify.sh` **PASS=25 FAIL=0**（九组全过，含组 6 的三段链
checkpoint 往返与组 9 的 8 路 tap），`verify_chain_death.py` **PASS=36 FAIL=0**；同一棵树上的
`seq_rm` 专项是 `PASS=4 FAIL=2`（红的两条都是 FULL 形态，§10）。这证明主补丁
不依赖增量补丁：拆出来的两份补丁里，主补丁自己就是一条可编译、可验收的线。日志：
`.qlh_tmp/pure-split-acceptance/{verify_pure,chain_death_pure,build_clean}_2026-11-08.log`
（上一轮 55 文件 / tree `639d0157` 的那三份在 `_prev_55files/` 下，数字相同，树不同）。

### 6.1 红的是环境还是代码：先看 93xx 上还有没有上一次的 stage

`verify.sh` 的组 4（`20:24` tail）与组 6（`8:16` relay + `16:24` tail）**都用 9331**。上一次
run 的 stage 没清干净时，组 6 的 client 会连到组 4 那台还在跑的 tail，报

```
stage identity mismatch: ... role tail vs relay
```

随后五项一起红：远端 KV 没被裁到 `pos_max=1`、`OP_SEQ_RM` 没到 relay/tail、checkpoint 不一致、
漂移检测失败。**这是环境，不是代码**——「一段接错、后面全崩」的形状本身就是信号。

**根因**：`kill_ours` 只杀「已经绑定 93xx」的进程，而 `netstat` 看不见一个**还在 loading 模型、
尚未 listen** 的 stage。上一轮被打断（Ctrl-C、终端关了、构建被 kill）之后紧接着再跑一轮，这个
孤儿进程会在组 6 起来之后才绑上端口——于是被当成 relay。

**跑验收前先做两步检查**（比读日志快）：

```
netstat -ano | grep LISTENING | grep -E ':93[0-9][0-9]'     # 必须空
tasklist | grep -i 'split-stage\|llama-server'              # 必须空
```

两条都空再跑。看到 `role tail vs relay` 时先做这两步，别去读代码。**9103/9104/9106 是常驻
服务，绝不要放进 `kill_ours` 的范围**——脚本只用 `:93xx` 匹配，就是为了这个。

**另一个坑：这些 Python 验收脚本不能用 `python -X utf8` 跑。** `verify_chain_death.py` 的
`kill_93xx()` 用 `subprocess.run(["netstat", "-ano"], text=True)` 读中文 Windows 的 netstat
输出，那是 GBK；UTF-8 模式会让子进程输出解码失败，`stdout` 变成 `None`，随后在
`.splitlines()` 上抛 `AttributeError`（2026-11-08 实测）。用默认 locale 跑，日志是 GBK，要留
档就用 `decode("gbk")` 转一次 UTF-8。

---

## 7. 未落盘 / 未复现

| 数字 | 谁在引用 | 现状 |
| --- | --- | --- |
| MTP k=2 **1.33× / 1.06× / 1.00×**（代码 / 混合 / 中文），口径见 §2 | [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §3.2、`ARCHITECTURE.md` | 日志里没有这三个数。本日重测是 **0.87× / 0.69× / 0.63×**；1.3× 只出现在第 2.1 节那种复读 prompt 上（1.32×）。**需在真实负载上重测，不能继续当结论用** |
| DFlash k=3 **+23.7%** | [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §3.2 | 日志复算是 **1.24×（+24.1%）**，差 0.4pp 在复现噪声内 |
| 级联整块全接受 **42.5% / 2.8%**（k=3 / k=15） | [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §2.3、[`SPECULATION-MATRIX.md`](SPECULATION-MATRIX.md) §4 | 42.5% / 2.8% 就是从 `.qlh_tmp/casc/srv_{k3,k15}.log` 数出来的：`3/3` 96/226 轮、`15/15` 5/177 轮。把 `1/1`、`2/2`、`4/4` 这类块尾也算成全接受是 **43.4%（98/226）/ 4.5%（8/177；复算到 9 轮，多出的一轮是 `4/4`）**。两个口径不完全可比（那边数的是 Python 驱动的级联路径），且 k=15 只有 5–9 轮全接受，比例本身噪声很大 |
| checkpoint **19.85 s → 0.71 s**，口径见 §1 | [`PATCHES.md`](PATCHES.md) 第四代、[`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §1 | 两个数各自成立，但「修前第二轮 19.85 s」没有日志：19.85 s 是整段重算的成本，修前实测到的是**失败**（`kill_ckpt_9415/9416`） |
| **3137 token** 历史 | 同上 | 日志是 **3103**（+16 生成 = 3119）；3137 在任何日志里都找不到 |
| 2B 上 k=3 与 k=2 的 **25.09 vs 25.87** | [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §3.2 | k=3 中位 25.09 对得上（第 2.1 节）；k=2 中位是 **28.32**，25.87 未复现 |
| 35B MTP k=2 **1.13–1.25×**，本周对照见 §3 | [`SPECULATION.md`](SPECULATION.md) §1 | 那份是 10-31 的草稿成本模型 + 单 prompt；本周链上端到端（4 prompt）是 **1.01×**。两个口径都要保留，不能互相替代 |
| v5 候选通道的等价性 / 带宽 / 延迟（§8） | [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §1 表格（数据见本文件第 8 节） | 只在 fork 的提交说明里，**`build-split/` 日志中没有这些行**；按「未落盘到日志」对待 |
| DFlash 头 / DSpark 头 / NextN 层「逐字节官方」（sha256 / LFS oid，§11.1） | 本文 §11.1 | 当日的人工比对，**`build-split/` 下没有留下日志**；独立复算没做过。它支撑的是「接受率退化不可能来自头是兼容版」这个排除项 |
| catch-up 让链上接受率回到整模（0.688 → ~0.91）、512 token 省 10–14 s（§11.4） | 本文 §11.4 | **估算，不是实测**：代码没改过。代价（每请求一次 N 行 draft 前向）与收益都按公式推，长 prompt 上净负 |

另有**两项能力做了但没有端到端证据**。它们不是数字，所以不进上表：

- **多模态全维位置**：`split_expand_pos()` 与 `req_header::n_pos` 让 4 个 M-RoPE section 跨段
  传下去，验收组 6 的三段跨进程在 `pos = [p,0,0,0]` 上逐位通过。但**没有图像输入**的端到端验证——本机
  `H:/qlh_models/mmproj` 有投影模型，分段链 + mtmd 的组合从未跑过。所以「图像位置能传」目前
  是代码级结论，不是实测结论。
- **本地纯 KV 链的 ctx_shift**：`can_shift=false` 对 qwen35/qwen35moe 是忠实的（整模同样
  不支持），但对非 MROPE 的本地链属于过度保守，从未验证过打开它是否安全。`seq_add` 是
  void、无返回值，要做到「逐段都真的移了同样量」只能靠 shift 后断言各段 `seq_pos_max`。

---

## 8. 远程 tail 的候选通道（协议 v5 引入，v8 仍在用）

**这一节的数字只出现在 fork 的提交说明里（`c5bd787b` 的候选通道一节），不在 `build-split/` 的日志里。**
所以它们按「未落盘到日志」对待：机制与数据可信到「提交说明」这一层，独立复算没做过。

口径：248320 词表的分段链、单序列；采样参数逐项注明在下面，fuzz 的 10 组随机参数是
「把整条链随机化」而不是只换 `top_p`。

- **为什么等价**：`common_sampler` 的链顺序是 `penalties → dry → top_n_sigma → top_k →
  typ_p → top_p → min_p → xtc → temp`，**`top_k` 之后的所有步骤只在 k 个候选内作用**。
  所以 tail 本地跑完 `logit_bias → penalties → top_n_sigma → top_k`，回传 40 个 `(id, logit)`，
  server 从 `cand_first`（`top_k` 在链里的下一个节点）接着跑，与回传整行**数学等价**。
  回退是白名单预判（`common_sampler_candidates_check()`，每请求一次），不是运行时兜底；
  条件见 [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §2.2。
- **惩罚历史按帧重放**：server 每帧带最近 `min(penalty_last_n, n_prev)` 个 token，tail 从零
  `reset` + 重放。代价 256 B/帧；换来的是重连、`seq_rm`、`state_restore`、slot 复用之后
  两端自动一致（增量式一旦两端对「哪些 token 被 accept」有分歧就会静默错）。
- **帧大小**：`rsp_header` 32 B + 40 × `token_cand` 8 B = **352 B/token**（两个 `static_assert`
  锁了 32 与 8 的尺寸），对整行 **993 KB** 是 2821 倍。
- **等价性实测**：4 prompt × 64 token、penalties 三参、`top_n_sigma`、xtc、temp=0、`top_k=1`、
  `logit_bias`、ignore_eos、post-sampling logprobs、cache 复用两轮、400 token 长生成、
  10 组随机参数 fuzz —— **全部逐 token 相同**。
- **带宽**（TCP 代理计数）：同一 64-token 请求，整行 67,932,452 B vs 候选 4,383,012 B →
  每 token 省 992,960 B。
- **loopback 延迟**：57.92 → 52.92 ms/token（+8.6%）。loopback 等效 ~200 MB/s，所以只值 5 ms；
  按跨机 LAN 的 24 ms/0.95 MiB 折算，跨机 decode 从 4.67 / 5.38 / 6.05 预计到
  **6.6 / 8.1 / 9.7 tok/s**（整模的 66–87%，原 45–54%）——**这是折算，不是实测**。
- **没做**：prefill 阶段的 tail 仍整行回传（只有「将被采样的那一行」才请求候选），prompt
  处理的带宽未优化。

---

## 9. 远程链的投机：五类 prompt 的真实收益，以及为什么弱机器与回环上是负的

口径：35B（`qwen35moe`，40 层 trunk + 1 个 NextN 层），head `0:32` + tail `32:40`，贪心，
32-token 生成。数据来自 ROG（LAN）、Surface（4C8T 弱机器）与本机回环三处；**`build-split/` 里
没有与之对应的日志文件**，按 §7 的口径，它们可信到「实测记录」这一层，独立复算没做过。

### 9.1 五类 prompt × 三配置（ROG，LAN）

| prompt（token） | 不投机 | ngram | MTP k=2 | MTP 变化 |
|---|---:|---:|---:|---:|
| 散文 882 | 8.18 | 7.33 | 8.47 | +3.6% |
| 代码 1205 | 8.74 | 6.92 | 9.47 | **+8.4%** |
| JSON 2379 | 9.47 | 7.09 | 9.36 | **−1.2%** |
| agent 2508 | 7.51 | 5.09 | 7.62 | +1.5%（当时无 draft） |
| 人为重复 `A+B+A+B+A` 89 | 9.73 | 7.75 | 10.75 | +10.5% |

**读法**：

- **中位约 +6%，区间 +8.4% ~ −1.2%。** 这就是跨机投机的真实量级，不是 +15%。
- 历史那个 **+15%（10.20 vs 8.88 tok/s）来自一个 27-token 的高确定任务**。同一个任务上 k=1 是
  10.11、k=2 是 10.20、k=3 是 9.76——k≥3 的验证帧要 k+1 行，而 **head 有 32 层、tail 只有 8 层，
  宽帧的代价几乎全在 head**，所以「k=2 是甜点」只在那个切点上成立。
- **agent 那行 +1.5% 是在「当时无 draft」下测的**（tail 没给出 NextN 草稿），它证明的是「没有
  draft 时投机不会变慢太多」，不是 MTP 在 agent 负载上的上限。
- **人为重复那行 +10.5% 是送分题**，与 ngram 那列 7.75 一样，只说明「可复制的 prompt 上投机
  好用」，不能代表真实负载。

### 9.2 弱机器与回环：收益全部转负

| 配置 | 不投机 | 投机 k=2 | 相对不投机 |
|---|---:|---:|---:|
| Surface（4C8T，纯 CPU）散文 | 8.65 | 5.21 | **−40%** |
| 本机回环，带 grammar（`toolcall.gbnf`） | 20.82 | 13.95 | **0.67×** |

**四个真实原因**：

1. **每帧两次 hop**：`OP_MTP_DRAFT` 出草稿，`OP_DECODE` 再验证——一轮走两趟链。
2. **MTP draft 不看 grammar / rbudget**：同一个 tail 上，接受率从无 grammar 时的 **2.76 token/帧**
   掉到 **1.46–1.73**。
3. **回退率 9–19%**：`toolcall.gbnf` 这种严格语法实测 **11/16 帧被拒**；而每次回退 ≈ 一整个
   普通 round（撤 KV + 重放 prompt + sampler 回滚）。
4. **回环上 993 KB 只要约 1 ms**：验证通道省掉的四个数量级带宽，在回环上本来就不值钱，于是
   这笔账只剩成本。

另外 **FORCING 期间 tail 回的未掩码 argmax 基本每轮都被拒**，有约 **2 倍白跑**。

### 9.3 两条通道的实现事实（协议 v7 / v8）

- **v7 验证通道**：tail 对验证帧的**每一行**回一个 argmax，回复体是 `n_tokens × int32`。贪心验证
  只需要 token id——`temp=0` 时 `common_sampler_sample_and_accept_n` 逐行等价于 argmax。单帧回复
  （4 行）**993,312 B → 48 B（20,694×）**。帧内行号与 batch 绝对下标只在帧首对齐时相同，所以
  `request_remote_verify_rows` 硬性要求 `off == 0`，不满足就不 arm（报错，不静默错位）。
- **v8 远程 MTP draft**：`OP_MTP_DRAFT`（op 8）带一行 hidden 与 `mtp_req{token,pos,k}`，回 k 个
  int32，**k 步递归在 stage 内部完成**；decode 回复可追加 `n_rows × n_embd` 个 float
  （`DECODE_FLAG_NEXTN_ROWS` / `RSP_FLAG_NEXTN_ROWS`）。`stage_id.has_nextn` 占原 `_pad` 字节，
  **故意不参与身份比较**——「能力」不是「身份」。`n_rows` 只能取 `rsp_header.n_tokens`，不能从
  字节数反推（否则帧边界错位）。
- **grammar**：head 侧先做单 token 自检（`common_sampler_grammar_rejects()`，内部是
  `llama_grammar_apply_impl` 只掩码不推进），被拒就返回 `LLAMA_TOKEN_NULL` → **整帧回退**
  （撤 KV + prompt + sampler，含 grammar 状态）→ `skip_draft_once` 防止同一条路径空转。
- **rbudget**：`common_sampler_rbudget_rejects()` → `common_reasoning_budget_rejects()`，纯读谓词
  （不调 `apply`，状态机在 `accept` 里推进）；预算耗尽不是放开而是**收紧**——整词表只剩 forced
  token。

### 9.4 正确性判据：为什么不是「与不投机逐字节相同」

这一节比 9.1/9.2 的所有 tok/s 都要紧。

**契约层**：llama.cpp 从不承诺 batch invariance。上游 `docs/DETERMINISM.md` 把
`--deterministic` 的覆盖范围限定在 **RMSNorm（全 backend）+ MatMul(CUDA) + Attention(CUDA)**：
"Determinism for attention on non-CUDA backends … planned"、"Quantized matmul is unchanged for
now (stretch goal)"。**CPU + 量化正落在这片空白里**——「1 行与 k+1 行必须算得一样」不是这份代码
承诺过的性质。（本机副本 `H:\llamacpp-c\.qlh_tmp\DETERMINISM_upstream.md`。）

**三个分界**（都在 ggml 的 CPU 代码里）：

| 分界 | 形状效应 |
|---|---|
| `ggml/src/ggml-cpu/ops.cpp` 的 `use_split_kv_path` | **单 token + KV ≥ 512** 才走 split-KV（切段并行 + `reduce_partials` 重加权），多 token 走 `f16_one_chunk` 整条累加 |
| `ggml/src/ggml-cpu/repack.cpp` 的 `nrows > 3` | `> 3` 行是 GEMM，1–3 行是逐行 GEMV |
| `ggml/src/ggml-cpu/tiled/tiled.cpp` 的 `ggml_tiled_min_batch` | `rows >= 8` 才启用 16×16 tile |

**实测**（整模、不分段、纯 embeddings 模式）：1 行 vs 2 行的 h **2048 个元素全部不同**
（`max|Δ| = 0.273`）；`rows=2` 与 `rows=3` 差异完全相同、`rows=4` 跳变——正是 `nrows > 3` 那条。

**v2 当年也没做到**：`chain_client.py` 的 `forward_batch(clients, block, n_past+1)` 同样是 k+1 行
一次 forward；它声称的逐位一致来自 `probe_perfect_draft.py`，而那个 probe 只测了 **2 行 + 30
token**。

**所以判据分三段**：

- **(a) 无回退时**：链上投机 == **整模投机**，必须逐字节相同（严格，能过）。
- **(b) 带回退时**：只能保证到「漂移翻掉的第一个近似并列处」。
- **「投机 == 不投机」降级为 INFO**：它测的是 batch invariance，不是接线。

**上游状态**（说明为什么我们不修）：

| 上游 | 结论 |
|---|---|
| PR #28108（已关闭） | FA 宽度 3/4 不能 bitwise 一致：`n_tokens 1、2 pass；3、4 FAIL`（正是 k+1 = 3/4）。修法代价 nb=3 **+16.5%**、nb=4 **+15.5%**；把 1..4 钉到一个实例会让 nb=1 慢 **+84.5%** |
| Issue #28111（open） | IQ 量化在 `n=5` / `n=8` 不 batch-invariant，定位到 MMVQ 内部 per-`ncols` 特化改了累加顺序 |
| Issue #25618（open） | 量化 target 上 ngram 无损、`draft-mtp` 分叉、bf16 target 一致 |
| vLLM | 把 batch invariance 当**必做项**：改写全部 GEMM/GEMV、给开关、写专文 |

要在 CPU 上 bit-exact 需要三处一起改（FA split-KV 门控、repack 分界固定、MoE 每专家形状钉死），
代价主要在第一项——长 context 下单 token decode 的 attention 只能单线程扫 KV，而那正是 split-KV
存在的理由。**我们不改上游，只记录。**

### 9.5 `split-stage` 的线程与 ubatch（顺带的实现遗漏）

`GGML_DEFAULT_N_THREADS = 4`，**而且 `n_threads_batch` 也是 4、不会自动跟随**——Surface（4C8T）
上原来只用了 4 个线程。现在 `split-stage` 有 `-t/--threads`（同时设两者）、`-tb/--threads-batch`
（单独覆盖 batch 那一半）、`-ub/--ubatch-size`（让验证帧在帧内逐行跑，是测量旋钮不是调优旋钮）。

| 配置（Surface） | 原来（4 线程） | `-t 8` | 变化 |
|---|---:|---:|---:|
| 不投机 | 7.41 | 7.67 | +3.5% |
| MTP k=2 | 5.12 | 5.48 | +7% |

**只设 `n_threads` 拿不到这笔收益**，batch 那一半必须一起设；head 的 `-t` 不会过网。

### 9.6 ngram 的真相

- **chat 模板下 ngram 全程 0 draft。** 机制是「复制上下文里最后 12 个 token 的后续」，而 chat
  模式下模型对重复文本的反应是**评论**（"It looks like you've pasted the same paragraph…"）而
  不是续写，所以永不命中；代码/JSON 生成的是新内容，同理。
- **只有 raw completion + 人为重复文本才有收益**：同一 prompt 上 ngram **27.85 vs none 9.59 tok/s
  = 2.9×**、接受率 **1.0**。这就是历史那个「4.3×」的全部来源，而那次用的 prompt 是
  `A + B + A + B + A`。
- 所以 9.1 表里 ngram 五类全部低于不投机，与「ngram 有用」不矛盾：**它们测的不是同一个东西**。
  测 ngram 必须用真实负载（散文、代码、agent 多轮）。

### 9.7 正确性验收（与 9.1/9.2 的性能数互不替代）

`verify_mtp.sh`：同一行 h（先证明与整模 `h_nextn` 逐位相同）、同一 token、同一 pos，远程 tail
内部递归 3 步的结果与「整模 + MTP」**逐个相同**（11751 / 13 / 271 三个 token 全中）。
`verify_mtp_e2e.sh`：远程链三组（不投机 / ngram / MTP）输出**逐字节相同**。

`verify.sh`（带构建）25 项（`--skip-build` 时 24 项）、`verify_chain_death.py` 36 项、
`verify_seq_id.py` 7 项、`verify_mtp.sh` 6 项、`verify_mtp_e2e.sh` 6 项、`verify_grammar_spec.sh`
22 项、`verify_rbudget_spec.sh` 36 项在 v8 之后全过；`seq_rm` 的两种形态另算，见 §10。

---

## 10. `seq_rm` 的精确性（`verify_seq_rm_exact.py`）

**判据为什么成立**：`seq_rm(p, -1)` 只删 p 之后的 KV，第 p 行的输入不变——它的注意力覆盖
`0..p`（含自己），第 p 个 token 也没变。所以重算第 p 行必须得到**逐位相同**的 logits。所有
「回滚后重算」的路径都建立在这个前提上：投机验证帧被拒后的回退、prefix cache 复用、checkpoint
恢复。

**口径**：2B 整模（`qwen35-2b-Q4_K_M`）、纯 CPU、`n_threads=4`、p=29、batch 里 14 个 prompt
token、vocab 248320；只用 llama API，**无 split、无 server、不占端口**。探针是
`build-split/bin/seq_rm_exact.exe`（源 `.qlh_tmp/seq_rm_exact.cpp`）。日志：
`.qlh_tmp/pure-split-acceptance/seq_rm_exact_2026-11-08.log`（在纯分段树 `b1646d6d` 上跑）。

| 形态 | `n_rs_seq` | k（回退量） | 结果 |
|---|---:|---:|---|
| A1 FULL | 0 | 1 | **红**：`seq_rm=false`，`pos_max 29 -> 29` |
| A2 FULL | 0 | 3 | **红**：`seq_rm=false`，`pos_max 31 -> 31` |
| B1 RS | 3 | 1 | 绿：`seq_rm=true`、`pos_max` 降到 28、重放 logits 逐位相同 |
| B2 RS | 3 | 3，重放 1 行（投机回退的真实形状） | 绿，同上 |
| B3 RS | 3 | 3，重放 3 行 | 绿，同上 |
| C1 RS | 3 | 4 > 快照窗口 | 绿：**明确拒绝**（false） |

`--expect-known-fail` 下是 `PASS=4 FAIL=2`、退出 0（它只承认「红的两条都是 FULL 形态」这一种
红）；不带开关时退出 1。

**FULL 形态的原始输出**（`seq_rm_exact.exe <model> 0 29 1 1`）：

```
[seq_rm_exact] n_rs_seq=0 p=29 k=1 k_replay=1 prompt_tokens=14 n_vocab=248320
[seq_rm_exact] seq_rm=false pos_max 29 -> 29
[seq_rm_exact] replay rc=-1 pos_max=-1
[seq_rm_exact] argmax first=27 second=-1
[seq_rm_exact] RESULT FAIL
```

读法：`seq_rm` 返回 false 而 `pos_max` **一点没降**——尾部 KV 一行都没删（hybrid 的短路：
recurrent 拒绝之后 attention 那半被整体跳过）。随后重放 `rc=-1`：decode 拒绝在「KV 比 prompt
长」的位置上写，于是没有第二份 logits 可比（`second=-1`）。**红的是「删除」这一步，不是
「重算」**——只要删除成功（B 组），重放就逐位相同。

B 组原始输出（`seq_rm_exact.exe <model> 3 29 3 1`）：`seq_rm=true pos_max 31 -> 28`、
`replay rc=0 pos_max=29`、`argmax first=27 second=27`、`logits diff=0/248320`、`RESULT PASS`。

**C 组为什么必须绿**：回退量超过快照窗口时，`seq_rm` 必须**明确拒绝**而不是删一半。删一半
比拒绝更糟——调用方拿到 true，以为位置已经一致。

**结论**：server 默认 `n_rs_seq = 0` 的 hybrid 上，「删尾部」永远失败，而这是上游缺陷
（open PR #28007 / issue #31656），不是投机接线的问题。调用方必须检查返回值——五个调用点
已按场景降级，见 [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §2.9。上游修好 FULL 缺陷后
A 组转绿，脚本会提示把 `--expect-known-fail` 关掉。

**shared-prefix 形态**（`verify_seq_rm_shared.py`，日志
`.qlh_tmp/pure-split-acceptance/seq_rm_shared_2026-11-08.log`）：`n_cmpl=2` 让 parent 与 child
共享前缀，再注入 `QLH_FORCE_SEQ_RM_FAIL=1`，比较注入前后两份 completion。实测两次各 2 条、
**逐字节相同**、`health` 都是 200；但日志里只有

```
[INFO] injection never fired: this shape does not reach a partial seq_rm
       (the comparison above then proves nothing about isolation)
```

——这个形状**不经过** partial `seq_rm`，所以「`prompt_clear` 误伤另一个 slot」不可达。**看到
这行就说明这次跑没测到东西**：脚本按 INFO 通过，不是按「隔离成立」通过。

---

## 11. 换模型：从蒸馏版到官方版（2026-11-08）

**这一节为什么在这里**：换模型的全部结论原来只写在 `H:\llamacpp-c\build-split\accept_baseline.md`
与 `build-split/acc/` 的日志里，而那个目录不进版本库，下一个接手的人看不到。数字原样搬进仓库，
出处逐条注明；`accept_baseline.md` 仍是原始记录，两份不一致时以日志为准（下文标出两处小出入：
imatrix 路径的 key 与整模加载耗时）。

### 11.1 两份权重、为什么换

| | 旧（**已弃用**） | 新（**现行**） |
| --- | --- | --- |
| 文件 | `Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-I-Quality.gguf` | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` |
| 来源 | lordx64 蒸馏 + **mudler 的 "apex" 工作流**做的混合量化（Q6_K 主 + Q8_0 nextn） | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` 的 `UD-Q4_K_M`（官方 MTP 仓） |
| 体积 | 23.48 GB（21.87 GiB） | **21.11 GiB** |
| 本机路径 | `H:/qlh_models/qwen3-6-35b-a3b-opus47-distill-apex-mtp/` | `H:/qlh_models/qwen3-6-35b-a3b-mtp-official/` |
| `general.name` | **`Safetensors`**（转换时没传 `--model-name`） | `Qwen3.6-35B-A3B` |
| 其他遗留 metadata | `general.dataset.0` 指向 `Lordx64 / Reasoning Distill Opus 4 7 Max Sft`；`quantize.imatrix.file` 是构建机的 `/home/ubuntu/work/apex/...`，`quantize.imatrix.dataset` 是 `/home/mudler/autoresearch-quant/...` | `quantize.imatrix.file` 是 `Qwen3.6-35B-A3B-GGUF/imatrix_unsloth...`，`quantize.imatrix.dataset` 是 `unsloth_calibration_...` |

出处：旧模型的 metadata 在 `build-split/acc/dist_pos_k8.log` 的 `llama_model_loader` 段
（`general.name` / `general.dataset.0.*` / `quantize.imatrix.*`）；新模型在
`build-split/acc/official_pos_k8.log` 的同一位置。两行的体积以两份日志的
`print_info: file size` 为准：官方 **21.10 GiB（5.10 BPW）**、蒸馏 **21.86 GiB（5.29 BPW）**；
上表的 21.11 / 21.87 与 `accept_baseline.md` 一致，是同一份文件的四舍五入差（0.01 GiB）。
「Q6_K 主 + Q8_0 nextn」与 mudler 的 apex 工作流来自该 GGUF 的模型卡，**不是日志**——日志只能
证到 `general.dataset.0` 指向 lordx64 的蒸馏仓、以及 5.29 BPW 比官方的 5.10 BPW 更高（与「主
权重放大到 Q6_K」一致）。

> **与 `accept_baseline.md` 的一处出入**：那份记录写 `quantize.imatrix.file =
> /home/mudder/autoresearch-quant/...`。日志里这是两条不同的 key——`imatrix.file` 指
> `/home/ubuntu/work/apex/...`，`/home/mudler/autoresearch-quant/...` 是 `imatrix.dataset`
> （作者名是 `mudler`，不是 `mudder`）。本节按日志写。

**新模型的形状与代码假设全部吻合**（`llama_model_loader` / `print_info` 段，
`build-split/acc/official_pos_k8.log`）：

| key | 值 |
| --- | --- |
| `general.architecture` | `qwen35moe` |
| `general.name` | `Qwen3.6-35B-A3B` |
| `qwen35moe.block_count` | **41** |
| `qwen35moe.nextn_predict_layers` | **1** |
| `qwen35moe.ssm.inner_size` | **4096** |
| `qwen35moe.ssm.state_size` | **128** |
| `qwen35moe.ssm.group_count` | **16** |
| `qwen35moe.ssm.conv_kernel` | **4** |
| `qwen35moe.full_attention_interval` | **4** |
| `qwen35moe.expert_count` / `expert_used_count` | **256** / 8 |

trunk depth 读作 **40**（41 block − 1 个 NextN），与 `n_layer_nextn == 1` 的断言一致；所以在
tail 段起 `32:40` 时 `--role tail` 的 `has_nextn=1` 成立、MTP draft context 建得起来。
**换模型零改代码**——形状这一层没有任何要适配的东西。

**头的来源也核对过**：DFlash 头、DSpark 头、NextN 层都与官方文件做过**逐字节**比对
（sha256 / Hugging Face LFS oid），三者都是官方的，不是「自己拼的兼容版」。这条重要，因为它
把「接受率退化」的候选原因砍掉了一类——**但它是当日的人工比对，`build-split/` 下没有留下
日志**，按 §7 的口径记为「未落盘」（见 §7 表格的倒数第二行）。

### 11.2 接受率基线（官方模型，整模，MTP）

口径：3 个真实 prompt（英文 prose / Python 代码 / 中文技术），`temperature 0`、`n_predict 64`、
`-c 4096 -b 512 -t 8 -ngl 0`、纯 CPU。prompt 原文在 `build-split/acc_test.py` 的 `PROMPTS`：
B-tree 插入时怎么保持平衡（prose，39 token）、`merge_intervals` 的实现（code，38 token）、
MoE 为什么比同等总量的稠密模型省算力（zh，46 token）；请求走 `/v1/chat/completions`、
`cache_prompt: false`（每次都是全新 prefill，避免前缀缓存污染 tok/s）。命令形状：

```
llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -c 4096 -b 512 -t 8 -ngl 0 --no-warmup \
             --spec-type draft-mtp --spec-draft-n-max <k>
```

指标取 server 日志的 `draft acceptance = x (a accepted / g generated), mean len = y`
（`mean len = 1 + accepted / verify_steps`），只统计 3 个正式请求（跳过 task 0 的 warmup）。

| k | accept_rate（a/g） | mean len | per-prompt acc（prose / code / zh） | per-prompt mean len | gen tok/s |
| ---: | ---: | ---: | --- | --- | ---: |
| 1 | 0.9479（91/96） | 1.95 | 1.000 / 0.938 / 0.909 | 2.00 / 1.94 / 1.91 | 8.95 |
| **2** | **0.9098**（121/133） | **2.81** | 0.932 / 0.909 / 0.889 | 2.86 / 2.82 / 2.74 | **10.54** |
| 3 | 0.8590（134/156） | 3.54 | 0.882 / 0.882 / 0.815 | 3.65 / 3.65 / 3.32 | 9.80 |
| 4 | 0.7933（142/179） | 4.05 | 0.907 / 0.758 / 0.730 | 4.50 / 3.94 / 3.71 | 9.68 |
| 6 | 0.6773（149/220） | 4.84 | 0.750 / 0.628 / 0.662 | 5.25 / 4.77 / 4.50 | 8.54 |
| 8 | 0.5667（153/270） | 5.48 | 0.761 / 0.537 / 0.462 | 7.00 / 5.25 / 4.20 | 8.84 |

出处：`build-split/acc/official_clean_k{1,2,3,4,6}.log` 与 `official_k8.log`（13:31–13:36），
每行 3 条 `draft acceptance`（task 编号逐 k 不同，见日志；同一 k 的三条就是 prose / code / zh）。
`gen tok/s` 是三个 prompt 的 `gen_tps` 均值（`official_clean_k2.json` 里是 8.39 / 11.24 / 11.99
→ 10.54）。**这一列受同机负载影响**，只在同一时段内相对可比：空闲时整模 k=2 单 prompt 到
13.99–16.86 tok/s（这一对只在当天的测量记录里，`acc/*.json` 复算不到，按 §7 当未落盘看），
表里的 10.54 是有背景负载时的数。

**k 的甜点仍是 2。** k=2 的接受率还有 0.910（mean len 2.81，上限 3），k=3 起每多一个 draft
就多一份 verify 成本，而接受率掉到 0.859、k=4 的 mean len 4.05 只换来 0.79。**这个结论与模型
无关，换模型不必重扫 k**：官方与蒸馏两版都落在 k=2（蒸馏版 k=2 是 0.699 / 2.39，k=4 掉到
0.545，k=8 掉到 0.321，衰减更快，更不可能往大 k 走）。甜点由「每轮 draft 的边际成本 / 每轮
verify 的边际收益」的比值决定（CPU 上 verify 是 k+1 行的批前向，draft 是 k 次单行前向），
这个比值对模型不敏感；只有当**接受率水平整体移动**（例如将来换更好的 MTP 头）时才需要重扫。

### 11.3 官方 vs 蒸馏：差的是 hidden 分布的一致性

同 prompt、同代码、同机：

| k | 官方 acc / mean len | 蒸馏 acc / mean len | accept_rate 提升 | gen tok/s（官方 / 蒸馏） |
| ---: | --- | --- | ---: | --- |
| 2 | 0.910 / 2.81（121/133） | 0.699 / 2.39（109/156） | **+30%** | 10.54 / 9.17 |
| 4 | 0.793 / 4.05（142/179） | 0.545 / 3.18（128/235） | **+46%** | 9.68 / 9.29 |
| 8 | 0.567 / 5.48（153/270） | 0.321 / 3.40（132/411） | **+76%** | 8.84 / 6.11 |

出处：官方见 §11.2 的日志；蒸馏是 `build-split/acc/dist_k{2,4,8}.log`（13:21–13:22）。蒸馏的
per-prompt acc：k=2 是 0.596 / 0.692 / 0.830，k=4 是 0.435 / 0.538 / 0.708，
k=8 是 0.278 / 0.303 / 0.398。tok/s 两列都在 13:12–13:36 这一段交错跑出来，**相对可比、绝对值
仍受同机负载影响**：k=2 是 10.54 vs 9.17、k=8 是 8.84 vs 6.11，而 k=4 只差 4%（9.68 vs 9.29）
——那一档差距不成比例，**不要拿单档 tok/s 反推接受率**（k=4 官方 9.68 甚至比 k=3 的 9.80 低）。

per-position 接受率（k=8，`-lv 4` 日志里的 `acc per pos`，pos0..pos7）：

| prompt | 官方 | 蒸馏 |
| --- | --- | --- |
| prose | 0.889 0.889 0.889 0.889 0.778 0.667 0.667 0.333 | 0.762 0.476 0.333 0.190 0.095 0.048 0.048 0.048 |
| code | 0.917 0.750 0.667 0.583 0.417 0.333 0.333 0.250 | 0.789 0.632 0.421 0.158 0.105 0.053 0.053 0.053 |
| zh | 0.800 0.667 0.400 0.333 0.333 0.333 0.267 0.067 | 0.875 0.688 0.438 0.312 0.188 0.188 0.188 0.062 |

出处：`build-split/acc/official_pos_k8.log`、`dist_pos_k8.log`（两份都是 3 条正式请求的
`acc per pos` 行；task 0 的 warmup 行是 `1.000, 1.000, 0.000...`，别数进去）。

**读法：pos0 两边接近，pos1 起分叉，之后蒸馏掉得很快。** 逐位看：「pos2 之后 0.16–0.44」这个
概括对应 pos2 的 0.33–0.44 与 pos3 的 0.16–0.31；pos4 之后再降到 0.05–0.19。pos0 接近说明蒸馏
版并不是「整体变笨」——它靠局部特征还能猜中第一个草稿；差的是 **hidden state 分布的一致性**，
所以 draft 走得越远（k 越大）误差累积越多，官方与蒸馏的差距也越大：k=2 +30%、k=8 +76%。
**推论：蒸馏版的任何接受率都不能拿来做容量规划**，它不是「同一模型慢一点」，而是另一个分布。

### 11.4 链上接受率低 24%：是我们自己加的代码

本地双段链（`--split-layer-range 0:32 --split-tail-range 32:40`，官方模型，`--spec-type draft-mtp
--spec-draft-n-max 2`）与整模同 prompt、同机：

| 配置 | accept_rate（a/g） | mean len | per-prompt acc | gen tok/s |
| --- | ---: | ---: | --- | ---: |
| 整模 + MTP k=2 | 0.9098（121/133） | 2.81 | 0.932 / 0.909 / 0.889 | 15.54（空闲）/ 8.4–12.0（有负载） |
| 双段链 + MTP k=2 | **0.6879**（108/157） | 2.37 | 0.648 / 0.725 / 0.692 | 8.76 |

出处：整模见 §11.2；链上 `build-split/acc/chain_off_k2c.log`（13:29，tag `chain_official_k2`，
`chain_off_k2c.json` 的 `gen_tps` 是 6.75 / 9.40 / 10.12 → 8.76）。链上比整模低
**(0.9098 − 0.6879) / 0.9098 ≈ 24%**。**这是本地双段链（进程内 tail）的数；远端链走的是同一个
`!is_chain && !is_remote` 分支，所以它同样没有 prompt 历史，但远程链的接受率没有单独测**
（§9 的 +6% 是蒸馏版上的端到端收益，不是接受率）。整模那格 tok/s 的「15.54（空闲）」是当天一次
空闲时单轮测量的数，`acc/` 里没有可复算的 json（同批 `acc_off_k2.json` 的均值是 10.38），按
§7 的口径当「未落盘」看；有负载那一档 8.4–12.0 与 §11.2 的 10.54 同量级。

**数值正确性没有退化**：3/3 prompt、64 token 与整模**逐字节相同**（含 `reasoning_content`，
250 / 244 / 249 字符）。可比的是 `acc/official_clean_k2.json` 与 `acc/chain_off_k2c.json` 的
`text` 字段——三行两两相等、长度也相等。所以这不是「链上算错了」，只是「链上的草稿更差」。

**根因在 `common/speculative.cpp` 的两处，而且两处都是本仓加的**（`H:\llamacpp-c` HEAD
`c5bd787b`）：

1. `speculative::process()` 的 catch-up decode 条件是
   `if (!is_mem_shared && !is_chain && !is_remote)`（HEAD 里在 `L1796`）。上游 `e0f142d5d`
   只有 `if (!is_mem_shared)`——**`&& !is_chain && !is_remote` 是 `c5bd787b` 新加的**。
2. `speculative::draft()` 在 chain 下每轮清空草稿 KV：`if (is_chain && !is_mem_shared)
   llama_memory_clear(llama_get_memory(ctx_dft), true);`（`L1978`）。

`draft()` 里那段注释就是理由原文（`L1979–L1986`）：

> the draft never saw the prompt (process() skips the catch-up), so a round must not inherit
> rows from whatever the previous round / previous request left at the same positions -- that is
> another token's state, and it would be read silently. One clear per round costs nothing (the
> draft KV is k+1 rows per sequence) and is far cheaper than replaying the prompt into the draft.
> Measured acceptance on the 2B model stays in the 70-93% range v1 reported.

**这是一个自洽的循环依赖，不是漏判**：

```
不 catch-up（链上补历史要整段 prompt 额外过一遍 draft）
  ↓ 所以
draft 没有 prompt 的 KV
  ↓ 所以
必须每轮清空（否则继承到「另一个 token 的状态」，静默读脏）
  ↓ 所以
即使 catch-up 了也读不到
  ↓ 所以
不做 catch-up 是"对的"
```

在**当前这套草稿 KV 策略**下，catch-up 建出来的 prompt 历史没有任何读者，跳过它是正确且必要
的。代价是链上 MTP 的接受率被封在 0.688（k=2），而不是整模的 0.910。**这是架构边界，不是 bug。**

**注意注释最后那句**：「Measured acceptance on the 2B model stays in the 70-93%」。这个设计在
2B 上是可接受的（70–93%），到 35B 上掉到 0.688 —— **问题被小模型掩盖了**。qwen35moe 的 NextN
块带 attention（MTP 图里有 `cache_k_l8` / `FLASH_ATTN_EXT`，见 `build-split/shape_tail.log`
的 `mtp_tok_embd-8` / `mtp_h_input` / `mtp_shared_head_norm` 那几行），所以「草稿看不到 prompt
的 KV」这个差异是真实存在的，不是理论洁癖。

**如果要做 catch-up，代价与收益**（以下**全部是估算，没有实测**；代码未改）：

| 场景 | 净收益 |
| --- | --- |
| 短 prompt + 短输出（40 prompt token / 64 生成） | **≈ 0**（catch-up 0.5–1.5 s，省 1.2–1.8 s） |
| 长生成（512 token） | **省 10–14 s** |
| 长 prompt（2000 token） | **净负**（catch-up 本身 30–60 s） |

代价是每请求一次 N 行 batch 前向（N = prompt 长度，output head 248k vocab 是大头，∝ N）；收益
是 k=2 的接受率 0.69 → 可能 ~0.91（若与整模持平）。按 64-token 生成从 ~28 步降到 ~23 步算，
省 1.2–1.8 s，所以短 prompt 短输出这一档基本白做。

**改动要点（将来要做，不是现在）**：必须**同时**改两处——catch-up 的 `is_chain` 判断，以及
`draft()` 的 chain 分支不再清空草稿 KV（改成按 target 的 `seq_rm` 严格同步）；只改一处会让
「历史建了没人读」或「读到别的 token 的状态」二者之一重新出现。上游非 chain 路径的 batch 构造
（token × h shift-right-1）在链上可以直接复用（`ctx_tgt` 已是 tail ctx，tail 的
`embeddings_nextn` 非 masked 通道正是逐行 h）。远端链还要额外把 prompt 的 token id 送过去
（N × 4 B，head 侧有、tail 侧没有）。判断：**值得做，但应做成 opt-in 开关**（按 prompt 长度 /
生成长度启用），风险在 KV 位置对齐与 rs-seq checkpoint 回滚。**不建议现在默认打开。**

### 11.5 「代码回归」与「新模型验收」是两件事

这两件事必须分开记，否则后来人会以为 `verify.sh` 能验新模型。

| 入口 | 测的是 | 结果 | 日志 |
| --- | --- | --- | --- |
| `bash build-split/verify.sh --skip-build` | **代码没退化**（2B，24 层） | **24/24 PASS，exit 0** | `build-split/verify_newmodel.log` |
| `bash build-split/verify_mtp.sh` | 链上 MTP 机制（2B，tail `20:24`） | **6/6 PASS，exit 0** | `build-split/verify_mtp_newmodel.log` |
| 本地双段链 vs 整模（35B 官方） | **形状对得上 + 数值正确** | **3/3 逐字节相同** | §11.4 的两份 `.json` |
| 形状复位验（split-stage tail `32:40`、整模 `--spec-type draft-mtp`） | 新模型的形状 | 通过 | `build-split/shape_tail.log`、`acc/official_k2.log` |

形状复位验的具体输出：tail 段报 `layers [32, 40) of 40, role=tail, n_embd=2048, n_rs_seq=8`、
`MTP draft context: 1 NextN block(s), n_ctx=4096, own KV, n_outputs_max=1; OP_MTP_DRAFT enabled`、
`identity: ... has_nextn=1`（`build-split/shape_tail.log`）；整模 `--spec-type draft-mtp` 加载
**17.63 s**、日志有 `creating MTP draft context against the target model`
（`acc/official_k2.log`）。**与 `accept_baseline.md` 的第二处出入**：那份记 17.9 s，是同一件事
的两次计时口径差，这里取日志。

`verify.sh` 逐组：组 1 本地链逐位 5/5、组 2 N=1 直通 1/1、组 3 N=2 经 server 1/1、组 4 跨进程
1/1、组 5 多 slot 2/2、组 6 三段跨进程 8/8、组 7 远程链经 server 2/2、组 8 链上 MTP 逐位 2/2、
组 9 DFlash 8 路 tap 2/2。`verify_mtp.sh`：tail 报 `has_nextn=1`、链上 5/5 行 logits 逐位、
wire 上 `h_nextn` 逐位、3 个 draft 逐个相同（11751 / 13 / 271）、tail 日志确认跑过
`OP_MTP_DRAFT`、协议版本一致。

**为什么这一步必须用 2B**：`verify.sh` 组 1 的断言按 24 层模型写死（`0:20+20:24`、
`0:8+8:16+16:24`、hidden 2048）；`verify_mtp.sh` 要求 tail `20:24` **恰好等于模型末层**，而
`--role tail` 的硬性校验是 `layer_hi == trunk depth`。对 40 层的 35B 新模型，`20:24` 只是 relay
段，**这两条结构上都不成立**。所以 2B 跑的是「代码没退化」，不是「新模型能用」。新模型的分段 /
MTP 验收只能走「本地双段链 `0:32+32:40` vs 整模、逐字节一致」这条路。

**闭环结论**：换模型不影响分段链与 MTP 的**数值正确性**（2B 24/24 + 6/6 回归；35B 链 vs 整模
逐字节一致）；影响只在**接受率**——整模上看是 trunk 漂移（0.910 vs 蒸馏 0.699），链上看是
草稿 KV 的差异（0.910 → 0.688），后者是设计选择（§11.4）。

### 11.6 复现脚本与日志

- `build-split/acc_test.py --port <p> --out <json>`：发 3 个 prompt（prose / code / zh），存 timings + 全文。
- `build-split/ksweep.sh <tag> <model> <k1,k2,...> <port>`：串行重启 server、逐 k 测、抓 `draft acceptance`。
  per-position 要额外给它 `LV="-lv 4"` 环境变量（脚本里是 `${LV:-}`）。
- `build-split/parse_pos.py <log>…`：从 `-lv 4` 的日志抓 per-position 接受率（默认扫 `build-split/acc/*.log`）。
- 日志全在 `build-split/acc/`：官方 k 扫描 `official_clean_k*.log`、蒸馏 `dist_k*.log`、
  per-position `{official,dist}_pos_k8.log`、链上 `chain_off_k2{,c}.log`。
- 验收/回归日志：`build-split/verify_newmodel.log`（24/24）、`verify_mtp_newmodel.log`（6/6）。

### 11.7 这一节之后，旧数字归谁

| 旧数字 | 在哪 | 属于谁 |
| --- | --- | --- |
| MTP 1.13–1.25×、接受率 70%（v2 链） | [`SPECULATION.md`](SPECULATION.md) §4 / §10 / §11 | 蒸馏版 + v2 的 Python 驱动链，**不是**本节的 v2.1 链 |
| DFlash 1.5×、接受 42–95% | [`SPECULATION.md`](SPECULATION.md) §5 | 旧模型（且是 `llama-speculative-simple` 的 40 层 target 口径） |
| DFlash k=3 的 1.12×、MTP k=2 的 1.01× | 本文 §3 | 蒸馏版（`.qlh_tmp/casc_probe.sh` 的 `TGT` 写的就是蒸馏版），换模型未重测 |
| 远程链 MTP k=2 中位 +6% | 本文 §9 | 蒸馏版（ROG 那轮在官方模型到位之前）。官方版接受率高 30–76%，收益预计更高，**但未重测** |
| 本文 §2 / §10 的 2B 数字 | 本文 | 2B，与换模型无关 |

**不要在两套数字之间做换算。** 能做的只有「同一模型、同一口径的 A/B」；跨模型只能比接受率
本身，且要说清哪个模型。
