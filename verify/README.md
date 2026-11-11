# `verify/` —— 分段链的验收装置

一句话：拿到 `docs/llamacpp-split.patch` 和本仓库的人，用这里的脚本能自己复验文档里每一条
「分段链没坏」的结论。这些脚本原来只存在于开发机的构建目录（`H:\llamacpp-c\build-split\`，
被 `.gitignore` 的 `/build*` 排除），外人拿到 patch 无从核实；2026-11-10 起入库到这里。

被测对象是**上游 llama.cpp 的构建产物**，不是本仓库的 Python 驱动链（那是 v2，已归档在
`slimarc/prototype`）。所以这里所有脚本都只依赖 `split-stage.exe` / `llama-server.exe` /
整模 gguf / 协议 v9，没有一条依赖 `chain_client`、`topology.toml` 或 `shim/`。

## 1. 一条命令

```bash
QLH_BUILD_DIR=H:/llamacpp-c/build-split bash verify/verify.sh --skip-build
```

退出码 0 = **被测到的**组全过。`--skip-build` 表示用现成二进制；去掉它会先跑一次
`cmake --build`（见 §7）。全套含 35B 组（组 9/10/11），要独占内存；只想确认 2B 那八组没有
回归就加 `QLH_VERIFY_FAST=1`：

```bash
QLH_VERIFY_FAST=1 bash verify/verify.sh --skip-build    # 组 1-8，组 9/10 明确 SKIP
```

## 2. 环境约定

| 变量 | 缺省 | 作用 |
|---|---|---|
| `QLH_BUILD_DIR` | `H:/llamacpp-c/build-split` | 上游构建目录，脚本只读它下面的 `bin/` |
| `QLH_SRC_DIR` | `QLH_BUILD_DIR` 的父目录 | 上游源码根（stage/server 的 cwd、构建脚本的 cwd） |
| `QLH_TMP` | `<本仓库>/.qlh_tmp/verify` | 日志、临时工件、锁（本机 `/tmp` 不可靠） |
| `QLH_SPLIT_MODEL` | 2B 的 `Qwen3.5-2B Q4_K_M` | 组 1-8 与多数单项脚本的模型 |
| `QLH_DFLASH_MODEL` / `QLH_DFLASH_DRAFT` | 35B target + DFlash 草稿 | 组 9/10/11 |

`H:/x`、`H:\x`、`/h/x` 三种写法都收。缺二进制时脚本**点名退出 3**（不是静默 skip、也不是
记一堆 FAIL）：

```
错误：H:/nope/bin 下缺少可执行文件：split-smoke.exe split-stage.exe llama-server.exe
  QLH_BUILD_DIR 现在 = H:/nope（env）
  请把它指向 llama.cpp 的构建目录（其下有 bin/），例如：
    QLH_BUILD_DIR=H:/llamacpp-c/build-split bash verify/verify.sh --skip-build
```

端口：全部在 93xx（`verify.sh` 用 9331/9332/9341-9343，组 11 用 9336-9338，链死验收用
9330/9331/9335，其余见各脚本头部）。`verify_seq_rm_shared.py` 用 9487。**9103 / 9104 / 9106
上的服务一律不碰**。同一台机器同时只允许一个 `verify.sh`（`.verify.lock`，并发跑会互抢端口
并互相强杀）。清理按「镜像名 + `$BIN` 路径」识别，不看端口（`ours_procs.ps1`）。

**同机还有别人的实验时**：清理会把「可执行文件在 `$BIN` 下」的 `split-stage` / `split-smoke` /
`split-rows` / `dflash-taps` / `llama-server` 一并杀掉，不区分是谁起的。要么等对方跑完，要么
把 `bin/` 原样复制一份、让 `QLH_BUILD_DIR` 指向副本（110 个文件、35 MB，逐字节相同）：

```bash
cp -r H:/llamacpp-c/build-split/bin /tmp-work/bin-copy/
QLH_BUILD_DIR=/tmp-work QLH_SRC_DIR=H:/llamacpp-c bash verify/verify.sh --skip-build
```

`verify/ours_procs.ps1` 必须是 **UTF-8 带 BOM**：无 BOM 时 Windows PowerShell 5.1 按 ANSI 解码
中文注释，注释乱码会把 `$prefix` 那条赋值吞掉，`$prefix` 变成空串、`StartsWith('')` 恒真——
表现是「清理正常」，实际杀的是整机上所有同名进程（含别人的）。入库版本已加 BOM，并加了一条
「prefix 为空就报错退出 3」的防线；改这个文件时别把 BOM 存掉。

## 3. 脚本清单

判据数取自各脚本自己的 `PASS=`/`FAIL=` 合计，实测数字见 `docs/measure/SPLIT-MEASUREMENTS.md`。
耗时是 2B、本机、无并发实验时的量级；35B 的组另算。

| 脚本 | 测什么 | 需要 | 判据 | 耗时 |
|---|---|---|---|---|
| `verify.sh` | 十一个组的总入口：本地链逐位、N=1 直通、经 server、跨进程 head+tail、多 slot、三段 relay、远程链经 server、MTP 投机、DFlash tap、DFlash/DSpark 远程 tap、DFlash 投机默认路径 | 五个目标全在 `bin/`；2B（组 1-8）+ 35B（组 9-11） | 52 | 组 1-8 实测 10-12 分钟（另有一个 2B 实验并行时也不慢于此）；加组 9-11 约 30-50 分钟 |
| `verify_chain_death.py` | 远端 stage 消失：head 活着、失败是请求级的、链死后快速失败、只重启 stage 能自愈、deep health 三态、重连节流、`-np 2 + --split-remote` 被拒 | `split-stage` `llama-server`，2B | 36 | 4-5 分钟（实测 4 分钟，含两次 120 s 读超时） |
| `verify_seq_id.py` | 越界 `seq_id` 回 `ST_BAD_REQ` 且连接可继续用；讲协议 v9 的帧 | `split-stage`，2B | 9 | 约 1 分钟 |
| `verify_seq_rm_exact.py` | `seq_rm(p,-1)` 后重算第 p 行必须逐位相同（不 split、不起服务） | `bin/seq_rm_exact.exe`（缺则用 `probes/` 现场编），2B | 6 个用例 | 约 2 分钟 |
| `verify_multislot.py` | hybrid + 分段链 + slot 复用：长度交替的十请求全 200、进程活着、prefix cache 仍生效 | `llama-server`，2B | 3 条硬判据 | 约 2 分钟/次 |
| `verify_seq_rm_shared.py` | shared-prefix（`n_cmpl=2`）+ 强制 `seq_rm` 失败时，应答隔离（两次输出逐字节相同） | `llama-server`，2B | 3 | 约 1 分钟 |
| `verify_state_stale.py` | v6 快照句柄过期：stage 回 `ST_STALE` 时请求仍 200、退化成重算、结果不变 | `split-stage` `llama-server`，2B | 7 | 约 3 分钟 |
| `verify_mtp.sh` | 远程 tail（`20:24`）的 MTP draft 与「整模 + MTP」逐 token 相同 | `split-rows` `split-stage`，2B（需 NextN 层） | 6 | 约 2 分钟 |
| `verify_mtp_e2e.sh` | 远程链 MTP 投机（temp=0）与整模 MTP 逐字节相同；附三组速度对照 | `llama-server` `split-stage`，2B | 6 | 约 4 分钟 |
| `verify_grammar_spec.sh` | grammar 请求真在投机、分段不改变结果、被拒的行能回退、走逐行通道 | `llama-server` `split-stage` + `grm/toolcall_open.gbnf`，2B | 23 | 约 6 分钟（8 次服务启动） |
| `verify_rbudget_spec.sh` | 真实 tools 请求不被 reasoning-budget 闸门挡掉；预算耗尽后强制段正确、回退不多扣 | `llama-server` `split-stage`，2B（thinking 模板） | 36 | 约 8 分钟（12 次服务启动） |
| `verify_dflash_dev_path.sh` | DFlash 投机的**产品路径**：整模仍走 device buffer，链回退 host copies 且输出逐字节相同 | 35B target + DFlash 草稿 | 8 | 约 6 分钟（35B 加载慢） |
| `measure_state_bytes.py` | 量每请求 tail 状态回传字节数（穿过计数代理，分 up/dn） | `split-stage` `llama-server`，2B | 度量，不计分 | 约 2 分钟 |
| `acc_test.py` | 三个固定 prompt 的接受率 / tok/s 基线，写 JSON | 已在跑的服务（`--port`） | 度量 | 请求时长 |
| `compare_out.py` | 两份 `acc_test.py` JSON 逐 prompt 逐字节比对 | 两个 JSON | 度量 | 秒级 |
| `long_prefill.py` | 约 1100 token 的长 prefill 耗时 | 已在跑的服务（`--port`） | 度量 | 请求时长 |
| `parse_pos.py` | 从 `-lv 4` 日志抓 per-position 接受率 | 日志文件 | 度量 | 秒级 |

## 4. 只跑某一块

单项脚本都能独立跑（自己起 stage/server、自己清进程）：

```bash
python verify/verify_chain_death.py                      # 36 项
python verify/verify_seq_id.py --port 9337
python verify/verify_multislot.py --np 2 --port 9350 --ctx 8192   # verify.sh 组 5 的一半
bash   verify/verify_mtp.sh
bash   verify/verify_dflash_dev_path.sh                  # verify.sh 组 11 本体
```

`verify.sh` 没有 `--group`：要复跑某一组就直接跑上面那个单项脚本（组 1/4/6 例外，它们跑的是
`bin/split-smoke.exe`，`ONLY_SPLIT=1` 与 `QLH_SPLIT_REMOTE2=a,b` 是它的环境变量，见
`verify.sh` 组 4/6）。`QLH_VERIFY_FAST=1` 跳过 35B 的三组，并把 SKIP 显著打印出来（SKIP 不算
通过，`FAIL=0` 也不等于完整验收）。

## 5. 已知的部分失败（**不是坏了**）

- `verify_seq_rm_exact.py` 的 **B 组**（`n_rs_seq=3` 的三条）在当前基线是红的，A 组（FULL
  形态）也红。原因是上游 `ab3768cdc` 把「可回退量」的口径从 `rollback <= n_rs_seq` 改成
  `rollback <= cell.rs_max`，而该探针逐 token decode（`rs_max = 0`），所以任何部分回退都被
  拒。这是**已被接受的基线漂移**，用 `--expect-known-fail` 跑（1 pass / 5 known-fail / 0
  unexpected，退出 0）；不带这个开关时退出码非 0 是预期行为。产品形状的回退由
  `verify_grammar_spec.sh` 与 `verify_rbudget_spec.sh` 覆盖，那两组是全绿的。
- 缺 35B 模型时组 9/10/11 会记 FAIL（不是 SKIP）；内存低于 `QLH_VERIFY_MIN_FREE_GB`（默认
  20 GB）时组 9/10 记 SKIP 并在 `结果` 段显著提示。
- `verify.sh` 里 `run_limited` 的超时提示用 `tee -a`；本机 MSYS 没有 `tee`，超时路径会退化
  成打到 stderr、日志里没有那一行。判据不受影响。

## 6. 判据口径（容易误读的两条）

1. **逐位/逐字节是有范围的**：投机组的判据是「链 == 整模」，不是「投机 == 不投机」。后者在
   llama.cpp 里本来就不成立——k+1 行批量 forward 与逐行 forward 走不同 kernel，h 有 0.27
   量级的确定性差异（见 `verify_mtp_e2e.sh` 头部的实测）。把它当硬判据，测的是 prompt 有
   没有近似并列，不是链。
2. **别用 `python -X utf8` 跑这里的 Python**：`verify_chain_death.py` 的判据里包含中文日志
   片段的匹配，`-X utf8` 会改变 stderr/stdout 的编码行为，历史上有过误判。

## 7. 构建（`--skip-build` 之外的路径）

```bash
QLH_BUILD_DIR=H:/llamacpp-c/build-split bash verify/verify.sh            # 增量
QLH_BUILD_DIR=H:/llamacpp-c/build-split bash verify/verify.sh --clean    # 全量
```

构建入口按顺序找：`$QLH_SRC_DIR/.qlh_tmp/build.bat`（开发机原有的那份）→ `verify/build.bat`
（入库的通用版，读 `QLH_SRC_DIR` / `QLH_BUILD_DIR` / `QLH_VCVARS`）。两者都没有时**明确报错
退出 3**，不静默跳过。

上游树的 ninja 依赖记录不可靠（多数对象 `deps=0`），改过 `.h` 必须 `--clean-first`，否则增量
构建会产出 ABI 混用的 dll。`verify/build.bat` 与 `probes/build_seq_rm_exact.bat` 必须保持
CRLF 行尾：写成 LF 时 cmd 解析 vcvars 会失效，任何 `cl` 调用都报 C1083。

## 8. 拿到 patch 的人怎么准备

1. 取一份匹配的 llama.cpp 源码树，应用 `docs/llamacpp-split.patch`。
2. 用 CMake 构建到某个目录（要 `llama` / `split-stage` / `llama-server` / `split-smoke` /
   `split-rows` / `dflash-taps` 六个目标，见 `verify/build.bat` 的 `--target`）。
3. 取一个 24 层的 2B gguf（组 1 的断言按 `0:20+20:24`、hidden 2048 写死；别的模型会假失败）。
4. `QLH_BUILD_DIR=<构建目录> bash verify/verify.sh --skip-build`。
5. 想复验 35B 的三组，再准备 target / DFlash 草稿 / DSpark 草稿三个 gguf，并留够内存。

## 9. 这套装置还没有覆盖的

- **跨机**：所有脚本都是 `127.0.0.1`，没有一条真的跑两台机器。远程链的协议、断线重连、时钟
  与 MTU 的影响没有实测覆盖（`docs/deploy/CROSS-MACHINE.md` 的结论来自手工实测，不可重跑）。
- **Android / ARM**：`docs/deploy/ANDROID-SURFACE-段接入评估.md` 的结论没有任何脚本支撑，全靠人工。
- **并发验收**：没有「按端口/按 pid」的隔离；同一份 `bin/` 上跑别的实验时会被清理一起带走，
  只能靠 `QLH_BUILD_DIR` 指向 bin 副本（§2）绕开。日志也共用 `<仓库>/.qlh_tmp/verify/`，两个
  会话同时跑会互相覆盖。
- **回归阈值**：只有「逐位相同」这种硬判据，没有性能回归闸门（tok/s、接受率漂移只记录不比）。
