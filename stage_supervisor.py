"""Stage supervisor: keep this machine's stages running, restarting only the one that dies.

层段分布式推理的部署形态常驻入口。读 topology，起本机段，然后一直盯着：哪一段退出就只
重启那一段，退避 2/4/8/16/30 s 封顶，稳定运行一段时间后退避归零。

例外是端口：端口被别的进程占着时 spawn 出来的 worker 只会在加载完模型之后 bind 失败退出
（`stage_worker.main` 先构造 StageEngine 再 serve），重生得越勤越浪费。所以每次重启前先探
一次端口，被占就只等不 spawn，约 60 s 仍不释放则放弃该段。

为什么不是 chain_client 的一个 `--keep-alive` 开关：三个入口各有一份 argparse，加参数
必须三处同步，历史上漏同步造成过静默行为分叉（AGENTS.md §7）。这个形态要的东西
`SA_TOPOLOGY` / `SA_CTX` / `SA_SHIM` 全都表达得了，所以没有第四份 argparse：

    SA_TOPOLOGY=topology-local-32-8.toml python stage_supervisor.py

为什么不是外层 `while true; do chain_client.py; done`：崩溃是按段发生的，外层只看得到
"主进程退了"，重启会把健康的段一起拉起来（每段几 GiB 到十几 GiB），一次单段崩溃变成
全链路停顿。这里要的是"谁死重启谁"。

为什么不管 remote 段：每台机器跑自己的 supervisor。本脚本只对 `stage.is_local` 的段有
进程句柄；remote 段没有可 poll 的东西，硬管只能靠端口探活——那是第二份就绪判据，真正的
判据是 HELLO。
"""
from __future__ import annotations

import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chain_client  # noqa: E402
import config  # noqa: E402
import topology  # noqa: E402

#: 退避台阶（秒），第 n 次连续失败等 `_backoff(n)`。2 s 起跳是因为"刚起就退出"这类失败
#: （显存不足、端口被占、模型路径错）值得快速重试；30 s 封顶是因为再长运维就看不见了——
#: 常驻进程的日志是唯一察觉通道，超过半分钟的静默等于死掉。
BACKOFF_S = (2, 4, 8, 16, 30)
#: 轮询间隔。不用 `proc.wait()` 阻塞等：要同时盯 N 段，且 wait 会吞掉 SIGTERM 后的收尾
#: 时机。0.2 s 的检测延迟相对 2 s 起跳的退避可以忽略。
POLL_S = 0.2
#: 连续失败计数归零所需的存活时长。为什么不是"spawn 成功就归零"：模型路径错或显存不足时
#: worker 起来一两秒就退出，一归零退避永远停在 2 s，日志刷屏，也看不出是"反复崩"还是
#: "偶发一次"。这个窗口要覆盖一次正常的模型加载（35B 一段几十秒）。
STABLE_S = 60.0
#: 端口被占时，隔多久再探一次。为什么不是把退避台阶接着用（2/4/8/16/30）：这里等的不是
#: "这个 worker 多久后重试"，而是"占着端口的那个进程多久后退"。上一轮的孤儿 worker 收尾
#: 通常在几秒内，5 s 足够；再长会让端口一旦释放就白等半分钟。
PORT_WAIT_S = 5.0
#: 端口被占的连续探测次数上限。12 × 5 s ≈ 60 s：超过这个窗口还没释放，说明占端口的是别的
#: 服务或者卡死的进程，继续等下去只是让这一段挂在那里假装在自愈。到这个数就放弃这一段。
PORT_GIVE_UP = 12
#: 本机端口探测的超时（秒）。`_port_open` 默认 1.5 s 是给跨机段留网络 RTT 的；本机段走
#: 内核，被占时 connect 在毫秒级返回（实测 2026-11-03：占着 9401 时 0.3 ms 成功）。而这台
#: Windows 上从未监听过的端口**不立刻回 RST**，要拖到超时才回 WSAECONNREFUSED（实测约
#: 2 s，疑似安全软件丢包），用默认值等于每次重启白等 1.5 s，"2 s 重启"就变成 3.5 s。
#: 0.3 s 足以把两种情形分开；判错的方向也只是"超时→当作空闲→照常 spawn"，即退回旧行为。
PORT_PROBE_S = 0.3


@dataclass
class _Run:
    """一段的运行时状态。``proc is None`` 表示"等 ``due`` 到点重生"。"""

    stage: topology.Stage
    proc: "subprocess.Popen | None" = None
    log: Path | None = None
    #: 连续失败次数，决定退避台阶；只有稳定存活 STABLE_S 才清零。
    failures: int = 0
    #: 这一次 spawn 的时刻，用来判"是否已经稳定"。
    started: float = 0.0
    #: 什么时候允许重启（monotonic）。None 表示没有在等退避。
    due: float | None = None
    #: 端口被占的连续探测次数，到 PORT_GIVE_UP 就放弃这一段。端口空着却仍在崩说明不是
    #: 端口问题，这个计数归零——否则一次显存不足的崩溃会替后面的端口问题预支配额。
    blocked: int = 0
    #: 已放弃：端口在窗口内始终没释放，不再 spawn、也不再探测。只影响这一段。
    given_up: bool = False


def _say(message: str) -> None:
    """带时间戳的一条 supervisor 日志。

    时间戳是验收依据：kill 在什么时候、重启在什么时候、退避是不是真按 2 s 的台阶走，
    只能靠时间线核对。flush 是因为常驻进程的输出常年被重定向到文件或 tee，块缓冲会让
    日志停在缓冲区里，看起来像卡死。
    """
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _backoff(failures: int) -> int:
    return BACKOFF_S[min(max(failures, 1), len(BACKOFF_S)) - 1]


def _code_text(code: int) -> str:
    """退出码附十六进制。被 TerminateProcess 杀掉时 Windows 给的 returncode 是
    0xFFFFFFFF，Python 原样报成 4294967295——既不像退出码也不像信号，运维看不出这是
    "被杀"。负数是 Python 的 -SIGnnn 约定，照原样留着。"""
    if 0 <= code <= 255:
        return str(code)
    return f"{code} (0x{code & 0xFFFFFFFF:08x})"


def _start(run: _Run, shim: Path, ctx: int) -> None:
    """起一段并登记为"活着"；异常往上抛，由调用方决定退避怎么走。"""
    proc, log = chain_client.spawn_worker(run.stage, shim, ctx)
    run.proc, run.log, run.started = proc, log, time.monotonic()
    run.due = None


def _keep_previous_log(run: _Run) -> None:
    """把上一份段日志留成 ``*.prev.log``，再让 spawn_worker 覆盖原文件。

    崩掉的原因几乎只在段日志的最后几十行里（加载失败、OOM、断言）。`spawn_worker` 用
    ``"w"`` 打开日志，重启会把它清空——不先留一份，"为什么崩"就在重启的瞬间没了，而
    supervisor 那行只记得退出码。只留一份、不做轮转：堆满日志比丢一份旧诊断更麻烦。
    """
    if run.log is None or not run.log.is_file():
        return
    try:
        shutil.copyfile(run.log, run.log.with_name(run.log.name + ".prev"))
    except OSError:
        pass  # 留不下来不该拦住重启


def _probe_host(stage: topology.Stage) -> str:
    """探测本机段端口时连哪个地址。

    `stage.host` 是 **bind** 地址：`0.0.0.0` / `::` 的意思是"绑所有网卡"，拿它去 connect
    不可靠（连不连得上取决于协议栈对方言的解释），所以换回环地址。`::` 也给 `127.0.0.1`
    是因为复用的 `chain_client._port_open` 只建 AF_INET 套接字；本仓库的部署（termux、
    ROG、各份验证拓扑）都是 IPv4，为 IPv6 再写一份探测属于重复造轮子。`localhost` 同样
    给定成 `127.0.0.1`：`getaddrinfo` 可能先返回 `::1`，而 worker 绑的是 IPv4 那一个。
    """
    return "127.0.0.1" if stage.host in ("0.0.0.0", "::", "localhost") else stage.host


def _port_held(stage: topology.Stage) -> bool:
    """这个段要用的端口是否已被别的进程占住（即现在 spawn 一个 worker 也绑不上）。

    复用 `chain_client._port_open`：它已经是"有没有人在 host:port 上 accept"的唯一实现。
    chain_client 不在本次改动范围内，所以和本文件既有的 `chain_client._terminate` 一样在
    导入侧使用私有名。

    **这不是就绪判据。** 段是否可用只由 HELLO 判定（`StageClient.connect` 开头那段身份
    校验），见本文件开头"为什么不管 remote 段"。这里的探测只在两个时刻做——初次起段前、
    以及每次崩溃后准备重生前——回答的是"这次 spawn 有没有意义"；它不参与健康检查，也不要
    拿去当"段还活着"的证据：端口上有人在 accept，只说明有进程在听，说明不了它加载的是
    这份拓扑、分层区间对不对——那些由 HELLO 逐项校验。

    探测会留下连接：对端不 close 时本端停在 FIN_WAIT_2，放弃前最多 `PORT_GIVE_UP` 条，
    `netstat` 里看到 supervisor 的 pid 挂在这几个端口上是正常的。
    """
    return chain_client._port_open(_probe_host(stage), stage.port, timeout=PORT_PROBE_S)


def _hold_off(run: _Run) -> None:
    """端口还被别的进程占着：不 spawn，只把下一次检查推后；到上限则放弃这一段。

    单独一处的理由：初次起段（`main`）和崩溃后重生（`_restart`）面对的是同一个判断和
    同一套计数，日志措辞也只能有一份——两处各写一遍就是 AGENTS.md §7 里的规则重复。
    """
    run.blocked += 1
    if run.blocked >= PORT_GIVE_UP:
        run.given_up = True
        run.due = None
        _say(f"stage {run.stage.index} ({run.stage.kind}) giving up: port "
             f"{run.stage.port} was still in use after {run.blocked} checks "
             f"(~{run.blocked * PORT_WAIT_S:.0f}s); it will not be restarted again. "
             f"Find what holds the port, then restart this supervisor "
             f"(other stages keep running)")
        return
    run.due = time.monotonic() + PORT_WAIT_S
    _say(f"stage {run.stage.index} ({run.stage.kind}) port {run.stage.port} is held by "
         f"another process (probably a worker from an earlier run that has not exited); "
         f"waiting for it to be released, next check in {PORT_WAIT_S:.0f}s "
         f"({run.blocked}/{PORT_GIVE_UP})")


def _restart(run: _Run, shim: Path, ctx: int) -> None:
    """到点重生一段；端口仍被占就先不 spawn。"""
    if _port_held(run.stage):
        # 为什么宁可不 spawn：`stage_worker.main` 是先构造 StageEngine（把整个模型读进
        # 内存）再在 `serve()` 里 bind 的，所以每一次"重生"都要先付一次模型加载，然后才在
        # bind 上失败退出。35B 一段几十秒，按 30 s 封顶退避重启就是"每半分钟白加载一次"，
        # 永远不会收敛。等端口释放比反复加载便宜得多。
        #
        # Windows 上还有第二种形态：新 worker 的 SO_REUSEADDR 能"成功"绑上别人已经占着
        # 的 127.0.0.1 端口（实测 2026-11-03：占位进程绑 0.0.0.0:9401 时，worker 绑
        # 127.0.0.1:9401 返回成功，两个 listener 并存），连接落到谁身上不确定。那比 bind
        # 失败更难查，同样属于"不该 spawn"。
        _hold_off(run)
        return
    # 端口空着却还是崩：不是端口问题。计数归零，退回原来的退避台阶。
    run.blocked = 0
    _keep_previous_log(run)
    try:
        _start(run, shim, ctx)
    except Exception as exc:  # noqa: BLE001 - 常驻进程不能因为一次拉起失败而死
        # spawn 自己失败（shim 被换掉、解释器没了、磁盘满）也走退避：supervisor 挂掉等于
        # 这台机器的段全部无人监管，比晚几秒重生严重得多。
        delay = _backoff(run.failures)
        run.due = time.monotonic() + delay
        _say(f"stage {run.stage.index} ({run.stage.kind}) could not restart: {exc}; "
             f"next try in {delay}s")
        return
    _say(f"stage {run.stage.index} ({run.stage.kind}) restarted, pid {run.proc.pid}")


def _supervise(runs: list[_Run], shim: Path, ctx: int, stop: threading.Event) -> None:
    """主循环：每 POLL_S 看一遍所有段，死的进退避，到点的重生。"""
    while not stop.is_set():
        now = time.monotonic()
        for run in runs:
            if run.proc is not None:
                code = run.proc.poll()
                if code is None:
                    if run.failures and now - run.started >= STABLE_S:
                        _say(f"stage {run.stage.index} ({run.stage.kind}) up for "
                             f"{STABLE_S:.0f}s, backoff reset")
                        run.failures = 0
                    continue
                # 只把这一段标成待重启，其余段继续跑。整批重启会让一次崩溃变成全链路的
                # 停顿，而单段崩溃才是常态（显存、端口、OOM）。
                run.proc = None
                run.failures += 1
                delay = _backoff(run.failures)
                run.due = now + delay
                _say(f"stage {run.stage.index} ({run.stage.kind}) exited "
                     f"{_code_text(code)}, restart in {delay}s (log: {run.log})")
                continue
            if run.given_up:
                # 放弃的段不再探测也不再 spawn，其余段照常。上面那行 "giving up" 是唯一
                # 线索，所以它把端口号和恢复办法都写出来了。
                continue
            if run.due is None or now >= run.due:
                _restart(run, shim, ctx)
        stop.wait(POLL_S)


def _shutdown(runs: list[_Run], procs: list | None = None) -> None:
    """SIGTERM/SIGINT 的收尾：terminate 全部、等它们退出、再退出自己。

    复用 `chain_client._terminate` 而不是另写一份：它已经处理了"worker 自己还有子进程"的
    情况（子进程可能只通过 stdin EOF 得知父进程死了，不会跟着 terminate 立刻退出），并且
    保证一个段收尾失败不会跳过其余段。``procs`` 只用于 `bring_up` 中途抛异常、``runs``
    还没建起来的路径。
    """
    live = [(run.proc, run.log, run.stage) for run in runs if run.proc is not None]
    if not live:
        live = list(procs or [])
    if not live:
        _say("stopped")
        return
    _say(f"stopping {len(live)} stage(s)")
    chain_client._terminate(live)
    _say("stopped")


def main() -> int:
    try:
        path = config.topology_path()
        stages = topology.load(path)
        shim = config.find_shim()
        ctx = config.ctx()
    except (config.ConfigError, topology.TopologyError) as exc:
        print(f"stage_supervisor: {exc}", file=sys.stderr)
        return 2

    local = topology.local_stages(stages)
    if not local:
        # 明确失败，不是"起来了但什么都不管"。常驻进程静默成功是最坏的一种：人看到进程
        # 还在，以为集群在跑。把每段在哪儿打出来，人才判断得出是拓扑选错了还是本机地址
        # 写错了（remote 主机会被 `is_loopback` 判为不是本机）。
        where = ", ".join(f"{s.kind} at {s.host}:{s.port}" for s in stages)
        print(f"stage_supervisor: {path} has no stage on this machine ({where}); "
              f"nothing would be supervised", file=sys.stderr)
        return 1

    # 整模校验放在起段之前：起完才发现两段不是同一个文件，已经白加载了一次 20+ GiB。
    # 只在拓扑声明了 sha256 时做（现有 8 份拓扑都没声明），否则这里要多花约 30 秒。
    bad = topology.verify_model_hashes(local, progress=_say)
    if bad:
        for line in bad:
            print(f"stage_supervisor: {line}", file=sys.stderr)
        return 3

    stop = threading.Event()

    def _on_signal(signum, _frame) -> None:
        # handler 里只置标志，不在里面 terminate：handler 可能正好打断一次 spawn，那会儿
        # 进程表处于半更新状态。收尾放到主循环退出之后统一做。
        _say(f"signal {signum}, shutting down")
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _on_signal)

    procs: list = []
    runs: list[_Run] = []
    try:
        # 只把**本机段**交给 bring_up。传整份拓扑的话，A 机的 supervisor 会按 start_cmd
        # ssh 去拉 B 机的段，而 B 机自己的 supervisor 已经占着那个端口——新 worker bind
        # 失败退出，然后被这里当成"崩了"反复重启。跨机拉起仍属于 CLI/TUI/server 的
        # bring_up；常驻形态下每台机器只对自己那几段有进程句柄。
        #
        # 初次起段也过同一道端口检查：worker 是"先把模型读进内存，再 bind"，端口被占着
        # 起段就是白加载一次（35B 一段几十秒）；而且 Windows 上 SO_REUSEADDR 还可能让它
        # "绑成功"、和占端口的进程各听一半连接（见 `_restart`）。被占的段不进 bring_up，
        # 登记成"等端口释放"，由 `_hold_off` 负责 5 s 一探、12 次后放弃。
        held = {s.index for s in local if _port_held(s)}
        chain_client.bring_up([s for s in local if s.index not in held], shim, ctx,
                              procs=procs, note=_say)
        by_index = {stage.index: (proc, log) for proc, log, stage in procs}
        # 先把"在监管"打出来再打各段的等待/重启：日志是按时间读的，一句 "supervising"
        # 夹在一段状态后面会让人以为那行段状态不属于这次启动。
        _say(f"supervising {len(local)} of {len(stages)} stage(s) from {path}")
        for stage in local:
            proc, log = by_index.get(stage.index, (None, None))
            run = _Run(stage=stage, proc=proc, log=log, started=time.monotonic())
            if stage.index in held:
                _hold_off(run)
            runs.append(run)
        _supervise(runs, shim, ctx, stop)
    finally:
        _shutdown(runs, procs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
