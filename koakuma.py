"""koakuma -- 一条命令从冷 shell 到能生成文本的链，四区 TUI。

项目名 K-Llama（层段分布式推理，目前不依赖 Slimarc 运行时）。启动器做完 TUI 出来之前
必须做完的事：找拓扑、找 shim、内存预检、拉起本机各段并连上。结果写进对象而不是 print()，
因为 Textual 接管终端后 print() 只会冲掉画面；启动画面上每一行因此都对应一件真发生过的
工作。TUI.md §7 的四个数字修正都在 Metrics：吞吐用相邻 token 间隔的滑窗（不是
first_token_at 到现在的跨度）、条形按当前最大段归一化（不是固定 50 ms）、TTFT 单列、
已用时间从第一个 token 起算。spawn_worker / _terminate / estimate_ram_bytes /
mem_budget_bytes 直接复用 chain_client，好让内存模型的修正到达每个入口。
"""
from __future__ import annotations

import argparse
import os
import re
import threading
import time
from collections import deque
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Input, Static

import config
import topology
from chain_client import (GPU_VRAM_OVERHEAD, VRAM_PER_BYTE, StageClient, _terminate,
                          estimate_ram_bytes, estimate_vram_bytes, mem_budget_bytes,
                          ram_ratio, spawn_worker, stage_bytes, vram_info)
from stream import generate_stream

HERE = Path(__file__).resolve().parent
GIB = 2 ** 30
SHIM_ENV, DEFAULT_SHIM = "SA_SHIM", Path("H:/slimarc-proto/shim/bin/qlh_keep_head.dll")
BAR, WINDOW, HISTORY = 12, 8, 16   # 条形格数 / 滑窗样本数 / 中位数样本数


def find_topology(explicit: str) -> Path:
    """--topology > SA_TOPOLOGY > topology.toml；都没有就列出候选，而不是空报错。"""
    wanted = explicit or os.environ.get(config.TOPOLOGY_ENV, "").strip() or config.DEFAULT_TOPOLOGY
    p = Path(wanted).expanduser()
    if p.is_file():
        return p
    if explicit:
        raise RuntimeError(f"--topology 指向的文件不存在：{p}")
    found = {c.resolve() for base in (Path.cwd(), HERE) for c in base.glob("topology*.toml")}
    names = "、".join(sorted(q.name for q in found)) or "（本目录没有 topology*.toml）"
    raise RuntimeError(f"找不到 {wanted}；可选：{names}。用 --topology 指定。")


def find_shim(explicit: str) -> Path:
    """--shim > SA_SHIM > 仓库内 shim/bin > 这台机器的默认安装位置。"""
    for raw in (explicit, os.environ.get(SHIM_ENV, ""), HERE / "shim/bin/qlh_keep_head.dll",
                DEFAULT_SHIM):
        if raw and Path(raw).expanduser().is_file():
            return Path(raw).expanduser()
    raise RuntimeError(f"找不到 keep-head shim：用 {SHIM_ENV} 或 --shim 指向 qlh_keep_head.dll"
                       f"（默认位置 {DEFAULT_SHIM} 也不存在）")


def precheck(stages) -> dict:
    """内存与显存账目。chain_client 的结论以 "Move a stage to another host" 结尾，既不说
    差多少也不说哪段最贵；这里两个都算出来，缺口就是启动画面直接打印的那个数。

    显存单独算，因为它的超额是静默的：WDDM 用系统内存兜住超出部分，llama.cpp 照样报
    offloaded N/N，只是每步慢到比纯 CPU 还差。见 chain_client.VRAM_PER_BYTE 的实测。"""
    need, budget = estimate_ram_bytes(stages), mem_budget_bytes()
    # Per-stage figures use the same weighted model as the total; computing them with the
    # old step function would make the rows disagree with the number they are explaining.
    source_total = sum(s.layers for s in stages)
    rows = [(s.index, s.kind, s.layers,
             int(stage_bytes(s, source_total) * ram_ratio(s)))
            for s in stages if s.is_local]
    out = {"need": need, "budget": budget, "short": max(0, need - budget), "rows": rows,
           "vram_need": 0, "vram_free": None, "vram_short": 0}
    if any(s.is_local and s.gpu_layers for s in stages):
        out["vram_need"] = estimate_vram_bytes(stages)
        info = vram_info()
        # A None here means nvidia-smi is unavailable, not that the card has no room;
        # leaving vram_free as None keeps the splash from claiming a check it never ran.
        if info is not None:
            out["vram_free"] = info[0] - info[1]
            out["vram_short"] = max(0, out["vram_need"] - out["vram_free"])
    return out


_TENSORS = re.compile(r"and (\d+) tensors from")


def log_state(log: Path | None) -> tuple[int, int]:
    """(create_tensor 行数, llama.cpp 声明的张量总数)。2B 两段实测 214/214 与 294/294 完全
    对上，所以进度是数出来的不是估的。轮询文件而不改 stage_worker 协议。"""
    try:
        text = log.read_text(encoding="utf-8", errors="replace") if log else ""
    except OSError:
        return 0, 0
    total = done = 0
    for line in text.splitlines():
        if not total and (found := _TENSORS.search(line)):
            total = int(found.group(1))
        elif total and line.startswith("create_tensor:"):
            done += 1
    return done, total


class Boot:
    """启动步骤与它拉起的进程：界面读它的状态，退出时它负责清理。"""

    def __init__(self, args) -> None:
        self.args, self.ctx, self.phase, self.error = args, 0, "", ""
        self.topology: Path | None = None
        self.shim: Path | None = None
        self.stages: tuple = ()
        self.mem, self.done, self.procs, self.clients = {}, set(), [], []
        self.status, self.errors, self._lock = {}, {}, threading.Lock()
        self.stopping = threading.Event()
        self.started_at = time.perf_counter()   # 启动画面显示真实已用时

    def run(self) -> None:
        """后台线程里跑完整准备流程；失败写进 self.error，不抛给界面。"""
        try:
            self.topology, self.ctx = find_topology(self.args.topology), config.ctx()
            self.stages = topology.load(self.topology)
            self.done.add("topology")
            self.shim = find_shim(self.args.shim)
            self.done.add("shim")
            self.mem = precheck(self.stages)
            self.done.add("memory")
            if self.mem["short"]:
                # 预检不过就不许起 worker：Windows 超内存是静默换页，会把整机拖死。
                # 返回后 status 仍是空的，画面不会出现任何 stage 行。
                return
            self.phase = "load"
            self.procs = [(*spawn_worker(s, self.shim, self.ctx), s)
                          for s in self.stages if s.is_local]
            self.clients = [StageClient(s) for s in self.stages]
            self.status = {s.index: ("loading" if s.is_local else "waiting") for s in self.stages}
            # 每段一个线程：段是并行加载的，连接也必须并行，否则等的是它们的和。
            procs = {st.index: proc for proc, _log, st in self.procs}
            for c in self.clients:
                threading.Thread(target=self._connect, daemon=True,
                                 args=(c, procs.get(c.stage.index))).start()
            while not self.error and not self.stopping.is_set():
                with self._lock:
                    states = dict(self.status)
                bad = [i for i, v in states.items() if v == "failed"]
                if bad:
                    self.error = self.errors.get(bad[0], f"stage {bad[0]} 连接失败")
                    return
                if all(v == "connected" for v in states.values()):
                    self.done.add("load")
                    return     # 全部连上就收工：线程必须结束，退出时才不会被 join 卡住
                time.sleep(0.1)
        except Exception as exc:  # noqa: BLE001 - 界面要显示它，不是崩掉
            self.error = f"{type(exc).__name__}: {exc}"

    def _connect(self, client: StageClient, proc) -> None:
        """连接一段。失败原因存起来给启动画面，不抛出（它跑在自己的线程里）。"""
        i = client.stage.index
        try:
            client.connect(proc=proc)
            with self._lock:
                self.status[i] = "connected"
        except Exception as exc:  # noqa: BLE001 - 原因要显示出来
            with self._lock:
                self.errors[i], self.status[i] = str(exc), "failed"

    def log_of(self, index: int) -> Path | None:
        """worker 日志路径。启动画面轮询它算真实进度，测试与调试也用它。"""
        return next((lg for _p, lg, st in self.procs if st.index == index), None)

    def shutdown(self) -> None:
        """关 socket 并收割 worker（含子进程）。幂等，由 finally 调用。

        _terminate 会沿每个 stage_worker 的子进程树清理，因为占着权重的是它的子进程而不是
        它自己。失败路径也必须走到这里，否则下一次启动会在上一轮的权重还驻留时做内存预检。"""
        self.stopping.set()   # 让仍在轮询的 Boot.run 立刻看到并退出
        for client in self.clients:
            try:
                client.close()
            except Exception:  # noqa: BLE001 - 关一个不该挡住关其它
                pass
        self.clients = []
        if self.procs:
            _terminate(self.procs)
            self.procs = []


class Metrics:
    """生成过程的实时数字。

    旧实现用 first_token_at 到此刻的跨度；分块灌入时第一批 token 几乎同时到达，跨度接近 0，
    于是 tok/s 被算成 6801.9。改用相邻 token 到达间隔的滑动窗口：单个异常间隔只污染窗口里
    的一个样本，窗口平均间隔的倒数就是稳态速率 —— 与「手工逐间隔求平均再取倒数」是同一个数。
    """

    def __init__(self, stages: int) -> None:
        self.t0 = self.first = self.last = None   # 请求发出 / 首 token / 上一 token
        self.gaps: deque[float] = deque(maxlen=WINDOW)
        self.tokens = 0
        self.hist: list[deque[float]] = [deque(maxlen=HISTORY) for _ in range(stages)]

    def add(self, chunk, now: float) -> None:
        if self.first is None:
            self.first = now          # TTFT 的终点，也是「已用时间」的零点
        elif self.last is not None:
            self.gaps.append(now - self.last)
        self.last, self.tokens = now, self.tokens + 1
        # stage_ms 由 stream 在 yield 之后填上一轮的值；长度不符说明还没有样本。
        if len(chunk.stage_ms) == len(self.hist):
            for h, ms in zip(self.hist, chunk.stage_ms):
                h.append(ms)

    @property
    def rate(self) -> float:
        """窗口内平均相邻间隔的倒数。单个异常间隔只污染一个样本。"""
        mean = sum(self.gaps) / len(self.gaps) if self.gaps else 0.0
        return 1.0 / mean if mean > 0 else 0.0

    def ttft_ms(self) -> float | None:
        """首 token 延迟，单独一个指标：它决定"感觉快不快"。"""
        return None if self.first is None or self.t0 is None else (self.first - self.t0) * 1000.0

    def elapsed(self, now: float) -> float:
        """从第一个 token 起算。旧版从请求发出起算，首个 token 前一直显示 0.0s。"""
        return 0.0 if self.first is None else now - self.first

    def stage_ms(self) -> list[float]:
        """每段取样本中位数：单次抖动不该让条形跳一下。"""
        return [0.0 if not h else sorted(h)[len(h) // 2] for h in self.hist]


class Ask(Input):
    """输入框：空的时候 q 交给 App 退出，有内容时 q 就是字母 q。

    不做成 App 的全局绑定：Input 会先拿到可打印字符，全局 q 就永远轮不到"用户想输入 q"
    的场景。priority=True 让这个绑定先于 Input 的插入逻辑。

    注意 App.action_quit 是 async 的：直接调用只会得到一个未 await 的协程（实测报
    "coroutine was never awaited"），所以这里走 app.exit()。
    """

    BINDINGS = [Binding("q", "quit_or_type", "退出", priority=True)]

    def action_quit_or_type(self) -> None:
        if self.value:
            self.insert_text_at_cursor("q")
        else:
            self.app.exit()


def splash_text(b: Boot) -> Text:
    """启动画面：每行一件真发生过的工作，进行中的行显示真实百分比与真实已用时。

    模块级函数而非 App 方法：它只依赖 Boot 的状态，独立出来更好读、也更好在测试里断言。
    """
    t = Text("  ▄▄▄  ", style="bold #4d8de0")

    def step(label: str, ok: bool, detail: str, bad: bool = False) -> None:
        mark, colour = ("✓", "#59c26a") if ok else ("✗", "#e3596b") if bad else ("·", "#7b8194")
        t.append(f"  {mark} ", style=f"bold {colour}")
        t.append(f"{label:<9}{detail}\n" if detail else f"{label:<9}\n",
                 style="#9aa3b8" if ok else "#7b8194")

    t.append("K-Llama\n", style="bold #eef0f8")
    t.append(" ▐███▌ ", style="bold #4d8de0")
    t.append("层段分布式推理 · 四区 TUI\n\n", style="#9aa3b8")
    shape = f"{len(b.stages)} 段 / {sum(s.layers for s in b.stages)} 层" if b.stages else ""
    bad = bool(b.error)
    step("topology", "topology" in b.done,
         f"{b.topology.name if b.topology else ''}  {shape}", bad and not b.topology)
    step("shim", "shim" in b.done, str(b.shim) if b.shim else "", bad and not b.shim)
    m = b.mem
    if not m:
        step("memory", False, "")
    elif not m["short"]:
        step("memory", True,
             f"预计 {m['need'] / GIB:.2f} / 预算 {m['budget'] / GIB:.2f} GiB（可用 ×0.75）")
    else:
        step("memory", False, f"差 {m['short'] / GIB:.2f} GiB（需 {m['need'] / GIB:.2f} / "
                              f"预算 {m['budget'] / GIB:.2f} GiB）", bad=True)
        # 指出哪一段最贵：只说「内存不够」没法让人动手。预检没过就没起任何 worker，
        # 因此后面不再画 stage 行，也不该提示"中止加载"。
        w = max(m["rows"], key=lambda r: r[3], default=(0, "", 0, 0))
        t.append(f"     最贵的是 stage {w[0]}（{w[1]} {w[2]}L，约 {w[3] / GIB:.2f} GiB）："
                 f"把它的 host 改成远端 IP，或减少本机段数\n", style="#e3a04a")
        t.append("\n  没有启动任何 worker。按 q 退出。\n", style="#7b8194")
        return t

    # Video memory gets its own row: an overflow there does not fail, it spills, and the
    # run merely crawls. Show it only when some local stage actually offloads.
    if m["vram_need"]:
        if m["vram_free"] is None:
            step("vram", False, "nvidia-smi 不可用，无法预算（溢出不报错）", bad=True)
        elif not m["vram_short"]:
            step("vram", True, f"预计 {m['vram_need'] / GIB:.2f} / 空闲 "
                              f"{m['vram_free'] / GIB:.2f} GiB")
        else:
            step("vram", False, f"差 {m['vram_short'] / GIB:.2f} GiB（需 "
                               f"{m['vram_need'] / GIB:.2f} / 空闲 "
                               f"{m['vram_free'] / GIB:.2f} GiB）", bad=True)
            gpu = max(((s, s.model.stat().st_size * VRAM_PER_BYTE + GPU_VRAM_OVERHEAD)
                       for s in b.stages if s.is_local and s.gpu_layers),
                      key=lambda p: p[1], default=None)
            if gpu:
                t.append(f"     最贵的是 stage {gpu[0].index}（{gpu[0].model.name}，约 "
                         f"{gpu[1] / GIB:.2f} GiB）：把它的 gpu_layers 降到 0 走 CPU。\n"
                         f"     Windows 不会拒绝这次分配，它会溢出到系统内存，"
                         f"然后比纯 CPU 还慢。\n", style="#e3a04a")
            t.append("\n  没有启动任何 worker。按 q 退出。\n", style="#7b8194")
            return t

    for stage in b.stages:
        i, status = stage.index, b.status.get(stage.index, "waiting")
        local = "local" if stage.is_local else stage.host
        t.append(f"  stage {i} {stage.kind:<4} {stage.layers:>2}L  ", style="#cfd2dc")
        if status == "connected":
            t.append(f"  ✓  已就绪  {local}\n", style="#59c26a")
            continue
        if status == "failed":
            t.append(f"  ✗  {b.errors.get(i, '连接失败')[:80]}\n", style="#e3596b")
            continue
        if not stage.is_local:
            t.append(f"  连接 {local}:{stage.port} …\n", style="#7b8194")
            continue
        done, total = log_state(b.log_of(i))
        filled = int((done / total if total else 0.0) * BAR)
        # Text.append 不解析 [...]，喂 markup 会把颜色标记原样打到屏幕上，条形分两段拼。
        t.append("█" * filled, style="#c9a227")
        t.append("░" * (BAR - filled), style="#3a3f4d")
        pct = f"{done / total * 100:5.1f}%" if total else "  —  "
        note = f"张量 {done}/{total}" if total else "启动解释器 / 加载后端…"
        # 已用时是真实数据，且比百分比更抗缓冲：llama.cpp 的 stdout 是块缓冲，几百行
        # create_tensor 成片落盘（实测 214 行在一次 8 KB 写里），百分比会跳变，秒数不会。
        t.append(f"  {pct}  {note}  {time.perf_counter() - b.started_at:5.1f}s\n", style="#9aa3b8")
    if bad:
        t.append(f"\n  ✗ {b.error}\n  按 q 退出（会先清理所有 worker）\n", style="#e3596b")
    elif "load" not in b.done:
        t.append("\n  按 q 中止（会先清理已启动的 worker）\n", style="#7b8194")
    return t


class KLlamaApp(App):
    CSS = """
    Screen { background: #14161c; color: #cfd2dc; }
    .main { display: none; }
    #top { height: 1; background: #1e212b; color: #eef0f8; padding: 0 1; }
    #chatbox { height: 1fr; padding: 0 1; scrollbar-size-vertical: 1; }
    #pipehead { height: 1; background: #191c24; padding: 0 1; }
    #pipe { height: auto; max-height: 8; background: #191c24; padding: 0 1;
            scrollbar-size-vertical: 1; }
    #params { height: 1; background: #191c24; padding: 0 1; }
    #ask { height: 3; border: tall #2c3140; background: #1a1d26; }
    #ask:focus { border: tall #4d8de0; }
    #splash { padding: 1 2; }
    """
    BINDINGS = [Binding("ctrl+c", "quit", "退出", system=True),
                Binding("q", "quit", "退出")]
    # Textual 会在屏幕顶行画 TITLE/SUB_TITLE；名字已经在自己的顶栏里，这里留空以免
    # 同一行出现两个标题（实测不设时它会画类名 KLlamaApp）。
    TITLE = ""
    SUB_TITLE = ""

    def __init__(self, args) -> None:
        super().__init__()
        self.args, self.boot, self.metrics = args, Boot(args), Metrics(0)
        self.buf: list[str] = []
        self.turns: list[tuple[str, str]] = []
        self.answer = ""
        self.busy = self.main_shown = False
        self.exit_code = 0

    def compose(self) -> ComposeResult:
        yield Static(id="splash")          # 启动画面，与主界面互斥
        # 四区：顶栏 / 对话 / 流水线 / 参数，外加输入行。main 类在主界面出现时统一显示。
        yield Static(id="top", classes="main")                  # 吞吐 · TTFT · 已用
        yield VerticalScroll(Static(id="chat"), id="chatbox", classes="main")
        yield Static(id="pipehead", classes="main")
        yield VerticalScroll(id="pipe", classes="main")          # 段数不定，必须可滚动
        yield Static(id="params", classes="main")
        yield Ask(id="ask", classes="main", placeholder="输入问题，Enter 发送　·　空输入按 q 退出")

    def on_mount(self) -> None:
        self.set_interval(0.08, self.tick)                       # 唯一的刷新节拍
        self.run_worker(self.boot.run, thread=True, name="boot")  # 启动在后台跑

    def enter_main(self) -> None:
        self.main_shown, self.metrics = True, Metrics(len(self.boot.stages))
        self.query_one("#pipe", VerticalScroll).mount_all(
            [Static(id=f"seg{i}") for i in range(len(self.boot.stages))])
        self.query_one("#splash").styles.display = "none"
        for widget in self.query(".main"):
            widget.styles.display = "block"
        self.query_one("#ask", Ask).focus()
        if self.args.prompt:
            self.ask(self.args.prompt)

    def tick(self) -> None:
        """唯一的刷新节拍：启动画面期间推 splash，之后推指标与增量文本。"""
        if not self.main_shown:
            self.query_one("#splash", Static).update(splash_text(self.boot))
            if "load" in self.boot.done:
                self.enter_main()
            elif self.boot.error and self.boot.phase != "load":
                self.exit_code = 1
                self.exit()     # 准备阶段失败：界面留不住，退出并清理
            return
        if self.buf:
            self.answer, self.buf = self.answer + "".join(self.buf), []
            self.show_chat()
        self.paint()

    def show_chat(self) -> None:
        """把当前对话推给对话区并滚到底。"""
        t = Text()
        for question, answer in self.turns:
            t.append(f"你  {question}\n", style="bold #4d8de0")
            t.append(f"{answer}\n\n", style="#e6e8f0")
        self.query_one("#chat", Static).update(t)
        self.query_one("#chatbox", VerticalScroll).scroll_end(animate=False)

    def paint(self) -> None:
        """刷新主界面四区。启动画面期间、或控件还没挂上时直接返回。"""
        if not self.main_shown:
            return
        try:
            self._paint()
        except Exception:  # noqa: BLE001 - 退出路径上屏幕可能已拆，刷新失败不该杀进程
            pass

    def _paint(self) -> None:
        m, boot = self.metrics, self.boot
        ttft = m.ttft_ms()
        ttft_s = f"TTFT {ttft:,.0f} ms" if ttft is not None else "TTFT —"
        # 四区各一块：顶栏 / 流水线头 / 每段一行 / 参数行。条形按当前最慢段归一化。
        self.query_one("#top", Static).update(
            f"[b]K-Llama[/]  [#59c26a]{m.rate:,.1f} tok/s[/]  [#9aa3b8]"
            f"{len(boot.stages)} 段 · {m.tokens} tok · {ttft_s} · "
            f"{m.elapsed(time.perf_counter()):.1f} s[/]")
        ms = m.stage_ms()
        top = max(ms) if ms else 0.0
        self.query_one("#pipehead", Static).update(
            f"[#7b8194]流水线 · 条形按当前最慢段归一化（{top:,.1f} ms = 满格）[/]")
        for stage in boot.stages:
            v = ms[stage.index] if stage.index < len(ms) else 0.0
            frac = v / top if top > 0 else 0.0
            n = int(round(frac * BAR))
            colour = "#59c26a" if frac < 0.5 else "#c9a227" if frac < 0.9 else "#e3596b"
            self.query_one(f"#seg{stage.index}", Static).update(
                f"[#e6e8f0]{stage.index}[/] {stage.kind:<4} {stage.layers:>2}L "
                f"[{colour}]{'█' * n}[/][#3a3f4d]{'░' * (BAR - n)}[/] [#cfd2dc]{v:7,.1f} ms[/]  "
                f"[#7b8194]{'local' if stage.is_local else stage.host}[/]")
        greedy = self.args.temperature <= 0.0
        mode = "贪心" if greedy else f"采样 temp={self.args.temperature:g}"
        drafts: list[str] = []
        if greedy and self.args.dflash:
            drafts.append(f"DFlash k={self.args.dflash_block - 1}")
        if greedy and self.args.spec_k:
            drafts.append(f"MTP k={self.args.spec_k}")
        draft_txt = (" · " + " + ".join(drafts)) if drafts else ""
        self.query_one("#params", Static).update(
            f"[#8f96a8]ctx {boot.ctx}  max_tokens {self.args.max_tokens}  "
            f"{mode}{draft_txt}  每问重置 KV[/]  [#7b8194]· "
            f"{boot.topology.name if boot.topology else ''}[/]")

    def ask(self, prompt: str) -> None:
        if self.busy or not self.boot.clients:
            return
        self.turns.append((prompt, ""))
        self.answer, self.busy = "", True
        self.metrics, self.metrics.t0 = Metrics(len(self.boot.stages)), time.perf_counter()
        self.paint()
        self.run_worker(lambda: self.generate(prompt), thread=True, name="gen")

    def generate(self, prompt: str) -> None:
        try:
            for client in self.boot.clients:  # 每问独立：KV 与位置都从一开始
                client.reset()
            args = self.args
            greedy = args.temperature <= 0.0
            tail = self.boot.clients[-1]
            # Sampler state lives in the shim and survives a KV reset, so it is set on
            # every request rather than once at boot: a run that switched modes must not
            # leave the tail sampling for the next question.
            if greedy:
                tail.clear_sampling()
            else:
                tail.set_sampling(temperature=args.temperature, top_k=args.top_k,
                                  top_p=args.top_p, seed=args.seed)
            # Drafting is only sound at temperature 0; a sampling tail would make the
            # verifier accept tokens the sampler never produced.
            for chunk in generate_stream(
                    self.boot.clients, prompt,
                    max_tokens=args.max_tokens,
                    spec_k=args.spec_k if greedy else 0,
                    dflash_path=args.dflash if greedy else "",
                    dflash_block=args.dflash_block):
                self.call_from_thread(self.on_chunk, chunk, time.perf_counter())
        except Exception as exc:  # noqa: BLE001 - 错误要显示在对话区
            self.call_from_thread(self.on_error, f"{type(exc).__name__}: {exc}")
            return
        self.call_from_thread(self.flush)

    def on_chunk(self, chunk, now: float) -> None:
        self.metrics.add(chunk, now)
        self.buf.append(chunk.text)

    def on_error(self, message: str) -> None: self.flush(extra=f"\n[错误] {message}")

    def flush(self, extra: str = "") -> None:
        """一轮生成结束（或出错）：把累积的文本落到对话区，然后刷新指标。"""
        self.busy, self.answer = False, self.answer + "".join(self.buf) + extra
        self.buf = []
        if self.turns:
            self.turns[-1] = (self.turns[-1][0], self.answer)
            self.show_chat()
        self.paint()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.input.value, prompt = "", event.value.strip()
        if prompt:
            self.ask(prompt)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="koakuma", description="K-Llama TUI，一条命令")
    ap.add_argument("--topology", default="", help="覆盖 SA_TOPOLOGY")
    ap.add_argument("--shim", default="", help="覆盖 SA_SHIM")
    ap.add_argument("--ctx", type=int, default=0, help="覆盖 SA_CTX")
    ap.add_argument("--prompt", default="", help="进 TUI 后自动发送的问题")
    ap.add_argument("--max-tokens", type=int, default=256)
    # The shim's sampler starts NULL, which is argmax -- so temperature 0 is both the
    # default and the only mode where drafting is valid. The flag exists to make that
    # explicit and to turn speculation on without also passing --dflash.
    ap.add_argument("--greedy", action="store_true",
                    help="temperature 0（默认即如此；显式给出便于与 --temperature 对照）")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="采样温度；>0 时关闭投机（验证器只暴露 argmax）")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--spec-k", type=int, default=0,
                    help="MTP 递归草稿数；只在 temperature 0 下生效")
    ap.add_argument("--dflash", default="",
                    help="DFlash 草稿 GGUF；只在 temperature 0 下生效")
    ap.add_argument("--dflash-block", type=int, default=6,
                    help="DFlash 每轮草稿数 = block-1（默认 6 = k=5）")
    args = ap.parse_args(argv)
    if args.greedy:
        args.temperature = 0.0
    if args.ctx:
        os.environ[config.CTX_ENV] = str(args.ctx)
    app = KLlamaApp(args)
    try:
        app.run()
        return app.exit_code
    except KeyboardInterrupt:
        return 130
    finally:
        # 正常退出、Ctrl+C、异常都走这里，否则 worker 会带着已加载的权重留下。
        app.boot.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
