"""koakuma -- 一条命令从冷 shell 到能生成文本的链，四区 TUI。

项目名 K-Llama（层段分布式推理，目前不依赖 Slimarc 运行时）。启动器做完 TUI 出来之前
必须做完的事：找拓扑、找 shim、内存预检、拉起拓扑里的每一段并连上。结果写进对象而不是
print()，因为 Textual 接管终端后 print() 只会冲掉画面；启动画面上每一行因此都对应一件真
发生过的工作。TUI.md §7 的四个数字修正都在 Metrics：吞吐用相邻 token 间隔的滑窗（不是
first_token_at 到现在的跨度）、条形按当前最大段归一化（不是固定 50 ms）、TTFT 单列、
已用时间从第一个 token 起算。bring_up / _terminate / estimate_ram_bytes /
mem_budget_bytes 直接复用 chain_client，好让内存模型与装配的修正到达每个入口。
"""
from __future__ import annotations

import argparse
import os
import re
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Input, Static

import chat
import config
import topology
from chain_client import (GPU_VRAM_OVERHEAD, VRAM_PER_BYTE, StageClient, _terminate,
                          bring_up, context_shortfall, estimate_ram_bytes,
                          estimate_vram_bytes, mem_budget_bytes, ram_ratio, stage_bytes,
                          tail_rs_depth, vram_after_load, vram_before_load, vram_info)
from stream import generate_stream

HERE = Path(__file__).resolve().parent
GIB = 2 ** 30
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


#: 找 shim 的规则只有一份，在 config.find_shim()：--shim > SA_SHIM > 仓库 shim/bin。
#: 这里原先还有第四级「这台机器的 H:\slimarc-proto\shim\bin」，那是 2026-10-30 的旧构建，
#: 34 个导出缺 15 个（tokenize / set_sampling / logits / draft_*），落到它上面 ChatML 路径直接失败。
find_shim = config.find_shim


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
        self.vram_before: tuple[int, int] | None = None
        #: 加载完成的显存实测（chain_client.vram_after_load）。预算是估算，只有这一份
        #: 能抓到真溢出；主界面据此报警，否则溢出完全静默。
        self.vram: dict | None = None
        self.status, self.errors, self._lock = {}, {}, threading.Lock()
        self.stopping = threading.Event()
        self.started_at = time.perf_counter()   # 启动画面显示真实已用时

    def run(self) -> None:
        """后台线程里跑完整准备流程；失败写进 self.error，不抛给界面。"""
        try:
            # ChatML 的 <|im_start|> 必须按 special token 解析，否则它会碎成普通片段，模型
            # 看到的是被拆坏的对话标记 —— 比裸文本更糟。必须在 spawn 之前设：worker 继承
            # 启动时的环境，晚一步就只影响下一次启动。--plain 不设，裸文本里那几个字符
            # 本来就该当普通文本。
            if not getattr(self.args, "plain", False):
                os.environ.update(chat.chat_token_env())
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
            # 拉段与连段分开：bring_up 是 CLI/TUI/server 共用的同一份装配，本机段 spawn、
            # 远端段按拓扑的 start_cmd 请求启动。这里传 note=None 是因为本函数的调用者
            # 在 Textual 的画布后面，print 会冲掉正在画的帧。
            self.vram_before = vram_before_load(self.stages)
            bring_up(self.stages, self.shim, self.ctx,
                     tail_rs_depth(getattr(self.args, "dflash", ""), self.args.dflash_block,
                                   getattr(self.args, "cascade_mtp", 0)),
                     procs=self.procs, note=None)
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
                    # 全部连上就是加载完成的时刻：此时实测显存才有意义。放在 done.add 之前，
                    # 因为 tick() 看到 "load" 就切主界面了。
                    self.vram = vram_after_load(self.stages, self.vram_before)
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


class Session:
    """真多轮的账本：累积的 token、KV 已写到哪、以及两者之间的差额。

    真多轮的地基只有一条不变量：KV 里的位置 ``[0, n_past)`` 恰好等于 ``ids[:n_past]``。
    守住它，下一轮才能把新 token 接在后面而不必重放整段历史；丢掉它，模型看到的就是一段
    用户从未见过的上下文，而输出「看起来像话」，没人会发现。

    破坏这条不变量的只有一件事：投机块在轮末被截断时，llama.cpp 已经把整块写进 KV，却
    不允许回退最后一个位置（见 docs/ISSUE-recurrent-snapshot-rollback.md），于是 KV 里
    留下若干「已写但没 emit」的 token。``align()`` 一发现这种差额就要求调用者「从 ids
    重放整段」——它不尝试裁剪，原因见 ``align()`` 的 docstring（hybrid 上 seq_rm 会静默
    失败）。重放慢一次，但 KV 与对话文本严格一致。

    ``ids`` 里包含 ``pending``——已 emit、但还没写进 KV 的最后一个 token。下一轮把它前置到
    ``prefix_ids`` 前面，位置就接上了（`stream.generate_stream` 的续写契约）。
    """

    def __init__(self) -> None:
        self.ids: list[int] = []
        self.n_past = 0                 # KV 已写到的位置数
        self.pending: int | None = None  # 已 emit、未写 KV 的最后一个 token
        self.last: int | None = None     # 本轮最后 emit 的 token（判 ChatML 是否收尾）
        self.rounds = 0
        self.replays = 0                # 被强制整段重放的次数（诊断用）
        self.dirty = False              # 上一轮异常退出，KV 状态未知

    def reset(self) -> None:
        self.ids, self.n_past, self.pending, self.last = [], 0, None, None
        self.rounds, self.replays, self.dirty = 0, 0, False

    def kv_target(self) -> int:
        """KV 里应该有多少个位置，即 ``ids`` 中已经写进 KV 的那部分。

        ``pending`` 那个 token 已经在 ``ids`` 里（它被 emit 出去了），但还没走过
        forward，所以 KV 应比 ``ids`` 少一个。
        """
        return len(self.ids) - (1 if self.pending is not None else 0)

    def stray(self) -> int:
        """KV 里多出来、不属于对话的位置数（投机块残留）。"""
        return max(0, self.n_past - self.kv_target())

    def align(self, clients) -> bool:
        """KV 里有多余位置时返回 True，调用者必须整段重放。

        这里**不尝试** `seq_rm` 把尾巴裁掉，因为实测它在 hybrid 模型上会静默失败：

            [seq_rm] p0=236 rc=-6
            ... the last position stored in the memory module ... is X = 237
            ... the tokens ... have a starting position of Y = 236
            llama_decode: failed to decode, ret = -1

        `-6` 表示「那段范围里没有可删的行」，shim 把它当 no-op 不报错（`keep_head_worker`
        的既有约定，对投机回滚是对的），于是调用方从返回值上分辨不出「裁掉了」和「没裁」。
        下一次 prefill 直接死在 `X < Y` 上，整条链报废。

        重放一次 prefill 是看得见的慢；一个静默坏掉的 KV 是看不见的错。选前者。这条路径
        只在「轮末停在投机块内」时触发——纯 decode 不会产生多余位置。
        """
        return self.stray() > 0

    def encode(self, clients, question: str, *, plain: bool, system: str | None) -> list[int]:
        """本轮新增部分的 token。渲染只交给 chat：第二个渲染器就是漂移的起点。"""
        if plain:
            text = question if not self.rounds else f"\n{question}\n"
        elif not self.rounds:
            text = chat.apply_chatml([{"role": "user", "content": question}], system=system)
        else:
            # 上一轮是否已经写了 <|im_end|>：自然停止时 pending 就是它；被 max_tokens
            # 截断时缺这个边界，next_turn 会补上。
            closed = self.last is not None and clients[-1].is_eog(self.last)
            text = chat.next_turn(question, assistant_closed=closed)
        return clients[0].tokenize(text)

    def prepare(self, clients, question: str, *, plain: bool,
                system: str | None) -> tuple[list[int], int, list[int]]:
        """返回 ``(prefix_ids, n_past, fresh)``：这一轮要写进链的 token、起始位置、新增部分。"""
        if self.dirty or self.align(clients):
            # 整段重放：清掉 KV，把累积的对话 token 重新写回去。dirty 时连位置账都不能
            # 信——上一轮的异常可能停在一次 forward 中间，KV 的实际内容无从推断。
            for client in clients:
                client.reset()
            self.n_past, self.pending, self.dirty = 0, None, False
            if self.ids:
                self.replays += 1
            prefix = list(self.ids)
        else:
            prefix = [] if self.pending is None else [self.pending]
        fresh = self.encode(clients, question, plain=plain, system=system)
        return prefix + fresh, self.n_past, fresh

    def record(self, stats: dict, fresh: list[int], generated: list[int]) -> None:
        """一轮结束（自然结束或被停止）后记账，使下一轮的起点有据可依。"""
        self.ids.extend(fresh)
        self.ids.extend(generated)
        self.n_past = int(stats.get("n_past", self.n_past))
        self.pending = stats.get("pending")
        self.last = stats.get("last_token")
        self.rounds += 1


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


#: 启动画面的大标题：6 行高的大写 K，右侧挂名字与副标题。
#: 逐行渐显（每行 50 ms）——启动要几秒到几十秒，一个立刻出现然后静止的 logo 会让人以为
#: 卡住了；而进度条式的渐显本身就是"我还在动"的证据。
LOGO = (
    ("██╗  ██╗", "#4d8de0"),
    ("██║ ██╔╝", "#4f93d8"),
    ("█████╔╝", "#4f9ad0"),
    ("██╔═██╗", "#4fa2c8"),
    ("██║  ██╗", "#4faac0"),
    ("╚═╝  ╚═╝", "#4fb2b8"),
)
#: 每行出现间隔。六行全部出现用 0.3 s，比一次 tick（80 ms）长，肉眼能看出是"长出来的"。
LOGO_STEP_S = 0.05


def logo_text(age: float) -> Text:
    """按已用时间渐显大标题；右侧第 3、4 行分别是名字与副标题。"""
    t = Text()
    shown = int(age / LOGO_STEP_S) + 1
    for i, (line, colour) in enumerate(LOGO):
        if i > shown:
            break
        t.append(line, style=f"bold {colour}")
        if i == 2:
            t.append("   K-Llama", style="bold #eef0f8")
        elif i == 3:
            t.append("   层段分布式推理 · 四区 TUI", style="#9aa3b8")
        t.append("\n")
    return t


def splash_text(b: Boot) -> Text:
    """启动画面：每行一件真发生过的工作，进行中的行显示真实百分比与真实已用时。

    模块级函数而非 App 方法：它只依赖 Boot 的状态，独立出来更好读、也更好在测试里断言。
    """
    t = logo_text(time.perf_counter() - b.started_at)

    def step(label: str, ok: bool, detail: str, bad: bool = False) -> None:
        mark, colour = ("✓", "#59c26a") if ok else ("✗", "#e3596b") if bad else ("·", "#7b8194")
        t.append(f"  {mark} ", style=f"bold {colour}")
        t.append(f"{label:<9}{detail}\n" if detail else f"{label:<9}\n",
                 style="#9aa3b8" if ok else "#7b8194")

    # 模型与运行参数。官方 TUI 第一屏就给 model / ftype，这边跑完 HELLO 校验却从不显示——
    # 校验通过了却看不见，等于没校验。每段可以是不同文件（弱设备那套就是这样），所以文件
    # 数不是 0/1 时如实报个数，不假装只有一个。
    if b.stages:
        names = {s.model.name for s in b.stages}
        model = names.pop() if len(names) == 1 else f"{len(b.stages)} 段 · {len(names) + 1} 个文件"
        gpu = sorted({s.gpu_layers for s in b.stages})
        t.append(f"  model    {model}\n", style="#9aa3b8")
        t.append(f"  ctx      {b.ctx or '—'}    "
                 f"{len(b.stages)} 段 / {sum(s.layers for s in b.stages)} 层    "
                 f"gpu_layers {'/'.join(str(x) for x in gpu) if gpu else '—'}\n\n",
                 style="#9aa3b8")
    else:
        t.append("\n")
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
            # 拓扑给了 start_cmd 的话 bring_up 已经请求它启动了；没给就只能等它自己在
            # 监听。两者的等待时间一样长，画面必须说清是哪一种，否则超时无人能查。
            how = "已请求启动" if stage.start_cmd else "未配 start_cmd，等待其已在监听"
            t.append(f"  {how} {local}:{stage.port} …\n", style="#7b8194")
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
                # Esc 是生成中唯一的软出口。Ctrl+C 会连着 worker 一起杀，而一段 35B 的
                # 权重要几十秒才装载回来；Esc 只让生成循环在下一个 token 处收尾。
                Binding("escape", "stop", "停止生成", priority=True),
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
        #: 真多轮的账本。启动画面期间是空的，第一问从位置 0 起。
        self.session = Session()
        self.stop_event = threading.Event()
        #: 参数行上的一句状态（已停止 / 停止中…）；下一问开始时清掉。
        self.note = ""

    def compose(self) -> ComposeResult:
        yield Static(id="splash")          # 启动画面，与主界面互斥
        # 四区：顶栏 / 对话 / 流水线 / 参数，外加输入行。main 类在主界面出现时统一显示。
        yield Static(id="top", classes="main")                  # 吞吐 · TTFT · 已用
        yield VerticalScroll(Static(id="chat"), id="chatbox", classes="main")
        yield Static(id="pipehead", classes="main")
        yield VerticalScroll(id="pipe", classes="main")          # 段数不定，必须可滚动
        yield Static(id="params", classes="main")
        yield Ask(id="ask", classes="main",
                  placeholder="输入问题，Enter 发送　·　/clear 清空上下文　·　Esc 停止生成")

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
            # `show_chat` 渲染的是 `self.turns`，而 `turns[-1]` 原先只在 `flush` 时写。
            # 结果生成期间对话区恒为空、结束后一次性出现——流式数据、0.08 s 节拍都在，
            # 只是接错了源。这里同步一次，才真的"边生成边显示"。
            if self.turns:
                self.turns[-1] = (self.turns[-1][0], self.answer)
            self.show_chat()
        self.paint()

    def show_chat(self) -> None:
        """把当前对话推给对话区并滚到底。question 为空的行是系统提示（/clear 分界）。"""
        t = Text()
        for question, answer in self.turns:
            if question:
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
        greedy = config.is_greedy({"temperature": self.args.temperature})
        mode = "贪心" if greedy else f"采样 temp={self.args.temperature:g}"
        drafts: list[str] = []
        if self.args.dflash:
            drafts.append(f"DFlash k={self.args.dflash_block - 1}")
        if self.args.spec_k:
            drafts.append(f"MTP k={self.args.spec_k}")
        draft_txt = (" · " + " + ".join(drafts)) if drafts else ""
        # 加载后的显存实测：预检是估算，段真正驻留后只有这一份能抓到溢出。溢出完全
        # 静默（WDDM 用系统内存兜住，llama.cpp 照样报 offloaded N/N），所以必须挂在
        # 常驻的参数行上，而不是启动画面上闪一下就消失。
        spill = ""
        v = boot.vram
        if v and v["spill"]:
            spill = (f"  [#e3596b]⚠ 显存溢出 {v['landed'] / GIB:.2f}/"
                     f"{v['want'] / GIB:.2f} GiB —— 降最贵段的 gpu_layers[/]")
        # 真多轮之后「每问重置 KV」不再成立，这里换成 KV 的已用位置：它既是下一问能否
        # 继续的唯一依据（满了会被拒绝，不静默截断），也是 /clear 前后唯一可见的差别。
        sess = self.session
        used = f"KV {sess.n_past}/{boot.ctx} tok" if boot.ctx else f"KV {sess.n_past} tok"
        turns = f"{sess.rounds} 轮" if sess.rounds else "首问"
        fmt = "plain" if self.args.plain else "ChatML"
        note = f"  [#e3a04a]{self.note}[/]" if self.note else ""
        replay = f"  [#e3a04a]重放 {sess.replays} 次[/]" if sess.replays else ""
        self.query_one("#params", Static).update(
            f"[#8f96a8]ctx {boot.ctx}  max_tokens {self.args.max_tokens}  "
            f"{mode}{draft_txt}  {fmt} · {turns} · {used}[/]{note}{replay}  "
            f"[#7b8194]· {boot.topology.name if boot.topology else ''}[/]{spill}")

    def ask(self, prompt: str) -> None:
        if self.busy or not self.boot.clients:
            return
        self.stop_event.clear()
        self.note = ""
        self.turns.append((prompt, ""))
        self.answer, self.busy = "", True
        self.metrics, self.metrics.t0 = Metrics(len(self.boot.stages)), time.perf_counter()
        self.paint()
        self.run_worker(lambda: self.generate(prompt), thread=True, name="gen")

    def action_stop(self) -> None:
        """Esc：让生成在下一个 token 处收尾，不碰 worker。

        Ctrl+C 走 action_quit，会连 worker 一起清理——那是对的退出，但对「这段答歪了」
        是灾难：35B 两段要几十秒才装载回来。这里只置一个 Event，解码循环每 token 看一次，
        停下来以后 KV 由 Session 对齐，链保持可用。
        """
        if not self.busy:
            return
        if not self.stop_event.is_set():
            self.stop_event.set()
            self.note = "停止中…"
            self.paint()

    def generate(self, prompt: str) -> None:
        """生成线程：真多轮续写 + ChatML + 采样投机 + 可被 Esc 打断。"""
        args, clients = self.args, self.boot.clients
        try:
            # 先把 KV 账对齐（裁掉上一轮的投机块残留，必要时整段重放），再决定本轮从哪
            # 个位置写。prefix 里可能带着上一轮 pending 的那个 token。
            prefix, n_past, fresh = self.session.prepare(
                clients, prompt, plain=args.plain, system=args.system or None)
            # 上下文检查必须在任何 decode 之前：超额时 llama_decode 返回 rc=-1，链会死在
            # 半路，看起来像「模型坏了」而不是「放不下」。给的是差额，不是一句 ctx 超了。
            if why := context_shortfall(n_past + len(prefix), args.max_tokens, self.boot.ctx):
                self.call_from_thread(
                    self.on_error,
                    f"{why}（历史 {n_past} tok + 本轮 {len(prefix)} tok；"
                    f"用 /clear 清空上下文，或调小 --max-tokens）")
                return
            greedy = config.is_greedy({"temperature": args.temperature})
            tail = clients[-1]
            # 采样器状态在 shim 里、能活过 KV reset，所以每问都设一次：否则上一问的模式
            # 会留到这一问。
            sampling = None
            if greedy:
                tail.clear_sampling()
            else:
                tail.set_sampling(temperature=args.temperature, top_k=args.top_k,
                                  top_p=args.top_p, seed=args.seed)
                # 采样投机：拒绝采样要 p、q 两张分布，而 top_k/top_p/penalties 在 shim 的
                # 采样链内部、不以分布形式暴露。能走到这里说明 main() 已经拒绝过不支持
                # 的组合（config.sampled_spec_problem）；这里只把 temperature 与 RNG 交给
                # 解码循环，使同一 --seed 的两次运行逐 token 可复现。
                sampling = {"temperature": args.temperature,
                            "rng": np.random.default_rng(args.seed)}
            stats: dict = {}
            generated: list[int] = []
            for chunk in generate_stream(
                    clients, "", prefix_ids=prefix, n_past=n_past,
                    max_tokens=args.max_tokens, spec_k=args.spec_k,
                    dflash_path=args.dflash, dflash_block=args.dflash_block,
                    cascade_mtp=getattr(args, "cascade_mtp", 0),
                    sampling=sampling, stop=self.stop_event, stats=stats):
                # token == -1 是组装器收尾补的剩余文本，不是模型生成的 id，不能进历史。
                if chunk.token >= 0:
                    generated.append(chunk.token)
                self.call_from_thread(self.on_chunk, chunk, time.perf_counter())
            self.session.record(stats, fresh, generated)
            note = ("⏹ 已停止：已生成的部分留在上下文里，下一问接着这段写"
                    if stats.get("stopped") else "")
            self.call_from_thread(self.flush, "", note)
        except Exception as exc:  # noqa: BLE001 - 错误要显示在对话区
            # 这一轮可能停在一次 forward 中间，KV 的实际内容无从推断：下一轮整段重放，
            # 而不是接着一个说不清的 n_past 写下去。
            self.session.dirty = True
            self.call_from_thread(
                self.on_error,
                f"{type(exc).__name__}: {exc}\n（本轮未计入上下文，下一问会从已确认的历史重放）")

    def on_chunk(self, chunk, now: float) -> None:
        self.metrics.add(chunk, now)
        self.buf.append(chunk.text)

    def on_error(self, message: str) -> None:
        self.note = ""
        self.flush(extra=f"\n[错误] {message}")

    def flush(self, extra: str = "", note: str = "") -> None:
        """一轮生成结束（正常 / 被停止 / 出错）：把累积的文本落到对话区，然后刷新指标。"""
        self.busy, self.answer = False, self.answer + "".join(self.buf) + extra
        self.buf = []
        if note:
            self.answer += f"\n{note}"
            self.note = "已停止"
        elif not extra:
            self.note = ""
        if self.turns:
            self.turns[-1] = (self.turns[-1][0], self.answer)
            self.show_chat()
        self.paint()

    def clear_context(self) -> None:
        """丢弃上下文：KV 置空、累积序列清零。

        对话文本留在屏幕上（否则用户会以为历史被吞了），但明确标出分界线：这条线以上
        的内容不再喂给模型。下一问从位置 0 重新开一段 ChatML 对话。
        """
        if self.busy:
            return
        failed: list[int] = []
        for client in self.boot.clients:
            try:
                client.reset()
            except Exception:  # noqa: BLE001 - 一段失败不该挡住其它段，但必须记住
                failed.append(client.stage.index)
        self.session.reset()
        self.note = ""
        if failed:
            # reset 失败的段 KV 状态未知，下一问必须先重放而不是从「位置 0」开始。
            self.session.dirty = True
            self.turns.append(("", f"── 上下文已清空，但 stage {failed} 的 reset 失败："
                                   f"下一问会重放已确认的历史 ──"))
        else:
            self.turns.append(("", "── 上下文已清空：上面的对话不再喂给模型，下一问从位置 0 开始 ──"))
        self.show_chat()
        self.paint()

    def command(self, text: str) -> None:
        name = text.split()[0].lower()
        if name in ("/clear", "/reset"):
            self.clear_context()
        else:
            self.turns.append(("", f"[未知命令 {name}；可用 /clear 清空上下文]"))
            self.show_chat()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.input.value, prompt = "", event.value.strip()
        if prompt.startswith("/"):
            self.command(prompt)
        elif prompt:
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
                    help="采样温度；>0 时默认仍可投机，但只支持 temperature（见 --spec-k）")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--plain", action="store_true",
                    help="不渲染 ChatML，直接发裸文本（默认走 ChatML；多轮也按 ChatML 续写）")
    ap.add_argument("--system", default="",
                    help="ChatML 的 system 段（默认不注入，与 server 的行为一致）")
    ap.add_argument("--spec-k", type=int, default=0,
                    help="MTP 递归草稿数；贪心与采样都支持，采样时要求 --top-k 0 --top-p 1.0")
    ap.add_argument("--dflash", default="",
                    help="DFlash 草稿 GGUF；只在 temperature 0 下生效")
    ap.add_argument("--dflash-block", type=int, default=6,
                    help="DFlash 每轮草稿数 = block-1（默认 6 = k=5）")
    ap.add_argument("--cascade-mtp", type=int, default=0,
                    help="DFlash 块整块接受后再用 MTP 追加几个草稿（0=关，实测 +6%%）")
    args = ap.parse_args(argv)
    if args.greedy:
        args.temperature = 0.0
    # 参数组合的冲突在装载任何权重之前就拒绝：跑到生成里才发现，代价是几十秒的加载和
    # 一个死掉的链。判据本身只有一份（config.sampled_spec_problem），CLI 与 TUI 共用。
    greedy = config.is_greedy({"temperature": args.temperature})
    if args.dflash and not greedy:
        raise SystemExit(config.DFLASH_NEEDS_GREEDY)
    if args.spec_k and not greedy:
        if why := config.sampled_spec_problem(args.temperature, args.top_k, args.top_p):
            raise SystemExit(why)
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
