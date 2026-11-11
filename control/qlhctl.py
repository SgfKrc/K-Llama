#!/usr/bin/env python3
"""qlhctl —— v2.1 分段链的控制面：读拓扑、起停 stage、探活。

三个子命令，不做第四件事：

    up      读拓扑，按链序（尾段先、链首最后）拉起每段进程，等它自己打印 ready
    status  对每段报 pid / 端口 / ready 行，一次调用内说明哪一段不在
    down    按 state 里记下的 pid 停掉这次 up 起的进程，并复核端口已释放

本文件**不**做的事（写在这里，是因为「又有一个 Python 了」很容易被读成 v2 回来了）：
不实现链协议（v9 在 `split-stage` 与 `llama_context` 里）、不碰 token、不理解层、不做
采样、不校验层范围与切点（那些是 C++ 侧的事）、不做切分求解与动态重平衡
（`docs/DEPLOYMENT.md` §7）。删掉本文件链照样能手工起，它省掉的只是「记住五条命令、
盯五个日志、猜哪一段死了」。

只依赖 Python 3.11+ 标准库（`tomllib`）。
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

# [QLH-CTL] 段的就绪信号是它自己打印的那一行，不是「端口能连上」。
#
# split-stage 从 v10 起先 bind 再加载，所以端口在模型读第一个字节之前就是通的——把
# TCP connect 当 ready 判据，会把一段还在读 20 GiB 权重的进程报成健康。ready 行在
# 模型、上下文与 offload 计划都定下来之后才打印，那才是「可以接请求」的时刻。
READY_SPLIT = re.compile(r"\[split-stage\] ready: layers \[(\d+), (\d+)\) of (\d+), role=(\w+)")
READY_HEAD = re.compile(r"listening on https?://")

# 停进程前用来认「这是我们起的那个」的映像名；pid 会被系统复用，只看 pid 会杀错。
OUR_IMAGES = ("split-stage.exe", "llama-server.exe", "split-stage", "llama-server")

LOG_TAIL_BYTES = 65536
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
# Windows 上 OpenSSH 的默认 shell 是 cmd.exe：脚本走参数时 `>` `&` `|` 会被 cmd 当自己的
# 元字符，引号还要按 cmd 的规则转义两次。所以那种远端脚本走 stdin 交给 Git Bash。
DEFAULT_WIN_BASH = "C:/Program Files/Git/bin/bash.exe"
PROBE_SCRIPT = "echo __QLH_SHELL__=$0\nuname -s || echo no-uname"


# ---------------------------------------------------------------- 基础设施


def die(msg: str, code: int = 1) -> "NoReturn":  # type: ignore[name-defined]
    print(f"qlhctl: {msg}", file=sys.stderr)
    raise SystemExit(code)


def note(msg: str) -> None:
    print(msg, flush=True)


def loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1", "0.0.0.0") or host.startswith("127.")


def run(cmd: list[str], timeout: float = 20.0, stdin_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def pid_alive(pid: int) -> bool:
    """进程还活着吗。Windows 上不能用 os.kill(pid, 0)——那在 CPython 里是 TerminateProcess。"""
    if pid <= 0:
        return False
    if os.name == "nt":
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        k32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def image_of(pid: int) -> str:
    """pid 对应的映像名，认不出返回空串。只用 tasklist：wmic 在新系统上已被移除。"""
    if os.name != "nt":
        r = run(["ps", "-o", "comm=", "-p", str(pid)], timeout=10)
        return r.stdout.strip()
    r = run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], timeout=10)
    line = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
    return line.split(",")[0].strip('"') if line.startswith('"') else ""


def proc_cmdline(pid: int) -> str:
    """pid 的完整命令行，拿不到返回空串。

    这是 state.json 之外的第二份身份：run_dir 跟着 cwd 走，换个目录跑 up 就看不到上次的
    记录，而进程还在。命令行不会丢——它同时带着 --port 与 --layer-range，足以把「我们的
    tail」和「别人的 split-stage」（比如 verify.sh 的 93xx）分开。
    Windows 上 wmic 已被移除但多数系统还在，先用它；拿不到再退到 PowerShell/CIM（慢 ~1s，
    只在端口冲突时走一次）。"""
    if os.name != "nt":
        return run(["ps", "-p", str(pid), "-o", "args="], timeout=10).stdout.strip()
    r = run(["wmic", "process", "where", f"processid={pid}", "get", "commandline"], timeout=15)
    for line in r.stdout.splitlines()[1:]:
        if line.strip():
            return line.strip()
    r = run(["powershell", "-NoProfile", "-Command",
             f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"], timeout=30)
    return r.stdout.strip()


def looks_like_ours(st: dict, pid: int) -> bool:
    """占着这个端口的进程，是不是「拓扑里的这一段」。只看两条证据：映像名，以及命令行里
    同时出现的 --port 与层范围。不解析、不重算，就是字符串比对。"""
    if image_of(pid).lower() not in OUR_IMAGES:
        return False
    cmd = proc_cmdline(pid)
    return bool(cmd) and f"--port {st['port']}" in cmd and st["layers"] in cmd


def listeners() -> dict[int, int]:
    """本机正在 LISTEN 的端口 -> pid。用 netstat/ss 而不是 TCP connect：split-stage 是
    「单连接、backlog=1」的服务，链首连着它时新 connect 只能排队——探出来是超时，还会
    往 backlog 里塞一个永不 accept 的连接。netstat 只看内核状态，没有副作用。"""
    table: dict[int, int] = {}
    if os.name != "nt":
        r = run(["ss", "-lntpH"], timeout=10)
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            port = parts[3].rsplit(":", 1)[-1]
            m = re.search(r"pid=(\d+)", line)
            if port.isdigit():
                table[int(port)] = int(m.group(1)) if m else 0
        return table
    r = run(["netstat", "-ano", "-p", "tcp"], timeout=15)
    for line in r.stdout.splitlines():
        f = line.split()
        if len(f) >= 5 and f[3].upper() == "LISTENING":
            port = f[1].rsplit(":", 1)[-1]
            if port.isdigit():
                try:
                    table[int(port)] = int(f[4])
                except ValueError:
                    pass
    return table


# ---------------------------------------------------------------- 远端（ssh）


def _ssh(st: dict) -> list[str]:
    return shlex.split(os.environ.get("QLH_SSH") or "ssh") + SSH_OPTS + [st["ssh"]]


def win_bash(st: dict) -> str:
    return st.get("remote_shell") or DEFAULT_WIN_BASH


def remote_style(st: dict) -> str:
    """远端是什么：`posix`（Linux / Termux）、`win-cmd`（Windows，默认 shell 是 cmd.exe）、
    `win-bash`（Windows，默认 shell 已经是 Git Bash / MSYS）。

    一条探测命令分得开：`$0` 在 cmd.exe 里不会被展开（原样打出 `$0`），`uname -s` 在
    Windows 上给出 MINGW/MSYS/CYGWIN 前缀。三种情形一条命令覆盖，Linux 用户不用多写
    一个字段；`remote_shell` 显式给了就直接当 `win-cmd` 处理（走那个 bash）。
    """
    if st.get("remote_shell"):
        return "win-cmd"
    if "_style" not in st:
        out = run(_ssh(st) + [PROBE_SCRIPT], timeout=25).stdout
        if "$0" in out:
            st["_style"] = "win-cmd"
        elif any(t in out for t in ("MINGW", "MSYS", "CYGWIN")):
            st["_style"] = "win-bash"
        else:
            st["_style"] = "posix"
    return st["_style"]


def remote(st: dict, script: str, timeout: float = 25.0) -> subprocess.CompletedProcess:
    """在段的机器上跑一段 sh。POSIX 远端：脚本作为 ssh 的最后一个参数。Windows 远端
    （默认 shell 是 cmd）：脚本走 stdin 交给 Git Bash——走参数的话 `>` `&` `|` 会先被 cmd
    当自己的元字符解释一遍（实测还会多出一个引号）。"""
    if remote_style(st) == "win-cmd":
        try:
            return run(_ssh(st) + [f'"{win_bash(st)}" -l'], timeout=timeout, stdin_text=script)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess([], 124, "", f"ssh timed out after {timeout:.0f}s")
    try:
        return run(_ssh(st) + [script], timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess([], 124, "", f"ssh timed out after {timeout:.0f}s")


def stage_alive(st: dict, pid: int) -> bool:
    if pid <= 0:
        return False
    if st.get("ssh"):
        if remote_style(st) == "posix":
            return remote(st, f"kill -0 {pid} 2>/dev/null && echo yes").stdout.strip() == "yes"
        # Windows 远端：pid 是 cmd.exe（它等 split-stage 结束），Git Bash 的 kill 不认识它。
        out = remote(st, f'tasklist //FI "PID eq {pid}" //NH 2>/dev/null').stdout
        return "split-stage" in out or "cmd.exe" in out
    return pid_alive(pid)


def read_log(st: dict, n: int = LOG_TAIL_BYTES) -> str:
    if st.get("ssh"):
        r = remote(st, f"tail -c {n} {shlex.quote(st['log'])} 2>/dev/null")
        return r.stdout
    try:
        with open(st["log"], "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def log_last_line(st: dict) -> str:
    for line in reversed(read_log(st).splitlines()):
        if line.strip():
            return line.strip()[:220]
    return ""


def _stage_extra(raw: dict, label: str) -> list[str]:
    """段上的 `extra` 必须是**字符串数组**，按序追加到该段命令行末尾。

    TOML 里写成 `extra = "--foo bar"`（单个字符串）时，`for a in extra` 会逐字符迭代，
    命令行静默变成一堆单字母参数——不报错，只是那一段起不来，日志里也看不出是配置写错了。
    `docs/DEPLOYMENT.md` §2 记了这个坑；这里把它变成启动前的显式错误，而不是留给运行期。
    """
    value = raw.get("extra")
    if value is None:
        return []
    if isinstance(value, str):
        die(
            f"{label} 的 extra 是单个字符串，必须写成字符串数组：\n"
            f'  extra = ["--foo", "bar"]\n'
            f"  写成字符串会被逐字符拆成参数（不报错，只是命令行变成一堆单字母）。"
        )
    if not isinstance(value, list):
        die(f"{label} 的 extra 必须是字符串数组，收到 {type(value).__name__}：{value!r}")
    return [str(a) for a in value]


def load_topology(path: Path) -> dict:
    if not path.is_file():
        die(f"拓扑文件不存在：{path}")
    try:
        doc = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        die(f"拓扑不是合法 TOML：{path}\n{exc}")

    chain = dict(doc.get("chain") or {})
    raw = list(doc.get("stage") or [])
    if not raw:
        die(f"拓扑里没有任何 [[stage]]：{path}")

    name = str(chain.get("name") or path.stem)
    bin_dir = str(chain.get("bin_dir") or os.environ.get("QLH_BIN") or "H:/llamacpp-c/build-split/bin")
    stages = []
    seen_ports = {}
    for i, s in enumerate(raw):
        for key in ("role", "port", "model", "layers"):
            if key not in s:
                die(f"第 {i + 1} 段缺字段 `{key}`（拓扑 {path}）")
        if s["role"] not in ("head", "relay", "tail"):
            die(f"第 {i + 1} 段的 role 是 {s['role']!r}，只能是 head / relay / tail")
        port = int(s["port"])
        if port in seen_ports:
            die(f"第 {i + 1} 段与 {seen_ports[port]} 用了同一个端口 {port}：两段不可能同时监听它")
        seen_ports[port] = f"第 {i + 1} 段"
        host = str(s.get("host") or "127.0.0.1")
        st_name = str(s.get("name") or f"{s['role']}{i + 1}")
        st = {
            "name": st_name,
            "role": s["role"],
            "host": host,
            # bind 只决定进程监听哪个地址。缺省按安全性取：本机回环就只绑回环；
            # 需要被别的机器连（host 不是回环）才张到 0.0.0.0。split-stage 自己的缺省
            # 也是回环，理由见它文件头（链上没有认证）。
            "bind": str(s.get("bind") or ("127.0.0.1" if loopback(host) else "0.0.0.0")),
            "port": port,
            "model": str(s["model"]),
            "layers": str(s["layers"]),
            "ssh": str(s["ssh"]) if s.get("ssh") else None,
            "remote_shell": str(s["remote_shell"]) if s.get("remote_shell") else None,
            "bin_dir": str(s.get("bin_dir") or bin_dir),
            "ctx": int(s.get("ctx") or chain.get("ctx") or 4096),
            "batch": int(s.get("batch") or chain.get("batch") or 512),
            "threads": s.get("threads"),
            "ngl": s.get("ngl"),
            "extra": _stage_extra(s, st_name),
            "workdir": s.get("workdir"),
            "log": None,
        }
        if not loopback(host) and not st["ssh"]:
            die(
                f"第 {i + 1} 段（{st['name']}）的 host 是 {host}，不是本机回环，但没给 `ssh`。\n"
                f"  本机段写 host = \"127.0.0.1\"；别的机器要写 ssh = \"user@host\"，"
                f"否则控制面会去本机 spawn 一个只存在于那台机器上的路径。"
            )
        stages.append(st)

    # [QLH-CTL] 下面两条不是「重算 split-stage 已经算过的」，而是它明确看不到的东西：
    #  - 容量：`--split-remote` 的文档写着 identity 握手只校验 range / 深度 / 宽度，
    #    **不校验容量**；某段 -c 比链首小，会在跑到长 prompt 时才报 ST_NO_SPACE。
    #  - rs_seq 一致性：两端的快照数由两个不同的开关（--rs-seq / --split-rs-seq）给，
    #    给不一致时的症状是回退静默失效，而不是启动报错。
    head = next((st for st in stages if st["role"] == "head"), None)
    head_ctx = head["ctx"] if head is not None else None
    if head_ctx is not None:
        for st in stages:
            if st["ctx"] < head_ctx:
                die(
                    f"{st['name']} 的 ctx={st['ctx']} 小于链首的 {head_ctx}："
                    f"split-stage 的握手不校验容量，这会变成跑到长 prompt 时才出现的 ST_NO_SPACE"
                )
            if st["batch"] < head["batch"]:
                die(f"{st['name']} 的 batch={st['batch']} 小于链首的 {head['batch']}")

    # 显式写 rs_seq = 0 不能被 `or` 吞成 8：0 是「配错了」而不是「没写」
    _rs = chain.get("rs_seq")
    rs_seq = 8 if _rs is None else int(_rs)
    if rs_seq < 1:
        die("chain.rs_seq 必须 >= 1：split-stage 拒绝 0（远程段没有 checkpoint 可回退）")

    run_root = chain.get("run_dir")
    run_dir = Path(run_root) if run_root else Path.cwd() / ".qlh_tmp" / "qlhctl" / name
    if run_root and not Path(run_root).is_absolute():
        run_dir = path.parent / Path(run_root)

    for st in stages:
        if st["ssh"]:
            wd = st.get("workdir") or "."
            st["log"] = f"{str(wd).rstrip('/')}/{st['name']}.log"
        else:
            st["log"] = str(run_dir / f"{st['name']}.log")

    return {
        "path": path,
        "name": name,
        "rs_seq": rs_seq,
        "ready_timeout": float(chain.get("ready_timeout") or 240),
        "stages": stages,
        "run_dir": run_dir,
    }


def exe(st: dict, tool: str) -> str:
    win = str(st.get("_style") or "").startswith("win")
    name = tool + (".exe" if (os.name == "nt" and not st["ssh"]) or win else "")
    if st["ssh"]:
        return f"{st['bin_dir'].rstrip('/')}/{name}"
    return str(Path(st["bin_dir"]) / name)


def stage_argv(topo: dict, idx: int) -> list[str]:
    """把一段翻译成命令行。这里只做转写，不做判断——range、role 的合法性由被启动的
    进程自己判（split-stage 的 validate_options / llama-server 的链首检查）。"""
    stages = topo["stages"]
    st = stages[idx]
    if st["role"] == "head":
        argv = [
            exe(st, "llama-server"),
            "-m", st["model"],
            "--split-layer-range", st["layers"],
            "--split-rs-seq", str(topo["rs_seq"]),
            "-c", str(st["ctx"]), "-b", str(st["batch"]), "-np", "1",
            "--host", st["bind"], "--port", str(st["port"]),
        ]
        for later in stages[idx + 1:]:
            argv += ["--split-remote", f"{later['host']}:{later['port']}"]
    else:
        argv = [
            exe(st, "split-stage"),
            "-m", st["model"],
            "--layer-range", st["layers"],
            "--role", st["role"],
            "--rs-seq", str(topo["rs_seq"]),
            "-c", str(st["ctx"]), "-b", str(st["batch"]),
            "--host", st["bind"], "--port", str(st["port"]),
        ]
    if st.get("threads") is not None:
        argv += ["-t", str(st["threads"])]
    if st.get("ngl") is not None:
        argv += ["-ngl", str(st["ngl"])]
    argv += st["extra"]
    return argv


# ---------------------------------------------------------------- 状态文件


def state_file(topo: dict) -> Path:
    return Path(topo["run_dir"]) / "state.json"


def load_state(topo: dict) -> dict | None:
    path = state_file(topo)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_state(topo: dict, state: dict) -> None:
    Path(topo["run_dir"]).mkdir(parents=True, exist_ok=True)
    state_file(topo).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def drop_state(topo: dict) -> None:
    try:
        state_file(topo).unlink()
    except OSError:
        pass


def alive_stages(state: dict | None) -> list[dict]:
    """state 里「还活着、且确实是我们的」那些段。

    只按 pid 判活会误报：pid 被系统复用给别的进程时，up 会报「链已经在跑」并拒绝启动，
    而 state 里记的那个 pid 早就不是我们的了。所以本机段再核一次映像名。
    远端段的 pid 是本机 tasklist 看不见的编号，这里不额外 ssh 一次（up 的第一次 ssh
    要留给 remote_style 探测），仍按 pid 判活。
    """
    if not state:
        return []
    alive = []
    for s in state.get("stages", []):
        pid = int(s.get("pid") or 0)
        if not pid_alive(pid):
            continue
        if not s.get("ssh"):
            img = image_of(pid)
            if img and img.lower() not in OUR_IMAGES:
                continue
        alive.append(s)
    return alive


# ---------------------------------------------------------------- 权重的便宜自检
#
# 两段加载的不是同一份权重时，链会产出「看着通顺的垃圾」：OP_HELLO 只报 range / 宽度 /
# 深度，区分不了同形状不同权重的 GGUF。这里补一道**启发式**，不是密码学证明：
# size 不同一定不是同一份（失败）；mtime 不同只警告（scp 会改它）；sha256 只在
# --verify-weights 时做（35B 要几分钟）。同 size 同 mtime 的不同权重比不出来。


def weight_stat(st: dict) -> tuple[int, float] | None:
    if st["ssh"]:
        r = remote(st, f"stat -c '%s %Y' {shlex.quote(st['model'])} 2>/dev/null")
        parts = r.stdout.split()
        if len(parts) < 2:
            return None
        try:
            return int(parts[0]), float(parts[1])
        except ValueError:
            return None
    try:
        s = os.stat(st["model"])
        return s.st_size, s.st_mtime
    except OSError:
        return None


def weight_sha256(st: dict) -> str | None:
    if st["ssh"]:
        r = remote(st, f"sha256sum {shlex.quote(st['model'])} 2>/dev/null", timeout=3600)
        parts = r.stdout.split()
        return parts[0] if parts else None
    h = hashlib.sha256()
    try:
        with open(st["model"], "rb") as f:
            for block in iter(lambda: f.read(8 << 20), b""):
                h.update(block)
    except OSError:
        return None
    return h.hexdigest()


def check_weights(topo: dict, deep: bool) -> None:
    stages = topo["stages"]
    note("权重自检：size + mtime（启发式，不是密码学证明；同大小同 mtime 的不同权重比不出来）")
    note("  " + ("--verify-weights 已开，另做完整 sha256（35B 要几分钟）" if deep
                else "完整 sha256 要几分钟，用 --verify-weights 才做"))

    stats = []
    for st in stages:
        got = weight_stat(st)
        if got is None:
            note(f"  {st['name']:<6} 取不到 model 的 stat：{st['model']}（跳过，不阻断）")
            stats.append(None)
            continue
        stats.append(got)

    known = [(st, s) for st, s in zip(stages, stats) if s]
    if len(known) < 2:
        return

    sizes = {s[0] for _, s in known}
    if len(sizes) > 1 or len({round(s[1]) for _, s in known}) > 1:
        for st, s in known:
            note(f"  {st['name']:<6} size={s[0]:,}  mtime={time.strftime('%Y-%m-%d %H:%M', time.localtime(s[1]))}")
    if len(sizes) > 1:
        die("各段的 model 大小不一致：这不是同一份权重的不同视角，链会产出通顺的垃圾。")
    if len({round(s[1]) for _, s in known}) > 1:
        note("  警告：各段 mtime 不同。dev_copy/scp 会改 mtime，所以这不一定是错——")
        note("        但它是唯一能免费看到的「可能不是同一份文件」的信号，请确认后继续。")
    else:
        note(f"  size={known[0][1][0]:,}  mtime 一致，{len(known)} 段看起来是同一份文件。")

    if deep:
        digests = {}
        for st, _ in known:
            d = weight_sha256(st)
            digests[st["name"]] = d
            note(f"  {st['name']:<6} sha256={d or '(取不到)'}")
        vals = {d for d in digests.values() if d}
        if len(vals) > 1:
            die("各段的 model sha256 不一致：不是同一份权重。")
        note("  sha256 一致。")


def _winpath(p: str) -> str:
    return p.replace("/", "\\") if len(p) > 2 and p[1] == ":" else p


def _win_launch(topo: dict, st: dict, argv: list[str]) -> int:
    """Windows 远端起进程：写一个 .cmd（负责重定向）再交给 WMI 创建。

    nohup 在这里不行：Windows OpenSSH 会话结束会杀掉整棵进程树（实测 `nohup sleep 60 &`
    在 ssh 退出 2 秒后就没了，Linux 上活得很好）。WMI 的 Win32_Process.Create 由 WMI 服务
    创建进程，不在那个 Job 里。代价是 pid 变成一层 cmd.exe（它等 split-stage 结束），
    停止要 taskkill /T 杀树——这也让「cmd 活着 == 段活着」成立。"""
    wd = str(st.get("workdir") or ".")
    cmd_f = f"{wd.rstrip('/')}/start-{st['name']}.cmd"
    ps_f = f"{wd.rstrip('/')}/start-{st['name']}.ps1"
    body = ("@echo off\r\ncd /d " + _winpath(wd) + "\r\n"
            + " ".join(f'"{_winpath(a)}"' for a in argv)
            + " > " + _winpath(st["log"]) + " 2>&1\r\n")
    ps = ("$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
          f"-Arguments @{{CommandLine='cmd /c \"{_winpath(cmd_f)}\"'}}\n\"$($r.ProcessId)\"\n")
    remote(st, f"mkdir -p {shlex.quote(wd)} && "
               f"cat > {shlex.quote(cmd_f)} <<'QLH_EOF'\n{body}QLH_EOF\n"
               f"cat > {shlex.quote(ps_f)} <<'QLH_EOF'\n{ps}QLH_EOF")
    out = remote(st, f'powershell -NoProfile -ExecutionPolicy Bypass -File "{_winpath(ps_f)}"')
    for line in reversed(out.stdout.strip().splitlines()):
        if line.strip().isdigit():
            return int(line.strip())
    die(f"{st['name']}: WMI 没有返回 pid。远端说：{out.stdout.strip()[:300] or out.stderr.strip()[:300]}")


def spawn(topo: dict, idx: int) -> int:
    st = topo["stages"][idx]
    if st["ssh"]:
        remote_style(st)  # 先探：argv 里的二进制要不要 .exe 取决于它
    argv = stage_argv(topo, idx)
    if st["ssh"] and remote_style(st) != "posix":
        return _win_launch(topo, st, argv)
    if st["ssh"]:
        # 远端起进程：pid 走文件，不走 ssh 的 stdout——后台进程只要继承了连接的 stdout，
        # 这条 ssh 就永远不返回，而调用方在等管道 EOF。所以不捕获输出，pid 写进文件再
        # 一条短命令读回来。代价是一次往返。
        log = st["log"]
        pidfile = log + ".pid"
        # 注意 `&` 的优先级：写成 `mkdir && cd && nohup … & echo $! > pidfile` 时，
        # 整条 `mkdir && cd && nohup` 都在后台，而 echo 立刻执行——第一次跑时 workdir
        # 还不存在，重定向直接失败（实测：`run/probe.pid: No such file or directory`）。
        # `{ … & echo $! > pid; }` 把两者放进同一个已 cd 过的子 shell。
        wd = shlex.quote(str(st.get("workdir") or "."))
        script = (
            f"mkdir -p {wd} && cd {wd} && {{ nohup {shlex.join(argv)} "
            f"> {shlex.quote(log)} 2>&1 < /dev/null & echo $! > {shlex.quote(pidfile)}; }}"
        )
        try:
            subprocess.run(_ssh(st) + [script],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=30)
        except subprocess.TimeoutExpired:
            note(f"        （ssh 30 s 没返回；后台进程可能继承了它的 stdout，pid 从 pidfile 取）")
        for _ in range(10):
            r = remote(st, f"cat {shlex.quote(pidfile)} 2>/dev/null")
            pid = r.stdout.strip().splitlines()[-1].strip() if r.stdout.strip() else ""
            if pid.isdigit():
                return int(pid)
            time.sleep(0.3)
        die(f"{st['name']}: 远端没有写出 pidfile（{pidfile}）。先手工 ssh {st['ssh']} 试一下。")

    Path(topo["run_dir"]).mkdir(parents=True, exist_ok=True)
    logf = open(st["log"], "wb")
    kwargs = {"start_new_session": True} if os.name != "nt" else {}
    try:
        proc = subprocess.Popen(
            argv, cwd=str(st.get("workdir") or topo["run_dir"]),
            stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT, **kwargs,
        )
    finally:
        logf.close()
    return proc.pid


def wait_ready(topo: dict, idx: int, pid: int) -> str:
    """等这一段自己打印 ready。循环看三样：进程还活着、日志末尾有没有 ready 行、还有
    多久超时。超时按失败处理——加载失败一定伴随进程退出或日志里一行明确的错误。"""
    st = topo["stages"][idx]
    pattern = READY_HEAD if st["role"] == "head" else READY_SPLIT
    deadline = time.time() + topo["ready_timeout"]
    while time.time() < deadline:
        for line in read_log(st).splitlines():
            if pattern.search(line):
                return line.strip()
        if not stage_alive(st, pid):
            tail = [ln for ln in read_log(st, 4096).splitlines() if ln.strip()][-6:]
            die(
                f"{st['name']}（pid {pid}）在 ready 之前就退出了。日志末尾：\n  "
                + "\n  ".join(tail)
                + f"\n完整日志：{st['log']}"
            )
        time.sleep(0.4 if not st["ssh"] else 1.0)
    tail = [ln for ln in read_log(st, 4096).splitlines() if ln.strip()][-6:]
    die(
        f"{st['name']} 在 {topo['ready_timeout']:.0f} s 内没有打印 ready（pid {pid} 还活着）。\n"
        f"  日志末尾：\n  " + "\n  ".join(tail)
    )


def preflight_ports(topo: dict, kill_stale: bool, old_state: dict | None) -> None:
    """起任何进程之前先探端口：split-stage 自己 214 ms 就绑失败退出，但控制面不知道就会
    傻等 ready 直到超时，然后报一个和端口无关的错。被占就当场说清谁占的、能不能杀。"""
    old_pids = {}
    for s in (old_state or {}).get("stages", []):
        try:
            old_pids[int(s["pid"])] = s
        except (KeyError, TypeError, ValueError):
            continue

    listening = listeners()
    conflicts = []
    for st in topo["stages"]:
        # 远端段的端口只从对端 ssh 才看得准，本机 netstat 里没有它；退化成「连得上就算占」
        # 也不行（单连接服务连着时会超时）。所以远端段不做端口预检，交给 split-stage 自己
        # 在 214 ms 内绑失败——它会把 "port already in use" 打进日志。
        if st["ssh"]:
            continue
        pid = listening.get(st["port"])
        if pid is None:
            continue
        # [C5] conflicts 只在 pid 非 None 时 append，所以下面的循环不必再处理 None
        # （以前那里有一条 `if pid is None:` 的不可达分支，已删）。
        conflicts.append((st, pid, old_pids.get(pid)))

    if not conflicts:
        return

    stale = []
    for st, pid, old in conflicts:
        where = f"{st['host']}:{st['port']}（{st['name']}）"
        img = image_of(pid) or "未知映像"
        if old is not None and img.lower() in OUR_IMAGES:
            note(f"端口 {where} 被 pid {pid}（{img}）占着，**这是上次 up 记录里的 {old.get('name')}**。")
            note("  处置：up --kill-stale 会停掉它再起；或先跑 qlhctl down。")
            stale.append((pid, st))
        elif looks_like_ours(st, pid):
            note(f"端口 {where} 被 pid {pid}（{img}）占着：命令行与拓扑的 {st['name']} 一致"
                 f"（--port {st['port']} 与层范围 {st['layers']}），像是上次没清掉的。")
            note("  处置：up --kill-stale 停掉它；state 里没有它的记录，down 管不到它。")
            stale.append((pid, st))
        else:
            note(f"端口 {where} 被 pid {pid}（{img}）占着，不是本控制面起的，也不像拓扑里的任何一段"
                 f"（命令行里没有 --port {st['port']} + 层范围 {st['layers']} 这一对）。")
            note("  处置：换一个端口，或由你确认后自行结束它——控制面不静默杀别人的进程。")

    if not kill_stale:
        die("端口已被占用，未启动任何进程。")
    if not stale:
        die("端口占用者都不是本控制面起的，--kill-stale 不动它们。")
    for pid, st in stale:
        note(f"  --kill-stale：停掉 pid {pid}")
        # 带 port / layers 走完整身份判据：state 记过的 pid 也可能已经被系统复用。
        stop_pid(pid, port=st["port"], layers=st["layers"])
    time.sleep(0.3)
    listening = listeners()
    left = [st for st in topo["stages"] if not st["ssh"] and st["port"] in listening]
    if left:
        die("停掉旧进程后端口仍然被占：" + ", ".join(f"{st['name']} {st['host']}:{st['port']}" for st in left))


def cmd_up(topo: dict, args: argparse.Namespace) -> int:
    old = load_state(topo)
    running = alive_stages(old)
    if running and not args.kill_stale:
        names = ", ".join(f"{s.get('name')}(pid {s.get('pid')})" for s in running)
        die(f"这条链已经在跑：{names}。先 qlhctl down，或 up --kill-stale 重来。")

    if args.dry_run:
        for idx, st in enumerate(topo["stages"]):
            if st["ssh"]:
                remote_style(st)
            note(f"[dry-run] {st['name']:<6} {' '.join(shlex.quote(a) for a in stage_argv(topo, idx))}")
        return 0

    preflight_ports(topo, args.kill_stale, old)
    check_weights(topo, args.verify_weights)

    Path(topo["run_dir"]).mkdir(parents=True, exist_ok=True)
    state = {"chain": topo["name"], "topology": str(topo["path"]), "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": []}
    note(f"启动链 {topo['name']}（{len(topo['stages'])} 段，尾段先起）")

    started = []
    try:
        for order, idx in enumerate(range(len(topo["stages"]) - 1, -1, -1)):
            st = topo["stages"][idx]
            note(f"  [{order + 1}/{len(topo['stages'])}] {st['name']:<6} {st['layers']:<8} {st['host']}:{st['port']} ...", )
            t0 = time.time()
            pid = spawn(topo, idx)
            entry = {"name": st["name"], "role": st["role"], "layers": st["layers"],
                     "host": st["host"], "port": st["port"], "pid": pid,
                     "ssh": st["ssh"], "log": st["log"], "style": st.get("_style")}
            state["stages"].append(entry)
            started.append(entry)
            # 先落盘：起进程之后任何一步失败，down 都要能找到它。
            save_state(topo, state)
            line = wait_ready(topo, idx, pid)
            entry["ready"] = True
            entry["ready_at"] = time.strftime("%H:%M:%S")
            entry["ready_line"] = line
            save_state(topo, state)
            note(f"        ready in {time.time() - t0:.1f}s, pid {pid}: {line}")
    except SystemExit:
        note("启动失败，停掉本次已经起来的段……")
        for entry in reversed(started):
            _stop_entry(entry, quiet=True)
        drop_state(topo)
        raise

    save_state(topo, state)
    head = next((s for s in topo["stages"] if s["role"] == "head"), None)
    if head:
        note(f"链就绪：head 在 http://{head['host']}:{head['port']}")
    note(f"日志与 state：{topo['run_dir']}")
    note("（控制面不管请求；链活着之后直接问 head 的 /v1/chat/completions）")
    return 0


def probe_stage(st: dict, entry: dict | None, listening: dict[int, int]) -> dict:
    pid = int(entry.get("pid") or 0) if entry else 0
    if entry and entry.get("style"):
        st["_style"] = entry["style"]          # up 时探过的，别为 status 再付一次 ssh
    # 远端只有一次 ssh 往返用来问 pid 是否还在；端口不查（对端有没有 ss 不可假设）。
    alive = stage_alive(st, pid)
    listen = None if st["ssh"] else st["port"] in listening
    # ready 的判据是「控制面亲眼见过那行」或「日志尾部还留着那行」，不是端口通。
    # 只靠日志尾部会骗人：head 跑一会儿之后，请求日志把 ready 行挤出窗口，一段健康的
    # 链会被报成 starting。所以 up 成功时那行已经写进 state，这里先信它。
    seen = bool(entry and entry.get("ready"))
    tail = read_log(st, 4096)
    ready = seen or bool(READY_HEAD.search(tail) if st["role"] == "head" else READY_SPLIT.search(tail))
    if entry is None:
        verdict = "unknown"
    elif not alive:
        verdict = "DEAD"
    elif not ready:
        verdict = "starting"
    elif listen is False:
        verdict = "LISTEN-GONE"   # 进程活着、ready 过，内核里却没有监听：不对劲
    else:
        verdict = "ok"
    return {"name": st["name"], "role": st["role"], "layers": st["layers"],
            "addr": f"{st['host']}:{st['port']}", "pid": pid, "alive": alive,
            "ready": ready, "listen": listen, "verdict": verdict,
            "last_log": log_last_line(st) if verdict != "ok" else "",
            "ready_at": (entry or {}).get("ready_at", "")}


def http_probe(host: str, port: int, path: str, timeout: float = 20.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}{path}", timeout=timeout) as r:
            return r.status, r.read(400).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(800).decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 —— 探针失败本身就是答案
        return 0, f"{type(exc).__name__}: {exc}"


def cmd_status(topo: dict, args: argparse.Namespace) -> int:
    state = load_state(topo)
    entries = {e["name"]: e for e in (state or {}).get("stages", [])}
    listening = listeners()
    rows = [probe_stage(st, entries.get(st["name"]), listening) for st in topo["stages"]]

    if args.json:
        print(json.dumps({"chain": topo["name"], "run_dir": str(topo["run_dir"]), "stages": rows},
                         ensure_ascii=False, indent=2))
    else:
        note(f"chain {topo['name']}    run_dir {topo['run_dir']}")
        note("")
        note(f"{'stage':<8}{'role':<7}{'layers':<9}{'addr':<22}{'pid':<9}{'proc':<6}{'port':<7}{'ready':<7}verdict")
        for r in rows:
            note(f"{r['name']:<8}{r['role']:<7}{r['layers']:<9}{r['addr']:<22}{r['pid'] or '-':<9}"
                 f"{'live' if r['alive'] else 'dead':<6}"
                 f"{('open' if r['listen'] else 'shut') if r['listen'] is not None else '-':<7}"
                 f"{'yes' if r['ready'] else 'no':<7}{r['verdict']}")
            if r["last_log"]:
                note(f"         └ 日志末尾：{r['last_log']}")
                if r["verdict"] == "DEAD":
                    note("           （日志里没有退出原因：多半是被外部结束，不是它加载失败）")
        note("")
        note(f"{sum(1 for r in rows if r['verdict'] == 'ok')}/{len(rows)} 段 ok")

    if args.deep:
        head = next((st for st in topo["stages"] if st["role"] == "head"), None)
        if head is None:
            note("没有 head 段，跳过链的深度探测。")
        else:
            code, body = http_probe(head["host"], head["port"], "/health?deep=1")
            # 浅 /health 是存活语义（链死也 200）；?deep=1 才真发 1-token 请求、链死 503 并点名。
            note("")
            note(f"链探测 GET /health?deep=1 -> {code}")
            note(f"  {body.strip()[:500]}")

    return 0 if all(r["verdict"] == "ok" for r in rows) else 1


def stop_pid(pid: int, quiet: bool = False, port: int | None = None,
             layers: str | None = None) -> bool:
    """停一个本控制面起的进程。

    身份判据有两档，**不要只看 pid**（系统会复用），也**不要只看映像名**（同一台机器上
    可能还跑着别人的 llama-server.exe，名字一样但端口/层范围不同）：

      - 给了 port + layers：走 `looks_like_ours`（映像名 + 命令行里的 --port 与层范围），
        这是唯一能区分「我们的这一段」与「别人的同名进程」的判据；
      - 没给：退到「映像名在名单里」。只该用在不掌握拓扑的场合（例如 state 里没有该字段）。

    `preflight_ports` / `_stop_entry` 都掌握 port 与 layers，一律走第一档。
    """
    if port is not None and layers is not None:
        if not looks_like_ours({"port": port, "layers": layers}, pid):
            note(f"  跳过 pid {pid}：命令行里没有 --port {port} + 层范围 {layers}，"
                 f"不是这一段（pid 被复用了）。")
            return False
    else:
        img = image_of(pid)
        if img and img.lower() not in OUR_IMAGES:
            note(f"  跳过 pid {pid}：它现在是 {img}，不是本控制面起的进程（pid 被复用了）。")
            return False
    if os.name == "nt":
        r = run(["taskkill", "/F", "/PID", str(pid)], timeout=15)
    else:
        r = run(["kill", "-9", str(pid)], timeout=15)
    for _ in range(30):
        if not pid_alive(pid):
            break
        time.sleep(0.1)
    ok = not pid_alive(pid)
    if not quiet:
        note(f"  pid {pid} {'已停' if ok else '没有停下'}（{' '.join(r.stdout.split()[:4])}）")
    return ok


def _stop_entry(entry: dict, quiet: bool) -> bool:
    """停一段，并**确认它真的不在了**。远端尤其不能只看命令退出码：kill 是发的，不是等的。"""
    pid = int(entry.get("pid") or 0)
    if entry.get("ssh"):
        st = {"ssh": entry["ssh"], "log": entry.get("log", "")}
        if remote_style(st) == "posix":
            remote(st, f"kill -9 {pid} 2>/dev/null; true")
        else:
            # Windows：pid 是包着 split-stage 的 cmd.exe，杀树。
            remote(st, f"taskkill //F //T //PID {pid} 2>/dev/null")
        for _ in range(20):
            if not stage_alive(st, pid):
                return True
            time.sleep(0.2)
        return False
    # 本机段：带上 state 里记的 port / layers，用完整判据核身份（旧版只看映像名，
    # pid 复用成别人的 llama-server.exe 时会杀错）。
    return stop_pid(pid, quiet=quiet,
                    port=entry.get("port"), layers=entry.get("layers"))


def cmd_down(topo: dict, args: argparse.Namespace) -> int:
    state = load_state(topo)
    if not state:
        note(f"没有 {topo['name']} 的运行记录（{state_file(topo)} 不存在）。")
        listening = listeners()
        leftovers = [st for st in topo["stages"] if not st["ssh"] and st["port"] in listening]
        if leftovers:
            note("但下面这些端口还有监听者，不是本控制面起的，不动它们：")
            for st in leftovers:
                pid = listening.get(st["port"])
                note(f"  {st['name']:<6} {st['host']}:{st['port']}  pid={pid} img={image_of(pid) if pid else '?'}")
            return 1
        return 0

    entries = state.get("stages", [])
    note(f"停链 {topo['name']}（head 先停，共 {len(entries)} 段）")
    # 逆序：先停链首（它连着所有下游），再停 relay，最后 tail。
    all_gone = True
    for entry in reversed(entries):
        pid = int(entry.get("pid") or 0)
        if pid <= 0:
            continue
        st = {"ssh": entry.get("ssh"), "log": entry.get("log", "")}
        if not stage_alive(st, pid):
            note(f"  {entry['name']:<6} pid {pid} 已经不在（跳过）")
            continue
        note(f"  {entry['name']:<6} pid {pid} ...", )
        ok = _stop_entry(entry, quiet=True)
        all_gone = all_gone and ok
        note("         已停" if ok else "         仍在（它没停；down 不假装成功）")

    if all_gone:
        drop_state(topo)

    time.sleep(0.3)
    listening = listeners()
    # 本机看端口；远端只能再问一次 pid（对端有没有 ss 不可假设）。
    still = [st for st in topo["stages"]
             if (not st["ssh"] and st["port"] in listening)
             or (st["ssh"] and any(e["name"] == st["name"] and stage_alive(
                 {"ssh": e.get("ssh"), "log": e.get("log", "")}, int(e.get("pid") or 0)) for e in entries))]
    if still:
        note("仍有残留（端口在监听，或远端进程没停）：")
        for st in still:
            pid = listening.get(st["port"])
            note(f"  {st['name']:<6} {st['host']}:{st['port']}  本机pid={pid or '-'}")
        return 1
    note(f"全部停掉（{len(topo['stages'])} 段），端口释放，state 已清。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qlhctl",
        description=__doc__.split("\n\n")[0],
        epilog="本工具只做进程与端口编排：不实现链协议、不碰 token、不理解层、不采样。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_up = sub.add_parser("up", help="按拓扑把链拉起来（尾段先，链首最后）")
    p_up.add_argument("--kill-stale", action="store_true",
                      help="停掉「上次 up 记录过的、还占着端口」的旧进程；不碰别的进程")
    p_up.add_argument("--verify-weights", action="store_true",
                      help="另做完整 sha256（35B 上几分钟）；默认只比 size + mtime")
    p_up.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令行，不启动")

    p_st = sub.add_parser("status", help="报每段的 pid / 端口 / ready，以及链的深度探测")
    p_st.add_argument("--deep", action="store_true",
                      help="另问 head 的 /health?deep=1（链死时 503 并点名哪段）；会清掉该 slot 的 KV")
    p_st.add_argument("--json", action="store_true")

    p_dn = sub.add_parser("down", help="停掉这次 up 起来的进程，并复核端口已释放")
    for p in (p_up, p_st, p_dn):
        p.add_argument("--topology", "-f", default=argparse.SUPPRESS,
                       help="拓扑 TOML（也可设 QLH_TOPOLOGY）")

    args = parser.parse_args(argv)
    given = getattr(args, "topology", None) or os.environ.get("QLH_TOPOLOGY")
    if not given:
        die("没有给拓扑：-f control/topology.example.toml（或设 QLH_TOPOLOGY）")
    topo_path = Path(given)
    topo = load_topology(topo_path)
    return {"up": cmd_up, "status": cmd_status, "down": cmd_down}[args.cmd](topo, args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    raise SystemExit(main())
