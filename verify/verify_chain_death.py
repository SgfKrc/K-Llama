#!/usr/bin/env python3
"""远端 stage 消失时的验收：head 必须活着、失败是请求级的、deep health 分得清三种状态。

为什么单独一个脚本，而不是 verify.sh 的第十组：这里真的要挂起一个 stage 进程
（NtSuspendProcess）再等它自己的 120 s 读超时，跑一趟约 4-5 分钟；verify.sh 的九组都是秒级
的，混在一起会让人不敢跑。它也真的要用到 120 s 超时那一层行为，不是能用 mock 代替的。

    python verify/verify_chain_death.py

验收内容（对应 docs/limits/SEGMENTED-LIMITS.md §2.7 记的那一条）：
  A. 挂起 stage 后：/completion 在 ~120 s 失败，但 head 进程活着、/health 仍 200
     （修改前是 GGML_ABORT，整个进程退出）
  B. 链被标记死之后：后续请求立即 500，不再等 120 s；/health?deep=1 立即 503
  C. 错误信息可诊断：哪一段、endpoint、seq、范围，以及「需要重建链」
  D. 自愈：只重启 stage、**不重启 head**，下一个请求就成功，且 prompt_n 等于完整 prompt
     长度、cache_n = 0、输出与链从未断过时逐字节相同（这三条是「slot 被正确重置」的证据）
  E. /health 的默认语义不变（挂起期间依然 200），/health?deep=1 在 125 s 内 503
  F. 链死期间：5 个连续请求只触发 1 次重连（5 s 节流），其余立即失败，且没有一个卡在
     120 s 读超时或 connect 超时上
  G. 多 slot：远程链单序列是设计决定，`-np 2 + --split-remote` 必须在启动时被拒

D 组以前验证的是「只重启 stage 不自愈」（请求仍 500）；自愈之后这条结论反过来了，脚本按新
行为改写：同一个动作（只重启 stage）现在必须成功，且成功的方式必须是「整段 prompt 重新
prefill」，不是「从旧 cache 接着算」。

跑完会杀掉自己启动的所有进程，并清掉 93xx 上的监听；9103/9104/9106 上的服务不碰。
"""

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _qlh_env import BIN, ROOT, TMP, require   # noqa: E402
MODEL     = os.environ.get("QLH_SPLIT_MODEL", r"H:/qlh_models/qwen3-5-2b-gguf/qwen35-2b-Q4_K_M.gguf")
HEAD_PORT = int(os.environ.get("QLH_HEAD_PORT", "9330"))
STAGE_PORT = int(os.environ.get("QLH_STAGE_PORT", "9331"))
HEAD_LOG  = os.path.join(TMP, "chain_death_head.log")
STAGE_LOG = os.path.join(TMP, "chain_death_stage.log")

PASS = 0
FAIL = 0


def ok(msg):
    global PASS
    PASS += 1
    print("  [PASS] " + msg)


def bad(msg):
    global FAIL
    FAIL += 1
    print("  [FAIL] " + msg)


def hdr(msg):
    print("\n=== %s ===" % msg)


# --------------------------------------------------------------------------- processes

def pid_alive(pid):
    """进程还活着吗。不能用 os.kill(pid, 0)：在 Windows 上 CPython 把它实现成 TerminateProcess。"""
    if pid <= 0:
        return False
    k32 = ctypes.windll.kernel32
    handle = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == 259                   # STILL_ACTIVE
    finally:
        k32.CloseHandle(handle)


def check_lock():
    """与 verify.sh / verify_dflash_dev_path.sh 互斥。

    三方都用 93xx，而本脚本的 kill_93xx() 是按端口无差别清场：并发跑时会把对方正在用的
    stage 当成自己的残留杀掉，两边各报一堆假失败（verify.sh 头注释与 docs/design/TESTING.md §6.1
    都记了这个坑）。持锁方的 pid 还活着就拒绝启动；留下的是死锁文件则直接接管。
    """
    lock = os.path.join(TMP, ".verify.lock")
    if not os.path.isdir(lock):
        return
    pid = 0
    try:
        with open(os.path.join(lock, "pid"), encoding="utf-8", errors="replace") as f:
            pid = int(f.read().strip() or 0)
    except (OSError, ValueError):
        pid = 0
    if pid and pid_alive(pid):
        sys.stderr.write(
            "另一个验收正在跑（%s，pid %d）。\n"
            "并发跑会互抢 93xx 并互相强杀，两边都会报假失败；等它结束再跑，\n"
            "或确认没有别的验收时删掉这个目录。\n" % (lock, pid))
        raise SystemExit(2)
    shutil.rmtree(lock, ignore_errors=True)


def kill_93xx():
    """只清 93xx 上的监听；9103/9104/9106 上的服务绝不碰（与 verify.sh 的 kill_ours 同一条规则）。"""
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
    pids = set()
    for line in out.splitlines():
        if "LISTENING" not in line:
            continue
        m = re.search(r":(93\d\d)\s", line)
        if not m:
            continue
        pids.add(line.split()[-1])
    for pid in pids:
        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
    if pids:
        time.sleep(2)


def start_stage():
    log = open(STAGE_LOG, "w", encoding="utf-8", errors="replace")
    p = subprocess.Popen(
        [os.path.join(BIN, "split-stage.exe"),
         "-m", MODEL, "--layer-range", "20:24", "--role", "tail",
         "--rs-seq", "8", "--port", str(STAGE_PORT), "-c", "4096", "-b", "256", "-v"],
        stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    for _ in range(120):
        if p.poll() is not None:
            break
        try:
            with open(STAGE_LOG, "r", encoding="utf-8", errors="replace") as f:
                # [v11] 端口 bind 在模型加载之前（v10），"listening" 只说明端口是它的；ready: 才是
                # 加载完、可以接帧的信号。等错信号会让 head 连一个还在加载的 stage，最后报成链不对。
                if "ready:" in f.read():
                    return p
        except OSError:
            pass
        time.sleep(0.5)
    return p


def start_head():
    log = open(HEAD_LOG, "w", encoding="utf-8", errors="replace")
    p = subprocess.Popen(
        [os.path.join(BIN, "llama-server.exe"),
         "-m", MODEL, "--host", "127.0.0.1", "--port", str(HEAD_PORT),
         "-c", "4096", "-ngl", "0", "--no-warmup",
         "--split-layer-range", "0:20", "--split-remote", "127.0.0.1:%d" % STAGE_PORT],
        stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    return p


def wait_health(timeout_s=180):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            code, _ = http_get("http://127.0.0.1:%d/health" % HEAD_PORT, timeout=2)
            if code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def killp(p):
    if p is None:
        return
    try:
        p.terminate()
    except Exception:
        pass
    try:
        p.wait(timeout=10)
    except Exception:
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(p.pid)], capture_output=True)
        except Exception:
            pass


def suspend(pid):
    ntdll = ctypes.WinDLL("ntdll")
    h = ctypes.windll.kernel32.OpenProcess(0x1F0FFF, False, pid)
    if not h:
        return -1
    st = ntdll.NtSuspendProcess(h)
    ctypes.windll.kernel32.CloseHandle(h)
    return st


def resume(pid):
    ntdll = ctypes.WinDLL("ntdll")
    h = ctypes.windll.kernel32.OpenProcess(0x1F0FFF, False, pid)
    if not h:
        return -1
    st = ntdll.NtResumeProcess(h)
    ctypes.windll.kernel32.CloseHandle(h)
    return st


# --------------------------------------------------------------------------- HTTP

def http_get(url, timeout=5):
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        # 连接被拒 == head 已经死了，这本身就是要报的结论，不要让异常打断脚本
        return 0, str(e)


def completion(timeout=300, max_tokens=8):
    """一个最小 /completion；返回 (http_code, body, seconds)。"""
    body = json.dumps({
        "prompt": "The capital of France is",
        "n_predict": max_tokens,
        "temperature": 0,
        "cache_prompt": True,
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%d/completion" % HEAD_PORT, data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace"), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace"), time.time() - t0
    except Exception as e:
        return 0, str(e), time.time() - t0


def readlog(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def main():
    require("split-stage.exe", "llama-server.exe")
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="结束时不要清理进程（调试用：保留现场）")
    args = ap.parse_args()
    check_lock()

    stage = None
    head = None
    started = []

    try:
        hdr("准备")
        kill_93xx()
        stage = start_stage()
        started.append(stage)
        if stage.poll() is not None:
            bad("split-stage 没起来，见 %s" % STAGE_LOG)
            return 1
        ok("split-stage 起来（pid=%d, port=%d）" % (stage.pid, STAGE_PORT))

        head = start_head()
        started.append(head)
        if not wait_health(180):
            bad("head 没就绪，见 %s" % HEAD_LOG)
            return 1
        ok("head 起来（pid=%d, port=%d）" % (head.pid, HEAD_PORT))

        # ---------------------------------------------------------------- 基线
        hdr("基线：链路正常")
        code, body = http_get("http://127.0.0.1:%d/health" % HEAD_PORT)
        shallow_ok = code == 200 and json.loads(body) == {"status": "ok"}
        if shallow_ok:
            ok("/health -> 200 %s（默认语义与原样一字不差）" % body.strip())
        else:
            bad("/health -> %d %s（期望 200 且 body 只有 status）" % (code, body[:200]))

        code, body = http_get("http://127.0.0.1:%d/health?deep=1" % HEAD_PORT, timeout=130)
        try:
            deep = json.loads(body)
        except Exception:
            deep = {}
        if code == 200 and deep.get("deep") == "probed":
            ok("/health?deep=1 -> 200（真的驱动了一个 token 过整条链）")
        else:
            bad("/health?deep=1 -> %d %s" % (code, body[:300]))

        code, body, dt = completion()
        baseline_ok = code == 200 and len(body) > 0
        if baseline_ok:
            ok("/completion -> 200（%.2f s）" % dt)
        else:
            bad("/completion -> %d（%.2f s）%s" % (code, dt, body[:300]))

        # D 组要拿这三个数做对照：完整 prompt 的 token 数、第一次的输出、以及基线自己用不用
        # 缓存。基线和自愈后的请求必须都是「从 0 开始 prefill 同一个 prompt」，所以 prompt_n
        # 应该完全相等——这就是「slot 被正确重置」而不是「从旧 cache 接着算」的直接证据。
        base_rec = {}
        try:
            base_rec = json.loads(body)
        except Exception:
            pass
        base_tim      = base_rec.get("timings") or {}
        base_prompt_n = base_tim.get("prompt_n")
        base_cache_n  = base_tim.get("cache_n")
        base_content  = base_rec.get("content")
        print("  [info] 基线：prompt_n=%s cache_n=%s content=%r"
              % (base_prompt_n, base_cache_n, (base_content or "")[:60]))

        # ---------------------------------------------------------------- A 挂起 stage
        hdr("A. 挂起 stage：请求失败，head 必须活着")
        st = suspend(stage.pid)
        print("  [info] NtSuspendProcess status=0x%08X" % (st & 0xFFFFFFFF))
        if st != 0:
            bad("挂起失败")
            return 1

        code, body = http_get("http://127.0.0.1:%d/health" % HEAD_PORT)
        if code == 200:
            ok("挂起期间 /health 仍 200（默认语义不变：不因为链路抖动摘机器）")
        else:
            bad("挂起期间 /health -> %d（期望仍是 200）" % code)

        t0 = time.time()
        code, body, dt = completion(timeout=300)
        head_alive = head.poll() is None
        code_after, _ = http_get("http://127.0.0.1:%d/health" % HEAD_PORT)
        print("  [info] /completion -> %d in %.1f s；head.poll()=%s；之后 /health=%d"
              % (code, dt, head.poll(), code_after))
        if code == 500:
            ok("请求级失败：/completion -> 500（不是进程死）")
        else:
            bad("/completion -> %d（期望 500；0 表示连接被拒，说明 head 死了）" % code)
        if head_alive and code_after == 200:
            ok("head 进程存活，/health 仍可连（实测问题里这里是 GGML_ABORT、端口消失）")
        else:
            bad("head 进程不在了（poll=%s, /health=%s）" % (head.poll(), code_after))
        if dt < 180:
            ok("失败发生在 stage 的 120 s 读超时之后、180 s 之内（%.1f s）" % dt)
        else:
            bad("失败用时 %.1f s，比超时上限还长" % dt)

        # ---------------------------------------------------------------- B 链死后快速失败
        hdr("B. 链已标记死：后续请求与 deep health 都必须立即失败")
        code, body, dt = completion(timeout=60)
        if code == 500 and dt < 10:
            ok("紧接着的 /completion -> 500 in %.2f s（不再等 120 s）" % dt)
        else:
            bad("紧接着的 /completion -> %d in %.2f s（期望 500 且 < 10 s）" % (code, dt))

        t_deep = time.time()
        code, body = http_get("http://127.0.0.1:%d/health?deep=1" % HEAD_PORT, timeout=130)
        dt = time.time() - t_deep
        try:
            deep = json.loads(body)
        except Exception:
            deep = {}
        if code == 503 and dt < 10:
            ok("/health?deep=1 -> 503 in %.2f s（链已标记死，不等超时）" % dt)
        else:
            bad("/health?deep=1 -> %d in %.2f s（期望 503 且 < 10 s）" % (code, dt))
        msg = json.dumps(deep, ensure_ascii=False)
        if "127.0.0.1:%d" % STAGE_PORT in msg:
            ok("503 body 指名了出问题的 endpoint：%s" % msg[:180])
        else:
            bad("503 body 没写 endpoint：%s" % msg[:300])

        code, body = http_get("http://127.0.0.1:%d/health" % HEAD_PORT)
        if code == 200:
            ok("链死后 /health（浅）仍 200：进程活着，链路状态由 deep=1 报告")
        else:
            bad("链死后 /health -> %d" % code)

        # ---------------------------------------------------------------- C 日志可诊断
        hdr("C. 日志可诊断：哪一段 / endpoint / 原因 / 需要重建")
        log = readlog(HEAD_LOG)
        ep = r"remote stage 1 \(127\.0\.0\.1:%d\)" % STAGE_PORT
        need = [
            ("哪一段 + endpoint", ep),
            # 这条请求走的是 checkpoint 恢复（槽里已经有缓存前缀），现场日志里的第一条就是这个
            ("具体失败的操作（恢复 / 载入）", r"failed the state (restore|load)"),
            ("链不可信的原因", r"the split chain is no longer trustworthy"),
            # 自愈之后这条措辞变了：不再是「重启 server」，而是「等到一次重建成功」，并说明
            # 重建会在下一个从位置 0 开始的请求上自动发生。
            ("需要重建链（且说清楚重建是自动的）", r"until a rebuild succeeds"),
        ]
        for label, pat in need:
            if re.search(pat, log):
                ok("日志里有「%s」" % label)
            else:
                bad("日志里缺「%s」（pattern=%s）" % (label, pat))
        if re.search(r"GGML_ABORT|failed to remove sequence 0 with p0=0|failed to restore context checkpoint\n", log):
            bad("日志里仍有 abort 痕迹（GGML_ABORT / common.cpp:1764 / 恢复 checkpoint 那条）")
        else:
            ok("日志里没有 GGML_ABORT / common.cpp / 恢复 checkpoint 的 abort 行")

        # ---------------------------------------------------------------- D 自愈
        hdr("D. 自愈：只重启 stage，不重启 head，链自己重建")
        killp(stage)
        started.remove(stage)
        time.sleep(1)
        stage = start_stage()
        started.append(stage)
        if stage.poll() is not None:
            bad("stage 重启失败")
            return 1

        # 重连节流是 5 s（src/llama-memory-split.cpp 的 kRebuildThrottleMs）。A 组那次真实尝试
        # 已经过去很久，但这里还是先把窗口等过去：本组量的是「自愈能不能成」，节流由 F 组
        # 单独考，别把两件事混在一条失败里。
        time.sleep(6)

        # 注意：这里**不重启 head**。这正是自愈要证明的事——stage 回来之后链自己重建。
        t0 = time.time()
        code, body, dt = completion(timeout=300)
        if code == 200:
            ok("只重启 stage、不重启 head：/completion -> 200（%.2f s，整段 prompt 重新 prefill）" % dt)
        else:
            bad("只重启 stage：/completion -> %d (%.2f s) %s" % (code, dt, body[:300]))

        rec = {}
        try:
            rec = json.loads(body)
        except Exception:
            pass
        tim      = rec.get("timings") or {}
        prompt_n = tim.get("prompt_n")
        cache_n  = tim.get("cache_n")
        content  = rec.get("content")

        # 「slot 被正确重置」的直接证据：自愈后第一个请求必须从 0 重新 prefill 整段 prompt。
        # 如果它从旧 cache 的 N 接着算，而链上每个段的 KV 都是空的，位置就会接着走而 KV 缺
        # 前面 N 个 cell——那正是这次改动绝不允许出现、也最不容易被看出来的错误。
        if base_prompt_n is not None and prompt_n == base_prompt_n and (prompt_n or 0) > 1:
            ok("自愈后第一个请求 prompt_n = %s，等于完整 prompt 长度（基线 %s）" % (prompt_n, base_prompt_n))
        else:
            bad("自愈后 prompt_n = %s（基线 %s）：slot 的 cache 没被正确重置" % (prompt_n, base_prompt_n))
        if cache_n == 0:
            ok("同一请求 cache_n = 0：旧 cache 真的被清掉了，不是接着算")
        else:
            bad("自愈后 cache_n = %s（期望 0：一个 token 都不该复用）" % cache_n)

        # 重新 prefill 出来的一定要和链从未断过时逐字节相同：自愈可以慢，不能算错。
        if base_content and content == base_content:
            ok("输出与链从未断过时逐字节相同（%d 字符）" % len(content))
        else:
            bad("输出与基线不同：%r vs %r" % ((content or "")[:120], (base_content or "")[:120]))

        log = readlog(HEAD_LOG)
        if re.search(r"rebuild attempt #\d+ succeeded", log):
            ok("head 日志里有重建成功那一条：rebuild attempt #N succeeded")
        else:
            bad("head 日志里没有重建成功的记录（自愈可能走了别的路径，或根本没发生）")

        code, body = http_get("http://127.0.0.1:%d/health?deep=1" % HEAD_PORT, timeout=130)
        try:
            deep = json.loads(body)
        except Exception:
            deep = {}
        if code == 200 and deep.get("deep") == "probed":
            ok("/health?deep=1 -> 200（重建后的链被真的探测过）")
        else:
            bad("/health?deep=1 -> %d %s" % (code, body[:300]))

        # ---------------------------------------------------------------- E 活的半死链：deep 在 125 s 内 503
        hdr("E. 新挂起（链还没被标记死）：deep=1 必须在 125 s 内 503")
        st = suspend(stage.pid)
        if st != 0:
            bad("第二次挂起失败")
            return 1
        code, body = http_get("http://127.0.0.1:%d/health" % HEAD_PORT)
        if code == 200:
            ok("挂起后 /health 仍 200")
        else:
            bad("挂起后 /health -> %d" % code)

        t0 = time.time()
        t_deep = time.time()
        code, body = http_get("http://127.0.0.1:%d/health?deep=1" % HEAD_PORT, timeout=200)
        dt = time.time() - t_deep
        if code == 503 and dt <= 125:
            ok("/health?deep=1 -> 503 in %.1f s（等到了 stage 自己的 120 s 超时，没有一直挂着）" % dt)
        else:
            bad("/health?deep=1 -> %d in %.1f s（期望 503 且 <= 125 s）" % (code, dt))
        print("  [info] body: %s" % body[:200])

        t_deep = time.time()
        code, body = http_get("http://127.0.0.1:%d/health?deep=1" % HEAD_PORT, timeout=130)
        dt = time.time() - t_deep
        # 自愈之后这里不再是「立即返回 0.02 s」：链死的状态下会先做一次有界重连尝试
        # （connect + clear 各 1.5 s 上限）。要求仍然远小于 120 s，且只试一次。
        if code == 503 and dt < 10:
            ok("再问一次 -> 503 in %.2f s（已标记死：只花一次有界重连尝试，不等 120 s）" % dt)
        else:
            bad("再问一次 -> %d in %.2f s（期望 503 且 < 10 s）" % (code, dt))

        # 探针本身是一次 cache_prompt=false 的请求，所以它的第一步就是 seq_rm(0,0,-1)：这条
        # 日志带着 seq 和范围，是「哪一段 / endpoint / seq / 范围」四件套里缺的那一半。
        log = readlog(HEAD_LOG)
        if re.search(r"remote stage 1 \(127\.0\.0\.1:%d\) failed to remove seq 0 positions \[0, -1\)"
                     % STAGE_PORT, log):
            ok("日志里有 seq_rm 那一半诊断：stage / endpoint / seq / 范围")
        else:
            bad("日志里缺 seq_rm 的 seq/范围诊断")
        if re.search(r"failed to remove sequence 0 with p0=0", log):
            bad("日志里出现了 common.cpp 的 abort 行")
        else:
            ok("日志里没有 common.cpp:1764 的 abort 行")

        # ---------------------------------------------------------------- F 节流 / 快速失败
        hdr("F. 链死期间：重连有节流，请求不卡在超时上")
        # E 组最后那次探针刚做过一次真实重连尝试；先等节流窗口过去，这一组量到的才是
        # 「一次尝试 + 若干次抑制」，而不是「全都被抑制所以看不出节流是否生效」。
        time.sleep(6.5)
        before = readlog(HEAD_LOG)
        t0 = time.time()
        results = [completion(timeout=60) for _ in range(5)]
        elapsed = time.time() - t0
        time.sleep(1)  # head 的日志由工作线程写盘，读之前让最后几行落下来
        after = readlog(HEAD_LOG)
        delta = after[len(before):] if after.startswith(before) else after

        attempts   = len(re.findall(r"rebuild attempt #\d+: the chain was dead", delta))
        suppressed = len(re.findall(r"rebuild suppressed", delta))
        codes = [c for c, _, _ in results]
        times = [d for _, _, d in results]
        print("  [info] codes=%s 耗时=%s 合计 %.2f s"
              % (codes, ["%.2f" % d for d in times], elapsed))
        print("  [info] 日志计数：rebuild attempt=%d, rebuild suppressed=%d" % (attempts, suppressed))

        if codes == [500] * 5:
            ok("5/5 全部 500（链仍死：没有任何一个请求被当成成功）")
        else:
            bad("5 个请求的返回码是 %s（期望全 500）" % codes)
        if max(times) < 10:
            ok("最慢一个 %.2f s：远小于 120 s 读超时，也没有卡在 connect 上" % max(times))
        else:
            bad("最慢一个 %.2f s：请求卡在超时上了" % max(times))
        fast = sum(1 for d in times if d < 1.0)
        if fast >= 4:
            ok("5 个里有 %d 个 < 1 s：被节流的请求是立即失败的" % fast)
        else:
            bad("只有 %d 个请求 < 1 s：每个请求都在重连，节流没生效" % fast)
        if 1 <= attempts <= 2:
            ok("5 个请求只触发 %d 次重连尝试（不是 5 次）" % attempts)
        else:
            bad("5 个请求触发了 %d 次重连尝试（期望 1~2 次）" % attempts)
        if suppressed >= 3:
            ok("另外 %d 次被明确记成 rebuild suppressed（日志可数）" % suppressed)
        else:
            bad("日志里只有 %d 条 rebuild suppressed（期望 >= 3）" % suppressed)

        # ---------------------------------------------------------------- G 多 slot
        hdr("G. 多 slot：远程链单序列，-np 2 必须在启动时被拒")
        # 第 5 条验收的答案不是一个测试而是一条设计边界：split_attach_remote 拒绝
        # n_seq_max > 1，所以远程链上根本不存在「一个 slot 重建、另一个跟着错乱」的形态。
        # 把「不存在」变成一条可重跑的结果，比在汇报里口头断言可靠。
        np2_log = os.path.join(TMP, "chain_death_np2.log")
        np2_port = HEAD_PORT + 5  # 9335：不碰 9331 的 stage 端口
        with open(np2_log, "w", encoding="utf-8", errors="replace") as f:
            p = subprocess.Popen(
                [os.path.join(BIN, "llama-server.exe"),
                 "-m", MODEL, "--host", "127.0.0.1", "--port", str(np2_port),
                 "-c", "4096", "-ngl", "0", "--no-warmup", "-np", "2",
                 "--split-layer-range", "0:20",
                 "--split-remote", "127.0.0.1:%d" % STAGE_PORT],
                stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
        rc = None
        try:
            rc = p.wait(timeout=180)
        except subprocess.TimeoutExpired:
            p.terminate()
        txt = readlog(np2_log)
        print("  [info] rc=%s" % rc)
        if rc not in (None, 0) and "single-sequence by design" in txt:
            ok("-np 2 + --split-remote 被拒（rc=%s），理由写明 single-sequence by design" % rc)
        else:
            bad("-np 2 + --split-remote 没被拒（rc=%s，日志见 %s）" % (rc, np2_log))

    finally:
        if args.keep:
            # --keep 以前是死参数：finally 无条件清理，想留现场留不住。
            hdr("保留现场（--keep）")
            print("  [keep] 不清理进程：stage pid=%s，head pid=%s"
                  % (stage.pid if stage is not None else None,
                     head.pid if head is not None else None))
            print("  [keep] 手工收：taskkill /F /PID <pid>；或按 93xx 端口清（别碰 9103/9104/9106）")
        else:
            hdr("清理")
            for p in list(started):
                killp(p)
            kill_93xx()
            left = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
            still = [l for l in left.splitlines() if "LISTENING" in l and re.search(r":(93\d\d)\s", l)]
            if still:
                bad("93xx 还有监听：%s" % still[:3])
            else:
                ok("93xx 无残留监听（9103/9104/9106 未被触碰）")

    hdr("结果")
    print("  PASS=%d  FAIL=%d" % (PASS, FAIL))
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
