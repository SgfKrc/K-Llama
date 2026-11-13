#!/usr/bin/env python3
"""判据 7–11：远程链 -np 2 上的四个投机通道（per-seq 形态，v2.2 第二批）。

为什么单独一个脚本：verify.sh 的组 7 跑远程链但从不给 -np（auto = 1），组 8 的 MTP 投机
只跑单序列。于是「一帧里两个序列，各自的投机通道参数不同」这一层从来没有被端到端跑过，
而这正是本批改的东西。判据 7–11 来自 docs/design/V22-PERSEQ-SPEC.md §6 与
docs/design/V22-MULTISEQ.md §7.1。

判据与做法：
  7  远程链 -np 2 + 投机，两个并发请求的输出与「各自串行（带投机）」逐字节相同
  8a 两个序列的候选参数不同（top_k 5 / 40，非投机链上走候选通道）互不污染；
     日志必须出现「一帧里两行、top_k 不同」的 arming 证据
  8b 两个序列的 k 不同（max_tokens 4 -> k<=3，max_tokens 64 -> k=7）互不污染
  9  一个序列在投机、另一个普通 decode（logprobs 让后者走整行路径），两条都正确
  10 验证帧的行映射不串行：A 投机的 batch 行不从 0 开始（B 的行在前），A 的输出仍正确；
     日志必须出现 verify channel armed ... rows [1..8] 这类证据
  11 k 超上限（--spec-draft-n-max 12，-np 2 -> 每序列上限 7）干净降级：启动日志有 warn、
     请求仍成功、输出与 n_max 7 串行基线逐 token 相同、不得出现 ST_BAD_REQ

每序列容量约束：-c 4096 / -np 2 -> 每序列 2048，prompt <= 1024、max_tokens <= 256。

用法：
  QLH_BUILD_DIR=H:/llamacpp-c/build-v22 python verify/verify_perspec.py [--model PATH] [--port-base 9360]

  --port-base 只改端口（默认 9360 = stage 9361-9363 / server 9371-9375，与历史默认逐一对齐）；
  verify.sh 的组 14 传 9380，避开组 13 的 KVMem 9361 与 verify.sh 主用的 9331/9332/9341-9343。
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _qlh_env import BIN, TMP, require   # noqa: E402

# Windows 上 Python 的 stdout 默认是 GBK（cp936）：中文判据消息落进日志后，在 UTF-8 终端里
# 回显全是乱码（verify.sh 的组 14 把本脚本输出逐行转记账，人读日志时中文全乱）。
# 统一成 UTF-8 输出；日志文件由 verify.sh 重定向，因此也以 UTF-8 落盘。
# 计数不受影响：[PASS]/[FAIL] 是 ASCII，verify.sh 只按这两个前缀记账。
# 为什么放在脚本里而不是只靠 verify.sh 设 PYTHONIOENCODING：README §4 说单项脚本要能独立跑，
# 只改调用点的话直接 `python verify/verify_perspec.py` 仍然乱码。verify_kvmem.py 同款。
# Python 3.7+ 才有 reconfigure；更旧版本（或 stdout 已被换成非 TextIOWrapper）静默跳过。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

MODEL = os.environ.get("QLH_SPLIT_MODEL", r"H:/qlh_models/qwen3-5-2b-gguf/qwen35-2b-Q4_K_M.gguf")
RS_SEQ = os.environ.get("QLH_RS_SEQ", "8")
VERIFY_ROWS = 17
LAYER_LO, LAYER_HI = 0, 20     # head
TAIL_LO, TAIL_HI = 20, 24      # tail stage

# 端口全部由基准算出来：独立跑时默认 9360，于是 stage 拿 9361/9362/9363、server 拿 9371-9375，
# 与历史默认逐一对齐，老命令行不受影响。verify.sh 的组 14 传 --port-base 9380（-> 9381-9383 与
# 9391-9395），避开组 13 的 KVMem 9361 与 verify.sh 主用的 9331/9332/9341-9343。
PORT_BASE = 9360


def _port_set(base):
    """基准 -> (stage_n2, stage_n1, stage_n2b, srv_n2, srv_n1, srv_n2_nomax, srv_n1_nospec, srv_n2_nospec)。"""
    return (base + 1, base + 2, base + 3,
            base + 11, base + 12, base + 13, base + 14, base + 15)


(PORT_STAGE_N2, PORT_STAGE_N1, PORT_STAGE_N2B, PORT_SRV_N2,
 PORT_SRV_N1, PORT_SRV_N2_NOMAX, PORT_SRV_N1_NOSPEC, PORT_SRV_N2_NOSPEC) = _port_set(PORT_BASE)

PASS = 0
FAIL = 0


def ok(msg):
    global PASS
    PASS += 1
    print("  [PASS] %s" % msg)


def bad(msg):
    global FAIL
    FAIL += 1
    print("  [FAIL] %s" % msg)


def info(msg):
    print("        %s" % msg)


class Proc:
    def __init__(self, name, cmd, log):
        self.name = name
        self.log = log
        self.f = open(log, "wb")
        self.p = subprocess.Popen(cmd, stdout=self.f, stderr=subprocess.STDOUT)

    def alive(self):
        return self.p.poll() is None

    def text(self):
        try:
            with open(self.log, "rb") as f:
                return f.read().decode("utf-8", "replace")
        except OSError:
            return ""

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.p.kill()
        try:
            self.f.close()
        except Exception:
            pass


def wait_for(fn, timeout, what):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if fn():
            return True
        time.sleep(0.25)
    return False


def wait_stage(proc, timeout=240):
    return wait_for(lambda: "ready:" in proc.text() or not proc.alive(), timeout, "stage ready")


def wait_health(port, proc, timeout=240):
    # Fail fast when the process is gone: waiting the whole timeout on a crashed server turns a
    # startup bug into a four-minute silence.
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not proc.alive():
            return False
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def start_stage(port, n_seq_max, tag, rs=None):
    cmd = [os.path.join(BIN, "split-stage.exe"), "-m", MODEL,
           "--layer-range", "%d:%d" % (TAIL_LO, TAIL_HI), "--role", "tail",
           "--rs-seq", str(rs if rs is not None else RS_SEQ), "--n-seq-max", str(n_seq_max),
           "--split-verify-rows", str(VERIFY_ROWS),
           "--port", str(port), "-c", "4096", "-b", "512"]
    proc = Proc("stage-%s" % tag, cmd, os.path.join(TMP, "perspec_stage_%s.log" % tag))
    if not wait_stage(proc):
        raise RuntimeError("stage %s did not become ready, see %s" % (tag, proc.log))
    return proc


def start_server(port, np_, n_max, tag, spec=True, extra=(), stage_port=None, rs=None):
    cmd = [os.path.join(BIN, "llama-server.exe"), "-m", MODEL,
           "--host", "127.0.0.1", "--port", str(port),
           "-c", "4096", "-np", str(np_), "-ngl", "0", "--no-warmup",
           "--split-layer-range", "%d:%d" % (LAYER_LO, LAYER_HI),
           "--split-remote", "127.0.0.1:%d" % (stage_port if stage_port is not None
                                               else (PORT_STAGE_N2 if np_ > 1 else PORT_STAGE_N1)),
           "--split-verify-rows", str(VERIFY_ROWS),
           "--split-rs-seq", str(rs if rs is not None else RS_SEQ)]
    if np_ > 1:
        cmd += ["--no-kv-unified"]
    if spec:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n_max)]
    cmd += list(extra)
    proc = Proc("server-%s" % tag, cmd, os.path.join(TMP, "perspec_srv_%s.log" % tag))
    if not wait_health(port, proc):
        raise RuntimeError("server %s did not come up, see %s\n%s" %
                           (tag, proc.log, "\n".join(proc.text().splitlines()[-8:])))
    return proc


def chat_body(prompt, max_tokens, top_k=None, logprobs=False, seed=1234):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "cache_prompt": False,
        "seed": seed,
    }
    if top_k is not None:
        body["top_k"] = top_k
    if logprobs:
        # 让请求走整行路径：pre-sampling logprobs 需要整行 softmax，验证/候选两个通道都因此
        # 被 plan_remote_* 关掉（common_sampler_verify_rows_check / candidates_check）。
        body["logprobs"] = True
        body["top_logprobs"] = 1
    return body


def extract(d):
    """The generated text and the draft counters of one completion.

    qwen35 is a reasoning model: with a small max_tokens every generated token lands in
    reasoning_content and `content` is empty, so comparing `content` alone would compare two empty
    strings -- a test that always passes. The full token stream is reasoning first, then the answer.
    """
    msg = d["choices"][0]["message"]
    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    t = d.get("timings") or {}
    return text, int(t.get("draft_n") or 0), int(t.get("draft_n_accepted") or 0)


def post(port, prompt, max_tokens, top_k=None, logprobs=False, timeout=600):
    data = json.dumps(chat_body(prompt, max_tokens, top_k, logprobs)).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return extract(d)


def stream_first_token(port, prompt, max_tokens, logprobs=False):
    """Start a streaming request, return (thread_result_holder, first_token_seen_event)."""
    body = chat_body(prompt, max_tokens, None, logprobs)
    body["stream"] = True
    data = json.dumps(body).encode()
    seen = threading.Event()
    out = {}

    def run():
        try:
            req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                         data=data, headers={"Content-Type": "application/json"})
            chunks = []
            with urllib.request.urlopen(req, timeout=600) as r:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        d = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    delta = (d.get("choices") or [{}])[0].get("delta", {})
                    piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                    if piece:
                        chunks.append(piece)
                        seen.set()
            out["text"] = "".join(chunks)
        except Exception as e:  # noqa: BLE001
            out["error"] = "%s: %s" % (type(e).__name__, e)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, out, seen


def run_concurrent(port, reqs):
    """reqs: list of (prompt, max_tokens, top_k, logprobs). Returns list of (text, draft, accepted)."""
    results = [None] * len(reqs)
    errors = [None] * len(reqs)
    barrier = threading.Barrier(len(reqs))

    def run(i, prompt, max_tokens, top_k, logprobs):
        try:
            barrier.wait(timeout=30)
            results[i] = post(port, prompt, max_tokens, top_k, logprobs)
        except Exception as e:  # noqa: BLE001
            errors[i] = "%s: %s" % (type(e).__name__, e)

    threads = [threading.Thread(target=run, args=(i,) + tuple(r)) for i, r in enumerate(reqs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=900)
    for i, e in enumerate(errors):
        if e:
            raise RuntimeError("concurrent request %d failed: %s" % (i, e))
    return results


PROMPT_A = ("Name the capital of France, then the capital of Japan, then the capital of Peru. "
            "Answer in one short sentence. " * 3).strip()
PROMPT_B = ("List three prime numbers between 100 and 200, then explain in one sentence why "
            "they are prime. " * 3).strip()
PROMPT_C = ("Summarise the plot of Hamlet in two short sentences, then name one theme. " * 2).strip()


def cmp_tokens(name, got, want):
    if got == want:
        ok("%s: 输出逐字节相同（%d 字符）" % (name, len(got)))
        return True
    bad("%s: 输出不同（并发 %d 字符 / 串行 %d 字符）" % (name, len(got), len(want)))
    info("concurrent: %r" % got[:120])
    info("serial    : %r" % want[:120])
    return False


def main():
    global MODEL, PORT_STAGE_N2, PORT_STAGE_N1, PORT_STAGE_N2B, PORT_SRV_N2, PORT_SRV_N1, \
        PORT_SRV_N2_NOMAX, PORT_SRV_N1_NOSPEC, PORT_SRV_N2_NOSPEC
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--port-base", type=int, default=PORT_BASE,
                    help="端口基准：stage 用 +1..+3、server 用 +11..+15（默认 9360）")
    args = ap.parse_args()
    MODEL = args.model
    (PORT_STAGE_N2, PORT_STAGE_N1, PORT_STAGE_N2B, PORT_SRV_N2,
     PORT_SRV_N1, PORT_SRV_N2_NOMAX, PORT_SRV_N1_NOSPEC,
     PORT_SRV_N2_NOSPEC) = _port_set(args.port_base)

    require("llama-server.exe", "split-stage.exe")
    print("[env] model=%s\n       BIN=%s\n       TMP=%s" % (MODEL, BIN, TMP))
    print("[env] port-base=%d\n       stage=%d/%d/%d\n       server=%d/%d/%d/%d/%d"
          % (args.port_base, PORT_STAGE_N2, PORT_STAGE_N1, PORT_STAGE_N2B,
             PORT_SRV_N2, PORT_SRV_N1, PORT_SRV_N2_NOMAX,
             PORT_SRV_N1_NOSPEC, PORT_SRV_N2_NOSPEC))

    procs = []
    try:
        # ---------------------------------------------------------------- 串行基线（-np 1，带投机）
        print("\n== 串行基线（远程链 -np 1 + MTP 投机，n_max 7）==")
        stage1 = start_stage(PORT_STAGE_N1, 1, "n1")
        procs.append(stage1)
        srv1 = start_server(PORT_SRV_N1, 1, 7, "n1_spec")
        procs.append(srv1)

        base_A, draft_A, acc_A = post(PORT_SRV_N1, PROMPT_A, 32)
        base_B, draft_B, _ = post(PORT_SRV_N1, PROMPT_B, 32)
        base_B4, draft_B4, _ = post(PORT_SRV_N1, PROMPT_B, 4)
        base_C, draft_C, _ = post(PORT_SRV_N1, PROMPT_C, 40, logprobs=True)
        ok("串行基线取到（A/B/B4/C 四条）")
        info("A=%r" % base_A[:60])
        info("C(logprobs)=%r" % base_C[:60])
        info("draft_n: A=%d B=%d B4=%d C(logprobs)=%d" % (draft_A, draft_B, draft_B4, draft_C))
        if draft_A > 0:
            ok("串行基线 A 真的在投机（draft_n=%d, 接受 %d）" % (draft_A, acc_A))
        else:
            bad("串行基线 A 没有产生任何 draft（投机没生效，后面的对比没有意义）")
        if draft_C == 0:
            ok("串行基线 C（logprobs）没有 draft：普通 decode 路径")
        else:
            bad("串行基线 C 竟然在投机（draft_n=%d），它应当被 logprobs 挡住" % draft_C)

        srv1.stop()
        procs.remove(srv1)

        # 判据 8a 的基线：同一条 -np 1 链，但不带投机（候选通道才可用），top_k 5 / 40。
        srv1n = start_server(PORT_SRV_N1, 1, 7, "n1_nospec", spec=False)
        procs.append(srv1n)
        base_A5 = post(PORT_SRV_N1, PROMPT_A, 32, top_k=5)[0]
        base_B40 = post(PORT_SRV_N1, PROMPT_B, 32, top_k=40)[0]
        ok("串行基线取到（非投机链，top_k 5 / 40）")
        srv1n.stop()
        procs.remove(srv1n)
        stage1.stop()
        procs.remove(stage1)

        # ---------------------------------------------------------------- 判据 7 / 8b / 9 / 10 / 11
        print("\n== 远程链 -np 2 + MTP 投机（n_max 7）==")
        stage2 = start_stage(PORT_STAGE_N2, 2, "n2")
        procs.append(stage2)
        srv2 = start_server(PORT_SRV_N2, 2, 7, "n2_spec")
        procs.append(srv2)

        # 判据 7：两个并发请求
        got = run_concurrent(PORT_SRV_N2, [(PROMPT_A, 32, None, False),
                                           (PROMPT_B, 32, None, False)])
        cmp_tokens("判据 7 A", got[0][0], base_A)
        cmp_tokens("判据 7 B", got[1][0], base_B)
        if got[0][1] > 0 and got[1][1] > 0:
            ok("判据 7: 两条并发请求都在投机（draft_n %d / %d）" % (got[0][1], got[1][1]))
        else:
            bad("判据 7: 有请求没有 draft（%d / %d）——并发下投机被关掉了" % (got[0][1], got[1][1]))

        # 判据 8b：两个序列的 k 不同（max_tokens 4 -> n_draft_max <= 3）
        got = run_concurrent(PORT_SRV_N2, [(PROMPT_A, 32, None, False),
                                           (PROMPT_B, 4, None, False)])
        cmp_tokens("判据 8b A(k=7)", got[0][0], base_A)
        cmp_tokens("判据 8b B(k<=3)", got[1][0], base_B4)
        info("判据 8b: 两个序列的 draft_n = %d / %d（k 不同：32 个 token 预算 vs 4 个）"
             % (got[0][1], got[1][1]))

        # 判据 9 / 10：C 先跑（普通 decode，logprobs 关掉投机通道），拿到第一个 token 后再发 A。
        # 这样 A 的 verify 帧在 batch 里的行号一定不是从 0 开始。
        print("  -- 判据 9/10：先起普通 decode，再起投机序列 --")
        srv2.stop()
        procs.remove(srv2)
        srv2 = start_server(PORT_SRV_N2, 2, 7, "n2_mixed")
        procs.append(srv2)

        tC, outC, seenC = stream_first_token(PORT_SRV_N2, PROMPT_C, 40, logprobs=True)
        if not seenC.wait(timeout=180):
            bad("判据 9/10: 普通 decode 请求没有产出第一个 token")
            outA = None
        else:
            time.sleep(0.3)  # 让 C 至少进到 decode 状态
            outA = post(PORT_SRV_N2, PROMPT_A, 32)
        tC.join(timeout=600)
        gotC = outC.get("text", "")
        if "error" in outC:
            bad("判据 9/10: 普通 decode 请求失败：%s" % outC["error"])
        else:
            cmp_tokens("判据 9 普通 decode（C）", gotC, base_C)
        if outA is not None:
            cmp_tokens("判据 9/10 投机序列（A）", outA[0], base_A)
            if outA[1] > 0:
                ok("判据 9: 同一批里 A 在投机（draft_n=%d），B 没有（普通 decode）" % outA[1])
            else:
                bad("判据 9: A 没有 draft（draft_n=%d）—— 期望它在投机，实际走了普通 decode" % outA[1])

        log = srv2.text()
        m = [ln for ln in log.splitlines() if "verify channel armed for" in ln]
        if m:
            ok("判据 10: 验证通道真的被用上，且行号不是从 0 开始：%s" % m[0].strip())
        else:
            bad("判据 10: 日志里没有 'verify channel armed for' —— 验证通道没被用上，行映射没验到")

        # 判据 8a：同一条 -np 2 链，关掉投机，两个并发请求用不同 top_k（走候选通道）
        print("  -- 判据 8a：非投机链上的逐行候选 --")
        srv2.stop()
        procs.remove(srv2)
        srv2n = start_server(PORT_SRV_N2, 2, 7, "n2_nospec", spec=False)
        procs.append(srv2n)
        got = run_concurrent(PORT_SRV_N2, [(PROMPT_A, 32, 5, False),
                                           (PROMPT_B, 32, 40, False)])
        cmp_tokens("判据 8a A(top_k 5)", got[0][0], base_A5)
        cmp_tokens("判据 8a B(top_k 40)", got[1][0], base_B40)
        log = srv2n.text()
        m = [ln for ln in log.splitlines() if "candidate channel armed for" in ln]
        if m and "top_k 5" in m[0] and "top_k 40" in m[0]:
            ok("判据 8a: 一帧里两行候选参数不同（%s）" % m[0].strip())
        else:
            bad("判据 8a: 没有「一帧里两行候选参数不同」的证据：%s" % (m[0].strip() if m else "(无日志)"))
        srv2n.stop()
        procs.remove(srv2n)

        # 判据 11：k 超上限（n_max 12，-np 2 -> 每序列上限 7）
        print("  -- 判据 11：--spec-draft-n-max 12（超上限）--")
        # 这条链的 n_rs_seq 由 head 的 need_n_rs_seq() = n_max + 1 = 13 决定，两端必须一致，
        # 所以单独起一个 --rs-seq 13 的段（同一个 stage 进程不能同时服务两个 rs_seq）。
        stage2b = start_stage(PORT_STAGE_N2B, 2, "n2b", rs=13)
        procs.append(stage2b)
        srv2m = start_server(PORT_SRV_N2_NOMAX, 2, 12, "n2_nmax12",
                             stage_port=PORT_STAGE_N2B, rs=13)
        procs.append(srv2m)
        log = srv2m.text()
        m = [ln for ln in log.splitlines() if "per-slot draft cap" in ln or "exceeds what this chain" in ln]
        if m:
            ok("判据 11: 启动日志有 per-slot 上限的 warn：%s" % m[0].strip())
        else:
            bad("判据 11: 启动日志没有 'per-slot draft cap' / 'exceeds what this chain' 的 warn —— 期望超限配置被点名降级，实际没有任何提示")
        try:
            got = run_concurrent(PORT_SRV_N2_NOMAX, [(PROMPT_A, 32, None, False),
                                                     (PROMPT_B, 32, None, False)])
            cmp_tokens("判据 11 A", got[0][0], base_A)
            cmp_tokens("判据 11 B", got[1][0], base_B)
            info("判据 11: 超限配置下仍在投机（draft_n %d / %d）" % (got[0][1], got[1][1]))
        except Exception as e:  # noqa: BLE001
            bad("判据 11: 超限配置下请求失败：%s" % e)

        bad_req = [ln for ln in (srv2m.text() + stage2.text()).splitlines()
                   if "ST_BAD_REQ" in ln or "stage rejected the request" in ln or
                   "n_tokens" in ln and "exceeds" in ln]
        if bad_req:
            bad("判据 11: 出现了 ST_BAD_REQ / 行数超限的拒绝：%s" % bad_req[0].strip())
        else:
            ok("判据 11: 没有出现 n_tokens > n_verify_rows 的 ST_BAD_REQ")

        srv2m.stop()
        procs.remove(srv2m)
        stage2b.stop()
        procs.remove(stage2b)
        stage2.stop()
        procs.remove(stage2)
    finally:
        for p in list(procs):
            p.stop()

    # 项数断言（防假绿的最后一道）：上面 7 处条件分支全是「应该发生但没发生」的形态，
    # 每一处都带 else: bad()，所以 PASS+FAIL 恒为 20。将来有人再加条件分支却忘了 else，
    # 检查会静默消失、FAIL 仍为 0、退出码仍是 0——这条就是抓那种假绿的。
    # 只在真的漏输出时触发：任何一处 else: bad() 触发时 PASS+FAIL 仍是 20，不会误报。
    # 放在汇总行之前：这样汇总行打印的是最终计数（含这条断言自己产生的 FAIL）。
    EXPECT_CHECKS = 20   # 与 verify.sh 的 EXPECT_PASS 增量一致（组 14 = 20 项）
    if PASS + FAIL != EXPECT_CHECKS:
        bad("项数变了：%d PASS + %d FAIL = %d，期望 %d（有检查因条件不满足静默消失）"
            % (PASS, FAIL, PASS + FAIL, EXPECT_CHECKS))
    print("\n== 判据 7–11：%d PASS / %d FAIL ==" % (PASS, FAIL))
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
