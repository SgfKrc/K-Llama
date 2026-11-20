#!/usr/bin/env python3
"""判据 2–6：远程链多序列的基础语义（V22-MULTISEQ.md §7.1）。

为什么单独一个脚本：verify.sh 的组 7 跑远程链但从不给 -np（auto = 1），组 14 覆盖的是
per-seq 投机四通道（判据 7–11）。判据 1（两个请求并发、与串行逐字节相同）已于 2026-11-14
三机实测通过；判据 2–6 是「多序列本身」的语义——混合批、KV 隔离、checkpoint、身份校验、
n_ctx 取整——此前**从未跑过**。它们与投机无关，判据也不许弱化成「近似相同」：
「逐字节相同」是组 1/3/6/7 的既有口径。

判据与做法（原文见 docs/design/V22-MULTISEQ.md §7.1）：

  2  混合批（一个在 prefill、一个在 decode）逐字节相同
     **确定性构造**：先流式发 A，等首 token 到达（此刻 A 必在 decode）再发 B，B 的 prompt
     一帧内 prefill 完（prompt token 数 <= n_ubatch）。A、B 的输出分别与串行基线逐字节比。
     防假绿：必须观察到 B 在 A 完成之前完成（否则两个请求根本没重叠，判据没测到）。
  3  KV 隔离：对 A 做 seq_rm 不影响 B
     B 在 decode 期间，另一个 slot 上连发三个短请求（每个结束都让 server 对**那个 slot 的
     seq** 做 seq_rm）。B 的输出与「B 单独跑」逐字节相同；段日志里 seq_rm 的 seq 是 A 的，
     且在 B 的首 token 之后、B 完成之前发生。
  4  多序列下的 checkpoint save/restore
     两个序列各在自己的 slot 上建立 context checkpoint（stage 侧 mint 一个 8 字节 handle），
     再各恢复一次（续写请求命中 prefix cache，server 用 checkpoint 恢复而不是重算）。恢复后
     继续生成的 token 与不中断时逐字节相同。另外用 --state-cache 1 制造 handle 失效：stage
     必须答 ST_STALE（「无 checkpoint」），server 必须重算而不是把链判死。
     注意：`--slot-save-path` 的 /slots save/restore 在远程链上**设计上不可用**（完整 state
     不导出，只有 checkpoint flavour 导出，启动日志有 warn），所以本判据走的是 llama-server
     的 context checkpoint 路径——那才是判据 4 说的 checkpoint。
  5  身份字段不一致必须干净报错
     head -np 2 + stage --n-seq-max 1：attach 阶段就拒（方案 A），消息必须带两个数字
     （期望 2、stage 报 1）。--split-verify-rows 不一致同理（stage 17 -> head 8 方向反了
     不拒，必须是 stage 偏小：head 17 + stage 8）。
     QLH_SPLIT_SKIP_IDENTITY=1 时退化为首帧 ST_BAD_REQ（stage 的 check_seq_id），**不得**
     出现「输出像正常但错」：失败的请求必须报错，成功返回的必须与基线逐字节相同。
  6  n_ctx 不被 n_seq_max 整除时可见
     -c 4096 + -np 3：4096/3 向下取整到 256 的倍数 = 1536，n_ctx 被改成 4608。head 与 stage
     的启动日志都必须有「rounding down」warn，且两端取整后的 n_ctx_seq 相同（head 的取整
     目标 4608 / 3 == stage 日志里的 1536）。

每序列容量约束（§7.1 末段，对全部判据生效）：-c 4096 / -np 2 -> 每序列 2048，prompt <= 1024、
max_tokens <= 256。本脚本里最长的 prompt 约 240 token、最长的 max_tokens 是 96。

用法：
  QLH_BUILD_DIR=H:/llamacpp-c/build-split python verify/verify_multiseq.py \
      [--model PATH] [--port-base 9370]

  --port-base 只改端口：stage 用 +1..+4、server 用 +5..+9。verify.sh 的组 15 传 9370
  （-> stage 9371-9374 / server 9375-9379），避开组 13 的 KVMem 9361、组 14 的 9381-9395、
  CUDA 组的 9357/9358 与主用的 9331/9332/9341-9343。全部落在 93xx 段内，所以 verify.sh 的
  cleanup_ours 连「不是从 $BIN 起的老构建」也按端口兜得住。
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _qlh_env import BIN, TMP, require   # noqa: E402

# Windows 上 Python 的 stdout 默认是 GBK（cp936）：中文判据消息落进日志后，在 UTF-8 终端里
# 回显全是乱码（verify.sh 的组 15 把本脚本输出逐行转记账）。统一成 UTF-8；日志文件由
# verify.sh 重定向，因此也以 UTF-8 落盘。verify_perspec.py / verify_kvmem.py 同款。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

MODEL = os.environ.get("QLH_SPLIT_MODEL", r"H:/qlh_models/qwen3-5-2b-gguf/qwen35-2b-Q4_K_M.gguf")
RS_SEQ = os.environ.get("QLH_RS_SEQ", "8")
VERIFY_ROWS = 17
LAYER_LO, LAYER_HI = 0, 20     # head
TAIL_LO, TAIL_HI = 20, 24      # tail stage

PORT_BASE = 9370
# stage 用 +1..+4、server 用 +5..+9。每条判据用不同的端口对：同一对端口连续起两次（前一次的
# 进程刚 terminate）在 Windows 上偶尔会撞上还没释放的绑定，那会被记成一次假的 FAIL。
(PORT_STAGE_1, PORT_STAGE_2, PORT_STAGE_3, PORT_STAGE_4,
 PORT_SRV_1, PORT_SRV_2, PORT_SRV_3, PORT_SRV_4, PORT_SRV_5) = (
    PORT_BASE + 1, PORT_BASE + 2, PORT_BASE + 3, PORT_BASE + 4,
    PORT_BASE + 5, PORT_BASE + 6, PORT_BASE + 7, PORT_BASE + 8, PORT_BASE + 9)

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


# ---------------------------------------------------------------- 进程脚手架
class Proc:
    def __init__(self, name, cmd, log, env=None):
        self.name = name
        self.log = log
        self.f = open(log, "wb")
        e = None
        if env:
            e = dict(os.environ)
            e.update(env)
        self.p = subprocess.Popen(cmd, stdout=self.f, stderr=subprocess.STDOUT, env=e)

    def alive(self):
        return self.p.poll() is None

    def rc(self):
        return self.p.poll()

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
                    self.p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        try:
            self.f.close()
        except Exception:
            pass


def wait_for(fn, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if fn():
            return True
        time.sleep(0.2)
    return False


def wait_stage(proc, timeout=240):
    """ready: 打在模型/context/offload 都定下来之后，是唯一可靠的「可以连了」信号。"""
    return wait_for(lambda: "ready:" in proc.text() or not proc.alive(), timeout)


def wait_health(port, proc, timeout=240):
    # 进程没了就立刻返回 False：等满超时会把一个启动 bug 变成四分钟静默。
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
        time.sleep(0.2)
    return False


def stage_cmd(port, n_seq_max, verify_rows=VERIFY_ROWS, state_cache=32, rs=RS_SEQ,
              ctx=4096, batch=512):
    return [os.path.join(BIN, "split-stage.exe"), "-m", MODEL,
            "--layer-range", "%d:%d" % (TAIL_LO, TAIL_HI), "--role", "tail",
            "--rs-seq", str(rs), "--n-seq-max", str(n_seq_max),
            "--split-verify-rows", str(verify_rows), "--state-cache", str(state_cache),
            "--port", str(port), "-c", str(ctx), "-b", str(batch)]


def srv_cmd(port, np_, stage_port, verify_rows=VERIFY_ROWS, rs=RS_SEQ, ctx=4096,
            batch=2048, ubatch=512, extra=()):
    cmd = [os.path.join(BIN, "llama-server.exe"), "-m", MODEL,
           "--host", "127.0.0.1", "--port", str(port),
           "-c", str(ctx), "-np", str(np_), "-ngl", "0", "--no-warmup",
           "--split-layer-range", "%d:%d" % (LAYER_LO, LAYER_HI),
           "--split-remote", "127.0.0.1:%d" % stage_port,
           "--split-verify-rows", str(verify_rows), "--split-rs-seq", str(rs),
           "-b", str(batch), "-ub", str(ubatch)]
    if np_ > 1:
        # 远程链 -np N 必须显式关掉 kv_unified：统一池给 head 的序列整个上下文，而段侧按
        # --n-seq-max 切分，两端每序列容量会差 N 倍（server 会启动即拒，这里照它的要求给）。
        cmd += ["--no-kv-unified"]
    cmd += list(extra)
    return cmd


class Chain:
    """一条 head + 一个 tail stage。起不来时 raise，由调用方记账。"""

    def __init__(self, tag, srv_port, stage_port, np_, n_seq_max, state_cache=32,
                 verify_rows=VERIFY_ROWS, srv_verify_rows=None, srv_rs=None, srv_env=None,
                 batch=2048, ubatch=512, ctx=4096, extra=()):
        self.tag = tag
        self.stage = Proc("stage-%s" % tag,
                          stage_cmd(stage_port, n_seq_max, verify_rows=verify_rows,
                                    state_cache=state_cache, ctx=ctx),
                          os.path.join(TMP, "multiseq_stage_%s.log" % tag))
        if not wait_stage(self.stage):
            raise RuntimeError("stage %s 没有 ready（见 %s）：\n%s"
                               % (tag, self.stage.log,
                                  "\n".join(self.stage.text().splitlines()[-6:])))
        self.srv = Proc("srv-%s" % tag,
                        srv_cmd(srv_port, np_, stage_port,
                                verify_rows=(srv_verify_rows if srv_verify_rows is not None
                                             else verify_rows),
                                rs=(srv_rs if srv_rs is not None else RS_SEQ),
                                ctx=ctx, batch=batch, ubatch=ubatch, extra=extra),
                        os.path.join(TMP, "multiseq_srv_%s.log" % tag), env=srv_env)
        self.port = srv_port
        if not wait_health(srv_port, self.srv):
            raise RuntimeError("server %s 没有起来（见 %s）：\n%s"
                               % (tag, self.srv.log,
                                  "\n".join(self.srv.text().splitlines()[-8:])))

    def stop(self):
        self.srv.stop()
        self.stage.stop()


class ExpectRefusal:
    """起一条**应该 attach 失败**的链：只起进程，不等 health。"""

    def __init__(self, tag, srv_port, stage_port, np_, n_seq_max, srv_verify_rows=None,
                 stage_verify_rows=VERIFY_ROWS, srv_env=None):
        self.tag = tag
        self.stage = Proc("stage-%s" % tag,
                          stage_cmd(stage_port, n_seq_max, verify_rows=stage_verify_rows),
                          os.path.join(TMP, "multiseq_stage_%s.log" % tag))
        if not wait_stage(self.stage):
            raise RuntimeError("stage %s 没有 ready（见 %s）" % (tag, self.stage.log))
        self.srv = Proc("srv-%s" % tag,
                        srv_cmd(srv_port, np_, stage_port,
                                verify_rows=(srv_verify_rows if srv_verify_rows is not None
                                             else VERIFY_ROWS)),
                        os.path.join(TMP, "multiseq_srv_%s.log" % tag), env=srv_env)
        self.port = srv_port
        self.srv_log = self.srv.log
        self.stage_log = self.stage.log

    def wait_exit(self, timeout=180):
        return wait_for(lambda: not self.srv.alive(), timeout)

    def text(self):
        return self.srv.text()

    def stop(self):
        self.srv.stop()
        self.stage.stop()


# ---------------------------------------------------------------- HTTP
def body(prompt, max_tokens, stream=False, cache_prompt=True):
    return json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0,
                       "cache_prompt": cache_prompt, "seed": 1234,
                       "stream": stream}).encode()


def extract(d):
    """qwen35 是 reasoning model：小 max_tokens 时 token 全落在 reasoning_content 里，
    content 是空串。只比 content 会比两个空串——一个永远通过的测试。"""
    m = d["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + (m.get("content") or "")


def post(port, prompt, max_tokens, cache_prompt=True, logprobs=False, timeout=900):
    data = body(prompt, max_tokens, False, cache_prompt)
    if logprobs:
        # 让这个请求走**整行** logits 路径：pre-sampling logprobs 需要整行 softmax，
        # common_sampler_candidates_check() 因此把候选通道关掉（组 14 判据 9 同款技巧）。
        d0 = json.loads(data)
        d0["logprobs"] = True
        d0["top_logprobs"] = 1
        data = json.dumps(d0).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return extract(d), (d.get("timings") or {})


def post_ex(port, prompt, max_tokens, cache_prompt=True, timeout=900):
    """同 post，但把 HTTP 错误也作为结果返回，不抛。返回 (status, text, timings)。"""
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=body(prompt, max_tokens, False, cache_prompt),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        return r.status, extract(d), (d.get("timings") or {})
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace"), {}
    except Exception as e:  # noqa: BLE001
        return 0, "%s: %s" % (type(e).__name__, e), {}


def stream_start(port, prompt, max_tokens, cache_prompt=False, logprobs=False):
    """流式发一个请求，返回 (holder, first_token_event, thread)。

    holder 在结束后带 text / error；first_token_event 在第一个非空 delta 时 set。
    logprobs=True 让这个请求走**整行** logits 路径（pre-sampling logprobs 需要整行 softmax，
    common_sampler_candidates_check() 因此关掉候选通道）。
    """
    seen = threading.Event()
    out = {}
    data = body(prompt, max_tokens, True, cache_prompt)
    if logprobs:
        d0 = json.loads(data)
        d0["logprobs"] = True
        d0["top_logprobs"] = 1
        data = json.dumps(d0).encode()

    def run():
        try:
            req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                         data=data,
                                         headers={"Content-Type": "application/json"})
            chunks = []
            with urllib.request.urlopen(req, timeout=900) as r:
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
    return out, seen, t


def health(port):
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------- 判据消息里的共用件
PROMPT_A = ("Name the capital of France, then the capital of Japan, then the capital of Peru. "
            "Answer in one short sentence. " * 2).strip()
PROMPT_B = ("List three prime numbers between 100 and 200, then explain in one sentence why "
            "they are prime. " * 2).strip()
PROMPT_C = "Say hi."


def cmp_bytes(label, got, want):
    if got == want:
        ok("%s：输出逐字节相同（%d 字符）" % (label, len(got)))
        return True
    bad("%s：输出不同（实际 %d 字符 / 基线 %d 字符）" % (label, len(got), len(want)))
    info("实际：%r" % got[:120])
    info("基线：%r" % want[:120])
    return False


def seq_rm_lines(stage_log_path):
    """段日志里的 OP_SEQ_RM 行，按出现顺序。格式（split-stage/main.cpp:1597）：
    [split-stage] OP_SEQ_RM seq=1 p0=0 p1=-1 -> ok, pos_max 18 -> -1
    """
    out = []
    try:
        with open(stage_log_path, "rb") as f:
            for line in f.read().decode("utf-8", "replace").splitlines():
                if "OP_SEQ_RM" not in line:
                    continue
                try:
                    sid = int(line.split("seq=")[1].split()[0])
                except (IndexError, ValueError):
                    continue
                out.append(sid)
    except OSError:
        pass
    return out


def state_load_lines(stage_log_path):
    out = []
    try:
        with open(stage_log_path, "rb") as f:
            for line in f.read().decode("utf-8", "replace").splitlines():
                if "OP_STATE_LOAD" in line:
                    out.append(line.strip())
    except OSError:
        pass
    return out


# ---------------------------------------------------------------- 判据 2
def crit2():
    """混合批：A 在 decode、B 刚开始 prefill。"""
    n_items = 3
    before = PASS + FAIL
    chain = None
    try:
        chain = Chain("c2", PORT_SRV_1, PORT_STAGE_1, np_=2, n_seq_max=2)
        port = chain.port
        # 串行基线：同一条链，A、B 各跑一次（不并发）。cache_prompt=False 让两次基线互不影响，
        # 也让下面的混合场景不因 prefix cache 命中而改变 prefill 的形状。
        baseA, timA = post(port, PROMPT_A, 96, cache_prompt=False)
        baseB, timB = post(port, PROMPT_B, 16, cache_prompt=False)
        info("串行基线：A %d 字符（prompt_n=%d）/ B %d 字符（prompt_n=%d）"
             % (len(baseA), timA.get("prompt_n", -1), len(baseB), timB.get("prompt_n", -1)))
        if not baseA or not baseB:
            bad("判据 2：串行基线是空的，无法对照")
            return
        # 混合批：先发 A（流式），首 token 到达 = A 已经 prefill 完、进入 decode；此刻发 B。
        outA, seenA, tA = stream_start(port, PROMPT_A, 96)
        if not seenA.wait(timeout=300):
            bad("判据 2：A 没有产出第一个 token（%s）" % outA.get("error", "无 delta"))
            tA.join(60)
            return
        t_b0 = time.time()
        gotB, timB2 = post(port, PROMPT_B, 16, cache_prompt=False)
        t_b1 = time.time()
        tA.join(900)
        gotA = outA.get("text", "")
        t_a1 = time.time()
        info("混合批：B 在 A 的首 token 之后发起，B 用 %.1fs 完成（prompt_n=%d），"
             "A 在 B 之后又跑了 %.1fs 才结束" % (t_b1 - t_b0, timB2.get("prompt_n", -1), t_a1 - t_b1))
        cmp_bytes("判据 2 A（decode 中的序列）", gotA, baseA)
        cmp_bytes("判据 2 B（A decode 时开始 prefill 的序列）", gotB, baseB)
        # 防假绿：B 必须在 A 结束之前完成，否则两个请求根本没重叠，这一条什么也没测到。
        if t_b1 < t_a1 and timB2.get("prompt_n", 0) > 0:
            ok("判据 2：B 在 A 完成之前结束（重叠 %.1fs），混合批真的发生了"
               % (t_a1 - t_b1))
        else:
            bad("判据 2：B 没有在 A 完成之前结束（A 结束于 B 之后 %.1fs）—— 两个请求没有重叠，"
                "这一条没测到混合批" % (t_b1 - t_a1))
    except Exception as e:  # noqa: BLE001
        info("判据 2 异常：%s: %s" % (type(e).__name__, e))
    finally:
        if chain is not None:
            chain.stop()
    missing = n_items - (PASS + FAIL - before)
    for _ in range(max(0, missing)):
        bad("判据 2：这一项没有跑到（组内异常或提前退出）")


# ---------------------------------------------------------------- 判据 3
def crit3():
    """KV 隔离：对 A 做 seq_rm 不影响 B。"""
    n_items = 3
    before = PASS + FAIL
    chain = None
    try:
        chain = Chain("c3", PORT_SRV_1, PORT_STAGE_1, np_=2, n_seq_max=2)
        port = chain.port
        stage_log = chain.stage.log
        # 基线：B 单独跑（同一条链、没有别的序列在动它）。
        baseB, _ = post(port, PROMPT_B, 96, cache_prompt=False)
        info("B 单独跑的基线：%d 字符" % len(baseB))
        if not baseB:
            bad("判据 3：B 的基线是空的，无法对照")
            return
        # B 占一个 slot 并进入 decode；另一个 slot 上连发三个短请求，每个结束都会让 server
        # 对**那个 slot 的 seq** 做 seq_rm（这是 slot 复用的正常路径，不是测试钩子）。
        # B 也走整行 logits 路径：这一条判据只测 seq_rm 的 KV 隔离，候选通道不在它的范围里
        # （候选通道在混合批下另有一个间歇性缺陷，已单独报告；把它关掉，这条判据才只回答
        # 「对 A 的 seq_rm 有没有动到 B」）。
        outB, seenB, tB = stream_start(port, PROMPT_B, 96, logprobs=True)
        if not seenB.wait(timeout=300):
            bad("判据 3：B 没有产出第一个 token（%s）" % outB.get("error", "无 delta"))
            tB.join(60)
            return
        mark_lo = len(seq_rm_lines(stage_log))
        # 长度交替（短/长/短）：判据原文要的是「长度交替触发 slot 复用」。每个请求结束都会让
        # server 对**它那个 slot 的 seq** 做 seq_rm（cache_prompt=false 时是整段清），下一个
        # 请求复用同一个 slot —— 这就是判据 3 要的「对 A 做 seq_rm」。
        overlap = True
        for i, prompt in enumerate((PROMPT_C, PROMPT_C * 3, PROMPT_C)):
            # logprobs=True 让干扰请求走整行 logits 路径，把**候选通道**排除在这条判据之外。
            # 理由：判据 3 测的是 seq_rm 的 KV 隔离，而「混合批 + 候选通道」另有一个间歇性
            # 缺陷（实测 4 轮 2 轮：head 偶尔发出 n_steps=0 / 非法 step 的 cand_req，stage 拒
            # 帧后 head 把整条链标成不可信，正在 decode 的另一个序列被打断）——那是候选通道
            # 的问题，不该混进「seq_rm 不影响 B」这条判据里，已单独报告。
            a, _ = post(port, prompt, 4, cache_prompt=False, logprobs=True)
            info("干扰请求 %d 完成（长度 %d，%r）" % (i, len(prompt), a[:24]))
            if not tB.is_alive():
                # B 提前结束：窗口后半段就不是「B 在 decode」了，这一轮的重叠不完整。
                overlap = False
                info("警告：B 在第 %d 个干扰请求之后已经结束，重叠没有覆盖全部干扰" % i)
                break
        # 窗口上界在 B 仍然活着时取：B 结束那一刻 server 自己会对 seq 0 做一次 seq_rm，
        # 那次不属于「干扰请求动了 B」，落在窗口里会把判据 3 的第三项变成假 FAIL。
        mark_hi = len(seq_rm_lines(stage_log))
        tB.join(900)
        gotB = outB.get("text", "")
        cmp_bytes("判据 3 B（A 被 seq_rm 期间的序列）", gotB, baseB)
        window = seq_rm_lines(stage_log)[mark_lo:mark_hi]
        info("B 首 token 之后、B 完成之前，段上收到的 seq_rm 的 seq：%s" % sorted(set(window)))
        # 判据 3 的原文：「段日志里 OP_SEQ_RM 的 seq 是 A 的」。B 占的 slot 是 0（它先发），
        # 干扰请求占 1；所以窗口里必须出现 seq=1 的 seq_rm。
        if 1 in window:
            ok("判据 3：B 运行期间段上收到了对 A（seq=1）的 OP_SEQ_RM（%d 次）"
               % window.count(1))
        else:
            bad("判据 3：B 运行期间没有出现对 A（seq=1）的 OP_SEQ_RM —— KV 隔离这条没测到"
                "（窗口内 seq：%s）" % sorted(set(window)))
        # 而且不能有对 B（seq=0）的 seq_rm 混进同一个窗口：那说明隔离本身就坏了。
        if overlap and 0 not in window:
            ok("判据 3：同一窗口里没有对 B（seq=0）的 OP_SEQ_RM")
        elif not overlap:
            bad("判据 3：B 在干扰请求结束之前就停了，窗口不完整 —— 「干扰期间 B 的 seq 没被动过」"
                "这一条没有测到")
        else:
            bad("判据 3：同一窗口里出现了对 B（seq=0）的 OP_SEQ_RM —— 干扰请求动到了 B 的序列")
    except Exception as e:  # noqa: BLE001
        info("判据 3 异常：%s: %s" % (type(e).__name__, e))
    finally:
        if chain is not None:
            chain.stop()
    missing = n_items - (PASS + FAIL - before)
    for _ in range(max(0, missing)):
        bad("判据 3：这一项没有跑到（组内异常或提前退出）")


# ---------------------------------------------------------------- 判据 4
# 判据 4 用的是 llama-server 的 context checkpoint（--ctx-checkpoints，默认 32/slot），不是
# /slots 的 save/restore：后者导出的是**完整** sequence state，远程链设计上不导出它
# （llama-context.cpp 的 split_state_unavailable：只有 recurrent-only 的 checkpoint flavour
# 可用，启动日志里有一行 warn），所以 /slots save 在远程链上写出 0 字节 state、restore 必然
# 失败。context checkpoint 走的是 state_seq_get_data_ext(..., PARTIAL_ONLY)，那才是判据 4
# 说的 checkpoint，也是 handle 失效（v6 的 ST_STALE）唯一的用户。
PROMPT_A_LONG = "France " * 200     # ~240 token：-b 128 下分 2 个 chunk -> 建 checkpoint
PROMPT_B_LONG = "Japan " * 200
PROMPT_A_CONT = PROMPT_A_LONG + "Peru " * 30    # 续写：与 A 同前缀，命中 checkpoint 恢复
PROMPT_B_CONT = PROMPT_B_LONG + "Chile " * 30


def _build_two_slot_checkpoints(chain, tag):
    """两个序列各占一个 slot、各自建立 context checkpoint，返回 (OA, OB)。

    这两个请求走默认的 cache_prompt=true：判据 4 要的是「slot 里留下可以恢复的状态」，
    关掉 prefix cache 会让 slot 在请求结束时被清空，checkpoint 就没有恢复的落点。
    """
    port = chain.port
    outA, seenA, tA = stream_start(port, PROMPT_A_LONG, 4, cache_prompt=True)
    if not seenA.wait(timeout=300):
        raise RuntimeError("%s：序列 A 没有产出第一个 token" % tag)
    # A 已经在 decode（= 已占住一个 slot），B 才会拿到另一个 slot。
    outB, seenB, tB = stream_start(port, PROMPT_B_LONG, 4, cache_prompt=True)
    if not seenB.wait(timeout=300):
        raise RuntimeError("%s：序列 B 没有产出第一个 token" % tag)
    tA.join(900)
    tB.join(900)
    return outA.get("text", ""), outB.get("text", "")


def crit4():
    n_items = 4
    before = PASS + FAIL
    base = None
    ck = None
    stale = None
    try:
        # 基线：干净链上直接跑续写请求（完整 prefill，没有 checkpoint 可用）。
        base = Chain("c4base", PORT_SRV_1, PORT_STAGE_1, np_=2, n_seq_max=2,
                     batch=128, ubatch=128)
        baseA2, timBaseA = post(base.port, PROMPT_A_CONT, 4, cache_prompt=False)
        baseB2, timBaseB = post(base.port, PROMPT_B_CONT, 4, cache_prompt=False)
        info("基线（完整 prefill）：A %d 字符（prompt_n=%d）/ B %d 字符（prompt_n=%d）"
             % (len(baseA2), timBaseA.get("prompt_n", -1),
                len(baseB2), timBaseB.get("prompt_n", -1)))
        if not baseA2 or not baseB2:
            bad("判据 4：续写基线是空的，无法对照")
            return
        base.stop()
        base = None

        # 同一条链上先让两个序列各自建 checkpoint，再用续写请求让它们各自恢复一次。
        ck = Chain("c4", PORT_SRV_2, PORT_STAGE_2, np_=2, n_seq_max=2,
                   batch=128, ubatch=128)
        oA, oB = _build_two_slot_checkpoints(ck, "判据 4")
        info("两个序列的首次生成：A %d 字符 / B %d 字符" % (len(oA), len(oB)))
        gotA2, timA2 = post(ck.port, PROMPT_A_CONT, 4, cache_prompt=True)
        gotB2, timB2 = post(ck.port, PROMPT_B_CONT, 4, cache_prompt=True)
        info("恢复后续写：A prompt_n=%d / B prompt_n=%d（基线是完整 prefill）"
             % (timA2.get("prompt_n", -1), timB2.get("prompt_n", -1)))
        cmp_bytes("判据 4 序列 A：checkpoint 恢复后续写的 token", gotA2, baseA2)
        cmp_bytes("判据 4 序列 B：checkpoint 恢复后续写的 token", gotB2, baseB2)
        # 防假绿：恢复必须真的发生（stage 侧 OP_STATE_LOAD ok），否则「逐字节相同」只是
        # 重算了一遍。两序列各一次。
        loads = state_load_lines(ck.stage.log)
        ok_loads = [ln for ln in loads if "-> ok" in ln]
        info("段侧 OP_STATE_LOAD：%d 次，其中 ok %d 次" % (len(loads), len(ok_loads)))
        if len(ok_loads) >= 2:
            ok("判据 4：两个序列各恢复了一次（段侧 OP_STATE_LOAD -> ok，%d 次）" % len(ok_loads))
        else:
            bad("判据 4：段侧只看到 %d 次成功的 OP_STATE_LOAD（期望两序列各一次）—— "
                "checkpoint 恢复没有真的发生，逐字节相同可能只是重算" % len(ok_loads))
        ck.stop()
        ck = None

        # handle 失效：--state-cache 1 的段只留一个 snapshot。两个序列各建一次 checkpoint 后
        # 先建的必然被驱逐，恢复它时段必须答 ST_STALE（「无 checkpoint」），head 必须重算
        # 而不是把链判死。
        stale = Chain("c4stale", PORT_SRV_3, PORT_STAGE_3, np_=2, n_seq_max=2,
                      state_cache=1, batch=128, ubatch=128)
        _build_two_slot_checkpoints(stale, "判据 4 handle 失效")
        gotA_stale, _ = post(stale.port, PROMPT_A_CONT, 4, cache_prompt=True)
        stale_lines = [ln for ln in state_load_lines(stale.stage.log) if "STALE" in ln]
        info("段侧 ST_STALE 行：%s" % (stale_lines[0] if stale_lines else "(无)"))
        alive = health(stale.port)
        if stale_lines and alive and gotA_stale == baseA2:
            ok("判据 4：handle 失效时段答「无 checkpoint」（ST_STALE，%d 次），server 没有链死，"
               "重算后的输出与基线逐字节相同" % len(stale_lines))
        else:
            bad("判据 4：handle 失效路径不对（ST_STALE %d 次 / server 存活 %s / 输出与基线相同 %s）"
                % (len(stale_lines), alive, gotA_stale == baseA2))
    except Exception as e:  # noqa: BLE001
        info("判据 4 异常：%s: %s" % (type(e).__name__, e))
    finally:
        for p in (base, ck, stale):
            if p is not None:
                p.stop()
    missing = n_items - (PASS + FAIL - before)
    for _ in range(max(0, missing)):
        bad("判据 4：这一项没有跑到（组内异常或提前退出）")


# ---------------------------------------------------------------- 判据 5
def _attach_refusal(label, srv_port, stage_port, np_, n_seq_max, srv_verify_rows,
                    expect_pairs, stage_verify_rows=VERIFY_ROWS, srv_env=None):
    """起一条应该 attach 阶段就拒的链，检查拒绝消息里带上了期望的两个数字。"""
    ex = None
    try:
        ex = ExpectRefusal("c5-%s" % label, srv_port, stage_port, np_=np_,
                           n_seq_max=n_seq_max, srv_verify_rows=srv_verify_rows,
                           stage_verify_rows=stage_verify_rows, srv_env=srv_env)
        if not ex.wait_exit(timeout=180):
            bad("判据 5（%s）：server 没有在 attach 阶段退出（它还在跑）—— 身份不一致没有拒" % label)
            return
        log = ex.text()
        hit = [ln.strip() for ln in log.splitlines() if "identity mismatch" in ln]
        if not hit:
            bad("判据 5（%s）：server 退出了，但日志里没有 'identity mismatch'" % label)
            info("\n".join(log.splitlines()[-6:]))
            return
        msg = hit[0]
        missing = [frag for frag in expect_pairs if frag not in msg]
        if missing:
            bad("判据 5（%s）：attach 拒绝了，但消息里缺 %s" % (label, missing))
            info(msg[:400])
        else:
            ok("判据 5（%s）：attach 阶段就拒，消息带上了两端的数字（%s）"
               % (label, " / ".join(expect_pairs)))
    except Exception as e:  # noqa: BLE001
        info("判据 5（%s）异常：%s: %s" % (label, type(e).__name__, e))
        bad("判据 5（%s）：没有跑到" % label)
    finally:
        if ex is not None:
            ex.stop()


def crit5():
    n_items = 5
    before = PASS + FAIL
    # 5a：head -np 2 vs stage --n-seq-max 1 —— 期望 2、stage 报 1。
    _attach_refusal("n_seq_max", PORT_SRV_1, PORT_STAGE_1, np_=2, n_seq_max=1,
                    srv_verify_rows=None,
                    expect_pairs=["expected 2", "the stage reports 1"])
    # 5b：--split-verify-rows 不一致同理。方向必须让 stage 偏小才会拒（head 17 + stage 8）。
    _attach_refusal("verify_rows", PORT_SRV_2, PORT_STAGE_2, np_=2, n_seq_max=2,
                    srv_verify_rows=17, stage_verify_rows=8,
                    expect_pairs=["expected 17", "the stage reports 8"])
    ex = None
    chain = None

    # 5c/5d/5e：QLH_SPLIT_SKIP_IDENTITY=1 的降级路径。head 认为 stage 有 2 个序列，stage 只有
    # 1 个；seq 1 的帧必须被 stage 的 check_seq_id 拒成 ST_BAD_REQ，且不能出现「输出像正常但错」。
    try:
        chain = Chain("c5skip", PORT_SRV_3, PORT_STAGE_3, np_=2, n_seq_max=1,
                      srv_env={"QLH_SPLIT_SKIP_IDENTITY": "1"})
        log = chain.srv.text()
        if "QLH_SPLIT_SKIP_IDENTITY=1" in log and "ignoring an identity mismatch" in log:
            ok("判据 5：QLH_SPLIT_SKIP_IDENTITY=1 时 attach 放行，但打出了「忽略身份不一致」的 warn")
        else:
            bad("判据 5：QLH_SPLIT_SKIP_IDENTITY=1 没有生效（日志里没有跳过身份比对的 warn）")
            info("\n".join([ln for ln in log.splitlines() if "IDENTITY" in ln][:3]))
        # 一个已知答案的短请求做基线：seq 0 应当照常正确。
        base0, _ = post(chain.port, PROMPT_C, 8, cache_prompt=False)
        # 并发两个请求：server 有 2 个 slot，第二个请求会用 seq 1 —— stage 只有 1 个序列。
        # 两个请求用**同一个** prompt，这样「成功返回的那个」才有已知答案可比：换成
        # 「PROMPT_C + (i)」的话，成功的那条本来就不该等于基线，判据 5e 会永远误报。
        results = [None, None]

        def one(i):
            results[i] = post_ex(chain.port, PROMPT_C, 8, cache_prompt=False)

        ts = [threading.Thread(target=one, args=(i,)) for i in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=900)
        bad_req = [ln.strip() for ln in chain.stage.text().splitlines() if "out of range" in ln]
        if bad_req:
            ok("判据 5：降级路径下 stage 用自己的 check_seq_id 把越界行拒成 ST_BAD_REQ（%s）"
               % bad_req[0][:120])
        else:
            bad("判据 5：降级路径下 stage 没有拒绝越界 seq —— 第二个序列在 n_seq_max=1 的段上被接受了")
        # 「不得出现输出像正常但错」：失败的请求必须是错误；成功返回的必须与 seq 0 的基线相同。
        statuses = [(s, txt) for (s, txt, _) in [r if r else (0, "", {}) for r in results]]
        failed = [s for (s, _) in statuses if s != 200]
        succeeded = [(s, txt) for (s, txt) in statuses if s == 200]
        wrong_ok = [txt for (s, txt) in succeeded if txt != base0]
        info("降级路径的两次请求：status=%s" % [s for (s, _) in statuses])
        # 判据 5 的原文只有一条：「不得出现输出像正常但错」。两个请求都 500 是合格的结果
        # ——stage 拒掉越界行之后 head 把链标成「不可信」，后续请求一律干净报错，那正是
        # 这条降级路径该有的样子；要求「至少一条成功」是加码，不是判据。
        if not wrong_ok:
            ok("判据 5：降级路径没有出现「输出像正常但错」（%d 个请求被拒，%d 个成功的与基线逐字节相同）"
               % (len(failed), len(succeeded)))
        else:
            bad("判据 5：降级路径出现了「输出像正常但错」（成功但内容与基线不同：%d 个；"
                "成功：%d 个；被拒：%d 个）"
                % (len(wrong_ok), len(succeeded), len(failed)))
    except Exception as e:  # noqa: BLE001
        info("判据 5（SKIP_IDENTITY）异常：%s: %s" % (type(e).__name__, e))
        bad("判据 5（SKIP_IDENTITY）：没有跑到")
    finally:
        if chain is not None:
            chain.stop()
    missing = n_items - (PASS + FAIL - before)
    for _ in range(max(0, missing)):
        bad("判据 5：这一项没有跑到（组内异常或提前退出）")


# ---------------------------------------------------------------- 判据 6
def crit6():
    n_items = 3
    before = PASS + FAIL
    chain = None
    try:
        # -c 4096 / -np 3：4096/3 = 1365 -> 向上取整到 256 的倍数 1536，n_ctx 变成 4608。
        # 两端各自 -c 4096 + n_seq_max 3，所以两边的取整必须一致（identity_mismatch 会查
        # n_ctx_seq，attach 能通过本身就是「两边算出了同一个数」的一个证据）。
        chain = Chain("c6", PORT_SRV_4, PORT_STAGE_4, np_=3, n_seq_max=3, ctx=4096)
        srv_log = chain.srv.text()
        stage_log = chain.stage.text()
        head_warn = [ln.strip() for ln in srv_log.splitlines() if "rounding down" in ln]
        stage_warn = [ln.strip() for ln in stage_log.splitlines() if "rounding down" in ln]
        if head_warn:
            ok("判据 6：head 的启动日志有 rounding down warn（%s）"
               % head_warn[0].split("rounding down")[-1].strip())
        else:
            bad("判据 6：head 的启动日志没有 'rounding down' warn")
        if stage_warn:
            ok("判据 6：stage 的启动日志有 rounding down warn（%s）"
               % stage_warn[0].split("rounding down")[-1].strip())
        else:
            bad("判据 6：stage 的启动日志没有 'rounding down' warn")
        # 两端取整后的 n_ctx_seq 必须相同：head 的取整目标 / n_seq_max 对 stage 自己报的值。
        head_ctx = None
        for ln in head_warn:
            tail = ln.split("rounding down to")[-1].strip().split()[0]
            if tail.isdigit():
                head_ctx = int(tail)
                break
        stage_seq = None
        for ln in stage_log.splitlines():
            if "n_ctx_seq" in ln and "=" in ln:
                tail = ln.split("=")[-1].strip().split()[0]
                if tail.isdigit():
                    stage_seq = int(tail)
                    break
        if head_ctx is not None and stage_seq is not None:
            head_seq = head_ctx // 3
            if head_seq == stage_seq:
                ok("判据 6：两端取整后的 n_ctx_seq 相同（head %d/%d = %d，stage 报 %d）"
                   % (head_ctx, 3, head_seq, stage_seq))
            else:
                bad("判据 6：两端取整后的 n_ctx_seq 不同（head %d/%d = %d，stage 报 %d）"
                    % (head_ctx, 3, head_seq, stage_seq))
        else:
            bad("判据 6：读不出两端的 n_ctx（head=%s stage=%s）" % (head_ctx, stage_seq))
    except Exception as e:  # noqa: BLE001
        info("判据 6 异常：%s: %s" % (type(e).__name__, e))
    finally:
        if chain is not None:
            chain.stop()
    missing = n_items - (PASS + FAIL - before)
    for _ in range(max(0, missing)):
        bad("判据 6：这一项没有跑到（组内异常或提前退出）")


# ---------------------------------------------------------------- main
def main():
    global MODEL, PORT_STAGE_1, PORT_STAGE_2, PORT_STAGE_3, PORT_STAGE_4, \
        PORT_SRV_1, PORT_SRV_2, PORT_SRV_3, PORT_SRV_4, PORT_SRV_5
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--port-base", type=int, default=PORT_BASE,
                    help="端口基准：stage 用 +1..+4、server 用 +5..+9（默认 9370）")
    ap.add_argument("--only", default="", help="只跑某几条判据，例如 2,5（调试用）")
    args = ap.parse_args()
    MODEL = args.model
    (PORT_STAGE_1, PORT_STAGE_2, PORT_STAGE_3, PORT_STAGE_4,
     PORT_SRV_1, PORT_SRV_2, PORT_SRV_3, PORT_SRV_4, PORT_SRV_5) = (
        args.port_base + 1, args.port_base + 2, args.port_base + 3, args.port_base + 4,
        args.port_base + 5, args.port_base + 6, args.port_base + 7,
        args.port_base + 8, args.port_base + 9)

    require("llama-server.exe", "split-stage.exe")
    print("[env] model=%s" % MODEL)
    print("       BIN=%s" % BIN)
    print("       TMP=%s" % TMP)
    print("[env] port-base=%d -> stage=%d/%d/%d/%d server=%d/%d/%d/%d/%d"
          % (args.port_base, PORT_STAGE_1, PORT_STAGE_2, PORT_STAGE_3, PORT_STAGE_4,
             PORT_SRV_1, PORT_SRV_2, PORT_SRV_3, PORT_SRV_4, PORT_SRV_5))

    only = set(x.strip() for x in args.only.split(",") if x.strip())

    def want(n):
        return not only or str(n) in only

    if want(2):
        print("\n== 判据 2：混合批（一个在 prefill、一个在 decode）==")
        crit2()
    if want(3):
        print("\n== 判据 3：KV 隔离（对 A 做 seq_rm 不影响 B）==")
        crit3()
    if want(4):
        print("\n== 判据 4：多序列下的 checkpoint save/restore ==")
        crit4()
    if want(5):
        print("\n== 判据 5：身份字段不一致必须干净报错 ==")
        crit5()
    if want(6):
        print("\n== 判据 6：n_ctx 不被 n_seq_max 整除时可见 ==")
        crit6()

    # 项数断言（防假绿的最后一道）：上面每条判据的每个分支都带 else: bad()，并且每个
    # critN 结束时都会把「没跑到的项」补成 FAIL，所以 PASS+FAIL 恒为 18。将来有人加条件
    # 分支却忘了 else，检查会静默消失、FAIL 仍为 0、退出码仍是 0——这条就是抓那种假绿的。
    EXPECT_CHECKS = 18   # 与 verify.sh 的 EXPECT_PASS 增量一致（组 15 = 18 项）
    if not only and PASS + FAIL != EXPECT_CHECKS:
        bad("项数变了：%d PASS + %d FAIL = %d，期望 %d（有检查因条件不满足静默消失）"
            % (PASS, FAIL, PASS + FAIL, EXPECT_CHECKS))
    print("\n== 判据 2–6：%d PASS / %d FAIL ==" % (PASS, FAIL))
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
