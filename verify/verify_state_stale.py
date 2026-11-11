"""v6 快照句柄的过期路径：stage 缓存把 checkpoint 淘汰掉之后，恢复必须「无害失败」。

要证明的三件事（都是这次改动引入的新失败模式）：
  1. stage 回 ST_STALE，head 不许把链判死：请求必须 200，不能 500、不能 abort；
  2. 恢复失败的 checkpoint 被丢弃，请求退化成「从头重算」，于是 prompt_n = 完整 prompt；
  3. 丢掉缓存只是慢，不能算错：同一个请求在不淘汰（--state-cache 32）的链上必须逐字节相同。

用法：python verify/verify_state_stale.py
退出码 0 = 全过。会杀掉 93xx 上的自己起的进程（不碰 9103/9104/9106）。
"""
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _qlh_env import BIN, ROOT, TMP, require   # noqa: E402
MODEL = os.environ.get("QLH_SPLIT_MODEL", r"H:\qlh_models\qwen3-5-2b-gguf\qwen35-2b-Q4_K_M.gguf")

HEAD_PORT = 9361
STAGE_PORT = 9362
STAGE_RANGE = "16:24"
HEAD_RANGE = "0:16"


def kill_ours():
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "LISTENING" in line and re.search(r":93[0-9][0-9]\s", line):
            pid = line.split()[-1]
            subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
    time.sleep(1.5)


def wait_health(port, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


def post(port, payload, timeout=600):
    req = urllib.request.Request(
        "http://127.0.0.1:%d/completion" % port, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def start_chain(tag, state_cache):
    stage_log = os.path.join(TMP, "stale_stage_%s.log" % tag)
    head_log = os.path.join(TMP, "stale_head_%s.log" % tag)
    sf = open(stage_log, "wb")
    stage = subprocess.Popen(
        [os.path.join(BIN, "split-stage.exe"), "-m", MODEL, "--layer-range", STAGE_RANGE,
         "--role", "tail", "--rs-seq", "8", "--host", "127.0.0.1", "--port", str(STAGE_PORT),
         "-c", "2048", "-b", "512", "--state-cache", str(state_cache)],
        stdout=sf, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 240:
        # [v11] ready: 而不是 listening：v10 把 bind 提到了加载之前，端口绑上不等于能服务。
        if b"ready:" in open(stage_log, "rb").read():
            break
        time.sleep(0.5)
    hf = open(head_log, "wb")
    head = subprocess.Popen(
        [os.path.join(BIN, "llama-server.exe"), "-m", MODEL, "--host", "127.0.0.1",
         "--port", str(HEAD_PORT), "-c", "2048", "-ngl", "0", "--no-warmup", "-np", "1",
         "-lv", "5", "--split-layer-range", HEAD_RANGE,
         "--split-remote", "127.0.0.1:%d" % STAGE_PORT],
        stdout=hf, stderr=subprocess.STDOUT)
    ok = wait_health(HEAD_PORT)
    return stage, head, ok, stage_log, head_log


def main():
    require("split-stage.exe", "llama-server.exe")
    kill_ours()
    fails = []

    def ok(msg):
        print("  [PASS] %s" % msg, flush=True)

    def bad(msg):
        print("  [FAIL] %s" % msg, flush=True)
        fails.append(msg)

    base = ("Name the capital of France and then three rivers of Europe. " * 10)
    # A 建出这份 checkpoint；B 往后长一段并建出新的（--state-cache 1 时挤掉 A 的）；
    # C 从 A 之后立刻分叉，它要的正好是 A 那份（pos_max=116、n_tokens=117）——已经没了。
    # B/C 必须在 base 之后第一个 token 就分开，否则它们会共享 B 那份更新的 checkpoint，
    # 测到的就是「命中」而不是「淘汰」（第一版脚本踩过：C 拿到 B 的 122 号快照）。
    prompt_a = base
    prompt_b = base + " Zebra. Zebra. Zebra."
    prompt_c = base + " Yak. Yak. Yak."

    print("== 1. 淘汰路径（stage --state-cache 1） ==", flush=True)
    stage, head, up, stage_log, head_log = start_chain("evict", 1)
    if not up:
        bad("head 没起来")
        kill_ours()
        return 1
    for tag, p in (("A", prompt_a), ("B", prompt_b)):
        r = post(HEAD_PORT, {"prompt": p, "n_predict": 4, "temperature": 0,
                             "cache_prompt": True})
        print("  [info] %s: prompt_n=%s cache_n=%s" % (tag, r["timings"].get("prompt_n"),
                                                       r["timings"].get("cache_n")), flush=True)
    r_c = post(HEAD_PORT, {"prompt": prompt_c, "n_predict": 4, "temperature": 0,
                           "cache_prompt": True})
    content_c = r_c.get("content")
    tim = r_c["timings"]
    print("  [info] C: prompt_n=%s cache_n=%s" % (tim.get("prompt_n"), tim.get("cache_n")), flush=True)

    slog = open(stage_log, "r", encoding="utf-8", errors="replace").read()
    hlog = open(head_log, "r", encoding="utf-8", errors="replace").read()

    if re.search(r"handle \d+ -> STALE", slog):
        ok("stage 回了 STALE（快照被淘汰）")
    else:
        bad("stage 日志里没有 STALE：这次没触发淘汰路径")
    if "re-processing the prompt" in hlog:
        ok("head 认出了过期 checkpoint 并选择重算（不是 abort、不是 500）")
    else:
        bad("head 日志里没有『re-processing the prompt』")
    if r_c is not None and tim.get("prompt_n") == tim.get("prompt_n") and (tim.get("prompt_n") or 0) > 100 \
            and tim.get("cache_n") == 0:
        ok("请求仍然 200，且退化成从头重算（prompt_n=%s, cache_n=0）" % tim.get("prompt_n"))
    else:
        bad("淘汰后的请求没有退化成重算：prompt_n=%s cache_n=%s"
            % (tim.get("prompt_n"), tim.get("cache_n")))
    alive = False
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % HEAD_PORT, timeout=3) as h:
            alive = h.status == 200
    except Exception:
        pass
    ok("head 进程还活着") if alive else bad("head 进程没了（abort）")

    head.kill(); stage.kill()
    kill_ours()

    print("== 2. 对照：同样三段请求、不淘汰（--state-cache 32） ==", flush=True)
    stage, head, up, stage_log2, head_log2 = start_chain("nocache", 32)
    if not up:
        bad("对照链的 head 没起来")
        kill_ours()
        return 1
    for tag, p in (("A", prompt_a), ("B", prompt_b), ("C", prompt_c)):
        r = post(HEAD_PORT, {"prompt": p, "n_predict": 4, "temperature": 0,
                             "cache_prompt": True})
        if tag == "C":
            content_ref = r.get("content")
            tim_ref = r["timings"]
    print("  [info] 对照 C: prompt_n=%s cache_n=%s" % (tim_ref.get("prompt_n"),
                                                      tim_ref.get("cache_n")), flush=True)
    if content_ref == content_c:
        ok("淘汰链与不淘汰链的输出逐字节相同（%d 字符）" % len(content_c or ""))
    else:
        bad("输出不同：%r vs %r" % ((content_c or "")[:120], (content_ref or "")[:120]))

    slog2 = open(stage_log2, "r", encoding="utf-8", errors="replace").read()
    if "-> STALE" not in slog2:
        ok("对照链没有出现 STALE（32 份缓存没有淘汰）")
    else:
        bad("对照链也出现了 STALE：缓存上限算错了")

    head.kill(); stage.kill()
    kill_ours()
    print()
    print("RESULT: %s" % ("PASS" if not fails else "FAIL (%d)" % len(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
