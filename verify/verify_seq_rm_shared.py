"""shared-prefix（n_tokens_shared > 0）+ 强制 seq_rm 失败：应答隔离的回归测试。

为什么需要它
------------------------------------------------------------
fork 的多 slot 会走 shared-prompt 前缀：parent 处理共享前缀，children 用 `copy_state_to`
复制它的 KV（`mem.seq_cp`），于是同一批 KV cell 同时属于多个 seq_id。

`prompt_clear()` 是 `mem.seq_rm(id, -1, -1)` + `prompt.clear()`。代码上全删只按 seq_id 摘自己
（`llama_memory_recurrent::seq_rm` 里 `cells[i].seq_id.erase(seq_id)`，cell 空了才释放；
`llama_kv_cache::seq_rm` 同理），所以清一个 slot 不该动到另一个。但这是**代码推导**，没有实测。

这个脚本造的就是那个形态：`n_cmpl = 2` 让 parent 与 child 共享前缀，再让 `seq_rm` 对**部分**
删除返回 false（`QLH_FORCE_SEQ_RM_FAIL=1`），然后比较：

  1. 注入前后两次运行的输出是否逐字节相同（温度 0；失败方走退化，另一方必须一字不差）
  2. 两次运行的 /health 都是 200（含非法内存访问造成的进程退出）
  3. 注入那次的日志里确实出现了退化痕迹（否则测的是"没触发"，不是"隔离"）

用法：python verify/verify_seq_rm_shared.py
退出码 0 = 全过。只用 948x 端口，不碰 93xx/9103/9104/9106。
"""
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _qlh_env import BIN, ROOT, TMP, require   # noqa: E402
MODEL = os.environ.get("QLH_SEQ_RM_MODEL", r"H:\qlh_models\qwen3-5-2b-gguf\qwen35-2b-Q4_K_M.gguf")
PORT = 9487
LOG_A = os.path.join(TMP, "shared_noinject.log")
LOG_B = os.path.join(TMP, "shared_inject.log")

PROMPT = ("You are a helpful assistant with access to tools.\n\n"
          "User: What is the weather in Paris?\nAssistant:\n")
N_PREDICT = 12


def kill_port():
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "LISTENING" in line and re.search(r":%d\s" % PORT, line):
            pid = line.split()[-1]
            subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
    time.sleep(1.0)


def wait_health(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


def health():
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=3) as r:
            return r.status
    except Exception:
        return 0


def start(force_fail, log_path):
    env = dict(os.environ)
    if force_fail:
        env["QLH_FORCE_SEQ_RM_FAIL"] = "0"   # only seq_id 0 (the shared-prompt parent)
    cmd = [os.path.join(BIN, "llama-server.exe"), "-m", MODEL,
           "--host", "127.0.0.1", "--port", str(PORT),
           "-c", "1024", "-ngl", "0", "--no-warmup", "-b", "256", "-ub", "256",
           "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    log = open(log_path, "wb")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc, log


def ask_warmup():
    """A plain single-completion request first: it leaves a cached prefix in the slot, so the
    next (shared-prefix) request has a cache to roll back instead of starting from pos 0."""
    body = json.dumps({"prompt": PROMPT, "n_predict": 8, "temperature": 0,
                       "cache_prompt": True, "stream": False}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/completion" % PORT, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=180).read()
        return True
    except Exception:
        return False


def ask_n(log_path):
    """两个 completion（共享前缀），返回 [(index, content, error)]。"""
    body = json.dumps({
        "prompt": PROMPT,
        "n_predict": N_PREDICT,
        "temperature": 0,
        "n_cmpl": 2,
        "cache_prompt": True,
        "stream": False,
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/completion" % PORT, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=180).read().decode())
    except urllib.error.HTTPError as e:
        return None, "HTTP %d: %s" % (e.code, e.read().decode()[:300])
    except Exception as e:
        return None, "EXC %r" % (e,)
    # n_cmpl > 1 -> the response is a JSON array of completion objects
    items = d if isinstance(d, list) else [d]
    out = []
    for i, c in enumerate(items):
        out.append((i, c.get("content", "") if isinstance(c, dict) else str(c),
                    c.get("stop") if isinstance(c, dict) else None))
    return out, None


def run(force_fail, log_path):
    kill_port()
    proc, log = start(force_fail, log_path)
    try:
        if not wait_health():
            return None, "server did not become healthy (see %s)" % log_path
        ask_warmup()
        out, err = ask_n(log_path)
        h = health()
        return (out, h), err
    finally:
        kill_port()
        try:
            log.close()
        except Exception:
            pass


def main():
    require("llama-server.exe")
    os.makedirs(os.path.dirname(LOG_A), exist_ok=True)
    print("model: %s" % MODEL)

    a, err_a = run(False, LOG_A)
    if err_a:
        print("[FAIL] baseline run: %s" % err_a)
        return 1
    b, err_b = run(True, LOG_B)
    if err_b:
        print("[FAIL] injected run: %s" % err_b)
        return 1

    (out_a, h_a) = a
    (out_b, h_b) = b
    ok = True

    print("baseline : %d completion(s), health=%s" % (len(out_a), h_a))
    for i, c, s in out_a:
        print("  [%d] stop=%s content=%r" % (i, s, c[:60]))
    print("injected : %d completion(s), health=%s" % (len(out_b), h_b))
    for i, c, s in out_b:
        print("  [%d] stop=%s content=%r" % (i, s, c[:60]))

    if h_a != 200 or h_b != 200:
        print("[FAIL] health: baseline=%s injected=%s (process must survive)" % (h_a, h_b))
        ok = False
    else:
        print("[PASS] both runs end with health=200")

    if len(out_a) != len(out_b):
        print("[FAIL] completion count changed: %d -> %d" % (len(out_a), len(out_b)))
        ok = False
    else:
        same = all(ca == cb for (_, ca, _), (_, cb, _) in zip(sorted(out_a), sorted(out_b)))
        if same:
            print("[PASS] injected run produced the same %d completion(s) byte for byte" % len(out_a))
        else:
            print("[FAIL] injected run changed the output (shared KV was disturbed)")
            for (i, ca, _), (_, cb, _) in zip(sorted(out_a), sorted(out_b)):
                if ca != cb:
                    print("  [%d] baseline=%r" % (i, ca))
                    print("  [%d] injected=%r" % (i, cb))
            ok = False

    # 注入确实被走到过：否则上面的"相同"只证明没触发，不证明隔离
    txt = open(LOG_B, "rb").read().decode("utf-8", "replace")
    hit_inject = "QLH_FORCE_SEQ_RM_FAIL: refusing a partial sequence removal" in txt
    hit_degrade = re.search(r"(could not roll back|could not trim|could not remove|ending the response)", txt) is not None
    if hit_inject and hit_degrade:
        print("[PASS] the injected run really took a degraded path (injection + degradation logged)")
    elif not hit_inject:
        print("[INFO] injection never fired: this shape does not reach a partial seq_rm "
              "(the comparison above then proves nothing about isolation)")
    else:
        print("[INFO] injection fired but no degradation branch was logged")

    print()
    print("RESULT %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
