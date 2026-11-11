"""量化「远程链每个请求的 tail 状态回传」：head -> proxy(9333) -> tail(9332)。

为什么要这个：13.17 MB/请求 的说法只有一个手写的 netstat 观感，没有可重跑的分方向计数。
这个脚本把 head 到 tail 的连接穿过一个计数代理，按「每次 HTTP 请求的时间窗」累加
up（head->tail）和 dn（tail->head）的字节数，把 blob 传输从 decode/logits 里分出来。

用法：
  python verify/measure_state_bytes.py                 # 默认 2B 模型，8 层尾段
  python verify/measure_state_bytes.py --tag before    # 输出里的标签

输出：每个请求一行（cache_prompt、prompt_n、cache_n、dn KB、up KB），最后是汇总。
dn 里包含该请求里 tail 回传的 checkpoint blob（每次 checkpoint 一份）和 logits。
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _qlh_env import BIN, ROOT, TMP, require   # noqa: E402
MODEL = os.environ.get("QLH_SPLIT_MODEL", r"H:\qlh_models\qwen3-5-2b-gguf\qwen35-2b-Q4_K_M.gguf")

HEAD_PORT = 9330
STAGE_PORT = 9332
PROXY_PORT = 9333


class Proxy:
    """单个连接的 TCP 计数代理：记录 (t, dir, n)，按时间窗求和。"""

    def __init__(self, listen, target):
        self.listen = listen
        self.target = target
        self.events = []
        self.lock = threading.Lock()
        self.t0 = time.perf_counter()
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", listen))
        self.srv.listen(4)
        self.stopped = False
        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.thread.start()

    def _accept_loop(self):
        while not self.stopped:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            try:
                r = socket.socket()
                r.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                r.connect(self.target)
                c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                c.close()
                continue
            threading.Thread(target=self._pump, args=(c, r, "up"), daemon=True).start()
            threading.Thread(target=self._pump, args=(r, c, "dn"), daemon=True).start()

    def _pump(self, a, b, tag):
        try:
            while True:
                data = a.recv(1 << 16)
                if not data:
                    break
                with self.lock:
                    self.events.append((time.perf_counter() - self.t0, tag, len(data)))
                b.sendall(data)
        except OSError:
            pass
        finally:
            try:
                b.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    def window(self, t_a, t_b):
        with self.lock:
            ev = list(self.events)
        up = sum(n for t, d, n in ev if d == "up" and t_a <= t < t_b)
        dn = sum(n for t, d, n in ev if d == "dn" and t_a <= t < t_b)
        return up, dn

    def close(self):
        self.stopped = True
        try:
            self.srv.close()
        except OSError:
            pass


def wait_health(port, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def post(port, payload, timeout=600):
    req = urllib.request.Request(
        "http://127.0.0.1:%d/completion" % port,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def kill_port(port):
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
    pids = set()
    for line in out.splitlines():
        if "LISTENING" in line and (":%d " % port) in line:
            pids.add(line.split()[-1])
    for pid in pids:
        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)


def main():
    require("split-stage.exe", "llama-server.exe")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--layer-range", default="0:16")
    ap.add_argument("--tail-range", default="16:24")
    ap.add_argument("-c", "--ctx", type=int, default=2048)
    args = ap.parse_args()

    logdir = os.path.join(TMP, "measure")
    os.makedirs(logdir, exist_ok=True)
    stage_log = open(os.path.join(logdir, "stage_%s.log" % args.tag), "wb")
    head_log = open(os.path.join(logdir, "head_%s.log" % args.tag), "wb")

    stage = subprocess.Popen(
        [os.path.join(BIN, "split-stage.exe"), "-m", MODEL,
         "--layer-range", args.tail_range, "--role", "tail", "--rs-seq", "8",
         "--host", "127.0.0.1", "--port", str(STAGE_PORT),
         "-c", str(args.ctx), "-b", "512"],
        stdout=stage_log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 240:
        stage_log.flush()
        with open(os.path.join(logdir, "stage_%s.log" % args.tag), "rb") as f:
            # [v11] ready: 而不是 listening：bind 在加载之前，端口绑上不代表模型已就绪。
            if b"ready:" in f.read():
                break
        time.sleep(0.5)
    else:
        print("stage did not start", file=sys.stderr)
        return 1

    proxy = Proxy(PROXY_PORT, ("127.0.0.1", STAGE_PORT))

    head = subprocess.Popen(
        [os.path.join(BIN, "llama-server.exe"), "-m", MODEL,
         "--host", "127.0.0.1", "--port", str(HEAD_PORT),
         "-c", str(args.ctx), "-ngl", "0", "--no-warmup", "-np", "1",
         "--split-layer-range", args.layer_range,
         "--split-remote", "127.0.0.1:%d" % PROXY_PORT],
        stdout=head_log, stderr=subprocess.STDOUT)
    if not wait_health(HEAD_PORT):
        print("head did not come up", file=sys.stderr)
        head.kill(); stage.kill(); proxy.close()
        return 1

    base = ("Name the capital of France and then three rivers of Europe. " * 12)
    prompts = [
        base,
        base + " Now name two mountains.",
        base + " Now name two mountains. And two oceans.",
    ]

    rows = []
    for cache_prompt in (False, True):
        for i, p in enumerate(prompts):
            ta = time.perf_counter() - proxy.t0
            t1 = time.time()
            rsp = post(HEAD_PORT, {"prompt": p, "n_predict": 4, "temperature": 0,
                                   "cache_prompt": cache_prompt})
            dt = time.time() - t1
            tb = time.perf_counter() - proxy.t0
            up, dn = proxy.window(ta, tb)
            tm = rsp.get("timings", {})
            rows.append(dict(cache_prompt=cache_prompt, i=i,
                             prompt_n=tm.get("prompt_n"), cache_n=tm.get("cache_n"),
                             dn=dn, up=up, dt=dt))
            print("cache_prompt=%-5s #%d prompt_n=%-4s cache_n=%-4s dn=%9d up=%8d %.2fs"
                  % (cache_prompt, i, tm.get("prompt_n"), tm.get("cache_n"), dn, up, dt), flush=True)

    print("\n== %s ==" % args.tag)
    for r in rows:
        print("  cache_prompt=%-5s #%d dn=%9d B (%6.2f MiB) up=%8d B"
              % (r["cache_prompt"], r["i"], r["dn"], r["dn"] / 1048576.0, r["up"]))

    # 句柄方案把状态留在 stage 里，所以 stage 的常驻内存是这次改动的成本项：量一下。
    def rss_mib(pid):
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-Process -Id %d).WorkingSet64" % pid],
                capture_output=True, text=True, timeout=30).stdout.strip()
            return float(out) / 1048576.0
        except Exception:
            return float("nan")

    print("  stage RSS = %.1f MiB, head RSS = %.1f MiB"
          % (rss_mib(stage.pid), rss_mib(head.pid)))

    head.kill(); stage.kill()
    proxy.close()
    time.sleep(1)
    kill_port(HEAD_PORT); kill_port(STAGE_PORT); kill_port(PROXY_PORT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
