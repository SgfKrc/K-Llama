#!/usr/bin/env python3
"""组 13：KVMem 的 KV 虚拟化（本地双段 + MTP + 驱逐）。

为什么单独一组：组 1-12 全部跑在不带 `--kvmem` 的构建上。KVMem 把 KV 虚拟化到 host RAM、
GPU 只留有界工作集，它接在 `llama_model::create_memory`（与分段链是嵌套关系），所以
「head 的 memory 换成 kvmem」这条路径在别处零覆盖。

四项判据，按重要性：

1. **能跑**：MTP + KVMem 的本地双段能起、能出 token、有真实 draft acceptance。
2. **驱逐**：`budget < prompt` 时 HTTP 200（不是 500），且日志里 `need_offload=1` 真的出现。
   **注意 pool = budget + gen_reserve**，所以触发驱逐的 prompt 必须超过两者之和。
3. **保留的 KV 是对的**：开头埋 secret code + 填充 + 结尾提问，驱逐后仍能答对 code。
   这是唯一能证明「驱逐没让模型看到错的前缀」的判据。
   **不要**用「与整模逐字相等」——`budget < prompt` 时被 stage_out 的 token 不参与 attention，
   逐字相等物理上做不到（详见 docs/design/KVMEM-评估.md §9）。
4. **确定性**：同配置跑两次，输出逐字节相同。

前置（缺任一条就 SKIP，并明说「这不是 PASS」）：

- `LLAMA_KVMEM_ROOT` 指向的 KVMem 源码存在
- 带 `LLAMA_KVMEM=ON` 的构建存在

退出码：0 = 全过；1 = 有 FAIL；2 = 前置缺失（SKIP）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

OK, BAD, SKIP = 0, 1, 2


def log(msg: str) -> None:
    print(msg, flush=True)


def http(url: str, payload: dict | None = None, timeout: float = 900.0):
    """POST /completion 或 GET；返回 (status, body)。"""
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:  # noqa: BLE001 - 网络层任何异常都当作「没拿到」
        return 0, str(e).encode()


def wait_health(port: int, deadline: float) -> bool:
    while time.time() < deadline:
        st, _ = http(f"http://127.0.0.1:{port}/health", timeout=5.0)
        if st == 200:
            return True
        time.sleep(0.5)
    return False


def gen(port: int, prompt: str, n_predict: int = 32, temp: float = 0.0,
        seed: int = 42, cache_prompt: bool = False, timeout: float = 900.0):
    """发一个 completion 请求；返回 (status, message_dict)。"""
    st, body = http(
        f"http://127.0.0.1:{port}/completion",
        {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temp,
            "seed": seed,
            "cache_prompt": cache_prompt,
            "top_k": 1,
        },
        timeout=timeout,
    )
    if st != 200:
        return st, {}
    try:
        return st, json.loads(body)
    except json.JSONDecodeError:
        return st, {}


def msg_of(m: dict) -> str:
    """取整条 message（content + reasoning_content）的规范串。"""
    c = m.get("content") or ""
    r = m.get("reasoning_content") or ""
    return c + "\x00" + r


def acceptance_of(logfile: str) -> str | None:
    try:
        with open(logfile, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    hits = re.findall(r"draft acceptance = ([0-9.]+ \(\s*\d+ accepted /\s*\d+ generated\))", text)
    return hits[-1] if hits else None


def need_offload_count(logfile: str) -> int:
    try:
        with open(logfile, "r", encoding="utf-8", errors="replace") as f:
            return len(re.findall(r"need_offload=1", f.read()))
    except OSError:
        return 0


def kvmem_hybrid_count(logfile: str) -> int:
    try:
        with open(logfile, "r", encoding="utf-8", errors="replace") as f:
            return len(re.findall(r"llama_memory_kvmem_hybrid:", f.read()))
    except OSError:
        return 0


def start_server(server: str, model: str, port: int, budget: int, reserve: int,
                 logfile: str, extra: list[str] | None = None) -> subprocess.Popen:
    args = [
        server,
        "-m", model,
        "--split-layer-range", "0:12",
        "--split-tail-range", "12:24",
        "-c", "4096",
        "-np", "1",
        "-ngl", "0",
        "--kvmem",
        "--kvmem-budget", str(budget),
        "--kvmem-gen-reserve", str(reserve),
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", "2",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--no-warmup",
        "-lv", "10",  # KVMem 的核心 INFO 在 -lv 3 下被过滤，必须 10 才看得见
    ]
    if extra:
        args += extra
    f = open(logfile, "wb")
    env = dict(os.environ)
    env["KVMEM_TRACE"] = "1"  # 驱逐计数靠它；关掉时 adapter 零开销、零输出
    p = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, env=env)
    p._qlh_log = f  # type: ignore[attr-defined]
    return p


def stop_server(p: subprocess.Popen) -> None:
    if p.poll() is None:
        try:
            p.terminate()
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=10)
    f = getattr(p, "_qlh_log", None)
    if f:
        f.close()


# 开头埋 code + 大量填充 + 结尾提问。填充要够长，保证 prompt 超过 pool（budget + gen_reserve）。
SECRET = "XK-7734-QP"
FILLER_SENT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
)
SECRET_PROMPT = (
    f"Remember this secret code: {SECRET}\n\n"
    + FILLER_SENT * 110   # ≈3300 token：超过 pool(3072) 触发驱逐，又不超 -c 4096
    + "\n\nQuestion: what was the secret code? Answer with the code only."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default=os.environ.get("QLH_KVMEM_BUILD_DIR",
                                                          "H:/llamacpp-c/build-kvmem2"))
    ap.add_argument("--model", default=os.environ.get("QLH_MODEL_2B", ""))
    ap.add_argument("--kvmem-root", default=os.environ.get("LLAMA_KVMEM_ROOT", ""))
    ap.add_argument("--port", type=int, default=int(os.environ.get("QLH_VERIFY_PORT_KVMEM", "9361")))
    ap.add_argument("--tmp", default=os.environ.get("QLH_TMP", ""))
    ap.add_argument("--evict-tokens", type=int, default=3600)
    args = ap.parse_args()

    server = os.path.join(args.build_dir, "bin", "llama-server.exe")
    if not os.path.isfile(server):
        log(f"  [SKIP] 组 13 没测：缺 {server}（带 LLAMA_KVMEM=ON 的构建不存在，"
            f"QLH_KVMEM_BUILD_DIR={args.build_dir}）—— 这不是 PASS，KVMem 没有覆盖")
        return SKIP
    if not args.kvmem_root or not os.path.isdir(args.kvmem_root):
        log(f"  [SKIP] 组 13 没测：LLAMA_KVMEM_ROOT 未设置或不存在（{args.kvmem_root!r}）"
            f"—— 这不是 PASS，KVMem 没有覆盖")
        return SKIP
    if not args.model or not os.path.isfile(args.model):
        log(f"  [SKIP] 组 13 没测：2B 模型不存在（{args.model!r}）—— 这不是 PASS")
        return SKIP

    tmp = args.tmp or os.path.join(os.path.dirname(__file__), ".qlh_tmp", "verify")
    os.makedirs(tmp, exist_ok=True)
    log_ok = os.path.join(tmp, "kvmem_ok.log")
    log_ev = os.path.join(tmp, "kvmem_evict.log")
    log_det = os.path.join(tmp, "kvmem_det.log")

    fails = 0
    procs: list[subprocess.Popen] = []
    try:
        # ---------------- ① MTP + KVMem 正常 ----------------
        log("  --- ① MTP + KVMem（budget 8192，不驱逐）---")
        p = start_server(server, args.model, args.port, 8192, 2048, log_ok)
        procs.append(p)
        if not wait_health(args.port, time.time() + 300):
            log(f"  [FAIL] 组 13①：server 没起来，见 {log_ok}")
            tail(log_ok)
            fails += 1
        else:
            st, m = gen(args.port, "The capital of France is", n_predict=16)
            acc = acceptance_of(log_ok)
            hybrids = kvmem_hybrid_count(log_ok)
            if st != 200:
                log(f"  [FAIL] 组 13①：请求失败（HTTP {st}）")
                fails += 1
            elif not acc:
                log("  [FAIL] 组 13①：没有 draft acceptance（MTP 没生效）")
                fails += 1
            elif hybrids < 2:
                log(f"  [FAIL] 组 13①：llama_memory_kvmem_hybrid 只出现 {hybrids} 次（head+tail 应各 ≥1）"
                    f"—— KVMem 没真的替换两段的 memory")
                fails += 1
            else:
                log(f"  [PASS] 组 13①：MTP + KVMem 正常（acceptance {acc}，hybrid ×{hybrids}）")
        stop_server(p)
        procs.remove(p)

        # ---------------- ②③ 驱逐 + secret-code ----------------
        log(f"  --- ②③ 驱逐（budget 1024 + gen_reserve 2048 = pool 3072，prompt ≈{args.evict_tokens} token）---")
        p = start_server(server, args.model, args.port + 1, 1024, 2048, log_ev)
        procs.append(p)
        if not wait_health(args.port + 1, time.time() + 300):
            log(f"  [FAIL] 组 13②：驱逐 server 没起来，见 {log_ev}")
            tail(log_ev)
            fails += 1
        else:
            # n_predict 要给够：qwen35 会先写 thinking，32 token 时还停在「Thinking Process」里，
            # 没走到答案。256 够它想完并答出 code。
            st, m = gen(args.port + 1, SECRET_PROMPT, n_predict=256, timeout=1800)
            offloads = need_offload_count(log_ev)
            if st != 200:
                log(f"  [FAIL] 组 13②：budget < prompt 时请求失败（HTTP {st}）—— 驱逐路径不可用")
                fails += 1
            elif offloads == 0:
                log(f"  [FAIL] 组 13②：请求过了但没有 need_offload=1 —— 没真的触发驱逐"
                    f"（prompt 可能没超过 pool 3072）")
                fails += 1
            else:
                log(f"  [PASS] 组 13②：驱逐下 HTTP 200（need_offload=1 ×{offloads}）")
                content = (m.get("content") or "") + (m.get("reasoning_content") or "")
                if SECRET in content:
                    log(f"  [PASS] 组 13③：驱逐后 secret code 仍答对（{SECRET}）")
                else:
                    log(f"  [FAIL] 组 13③：驱逐后 secret code 丢了（答的是 {content[:80]!r}）"
                        f"—— 被驱逐的块里有关键信息，说明选块策略有问题")
                    fails += 1
        stop_server(p)
        procs.remove(p)

        # ---------------- ④ 确定性 ----------------
        log("  --- ④ 确定性（同配置跑两次）---")
        p = start_server(server, args.model, args.port + 2, 8192, 2048, log_det)
        procs.append(p)
        if not wait_health(args.port + 2, time.time() + 300):
            log(f"  [FAIL] 组 13④：确定性 server 没起来，见 {log_det}")
            fails += 1
        else:
            p1 = "Count from one to ten, comma separated."
            st1, m1 = gen(args.port + 2, p1, n_predict=32)
            st2, m2 = gen(args.port + 2, p1, n_predict=32)
            if st1 != 200 or st2 != 200:
                log(f"  [FAIL] 组 13④：两次请求有失败（HTTP {st1}/{st2}）")
                fails += 1
            else:
                a = msg_of(m1).encode()
                b = msg_of(m2).encode()
                if a == b:
                    log(f"  [PASS] 组 13④：两次输出 byte-identical"
                        f"（sha256 {hashlib.sha256(a).hexdigest()[:16]}…）")
                else:
                    log(f"  [FAIL] 组 13④：两次输出不一致"
                        f"（{hashlib.sha256(a).hexdigest()[:16]} vs {hashlib.sha256(b).hexdigest()[:16]}）")
                    fails += 1
        stop_server(p)
        procs.remove(p)
    finally:
        for p in procs:
            stop_server(p)

    return BAD if fails else OK


def tail(path: str, n: int = 8) -> None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[-n:]
        for ln in lines:
            log("    " + ln.rstrip())
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
