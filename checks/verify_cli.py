"""CLI 侧的等价判据：llama-server 的 raw /completion（不走 chat 模板）。

为什么不用 llama-cli：本仓库 H:\\llamacpp-c 的 llama-cli 已经不是 completion 工具，
而是纯 chat 客户端（tools/cli/cli-client.cpp:55 只 POST /v1/chat/completions，main.cpp
只有 116 字节的转发壳）。它无法发 raw prompt，也不打印生成 token id，输出还经过 UI
折行。llama-server 的 /completion 是 raw prompt + 同一套 llama.cpp 采样代码，且自带
/tokenize，能把生成文本还原成 token id 供比对（还原是 detokenize∘tokenize，属于
本判据的弱点：文本边界可能改变 BPE 合并）。

判据 E1-cli：raw prompt 下 greedy 生成 N token，本项目 chain 的 token id 序列应等于
            "server 文本再 tokenize" 的 id 序列；文本应逐字符相同。
判据 E2-cli：greedy 下同一请求重复两次必须给出同一文本（确定性检查，抓采样状态泄漏）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BIN = Path(r"H:\llamacpp-c\build-cuda\bin")
MODEL = Path(r"H:\qlh_models\qwen3-6-35b-a3b-opus47-distill-apex-mtp"
             r"\Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-I-Quality.gguf")


def _post(url: str, obj: dict, timeout: float = 900.0) -> dict:
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _healthy(base: str) -> bool:
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001 - 503 while loading is the normal path
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-text", default="def quicksort(arr):")
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--port", type=int, default=8137)
    ap.add_argument("--gpu-layers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--chat", action="store_true",
                    help="use /v1/chat/completions (jinja template) instead of raw /completion")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    log = Path(".qlh_tmp/server-cli.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    exe = BIN / "llama-server.exe"
    if not exe.is_file():
        raise SystemExit(f"llama-server not found: {exe}")
    env = dict(os.environ)
    env["PATH"] = str(BIN) + os.pathsep + env.get("PATH", "")
    handle = open(log, "w", encoding="utf-8", errors="replace")
    t0 = time.time()
    proc = subprocess.Popen(
        [str(exe), "-m", str(MODEL), "-c", str(args.ctx), "-ngl", str(args.gpu_layers),
         "-t", str(args.threads), "--host", "127.0.0.1", "--port", str(args.port),
         "--no-webui"],
        stdout=handle, stderr=subprocess.STDOUT, cwd=str(BIN))
    try:
        deadline = time.time() + 300
        while time.time() < deadline:
            if proc.poll() is not None:
                raise SystemExit(f"llama-server died early, see {log}")
            if _healthy(base):
                break
            time.sleep(1.0)
        else:
            raise SystemExit(f"llama-server never became healthy, see {log}")
        load_s = time.time() - t0

        req = {"prompt": args.prompt_text, "n_predict": args.gen, "temperature": 0.0,
               "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repeat_last_n": 0,
               "repeat_penalty": 1.0, "cache_prompt": False, "stream": False}
        chat_req = {"messages": [{"role": "user", "content": args.prompt_text}],
                    "n_predict": args.gen, "temperature": 0.0, "top_k": 0, "top_p": 1.0,
                    "min_p": 0.0, "stream": False}
        path = "/v1/chat/completions" if args.chat else "/completion"
        t1 = time.time()
        first = _post(base + path, chat_req if args.chat else req)
        run_s = time.time() - t1
        second = _post(base + path, chat_req if args.chat else req)
        if args.chat:
            choice = first.get("choices", [{}])[0]
            msg = choice.get("message", {}) or {}
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            again = ((second.get("choices", [{}])[0].get("message", {}) or {})
                     .get("content") or "")
            predicted = first.get("usage", {}).get("completion_tokens")
        else:
            content = first.get("content") or ""
            reasoning = ""
            again = second.get("content") or ""
            predicted = first.get("tokens_predicted")
        ids = _post(base + "/tokenize", {"content": content})
        prompt_ids = _post(base + "/tokenize", {"content": args.prompt_text})
        applied = None
        if args.chat:
            try:
                applied = _post(base + "/apply-template",
                                {"messages": [{"role": "user", "content": args.prompt_text}]}
                                ).get("prompt")
            except Exception as exc:  # noqa: BLE001 - endpoint may be absent
                applied = f"<apply-template failed: {exc}>"

        result = {
            "side": ("llama-server /v1/chat/completions (jinja)"
                     if args.chat else "llama-server /completion (raw)"),
            "prompt_text": args.prompt_text,
            "prompt_ids": prompt_ids.get("tokens"),
            "gen": args.gen,
            "load_s": round(load_s, 1), "run_s": round(run_s, 1),
            "content": content,
            "reasoning_content": reasoning,
            "content_again": again,
            "deterministic": content == again,
            "tokens_predicted": predicted,
            "content_ids": ids.get("tokens"),
            "applied_prompt": applied,
        }
        print(json.dumps(result, ensure_ascii=False, indent=1))
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
