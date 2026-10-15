#!/usr/bin/env python
"""relay_mid_service.py — 跨机接力的**远端段**服务（P2 多段拓扑的网络侧）。

链路：`torch 上游 → [网络] → 远端段 → [网络] → 下一段/末段`。
本服务提供两种远端角色，正好是 `src/relay_transport.py` 的两种往返：

| `--role` | 语义 | 协议 | 本地等价物 |
|---|---|---|---|
| `middle` | 吃 hidden → 吐 **hidden**（末层输出，`output_norm` 之前） | HIDDEN → HIDDEN | `llama_keep_head.KeepHeadUpstream.forward_hidden_to_hidden` |
| `tail` | 吃 hidden → 吐 **token**（末位 argmax） | HIDDEN → TOKEN | `llama_engine.LlamaCppEngine.forward_layers_from_hidden` |

⚠️ 语义纪律：`middle` 必须用 **keep-head 通道**（补丁导出的 nextn / 层输入），
不能用 `llama_get_embeddings_ith`（那是 `output_norm(H)`，多一次归一化 ⇒ 下游分叉）。
Android 段用同一语义的 JNI 入口：`nativeLayerForwardHiddenKeepHead`。

本服务只允许 **loopback 绑定**（`open_loopback_listener` 强制）—— 跨机时用 SSH 隧道把
远端端口映射到本机 loopback（与本仓既有 relay 纪律一致），不直接把端口暴露到 LAN/tailnet。

用法::

    # 远端（或本机另一进程）：中间段
    python scripts/relay_mid_service.py --role middle --listen 127.0.0.1:50161 \
        --keep-head-shim build/keephead/build-cpu/bin/qlh_keep_head.dll \
        --model build/cross-framework-layer-poc/out/qwen25-05b-f16-mid8-16.gguf \
        --threads 8 --ready-file build/relay-records/mid.ready
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.relay_transport import (  # noqa: E402
    RELAY_DEFAULT_MAX_TOKENS,
    open_loopback_listener,
    serve_relay_connection,
    serve_relay_middle_connection,
)


class KeepHeadMiddleRunner:
    """中间段：吃 hidden → 吐 hidden（keep-head 语义）。模型只加载一次，多次连接复用。"""

    def __init__(self, *, shim: str, model: str, n_ctx: int, n_threads: int,
                 n_seq_max: int, n_batch: int, mode: str = "nextn",
                 cut_layer: int | None = None, extra_dll_dirs: list[str]) -> None:
        import os  # noqa: PLC0415

        from llama_keep_head import KeepHeadUpstream  # noqa: PLC0415

        dirs = list(extra_dll_dirs)
        dirs.extend(d for d in (os.environ.get("QLH_KEEP_HEAD_DLL_DIRS") or "")
                    .split(os.pathsep) if d)
        self._upstream = KeepHeadUpstream(shim, model, mode=mode, cut_layer=cut_layer,
                                          n_ctx=n_ctx,
                                          n_threads=n_threads,
                                          n_seq_max=max(1, int(n_seq_max)),
                                          n_batch=max(512, int(n_batch)),
                                          extra_dll_dirs=dirs)
        self.n_embd = self._upstream.n_embd
        self.n_layer = self._upstream.n_layer
        self.n_seq_max = max(1, int(n_seq_max))
        self.mode = mode
        self.cut_layer = cut_layer
        self._pos = 0
        print(json.dumps({"role": "middle", "n_embd": self.n_embd,
                          "n_layer": self.n_layer, "n_seq_max": self.n_seq_max,
                          "n_batch": max(512, int(n_batch)), "mode": mode,
                          "cut_layer": cut_layer,
                          "channel": "keep_head_nextn" if mode == "nextn"
                                     else "layer_inp"}), flush=True)

    def reset(self) -> None:
        """每条连接从干净状态开始：位置归零 + **清 KV/recurrent 记忆**（同进程多连接必需）。"""
        self._pos = 0
        self._upstream.reset()

    def request_hidden(self, hidden_bytes: bytes, *, n_tokens: int) -> bytes:
        import numpy as np  # noqa: PLC0415

        count = int(n_tokens)
        incoming = np.frombuffer(hidden_bytes, dtype=np.float32).reshape(count, self.n_embd)
        produced = self._upstream.forward_hidden_to_hidden(incoming, n_past=self._pos)
        self._pos += count
        return produced.astype(np.float32).tobytes()

    def request_hidden_seq(self, hidden_bytes: bytes, *, n_tokens: int,
                           meta: dict[str, object]) -> bytes:
        """★ P3 多序列：按帧里的 `seq_ids` / `positions` 显式绑定（不依赖本段位置累加）。

        远端每步都拿到**完整位置**，因此不需要（也不应该）维护 `self._pos`。
        """
        import numpy as np  # noqa: PLC0415

        count = int(n_tokens)
        incoming = np.frombuffer(hidden_bytes, dtype=np.float32).reshape(count, self.n_embd)
        produced = self._upstream.forward_hidden_to_hidden(
            incoming,
            seq_ids=meta.get("seq_ids"),
            positions=meta.get("positions"))
        return produced.astype(np.float32).tobytes()

    def close(self) -> None:
        """连接级清理：**不卸载模型**（下一个连接继续复用）。"""
        self.reset()


class TailRunner:
    """末段：吃 hidden → 吐 token（主仓 llama_engine）。

    ★ 2026-09-23：pip 绑定的 `Llama.__init__` 会**无条件**把 `flash_attn_type` 写成
    `DISABLED`、把 `n_threads_batch` 写成 `multiprocessing.cpu_count()`；而自建 shim 那份
    llama.cpp 是 `flash_attn = auto → enabled`、`n_threads_batch = n_threads`。这两项在
    32-token prefill（`ubatch.n_tokens > 1`）上都会改变浮点结果 ⇒ 与 shim 段**混用**时可能把
    近并列的 top-1 翻转（实测：同一 head 段下 9B 两段 pip tail 1/32、shim tail 32/32）。
    这里把两者暴露为可显式对齐的参数；不传则保持 pip 默认（便于复现差异本身）。
    """

    def __init__(self, *, model: str, n_ctx: int, n_threads: int,
                 flash_attn: bool = True, n_threads_batch: int | None = None) -> None:
        from llama_engine import LlamaCppEngine  # noqa: PLC0415

        # 默认与自建 shim 取齐（`flash_attn=True`、`n_threads_batch=n_threads`）：
        #   - flash attention 是**判据关键**：pip 绑定默认 DISABLED 时 9B 两段 1/32，
        #     打开后 32/32（单变量实测：只对齐 batch 线程仍 1/32）；
        #   - batch 线程数不是判据关键，但取齐可避免无谓的浮点差异。
        load_kwargs: dict[str, object] = {
            "flash_attn": bool(flash_attn),
            "n_threads_batch": (int(n_threads_batch)
                                if (n_threads_batch is not None and int(n_threads_batch) > 0)
                                else int(n_threads)),
        }
        self._engine = LlamaCppEngine()
        self._engine.load_model(model_path=str(model), n_ctx=n_ctx, n_threads=n_threads,
                               n_seq_max=1, **load_kwargs)
        if not self._engine.is_loaded:
            raise RuntimeError(f"下游模型加载失败：{model}")
        import llama_cpp.llama_cpp as M  # noqa: PLC0415

        native = self._engine._model._model.model
        self.n_embd = int(M.llama_model_n_embd_inp(native))
        self.n_layer = int(M.llama_model_n_layer(native))
        self._pos = 0
        # ★ 逐段引擎/参数标识（§10.2 待办的一部分）：写清实际生效的构建与开关
        print(json.dumps({"role": "tail", "n_embd": self.n_embd, "n_layer": self.n_layer,
                          "channel": "llama_engine.forward_layers_from_hidden",
                          "flash_attn": bool(flash_attn),
                          "n_threads_batch": (int(n_threads_batch)
                                              if n_threads_batch is not None else None),
                          "llama_cpp_version": getattr(
                              __import__("llama_cpp"), "__version__", "?")}), flush=True)

    def reset(self) -> None:
        """★ 会话级清理：下一个任务必须能从位置 0 重新开始。

        与 head/middle 不同（它们的 KV 在 `self._upstream` 内，由
        `KeepHeadUpstream.reset()` 清），本类**直连** `LlamaCppEngine`，而
        `forward_layers_from_hidden` 是"**直接在 KV 里占 `[n_past, n_past+n_tokens)`**"
        （见 `src/llama_engine.py` 该方法说明）—— 下一个任务又从位置 0 开始
        ⇒ 不在同一位置重跑前清 KV 就会 `llama_decode rc=-1`。

        实测（2026-09-30 三机产品路径验收）：只重置 `self._pos` 时**第二个**
        `POST /api/chat` 必失败，远端日志
        `Relay tail runner (seq) failed: code=runner_failed` + `llama_decode 失败 rc=-1`，
        `[session] frames=0 tokens=0 closed_cleanly=False error=runner_failed`。
        `LlamaCppEngine.reset_kv_cache()` 是既有的 stateless no-op，**不能**替代
        （`llama_engine` 的原话）。
        """
        self._pos = 0
        self._clear_native_kv()

    def _clear_native_kv(self) -> None:
        """清本段直连的那份原生 KV（`llama_cpp.LlamaContext.kv_cache_clear`）。"""
        ctx = getattr(getattr(self._engine, "_model", None), "_ctx", None)
        clear = getattr(ctx, "kv_cache_clear", None)
        if callable(clear):
            clear()

    def request_token(self, hidden_bytes: bytes, *, n_tokens: int) -> int:
        import numpy as np  # noqa: PLC0415

        count = int(n_tokens)
        incoming = np.frombuffer(hidden_bytes, dtype=np.float32).reshape(count, self.n_embd)
        logits = self._engine.forward_layers_from_hidden(incoming, n_past=self._pos,
                                                         all_logits=True)
        if logits is None:
            return -1
        self._pos += count
        return int(np.asarray(logits)[-1].argmax())

    def request_token_seq(self, hidden_bytes: bytes, *, n_tokens: int,
                          meta: dict[str, object]):
        """★ 2026-10-14（PIPE-COMM-01）：多位置回程 —— 返回 **每个位置各一个 argmax token**。

        返回 `list[int]` 时，`serve_relay_connection` 会走 `TOKEN_SEQ` 帧一次回 γ 个 token
        （见 `RelayFrameKind.TOKEN_SEQ` 的说明）。历史实现返回 int 仍被支持（单 token 回程）。

        ⚠️ 本路径的逐位置 argmax **不保证与逐 token 顺序推进等价**（各位置只依赖其之前的
        上下文），只用于验证 RTT 摊薄的物理收益。
        """
        import numpy as np  # noqa: PLC0415

        count = int(n_tokens)
        incoming = np.frombuffer(hidden_bytes, dtype=np.float32).reshape(count, self.n_embd)
        logits = self._engine.forward_layers_from_hidden(
            incoming,
            seq_ids=meta.get("seq_ids"),
            positions=meta.get("positions"),
            all_logits=True,
        )
        if logits is None:
            return -1
        arr = np.asarray(logits)
        # `all_logits=True` ⇒ 形状 [n_tokens, vocab]；每行是该位置的 logits。
        if arr.ndim >= 2 and arr.shape[0] == count:
            return [int(row.argmax()) for row in arr]
        # 形状不符（例如只给了末位）⇒ 如实退化，不冒充多位置结果。
        if arr.ndim == 1:
            return -1
        return -1

    def close(self) -> None:
        self.reset()


class HeadRunner:
    """★ P4.5 **上游段**（无 PC 集群）：吃 **token** → 吐 hidden（本节点自己的 head 段）。

    为什么需要：层接力要能**完全不依赖本机**运行（集群里可能没有任何 PC / 能跑 torch 的节点）。
    上游段只需 keep-head 的 token 入口（`forward_tokens_to_hidden`，shim 早已支持），
    因此设备侧无需 pip `llama_cpp` —— 与 `middle` / `ShimTailRunner` 共用同一份 shim。

    走 `serve_relay_middle_connection` 的会话循环（它支持 `TOKENS → HIDDEN`）。
    """

    def __init__(self, *, shim: str, model: str, n_ctx: int, n_threads: int,
                 mode: str = "nextn", cut_layer: int | None = None,
                 extra_dll_dirs: list[str], n_seq_max: int = 1,
                 n_batch: int = 512) -> None:
        import os  # noqa: PLC0415

        from llama_keep_head import KeepHeadUpstream  # noqa: PLC0415

        dirs = list(extra_dll_dirs)
        dirs.extend(d for d in (os.environ.get("QLH_KEEP_HEAD_DLL_DIRS") or "")
                    .split(os.pathsep) if d)
        self._upstream = KeepHeadUpstream(shim, model, mode=mode, cut_layer=cut_layer,
                                          n_ctx=n_ctx, n_threads=n_threads,
                                          n_seq_max=max(1, int(n_seq_max)),
                                          n_batch=max(512, int(n_batch)),
                                          extra_dll_dirs=dirs)
        self.n_embd = self._upstream.n_embd
        self.n_layer = self._upstream.n_layer
        self._pos = 0
        print(json.dumps({"role": "head", "n_embd": self.n_embd, "n_layer": self.n_layer,
                          "channel": "keep_head.forward_tokens_to_hidden",
                          "mode": mode, "cut_layer": cut_layer}), flush=True)

    def reset(self) -> None:
        self._pos = 0
        self._upstream.reset()

    def request_hidden_from_tokens(self, tokens: list[int]) -> bytes:
        import numpy as np  # noqa: PLC0415

        hidden = self._upstream.forward_tokens_to_hidden([int(t) for t in tokens],
                                                         n_past=self._pos)
        self._pos += len(tokens)
        return np.ascontiguousarray(hidden, dtype=np.float32).tobytes()

    def close(self) -> None:
        self.reset()
        self._upstream.close()


class ShimTailRunner:
    """末段（**无 pip `llama_cpp` 的设备**）：吃 hidden → 吐 token，走 **keep-head shim**。

    为什么需要它（而不是"在设备上装 llama-cpp-python"）：

    1. `TailRunner` 依赖 pip `llama_cpp`，而 head/middle 需要的
       `llama_get_embeddings_nextn_ith` / `llama_get_embeddings_layer_inp` /
       `llama_set_embeddings_layer_inp` 这些**补丁符号在 pip 的绑定层里不存在**（实测：三个符号
       在 `llama_cpp.llama_cpp` 里全部 MISSING，而标准符号都在）⇒ 用 pip 版做 head/middle
       不仅要换它 vendor 的 llama.cpp，**还要改 `llama_cpp.py` 补声明并长期跟进上游**；
    2. shim 的 C 源与 NDK 交叉编译流程本仓已有且**已在真机验证过**，编译发生在**开发机**，
       设备只收产物 —— 边缘设备不该承担编译；
    3. 末段走同一份 shim ⇒ **同一份 C 源、同一份 llama.cpp**，与 head/middle 的数值口径天然对齐，
       不会在同一链路里混入**第二套构建配置**的 llama.cpp。

    因此设备侧统一走 shim：本类的 `request_token` 与 `TailRunner.request_token` 同语义
    （吃 hidden、吐末位 argmax），可被 `serve_relay_middle_connection` 直接替换。
    """

    def __init__(self, *, shim: str, model: str, n_ctx: int, n_threads: int,
                 mode: str = "nextn", cut_layer: int | None = None,
                 extra_dll_dirs: list[str], n_seq_max: int = 1,
                 n_batch: int = 512) -> None:
        import os  # noqa: PLC0415

        from llama_keep_head import KeepHeadUpstream  # noqa: PLC0415

        dirs = list(extra_dll_dirs)
        dirs.extend(d for d in (os.environ.get("QLH_KEEP_HEAD_DLL_DIRS") or "")
                    .split(os.pathsep) if d)
        self._upstream = KeepHeadUpstream(shim, model, mode=mode, cut_layer=cut_layer,
                                          n_ctx=n_ctx, n_threads=n_threads,
                                          n_seq_max=max(1, int(n_seq_max)),
                                          n_batch=max(512, int(n_batch)),
                                          extra_dll_dirs=dirs)
        self.n_embd = self._upstream.n_embd
        self.n_layer = self._upstream.n_layer
        self._pos = 0
        print(json.dumps({"role": "tail", "n_embd": self.n_embd, "n_layer": self.n_layer,
                          "channel": "keep_head.forward_hidden_to_token",
                          "mode": mode, "cut_layer": cut_layer}), flush=True)

    def reset(self) -> None:
        self._pos = 0
        self._upstream.reset()

    def request_token(self, hidden_bytes: bytes, *, n_tokens: int) -> int:
        import numpy as np  # noqa: PLC0415

        count = int(n_tokens)
        expected = count * int(self.n_embd) * 4
        if os.environ.get("QLH_TAIL_DIAG") == "1":
            # 诊断用（默认关闭）：定位 A 组（上游在 y700）通、B 组（上游在本机）报 rc=-5 的差异。
            # rc=-5 的语义是"embd/n_tokens 到了非法值"，所以要看的是**帧的字节数**而不是数值。
            print(json.dumps({"diag": "tail_frame", "n_tokens": count, "n_embd": self.n_embd,
                              "got_bytes": len(hidden_bytes), "expected_bytes": expected,
                              "n_past": self._pos, "ok": len(hidden_bytes) == expected}),
                  flush=True)
        if len(hidden_bytes) != expected:
            raise ValueError(
                f"tail 段收到 {len(hidden_bytes)} 字节，但 n_tokens={count} × n_embd={self.n_embd}"
                f" × 4 应为 {expected} 字节（帧与段划分不匹配）")
        incoming = np.frombuffer(hidden_bytes, dtype=np.float32).reshape(count, self.n_embd)
        token = self._upstream.forward_hidden_to_token(incoming, n_past=self._pos)
        self._pos += count
        return int(token)

    def request_token_seq(self, hidden_bytes: bytes, *, n_tokens: int,
                          meta: dict[str, object]):
        """★ 2026-10-14（PIPE-COMM-01）：多位置回程（keep-head 通道）。

        `forward_hidden_to_token` 每次只回一个位置 ⇒ 对 γ 个位置**逐位调用**（每次只喂
        `incoming[i:i+1]` 与 `positions=[pos_i]`），最后把 γ 个 token 作为 `list` 一次交回。
        注意：**网络往返仍只有一次** —— 这正是"摊薄 RTT"的机制所在。

        ⚠️ 逐位置 argmax 不保证与逐 token 顺序推进等价（各位置只依赖其之前的上下文）。
        """
        import numpy as np  # noqa: PLC0415

        count = int(n_tokens)
        expected = count * int(self.n_embd) * 4
        if len(hidden_bytes) != expected:
            raise ValueError(
                f"tail hidden bytes={len(hidden_bytes)} expected={expected}")
        incoming = np.frombuffer(hidden_bytes, dtype=np.float32).reshape(count, self.n_embd)
        seq_ids = meta.get("seq_ids")
        positions = meta.get("positions")
        out: list[int] = []
        for index in range(count):
            seq_i = None if seq_ids is None else [seq_ids[index]]
            pos_i = None if positions is None else [positions[index]]
            out.append(int(self._upstream.forward_hidden_to_token(
                incoming[index:index + 1],
                seq_ids=seq_i,
                positions=pos_i,
            )))
        return out

    def close(self) -> None:
        self.reset()
        self._upstream.close()


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="跨机接力的远端段服务（keep-head / 末段）")
    ap.add_argument("--role", choices=("head", "middle", "tail"), default="middle")
    ap.add_argument("--listen", required=True, help="loopback 端点，host:port（跨机用 SSH 隧道）")
    ap.add_argument("--keep-head-shim", default=None, help="--role middle 必需")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-embd", type=int, default=None, help="可选：与本地期望宽度核对")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-seq-max", type=int, default=1,
                    help="★ P3：允许的并行序列上限（跨机多序列要求 ≥ 调用方 batch）")
    ap.add_argument("--n-batch", type=int, default=1024,
                    help="★ P3：batch 容量下限（≥ 调用方 batch × prefill 长度）")
    ap.add_argument("--mode", choices=("nextn", "layer_inp"), default="nextn",
                    help="中间段的 keep-head 通道：nextn（配 head 裁层工件，只跑本段层）/"
                         "layer_inp（配整模工件 + --cut-layer，取第 cut-layer 层输入；"
                         "会跑满全部层，只适合验证或没有裁层工件时）")
    ap.add_argument("--cut-layer", type=int, default=None,
                    help="--mode layer_inp 必需：切点 K（取第 K 层输入 = 前 K 层输出）")
    ap.add_argument("--tail-no-flash-attn", action="store_true",
                    help="★ 仅 --role tail（pip llama_engine）：**关闭** flash attention。"
                         "默认打开（与自建 shim 的 AUTO→enabled 取齐）—— pip 绑定默认写成 "
                         "DISABLED，那是 9B 两段 1/32 的根因（单变量实测：只对齐 batch 线程"
                         "仍 1/32，只打开 FA 即 32/32）。此开关只用于复现差异")
    ap.add_argument("--tail-threads-batch", type=int, default=None,
                    help="★ 仅 --role tail（pip llama_engine）：批处理线程数；缺省取 --threads "
                         "（与 shim 一致。pip 绑定默认 cpu_count()）")
    ap.add_argument("--max-tokens", type=int, default=RELAY_DEFAULT_MAX_TOKENS)
    ap.add_argument("--hidden-quant", default="none",
                    choices=("none", "f16", "int8_block128", "int4_block128"),
                    help="★ A5：**下行**（本服务 → 客户端）的 hidden 压缩档。只有 --role head/middle "
                         "会回 HIDDEN（tail 回 TOKEN，不受影响）；客户端按帧里的档位解压。"
                         "实测判据见 docs/跨框架接力 §5.1③（f16 / int8 32/32 PASS，int4 FAIL）")
    ap.add_argument("--dll-dir", action="append", default=[])
    ap.add_argument("--ready-file", default=None,
                    help="写就绪标记（含实际端点与**构建标识**），供驱动等待与逐段对账")
    ap.add_argument("--digest-artifacts", action="store_true",
                    help="★ ready 文件里对**段工件**也算 sha256（GB 级会明显变慢；默认只记大小/名字）")
    ap.add_argument("--max-connections", type=int, default=0, help="0 = 不限制")
    ap.add_argument("--heartbeat-interval", type=float, default=5.0,
                    help="★ P4.5 健康检查：定期刷新 ready 文件的时间戳（秒；0 = 关闭）。"
                         "外部据此判断服务是否还活着 —— 服务跑在 ssh 会话里时会被网络抖动静默带走，"
                         "没有心跳就分不清'服务已退出'与'模型算错'")
    ap.add_argument("--detach", action="store_true",
                    help="★ R-R9：**脱离发起会话**运行（重起为无终端子进程后父进程退出）—— "
                         "§8.6 的实测教训：ssh 起的段会被网络抖动带走，而隧道端口仍在监听、"
                         "新连接被接受后立刻 reset，与'模型层错误'难以区分。需配合 --log-file")
    ap.add_argument("--log-file", default=None,
                    help="★ R-R9：--detach 时 stdout/stderr 的落点（detach 后没有终端可写）")
    ap.add_argument("--pid-file", default=None,
                    help="★ R-R9：写入服务 PID，便于停止与对账（服务本身不删该文件："
                         "判断活性请用 scripts/relay_health.py --probe）")
    return ap.parse_args(argv)


def _split_endpoint(endpoint: str) -> tuple[str, int]:
    host, _, port = str(endpoint).rpartition(":")
    if not host or not port.isdigit():
        raise SystemExit(f"FAIL: --listen 需要 host:port，实得 {endpoint!r}")
    return host, int(port)


def _runner_build(runner: Any, *, digest_artifacts: bool = False) -> dict[str, Any]:
    """收集**本段**构建标识，写进 ready 文件（供探针逐段对账；§10.2 待办）。

    shim 路径的 runner（`KeepHeadMiddleRunner` / `HeadRunner` / `ShimTailRunner`）都持有 `_upstream`，
    于是能记下 shim 与同目录 `libllama`/`ggml*` 的摘要；pip 的 `TailRunner` 没有 shim，
    就记 `llama_cpp` 版本与它自带的 `lib/llama.dll` 摘要 —— 这正是 §10.7 里 1/32 vs 32/32 的分界。
    """
    from relay_segment_info import collect_local_build  # noqa: PLC0415

    upstream = getattr(runner, "_upstream", None)
    shim = getattr(upstream, "shim_path", None)
    model = getattr(upstream, "model_path", None)
    module = None
    if upstream is None:  # pip 绑定路径：没有 shim
        try:
            import llama_cpp as module  # noqa: PLC0415
        except Exception:  # noqa: BLE001 - 未安装不影响服务本身
            module = None
    return collect_local_build(shim=shim, model=model, llama_cpp_module=module,
                               digest_artifacts=digest_artifacts)


def _detach_self(args: argparse.Namespace) -> int:
    """★ R-R9：把本服务**重起为脱离会话的进程**，父进程随即退出。

    ## 为什么需要（§8.6 的实测教训）

    用 `ssh ... python3 relay_mid_service.py` 起的段，**网络抖动会把 ssh 会话带走、服务随之退出**；
    此后隧道端口**仍在本机监听**，新连接被"接受"后立刻 reset（`ConnectionResetError`）——
    现象与"模型层错误"**难以区分**。生产形态要求服务**不依赖发起会话**。

    ## 做法（零依赖，不引入 `systemd` / `pywin32`）

    - `subprocess.Popen` **重起自己**，并用 `QLH_RELAY_DETACHED=1` 防止无限递归；
    - Windows：`CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS`；POSIX：`start_new_session=True`
      （等效 `setsid`，Termux 无 `setsid` 也可用）；
    - stdout/stderr 重定向到 `--log-file`（detach 后没有终端可写）；
    - 打印子进程 PID（并按需写 `--pid-file`）后**父进程退出** —— 调用方会话断了也不影响服务。

    ## 怎么停

    先用 `scripts/relay_health.py --probe <name>=tcp:<host>:<port>` 确认该段是否真在服务
    （**协议级**握手，能识别"端口在监听但对端已死"），再按 `--pid-file` 或 PID 停：
    Windows `taskkill /PID <pid> /T /F`；POSIX `kill <pid>`。
    """
    if not args.log_file:
        print("FAIL: --detach 需要 --log-file（detach 后没有终端可接管输出）", file=sys.stderr)
        return 2

    env = dict(os.environ, QLH_RELAY_DETACHED="1")
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        if os.name == "nt":
            flags = (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                     | getattr(subprocess, "DETACHED_PROCESS", 0))
            child = subprocess.Popen(command, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                                     env=env, creationflags=flags, close_fds=True)
        else:
            child = subprocess.Popen(command, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                                     env=env, start_new_session=True, close_fds=True)
    print(f"[detached] pid={child.pid} log={log_path}")
    if args.pid_file:
        Path(args.pid_file).write_text(str(child.pid), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    # ★ R-R8 顺手修：服务端日志里含 `⇒` 等符号时，Windows GBK 控制台会让写日志抛
    #   `UnicodeEncodeError`（同 `relay_health.py` 踩过的坑）。统一降级为 replace。
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    args = _parse(argv)

    # ★ R-R9：先处理"脱离会话"，再进入真正的服务流程。
    if args.detach and os.environ.get("QLH_RELAY_DETACHED") != "1":
        return _detach_self(args)

    if args.pid_file:
        try:
            Path(args.pid_file).parent.mkdir(parents=True, exist_ok=True)
            Path(args.pid_file).write_text(str(os.getpid()), encoding="utf-8")
        except OSError:
            pass        # 写不了 pid 文件不该让服务起不来；判断活性请用 relay_health --probe

    host, port = _split_endpoint(args.listen)

    if args.role == "middle":
        if not args.keep_head_shim:
            raise SystemExit("FAIL: --role middle 需要 --keep-head-shim")
        if args.mode == "layer_inp" and args.cut_layer is None:
            raise SystemExit("FAIL: --mode layer_inp 需要 --cut-layer（整模 + 切点 K）")
        runner: Any = KeepHeadMiddleRunner(shim=args.keep_head_shim, model=args.model,
                                           n_ctx=args.n_ctx, n_threads=args.threads,
                                           n_seq_max=args.n_seq_max, n_batch=args.n_batch,
                                           mode=args.mode, cut_layer=args.cut_layer,
                                           extra_dll_dirs=list(args.dll_dir))
        serve = serve_relay_middle_connection
    elif args.role == "head":
        # ★ P4.5：上游段（吃 token 吐 hidden）—— 同样走 shim，设备侧无需 pip llama_cpp。
        # 会话循环复用 middle 的那套（它支持 TOKENS → HIDDEN）。
        if not args.keep_head_shim:
            raise SystemExit("FAIL: --role head 需要 --keep-head-shim")
        runner = HeadRunner(shim=args.keep_head_shim, model=args.model,
                            n_ctx=args.n_ctx, n_threads=args.threads,
                            mode=args.mode, cut_layer=args.cut_layer,
                            extra_dll_dirs=list(args.dll_dir),
                            n_seq_max=args.n_seq_max, n_batch=args.n_batch)
        serve = serve_relay_middle_connection
    else:
        if args.keep_head_shim:
            # ★ P4.5：末段走 shim（设备侧既没有 pip llama_cpp，也不该为它维护打补丁的 fork）
            runner = ShimTailRunner(shim=args.keep_head_shim, model=args.model,
                                    n_ctx=args.n_ctx, n_threads=args.threads,
                                    mode=args.mode, cut_layer=args.cut_layer,
                                    extra_dll_dirs=list(args.dll_dir),
                                    n_seq_max=args.n_seq_max, n_batch=args.n_batch)
        else:
            runner = TailRunner(model=args.model, n_ctx=args.n_ctx, n_threads=args.threads,
                                flash_attn=not bool(args.tail_no_flash_attn),
                                n_threads_batch=args.tail_threads_batch)
        serve = serve_relay_connection

    if args.n_embd is not None and int(args.n_embd) != int(runner.n_embd):
        raise SystemExit(f"FAIL: 模型 n_embd={runner.n_embd} 与期望 {args.n_embd} 不一致")

    listener = open_loopback_listener(host, port)
    utc_now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    ready = {"role": args.role, "host": host, "port": port, "n_embd": runner.n_embd,
             "ready_at": utc_now(),
             # ★ 2026-09-23（A5）：本服务**下行**（→ 客户端）的 hidden 压缩档。
             #   tail 角色回 TOKEN、不回 HIDDEN ⇒ 记 `None`，避免记录里出现误导性档位。
             "hidden_quant_downlink": (args.hidden_quant
                                       if serve is serve_relay_middle_connection else None),
             # ★ 2026-09-23（§10.2）：把**本段构建标识**写进 ready 文件 —— 探针据此在记录里
             #   逐段写出 runner/构建，避免「记录里看不出用的是 pip 绑定还是自建 shim」。
             "build": _runner_build(runner,
                                    digest_artifacts=bool(args.digest_artifacts))}
    if args.ready_file:
        target = Path(args.ready_file)
        target.parent.mkdir(parents=True, exist_ok=True)

        def _write_ready() -> None:
            payload = dict(ready)
            payload["alive_at"] = utc_now()
            payload["pid"] = os.getpid()
            target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        _write_ready()
        if args.heartbeat_interval and float(args.heartbeat_interval) > 0:
            # ★ P4.5 健康检查：定期刷新 ready 文件的时间戳，外部据此判断"服务是否还活着"。
            # 教训：服务跑在 ssh 会话里时，一次网络抖动就会把它**静默带走**，而客户端只会
            # 看到一个与"模型层错误"难以区分的连接错误（实测踩到，见文档 §8.4）。
            # 有了心跳，"服务已退出"与"模型算错"就能被分开。
            def _beat() -> None:
                while True:
                    time.sleep(float(args.heartbeat_interval))
                    try:
                        _write_ready()
                    except Exception:  # noqa: BLE001 - 心跳写失败不应终止服务
                        pass

            threading.Thread(target=_beat, daemon=True).start()
    # ★ 2026-09-24：把**每序列可用 ctx** 讲清楚 —— llama.cpp 会把 `n_ctx` 按 `n_seq_max`
    #   **均分**；单序列长 decode 一旦超过 `n_ctx / n_seq_max` 就会在中途报 llama.cpp 的
    #   "failed to find a memory slot for batch"（`llama_decode rc=1`）。
    #   实测（tail 段）：`n_ctx=2048, n_seq_max=8` ⇒ 单序列仅 256 槽位，decode 到第 256 步即失败；
    #   `4096/8=512` ⇒ ~481 步失败；`8192/8=1024` ⇒ ~1000 步失败（三者完全吻合）。
    #   ⇒ 单序列用法请显式 `--n-seq-max 1`，或把 `--n-ctx` 放大到 `n_seq_max` 倍。
    _per_seq = int(args.n_ctx) // max(1, int(args.n_seq_max))
    if int(args.n_seq_max) > 1:
        print(f"[warn] n_seq_max={args.n_seq_max} ⇒ 每序列 ctx ≈ {_per_seq}（单序列长 decode "
              f"超过它会在中途 rc=1 失败）；单序列请用 --n-seq-max 1 或放大 --n-ctx", flush=True)
    print(f"[ready] role={args.role} listening {host}:{port} n_embd={runner.n_embd} "
          f"ctx_per_seq={_per_seq} (heartbeat={args.heartbeat_interval}s)", flush=True)

    served = 0
    degraded: str | None = None
    try:
        while True:
            if args.max_connections and served >= int(args.max_connections):
                break
            sock, _addr = listener.accept()
            # ★ 2026-09-24：**fail-closed 但不退出** —— runner 一旦不可用（例如长 decode 触发
            #   `llama_decode rc=1` 找不到 KV 槽位），后续会话必须**明确拒绝**，而不是让异常冒到
            #   顶层把服务进程带走（那样现象与"模型算错"难以区分，实测踩到）。保持进程存活，
            #   心跳与日志才继续可见。
            if degraded is not None:
                print(f"[reject] runner unavailable: {degraded}", flush=True)
                try:
                    sock.close()
                except OSError:
                    pass
                served += 1
                continue
            try:
                try:
                    runner.reset()
                except Exception as exc:  # noqa: BLE001 - 引擎已不可用
                    degraded = f"{type(exc).__name__}: {exc}"
                    print(f"[degraded] runner reset failed -> marking unavailable: {degraded}",
                          flush=True)
                    try:
                        sock.close()
                    except OSError:
                        pass
                    served += 1
                    continue
                # ★ A5：只有 head/middle 角色会回 HIDDEN ⇒ 下行压缩档仅对它们有意义。
                if serve is serve_relay_middle_connection:
                    result = serve(sock, runner, n_embd=int(runner.n_embd),
                                   max_tokens=int(args.max_tokens),
                                   hidden_quant=args.hidden_quant)
                else:
                    result = serve(sock, runner, n_embd=int(runner.n_embd),
                                   max_tokens=int(args.max_tokens))
                print(f"[session] frames={result.frames} tokens={result.tokens} "
                      f"closed_cleanly={result.closed_cleanly} error={result.error or '-'}",
                      flush=True)
            finally:
                try:
                    sock.close()
                except OSError:
                    pass
            served += 1
    except KeyboardInterrupt:
        print("[stop] interrupted", flush=True)
    finally:
        listener.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
