"""Stage worker: one cut artifact, one TCP endpoint.

Runs as a separate process so every stage owns its llama context, KV cache and
(for GPU-offloaded stages) VRAM. Protocol is the six-message set in protocol.py:
hello / forward / result / error / reset / bye.

Three roles, three payload shapes:

  gpu   tokens  (JSON list)  -> hidden (raw bytes, wire dtype)
  cpu   hidden  (wire dtype) -> hidden (raw bytes, wire dtype)
  tail  hidden  (wire dtype) -> token  (JSON int)

The wire dtype is float32 by default and float16 when SA_WIRE_DTYPE says so; see
wire.py for why it is a switch. Inside the stage everything stays float32.
"""
from __future__ import annotations

import argparse
import os
import select
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import protocol  # noqa: E402
import wire  # noqa: E402
from engine import ROLES, StageEngine  # noqa: E402


def _trace(message: str) -> None:
    """Per-request breadcrumb on stderr, off unless SA_STAGE_TRACE is set."""
    if os.environ.get("SA_STAGE_TRACE"):
        print(f"[trace {os.getpid()}] {message}", file=sys.stderr, flush=True)


#: 心跳文件多久 touch 一次，也是 select 等一次数据的最长空转时间。
HEARTBEAT_S = 10.0
#: 一个连接空转多久就放弃它（秒）。单线程 `accept → _session` 的代价：一条客户端崩溃
#: 但 TCP 未关的连接会把整段堵死，而它自己永远不会超时。60 s 的取法——正常推理每步都有
#: 请求（150 ms 级），空闲到这个量级的只可能是死连接；环境变量可覆盖，给交互式客户端留
#: 余量（用户思考间隔可能比这长）。
SESSION_IDLE_S = float(os.environ.get("SA_SESSION_IDLE_S", "60"))


class StageWorker:
    def __init__(self, engine: StageEngine, role: str,                 identity: dict | None = None) -> None:
        self._engine = engine
        self._role = role
        # What this process actually loaded, announced before the first request is
        # answered. Until this existed the chain's whole readiness test was "the port
        # accepts", which is equally true of a worker holding the wrong cut.
        self._identity = dict(identity or {})
        #: supervisor 通过 `SA_HEARTBEAT` 指定这个文件，每转一圈 touch 一次。
        #: 为什么用文件 mtime 而不是网络探针：worker 是单线程 `accept → _session`，
        #: 正在服务一个长连接时，新的探测连接只能躺在 backlog 里等 accept，探针永远
        #: 拿不到响应（实测见 docs/ISSUE-tail-worker-hang.md）。心跳是"我主动报告"，
        #: 不受这个限制。
        beat = os.environ.get("SA_HEARTBEAT")
        self._heartbeat = Path(beat) if beat else None

    def _beat(self) -> None:
        """报告「我还在走」。卡在 engine 调用里时走不到这里，心跳就停了。"""
        if self._heartbeat is not None:
            self._heartbeat.touch(exist_ok=True)

    def serve(self, host: str, port: int) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(4)
        print(f"[{self._role}] listening on {host}:{port} (serving repeatedly)", flush=True)

        # Serve connections in a loop rather than handling exactly one.
        #
        # The old behaviour made every chain run require restarting the remote worker --
        # on ROG that meant an schtasks call plus 30 s of model loading before each
        # measurement, and a stale worker silently refusing the next connection. The model
        # stays loaded across sessions; only the KV cache is reset (`reset` op), which is
        # what a fresh chain wants anyway.
        # 空闲时也要报心跳。supervisor 判的是「这一段还在走」，不是「有请求正在处理」；
        # 没有这个超时，accept 会一直阻塞，一个空闲的段就会被自己的心跳机制杀掉。
        server.settimeout(HEARTBEAT_S)
        while True:
            try:
                conn, addr = server.accept()
            except socket.timeout:
                self._beat()
                continue
            except OSError:
                _trace("accept raised OSError, leaving serve loop")
                break
            self._beat()
            # accept 之后先查一次：子进程可能在上一个 session 之后的空闲期崩掉，这时候
            # 服务这个客户端只会让它第一帧拿到 "keep-head worker is not running"。
            self._exit_if_keep_head_lost(server, conn)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[{self._role}] client {addr}", flush=True)
            try:
                self._session(conn)
            except Exception as exc:  # noqa: BLE001 - one bad session must not kill the worker
                # 这条 except 兜的是"一个坏请求"，不是"模型没了"：后者由下面的健康检查
                # 主动退出处理，那种情况本来也不该继续服务。
                print(f"[{self._role}] session error: {type(exc).__name__}: {exc}", flush=True)
                _trace("session error handler: engine.reset()")
                self._engine.reset()
                _trace("session error handler: reset done")
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
                print(f"[{self._role}] client {addr} closed, ready for next", flush=True)
            # session 结束后再查一次：崩在 session 中间的那一次已经把错误回给了客户端，
            # 这里退出，supervisor 下次拉起的是一个模型重新加载好的段。
            self._exit_if_keep_head_lost(server)
        server.close()

    def _exit_if_keep_head_lost(self, server: socket.socket,
                                conn: socket.socket | None = None) -> None:
        """keep-head 子进程没了就关掉监听、以非零码退出，而不是继续服务。

        为什么退出：这一段真正持有模型权重的是子进程，本进程只是转发。子进程崩了以后本进程
        照样 accept、照样回 "keep-head worker is not running"，而 stage_supervisor 只看本进程
        的 proc.poll()，于是这一段永久不可用、supervisor 日志一片安静。非零退出是唯一能让
        重启逻辑看见这个事实的信号；监听先关掉，是为了让接手的段能立刻绑回同一个端口。
        """
        if self._engine.healthy:
            return
        # 子进程自己的崩溃输出（llama_decode 失败、断言）就在上面几行的 stderr 里。
        print(f"[{self._role}] keep-head child process is gone; exiting non-zero so "
              f"stage_supervisor restarts this stage", flush=True)
        for sock in (conn, server):
            if sock is None:
                continue
            try:
                sock.close()
            except OSError:
                pass
        raise SystemExit(1)

    def _session(self, conn: socket.socket) -> None:
        # HELLO goes first, unsolicited: the driver reads exactly one frame after connecting
        # and validates it before it trusts anything else on this socket. Both ends must be
        # from the same checkout -- the previous protocol had no handshake at all.
        protocol.send(conn, protocol.HELLO, self._identity)
        idle = 0.0
        while True:
            # 先等可读、再读。中间那一跳是心跳：阻塞在 read 上时没法报告自己还活着，
            # 而「活着但卡住」正是要区分的两种状态。
            ready, _, _ = select.select([conn], [], [], HEARTBEAT_S)
            self._beat()
            if not ready:
                # 空闲累计。worker 是单线程 `accept → _session`，一条客户端崩溃但 TCP
                # 未关的连接会把整段堵在这里——期间新连接只能在 listen(4) 的 backlog 里
                # 排队，拿不到 HELLO（实测见 docs/ISSUE-tail-worker-hang.md §1）。
                # 断开它需要 `SESSION_IDLE_S` 秒内一个请求都没收到；正常推理每步都有
                # 请求，间隔远小于这个值，所以不会误伤。
                idle += HEARTBEAT_S
                if idle >= SESSION_IDLE_S:
                    _trace(f"session idle for {idle:.0f}s, dropping it")
                    return
                continue
            idle = 0.0
            try:
                msg = protocol.read(conn)
            except protocol.ProtocolError:
                return

            if msg.kind == protocol.BYE:
                return
            if msg.kind == protocol.RESET:
                self._engine.reset()
                protocol.send(conn, protocol.RESULT, {"ok": True})
                continue
            if msg.kind != protocol.FORWARD:
                protocol.send(conn, protocol.ERROR, {"code": "unexpected_message", "name": msg.name})
                continue

            n_tokens = int(msg.meta.get("n_tokens", 0))
            n_past = int(msg.meta.get("n_past", 0))
            _trace(f"forward op={msg.meta.get('op') or '-'} n_tokens={n_tokens} n_past={n_past}")
            # Text helpers ride on FORWARD's meta instead of new message types, so the
            # frame format stays at six messages and `op` is just another key.
            op = msg.meta.get("op", "")
            if op:
                try:
                    if op == "tokenize":
                        tokens = self._engine.tokenize(str(msg.meta.get("text", "")))
                        # n_embd rides along so a client can learn the width without
                        # running a forward first (single-stage topologies never do).
                        protocol.send(conn, protocol.RESULT,
                                      {"tokens": tokens, "n_embd": self._engine.n_embd})
                    elif op == "piece":
                        piece = self._engine.piece(int(msg.meta["token"]))
                        # latin-1 keeps arbitrary bytes intact through the JSON hop.
                        protocol.send(conn, protocol.RESULT,
                                      {"piece": piece.decode("latin-1")})
                    elif op == "last_hidden":
                        h = self._engine.last_hidden(int(msg.meta.get("row", -1)))
                        if h is None:
                            protocol.send(conn, protocol.ERROR,
                                          {"code": "no_nextn_head"})
                        else:
                            # `dtype` is only resolved further down (the forward path needs it
                            # after decoding the blob); this op sends bytes without one, so it
                            # has to resolve its own or the NameError gets swallowed by the
                            # handler's except and the caller silently sees "no hidden".
                            d = wire.resolve(msg.meta.get("dtype"))
                            protocol.send(conn, protocol.RESULT,
                                          {"n_embd": int(h.shape[0]), "n_tokens": 1,
                                           "dtype": d},
                                          wire.encode(h.reshape(1, -1), d))
                    elif op == "set_skip_logits":
                        self._engine.set_skip_hidden_logits(bool(msg.meta.get("skip", True)))
                        protocol.send(conn, protocol.RESULT, {"ok": True})
                    elif op == "logits":
                        values = self._engine.logits(int(msg.meta.get("row", -1)))
                        protocol.send(conn, protocol.RESULT,
                                      {"n_vocab": int(values.size), "n_tokens": 1},
                                      wire.encode(values.reshape(1, -1), None))
                    elif op == "forward_hidden":
                        # DFLASH-01: hidden -> hidden on ANY stage, including tail.
                        # tail 的主路径只出 token（mtp_draft / forward_tokens_out /
                        # forward_token），但 DFlash 需要 tail 真的前向一次来产生 33/38
                        # 两路 layer_inp。这条 op 显式要求 hidden-in/hidden-out，绕开
                        # 主分支的角色假设。
                        dtype = wire.resolve(msg.meta.get("dtype"))
                        n_tokens = int(msg.meta["n_tokens"])
                        n_embd = self._engine.n_embd
                        hidden = wire.decode(msg.blob, dtype, (n_tokens, n_embd))
                        try:
                            out = self._engine.forward_hidden(hidden, n_past=n_past)
                            if out is None:
                                # 前向发生了，只是这一段不提供 dense hidden（tail 段
                                # 必须开 logits，而 embeddings = !needs_logits）。
                                raise RuntimeError("stage offers no dense hidden")
                        except Exception as exc:  # noqa: BLE001
                            # tail 段为了出 token 必须开 logits，而 logits 与
                            # cparams.embeddings 互斥 -> dense hidden 取不到。这不是错误：
                            # 调用方要的往往是这次前向顺手产生的 layer_inp（DFlash 的
                            # 33/38 两路）。前向已经发生，照实回"没有 hidden"。
                            _trace(f"forward_hidden produced no dense hidden: {exc}")
                            protocol.send(conn, protocol.RESULT,
                                          {"n_past": n_past, "no_hidden": True})
                            continue
                        protocol.send(conn, protocol.RESULT,
                                      {"n_tokens": int(out.shape[0]),
                                       "n_embd": int(out.shape[1]),
                                       "n_past": n_past, "dtype": dtype},
                                      wire.encode(out, dtype))
                    elif op == "draft_open":
                        rc = self._engine.draft_open(str(msg.meta["path"]),
                                                     int(msg.meta.get("n_ctx", 256)),
                                                     int(msg.meta.get("n_batch", 32)),
                                                     int(msg.meta.get("n_gpu_layers", 0)))
                        protocol.send(conn, protocol.RESULT, {"draft_rc": int(rc)})
                    elif op == "draft_n_embd_enc":
                        protocol.send(conn, protocol.RESULT,
                                      {"n_embd_enc": int(self._engine.draft_n_embd_enc())})
                    elif op == "draft_inject":
                        # 与 forward 分支同样的内联解码（本文件没有 _decode_hidden）
                        n_rows = int(msg.meta["n_tokens"])
                        n_embd = int(msg.meta["shape"][1])
                        d = wire.resolve(msg.meta.get("dtype"))
                        arr = wire.decode(msg.blob, d, (n_rows, n_embd))
                        rc = self._engine.draft_inject(arr)
                        protocol.send(conn, protocol.RESULT, {"draft_rc": int(rc)})
                    elif op == "draft_block":
                        ids = self._engine.draft_block(int(msg.meta["anchor"]),
                                                       int(msg.meta.get("n_block", 16)))
                        protocol.send(conn, protocol.RESULT, {"drafts": [int(v) for v in ids]})
                    elif op == "draft_reset":
                        rc = self._engine.draft_reset()
                        protocol.send(conn, protocol.RESULT, {"draft_rc": int(rc)})
                    elif op == "draft_rollback":
                        rc = self._engine.draft_rollback(int(msg.meta["n_keep"]))
                        protocol.send(conn, protocol.RESULT, {"draft_rc": int(rc)})
                    elif op == "draft_close":
                        self._engine.draft_close()
                        protocol.send(conn, protocol.RESULT, {})
                    elif op == "add_layer_inp":
                        # DFLASH-01: open one more intermediate-hidden slot. The shim keeps
                        # the list; this op is idempotent so a reconnecting driver can re-add.
                        self._engine.add_layer_inp(int(msg.meta["lid"]))
                        protocol.send(conn, protocol.RESULT, {})
                    elif op == "layer_inp":
                        # DFLASH-01: all requested slots from the most recent forward, in one
                        # reply. Per-slot round trips would be 6 socket hops per decode step.
                        arr = self._engine.layer_inp([int(v) for v in msg.meta.get("lids", [])])
                        if arr is None:
                            protocol.send(conn, protocol.ERROR,
                                          {"code": "no_layer_inp",
                                           "detail": "shim has no data for those slots"})
                        else:
                            d = wire.resolve(msg.meta.get("dtype"))
                            protocol.send(conn, protocol.RESULT,
                                          {"n_layers": int(arr.shape[0]),
                                           "n_tokens": int(arr.shape[1]),
                                           "n_embd": int(arr.shape[2]),
                                           "dtype": d},
                                          wire.encode(arr.reshape(-1, arr.shape[-1]), d))
                    elif op == "seq_rm":
                        self._engine.seq_rm(n_past)
                        protocol.send(conn, protocol.RESULT, {})
                    elif op == "mtp_reset":
                        self._engine.mtp_reset()
                        protocol.send(conn, protocol.RESULT, {})
                    elif op == "is_eog":
                        protocol.send(conn, protocol.RESULT,
                                      {"eog": self._engine.is_eog(int(msg.meta["token"]))})
                    elif op == "set_sampling":
                        # Sampling lives on the tail stage only (it is the one that
                        # turns logits into an id). Every field is optional; the
                        # reply is the shim's own view of what took effect, because
                        # the clamping rules live there and not here.
                        effective = self._engine.set_sampling(
                            temperature=float(msg.meta.get("temperature", 0.0)),
                            top_p=float(msg.meta.get("top_p", 1.0)),
                            top_k=int(msg.meta.get("top_k", 0)),
                            repeat_penalty=float(msg.meta.get("repeat_penalty", 1.0)),
                            repeat_last_n=int(msg.meta.get("repeat_last_n", 0)),
                            frequency_penalty=float(msg.meta.get("frequency_penalty", 0.0)),
                            presence_penalty=float(msg.meta.get("presence_penalty", 0.0)),
                            seed=int(msg.meta.get("seed", 0xFFFFFFFF)),
                        )
                        protocol.send(conn, protocol.RESULT, {"sampling": effective})
                    elif op == "get_sampling":
                        protocol.send(conn, protocol.RESULT,
                                      {"sampling": self._engine.sampling()})
                    elif op == "clear_sampling":
                        protocol.send(conn, protocol.RESULT,
                                      {"sampling": self._engine.clear_sampling()})
                    else:
                        protocol.send(conn, protocol.ERROR,
                                      {"code": "unknown_op", "op": op})
                except Exception as exc:  # noqa: BLE001
                    protocol.send(conn, protocol.ERROR,
                                  {"code": "op_failed", "op": op, "detail": str(exc)[:300]})
                continue

            try:
                # The client declares the transport dtype per request; a frame without
                # one is float32, which is what an older client sends. The answer is
                # sent back in the same dtype, so one side can switch without the other.
                dtype = wire.resolve(msg.meta.get("dtype"))
                if self._role == "gpu":
                    tokens = list(msg.meta["tokens"])
                    hidden = self._engine.forward_tokens(tokens, n_past=n_past)
                    protocol.send(
                        conn, protocol.RESULT,
                        {"n_tokens": hidden.shape[0], "n_embd": hidden.shape[1],
                         "n_past": n_past, "dtype": dtype},
                        wire.encode(hidden, dtype),
                    )
                else:
                    n_embd = self._engine.n_embd
                    expected = wire.payload_size((n_tokens, n_embd), dtype)
                    if len(msg.blob) != expected:
                        protocol.send(conn, protocol.ERROR, {
                            "code": "hidden_size_mismatch",
                            "expected": expected,
                            "got": len(msg.blob),
                            "dtype": dtype,
                        })
                        continue
                    # A view, not a copy: the engine converts to float32 exactly once,
                    # at the point where the shim's ABI requires it.
                    hidden = wire.decode(msg.blob, dtype, (n_tokens, n_embd))
                    if self._role == "tail":
                        if "token" in msg.meta:
                            if msg.meta.get("want_logits"):
                                # SPEC-SAMPLE-01: same draft forward, distribution attached.
                                # The blob is always float32: the wire dtype is for hidden
                                # state, and quantizing logits to fp16 would move p and q
                                # differently and break the acceptance ratio.
                                token, values = self._engine.mtp_draft_logits(
                                    hidden, int(msg.meta["token"]), n_past=n_past)
                                protocol.send(conn, protocol.RESULT,
                                              {"token": token, "n_vocab": int(values.size),
                                               "n_past": n_past},
                                              wire.encode(values.reshape(1, -1), None))
                            else:
                                token = self._engine.mtp_draft(hidden, int(msg.meta["token"]),
                                                               n_past=n_past)
                                protocol.send(conn, protocol.RESULT,
                                              {"token": token, "n_past": n_past})
                        elif n_tokens > 1:
                            # Speculative verification: one argmax per submitted position.
                            toks = self._engine.forward_tokens_out(hidden, n_past=n_past)
                            protocol.send(conn, protocol.RESULT,
                                          {"tokens": toks, "n_past": n_past})
                        else:
                            token = self._engine.forward_token(hidden, n_past=n_past)
                            protocol.send(conn, protocol.RESULT,
                                          {"token": token, "n_past": n_past})
                    else:
                        out = self._engine.forward_hidden(hidden, n_past=n_past)
                        protocol.send(
                            conn, protocol.RESULT,
                            {"n_tokens": out.shape[0], "n_embd": out.shape[1],
                             "n_past": n_past, "dtype": dtype},
                            wire.encode(out, dtype),
                        )
            except Exception as exc:  # noqa: BLE001 - the peer must see the failure
                _trace(f"forward raised: {type(exc).__name__}: {exc}")
                protocol.send(conn, protocol.ERROR, {"code": "forward_failed", "detail": str(exc)[:400]})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=ROLES)
    ap.add_argument("--model", required=True)
    ap.add_argument("--shim", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    ap.add_argument("--n-cpu-moe", type=int, default=0)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--cpus", default="", help="comma-separated logical CPU ids to pin to")
    args = ap.parse_args()

    if args.cpus:
        try:
            import psutil

            allowed = [int(x) for x in args.cpus.split(",") if x.strip()]
            psutil.Process().cpu_affinity(allowed)
            print(f"[{args.role}] pinned to CPUs {allowed}", flush=True)
        except Exception as exc:  # noqa: BLE001 - pinning is best effort
            print(f"[{args.role}] cpu pin failed: {exc}", flush=True)

    engine = StageEngine(
        shim_path=Path(args.shim),
        model_path=Path(args.model),
        role=args.role,
        ctx=args.ctx,
        n_threads=args.threads,
        n_gpu_layers=args.n_gpu_layers,
        n_cpu_moe=args.n_cpu_moe,
    )
    try:
        # --layers is what the topology declares; the artifact is what actually runs.
        # A mismatch means the topology points at a cut for a different range. It used to
        # be a warning, so the run proceeded and returned plausible-but-wrong hidden
        # states -- the failure this whole file's handshake exists to prevent. Now fatal.
        if engine.n_layer != args.layers:
            raise SystemExit(
                f"[{args.role}] the topology declares {args.layers} layers but "
                f"{Path(args.model).name} holds {engine.n_layer}.\n"
                f"  Fix the topology's `layers` (or its layer_range). Loading anyway would "
                f"run, answer every request, and quietly compute the wrong thing.")
        model_path = Path(args.model)
        # Model identity by name and size, not a hash: a 14.79 GiB digest costs seconds on
        # every start and the layer count above already catches the case that matters.
        identity = {
            "role": args.role,
            "n_layer": int(engine.n_layer),
            "n_embd": int(engine.n_embd),
            "layer_range": os.environ.get("LLAMA_LAYER_RANGE", ""),
            "dtype": wire.current_dtype(),
            "model": model_path.name,
            "model_bytes": model_path.stat().st_size if model_path.is_file() else 0,
            "layers_declared": int(args.layers),
        }
        # A relay stage returns hidden; the output projection it computes on the way is
        # discarded. The shim has always had the switch, but nothing on this side called it,
        # so every relay stage was paying for 248,320 rows per decode step. The tail must
        # keep logits -- that is how it picks a token.
        # SA_SKIP_LOGITS=0 keeps the old (wasteful) behaviour for A/B measurement.
        if args.role != "tail" and os.environ.get("SA_SKIP_LOGITS", "1") != "0":
            engine.set_skip_hidden_logits(True)
            print(f"[{args.role}] skipping logits (hidden-only stage)", flush=True)
        print(f"[{args.role}] {engine.n_layer} layers, n_embd={engine.n_embd}",
              flush=True)
        StageWorker(engine, args.role, identity).serve(args.host, args.port)
    finally:
        _trace("main finally: engine.close()")
        engine.close()
        _trace("main finally: engine closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
