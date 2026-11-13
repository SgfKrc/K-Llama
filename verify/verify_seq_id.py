#!/usr/bin/env python3
"""An out-of-range seq_id must answer ST_BAD_REQ and leave the connection usable.

The stage used to pass req.seq_id straight from the wire into llamea_memory_seq_rm /
llama_state_seq_*, so seq_id = -1 or 999999 reached the memory layer. This script speaks the
wire protocol from src/llama-split-proto.h directly (OP_HELLO, then a few OP_SEQ_RM frames) and
checks:

  1. seq_id = -1      -> ST_BAD_REQ, connection still serving
  2. seq_id = 999999  -> ST_BAD_REQ, connection still serving
  3. seq_id = 0 (legal, pos0=0 pos1=-1) -> ST_OK, twice, on the same connection

It starts its own stage, so it needs a model file and a free 93xx port.

usage: python verify/verify_seq_id.py [--model PATH] [--port 9337]
"""

import argparse
import os
import socket
import struct
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _qlh_env import BIN, ROOT, TMP, require   # noqa: E402
MODEL_DEFAULT = os.environ.get(
    "QLH_SPLIT_MODEL", "H:/qlh_models/qwen3-5-2b-gguf/qwen35-2b-Q4_K_M.gguf")

MAGIC = 0x514C4853
# Mirror of qlh_split::PROTO_VER (src/llama-split-proto.h). This script speaks the wire protocol
# itself, so a bump has to be mirrored here -- that is the point: the stage refuses a frame whose
# version it does not speak, and this is what proves the check is on the frame and not on the op.
# v10 added the per-row seq_id records (the `rows` section at the end of an OP_DECODE frame), the
# rsp_header::n_logits field (32 -> 36 bytes) and four chain-identity numbers in stage_id (28 -> 40
# bytes). This script only speaks OP_HELLO / OP_SEQ_RM / OP_CLEAR / OP_BYE, so none of the *decode*
# layout changes reach it -- but the two struct sizes it does parse moved, which is exactly what the
# version bump is here to catch.
PROTO_VER = 10

OP_HELLO = 7
OP_SEQ_RM = 2
OP_CLEAR = 3
OP_BYE = 6

ST_OK = 0
ST_BAD_REQ = 2

STAGE_ID = 40
# req_header: magic, version, op, flags, pad, n_tokens, n_embd, n_pos, seq_id, pos0, pos1, n_bytes
REQ = struct.Struct("<IIBBHIIIiiiI")
# rsp_header: magic, version, status, flags, (n_taps + pad, read as one H), (v10: n_logits),
# n_tokens, n_embd, n_vocab, pos_max, n_bytes
RSP = struct.Struct("<IIBBHIIIIiI")

assert REQ.size == 40, REQ.size
assert RSP.size == 36, RSP.size


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("connection closed with %d/%d bytes read" % (len(buf), n))
        buf += chunk
    return buf


class Stage:
    def __init__(self, model, port):
        self.port = port
        self.log = os.path.join(TMP, "seq_id_stage.log")
        self.logf = open(self.log, "wb")
        self.proc = subprocess.Popen(
            [os.path.join(BIN, "split-stage.exe"), "-m", model,
             "--layer-range", "20:24", "--role", "tail", "--rs-seq", "8",
             "--port", str(port), "-c", "512", "-b", "256"],
            stdout=self.logf, stderr=subprocess.STDOUT)

    def wait_ready(self, timeout=240):
        # [v11] 等 ready: 而不是 listening。v10 把 split-stage 的 bind 提到了模型加载之前，端口一绑
        # 上就打 "listening on ... (model not loaded yet)"，所以那个字符串只说明端口是它的。ready:
        # 是模型、ctx 与 offload 计划都落定、即将进 accept 循环时打的那一行。
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("stage exited early, see " + self.log)
            try:
                with open(self.log, "rb") as f:
                    if b"ready:" in f.read():
                        return
            except OSError:
                pass
            time.sleep(0.5)
        raise RuntimeError("stage did not report ready:, see " + self.log)

    def stop(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self.logf.close()

    def log_text(self):
        with open(self.log, "r", errors="replace") as f:
            return f.read()


def connect(port, timeout=20.0):
    sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    sock.settimeout(timeout)
    return sock


def send_req(sock, op, seq_id=0, pos0=0, pos1=-1, n_tokens=0, n_embd=0, n_pos=0,
             n_bytes=0, flags=0, version=PROTO_VER):
    sock.sendall(REQ.pack(MAGIC, version, op, flags, 0, n_tokens, n_embd, n_pos,
                          seq_id, pos0, pos1, n_bytes))


def read_rsp(sock):
    raw = recv_exact(sock, RSP.size)
    magic, version, status, flags, _pad, _n_logits, n_tokens, n_embd, n_vocab, pos_max, n_bytes = \
        RSP.unpack(raw)
    assert magic == MAGIC, "bad reply magic 0x%08x" % magic
    assert version == PROTO_VER, "bad reply version %d" % version
    return status, n_bytes, n_vocab, pos_max


def handshake(sock):
    # OP_HELLO expects one stage_id; the stage only logs it, so unpinned values are fine.
    send_req(sock, OP_HELLO, n_bytes=STAGE_ID)
    # stage_id (v10, 40 B): layer_lo, layer_hi, n_layer_src, n_embd_inp, n_embd_out, n_vocab,
    # role, has_nextn, n_seq_max, n_ctx_seq, n_rs_seq, n_verify_rows. The stage only logs the
    # expectation, so unpinned values are fine.
    sock.sendall(struct.pack("<iiiIIIBBHIII", -1, -1, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    status, n_bytes, _n_vocab, _pm = read_rsp(sock)
    assert status == ST_OK, "hello answered status %d" % status
    body = recv_exact(sock, n_bytes)
    assert len(body) == STAGE_ID, "hello identity is %d bytes" % len(body)


def main():
    require("split-stage.exe")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--port", type=int, default=9337)
    args = ap.parse_args()

    failures = []
    n_pass = [0]

    def check(cond, what):
        print(("  [PASS] " if cond else "  [FAIL] ") + what)
        if cond:
            n_pass[0] += 1
        else:
            failures.append(what)

    stage = Stage(args.model, args.port)
    try:
        stage.wait_ready()
        sock = connect(args.port)
        try:
            handshake(sock)

            for bad in (-1, 999999):
                send_req(sock, OP_SEQ_RM, seq_id=bad, pos0=0, pos1=-1)
                status, _nb, _nv, _pm = read_rsp(sock)
                check(status == ST_BAD_REQ,
                      "seq_id=%d -> ST_BAD_REQ (got %d)" % (bad, status))

            # The connection has to still be parseable: a legal frame now must answer OK.
            for i in (1, 2):
                send_req(sock, OP_SEQ_RM, seq_id=0, pos0=0, pos1=-1)
                status, _nb, _nv, _pm = read_rsp(sock)
                check(status == ST_OK,
                      "legal seq_rm #%d on the same connection -> ST_OK (got %d)" % (i, status))

            send_req(sock, OP_CLEAR)
            status, _nb, _nv, _pm = read_rsp(sock)
            check(status == ST_OK, "OP_CLEAR on the same connection -> ST_OK (got %d)" % status)

            send_req(sock, OP_BYE)
        finally:
            sock.close()

        # [QLH-SPLIT] v9: a peer from the *previous* protocol version has to be refused on its
        # first frame, not misparsed. The rsp_header change (a padding byte became n_taps) is
        # layout-compatible, so nothing but PROTO_VER would catch a v8 client -- which is exactly
        # why the version is checked before any op is dispatched. Only the header is sent: the
        # stage answers and closes without reading the body it would have carried.
        old = connect(args.port)
        try:
            send_req(old, OP_HELLO, n_bytes=STAGE_ID, version=PROTO_VER - 1)
            status, _nb, _nv, _pm = read_rsp(old)
            check(status == ST_BAD_REQ,
                  "a v%d HELLO -> ST_BAD_REQ (got %d)" % (PROTO_VER - 1, status))
        finally:
            old.close()
    finally:
        stage.stop()

    text = stage.log_text()
    check("out of range" in text, "stage log states the reason (out of range)")
    check('seq_id=-1' in text and 'seq_id=999999' in text,
          "stage log names both rejected seq_ids")
    check("protocol version mismatch" in text,
          "stage log calls the v%d frame a version mismatch, not a bad request" % (PROTO_VER - 1))

    print()
    print("PASS=%d FAIL=%d" % (n_pass[0], len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
