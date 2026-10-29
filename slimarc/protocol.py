"""Six message types over TCP, length-prefixed frames.

Frame: [4-byte big-endian payload length][1-byte type][payload]
Payload is raw bytes: hidden states travel as raw float32, never base64/JSON.
"""
from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass

HEADER = struct.Struct(">IB")
MAX_PAYLOAD = 64 * 1024 * 1024

HELLO = 1
FORWARD = 2
RESULT = 3
ERROR = 4
RESET = 5
BYE = 6

_NAMES = {
    HELLO: "hello",
    FORWARD: "forward",
    RESULT: "result",
    ERROR: "error",
    RESET: "reset",
    BYE: "bye",
}


class ProtocolError(RuntimeError):
    pass


@dataclass
class Message:
    kind: int
    meta: dict
    blob: bytes

    @property
    def name(self) -> str:
        return _NAMES.get(self.kind, f"unknown({self.kind})")


def encode(kind: int, meta: dict | None = None, blob: bytes = b"") -> bytes:
    head = json.dumps(meta or {}, separators=(",", ":")).encode()
    payload = struct.pack(">I", len(head)) + head + blob
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError(f"payload too large: {len(payload)}")
    return HEADER.pack(len(payload), kind) + payload


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    got = 0
    while got < n:
        chunk = sock.recv(n - got)
        if not chunk:
            raise ProtocolError("peer closed")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def read(sock: socket.socket) -> Message:
    raw_head = _recv_exact(sock, HEADER.size)
    length, kind = HEADER.unpack(raw_head)
    if length > MAX_PAYLOAD:
        raise ProtocolError(f"declared payload too large: {length}")
    payload = _recv_exact(sock, length) if length else b""
    if len(payload) < 4:
        raise ProtocolError("payload missing meta length")
    (meta_len,) = struct.unpack(">I", payload[:4])
    if 4 + meta_len > len(payload):
        raise ProtocolError("meta length exceeds payload")
    meta = json.loads(payload[4:4 + meta_len] or b"{}")
    return Message(kind=kind, meta=meta, blob=payload[4 + meta_len:])


def send(sock: socket.socket, kind: int, meta: dict | None = None, blob: bytes = b"") -> None:
    sock.sendall(encode(kind, meta, blob))
