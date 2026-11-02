"""Six message types over TCP, length-prefixed frames.

Frame: [4-byte big-endian payload length][1-byte type][payload]
Payload is [4-byte meta length][meta JSON][blob]. Hidden states travel as raw bytes
(float32 by default, float16 or int16 under SA_WIRE_DTYPE -- see wire.py), never
base64/JSON.

The two ends are deliberately asymmetric in how they move those bytes:

Sending builds the whole frame in one buffer and calls sendall once. Two sendalls would
let the 5-byte frame header go out as its own packet, and on the 1280-byte-MTU Tailscale
link a packet costs a whole RTT (~5 ms) -- which is far more than the one memcpy that
assembling the frame costs.

Receiving reads straight into a preallocated bytearray with recv_into and hands back a
memoryview over it, so the payload is never copied between the socket and the caller's
np.frombuffer. The old path copied twice (join the chunks, then slice off the meta) and
that is exactly the work this removes.
"""
from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass

HEADER = struct.Struct(">IB")        # payload length, message kind
META_LEN = struct.Struct(">I")       # meta JSON length, first 4 bytes of the payload
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
    # memoryview for a payload read off a socket, bytes for anything constructed by
    # hand. Both are buffer-protocol objects, which is all the callers need.
    blob: object

    @property
    def name(self) -> str:
        return _NAMES.get(self.kind, f"unknown({self.kind})")


def encode(kind: int, meta: dict | None = None, blob=b"") -> bytes:
    head = json.dumps(meta or {}, separators=(",", ":")).encode()
    payload_len = META_LEN.size + len(head) + len(blob)
    if payload_len > MAX_PAYLOAD:
        raise ProtocolError(f"payload too large: {payload_len}")
    # One allocation, one pass: b"".join walks the parts exactly once. The previous
    # version concatenated twice (pack+head, then +blob, then +header), which copied
    # an 8 KB hidden three times per step instead of once.
    return b"".join((HEADER.pack(payload_len, kind), META_LEN.pack(len(head)), head, blob))


def _recv_into(sock: socket.socket, view: memoryview) -> None:
    """Fill `view` completely, raising if the peer closes first."""
    got = 0
    total = len(view)
    while got < total:
        n = sock.recv_into(view[got:], total - got)
        if not n:
            raise ProtocolError("peer closed")
        got += n


def read(sock: socket.socket) -> Message:
    header = bytearray(HEADER.size)
    _recv_into(sock, memoryview(header))
    length, kind = HEADER.unpack(header)
    if length > MAX_PAYLOAD:
        raise ProtocolError(f"declared payload too large: {length}")
    if length < META_LEN.size:
        raise ProtocolError("payload missing meta length")
    # The payload goes from the kernel straight into this buffer; blob is a view, so
    # _recv_into plus the slice below are the only two touches and neither copies.
    payload = bytearray(length)
    _recv_into(sock, memoryview(payload))
    (meta_len,) = META_LEN.unpack_from(payload, 0)
    end = META_LEN.size + meta_len
    if end > length:
        raise ProtocolError("meta length exceeds payload")
    meta = json.loads(bytes(payload[META_LEN.size:end]) or b"{}")
    return Message(kind=kind, meta=meta, blob=memoryview(payload)[end:])


def send(sock: socket.socket, kind: int, meta: dict | None = None, blob=b"") -> None:
    sock.sendall(encode(kind, meta, blob))
