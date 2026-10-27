"""Relay fixed model-intent requests through the worker's live bootstrap binding.

HTTP source/auth checks still apply on both API servers. The hop marker only
prevents another relay; it is never a credential or a control-plane permit.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import ipaddress
import json
import re
from typing import Any

import httpx
from fastapi import HTTPException, Request
from starlette.responses import Response

import node_config
from model_intent_access import require_model_intent_source
from network_address import build_url, canonical_host

INTENT_PATH = "/api/models/intent"
HOP_HEADER = "X-QLH-Intent-Hop"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_JSON_NESTING_DEPTH = 64
FORWARD_TIMEOUT_SECONDS = 30.0
_OPERATION_ID = re.compile(r"supply_[0-9a-f]{32}")
_REQUEST_HEADERS = ("X-Request-ID",)
_RESPONSE_HEADERS = ("Content-Type", "X-Request-ID", "X-QLH-Error-Code", "Retry-After")


@dataclass(frozen=True)
class _Connection:
    client: Any
    sock: Any
    identity: tuple


def _host(value) -> str:
    if not isinstance(value, str):
        return ""
    value = canonical_host(value)
    if not value or any(char in value for char in "/\\@?#\r\n\t "):
        return ""
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        # Accept DNS names, never a URL, a host:port pair or encoded authority.
        if len(value) > 253 or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.?", value):
            return ""
        if any(not part or len(part) > 63 or part.startswith("-") or part.endswith("-")
               for part in value.rstrip(".").split(".")):
            return ""
        return value.rstrip(".").lower()
    if address.is_unspecified or address.is_multicast:
        return ""
    return str(address)


def _port(value) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, str):
        if not re.fullmatch(r"[0-9]{1,5}", value):
            return 0
        value = int(value)
    return value if isinstance(value, int) and 1 <= value <= 65535 else 0


def _view(status, is_master, code="", reason="", **fields) -> dict:
    return {"status": status, "is_master": is_master, "master_host": "",
            "master_api_port": 0, "reason_code": code, "reason": reason, **fields}


def _role_binding(scheduler, api_port) -> dict | None:
    try:
        role = scheduler._effective_role()
        facade = getattr(scheduler, "_runtime_setting", None)
        if callable(facade):
            run_mode = facade("RUN_MODE")
        else:
            from config import RUN_MODE
            run_mode = RUN_MODE
        enabled = scheduler.get_distributed_inference_enabled()
    except Exception:
        return _view("not_found", False, "INTENT_ROLE_UNAVAILABLE",
                     "The current node role or inference mode is unavailable")
    is_master = role == "master"
    if run_mode != "distributed" or not enabled:
        return _view("local", is_master, "INTENT_LOCAL_MODE",
                     "Switch to distributed inference before selecting a cluster model")
    if is_master:
        # The master uses its own API, never a worker connection or saved peer.
        port = _port(api_port)
        if not port:
            return _view("not_found", True, "INTENT_MASTER_BINDING_INVALID",
                         "The local master API port is unavailable")
        return _view("connected", True, master_host="127.0.0.1", master_api_port=port)
    if role != "client":
        return _view("not_found", False, "INTENT_CLIENT_REQUIRED",
                     "Only a connected worker can relay model selection")

    return None


def _resolve_binding(scheduler, api_port) -> tuple[dict, _Connection | None]:
    role_binding = _role_binding(scheduler, api_port)
    if role_binding is not None:
        return role_binding, None
    client = getattr(scheduler, "_tcp_client", None)
    sock = getattr(client, "sock", None)
    if (client is None or not getattr(client, "_running", False)
            or not getattr(client, "is_registered", False) or sock is None):
        return _view("not_found", False, "INTENT_MASTER_DISCONNECTED",
                     "Connect to the master before selecting a cluster model"), None
    tcp_host = _host(getattr(client, "server_host", None))
    tcp_port = _port(getattr(client, "server_port", None))
    registration_epoch = getattr(client, "registration_epoch", None)
    try:
        data = node_config.load_node_config()
        if not isinstance(data, dict) or data.get("bootstrapped") is not True:
            return _view("not_found", False, "INTENT_MASTER_NOT_FOUND",
                         "A completed master bootstrap binding is required"), None
        cluster = data.get("cluster")
        node = data.get("node")
        if not isinstance(cluster, dict) or not isinstance(node, dict):
            raise ValueError("missing bootstrap identity")
        cluster_id = cluster.get("cluster_id")
        node_id = node.get("node_id")
        if (not isinstance(cluster_id, str) or not cluster_id.strip()
                or not isinstance(node_id, str) or not node_id.strip()
                or node_id != getattr(client, "client_id", None)):
            return _view("not_found", False, "INTENT_MASTER_BINDING_MISMATCH",
                         "The bootstrap identity does not match the registered worker"), None
        # These are bootstrap TCP fields, not preferred_master_endpoint. A new
        # preferred peer cannot inherit the previous master's API port.
        boot_host = _host(cluster.get("master_tcp_host") or cluster.get("master_host"))
        boot_port = _port(cluster.get("master_tcp_port") or cluster.get("master_port"))
        api_host = _host(cluster.get("master_api_host"))
        port = _port(cluster.get("master_api_port"))
        if not all((tcp_host, tcp_port, boot_host, boot_port, api_host, port)):
            return _view("not_found", False, "INTENT_MASTER_BINDING_INVALID",
                         "The bootstrap API or TCP endpoint is invalid"), None
        if (tcp_host, tcp_port) != (boot_host, boot_port):
            return _view("not_found", False, "INTENT_MASTER_BINDING_MISMATCH",
                         "The live master differs from the bootstrap binding; reconnect with a fresh binding"), None
        identity = (cluster_id, node_id, cluster.get("cluster_secret_epoch"),
                    tcp_host, tcp_port, api_host, port, registration_epoch)
    except Exception:
        return _view("not_found", False, "INTENT_MASTER_BINDING_INVALID",
                     "The master bootstrap binding is unavailable"), None
    # Config reads can yield to role changes or a renewed registration.
    role_binding = _role_binding(scheduler, api_port)
    if role_binding is not None:
        return role_binding, None
    if (getattr(scheduler, "_tcp_client", None) is not client
            or getattr(client, "sock", None) is not sock
            or not getattr(client, "_running", False)
            or not getattr(client, "is_registered", False)
            or getattr(client, "client_id", None) != node_id
            or getattr(client, "registration_epoch", None) != registration_epoch
            or _host(getattr(client, "server_host", None)) != tcp_host
            or _port(getattr(client, "server_port", None)) != tcp_port):
        return _view("not_found", False, "INTENT_MASTER_DISCONNECTED",
                     "The master connection changed while resolving its binding"), None
    return _view("connected", False, master_host=api_host, master_api_port=port,
                 master_tcp_host=tcp_host, master_tcp_port=tcp_port,
                 cluster_id=cluster_id, node_id=node_id), _Connection(client, sock, identity)


def get_binding(scheduler, api_port) -> dict:
    """Return safe endpoint metadata; saved discovery alone never means online."""
    return _resolve_binding(scheduler, api_port)[0]


def _error(status, code, message) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _require_worker(binding, connection) -> None:
    if binding["status"] == "local":
        raise _error(409, binding["reason_code"], binding["reason"])
    if binding["is_master"] or binding["reason_code"] == "INTENT_CLIENT_REQUIRED":
        raise _error(403, "INTENT_CLIENT_REQUIRED", "The master serves model intent locally")
    if binding["status"] != "connected" or connection is None:
        status = 409 if binding["reason_code"] in {
            "INTENT_MASTER_BINDING_INVALID", "INTENT_MASTER_BINDING_MISMATCH",
        } else 503
        raise _error(status, binding["reason_code"], binding["reason"])


def _check_connection(scheduler, original: _Connection) -> None:
    binding, current = _resolve_binding(scheduler, 0)
    _require_worker(binding, current)
    if (current.client is not original.client or current.sock is not original.sock
            or current.identity != original.identity):
        raise _error(409, "INTENT_MASTER_CONNECTION_CHANGED",
                     "The master connection or bootstrap binding changed during the request")


def _path(method: str, suffix: str) -> str:
    if not isinstance(suffix, str):
        raise _error(400, "INTENT_FORWARD_ROUTE_INVALID", "This model intent route cannot be relayed")
    if (method == "POST" and suffix == ""
            or method == "GET" and suffix in {"catalog", "current"}
            or method in {"GET", "DELETE"}
            and _OPERATION_ID.fullmatch(suffix)):
        return INTENT_PATH + ("/" + suffix if suffix else "")
    raise _error(400, "INTENT_FORWARD_ROUTE_INVALID", "This model intent route cannot be relayed")


def _success_json(content: bytearray):
    # Limit containers before decoding creates a recursive object graph. Match
    # json.loads' byte encodings, but never rely on the process recursion limit.
    try:
        text = content.decode(json.detect_encoding(content), errors="surrogatepass")
        depth = 0
        in_string = False
        escaped = False
        for char in text:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char in "[{":
                depth += 1
                if depth > MAX_JSON_NESTING_DEPTH:
                    raise _error(502, "INTENT_MASTER_RESPONSE_INVALID",
                                 f"The master JSON response exceeds the nesting limit of {MAX_JSON_NESTING_DEPTH}")
            elif char in "]}":
                depth -= 1
        return json.loads(text)
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise _error(502, "INTENT_MASTER_RESPONSE_INVALID", "The master API returned invalid JSON") from exc


async def forward_intent(scheduler, request: Request, suffix: str, body=None):
    """Return successful JSON, or a Response preserving the master's error body.

    Callers must retain their normal machine-source boundary. No account token
    is relayed.
    A timeout after a write does not prove that the master rejected the intent.
    """
    if request is None:
        raise _error(400, "INTENT_FORWARD_REQUEST_INVALID", "An HTTP request is required")
    require_model_intent_source(request, scheduler)
    path = _path(request.method, suffix)
    if HOP_HEADER in request.headers:
        raise _error(508, "INTENT_FORWARD_LOOP", "Model intent cannot be relayed twice")
    if request.method != "POST" and body is not None:
        raise _error(400, "INTENT_FORWARD_ROUTE_INVALID", "This model intent route does not accept a body")
    binding, connection = _resolve_binding(scheduler, 0)
    _require_worker(binding, connection)
    url = build_url("http", binding["master_host"], binding["master_api_port"], path)
    headers = {name: request.headers[name] for name in _REQUEST_HEADERS if name in request.headers}
    if "idempotency-key" in request.headers:
        headers["Idempotency-Key"] = request.headers["idempotency-key"]
    headers[HOP_HEADER] = "1"
    # Bound the wire body without accepting compression bombs from the peer.
    headers["Accept-Encoding"] = "identity"
    kwargs = {"json": body} if request.method == "POST" else {}
    try:
        async with asyncio.timeout(FORWARD_TIMEOUT_SECONDS):
            async with httpx.AsyncClient(
                trust_env=False, follow_redirects=False,
                timeout=httpx.Timeout(FORWARD_TIMEOUT_SECONDS, connect=5.0, write=10.0, pool=5.0),
            ) as client:
                _check_connection(scheduler, connection)
                async with client.stream(request.method, url, headers=headers, **kwargs) as upstream:
                    _check_connection(scheduler, connection)
                    if 300 <= upstream.status_code < 400:
                        raise _error(502, "INTENT_MASTER_REDIRECT_FORBIDDEN", "The master API returned a redirect")
                    encoding = upstream.headers.get("Content-Encoding", "identity").strip().lower()
                    if encoding not in {"", "identity"}:
                        raise _error(502, "INTENT_MASTER_RESPONSE_INVALID", "The master API returned an unexpected content encoding")
                    length = upstream.headers.get("Content-Length", "")
                    significant_length = length.lstrip("0") or "0"
                    limit = str(MAX_RESPONSE_BYTES)
                    if length.isdigit() and (len(significant_length) > len(limit)
                            or len(significant_length) == len(limit) and significant_length > limit):
                        raise _error(502, "INTENT_MASTER_RESPONSE_TOO_LARGE", "The master response exceeds 2 MiB")
                    content = bytearray()
                    async for chunk in upstream.aiter_raw(chunk_size=64 * 1024):
                        _check_connection(scheduler, connection)
                        if len(content) + len(chunk) > MAX_RESPONSE_BYTES:
                            raise _error(502, "INTENT_MASTER_RESPONSE_TOO_LARGE", "The master response exceeds 2 MiB")
                        content.extend(chunk)
                    status = upstream.status_code
                    response_headers = {name: upstream.headers[name] for name in _RESPONSE_HEADERS if name in upstream.headers}
            _check_connection(scheduler, connection)
    except (httpx.TimeoutException, TimeoutError) as exc:
        raise _error(504, "INTENT_MASTER_TIMEOUT", "The master API did not respond before the request deadline") from exc
    except httpx.RequestError as exc:
        raise _error(502, "INTENT_MASTER_UNREACHABLE", "The master API could not complete the request") from exc
    if not 200 <= status < 300:
        return Response(content=bytes(content), status_code=status, headers=response_headers)
    return _success_json(content)
