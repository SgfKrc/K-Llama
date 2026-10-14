"""Source gate for path-free model intents and required distributed text chat.

A Tailnet address is not a credential. Extra source admission requires the
master's live HMAC-authenticated, ACK-confirmed TCP registration and its real
socket peer. Bearer auth and the master control fence remain separate gates.
"""
from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import re
from typing import Any

from fastapi import HTTPException, Request

from model_api_access import (
    is_model_api_source_trusted, request_client_host, require_model_api_source,
)
from network_address import TAILSCALE_IPV4_NETWORK, TAILSCALE_IPV6_NETWORK

_BASE_PATH = "/api/models/intent"
_OPERATION = re.compile(r"supply_[0-9a-f]{32}")
_PEER_KEY = "_qlh_model_intent_source_peer"

_CHAT_KEY = "_qlh_registered_distributed_chat"
_CHAT_FIELDS = frozenset({
    "message", "session_id", "generation_id", "max_new_tokens", "temperature", "top_p",
    "show_thinking", "enable_thinking", "streaming_mode", "routing_preference", "execution_mode",
    "client_node_id", "client_node_type", "client_mode", "client_app_variant",
    "image_data_urls", "allow_external", "prefer_external",
})


async def mark_registered_chat_request(request: Request, scheduler) -> bool:
    """Classify only the fixed required text route, without granting peer access.

    Reading the same cached ASGI body leaves the existing ChatRequest validator
    authoritative for types and generation limits. Unknown/local execution fields
    cannot enter this source exception.
    """
    request.scope.pop(_CHAT_KEY, None)
    if request.method != "POST" or request.url.path not in {"/api/chat", "/api/chat/stream"}:
        return False
    if not is_model_api_source_trusted(request):
        source = _ip(request_client_host(request))
        tailnet = source is not None and (
            source.version == 4 and source in TAILSCALE_IPV4_NETWORK
            or source.version == 6 and source in TAILSCALE_IPV6_NETWORK
        )
        if not tailnet:
            return False
        peer = _registered_peer(scheduler, source, request.scope.get(_PEER_KEY))
        if peer is None:
            return False
        request.scope[_PEER_KEY] = peer
    try:
        payload = await request.json()
    except (ValueError, UnicodeError):
        return False
    if (not isinstance(payload, dict) or set(payload) - _CHAT_FIELDS
            or payload.get("routing_preference") != "distributed_required"
            or payload.get("execution_mode", "auto") != "auto"
            or not isinstance(payload.get("message"), str) or not payload["message"]
            or payload.get("image_data_urls", []) != []
            or payload.get("allow_external") is not None and payload.get("allow_external") is not False
            or payload.get("prefer_external") is not None and payload.get("prefer_external") is not False):
        return False
    request.scope[_CHAT_KEY] = (request.method, request.url.path)
    return True


def is_registered_chat_request(request: Request | None) -> bool:
    return bool(request is not None and request.method == "POST"
                and request.url.path in {"/api/chat", "/api/chat/stream"}
                and request.scope.get(_CHAT_KEY) == (request.method, request.url.path))


@dataclass(frozen=True)
class _Peer:
    scheduler: Any
    server: Any
    connection: Any
    sock: Any
    node_id: str
    epoch: int
    endpoint: tuple


def is_path_free_intent_request(request: Request) -> bool:
    """Keep connect, unknown verbs/routes and every model-management API out."""
    path = request.url.path
    method = request.method
    if path == _BASE_PATH:
        return method == "POST"
    if not path.startswith(_BASE_PATH + "/"):
        return False
    suffix = path[len(_BASE_PATH) + 1:]
    return (method == "GET" and suffix in {"catalog", "current"}
            or method in {"GET", "DELETE"} and _OPERATION.fullmatch(suffix) is not None)


def _ip(value):
    if not isinstance(value, str) or "%" in value:
        return None
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address


def _endpoint(value):
    if not isinstance(value, tuple) or len(value) < 2:
        return None
    address, port = _ip(value[0]), value[1]
    if address is None or isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        return None
    return address, port


def _master_server(scheduler):
    # Runtime authority, not discovery, node-list projections or saved config.
    if scheduler._effective_role() != "master":
        return None
    facade = getattr(scheduler, "_scheduler_facade_global", None)
    if callable(facade):
        run_mode = facade("RUN_MODE")
    else:
        from config import RUN_MODE
        run_mode = RUN_MODE
    if run_mode != "distributed" or not scheduler.get_distributed_inference_enabled():
        return None
    server = getattr(scheduler, "tcp_server", None)
    return server if server is not None and server._running else None


def _evidence(scheduler, server, node_id, connection, source):
    if (not isinstance(node_id, str) or not node_id or node_id == "master"
            or connection.client_id != node_id or connection.registration_confirmed is not True):
        return None
    # Pending registrations and their advertised endpoints grant nothing.
    if getattr(server, "_pending_registrations", {}).get(node_id) is not None:
        return None
    epoch = connection.registration_epoch
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        return None
    sock = connection.sock
    if sock is None:
        return None
    descriptor = sock.fileno()
    if isinstance(descriptor, bool) or not isinstance(descriptor, int) or descriptor < 0:
        return None
    endpoint = _endpoint(sock.getpeername())
    if endpoint is None or endpoint != _endpoint(connection.addr) or endpoint[0] != source:
        return None
    if (server.clients.get(node_id) is not connection or connection.sock is not sock
            or connection.registration_epoch != epoch or connection.registration_confirmed is not True
            or _endpoint(connection.addr) != endpoint or sock.fileno() != descriptor
            or _master_server(scheduler) is not server):
        return None
    return _Peer(scheduler, server, connection, sock, node_id, epoch, endpoint)


def _registered_peer(scheduler, source, original):
    try:
        server = _master_server(scheduler)
        if server is None or original is not None and original.server is not server:
            return None
        # Use the TCP owner's lock so reconnect/credential rotation cannot
        # publish a different registration in the middle of this snapshot.
        with server._clients_lock:
            if original is not None:
                if original.scheduler is not scheduler:
                    return None
                connection = server.clients.get(original.node_id)
                if connection is not original.connection:
                    return None
                current = _evidence(scheduler, server, original.node_id, connection, source)
                if (current is None or current.sock is not original.sock
                        or current.epoch != original.epoch or current.endpoint != original.endpoint):
                    return None
                return current
            for node_id, connection in server.clients.items():
                try:
                    current = _evidence(scheduler, server, node_id, connection, source)
                except (AttributeError, OSError, TypeError, ValueError):
                    continue
                if current is not None:
                    return current
    except Exception:
        # Missing real peer/owner evidence never falls back to claimed hosts.
        return None
    return None


def require_model_intent_source(request: Request | None, scheduler) -> str:
    """Apply the original gate plus live registered Tailnet peers for intent.

    Store only in-process evidence in the ASGI scope, then recheck that exact
    owner at the handler and response boundaries. No request header conveys it.
    """
    if request is None:
        return require_model_api_source(request)
    original = request.scope.get(_PEER_KEY)
    if original is None and is_model_api_source_trusted(request):
        return request_client_host(request)
    source = _ip(request_client_host(request))
    tailnet = source is not None and (
        source.version == 4 and source in TAILSCALE_IPV4_NETWORK
        or source.version == 6 and source in TAILSCALE_IPV6_NETWORK
    )
    if (tailnet and (is_path_free_intent_request(request) or is_registered_chat_request(request))
            and (original is None or isinstance(original, _Peer))):
        peer = _registered_peer(scheduler, source, original)
        if peer is not None:
            request.scope[_PEER_KEY] = peer
            return request_client_host(request)
    # Preserve the established public rejection code and explicit trust policy.
    # A request with captured evidence must not adopt a replacement connection.
    raise HTTPException(403, {"code": "MODEL_API_SOURCE_UNTRUSTED",
                              "message": "model intent requires a trusted source or a current registered Tailnet peer"})
