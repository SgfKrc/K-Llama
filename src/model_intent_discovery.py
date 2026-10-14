"""Small, unauthenticated UDP master hints; no runtime imports or credentials.

Consumers send UTF-8 JSON {"protocol":"qlh-master-discovery-v1","nonce":32hex}
(<=1024 bytes) to 239.255.81.76:29876, or a LAN broadcast on that port.
The listener binds 0.0.0.0 and replies only to the sender's unicast endpoint.
Replies contain exactly protocol, nonce, master_api_host (bare IP literal),
master_api_port, master_tcp_port and node_id, and are also <=1024 bytes.

A matching nonce correlates a probe ONLY. It does not authenticate the host,
node_id or ports. Consumers must validate the candidate's fields, GET its
/api/bootstrap/info, then use the existing encrypted first-connect flow;
never use this hint as identity, credential or admission evidence.

API integration: responder = MasterDiscoveryResponder(scheduler, api_port);
call start() at startup and stop() at shutdown. api_port may be an integer or
an inexpensive zero-argument callback returning the actual listening port.
Scheduler callbacks are read afresh for each reply: _effective_role(),
_scheduler_facade_global("RUN_MODE"), get_distributed_inference_enabled(),
get_effective_node_id(), plus the current _tcp_server.port / _running.
start() only schedules a daemon thread; bound_port/running and named logs
expose asynchronous readiness/failure. No retries are hidden from the owner.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import socket
import threading
import time

if __package__:
    from .network_address import is_tailscale_ip
else:
    from network_address import is_tailscale_ip

PROTOCOL = "qlh-master-discovery-v1"
MULTICAST_GROUP = "239.255.81.76"
DISCOVERY_PORT = 29876
MAX_DATAGRAM_BYTES = 1024
GLOBAL_REPLIES_PER_SECOND = 10
SOURCE_REPLIES_PER_SECOND = 2
_SOCKET_TIMEOUT = 0.2
_NONCE = re.compile(r"[0-9a-fA-F]{32}")
logger = logging.getLogger("model_intent_discovery")


def _ip_literal(value):
    if not isinstance(value, str) or "%" in value:
        return None
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return None
    if address.is_unspecified or address.is_multicast or str(address) == "255.255.255.255":
        return None
    return str(address)


def _interface_addresses():
    """Optional local interface enumeration, never a subprocess or DNS lookup."""
    try:
        import psutil
    except ImportError:
        return []
    try:
        return list(dict.fromkeys(
            host
            for addresses in psutil.net_if_addrs().values()
            for entry in addresses
            if entry.family in (socket.AF_INET, socket.AF_INET6)
            if (host := _ip_literal(entry.address)) is not None
        ))
    except Exception:
        logger.warning("master_discovery_interfaces_failed", exc_info=True)
        return []


class MasterDiscoveryResponder:
    """Owned background listener. stop() closes the socket and joins its thread.

    Optional port=0 enables isolated tests; lan_interface is a local IPv4
    interface address for multicast membership, never an advertised URL.
    Without it, join each detected LAN IPv4 interface, or the default route
    when optional interface enumeration is unavailable. Membership failure
    leaves unicast/broadcast listening available and produces a named log.
    Limits use one-second windows: <=10 replies total, <=2 per source IP.
    Source accounting is bounded by the global limit and resets each window.
    """

    def __init__(self, scheduler, api_port, *, port=DISCOVERY_PORT, lan_interface=None):
        self.scheduler = scheduler
        self.api_port = api_port
        self.port = port
        self.lan_interface = lan_interface
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self._socket = None
        self._bound_port = None
        self._addresses = []
        self._addresses_at = float("-inf")
        self._window_at = float("-inf")
        self._reply_count = 0
        self._sources = {}

    @property
    def bound_port(self):
        with self._lock:
            return self._bound_port

    @property
    def running(self):
        with self._lock:
            return self._socket is not None and not self._stop_event.is_set()

    def start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event = threading.Event()
            self._ready.clear()
            self._window_at = float("-inf")
            self._addresses_at = float("-inf")
            self._thread = threading.Thread(
                target=self._listen, args=(self._stop_event,),
                name="master-discovery-responder", daemon=True,
            )
            try:
                self._thread.start()
            except Exception:
                self._thread = None
                self._ready.set()
                logger.exception("master_discovery_thread_start_failed")

    def stop(self):
        with self._lock:
            self._stop_event.set()
            sock, thread = self._socket, self._thread
        if sock is not None:
            self._close(sock)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
            if thread.is_alive():
                logger.warning("master_discovery_stop_timeout: listener still alive")

    @staticmethod
    def _close(sock):
        try:
            sock.close()
        except OSError:
            logger.warning("master_discovery_socket_close_failed", exc_info=True)

    def _local_addresses(self):
        now = time.monotonic()
        if now - self._addresses_at >= 5.0:
            self._addresses = _interface_addresses()
            self._addresses_at = now
        return self._addresses

    def _join_multicast(self, sock):
        if self.lan_interface is not None:
            host = _ip_literal(self.lan_interface)
            if host is None or ":" in host:
                raise ValueError("lan_interface must be a local IPv4 literal")
            interfaces = [host]
        else:
            interfaces = [host for host in self._local_addresses()
                          if ":" not in host and not is_tailscale_ip(host)
                          and not ipaddress.ip_address(host).is_loopback]
            interfaces = interfaces or ["0.0.0.0"]
        for interface in interfaces:
            try:
                membership = socket.inet_aton(MULTICAST_GROUP) + socket.inet_aton(interface)
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
            except OSError:
                logger.warning("master_discovery_multicast_join_failed: interface=%s",
                               interface, exc_info=True)

    def _listen(self, stop_event):
        sock = None
        stage = "socket"
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Do not reuse the port: competing listeners must be observable.
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            sock.settimeout(_SOCKET_TIMEOUT)
            stage = "bind"
            sock.bind(("0.0.0.0", self.port))
            with self._lock:
                if stop_event.is_set():
                    return
                self._socket = sock
                self._bound_port = sock.getsockname()[1]
            stage = "multicast_setup"
            self._join_multicast(sock)
            self._ready.set()
            logger.info("master_discovery_listening: 0.0.0.0:%s group=%s",
                        self.bound_port, MULTICAST_GROUP)
            stage = "receive"
            while not stop_event.is_set():
                try:
                    data, peer = sock.recvfrom(MAX_DATAGRAM_BYTES + 1)
                except socket.timeout:
                    continue
                except OSError as exc:
                    # Windows raises WSAEMSGSIZE for an oversized datagram.
                    if getattr(exc, "winerror", None) == 10040:
                        continue
                    raise
                try:
                    response = self._response(data, peer)
                except Exception:
                    logger.warning("master_discovery_snapshot_failed", exc_info=True)
                    continue
                if response is not None and not stop_event.is_set():
                    try:
                        sock.sendto(response, peer)
                    except OSError:
                        if not stop_event.is_set():
                            logger.warning("master_discovery_send_failed", exc_info=True)
        except Exception:
            if not stop_event.is_set():
                logger.exception("master_discovery_%s_failed", stage)
        finally:
            if sock is not None:
                self._close(sock)
            with self._lock:
                if self._socket is sock:
                    self._socket = None
                    self._bound_port = None
            self._ready.set()
            logger.info("master_discovery_stopped")

    def _allow_reply(self, source):
        now = time.monotonic()
        if now - self._window_at >= 1.0:
            self._window_at = now
            self._reply_count = 0
            self._sources.clear()
        count = self._sources.get(source, 0)
        if self._reply_count >= GLOBAL_REPLIES_PER_SECOND or count >= SOURCE_REPLIES_PER_SECOND:
            return False
        self._reply_count += 1
        self._sources[source] = count + 1
        return True

    def _advertised_host(self, peer):
        candidates = [getattr(self.scheduler, "_lan_ip", None), *self._local_addresses()]
        hosts = [host for value in candidates if (host := _ip_literal(value)) is not None]
        tailnet = sorted((host for host in hosts if is_tailscale_ip(host)),
                         key=lambda host: (":" in host, host))
        if tailnet:
            return tailnet[0]
        for host in hosts[:1]:
            if not ipaddress.ip_address(host).is_loopback:
                return host
        # Ask the local routing table without transmitting a datagram or using DNS.
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
                route.settimeout(_SOCKET_TIMEOUT)
                route.connect(peer)
                if (host := _ip_literal(route.getsockname()[0])) is not None:
                    return host
        except OSError:
            logger.warning("master_discovery_route_failed", exc_info=True)
        return next((host for host in hosts if not ipaddress.ip_address(host).is_loopback), None)

    def _response(self, data, peer):
        if not data or len(data) > MAX_DATAGRAM_BYTES:
            return None
        try:
            request = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeError, RecursionError):
            return None
        if not isinstance(request, dict) or set(request) != {"protocol", "nonce"}:
            return None
        nonce = request["nonce"]
        if request["protocol"] != PROTOCOL or not isinstance(nonce, str) or not _NONCE.fullmatch(nonce):
            return None
        source = _ip_literal(peer[0])
        if source is None or ":" in source or not 1024 <= peer[1] <= 65535:
            return None
        if not self._allow_reply(source):
            return None
        scheduler = self.scheduler
        if (scheduler._effective_role() != "master"
                or scheduler._scheduler_facade_global("RUN_MODE") != "distributed"
                or scheduler.get_distributed_inference_enabled() is not True):
            return None
        server = scheduler._tcp_server
        if server is None or not server._running:
            return None
        api_port = self.api_port() if callable(self.api_port) else self.api_port
        tcp_port = server.port
        if any(type(port) is not int or not 1 <= port <= 65535 for port in (api_port, tcp_port)):
            return None
        node_id = scheduler.get_effective_node_id()
        if not isinstance(node_id, str) or not 1 <= len(node_id) <= 128 or not node_id.isprintable():
            return None
        host = self._advertised_host(peer)
        if host is None:
            return None
        payload = json.dumps({
            "protocol": PROTOCOL, "nonce": nonce, "master_api_host": host,
            "master_api_port": api_port, "master_tcp_port": tcp_port, "node_id": node_id,
        }, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return payload if len(payload) <= MAX_DATAGRAM_BYTES else None
