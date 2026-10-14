"""First-connect bootstrap for trusted Tailscale nodes."""

from __future__ import annotations

import ipaddress
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from bootstrap_credentials import (
    BootstrapCredentialError,
    create_bootstrap_credential_request,
    open_bootstrap_credential,
)
from node_config import apply_runtime_config, persist_bootstrap_response
from network_address import build_url, canonical_host, is_tailscale_ip


DEFAULT_TRUSTED_CIDRS = "100.64.0.0/10,127.0.0.0/8,::1/128,fd7a:115c:a1e0::/48"
_WINDOWS_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _cidr_list(raw: str | None = None) -> list[ipaddress._BaseNetwork]:
    value = raw if raw is not None else os.environ.get(
        "QLH_TRUSTED_BOOTSTRAP_CIDRS", DEFAULT_TRUSTED_CIDRS
    )
    networks = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            networks.append(ipaddress.ip_network(item, strict=False))
        except ValueError:
            continue
    return networks


def is_trusted_bootstrap_source(host: str, cidrs: str | None = None) -> bool:
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(ip in network for network in _cidr_list(cidrs))


def is_secure_bootstrap_source(host: str) -> bool:
    """Allow credential delivery only over loopback or Tailnet overlay."""
    try:
        address = ipaddress.ip_address((host or "").split("%", 1)[0])
    except ValueError:
        return False
    return bool(address.is_loopback or is_tailscale_ip(str(address)))


def normalize_node_type(node_type: str | None) -> str:
    value = (node_type or "pc").strip().lower()
    if value in {"android", "mobile"}:
        return "android"
    return "pc"


def normalize_node_id(node_id: str | None, node_type: str = "pc") -> str:
    raw = (node_id or "").strip()
    if not raw or raw == "master":
        prefix = "android" if node_type == "android" else "client"
        raw = f"{prefix}_{socket.gethostname()}"
    allowed = []
    for ch in raw:
        if ch.isalnum() or ch in {"_", "-", "."}:
            allowed.append(ch)
        else:
            allowed.append("_")
    normalized = "".join(allowed).strip("._-")
    if not normalized or normalized == "master":
        normalized = f"client_{socket.gethostname()}"
    return normalized[:64]


def select_advertised_master_host(requested_host: str, lan_ip: str = "") -> str:
    """Keep Tailnet joins on the routable Tailnet address instead of a LAN IP."""
    host = canonical_host(requested_host)
    if is_tailscale_ip(host) or host.lower().endswith(".ts.net"):
        return host
    if lan_ip and lan_ip not in {"0.0.0.0", "127.0.0.1", "localhost"}:
        return lan_ip
    return host


def _find_tailscale_executable() -> str | None:
    found = shutil.which("tailscale") or shutil.which("tailscale.exe")
    if found:
        return found
    if sys.platform == "win32":
        candidate = os.path.join(
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            "Tailscale",
            "tailscale.exe",
        )
        if os.path.isfile(candidate):
            return candidate
    return None


def get_tailnet_peer_ips(timeout: float = 3.0) -> list[str]:
    """Return online Tailnet peer IPv4 addresses without requiring DB access."""
    configured = os.environ.get("QLH_TAILNET_PEERS", "")
    peers = [item.strip() for item in configured.split(",") if item.strip()]

    executable = _find_tailscale_executable()
    if executable:
        try:
            result = subprocess.run(
                [executable, "status", "--json"],
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
                creationflags=_WINDOWS_NO_WINDOW,
            )
            if result.returncode == 0 and result.stdout.strip():
                status = json.loads(result.stdout)
                for peer in (status.get("Peer") or {}).values():
                    if peer.get("Online") is False:
                        continue
                    for address in peer.get("TailscaleIPs") or []:
                        try:
                            ip = ipaddress.ip_address(address)
                        except ValueError:
                            continue
                        # v4（100.64/10）与 v6（fd7a:115c:a1e0::/48 ULA）对等点均放行
                        if is_trusted_bootstrap_source(address):
                            peers.append(address)
        except Exception:
            pass

    return list(dict.fromkeys(peers))


def discover_master_via_tailnet(
    api_port: int = 8000,
    timeout: float = 1.5,
) -> dict[str, Any]:
    """Probe approved Tailnet peers for a confirmed QLH master."""
    peers = get_tailnet_peer_ips()
    if not peers:
        return {"found": False, "source": "tailnet"}

    def _probe(host: str) -> dict[str, Any] | None:
        canonical = canonical_host(host)
        url = build_url("http", canonical, api_port, "/api/bootstrap/info")
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8") or "{}")
            if not data.get("is_master"):
                return None
            return {
                "found": True,
                "master_host": canonical,
                "master_port": int(data.get("master_tcp_port") or 8888),
                "master_api_port": int(data.get("master_api_port") or api_port),
                "stale": False,
                "source": "tailnet",
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=min(12, len(peers))) as executor:
        futures = [executor.submit(_probe, peer) for peer in peers]
        for future in as_completed(futures):
            result = future.result()
            if result:
                for pending in futures:
                    pending.cancel()
                return result
    return {"found": False, "source": "tailnet"}


def first_connect(
    master_api_host: str,
    master_api_port: int = 8000,
    node_id: str | None = None,
    node_type: str = "pc",
    app_variant: str = "",
    capabilities: dict[str, Any] | None = None,
    timeout: float = 8.0,
) -> dict[str, Any]:
    node_type = normalize_node_type(node_type)
    normalized_node_id = normalize_node_id(node_id, node_type)
    credential_request = create_bootstrap_credential_request()
    payload = {
        "node_id": normalized_node_id,
        "node_type": node_type,
        "hostname": socket.gethostname(),
        "platform": platform.system().lower(),
        "app_variant": app_variant,
        "capabilities": capabilities or {},
        **credential_request.fields(),
    }
    url = build_url(
        "http",
        master_api_host,
        master_api_port,
        "/api/bootstrap/first-connect",
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"bootstrap HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"bootstrap request failed: {exc.reason}") from exc

    data = json.loads(body or "{}")
    if data.get("status") not in {"ok", "registered", "updated"}:
        raise RuntimeError("bootstrap rejected")

    cluster = data.get("cluster") if isinstance(data.get("cluster"), dict) else {}
    response_node = data.get("node") if isinstance(data.get("node"), dict) else {}
    try:
        credential = open_bootstrap_credential(
            data.get("credential_envelope"),
            private_key=credential_request.private_key,
            expected_request_nonce=credential_request.request_nonce,
            expected_cluster_id=str(cluster.get("cluster_id") or ""),
            expected_node_id=str(response_node.get("node_id") or normalized_node_id),
        )
    except BootstrapCredentialError as exc:
        raise RuntimeError(f"bootstrap credential rejected: {exc.code}") from exc
    cluster["cluster_secret"] = credential["cluster_secret"]
    cluster["cluster_secret_epoch"] = credential["cluster_secret_epoch"]
    data["cluster"] = cluster
    data.pop("credential_envelope", None)

    persist_bootstrap_response(data)
    apply_runtime_config(data)
    return data
