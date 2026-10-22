"""Network boundary for model control and local-asset APIs."""

from __future__ import annotations

import ipaddress
import os

from fastapi import HTTPException, Request


_LOCAL_CLIENT_ALIASES = {
    "127.0.0.1",
    "::1",
    "::ffff:127.0.0.1",
    "localhost",
}

_TEST_CLIENT_ALIASES = {"testclient", "testserver"}


def request_client_host(request: Request | None) -> str:
    client = getattr(request, "client", None)
    return (getattr(client, "host", "") if client else "") or "unknown"


def is_model_api_source_trusted(
    request: Request | None,
    *,
    trusted_cidrs: str | None = None,
) -> bool:
    """Allow loopback by default; remote model access is explicit opt-in."""
    if request is None:
        # Direct in-process callers do not cross an HTTP trust boundary.
        return True

    host = request_client_host(request)
    if host in _LOCAL_CLIENT_ALIASES:
        return True
    if host in _TEST_CLIENT_ALIASES and (
        os.environ.get("NODE_ENV", "").strip().lower() == "test"
        or os.environ.get("PYTEST_CURRENT_TEST")
    ):
        return True
    try:
        if ipaddress.ip_address(host).is_loopback:
            return True
    except ValueError:
        return False

    raw_cidrs = trusted_cidrs
    if raw_cidrs is None:
        raw_cidrs = os.environ.get("QLH_MODEL_API_TRUSTED_CIDRS", "").strip()
    if not raw_cidrs:
        return False
    try:
        peer = ipaddress.ip_address(host)
    except ValueError:
        return False
    for item in raw_cidrs.split(","):
        try:
            if peer in ipaddress.ip_network(item.strip(), strict=False):
                return True
        except ValueError:
            continue
    return False


def require_model_api_source(
    request: Request | None,
    *,
    trusted_cidrs: str | None = None,
) -> str:
    """Fail closed before model paths, registry rows, or download jobs are read."""
    host = request_client_host(request)
    if is_model_api_source_trusted(request, trusted_cidrs=trusted_cidrs):
        return host
    raise HTTPException(
        status_code=403,
        detail={
            "code": "MODEL_API_SOURCE_UNTRUSTED",
            "message": "model control APIs require loopback or an explicitly trusted CIDR",
        },
    )
