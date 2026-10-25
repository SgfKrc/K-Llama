"""Stable API error codes without breaking existing ``detail`` consumers."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


ERROR_CODE_HEADER = "X-QLH-Error-Code"


class CodedHTTPException(HTTPException):
    """HTTPException carrying a machine-readable code beside legacy detail."""

    def __init__(
        self,
        status_code: int,
        code: str,
        detail: Any,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        resolved_headers = dict(headers or {})
        resolved_headers[ERROR_CODE_HEADER] = code
        self.error_code = code
        super().__init__(status_code=status_code, detail=detail, headers=resolved_headers)


def coded_http_error(status_code: int, code: str, detail: Any) -> CodedHTTPException:
    return CodedHTTPException(status_code, code, detail)


def error_code_from_exception(exc: HTTPException) -> str:
    code = str(getattr(exc, "error_code", "") or "")
    if code:
        return code
    code = str((exc.headers or {}).get(ERROR_CODE_HEADER, "") or "")
    if code:
        return code
    if isinstance(exc.detail, dict):
        return str(
            exc.detail.get("code") or exc.detail.get("reason_code") or ""
        )
    return ""


def error_response_content(exc: HTTPException, **extra: Any) -> dict[str, Any]:
    content: dict[str, Any] = {"detail": exc.detail, **extra}
    code = error_code_from_exception(exc)
    if code:
        content["error_code"] = code
        from qlh_core.contracts.snapshots import reason
        supplied = exc.detail.get("reason_detail") if isinstance(exc.detail, dict) else None
        content["reason_detail"] = supplied or reason("api", code, "http_adapter", details={"message": exc.detail})
    return content


def install_http_error_handler(app: FastAPI) -> None:
    """Install the compatibility response shape on a standalone service app."""

    @app.exception_handler(HTTPException)
    async def _coded_http_exception_handler(
        _request: Request, exc: HTTPException,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response_content(exc),
            headers=dict(exc.headers or {}),
        )
