"""Regression coverage for retired TUI human account authentication."""

from __future__ import annotations

import inspect

import tui_api
import tui_textual
from tui_shared import API_PATHS, COMMAND_SPECS


ACCOUNT_COMMANDS = {"/login", "/logout", "/whoami", "/users", "/totp"}
ACCOUNT_API_WRAPPERS = {
    "auth_capability", "auth_login", "auth_logout", "auth_me",
    "auth_totp_provision", "auth_totp_verify", "list_auth_users",
    "create_auth_user", "patch_auth_user", "delete_auth_user",
}


def test_account_commands_handlers_and_api_wrappers_are_removed():
    command_names = {command["name"] for command in COMMAND_SPECS}
    assert ACCOUNT_COMMANDS.isdisjoint(command_names)

    for handler in ("cmd_login", "cmd_logout", "cmd_whoami", "cmd_users", "cmd_totp", "auth_call"):
        assert not hasattr(tui_textual.ChatPane, handler)
    for wrapper in ACCOUNT_API_WRAPPERS:
        assert not hasattr(tui_api, wrapper)
    assert not any(path.startswith("/auth/") or path == "/users" for path in API_PATHS.values())


def test_api_client_has_no_account_token_or_authorization_header():
    assert "auth_token" not in inspect.signature(tui_api.ApiClient).parameters

    api = tui_api.ApiClient(log_token="logs-only")
    headers = api._request_headers(with_log_token=True)

    assert "Authorization" not in headers
    assert headers["X-QLH-Log-Token"] == "logs-only"
