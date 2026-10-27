"""Tests for the journal replica receiver launcher (#93 deployment gap).

The launcher must be safe to dry-run and must fail closed on a missing/short
shared secret, so the receiver never accepts unauthenticated checkpoint traffic.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
SCRIPT = REPO_ROOT / "scripts" / "start_journal_replica_receiver.py"
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_launcher():
    spec = importlib.util.spec_from_file_location("qlh_jr_receiver_launcher", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def launcher():
    return _load_launcher()


class _Args:
    def __init__(self, **kwargs):
        self.replica = kwargs.get("replica", "")
        self.port = kwargs.get("port", 9200)
        self.host = kwargs.get("host", "0.0.0.0")
        self.source_node_id = kwargs.get("source_node_id", "node-leader")
        self.stream_id = kwargs.get("stream_id", "task-graph-journal")
        self.secret_env = kwargs.get("secret_env", "QLH_JOURNAL_REPLICATION_SECRET")
        self.secret = kwargs.get("secret", "")
        self.timeout_s = kwargs.get("timeout_s", 5.0)


def test_print_plan_succeeds_without_binding_a_port(tmp_path, launcher, monkeypatch, capsys):
    monkeypatch.setenv("QLH_JOURNAL_REPLICATION_SECRET", "s" * 40)
    code = launcher.main([
        "--replica", str(tmp_path / "recv.sqlite3"),
        "--port", "9299",
        "--source-node-id", "node-leader",
        "--print-plan",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "plan OK (port not bound)" in out
    # The plan must not leak the secret itself.
    assert "s" * 40 not in out
    assert "secret_bytes = 40" in out


def test_short_secret_is_rejected(monkeypatch, tmp_path, launcher, capsys):
    monkeypatch.setenv("QLH_JOURNAL_REPLICATION_SECRET", "too-short")
    code = launcher.main([
        "--replica", str(tmp_path / "recv.sqlite3"),
        "--port", "9299",
        "--source-node-id", "node-leader",
        "--print-plan",
    ])
    assert code == 2
    assert "at least" in capsys.readouterr().err


def test_env_secret_wins_over_argument(launcher, monkeypatch):
    monkeypatch.setenv("QLH_JOURNAL_REPLICATION_SECRET", "e" * 40)
    assert launcher.resolve_secret(_Args(secret="a" * 40)) == "e" * 40


def test_argument_secret_is_used_when_env_is_empty(launcher, monkeypatch):
    monkeypatch.delenv("QLH_JOURNAL_REPLICATION_SECRET", raising=False)
    assert launcher.resolve_secret(_Args(secret="a" * 40)) == "a" * 40


def test_missing_replica_directory_is_rejected(launcher, monkeypatch, capsys):
    monkeypatch.setenv("QLH_JOURNAL_REPLICATION_SECRET", "s" * 40)
    missing = Path(os.environ.get("TEMP", ".")) / "qlh-does-not-exist-dir" / "recv.sqlite3"
    code = launcher.main([
        "--replica", str(missing),
        "--port", "9299",
        "--source-node-id", "node-leader",
        "--print-plan",
    ])
    assert code == 2
    assert "replica directory does not exist" in capsys.readouterr().err


def test_invalid_port_is_rejected(launcher, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("QLH_JOURNAL_REPLICATION_SECRET", "s" * 40)
    code = launcher.main([
        "--replica", str(tmp_path / "recv.sqlite3"),
        "--port", "70000",
        "--source-node-id", "node-leader",
        "--print-plan",
    ])
    assert code == 2
    assert "port must be within" in capsys.readouterr().err
