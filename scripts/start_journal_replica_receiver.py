"""Start one journal replica receiver for explicit journal recovery.

Why this exists: the receiver was only constructible from Python
(``JournalReplicaReceiver`` has no product entrypoint), so the 2026-10-11 drills
had to use a throwaway script. A configured recovery donor needs this process
to be *running and persistent*.

Typical use (Windows):

    set QLH_JOURNAL_REPLICATION_SECRET=<40+ byte secret>   # prefer env over argv
    python scripts/start_journal_replica_receiver.py --replica D:\\qlh\\jr\\node.sqlite3 ^
        --port 9200 --source-node-id node-leader

Termux (Y700):

    python scripts/start_journal_replica_receiver.py --replica ~/qlh-jr/node.sqlite3 \
        --port 9200 --host 0.0.0.0 --source-node-id node-leader

Notes:
  * ``--print-plan`` validates everything and prints what would run, without
    binding a port (safe dry-run).
  * ``--pid-file`` writes the receiver PID so an operator (or boot script) can
    stop it deterministically.
  * The secret is read from ``QLH_JOURNAL_REPLICATION_SECRET`` by default:
    command lines leak into process listings.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from journal_replication import JournalReplicationError  # noqa: E402
from journal_replication_service import (  # noqa: E402
    JournalReplicaReceiver,
    JournalReplicationSettings,
)

DEFAULT_STREAM_ID = "task-graph-journal"
MIN_SECRET_BYTES = 32


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start a journal replica receiver for journal recovery.",
    )
    parser.add_argument("--replica", required=True, help="sqlite path owned by this receiver")
    parser.add_argument("--port", type=int, required=True, help="TCP port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="bind host (default 0.0.0.0)")
    parser.add_argument("--source-node-id", required=True, help="journal source identity")
    parser.add_argument("--stream-id", default=DEFAULT_STREAM_ID)
    parser.add_argument(
        "--secret-env",
        default="QLH_JOURNAL_REPLICATION_SECRET",
        help="env var holding the shared secret (default QLH_JOURNAL_REPLICATION_SECRET)",
    )
    parser.add_argument(
        "--secret",
        default="",
        help="shared secret (discouraged: visible in process listings; prefer --secret-env)",
    )
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--pid-file", default="", help="write receiver PID here after start")
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="validate and print the plan without binding a port (dry-run)",
    )
    return parser


def resolve_secret(args: argparse.Namespace) -> str:
    """Prefer the env var; fall back to ``--secret``.  Never invent a default."""

    from_env = os.environ.get(args.secret_env, "") if args.secret_env else ""
    secret = (from_env or args.secret or "").strip()
    if len(secret.encode("utf-8")) < MIN_SECRET_BYTES:
        raise JournalReplicationError(
            f"shared secret must be at least {MIN_SECRET_BYTES} bytes "
            f"(set {args.secret_env} or pass --secret)"
        )
    return secret


def build_settings(args: argparse.Namespace, secret: str) -> JournalReplicationSettings:
    if not (0 < int(args.port) <= 65535):
        raise JournalReplicationError("port must be within 1..65535")
    if not str(args.replica).strip():
        raise JournalReplicationError("--replica is required")
    replica_path = str(Path(args.replica).expanduser())
    parent = Path(replica_path).parent
    if not parent.exists():
        raise JournalReplicationError(f"replica directory does not exist: {parent}")
    settings = JournalReplicationSettings(
        enabled=True,
        source_node_id=str(args.source_node_id).strip(),
        stream_id=str(args.stream_id).strip() or DEFAULT_STREAM_ID,
        replica_path=replica_path,
        bind_host=str(args.host).strip() or "0.0.0.0",
        bind_port=int(args.port),
        timeout_s=float(args.timeout_s),
        secret=secret.encode("utf-8"),
    )
    settings.validate()
    return settings


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        secret = resolve_secret(args)
        settings = build_settings(args, secret)
    except JournalReplicationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    plan = {
        "replica_path": settings.replica_path,
        "bind": f"{settings.bind_host}:{settings.bind_port}",
        "source_node_id": settings.source_node_id,
        "stream_id": settings.stream_id,
        "secret_bytes": len(settings.secret_bytes),
    }
    if args.print_plan:
        for key, value in plan.items():
            print(f"{key} = {value}")
        print("plan OK (port not bound)")
        return 0

    receiver = JournalReplicaReceiver(settings=settings)
    host, port = receiver.start()
    if args.pid_file:
        Path(args.pid_file).expanduser().write_text(str(os.getpid()), encoding="ascii")
    print(f"RECEIVER READY host={host} port={port} replica={settings.replica_path}", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("stopping receiver", flush=True)
    finally:
        receiver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
