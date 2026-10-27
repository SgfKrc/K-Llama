from __future__ import annotations

import base64
import sqlite3
import threading

import pytest

from bootstrap_nonce_ledger import BootstrapNonceError, BootstrapRequestNonceLedger


def _nonce(value: bytes = b"n" * 18) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def test_bootstrap_nonce_is_durable_and_only_creates_its_ledger(tmp_path):
    path = tmp_path / "bootstrap.sqlite3"
    nonce = _nonce()
    digest = "a" * 64

    BootstrapRequestNonceLedger(path).consume(
        nonce=nonce, request_digest=digest, consumed_at=10,
    )

    with pytest.raises(BootstrapNonceError) as replay:
        BootstrapRequestNonceLedger(path).consume(
            nonce=nonce, request_digest=digest, consumed_at=11,
        )
    assert replay.value.code == "nonce_replayed"

    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'",
            )
        }
    assert tables == {"cluster_bootstrap_request_nonces"}
    assert not (tmp_path / "cluster_join_secrets.json").exists()


def test_concurrent_bootstrap_nonce_consumption_has_one_winner(tmp_path):
    path = tmp_path / "bootstrap-concurrent.sqlite3"
    ledger_a = BootstrapRequestNonceLedger(path)
    ledger_b = BootstrapRequestNonceLedger(path)
    barrier = threading.Barrier(3)
    outcomes = []

    def consume(ledger):
        barrier.wait(timeout=5)
        try:
            ledger.consume(nonce=_nonce(), request_digest="b" * 64, consumed_at=12)
            outcomes.append("consumed")
        except BootstrapNonceError as exc:
            outcomes.append(exc.code)

    workers = [
        threading.Thread(target=consume, args=(ledger_a,)),
        threading.Thread(target=consume, args=(ledger_b,)),
    ]
    for worker in workers:
        worker.start()
    barrier.wait(timeout=5)
    for worker in workers:
        worker.join(timeout=5)

    assert not any(worker.is_alive() for worker in workers)
    assert sorted(outcomes) == ["consumed", "nonce_replayed"]


@pytest.mark.parametrize(
    ("nonce", "digest", "consumed_at"),
    [
        ("not-base64", "c" * 64, 1),
        (_nonce(b"short"), "c" * 64, 1),
        (_nonce(), "not-a-digest", 1),
        (_nonce(), "c" * 64, float("nan")),
    ],
)
def test_invalid_bootstrap_nonce_record_is_rejected(tmp_path, nonce, digest, consumed_at):
    with pytest.raises(BootstrapNonceError):
        BootstrapRequestNonceLedger(tmp_path / "invalid.sqlite3").consume(
            nonce=nonce, request_digest=digest, consumed_at=consumed_at,
        )
