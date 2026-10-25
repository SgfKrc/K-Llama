"""Compose the model control owner without opening storage at import/construction."""
from __future__ import annotations

from contextlib import contextmanager
import time
import uuid
from qlh_core.adapters.model_control_repository import SqliteModelControlRepository
from qlh_core.control.deployment import DeploymentService


def build_model_control(store=None):
    from local_store import _connect, _write_connection
    def connect():
        if store is None:
            return _connect()
        with store.scope():
            return _connect()
    @contextmanager
    def transaction():
        if store is None:
            with _write_connection() as connection:
                yield connection
        else:
            with store.scope(), _write_connection() as connection:
                yield connection
    return DeploymentService(SqliteModelControlRepository(connect=connect, transaction=transaction),
                             boot_id=uuid.uuid4().hex, clock=time.time)
