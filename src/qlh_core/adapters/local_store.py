"""Runtime-owned path binding for retained SQLite operations."""
from __future__ import annotations
from pathlib import Path
from contextlib import contextmanager


class RuntimeLocalStore:
    def __init__(self, config):
        environment = config.environment_snapshot()
        self.path = str(Path(environment.get("QLH_SQLITE_PATH") or Path(config.state_dir) / "qlh-control.sqlite3").expanduser().absolute())

    def scope(self):
        from local_store import bind_local_store
        return bind_local_store(self.path)

    def initialize(self):
        from local_store import initialize_local_store
        return initialize_local_store(self.path, migrate_legacy=False)

    def invoke(self, operation, *args, **kwargs):
        with self.scope():
            return operation(*args, **kwargs)

    def catalog_repository(self):
        from local_store import _connect, _write_connection, _write_model_registry_row, _now
        from qlh_core.adapters.catalog_repository import SqliteCatalogRepository
        def connect():
            with self.scope():
                return _connect()
        @contextmanager
        def transaction():
            with self.scope(), _write_connection() as connection:
                yield connection
        return SqliteCatalogRepository(connect=connect, transaction=transaction,
                                       write_row=_write_model_registry_row, now=_now)
