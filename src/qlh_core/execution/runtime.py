"""Instance owner of the active native handle and its diagnostic state."""
from __future__ import annotations
from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class RuntimeSnapshot:
    revision: int
    model_loaded: bool
    current_quant: str
    metadata: tuple


class RuntimeController:
    def __init__(self, write_metadata=None):
        self._lock = threading.RLock()
        self.execution_lock = threading.RLock()
        self._write_metadata = write_metadata
        self._manager = None
        self.model_loaded = False
        self.current_quant = "int4"
        self._metadata = {}
        self._revision = 0

    def manager(self):
        with self._lock:
            return self._manager

    def install(self, manager):
        with self._lock:
            self._manager = manager
            self._metadata.clear()
            self._revision += 1

    def update(self, *, model_loaded=None, current_quant=None):
        with self._lock:
            if model_loaded is not None:
                self.model_loaded = bool(model_loaded)
            if current_quant is not None:
                self.current_quant = current_quant
            self._revision += 1

    def set_quant(self, value):
        with self._lock:
            self.current_quant = value
            self._revision += 1

    def _metadata_manager(self):
        manager = self._manager
        # Observe an already-created native owner without triggering a cold
        # lazy factory. The proxy remains installed after the first torch load.
        if type(manager).__name__ == "_LazyModelManager":
            return object.__getattribute__(manager, "_instance")
        return manager

    def publish_metadata(self, name, value):
        with self._lock:
            manager = self._metadata_manager()
            # Published fields have one read path; preserve lazy construction.
            if manager is not None:
                try:
                    self._write_metadata(manager, name, value) if self._write_metadata else None
                except (AttributeError, TypeError):
                    pass  # Read-only native descriptors already own their value.
            self._metadata[name] = value
            self._revision += 1

    def read_metadata(self, name):
        with self._lock:
            manager = self._metadata_manager()
            if manager is not None:
                try:
                    return getattr(manager, name)
                except AttributeError:
                    pass
            if name in self._metadata:
                return self._metadata[name]
            raise AttributeError(name)

    def publication(self):
        return self._lock

    def clear_metadata(self):
        with self._lock:
            self._metadata.clear()
            self._revision += 1

    def snapshot(self):
        with self._lock:
            metadata = tuple((key, self.read_metadata(key)) for key in sorted(self._metadata))
            return RuntimeSnapshot(self._revision, self.model_loaded, self.current_quant, metadata)
