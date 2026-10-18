"""Body-free deterministic FIFO, cancellation and physical drain scenarios."""
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import threading
import time
from qlh_core.adapters.request_repository import SqliteRequestRepository
from qlh_core.execution.requests import RequestService
from qlh_core.contracts.deadline import RequestDeadline
from tests.test_refactor_execution import plan

@dataclass(frozen=True)
class Scenario:
    scenario_id: str

def available_scenarios():
    return tuple(Scenario(name) for name in ("fifo", "queued_cancel", "terminal_before_drain"))

class LinearSimulationHarness:
    def run(self, scenario_id):
        if scenario_id not in {s.scenario_id for s in available_scenarios()}:
            raise ValueError("unknown_linear_scenario")
        with TemporaryDirectory() as folder:
            path = Path(folder) / "control.db"
            def connect():
                c = sqlite3.connect(path); c.row_factory = sqlite3.Row; return c
            @contextmanager
            def transaction():
                c = connect()
                try:
                    c.execute("BEGIN IMMEDIATE"); yield c; c.commit()
                finally: c.close()
            owner = RequestService(SqliteRequestRepository(connect=connect, transaction=transaction), boot_id="simulation", capacity=2)
            started, release = threading.Event(), threading.Event()
            order = []
            def execute_first(_):
                started.set()
                assert release.wait(3)
                order.append(1)
                return {"response": ""}
            def invoke(key, execute):
                owner.run(plan=plan(), deadline=RequestDeadline.start(4), execute=execute, cleanup=lambda _:True, request_id=key)
            first = threading.Thread(target=invoke, args=("first", execute_first))
            second = threading.Thread(target=invoke, args=("second", lambda _: order.append(2) or {"response":""}))
            try:
                first.start(); assert started.wait(3)
                if scenario_id == "terminal_before_drain":
                    owner.cancel("first"); first.join(2)
                    assert not first.is_alive()
                    record = owner.recent()["requests"][0]
                    assert record["terminal"] == "generation_cancelled" and record["execution_state"] == "running"
                else:
                    second.start()
                    deadline = time.monotonic() + 3
                    while owner.queue_snapshot()["queue_size"] != 1 and time.monotonic() < deadline:
                        threading.Event().wait(.002)
                    assert owner.queue_snapshot()["queue_size"] == 1
                    if scenario_id == "queued_cancel": owner.cancel("second")
                release.set();first.join(3)
                if second.ident:second.join(3)
                owner.close()
                assert order == ([1,2] if scenario_id == "fifo" else [1])
                return {"scenario_id":scenario_id, "outcome":"passed", "native_calls":len(order), "quarantined":0}
            finally:
                release.set();first.join(3)
                if second.ident:second.join(3)
                owner.close()
