"""#117: a worker that holds a whole GGUF may join the layer pool without a segment.

Before this change `layer_worker` only looked at "do I already have a layer
configuration".  A node with no pre-cut segment therefore never advertised
`layer_forward`, so the master never assigned it one -- a chicken-and-egg that
made historically mis-cut segments impossible to clear and recompute.
"""
from types import SimpleNamespace

from scheduler import Scheduler
from model_supply_worker import ModelSupplyWorker


def _capabilities(monkeypatch, *, cut: bool, layers: int | None):
    scheduler = Scheduler()
    # 本用例正是要验证自裁路径本身，显式打开（conftest 默认关闭以保持隔离）。
    monkeypatch.setenv("QLH_LOCAL_CUT_SELF", "1")
    monkeypatch.setattr(
        scheduler, "_host",
        SimpleNamespace(model_loaded=False, layer_range=None),
    )
    monkeypatch.setattr(scheduler, "_task_worker_local_cut_available", lambda: cut)
    monkeypatch.setattr(
        scheduler, "_task_worker_local_full_source",
        lambda: {"total_layers": layers, "size_bytes": 3200} if layers else None,
    )
    return scheduler._task_worker_capabilities()


def test_whole_gguf_worker_joins_layer_pool_without_a_segment(monkeypatch):
    capabilities = _capabilities(monkeypatch, cut=True, layers=32)
    assert "layer_forward" in capabilities["stage_types"]
    assert capabilities["layer_budget"]["local_cut"] is True
    assert capabilities["layer_budget"]["per_layer_bytes"] == 100
    # Declares the whole domain: capacity solving may then split by capability
    # instead of inheriting whatever range an earlier run happened to cut.
    assert capabilities["layer_ranges"] == [[0, 32]]


def test_missing_local_source_keeps_node_out_of_layer_pool(monkeypatch):
    capabilities = _capabilities(monkeypatch, cut=True, layers=None)
    assert "layer_forward" not in capabilities["stage_types"]
    assert capabilities["layer_ranges"] == []
    assert "layer_budget" not in capabilities


def test_missing_cut_tooling_keeps_node_out_of_layer_pool(monkeypatch):
    capabilities = _capabilities(monkeypatch, cut=False, layers=32)
    assert "layer_forward" not in capabilities["stage_types"]
    assert capabilities["layer_ranges"] == []


def test_loaded_full_model_keeps_its_full_inference_role(monkeypatch):
    scheduler = Scheduler()
    monkeypatch.setattr(
        scheduler, "_host",
        SimpleNamespace(model_loaded=True, layer_range=None),
    )
    monkeypatch.setattr(scheduler, "_task_worker_local_cut_available", lambda: True)
    monkeypatch.setattr(
        scheduler, "_task_worker_local_full_source",
        lambda: {"total_layers": 32, "size_bytes": 3200},
    )
    capabilities = scheduler._task_worker_capabilities()
    # A node already running a whole model must not also be pulled into the
    # layer pool: that would contend for the same memory with two paths.
    assert "layer_forward" not in capabilities["stage_types"]


def test_capacity_source_probe_uses_inventory_candidates_outside_models_root(tmp_path, monkeypatch):
    import model_supply_worker
    from tests.test_model_supply_worker import gguf

    source = tmp_path / "registered" / "full-model.gguf"
    source.parent.mkdir()
    gguf(source, layers=3)
    monkeypatch.setattr(
        model_supply_worker, "local_gguf_source_candidates",
        lambda: iter([("registered-model", source)]),
    )

    scheduler = Scheduler()
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    scheduler._model_supply_worker = worker

    assert list(worker._source_candidates()) == [("registered-model", source)]
    assert scheduler._task_worker_local_full_source() == {
        "source_model_id": "registered-model", "path": source,
        "total_layers": 3, "size_bytes": source.stat().st_size,
    }
    assert scheduler._task_worker_local_full_layers() == 3
    assert scheduler._task_worker_local_full_source_bytes() == source.stat().st_size
