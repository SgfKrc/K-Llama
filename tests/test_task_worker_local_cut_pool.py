"""#117: a worker that holds a whole GGUF may join the layer pool without a segment.

Before this change `layer_worker` only looked at "do I already have a layer
configuration".  A node with no pre-cut segment therefore never advertised
`layer_forward`, so the master never assigned it one -- a chicken-and-egg that
made historically mis-cut segments impossible to clear and recompute.
"""
from types import SimpleNamespace

from scheduler import Scheduler


def _capabilities(monkeypatch, *, cut: bool, layers: int | None):
    scheduler = Scheduler()
    monkeypatch.setattr(
        scheduler, "_host",
        SimpleNamespace(model_loaded=False, layer_range=None),
    )
    monkeypatch.setattr(scheduler, "_task_worker_local_cut_available", lambda: cut)
    monkeypatch.setattr(scheduler, "_task_worker_local_full_layers", lambda: layers)
    return scheduler._task_worker_capabilities()


def test_whole_gguf_worker_joins_layer_pool_without_a_segment(monkeypatch):
    capabilities = _capabilities(monkeypatch, cut=True, layers=32)
    assert "layer_forward" in capabilities["stage_types"]
    # Declares the whole domain: capacity solving may then split by capability
    # instead of inheriting whatever range an earlier run happened to cut.
    assert capabilities["layer_ranges"] == [[0, 32]]


def test_missing_local_source_keeps_node_out_of_layer_pool(monkeypatch):
    capabilities = _capabilities(monkeypatch, cut=True, layers=None)
    assert "layer_forward" not in capabilities["stage_types"]
    assert capabilities["layer_ranges"] == []


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
    monkeypatch.setattr(scheduler, "_task_worker_local_full_layers", lambda: 32)
    capabilities = scheduler._task_worker_capabilities()
    # A node already running a whole model must not also be pulled into the
    # layer pool: that would contend for the same memory with two paths.
    assert "layer_forward" not in capabilities["stage_types"]
