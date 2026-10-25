"""Executable architecture boundaries; synthetic violations exercise the gate itself."""
from pathlib import Path

import pytest

from scripts.check_refactor_boundaries import check_repository, check_source, load_policy

pytestmark = pytest.mark.refactor_contract
ROOT = Path(__file__).resolve().parents[1]
POLICY = load_policy()


def codes(source, module="qlh.control.operations"):
    return {v.code for v in check_source(source, "synthetic.py", module, POLICY)}


def test_repository_boundary_gate_checks_real_modules():
    violations, checked = check_repository(ROOT, POLICY)
    assert not violations, violations
    assert set(POLICY["legacy_pure_modules"]) <= set(checked)
    assert len(checked) >= 5


@pytest.mark.parametrize("source,module,code", [
    ("import torch", "qlh.contracts.identity", "non_stdlib_import"),
    ("from fastapi import Request", "qlh.control.operations", "non_stdlib_import"),
    ("from qlh.adapters.engine import load", "qlh.control.operations", "layer_import"),
    ("from ..adapters import engine", "qlh.control.operations", "layer_import"),
    ("from qlh import adapters", "qlh.planning.solver", "layer_import"),
    ("import scheduler", "qlh.adapters.transport", "retired_or_facade_import"),
    ("from task_graph import Workflow", "qlh.execution.relay", "retired_or_facade_import"),
    ("from src.model_host import ModelHost", "qlh.api.models", "noncanonical_import"),
    ("from pathlib import Path\ndef move():\n    Path('a').replace('b')", "qlh.contracts.identity", "pure_io"),
    ("import src.qlh.execution.runtime", "qlh.api.models", "noncanonical_import"),
    ("from inference_service.engine_host import EngineHost", "qlh.api.models", "api_engine_import"),
    ("from model_host import ModelHost", "qlh.api.models", "api_engine_import"),
    ("from llama_engine import LlamaCppEngine", "qlh.api.models", "api_engine_import"),
    ("import importlib\nimportlib.import_module('scheduler')", "qlh.control.operations", "dynamic_code"),
    ("from importlib import import_module\nimport_module('torch')", "qlh.contracts.identity", "dynamic_code"),
    ("import importlib as loader\nloader.import_module('torch')", "qlh.control.operations", "dynamic_code"),
    ("from importlib import import_module as load\nload('torch')", "qlh.contracts.identity", "dynamic_code"),
    ("import os as fs\ndef erase():\n    fs.replace('a', 'b')", "qlh.contracts.identity", "pure_io"),
    ("exec('model_host.model_loaded = True')", "qlh.control.operations", "dynamic_code"),
    ("def solve():\n    return open('weights.gguf', 'rb')", "qlh.planning.solver", "pure_io"),
    ("from pathlib import Path\ndef read():\n    return Path('model').read_bytes()", "qlh.contracts.identity", "pure_io"),
    ("import sqlite3\nstore = sqlite3.connect('state.sqlite3')", "qlh.control.operations", "import_resource"),
    ("def f(value: open('state')):\n    pass", "qlh.control.operations", "import_resource"),
    ("resource = (lambda: open('state'))()", "qlh.control.operations", "import_resource"),
    ("def start(store=open('state')):\n    pass", "qlh.control.operations", "import_resource"),
    ("class Runtime:\n    data = open('state')", "qlh.execution.runtime", "import_resource"),
    ("def start():\n    pass", "qlh.surprise.runtime", "unclassified_layer"),
])
def test_dependency_and_import_effect_violations_are_rejected(source, module, code):
    assert code in codes(source, module)


@pytest.mark.parametrize("source", [
    "model_host.model_loaded = True",
    "for host.model_loaded in [True]:\n    pass",
    "with lock as host.current_quant:\n    pass",
    "host.__dict__.update({'model_loaded': True})",
    "host.__dict__.update(model_loaded=True)",
    "host.__dict__.setdefault('model_loaded', True)",
    "model_host.current_quant: str = 'int4'",
    "scheduler.active_deployment += 1",
    "del model_host._manager",
    "setattr(model_host, 'model_loaded', True)",
    "delattr(model_host, 'model_loaded')",
    "scheduler._worker_assignments['node'] = 'ready'",
    "scheduler._worker_assignments.clear()",
    "scheduler._worker_assignments.update({'node': 'ready'})",
    "model_host.__dict__['model_loaded'] = True",
])
def test_cross_owner_writes_are_rejected(source):
    assert "foreign_write" in codes(source, "qlh.api.models")


def test_computed_attribute_write_cannot_hide_owner():
    assert "dynamic_write" in codes("setattr(target, request.field, request.value)")


@pytest.mark.parametrize("source,module", [
    ("if __name__ == '__main__':\n    app.run()", "qlh.bootstrap.main"),
    ("from __future__ import annotations\ndef f(value: open('state')):\n    pass", "qlh.control.operations"),
    ("from dataclasses import replace\ndef evolve(value):\n    return replace(value, revision=2)", "qlh.contracts.identity"),
    ("def clean(items):\n    items.remove(1)\n    return 'x'.replace('x', 'y')", "qlh.planning.solver"),
    ("from dataclasses import dataclass\nfrom typing import Mapping", "qlh.contracts.identity"),
    ("from ..contracts import ModelRevision", "qlh.planning.solver"),
    ("from .identity import ModelRevision", "qlh.contracts.plan"),
    ("def __init__(self):\n    self.model_loaded = False", "qlh.execution.runtime"),
    ("self._worker_assignments.clear()", "qlh.control.deployment"),
    ("value = model_host.model_loaded", "qlh.api.models"),
    ("def read(store):\n    return store.read_bytes()", "qlh.adapters.files"),
])
def test_legal_dependencies_reads_and_owner_writes_pass(source, module):
    assert codes(source, module) == set()


def test_syntax_errors_are_not_skipped():
    assert codes("def broken(:") == {"syntax_error"}


def test_required_legacy_scope_cannot_silently_disappear(tmp_path):
    violations, checked = check_repository(tmp_path, POLICY)
    assert checked == []
    assert len(violations) == len(POLICY["legacy_pure_modules"])
    assert {v.code for v in violations} == {"missing_required_module"}


def test_retained_pure_contracts_reject_function_body_io():
    violations = check_source("def read():\n    return open('weights')", "legacy.py",
                              "worker_prepare_contract", POLICY, legacy_pure=True)
    assert {v.code for v in violations} == {"pure_io"}


def test_deadline_constructor_exception_does_not_allow_thread_start():
    safe = check_source("def coerce():\n    return RequestDeadline.start(1)", "request_deadline.py",
                        "request_deadline", POLICY, legacy_pure=True)
    unsafe = check_source("def run():\n    worker.start()", "request_deadline.py",
                          "request_deadline", POLICY, legacy_pure=True)
    assert safe == []
    assert {v.code for v in unsafe} == {"pure_io"}


@pytest.mark.parametrize("module", ["qlh.control.catalog", "qlh_core.control.catalog"])
def test_installed_catalog_namespace_is_the_same_owner(module):
    assert codes("self.registry_revision = 2", module) == set()


@pytest.mark.parametrize("module", ["qlh_core.api.models", "qlh_core.control.operations"])
def test_installed_namespace_keeps_foreign_write_protection(module):
    assert "foreign_write" in codes("self.registry_revision = 2", module)


@pytest.mark.parametrize("module,source,allowed", [
    ("pipeline_capacity", "from qlh_core.planning.capacity import solve_pipeline_capacity", True),
    ("worker_prepare_contract", "from qlh_core.contracts.load import RuntimeLoadSpec", True),
    ("pipeline_capacity", "from qlh_core.execution.runtime import load", False),
    ("worker_prepare_contract", "from qlh_core.control.catalog import CatalogService", False),
    ("pipeline_capacity", "from qlh_core.planning.capacity_extra import solve", False),
])
def test_legacy_pure_facade_delegates_only_to_named_pure_core(module, source, allowed):
    result = check_source(source, "synthetic.py", module, POLICY, legacy_pure=True)
    assert (result == []) is allowed
