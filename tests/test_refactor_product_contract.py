"""Fixed external schemas and a complete baseline entrypoint disposition inventory.

Cosmetic OpenAPI text and handler names are excluded. Changes require explicit
contract review; there is no update-from-HEAD path or automatic snapshot rewrite.
Actual auth, model-mode, SSE and state behavior are run by the R0 suite manifest.
"""
import ast
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.refactor_contract
ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
SNAPSHOT = json.loads((FIXTURES / "refactor_product_api_v1.json").read_text(encoding="utf-8"))
ENTRIES = json.loads((FIXTURES / "refactor_entrypoints_v1.json").read_text(encoding="utf-8"))


def semantic(value, *, named_mapping=False):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not named_mapping and key in SNAPSHOT["ignored_presentation_keys"]:
                continue
            if key in {"default", "const", "enum"} and not named_mapping:
                result[key] = item  # JSON payload values are part of the contract.
            else:
                result[key] = semantic(item, named_mapping=(
                    not named_mapping and key in {"properties", "patternProperties", "$defs", "dependentSchemas"}
                ))
        return result
    if isinstance(value, list):
        return [semantic(item) for item in value]
    return value


@pytest.fixture(scope="module")
def openapi():
    import api_server
    return api_server.app.openapi()


@pytest.mark.parametrize("path", sorted(SNAPSHOT["paths"]))
def test_product_parameters_responses_and_validation_are_fixed(openapi, path):
    for method, expected in SNAPSHOT["paths"][path].items():
        assert semantic(openapi["paths"][path][method]) == expected


@pytest.mark.parametrize("name", sorted(SNAPSHOT["schemas"]))
def test_product_request_and_response_fields_are_fixed(openapi, name):
    assert semantic(openapi["components"]["schemas"][name]) == SNAPSHOT["schemas"][name]


def test_product_inventory_accounts_for_every_existing_http_operation(openapi):
    actual = {p: sorted(methods) for p, methods in openapi["paths"].items()}
    assert actual == {p: entry["methods"] for p, entry in ENTRIES["http_surfaces"]["product_api"].items()}


def test_native_service_inventory_accounts_for_parallel_load_and_decode_entries():
    from fastapi import FastAPI
    from inference_service.routes import router
    app = FastAPI()
    app.include_router(router)
    actual = {p: sorted(methods) for p, methods in app.openapi()["paths"].items()}
    assert actual == {p: entry["methods"] for p, entry in ENTRIES["http_surfaces"]["native_service"].items()}


def test_entry_dispositions_have_explicit_migration_phase():
    for entries in ENTRIES["http_surfaces"].values():
        for entry in entries.values():
            assert entry["disposition"] in {"migrate", "retire", "extension"}
            assert entry["phase"] in {f"R{i}" for i in range(7)}
    for entry in ENTRIES["non_http"]:
        path = ROOT / entry["path"]
        assert path.is_file()
        assert entry["responsibility"] and entry["phase"] in {f"R{i}" for i in range(7)}
        if path.suffix == ".py":
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            assert any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                       and n.name == entry["symbol"] for n in ast.walk(tree)), entry
        else:
            assert entry["symbol"] in path.read_text(encoding="utf-8")


def test_graph_poc_and_priority_strategy_are_explicitly_retired():
    product = ENTRIES["http_surfaces"]["product_api"]
    retiring = [entry for path, entry in product.items()
                if path.startswith(("/api/workflows", "/api/experimental/speculative", "/api/cluster/config/task-graph"))]
    assert retiring and all(e["disposition"] == "retire" and e["phase"] == "R5" for e in retiring)
    assert product["/api/cluster/queue/strategy"]["disposition"] == "retire"
    assert ENTRIES["http_surfaces"]["native_service"]["/v1/speculative/run"]["disposition"] == "retire"


def test_product_snapshot_does_not_freeze_private_handler_names():
    assert "operationId" in SNAPSHOT["ignored_presentation_keys"]
    assert SNAPSHOT["baseline_revision"] == ENTRIES["baseline_revision"]
    assert len(SNAPSHOT["baseline_revision"]) == 40


def test_schema_normalization_keeps_user_fields_and_literal_defaults():
    value = {
        "title": "Cosmetic Schema Name",
        "properties": {
            "title": {"type": "string", "title": "Cosmetic Field Name"},
            "description": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "default": {"title": "user title", "description": "user data"},
    }
    normalized = semantic(value)
    assert normalized["properties"] == {
        "title": {"type": "string"}, "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    }
    assert "title" not in normalized
    assert normalized["default"] == value["default"]
