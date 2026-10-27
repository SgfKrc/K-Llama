"""Fixed product schemas and explicit active/retired entrypoint inventories.

Cosmetic OpenAPI text and handler names are excluded. Changes require explicit
contract review; there is no update-from-HEAD path or automatic snapshot rewrite.
Static roles, machine authentication, model-mode, SSE and state behavior are
covered by staged suites.
"""
import ast
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.refactor_contract
ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
SNAPSHOT = json.loads((FIXTURES / "refactor_product_api_v2.json").read_text(encoding="utf-8"))
ENTRIES = json.loads((FIXTURES / "refactor_entrypoints_v2.json").read_text(encoding="utf-8"))


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
    product = ENTRIES["retired_http"]["product_api"]
    retiring = [entry for path, entry in product.items()
                if path.startswith(("/api/workflows", "/api/experimental/speculative", "/api/cluster/config/task-graph"))]
    assert retiring and all(e["disposition"] == "retire" and e["phase"] == "R5" for e in retiring)
    assert product["/api/cluster/queue/strategy"]["disposition"] == "retire"
    assert ENTRIES["retired_http"]["native_service"]["/v1/speculative/run"]["disposition"] == "retire"


R6_RETIRED_ROUTES = {
    "/api/cluster/join/request": ["post"],
    "/api/cluster/join/grant": ["post"],
    "/api/cluster/credentials/rotate": ["post"],
    "/api/cluster/join/consume": ["post"],
    "/api/cluster/control-plane/certificate": ["post"],
    "/api/cluster/quorum/voter": ["post"],
    "/api/cluster/management-score": ["get"],
    "/api/cluster/transfer-master": ["post"],
    "/api/cluster/transfer-logs": ["get"],
    "/api/cluster/spare-master": ["delete", "get", "post"],
    "/api/cluster/spare-master/logs": ["get"],
    "/api/cluster/review/create": ["post"],
    "/api/cluster/review/vote": ["post"],
    "/api/cluster/review/tickets": ["delete", "get"],
    "/api/cluster/review/tickets/{ticket_id}": ["delete", "get"],
    "/api/cluster/review/can-vote": ["get"],
    "/api/cluster/review/expire-check": ["post"],
    "/api/auth/capability": ["get"],
    "/api/auth/login": ["post"],
    "/api/auth/logout": ["post"],
    "/api/auth/me": ["get"],
    "/api/auth/totp/provision": ["post"],
    "/api/auth/totp/verify": ["post"],
    "/api/users": ["get", "post"],
    "/api/users/{username}": ["delete", "patch"],
}
R6_RETIRED_SCHEMAS = {
    "ClusterJoinRequestCreate",
    "ClusterJoinGrantIssue",
    "ClusterJoinConsume",
    "ClusterSecretRotateRequest",
    "ControlCertificateRequest",
    "LoginRequest",
    "CreateUserRequest",
    "PatchUserRequest",
    "TotpVerifyRequest",
    "TransferMasterRequest",
    "SpareMasterRequest",
    "CreateReviewRequest",
    "CastVoteRequest",
    "quorum-voter RPC request/response (inline)",
}


def test_r6_retired_routes_are_absent_and_explicitly_accounted(openapi):
    expected = {
        path: {"methods": methods, "disposition": "retire", "phase": "R6"}
        for path, methods in R6_RETIRED_ROUTES.items()
    }
    assert {path: ENTRIES["retired_http"]["product_api"][path] for path in expected} == expected
    assert {path: SNAPSHOT["retired_http"][path] for path in expected} == expected
    assert set(R6_RETIRED_ROUTES).isdisjoint(ENTRIES["http_surfaces"]["product_api"])
    assert set(R6_RETIRED_ROUTES).isdisjoint(openapi["paths"])


def test_r6_retired_schemas_and_user_ruling_are_explicit(openapi):
    retired = {entry["name"]: entry for entry in SNAPSHOT["retired_schemas"]}
    assert set(retired) == R6_RETIRED_SCHEMAS
    assert all(entry["disposition"] == "retire" and entry["phase"] == "R6" for entry in retired.values())
    assert retired["quorum-voter RPC request/response (inline)"]["openapi_components"] == []
    assert R6_RETIRED_SCHEMAS.isdisjoint(openapi["components"]["schemas"])

    for inventory in (ENTRIES, SNAPSHOT):
        ruling = inventory["retirement_decision"]
        assert ruling["id"] == "R6-static-role-account-api-retirement"
        assert ruling["date"] == "2026-10-27"
        assert ruling["recorded_date"] == "2026-10-28"
        assert "静态主节点部署与认证退役决策-2026-10-27.md" in ruling["source"]
        assert "TCP HMAC node authentication" in ruling["preserve"]
        assert "bootstrap credential exchange" in ruling["preserve"]
        assert "local user settings and chat sessions" in ruling["preserve"]


def test_retained_bootstrap_machine_identity_and_local_settings_remain_active(openapi):
    active = ENTRIES["http_surfaces"]["product_api"]
    assert active["/api/bootstrap/first-connect"]["methods"] == ["post"]
    assert active["/api/bootstrap/info"]["methods"] == ["get"]
    assert active["/api/user/settings"]["methods"] == ["get", "put"]
    assert "/api/cluster/nodes/register" in openapi["paths"]


def test_refactor_manifests_load_and_use_static_role_regressions():
    from scripts import run_refactor_contracts

    stages = {
        "contract": run_refactor_contracts.suite_targets(),
        "bootstrap": run_refactor_contracts.bootstrap_targets(),
        "catalog": run_refactor_contracts.catalog_targets(),
        "deployment": run_refactor_contracts.deployment_targets(),
        "execution": run_refactor_contracts.execution_targets(),
        "client": run_refactor_contracts.client_targets(),
        "product": run_refactor_contracts.product_targets(),
    }
    assert {
        "tests/test_static_node_roles.py",
        "tests/test_static_role_config.py",
        "tests/test_static_cluster_api.py",
    }.issubset(stages["contract"])
    assert "tests/test_bootstrap_credentials.py" in stages["bootstrap"]
    assert "tests/test_api_security_boundaries.py" in stages["contract"]
    assert "tests/test_tui_auth_ui.py" in stages["client"]
    assert "tests/test_tui_sse_auth.py" in stages["client"]

    active = set().union(*(set(targets) for targets in stages.values()))
    retired = set()
    for path in FIXTURES.glob("refactor_*suite_v1.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if "retirement_decision" not in manifest:
            continue
        assert manifest["retirement_decision"]["date"] == "2026-10-27"
        assert manifest["retirement_decision"]["recorded_date"] == "2026-10-28"
        retired.update(entry["target"] for entry in manifest["retired_targets"])
    assert retired.isdisjoint(active)
    assert "tests/test_api_cluster_join.py" in retired


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
