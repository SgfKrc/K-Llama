from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]

# 证据工件只存在于本机（`local_docs/evidence/`，不入库）；缺它按约定跳过。
_EVIDENCE = ROOT / "local_docs" / "evidence"
pytestmark = pytest.mark.skipif(
    not (_EVIDENCE / "torch-hetero-plan").exists() or not (_EVIDENCE / "torch-op-profile").exists(),
    reason="缺本机证据工件 local_docs/evidence/{torch-hetero-plan,torch-op-profile}",
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.torch_operator_registry import (  # noqa: E402
    DEFAULT_IMPLEMENTATIONS,
    DEFAULT_TORCH_OPERATOR_REGISTRY,
    ImplementationKind,
    ImplementationSpec,
    LOGICAL_OPERATORS,
    NumericalEvidence,
    OperatorRegistry,
    SelectionPolicy,
)
from src.torch_hetero_plan import (  # noqa: E402
    INPUT_SCHEMA,
    SCHEDULE_MODEL,
    ContinuousLayerBaseline,
    DeviceResource,
    HeterogeneousPlanInput,
    LayerFitEvidence,
    LinkProfile,
    MeasuredOperatorCost,
    OperatorNode,
    load_operator_profile_evidence,
    load_layer_profile_parts,
    TensorEdge,
    main,
    plan_operator_placement,
)


MODEL_FP = "model:sha256:abc"
WORKLOAD_FP = "workload:decode:8xshape-a"


def _scenario(
    *,
    cpu_memory: int = 1000,
    cuda_memory: int = 60,
    links: tuple[LinkProfile, ...] | None = None,
    costs: tuple[MeasuredOperatorCost, ...] | None = None,
    max_search_states: int = 100_000,
) -> HeterogeneousPlanInput:
    nodes = tuple(
        OperatorNode(
            node_id=f"layer-{index}",
            operator_id="linear_projection",
            layer_index=index,
            dtype="float32",
            shape_fingerprint=f"shape-{index}",
            resident_bytes=10,
            workspace_bytes=1,
        )
        for index in range(8)
    )
    edges = tuple(
        TensorEdge(f"layer-{index}", f"layer-{index + 1}", 8)
        for index in range(7)
    )
    devices = (
        DeviceResource("cpu", "cpu", cpu_memory, frozenset({"cpu"}),
                       memory_safety_margin=1.0, profile_ref="device/cpu.json"),
        DeviceResource("cuda", "cuda", cuda_memory, frozenset({"cuda"}),
                       memory_safety_margin=1.0, profile_ref="device/cuda.json"),
    )
    if links is None:
        links = (
            LinkProfile("cpu", "cuda", 1_000_000_000, 0.02, 3, True, "link/cpu-cuda.json"),
            LinkProfile("cuda", "cpu", 1_000_000_000, 0.02, 3, True, "link/cuda-cpu.json"),
        )
    if costs is None:
        entries = []
        for node in nodes:
            for device_id, values in (
                ("cpu", (2.8, 3.0, 3.2)),
                ("cuda", (0.8, 1.0, 1.2)),
            ):
                device_type = "cpu" if device_id == "cpu" else "cuda"
                entries.append(MeasuredOperatorCost(
                    node_id=node.node_id,
                    device_id=device_id,
                    operator_id=node.operator_id,
                    implementation_id=f"pytorch.eager.{device_type}.linear_projection",
                    phase="decode",
                    dtype=node.dtype,
                    shape_fingerprint=node.shape_fingerprint,
                    model_fingerprint=MODEL_FP,
                    workload_fingerprint=WORKLOAD_FP,
                    samples_ms=values,
                    warmup_consistent=True,
                    instrumented=False,
                    source_ref=f"costs/{node.node_id}-{device_id}.json",
                ))
        costs = tuple(entries)
    layer_bytes = (10,) * len(nodes)
    baseline = ContinuousLayerBaseline(
        device_order=("cpu", "cuda"),
        layer_fits=(
            LayerFitEvidence(
                "cpu", MODEL_FP, WORKLOAD_FP, "decode", 0.1, 3.0,
                layer_bytes, 5, 0.99, "fits/cpu.json",
            ),
            LayerFitEvidence(
                "cuda", MODEL_FP, WORKLOAD_FP, "decode", 0.1, 1.0,
                layer_bytes, 5, 0.99, "fits/cuda.json",
            ),
        ),
        total_layers=len(nodes),
        hidden_size=2,
        hidden_dtype="float32",
        prefill_tokens=8,
        decode_tokens=1,
        minimum_layers_per_segment=1,
    )
    return HeterogeneousPlanInput(
        model_fingerprint=MODEL_FP,
        workload_fingerprint=WORKLOAD_FP,
        model_parameter_count=500_000_000,
        phase="decode",
        nodes=nodes,
        edges=edges,
        devices=devices,
        links=links,
        costs=costs,
        numerical_evidence={},
        continuous_baseline=baseline,
        max_search_states=max_search_states,
    )


def test_exact_operator_plan_compares_with_existing_contiguous_layer_objective():
    result = plan_operator_placement(_scenario())

    assert result.admitted
    assert result.reason == "best_exact_feasible_assignment"
    assert result.total_ms is not None
    assert result.transfer_work_ms > 0
    assert result.schedule_model == SCHEDULE_MODEL
    assert {item.device_id for item in result.placements} == {"cpu", "cuda"}
    comparison = result.baseline_comparison
    assert comparison["admitted"]
    assert comparison["reason"] == "compared_same_model_workload_phase"
    assert comparison["cuts"] == [3]
    assert comparison["speedup_vs_contiguous"] > 1.0
    assert comparison["resource_adjustment"] == {
        "peak_workspace_bytes": 1,
        "peak_layer_boundary_bytes": 8,
        "effective_capacity_by_device": {"cpu": 991, "cuda": 51},
        "safety_margin": 1.0,
    }


def test_memory_accounting_includes_weights_workspace_and_boundary_buffers():
    result = plan_operator_placement(_scenario())
    assert result.admitted
    cuda = result.memory_by_device["cuda"]
    assert cuda["resident_bytes"] == 50
    assert cuda["workspace_peak_bytes"] == 1
    assert cuda["boundary_buffer_bytes_upper_bound"] == 8
    assert cuda["required_with_margin_bytes"] == 59


def test_independent_dag_branches_overlap_across_device_queues():
    request = _scenario()
    result = plan_operator_placement(replace(request, edges=()))

    assert result.admitted
    assert result.compute_work_ms == 14.0
    assert result.transfer_work_ms == 0.0
    assert result.total_ms == 9.0
    assert result.total_ms < result.compute_work_ms + result.transfer_work_ms


def test_continuous_baseline_reserves_transient_memory_before_safety_margin():
    request = _scenario()
    devices = tuple(replace(device, memory_safety_margin=1.2) for device in request.devices)
    result = plan_operator_placement(replace(request, devices=devices))

    assert result.admitted
    comparison = result.baseline_comparison
    assert comparison["admitted"]
    assert comparison["cuts"] == [4]
    assert comparison["resource_adjustment"]["effective_capacity_by_device"] == {
        "cpu": 989,
        "cuda": 49,
    }
    assert result.memory_by_device["cuda"]["required_with_margin_bytes"] <= 60


def test_no_assignment_is_admitted_when_every_partition_exceeds_memory():
    result = plan_operator_placement(_scenario(cpu_memory=40, cuda_memory=40))

    assert not result.admitted
    assert result.reason == "no_feasible_assignment"


def test_missing_cross_device_link_fails_closed_when_memory_requires_a_split():
    result = plan_operator_placement(_scenario(
        cpu_memory=40, cuda_memory=40, links=(),
    ))

    assert not result.admitted
    assert result.reason == "no_feasible_assignment"


def test_instrumented_operator_costs_are_not_used_for_latency_placement():
    request = _scenario()
    costs = tuple(replace(item, instrumented=True) for item in request.costs)
    result = plan_operator_placement(replace(request, costs=costs))

    assert not result.admitted
    assert result.reason == "no_matched_eligible_cost:layer-0"
    assert any("instrumented cost" in item for item in result.rejected_candidates)


def test_mixed_workload_cost_rows_are_rejected_instead_of_combined():
    request = _scenario()
    costs = (replace(request.costs[0], workload_fingerprint="another-run"), *request.costs[1:])

    result = plan_operator_placement(replace(request, costs=costs))

    assert not result.admitted
    assert result.reason == "mixed_operator_cost_experiment_identity"


def test_exact_search_limit_rejects_instead_of_returning_partial_best():
    result = plan_operator_placement(_scenario(max_search_states=1))

    assert not result.admitted
    assert result.reason == "exact_search_state_limit_exceeded"


def test_experimental_implementation_requires_registry_policy_and_matching_evidence():
    request = _scenario()
    experimental_id = "pytorch.experimental.fast_linear"
    spec = ImplementationSpec(
        experimental_id,
        "linear_projection",
        ImplementationKind.EXPERIMENTAL,
        frozenset({"cuda"}),
        frozenset({"float32"}),
        frozenset({"decode"}),
        required_capabilities=frozenset({"cuda"}),
        feature_gate="fast_linear_experiment",
        priority=50,
    )
    registry = OperatorRegistry(LOGICAL_OPERATORS, (*DEFAULT_IMPLEMENTATIONS, spec))
    devices = tuple(
        replace(
            device,
            enabled_feature_gates=frozenset({"fast_linear_experiment"}),
        ) if device.device_id == "cuda" else device
        for device in request.devices
    )
    extra_costs = tuple(MeasuredOperatorCost(
        node_id=node.node_id,
        device_id="cuda",
        operator_id=node.operator_id,
        implementation_id=experimental_id,
        phase=request.phase,
        dtype=node.dtype,
        shape_fingerprint=node.shape_fingerprint,
        model_fingerprint=MODEL_FP,
        workload_fingerprint=WORKLOAD_FP,
        samples_ms=(0.18, 0.2, 0.22),
        warmup_consistent=True,
        instrumented=False,
        source_ref=f"costs/{node.node_id}-experimental.json",
    ) for node in request.nodes)
    request = replace(request, devices=devices, costs=(*request.costs, *extra_costs))
    policy = SelectionPolicy(allow_experimental=True)
    denied = plan_operator_placement(request, registry=registry, policy=policy)
    evidence = NumericalEvidence(
        MODEL_FP, WORKLOAD_FP, "evidence/fast-linear.json", True, True,
        0.0, 0.0, 8,
    )
    admitted = plan_operator_placement(
        replace(request, numerical_evidence={experimental_id: evidence}),
        registry=registry,
        policy=policy,
    )

    assert denied.admitted and all(
        item.implementation_id != experimental_id for item in denied.placements
    )
    assert admitted.admitted and any(
        item.implementation_id == experimental_id for item in admitted.placements
    )


def test_device_resource_reads_available_ram_and_selected_free_vram():
    cpu = DeviceResource.from_device_profile(
        "cpu", "cpu", {"ram": {"available_gb": 2.0}}, profile_ref="cpu.json",
    )
    cuda = DeviceResource.from_device_profile(
        "cuda", "cuda",
        {
            "selected_gpu_index": 1,
            "gpu": {"index": 0, "cuda_available": True, "vram_free_gb": 7.0},
            "gpus": [
                {"index": 0, "cuda_available": True, "vram_free_gb": 1.0},
                {"index": 1, "cuda_available": True, "vram_free_gb": 4.0},
            ],
        },
        capabilities=frozenset({"cuda"}),
        profile_ref="cuda.json",
    )

    assert cpu.available_memory_bytes == 2 * 1024 ** 3
    assert cuda.available_memory_bytes == 4 * 1024 ** 3


def _scenario_payload(request: HeterogeneousPlanInput) -> dict:
    return {
        "schema_version": INPUT_SCHEMA,
        "request": {
            "model_fingerprint": request.model_fingerprint,
            "workload_fingerprint": request.workload_fingerprint,
            "model_parameter_count": request.model_parameter_count,
            "phase": request.phase,
            "max_search_states": request.max_search_states,
        },
        "nodes": [asdict(item) for item in request.nodes],
        "edges": [asdict(item) for item in request.edges],
        "devices": [
            {
                **asdict(item),
                "capabilities": sorted(item.capabilities),
                "enabled_feature_gates": sorted(item.enabled_feature_gates),
            }
            for item in request.devices
        ],
        "links": [asdict(item) for item in request.links],
        "costs": [asdict(item) for item in request.costs],
        "numerical_evidence": [],
        "continuous_baseline": {
            **asdict(request.continuous_baseline),
            "device_order": list(request.continuous_baseline.device_order),
            "layer_fits": [asdict(item) for item in request.continuous_baseline.layer_fits],
        },
    }


def test_json_cli_writes_reproducible_plan_report(tmp_path: Path):
    input_path = tmp_path / "scenario.json"
    output_path = tmp_path / "plan" / "report.json"
    input_path.write_text(json.dumps(_scenario_payload(_scenario())), encoding="utf-8")

    code = main(["--input", str(input_path), "--json-out", str(output_path)])
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert code == 0
    assert report["schema_version"] == "qlh.torch_hetero_plan_report.v1"
    assert report["admitted"] is True
    assert report["baseline_comparison"]["admitted"] is True


def test_cli_rejects_unknown_scenario_schema(tmp_path: Path):
    input_path = tmp_path / "scenario.json"
    output_path = tmp_path / "report.json"
    input_path.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")

    assert main(["--input", str(input_path), "--json-out", str(output_path)]) == 2
    assert not output_path.exists()


def test_cli_rejects_duplicate_numerical_evidence(tmp_path: Path):
    payload = _scenario_payload(_scenario())
    evidence = {
        "implementation_id": "candidate",
        "evidence": {
            "model_fingerprint": MODEL_FP,
            "workload_fingerprint": WORKLOAD_FP,
            "artifact_ref": "evidence/candidate.json",
            "passed": True,
            "argmax_exact": True,
            "max_abs_error": 0.0,
            "max_relative_error": 0.0,
            "sample_count": 3,
        },
    }
    payload["numerical_evidence"] = [evidence, evidence]
    input_path = tmp_path / "scenario.json"
    output_path = tmp_path / "report.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["--input", str(input_path), "--json-out", str(output_path)]) == 2
    assert not output_path.exists()


def test_cli_refuses_to_overwrite_its_scenario_input(tmp_path: Path):
    input_path = tmp_path / "scenario.json"
    payload = json.dumps(_scenario_payload(_scenario()))
    input_path.write_text(payload, encoding="utf-8")

    assert main(["--input", str(input_path), "--json-out", str(input_path)]) == 2
    assert input_path.read_text(encoding="utf-8") == payload


def test_default_registry_remains_runtime_and_edge_dependency_free():
    import ast

    source = (ROOT / "src" / "torch_hetero_plan.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    torch_imports = [
        node for node in tree.body
        if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names)
        or isinstance(node, ast.ImportFrom) and node.module == "torch"
    ]

    assert torch_imports == []
    assert DEFAULT_TORCH_OPERATOR_REGISTRY.catalog()["schema_version"] == (
        "qlh.torch_operator_registry.v1"
    )


@pytest.mark.parametrize(
    ("phase", "operator_count", "dispatch_count"),
    (("prefill", 97, 1367), ("decode", 250, 10456)),
)
def test_real_operator_profile_is_validated_without_treating_profiler_time_as_cost(
    phase: str, operator_count: int, dispatch_count: int,
):
    profile_path = (
        ROOT / "local_docs" / "evidence" / "torch-op-profile"
        / "2026-09-23-qwen25-05b-rtx4060-cuda-steady.json"
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    evidence = load_operator_profile_evidence(
        profile,
        phase=phase,
        device_id="cuda",
        profile_ref=str(profile_path),
        model_fingerprint="model:sha256:qwen25-05b",
        workload_fingerprint=f"workload:{phase}:same-profile",
    )

    assert evidence.operator_count == operator_count
    assert evidence.dispatch_call_count == dispatch_count
    assert evidence.unmatched_dispatch_calls == 0
    assert len(evidence.wall_samples_ms) == 5
    assert evidence.profiler_costs_usable is False
    assert evidence.logical_operator_counts["linear_projection"] > 0
    assert evidence.logical_operator_counts["attention_core"] > 0
    assert evidence.unmapped_profile_operators


def test_real_layer_profile_parts_preserve_identity_and_cost_gates():
    for device_id, report_name in (
        ("cpu", "cpu-layer-profile-smoke-20260924.json"),
        ("cuda", "cuda-layer-profile-smoke-20260924.json"),
    ):
        profile_path = ROOT / "local_docs" / "evidence" / "torch-hetero-plan" / report_name
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        nodes, edges, costs = load_layer_profile_parts(
            profile,
            phase="prefill",
            device_id=device_id,
            profile_ref=str(profile_path),
        )

        assert len(nodes) == 2
        assert len(edges) == 1
        assert {item.node_id for item in costs} == {item.node_id for item in nodes}
        assert all(item.instrumented is False for item in costs)
        assert all(isinstance(item.warmup_consistent, bool) for item in costs)
        assert all(item.model_fingerprint == profile["model_fingerprint"] for item in costs)


def test_layer_profile_adapter_rejects_duplicate_or_mismatched_cost_rows():
    profile_path = ROOT / "local_docs" / "evidence" / "torch-hetero-plan" / "cpu-layer-profile-smoke-20260924.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    costs = profile["phase_profiles"]["prefill"]["costs"]

    duplicate = json.loads(json.dumps(profile))
    duplicate["phase_profiles"]["prefill"]["costs"].append(dict(costs[0]))
    with pytest.raises(ValueError, match="duplicate cost node ids"):
        load_layer_profile_parts(
            duplicate, phase="prefill", device_id="cpu", profile_ref=str(profile_path),
        )

    mismatch = json.loads(json.dumps(profile))
    mismatch["phase_profiles"]["prefill"]["costs"][0]["shape_fingerprint"] = "shape:wrong"
    with pytest.raises(ValueError, match="shape_fingerprint does not match node"):
        load_layer_profile_parts(
            mismatch, phase="prefill", device_id="cpu", profile_ref=str(profile_path),
        )


def test_full_real_layer_matrix_has_matched_identity_and_phase_edges():
    root = ROOT / "local_docs" / "evidence" / "torch-hetero-plan"
    reports = {
        device: json.loads(
            (root / f"{device}-layer-profile-fp32-full-20260924.json").read_text(
                encoding="utf-8",
            )
        )
        for device in ("cpu", "cuda")
    }
    assert reports["cpu"]["model_fingerprint"] == reports["cuda"]["model_fingerprint"]
    assert reports["cpu"]["workload_fingerprint"] == reports["cuda"]["workload_fingerprint"]
    for phase in ("prefill", "decode"):
        cpu_nodes, cpu_edges, cpu_costs = load_layer_profile_parts(
            reports["cpu"], phase=phase, device_id="cpu", profile_ref=str(root / "cpu"),
        )
        cuda_nodes, cuda_edges, cuda_costs = load_layer_profile_parts(
            reports["cuda"], phase=phase, device_id="cuda", profile_ref=str(root / "cuda"),
        )
        assert len(cpu_nodes) == len(cuda_nodes) == 24
        assert tuple(node.shape_fingerprint for node in cpu_nodes) == tuple(
            node.shape_fingerprint for node in cuda_nodes
        )
        assert len(cpu_edges) == len(cuda_edges) == 23
        expected_tensor_bytes = (64 if phase == "prefill" else 1) * 896 * 4
        assert cpu_edges[0].tensor_bytes == expected_tensor_bytes
        assert cuda_edges[0].tensor_bytes == expected_tensor_bytes
        assert len(cpu_costs) == len(cuda_costs) == 24
