"""离线复现：同机 CPU 段为什么没被求解器选中。

不需要集群。用 `tests/test_pipeline_capacity_freshness.py` 的 descriptor 构造器，
配上生产日志里观察到的真实节点记录形态（四个执行器），直接把求解器跑起来看它选什么。

用法：python scripts/probe_local_executor_placement.py
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _d in (ROOT, ROOT / "src", ROOT / "tests"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from qlh_core.planning.capacity import solve_pipeline_capacity  # noqa: E402
from tests.test_model_supply import descriptor  # noqa: E402

TOTAL_LAYERS = 32


def build_descriptor(layers=TOTAL_LAYERS):
    model = descriptor(layers=layers, layer_bytes=100, components=0)
    model["component_weight_bytes"]["lm_head"] = 400
    return model


def node(node_id, *, executor_id="", capacity, layer_ranges, engine="llama_cpp",
         cost_ms=None, segment_mode=None, candidate=None, domain_total=None,
         domain_allocatable=None, score=1.0, host_capacity=None, local_cut=False):
    record = {
        "node_id": node_id,
        "role": "master" if node_id == "master" else "client",
        "capacity_bytes": capacity,
        "reserve_bytes": 0,
        "runtime_multiplier": 1.0,
        "execution_device": "cpu",
        "capacity_source": "probe",
        "engine": engine,
        "score": score,
        "layer_ranges": layer_ranges,
    }
    # ⚠️ 关键：判据是 `if advertised_budget is None or not budget["local_cut"]` **才**检查
    #    `layer_ranges`。而 `layer_ranges=[[0,32]]` 的语义是**精确匹配** `(cursor, end)`
    #    （`start == cursor and end == allowed_end`），**不是**"覆盖范围" ——
    #    没有 `local_cut` 的节点只能拿整段 `[0,32)`，无法分担部分层。
    #    生产日志里 rog/android 正是靠 `local_cut=True` 才拿到 `[8,24)` 这种部分区间。
    if local_cut:
        record["layer_budget"] = {
            "available_bytes": int(capacity),
            "per_layer_bytes": 100,
            "max_layers": max(1, int(capacity) // 100),
            "local_cut": True,
        }
    if executor_id:
        record["executor_id"] = executor_id
    if cost_ms is not None:
        record["layer_cost_ms"] = cost_ms
    if segment_mode:
        record["segment_mode"] = segment_mode
    if candidate is not None:
        record["candidate_layer_ranges"] = candidate
    if domain_total is not None:
        record["domain_total_bytes"] = domain_total
    if domain_allocatable is not None:
        record["domain_allocatable_bytes"] = domain_allocatable
    if host_capacity is not None:
        record["host_capacity_bytes"] = host_capacity
    return record


def scenario(label, nodes, *, prefer_all_workers=True, total_layers=TOTAL_LAYERS,
             local_layer_budget=None):
    model = build_descriptor(total_layers)
    # 先看求解器眼里的可分配集合 —— CPU 段"完全没进候选"时，光看结果无从判断
    # 它是在 `_normalize_nodes` 被排除，还是进了 `usable` 却在 DFS 里被静默跳过。
    from qlh_core.planning.capacity import _normalize_nodes

    usable, excluded = _normalize_nodes(deepcopy(nodes), total_layers=total_layers)
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    print("  usable:")
    for item in usable:
        print(f"    {item['node_id']:<18} executor={str(item.get('executor_id') or ''):<6} "
              f"ranges={item.get('layer_ranges')} mode={item.get('segment_mode')} "
              f"cap={item.get('capacity_bytes')} budget={'Y' if item.get('layer_budget') else 'N'}")
    if excluded:
        print(f"  excluded(normalize): {[e.get('node_id') for e in excluded]}")

    result = solve_pipeline_capacity(
        model, deepcopy(nodes),
        safety_margin=1.0,
        require_distributed=True,
        local_layer_budget=local_layer_budget,
        prefer_all_workers=prefer_all_workers,
    )
    print(f"  admitted        = {result.get('admitted')}")
    print(f"  reason_code     = {result.get('reason_code')}")
    for item in result.get("assignments", []):
        print(f"    {item.get('node_id'):<18} executor={item.get('executor_id', ''):<6} "
              f"[{item.get('start_layer')},{item.get('end_layer')})")
    excl = result.get("candidate_exclusions") or []
    if excl:
        # 按 (node_id, reason) 汇总，避免刷屏
        counts: dict[tuple, int] = {}
        for item in excl:
            key = (item.get("node_id"), item.get("executor_id", ""), item.get("reason_code"))
            counts[key] = counts.get(key, 0) + 1
        print("  candidate_exclusions:")
        for (nid, ex, reason), count in sorted(counts.items()):
            print(f"    {nid:<18} executor={ex:<6} {reason}  x{count}")
    return result


def main() -> int:
    # 真实形态：master(GPU, [0,8)) + master/cpu([8,24) middle) + rog([0,32)) + android([24,32))
    # 容量按生产日志量级：host 总账 ~14.9 GB，CPU 段拿 70%。
    GB = 1024 ** 3
    host_total = int(14.9 * GB)
    cpu_share = int(host_total * 0.7)
    primary_alloc = host_total - cpu_share

    base_nodes = [
        # 主段：生产里它从段工件 manifest 拿到 mode=head（`_configured_layer_artifact`）。
        node("master", capacity=int(6.0 * GB), layer_ranges=[[0, 8]],
             engine="pytorch", host_capacity=host_total, segment_mode="head",
             domain_total=host_total, domain_allocatable=primary_alloc,
             candidate=[[0, 8]], score=1.0),
        node("master", executor_id="cpu", capacity=cpu_share, layer_ranges=[[8, 24]],
             engine="llama_cpp", cost_ms=6.84, segment_mode="middle",
             domain_total=host_total, domain_allocatable=cpu_share, score=1.0),
        # 两个远端都是 tail（生产日志 segment_mode={'android':'tail','rog-5060':'tail'}）
        # ⇒ 只有一个能当末段，另一个必然闲置。这是既有拓扑的真实约束。
        node("rog-5060", capacity=int(4.3 * GB), layer_ranges=[[0, 32]],
             cost_ms=8.7, score=1.0, local_cut=True, segment_mode="tail"),
        node("android-542cd254", capacity=int(7.2 * GB), layer_ranges=[[24, 32]],
             cost_ms=20.1, score=1.0, local_cut=True, segment_mode="tail"),
    ]

    scenario("A) 四执行器，prefer_all_workers=True（fastest），total_layers=32",
             base_nodes, total_layers=32)

    scenario("B) 同上但 total_layers=8（小模型，看是否仍不选 CPU）",
             [n for n in base_nodes if n["node_id"] != "android-542cd254"],
             total_layers=32)

    scenario("C) 只留 master + master/cpu + rog（与 B 同节点，改用 32 层）",
             [n for n in base_nodes if n["node_id"] != "android-542cd254"],
             total_layers=32)

    scenario("D) 四执行器但 prefer_all_workers=False（balanced）",
             base_nodes, prefer_all_workers=False, total_layers=32)

    # E) 关键对照：把 CPU 段的层区间放宽到 [0,32)，看它是否被选
    wide = deepcopy(base_nodes)
    for item in wide:
        if item.get("executor_id") == "cpu":
            item["layer_ranges"] = [[0, 32]]
    scenario("E) CPU 段区间放宽为 [0,32)（middle 约束仍在），看是否入选",
             wide, total_layers=32)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
