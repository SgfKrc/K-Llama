"""A1 relay（legacy 探针）的**隔离边界**。

背景（`docs/archive/misc/DIST后续核心链路缺陷审计-2026-10-07.md` 的设计债务一节）
--------------------------------------------------------------------
A1 relay 已从产品调度入口剔除：`PIPELINE_RELAY_PROBE_ONLY` 默认 1 ⇒
`scheduler_pipeline._relay_segment_for_worker()` 直接返回 `None`，生产请求走 A3
`stage_offer_v3`。但 A1 的 relay session、`relay_middle` assignment、段映射缓存与
peer 缓存仍散落主仓，并继续出现在 readiness、角色与配置分支里。

本模块是它的**唯一入口与诊断来源**（DIST-NEXT-7）：

* [a1_production_enabled] —— 「A1 是否允许进入生产调度」的唯一判据；生产供给只经它。
* [a1_isolation_status] —— 独立诊断 namespace（`relay_a1.*`），把开关、配置、
  角色计数与拒绝原因集中在一处，不混进 Route A 的原因码。
* [parse_relay_segment_map] —— `QLH_RELAY_SEGMENTS` 的解析（自
  `scheduler_pipeline` 迁入；判据仍复用 `_normalize_relay_segment`，单一真源）。

**默认不可被 scheduler assignment 选中**：`a1_production_enabled()` 默认 `False`，
本模块也不导出任何「给生产 assignment 用」的构造函数 —— 需要 A1 行为的只有探针入口
（`QLH_RELAY_PROBE_ONLY=0` + `QLH_RELAY_ENABLED=1`）与 relay 相关单测。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger("scheduler")

#: 独立诊断 namespace —— Route A 的原因码不带该前缀，便于区分「谁拒绝的」。
DIAGNOSTIC_NAMESPACE = "relay_a1"

#: 恢复 A1 行为所需的环境开关（探针入口，写在这里作为唯一说明）。
PROBE_ENTRY_ENV = "QLH_RELAY_PROBE_ONLY=0 + QLH_RELAY_ENABLED=1"


def a1_production_enabled(*, probe_only: bool, relay_enabled: bool) -> bool:
    """A1 是否允许进入**生产调度**（唯一判据）。

    默认（`probe_only=True`）恒 `False`：生产请求继续走 A3，不做静默切换。
    探针/实验必须同时显式给出 `QLH_RELAY_PROBE_ONLY=0` 与 `QLH_RELAY_ENABLED=1`。
    """
    return bool(relay_enabled) and not bool(probe_only)


def a1_isolation_status(
    *,
    probe_only: bool,
    relay_enabled: bool,
    segments_raw: str = "",
    configured_nodes: Optional[Mapping[str, Any]] = None,
    relay_host_count: int = 0,
) -> dict[str, Any]:
    """A1 的独立诊断快照（`relay_a1` namespace）。

    `production_enabled` 与 `assignment_selectable` 同源取值 —— 调用方（health /
    日志 / 测试）据此判断「A1 现在能不能被调度选中」，而不是逐个开关去猜。
    """
    production_enabled = a1_production_enabled(
        probe_only=probe_only, relay_enabled=relay_enabled,
    )
    return {
        "namespace": DIAGNOSTIC_NAMESPACE,
        "production_enabled": production_enabled,
        "assignment_selectable": production_enabled,
        "probe_only": bool(probe_only),
        "relay_enabled": bool(relay_enabled),
        "segments_configured": bool(str(segments_raw or "").strip()),
        "configured_node_count": len(dict(configured_nodes or {})),
        "relay_host_count": int(relay_host_count),
        "probe_entry": PROBE_ENTRY_ENV,
    }


def log_isolation_status(status: Mapping[str, Any]) -> None:
    """启动期把隔离状态打一条具名日志（独立 namespace，可 grep）。"""
    logger.info(
        "event=relay_a1_isolation production_enabled=%s probe_only=%s "
        "relay_enabled=%s segments_configured=%s configured_node_count=%d "
        "probe_entry=%s",
        status.get("production_enabled"),
        status.get("probe_only"),
        status.get("relay_enabled"),
        status.get("segments_configured"),
        int(status.get("configured_node_count") or 0),
        status.get("probe_entry"),
    )


def parse_relay_segment_map(
    raw: str,
    normalize: Callable[[Mapping[str, Any]], dict[str, object]],
) -> dict[str, dict[str, object]]:
    """解析 `QLH_RELAY_SEGMENTS`（条目 `node=role@host:port#n_embd#start-end`）。

    条目格式与校验**完全**沿用迁移前的实现：`;`/`,` 分隔，任何不合法、或未声明层区间
    的条目**整条丢弃**（宁可不下发，也不下发"只带 n_embd"的半懂规格 —— 后者会让主节点
    无从扣除该段的层）。空配置 ⇒ `{}`（行为与接线前一致）。

    `normalize` 由调用方注入（`SchedulerPipelineMixin._normalize_relay_segment`），
    避免本模块反向依赖 scheduler 层。
    """
    result: dict[str, dict[str, object]] = {}
    for chunk in str(raw or "").replace(",", ";").split(";"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk or "@" not in chunk:
            continue
        name, _, value = chunk.partition("=")
        body, _, fields_text = value.partition("#")
        role, _, host_port = body.partition("@")
        host, _, port_text = host_port.rpartition(":")
        fields = fields_text.split("#")
        if len(fields) != 2:
            continue        # 缺层区间（或多余字段）⇒ 整条丢弃
        n_embd_text, range_text = fields
        range_start, _, range_end = range_text.partition("-")
        try:
            spec = normalize({
                "role": role.strip(),
                "host": host.strip(),
                "port": int(port_text),
                "n_embd": int(n_embd_text),
                "layer_start": int(range_start),
                "layer_end": int(range_end),
            })
        except ValueError:
            continue
        if name.strip() and spec is not None:
            result[name.strip()] = spec
    return result
