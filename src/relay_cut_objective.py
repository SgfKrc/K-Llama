"""P2：跨框架接力的切点求解（容量 / 延迟 / 风险 三输出）。

票面（`docs/跨框架接力-当前有效基线与后续优化计划-2026-09-21.md` §4 P2）要求：
按候选节点的**单位层耗时、可用内存、网络带宽、首 token 延迟和稳定性**求解切点，
并同时输出 `capacity_feasible`、`latency_estimate`、`risk_penalty`。

与既有模块的分工（不重复造轮子）：

* `src/pipeline_capacity.solve_pipeline_capacity` —— 只做**容量**：连续区间放置、含
  embedding/lm_head 归属、safety margin、节点 score 作次级目标；
* `src/relay_planner.plan_relay_cut` —— 只做 **L→L 两段**的保守切点与拒绝理由；
* **本模块** —— 把容量、延迟与风险放进**同一个目标函数**，支持 **n 段**，
  并强制**合法切点约束**（如 Qwen3.5 的切点必须是 `full_attention_interval` 的整倍数，
  否则裁层 GGUF 的层类型错位、无法加载）。

段画像（`SegmentProfile`）既能来自 `DeviceProfiler.to_dict()`，也能**从实测拟合**
（`fit_segment_profile` / `fit_two_segment`）—— 后者是本票「切点重搜」的判据来源：
先在实测点上回归出每段"固定开销 + 每层耗时"，再用求解器预测最优切点，最后与实测最优对比。

本模块 stdlib-only，可在任何环境被测试（不需要模型/GPU）。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.relay_contract import CUT_LAYER_MIN, RelayHiddenSpec

SCHEMA_VERSION = "qlh.relay_cut_plan.v1"
DEFAULT_SAFETY_MARGIN = 1.2
#: 目标函数权重：容量收益、速度（相对单段）、风险惩罚。
DEFAULT_WEIGHTS = {"capacity": 1.0, "latency": 1.0, "risk": 1.0}
#: 风险项阈值（低于/高于该值开始计罚）。
RISK_BANDWIDTH_MBPS_FLOOR = 5.0
RISK_RTT_MS_CEIL = 50.0
RISK_THERMAL_PENALTY = 0.25
RISK_ARTIFACT_PENALTY = 0.5


def rtt_ms_from_heartbeat(
    samples: Sequence[float],
    *,
    percentile: float = 0.5,
) -> float:
    """★ 2026-10-14（22c 缺口①）：把**在线心跳 RTT 样本**折算成 `SegmentProfile.rtt_ms`。

    动机：切点求解的往返税项是 `STAGE_ROUND_TRIPS * rtt_ms`（`docs/已知问题记录.md` #71 判据
    `n·c_master > 2·RTT + n·c_D` 的核心），但此前 `rtt_ms` 只有两个来源 —— 人工常量
    `MEASURED_*_RTT_MS` 与手工 `--rtt-ms`。而**集群在线 RTT 本就有独立链路**：
    `src/tcp_comm.py:2134` 的 `avg_rtt_ms`（EWMA，`:2736` `avg = 0.9*avg + 0.1*rtt`）、
    `src/network_path.py` 的 `HeartbeatQualityWindow`（`rtt_ms_p50/p95/jitter`）、
    `src/scheduler.py:1687-1690` 落库 —— **此前没有任何地方把它们接进切点求解**
    （`grep avg_rtt_ms` 在本模块零命中）。

    本函数只做「样本 → 数值」的**纯计算**，**不 import 任何网络模块** ⇒ 保持本模块
    **不在产品执行路径上**的干净性（它仍是纯函数库，由调用方喂数据）。

    ⚠️ `#71` 要求「**RTT 必须按段取**」（PC↔PC=3ms、PC↔Y700=19~43ms），所以每个段应各自
    喂自己的样本，不要共用一条全局中位。

    口径：线性插值分位（显式写清以便与实测对拍）；`percentile=0.5` 即中位（默认），
    尾值场景可传 `0.95`。空样本返回 `0.0`（与既有"缺字段如实置 0，不猜"一致）。
    """
    values = sorted(float(x) for x in samples if float(x) >= 0.0)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    q = min(1.0, max(0.0, float(percentile)))
    pos = q * (len(values) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (pos - low)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


@dataclass(frozen=True)
class SegmentProfile:
    """一个候选段（设备 + 引擎）的画像。耗时单位 ms，容量单位 byte。"""

    node_id: str
    engine: str = "llama.cpp"
    capacity_bytes: int = 0
    ms_per_layer_decode: float = 0.0
    ms_per_layer_prefill: float = 0.0
    ms_fixed_decode: float = 0.0
    ms_fixed_prefill: float = 0.0
    rtt_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    thermal_throttled: bool = False
    retry_rate: float = 0.0
    artifacts_ready: bool = True
    source: str = "declared"

    def with_rtt(self, rtt_ms: float, *, source: str | None = None) -> "SegmentProfile":
        """★ 2026-10-14（22c 缺口①）：返回一个只替换 `rtt_ms`（与可选 `source`）的副本。

        本类是 `frozen=True`，所以用 `dataclasses.replace` 而不是原地赋值。
        典型用法：先 `from_device_profile(...)` 拿画像，再用**该段的在线心跳 RTT**
        （见 `rtt_ms_from_heartbeat`）覆盖人工默认值 —— 这就是"把在线 RTT 接进切点求解"。

        ⚠️ `#71`：RTT 必须**按段**取，不要用一个全局中位覆盖所有段。
        """
        replacement: dict[str, Any] = {"rtt_ms": float(max(0.0, rtt_ms))}
        if source is not None:
            replacement["source"] = str(source)
        return replace(self, **replacement)

    @classmethod
    def from_device_profile(
        cls,
        info: Mapping[str, Any] | None,
        *,
        node_id: str = "segment",
        engine: str = "llama.cpp",
        **overrides: Any,
    ) -> "SegmentProfile":
        """从 `DeviceProfiler.to_dict()` 风格的字典构造（缺字段如实置 0，不猜）。"""
        data = _mapping(info)
        ram = _mapping(data.get("ram"))
        net = _mapping(data.get("network"))
        gpu = _mapping(data.get("gpu"))
        available_gb = _number(ram.get("available_gb", data.get("ram_available_gb")), 0.0)
        values: dict[str, Any] = {
            "node_id": str(node_id),
            "engine": str(engine),
            "capacity_bytes": int(max(0.0, available_gb) * (1024 ** 3)),
            "ms_per_layer_decode": _number(data.get("ms_per_layer_decode")),
            "ms_per_layer_prefill": _number(data.get("ms_per_layer_prefill")),
            "ms_fixed_decode": _number(data.get("ms_fixed_decode")),
            "ms_fixed_prefill": _number(data.get("ms_fixed_prefill")),
            # ★ 2026-10-14（22c 缺口①）：除 `rtt_ms` 外，也认 `network_path.HeartbeatQualityWindow`
            #   风格的分位字段 —— 取 `rtt_ms_p50`（典型值）优先，再退到 p95。
            "rtt_ms": _number(
                data.get(
                    "rtt_ms",
                    net.get("rtt_ms", net.get("rtt_ms_p50", net.get("rtt_ms_p95"))),
                )
            ),
            "bandwidth_mbps": _number(data.get("bandwidth_mbps", net.get("bandwidth_mbps"))),
            "thermal_throttled": _bool(data.get("thermal_throttled"), False),
            "retry_rate": _number(data.get("retry_rate")),
            "artifacts_ready": _bool(data.get("artifacts_ready"), True),
            "source": str(data.get("source", "device_profile")),
        }
        if gpu and not values["capacity_bytes"]:
            values["capacity_bytes"] = int(max(0.0, _number(gpu.get("vram_total_gb"))) * (1024 ** 3))
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelayCutPlan:
    admitted: bool
    reason: str
    cuts: tuple[int, ...]
    segment_layers: tuple[int, ...]
    capacity_feasible: bool
    latency_estimate: dict[str, Any]
    risk_penalty: dict[str, Any]
    score: float
    capacity_gain_x: float | None
    speedup_vs_single: float | None
    segment_bytes: tuple[int, ...] = ()
    candidates: tuple[dict[str, Any], ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "admitted": self.admitted,
            "reason": self.reason,
            "cuts": list(self.cuts),
            "segment_layers": list(self.segment_layers),
            "segment_bytes": list(self.segment_bytes),
            "capacity_feasible": self.capacity_feasible,
            "latency_estimate": self.latency_estimate,
            "risk_penalty": self.risk_penalty,
            "score": self.score,
            "capacity_gain_x": self.capacity_gain_x,
            "speedup_vs_single": self.speedup_vs_single,
            "candidates": [dict(item) for item in self.candidates],
        }


def _transfer_ms(bytes_total: float, profile: SegmentProfile, *, with_rtt: bool = True) -> float:
    """把 ``bytes_total`` 传过一跳的时间（ms）：串行化 + 可选 RTT。带宽为 0 ⇒ 视为不可用。"""
    bandwidth_mbps = max(0.0, profile.bandwidth_mbps)
    if bandwidth_mbps <= 0:
        return float("inf")
    serialization_ms = bytes_total * 8.0 / (bandwidth_mbps * 1e6) * 1000.0
    return serialization_ms + (profile.rtt_ms if with_rtt else 0.0)


def _segment_bytes(
    layer_bytes: Sequence[int], start: int, end: int, *, has_embedding: bool,
    has_lm_head: bool, non_split_bytes: int,
) -> int:
    total = sum(int(layer_bytes[i]) for i in range(start, end))
    if has_embedding or has_lm_head:
        total += int(non_split_bytes)
    return total


# --------------------------------------------------------------------------- 分段收益账
#: ★ 2026-10-09（③）：以下默认值**全部来自实测**（见 `docs/已知问题记录.md` #70/#71/#72）：
#:   - master（RTX 4060，GPU）16 层 ≈ 49ms ⇒ **3.06 ms/层**
#:   - Surface（Intel 集显轻薄本）：单段 53ms，减 2×3ms 往返 ⇒ **11.75 ms/层**；RTT **3ms**
#:   - Y700（Android + llama.cpp）：单段 84ms，减 2×19ms 往返 ⇒ **11.5 ms/层**；RTT **19ms**
#: 这些数字是"要不要把层分出去"的唯一依据，不要凭感觉改。
MEASURED_MASTER_MS_PER_LAYER = 3.06
MEASURED_TABLET_MS_PER_LAYER = 11.75
MEASURED_TABLET_RTT_MS = 3.0
MEASURED_ANDROID_MS_PER_LAYER = 11.5
MEASURED_ANDROID_RTT_MS = 19.0
#: 一次 stage 的固定往返**次数**：offer→accept、accept→result（实测 `perf2`，见 #70）。
STAGE_ROUND_TRIPS = 2


@dataclass(frozen=True)
class SegmentEconomics:
    """把 `n_layers` 交给某设备作为**独立段**的账。

    判据（由实测分解推出）：一次 stage 要付 `STAGE_ROUND_TRIPS` 次跨机往返，故

        值得 ⟺ n·c_master > STAGE_ROUND_TRIPS·rtt + n·c_device
        ⟹ n > STAGE_ROUND_TRIPS·rtt / (c_master − c_device)   （仅当 c_master > c_device 有解）

    ⇒ **若该设备单层耗时不低于 master，则任何层数都不值得** —— 它不该参与逐 token 分段，
    它的价值在**容量**（把装不下的模型切开），不在**速度**。这正是实测"去掉一个段 +49%"的成因。

    ⚠️ 边界：这些 `c_*` 是**单请求、单 token**（batch=1）的值；合批后往返税与 `c_master`
    都摊到 N 个请求上，不等式必须重算（见 #66/#70）。
    """

    n_layers: int
    c_master_ms: float = MEASURED_MASTER_MS_PER_LAYER
    c_device_ms: float = MEASURED_ANDROID_MS_PER_LAYER
    rtt_ms: float = MEASURED_ANDROID_RTT_MS
    round_trips: int = STAGE_ROUND_TRIPS

    @property
    def saved_ms(self) -> float:
        """留在 master 上要花的时间。"""
        return float(self.n_layers) * float(self.c_master_ms)

    @property
    def cost_ms(self) -> float:
        """分出去要付的时间：固定往返税 + 对端自己算。"""
        return (
            float(self.round_trips) * float(self.rtt_ms)
            + float(self.n_layers) * float(self.c_device_ms)
        )

    @property
    def net_ms(self) -> float:
        """净收益（正=值得分出去）。"""
        return self.saved_ms - self.cost_ms

    @property
    def worthwhile(self) -> bool:
        """严格大于 0 才算值得（恰好打平不值得——多一次往返就亏）。"""
        return self.net_ms > 0.0

    @property
    def min_layers(self) -> float | None:
        """达到盈亏平衡所需的最少层数；该设备**注定不划算**时返回 `None`。"""
        gain = float(self.c_master_ms) - float(self.c_device_ms)
        if gain <= 0.0:
            return None
        return float(self.round_trips) * float(self.rtt_ms) / gain


def segment_worthwhile(
    n_layers: int,
    c_master_ms: float = MEASURED_MASTER_MS_PER_LAYER,
    c_device_ms: float = MEASURED_ANDROID_MS_PER_LAYER,
    rtt_ms: float = MEASURED_ANDROID_RTT_MS,
) -> bool:
    """便捷判据：把 `n_layers` 交给该设备是否值得（细节见 `SegmentEconomics`）。"""
    return SegmentEconomics(n_layers, c_master_ms, c_device_ms, rtt_ms).worthwhile


def cut_multiple_from_gguf(gguf_path: "str | Path") -> int | None:
    """★ `#67`：**从源 GGUF 直接读**裁层整数倍（`<arch>.full_attention_interval`）。

    为什么需要它：`cut_multiple` 此前**全靠人工传参**，而 `relay_cut_plan` 与
    `HeteroBaseline` 的**默认值都是 1** —— 对 hybrid 架构（Qwen3.5，interval=4）**1 是错的**：
    裁层 GGUF 的层类型按"（重编号后的）本地层号对 interval 取模"推导，非整数倍切点会整体错位，
    llama.cpp 报 `missing tensor 'blk.x.<...>'`。本函数让人工参数成为**可选覆盖**而非必需知识。

    返回 `None` 表示"源没有该约束"（非 hybrid 件，或 `interval <= 1`）⇒ 调用方保持既有行为；
    这是**探测**语义而非校验，故文件缺失/读不动一律返回 `None` 而不抛。
    """
    try:
        import gguf
    except ImportError:  # pragma: no cover - torch-free / minimal 环境
        return None
    try:
        reader = gguf.GGUFReader(str(gguf_path))
    except Exception:  # noqa: BLE001 - 探测失败即"无约束"
        return None
    try:
        arch_field = reader.fields.get("general.architecture")
        if arch_field is None:
            return None
        arch = str(arch_field.contents())
        interval_field = reader.fields.get(f"{arch}.full_attention_interval")
        if interval_field is None:
            return None
        interval = int(interval_field.contents())
    except Exception:  # noqa: BLE001
        return None
    return interval if interval > 1 else None


def legal_cuts(total_layers: int, *, cut_multiple: int = 1,
               min_layers_per_segment: int = 1) -> tuple[int, ...]:
    """合法切点集合：满足步长约束、且切点两侧都留够层数。

    ⚠️ `cut_multiple > 1` 是**硬约束**（Qwen3.5 = 4）：裁层 GGUF 的层类型按
    `full_attention_interval` 交替，切错会导致下游无法加载（实测 N=2 报错）。
    """
    multiple = max(1, _int(cut_multiple, 1))
    minimum = max(CUT_LAYER_MIN, _int(min_layers_per_segment, 1))
    total = max(0, _int(total_layers))
    upper = total - minimum
    return tuple(k for k in range(multiple, upper + 1, multiple) if k >= minimum)


def _estimate_latency(
    segment_layers: Sequence[int],
    segments: Sequence[SegmentProfile],
    hidden: RelayHiddenSpec,
    *,
    prefill_tokens: int,
    decode_tokens: int,
) -> dict[str, Any]:
    """端到端估计：各段自身耗时之和 + 每跳激活传输（decode 逐 token）。"""
    decode_ms = 0.0
    prefill_ms = 0.0
    for profile, layers in zip(segments, segment_layers):
        decode_ms += profile.ms_fixed_decode + profile.ms_per_layer_decode * layers
        prefill_ms += profile.ms_fixed_prefill + profile.ms_per_layer_prefill * layers

    decode_bytes = hidden.bytes_per_token * max(0, _int(decode_tokens))
    prefill_bytes = hidden.bytes_per_token * max(0, _int(prefill_tokens))
    transfer_decode_ms = 0.0
    transfer_prefill_ms = 0.0
    for producer, consumer in zip(segments, segments[1:]):
        # 生产端带宽决定出站；RTT 记在消费端一侧（各计一次，避免双算）
        hop_profile = SegmentProfile(
            node_id=consumer.node_id, bandwidth_mbps=producer.bandwidth_mbps,
            rtt_ms=consumer.rtt_ms)
        transfer_decode_ms += _transfer_ms(decode_bytes, hop_profile)
        transfer_prefill_ms += _transfer_ms(prefill_bytes, hop_profile)
    return {
        "prefill_ms": None if math.isinf(prefill_ms + transfer_prefill_ms)
        else round(prefill_ms + transfer_prefill_ms, 4),
        "decode_ms": None if math.isinf(decode_ms + transfer_decode_ms)
        else round(decode_ms + transfer_decode_ms, 4),
        "compute_prefill_ms": round(prefill_ms, 4),
        "compute_decode_ms": round(decode_ms, 4),
        "transfer_prefill_ms": None if math.isinf(transfer_prefill_ms) else round(transfer_prefill_ms, 4),
        "transfer_decode_ms": None if math.isinf(transfer_decode_ms) else round(transfer_decode_ms, 4),
        "hops": max(0, len(segments) - 1),
        "hidden_bytes_per_token": hidden.bytes_per_token,
        "prefill_tokens": max(0, _int(prefill_tokens)),
        "decode_tokens": max(0, _int(decode_tokens)),
    }


def _risk_penalty(segments: Sequence[SegmentProfile]) -> dict[str, Any]:
    """风险分项：弱网 / 高 RTT / 热降频 / 重试 / 工件不可得。取值越大越差（0 最好）。"""
    items: list[dict[str, Any]] = []
    for profile in segments:
        bandwidth = max(0.0, profile.bandwidth_mbps)
        bandwidth_item = 0.0
        if bandwidth < RISK_BANDWIDTH_MBPS_FLOOR:
            bandwidth_item = 1.0 - (bandwidth / RISK_BANDWIDTH_MBPS_FLOOR)
        rtt_item = min(1.0, max(0.0, profile.rtt_ms) / RISK_RTT_MS_CEIL)
        item = {
            "node_id": profile.node_id,
            "bandwidth": round(bandwidth_item, 4),
            "rtt": round(rtt_item, 4),
            "thermal": RISK_THERMAL_PENALTY if profile.thermal_throttled else 0.0,
            "retry": round(min(1.0, max(0.0, profile.retry_rate)), 4),
            "artifacts": 0.0 if profile.artifacts_ready else RISK_ARTIFACT_PENALTY,
        }
        item["total"] = round(min(1.0, sum(v for k, v in item.items()
                                           if k not in ("node_id", "total"))), 4)
        items.append(item)
    total = round(min(1.0, sum(i["total"] for i in items) / max(1, len(items))), 4) if items else 1.0
    return {"total": total, "items": items}


def _single_segment_ms(segments: Sequence[SegmentProfile], total_layers: int) -> float | None:
    """单段（整模放一个设备）的 decode 估计：取最快的可用段。"""
    values = [p.ms_fixed_decode + p.ms_per_layer_decode * total_layers for p in segments]
    values = [v for v in values if v > 0]
    return min(values) if values else None


def plan_relay_cut_n_segments(
    *,
    total_layers: int,
    layer_bytes: Sequence[int],
    n_embd: int,
    segments: Sequence[SegmentProfile],
    non_split_bytes: int = 0,
    hidden_dtype: str = "float32",
    cut_multiple: int = 1,
    min_layers_per_segment: int = 1,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    weights: Mapping[str, float] | None = None,
    prefill_tokens: int = 32,
    decode_tokens: int = 32,
    max_candidates: int = 512,
) -> RelayCutPlan:
    """在合法切点空间里选**容量可行 + 目标函数最优**的 n 段方案，或给出具名拒绝。"""
    total = max(0, _int(total_layers))
    hidden = RelayHiddenSpec(n_embd=_int(n_embd), dtype=str(hidden_dtype))
    weight_map = dict(DEFAULT_WEIGHTS)
    weight_map.update({k: _number(v) for k, v in _mapping(weights).items()})
    margin = max(1.0, _number(safety_margin, DEFAULT_SAFETY_MARGIN))
    profiles = list(segments)

    def reject(reason: str, *, feasible: bool = False, risk: dict[str, Any] | None = None) -> RelayCutPlan:
        return RelayCutPlan(
            admitted=False, reason=reason, cuts=(), segment_layers=(),
            capacity_feasible=feasible,
            latency_estimate={}, risk_penalty=risk or {"total": 1.0, "items": []},
            score=float("-inf"), capacity_gain_x=None, speedup_vs_single=None)

    if not profiles:
        return reject("no_usable_segments")
    if len(layer_bytes) != total:
        return reject("layer_bytes_length_mismatch")
    if total < 2:
        return reject("model_has_no_relay_range")
    if not hidden.supported:
        return reject("unsupported_hidden_format")
    if len(profiles) < 2:
        return reject("segment_count_mismatch")

    cuts = legal_cuts(total, cut_multiple=cut_multiple,
                      min_layers_per_segment=min_layers_per_segment)
    if not cuts:
        return reject("no_legal_cut_point")

    n_segments = len(profiles)
    if n_segments - 1 > len(cuts):
        return reject("no_legal_cut_combination")
    candidate_limit = _int(max_candidates, 0)
    if candidate_limit <= 0:
        return reject("candidate_limit_invalid")
    candidate_count = math.comb(len(cuts), n_segments - 1)
    if candidate_count > candidate_limit:
        return reject("candidate_space_too_large")
    combos = list(itertools.combinations(cuts, n_segments - 1))

    evaluated: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    # The non-split weights live on both endpoint segments, so the resident
    # total must match the per-segment accounting below.
    total_bytes = sum(int(b) for b in layer_bytes) + 2 * int(non_split_bytes)
    single_ms = _single_segment_ms(profiles, total)

    for combo in combos:
        boundaries = (0,) + tuple(combo) + (total,)
        sizes = tuple(boundaries[i + 1] - boundaries[i] for i in range(n_segments))
        if any(size < max(CUT_LAYER_MIN, _int(min_layers_per_segment, 1)) for size in sizes):
            continue
        sizes_bytes = tuple(
            _segment_bytes(layer_bytes, boundaries[i], boundaries[i + 1],
                           has_embedding=(i == 0), has_lm_head=(i == n_segments - 1),
                           non_split_bytes=non_split_bytes)
            for i in range(n_segments))
        feasible = all(
            size_bytes * margin <= max(0, profiles[i].capacity_bytes)
            for i, size_bytes in enumerate(sizes_bytes))
        latency = _estimate_latency(sizes, profiles, hidden,
                                    prefill_tokens=prefill_tokens, decode_tokens=decode_tokens)
        risk = _risk_penalty(profiles)
        max_segment = max(sizes_bytes) if sizes_bytes else 0
        capacity_gain = round(total_bytes / max_segment, 4) if max_segment else None
        speedup = (round(single_ms / latency["decode_ms"], 4)
                   if single_ms and latency["decode_ms"] else None)
        score = (weight_map["capacity"] * (capacity_gain or 0.0)
                 + weight_map["latency"] * (speedup or 0.0)
                 - weight_map["risk"] * risk["total"])
        item = {
            "cuts": list(combo),
            "segment_layers": list(sizes),
            "segment_bytes": list(sizes_bytes),
            "capacity_feasible": feasible,
            "capacity_gain_x": capacity_gain,
            "speedup_vs_single": speedup,
            "latency_estimate": latency,
            "risk_penalty": risk["total"],
            "score": round(score, 4),
        }
        evaluated.append(item)
        if not feasible:
            continue
        if best is None or score > best["score"]:
            best = item

    if best is None:
        any_feasible = any(item["capacity_feasible"] for item in evaluated)
        plan = reject("all_placements_infeasible" if not any_feasible else "no_scoring_candidate",
                      feasible=any_feasible, risk=_risk_penalty(profiles))
        return RelayCutPlan(**{**plan.__dict__, "candidates": tuple(evaluated[:200])})

    plan = RelayCutPlan(
        admitted=True,
        reason="best_objective",
        cuts=tuple(best["cuts"]),
        segment_layers=tuple(best["segment_layers"]),
        capacity_feasible=bool(best["capacity_feasible"]),
        latency_estimate=best["latency_estimate"],
        risk_penalty=_risk_penalty(profiles),
        score=best["score"],
        capacity_gain_x=best["capacity_gain_x"],
        speedup_vs_single=best["speedup_vs_single"],
        segment_bytes=tuple(best["segment_bytes"]),
        candidates=tuple(evaluated[:200]),
    )
    return plan


def fit_segment_profile(
    measurements: Sequence[Mapping[str, Any]],
    *,
    node_id: str,
    engine: str,
    layers_key: str,
    ms_key: str,
    capacity_bytes: int = 0,
    bandwidth_mbps: float = 0.0,
    rtt_ms: float = 0.0,
    source: str = "fitted_from_measurements",
) -> tuple[SegmentProfile, dict[str, Any]]:
    """从实测点线性回归出「固定开销 + 每层耗时」（最小二乘）。

    `measurements` 里每条给该段**实际层数**与**实测耗时**。返回 `(profile, fit_info)`，
    `fit_info` 带斜率/截距/`r2`/样本数 —— 报告必须给出这些，才能复算。

    为什么需要：P2 的「切点重搜」要用**该设备对的实测**而不是通用常数（早期用
    32 MiB/层 的常量估计偏差巨大）。
    """
    points = [(_number(m.get(layers_key)), _number(m.get(ms_key)))
              for m in measurements if _mapping(m).get(layers_key) is not None
              and m.get(ms_key) is not None]
    points = [(x, y) for x, y in points if x > 0 and y > 0]
    if len(points) < 2:
        return (SegmentProfile(node_id=node_id, engine=engine, capacity_bytes=capacity_bytes,
                               bandwidth_mbps=bandwidth_mbps, rtt_ms=rtt_ms, source=source),
                {"samples": len(points), "r2": None, "slope": None, "intercept": None})
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    slope = 0.0 if denom == 0 else sum((x - mean_x) * (y - mean_y) for x, y in points) / denom
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    r2 = None if ss_tot == 0 else round(1.0 - ss_res / ss_tot, 6)
    profile = SegmentProfile(
        node_id=node_id, engine=engine, capacity_bytes=capacity_bytes,
        ms_per_layer_decode=round(max(0.0, slope), 6),
        ms_fixed_decode=round(intercept, 6),
        bandwidth_mbps=bandwidth_mbps, rtt_ms=rtt_ms, source=source)
    return profile, {
        "samples": len(points),
        "slope": round(slope, 6),
        "intercept": round(intercept, 6),
        "r2": r2,
    }


def fit_two_segment(
    measurements: Sequence[Mapping[str, Any]],
    *,
    total_layers: int,
    upstream_layers_key: str = "upstream_layers",
    upstream_ms_key: str = "upstream_decode_ms",
    downstream_ms_key: str = "downstream_decode_ms",
    upstream_profile: SegmentProfile | None = None,
    downstream_profile: SegmentProfile | None = None,
) -> dict[str, Any]:
    """两段（D→L）专用便捷入口：分别拟合上游/下游，返回两段的 profile 与拟合质量。"""
    upstream_ms = [{"k": m.get(upstream_layers_key), "ms": m.get(upstream_ms_key)}
                   for m in measurements]
    downstream_ms = []
    for m in measurements:
        upstream_used = _number(m.get(upstream_layers_key))
        if upstream_used <= 0:
            continue
        downstream_ms.append({"k": total_layers - upstream_used,
                              "ms": m.get(downstream_ms_key)})
    up_profile, up_fit = fit_segment_profile(
        [{"k": p["k"], "ms": p["ms"]} for p in upstream_ms],
        node_id=(upstream_profile.node_id if upstream_profile else "upstream_pytorch"),
        engine="pytorch",
        layers_key="k", ms_key="ms",
        capacity_bytes=(upstream_profile.capacity_bytes if upstream_profile else 0),
        bandwidth_mbps=(upstream_profile.bandwidth_mbps if upstream_profile else 0.0),
        rtt_ms=(upstream_profile.rtt_ms if upstream_profile else 0.0))
    dn_profile, dn_fit = fit_segment_profile(
        [{"k": p["k"], "ms": p["ms"]} for p in downstream_ms],
        node_id=(downstream_profile.node_id if downstream_profile else "downstream_llama"),
        engine="llama.cpp",
        layers_key="k", ms_key="ms",
        capacity_bytes=(downstream_profile.capacity_bytes if downstream_profile else 0),
        bandwidth_mbps=(downstream_profile.bandwidth_mbps if downstream_profile else 0.0),
        rtt_ms=(downstream_profile.rtt_ms if downstream_profile else 0.0))
    return {"upstream": up_profile, "downstream": dn_profile,
            "fit": {"upstream": up_fit, "downstream": dn_fit}}
