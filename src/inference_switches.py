"""设备画像 → llama.cpp 推理开关的显式映射（票 20 `DEVICE-AWARE-INFER-01` 第一步）。

本模块解决的是「开关各自为政」：`-ctk`/`-ctv`（KV 量化）、`flash_attn` 等此前要么
只有 docstring、要么硬编码，既不按设备也不按**该开关在本机到底有没有用**决定。

三条设计约束：

1. **默认值 = 接线前的既有行为**（KV 不量化、flash-attn 关、线程数沿用各调用点策略）。
   本模块只把「为什么这么定」写成一张显式表并给出可观测出口，**不改变默认路径**；
   非默认值只可能来自显式配置（环境变量），便于回滚与对拍。
2. **没有实测支撑的开关一律保持关闭**：「有开关」不等于「该开」。KV 量化在 CPU 边缘段
   实测净亏（见 `EVIDENCE`），而 CUDA 路径尚无同口径实测 ⇒ 默认关闭。
3. **事实源分两层且不可混用**：`engine_isa` 是 llama.cpp **构建实际启用**的指令集
   （`llama_print_system_info()`），`host_isa` 是 CPU **硬件支持**的指令集。二者不等价
   （本机实测：host 支持 AVX512，engine 只启用 AVX2 + REPACK），换 ISA 必须重跑判据。

`n_ctx` 的「每段 KV 预算」求解属第二步，本模块只提供 `kv_budget_bytes()` 所需的
只读事实（层数 × 每层字节 × ctx 的算法在调用方）。
"""
from __future__ import annotations

import ctypes
import logging
import os
import platform
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

#: 映射表版本；`to_dict()` 会带上，便于与实测记录对齐。
SCHEMA_VERSION = 1

#: 允许的 KV 缓存类型（对应 `GGML_TYPE_*`）。刻意只列被本仓实测覆盖过的档位。
KV_CACHE_TYPES: tuple = ("q8_0", "q4_0")

#: 环境变量：显式覆盖开关。未设置一律走「默认 = 现状」。
ENV_KV_CACHE = "QLH_INFER_KV_CACHE"
ENV_FLASH_ATTN = "QLH_INFER_FLASH_ATTN"
ENV_THREADS = "QLH_INFER_THREADS"

#: 每条映射的实测依据。数值口径随原始记录，跨设备不可直接套用。
EVIDENCE: Dict[str, str] = {
    "kv_cache_type": (
        "CPU 边缘段实测净亏：q8_0 0.89×、q4_0 0.84×（输出 sha8 全同），"
        "见 archive/relay/KV缓存优化与注意力结构调研-GQA与MLA-2026-10-14.md"
    ),
    "flash_attn": (
        "CPU 上仅 1.06×（噪声级）；CUDA 路径无同口径实测，见同一调研 §实测"
    ),
    "n_threads": (
        "Surface（4C8T）实测 n_threads=8 最优；线程数属每机标定，不按画像盲推"
    ),
    "cpu_moe": (
        "35B-A3B 实测 `-ngl 99 --cpu-moe` 15.94 tok/s（2.66×）且仅占 2863 MiB 显存；"
        "沿用引擎既有的 MoE 判定，不由画像决定"
    ),
}

#: `llama_print_system_info()` 里需要识别的引擎 ISA 标记。
_ENGINE_ISA_KEYS: tuple = (
    "AVX", "AVX2", "AVX512", "AVX512_VNNI", "AVX512_BF16", "FMA", "F16C",
    "SSE3", "SSSE3", "NEON", "DOTPROD", "I8MM", "REPACK", "LLAMAFILE", "OPENMP",
)

#: host 侧关心的硬件特性（与引擎标记同名，便于对照）。
_HOST_ISA_KEYS: tuple = (
    "AVX2", "AVX512", "AVX512_VNNI", "AVX512_BF16", "FMA", "DOTPROD", "I8MM",
)

#: Win32 `IsProcessorFeaturePresent` 常量（Windows 无 /proc/cpuinfo）。
_WIN_PF: Dict[str, int] = {
    "AVX": 39,
    "AVX2": 40,
    "AVX512": 41,
}

# Linux `/proc/cpuinfo` / 内核 flags → 本模块的统一键名。
_LINUX_FLAG_MAP: Dict[str, str] = {
    "avx": "AVX",
    "avx2": "AVX2",
    "avx512f": "AVX512",
    "avx512_vnni": "AVX512_VNNI",
    "avx512bf16": "AVX512_BF16",
    "fma": "FMA",
    "asimd": "NEON",
    "asimddp": "DOTPROD",
    "i8mm": "I8MM",
}


def parse_engine_isa(text: str) -> Dict[str, int]:
    """把 `llama_print_system_info()` 的输出解析成 ``{"AVX2": 1, ...}``。

    只保留值为 1 的标记；未知标记忽略（上游新增特性不应让本模块报错）。
    """
    found: Dict[str, int] = {}
    for name, value in re.findall(r"([A-Z][A-Z0-9_]{1,31})\s*=\s*(\d+)", text or ""):
        if name in _ENGINE_ISA_KEYS and int(value) == 1:
            found[name] = 1
    return found


def detect_engine_isa() -> Dict[str, int]:
    """读取 llama.cpp **构建实际启用**的指令集；绑定不可用时返回空表（不猜测）。"""
    try:
        import llama_cpp  # noqa: PLC0415 - 可选依赖，刻意惰性

        raw = llama_cpp.llama_print_system_info()
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        return parse_engine_isa(text)
    except Exception as exc:  # pragma: no cover - 环境相关
        logger.debug("llama_print_system_info 不可用: %s", exc)
        return {}


def _detect_host_isa_linux() -> Dict[str, int]:
    flags: Dict[str, int] = {}
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                if key.strip().lower() in ("flags", "features"):
                    for token in value.split():
                        mapped = _LINUX_FLAG_MAP.get(token.strip().lower())
                        if mapped:
                            flags[mapped] = 1
                    break
    except OSError as exc:  # pragma: no cover - 平台相关
        logger.debug("/proc/cpuinfo 不可读: %s", exc)
    return flags


def _detect_host_isa_windows() -> Dict[str, int]:
    flags: Dict[str, int] = {}
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - 平台相关
        logger.debug("Win32 API 不可用: %s", exc)
        return flags
    for name, feature in _WIN_PF.items():
        try:
            if kernel32.IsProcessorFeaturePresent(ctypes.c_uint(feature)):
                flags[name] = 1
        except Exception:  # pragma: no cover - 平台相关
            continue
    # VNNI/BF16 无对应 PF 常量 ⇒ 保持未知，不虚报（Windows 下需 CPUID 才能判）。
    return flags


def detect_host_isa() -> Dict[str, int]:
    """读取 CPU **硬件支持**的指令集；无法判定时返回空表（不猜测）。"""
    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64", "armv7l"):
        return _detect_host_isa_linux() if sys.platform != "win32" else {}
    if sys.platform == "win32":
        return _detect_host_isa_windows()
    if sys.platform.startswith("linux"):
        return _detect_host_isa_linux()
    return {}


@dataclass
class InferenceSwitches:
    """一次解析得到的推理开关集合。字段默认值即「接线前的既有行为」。"""

    kv_cache_type_k: Optional[str] = None
    kv_cache_type_v: Optional[str] = None
    flash_attn: bool = False
    n_threads: Optional[int] = None
    cpu_moe: str = "auto"
    engine_isa: Dict[str, int] = field(default_factory=dict)
    host_isa: Dict[str, int] = field(default_factory=dict)
    overrides: Dict[str, str] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @property
    def is_default(self) -> bool:
        """是否与接线前的默认行为完全一致（用于「开关未改变路径」自证）。"""
        return (
            self.kv_cache_type_k is None
            and self.kv_cache_type_v is None
            and not self.flash_attn
            and self.n_threads is None
            and self.cpu_moe == "auto"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kv_cache_type_k": self.kv_cache_type_k,
            "kv_cache_type_v": self.kv_cache_type_v,
            "flash_attn": self.flash_attn,
            "n_threads": self.n_threads,
            "cpu_moe": self.cpu_moe,
            "engine_isa": dict(sorted(self.engine_isa.items())),
            "host_isa": dict(sorted(self.host_isa.items())),
            "overrides": dict(sorted(self.overrides.items())),
            "is_default": self.is_default,
            "evidence": dict(EVIDENCE),
        }

    def describe(self) -> str:
        """单行可读摘要（进日志，便于回滚与对拍）。"""
        kv = self.kv_cache_type_k or "off"
        threads = str(self.n_threads) if self.n_threads else "auto"
        engine = ",".join(sorted(self.engine_isa)) or "unknown"
        return (
            f"kv={kv} flash_attn={'on' if self.flash_attn else 'off'} "
            f"threads={threads} cpu_moe={self.cpu_moe} "
            f"engine_isa={engine} overrides={sorted(self.overrides) or 'none'}"
        )


def _read_env(env: Mapping[str, str], name: str) -> Optional[str]:
    raw = env.get(name)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def resolve_inference_switches(
    profile: Optional[Mapping[str, Any]] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    engine_isa: Optional[Mapping[str, int]] = None,
) -> InferenceSwitches:
    """把设备画像与显式配置解析成一组开关。

    Args:
        profile: `device_profiler.DeviceProfiler.to_dict()` 的结果；`None` 表示无画像。
        env: 环境变量映射（默认取 `os.environ`，测试可注入）。
        engine_isa: 引擎 ISA 事实；`None` 时实时探测（结果只在进程内缓存一次）。

    Returns:
        `InferenceSwitches`。**默认（无环境变量覆盖）等于接线前的既有行为**：
        KV 不量化、flash-attn 关、线程数交给调用点、`cpu_moe="auto"`。
    """
    environ: Mapping[str, str] = os.environ if env is None else env
    switches = InferenceSwitches(
        engine_isa=dict(engine_isa if engine_isa is not None else _cached_engine_isa()),
        host_isa=_cached_host_isa(),
    )

    raw_kv = _read_env(environ, ENV_KV_CACHE)
    if raw_kv is not None:
        value = raw_kv.strip().lower()
        if value in ("off", "none", "disable", "disabled", "0"):
            switches.overrides["kv_cache_type"] = "off"
        elif value in KV_CACHE_TYPES:
            switches.kv_cache_type_k = value
            switches.kv_cache_type_v = value
            switches.overrides["kv_cache_type"] = value
        else:
            logger.warning(
                "忽略无效的 %s=%r（允许值：%s 或 off）",
                ENV_KV_CACHE, raw_kv, "、".join(KV_CACHE_TYPES),
            )

    raw_fa = _read_env(environ, ENV_FLASH_ATTN)
    if raw_fa is not None:
        enabled = raw_fa.strip().lower() in ("1", "true", "yes", "on")
        disabled = raw_fa.strip().lower() in ("0", "false", "no", "off")
        if enabled or disabled:
            switches.flash_attn = enabled
            switches.overrides["flash_attn"] = "on" if enabled else "off"
        else:
            logger.warning("忽略无效的 %s=%r（允许 1/0）", ENV_FLASH_ATTN, raw_fa)

    raw_threads = _read_env(environ, ENV_THREADS)
    if raw_threads is not None:
        try:
            parsed = int(raw_threads)
        except ValueError:
            parsed = 0
        if parsed > 0:
            switches.n_threads = parsed
            switches.overrides["n_threads"] = str(parsed)
        else:
            logger.warning("忽略无效的 %s=%r（需正整数）", ENV_THREADS, raw_threads)

    if profile is not None:
        switches.overrides.setdefault("profile_tier", str(profile.get("tier", "unknown")))
    return switches


_ENGINE_ISA_CACHE: Dict[str, Dict[str, int]] = {}
_HOST_ISA_CACHE: Dict[str, Dict[str, int]] = {}


def _cached_engine_isa() -> Dict[str, int]:
    if "value" not in _ENGINE_ISA_CACHE:
        _ENGINE_ISA_CACHE["value"] = detect_engine_isa()
    return _ENGINE_ISA_CACHE["value"]


def _cached_host_isa() -> Dict[str, int]:
    if "value" not in _HOST_ISA_CACHE:
        _HOST_ISA_CACHE["value"] = detect_host_isa()
    return _HOST_ISA_CACHE["value"]


def reset_caches() -> None:
    """清空 ISA 探测缓存（测试用）。"""
    _ENGINE_ISA_CACHE.clear()
    _HOST_ISA_CACHE.clear()


def apply_to_load_kwargs(switches: InferenceSwitches, load_kwargs: Dict[str, Any]) -> None:
    """把开关写进 `llama_cpp.Llama(**load_kwargs)` 的参数字典（原地修改）。

    默认档位**不新增任何键**，调用点的既有行为逐字不变；已由调用方显式给出的
    参数不被覆盖（显式调用参数优先于画像映射）。
    """
    if switches.kv_cache_type_k and "type_k" not in load_kwargs:
        load_kwargs["type_k"] = switches.kv_cache_type_k
    if switches.kv_cache_type_v and "type_v" not in load_kwargs:
        load_kwargs["type_v"] = switches.kv_cache_type_v
    if switches.flash_attn and "flash_attn" not in load_kwargs:
        load_kwargs["flash_attn"] = True
    if switches.n_threads and "n_threads" not in load_kwargs:
        load_kwargs["n_threads"] = switches.n_threads


def server_argv_args(switches: InferenceSwitches) -> List[str]:
    """把开关翻成 `llama-server` 命令行片段；默认档位返回**空列表**。

    参数名以本机 `llama-server --help` 实测为准：`--threads N`、
    `--cache-type-k TYPE`、`--cache-type-v TYPE`、`--flash-attn [on|off|auto]`。
    flash-attn 只在**显式覆盖**时才出现 —— 不传等于 server 自己的默认 `auto`，
    这样「未介入」与「显式关闭」在命令行上可区分。
    """
    args: List[str] = []
    if switches.n_threads:
        args += ["--threads", str(int(switches.n_threads))]
    if switches.kv_cache_type_k:
        args += ["--cache-type-k", str(switches.kv_cache_type_k)]
    if switches.kv_cache_type_v:
        args += ["--cache-type-v", str(switches.kv_cache_type_v)]
    if "flash_attn" in switches.overrides:
        args += ["--flash-attn", "on" if switches.flash_attn else "off"]
    return args


def resolve_from_device_profile() -> InferenceSwitches:
    """从本进程设备画像解析开关；画像不可用或异常时退回默认档位。

    两条推理路径（`llama_engine` 的绑定、`llama_server_engine` 的子进程）共用本
    入口，同一台设备得到同一张映射，而不是各自读一套默认值。
    """
    profile = None
    try:
        from device_profiler import get_profile

        profile = get_profile().to_dict()
    except Exception as exc:  # pragma: no cover - 画像属可选事实
        logger.debug("设备画像不可用，推理开关按默认档位继续: %s", exc)
    return resolve_inference_switches(profile)
