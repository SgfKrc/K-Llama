"""
设备能力检测模块 — 硬件画像 + 档位分级 + 自适应配置推荐
==========================================================
纯硬件检测模块，零项目内部依赖（只依赖 psutil、torch、标准库）。

功能:
1. 检测 CPU / RAM / GPU / 磁盘 / 操作系统
2. 5 档设备分级 (workstation → mobile)
3. 加权评分 (GPU 50% + RAM 30% + CPU 20%)
4. 根据档位生成自适应推理配置
5. Android/移动端部署路径提示

使用:
    profiler = DeviceProfiler()
    print(profiler.to_dict())        # 完整 JSON 画像
    config = profiler.recommend_config()  # 自适应配置 dict

模拟测试:
    profiler = DeviceProfiler.mock_mobile()   # 模拟 4GB RAM 手机
    profiler = DeviceProfiler.mock_edge()     # 模拟树莓派
"""

from __future__ import annotations

import enum
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple

import psutil

from task_worker_protocol import (
    RUNTIME_PROFILES as TASK_WORKER_RUNTIME_PROFILES,
    RUNTIME_PROFILE_UNSPECIFIED,
)

logger = logging.getLogger(__name__)

_WINDOWS_NO_WINDOW = (
    getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
)


# ============================================================
# 设备档位枚举
# ============================================================

class DeviceTier(enum.Enum):
    """设备档位，从高到低"""
    WORKSTATION = "workstation"   # 桌面工作站 / 高端游戏本 (≥8GB VRAM, ≥32GB RAM, ≥8核)
    LAPTOP = "laptop"             # 普通游戏本 / 独显本 (4-8GB VRAM, 16-32GB RAM)
    ULTRABOOK = "ultrabook"       # 轻薄本 / 集显本 (≤2GB VRAM 共享, 8-16GB RAM)
    EDGE = "edge"                 # 树莓派 / Jetson Nano / 旧笔记本 (无独显, 4-8GB RAM)
    MOBILE = "mobile"             # 安卓手机 / 低端平板 (无 CUDA, ≤4GB RAM)

    @property
    def label(self) -> str:
        labels = {
            DeviceTier.WORKSTATION: "桌面工作站",
            DeviceTier.LAPTOP: "游戏本 / 独显本",
            DeviceTier.ULTRABOOK: "轻薄本 / 集显",
            DeviceTier.EDGE: "边缘设备",
            DeviceTier.MOBILE: "移动设备",
        }
        return labels.get(self, "未知")

    @property
    def color(self) -> str:
        """前端 CSS 颜色变量名"""
        colors = {
            DeviceTier.WORKSTATION: "gold",
            DeviceTier.LAPTOP: "green",
            DeviceTier.ULTRABOOK: "blue",
            DeviceTier.EDGE: "orange",
            DeviceTier.MOBILE: "red",
        }
        return colors.get(self, "gray")

    @property
    def icon(self) -> str:
        icons = {
            DeviceTier.WORKSTATION: "🖥️",
            DeviceTier.LAPTOP: "💻",
            DeviceTier.ULTRABOOK: "📔",
            DeviceTier.EDGE: "📟",
            DeviceTier.MOBILE: "📱",
        }
        return icons.get(self, "❓")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class CPUInfo:
    model_name: str = ""
    physical_cores: int = 0
    logical_cores: int = 0
    freq_mhz: float = 0.0       # 当前频率
    freq_max_mhz: float = 0.0   # 最大频率
    architecture: str = ""      # x86_64 / aarch64 / armv7l
    usage_percent: float = 0.0


@dataclass
class RAMInfo:
    total_gb: float = 0.0
    available_gb: float = 0.0
    used_gb: float = 0.0
    percent_used: float = 0.0


@dataclass
class GPUInfo:
    name: str = ""
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    cuda_available: bool = False
    compute_capability: str = ""   # e.g. "8.6"
    is_integrated: bool = False    # 集显 / 共享内存 GPU
    gpu_type: str = "unknown"      # "integrated" | "discrete" | "unknown"
    driver_version: str = ""
    mps_available: bool = False    # Apple Metal
    index: int = 0                 # GPU 列表中的序号


@dataclass
class DiskInfo:
    total_gb: float = 0.0
    free_gb: float = 0.0
    used_gb: float = 0.0
    path: str = ""


@dataclass
class PlatformInfo:
    os: str = ""                   # Windows / Linux / Darwin
    os_version: str = ""
    architecture: str = ""         # AMD64 / ARM64
    machine: str = ""              # platform.machine()
    hostname: str = ""
    python_version: str = ""


# ============================================================
# 发行 profile（运行时分档）环境变量读取
# ============================================================

#: 发行 profile 取值集合。**必须与 `packaging/packaging/runtime_guard.py` 的
#: `RUNTIME_PROFILES` 一致**，由 `tests/test_runtime_profile_advertise.py` 校验。
#: `llama_cpp_only` 是边缘/无 CUDA 发行的 profile：包体与外部 runtime venv 都不含 torch。
RUNTIME_PROFILES: tuple = TASK_WORKER_RUNTIME_PROFILES

#: 未声明 profile 时的取值：源码运行，或启动时既没给 `--runtime-profile`
#: 也没设 `QLH_RUNTIME_PROFILE`。它表示"这个进程没有发行档"，不是硬件未知。
def detect_runtime_profile() -> str:
    """
    读取本进程的发行 profile（环境变量 `QLH_RUNTIME_PROFILE`）。

    该变量由应用首启选定并写入环境，子进程继承。只认 `RUNTIME_PROFILES`
    内的取值；未设置或不在此集合内一律返回 `unspecified`。

    字段随注册 device_info 上报主节点，使调度能区分"无 torch 的边缘发行节点"
    与"带 torch 的 PC 节点"，而不是靠目标机器碰巧装了什么反推。
    """
    raw = os.environ.get("QLH_RUNTIME_PROFILE", "").strip().lower()
    return raw if raw in RUNTIME_PROFILES else RUNTIME_PROFILE_UNSPECIFIED


@dataclass
class DeviceProfile:
    """完整设备画像"""
    tier: str = ""
    tier_label: str = ""
    tier_icon: str = ""
    score_total: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    cpu: Optional[CPUInfo] = None
    ram: Optional[RAMInfo] = None
    gpu: Optional[GPUInfo] = None           # 当前选中的 GPU（向后兼容）
    gpus: list = field(default_factory=list)  # 全部检测到的 GPU 列表
    selected_gpu_index: int = 0               # 当前选中的 GPU 序号
    disk: Optional[DiskInfo] = None
    platform: Optional[PlatformInfo] = None
    recommendations: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    android_ready: bool = False
    # 发行 profile（`llama_cpp_only` / `torch_cpu` / `torch_cuda`；
    # `unspecified` = 源码运行或未声明）。随注册 device_info 上报主节点。
    runtime_profile: str = RUNTIME_PROFILE_UNSPECIFIED
    # TP 孤岛聚合能力段（网关节点专用，随注册 device_info 上报主节点；
    # None = 非孤岛网关节点，旧版本节点无感）
    island: Optional[Dict[str, Any]] = None
    # 推理开关映射（票 20 `DEVICE-AWARE-INFER-01`）：画像 → llama.cpp 开关的
    # 显式化结果，含每条映射的实测依据；None = 未解析（如映射模块不可用）。
    inference_switches: Optional[Dict[str, Any]] = None


# ============================================================
# TP 孤岛环境变量读取（保持本模块零项目内部依赖，直接读 QLH_ISLAND_*）
# ============================================================

def _island_env_bool(name: str) -> bool:
    raw = os.environ.get(name, "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _island_env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, "").strip() or default)
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def _island_env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "").strip() or default)
        return value if value >= 0 else default
    except (TypeError, ValueError):
        return default


def _mask_island_base_url(url: str) -> str:
    """脱敏孤岛端点：去除内嵌账号密码与查询串。"""
    if not url:
        return ""
    try:
        from urllib.parse import urlsplit, urlunsplit
        parts = urlsplit(url)
        netloc = parts.netloc
        if "@" in netloc:
            netloc = netloc.rsplit("@", 1)[-1]
        return urlunsplit((parts.scheme, netloc, parts.path.rstrip("/"), "", ""))
    except (ValueError, AttributeError):
        return "<无效 URL>"


def detect_island_profile() -> Optional[Dict[str, Any]]:
    """
    读取 TP 孤岛聚合能力配置（QLH_ISLAND_* 环境变量）。

    网关无法探测孤岛内部硬件，聚合 GPU 数 / 显存 / 并行度由部署者声明。
    未启用时返回 None（画像中不出现 island 段，旧节点/主节点无感）。
    """
    if not _island_env_bool("QLH_ISLAND_ENABLED"):
        return None
    return {
        "enabled": True,
        "backend": os.environ.get("QLH_ISLAND_BACKEND", "").strip() or "openai-compatible",
        "base_url": _mask_island_base_url(
            os.environ.get("QLH_ISLAND_BASE_URL", "").strip()
        ),
        "model": os.environ.get("QLH_ISLAND_MODEL", "").strip(),
        "tp_size": _island_env_int("QLH_ISLAND_TP_SIZE", 1),
        "gpu_count": _island_env_int("QLH_ISLAND_GPU_COUNT", 1),
        "vram_gb": _island_env_float("QLH_ISLAND_VRAM_GB", 0.0),
    }


# ============================================================
# 核心检测类
# ============================================================

class DeviceProfiler:
    """
    设备硬件能力检测器。

    检测 CPU / RAM / GPU / 磁盘 / 操作系统，
    按 5 档分级并生成自适应配置建议。
    """

    def __init__(self):
        self._cpu: Optional[CPUInfo] = None
        self._ram: Optional[RAMInfo] = None
        self._gpu: Optional[GPUInfo] = None
        self._gpus: list[GPUInfo] = []            # 全部 GPU
        self._selected_gpu_index: int = 0          # 当前选中的 GPU 序号
        self._disk: Optional[DiskInfo] = None
        self._platform: Optional[PlatformInfo] = None
        self._tier: DeviceTier = DeviceTier.LAPTOP  # 默认
        self._score: float = 0.0
        self._score_breakdown: Dict[str, float] = {}

        # 执行检测
        self._detect_all()

    # ================================================================
    # 公开方法
    # ================================================================

    @property
    def tier(self) -> DeviceTier:
        return self._tier

    @property
    def score(self) -> float:
        return self._score

    @property
    def cpu(self) -> CPUInfo:
        return self._cpu

    @property
    def ram(self) -> RAMInfo:
        return self._ram

    @property
    def gpu(self) -> GPUInfo:
        return self._gpu

    @property
    def gpus(self) -> list:
        return self._gpus

    @property
    def selected_gpu_index(self) -> int:
        return self._selected_gpu_index

    @property
    def has_multiple_gpus(self) -> bool:
        """是否检测到多个 GPU（集显+独显混合）"""
        return len(self._gpus) >= 2

    @property
    def is_gaming_laptop(self) -> bool:
        """
        是否检测为游戏本（集显+独显混合，且非桌面工作站）。

        判定条件：
        1. 同时有集显和 CUDA 独显
        2. 独显名称含 "Laptop" 关键字（移动版 GPU）
           → 或者：独显 VRAM ≤ 8GB（桌面显卡通常 > 8GB）
        3. 总得分 < workstation 阈值（否则是高端笔记本，当作工作站）
        """
        if not self.has_multiple_gpus:
            return False
        has_igpu = any(g.is_integrated for g in self._gpus)
        dgpu = next((g for g in self._gpus if not g.is_integrated and g.cuda_available), None)
        if not has_igpu or not dgpu:
            return False
        # 独显名称暗示移动版
        is_mobile_gpu = "laptop" in dgpu.name.lower()
        # 或者 VRAM ≤ 8GB 的非旗舰 GPU
        is_mid_range = dgpu.vram_total_gb <= 8.0
        # 排除工作站级（RTX 4090 Laptop / 专业卡 + 大内存）
        is_workstation_gpu = (
            dgpu.vram_total_gb >= 12.0
            or "quadro" in dgpu.name.lower()
            or "rtx a" in dgpu.name.lower()
        )
        return (is_mobile_gpu or is_mid_range) and not is_workstation_gpu

    def select_gpu(self, index: int) -> bool:
        """
        切换当前使用的 GPU。

        Args:
            index: GPU 列表中的序号 (0 = 集显/CPU, 1 = 独显/CUDA)

        Returns:
            是否切换成功
        """
        if index < 0 or index >= len(self._gpus):
            return False
        self._selected_gpu_index = index
        self._gpu = self._gpus[index]
        return True

    @property
    def disk(self) -> DiskInfo:
        return self._disk

    @property
    def platform(self) -> PlatformInfo:
        return self._platform

    def recommend_config(self) -> Dict[str, Any]:
        """
        根据设备档位 + 当前选中的 GPU 生成自适应推理配置。

        Returns:
            dict — 可直接用于覆盖 config.py 的配置参数
        """
        # device 由当前选中的 GPU 决定
        if self._gpu and self._gpu.cuda_available:
            device = "cuda"
        elif self._gpu and self._gpu.mps_available:
            device = "mps"
        else:
            device = "cpu"

        tier = self._tier
        configs = {
            DeviceTier.WORKSTATION: {
                "quant_type": "fp16",
                "page_size": 128,
                "max_pages": 512,
                "max_seq_len": 8192,    # Qwen-1.8B 原生上下文窗口
                "max_new_tokens": 4096,
                "use_compile": True,
                "device": device,
                "description": "桌面级性能，可运行 FP16 原版模型 + 算子融合",
            },
            DeviceTier.LAPTOP: {
                "quant_type": "int4",
                "page_size": 128,
                "max_pages": 256,
                "max_seq_len": 4096,    # 8GB VRAM 可容纳约 7K token 的 KV 缓存
                "max_new_tokens": 2048,
                "use_compile": False,
                "device": device,
                "description": "游戏本级性能，推荐 INT4 量化平衡速度与显存",
            },
            DeviceTier.ULTRABOOK: {
                "quant_type": "int4",
                "page_size": 64,
                "max_pages": 128,
                "max_seq_len": 2048,
                "max_new_tokens": 1024,
                "use_compile": False,
                "device": device,
                "description": "轻薄本级性能，INT4 量化 + 缩减 KV 缓存",
            },
            DeviceTier.EDGE: {
                "quant_type": "int4",
                "page_size": 64,
                "max_pages": 64,
                "max_seq_len": 1024,
                "max_new_tokens": 512,
                "use_compile": False,
                "device": "cpu",
                "description": "边缘设备，CPU-only 推理 + 最小 KV 缓存",
            },
            DeviceTier.MOBILE: {
                "quant_type": "int4",
                "page_size": 32,
                "max_pages": 32,
                "max_seq_len": 512,
                "max_new_tokens": 256,
                "use_compile": False,
                "device": "cpu",
                "description": "移动设备级，极限压缩配置（建议导出 ONNX/GGUF）",
            },
        }
        return configs.get(tier, configs[DeviceTier.LAPTOP])

    def to_dict(self) -> dict:
        """完整设备画像，JSON 可序列化"""
        profile = DeviceProfile(
            tier=self._tier.value,
            tier_label=self._tier.label,
            tier_icon=self._tier.icon,
            score_total=round(self._score, 1),
            score_breakdown={k: round(v, 1) for k, v in self._score_breakdown.items()},
            cpu=self._cpu,
            ram=self._ram,
            gpu=self._gpu,
            gpus=[_dataclass_to_dict(g) for g in self._gpus],
            selected_gpu_index=self._selected_gpu_index,
            disk=self._disk,
            platform=self._platform,
            recommendations=[],
            warnings=[],
            android_ready=self._check_android_ready(),
            runtime_profile=detect_runtime_profile(),
            island=detect_island_profile(),
        )

        # 推理开关映射（票 20 `DEVICE-AWARE-INFER-01`）：把「画像 → 开关」显式化，
        # 关闭态同样可观测。惰性 import：本模块刻意保持零项目内部依赖。
        try:
            import inference_switches

            profile.inference_switches = inference_switches.resolve_inference_switches(
                _dataclass_to_dict(profile)
            ).to_dict()
        except Exception:  # pragma: no cover - 画像异常不得拖垮设备画像本身
            profile.inference_switches = None

        # 生成建议
        profile.recommendations = self._generate_recommendations()
        profile.warnings = self._generate_warnings()

        return _dataclass_to_dict(profile)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    # ================================================================
    # 模拟工厂（测试用）
    # ================================================================

    @classmethod
    def mock_mobile(cls) -> "DeviceProfiler":
        """模拟 4GB RAM + ARM + 无 GPU 的手机环境"""
        profiler = cls.__new__(cls)
        profiler._cpu = CPUInfo(
            model_name="ARM Cortex-A78",
            physical_cores=4,
            logical_cores=4,
            freq_mhz=2400.0,
            freq_max_mhz=2800.0,
            architecture="aarch64",
            usage_percent=30.0,
        )
        profiler._ram = RAMInfo(total_gb=4.0, available_gb=1.5, used_gb=2.5, percent_used=62.5)
        gpu = GPUInfo(name="Adreno 730 (集成)", vram_total_gb=0.0, is_integrated=True, gpu_type="integrated", index=0)
        profiler._gpu = gpu
        profiler._gpus = [gpu]
        profiler._selected_gpu_index = 0
        profiler._disk = DiskInfo(total_gb=64.0, free_gb=12.0, used_gb=52.0, path="/data")
        profiler._platform = PlatformInfo(
            os="Android", os_version="14", architecture="ARM64",
            machine="aarch64", hostname="localhost", python_version=sys.version,
        )
        profiler._score = 8.0
        profiler._score_breakdown = {"gpu": 0, "ram": 3.0, "cpu": 5.0}
        profiler._tier = DeviceTier.MOBILE
        return profiler

    @classmethod
    def mock_edge(cls) -> "DeviceProfiler":
        """模拟树莓派 4B (4GB) 环境"""
        profiler = cls.__new__(cls)
        profiler._cpu = CPUInfo(
            model_name="ARM Cortex-A72",
            physical_cores=4, logical_cores=4,
            freq_mhz=1500.0, freq_max_mhz=1500.0,
            architecture="aarch64", usage_percent=20.0,
        )
        profiler._ram = RAMInfo(total_gb=4.0, available_gb=2.5, used_gb=1.5, percent_used=37.5)
        gpu = GPUInfo(name="VideoCore VI (无 CUDA)", vram_total_gb=0.0, is_integrated=True, gpu_type="integrated", index=0)
        profiler._gpu = gpu
        profiler._gpus = [gpu]
        profiler._selected_gpu_index = 0
        profiler._disk = DiskInfo(total_gb=32.0, free_gb=18.0, used_gb=14.0, path="/")
        profiler._platform = PlatformInfo(
            os="Linux", os_version="Raspbian 12", architecture="ARM64",
            machine="aarch64", hostname="raspberrypi", python_version=sys.version,
        )
        profiler._score = 16.0
        profiler._score_breakdown = {"gpu": 0, "ram": 6.0, "cpu": 10.0}
        profiler._tier = DeviceTier.EDGE
        return profiler

    # ================================================================
    # 内部检测方法
    # ================================================================

    def _detect_all(self):
        """执行全部硬件检测"""
        self._cpu = self._detect_cpu()
        self._ram = self._detect_ram()
        self._gpu = self._detect_gpu()
        self._disk = self._detect_disk()
        self._platform = self._detect_platform()

        # 评分与分级（基于当前选中 GPU）
        self._score_breakdown = self._compute_scores()
        self._score = sum(self._score_breakdown.values())
        self._tier = self._classify_tier()

        # 游戏本：检测到集显+独显混合，默认使用独显（CUDA 加速）
        # 用户可在设备面板手动切换到集显以降低功耗
        if self.is_gaming_laptop:
            dgpu = next((g for g in self._gpus if not g.is_integrated and g.cuda_available), None)
            igpu = next((g for g in self._gpus if g.is_integrated), None)
            if dgpu and igpu:
                logger.info(
                    f"检测到游戏本: {igpu.name}（集显）+ {dgpu.name}（独显），"
                    f"默认使用独显推理。可在设备面板切换至集显省电。"
                )

    def _detect_cpu(self) -> CPUInfo:
        """检测 CPU 信息"""
        info = CPUInfo()

        # 型号名称
        try:
            if sys.platform == "win32":
                # Windows: 从注册表或 wmic 获取
                result = subprocess.run(
                    ["wmic", "cpu", "get", "Name"],
                    capture_output=True, text=True, timeout=5,
                    encoding="utf-8", errors="replace",
                    creationflags=_WINDOWS_NO_WINDOW,
                )
                lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
                info.model_name = lines[1] if len(lines) > 1 else platform.processor()
            elif sys.platform == "darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=5,
                )
                info.model_name = result.stdout.strip()
            else:
                # Linux: 读取 /proc/cpuinfo
                try:
                    with open("/proc/cpuinfo") as f:
                        for line in f:
                            if "model name" in line:
                                info.model_name = line.split(":", 1)[1].strip()
                                break
                except FileNotFoundError:
                    info.model_name = platform.processor()
        except Exception:
            info.model_name = platform.processor() or "Unknown"

        # 核心数
        info.physical_cores = psutil.cpu_count(logical=False) or 1
        info.logical_cores = psutil.cpu_count(logical=True) or 1

        # 频率
        try:
            freq = psutil.cpu_freq()
            if freq:
                info.freq_mhz = round(freq.current, 1)
                info.freq_max_mhz = round(freq.max, 1) if freq.max else freq.current
        except Exception:
            info.freq_mhz = 0.0
            info.freq_max_mhz = 0.0

        # 架构
        info.architecture = platform.machine()

        # 使用率
        info.usage_percent = round(psutil.cpu_percent(interval=0.1), 1)

        return info

    def _detect_ram(self) -> RAMInfo:
        """检测系统内存"""
        mem = psutil.virtual_memory()
        return RAMInfo(
            total_gb=round(mem.total / (1024 ** 3), 1),
            available_gb=round(mem.available / (1024 ** 3), 1),
            used_gb=round(mem.used / (1024 ** 3), 1),
            percent_used=round(mem.percent, 1),
        )

    def _detect_gpu(self) -> GPUInfo:
        """
        检测 GPU 信息（单 GPU 兼容接口）。

        内部调用 _detect_all_gpus() 获取全部 GPU 列表，
        返回当前选中的 GPU。
        """
        self._gpus = self._detect_all_gpus()

        if not self._gpus:
            # 无 GPU
            return GPUInfo(name="CPU-only (无 GPU)", is_integrated=True, gpu_type="integrated")

        # 默认选择策略：
        # - 有独显（CUDA）→ 优先选独显（最佳推理性能）
        # - 游戏本（集显+独显混合）→ 默认选独显，用户可在设备面板切换到集显省电
        # - 无独显 → 选集显 / CPU-only
        dgpu_idx = next(
            (i for i, g in enumerate(self._gpus)
             if not g.is_integrated and g.cuda_available),
            None,
        )
        if dgpu_idx is not None:
            self._selected_gpu_index = dgpu_idx
        else:
            # 无独显：选第一个集显或最后一个
            igpu_idx = next(
                (i for i, g in enumerate(self._gpus) if g.is_integrated),
                len(self._gpus) - 1,
            )
            self._selected_gpu_index = igpu_idx

        self._gpu = self._gpus[self._selected_gpu_index]
        return self._gpu

    def _detect_all_gpus(self) -> list:
        """
        检测系统中所有 GPU（集显 + 独显）。

        检测优先级：
        1. NVIDIA CUDA GPU（torch.cuda）
        2. 系统显示适配器（WMI on Windows）
        3. Apple MPS
        4. CPU-only fallback
        """
        all_gpus: list[GPUInfo] = []
        seen_names: set[str] = set()

        # ---- 第 1 层：NVIDIA CUDA ----
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    vram = round(props.total_memory / (1024 ** 3), 1)

                    # 判断独显还是集显
                    # 注意：CUDA 路径仅检测 NVIDIA GPU，均为独显/独立 GPU
                    # "mx" 不应在此列表中 — GeForce MX 系列（MX150/250/350）是独显
                    name_lower = name.lower()
                    integrated_kw = ["intel", "radeon", "adreno", "mali",
                                     "iris", "uhd", "hd graphics"]
                    is_integrated = (
                        any(kw in name_lower for kw in integrated_kw)
                        or vram < 0.5  # VRAM < 0.5GB 才可能为虚拟/集显
                    )

                    free_mem = (props.total_memory - torch.cuda.memory_allocated(i)) / (1024 ** 3)

                    gpu = GPUInfo(
                        name=name,
                        vram_total_gb=vram,
                        vram_free_gb=round(max(0, free_mem), 1),
                        cuda_available=True,
                        compute_capability=f"{props.major}.{props.minor}",
                        is_integrated=is_integrated,
                        gpu_type="integrated" if is_integrated else "discrete",
                        index=len(all_gpus),
                    )
                    all_gpus.append(gpu)
                    seen_names.add(name.lower())
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"CUDA GPU 检测异常: {e}")

        # ---- 第 2 层：系统级显示适配器检测（Windows WMI） ----
        # WMI 输出格式: Node,AdapterRAM,Name
        if sys.platform == "win32":
            try:
                result = subprocess.run(
                    ["wmic", "path", "win32_VideoController",
                     "get", "Name,AdapterRAM", "/format:csv"],
                    capture_output=True, text=True, timeout=10,
                    encoding="utf-8", errors="replace",
                    creationflags=_WINDOWS_NO_WINDOW,
                )
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    # 跳过标题行
                    if "Node," in line and "AdapterRAM" in line:
                        continue

                    # 解析 CSV: Node,AdapterRAM,Name
                    # 使用逗号分割，第1列=Node, 第2列=AdapterRAM, 剩余=Name
                    first_comma = line.find(",")
                    if first_comma < 0:
                        continue
                    rest = line[first_comma + 1:]
                    second_comma = rest.find(",")
                    if second_comma < 0:
                        continue

                    vram_bytes_str = rest[:second_comma].strip()
                    name = rest[second_comma + 1:].strip()

                    # 过滤无效名称
                    if not name:
                        continue
                    if len(name) < 5:
                        continue
                    # 已由 CUDA 检测到则跳过
                    name_lower = name.lower()
                    if name_lower in seen_names:
                        continue
                    # 过滤列名噪声
                    skip_names = {"adapterram", "name", "node", "driverversion"}
                    if name_lower in skip_names:
                        continue

                    # 解析 VRAM（字节 → GB）
                    vram_gb = 0.0
                    try:
                        adapter_ram = int(vram_bytes_str) if vram_bytes_str else 0
                        vram_gb = round(adapter_ram / (1024 ** 3), 1) if adapter_ram > 0 else 0.0
                    except (ValueError, TypeError):
                        vram_gb = 0.0

                    # 集显关键词
                    igpu_kw = ["intel", "uhd", "iris", "hd graphics",
                               "adreno", "mali", "microsoft basic",
                               "amd radeon(tm)"]
                    is_igpu = any(kw in name_lower for kw in igpu_kw)
                    # 独显关键词（radeon pro 必须在 radeon 通用匹配之前）
                    dgpu_kw = ["nvidia", "rtx", "gtx", "geforce", "quadro",
                               "tesla", "radeon rx", "radeon pro", "radeon w", "arc a"]
                    is_dgpu = any(kw in name_lower for kw in dgpu_kw)

                    # AMD Radeon 不带独立显卡关键词 → 集显（如 Radeon Graphics on laptop APU）
                    if "radeon" in name_lower and not is_dgpu:
                        is_igpu = True

                    # 分类
                    if is_igpu and not is_dgpu:
                        vram_gb = 0.0  # 集显共享系统内存
                        gpu_type = "integrated"
                    elif is_dgpu:
                        gpu_type = "discrete"
                    else:
                        gpu_type = "unknown"
                        # 未知类型：有 VRAM → 可能是独显，无 VRAM → 可能是集显
                        if vram_gb > 0.5:
                            gpu_type = "discrete"

                    gpu = GPUInfo(
                        name=name,
                        vram_total_gb=vram_gb,
                        cuda_available=False,  # WMI 检测到的通常不是 CUDA 设备
                        is_integrated=(gpu_type == "integrated"),
                        gpu_type=gpu_type,
                        index=len(all_gpus),
                    )
                    all_gpus.append(gpu)
                    seen_names.add(name_lower)
            except Exception as e:
                logger.debug(f"WMI GPU 检测异常: {e}")

        # ---- 第 2.5 层：Linux 系统级 GPU 检测 ----
        elif sys.platform == "linux":
            # 2.5a: nvidia-smi（NVIDIA GPU 的权威来源）
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 2:
                            gpu_name = parts[0]
                            name_lower = gpu_name.lower()
                            if name_lower in seen_names:
                                continue
                            try:
                                vram_mb = float(parts[1])
                                vram_gb = round(vram_mb / 1024.0, 1)
                            except (ValueError, TypeError):
                                vram_gb = 0.0
                            gpu = GPUInfo(
                                name=gpu_name,
                                vram_total_gb=vram_gb,
                                cuda_available=True,
                                is_integrated=False,
                                gpu_type="discrete",
                                index=len(all_gpus),
                            )
                            all_gpus.append(gpu)
                            seen_names.add(name_lower)
            except FileNotFoundError:
                logger.debug("nvidia-smi 未找到，跳过 NVIDIA GPU 检测")
            except Exception as e:
                logger.debug(f"nvidia-smi GPU 检测异常: {e}")

            # 2.5b: lspci 检测所有 VGA/3D 设备（集显 + 未安装 nvidia-smi 的独显）
            try:
                result = subprocess.run(
                    ["lspci"], capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        line_lower = line.lower()
                        if "vga" not in line_lower and "3d" not in line_lower:
                            continue
                        # 提取 GPU 名称: "01:00.0 VGA compatible controller: NVIDIA ..."
                        if ":" not in line:
                            continue
                        # 跳过已检测到的
                        name_part = line.split(":", 2)[-1].strip()
                        # 去掉前缀如 "VGA compatible controller: "
                        for prefix in ("vga compatible controller: ", "3d controller: "):
                            if name_part.lower().startswith(prefix):
                                name_part = name_part[len(prefix):]
                                break
                        gpu_name = name_part.strip()
                        if not gpu_name or len(gpu_name) < 3:
                            continue
                        name_lower = gpu_name.lower()
                        if name_lower in seen_names:
                            continue

                        # 去重：若已有 NVIDIA CUDA GPU 且 lspci 也报告 NVIDIA，则跳过
                        # （nvidia-smi 提供了更准确的名称和 VRAM 信息）
                        if "nvidia" in name_lower and any(
                            "nvidia" in g.name.lower() for g in all_gpus
                        ):
                            continue

                        # 集显关键词
                        igpu_kw = ["intel", "uhd", "iris", "hd graphics",
                                   "adreno", "mali", "radeon graphics",
                                   "microsoft basic", "virtio"]
                        is_igpu = any(kw in name_lower for kw in igpu_kw)
                        # 独显关键词（radeon pro 必须在 radeon 通用匹配之前）
                        dgpu_kw = ["nvidia", "rtx", "gtx", "geforce", "quadro",
                                   "tesla", "radeon rx", "radeon pro", "radeon w",
                                   "arc a"]
                        is_dgpu = any(kw in name_lower for kw in dgpu_kw)
                        # AMD Radeon 不带独立显卡关键词 → 集显
                        if "radeon" in name_lower and not is_dgpu:
                            is_igpu = True

                        if is_igpu and not is_dgpu:
                            gpu_type = "integrated"
                            vram_gb = 0.0
                        elif is_dgpu:
                            gpu_type = "discrete"
                            vram_gb = 0.0  # lspci 不报告 VRAM
                        else:
                            gpu_type = "unknown"
                            vram_gb = 0.0

                        gpu = GPUInfo(
                            name=gpu_name,
                            vram_total_gb=vram_gb,
                            cuda_available=False,
                            is_integrated=(gpu_type == "integrated"),
                            gpu_type=gpu_type,
                            index=len(all_gpus),
                        )
                        all_gpus.append(gpu)
                        seen_names.add(name_lower)
            except FileNotFoundError:
                logger.debug("lspci 未找到，跳过 PCI GPU 检测")
            except Exception as e:
                logger.debug(f"lspci GPU 检测异常: {e}")

        # ---- 第 3 层：Apple Metal ----
        try:
            import torch
            if torch.backends.mps.is_available():
                if "apple mps" not in seen_names:
                    gpu = GPUInfo(
                        name="Apple MPS (集成 GPU)",
                        vram_total_gb=0.0,
                        cuda_available=False,
                        is_integrated=True,
                        gpu_type="integrated",
                        mps_available=True,
                        index=len(all_gpus),
                    )
                    all_gpus.append(gpu)
        except (ImportError, Exception):
            pass

        # ---- 重新分配 index ----
        for i, gpu in enumerate(all_gpus):
            gpu.index = i

        # ---- 无 GPU 时给一个 CPU-only 条目 ----
        if not all_gpus:
            all_gpus.append(GPUInfo(
                name="CPU-only (无 GPU)",
                vram_total_gb=0.0,
                is_integrated=True,
                gpu_type="integrated",
                index=0,
            ))

        return all_gpus

    def _detect_disk(self) -> DiskInfo:
        """检测磁盘空间（模型目录所在分区）"""
        # 优先检测项目根目录，fallback 到当前目录
        model_path = os.environ.get("MODEL_PATH", ".")
        # 解析到绝对路径
        abs_path = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "models",
        ))
        if not os.path.exists(abs_path):
            abs_path = "."

        try:
            usage = psutil.disk_usage(abs_path)
            return DiskInfo(
                total_gb=round(usage.total / (1024 ** 3), 1),
                free_gb=round(usage.free / (1024 ** 3), 1),
                used_gb=round(usage.used / (1024 ** 3), 1),
                path=abs_path,
            )
        except Exception:
            return DiskInfo(path=abs_path)

    def _detect_platform(self) -> PlatformInfo:
        """检测操作系统信息"""
        return PlatformInfo(
            os=platform.system(),
            os_version=platform.version(),
            architecture=platform.architecture()[0],
            machine=platform.machine(),
            hostname=platform.node(),
            python_version=sys.version.split()[0],
        )

    # ================================================================
    # 评分与分级
    # ================================================================

    def _compute_scores(self) -> Dict[str, float]:
        """
        计算设备加权评分 (0-100，孤岛网关按聚合能力上浮)。

        GPU 50% — 有独显+显存 → 高分，集显 → 中分，无 → 0
        RAM 30% — 越大越好，封顶 64GB
        CPU 20% — 核心数+频率综合
        island — TP 孤岛聚合能力附加项（网关节点专用）:
                 min(50, 聚合显存GB/24 × 50) + min(GPU数-1, 4) × 2.5
                 与 GPU 评分同刻度（24GB 单卡 = 50 分），多卡按数量加成
        """
        scores = {}

        # --- GPU 评分 (max 50) ---
        # 评分基于「最强可用 GPU」（通常是独显），而非当前选中的 GPU
        best_gpu = self._gpu
        if self._gpus:
            # 找到最强 GPU：优先独显+CUDA → 独显 → 集显+CUDA → 集显
            for g in sorted(self._gpus,
                            key=lambda g: (g.cuda_available and not g.is_integrated,
                                           not g.is_integrated,
                                           g.vram_total_gb),
                            reverse=True):
                best_gpu = g
                break

        if best_gpu and best_gpu.cuda_available and not best_gpu.is_integrated:
            vram = best_gpu.vram_total_gb
            if vram >= 24:
                scores["gpu"] = 50.0
            elif vram >= 16:
                scores["gpu"] = 45.0
            elif vram >= 12:
                scores["gpu"] = 40.0
            elif vram >= 8:
                scores["gpu"] = 35.0
            elif vram >= 6:
                scores["gpu"] = 25.0
            elif vram >= 4:
                scores["gpu"] = 18.0
            else:
                scores["gpu"] = 10.0
        elif best_gpu and (best_gpu.cuda_available or best_gpu.mps_available):
            # 集显 / Apple MPS
            scores["gpu"] = 8.0
        else:
            scores["gpu"] = 0.0

        # --- RAM 评分 (max 30) ---
        ram_gb = self._ram.total_gb if self._ram else 4.0
        ram_score = min(30.0, (ram_gb / 64.0) * 30.0)
        scores["ram"] = round(ram_score, 1)

        # --- CPU 评分 (max 20) ---
        cpu = self._cpu
        if cpu:
            core_score = min(10.0, cpu.physical_cores * 1.5)   # 6核封顶
            freq_score = min(10.0, cpu.freq_max_mhz / 400.0)   # 4GHz 封顶
            cpu_score = core_score + freq_score
        else:
            cpu_score = 5.0
        scores["cpu"] = round(cpu_score, 1)

        # --- TP 孤岛聚合能力附加分（网关节点专用） ---
        # 孤岛内部硬件对网关不可见，按部署者声明的 gpu_count/vram_gb 计算，
        # 使主节点排序反映"单个逻辑高算力节点"的聚合能力（调研方案 §2.1）。
        island = detect_island_profile()
        if island:
            vram_gb_total = float(island.get("vram_gb", 0.0) or 0.0)
            gpu_count = int(island.get("gpu_count", 1) or 1)
            island_score = (
                min(50.0, vram_gb_total / 24.0 * 50.0)
                + min(max(gpu_count, 1) - 1, 4) * 2.5
            )
            scores["island"] = round(island_score, 1)

        return scores

    def _classify_tier(self) -> DeviceTier:
        """
        根据硬件指标分级。

        分级基于「最强可用硬件」而非当前选中的 GPU，
        因为用户可能选集显模式但预期看到游戏本级配置建议。

        判定规则（优先级从高到低）：
        1. ARM 架构 + RAM < 4GB → MOBILE
        2. ARM 架构 + RAM < 8GB → EDGE
        3. RAM < 4GB → MOBILE
        4. RAM < 8GB + 无独显 → EDGE
        5. 无独显 / 集显 + RAM < 16GB → ULTRABOOK
        6. GPU VRAM < 4GB → ULTRABOOK
        7. GPU VRAM 4-8GB → LAPTOP
        8. GPU VRAM ≥ 8GB + RAM ≥ 32GB → WORKSTATION
        """
        # 规则 0: TP 孤岛网关 → WORKSTATION
        # 网关对集群呈现孤岛的聚合算力，档位按聚合能力而非本机硬件判定
        if detect_island_profile():
            return DeviceTier.WORKSTATION

        ram = self._ram
        plat = self._platform
        is_arm = plat and plat.machine in ("aarch64", "armv7l", "arm64")

        ram_gb = ram.total_gb if ram else 4.0

        # 找到最强 GPU（独显 > 集显，按 VRAM 排序）
        best_gpu = None
        for g in sorted(self._gpus,
                        key=lambda g: (
                            g.gpu_type == "discrete" or not g.is_integrated,
                            g.cuda_available,
                            g.vram_total_gb,
                        ),
                        reverse=True):
            best_gpu = g
            break

        vram_gb = best_gpu.vram_total_gb if best_gpu else 0.0
        # 独显判定：基于 GPU 物理类型（gpu_type/is_integrated），而非 PyTorch 是否编译了 CUDA
        # CPU-only PyTorch 环境下 CUDA 不可用，但硬件独显仍然是独显
        has_dgpu = best_gpu and (best_gpu.gpu_type == "discrete" or not best_gpu.is_integrated) if best_gpu else False

        # 规则 1: ARM + 极低 RAM → MOBILE
        if is_arm and ram_gb < 4.0:
            return DeviceTier.MOBILE

        # 规则 2: ARM + 低 RAM → EDGE
        if is_arm and ram_gb < 8.0:
            return DeviceTier.EDGE

        # 规则 3: 极低 RAM → MOBILE
        if ram_gb < 4.0:
            return DeviceTier.MOBILE

        # 规则 4: 低 RAM + 无独显 → EDGE
        if ram_gb < 8.0 and not has_dgpu:
            return DeviceTier.EDGE

        # 规则 5: 无独显 + 中低 RAM → ULTRABOOK
        if not has_dgpu and ram_gb < 16.0:
            return DeviceTier.ULTRABOOK

        # 规则 6: 低 VRAM 独显 → ULTRABOOK
        if has_dgpu and vram_gb < 4.0:
            return DeviceTier.ULTRABOOK

        # 规则 7: 中 VRAM → LAPTOP
        if has_dgpu and vram_gb < 8.0:
            return DeviceTier.LAPTOP

        # 规则 8: 高 VRAM + 大 RAM → WORKSTATION
        if has_dgpu and vram_gb >= 8.0 and ram_gb >= 32.0:
            return DeviceTier.WORKSTATION

        # 规则 9: 高 VRAM 但 RAM 不够 → LAPTOP
        if has_dgpu and vram_gb >= 8.0:
            return DeviceTier.LAPTOP

        # 兜底
        return DeviceTier.LAPTOP

    # ================================================================
    # 建议与警告生成
    # ================================================================

    def _check_android_ready(self) -> bool:
        """检测是否具备 Android 部署条件"""
        plat = self._platform
        if plat and plat.os == "Android":
            return True
        # 检测 ARM 架构（未来可运行）
        return plat.machine in ("aarch64", "armv7l", "arm64") if plat else False

    def _generate_recommendations(self) -> list:
        """根据设备画像生成操作建议"""
        recs = []
        gpu = self._gpu
        ram = self._ram
        tier = self._tier

        # GPU 切换提示
        if self.is_gaming_laptop:
            dgpu = next((g for g in self._gpus if not g.is_integrated and g.cuda_available), None)
            igpu_name = self._gpu.name if self._gpu and self._gpu.is_integrated else "集显"
            dgpu_name = dgpu.name if dgpu else "独显"
            recs.append(
                f"检测到游戏本：{igpu_name}（集显）+ {dgpu_name}（独显）\n"
                f"   默认使用独显（CUDA 加速），可在下方切换至集显（低功耗）"
            )
        elif self.has_multiple_gpus:
            recs.append("🔄 检测到多个 GPU，可在下方切换")

        # 量化建议
        if tier == DeviceTier.WORKSTATION:
            recs.append("💡 您的设备可运行 FP16 原版模型，速度最快 (~53 tok/s)")
        elif tier in (DeviceTier.LAPTOP, DeviceTier.ULTRABOOK):
            recs.append("💡 推荐使用 INT4 量化，兼顾速度与显存（显存仅需 ~1.8 GB）")
        elif tier in (DeviceTier.EDGE, DeviceTier.MOBILE):
            recs.append("💡 推荐 INT4 量化 + CPU 推理（当前 PyTorch 栈不支持移动端 GPU）")

        # 显存建议（基于当前选中 GPU）
        if gpu and gpu.cuda_available:
            vram = gpu.vram_total_gb
            if vram >= 8.0:
                recs.append(f"🟢 {vram:.0f} GB 显存充足，可加载全部 KV 缓存页")
            elif vram >= 4.0:
                recs.append(f"🟡 {vram:.0f} GB 显存适中，KV 缓存已自动缩减")
            elif vram > 0:
                recs.append(f"🔴 显存仅 {vram:.1f} GB，建议关闭其他 GPU 应用")
        elif gpu and gpu.is_integrated:
            recs.append("ℹ️ 当前使用集显 / CPU 推理模式（低功耗）")

        # RAM 建议
        if ram and ram.available_gb < 2.0:
            recs.append("⚠️ 可用内存不足 2 GB，推理可能 OOM")
        elif ram and ram.available_gb < 4.0:
            recs.append("⚠️ 可用内存较低，建议关闭其他应用")

        # 编译建议
        if tier == DeviceTier.WORKSTATION:
            recs.append("⚡ 支持 torch.compile 算子融合（FP16 下 +3.6% 加速）")

        # 移动端建议
        if tier == DeviceTier.MOBILE or self._check_android_ready():
            recs.append("📱 Android 部署建议：导出模型为 ONNX/GGUF 格式，搭配 ONNX Runtime Mobile 或 llama.cpp")

        return recs

    def _generate_warnings(self) -> list:
        """生成硬件相关的警告"""
        warnings = []
        gpu = self._gpu
        ram = self._ram
        cpu = self._cpu

        # 检测是否存在可用的独显（即使当前未选中）
        _dgpu = None
        for g in self._gpus:
            if not g.is_integrated and g.cuda_available:
                _dgpu = g
                break

        # 基于当前选中 GPU 的警告
        if gpu and gpu.is_integrated and gpu.cuda_available:
            warnings.append("当前使用集成显卡（含 CUDA），推理速度可能较慢")

        if gpu and gpu.is_integrated and not gpu.cuda_available:
            if _dgpu:
                # 集显模式，但有独显可用 — 明确告知可切换
                warnings.append(
                    f"当前使用集显 ({gpu.name})，CPU 推理速度约为独显的 1/5。"
                    f"可切换至独显 ({_dgpu.name}) 获得 CUDA 加速。"
                )
            else:
                # 真·无独显
                warnings.append("当前使用集显，CPU 推理速度约为 CUDA 的 1/5 ~ 1/10")

        if not gpu or (not gpu.cuda_available and not gpu.mps_available):
            if _dgpu:
                # 已有第一条集显警告，此处不再重复
                pass
            else:
                # 真·无任何加速器
                warnings.append("未检测到 GPU 加速器，将使用 CPU 推理（可安装 llama.cpp + GGUF 加速）")

        if ram and ram.total_gb < 8.0:
            warnings.append("系统内存不足 8 GB，大序列对话可能 OOM")

        if cpu and cpu.physical_cores < 4:
            warnings.append("CPU 核心数不足 4，多线程推理受限")

        if gpu and gpu.cuda_available and gpu.vram_total_gb < 2.0:
            warnings.append("显存不足 2 GB，仅 INT4 量化可运行")

        # 游戏本切换警告
        if self.is_gaming_laptop and self._gpu and not self._gpu.is_integrated:
            warnings.append("⚡ 已切换至独显模式，功耗和发热将显著增加")

        # ARM 架构警告
        plat = self._platform
        if plat and plat.machine in ("aarch64", "armv7l", "arm64"):
            warnings.append("📱 ARM 架构：bitsandbytes 量化不支持 ARM，需使用 CPU-only FP16 或导出 ONNX")

        return warnings


# ============================================================
# 辅助函数
# ============================================================

def _dataclass_to_dict(obj) -> dict:
    """递归将 dataclass 转为 dict"""
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            result[field_name] = _dataclass_to_dict(value)
        return result
    elif isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, enum.Enum):
        return obj.value
    else:
        return obj


# ============================================================
# 便捷函数
# ============================================================

_profile_cache: Optional[DeviceProfiler] = None


def get_profile() -> DeviceProfiler:
    """获取设备画像（全局单例，首次调用时检测）"""
    global _profile_cache
    if _profile_cache is None:
        _profile_cache = DeviceProfiler()
        logger.info(
            f"设备检测完成: tier={_profile_cache.tier.value} "
            f"score={_profile_cache.score:.1f}/100 "
            f"cpu={_profile_cache.cpu.physical_cores}核 "
            f"ram={_profile_cache.ram.total_gb}GB "
            f"gpu={_profile_cache.gpu.name}"
        )
    return _profile_cache


def reset_profile():
    """重置缓存画像（如硬件热插拔后）"""
    global _profile_cache
    _profile_cache = None


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    profiler = DeviceProfiler()
    print(profiler.to_json())
