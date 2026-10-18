"""keep-head shim 的唯一定位入口。

解析顺序：显式入参 > QLH_KEEP_HEAD_SHIM > 当前运行形态的默认路径。
源码运行使用仓库内构建产物；冻结运行使用包内独立的 keep_head 目录，
不回退到开发仓库。找不到时返回空串，由调用方给出具名错误。
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

ENV_VAR = "QLH_KEEP_HEAD_SHIM"

# shim 和配套 patched libllama/ggml 必须在同一目录，与 pip native 隔离。
DEFAULT_SHIM_RELATIVE = (
    Path("build") / "keephead" / "build-cpu" / "bin" / "qlh_keep_head.dll"
)
BUNDLED_SHIM_RELATIVE = Path("keep_head") / "qlh_keep_head.dll"


def default_shim_path() -> Path:
    """冻结态定位包内私有目录；源码态定位仓库默认构建产物。"""
    if getattr(sys, "frozen", False):
        bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        return bundle_root / BUNDLED_SHIM_RELATIVE
    return Path(__file__).resolve().parent.parent / DEFAULT_SHIM_RELATIVE


def resolve_keep_head_shim(explicit: str | None = None) -> str:
    """显式入参 > 环境变量 > 当前运行形态的默认路径；缺失返回空串。"""
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    from_env = os.environ.get(ENV_VAR, "").strip()
    if from_env:
        return from_env
    candidate = default_shim_path()
    if candidate.is_file():
        return str(candidate)
    return ""


def missing_shim_hint() -> str:
    """缺 shim 时列出环境变量和当前运行形态实际查找的路径。"""
    return (
        f"未找到 keep-head shim：已尝试 环境变量 {ENV_VAR} 与默认路径 "
        f"{default_shim_path()}；请构建 shim 或显式设置该环境变量"
    )
