"""整模推理路径选择与能力声明（票 15a `INFER-PATH-SELECT-01`）。

**裁决**（2026-10-12）：

    绑定的 ngram 主要是给边缘设备用的，所以应该有 CUDA 走子进程，
    没 CUDA 老实走绑定。

**为什么必须有这个选择器**

* 绑定 wheel（`llama-cpp-python`）实测是 **CPU-only**（`llama_cpp/lib/` 下没有
  `ggml-cuda.dll`，加载后 `vram` 几乎不动）=> **要 GPU 就必须走 server**；
* server 子进程拿得到 `--spec-type` 投机族、`--cpu-moe`（实测 35B-A3B 上
  `-ngl 99 --cpu-moe` = 15.94 tok/s，是纯 CPU 的 2.66x，显存只 2863 MiB）、
  `--cache-reuse` 等绑定给不了的能力；
* 但 **server 只能整模推理**：层段/keep-head 前向依赖低层 C API 与本仓
  llama.cpp 补丁（`t_layer_inp` 槽位）=> **层段接力永远走绑定**。

**两条硬规矩**（本模块的全部价值所在）

1. **层段接力强制绑定**，与有没有 CUDA 无关 —— 守卫，防止"为了统一把层段塞进
   server"而悄悄断掉接力；
2. 选择**绝不静默降级**：显式请求 server 而没有 CUDA（或没有可执行文件）=> 具名
   报错，不是"那就用绑定吧"。`auto` 允许降级，但原因会写进 `reason`（可观测），
   调用方据此记 warning。
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger("inference_path_selector")

SERVER = "llama_server"
BINDING = "llama_cpp_binding"
AUTO = "auto"

VALID_REQUESTS = (AUTO, SERVER, BINDING)


class InferencePathError(RuntimeError):
    """路径选择被具名拒绝（绝不静默降级）。"""


@dataclass(frozen=True)
class InferencePathDecision:
    """一次路径选择的结论；`to_capability()` 供能力广告投影。"""

    path: str
    reason: str
    cuda_available: bool
    requested: str

    def to_capability(self) -> dict:
        """投影成能力声明。

        `serves_layer_segments` 只对绑定路径为真 —— **server 路径绝不得宣称
        承担层段**，否则容量求解与路由会误判这台机器"能做什么"。
        """
        return {
            "inference_path": self.path,
            "cuda_available": bool(self.cuda_available),
            "serves_layer_segments": self.path == BINDING,
            "reason": str(self.reason),
        }


def detect_cuda_available() -> bool:
    """实时探测本机是否有可用的 CUDA device。

    ⚠️ **不能只看 `torch.cuda.is_available()`** —— 实测本仓 `.venv-test` 装的是
    **CPU-only torch**（`2.13.0+cpu`），于是明明有 RTX 4060 也返回 False，`auto`
    会永远选不到 server 路径。硬件级判据优先：`nvidia-smi` 能列出 device 即认为
    可用 —— 这也正是 llama-server 能否用 GPU 的真正前提（它自带 `ggml-cuda.dll`，
    不依赖 Python 侧 torch 的编译变体）。
    """
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
            # 中文 Windows 下 nvidia-smi 输出按 GBK 解可能崩（历史教训）
            encoding="utf-8", errors="replace",
        )
        if completed.returncode == 0 and (completed.stdout or "").strip():
            return True
    except Exception:  # noqa: BLE001 - 没有 nvidia-smi 就是没有 CUDA device
        pass
    try:
        import torch_runtime as _tr  # noqa: PLC0415 - 惰性，避免硬依赖

        return bool(_tr.cuda_available(load=False))
    except Exception:  # noqa: BLE001 - 探测失败按"无 CUDA"处理（最保守）
        return False


def select_inference_path(
    *,
    requested: str = AUTO,
    cuda_available: bool | None = None,
    layer_segment_required: bool = False,
    server_available: bool = True,
) -> InferencePathDecision:
    """选择整模推理路径。

    Args:
        requested: `auto` / `llama_server` / `llama_cpp_binding`。
        cuda_available: 注入用（测试）；`None` 时走实时探测。
        layer_segment_required: 该调用是否需要层段接力能力（守卫入口）。
        server_available: llama-server 可执行文件是否就位。只影响 `auto` 的降级：
            **显式**请求 server 时永远 fail-closed，不看这个开关。
    """
    normalized = (requested or AUTO).strip().lower() or AUTO
    if normalized not in VALID_REQUESTS:
        raise InferencePathError(
            f"未知的推理路径请求：{requested!r}（可用：{'/'.join(VALID_REQUESTS)}）",
        )
    if cuda_available is None:
        cuda_available = detect_cuda_available()
    cuda = bool(cuda_available)

    if layer_segment_required:
        if normalized == SERVER:
            raise InferencePathError(
                "层段接力必须走绑定路径：llama-server 仅承担整模推理，"
                "层段/keep-head 前向依赖低层 C API 与本仓 llama.cpp 补丁"
                "（t_layer_inp 槽位），server 进程给不了这种能力",
            )
        return InferencePathDecision(
            path=BINDING,
            reason="层段接力强制绑定路径（与是否有 CUDA 无关）",
            cuda_available=cuda,
            requested=normalized,
        )

    if normalized == SERVER:
        if not cuda:
            raise InferencePathError(
                "请求了 server 路径但本机没有可用的 CUDA：绑定 wheel 是 CPU-only，"
                f"无 CUDA 时 server 路径无意义；请改用 requested={BINDING}",
            )
        if not server_available:
            raise InferencePathError(
                "显式请求了 server 路径但找不到 llama-server 可执行文件："
                "请设置 QLH_LLAMA_SERVER_EXE，或把构建放到 runtime/llama-cpp/<build>/ 下。"
                f"（显式请求不降级；若要自动挑选请用 requested={AUTO}）",
            )
        return InferencePathDecision(
            path=SERVER,
            reason="显式请求 server 路径，且 CUDA 与可执行文件均可用",
            cuda_available=cuda,
            requested=normalized,
        )

    if normalized == BINDING:
        return InferencePathDecision(
            path=BINDING,
            reason="显式请求绑定路径（例如做数值对照或边缘部署）",
            cuda_available=cuda,
            requested=normalized,
        )

    # auto：只在「llama.cpp 家族内」挑传输。有 CUDA 走 server（绑定是 CPU-only，
    # 拿不到 GPU）；没有 CUDA 老实走绑定。server 不可用时**可观测地**退回绑定
    # —— 不静默：reason 里写明原因，调用方据此记 warning。
    if cuda and server_available:
        return InferencePathDecision(
            path=SERVER,
            reason="auto：检测到 CUDA 且 llama-server 可用，走 server 子进程"
                   "（可用 --cpu-moe 与投机族）",
            cuda_available=cuda,
            requested=normalized,
        )
    if cuda and not server_available:
        return InferencePathDecision(
            path=BINDING,
            reason="auto：检测到 CUDA 但找不到 llama-server 可执行文件，退回绑定"
                   "（可观测降级；设置 QLH_LLAMA_SERVER_EXE 可启用 server 路径）",
            cuda_available=cuda,
            requested=normalized,
        )
    return InferencePathDecision(
        path=BINDING,
        reason="auto：没有 CUDA，走绑定（边缘路径，可用 n-gram 自投机）",
        cuda_available=cuda,
        requested=normalized,
    )


__all__ = [
    "AUTO",
    "BINDING",
    "SERVER",
    "InferencePathDecision",
    "InferencePathError",
    "detect_cuda_available",
    "select_inference_path",
]
