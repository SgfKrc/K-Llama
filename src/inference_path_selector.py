"""整模推理路径选择与能力声明（票 15a `INFER-PATH-SELECT-01`）。

**裁决**（2026-10-12）：

    绑定的 ngram 主要是给边缘设备用的，所以应该有 CUDA 走子进程，
    没 CUDA 老实走绑定。

**为什么必须有这个选择器**

* 绑定 wheel（`llama-cpp-python`）实测是 **CPU-only**（`llama_cpp/lib/` 下没有
  `ggml-cuda.dll`，加载后 `vram` 几乎不动）⇒ **要 GPU 就必须走 server**；
* server 子进程拿得到 `--spec-type` 投机族、`--cpu-moe`（实测 35B-A3B 上
  `-ngl 99 --cpu-moe` = 15.94 tok/s，是纯 CPU 的 2.66×，显存只 2863 MiB）、
  `--cache-reuse` 等绑定给不了的能力；
* 但 **server 只能整模推理**：层段/keep-head 前向依赖低层 C API 与本仓
  llama.cpp 补丁（`t_layer_inp` 槽位）⇒ **层段接力永远走绑定**。

**两条硬规矩**（本模块的全部价值所在）：

1. **层段接力强制绑定**，与有没有 CUDA 无关 —— 守卫，防止"为了统一把层段塞进
   server"而悄悄断掉接力；
2. 选择**绝不静默降级**：显式请求 server 而没有 CUDA ⇒ 具名报错，不是"那就用
   绑定吧"。同理，层段接力却请求 server 也具名报错。
"""

from __future__ import annotations

import logging
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
    """实时探测 CUDA（不用画像快照，与 `_decide_pipeline_load_mode` 同口径）。"""
    try:
        import torch_runtime as _tr  # noqa: PLC0415 - 惰性，避免无 CUDA 环境硬依赖

        return bool(_tr.cuda_available(load=False))
    except Exception:  # noqa: BLE001 - 探测失败按"无 CUDA"处理（最保守）
        return False


def select_inference_path(
    *,
    requested: str = AUTO,
    cuda_available: bool | None = None,
    layer_segment_required: bool = False,
) -> InferencePathDecision:
    """选择整模推理路径。

    Args:
        requested: `auto` / `llama_server` / `llama_cpp_binding`。
        cuda_available: 注入用（测试）；`None` 时走实时探测。
        layer_segment_required: 该调用是否需要层段接力能力（守卫入口）。
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
        return InferencePathDecision(
            path=SERVER,
            reason="显式请求 server 路径，且 CUDA 可用",
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

    if cuda:
        return InferencePathDecision(
            path=SERVER,
            reason="auto：检测到 CUDA ⇒ 走 server 子进程（可用 --cpu-moe 与投机族）",
            cuda_available=cuda,
            requested=normalized,
        )
    return InferencePathDecision(
        path=BINDING,
        reason="auto：没有 CUDA ⇒ 走绑定（边缘路径，可用 n-gram 自投机）",
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
