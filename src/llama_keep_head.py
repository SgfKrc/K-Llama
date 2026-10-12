"""keep-head 上游：用**带补丁**的 llama.cpp 取「前 K 层输出」（`output_norm` 之前）。

为什么需要（P1 的否证 + 用户裁定 C 路线）：
`pip` 绑定的 `llama_get_embeddings_ith` 返回的是 **`output_norm(H)`**
（证据 `scripts/relay_diag_head_norm.py`：`rel_err=0.0018` / `cos=0.999998`），
比层接力上游所需的 hidden 多一次归一化 ⇒ L 段无法当上游/中间段。

项目已有补丁 `scripts/model_tools/patches/llama-cpp-layer-forward-api.patch`
（登记在 `scripts/model_tools/llama_quantize.lock.json`，marker `★ QLH 2026-09-20`），
把 `src/llama-context.cpp` 里**早已实现但未导出**的 6 个 API 补进 `include/llama.h`：

* `llama_set_embeddings_layer_inp(ctx, lid, enable)` / `llama_get_embeddings_layer_inp(ctx, lid)`
  —— **第 lid 层的输入**（＝第 lid-1 层的输出，残差流，未经任何 norm）；
* `llama_set_embeddings_nextn(ctx, masked)` / `llama_get_embeddings_nextn_ith(ctx, i)`
  —— **末层输出（`output_norm` 之前）**，按 token 稠密存放（`masked=false`）。

Python 侧**不直接 ctypes 调 libllama**：`llama_context_params` 是按值传参的大结构体，
用 PyPI 绑定的声明去调自建 DLL 会字段错位（实测 `Unsupported ctx type`）。
因此走 C shim `scripts/model_tools/keep_head_shim/qlh_keep_head.c`（与被调 DLL 同一份
`include/llama.h` 编译，Python 侧只剩 4 个平凡签名）。

两种模式：

* ``mode="nextn"``（默认）：配 **head 裁层工件**（保留 `blk.0..K-1`）⇒ 只跑 K 层，
  末层输出即 relay 上游该交的 hidden；
* ``mode="layer_inp"``：配**整模工件** + `cut_layer=K`，取第 K 层输入（语义等价，
  但会跑满全部层，只适合数值对照）。

⚠️ **层输出的实现通道（2026-09-23 起）**：上面两个模式现在都走 llama.cpp 的 `layer_inp`
通道（shim 的 `qlh_kh_load` 里 `mode 0` ⇒ `lid = n_layer`，`mode 1` ⇒ `lid = cut_layer`）。
`lid == n_layer` 是「第 n_layer 层的输入」= **末层输出（`output_norm` 之前）**，由 llama.cpp
侧多分配一个槽位实现（见 `scripts/model_tools/patches/llama-cpp-layer-forward-api.patch`）。
为什么不能继续用 `llama_set_embeddings_nextn`：各架构的 `t_h_nextn` 挂点不同（qwen2 在
`output_norm` **之前**，qwen35 在**之后** —— 后者多一次 RMSNorm，实测会让 9B 接力首步分叉）。
**目前登记该槽位的架构只有 `qwen2` 与 `qwen35`**；其他架构会因槽位为空而在 decode 时触发
GGML_ASSERT（fail-closed，不会静默给出错值）。

⚠️ 运行时依赖：shim 由 MinGW 构建 ⇒ 除同目录的 `libllama.dll` / `ggml*.dll` 外还需要
`libgcc_s_seh-1.dll` / `libstdc++-6.dll` / `libwinpthread-1.dll` 与 MSYS 的
`api-ms-win-crt-*` 副本。本模块会依次把「shim 所在目录」与
`QLH_KEEP_HEAD_DLL_DIRS`（`os.pathsep` 分隔）加入 DLL 搜索路径。
"""

from __future__ import annotations

import base64
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

#: 允许额外 DLL 搜索目录（例如 `C:\msys64\ucrt64\bin`），`os.pathsep` 分隔。
EXTRA_DLL_DIRS_ENV = "QLH_KEEP_HEAD_DLL_DIRS"

#: shim 必须导出的符号（缺任何一个都说明编译产物不对）。
SHIM_SYMBOLS = ("qlh_kh_load", "qlh_kh_forward", "qlh_kh_forward_embd",
                "qlh_kh_forward_embd_seq", "qlh_kh_reset", "qlh_kh_n_embd",
                "qlh_kh_n_layer", "qlh_kh_close")

#: `qlh_kh_forward*` 的错误码 → 说明。
FORWARD_ERRORS = {
    -1: "参数非法（句柄 / tokens / 输出缓冲）",
    -2: "llama_decode 失败",
    -3: "llama_get_embeddings_layer_inp 返回空（该槽位导出没打开，或架构没登记该槽位）",
    -4: "保留（旧 nextn 通道已停用：现统一走 layer_inp 的 n_layer 槽位）",
    -5: "参数非法（句柄 / embd / 输出缓冲）",
}

MODE_CODES = {"nextn": 0, "layer_inp": 1}


def ctx_per_seq_for(n_ctx: int, n_seq_max: int) -> int:
    """★ 2026-09-24：**每序列可用 ctx** = `n_ctx // n_seq_max`（`n_seq_max <= 0` 视为 1）。

    llama.cpp 把 `n_ctx` 按 `n_seq_max` **均分**；单序列长 decode 一旦超过这个预算，就会在
    **中途**报 "failed to find a memory slot for batch"（`llama_decode rc=1`）—— 症状是
    `runner_failed`，与"模型算错"难以区分（实测：`n_ctx=2048 / n_seq_max=8` ⇒ 单序列仅 256
    槽位，decode 到第 256 步即失败；`4096/8=512` ⇒ ~481 步；`8192/8=1024` ⇒ ~1000 步）。
    抽成纯函数：既可单测，也便于调用方在启动前校验自己的 `prefill + gen` 预算。
    """
    return max(0, int(n_ctx)) // max(1, int(n_seq_max))


class KeepHeadUnavailable(RuntimeError):
    """shim 缺失、符号不对或加载/初始化失败（都属于「不能用」，绝不降级成 embeddings 通道）。"""


def _add_dll_dirs(shim_dir: Path, extra: Sequence[str] = ()) -> tuple[list[str], list[object]]:
    dirs: list[str] = [str(shim_dir)]
    dirs.extend(extra)
    dirs.extend(d for d in os.environ.get(EXTRA_DLL_DIRS_ENV, "").split(os.pathsep) if d)
    added: list[str] = []
    handles: list[object] = []
    for candidate in dirs:
        if not candidate or candidate in added or not Path(candidate).is_dir():
            continue
        try:
            handle = os.add_dll_directory(candidate)
        except (AttributeError, OSError):
            continue
        added.append(candidate)
        handles.append(handle)
    return added, handles


def _llama_cpp_loaded() -> bool:
    return any(name == "llama_cpp" or name.startswith("llama_cpp.")
               for name in sys.modules)


def _shim_abi_collides_with_llama_cpp(shim_dir: Path) -> bool:
    """shim 所在目录是否带**与 pip llama_cpp 同名**的 ggml DLL（⇒ 必须隔离进程）。

    ★ 2026-09-23：这是 `WinError 127` 的根因所在，别退回「只在 llama_cpp 已加载时隔离」。

    `qlh_keep_head.dll` 经 `libllama.dll` 依赖 **按 basename** 解析的 `ggml-base.dll` /
    `ggml.dll`；而 pip 的 `llama_cpp/lib` 用的是**同名**文件但是**更新的** llama.cpp 构建。
    Windows 的 loader 对同一 basename 在**进程生命周期内只认第一个加载的模块**，
    且该绑定**不可撤销** —— `os.add_dll_directory` / `PATH` 之后怎么改都换不回来。

    实测（.venv-test，llama_cpp_python 0.3.35）：
      * `llama_cpp/lib/ggml-base.dll` 导出 `ggml_dsv4_hc_comb` / `_pre` / `_post` /
        `ggml_lightning_indexer`；
      * `build/keephead/build-cpu/bin/ggml-base.dll`（更早的构建）**不导出**这 4 个符号；
      * 只要 keep-head 的 ggml 先被加载，随后 `import llama_cpp.llama_cpp` 就在
        `CDLL(llama_cpp/lib/llama.dll)` 处抛
        `RuntimeError: ... [WinError 127] 找不到指定的程序`
        （xdist worker 更早一步：进程绑定期直接 `0xc0000139`＝STATUS_ENTRYPOINT_NOT_FOUND）。

    ⚠️ 因此隔离判据**不能**是「llama_cpp 是否已导入」—— 那个方向只覆盖了一半：
    llama_cpp 先导入 ⇒ 走 worker（安全）；keep-head 先导入 ⇒ 就地加载 shim（污染进程，
    后续 `tests/test_llama_relay_entry.py` 的 `import llama_cpp.llama_cpp` 必炸）。
    判据必须是「**同名依赖是否真的冲突**」这个**顺序无关**的事实。

    没有同名 ggml 的 shim 目录（例如只带 MinGW 运行时的目录）不冲突 ⇒ 允许就地加载。
    """
    for name in ("ggml-base.dll", "ggml.dll"):
        if (shim_dir / name).is_file():
            return True
    return False


class KeepHeadUpstream:
    """最小 keep-head 上游：吃 token → 吐「前 K 层输出」`[n_tokens, n_embd]` f32。

    只做上游该做的事：加载模型、按 token decode、导出 hidden。不提供 logits、不管采样，
    避免把实验路径膨胀成第二个引擎。
    """

    def __init__(
        self,
        shim_path: str | Path,
        model_path: str | Path,
        *,
        mode: str = "nextn",
        cut_layer: int | None = None,
        n_ctx: int = 4096,
        n_threads: int = 8,
        n_batch: int = 512,
        n_seq_max: int = 1,
        extra_dll_dirs: Sequence[str] = (),
        _worker_process: bool = False,
    ) -> None:
        if mode not in MODE_CODES:
            raise KeepHeadUnavailable(f"mode 必须是 {sorted(MODE_CODES)}，实得 {mode!r}")
        if mode == "layer_inp" and cut_layer is None:
            raise KeepHeadUnavailable("layer_inp 模式必须给 cut_layer")
        self.mode = mode
        self.cut_layer = None if cut_layer is None else int(cut_layer)
        self.shim_path = Path(shim_path)
        self.model_path = Path(model_path)
        if not self.shim_path.is_file():
            raise KeepHeadUnavailable(
                f"找不到 keep-head shim：{self.shim_path}（用 "
                "scripts/model_tools/build_keep_head_shim.ps1 生成）")
        if not self.model_path.is_file():
            raise KeepHeadUnavailable(f"找不到模型：{self.model_path}")

        self._worker = None
        self.n_seq_max = max(1, int(n_seq_max))
        self.n_batch = max(1, int(n_batch))
        # ★ 2026-09-24：**每序列可用 ctx** —— llama.cpp 把 `n_ctx` 按 `n_seq_max` **均分**，
        #   单序列长 decode 一旦超过 `n_ctx / n_seq_max` 就会在中途报 llama.cpp 的
        #   "failed to find a memory slot for batch"（`llama_decode rc=1`），现象是 `runner_failed`，
        #   与"模型算错"难以区分（实测：tail 段 `n_ctx=2048 / n_seq_max=8` ⇒ 单序列仅 **256** 槽位，
        #   decode 到第 256 步即失败）。这里把不变式**显式暴露**出来，调用方应据此校验自己的
        #   `prefill + gen` 预算；`n_seq_max > 1` 时额外给一行告警。
        self.ctx_per_seq = ctx_per_seq_for(n_ctx, self.n_seq_max)
        if self.n_seq_max > 1:
            print(f"[keep-head][warn] n_seq_max={self.n_seq_max} ⇒ 每序列 ctx ≈ "
                  f"{self.ctx_per_seq}（n_ctx={n_ctx} 被均分）；单序列长 decode 请用 n_seq_max=1 "
                  f"或把 n_ctx 放大到 {self.n_seq_max} 倍", file=sys.stderr, flush=True)
        # ★ 隔离判据必须**顺序无关**（2026-09-23 修 `WinError 127`）：
        #   旧代码只判 `_llama_cpp_loaded()` ⇒ 「keep-head 先加载、llama_cpp 后导入」这一半
        #   会就地加载 shim，把 keep-head 的 ggml-base.dll/ggml.dll 永久绑进进程，
        #   后续 `import llama_cpp.llama_cpp` 便以 WinError 127 失败（详见
        #   `_shim_abi_collides_with_llama_cpp` 的实测说明）。
        #   现在：同名的 shim 目录**一律**走独立 worker（两个方向都安全）；
        #   llama_cpp 已导入时也仍然走 worker（保持原有行为与理由）。
        #   ⚠️ 但 worker 进程自身必须**就地**加载 shim —— 它就是隔离边界，再隔离就是递归。
        shim_dir = self.shim_path.parent
        needs_isolation = (_llama_cpp_loaded()
                           or _shim_abi_collides_with_llama_cpp(shim_dir))
        if needs_isolation and not _worker_process:
            self._init_isolated_worker(
                n_ctx=n_ctx, n_threads=n_threads, n_batch=n_batch,
                n_seq_max=self.n_seq_max,
                extra_dll_dirs=extra_dll_dirs)
            return

        self._dll_dirs, self._dll_dir_handles = _add_dll_dirs(
            self.shim_path.parent, extra_dll_dirs)
        try:
            self._lib = ctypes.CDLL(str(self.shim_path))
        except OSError as exc:  # noqa: PERF203
            raise KeepHeadUnavailable(
                f"加载 {self.shim_path} 失败：{exc}；若缺 MinGW 运行时，请把 MSYS 的 bin "
                f"目录放进 {EXTRA_DLL_DIRS_ENV}") from exc

        missing = [name for name in SHIM_SYMBOLS if not hasattr(self._lib, name)]
        if missing:
            raise KeepHeadUnavailable(f"{self.shim_path} 缺符号 {missing}")

        lib = self._lib
        lib.qlh_kh_load.argtypes = [
            ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.c_char_p, ctypes.c_size_t,
        ]
        lib.qlh_kh_load.restype = ctypes.c_void_p
        lib.qlh_kh_forward.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
            ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
        ]
        lib.qlh_kh_forward.restype = ctypes.c_int32
        lib.qlh_kh_forward_embd_seq.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
        ]
        lib.qlh_kh_forward_embd_seq.restype = ctypes.c_int32
        # ★ P4.5 末段能力（可选符号）：**必须显式设 argtypes** —— 否则 ctypes 把 64 位句柄按
        #   `c_int` 处理 ⇒ `OverflowError: int too long to convert`（实测踩到）。
        #   旧版 shim 没有这个符号，因此只在存在时设置，保持向后兼容。
        if hasattr(lib, "qlh_kh_forward_embd_token"):
            lib.qlh_kh_forward_embd_token.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ]
            lib.qlh_kh_forward_embd_token.restype = ctypes.c_int32
        lib.qlh_kh_forward_embd.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
            ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
        ]
        lib.qlh_kh_forward_embd.restype = ctypes.c_int32
        lib.qlh_kh_n_embd.argtypes = [ctypes.c_void_p]
        lib.qlh_kh_n_embd.restype = ctypes.c_int32
        lib.qlh_kh_n_layer.argtypes = [ctypes.c_void_p]
        lib.qlh_kh_n_layer.restype = ctypes.c_int32
        lib.qlh_kh_close.argtypes = [ctypes.c_void_p]
        lib.qlh_kh_close.restype = None
        lib.qlh_kh_reset.argtypes = [ctypes.c_void_p]
        lib.qlh_kh_reset.restype = None
        # ★ 2026-09-24：**可选**符号 —— 透出最近一次 `llama_decode` 的原始 rc（旧 shim 无此符号）。
        if hasattr(lib, "qlh_kh_last_error"):
            lib.qlh_kh_last_error.argtypes = [ctypes.c_void_p]
            lib.qlh_kh_last_error.restype = ctypes.c_int32

        n_embd_out = ctypes.c_int32(0)
        n_layer_out = ctypes.c_int32(0)
        err = ctypes.create_string_buffer(512)
        handle = lib.qlh_kh_load(
            str(self.model_path).encode("utf-8"), int(n_ctx), int(n_threads), int(n_batch),
            int(self.n_seq_max),
            MODE_CODES[mode], int(cut_layer or 0),
            ctypes.byref(n_embd_out), ctypes.byref(n_layer_out), err, len(err))
        if not handle:
            message = err.value.decode("utf-8", "replace") or "未知错误"
            raise KeepHeadUnavailable(f"keep-head 初始化失败：{message}")
        self._handle = handle
        self.n_embd = int(n_embd_out.value)
        self.n_layer = int(n_layer_out.value)
        self.cut_layer = None if cut_layer is None else int(cut_layer)
        if self.n_embd <= 0:
            self.close()
            raise KeepHeadUnavailable("shim 报告 n_embd <= 0")

    # ------------------------------------------------------------------ 前向
    def _init_isolated_worker(self, *, n_ctx: int, n_threads: int, n_batch: int,
                              n_seq_max: int, extra_dll_dirs: Sequence[str]) -> None:
        """Keep the patched llama.cpp ABI out of the pip llama.cpp process."""
        worker = Path(__file__).with_name("llama_keep_head_worker.py")
        if not worker.is_file():
            raise KeepHeadUnavailable(f"keep-head worker missing: {worker}")
        # ★ 冻结包（PyInstaller）内 `sys.executable` 是**主 exe 自己**，不是 Python
        #   解释器 ⇒ 不能再写 `[sys.executable, worker.py, ...]`：那只会让 exe 重跑
        #   自己的入口，worker 的 stdin/stdout 协议永不建立（2026-10-06 用 Edge 包当
        #   layer worker 实测踩到：子进程在、keephead-worker.log 为 0 字节、master 等满
        #   60s 超时）。冻结态改走 launcher 的 `--keephead-worker` 早期分派，把同一份
        #   worker 脚本当 __main__ 跑；源码态保持原样。
        if getattr(sys, "frozen", False):
            worker_command = [sys.executable, "--keephead-worker"]
        else:
            worker_command = [sys.executable, str(worker)]
        command = [
            *worker_command,
            "--shim", str(self.shim_path), "--model", str(self.model_path),
            "--mode", self.mode, "--n-ctx", str(int(n_ctx)),
            "--n-threads", str(int(n_threads)), "--n-batch", str(int(n_batch)),
            "--n-seq-max", str(int(n_seq_max)),
        ]
        if self.cut_layer is not None:
            command.extend(["--cut-layer", str(self.cut_layer)])
        for dll_dir in extra_dll_dirs:
            command.extend(["--dll-dir", str(dll_dir)])
        popen_kwargs: dict[str, object] = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        # worker 的 stderr 必须落盘：llama.cpp 的日志量很大，用 PIPE 会填满缓冲而死锁，
        # 用 DEVNULL 则失败时无信息。落盘后可随时 tail 诊断。
        log_path = Path(os.environ.get("QLH_KEEP_HEAD_WORKER_LOG")
                        or (Path.cwd() / "build" / "relay-records" / "_keephead_worker.err"))
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_target: object = open(log_path, "w", encoding="utf-8", errors="replace")
        except OSError:
            stderr_target = subprocess.DEVNULL
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=stderr_target, text=True, encoding="utf-8",
                bufsize=1, **popen_kwargs)
        except OSError as exc:
            raise KeepHeadUnavailable(f"keep-head worker start failed: {exc}") from exc
        self._worker = process
        self._worker_log = log_path
        try:
            response = self._read_worker_response()
            if not response.get("ok"):
                raise KeepHeadUnavailable(
                    f"{response.get('error') or 'worker init failed'}"
                    f"（worker stderr: {log_path}）")
            self.n_embd = int(response["n_embd"])
            self.n_layer = int(response["n_layer"])
        except Exception:
            self.close()
            raise
        self._dll_dirs = []
        self._dll_dir_handles = []
        self._lib = None
        self._handle = None

    def _read_worker_response(self) -> dict[str, object]:
        process = self._worker
        if process is None or process.poll() is not None or process.stdout is None:
            raise KeepHeadUnavailable("keep-head worker unavailable")
        line = process.stdout.readline()
        if not line:
            raise KeepHeadUnavailable("keep-head worker exited before response")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise KeepHeadUnavailable("keep-head worker returned invalid response") from exc
        if not isinstance(value, dict):
            raise KeepHeadUnavailable("keep-head worker response is not an object")
        return value

    @property
    def is_available(self) -> bool:
        if self._worker is not None:
            return self._worker.poll() is None
        return self._lib is not None and self._handle is not None

    def _worker_request(self, payload: dict[str, object]) -> dict[str, object]:
        process = self._worker
        if process is None or process.poll() is not None or process.stdin is None:
            raise KeepHeadUnavailable("keep-head worker unavailable")
        try:
            process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            process.stdin.flush()
            response = self._read_worker_response()
        except (OSError, BrokenPipeError) as exc:
            raise KeepHeadUnavailable(f"keep-head worker communication failed: {exc}") from exc
        if not response.get("ok"):
            raise KeepHeadUnavailable(str(response.get("error") or "worker request failed"))
        return response

    @staticmethod
    def _worker_array(response: dict[str, object]):
        import numpy as np

        try:
            shape = tuple(int(value) for value in response["shape"])
            raw = base64.b64decode(str(response["data"]))
            array = np.frombuffer(raw, dtype=np.float32)
            return array.reshape(shape).copy()
        except (KeyError, TypeError, ValueError) as exc:
            raise KeepHeadUnavailable("keep-head worker returned invalid hidden") from exc

    def last_decode_error(self) -> int:
        """★ 2026-09-24：最近一次 `llama_decode` 的**原始**返回码（0 = 无错误，或旧 shim）。

        与 `qlh_kh_forward*` 的稳定返回码（-2 = "decode 失败"）不同，这里给的是 **llama.cpp 自己**
        的错误码 —— 长时 decode 在特定 ctx 下失败时，只有它能区分"KV/ctx 相关"与"参数/状态相关"。
        旧版 shim 没有 `qlh_kh_last_error` ⇒ 返回 0（向后兼容）。
        """
        if self._lib is None or self._handle is None:
            return 0
        if not hasattr(self._lib, "qlh_kh_last_error"):
            return 0
        try:
            return int(self._lib.qlh_kh_last_error(self._handle))
        except Exception:  # noqa: BLE001 - 诊断信息绝不能变成新的失败点
            return 0

    def _decode_error_suffix(self) -> str:
        """把 `last_decode_error()` 渲染成可读后缀（为 0 时返回空串）。"""
        value = self.last_decode_error()
        return f"（shim 记录 llama_decode rc={value}）" if value else ""

    def forward_tokens_to_hidden(self, tokens: Sequence[int], *, n_past: int = 0):
        """跑模型（`nextn` 模式即前 K 层），返回 `[n_tokens, n_embd]` 的 f32 hidden。"""
        import numpy as np

        toks = [int(t) for t in tokens]
        n_tokens = len(toks)
        if n_tokens == 0:
            raise ValueError("tokens 不能为空")
        if n_tokens > self.n_batch:
            raise KeepHeadUnavailable(
                f"keep-head batch exceeds n_batch: tokens={n_tokens} limit={self.n_batch}"
            )
        if n_past < 0 or n_past + n_tokens > self.ctx_per_seq:
            raise KeepHeadUnavailable(
                f"keep-head context exceeds per-sequence limit: "
                f"past={n_past} tokens={n_tokens} limit={self.ctx_per_seq}"
            )
        if self._worker is not None:
            return self._worker_array(self._worker_request({
                "op": "tokens", "tokens": toks, "n_past": int(n_past)}))
        tokens_arr = (ctypes.c_int32 * n_tokens)(*toks)
        out = np.zeros((n_tokens, self.n_embd), dtype=np.float32)
        rc = self._lib.qlh_kh_forward(
            self._handle, tokens_arr, n_tokens, int(n_past),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if rc != 0:
            raise KeepHeadUnavailable(
                f"keep-head 前向失败 rc={rc}：{FORWARD_ERRORS.get(rc, '未知错误码')}"
                f"{self._decode_error_suffix()}")
        return out

    def forward_hidden_to_token(self, hidden, *, n_past: int = 0,
                                seq_ids: Sequence[int] | None = None,
                                positions: Sequence[int] | None = None) -> int:
        """★ **末段能力（P4.5 退化路径）**：吃上游 hidden（`embd` 注入）→ 吐**末位 argmax**。

        用途：集群里可能没有任何能跑 torch 的节点（无 PC 边缘集群），此时层接力必须全部由
        llama.cpp 承载 —— 末段就得能"吃 hidden 出 token"。本入口与 `forward_hidden_to_hidden`
        共用同一条 shim decode 路径，只把输出换成 argmax ⇒ **数值口径天然对齐**
        （同一份 C 源、同一份 llama.cpp，不引入第二个 llama.cpp 版本）。

        ⚠️ **可选能力**：需要 shim 带 `qlh_kh_forward_embd_token`（P4.5 新增）。旧版 shim 只缺
        这一个符号，不影响上游/中间段；缺失时给明确错误，绝不静默降级。
        """
        import numpy as np

        # ⚠️ worker 隔离路径下符号在**子进程**里（`self._lib` 为 None）⇒ 本检查只对就地加载有意义。
        #    漏掉这个前置条件会把「worker 模式」误判成「缺符号」（实测踩到：主进程一旦 import 过
        #    llama_cpp / 或 shim 目录带同名 ggml，就会走 worker，然后这里必然抛"缺符号"）。
        if self._worker is None and not hasattr(self._lib, "qlh_kh_forward_embd_token"):
            raise KeepHeadUnavailable(
                f"{self.shim_path} 缺 qlh_kh_forward_embd_token（末段能力）"
                "—— 需用含 P4.5 入口的 shim 重新编译")

        arr = np.ascontiguousarray(np.asarray(hidden, dtype=np.float32))
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != self.n_embd:
            raise ValueError(f"hidden 形状应为 [n_tokens, {self.n_embd}]，实得 {arr.shape}")
        n_tokens = int(arr.shape[0])
        if n_tokens == 0:
            raise ValueError("hidden 的 token 数不能为 0")
        seq_list = self._token_list(seq_ids, n_tokens, "seq_ids")
        pos_list = self._token_list(positions, n_tokens, "positions")
        if seq_list is not None and max(set(seq_list)) >= self.n_seq_max:
            raise ValueError(
                f"seq_ids 最大 {max(set(seq_list))} ≥ n_seq_max {self.n_seq_max}；"
                "多序列必须用 n_seq_max 构造 KeepHeadUpstream")

        if self._worker is not None:
            payload = {
                "op": "hidden_token", "shape": list(arr.shape),
                "data": base64.b64encode(arr.tobytes()).decode("ascii"),
                "n_past": int(n_past),
            }
            if seq_list is not None:
                payload["seq_ids"] = seq_list
            if pos_list is not None:
                payload["positions"] = pos_list
            response = self._worker_request(payload)
            token = int(response.get("token", -1))
            if token < 0:
                raise KeepHeadUnavailable(
                    f"keep-head 末段前向失败 rc={response.get('rc')}："
                    f"{FORWARD_ERRORS.get(response.get('rc'), '未知错误码')}")
            return token

        out_token = ctypes.c_int32(-1)
        rc = self._lib.qlh_kh_forward_embd_token(
            self._handle, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_tokens, int(n_past),
            None,
            (ctypes.c_int32 * n_tokens)(*seq_list) if seq_list is not None else None,
            (ctypes.c_int32 * n_tokens)(*pos_list) if pos_list is not None else None,
            ctypes.byref(out_token))
        if rc != 0:
            raise KeepHeadUnavailable(
                f"keep-head 末段前向失败 rc={rc}：{FORWARD_ERRORS.get(rc, '未知错误码')}"
                f"{self._decode_error_suffix()}")
        return int(out_token.value)

    def forward_hidden_to_hidden(self, hidden, *, n_past: int = 0,
                                 seq_ids: Sequence[int] | None = None,
                                 positions: Sequence[int] | None = None):
        """★ **中间段能力**：吃上游 hidden（`embd` 注入）→ 吐本段的 hidden。

        这是「1 个 torch 上游 + n 个 llama 下游」链式拼接的关键：中间的 llama 段必须能
        既接受上游 hidden 又交出 hidden。

        ★ P3 多序列数据流：`seq_ids` / `positions` 与
        `llama_engine.forward_layers_from_hidden()` **同一契约** —— 都给则逐 token 显式绑定
        （多序列交错推进时必须显式给）；都省则退化为单序列、位置自 `n_past` 起递增。
        多序列要求本实例以 `n_seq_max >= 序列数` 构造（见 `KeepHeadUpstream.__init__`）。
        """
        import numpy as np

        arr = np.ascontiguousarray(np.asarray(hidden, dtype=np.float32))
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != self.n_embd:
            raise ValueError(f"hidden 形状应为 [n_tokens, {self.n_embd}]，实得 {arr.shape}")
        n_tokens = int(arr.shape[0])
        if n_tokens == 0:
            raise ValueError("hidden 的 token 数不能为 0")
        seq_list = self._token_list(seq_ids, n_tokens, "seq_ids")
        pos_list = self._token_list(positions, n_tokens, "positions")
        if seq_list is not None:
            distinct = {value for value in seq_list}
            if max(distinct) >= self.n_seq_max:
                raise ValueError(
                    f"seq_ids 最大 {max(distinct)} ≥ n_seq_max {self.n_seq_max}；"
                    "多序列必须用 n_seq_max 构造 KeepHeadUpstream")

        if self._worker is not None:
            payload = {
                "op": "hidden", "shape": list(arr.shape),
                "data": base64.b64encode(arr.tobytes()).decode("ascii"),
                "n_past": int(n_past)}
            if seq_list is not None:
                payload["seq_ids"] = seq_list
            if pos_list is not None:
                payload["positions"] = pos_list
            return self._worker_array(self._worker_request(payload))

        out = np.zeros((n_tokens, self.n_embd), dtype=np.float32)
        rc = self._lib.qlh_kh_forward_embd_seq(
            self._handle, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_tokens, int(n_past),
            None,
            (ctypes.c_int32 * n_tokens)(*seq_list) if seq_list is not None else None,
            (ctypes.c_int32 * n_tokens)(*pos_list) if pos_list is not None else None,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if rc != 0:
            raise KeepHeadUnavailable(
                f"keep-head embd 前向失败 rc={rc}：{FORWARD_ERRORS.get(rc, '未知错误码')}"
                f"{self._decode_error_suffix()}")
        return out

    @staticmethod
    def _token_list(values: Sequence[int] | None, n_tokens: int,
                    field: str) -> list[int] | None:
        if values is None:
            return None
        result = [int(value) for value in values]
        if len(result) != n_tokens:
            raise ValueError(f"{field} 长度 {len(result)} != n_tokens {n_tokens}")
        return result

    # ------------------------------------------------------------------ 资源
    def reset(self) -> None:
        """★ P3：清空 KV / recurrent 记忆。

        跨机的中间段服务在**同一进程**里服务多条连接（`--max-connections`），新连接必须
        从干净的记忆开始 —— 否则上一条连接留下的位置会让本篇的 position 0 触发
        "tokens ... have inconsistent sequence positions"（远端表现为 `runner_failed`）。
        """
        if self._worker is not None:
            self._worker_request({"op": "reset"})
            return
        if getattr(self, "_handle", None) and getattr(self, "_lib", None) is not None:
            self._lib.qlh_kh_reset(self._handle)

    def close(self) -> None:
        process = getattr(self, "_worker", None)
        if process is not None:
            try:
                if process.poll() is None and process.stdin is not None:
                    process.stdin.write('{"op":"close"}\n')
                    process.stdin.flush()
                    self._read_worker_response()
            except (OSError, BrokenPipeError, KeepHeadUnavailable):
                pass
            finally:
                try:
                    if process.stdin is not None:
                        process.stdin.close()
                except OSError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                self._worker = None
        handle = getattr(self, "_handle", None)
        if handle and getattr(self, "_lib", None) is not None:
            self._lib.qlh_kh_close(handle)
            self._handle = None
        for dll_handle in reversed(getattr(self, "_dll_dir_handles", ())):
            try:
                dll_handle.close()
            except (AttributeError, OSError):
                pass
        if hasattr(self, "_dll_dir_handles"):
            self._dll_dir_handles.clear()

    def __enter__(self) -> "KeepHeadUpstream":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - 解释器退出时的兜底
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
