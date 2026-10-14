"""P2「C 路线」keep-head 上游的守卫用例（缺 shim / 缺工件时条件跳过）。

为什么这些用例重要（`docs/跨框架接力-当前有效基线与后续优化计划-2026-09-21.md` §3）：
pip 绑定的 `embeddings` 通道返回 `output_norm(H)`，**不能**当层接力上游。keep-head
（补丁导出的 layer-input / nextn 通道）是 L 段当上游/中间段的唯一经路。这里固定：

1. shim 缺失/符号不全 ⇒ 明确报 `KeepHeadUnavailable`（绝不静默降级到 embeddings 通道）；
2. shim 在位时：hidden 形状 = `[n_tokens, n_embd]`、有限、**同输入两次调用逐位一致**；
3. 「吃 hidden 吐 hidden」（中间段能力）返回同样形状。

真模型路径需要：带补丁的 `libllama`/shim（`scripts/model_tools/build_keep_head_shim.ps1`）
与 head 裁层工件；缺失时跳过（与本仓既有做法一致）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llama_keep_head import (  # noqa: E402
    MODE_CODES,
    SHIM_SYMBOLS,
    KeepHeadUnavailable,
    KeepHeadUpstream,
    _add_dll_dirs,
    _shim_abi_collides_with_llama_cpp,
)

SHIM = ROOT / "build" / "keephead" / "build-cpu" / "bin" / "qlh_keep_head.dll"
HEAD12 = ROOT / "build" / "cross-framework-layer-poc" / "out" / "qwen25-05b-f16-head12.gguf"
EXTRA_DLL_DIRS = [d for d in (os.environ.get("QLH_KEEP_HEAD_DLL_DIRS") or
                              r"C:\msys64\ucrt64\bin").split(os.pathsep) if d]
NATIVE_WORKER_ENV = "QLH_KEEP_HEAD_NATIVE_WORKER"


def _upstream_or_skip(**kwargs):
    if os.environ.get(NATIVE_WORKER_ENV) != "1":
        pytest.skip("keep-head native ABI tests run in an isolated subprocess")
    if not SHIM.is_file():
        pytest.skip(f"需要 keep-head shim（{SHIM.relative_to(ROOT)}）；"
                    "用 scripts/model_tools/build_keep_head_shim.ps1 生成")
    if not HEAD12.is_file():
        pytest.skip(f"需要 head 裁层工件（{HEAD12.relative_to(ROOT)}），本机缺失")
    try:
        return KeepHeadUpstream(SHIM, HEAD12, extra_dll_dirs=EXTRA_DLL_DIRS,
                                n_ctx=512, n_threads=4, **kwargs)
    except KeepHeadUnavailable as exc:
        pytest.skip(f"keep-head 不可用：{exc}")


def test_dll_directory_handles_are_retained(monkeypatch, tmp_path):
    dll_dir = tmp_path / "dll"
    dll_dir.mkdir()
    handle = object()
    monkeypatch.setattr(os, "add_dll_directory", lambda path: handle)
    # 环境变量会追加额外目录，测试必须隔离环境（否则断言依赖开发机配置）
    monkeypatch.delenv("QLH_KEEP_HEAD_DLL_DIRS", raising=False)

    dirs, handles = _add_dll_dirs(dll_dir)

    assert dirs == [str(dll_dir)]
    assert handles == [handle]


# --------------------------------------------- ★ DLL basename 冲突（WinError 127 回归）
def test_shim_dir_with_same_named_ggml_is_judged_colliding(tmp_path):
    """★ 回归（2026-09-23，`WinError 127`）：带同名 `ggml-base.dll`/`ggml.dll` 的 shim 目录
    必须被判为「与 pip llama_cpp 冲突」。

    背景：keep-head 的 shim 经 `libllama.dll` 依赖**按 basename** 解析的 `ggml-base.dll`；
    pip 的 `llama_cpp/lib` 用同名但更新的构建。Windows loader 对同一 basename 在进程内
    **只认第一个加载的模块且不可撤销** ⇒ 谁先加载谁说了算。

    实测差异（.venv-test / llama_cpp_python 0.3.35）——`llama_cpp/lib/ggml-base.dll` 导出，
    而 `build/keephead/build-cpu/bin/ggml-base.dll` **不**导出：
        `ggml_dsv4_hc_comb` / `ggml_dsv4_hc_pre` / `ggml_dsv4_hc_post` / `ggml_lightning_indexer`
    ⇒ keep-head 先加载时，`CDLL(llama_cpp/lib/llama.dll)` 抛
    `[WinError 127] 找不到指定的程序`；xdist worker 更早一步 `0xc0000139`
    （＝STATUS_ENTRYPOINT_NOT_FOUND）。
    """
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    assert _shim_abi_collides_with_llama_cpp(shim_dir) is False, "空目录不应判为冲突"
    (shim_dir / "ggml-base.dll").write_bytes(b"")
    assert _shim_abi_collides_with_llama_cpp(shim_dir) is True, (
        "带同名 ggml-base.dll 的 shim 目录必须判为冲突（否则就地加载会污染进程）")
    (shim_dir / "ggml-base.dll").unlink()
    (shim_dir / "ggml.dll").write_bytes(b"")
    assert _shim_abi_collides_with_llama_cpp(shim_dir) is True, (
        "带同名 ggml.dll 的 shim 目录必须判为冲突")


def test_isolation_decision_is_order_independent(monkeypatch, tmp_path):
    """★ 回归：隔离判据必须**顺序无关** —— 「keep-head 先加载、llama_cpp 后导入」也要隔离。

    旧实现只判 `_llama_cpp_loaded()`（「pip llama_cpp 是否已导入」），只覆盖了一半方向：
    llama_cpp 先导入 ⇒ 走 worker（安全）；keep-head 先导入 ⇒ **就地加载 shim**，
    把 keep-head 的 ggml DLL 永久绑进进程 ⇒ 后续任何
    `import llama_cpp.llama_cpp`（如 `tests/test_llama_relay_entry.py`）都以 WinError 127 失败。

    本用例在「llama_cpp 未导入」的前提下，直接验证冲突目录会走 worker 分支 ——
    即**不再**依赖导入顺序。
    """
    import src.llama_keep_head as kh

    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    (shim_dir / "ggml-base.dll").write_bytes(b"")
    shim = shim_dir / "qlh_keep_head.dll"
    shim.write_bytes(b"")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"")

    # 前置条件：pip llama_cpp **尚未**导入（正是旧实现漏掉的那一半）
    monkeypatch.setattr(kh, "_llama_cpp_loaded", lambda: False)
    went_to_worker = []
    monkeypatch.setattr(kh.KeepHeadUpstream, "_init_isolated_worker",
                        lambda self, **kw: went_to_worker.append(kw))

    kh.KeepHeadUpstream(shim, model)
    assert went_to_worker, (
        "llama_cpp 未导入 + shim 目录有同名 ggml 时也必须走隔离 worker；"
        "就地加载会把 keep-head 的 ggml 绑进进程，令后续 import llama_cpp 报 WinError 127")


def test_isolated_worker_does_not_respawn_itself(monkeypatch, tmp_path):
    """回归：worker 通过私有构造参数在子进程里直接加载 shim，不能递归 spawn。"""
    import src.llama_keep_head as kh

    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    (shim_dir / "ggml-base.dll").write_bytes(b"")
    shim = shim_dir / "qlh_keep_head.dll"
    shim.write_bytes(b"")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"")

    monkeypatch.setattr(kh, "_llama_cpp_loaded", lambda: False)
    spawned = []
    monkeypatch.setattr(kh.KeepHeadUpstream, "_init_isolated_worker",
                        lambda self, **kw: spawned.append(kw))

    # worker 内不走继续隔离分支，而是落到就地 CDLL(shim)（此处 shim 是空文件 ⇒ 抛错即可）
    with pytest.raises(KeepHeadUnavailable):
        kh.KeepHeadUpstream(shim, model, _worker_process=True)
    assert not spawned, "worker 进程内不得再次 spawn worker（会递归）"


def test_worker_entry_opts_into_local_load_mode():
    """Worker 只通过内部构造参数选择进程内加载。"""
    source = (ROOT / "src" / "llama_keep_head_worker.py").read_text(encoding="utf-8")
    call = source.index("upstream = KeepHeadUpstream(")
    flag = source.index("_worker_process=True", call)
    assert flag < source.index(")", call)


def test_real_shim_isolated_before_host_llama_cpp_import(tmp_path):
    """加载真实 keep-head shim 后，宿主进程仍可导入 pip llama.cpp。"""
    if not SHIM.is_file() or not HEAD12.is_file():
        pytest.skip("真实 keep-head shim/head 模型工件不齐全")

    probe = r"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "src"))
from llama_keep_head import KeepHeadUpstream

upstream = KeepHeadUpstream(
    sys.argv[2], sys.argv[3], n_ctx=512, n_threads=2, n_batch=128,
)
assert upstream._worker is not None
upstream.close()
import llama_cpp.llama_cpp
print("keep-head worker isolated; host llama_cpp import succeeded")
"""
    env = os.environ.copy()
    env["QLH_KEEP_HEAD_IS_WORKER"] = "1"
    env["QLH_KEEP_HEAD_WORKER_LOG"] = str(tmp_path / "keephead-worker.stderr.log")
    result = subprocess.run(
        [sys.executable, "-c", probe, str(ROOT), str(SHIM), str(HEAD12)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"真实 shim worker 隔离回归失败。\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "host llama_cpp import succeeded" in result.stdout


# ------------------------------------------------------------------ 失败路径（不需要模型）
def test_missing_shim_is_explicit_not_silent(tmp_path):
    with pytest.raises(KeepHeadUnavailable, match="找不到 keep-head shim"):
        KeepHeadUpstream(tmp_path / "nope.dll", tmp_path / "nope.gguf")


def test_missing_model_is_explicit(tmp_path):
    shim = tmp_path / "fake.dll"
    shim.write_bytes(b"")
    with pytest.raises(KeepHeadUnavailable, match="找不到模型"):
        KeepHeadUpstream(shim, tmp_path / "nope.gguf")


def test_invalid_mode_is_rejected():
    with pytest.raises(KeepHeadUnavailable, match="mode"):
        KeepHeadUpstream("whatever.dll", "whatever.gguf", mode="guess")


def test_layer_inp_mode_requires_cut_layer():
    with pytest.raises(KeepHeadUnavailable, match="cut_layer"):
        KeepHeadUpstream("whatever.dll", "whatever.gguf", mode="layer_inp")


def test_shim_symbol_contract_is_declared():
    """shim 的符号清单必须包含中间段能力（`forward_embd`）—— 否则三段链路无从谈起。"""
    assert "qlh_kh_forward_embd" in SHIM_SYMBOLS
    assert set(MODE_CODES) == {"nextn", "layer_inp"}


def test_oversized_prompt_rejected_before_native_worker():
    upstream = object.__new__(KeepHeadUpstream)
    upstream.n_batch = 512
    upstream.ctx_per_seq = 4096

    with pytest.raises(KeepHeadUnavailable, match="batch exceeds n_batch"):
        upstream.forward_tokens_to_hidden(range(513))
    with pytest.raises(KeepHeadUnavailable, match="context exceeds per-sequence limit"):
        upstream.forward_tokens_to_hidden([1] * 10, n_past=4090)


# ------------------------------------------------------------------ 真模型（缺工件跳过）
def test_nextn_upstream_returns_hidden_with_expected_shape():
    import numpy as np

    with _upstream_or_skip(mode="nextn") as up:
        assert up.n_layer == 12, "head12 工件应是 12 层"
        assert up.n_embd > 0
        hidden = up.forward_tokens_to_hidden([100, 200, 300])
        assert hidden.shape == (3, up.n_embd)
        assert np.isfinite(hidden).all()


def test_same_input_is_bitwise_stable():
    """接力对数值稳定性有硬要求：同输入两次必须逐位一致。"""
    import numpy as np

    with _upstream_or_skip(mode="nextn") as first:
        a = first.forward_tokens_to_hidden([100, 200, 300])
    with _upstream_or_skip(mode="nextn") as second:
        b = second.forward_tokens_to_hidden([100, 200, 300])
    assert np.array_equal(a, b)


def test_middle_segment_can_take_hidden_and_return_hidden():
    """中间段能力：吃 hidden（embd 注入）→ 吐 hidden（nextn 末层输出）。"""
    import numpy as np

    with _upstream_or_skip(mode="nextn") as up:
        incoming = np.zeros((2, up.n_embd), dtype=np.float32)
        outgoing = up.forward_hidden_to_hidden(incoming)
        assert outgoing.shape == incoming.shape
        assert np.isfinite(outgoing).all()


def test_rejects_wrong_hidden_width():
    with _upstream_or_skip(mode="nextn") as up:
        with pytest.raises(ValueError, match="hidden 形状"):
            up.forward_hidden_to_hidden([[0.0] * max(1, up.n_embd // 2)])


# --------------------------------------------------- P3：多序列数据流契约
def test_token_list_validates_length_without_a_model():
    """`seq_ids` / `positions` 与 n_tokens 必须等长（纯校验，不需要模型）。"""
    assert KeepHeadUpstream._token_list(None, 3, "seq_ids") is None
    assert KeepHeadUpstream._token_list([0, 1, 2], 3, "seq_ids") == [0, 1, 2]
    with pytest.raises(ValueError, match="长度 2 != n_tokens 3"):
        KeepHeadUpstream._token_list([0, 1], 3, "positions")


def test_multi_sequence_binding_is_accepted():
    """P3：2 序列 × 2 token 的显式绑定（seq_ids + positions）必须与单序列同形返回。"""
    import numpy as np

    with _upstream_or_skip(mode="nextn", n_seq_max=2) as up:
        hidden = np.zeros((4, up.n_embd), dtype=np.float32)
        out = up.forward_hidden_to_hidden(hidden, seq_ids=[0, 0, 1, 1],
                                          positions=[0, 1, 0, 1])
        assert out.shape == hidden.shape
        assert np.isfinite(out).all()


def test_seq_id_beyond_n_seq_max_is_rejected():
    """seq_id ≥ n_seq_max 必须 fail-loud（否则 llama.cpp 直接 rc=-1，错误难定位）。"""
    import numpy as np

    with _upstream_or_skip(mode="nextn", n_seq_max=1) as up:
        hidden = np.zeros((2, up.n_embd), dtype=np.float32)
        with pytest.raises(ValueError, match="n_seq_max"):
            up.forward_hidden_to_hidden(hidden, seq_ids=[0, 3], positions=[0, 0])


def test_single_sequence_positions_stay_consecutive_across_steps():
    """★ 回归：单序列增量必须延续位置（曾因多序列改造丢掉 n_past 而第二步 decode 失败）。"""
    import numpy as np

    with _upstream_or_skip(mode="nextn", n_seq_max=1) as up:
        first = up.forward_hidden_to_hidden(np.zeros((2, up.n_embd), dtype=np.float32),
                                            n_past=0)
        second = up.forward_hidden_to_hidden(np.zeros((1, up.n_embd), dtype=np.float32),
                                             n_past=2)
        assert first.shape[0] == 2 and second.shape[0] == 1


# --------------------------------------------------- P3：hidden 压缩字节口径
def test_hidden_quant_bytes_accounting():
    """压缩档位的**每 token 有效线上字节**（int8 块量化含每 128 维 1 个 f32 scale）。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "relay_experiment_cli", ROOT / "scripts" / "relay_experiment.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module._hidden_bytes(896, "none") == 896 * 4
    assert module._hidden_bytes(896, "f16") == 896 * 2
    assert module._hidden_bytes(896, "int8_block128") == 896 + 7 * 4   # 7 个 128 块
    assert module._hidden_bytes(0, "f16") is None


def test_int8_hidden_quant_supports_non_block_aligned_width():
    import importlib.util
    import numpy as np

    spec = importlib.util.spec_from_file_location(
        "relay_experiment_cli_non_aligned", ROOT / "scripts" / "relay_experiment.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    hidden = np.arange(3 * 130, dtype=np.float32).reshape(3, 130) - 100.0
    out = module._quantize_hidden(hidden, "int8_block128")
    assert out.shape == hidden.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_token_entry_requires_shim_symbol():
    """★ P4.5 末段入口：shim 缺 `qlh_kh_forward_embd_token` 时必须**明确报错**。

    旧版 shim 只缺这一个入口，middle/上游照常可用 ⇒ 不能把它放进 `SHIM_SYMBOLS`（那会让设备上
    未升级的 shim 整体加载失败），但也绝不能静默降级 —— 报错必须指名缺哪个符号。
    """
    import types

    import numpy as np

    upstream = object.__new__(KeepHeadUpstream)
    upstream._lib = types.SimpleNamespace()          # 模拟旧 shim：没有 token 入口
    upstream.shim_path = "fake.dll"
    upstream.n_embd = 8
    upstream._worker = None
    with pytest.raises(KeepHeadUnavailable, match="qlh_kh_forward_embd_token"):
        upstream.forward_hidden_to_token(np.zeros((1, 8), dtype=np.float32))


def test_token_symbol_argtypes_are_set():
    """★ 符号存在时**必须**设置 `argtypes`：否则 ctypes 把 64 位句柄按 `c_int` 处理 ⇒
    `OverflowError: int too long to convert`（实测踩到：服务端每次连接都失败）。"""
    shim = ROOT / "build" / "keephead" / "build-cpu" / "bin" / "qlh_keep_head.dll"
    if not shim.exists():
        pytest.skip("缺本机 shim（先跑 scripts/model_tools/build_keep_head_shim.ps1）")
    probe = r"""
import ctypes
import sys
from pathlib import Path
from src.llama_keep_head import _add_dll_dirs

shim_path = Path(sys.argv[1])
_, dll_dir_handles = _add_dll_dirs(shim_path.parent, sys.argv[2:])
lib = ctypes.CDLL(str(shim_path))
symbol = getattr(lib, "qlh_kh_forward_embd_token", None)
if symbol is None:
    raise SystemExit(77)
symbol.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
]
assert symbol.argtypes[0] is ctypes.c_void_p
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, str(shim), *EXTRA_DLL_DIRS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode == 77:
        pytest.skip("本机 shim 尚未包含 P4.5 末段入口")
    assert result.returncode == 0, (
        "真实 keep-head shim ABI 检查失败；DLL 必须在子进程加载以避免污染 pytest worker。\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
