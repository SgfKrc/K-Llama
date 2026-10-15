import os
import sys
import threading
import ctypes
import types
import hashlib
import json
import io
import subprocess

import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llama_engine import LlamaCppEngine


def test_gguf_qwen35_pipeline_descriptor_uses_canonical_model_type(
    monkeypatch, tmp_path,
):
    model = tmp_path / "qwen35.gguf"
    model.write_bytes(b"GGUF fixture")
    tokenizer_root = tmp_path / "tokenizer-asset"
    tokenizer_root.mkdir()
    tokenizer_payload = b'{"version":1}'
    (tokenizer_root / "tokenizer.json").write_bytes(tokenizer_payload)
    tokenizer_sha256 = hashlib.sha256(tokenizer_payload).hexdigest()
    (tokenizer_root / ".qlh-model-asset.json").write_text(json.dumps({
        "schema_version": 1,
        "files": [{
            "path": "tokenizer.json",
            "size": len(tokenizer_payload),
            "sha256": tokenizer_sha256,
        }],
    }), encoding="utf-8")
    monkeypatch.setattr(
        "relay_segment_info.read_gguf_layer_info",
        lambda _path: {"architecture": "qwen35", "n_layer": 2, "n_embd": 2048},
    )
    monkeypatch.setattr(
        "relay_segment_info.read_artifact_manifest",
        lambda _path: {
            "source_model_id": "qwen3-5-2b",
            "source_model_sha256": "e" * 64,
        },
    )
    monkeypatch.setattr(
        "model_config.get_builtin_model",
        lambda _model_id: types.SimpleNamespace(model_path=str(tokenizer_root)),
    )
    monkeypatch.setattr(
        "model_config.resolve_model_path", lambda value: value,
    )
    monkeypatch.setattr(
        "relay_segment_info.read_gguf_tensor_bytes",
        lambda _path: {
            "token_embd.weight": 8,
            "blk.0.attn_norm.weight": 4,
            "blk.1.attn_norm.weight": 4,
            "output_norm.weight": 4,
        },
    )

    engine = LlamaCppEngine()
    engine._model_path = str(model)
    engine.active_model_id = "qwen3-5-2b"

    descriptor = engine.get_pipeline_descriptor()

    assert descriptor["model_type"] == "qwen3_5"
    assert descriptor["pipeline_runtime_supported"] is True
    assert descriptor["hidden_size"] == 2048
    assert descriptor["tokenizer_sha256"] == tokenizer_sha256
    assert descriptor["source_model_id"] == "qwen3-5-2b"
    assert descriptor["source_model_sha256"] == "e" * 64
    assert descriptor["layer_weight_bytes"] == [4, 4]


def test_prepare_pipeline_model_is_metadata_only(monkeypatch, tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF metadata probe")
    engine = LlamaCppEngine()
    descriptor = {
        "model_id": "probe",
        "model_sha256": "a" * 64,
        "model_type": "qwen2",
        "total_layers": 24,
        "pipeline_runtime_supported": True,
    }
    monkeypatch.setattr(
        engine, "get_pipeline_descriptor", lambda: dict(descriptor),
    )

    result = engine.prepare_pipeline_model(
        model_id="probe", model_path=str(path), quant_type="Q4_K_M",
    )

    assert result == descriptor
    assert engine.is_loaded is False
    assert engine.is_pipeline_prepared is True


def test_dead_keep_head_worker_is_recreated(monkeypatch, tmp_path):
    from llama_keep_head import KeepHeadUpstream

    artifact = tmp_path / "head.gguf"
    artifact.write_bytes(b"head")
    shim = tmp_path / "keep-head.dll"
    shim.write_bytes(b"shim")
    closed = []
    old = types.SimpleNamespace(
        is_available=False, close=lambda: closed.append(True),
    )
    fresh = types.SimpleNamespace(is_available=True)
    engine = LlamaCppEngine()
    engine._model_path = str(artifact)
    engine._pipeline_descriptor = {
        "loaded_artifact": str(artifact),
        "assignment_layer_range": [0, 4],
    }
    engine._keep_head_key = (str(artifact), 4)
    engine._keep_head = old
    monkeypatch.setattr("llama_engine.resolve_keep_head_shim", lambda: str(shim))
    # ⚠️ 千万不要 patch `KeepHeadUpstream.__new__` —— CPython 的 `tp_new` 槽一旦因子类
    #   赋值而变成 `slot_tp_new`，**删掉类属性也不会复位**（实测：删后
    #   `"__new__" in cls.__dict__` 为 False、`cls.__new__` 看着仍是继承来的
    #   `object.__new__`，但 `cls(a, b)` 照样抛
    #   `TypeError: object.__new__() takes exactly one argument`）。那会**永久污染
    #   整个进程**里这个类，害得后续构造全崩（实测本文件
    #   `test_keep_head_initialization_and_cleanup_failure_exposes_retryable_owner`
    #   就是这么被弄红的）。改 patch **模块属性**：`_keep_head_upstream()` 里是
    #   函数内 `from llama_keep_head import KeepHeadUpstream`，运行时取到的正是它。
    monkeypatch.setattr("llama_keep_head.KeepHeadUpstream", lambda *args, **kwargs: fresh)

    assert engine._keep_head_upstream() is fresh
    assert closed == [True]


def test_gguf_pipeline_descriptor_rejects_manifest_tokenizer_mismatch(monkeypatch, tmp_path):
    model = tmp_path / "head.gguf"
    model.write_bytes(b"GGUF metadata-only fixture")
    model.with_suffix(".manifest.json").write_text(json.dumps({
        "source_layer_range": [0, 1], "n_layer": 2,
        "tokenizer_sha256": "b" * 64,
    }), encoding="utf-8")
    monkeypatch.setattr("relay_segment_info.read_gguf_layer_info", lambda _: {
        "architecture": "qwen35", "n_layer": 1, "n_embd": 4096,
    })
    monkeypatch.setattr("relay_segment_info.read_gguf_tensor_bytes", lambda _: {
        "token_embd.weight": 8, "blk.0.attn_q.weight": 24,
    })
    monkeypatch.setattr("relay_segment_info.read_gguf_tokenizer_sha256", lambda _: "a" * 64)
    engine = LlamaCppEngine()
    engine._model_path = str(model)
    assert engine.get_pipeline_descriptor() == {}

    monkeypatch.setattr("relay_segment_info.read_gguf_tokenizer_sha256", lambda _: "b" * 64)
    descriptor = engine.get_pipeline_descriptor()
    assert descriptor["tokenizer_sha256"] == "b" * 64
    assert descriptor["total_layers"] == 2


def test_abort_pipeline_materialization_restores_whole_descriptor():
    engine = LlamaCppEngine()
    closed = []

    class LoadedSegment:
        def close(self):
            closed.append(True)

    engine._model = LoadedSegment()
    engine._loaded = True
    engine._model_path = "segment.gguf"
    engine._quant_type = "F16"
    engine.active_model_id = "source-model"
    engine._pipeline_distributed_only = True
    engine._pipeline_descriptor = {
        "model_id": "source-model",
        "model_path": "whole.gguf",
        "model_sha256": "a" * 64,
        "model_type": "qwen2",
        "total_layers": 24,
        "quant_type": "F16",
        "pipeline_runtime_supported": True,
        "assignment_layer_range": [0, 16],
        "partial_assignment": True,
        "loaded_artifact": "segment.gguf",
    }

    engine.abort_pipeline_materialization()

    assert closed == [True]
    assert engine.is_loaded is False
    assert engine.is_pipeline_prepared is True
    assert engine._model_path == "whole.gguf"
    descriptor = engine.get_pipeline_descriptor()
    assert descriptor["model_path"] == "whole.gguf"
    assert "assignment_layer_range" not in descriptor
    assert "partial_assignment" not in descriptor
    assert "loaded_artifact" not in descriptor


def make_fake_mtmd():
    calls = {
        "free_context": 0,
        "free_bitmap": 0,
        "free_chunks": 0,
        "free_batch": 0,
    }
    mtmd = types.SimpleNamespace()
    class FakeInputText(ctypes.Structure):
        _fields_ = [
            ("text", ctypes.c_char_p),
            ("add_special", ctypes.c_bool),
            ("parse_special", ctypes.c_bool),
        ]

    mtmd.MTMD_INPUT_CHUNK_TYPE_TEXT = 0
    mtmd.MTMD_INPUT_CHUNK_TYPE_IMAGE = 1
    mtmd.mtmd_bitmap_p_ctypes = ctypes.c_void_p
    mtmd.mtmd_input_text = lambda data, add_special, parse_special: FakeInputText(
        data, add_special, parse_special,
    )
    mtmd.mtmd_helper_post_decode_callback = lambda callback: callback
    mtmd.mtmd_context_params_default = lambda: types.SimpleNamespace(use_gpu=False, n_threads=0)
    mtmd.mtmd_init_from_file = lambda path, model, params: "mtmd-context"
    mtmd.mtmd_free = lambda context: calls.__setitem__("free_context", calls["free_context"] + 1)
    mtmd.mtmd_support_vision = lambda context: True
    mtmd.mtmd_support_audio = lambda context: False
    mtmd.mtmd_default_marker = lambda: b"<image>"
    mtmd.mtmd_helper_bitmap_init_from_file = lambda context, path, placeholder: types.SimpleNamespace(
        bitmap=123, video_ctx=None,
    )
    mtmd.mtmd_bitmap_free = lambda bitmap: calls.__setitem__("free_bitmap", calls["free_bitmap"] + 1)
    mtmd.mtmd_input_chunks_init = lambda: [0, 1]
    mtmd.mtmd_input_chunks_free = lambda chunks: calls.__setitem__("free_chunks", calls["free_chunks"] + 1)
    mtmd.mtmd_input_chunks_size = lambda chunks: len(chunks)
    mtmd.mtmd_input_chunks_get = lambda chunks, index: {"type": chunks[index]}
    mtmd.mtmd_input_chunk_get_type = lambda chunk: chunk["type"]
    mtmd.mtmd_tokenize = lambda context, chunks, text, bitmaps, count: 0

    def eval_text(context, llama_context, chunk, n_past, seq_id, n_batch, logits_last, new_n_past):
        new_n_past._obj.value += 2
        return 0

    def decode_image(context, llama_context, chunk, embedding, n_past, seq_id, n_batch, new_n_past, callback, user_data):
        new_n_past._obj.value += 3
        return 0

    mtmd.mtmd_helper_eval_chunk_single = eval_text
    mtmd.mtmd_batch_init = lambda context: "mtmd-batch"
    mtmd.mtmd_batch_free = lambda batch: calls.__setitem__("free_batch", calls["free_batch"] + 1)
    mtmd.mtmd_batch_add_chunk = lambda batch, chunk: 0
    mtmd.mtmd_batch_encode = lambda batch: 0
    mtmd.mtmd_batch_get_output_embd = lambda batch, chunk: object()
    mtmd.mtmd_helper_decode_image_chunk = decode_image
    mtmd.calls = calls
    return mtmd


class FakeNativeContext:
    def __init__(self):
        self.ctx = object()
        self.cleared = 0

    def kv_cache_clear(self):
        self.cleared += 1

    def decode(self, batch):
        return None


class FakeNativeModel:
    def __init__(self):
        self.model = object()
        self._ctx = FakeNativeContext()
        self.n_tokens = 4

    def detokenize(self, tokens):
        assert tokens == [7]
        return b"answer"


class FakeNativeBatch:
    closed = 0

    def __init__(self, **kwargs):
        self.last = None

    def set_batch(self, tokens, n_past, logits_all):
        self.last = (tokens, n_past, logits_all)

    def close(self):
        FakeNativeBatch.closed += 1


class FakeNativeSampler:
    closed = 0

    def __init__(self):
        self.tokens = iter([101, 7, 9])

    def add_greedy(self):
        pass

    def add_top_k(self, value):
        pass

    def add_top_p(self, value):
        pass

    def add_temp(self, value):
        pass

    def add_dist(self, value):
        pass

    def sample(self, context, index):
        return next(self.tokens)

    def accept(self, token):
        pass

    def close(self):
        FakeNativeSampler.closed += 1


class FakeChatStream:
    def __init__(self, cancel_event):
        self.cancel_event = cancel_event
        self.closed = False
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= 3:
            raise StopIteration
        self.index += 1
        if self.index == 1:
            self.cancel_event.set()
        return {
            "choices": [{
                "delta": {"content": f"token-{self.index}"},
                "finish_reason": None,
            }],
        }

    def close(self):
        self.closed = True


class FakeLlamaModel:
    def __init__(self, cancel_event):
        self.stream = FakeChatStream(cancel_event)
        self.call_kwargs = None

    def create_chat_completion(self, **kwargs):
        self.call_kwargs = kwargs
        return self.stream

    def tokenize(self, text, add_bos=True, special=False):
        return [1] if text else []


def test_chat_cancel_event_stops_llama_stream_at_token_boundary():
    cancel_event = threading.Event()
    model = FakeLlamaModel(cancel_event)
    engine = LlamaCppEngine()
    engine._model = model
    engine._model_path = "fake.gguf"
    engine._loaded = True

    result = engine.chat(
        [{"role": "user", "content": "question"}],
        max_tokens=10,
        _cancel_event=cancel_event,
    )

    assert model.call_kwargs["stream"] is True
    assert model.stream.index == 1
    assert model.stream.closed is True
    assert result["content"] == "token-1"
    assert result["finish_reason"] == "cancelled"
    assert result["usage"]["completion_tokens"] == 1
    assert result["usage_estimated"] is True


def test_chat_pre_cancelled_does_not_start_llama_generation():
    cancel_event = threading.Event()
    cancel_event.set()
    model = FakeLlamaModel(cancel_event)
    engine = LlamaCppEngine()
    engine._model = model
    engine._model_path = "fake.gguf"
    engine._loaded = True

    result = engine.chat(
        [{"role": "user", "content": "question"}],
        _cancel_event=cancel_event,
    )

    assert model.call_kwargs is None
    assert result["content"] == ""
    assert result["finish_reason"] == "cancelled"


def test_qwen3_thinking_flag_updates_registered_chat_template_state():
    cancel_event = threading.Event()
    model = FakeLlamaModel(cancel_event)
    engine = LlamaCppEngine()
    engine._model = model
    engine._model_path = "qwen3-0.6b.gguf"
    engine._loaded = True
    engine._chat_template = "qwen3_chat_v1"
    engine._thinking_controlled = True

    engine.chat(
        [{"role": "user", "content": "question"}],
        max_tokens=4,
        show_thinking=False,
        _cancel_event=cancel_event,
    )

    assert engine._thinking_enabled is False
    assert engine._chat_template_kwargs == {"enable_thinking": False}


def test_chat_stream_closes_native_stream_and_forwards_thinking_flag():
    class Stream:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            return iter([
                {"choices": [{"delta": {"content": "one"}}]},
                {"choices": [{"delta": {"content": "two"}}]},
            ])

        def close(self):
            self.closed = True

    class Model:
        def __init__(self):
            self.stream = Stream()
            self.call_kwargs = None

        def create_chat_completion(self, **kwargs):
            self.call_kwargs = kwargs
            return self.stream

    model = Model()
    engine = LlamaCppEngine()
    engine._model = model
    engine._model_path = "qwen25.gguf"
    engine._loaded = True

    assert list(engine.chat_stream(
        [{"role": "user", "content": "question"}],
        show_thinking=False,
    )) == ["one", "two"]
    assert model.stream.closed is True
    assert model.call_kwargs["stream"] is True


def test_mtmd_capabilities_are_registered_and_released(tmp_path):
    mtmd = make_fake_mtmd()
    engine = LlamaCppEngine()
    engine._model = FakeNativeModel()
    engine._model_path = "gemma-4.gguf"
    engine._loaded = True
    engine._mtmd_module = mtmd
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"projector")

    capabilities = engine.load_mmproj(str(mmproj))

    assert capabilities["native_mtmd"] is True
    assert capabilities["vision"] is True
    assert capabilities["audio"] is False
    assert engine.get_model_info()["capabilities"] == capabilities
    engine.unload()
    assert mtmd.calls["free_context"] == 1


def test_chat_image_runs_native_pipeline_and_frees_resources(monkeypatch, tmp_path):
    mtmd = make_fake_mtmd()
    fake_llama = types.ModuleType("llama_cpp")
    fake_llama.llama_pos = ctypes.c_int
    fake_llama.llama_token_bos = lambda context: 1
    fake_llama.llama_token_eos = lambda context: 9
    fake_internals = types.SimpleNamespace(LlamaBatch=FakeNativeBatch, LlamaSampler=FakeNativeSampler)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama)
    monkeypatch.setitem(sys.modules, "llama_cpp._internals", fake_internals)

    image = tmp_path / "sample.png"
    image.write_bytes(b"not-a-real-image-for-fake-binding")
    engine = LlamaCppEngine()
    engine._model = FakeNativeModel()
    engine._model_path = "gemma-4.gguf"
    engine._loaded = True
    engine._mtmd_module = mtmd
    engine._mtmd_context = "mtmd-context"
    engine._mtmd_capabilities = {"vision": True, "audio": False}

    result = engine.chat_image(str(image), prompt="What is this? <__media__>", max_tokens=8)

    assert result["content"] == "answer"
    assert result["finish_reason"] == "stop"
    assert result["native_mtmd"] is True
    assert result["usage"]["prompt_tokens"] == 5
    assert result["usage"]["completion_tokens"] == 2
    assert engine._model._ctx.cleared == 1
    assert mtmd.calls["free_bitmap"] == 1
    assert mtmd.calls["free_chunks"] == 1
    assert mtmd.calls["free_batch"] == 1
    assert FakeNativeBatch.closed == 1
    assert FakeNativeSampler.closed == 1


def test_model_name_reports_actual_gguf_not_static_default(tmp_path):
    """★ 2026-09-24 回归（B2）：`LlamaCppEngine.model_name` 必须反映**实际加载物**。

    缺陷史：`get_model_info()` 原先**不报** `model_name` ⇒ 消费方（`/status`、`/models/current`）
    各自兜底到静态 `config.MODEL_NAME`（= 默认 0.6B）⇒ 实测加载 2B（`qwen35-2b-Q4_K_M.gguf`）
    却对外报 `Qwen/Qwen3-0.6B`（见 `docs/未完成工作备忘-2026-09-23.md` §4.8 B2）。

    「该红必须红」：若有人把 `model_name` 改回"取不到就返回 `MODEL_NAME`"，本用例立刻红。
    """
    engine = LlamaCppEngine()

    # ① 未加载 ⇒ 空串；**不得**是任何静态默认模型名
    assert engine.model_name == "", "未加载时不应报出模型名，更不能是静态默认值"

    # ② 有路径但读不到 metadata ⇒ 回退到 GGUF **文件名**（去扩展名）
    gguf = tmp_path / "qwen35-2b-Q4_K_M.gguf"
    gguf.write_bytes(b"")          # 刻意不真加载：只验证取值链
    engine._model_path = str(gguf)
    assert engine.model_name == "qwen35-2b-Q4_K_M"

    # ③ 真加载时 llama_cpp 提供 metadata ⇒ 优先取 GGUF 自带的 general.name
    class _ModelWithMetadata:
        metadata = {"general.name": "Qwen3 5 2b"}

    engine._model = _ModelWithMetadata()
    assert engine.model_name == "Qwen3 5 2b"

    # ④ `get_model_info()` 必须带上该键（消费方正是从这里取）
    info = engine.get_model_info()
    assert info["model_name"] == "Qwen3 5 2b"
    assert info["engine"] == "llama.cpp"


def test_reporting_paths_do_not_fall_back_to_static_model_name():
    """★ 2026-09-24 回归（B2）：`/status` 与 `/models/current` 的 `model_name` 兜底**不得**
    再用静态 `config.MODEL_NAME`（那会张冠李戴）。

    这是**源码级守卫**：兜底位置在 API 层、端到端验证要起真后端 + 真模型，
    所以这里直接锁住"不该出现的写法" —— 任何人改回去立刻红。
    """
    root = Path(__file__).resolve().parents[1]
    for relative in ("src/api/routes_health.py", "src/api/routes_models.py"):
        text = (root / relative).read_text(encoding="utf-8")
        offenders = [
            line.strip() for line in text.splitlines()
            if "model_name" in line and "MODEL_NAME" in line
        ]
        assert not offenders, (
            f"{relative} 的 model_name 兜底又用回了静态 MODEL_NAME: {offenders}"
        )


def test_chat_image_requires_registered_vision(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"image")
    engine = LlamaCppEngine()
    engine._model = FakeNativeModel()
    engine._loaded = True

    try:
        engine.chat_image(str(image))
    except RuntimeError as exc:
        assert "vision capability" in str(exc)
    else:
        raise AssertionError("chat_image should require native vision registration")


# ================================================================
# G4.5：GPU offload 预算门与便捷加载
# ================================================================

def test_estimate_gpu_layers_budget_gate():
    """8GB 显存场景：部分 offload 层数在合理区间（不承诺全量，不归零）。"""
    layers = LlamaCppEngine.estimate_gpu_layers(
        36, 7_662_533_088, int(7.4 * 2**30),
    )
    # Safety headroom reduces the usable budget; it must never create VRAM.
    assert 0 < layers < 36
    per_layer = 7_662_533_088 // 36
    assert (layers * per_layer + 512 * 2**20) * 1.15 <= int(7.4 * 2**30)
    # 显存紧张（4GB，与 SD 并驻留场景）：显著少于全量
    low = LlamaCppEngine.estimate_gpu_layers(
        36, 7_662_533_088, int(4 * 2**30),
    )
    assert 0 < low < 36
    # 显存不足 → 0（不强行 offload）
    zero = LlamaCppEngine.estimate_gpu_layers(
        36, 7_662_533_088, int(0.4 * 2**30),
    )
    assert zero == 0


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "models", "gemma4-native")),
    reason="缺本地工件 models/gemma4-native（CI 不拉模型）",
)
def test_load_gemma4_native_fails_closed_on_vram_shortage(monkeypatch):
    """require_gpu_layers 不满足时 fail-closed（不加载）。"""
    monkeypatch.setattr(LlamaCppEngine, "_vram_free_bytes", staticmethod(lambda: int(0.5 * 2**30)))
    engine = LlamaCppEngine()
    try:
        engine.load_gemma4_native(gpu_layers=-1, require_gpu_layers=8)
    except RuntimeError as exc:
        assert "显存预算不足" in str(exc)
    else:
        raise AssertionError("显存不足时应 fail-closed")


def test_load_gemma4_native_rejects_invalid_gpu_layer_request():
    engine = LlamaCppEngine()
    try:
        engine.load_gemma4_native(gpu_layers=-2)
    except ValueError as exc:
        assert "gpu_layers" in str(exc)
    else:
        raise AssertionError("invalid gpu_layers should be rejected")


def test_gemma4_native_asset_lock_rejects_tampering(tmp_path):
    model = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    model.write_bytes(b"model")
    mmproj.write_bytes(b"mmproj")
    lock = tmp_path / "lock.json"

    def entry(path):
        return {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    lock.write_text(json.dumps({
        "schema_version": 1,
        "artifacts": {"main_gguf": entry(model), "mmproj": entry(mmproj)},
    }), encoding="utf-8")
    LlamaCppEngine._verify_gemma4_native_assets(lock, model, mmproj)
    model.write_bytes(b"tampered")
    try:
        LlamaCppEngine._verify_gemma4_native_assets(lock, model, mmproj)
    except RuntimeError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("tampered Gemma asset should be rejected")


def test_load_gemma4_native_rejects_n_gpu_layers_via_kwargs():
    """n_gpu_layers 由 gpu_layers/require_gpu_layers 预算门管理，
    不得经 **llama_kwargs 绕开（否则预算门形同虚设）。"""
    engine = LlamaCppEngine()
    try:
        engine.load_gemma4_native(gpu_layers=0, n_gpu_layers=99)
    except ValueError as exc:
        assert "gpu_layers" in str(exc)
    else:
        raise AssertionError("应拒绝经 **llama_kwargs 传入 n_gpu_layers")


def test_load_gemma4_native_forwards_llama_kwargs(monkeypatch, tmp_path):
    """**llama_kwargs 应落到最终 Llama() 的加载参数里。

    这是「主仓缺口」的修复点：load_gemma4_native 此前只透传 n_gpu_layers，
    使得 tensor_split / use_mmap / use_mlock / kv_overrides 等容量与放置相关的
    参数无法经该路径使用。
    """
    captured: dict = {}

    # 伪造受管工件 + lock，绕过真实文件校验
    model = tmp_path / "fake.gguf"
    mmproj = tmp_path / "fake.mmproj"
    model.write_bytes(b"gguf")
    mmproj.write_bytes(b"mmproj")
    lock_dir = tmp_path / "models" / "gemma4-native"
    lock_dir.mkdir(parents=True)
    lock = lock_dir / "gemma4-native.lock.json"

    def entry(path):
        return {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    lock.write_text(json.dumps({
        "schema_version": 1,
        "artifacts": {"main_gguf": entry(model), "mmproj": entry(mmproj)},
    }), encoding="utf-8")

    monkeypatch.setattr(LlamaCppEngine, "_verify_gemma4_native_assets",
                        staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(LlamaCppEngine, "load_model",
                        lambda self, **kw: captured.update(kw))

    engine = LlamaCppEngine()
    # 直接以显式路径调用，避开缺省工件解析
    engine.load_gemma4_native(
        gguf_path=str(model),
        mmproj_path=str(mmproj),
        gpu_layers=4,
        tensor_split=[0.6, 0.4],
        use_mmap=False,
        kv_overrides={"qwen35.block_count": 20},
    )
    assert captured.get("tensor_split") == [0.6, 0.4]
    assert captured.get("use_mmap") is False
    assert captured.get("kv_overrides") == {"qwen35.block_count": 20}
    # gpu_layers > 0 时由预算门显式写入 n_gpu_layers
    assert captured.get("n_gpu_layers") == 4
    # 显式路径必须优先于缺省工件解析
    assert captured.get("model_path") == str(model)


def test_unload_reaps_keep_head_child_without_close_response():
    from llama_keep_head import KeepHeadUpstream

    process = subprocess.Popen(
        [sys.executable, "-c", "import sys,time; sys.stdin.readline(); time.sleep(60)"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    upstream = KeepHeadUpstream.__new__(KeepHeadUpstream)
    upstream._worker = process
    engine = LlamaCppEngine()
    engine._keep_head = upstream
    engine._keep_head_key = ("head.gguf", 8)
    engine._pipeline_distributed_only = True
    engine._pipeline_descriptor = {"assignment_layer_range": [0, 8]}
    try:
        engine.unload()
        assert process.poll() is not None
        assert process.stdin.closed and process.stdout.closed
        assert upstream._worker is None
        assert engine._keep_head is None and engine._keep_head_key is None
        assert not engine.is_loaded and not engine.is_pipeline_prepared
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        for pipe in (process.stdin, process.stdout):
            if not pipe.closed:
                pipe.close()


def test_keep_head_close_failure_retains_child_for_retry():
    from llama_keep_head import KeepHeadUpstream

    class Process:
        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = io.StringIO()
            self.fail = True
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            if self.fail:
                raise OSError("cannot confirm process exit")
            self.returncode = 0
            return 0

    process = Process()
    upstream = KeepHeadUpstream.__new__(KeepHeadUpstream)
    upstream._worker = process
    with pytest.raises(OSError, match="cannot confirm process exit"):
        upstream.close()
    assert upstream._worker is process and process.poll() is None
    assert process.stdin.closed and not process.stdout.closed
    process.fail = False
    upstream.close()
    assert upstream._worker is None and process.poll() == 0
    assert process.stdin.closed and process.stdout.closed


def test_unload_keep_head_failure_preserves_model_and_identity():
    engine = LlamaCppEngine()
    calls = []
    should_fail = [True]

    def close():
        calls.append("head")
        if should_fail[0]:
            raise OSError("head still running")

    upstream = types.SimpleNamespace(close=close)
    model = types.SimpleNamespace(close=lambda: calls.append("binding"))
    descriptor = {"assignment_layer_range": [0, 8]}
    engine._keep_head = upstream
    engine._keep_head_key = ("head.gguf", 8)
    engine._model = model
    engine._loaded = True
    engine._model_path = "head.gguf"
    engine._pipeline_descriptor = descriptor
    with pytest.raises(OSError, match="head still running"):
        engine.unload()
    assert engine._keep_head is upstream and engine._keep_head_key == ("head.gguf", 8)
    assert engine._model is model and engine.is_loaded
    assert engine._pipeline_descriptor is descriptor and engine._model_path == "head.gguf"
    assert calls == ["head"]
    should_fail[0] = False
    engine.unload()
    assert calls == ["head", "head", "binding"]
    assert engine._keep_head is None and engine._model is None and not engine.is_loaded


def test_unload_binding_failure_retains_binding_for_retry():
    engine = LlamaCppEngine()
    should_fail = [True]

    def close():
        if should_fail[0]:
            raise OSError("binding release failed")

    model = types.SimpleNamespace(close=close)
    descriptor = {"assignment_layer_range": [0, 8]}
    engine._model = model
    engine._loaded = True
    engine._model_path = "head.gguf"
    engine._pipeline_descriptor = descriptor
    with pytest.raises(OSError, match="binding release failed"):
        engine.unload()
    assert engine._model is model and engine.is_loaded
    assert engine._pipeline_descriptor is descriptor and engine._model_path == "head.gguf"
    should_fail[0] = False
    engine.unload()
    assert engine._model is None and not engine.is_loaded
    assert engine._model_path == "" and engine._pipeline_descriptor is None


def test_keep_head_replacement_close_failure_preserves_old_owner(monkeypatch):
    engine = LlamaCppEngine()
    descriptor = {"loaded_artifact": "new.gguf", "assignment_layer_range": [0, 4]}
    engine._pipeline_descriptor = descriptor

    def fail_close():
        raise OSError("old head still running")

    old = types.SimpleNamespace(is_available=True, close=fail_close)
    engine._keep_head = old
    engine._keep_head_key = ("old.gguf", 8)
    monkeypatch.setattr("llama_engine.resolve_keep_head_shim", lambda: pytest.fail("must retire old owner first"))
    with pytest.raises(OSError, match="old head still running"):
        engine._keep_head_upstream()
    assert engine._keep_head is old and engine._keep_head_key == ("old.gguf", 8)
    assert engine._pipeline_descriptor is descriptor


def test_keep_head_initialization_and_cleanup_failure_exposes_retryable_owner(monkeypatch, tmp_path):
    from llama_keep_head import KeepHeadCleanupError, KeepHeadUpstream

    class Process:
        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = io.StringIO('{"ok":false,"error":"native init failed"}\n')
            self.returncode = None
            self.fail = True

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            if self.fail:
                raise PermissionError("process exit could not be confirmed")
            self.returncode = 0
            return 0

    process = Process()
    shim = tmp_path / "shim.dll"
    artifact = tmp_path / "head.gguf"
    shim.write_bytes(b"shim")
    artifact.write_bytes(b"segment")
    monkeypatch.setattr("llama_keep_head._llama_cpp_loaded", lambda: True)
    monkeypatch.setattr("llama_keep_head.subprocess.Popen", lambda *args, **kw: process)
    monkeypatch.setenv("QLH_KEEP_HEAD_WORKER_LOG", str(tmp_path / "child.log"))
    with pytest.raises(KeepHeadCleanupError) as failed:
        KeepHeadUpstream(shim, artifact)
    owner = failed.value.cleanup_owner
    assert owner._worker is process and process.poll() is None
    assert not owner.is_available
    process.fail = False
    owner.close()
    assert owner._worker is None and process.poll() == 0
    assert process.stdin.closed and process.stdout.closed


def test_engine_retains_failed_initializer_before_propagating(monkeypatch, tmp_path):
    from llama_keep_head import KeepHeadCleanupError

    class Partial:
        is_available = False
        def close(self):
            self.closed = True

    owner = Partial()
    def initialize(*args, **kwargs):
        raise KeepHeadCleanupError(owner, RuntimeError("init failed"), OSError("close failed"))

    artifact = tmp_path / "head.gguf"
    shim = tmp_path / "shim.dll"
    artifact.write_bytes(b"segment")
    shim.write_bytes(b"shim")
    monkeypatch.setattr("llama_keep_head.KeepHeadUpstream", initialize)
    monkeypatch.setattr("llama_engine.resolve_keep_head_shim", lambda: str(shim))
    engine = LlamaCppEngine()
    engine._pipeline_descriptor = {"loaded_artifact": str(artifact), "assignment_layer_range": [0, 8]}
    with pytest.raises(KeepHeadCleanupError):
        engine._keep_head_upstream()
    assert engine._keep_head is owner and engine._keep_head_key is None
    engine.unload()
    assert owner.closed and engine._keep_head is None
