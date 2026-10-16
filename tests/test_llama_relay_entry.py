"""★ 主仓下游接力入口（`LlamaCppEngine.forward_layers_from_hidden`）的单元测试。

背景：QLH 的跨框架层接力（D→L / L→L）此前**只有实验脚本**用裸 `llama_cpp` API 实现
（`build/cross-framework-layer-poc/relay_mainrepo_upstream.py`），主仓 `llama_engine` 没有入口。
2026-09-19 在 `src/llama_engine.py` 新增 `forward_layers_from_hidden()`：把上游 hidden 注入
`llama_batch.embd`，只跑本模型（裁层 GGUF）的层、返回末位 logits —— 使「上下游都走主仓引擎」成立。

Real-model cases use the explicit real_model channel and skip missing cut GGUF artifacts.
Offline error paths and constructor restoration assertions run without native DLLs or weights.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CUT_GGUF = ROOT / "build" / "cross-framework-layer-poc" / "out" / "qwen25-05b-f16-cut-k12.gguf"


def _require_real_model_opt_in():
    if os.environ.get("QLH_RUN_PHYSICAL_TESTS") != "1" and os.environ.get("QLH_RUN_REAL_MODEL_SMOKE") != "1":
        pytest.skip("real_model requires QLH_RUN_PHYSICAL_TESTS=1 or QLH_RUN_REAL_MODEL_SMOKE=1")


def _engine_or_skip(**load_kwargs):
    _require_real_model_opt_in()
    if not CUT_GGUF.is_file():
        pytest.skip(f"missing cut GGUF artifact: {CUT_GGUF.relative_to(ROOT)}")
    from llama_engine import LlamaCppEngine

    eng = LlamaCppEngine()
    try:
        eng.load_model(
            model_path=str(CUT_GGUF),
            **{"n_ctx": 512, "n_threads": 4, **load_kwargs},
        )
    except Exception:
        eng.unload()
        raise  # Explicit native smoke must expose binding/load failures.
    return eng


@contextmanager
def _isolated_llama_binding():
    """Install a coherent stub package and restore the entire prior family."""
    names = [name for name in sys.modules if name == "llama_cpp" or name.startswith("llama_cpp.")]
    originals = {name: sys.modules[name] for name in names}
    package = ModuleType("llama_cpp")
    package.__path__ = []
    low_level = ModuleType("llama_cpp.llama_cpp")
    low_level.llama_context_default_params = lambda: SimpleNamespace(n_seq_max=1)
    package.llama_cpp = low_level
    seen = []
    binding = SimpleNamespace(package=package, low_level=low_level, seen=seen, fail=False)

    class ConstructorFailure(RuntimeError):
        pass

    binding.ConstructorFailure = ConstructorFailure

    def construct(**kwargs):
        params = low_level.llama_context_default_params()
        seen.append((dict(kwargs), params.n_seq_max))
        if binding.fail:
            raise ConstructorFailure("controlled constructor failure")
        return SimpleNamespace(kwargs=kwargs, n_seq_max=params.n_seq_max)

    package.Llama = construct
    try:
        for name in names:
            del sys.modules[name]
        sys.modules["llama_cpp"] = package
        sys.modules["llama_cpp.llama_cpp"] = low_level
        yield binding
    finally:
        for name in list(sys.modules):
            if name == "llama_cpp" or name.startswith("llama_cpp."):
                del sys.modules[name]
        sys.modules.update(originals)


@pytest.fixture
def fake_llama_binding():
    with _isolated_llama_binding() as binding:
        yield binding


class TestNotLoaded:
    """未加载路径：不需要任何模型，任何环境都能跑。"""

    def test_returns_none_when_not_loaded(self):
        from llama_engine import LlamaCppEngine

        eng = LlamaCppEngine()
        assert eng.is_loaded is False, "新实例不应处于已加载态"
        assert eng.forward_layers_from_hidden([[0.0] * 8], n_past=0) is None

    def test_is_loaded_is_property_not_method(self):
        """⚠️ 实测坑：`is_loaded` 是 **bool 属性**，调用它会 `TypeError: 'bool' object is not callable`。"""
        from llama_engine import LlamaCppEngine

        eng = LlamaCppEngine()
        assert isinstance(eng.is_loaded, bool)
        with pytest.raises(TypeError):
            eng.is_loaded()  # type: ignore[operator]

    def test_partial_upstream_fails_closed_without_keep_head(self, monkeypatch):
        """A partial GGUF must never hand output_norm hidden to a relay tail."""
        from llama_engine import LlamaCppEngine

        eng = LlamaCppEngine()
        eng._model = object()
        eng._loaded = True
        eng._pipeline_descriptor = {
            "partial_assignment": True,
            "assignment_layer_range": [0, 8],
        }
        monkeypatch.setattr(eng, "_keep_head_upstream", lambda: None)

        with pytest.raises(RuntimeError, match="keep-head"):
            eng.forward_layers_to_hidden([1, 2], n_past=0)

    def test_layer_load_marks_partial_without_prior_descriptor(self, monkeypatch, tmp_path):
        """A first direct stage load must still block full-model fallback."""
        from llama_engine import LlamaCppEngine

        artifact = tmp_path / "head8.gguf"
        artifact.write_bytes(b"test")
        eng = LlamaCppEngine()
        monkeypatch.setattr(eng, "_find_layer_artifact", lambda _start, _end: str(artifact))
        monkeypatch.setattr(
            eng,
            "load_model",
            lambda path: setattr(eng, "_model_path", str(path)),
        )

        eng.load_layer_range(0, 8, has_embedding=True, has_lm_head=False)

        descriptor = eng.get_pipeline_descriptor()
        assert descriptor["partial_assignment"] is True
        assert descriptor["assignment_layer_range"] == [0, 8]

    def test_layer_artifact_selection_requires_same_source_and_valid_digest(
        self, monkeypatch, tmp_path,
    ):
        from llama_engine import LlamaCppEngine

        source = tmp_path / "whole.gguf"
        source.write_bytes(b"source-model-a")
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()

        wrong = tmp_path / "a-wrong.gguf"
        wrong.write_bytes(b"segment-from-model-b")
        (tmp_path / "a-wrong.manifest.json").write_text(json.dumps({
            "source_layer_range": [0, 8],
            "source_model_sha256": "b" * 64,
            "artifact_sha256": hashlib.sha256(wrong.read_bytes()).hexdigest(),
        }), encoding="utf-8")

        right = tmp_path / "z-right.gguf"
        right.write_bytes(b"segment-from-model-a")
        (tmp_path / "z-right.gguf.manifest.json").write_text(json.dumps({
            "source_layer_range": [0, 8],
            "source_model_sha256": source_sha,
            "artifact_sha256": hashlib.sha256(right.read_bytes()).hexdigest(),
        }), encoding="utf-8")

        eng = LlamaCppEngine()
        eng._pipeline_descriptor = {
            "model_path": str(source),
            "model_sha256": source_sha,
        }
        monkeypatch.setenv("QLH_LAYER_ARTIFACT_DIR", str(tmp_path))

        assert eng._find_layer_artifact(0, 8) == str(right)

    def test_precut_head_uses_source_digest_not_head_digest(
        self, monkeypatch, tmp_path,
    ):
        from llama_engine import LlamaCppEngine

        source_sha = hashlib.sha256(b"unloaded-whole-model").hexdigest()
        head = tmp_path / "head4.gguf"
        head.write_bytes(b"verified-head-segment")
        head_sha = hashlib.sha256(head.read_bytes()).hexdigest()
        manifest = tmp_path / "head4.manifest.json"
        manifest.write_text(json.dumps({
            "source_layer_range": [0, 4],
            "source_model_sha256": source_sha,
            "artifact_sha256": head_sha,
        }), encoding="utf-8")

        engine = LlamaCppEngine()
        engine._model_path = str(head)
        engine._pipeline_descriptor = {
            "model_path": str(head),
            "model_sha256": head_sha,
            "source_model_sha256": source_sha,
        }
        monkeypatch.delenv("QLH_LAYER_ARTIFACT_DIR", raising=False)

        assert engine._find_layer_artifact(0, 4) == str(head)
        manifest.write_text(json.dumps({
            "source_layer_range": [0, 4],
            "source_model_sha256": head_sha,
            "artifact_sha256": head_sha,
        }), encoding="utf-8")
        assert engine._find_layer_artifact(0, 4) == ""

    @pytest.mark.parametrize("legacy_exists", [False, True])
    def test_supplied_segment_is_authoritative_over_legacy_artifact_directory(
        self, monkeypatch, tmp_path, legacy_exists,
    ):
        from llama_engine import LlamaCppEngine
        import model_supply_worker

        selected_dir = tmp_path / "supply"
        selected_dir.mkdir()
        head = selected_dir / "segment.gguf"
        head.write_bytes(b"newly-built-head-twelve")
        source_sha = hashlib.sha256(b"unloaded-source").hexdigest()
        info = {
            "source_layer_range": [0, 12],
            "source_model_sha256": source_sha,
            "artifact_sha256": hashlib.sha256(head.read_bytes()).hexdigest(),
        }
        manifest = head.with_suffix(".manifest.json")
        manifest.write_text(json.dumps(info), encoding="utf-8")
        legacy = tmp_path / "legacy"
        if legacy_exists:
            legacy.mkdir()
        monkeypatch.setenv("QLH_LAYER_ARTIFACT_DIR", str(legacy))
        monkeypatch.setattr(model_supply_worker, "selected_gguf_path", lambda: str(head))
        engine = LlamaCppEngine()
        engine._model_path = str(head)
        engine._pipeline_descriptor = {
            "model_path": str(head), "source_model_sha256": source_sha,
        }
        assert engine._find_layer_artifact(0, 12) == str(head)
        assert engine._find_layer_artifact(0, 4) == ""
        # A selected receipt still needs its precise range/source/payload proof.
        info["artifact_sha256"] = "f" * 64
        manifest.write_text(json.dumps(info), encoding="utf-8")
        assert engine._find_layer_artifact(0, 12) == ""

    def test_layer_artifact_selection_rejects_digest_mismatch(
        self, monkeypatch, tmp_path,
    ):
        from llama_engine import LlamaCppEngine

        source = tmp_path / "whole.gguf"
        source.write_bytes(b"source-model")
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        artifact = tmp_path / "head8.gguf"
        artifact.write_bytes(b"tampered-segment")
        (tmp_path / "head8.manifest.json").write_text(json.dumps({
            "source_layer_range": [0, 8],
            "source_model_sha256": source_sha,
            "artifact_sha256": "c" * 64,
        }), encoding="utf-8")

        eng = LlamaCppEngine()
        eng._pipeline_descriptor = {
            "model_path": str(source),
            "model_sha256": source_sha,
        }
        monkeypatch.setenv("QLH_LAYER_ARTIFACT_DIR", str(tmp_path))

        assert eng._find_layer_artifact(0, 8) == ""

    def test_layer_artifact_selection_rehashes_same_size_replacement(
        self, monkeypatch, tmp_path,
    ):
        from llama_engine import LlamaCppEngine

        source = tmp_path / "whole.gguf"
        source.write_bytes(b"source-model")
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        artifact = tmp_path / "head8.gguf"
        artifact.write_bytes(b"segment-good")
        original_stat = artifact.stat()
        (tmp_path / "head8.manifest.json").write_text(json.dumps({
            "source_layer_range": [0, 8],
            "source_model_sha256": source_sha,
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }), encoding="utf-8")

        eng = LlamaCppEngine()
        eng._pipeline_descriptor = {
            "model_path": str(source),
            "model_sha256": source_sha,
        }
        monkeypatch.setenv("QLH_LAYER_ARTIFACT_DIR", str(tmp_path))

        assert eng._find_layer_artifact(0, 8) == str(artifact)
        artifact.write_bytes(b"segment-evil")
        os.utime(
            artifact,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )

        assert eng._find_layer_artifact(0, 8) == ""

    def test_layer_artifact_selection_rejects_multiple_valid_matches(
        self, monkeypatch, tmp_path,
    ):
        from llama_engine import LlamaCppEngine

        source = tmp_path / "whole.gguf"
        source.write_bytes(b"source-model")
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        for stem in ("head8-a", "head8-b"):
            artifact = tmp_path / f"{stem}.gguf"
            artifact.write_bytes(stem.encode("ascii"))
            (tmp_path / f"{stem}.manifest.json").write_text(json.dumps({
                "source_layer_range": [0, 8],
                "source_model_sha256": source_sha,
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }), encoding="utf-8")

        eng = LlamaCppEngine()
        eng._pipeline_descriptor = {
            "model_path": str(source),
            "model_sha256": source_sha,
        }
        monkeypatch.setenv("QLH_LAYER_ARTIFACT_DIR", str(tmp_path))

        with pytest.raises(RuntimeError, match="多个同源"):
            eng._find_layer_artifact(0, 8)


@pytest.mark.real_model
class TestWithRealModel:
    """需要裁层 GGUF 的用例（缺工件则整类跳过）。"""

    @pytest.fixture(autouse=True)
    def _release_engines(self, monkeypatch):
        engines = []
        original = _engine_or_skip

        def tracked_engine(**kwargs):
            engine = original(**kwargs)
            engines.append(engine)
            return engine

        monkeypatch.setattr(sys.modules[__name__], "_engine_or_skip", tracked_engine)
        try:
            yield
        finally:
            for engine in engines:
                engine.unload()

    @pytest.fixture(autouse=True)
    def _allow_normed_upstream(self, monkeypatch):
        """★ 2026-10-06：本类有意覆盖 `forward_layers_to_hidden` 的 pip-embeddings 通道
        （它返回 `output_norm(H)`）。该通道自 #35 加固起**默认 fail-closed**（因为对层接力
        而言语义不兼容），所以这里显式放行，让函数级覆盖继续有效。
        「默认拒绝」本身另由 `test_default_rejects_normed_upstream` 覆盖。"""
        monkeypatch.setenv("QLH_ALLOW_NORMED_UPSTREAM", "1")

    def test_default_rejects_normed_upstream(self, monkeypatch):
        """★ #35 加固（2026-10-06）：**默认**（不设 `QLH_ALLOW_NORMED_UPSTREAM`）必须
        fail-closed，不得静默交付 `output_norm(H)`。本类有 autouse fixture 放行 ⇒
        这里显式清除以验证默认行为；keep-head 也 mock 成不可用以走到该分支。"""
        monkeypatch.delenv("QLH_ALLOW_NORMED_UPSTREAM", raising=False)
        eng = _engine_or_skip()
        monkeypatch.setattr(eng, "_keep_head_upstream", lambda: None)
        with pytest.raises(RuntimeError, match="output_norm"):
            eng.forward_layers_to_hidden([1, 2], n_past=0)

    def test_logits_shape_and_determinism(self):
        import numpy as np

        eng = _engine_or_skip()
        nm = eng._model._model.model  # ⚠️ 原生指针在 `.model`（`. _model` 是 LlamaModel 包装）
        import llama_cpp.llama_cpp as M

        n_embd = int(M.llama_model_n_embd_inp(nm))
        n_vocab = int(M.llama_vocab_n_tokens(M.llama_model_get_vocab(nm)))

        h = (np.random.default_rng(0).standard_normal((1, n_embd)) * 0.02).astype(np.float32)
        lg1 = eng.forward_layers_from_hidden(h, n_past=0)
        assert lg1 is not None
        assert lg1.shape == (n_vocab,), f"logits 形状应为 (n_vocab,)，实得 {lg1.shape}"
        assert np.isfinite(lg1).all(), "logits 不应出现 NaN/Inf"

        # 同输入重跑 ⇒ 逐位一致（接力对数值稳定性有硬要求）
        # ⚠️ 必须先清 KV：本方法直接在 KV 里占位置，同 n_past 重跑会 llama_decode rc=-1
        #    （注意 `reset_kv_cache()` 是既有的 stateless no-op，用不了）
        eng._model._ctx.kv_cache_clear()
        lg2 = eng.forward_layers_from_hidden(h, n_past=0)
        assert np.array_equal(lg1, lg2), "相同输入两次调用结果必须逐位一致"

    def test_rejects_width_mismatch(self):
        import numpy as np

        eng = _engine_or_skip()
        import llama_cpp.llama_cpp as M

        n_embd = int(M.llama_model_n_embd_inp(eng._model._model.model))
        bad = np.zeros((1, n_embd // 2), dtype=np.float32)
        with pytest.raises(ValueError, match="n_embd"):
            eng.forward_layers_from_hidden(bad, n_past=0)

    def test_accepts_1d_hidden(self):
        """1D 输入应被当作单 token（`[n_embd]` ⇒ `[1, n_embd]`）。"""
        import numpy as np

        eng = _engine_or_skip()
        import llama_cpp.llama_cpp as M

        n_embd = int(M.llama_model_n_embd_inp(eng._model._model.model))
        lg = eng.forward_layers_from_hidden(np.zeros(n_embd, dtype=np.float32), n_past=0)
        assert lg is not None and lg.ndim == 1

    def test_upstream_entry_returns_hidden(self):
        """★ 上游入口（对称）：裁层 GGUF 跑前 k 层 ⇒ 出末位 hidden（长度 n_embd），且同输入逐位一致。"""
        import numpy as np

        eng = _engine_or_skip()
        import llama_cpp.llama_cpp as M

        n_embd = int(M.llama_model_n_embd(eng._model._model.model))
        ids = [100, 200, 300, 400]
        h1 = eng.forward_layers_to_hidden(ids, n_past=0)
        assert h1 is not None
        assert h1.shape == (n_embd,), f"hidden 形状应为 (n_embd,)，实得 {h1.shape}"
        assert np.isfinite(h1).all()

        # ⚠️ 同 n_past 重跑前须清 KV（reset_kv_cache 是既有 no-op）
        eng._model._ctx.kv_cache_clear()
        h2 = eng.forward_layers_to_hidden(ids, n_past=0)
        assert np.array_equal(h1, h2), "相同输入两次调用结果必须逐位一致"

    def test_upstream_entry_rejects_empty(self):
        eng = _engine_or_skip()
        with pytest.raises(ValueError, match="不能为空"):
            eng.forward_layers_to_hidden([], n_past=0)


class TestMultiSequence:
    """★ `n_seq_max` 工程化收口（2026-09-19，待办 3）。

    背景：`llama-cpp-python` 的 `Llama.__init__` **只在 `embedding=True` 时**才设
    `context_params.n_seq_max`（源码 :341-346）⇒ 非 embedding 时恒为 1 ⇒ **`seq_id>0` 必然
    `llama_decode rc=-1`**。主仓现经模块级 `llama_context_default_params()` 在建 context 前注入
    （`llama_engine._new_llama_with_seq_max`）。
    """

    @pytest.fixture(autouse=True)
    def _release_engines(self, monkeypatch):
        engines = []
        original = _engine_or_skip

        def tracked_engine(**kwargs):
            engine = original(**kwargs)
            engines.append(engine)
            return engine

        monkeypatch.setattr(sys.modules[__name__], "_engine_or_skip", tracked_engine)
        try:
            yield
        finally:
            for engine in engines:
                engine.unload()

    @pytest.fixture(autouse=True)
    def _allow_normed_upstream(self, monkeypatch):
        """同 `TestWithRealModel`：本类直接覆盖 `forward_layers_to_hidden`，需显式放行
        （该函数的 embeddings 通道自 #35 加固起默认 fail-closed）。"""
        monkeypatch.setenv("QLH_ALLOW_NORMED_UPSTREAM", "1")

    @pytest.mark.parametrize("constructor_fails", [False, True])
    def test_default_path_unchanged(self, fake_llama_binding, constructor_fails):
        """Default construction preserves kwargs and the original params API."""
        from llama_engine import _new_llama_with_seq_max

        binding = fake_llama_binding
        before = binding.low_level.llama_context_default_params
        binding.fail = constructor_fails
        kwargs = {"model_path": "offline-stub.gguf", "n_ctx": 512, "verbose": False}
        if constructor_fails:
            with pytest.raises(binding.ConstructorFailure, match="controlled"):
                _new_llama_with_seq_max(kwargs, None)
        else:
            model = _new_llama_with_seq_max(kwargs, None)
            assert model.n_seq_max == 1
            assert model.kwargs == kwargs
        assert binding.seen == [(kwargs, 1)]
        assert binding.low_level.llama_context_default_params is before
        assert kwargs == {"model_path": "offline-stub.gguf", "n_ctx": 512, "verbose": False}

    @pytest.mark.parametrize("constructor_fails", [False, True])
    @pytest.mark.parametrize("requested, expected", [(4, 4), (0, 1)])
    def test_seq_max_is_restored_after_load(self, fake_llama_binding, constructor_fails, requested, expected):
        """The constructor sees the raised limit; success/error both restore it."""
        from llama_engine import _new_llama_with_seq_max

        binding = fake_llama_binding
        before = binding.low_level.llama_context_default_params
        binding.fail = constructor_fails
        kwargs = {"model_path": "offline-stub.gguf", "n_ctx": 512}
        if constructor_fails:
            with pytest.raises(binding.ConstructorFailure, match="controlled"):
                _new_llama_with_seq_max(kwargs, requested)
        else:
            model = _new_llama_with_seq_max(kwargs, requested)
            assert model.n_seq_max == expected
            assert model.kwargs == kwargs
        assert binding.seen == [(kwargs, expected)]
        assert binding.low_level.llama_context_default_params is before
        assert before().n_seq_max == 1

    @pytest.mark.real_model
    def test_multi_sequence_decode_matches_single(self):
        """多序列（`seq_id>0`）能解码，且与单序列结果逐位一致 —— 数值判据。"""
        import numpy as np

        eng = _engine_or_skip(n_ctx=1024, n_seq_max=4)
        import llama_cpp.llama_cpp as M

        n_embd = int(M.llama_model_n_embd_inp(eng._model._model.model))
        rng = np.random.default_rng(0)
        h = rng.standard_normal((2, n_embd)).astype(np.float32)

        # 单序列（一次一条；每次前清 KV，因为每条序列的 seq_id=0 位置都从 0 起）
        singles = []
        for i in range(2):
            eng._model._ctx.kv_cache_clear()
            singles.append(eng.forward_layers_from_hidden(h[i], n_past=0, all_logits=False))
        single = np.stack(singles)
        # 多序列（一次 decode，两个 seq_id）
        eng._model._ctx.kv_cache_clear()
        lg = eng.forward_layers_from_hidden(h, seq_ids=[0, 1], positions=[0, 0], all_logits=True)
        # ⚠️ 判据用 **`argmax` 相同**，而非逐位相等或 `allclose`：llama.cpp 与 torch **同源** ——
        #    batch=1 与 batch=2 会走不同 kernel（GEMV vs GEMM），f16 下 logits 尾部会出现
        #    超过 1% 的舍入差异（本测试实测 max|diff| 已大于 1e-2）。**下游实际只用 argmax**，
        #    故「选取的 token 一致」才是正确的验收判据。
        assert np.array_equal(single.argmax(axis=-1), lg.argmax(axis=-1)), (
            "多序列与单序列的 argmax 必须相同，实得 "
            f"{single.argmax(axis=-1)} vs {lg.argmax(axis=-1)}")

    @pytest.mark.real_model
    def test_roundtrip_hidden_width_matches_downstream(self):
        """★ 对称性：上游出的 hidden 宽度 == 下游入口接受的宽度（两端可直连）。"""
        eng = _engine_or_skip()
        h = eng.forward_layers_to_hidden([100, 200], n_past=0)
        eng._model._ctx.kv_cache_clear()
        lg = eng.forward_layers_from_hidden(h, n_past=0)
        assert lg is not None, "上游 hidden 应能被下游入口直接接受"


@pytest.mark.parametrize("failure", [False, True])
def test_fake_binding_restores_preexisting_orphan_child(monkeypatch, failure):
    """A stub leftover from another unit cannot force native import or leak."""
    from llama_engine import _new_llama_with_seq_max

    for name in list(sys.modules):
        if name == "llama_cpp" or name.startswith("llama_cpp."):
            monkeypatch.delitem(sys.modules, name)
    orphan = ModuleType("llama_cpp.llama_cpp")
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_cpp", orphan)
    with _isolated_llama_binding() as binding:
        binding.fail = failure
        if failure:
            with pytest.raises(binding.ConstructorFailure):
                _new_llama_with_seq_max({"model_path": "offline.gguf"}, 4)
        else:
            assert _new_llama_with_seq_max({"model_path": "offline.gguf"}, 4).n_seq_max == 4
        assert sys.modules["llama_cpp.llama_cpp"] is binding.low_level
    assert "llama_cpp" not in sys.modules
    assert sys.modules["llama_cpp.llama_cpp"] is orphan


def test_real_model_paths_require_opt_in_and_keep_their_marks(monkeypatch):
    monkeypatch.delenv("QLH_RUN_PHYSICAL_TESTS", raising=False)
    monkeypatch.delenv("QLH_RUN_REAL_MODEL_SMOKE", raising=False)
    # Stop before even constructing an engine or checking an existing GGUF.
    with pytest.raises(pytest.skip.Exception, match="requires"):
        _engine_or_skip()
    assert any(mark.name == "real_model" for mark in TestWithRealModel.pytestmark)
    for name in ("test_multi_sequence_decode_matches_single", "test_roundtrip_hidden_width_matches_downstream"):
        assert any(mark.name == "real_model" for mark in getattr(TestMultiSequence, name).pytestmark)
    for name in ("test_default_path_unchanged", "test_seq_max_is_restored_after_load"):
        assert not any(mark.name == "real_model" for mark in getattr(TestMultiSequence, name).pytestmark)


@pytest.mark.parametrize("flag", ["QLH_RUN_PHYSICAL_TESTS", "QLH_RUN_REAL_MODEL_SMOKE"])
def test_real_model_gate_accepts_explicit_switch(monkeypatch, flag):
    monkeypatch.delenv("QLH_RUN_PHYSICAL_TESTS", raising=False)
    monkeypatch.delenv("QLH_RUN_REAL_MODEL_SMOKE", raising=False)
    monkeypatch.setenv(flag, "1")
    assert _require_real_model_opt_in() is None


@pytest.mark.parametrize("load_fails", [False, True])
def test_explicit_model_load_exposes_failures_and_forwards_options(monkeypatch, tmp_path, load_fails):
    """Exercise the smoke helper using an offline engine, never real weights."""
    import llama_engine

    artifact = tmp_path / "stub.gguf"
    artifact.write_bytes(b"offline-helper-test")
    monkeypatch.setattr(sys.modules[__name__], "CUT_GGUF", artifact)
    monkeypatch.setenv("QLH_RUN_REAL_MODEL_SMOKE", "1")
    calls = []
    unloaded = []

    class BindingFailure(RuntimeError):
        pass

    class FakeEngine:
        def load_model(self, **kwargs):
            calls.append(kwargs)
            if load_fails:
                raise BindingFailure("binding load failed")

        def unload(self):
            unloaded.append(self)

    monkeypatch.setattr(llama_engine, "LlamaCppEngine", FakeEngine)
    if load_fails:
        with pytest.raises(BindingFailure, match="binding load failed"):
            _engine_or_skip(n_ctx=1024, n_seq_max=4)
        assert len(unloaded) == 1
    else:
        assert isinstance(_engine_or_skip(n_ctx=1024, n_seq_max=4), FakeEngine)
        assert not unloaded
    assert calls == [{"model_path": str(artifact), "n_ctx": 1024, "n_threads": 4, "n_seq_max": 4}]
