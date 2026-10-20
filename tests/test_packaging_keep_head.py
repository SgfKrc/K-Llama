"""Keep-head package closure and frozen resolution, without loading native DLLs."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import keep_head_shim

ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging/packaging"
_spec = importlib.util.spec_from_file_location("keep_head_resources", PACKAGING / "keep_head_resources.py")
resources = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(resources)


@pytest.fixture
def native_group(tmp_path, monkeypatch):
    root = tmp_path / "core"
    directory = root / resources.NATIVE_RELATIVE
    directory.mkdir(parents=True)
    for name in resources.REQUIRED_DLLS:
        (directory / name).write_bytes(b"native fixture")
    metadata = {name: ({}, set(resources.REQUIRED_EXPORTS.get(name, ())))
                for name in resources.REQUIRED_DLLS}
    metadata["qlh_keep_head.dll"][0]["libllama.dll"] = {"llama_get_embeddings_layer_inp"}
    metadata["libllama.dll"][0]["ggml-base.dll"] = {"ggml_backend_init"}
    metadata["ggml-base.dll"][1].add("ggml_backend_init")
    metadata["ggml.dll"][0]["ggml-cpu.dll"] = {"ggml_backend_cpu_init"}
    metadata["ggml-cpu.dll"][1].add("ggml_backend_cpu_init")
    metadata["ggml-base.dll"][0]["libgomp-1.dll"] = {"omp_get_max_threads"}
    metadata["libgomp-1.dll"][1].add("omp_get_max_threads")
    metadata["libstdc++-6.dll"][0].update({
        "kernel32.dll": {"GetLastError"},
        "api-ms-win-crt-runtime-l1-1-0.dll": {"abort"},
    })
    monkeypatch.setattr(resources, "_inspect_pe", lambda path: metadata[path.name])
    return root, directory, metadata


def test_source_default_uses_repo_not_cwd_or_meipass(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "unrelated-bundle"), raising=False)
    monkeypatch.setattr(keep_head_shim, "__file__", str(tmp_path / "core/src/keep_head_shim.py"))
    monkeypatch.delenv(keep_head_shim.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    candidate = tmp_path / "core" / keep_head_shim.DEFAULT_SHIM_RELATIVE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"shim")
    assert keep_head_shim.default_shim_path() == candidate
    assert keep_head_shim.resolve_keep_head_shim() == str(candidate)
    assert candidate.parent.relative_to(tmp_path / "core") == resources.NATIVE_RELATIVE


@pytest.mark.parametrize("has_meipass", [True, False])
def test_frozen_resolves_private_group(tmp_path, monkeypatch, has_meipass):
    internal = tmp_path / "product/_internal"
    module_root = tmp_path / "module-location" if has_meipass else internal
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(keep_head_shim, "__file__", str(module_root / "keep_head_shim.py"))
    monkeypatch.delenv(keep_head_shim.ENV_VAR, raising=False)
    if has_meipass:
        monkeypatch.setattr(sys, "_MEIPASS", str(internal), raising=False)
    else:
        monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    candidate = internal / resources.PRIVATE_NATIVE_DIR / "qlh_keep_head.dll"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"shim")
    monkeypatch.chdir(tmp_path)
    assert keep_head_shim.default_shim_path() == candidate
    assert keep_head_shim.resolve_keep_head_shim() == str(candidate)


def test_frozen_missing_does_not_fall_back_to_developer_group(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    internal = tmp_path / "core/_internal"
    monkeypatch.setattr(sys, "_MEIPASS", str(internal), raising=False)
    monkeypatch.setattr(keep_head_shim, "__file__", str(tmp_path / "core/src/keep_head_shim.py"))
    monkeypatch.delenv(keep_head_shim.ENV_VAR, raising=False)
    for candidate in (tmp_path / "core" / keep_head_shim.DEFAULT_SHIM_RELATIVE,
                      internal / "qlh_keep_head.dll"):
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(b"unbundled shim")
    assert keep_head_shim.resolve_keep_head_shim() == ""
    hint = keep_head_shim.missing_shim_hint()
    assert str(internal / keep_head_shim.BUNDLED_SHIM_RELATIVE) in hint
    assert keep_head_shim.ENV_VAR in hint


@pytest.mark.parametrize(("explicit", "environment", "expected"), [
    (" C:/dev/explicit.dll ", " C:/dev/env.dll ", "C:/dev/explicit.dll"),
    (None, " C:/dev/env.dll ", "C:/dev/env.dll"),
    (" ", " C:/dev/env.dll ", "C:/dev/env.dll"),
])
def test_frozen_keeps_explicit_development_override(monkeypatch, explicit, environment, expected):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv(keep_head_shim.ENV_VAR, environment)
    assert keep_head_shim.resolve_keep_head_shim(explicit) == expected


def test_collects_only_approved_native_files_in_private_directory(native_group):
    root, directory, _ = native_group
    (directory / "unrelated.dll").write_bytes(b"must not be collected")
    (directory / "llama-bench.exe").write_bytes(b"must not be collected")
    toc = resources.collect_keep_head_binaries(root)
    assert len(toc) == 9
    assert {Path(dest).name for dest, _, _ in toc} == set(resources.REQUIRED_DLLS)
    assert all(Path(dest).parent.as_posix() == resources.PRIVATE_NATIVE_DIR for dest, _, _ in toc)
    assert all(Path(source).parent == directory and kind == "BINARY" for _, source, kind in toc)


@pytest.mark.parametrize("missing", resources.REQUIRED_DLLS)
def test_missing_any_required_file_fails_with_name(native_group, missing):
    root, directory, _ = native_group
    (directory / missing).unlink()
    with pytest.raises(FileNotFoundError, match="keep_head_native_missing") as error:
        resources.collect_keep_head_binaries(root)
    assert str(directory / missing) in str(error.value)


@pytest.mark.parametrize(("library", "symbol"), [
    (library, symbol) for library, symbols in resources.REQUIRED_EXPORTS.items() for symbol in sorted(symbols)
])
def test_required_shim_or_patched_export_cannot_be_missing(native_group, library, symbol):
    root, _, metadata = native_group
    metadata[library][1].remove(symbol)
    with pytest.raises(RuntimeError, match="keep_head_native_symbol_missing") as error:
        resources.collect_keep_head_binaries(root)
    assert library in str(error.value) and symbol in str(error.value)


def test_legacy_shim_without_load_receipts_is_rejected_before_build(native_group):
    root, _, metadata = native_group
    metadata["qlh_keep_head.dll"][1].difference_update({
        "qlh_kh_model_bytes", "qlh_kh_context_size", "qlh_kh_batch_size",
    })
    with pytest.raises(RuntimeError, match="keep_head_native_symbol_missing"):
        resources.collect_keep_head_binaries(root)


def test_unknown_dependency_fails_even_if_a_sibling_file_exists(native_group):
    root, directory, metadata = native_group
    metadata["ggml-cpu.dll"][0]["unexpected-backend.dll"] = {"backend_init"}
    (directory / "unexpected-backend.dll").write_bytes(b"not approved")
    with pytest.raises(FileNotFoundError, match="keep_head_native_dependency_missing") as error:
        resources.collect_keep_head_binaries(root)
    assert "ggml-cpu.dll" in str(error.value) and "unexpected-backend.dll" in str(error.value)


def test_mismatched_same_name_ggml_fails_before_packaging(native_group):
    root, _, metadata = native_group
    metadata["ggml-base.dll"][1].remove("ggml_backend_init")
    with pytest.raises(RuntimeError, match="keep_head_native_import_missing") as error:
        resources.collect_keep_head_binaries(root)
    assert "ggml-base.dll" in str(error.value) and "ggml_backend_init" in str(error.value)


def test_invalid_native_binary_is_rejected_without_loading(tmp_path):
    path = tmp_path / "qlh_keep_head.dll"
    path.write_bytes(b"invalid PE")
    with pytest.raises(RuntimeError, match="keep_head_native_invalid") as error:
        resources._inspect_pe(path)
    assert str(path) in str(error.value)


def _execute_edge_spec(monkeypatch, native_group):
    root, _, _ = native_group
    (root / "src").mkdir(exist_ok=True)
    pip_lib = root / "venv/llama_cpp/lib"
    pip_lib.mkdir(parents=True, exist_ok=True)
    for name in ("llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll"):
        (pip_lib / name).write_bytes(b"pip ABI")
    calls = []

    def analysis(*args, **kwargs):
        calls.append(("Analysis", args, kwargs))
        return SimpleNamespace(pure=[], zipped_data=[], scripts=[], binaries=[
            (f"{dest}/{Path(source).name}", source, "BINARY") for source, dest in kwargs["binaries"]
        ], datas=[])

    def collect(*args, **kwargs):
        calls.append(("COLLECT", args, kwargs))

    monkeypatch.setenv("QLH_CORE_ROOT", str(root))
    monkeypatch.setattr(sys, "path", sys.path[:])
    monkeypatch.setitem(sys.modules, "keep_head_resources", resources)
    monkeypatch.setitem(sys.modules, "model_supply_resources", SimpleNamespace(
        collect_model_supply_resources=lambda _, **_kw: ([], [])))
    monkeypatch.setitem(sys.modules, "scripts.model_tools.llama_quantize_toolchain", SimpleNamespace(
        verify_managed_package=lambda *args, **kwargs: {"valid": False}))
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(__file__=str(pip_lib.parent / "__init__.py")))
    namespace = {"SPECPATH": str(PACKAGING), "Analysis": analysis, "COLLECT": collect,
                 "PYZ": lambda *a, **k: None, "EXE": lambda *a, **k: None}
    return calls, namespace


def test_edge_spec_preserves_both_native_groups_without_shared_root_dlls(monkeypatch, native_group):
    calls, namespace = _execute_edge_spec(monkeypatch, native_group)
    exec(compile((PACKAGING / "qlh-edge.spec").read_text(encoding="utf-8"), "qlh-edge.spec", "exec"), namespace)
    analysis, collection = calls
    pip_inputs = analysis[2]["binaries"]
    assert len(pip_inputs) == 4 and all(dest == "llama_cpp/lib" for _, dest in pip_inputs)
    toc = [entry for group in collection[1][1:] for entry in group]
    expected_private = {f"{resources.PRIVATE_NATIVE_DIR}/{name}" for name in resources.REQUIRED_DLLS}
    assert expected_private <= {dest for dest, _, _ in toc}
    collisions = {"ggml.dll", "ggml-base.dll", "ggml-cpu.dll"}
    for name in collisions:
        matches = [(dest, source) for dest, source, _ in toc if Path(dest).name == name]
        assert len(matches) == 2
        assert {Path(dest).parent.as_posix() for dest, _ in matches} == {"llama_cpp/lib", "keep_head"}
        assert len({source for _, source in matches}) == 2
    assert all(Path(dest).parent.as_posix() != "." for dest, _, _ in toc)
    assert collection[2]["strip"] is False and collection[2]["upx"] is False


def test_edge_spec_missing_group_stops_before_analysis(monkeypatch, native_group):
    calls, namespace = _execute_edge_spec(monkeypatch, native_group)
    (native_group[1] / "libgomp-1.dll").unlink()
    with pytest.raises(FileNotFoundError, match="keep_head_native_missing.*libgomp-1.dll"):
        exec(compile((PACKAGING / "qlh-edge.spec").read_text(encoding="utf-8"), "qlh-edge.spec", "exec"), namespace)
    assert calls == []


@pytest.mark.parametrize("cuda_version", [None, "12.6"])
def test_cuda_tui_spec_requires_gpu_torch_and_keeps_both_native_groups(monkeypatch, native_group, cuda_version):
    calls, namespace = _execute_edge_spec(monkeypatch, native_group)
    monkeypatch.setenv("QLH_EDGE_FREEZE_TORCH", "1")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(cuda=cuda_version)))
    code = compile((PACKAGING / "qlh-edge.spec").read_text(encoding="utf-8"), "qlh-edge.spec", "exec")
    if cuda_version is None:
        with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
            exec(code, namespace)
        assert calls == []
        return
    exec(code, namespace)
    analysis, collection = calls
    imports = analysis[2]["hiddenimports"]
    assert {"torch", "model_module", "transformers.models.qwen3_5.modeling_qwen3_5", "tokenizers"} <= set(imports)
    assert "torch" not in analysis[2]["excludes"] and "transformers" not in analysis[2]["excludes"]
    assert collection[2]["name"] == "QLH-Edge-Inference-CUDA"
    toc = [entry for group in collection[1][1:] for entry in group]
    assert {f"keep_head/{name}" for name in resources.REQUIRED_DLLS} <= {dest for dest, _, _ in toc}
    assert any(dest == "llama_cpp/lib/llama.dll" for dest, _, _ in toc)


def test_cpu_tui_spec_continues_excluding_torch(monkeypatch, native_group):
    calls, namespace = _execute_edge_spec(monkeypatch, native_group)
    monkeypatch.setenv("QLH_EDGE_FREEZE_TORCH", "0")
    exec(compile((PACKAGING / "qlh-edge.spec").read_text(encoding="utf-8"), "qlh-edge.spec", "exec"), namespace)
    assert {"torch", "transformers"} <= set(calls[0][2]["excludes"])
    assert "model_module" not in calls[0][2]["hiddenimports"]
    assert calls[1][2]["name"] == "QLH-Edge-Inference"


def test_tui_spec_rejects_unknown_torch_build_choice(monkeypatch, native_group):
    calls, namespace = _execute_edge_spec(monkeypatch, native_group)
    monkeypatch.setenv("QLH_EDGE_FREEZE_TORCH", "true")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        exec(compile((PACKAGING / "qlh-edge.spec").read_text(encoding="utf-8"), "qlh-edge.spec", "exec"), namespace)
    assert calls == []
