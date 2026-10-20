"""Compile and execute the real tail argmax with stub llama outputs; no model load."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def native_argmax(tmp_path_factory, native_executable_guard):
    compiler = os.environ.get("QLH_NATIVE_TEST_CC") or shutil.which("gcc") or shutil.which("clang")
    if not compiler and os.name == "nt":
        candidate = Path("C:/msys64/ucrt64/bin/gcc.exe")
        compiler = str(candidate) if candidate.is_file() else None
    if not compiler:
        pytest.skip("native argmax regression requires a C compiler (QLH_NATIVE_TEST_CC)")
    destination = tmp_path_factory.mktemp("native-argmax")
    executable = destination / ("argmax.exe" if os.name == "nt" else "argmax")
    environment = dict(os.environ)
    environment["PATH"] = str(Path(compiler).resolve().parent) + os.pathsep + environment.get("PATH", "")
    command = [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-Wno-unused-parameter",
               str(Path(__file__).with_name("keep_head_argmax_harness.c")),
               "-I" + str(ROOT / "android/app/src/main/cpp/llama.cpp/include"),
               "-I" + str(ROOT / "android/app/src/main/cpp/llama.cpp/ggml/include"),
               "-o", str(executable)]
    built = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=60)
    assert built.returncode == 0, built.stdout + built.stderr
    native_executable_guard(executable)

    def run(logits, *, decode_rc=0, missing_row=-1, missing_model=False, missing_vocab=False):
        columns = len(logits[0])
        assert all(len(row) == columns for row in logits)
        result = subprocess.run(
            [str(executable), str(len(logits)), str(columns), str(decode_rc), str(missing_row),
             str(int(missing_model)), str(int(missing_vocab)),
             *(str(value) for row in logits for value in row)],
            capture_output=True, text=True, env=environment, timeout=10,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        data = json.loads(result.stdout)
        assert data["guard"] == 1 and data["null_n_vocab"] == 0
        assert data["allocated"] == data["freed"] == 1
        return data, result.stderr

    return run


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "posinf", "neginf"])
@pytest.mark.parametrize("index", [0, 1, 3], ids=["first", "middle", "last"])
def test_native_tail_rejects_every_nonfinite_logit(native_argmax, bad, index):
    row = [-4.0, -2.0, -1.0, -3.0]
    row[index] = bad
    data, error = native_argmax([row])
    assert data["rc"] == -7 and data["token"] == -1
    assert data["decode_error"] == 0
    assert f"non-finite logits (row=0, vocab_index={index})" in error


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "posinf", "neginf"])
def test_native_tail_rejects_nonfinite_prefill_row_even_if_last_row_is_finite(native_argmax, bad):
    data, error = native_argmax([[bad, 0.0, 1.0], [0.0, 2.0, 1.0]])
    assert data["rc"] == -7 and data["token"] == -1
    assert "row=0, vocab_index=0" in error


@pytest.mark.parametrize("row,token", [
    ([-4.0, -2.0, -1.0, -3.0], 2),
    ([7.0, 7.0, 6.0], 0),
    ([-0.0, 0.0], 0),
    ([-3.402823466e38, 3.402823466e38, 0.0], 1),
    ([1e-40, 2e-40, 0.0], 1),
    ([0.0], 0),
])
def test_native_tail_finite_argmax_and_vocab_query(native_argmax, row, token):
    data, error = native_argmax([row])
    assert data["rc"] == 0 and data["token"] == token
    assert data["n_vocab"] == len(row) and data["decode_error"] == 0
    assert not error


def test_native_tail_uses_last_batch_row(native_argmax):
    data, _ = native_argmax([[3.0, 0.0], [0.0, 4.0]])
    assert data["rc"] == 0 and data["token"] == 1


@pytest.mark.parametrize("missing_row", [0, 1])
def test_native_tail_missing_logits_fails_without_stale_token(native_argmax, missing_row):
    data, _ = native_argmax([[3.0, 0.0], [0.0, 4.0]], missing_row=missing_row)
    assert data["rc"] == -6 and data["token"] == -1


@pytest.mark.parametrize("missing", ["missing_model", "missing_vocab"])
def test_native_tail_missing_vocabulary_fails(native_argmax, missing):
    data, _ = native_argmax([[1.0]], **{missing: True})
    assert data["rc"] == -6 and data["token"] == -1 and data["n_vocab"] == 0


def test_native_tail_preserves_decode_failure_code(native_argmax):
    data, _ = native_argmax([[1.0]], decode_rc=1)
    assert data["rc"] == -2 and data["token"] == -1 and data["decode_error"] == 1
