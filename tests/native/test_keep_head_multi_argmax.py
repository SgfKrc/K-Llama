"""Exercise output atomicity and input boundaries of the real multi-position C ABI."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SENTINEL = 12345


@pytest.fixture(scope="module")
def multi_argmax(tmp_path_factory, native_executable_guard):
    compiler = os.environ.get("QLH_NATIVE_TEST_CC") or shutil.which("gcc") or shutil.which("clang")
    if not compiler:
        pytest.skip("native multi-position harness requires a C compiler")
    destination = tmp_path_factory.mktemp("native-multi-argmax")
    executable = destination / ("argmax.exe" if os.name == "nt" else "argmax")
    environment = dict(os.environ)
    environment["PATH"] = str(Path(compiler).resolve().parent) + os.pathsep + environment.get("PATH", "")
    result = subprocess.run(
        [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-Wno-unused-parameter",
         str(Path(__file__).with_name("keep_head_multi_argmax_harness.c")),
         "-I" + str(ROOT / "android/app/src/main/cpp/llama.cpp/include"),
         "-I" + str(ROOT / "android/app/src/main/cpp/llama.cpp/ggml/include"),
         "-o", str(executable)],
        capture_output=True, text=True, env=environment, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    native_executable_guard(executable)

    def run(logits, *, decode_rc=0, missing_row=-1, missing_model=False, missing_vocab=False,
            **options):
        columns = len(logits[0])
        command = [str(executable), str(len(logits)), str(columns), str(decode_rc),
                   str(missing_row), str(int(missing_model)), str(int(missing_vocab)),
                   *(str(value) for row in logits for value in row)]
        for name, value in options.items():
            command.append("--" + name.replace("_", "-"))
            if not isinstance(value, bool):
                command.extend(str(item) for item in (value if isinstance(value, list) else [value]))
        result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=10)
        assert result.returncode == 0, result.stdout + result.stderr
        data = json.loads(result.stdout)
        assert data["allocated"] == data["freed"]
        return data

    return run


@pytest.mark.parametrize("rows", [1, 2, 4, 8])
@pytest.mark.parametrize("rope", [1, 4])
def test_each_row_argmax_with_capacity_canaries(multi_argmax, rows, rope):
    logits = [[float(i % 3 == j) for j in range(3)] for i in range(rows)]
    result = multi_argmax(logits, capacity=10, rope=rope, past=7,
                          seq=[3] * rows, pos=list(range(7, 7 + rows)), nseq=[1] * rows)
    assert result["rc"] == result["decode_error"] == 0
    assert result["output"] == [SENTINEL] + [i % 3 for i in range(rows)] + [SENTINEL] * (11 - rows)
    assert result["allocated"] == 1


def test_default_metadata_and_ties_use_first_vocab_index(multi_argmax):
    result = multi_argmax([[-2, -1, -1], [0.0, -0.0, 0.0], [3e38, -3e38, 0.0]])
    assert result["rc"] == 0 and result["output"][1:4] == [1, 0, 0]


@pytest.mark.parametrize("row", [0, 1, 2])
@pytest.mark.parametrize("column", [0, 1, 2])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_in_any_row_does_not_publish_any_result(multi_argmax, row, column, bad):
    logits = [[0.0, 2.0, 1.0] for _ in range(3)]
    logits[row][column] = bad
    result = multi_argmax(logits)
    assert result["rc"] == -7 and result["decode_error"] == 0
    assert result["output"] == [SENTINEL] * 12


@pytest.mark.parametrize("options", [
    {"capacity": 0}, {"capacity": 1}, {"capacity": -1},
    {"count": 0}, {"count": -1}, {"past": -1}, {"past": 2**31 - 1},
    {"pos": [0, 0]}, {"pos": [-1, 0]}, {"pos": [0, 2]},
    {"pos": [2**31 - 1, -2**31]},
    {"seq": [0, 1]}, {"seq": [-1, -1]}, {"nseq": [0, 1]}, {"nseq": [1, 2]},
    {"null_handle": True}, {"null_embd": True}, {"null_out": True},
])
def test_invalid_metadata_and_capacity_fail_before_decode(multi_argmax, options):
    result = multi_argmax([[0, 1], [1, 0]], **options)
    assert result["rc"] == -5
    assert result["allocated"] == 0
    assert result["output"] == [SENTINEL] * 12
    assert result["decode_error"] == (99 if options.get("null_handle") else 0)


def test_last_int32_position_is_valid_without_overflow(multi_argmax):
    result = multi_argmax([[1, 0]], past=2**31 - 1, pos=[2**31 - 1])
    assert result["rc"] == 0 and result["output"][1] == 0


@pytest.mark.parametrize("missing_row", [0, 1, 2])
def test_missing_logits_does_not_publish_partial_result(multi_argmax, missing_row):
    result = multi_argmax([[1, 0]] * 3, missing_row=missing_row)
    assert result["rc"] == -6 and result["output"] == [SENTINEL] * 12


@pytest.mark.parametrize("options", [{"missing_model": True}, {"missing_vocab": True}])
def test_missing_vocabulary_fails_atomically(multi_argmax, options):
    result = multi_argmax([[1, 0]] * 2, **options)
    assert result["rc"] == -6 and result["output"] == [SENTINEL] * 12


def test_empty_vocabulary_fails_atomically(multi_argmax):
    result = multi_argmax([[]])
    assert result["rc"] == -6 and result["output"] == [SENTINEL] * 12


@pytest.mark.parametrize("decode_rc", [1, -1, -3])
def test_decode_failure_preserves_error_without_output(multi_argmax, decode_rc):
    result = multi_argmax([[1, 0]], decode_rc=decode_rc)
    assert result["rc"] == -2 and result["decode_error"] == decode_rc
    assert result["output"] == [SENTINEL] * 12


@pytest.mark.parametrize("past", [0, -1, 2**31 - 1])
@pytest.mark.parametrize("rope", [1, 4])
def test_explicit_positions_override_past_including_overflow_check(multi_argmax, past, rope):
    result = multi_argmax([[0, 1], [1, 0], [0, 2]], past=past, pos=[41, 42, 43], rope=rope)
    assert result["rc"] == 0 and result["output"][1:4] == [1, 0, 1]


def test_explicit_position_overflow_is_rejected_even_with_zero_past(multi_argmax):
    result = multi_argmax([[0, 1], [1, 0]], pos=[2**31 - 1, 0])
    assert result["rc"] == -5 and result["allocated"] == 0
    assert result["output"] == [SENTINEL] * 12
