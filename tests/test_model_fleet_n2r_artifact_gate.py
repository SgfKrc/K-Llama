"""MF-MEM-N2R artifact admission tests.

★ 2026-09-30：原测试把"候选工件存在"**硬编码为 1**，依赖真实仓库里
`models/qwen-1_8b-chat` 的存在。该模型已退役（仓库里不再有 1.8B 目录），
于是断言在真实环境恒失败，被误记为"环境问题"。

现在改为**自包含**：用 `tmp_path` 造出 `project_root` 的工件布局（gate 的所有
路径都从传入的 root 派生，天然可注入），候选有/无、目标有/无两个方向都**确定性**
覆盖 —— 既不再依赖本机 `models/`，也不靠"条件分支永不进入"来假装通过。
"""

# Owned by the retired MF-MEM benchmark workspace; its one-off harness was
# removed from the main repo in aa889bcb (see .gitignore, scripts policy).
# Legacy local run: QLH_INCLUDE_WORKSPACE_TOOL_TESTS=1 python -m pytest <this file>.
# Opting in requires the real harness; keep missing imports fatal.

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import model_fleet_n2r_artifact_gate as gate  # noqa: E402
from model_fleet_n2r_artifact_gate import TARGET_FAMILY, inspect_artifacts  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]

_CANDIDATE_SHARD = "model-00001-of-00002.safetensors"
_TARGET_SHARDS = (
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
)


def _make_project(
    root: Path, *, candidate: bool, target: bool, gguf: bool = True,
) -> Path:
    """构造一个自包含的 `project_root`（不触碰真实仓库的 `models/`）。"""
    models = root / "models"
    models.mkdir(parents=True)
    if gguf:
        (models / "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf").write_bytes(b"gguf")
    if candidate:
        candidate_dir = models / "qwen-1_8b-chat"
        candidate_dir.mkdir()
        (candidate_dir / _CANDIDATE_SHARD).write_bytes(b"candidate")
    if target:
        target_dir = models / "deepseek-r1-distill-qwen-7b"
        target_dir.mkdir()
        for shard in _TARGET_SHARDS:
            (target_dir / shard).write_bytes(b"target")
        manifest_path = (
            root
            / "build"
            / "model-fleet"
            / "model-store-20260808"
            / "manifests"
            / "migration"
            / "deepseek-r1-distill-qwen-7b-safetensors"
            / "builtin-20260808.json"
        )
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "quantization": "bf16",
                    "files": [{"path": shard} for shard in _TARGET_SHARDS],
                }
            ),
            encoding="utf-8",
        )
    return root


def _force_bitsandbytes(monkeypatch, available: bool) -> None:
    """把 bitsandbytes 的"本机是否装了"钉死，让两条分支都确定性可测。"""
    monkeypatch.setattr(
        gate.importlib.util,
        "find_spec",
        lambda name: (object() if name == "bitsandbytes" else None) if available else None,
    )


def test_gate_does_not_promote_a_smaller_quant_candidate_to_target(monkeypatch, tmp_path):
    """★ 核心语义：**有**非目标量化候选 ≠ 有目标工件 —— 候选绝不能顶替 7B 目标。"""
    root = _make_project(tmp_path / "with-candidate", candidate=True, target=False)
    _force_bitsandbytes(monkeypatch, True)

    result = inspect_artifacts(root)

    assert result["candidate_quant_artifact_available"] == 1
    assert "available_quant_candidate_is_qwen_1_8b_not_7b" in result["reason_codes"]
    # 候选在，但目标工件缺 ⇒ 仍必须资源拒绝，且候选不得被计入目标。
    assert result["target_artifact_available"] == 0
    assert result["candidate_target_match"] == 0
    assert result["resource_rejected"] == 1
    assert result["status"] == "resource_rejected"
    assert "target_7b_safetensors_weights_missing" in result["reason_codes"]


def test_gate_without_any_smaller_candidate_omits_the_candidate_reason(monkeypatch, tmp_path):
    """★ 另一半分支：无候选 ⇒ 计数为 0 且**不**报"候选是 1.8B"这条 reason。"""
    root = _make_project(tmp_path / "no-candidate", candidate=False, target=True)
    _force_bitsandbytes(monkeypatch, True)

    result = inspect_artifacts(root)

    assert result["candidate_quant_artifact_available"] == 0
    assert "available_quant_candidate_is_qwen_1_8b_not_7b" not in result["reason_codes"]
    # 目标齐备 + bitsandbytes 可用 ⇒ 这次应当放行。
    assert result["target_artifact_available"] == 1
    assert result["resource_rejected"] == 0
    assert result["status"] == "ready_for_compare"
    assert "fixed_7b_bf16_weights_with_explicit_runtime_int8_nf4_recipe" in result["reason_codes"]


def test_gate_keeps_candidate_and_target_verdicts_independent(monkeypatch, tmp_path):
    """★ 候选存在时**也不得**干扰目标判定（两个计数互不牵连）。"""
    root = _make_project(tmp_path / "both", candidate=True, target=True)
    _force_bitsandbytes(monkeypatch, True)

    result = inspect_artifacts(root)

    assert result["candidate_quant_artifact_available"] == 1
    assert result["target_artifact_available"] == 1
    assert result["candidate_target_match"] == 0
    assert result["resource_rejected"] == 0
    assert result["status"] == "ready_for_compare"


def test_gate_never_promotes_gguf_q4_as_the_target_quantization(monkeypatch, tmp_path):
    """★ GGUF Q4 存在也不得当成目标量化；bitsandbytes 缺失须 fail-loud。"""
    root = _make_project(tmp_path / "gguf-only", candidate=False, target=False, gguf=True)
    _force_bitsandbytes(monkeypatch, False)

    result = inspect_artifacts(root)

    assert result["target_artifact_available"] == 0
    assert "7b_gguf_q4_k_m_is_not_target_quantization" in result["reason_codes"]
    assert "bitsandbytes_unavailable" in result["reason_codes"]
    assert result["bitsandbytes_available"] == 0


def test_gate_records_missing_7b_safetensors_weight_files(monkeypatch, tmp_path):
    """★ manifest 声明的分片与实际文件对不上时，必须如实记录（不得算作齐备）。"""
    root = _make_project(tmp_path / "manifest-complete", candidate=False, target=True)
    _force_bitsandbytes(monkeypatch, True)
    complete = inspect_artifacts(root)
    manifest_records = [
        record for record in complete["local_artifacts"] if record["format"] == "manifest"
    ]
    assert manifest_records
    assert manifest_records[0]["expected_safetensors"]
    assert manifest_records[0]["all_expected_weights_present"] is True

    # 删掉一个分片 ⇒ 同一 root 上的判定必须翻转（证明这条断言有真实判别力）。
    missing = _make_project(tmp_path / "manifest-missing", candidate=False, target=True)
    (missing / "models" / "deepseek-r1-distill-qwen-7b" / _TARGET_SHARDS[0]).unlink()
    dropped = inspect_artifacts(missing)
    dropped_records = [
        record for record in dropped["local_artifacts"] if record["format"] == "manifest"
    ]
    assert dropped_records[0]["expected_safetensors"]
    assert dropped_records[0]["all_expected_weights_present"] is False
    assert dropped["target_weight_files_present"] == 0


def test_gate_json_is_serializable(tmp_path):
    """真实 `project_root` 上跑一次 smoke（只验结构稳定 + 可序列化，不依赖工件是否存在）。"""
    result = inspect_artifacts(PROJECT_ROOT)

    assert result["ticket"] == "MF-MEM-N2R"
    assert result["target"]["family"] == TARGET_FAMILY
    json.dumps(result, ensure_ascii=False)
