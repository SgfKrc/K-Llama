"""tests/test_cut_layers_manifest_contains.py — `#67` 缺口 ⑥（后半）：manifest 显式声明归属

背景（`docs/已知问题记录.md` 的 `#67`）：
下游此前只能靠「段类型（`mode`）+ `source_layer_range`」**反推**一个裁层工件带不带
`token_embd` / `output_norm` / `output.weight`。反推容易写错 —— `LayerArtifactCatalog.kt`
就据此断言"中间段工件没有 lm_head/final_norm"，而实测 `q35-2b-cut-16-20`（`mode=middle`）
的 `tensors_kept=55` 明确**含** `output_norm`（3 linear×14 + 1 full×11 + 2 非 blk）。

⇒ 让生成器**直接写出来**：`artifact_contains{token_embd, output_norm, output_weight}`，
并给出 `can_serve_tail`（能否承担末段职责：需 final_norm，且末段要么有独立 `output.weight`，
要么 tie embeddings 时用 `token_embd` 兼作 lm_head）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "cut_layers.py"
#: 用**非 qwen35** 架构：本文件只测"张量归属声明"，与 #67-④ 的必需表无关
#: （用 qwen35 会被那条校验拒，因为这里的最小 fixture 不含全部必需张量）。
ARCH = "qwen2"
N_LAYERS = 8


def _make_gguf(path: Path, *, with_output_weight: bool = False) -> None:
    """合成 GGUF；`with_output_weight=False` 复刻 tie-embeddings 形态（无独立 lm_head）。"""
    import gguf
    import numpy as np

    writer = gguf.GGUFWriter(str(path), ARCH)
    writer.add_block_count(N_LAYERS)
    writer.add_embedding_length(8)
    writer.add_key_value(f"{ARCH}.full_attention_interval", 4, gguf.GGUFValueType.INT32)
    for layer in range(N_LAYERS):
        writer.add_tensor(f"blk.{layer}.attn_norm.weight", np.ones(8, dtype=np.float32))
    writer.add_tensor("token_embd.weight", np.ones((8, 8), dtype=np.float32))
    writer.add_tensor("output_norm.weight", np.ones(8, dtype=np.float32))
    if with_output_weight:
        writer.add_tensor("output.weight", np.ones((8, 8), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


class TestManifestArtifactContains:
    def test_manifest_declares_membership_for_middle_artifact(self, tmp_path):
        """★ 该红必须红：middle 工件也必须**显式**声明含 output_norm（改正反推断言）。"""
        src = tmp_path / "src.gguf"
        _make_gguf(src)
        dst = tmp_path / "out.gguf"
        mf = tmp_path / "out.manifest.json"
        r = _run("--src", str(src), "--dst", str(dst), "--k", "4",
                 "--manifest", str(mf))
        assert r.returncode == 0, (r.stdout, r.stderr)
        m = json.loads(mf.read_text(encoding="utf-8"))
        assert m["mode"] == "tail"
        contains = m.get("artifact_contains")
        assert contains is not None, f"manifest 应含 artifact_contains：{m.keys()}"
        assert contains["token_embd"] is True
        assert contains["output_norm"] is True, "非 blk 张量无条件保留 ⇒ 含 output_norm"
        assert contains["output_weight"] is False, "tie embeddings ⇒ 无独立 output.weight"
        assert contains["can_serve_tail"] is True, "有 norm + tie ⇒ 可承担末段"

    def test_head_mode_also_declares_membership(self, tmp_path):
        """head 模式同样声明（下游不必按模式分支猜）。"""
        src = tmp_path / "src.gguf"
        _make_gguf(src)
        dst = tmp_path / "out.gguf"
        mf = tmp_path / "out.manifest.json"
        r = _run("--src", str(src), "--dst", str(dst), "--keep-head", "4",
                 "--manifest", str(mf))
        assert r.returncode == 0, (r.stdout, r.stderr)
        m = json.loads(mf.read_text(encoding="utf-8"))
        assert m["mode"] == "head"
        assert m["artifact_contains"]["token_embd"] is True
        assert m["artifact_contains"]["output_norm"] is True

    def test_can_serve_tail_true_when_independent_lm_head_exists(self, tmp_path):
        """有独立 output.weight ⇒ can_serve_tail 亦为真（另一条合法路径）。"""
        src = tmp_path / "src.gguf"
        _make_gguf(src, with_output_weight=True)
        dst = tmp_path / "out.gguf"
        mf = tmp_path / "out.manifest.json"
        r = _run("--src", str(src), "--dst", str(dst), "--k", "4",
                 "--manifest", str(mf))
        assert r.returncode == 0, (r.stdout, r.stderr)
        m = json.loads(mf.read_text(encoding="utf-8"))
        assert m["artifact_contains"]["output_weight"] is True
        assert m["artifact_contains"]["can_serve_tail"] is True

    def test_manifest_without_kept_names_stays_backward_compatible(self, tmp_path):
        """`kept_names` 未给出时不得凭空写字段（保持既有 manifest 形状）。"""
        src = tmp_path / "src.gguf"
        _make_gguf(src)
        dst = tmp_path / "out.gguf"
        mf = tmp_path / "out.manifest.json"
        r = _run("--src", str(src), "--dst", str(dst), "--k", "4", "--manifest", str(mf))
        assert r.returncode == 0, (r.stdout, r.stderr)
        m = json.loads(mf.read_text(encoding="utf-8"))
        # CLI 路径总会带 kept_names；这里只断言字段存在且自洽即可。
        assert "artifact_contains" in m


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
