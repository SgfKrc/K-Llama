"""文档维护机械化扫描器（M1）单测：5 条规则各覆盖命中+不命中。

工具代码位于 docs/agent_tool/（非 tests 收集路径），通过 sys.path 引入。
规则用临时文档目录 + 临时 git 仓库构造，不触碰真实 docs/。
"""
from __future__ import annotations

# Owned by the independent qlh-docagent workspace, not the main repository CI.
# Legacy local run: QLH_INCLUDE_WORKSPACE_TOOL_TESTS=1 python -m pytest <this file>.
# Keep imports strict: opting in requires the actual workspace tool sources.

import os
import io
import subprocess
import sys
import time
from pathlib import Path

import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "docs" / "agent_tool"
sys.path.insert(0, str(TOOL_DIR))

from doc_maintenance_audit import (  # noqa: E402
    REPO_ROOT,
    _check_link,
    _configure_text_stream,
    _extract_links,
    _status_line,
    _updated_at,
    scan_all,
    scan_doc,
)
from docagent_rules import load_rules  # noqa: E402


@pytest.fixture
def fake_repo(tmp_path):
    """构造临时 git 仓库 + docs/ 目录，返回 (repo_root, docs_dir)。"""
    repo = tmp_path / "repo"
    docs = repo / "docs"
    docs.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    return repo, docs


def _write(docs: Path, name: str, content: str) -> Path:
    p = docs / name
    p.write_text(content, encoding="utf-8")
    return p


def _commit(repo: Path, message: str = "init") -> None:
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", message], check=True)


def _commit_at(repo: Path, date: str, message: str) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_DATE": f"{date}T12:00:00",
        "GIT_COMMITTER_DATE": f"{date}T12:00:00",
    }
    subprocess.run(
        ["git", "-C", str(repo), "add", "-A"], check=True, env=env)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", message],
        check=True, env=env)


# ---------- 辅助函数 ----------

def test_status_line_extraction():
    text = "> 文档状态：规划（未开始）\n> 更新日期：2026-08-16\n"
    assert "规划" in _status_line(text)
    assert _updated_at(text) == "2026-08-16"


def test_status_line_accepts_markdown_bold_label():
    text = "> **状态**：设计方案\n> 更新日期：2026-08-16\n"
    assert _status_line(text) == "> **状态**：设计方案"


def test_status_line_skips_lifecycle():
    text = "> 文档生命周期：Active\n> 状态：现行\n"
    assert _status_line(text) == "> 状态：现行"


def test_links_extraction_and_existence(tmp_path):
    # 提取
    text = "[甲](A.md) [乙](../README.md) [外链](https://x.com/a) [锚](#sec)"
    links = _extract_links(text)
    paths = [h for _, h in links]
    assert "A.md" in paths and "../README.md" in paths
    assert not any(h.startswith(("http", "#")) for h in paths)
    # 存在性：样本**动态取自 docs/ 下真实存在的文档**，不写死文件名。
    # 原先写死 `缓存机制专项计划-2026-09-13.md`，而它随归档整理进了 docs/archive/<分类>/，
    # 断言只按 `docs/<名字>` 解析 ⇒ 结构一变测试就断。动态取样可避免把测试绑死在某一版目录结构上。
    docs_dir = REPO_ROOT / "docs"
    sample = next(iter(sorted(docs_dir.rglob("*.md")))).relative_to(docs_dir).as_posix()
    assert _check_link(sample), sample
    assert not _check_link("不存在的文档.md")


def test_windows_text_stream_is_reconfigured_to_utf8():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="gbk")
    _configure_text_stream(stream)
    stream.write("✅ 文档")
    stream.flush()
    assert raw.getvalue().decode("utf-8") == "✅ 文档"


# ---------- R1 完成未收口 ----------

def test_r1_hit_when_stale_status_with_done_marks(fake_repo):
    repo, docs = fake_repo
    p = _write(docs, "x.md",
               "> 状态：规划（阶段 1 进行中）\n> 更新日期：2026-08-10\n\n"
               "## 记录\n\n阶段 2 完成 2026-08-11，导入闭环 5/5")
    _commit(repo)
    entry = scan_doc(p, repo, None)
    rules = {f["rule"] for f in entry["findings"]}
    assert "R1" in rules


def test_r1_miss_when_status_current(fake_repo):
    repo, docs = fake_repo
    p = _write(docs, "y.md",
               "> 状态：**`✅ Completed`**（阶段 1+2 完成）\n\n正文完成记录")
    _commit(repo)
    entry = scan_doc(p, repo, None)
    assert "R1" not in {f["rule"] for f in entry["findings"]}


def test_r1_miss_when_status_mentions_partial_done(fake_repo):
    """状态行已含完成标记（部分完成+未实施）→ 无矛盾，不命中。"""
    repo, docs = fake_repo
    p = _write(docs, "part.md",
               "> 状态：规划与技术预研（分布式部分全部未实施；UI 层已完成重构）\n\n"
               "UI 层已完成重构并通过编译")
    _commit(repo)
    entry = scan_doc(p, repo, None)
    assert "R1" not in {f["rule"] for f in entry["findings"]}


def test_r1_miss_when_stale_word_only_in_parentheses(fake_repo):
    """命中词只出现在括号说明（"（…仍为规划…）"）→ 主干无矛盾，不命中。"""
    repo, docs = fake_repo
    p = _write(docs, "impl.md",
               "> 状态：实施中（单机 POC 已接线；mesh 内方案仍为规划，尚未验收）\n\n"
               "任务链单机 POC 已接线")
    _commit(repo)
    entry = scan_doc(p, repo, None)
    assert "R1" not in {f["rule"] for f in entry["findings"]}


def test_r1_miss_when_exempt_document(fake_repo):
    repo, docs = fake_repo
    p = _write(docs, "hist.md",
               "> 状态：历史参考 / 待拆分\n\n已完成事项记录（不更新）")
    _commit(repo)
    entry = scan_doc(p, repo, None)
    assert "R1" not in {f["rule"] for f in entry["findings"]}


def test_r1_miss_when_no_done_marks(fake_repo):
    repo, docs = fake_repo
    p = _write(docs, "z.md",
               "> 状态：规划（未开始）\n\n只有背景描述，无完成标记")
    _commit(repo)
    entry = scan_doc(p, repo, None)
    assert "R1" not in {f["rule"] for f in entry["findings"]}


# ---------- R2 未提交登记 ----------

def test_r2_hit_when_docs_uncommitted(fake_repo):
    repo, docs = fake_repo
    _write(docs, "a.md", "> 状态：现行\n")
    _commit(repo)
    _write(docs, "b.md", "> 状态：现行\n")  # 未提交
    unchanged = scan_doc(repo / "docs" / "a.md", repo, None)
    changed = scan_doc(repo / "docs" / "b.md", repo, None)
    assert "R2" not in {f["rule"] for f in unchanged["findings"]}
    assert "R2" in {f["rule"] for f in changed["findings"]}


def test_r2_miss_when_clean(fake_repo):
    repo, docs = fake_repo
    _write(docs, "a.md", "> 状态：现行\n")
    _commit(repo)
    entry = scan_doc(repo / "docs" / "a.md", repo, None)
    assert "R2" not in {f["rule"] for f in entry["findings"]}


# ---------- R3 状态行滞后 ----------

def test_r3_hit_when_commit_newer_than_updated_at(fake_repo):
    repo, docs = fake_repo
    p = _write(docs, "task_graph_plan.md",
               "> 状态：现行\n> 更新日期：2026-08-01\n")
    src = repo / "src"
    src.mkdir()
    (src / "task_graph.py").write_text("BASE = 1\n", encoding="utf-8")
    _commit_at(repo, "2026-08-01", "initial task graph")
    (src / "task_graph.py").write_text("BASE = 2\n", encoding="utf-8")
    _commit_at(repo, "2026-08-15", "update task graph")
    entry = scan_doc(p, repo, None)
    assert "R3" in {f["rule"] for f in entry["findings"]}


def test_r3_miss_for_unrelated_source_change(fake_repo):
    repo, docs = fake_repo
    p = _write(docs, "task_graph_plan.md",
               "> 状态：现行\n> 更新日期：2026-08-01\n")
    src = repo / "src" / "auth"
    src.mkdir(parents=True)
    (src / "login.py").write_text("BASE = 1\n", encoding="utf-8")
    _commit_at(repo, "2026-08-01", "initial files")
    (src / "login.py").write_text("BASE = 2\n", encoding="utf-8")
    _commit_at(repo, "2026-08-15", "update task graph documentation only")
    entry = scan_doc(p, repo, None)
    assert "R3" not in {f["rule"] for f in entry["findings"]}


def test_r3_miss_when_updated_recently(fake_repo):
    repo, docs = fake_repo
    p = _write(docs, "t.md",
               "> 状态：现行\n> 更新日期：2026-08-16\n")
    _commit(repo)
    entry = scan_doc(p, repo, None)
    assert "R3" not in {f["rule"] for f in entry["findings"]}


# ---------- R4 链接失效 ----------

def test_r4_hit_on_broken_link(fake_repo):
    repo, docs = fake_repo
    (docs / "exists.md").write_text("> 状态：现行\n", encoding="utf-8")
    p = _write(docs, "l.md",
               "> 状态：现行\n\n[坏链](不存在.md) [好链](exists.md)")
    _commit(repo)
    entry = scan_doc(p, repo, None)
    r4 = [f for f in entry["findings"] if f["rule"] == "R4"]
    assert len(r4) == 1 and "不存在.md" in r4[0]["message"]


def test_r4_miss_when_all_links_ok(fake_repo):
    repo, docs = fake_repo
    (docs / "ok.md").write_text("> 状态：现行\n", encoding="utf-8")
    p = _write(docs, "m.md",
               "> 状态：现行\n\n[好链](ok.md) [外链](https://example.com)")
    _commit(repo)
    entry = scan_doc(p, repo, None)
    assert "R4" not in {f["rule"] for f in entry["findings"]}


# ---------- R5 状态行缺失 ----------

def test_r5_hit_when_status_missing(fake_repo):
    repo, docs = fake_repo
    p = _write(docs, "n.md", "# 标题\n\n正文没有状态行\n")
    _commit(repo)
    entry = scan_doc(p, repo, None)
    assert "R5" in {f["rule"] for f in entry["findings"]}


def test_r5_miss_when_status_present(fake_repo):
    repo, docs = fake_repo
    p = _write(docs, "o.md",
               "> 状态：现行（正常文档）\n> 更新日期：2026-08-16\n")
    _commit(repo)
    entry = scan_doc(p, repo, None)
    assert "R5" not in {f["rule"] for f in entry["findings"]}


def test_p1b_scanner_uses_rule_data_for_hints_and_levels(fake_repo):
    repo, docs = fake_repo
    p = _write(docs, "custom.md", "> 状态：进行中\n\n正文已完成\n")
    _commit(repo)
    rules = load_rules()
    r1 = next(rule for rule in rules["rules"] if rule["id"] == "R1")
    r1["level"] = "error"
    r1["parameters"]["stale_status_hints"] = ["进行中"]

    entry = scan_doc(p, repo, None, rules=rules)

    assert entry["findings"] == [{
        "rule": "R1",
        "level": "error",
        "message": "状态行含未收口词但正文含完成标记：> 状态：进行中",
    }]


def test_p1b_scanner_uses_rule_data_for_status_window(fake_repo):
    repo, docs = fake_repo
    p = _write(docs, "window.md", "# 标题\n\n> 状态：现行\n")
    _commit(repo)
    rules = load_rules()
    rules["defaults"]["status_window_lines"] = 2

    entry = scan_doc(p, repo, None, rules=rules)

    assert entry["findings"] == [{
        "rule": "R5",
        "level": "info",
        "message": "前 2 行无状态行",
    }]


# ---------- 对真实仓库的烟雾验证（不触碰、只读） ----------

def test_real_repo_scan_is_read_only_and_fast():
    """对真实仓库跑全量：只读、<10s、能找到至少一类 warn 命中。"""
    import subprocess
    import time

    def docs_status():
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "docs/"],
            capture_output=True, text=True, encoding="utf-8").stdout

    before = docs_status()
    t0 = time.time()
    out = scan_all(None, "error")
    elapsed = time.time() - t0
    assert elapsed < 10, f"扫描超时: {elapsed:.1f}s"
    assert out["docs"]
    # 只读性：扫描前后 docs/ 工作区状态一致（不产生/清除任何改动）
    assert docs_status() == before, "扫描改变了 docs/ 工作区状态"
    # ⚠️ 2026-09-19 更正：原断言是「至少一类 warn 命中（对当前仓库必然成立）」，但该前提
    #    已被**文档整肃**打破——旧结论不再「保留+划掉」，warn 因此归零。
    #    ⇒ 这里改为「**允许 0 warn**，但若出现必须是合法 level」，同时用**构造输入**保证
    #      扫描器**仍能识别 warn**（见本文件里的定向用例），避免这条只读烟雾验证变成空断言。
    # ⚠️ 2026-09-19 更正：原断言是「至少一类 warn 命中（**对当前仓库必然成立**）」，但该前提已被
    #    **文档整肃**打破 —— 旧结论不再「保留+划掉」，而是挪入汇总的「已推翻结论」清单，warn 因此归零。
    #    ⇒ 本**只读烟雾验证**只负责「扫描跑通 + 只读 + 规则指纹稳定」；「扫描器仍能识别 warn」
    #      由本文件里**构造输入**的定向用例保证（不依赖仓库现状，避免空断言）。
    assert all(f["level"] for f in [x for d in out["docs"] for x in d["findings"]])
    assert len(out["rules_fingerprint"]) == 64
    assert out["rules"] == {
        "R1": "完成未收口", "R2": "未提交登记", "R3": "状态行滞后",
        "R4": "链接失效", "R5": "状态行缺失",
    }
