"""`scripts/check_doc_links.py` 的行为守卫。

重点守两类**约定**，它们直接决定 CI 是绿是红：

1. **仓库外的相对路径不计入链接门**。`docs/` 里有形如 `../../qlh-release/docs/…` 的引用 ——
   它们是"本地工作区里与主仓并列的独立仓库"这一事实的记录，换台机器或 CI 上必然不存在。
   若把它们当死链，CI 会永久红灯（2026-09-23 实际发生过：一次 push 后 CI 报 16 处，其中 8 处是这类）。
2. **子模块内的文档引用要靠"检出子模块"来满足**（`android/`），而不是靠放宽检查
   —— 所以这里也断言 `android/…` 这类路径会被**正常校验**（在仓库内、必须存在）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_doc_links", ROOT / "scripts" / "check_doc_links.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _probe(module, tmp_path: Path, monkeypatch, body: str) -> list[tuple[int, str, str]]:
    """把 ROOT 指到临时目录，写一个探针 md 并返回 check_file 的问题列表。"""
    monkeypatch.setattr(module, "ROOT", str(tmp_path))
    target = tmp_path / "docs" / "probe.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "存在.md").write_text("# ok\n", encoding="utf-8")
    (tmp_path / "outside.md").write_text("# outside\n", encoding="utf-8")
    target.write_text(body, encoding="utf-8")
    return module.check_file(str(target))


def test_missing_inside_repo_is_reported(tmp_path, monkeypatch) -> None:
    module = _load()
    problems = _probe(module, tmp_path, monkeypatch, "[缺](不存在.md)\n")
    assert [why for _, _, why in problems] == ["目标不存在"]


def test_existing_inside_repo_passes(tmp_path, monkeypatch) -> None:
    module = _load()
    assert _probe(module, tmp_path, monkeypatch, "[在](存在.md)\n") == []


def test_directory_target_is_reported(tmp_path, monkeypatch) -> None:
    module = _load()
    problems = _probe(module, tmp_path, monkeypatch, "[目录](../docs)\n")
    assert [why for _, _, why in problems] == ["指向目录而非文件"]


def test_outside_repo_reference_is_skipped(tmp_path, monkeypatch) -> None:
    """★ 约定 1：指到仓库外的相对路径必须**跳过**（否则 CI 永久红灯）。"""
    module = _load()
    body = "[外部](../outside.md)\n[更外](../../qlh-release/docs/x.md)\n"
    assert _probe(module, tmp_path, monkeypatch, body) == []


@pytest.mark.skipif(sys.platform != "win32", reason="跨盘符是 Windows 语义（os.path.relpath 会抛 ValueError）")
def test_outside_repo_skip_survives_other_drive(tmp_path, monkeypatch) -> None:
    """★ 跨盘符不能崩：Windows 上 `os.path.relpath` 对另一个盘会抛 ValueError。

    这个分支正是实测中暴露的 —— 检查工具自己崩掉，比漏报一条链接更糟。
    """
    module = _load()
    # 临时目录位于系统盘，与真实仓库（可能在不同盘符）天然构成跨盘场景
    assert _probe(module, tmp_path, monkeypatch, "[别盘](Z:/绝对/其他.md)\n") == []


def test_absolute_urls_and_anchors_still_skipped(tmp_path, monkeypatch) -> None:
    module = _load()
    body = "[网](https://example.com/x) [锚](#sec) [邮](mailto:a@b.c)\n"
    assert _probe(module, tmp_path, monkeypatch, body) == []


def test_iter_md_files_excludes_archive() -> None:
    """归档区不在链接质量门内（历史引用刻意保留），所以扫描必须跳过它。"""
    module = _load()
    scanned = {Path(p).resolve() for p in module.iter_md_files()}
    archive = (ROOT / "docs" / "archive").resolve()
    assert scanned, "至少要扫到一些文档"
    assert not any(archive in path.parents for path in scanned)


def test_android_submodule_documents_are_checked_in_repo() -> None:
    """★ 约定 2：`android/…` 属于**仓库内**目标，必须正常校验（靠检出子模块满足，不放宽）。"""
    module = _load()
    android_doc = ROOT / "android" / "Android验证替代路径-2026-09-18.md"
    if not android_doc.exists():
        # 未检出子模块（例如 CI 里没跑 git submodule update）——此时跳过，
        # 但**不改变**检查逻辑：CI workflow 已负责按需拉这个子模块。
        import pytest

        pytest.skip("android 子模块未检出")
    result = module.check_file(str(ROOT / "README.md"))
    assert isinstance(result, list)
    assert not any("Android验证替代路径" in raw for _, raw, _ in result)


def test_wrong_committed_reference_is_still_caught() -> None:
    """真实仓库里不许出现指向缺失文件的仓库内链接（本仓库当前应为零）。"""
    module = _load()
    problems = []
    for md in module.iter_md_files():
        problems += [(md, raw, why) for _, raw, why in module.check_file(md)]
    assert problems == [], problems


def test_gitignored_inside_repo_target_is_flagged() -> None:
    """★ 约定 3：**仓库内但未入库**的目标必须被标记。

    本机有工作区独立目录（`tools/`、`docs/agent_tool/` 等已被主仓裁撤的路径），所以
    「文件存在」在本机恒真 —— 放过这类引用就会**本地全绿、CI 红**（2026-09-23 连挂两次，
    根因都是这个）。主仓文档本就不该链接已从主仓移除的东西。
    """
    module = _load()
    assert module._is_gitignored("tools/docagent/docs/文档维护Agent工具设计.md") is True
    assert module._is_gitignored("docs/agent_tool/doc_maintenance_audit.py") is True
    assert module._is_gitignored("README.md") is False
    assert module._is_gitignored("docs/项目速览-QLH-at-a-Glance.md") is False
