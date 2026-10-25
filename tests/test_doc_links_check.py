"""`scripts/check_doc_links.py` 的行为守卫。

重点守两类**约定**，它们直接决定 CI 是绿是红：

1. **仓库外的相对路径不计入链接门**。`docs/` 里有形如 `../../qlh-release/docs/…` 的引用 ——
   它们是"本地工作区里与主仓并列的独立仓库"这一事实的记录，换台机器或 CI 上必然不存在。
   若把它们当死链，CI 会永久红灯（2026-09-23 实际发生过：一次 push 后 CI 报 16 处，其中 8 处是这类）。
2. **子模块内的目标按"是否检出"分流，而不是一律跳过或一律报错**。
   子模块是主仓的正式组成部分（`.gitmodules` 声明、指针入库），文档链接其中的实现文件是
   合理的；但**是否检出取决于 CI 配置**（`doc-checks` 只拉 `android`、`core-tests` 多拉
   `packaging`、谁都不拉 `harness_workbench`）。若把"未检出"判成死链，同一份文档在本地全绿、
   在 CI 必红，门禁变成 CI 配置的镜像（2026-10-18 起连挂 8 次，根因即此）。
   因此：所属子模块**未检出 ⇒ 不判定并显式列出跳过项**；**已检出 ⇒ 照常校验**，
   子模块内写错路径或改名仍会被抓。
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
    """★ 约定 2 的「已检出」分支：`android/…` 检出后必须**正常校验**，不放宽。"""
    module = _load()
    android_doc = ROOT / "android" / "Android验证替代路径-2026-09-18.md"
    if not android_doc.exists():
        # 未检出子模块 ⇒ 按约定 2 跳过该引用（check_file 不判定），这里直接跳过用例。
        pytest.skip("android 子模块未检出")
    result = module.check_file(str(ROOT / "README.md"))
    assert isinstance(result, list)
    assert not any("Android验证替代路径" in raw for _, raw, _ in result)


def _stub_submodules(module, root: Path, declarations: list[tuple[str, bool]]) -> None:
    """伪造子模块声明表，避免测试依赖真实子模块检出状态与 git 仓库布局。"""
    module._SUBMODULE_CACHE = (str(root), declarations)
    module.SKIPPED_SUBMODULES.clear()


def test_unchecked_submodule_target_is_skipped(tmp_path, monkeypatch) -> None:
    """★ 约定 2 的「未检出」分支：所属子模块未检出 ⇒ 不判定，并记录跳过项。"""
    module = _load()
    _stub_submodules(module, tmp_path, [("packaging", False)])
    problems = _probe(module, tmp_path, monkeypatch,
                      "[打包](../packaging/packaging/launcher.py)\n")
    assert problems == []
    assert module.SKIPPED_SUBMODULES == {"packaging"}


def test_checked_submodule_target_is_still_validated(tmp_path, monkeypatch) -> None:
    """★ 约定 2 的关键性质：子模块**已检出**时真死链仍必须被抓 —— 否则就是放宽检查。"""
    module = _load()
    _stub_submodules(module, tmp_path, [("packaging", True)])
    problems = _probe(module, tmp_path, monkeypatch,
                      "[缺](../packaging/packaging/no_such_module.py)\n")
    assert [why for _, _, why in problems] == ["目标不存在"]
    assert module.SKIPPED_SUBMODULES == set()


def test_nested_submodule_under_checked_parent_is_skipped(tmp_path, monkeypatch) -> None:
    """父模块已检出、嵌套子模块未检出（android + llama.cpp）也要取**最长匹配**判定。"""
    module = _load()
    _stub_submodules(module, tmp_path, [
        ("android", True),
        ("android/app/src/main/cpp/llama.cpp", False),
    ])
    nested = tmp_path / "android" / "app" / "src" / "main" / "cpp" / "llama.cpp"
    nested.mkdir(parents=True)
    problems = _probe(
        module, tmp_path, monkeypatch,
        "[repack](../android/app/src/main/cpp/llama.cpp/ggml/src/ggml-cpu/repack.cpp)\n")
    assert problems == []
    assert module.SKIPPED_SUBMODULES == {"android/app/src/main/cpp/llama.cpp"}


def test_declared_submodule_paths_are_read_from_gitmodules(tmp_path) -> None:
    """`.gitmodules` 解析是分层判定的输入，必须按声明顺序取到 path（不是 url/name）。"""
    module = _load()
    (tmp_path / ".gitmodules").write_text(
        '[submodule "packaging"]\n\tpath = packaging\n\turl = https://example.invalid/a.git\n'
        '[submodule "android"]\n\tpath = android\n\turl = https://example.invalid/b.git\n',
        encoding="utf-8")
    assert module._read_submodule_paths(str(tmp_path)) == ["packaging", "android"]


def test_absent_gitmodules_declares_no_submodules(tmp_path) -> None:
    """没有 `.gitmodules` 的仓库（或子目录）不能被误判出子模块。"""
    module = _load()
    assert module._read_submodule_paths(str(tmp_path)) == []


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
