# -*- coding: utf-8 -*-
"""P1-6 文档收口：检查仓库内 Markdown 相对链接是否指向存在文件。

覆盖范围：docs/、README.md、tests/simulation/README.md 等主仓文档。
- 提取 [text](url) 与 [text](url "title") 形式链接
- 跳过 http(s)://、mailto:、# 锚点、<...> 自动链接
- 解码 URL 编码，去掉 #锚点 后按相对路径解析
- 仓库外路径与「未检出的子模块内」路径不判定（显式打印跳过项）；子模块已检出时照常校验
- 报告：目标不存在 / 指向目录 / 目标未入库
"""
import os
import re
import subprocess
import sys
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LINK_RE = re.compile(r'\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)')

SKIP_PREFIX = ('http://', 'https://', 'mailto:', 'ftp://', '#')

_IGNORED_CACHE: dict[str, bool] = {}


def _is_gitignored(rel_path: str) -> bool:
    """目标路径是否被 .gitignore 忽略（= 不在版本控制里 ⇒ CI 上必然不存在）。

    为什么必须单独查这一点：本机会有**工作区独立目录**（`tools/`、`docs/agent_tool/` 等已被
    主仓裁撤的路径），所以「文件存在」在本机恒真 —— 本地检查全绿、CI 却红。
    主仓文档本就不应链接已从主仓移除的东西，因此这里**报错**而不是放过。
    """
    if rel_path in _IGNORED_CACHE:
        return _IGNORED_CACHE[rel_path]
    try:
        done = subprocess.run(['git', 'check-ignore', '-q', rel_path], cwd=ROOT,
                              capture_output=True, check=False)
        result = done.returncode == 0
    except (OSError, ValueError):
        result = False          # 环境里没有 git ⇒ 不应用这条规则，其余检查照旧
    _IGNORED_CACHE[rel_path] = result
    return result


_SUBMODULE_CACHE: tuple[str, list[tuple[str, bool]]] | None = None
SKIPPED_SUBMODULES: set[str] = set()


def _read_submodule_paths(repo_dir: str) -> list[str]:
    """读取 `repo_dir/.gitmodules` 里声明的子模块路径（相对该仓库）。

    `git config -f` 比手解析 INI 更稳：忽略注释、引号与重复键的细节差异。
    """
    gitmodules = os.path.join(repo_dir, '.gitmodules')
    if not os.path.exists(gitmodules):
        return []
    try:
        done = subprocess.run(
            ['git', 'config', '-f', gitmodules, '--get-regexp', r'^submodule\..*\.path$'],
            capture_output=True, check=False, text=True, encoding='utf-8', errors='replace')
    except (OSError, ValueError):
        return []
    if done.returncode != 0:
        return []
    paths = []
    for line in done.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[1].strip():
            paths.append(parts[1].strip().replace('\\', '/'))
    return paths


def _is_checked_out(abs_dir: str) -> bool:
    """子模块目录是否真检出了工作树。

    `actions/checkout` 在 `submodules: false` 下仍会创建**空目录**，所以 `isdir` 不够；
    空目录 + `git rev-parse HEAD` 失败 ⇒ 视为未检出。
    """
    if not os.path.isdir(abs_dir):
        return False
    try:
        if not os.listdir(abs_dir):
            return False
        done = subprocess.run(['git', '-C', abs_dir, 'rev-parse', '--verify', 'HEAD'],
                              capture_output=True, check=False)
    except (OSError, ValueError):
        return False
    return done.returncode == 0


def _submodule_declarations() -> list[tuple[str, bool]]:
    """列出所有声明的子模块 (仓库内相对路径, 是否已检出)，递归进入已检出的子模块。

    必须递归：`llama.cpp` 是 `android` 的子模块，只有 `android` 检出后才枚举得到它。
    深度上限防止异常嵌套把检查拖死。
    """
    global _SUBMODULE_CACHE
    # 缓存按 ROOT 绑定：测试会把 ROOT 指到临时目录，只按模块级缓存会串味。
    if _SUBMODULE_CACHE is not None and _SUBMODULE_CACHE[0] == ROOT:
        return _SUBMODULE_CACHE[1]
    found: list[tuple[str, bool]] = []

    def walk(abs_dir: str, prefix: str, depth: int) -> None:
        if depth > 3:
            return
        for rel_path in _read_submodule_paths(abs_dir):
            rel = f'{prefix}{rel_path}'
            abs_sub = os.path.join(ROOT, rel.replace('/', os.sep))
            checked = _is_checked_out(abs_sub)
            found.append((rel, checked))
            if checked:
                walk(abs_sub, rel + '/', depth + 1)

    walk(ROOT, '', 0)
    _SUBMODULE_CACHE = (ROOT, found)
    return found


def _unchecked_submodule_owner(rel_path: str) -> str | None:
    """目标落在某个**未检出**的子模块内时返回该子模块路径，否则 None。

    为什么需要这一条：子模块是主仓的正式组成部分（`.gitmodules` 声明、指针入库），文档
    链接到其中的实现文件是合理的。但**子模块是否检出取决于 CI 配置** —— `doc-checks`
    只浅拉 `android`、`core-tests` 多拉 `packaging`、两者都不拉 `harness_workbench`。
    若把「未检出」判成死链，同一份文档在本地（子模块齐全）全绿、在 CI 上必红，这个门就
    退化成「CI 配置的镜像」而不是文档质量的判据。

    因此：目标所属子模块未检出 ⇒ 不计入本门；已检出 ⇒ 照常逐条校验存在性（本地仍能查出
    子模块内写错的路径或改名）。取**最长匹配**，这样「父模块已检出、嵌套子模块未检出」的
    情况（android + llama.cpp）也判为不校验。
    """
    best_rel: str | None = None
    best_checked = True
    for rel, checked in _submodule_declarations():
        if rel_path == rel or rel_path.startswith(rel + '/'):
            if best_rel is None or len(rel) > len(best_rel):
                best_rel, best_checked = rel, checked
    if best_rel is None or best_checked:
        return None
    return best_rel


def iter_md_files():
    for dirpath, dirs, files in os.walk(os.path.join(ROOT, 'docs')):
        # Archived documents keep historical references and are outside the
        # active link quality gate after their one-time migration repair.
        dirs[:] = [name for name in dirs if name != 'archive']
        for f in files:
            if f.endswith('.md'):
                yield os.path.join(dirpath, f)
    for f in ['README.md', 'tests/simulation/README.md']:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            yield p


def check_file(md_path):
    """返回 (ok, problems)"""
    problems = []
    base = os.path.dirname(md_path)
    with open(md_path, encoding='utf-8') as fh:
        for lineno, line in enumerate(fh, 1):
            for m in LINK_RE.finditer(line):
                raw = m.group(1).strip()
                if raw.startswith(SKIP_PREFIX):
                    continue
                # 解码 + 去锚点
                target = urllib.parse.unquote(raw)
                target = target.split('#')[0]
                if not target:
                    continue
                # 相对路径解析（相对当前文件所在目录）
                abs_target = os.path.normpath(os.path.join(base, target))
                # 仓库外的相对路径（如 `../../qlh-release/docs/…`）是**本地工作区路径说明**，
                # 不是可校验的仓库内链接：它们指向与主仓并列的独立仓库/工作区目录，
                # 换一台机器或 CI 上必然不存在。这类引用不计入链接质量门。
                try:
                    inside_repo = not os.path.relpath(abs_target, ROOT).startswith('..')
                except ValueError:
                    # Windows 上跨盘符时 relpath 会抛 ValueError ⇒ 目标必然在仓库外。
                    # 不接住它，检查工具自己就会崩（比漏报一条链接更糟）。
                    inside_repo = False
                if not inside_repo:
                    continue
                rel_to_root = os.path.relpath(abs_target, ROOT).replace(os.sep, '/')
                # 仓库内但**未入库**的目标（工作区独立目录、已裁撤的工具目录）：
                # 本机存在、CI 上不存在 ⇒ 本地会全绿而 CI 红。主仓文档不应链接这类路径。
                if _is_gitignored(rel_to_root):
                    problems.append((lineno, raw, '目标未入库（CI 上不存在）'))
                    continue
                # 目标属于**尚未检出**的子模块 ⇒ 本机与 CI 可见性不同，本门不作判定。
                owner = _unchecked_submodule_owner(rel_to_root)
                if owner is not None:
                    SKIPPED_SUBMODULES.add(owner)
                    continue
                if not os.path.exists(abs_target):
                    problems.append((lineno, raw, '目标不存在'))
                elif os.path.isdir(abs_target):
                    problems.append((lineno, raw, '指向目录而非文件'))
    return problems


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    total_links = 0
    all_problems = []
    for md in iter_md_files():
        rel = os.path.relpath(md, ROOT)
        problems = check_file(md)
        n = 0
        with open(md, encoding='utf-8') as fh:
            for line in fh:
                n += len(LINK_RE.findall(line))
        total_links += n
        for lineno, raw, why in problems:
            all_problems.append((rel, lineno, raw, why))
    print(f'扫描文件数: {len(list(iter_md_files()))}  链接总数: {total_links}')
    # 透明化跳过项：本机子模块齐全时这里通常为空；CI 上会列出未检出的子模块，
    # 避免「门为绿」被误读成「这些链接也校验过了」。
    if SKIPPED_SUBMODULES:
        print(f'ℹ 未校验指向未检出子模块的链接（子模块: {", ".join(sorted(SKIPPED_SUBMODULES))}）')
    if not all_problems:
        print('✅ 全部相对链接目标存在')
        return 0
    print(f'❌ 发现 {len(all_problems)} 个问题:')
    for rel, lineno, raw, why in all_problems:
        print(f'  {rel}:{lineno}  [{raw}]  ->  {why}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
