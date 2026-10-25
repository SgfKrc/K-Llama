"""Validate an installed QLH Edge L runtime without importing the full stack."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

FORBIDDEN_MODULES = (
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "pandas",
    "einops",
    "tiktoken",
)
REQUIRED_MODULES = {
    "llama_cpp": "llama-cpp-python",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "psutil": "psutil",
    "httpx": "httpx",
}
REQUIRED_ROUTES = ("/health", "/status", "/generate", "/capabilities", "/rpc/status")
PROBE_MARKER = "QLH_EDGE_PREFLIGHT="

#: `REQUIRED_ROUTES` 描述的是**这个入口**的契约（裸 `/health` 等）。
ROUTES_CONTRACT_ENTRY = "qlh_edge"
#: ★ P0-3：「干净 venv 运行扫描」的第二个入口 —— **SLIM 的真实载荷**。
#: `packaging/packaging/slim_entry.py` 分派正式应用入口，
#: 而它的路由带 `/api/` 前缀 ⇒ **不能**拿 `REQUIRED_ROUTES` 去判它；只判
#: 「**导入成功**且没把 forbidden 模块拽进 `sys.modules`」。
SLIM_ENTRY = "src.api_server"
#: 要做运行扫描的入口。两者各跑**独立子进程** —— 同一进程里先 import 谁会把
#: `sys.modules` 污染给后一个，那样第二个入口的判据就不可信了。
PROBE_ENTRIES = (ROUTES_CONTRACT_ENTRY, SLIM_ENTRY)


def _venv_root(python_executable: Path) -> Path:
    # Prefer the on-disk layout: a venv keeps its interpreter under ``bin/`` (POSIX)
    # or ``Scripts/`` (Windows) next to ``pyvenv.cfg``. Deciding from the path as
    # given (before ``resolve``) keeps symlink-shimmed venvs -- e.g. ``uv venv``,
    # whose ``bin/python`` points at a managed base interpreter -- reporting the
    # venv directory rather than the base interpreter that has no site packages.
    parent = python_executable.parent
    if parent.name.lower() in {"scripts", "bin"}:
        candidate = parent.parent
        if (candidate / "pyvenv.cfg").is_file():
            return candidate
    executable = python_executable.resolve()
    if executable.parent.name.lower() in {"scripts", "bin"}:
        return executable.parent.parent
    return executable.parent


def _directory_size_mb(root: Path) -> float:
    if not root.exists():
        return 0.0
    total = sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
    return round(total / 2**20, 2)


def _probe_code(entry: str) -> str:
    forbidden = repr(list(FORBIDDEN_MODULES))
    required = repr(list(REQUIRED_MODULES))
    return f"""
import importlib
import importlib.util
import json
import sys
import time

started = time.perf_counter()
payload = {{}}
try:
    module = importlib.import_module({entry!r})

    payload["entry"] = {entry!r}
    payload["import_elapsed_s"] = round(time.perf_counter() - started, 4)
    app = getattr(module, "app", None)
    payload["has_app"] = app is not None
    # 不是每条 route 都有 `.path`（FastAPI 的惰性 ``_IncludedRouter`` 就没有）⇒ 只取真有的。
    payload["routes"] = sorted(
        path
        for path in (getattr(route, "path", None) for route in getattr(app, "routes", []))
        if path
    )
    payload["forbidden_imported"] = [name for name in {forbidden} if name in sys.modules]
    payload["forbidden_installed"] = []
    for name in {forbidden}:
        try:
            if importlib.util.find_spec(name) is not None:
                payload["forbidden_installed"].append(name)
        except (ImportError, ModuleNotFoundError, ValueError):
            pass
    payload["missing_required"] = []
    for name in {required}:
        try:
            if importlib.util.find_spec(name) is None:
                payload["missing_required"].append(name)
        except (ImportError, ModuleNotFoundError, ValueError):
            payload["missing_required"].append(name)
    payload["ok"] = True
except Exception as exc:
    payload = {{"ok": False, "error": f"{{type(exc).__name__}}: {{exc}}"}}

print({PROBE_MARKER!r} + json.dumps(payload, sort_keys=True))
"""


def _run_probe(
    python_executable: Path,
    repository_root: Path,
    entry: str = ROUTES_CONTRACT_ENTRY,
) -> dict[str, Any]:
    environment = os.environ.copy()
    python_path = [str(repository_root), str(repository_root / "src")]
    existing_python_path = environment.get("PYTHONPATH")
    if existing_python_path:
        python_path.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_path)

    try:
        completed = subprocess.run(
            [str(python_executable), "-c", _probe_code(entry)],
            cwd=repository_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return {"ok": False, "error": f"cannot start edge Python: {exc}", "probe_returncode": -1}
    payload: dict[str, Any] | None = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(PROBE_MARKER):
            payload = json.loads(line[len(PROBE_MARKER) :])
            break
    if payload is None:
        payload = {
            "ok": False,
            "error": "edge import probe produced no result",
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-2000:],
        }
    payload["probe_returncode"] = completed.returncode
    return payload


def run_preflight(
    python_executable: str | os.PathLike[str],
    repository_root: str | os.PathLike[str] | None = None,
    *,
    max_size_mb: float = 300.0,
    max_startup_s: float = 15.0,
) -> dict[str, Any]:
    root = Path(repository_root or Path(__file__).resolve().parents[1]).resolve()
    # Do not ``resolve`` the interpreter: that would follow a venv's symlink shim
    # (``uv venv`` and POSIX venvs in general) to the base interpreter, which lacks
    # the venv's site packages. Passing the path as given keeps the venv active.
    python_path = Path(python_executable)
    venv_root = _venv_root(python_path)
    size_mb = _directory_size_mb(venv_root)
    probe = _run_probe(python_path, root, ROUTES_CONTRACT_ENTRY)
    routes = set(probe.get("routes", []))
    import_elapsed_s = probe.get("import_elapsed_s")

    # ★ P0-3 运行半边的第二个入口（SLIM 真实载荷）。**独立子进程** —— 同进程里先 import
    #   谁都会把 `sys.modules` 污染给后一个。仅当该入口在本仓存在时才判：Edge 安装包
    #   可以不含 `src/api_server.py`，那时这条不适用（SLIM 包的验收由 spec 决定），
    #   **不得**因此判成失败。
    slim_present = (root / "src" / "api_server.py").is_file()
    slim_probe: dict[str, Any] = {}
    if slim_present:
        slim_probe = _run_probe(python_path, root, SLIM_ENTRY)

    checks = {
        "python_exists": python_path.is_file(),
        "venv_size": size_mb <= max_size_mb,
        "cold_start": isinstance(import_elapsed_s, (int, float)) and import_elapsed_s <= max_startup_s,
        "required_modules": not probe.get("missing_required"),
        "no_forbidden_import": not probe.get("forbidden_imported"),
        "no_forbidden_install": not probe.get("forbidden_installed"),
        "required_routes": set(REQUIRED_ROUTES).issubset(routes),
        "probe": probe.get("ok") is True and probe.get("probe_returncode") == 0,
        # ★ P0-3：SLIM 真实载荷同样要能**导入成功**、**不把 forbidden 拽进 sys.modules**、
        #   且必需依赖齐全。注意**不查路由** —— `src.api_server` 的路径带 `/api/` 前缀，
        #   `REQUIRED_ROUTES` 是 `qlh_edge` 的契约，混用会误判。
        "slim_entry_import": (not slim_present)
        or (slim_probe.get("ok") is True and slim_probe.get("probe_returncode") == 0),
        "slim_entry_no_forbidden": (not slim_present) or not slim_probe.get("forbidden_imported"),
        "slim_entry_required_modules": (not slim_present) or not slim_probe.get("missing_required"),
    }
    return {
        "ok": all(checks.values()),
        "python": str(python_path),
        "venv_root": str(venv_root),
        "venv_size_mb": size_mb,
        "max_size_mb": max_size_mb,
        "max_startup_s": max_startup_s,
        "checks": checks,
        "probe": probe,
        "slim_entry": SLIM_ENTRY,
        "slim_entry_present": slim_present,
        "slim_probe": slim_probe,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the QLH Edge L runtime contract.")
    parser.add_argument("--python", default=sys.executable, help="Edge venv Python executable")
    parser.add_argument("--max-size-mb", type=float, default=300.0)
    parser.add_argument("--max-startup-s", type=float, default=15.0)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    result = run_preflight(
        args.python,
        max_size_mb=args.max_size_mb,
        max_startup_s=args.max_startup_s,
    )
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"edge python: {result['python']}")
        print(f"venv size: {result['venv_size_mb']} MB / {result['max_size_mb']} MB")
        print(f"cold start: {result['probe'].get('import_elapsed_s', 'n/a')} s / {result['max_startup_s']} s")
        if result.get("slim_entry_present"):
            slim = result.get("slim_probe", {})
            print(
                f"SLIM entry ({result['slim_entry']}): "
                f"{'import OK' if slim.get('ok') else 'IMPORT FAILED'} "
                f"in {slim.get('import_elapsed_s', 'n/a')} s"
            )
            if slim.get("error"):
                print(f"  {slim['error']}", file=sys.stderr)
        for name, passed in result["checks"].items():
            print(f"{'PASS' if passed else 'FAIL'} {name}")
        if result["probe"].get("error"):
            print(result["probe"]["error"], file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
