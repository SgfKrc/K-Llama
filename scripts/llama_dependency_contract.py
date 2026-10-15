"""Check the QLH llama-cpp-python declaration and lock contract.

The ordinary CPU/GGUF profile is pinned to one wheel version. Gemma native
is intentionally a separate managed MTMD binding with its own frozen ABI
version; it must never be upgraded by the ordinary runtime lock.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERSION = "0.3.35"
GEMMA_NATIVE_VERSION = "0.3.28"
PACKAGE_RE = re.compile(
    r"^\s*llama[-_]cpp[-_]python\s*(?P<spec>==|>=|<=|~=|!=|>|<)\s*"
    r"(?P<version>[0-9]+(?:\.[0-9]+){1,3})\b",
    re.IGNORECASE,
)
LOCK_RE = re.compile(
    r"^\s*llama[-_]cpp[-_]python==(?P<version>[0-9]+(?:\.[0-9]+){1,3})\s*$",
    re.IGNORECASE,
)

DECLARATIONS = {
    "requirements.txt": DEFAULT_VERSION,
    "requirements-edge.txt": DEFAULT_VERSION,
    "requirements-test.txt": DEFAULT_VERSION,
    "requirements/requirements-gemma4-native.txt": GEMMA_NATIVE_VERSION,
    "packaging/packaging/requirements-cpu.txt": DEFAULT_VERSION,
    "packaging/packaging/requirements-runtime-cpu.txt": DEFAULT_VERSION,
    "packaging/packaging/requirements-runtime-cuda.txt": DEFAULT_VERSION,
}
LOCKS = {
    "requirements-lock/main.lock.txt": DEFAULT_VERSION,
    "requirements-lock/edge.lock.txt": DEFAULT_VERSION,
    "requirements-lock/test.lock.txt": DEFAULT_VERSION,
    "requirements-lock/gemma4-native.lock.txt": GEMMA_NATIVE_VERSION,
    "packaging/requirements-lock/packaging.lock.txt": DEFAULT_VERSION,
    "packaging/requirements-lock/packaging-cuda.lock.txt": DEFAULT_VERSION,
}
EDGE_FORBIDDEN = ("torch", "transformers", "accelerate", "bitsandbytes")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _declaration_version(path: Path) -> tuple[str | None, str | None]:
    matches: list[tuple[str, str]] = []
    for line in _read(path).splitlines():
        line = line.split("#", 1)[0].strip()
        match = PACKAGE_RE.match(line)
        if match:
            matches.append((match.group("spec"), match.group("version")))
    if len(matches) != 1:
        return None, f"expected one declaration, found {len(matches)}"
    spec, version = matches[0]
    return (version, None) if spec == "==" else (None, f"declaration is {spec}{version}, not ==")


def _lock_version(path: Path) -> tuple[str | None, str | None]:
    matches = [
        match.group("version")
        for line in _read(path).splitlines()
        if (match := LOCK_RE.match(line.split("#", 1)[0]))
    ]
    if len(matches) != 1:
        return None, f"expected one locked package, found {len(matches)}"
    return matches[0], None


def _check_file_set(
    entries: dict[str, str],
    parser: Callable[[Path], tuple[str | None, str | None]],
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for relative, expected in entries.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append({"file": relative, "reason": "missing"})
            continue
        actual, reason = parser(path)
        if actual != expected:
            failures.append({
                "file": relative,
                "expected": expected,
                "actual": actual or "",
                "reason": reason or "version mismatch",
            })
    return failures


def check_contract() -> dict[str, Any]:
    failures = _check_file_set(DECLARATIONS, _declaration_version)
    failures.extend(_check_file_set(LOCKS, _lock_version))

    edge_text = _read(ROOT / "requirements-edge.txt")
    for package in EDGE_FORBIDDEN:
        if re.search(rf"^\s*{re.escape(package)}(?:[-_]|\s|[<>=])", edge_text, re.IGNORECASE | re.MULTILINE):
            failures.append({
                "file": "requirements-edge.txt",
                "reason": f"forbidden dependency: {package}",
            })

    marker_path = ROOT / "scripts/model_tools/gemma4_native_binding.lock.json"
    try:
        marker = json.loads(_read(marker_path))
        marker_version = marker["package"]["version"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
        marker_version = None
    if marker_version != GEMMA_NATIVE_VERSION:
        failures.append({
            "file": "scripts/model_tools/gemma4_native_binding.lock.json",
            "expected": GEMMA_NATIVE_VERSION,
            "actual": str(marker_version or ""),
            "reason": "managed Gemma binding marker version mismatch",
        })

    return {
        "ok": not failures,
        "default_cpu_gguf_version": DEFAULT_VERSION,
        "gemma_native_version": GEMMA_NATIVE_VERSION,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    result = check_contract()
    if args.as_json:
        print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    else:
        print(
            f"llama dependency contract: {'OK' if result['ok'] else 'FAILED'} "
            f"(default={DEFAULT_VERSION}, gemma-native={GEMMA_NATIVE_VERSION})"
        )
        for failure in result["failures"]:
            print(f"- {failure['file']}: {failure['reason']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
