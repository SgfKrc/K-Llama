#!/usr/bin/env bash
# Application, TUI and signed maintenance entry for source checkouts.
SELF="$0"
if command -v readlink >/dev/null 2>&1 && readlink -f "$0" >/dev/null 2>&1; then
    SELF="$(readlink -f "$0")"
fi
PROJECT_ROOT="$(cd "$(dirname "$SELF")" && pwd)" || exit 1
cd "$PROJECT_ROOT" || exit 1
export QLH_CORE_ROOT="${QLH_CORE_ROOT:-$PROJECT_ROOT}"
export QLH_RELEASE_ROOT="${QLH_RELEASE_ROOT:-$PROJECT_ROOT/packaging}"
export QLH_SHELL_ROOT="${QLH_SHELL_ROOT:-$PROJECT_ROOT/frontend_cybergothic}"
PY="${PYTHON:-python3}"
if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then PY="$PROJECT_ROOT/.venv/bin/python"; fi
export PYTHONIOENCODING=utf-8
case "${1:-tui}" in
    ui|web|headless)
        mode="$1"; shift
        exec "$PY" "$PROJECT_ROOT/src/app_bootstrap.py" "--$mode" "$@" ;;
    update)
        shift
        exec "$PY" "$PROJECT_ROOT/qlh.py" maintenance check "$@" ;;
    check|download|install|version-status|version-stage|version-activate|version-rollback|version-recover|diagnostics|verify|diagnose|repair|data-status|retain-data|reassociate-data|reinstall)
        exec "$PY" "$PROJECT_ROOT/qlh.py" maintenance "$@" ;;
    version) exec "$PY" -c 'import json; print(json.load(open("release-contract.json"))["version"]["product"])' ;;
    *) exec "$PY" "$PROJECT_ROOT/qlh.py" "$@" ;;
esac
