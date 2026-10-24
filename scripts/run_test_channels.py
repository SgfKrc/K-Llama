"""Run unit, external-resource and opted-in real-model test channels.

Physical markers require QLH_RUN_PHYSICAL_TESTS=1 (or the legacy
QLH_RUN_REAL_MODEL_SMOKE=1). Ordinary all-channel CI validates unit/external
and explicitly records physical/smoke NOT_RUN. Every expected run needs both
an integer zero return code and JUnit evidence containing actual passed tests.

--session NAME creates a new, non-reusable directory under --artifacts-dir.
--require-physical requires opt-in and channel=all, with attributed physical
passes in every external and smoke repetition; skips never count as passes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]

# Support both direct CLI execution and imports from the repository's tests.
try:
    from scripts.summarize_test_channels import (
        evaluate_manifest, evaluate_run, execution_status, expected_runs, session_path,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from summarize_test_channels import (
        evaluate_manifest, evaluate_run, execution_status, expected_runs, session_path,
    )


def _write_console(text: str) -> None:
    """Write subprocess output without letting a legacy Windows code page abort a run."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    encoded = text.encode(encoding, errors="replace")
    stream = getattr(sys.stdout, "buffer", None)
    if stream is not None:
        stream.write(encoded)
        stream.flush()
    else:
        sys.stdout.write(encoded.decode(encoding, errors="replace"))
        sys.stdout.flush()


#: 物理设备 / 重档用例的显式 opt-in 开关（票 13 第三条）。
#:
#: 未开启时这些用例**不被收集**，而不是"先尝试跑、依赖缺失再 skip" —— 后者在有 GPU 的
#: 机器上会真的跑起来，等于隐式 opt-in。`QLH_RUN_REAL_MODEL_SMOKE` 保留为向后兼容的
#: 等价开关（它此前只覆盖 `real_model`）。
PHYSICAL_OPT_IN_ENV = "QLH_RUN_PHYSICAL_TESTS"
LEGACY_SMOKE_ENV = "QLH_RUN_REAL_MODEL_SMOKE"
#: 需要显式 opt-in 才收集的标记集合。
PHYSICAL_MARKERS = ("real_model", "requires_gpu", "slow", "requires_db")


def _physical_opt_in() -> bool:
    """物理设备 / 重档用例是否已显式开启。"""
    return (
        os.environ.get(PHYSICAL_OPT_IN_ENV) == "1"
        or os.environ.get(LEGACY_SMOKE_ENV) == "1"
    )


def _external_marker_expression(*, physical: bool) -> str:
    """external 段的 `-m` 表达式。

    external 段承载两类用例：**必须串行**的（`external`，如起真实 socket / 子进程）与
    **物理设备/重档**的。后者的收集与否由显式 opt-in 决定；未开启时它们连收集都不进，
    因此既不会占用串行段的时间，也不会以 skip 的形式混进结果。
    """
    if physical:
        return 'not real_model and (external or requires_db or slow or requires_gpu)'
    # `PHYSICAL_MARKERS` 已包含 `real_model`，这里不再重复写一遍。
    excluded = ' and '.join(f'not {marker}' for marker in PHYSICAL_MARKERS)
    return f'{excluded} and (external)'


def pytest_collection_finish(session) -> None:
    """Persist marker attribution in JUnit without importing pytest in the CLI.

    Loaded with -p scripts.run_test_channels only for opted-in serial channels.
    Runs after all collection_modifyitems hooks, so classification is final.
    Ordinary external passes therefore cannot masquerade as physical execution.
    """
    if not _physical_opt_in():
        return
    for item in session.items:
        markers = sorted({marker.name for marker in item.iter_markers()}
                         & set(PHYSICAL_MARKERS))
        if markers:
            item.user_properties.append(("qlh_physical_markers", ",".join(markers)))


def _write_json_atomic(path: Path, data: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}-{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _pytest_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (src, current) if value
    )
    return env


def _run_pytest(
    arguments: Sequence[str],
    env: dict[str, str],
    *,
    log_path: Path,
) -> tuple[int, float]:
    command = [sys.executable, "-m", "pytest", *arguments]
    print("+", " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = None
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                _write_console(line)
                log.write(line)
            returncode = process.wait()
        except (Exception, KeyboardInterrupt) as exc:  # runner failures need an artifact
            log.write(f"\n[test-channels] runner error: {exc!r}\n")
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            if isinstance(exc, KeyboardInterrupt):
                raise
            returncode = 2
    return returncode, time.monotonic() - started


def _quality_args(*, junitxml: Path, order_seed: int) -> list[str]:
    return [
        "--junitxml",
        str(junitxml),
        "--maxfail=0",
        "--tb=long",
        "--qlh-order-seed",
        str(order_seed),
    ]


def _unit_args(
    workers: int,
    *,
    dist: str = "loadscope",
    junitxml: Path | None = None,
    order_seed: int | None = None,
) -> list[str]:
    arguments = [
        "tests",
        "-q",
        "-m",
        "not external and not requires_db and not real_model and not slow and not requires_gpu",
        # 覆盖 pytest.ini 的 --maxfail=3（0 = 不设上限，跑完为止）。两者一起用时
        # `--maxfail` 与 `--junitxml` 会互相拆台：pytest 提前停止后
        # <testsuite tests=N> 仍写**全部收集数**，而 <testcase> 只有**已执行的**，
        # 于是 summarize_test_channels 判定 "JUnit tests count disagrees with testcase
        # outcomes" 并**放弃解析**，总结里只剩 `passed=? failed=? errors=?`。
        # 结果是 CI 越红给出的信息越少：2026-10-24 排查 unit 通道时，只能靠本机
        # 跑 20 分钟全量（`run_test_channels.py --channel unit`）才拿到失败清单。
        # 代价是失败时多跑几分钟，换来每次红都能在日志里直接看到失败测试名。
        "--maxfail=0",
    ]
    if importlib.util.find_spec("xdist") is None:
        print(
            "[test-channels] pytest-xdist is missing; the unit channel will "
            "fall back to serial execution. Install requirements-test.txt "
            "in the project virtual environment to enable parallel tests.",
            flush=True,
        )
    else:
        arguments.extend(["-n", str(workers), "--dist", dist])
    if junitxml is not None and order_seed is not None:
        arguments.extend(_quality_args(junitxml=junitxml, order_seed=order_seed))
    return arguments


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--channel',
        choices=('all', 'unit', 'external', 'smoke'),
        default='all',
        help='channel to run; all uses unit -> external -> smoke order',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=int(os.environ.get('QLH_TEST_WORKERS', '4')),
        help='parallel unit-test workers (default: 4)',
    )
    parser.add_argument(
        '--allow-system-python',
        action='store_true',
        help=(
            'bypass the virtual-environment guard for disposable CI images or '
            'dependency diagnostics'
        ),
    )
    parser.add_argument(
        '--dist',
        choices=('loadscope', 'load'),
        default='loadscope',
        help=(
            'unit-channel xdist distribution: loadscope is the stable default; '
            'load enables true test-level concurrency for race audits'
        ),
    )
    parser.add_argument(
        '--repeat',
        type=int,
        default=1,
        help=(
            'run each selected channel this many complete times; failures are '
            'recorded and are never hidden by retries'
        ),
    )
    parser.add_argument(
        '--order-seed',
        type=int,
        default=(
            int(os.environ['QLH_TEST_ORDER_SEED'])
            if os.environ.get('QLH_TEST_ORDER_SEED')
            else None
        ),
        help='deterministic test-order seed; generated and recorded when omitted',
    )
    parser.add_argument(
        '--artifacts-dir',
        type=Path,
        default=Path(
            os.environ.get('QLH_TEST_ARTIFACTS_DIR', str(ROOT / 'build' / 'audit'))
        ),
        help='directory for per-run JUnit XML, logs, and manifest.json',
    )
    parser.add_argument(
        '--session',
        help='exact new session directory name under --artifacts-dir; never reused',
    )
    parser.add_argument(
        '--require-physical',
        action='store_true',
        help='require opted-in all-channel execution with actual physical passes',
    )
    return parser.parse_args(argv)


def _in_virtual_environment() -> bool:
    return sys.prefix != getattr(sys, 'base_prefix', sys.prefix)


def _check_python_environment(*, allow_system_python: bool) -> bool:
    if _in_virtual_environment() or allow_system_python:
        return True
    test_python = ROOT / '.venv-test' / (
        'Scripts/python.exe' if os.name == 'nt' else 'bin/python'
    )
    print(
        '[test-channels] refusing to run with system Python because test '
        'dependencies can change the shared runtime.',
        file=sys.stderr,
    )
    print(
        '[test-channels] prepare it with: python scripts/setup_test_env.py',
        file=sys.stderr,
    )
    print(
        f'[test-channels] then run: {test_python} '
        'scripts/run_test_channels.py',
        file=sys.stderr,
    )
    print(
        '[test-channels] use --allow-system-python only in a disposable '
        'environment.',
        file=sys.stderr,
    )
    return False


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not 1 <= args.workers <= 16:
        raise SystemExit('--workers must be between 1 and 16')
    if not 1 <= args.repeat <= 100:
        raise SystemExit('--repeat must be between 1 and 100')
    if not _check_python_environment(allow_system_python=args.allow_system_python):
        return 2

    order_seed = args.order_seed
    if order_seed is None:
        order_seed = random.SystemRandom().randrange(1, 2**32)
    artifact_base = args.artifacts_dir.expanduser().resolve()
    artifact_base.mkdir(parents=True, exist_ok=True)
    session_name = args.session if args.session is not None else (
        'test-channels-'
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f'{os.getpid()}'
    )
    try:
        artifact_dir = session_path(artifact_base, session_name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    suffix = 1
    while True:
        try:
            artifact_dir.mkdir()
            break
        except FileExistsError:
            if args.session is not None:
                raise SystemExit(f'session already exists: {artifact_dir}')
            suffix += 1
            artifact_dir = artifact_base / f'{session_name}-{suffix}'
    # Select this invocation before writing its manifest or launching anything.
    # A missing/partial manifest in this session must never reveal an older pass.
    _write_json_atomic(artifact_base / 'latest.json', {'session': artifact_dir.name})
    print(f'[test-channels] artifacts: {artifact_dir}', flush=True)
    print(f'[test-channels] order seed: {order_seed}', flush=True)

    env = _pytest_env()
    physical = _physical_opt_in()
    runs: list[dict[str, object]] = []
    evidence_runs: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        'schema_version': 2,
        'session': artifact_dir.name,
        'state': 'running',
        'started_at': datetime.now(timezone.utc).isoformat(),
        'channel': args.channel,
        'workers': args.workers,
        'dist': args.dist,
        'repeat': args.repeat,
        'order_seed': order_seed,
        'require_physical': args.require_physical,
        'expected_runs': expected_runs(args.channel, args.repeat, physical),
        'runs': runs,
    }

    def record_manifest() -> None:
        manifest['smoke'] = execution_status(
            evidence_runs, requested=args.channel in ('all', 'smoke'),
            opt_in=physical, smoke=True,
        )
        manifest['physical'] = {
            'opt_in_env': PHYSICAL_OPT_IN_ENV,
            'legacy_env': LEGACY_SMOKE_ENV,
            'covers_markers': list(PHYSICAL_MARKERS),
            'external_marker_expression': _external_marker_expression(physical=physical),
            **execution_status(evidence_runs,
                               requested=args.channel in ('all', 'external', 'smoke'),
                               opt_in=physical),
        }
        for label in ('smoke', 'physical'):
            status = manifest[label]
            if status['status'] == 'PASS' and manifest['state'] != 'completed':
                status.update(status='FAIL' if manifest['state'] == 'interrupted' else 'RUNNING',
                              reason='session has not completed')
        _write_json_atomic(artifact_dir / 'manifest.json', manifest)

    record_manifest()
    invalid_physical_request = args.require_physical and (not physical or args.channel != 'all')
    if not invalid_physical_request:
        # Keep the established order: unit/external pairs, then smoke repeats.
        run_order = [(name, repetition) for repetition in range(1, args.repeat + 1)
                     for name in ('unit', 'external') if args.channel in ('all', name)]
        if physical and args.channel in ('all', 'smoke'):
            run_order.extend(('smoke', repetition) for repetition in range(1, args.repeat + 1))
        for name, repetition in run_order:
            seed = (order_seed + repetition - 1) % (2**32)
            junitxml = artifact_dir / f'{name}-{repetition:03d}.xml'
            log_path = artifact_dir / f'{name}-{repetition:03d}.log'
            if name == 'unit':
                command = _unit_args(args.workers, dist=args.dist,
                                     junitxml=junitxml, order_seed=seed)
            else:
                expression = 'real_model' if name == 'smoke' else _external_marker_expression(physical=physical)
                command = ['tests', '-q', '-m', expression]
                if importlib.util.find_spec('xdist') is not None:
                    command.extend(['-n', '0'])
                if physical:
                    command.extend(['-p', 'scripts.run_test_channels'])
                command.extend(_quality_args(junitxml=junitxml, order_seed=seed))
            run: dict[str, object] = {
                'channel': name, 'run': repetition, 'order_seed': seed,
                'state': 'running', 'returncode': None,
                'junitxml': junitxml.name, 'log': log_path.name,
                'command': [sys.executable, '-m', 'pytest', *command],
            }
            runs.append(run)
            # Record the attempt before launch, including when launch interrupts.
            evidence_runs.append(dict(run))
            record_manifest()
            print(f'\n[test-channels] {name} channel (run {repetition}/{args.repeat})', flush=True)
            try:
                returncode, duration = _run_pytest(command, env, log_path=log_path)
            except (Exception, KeyboardInterrupt) as exc:
                manifest['state'] = 'interrupted'
                manifest['runner_error'] = repr(exc)
                record_manifest()
                print(f'[test-channels] interrupted: {exc!r}', flush=True)
                return 1
            run.update(state='completed', returncode=returncode, duration_s=round(duration, 3))
            evidence = evaluate_run(run, artifact_dir)
            run.update({key: evidence[key] for key in ('counts', 'physical_counts', 'status', 'errors')})
            evidence_runs[-1] = evidence
            record_manifest()

    if not physical and args.channel in ('all', 'smoke'):
        print(f'\n[test-channels] smoke NOT_RUN: set {PHYSICAL_OPT_IN_ENV}=1 or {LEGACY_SMOKE_ENV}=1.', flush=True)
    manifest['state'] = 'completed'
    manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
    record_manifest()
    code, result = evaluate_manifest(manifest, artifact_dir,
                                     require_physical=args.require_physical)
    for run in result['runs']:
        for error in run['errors']:
            print(f"[test-channels] {run['channel']}#{run['run']}: {error}", flush=True)
    for error in result['errors']:
        print(f'[test-channels] {error}', flush=True)
    print(f"\n[test-channels] {result['status']}", flush=True)
    print(f'[test-channels] evidence: {artifact_dir}', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
