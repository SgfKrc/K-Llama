"""Tests for the dedicated test-environment guard and bootstrapper."""

from pathlib import Path
import sys
from types import SimpleNamespace

from scripts import run_test_channels
from scripts import setup_envs
from scripts import setup_qwen3_sidecar_env
from scripts import setup_test_env


def test_test_channel_guard_rejects_system_python(monkeypatch, capsys):
    monkeypatch.setattr(sys, "prefix", "C:/Python312")
    monkeypatch.setattr(sys, "base_prefix", "C:/Python312")

    assert run_test_channels._check_python_environment(
        allow_system_python=False,
    ) is False
    error = capsys.readouterr().err
    assert "setup_test_env.py" in error
    assert "--reuse-runtime" not in error


def test_test_channel_guard_accepts_virtual_environment(monkeypatch):
    monkeypatch.setattr(sys, "prefix", "G:/qlh/.venv-test")
    monkeypatch.setattr(sys, "base_prefix", "C:/Python312")

    assert run_test_channels._check_python_environment(
        allow_system_python=False,
    ) is True


def test_test_channel_console_writer_replaces_unrepresentable_characters(capsys):
    run_test_channels._write_console("test\u036c\n")

    assert "test" in capsys.readouterr().out


def test_test_channel_cli_defaults_to_stable_scope_distribution():
    args = run_test_channels._parse_args(["--channel", "unit"])

    assert args.dist == "loadscope"
    assert args.repeat == 1
    assert args.order_seed is None
    assert args.artifacts_dir == run_test_channels.ROOT / "build" / "audit"


def test_test_channel_unit_args_can_enable_true_concurrency(tmp_path):
    args = run_test_channels._unit_args(
        4,
        dist="load",
        junitxml=tmp_path / "unit.xml",
        order_seed=123,
    )

    assert args[4:8] == ["-n", "4", "--dist", "load"]
    assert args[-6:] == [
        "--junitxml",
        str(tmp_path / "unit.xml"),
        "--maxfail=0",
        "--tb=long",
        "--qlh-order-seed",
        "123",
    ]


def test_test_channel_repeat_keeps_all_runs_and_failure_evidence(monkeypatch, tmp_path):
    calls = []

    def fake_run(arguments, _env, *, log_path):
        log_path.write_text("captured\n", encoding="utf-8")
        calls.append((list(arguments), log_path.name))
        return (1 if len(calls) == 1 else 0), 0.01

    monkeypatch.setattr(run_test_channels, "_check_python_environment", lambda **_: True)
    monkeypatch.setattr(run_test_channels, "_run_pytest", fake_run)

    assert run_test_channels.main([
        "--channel", "unit",
        "--repeat", "2",
        "--order-seed", "7",
        "--artifacts-dir", str(tmp_path),
        "--allow-system-python",
    ]) == 1

    session_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(session_dirs) == 1
    manifest = (session_dirs[0] / "manifest.json").read_text(encoding="utf-8")
    data = __import__("json").loads(manifest)
    assert data["order_seed"] == 7
    assert [run["returncode"] for run in data["runs"]] == [1, 0]
    assert [run["order_seed"] for run in data["runs"]] == [7, 8]
    assert [run["order_seed"] for run in data["runs"]] == [7, 8]
    assert [run["log"] for run in data["runs"]] == ["unit-001.log", "unit-002.log"]
    assert all("--junitxml" in run["command"] for run in data["runs"])


def test_install_command_keeps_proxy_and_requirements_in_venv(monkeypatch):
    test_python = Path("G:/qlh/.venv-test/Scripts/python.exe")
    monkeypatch.setattr(setup_test_env, "_python_path", lambda: test_python)

    command = setup_test_env._install_command(proxy="http://127.0.0.1:7897")

    assert command[:4] == [
        str(test_python), "-m", "pip", "install",
    ]
    assert command[4:6] == ["--proxy", "http://127.0.0.1:7897"]
    assert command[-2:] == ["-r", str(setup_test_env.REQUIREMENTS)]


def test_wheelhouse_disables_network_proxy(monkeypatch, tmp_path):
    test_python = Path("G:/qlh/.venv-test/Scripts/python.exe")
    monkeypatch.setattr(setup_test_env, "_python_path", lambda: test_python)

    command = setup_test_env._install_command(
        proxy="http://127.0.0.1:7897",
        wheelhouse=tmp_path,
    )

    assert "--no-index" in command
    assert f"--find-links={tmp_path}" in command
    assert "--proxy" not in command


def test_check_rejects_overlay_when_isolation_is_requested(monkeypatch, capsys):
    monkeypatch.setattr(setup_test_env, "_uses_system_site_packages", lambda: True)
    monkeypatch.setattr(
        setup_test_env,
        "_ready",
        lambda: (_ for _ in ()).throw(AssertionError("health check must not run")),
    )

    assert setup_test_env.main(["--check"]) == 2
    assert "existing environment is overlay" in capsys.readouterr().err


def test_check_accepts_explicit_overlay_mode(monkeypatch):
    monkeypatch.setattr(setup_test_env, "_uses_system_site_packages", lambda: True)
    monkeypatch.setattr(setup_test_env, "_ready", lambda: True)

    assert setup_test_env.main(["--check", "--reuse-runtime"]) == 0


def test_unified_setup_keeps_test_environment_isolated():
    test_env = setup_envs.ENV_BY_NAME["test"]

    assert test_env.system_site_packages is False
    assert {"pytest", "xdist", "pytest_timeout"} <= set(test_env.required_modules)


def test_unified_setup_defaults_to_mainline_python_only():
    args = SimpleNamespace(
        only=None,
        all=True,
        check=False,
        snapshot=False,
        no_node=False,
        with_node=False,
        skip=None,
    )

    _, node_projects = setup_envs._select(args)

    assert node_projects == []


def test_unified_setup_keeps_product_shell_out_of_mainline():
    args = SimpleNamespace(
        only=None,
        all=True,
        check=False,
        snapshot=False,
        no_node=False,
        with_node=True,
        skip=None,
    )

    _, node_projects = setup_envs._select(args)

    assert node_projects == []


def test_bjtu_routes_extracted_shell_and_release_repositories():
    windows = Path("bjtu.bat").read_text(encoding="utf-8")
    unix = Path("bjtu.sh").read_text(encoding="utf-8")

    for script in (windows, unix):
        assert "QLH_SHELL_ROOT" in script
        assert "QLH_RELEASE_ROOT" in script
        assert "QLH_CORE_ROOT" in script
    assert r"src\app_bootstrap.py" in windows and "--maintenance" in windows
    assert "$PROJECT_ROOT/src/app_bootstrap.py" in unix
    assert 'qlh.py" maintenance' in unix


def test_sidecar_checks_require_torch_runtime_modules():
    qwen = setup_envs.ENV_BY_NAME["qwen3-sidecar"]
    gemma = setup_envs.ENV_BY_NAME["gemma4-pipeline"]

    assert {"torch", "torchvision", "transformers"} <= set(qwen.required_modules)
    assert {"torch", "accelerate", "transformers"} <= set(gemma.required_modules)


def test_qwen_sidecar_installs_torch_and_torchvision_from_one_index(monkeypatch):
    readiness = iter((False, True))
    commands: list[list[str]] = []
    monkeypatch.setattr(
        setup_qwen3_sidecar_env,
        "_ready",
        lambda **_kwargs: next(readiness),
    )
    monkeypatch.setattr(
        setup_qwen3_sidecar_env,
        "_python_path",
        lambda: Path(sys.executable),
    )
    monkeypatch.setattr(
        setup_qwen3_sidecar_env.subprocess,
        "run",
        lambda command, **_kwargs: (
            commands.append(list(command)) or SimpleNamespace(returncode=0)
        ),
    )

    assert setup_qwen3_sidecar_env.main([
        "--pipeline",
        "--torch-index-url", "https://download.pytorch.org/whl/cu126",
    ]) == 0

    assert commands[0][-2:] == [
        "torch>=2.0", "torchvision>=0.28,<0.29",
    ]
    assert "https://download.pytorch.org/whl/cu126" in commands[0]


def test_qwen_unified_setup_hint_keeps_torch_wheels_on_one_index():
    hint = setup_envs._torch_hint(
        setup_envs.ENV_BY_NAME["qwen3-sidecar"],
        "https://download.pytorch.org/whl/cu126",
    )

    assert "torchvision>=0.28,<0.29" in hint
    assert hint.count("https://download.pytorch.org/whl/cu126") == 1


def test_qwen_runtime_probe_reports_mixed_wheels(monkeypatch):
    seen: list[list[str]] = []
    monkeypatch.setattr(
        setup_envs.subprocess,
        "run",
        lambda command, **_kwargs: (
            seen.append(list(command))
            or SimpleNamespace(returncode=1, stdout="mixed Torch builds", stderr="")
        ),
    )

    assert setup_envs._qwen3_torchvision_runtime_issue(Path("python")) == "mixed Torch builds"
    assert "torchvision.ops.nms" in seen[0][2]


# ================================================================
# T-4：测试分类报告脚本（scripts/test_classification_report.py）
# ================================================================

def test_classify_places_quality_gate_and_contract_files():
    from scripts import test_classification_report as tcr

    files_by_class = tcr.classify_test_files(Path("tests"))
    # 质量门文件进 quality_gate 类（marker 入口），不被其他类截胡
    # 契约类文件（命名启发式）
    assert any("contract" in name for name in files_by_class["contract"])
    # 每个文件恰好落入一个分类（不重不漏）。用递归统计，别只数顶层 + simulation：
    # tests/ 下还有 native/ 等子目录，写死目录清单会在新增子目录时误报。
    expected_total = len(list(Path("tests").rglob("test_*.py")))
    total_classified = sum(len(v) for v in files_by_class.values())
    assert total_classified == expected_total


def test_classify_unit_is_fallback_for_unmatched_files():
    from scripts import test_classification_report as tcr

    files_by_class = tcr.classify_test_files(Path("tests"))
    # unit 是兜底类，必有内容（多数普通单测文件按命名落入）
    assert len(files_by_class["unit"]) > 0
