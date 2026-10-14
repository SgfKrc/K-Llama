"""任务图 journal 冷启动 fence 定向测试。

背景（DEADLINE-01 遗留隐患）：`api_server` 的 TaskGraph coordinator 曾在
import 模块阶段就调用 `_create_task_graph_coordinator()`，因此任何只为了
**import** `api_server` 的冷启动子进程（Windows ``multiprocessing`` spawn
worker、pytest 探针、命令行探针……）都会去获取生产 task-journal 的单写者独占
锁。当父进程（真正的生产服务）已经持有该锁时，子进程会被错误记录为
「任务图 journal 初始化失败，任务图已禁用」，并且子进程在 import 阶段就成了
一个失败关闭的 writer。

修复语义：journal 锁的获取推迟到**首次真实使用**。真正常用的进程（执行任务
链、查询任务工作流、运行时热切换）依然是唯一 writer 并保持单写者 fencing；
仅 import / 非任务链探针调用绝不争用生产锁。

以下用例守护：
1. lazy 代理在首次使用前不调用工厂（import 无副作用）。
2. 生产锁被占用时，writer 工厂仍 fail-closed（不削弱 fencing）。
3. 真实冷启动子进程（子解释器 import）不碰生产锁；首个真实使用才争锁并失败
   关闭。
"""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server
from task_graph import StageSpec, TaskGraphCoordinator
from task_journal import SQLiteTaskJournal


def test_lazy_coordinator_defers_writer_until_first_use():
    """首次访问前不调用工厂；首次使用才物化 writer。"""
    calls = []

    def factory():
        calls.append(True)
        return TaskGraphCoordinator(max_records=10)

    proxy = api_server._LazyTaskGraphCoordinator(factory)

    assert calls == []
    assert proxy._is_materialized() is False

    status = proxy.journal_status()
    assert status["backend"] == "memory"
    assert calls == [True]
    assert proxy._is_materialized() is True
    assert status["available"] is True

    proxy.close()
    assert proxy._is_materialized() is False


def test_writer_factory_fails_closed_when_production_lock_is_claimed(
    tmp_path, monkeypatch,
):
    """生产锁被另一个进程持有 → 工厂必须 fail-closed（不得成为第二个 writer）。"""
    path = str(tmp_path / "state" / "task_graph" / "instance-fence.sqlite3")
    owner = SQLiteTaskJournal(path)
    try:
        monkeypatch.setattr(api_server, "TASK_GRAPH_ENABLED", True)
        monkeypatch.setattr(api_server, "TASK_GRAPH_JOURNAL_PATH", path)
        monkeypatch.setattr(api_server, "TASK_GRAPH_RETENTION_DAYS", 0)
        monkeypatch.setattr(api_server, "TASK_GRAPH_RETENTION_MAX_RECORDS", 0)

        coordinator = api_server._create_task_graph_coordinator()

        status = coordinator.journal_status()
        assert status["available"] is False
        assert status["backend"] == "memory"
        assert "another process" in (status.get("error") or "")
        assert "another process" in coordinator.availability_error

        # 真正写 journal 的执行路径必须在 executor 调用前被拒绝，绝不静默
        # 成为 memory writer。
        executor_calls = []
        with pytest.raises(api_server.TaskGraphUnavailable):
            coordinator.run(
                stages=[StageSpec("only", "full_inference")],
                final_stage_id="only",
                root_input={"message": "must not execute"},
                execute_stage=lambda *args: executor_calls.append(args),
                workflow_id="wf_coldstart_fence",
            )
        assert executor_calls == []
    finally:
        owner.close()


def test_cold_start_subprocess_import_does_not_claim_production_journal_lock(
    tmp_path,
):
    """冷启动子进程仅 import api_server 不得争用生产 journal 锁。

    模拟 Windows ``multiprocessing`` spawn / pytest 探针子进程：父进程（测试
    进程）持有生产 journal 锁，子进程在一个全新解释器里 import api_server，
    必须保持惰性；只有首个真实使用（如查询 journal 状态）才允许争锁，且因锁
    被占必须 fail-closed。
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"

    env = os.environ.copy()
    env.update({
        "QLH_TASK_GRAPH_ENABLED": "true",
        "QLH_TASK_WORKER_EXPERIMENTAL_ENABLED": "false",
        "QLH_STATE_DIR": str(tmp_path / "state"),
        "QLH_TASK_GRAPH_INSTANCE_ID": "cold-start-fence-writer",
    })
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(src_root), env.get("PYTHONPATH", "")) if item
    )

    # 先在一个独立子解释器里计算「生产」journal 路径（与探针子进程相同的
    # STATE_DIR / INSTANCE_ID），父进程据此持有独占锁。
    calc = subprocess.run(
        [sys.executable, "-c", "import config; print(config.TASK_GRAPH_JOURNAL_PATH)"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert calc.returncode == 0, calc.stderr
    journal_path = calc.stdout.strip()
    assert journal_path

    owner = SQLiteTaskJournal(journal_path)
    try:
        probe = textwrap.dedent(
            f"""
            import asyncio
            import sys

            sys.path.insert(0, {str(src_root)!r})

            import api_server

            cold = api_server.task_graph_coordinator
            assert cold._is_materialized() is False, (
                "importing api_server must not claim the production journal lock"
            )

            # 非任务链的冷启动探针调用不得触发 writer 初始化。
            asyncio.run(api_server.health())
            assert cold._is_materialized() is False, (
                "non-task-graph probe calls must not initialize the writer"
            )

            # 首个真实使用：此时才允许争用锁，且必须 fail-closed。
            status = cold.journal_status()
            assert cold._is_materialized() is True
            assert status["available"] is False, (
                "claimed production lock must leave journal unavailable"
            )
            assert "another process" in (status.get("error") or "")

            print("COLD_START_FENCE_OK")
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        assert completed.returncode == 0, completed.stderr or completed.stdout
        assert "COLD_START_FENCE_OK" in completed.stdout
    finally:
        owner.close()
