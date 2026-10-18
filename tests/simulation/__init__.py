"""
分布式推理仿真测试框架

提供端到端的仿真测试能力，验证分布式推理在真实环境下的可用性。
"""

from .framework import (
    TestOrchestrator,
    TestConfig,
    TestResult,
    BackendManager,
    RequestSender,
    HumanSimulator,
    ResponseValidator,
    ValidationResult,
)

from .scenarios import (
    SCENARIOS,
    Scenario,
)


from .task_worker_harness import (
    SIMULATION_SCHEMA_VERSION as TASK_WORKER_SIMULATION_SCHEMA_VERSION,
    SimulationScenario as TaskWorkerSimulationScenario,
    SimulationScenarioError as TaskWorkerSimulationScenarioError,
    TaskWorkerControlSimulationHarness,
    available_scenarios as available_task_worker_scenarios,
)


from .html_report import generate_html_report

# 测试模块
from . import test_single_node
from . import test_distributed
from . import test_degradation
from . import test_stress
from . import test_exceptions

__all__ = [
    "TestOrchestrator",
    "TestConfig",
    "TestResult",
    "BackendManager",
    "RequestSender",
    "HumanSimulator",
    "ResponseValidator",
    "ValidationResult",
    "SCENARIOS",
    "Scenario",
    "TASK_WORKER_SIMULATION_SCHEMA_VERSION",
    "TaskWorkerSimulationScenario",
    "TaskWorkerSimulationScenarioError",
    "TaskWorkerControlSimulationHarness",
    "available_task_worker_scenarios",
    "generate_html_report",
    "test_single_node",
    "test_distributed",
    "test_degradation",
    "test_stress",
    "test_exceptions",
]

__version__ = "3.0.0"
