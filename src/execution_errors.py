"""Stable worker-stage failures shared by API and native adapters."""
class StageExecutionError(RuntimeError):
    def __init__(self, message, *, code="stage_execution_failed"):
        self.code = code
        super().__init__(message)

DEPENDENCY_FAILURES_KEY = "__qlh_failed_dependencies__"
