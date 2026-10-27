"""节点运行时状态（node_runtime）—— 阶段 0 边界治理（0.3）

承接 config.py 中被运行时写回的可变项（NODE_ID / NODE_ROLE），
config.py 退化为纯静态配置（端口、路径、阈值等只读常量）。

迁移来源：
  - scheduler._sync_runtime_node_config（scheduler.py:126-140）写回 cfg.NODE_ID
  - node_config.py:230 写回 cfg.NODE_ID

用法：
  from node_runtime import node_runtime
  node_runtime.set_node_id("node-2")
  current = node_runtime.get_node_id()
"""
import threading
from typing import Optional


class NodeRuntime:
    """节点运行时可变状态单例（线程安全）。"""

    def __init__(self):
        try:
            import config as _cfg
            self._node_id: str = str(getattr(_cfg, "NODE_ID", "client"))
            self._node_role: str = str(getattr(_cfg, "NODE_ROLE", "client"))
        except Exception:
            self._node_id = "client"
            self._node_role = "client"
        self._lock = threading.RLock()

    # ---- node_id ----
    def get_node_id(self) -> str:
        with self._lock:
            return self._node_id

    def set_node_id(self, node_id: str) -> None:
        with self._lock:
            self._node_id = str(node_id)

    # ---- node_role ----
    def get_node_role(self) -> str:
        with self._lock:
            return self._node_role

    def set_node_role(self, role: str) -> None:
        with self._lock:
            self._node_role = str(role)

    # ---- 组合 ----
    def update(self, node_id: Optional[str] = None, node_role: Optional[str] = None) -> None:
        """等价原 scheduler._sync_runtime_node_config 的写回语义。"""
        with self._lock:
            if node_id:
                self._node_id = str(node_id)
            if node_role:
                self._node_role = str(node_role)

    def __repr__(self):
        return "<NodeRuntime node_id=%r node_role=%r>" % (
            self._node_id, self._node_role)


# 全局单例
node_runtime = NodeRuntime()
