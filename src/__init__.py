"""
轻量化大模型分布式边缘推理优化系统
====================================
基于分层流水线的边缘分布式大模型推理系统

核心模块:
- model_module: 模型加载、量化、算子融合
- paged_kv_cache: 轻量化分页KV缓存
- tcp_comm: TCP主从通信
- scheduler: 任务调度控制
- api_server: FastAPI 后端 + React 前端一体化服务
"""

from .release_contract import PRODUCT_VERSION as __version__
