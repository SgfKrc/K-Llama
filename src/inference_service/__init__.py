"""inference-svc —— 推理数据面独立进程（微服务架构改造计划 阶段 1，§1.1）。

并行共存（§1.4）：本包独立实现，不修改现有 Python 后端任何代码；
api_server.py / scheduler.py 保持原样作为可运行基线。

包结构：
  protocol.py         冻结的 HTTP 契约模型（§4.1）
  routes.py           FastAPI 子应用（/v1/*）
  engine_host.py      模型托管宿主（进程内 ModelHost + 执行段宿主）
  kv_host.py          任务级 KV 缓存生命周期
  tensor_transport.py loopback 张量序列化（复用 tcp_comm）
"""

try:
    from release_contract import (
        INFERENCE_SERVICE_CONTRACT_VERSION as __contract_version__,
        PRODUCT_VERSION as __version__,
    )
except ImportError:  # package import through ``src.inference_service``
    from ..release_contract import (  # type: ignore[no-redef]
        INFERENCE_SERVICE_CONTRACT_VERSION as __contract_version__,
        PRODUCT_VERSION as __version__,
    )
