"""inference-svc 进程入口（微服务架构改造计划 §1.3）。

并行共存（§1.4）：独立新入口，不修改 api_server.py / scheduler.py
任何代码，也不改变现有启动方式。

角色感知：`QLH_NODE_ROLE=client` 时仅提供层段/KV/流水线接口
（从节点无完整模型，chat 端点 404，由 routes._require_master_role 门控）；
`master`（默认）时全部启用。

模型加载延迟化：本入口顶层**不** import model_module / torch /
transformers —— EngineHost 内部 `_LazyModelManager` 保证首次
load 请求时才加载（§2.3：model_module 子树 9.7s 从启动成本变为
首次加载成本）。

启动日志：logs/inference_svc_startup.log（各阶段耗时）。
"""
import logging
import os
import time

from node_config import apply_node_config_to_env, resolve_initial_node_role
from pathlib import Path
from release_contract import PRODUCT_VERSION

_LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "inference_svc_startup.log"


def _setup_logging() -> None:
    """控制台 + 启动日志文件双写。"""
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(_LOG_FILE, encoding="utf-8"),
        ],
    )


def build_app(node_role: str = "master", *, engine_host=None, kv_host=None):
    """组装 FastAPI 应用（routes + EngineHost + KVHost）。

    node_role: "master"（默认，全部端点）/ "client"（仅层段/KV/流水线）。
    engine_host / kv_host: 可选依赖注入入口；默认仍创建生产宿主，测试可直接
    注入轻量实现，避免先构造真实宿主再覆盖。
    顶层 import 仅 fastapi 与 inference_service 包自身，不触发
    model_module / torch（EngineHost 构造只 import model_host 与 config）。
    """
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    from api_errors import install_http_error_handler
    from inference_service.kv_host import KVHost
    from inference_service.routes import router

    owns_engine_host = engine_host is None
    host = engine_host
    if host is None:
        from inference_service.engine_host import EngineHost

        host = EngineHost()

    @asynccontextmanager
    async def lifespan(_app):
        import asyncio

        if owns_engine_host and node_role == "master":
            sqlite_path = await asyncio.to_thread(host.initialize_storage)
            logging.getLogger("inference_svc_main").info(
                "主节点 SQLite 已就绪: %s",
                sqlite_path,
            )
        yield
        await asyncio.to_thread(host.close)

    app = FastAPI(title="inference-svc", version=PRODUCT_VERSION, lifespan=lifespan)
    host.role = node_role
    app.state.engine_host = host
    app.state.kv_host = kv_host if kv_host is not None else KVHost()
    app.state.node_role = node_role
    install_http_error_handler(app)
    app.include_router(router)
    return app


def main() -> None:
    _setup_logging()
    logger = logging.getLogger("inference_svc_main")
    t_start = time.time()

    # Keep the lightweight service entrypoint aligned with config.py: an
    # unconfigured source checkout is a client, never an accidental master.
    apply_node_config_to_env()
    node_role = resolve_initial_node_role()

    if node_role == "client":
        # 1.5 从节点入口：不 import fastapi/uvicorn，直接起 peer
        from inference_service.peer import run_peer

        logger.info(
            f"从节点模式启动（{time.time() - t_start:.2f}s 内完成 import），连接主节点"
        )
        run_peer()
        return

    host = os.environ.get("QLH_INFERENCE_HOST", "127.0.0.1")
    port = int(os.environ.get("QLH_INFERENCE_PORT", "8010"))

    logger.info(f"inference-svc 启动: role={node_role} 监听 {host}:{port}")
    app = build_app(node_role)
    logger.info(f"应用组装完成 ({time.time() - t_start:.2f}s)，导入 uvicorn")

    import uvicorn

    logger.info(f"uvicorn 就绪，开始监听 {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
