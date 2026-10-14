"""Routes extracted from api_server; shared state remains facade-owned."""

from __future__ import annotations

from types import ModuleType

from fastapi import APIRouter
from api._routing import configure_route_module

router = APIRouter()
_api_module: ModuleType | None = None
_RESOLUTION_NAMES = (
    "CreateSessionRequest",
    "Optional",
    "RenameSessionRequest",
)

def configure_api_module(module: ModuleType) -> None:
    configure_route_module(globals(), module, _RESOLUTION_NAMES)

def exported_handlers() -> dict[str, object]:
    return {name: globals()[name] for name in ['get_conversation_sync_status', 'get_conversations', 'delete_conversations', 'create_session', 'list_sessions', 'get_session_info', 'rename_session', 'delete_session', 'activate_session', 'delete_turn']}

async def get_conversation_sync_status():
    """获取用户本人主节点上的本地持久化状态。"""
    try:
        save_history = _api_module._local_store.get_local_save_history()
        local_health = _api_module._local_store.local_store_health()
    except Exception as exc:
        raise _api_module.HTTPException(503, f"本地 SQLite 不可用: {exc}")

    return {
        "save_history": save_history,
        "db_connected": False,
        "local_save_enabled": True,  # localStorage 始终可用
        "local_store_enabled": local_health.get("status") == "ok",
        "cloud_sync_enabled": False,
        "storage_backend": "sqlite",
        "effective_mode": "local_only",
    }

async def get_conversations(session_id: str = "default", limit: int = 200):
    """
    从主节点 SQLite 加载指定会话的对话历史。

    旧数据库仅为迁移期只读兼容源；内存仅承接未持久化的新会话。
    """
    try:
        local_messages = _api_module._local_store.load_local_conversation(session_id, limit)
        local_session = _api_module._local_store.get_local_session(session_id)
        if local_messages or local_session:
            return {
                "messages": [
                    {
                        "role": m["role"],
                        "content": m["content"],
                        "created_at": m.get("created_at"),
                        **({"metrics": m["metrics"]} if "metrics" in m else {}),
                    }
                    for m in local_messages
                ],
                "count": _api_module._local_store.get_local_conversation_count(session_id),
                "source": "sqlite",
            }
    except Exception as e:
        _api_module.logger.error(f"SQLite 读取对话历史失败: {e}")
        raise _api_module.HTTPException(503, f"本地对话存储不可用: {e}")

    targeted_history = _api_module.session_histories.get(session_id, [])
    return {
        "messages": [
            {"role": m["role"], "content": m["content"]}
            for m in targeted_history
        ],
        "count": len(targeted_history),
        "source": "memory_fallback",
    }

def delete_conversations(session_id: str = "default"):
    """
    清空指定会话的对话历史（数据库 + 内存同步）。

    单机模式下仅清空当前会话上下文；分布式模式下可跨节点同步。
    """
    global kv_cache
    resolved_session_id = (
        _api_module.active_session_id if session_id == "default" and _api_module.active_session_id
        else session_id
    )
    deleted_count = 0

    try:
        deleted_count = _api_module._local_store.clear_local_conversation(resolved_session_id)
        _api_module.logger.info(
            "SQLite 对话历史已清空: session=%s, %s 条",
            resolved_session_id,
            deleted_count,
        )
    except Exception as e:
        _api_module.logger.error(f"SQLite 清空对话历史失败: {e}")
        raise _api_module.HTTPException(503, f"本地对话存储不可用: {e}")

    _api_module._sync_chat_context_facade()
    _api_module._chat_context.clear_session(resolved_session_id)
    _api_module._publish_chat_context_facade()
    if resolved_session_id == _api_module.active_session_id:
        if _api_module.kv_cache:
            _api_module.kv_cache.clear()
        _api_module._init_kv_cache()
    _api_module.logger.info(f"对话历史已清空 (内存)")
    return {
        "status": "cleared",
        "session_id": resolved_session_id,
        "deleted_count": deleted_count,
    }

def create_session(req: Optional[CreateSessionRequest] = None):
    """
    创建新会话并自动激活。

    如果提供了 first_message，用它自动生成标题（截取前30字）；
    否则使用 req.title（默认"新对话"）。
    """
    import uuid
    session_id = str(uuid.uuid4())
    title = "新对话"

    if req and req.first_message:
        title = req.first_message.strip()[:30]
        if len(req.first_message.strip()) > 30:
            title += "..."
    elif req and req.title:
        title = req.title

    try:
        _api_module._local_store.create_local_session(session_id, title)
    except Exception as e:
        _api_module.logger.error(f"SQLite 创建会话失败: {e}")
        raise _api_module.HTTPException(503, f"本地会话存储不可用: {e}")

    # 注册到内存并激活
    _api_module._sync_chat_context_facade()
    _api_module._chat_context.set_history(session_id, [])
    _api_module._publish_chat_context_facade()
    _api_module._switch_session(session_id)

    _api_module.logger.info(f"会话已创建: {session_id} ({title})")
    return {
        "id": session_id,
        "title": title,
        "message_count": 0,
        "active": True,
    }

async def list_sessions(limit: int = 50, offset: int = 0):
    """
    获取所有会话列表（按 updated_at DESC 排序）。
    """
    # SQLite sessions 与尚未落盘的内存会话合并。
    mem_sessions = []
    seen_ids = set()

    try:
        local_sessions = _api_module._local_store.get_all_local_sessions(limit, offset)
        for ls in local_sessions:
            sid = ls["id"]
            seen_ids.add(sid)
            # 用内存中的实际消息数更新计数
            hist = _api_module.session_histories.get(sid, [])
            mem_sessions.append({
                "id": sid,
                "title": ls.get("title", "新对话"),
                "message_count": len(hist) or ls.get("message_count", 0),
                "created_at": ls.get("created_at"),
                "updated_at": ls.get("updated_at"),
            })
    except Exception as e:
        _api_module.logger.error(f"SQLite 读取会话列表失败: {e}")
        raise _api_module.HTTPException(503, f"本地会话存储不可用: {e}")

    for sid, hist in _api_module.session_histories.items():
        if sid not in seen_ids:
            mem_sessions.append({
                "id": sid,
                "title": "会话" if not hist else (hist[0].get("content", "")[:30] if hist else "新对话"),
                "message_count": len(hist),
                "created_at": None,
                "updated_at": None,
            })

    return {
        "sessions": mem_sessions,
        "active_session_id": _api_module.active_session_id,
        "total": _api_module._local_store.get_local_session_count() + sum(
            1 for sid in _api_module.session_histories if sid not in seen_ids
        ),
        "source": "sqlite",
    }

async def get_session_info(session_id: str):
    """获取单个会话的元数据"""
    try:
        local_session = _api_module._local_store.get_local_session(session_id)
        if local_session:
            hist = _api_module.session_histories.get(session_id, [])
            local_session["message_count"] = len(hist) or local_session.get("message_count", 0)
            local_session["active"] = session_id == _api_module.active_session_id
            return local_session
    except Exception as e:
        _api_module.logger.error(f"SQLite 读取会话失败: {e}")
        raise _api_module.HTTPException(503, f"本地会话存储不可用: {e}")

    hist = _api_module.session_histories.get(session_id, [])
    if not hist:
        raise _api_module.HTTPException(404, f"会话不存在: {session_id}")
    return {
        "id": session_id,
        "title": "新对话",
        "message_count": len(hist),
        "active": session_id == _api_module.active_session_id,
    }

async def rename_session(session_id: str, req: RenameSessionRequest):
    """重命名会话"""
    try:
        updated = _api_module._local_store.update_local_session_title(session_id, req.title)
    except Exception as e:
        _api_module.logger.error(f"SQLite 重命名会话失败: {e}")
        raise _api_module.HTTPException(503, f"本地会话存储不可用: {e}")

    if updated is None:
        raise _api_module.HTTPException(404, f"会话不存在: {session_id}")

    return updated

def delete_session(session_id: str):
    """
    删除会话及其所有对话消息。

    如果删除的是当前活跃会话，自动切换到另一个会话（或清空状态）。
    """
    global active_session_id

    try:
        deleted = _api_module._local_store.delete_local_session(session_id)
    except Exception as e:
        _api_module.logger.error(f"SQLite 删除会话失败: {e}")
        raise _api_module.HTTPException(503, f"本地会话存储不可用: {e}")

    # 从内存中移除
    _api_module._sync_chat_context_facade()
    was_active = _api_module._chat_context.drop_session(session_id)
    _api_module._publish_chat_context_facade()

    # 如果删除的是活跃会话，清除状态
    if was_active:
        if _api_module.kv_cache:
            _api_module.kv_cache.clear()
        _api_module._init_kv_cache()

    _api_module.logger.info(f"会话已删除: {session_id} ({deleted} DB rows)")
    return {"status": "deleted", "session_id": session_id}

def activate_session(session_id: str):
    """
    切换到指定会话，返回该会话的消息历史。
    """
    _api_module._switch_session(session_id)

    # 返回该会话的消息历史
    history = _api_module._get_active_history()
    return {
        "session_id": session_id,
        "messages": [
            {"role": m["role"], "content": m["content"]}
            for m in history
        ],
        "count": len(history),
    }

def delete_turn(session_id: str, turn_index: int):
    """
    删除指定会话中的单轮对话（user + assistant 两条消息）。

    turn_index: 0-based 对话轮次索引。
    """
    global kv_cache

    _api_module._sync_chat_context_facade()
    history = _api_module.session_histories.get(session_id, [])
    try:
        persisted_message_count = (
            _api_module._local_store.get_local_conversation_count(session_id)
        )
    except Exception as e:
        _api_module.logger.error(f"SQLite 读取待删除轮次失败: {e}")
        raise _api_module.HTTPException(503, f"本地会话存储不可用: {e}")

    message_count = persisted_message_count or len(history)
    if message_count < 2:
        raise _api_module.HTTPException(404, f"会话不存在或无消息: {session_id}")

    max_turn = (message_count // 2) - 1
    if turn_index < 0 or turn_index > max_turn:
        raise _api_module.HTTPException(400, f"无效的轮次索引: {turn_index}（有效范围: 0-{max_turn}）")

    try:
        deleted_count = _api_module._local_store.delete_local_message_range(session_id, turn_index)
    except Exception as e:
        _api_module.logger.error(f"SQLite 删除消息失败: {e}")
        raise _api_module.HTTPException(503, f"本地会话存储不可用: {e}")

    reloaded_active = False
    if persisted_message_count:
        if deleted_count != 2:
            raise _api_module.HTTPException(
                409, "会话已被并发修改，请刷新后重试",
            )
        # The context cache contains only the latest window, while turn_index
        # is relative to the complete SQLite transcript. Never splice the
        # window with a full-history index; invalidate and reload instead.
        _api_module._chat_context.invalidate_session(session_id)
        _api_module._publish_chat_context_facade()
        if session_id == _api_module.active_session_id:
            _api_module._switch_session(session_id)
            reloaded_active = True
        remaining_turns = max(0, (persisted_message_count - 2) // 2)
    else:
        idx = turn_index * 2
        if idx + 1 < len(history):
            del history[idx:idx + 2]
        _api_module._chat_context.mark_current(session_id)
        _api_module._publish_chat_context_facade()
        remaining_turns = len(history) // 2

    # 如果删除的是活跃会话的轮次，清 KV Cache（token 位置已变）
    if (
        not reloaded_active
        and session_id == _api_module.active_session_id
        and _api_module.kv_cache
    ):
        _api_module.kv_cache.clear()
        _api_module._init_kv_cache()

    _api_module.logger.info(f"已删除会话 {session_id} 第 {turn_index} 轮对话（{deleted_count} DB rows），剩余 {remaining_turns} 轮")
    return {
        "status": "deleted",
        "session_id": session_id,
        "turn_index": turn_index,
        "deleted_count": deleted_count,
        "remaining_turns": remaining_turns,
    }


def register_routes() -> None:
    for name in ('delete_conversations', 'create_session', 'delete_session', 'activate_session', 'delete_turn'):
        handler = _api_module._serialized_conversation_mutation(globals()[name])
        globals()[name] = handler
    router.add_api_route('/api/conversations/sync-status', get_conversation_sync_status, methods=['GET'])
    router.add_api_route('/api/conversations', get_conversations, methods=['GET'])
    router.add_api_route('/api/conversations', delete_conversations, methods=['DELETE'])
    router.add_api_route('/api/sessions', create_session, methods=['POST'])
    router.add_api_route('/api/sessions', list_sessions, methods=['GET'])
    router.add_api_route('/api/sessions/{session_id}', get_session_info, methods=['GET'])
    router.add_api_route('/api/sessions/{session_id}', rename_session, methods=['PUT'])
    router.add_api_route('/api/sessions/{session_id}', delete_session, methods=['DELETE'])
    router.add_api_route('/api/sessions/{session_id}/activate', activate_session, methods=['POST'])
    router.add_api_route('/api/sessions/{session_id}/turns/{turn_index}', delete_turn, methods=['DELETE'])
