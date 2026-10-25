"""Opt-in F3: real physical LL/DL relay through Textual, v2 snapshots and request journal.

The bench prepares a normal automatic model selection and at least one physical
worker. This smoke never loads a full model, configures ranges or uses workflows.
Missing prerequisites skip before sending inference; all later failures are errors.
"""
from __future__ import annotations
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import threading
import uuid
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'src', ROOT/'tests'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from tui_e2e_flow import wait_for
from scripts.package_contract.refactor_product_gate import prepared, completed, ProductGateError

pytestmark = [pytest.mark.real_model, pytest.mark.slow]
HOST = os.environ.get('QLH_TUI_E2E_HOST', '127.0.0.1')
PORT = int(os.environ.get('QLH_TUI_E2E_PORT', '8000'))
TIMEOUT = float(os.environ.get('QLH_TUI_E2E_TIMEOUT', '300'))
MIN_NODES = int(os.environ.get('QLH_TUI_E2E_MIN_NODES', '2'))
PROMPT = 'What is 1 + 1? Reply with only the result, no explanation.'


def test_dual_host_tui_routed_pipeline_streams_text(monkeypatch):
    if os.environ.get('QLH_RUN_DUAL_HOST_TUI') != '1':
        pytest.skip('Set QLH_RUN_DUAL_HOST_TUI=1 with a prepared physical cluster')
    pytest.importorskip('textual')
    from tui_api import ApiClient, ApiError
    import tui_textual
    from textual.widgets import Input, Static
    api = ApiClient(HOST, PORT, timeout=15, auth_token=os.environ.get('QLH_PRODUCT_AUTH_TOKEN', ''))
    try:
        before = api.get('/client/state')
        prepared(before, minimum_nodes=MIN_NODES)
        nodes = api.get('/cluster/nodes')['nodes']
        assert sum(n.get('is_available') is True for n in nodes) >= MIN_NODES
    except (ApiError, OSError, ValueError, KeyError, AssertionError) as exc:
        pytest.skip(f'F3 physical prerequisites unavailable: {exc}')
    previous = {r['request_id'] for r in api.get('/requests')['requests']}
    original = tui_textual.iter_chat_payloads
    done, errors, stream = [], [], []
    ended = threading.Event()

    def observe(*args, **kwargs):
        kwargs.update(max_new_tokens=64, read_timeout=min(TIMEOUT,120))
        try:
            assert kwargs.get('routing_preference') == 'distributed_required'
            for payload in original(*args, **kwargs):
                stream.append(deepcopy(payload))
                if payload.get('done'):
                    done.append(deepcopy(payload))
                if payload.get('error') or payload.get('cancelled'):
                    errors.append(deepcopy(payload))
                yield payload
        finally:
            ended.set()
    monkeypatch.setattr(tui_textual, 'iter_chat_payloads', observe)

    async def drive():
        app = tui_textual.KoakumaApp(api, interval=2, enable_thinking=False)
        async with app.run_test(size=(120,40)) as pilot:
            await pilot.press('enter')
            assert await wait_for(pilot, lambda: app.main_screen is not None
                and app.main_screen.runtime_ready is True and not app.main_screen._core_inflight
                and app.main_screen.model_intent.get('operation_id') == before['operation']['operation_id']
                and not app.main_screen.chat_model_gate())
            box = app.screen.query_one('#chat-input', Input)
            await pilot.click('#chat-input')
            box.value = '/new R6-F3-' + uuid.uuid4().hex
            await pilot.press('enter')
            assert await wait_for(pilot, lambda: bool(app.session_id) and not any(w.group=='sessions' and not w.is_finished for w in app.workers))
            await pilot.pause()
            session_id = app.session_id
            await pilot.click('#chat-input')
            box.value = '/route required'; await pilot.press('enter')
            assert await wait_for(pilot, lambda: app.routing_preference == 'distributed_required')
            await pilot.click('#chat-input')
            box.value = PROMPT; await pilot.press('enter')
            assert await wait_for(pilot, lambda: ended.is_set() and not any(w.group=='chat' and not w.is_finished for w in app.workers),timeout=TIMEOUT)
            await pilot.pause()
            assert app.session_id == session_id
            transcript = app.screen.query_one('#chat-log',Static).render().plain
            status = app.screen.query_one('#chat-status',Static).render().plain
            out = os.environ.get('QLH_TUI_E2E_OUTPUT')
            if out:
                path = Path(out); path.mkdir(parents=True, exist_ok=False)
                (path/'screen.svg').write_text(app.export_screenshot(),encoding='utf-8')
                (path/'ui.json').write_text(json.dumps({'session_id':session_id,'transcript':transcript,'status':status,'stream':stream,'model_context':app.main_screen.chat_model_context(),'gate':app.main_screen.chat_model_gate()},ensure_ascii=False,indent=2),encoding='utf-8')
            assert PROMPT in transcript and transcript.rsplit('assistant ',1)[-1].strip() == '2', (transcript,status,stream)
    asyncio.run(drive())
    assert len(done) == 1 and not errors, (done,errors)
    result = completed(done[0],before,api.get('/client/state'),api.get('/requests'),previous_request_ids=previous)
    assert result['fallback'] is False
