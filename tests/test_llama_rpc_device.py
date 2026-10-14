"""RPC device 接入层测试（fail-closed 行为 + 引擎接入点）。

不需要真实 ggml-rpc 二进制：用 fake 的 ggml API 与 fake 的 llama_cpp 模块覆盖
注入顺序、错误路径与生命周期。真实注入的集成用例见文件末尾（需环境变量显式开启）。
"""

import ctypes
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llama_engine import LlamaCppEngine  # noqa: E402
from llama_rpc_device import (  # noqa: E402
    DEFAULT_WORKER_EXE,
    LocalRpcWorker,
    RpcBackendInjector,
    RpcInjectionError,
    RpcSession,
)


# --------------------------------------------------------------------- fakes

class _FakeGgml:
    def __init__(self, reg_ptr=0x1000, add_server_ret=0x9000):
        self.registered = []
        self.load_all_calls = 0
        self.reg_ptr = reg_ptr
        self.add_server_ret = add_server_ret
        self._add_server = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)(
            lambda endpoint: self.add_server_ret
        )
        self.add_server_endpoints = []

    def ggml_backend_load_all(self):
        self.load_all_calls += 1

    def ggml_backend_reg_by_name(self, name):
        assert name == b"RPC"
        return self.reg_ptr

    def ggml_backend_register(self, reg):
        self.registered.append(int(reg.value) if isinstance(reg, ctypes.c_void_p) else reg)

    def ggml_backend_dev_count(self):
        return 1

    def ggml_backend_dev_get(self, index):
        return 0x5000 + index


class _FakeBase:
    def __init__(self, add_server_addr):
        self.add_server_addr = add_server_addr
        self.proc_calls = []

    def ggml_backend_reg_get_proc_address(self, reg, name):
        self.proc_calls.append(name)
        return self.add_server_addr

    def ggml_backend_dev_name(self, ptr):
        return b"RPC0"

    def ggml_backend_dev_description(self, ptr):
        return b"127.0.0.1:50163"


class _FakeRpc:
    def ggml_backend_rpc_reg(self):
        return 0x2000


class _FakeApi:
    """替换 llama_rpc_device._GgmlApi：记录调用顺序，device 列表可定制。"""

    last: "_FakeApi | None" = None

    def __init__(self, lib_dir=None, *, add_server_ret=0x9000, rpc_reg_present=True,
                 devices=((0x5000, "RPC0", "127.0.0.1:50163"),)):
        self.ggml = _FakeGgml(reg_ptr=0x1000 if rpc_reg_present else 0,
                              add_server_ret=add_server_ret)
        self.base = _FakeBase(ctypes.cast(self.ggml._add_server, ctypes.c_void_p).value)
        self.rpc = _FakeRpc()
        self._devices = list(devices)
        _FakeApi.last = self

    def device_count(self):
        return len(self._devices)

    def device_ptr(self, index):
        return self._devices[index][0]

    def device_name(self, ptr):
        for dev_ptr, name, _ in self._devices:
            if dev_ptr == ptr:
                return name
        return "?"

    def device_description(self, ptr):
        for dev_ptr, _, desc in self._devices:
            if dev_ptr == ptr:
                return desc
        return ""


@pytest.fixture
def fake_injector(monkeypatch):
    """RpcBackendInjector，其 ggml API 与 DLL 路径处理都被替换。"""
    monkeypatch.setattr("llama_rpc_device.ensure_dll_search_path", lambda lib_dir: None)
    monkeypatch.setattr("llama_rpc_device._GgmlApi", _FakeApi)
    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
    lc_module = types.ModuleType("llama_cpp.llama_cpp")
    lc_module.llama_backend_init = lambda: None
    lc_module.llama_model_load_from_file = lambda path, params: "orig-model"
    sys.modules["llama_cpp.llama_cpp"] = lc_module
    return RpcBackendInjector(lib_dir=".")


# --------------------------------------------------------------------- 注入

def test_attach_registers_the_reg_returned_by_add_server(fake_injector):
    """官方流程的最后一步必须发生：add_server 的返回值要 ggml_backend_register。"""
    devices = fake_injector.attach(["127.0.0.1:50163"])

    assert [d.name for d in devices] == ["RPC0"]
    assert devices[0].endpoint == "127.0.0.1:50163"
    assert _FakeApi.last.ggml.registered == [0x9000], "缺少 ggml_backend_register(reg)"
    assert _FakeApi.last.ggml.load_all_calls == 1
    assert _FakeApi.last.base.proc_calls == [b"ggml_backend_rpc_add_server"]


def test_attach_requires_endpoints(fake_injector):
    with pytest.raises(RpcInjectionError):
        fake_injector.attach([])


def test_attach_falls_back_to_rpc_reg_then_fails_closed_without_add_server(monkeypatch):
    """注册表里没有 RPC backend 时用 ggml_backend_rpc_reg() 兜底；再失败即抛。"""

    class _NoProcAddrApi(_FakeApi):
        def __init__(self, lib_dir=None, **kwargs):
            kwargs.setdefault("rpc_reg_present", False)  # 注册表里没有 RPC backend
            super().__init__(lib_dir, **kwargs)
            self.base.add_server_addr = 0

    monkeypatch.setattr("llama_rpc_device.ensure_dll_search_path", lambda lib_dir: None)
    monkeypatch.setattr("llama_rpc_device._GgmlApi", _NoProcAddrApi)
    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
    lc_module = types.ModuleType("llama_cpp.llama_cpp")
    lc_module.llama_backend_init = lambda: None
    sys.modules["llama_cpp.llama_cpp"] = lc_module

    injector = RpcBackendInjector(lib_dir=".")
    with pytest.raises(RpcInjectionError, match="add_server"):
        injector.attach(["127.0.0.1:50163"])
    # 兜底路径确实注册了 backend 自带 reg
    assert _NoProcAddrApi.last.ggml.registered == [0x2000]


def test_attach_fail_closed_when_add_server_returns_null(monkeypatch):
    """worker 没起 / 协议不匹配 -> add_server 返回 NULL -> 抛错，绝不静默用 CPU。"""
    monkeypatch.setattr("llama_rpc_device.ensure_dll_search_path", lambda lib_dir: None)
    monkeypatch.setattr(
        "llama_rpc_device._GgmlApi",
        lambda lib_dir=None: _FakeApi(lib_dir, add_server_ret=0),
    )
    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
    lc_module = types.ModuleType("llama_cpp.llama_cpp")
    lc_module.llama_backend_init = lambda: None
    sys.modules["llama_cpp.llama_cpp"] = lc_module

    injector = RpcBackendInjector(lib_dir=".")
    with pytest.raises(RpcInjectionError, match="add_server"):
        injector.attach(["127.0.0.1:50163"])


def test_device_array_is_null_terminated(fake_injector):
    fake_injector.attach(["127.0.0.1:50163"])
    array = fake_injector.device_array()
    assert array[0] == 0x5000
    assert array[1] is None
    assert len(array) == 2


def test_patched_llama_loader_injects_devices_and_restores(fake_injector):
    fake_injector.attach(["127.0.0.1:50163"])
    import llama_cpp.llama_cpp as lc

    original = lc.llama_model_load_from_file
    seen = {}

    def fake_load(path, params):
        seen["devices"] = params.devices
        return "model"

    lc.llama_model_load_from_file = fake_load
    try:
        params = types.SimpleNamespace(devices=None)
        with fake_injector.patched_llama_loader() as array:
            assert lc.llama_model_load_from_file(b"m.gguf", params) == "model"
        assert seen["devices"] == ctypes.cast(array, ctypes.c_void_p).value
        assert lc.llama_model_load_from_file is fake_load, "patch 必须恢复"
    finally:
        lc.llama_model_load_from_file = original


def test_patched_llama_loader_can_disable_repack(fake_injector):
    """use_extra_bufts=False 必须落到 params 上（llama-cpp-python 的 Llama 不暴露该开关）。"""
    fake_injector.attach(["127.0.0.1:50163"])
    import llama_cpp.llama_cpp as lc

    original = lc.llama_model_load_from_file
    seen = {}
    lc.llama_model_load_from_file = lambda path, params: seen.setdefault("repack", params.use_extra_bufts)
    try:
        params = types.SimpleNamespace(devices=None, use_extra_bufts=True)
        with fake_injector.patched_llama_loader(use_extra_bufts=False):
            lc.llama_model_load_from_file(b"m.gguf", params)
        assert seen["repack"] is False

        # 不传该参数时不干预（保持 llama.cpp 默认 = 启用 repack）
        params2 = types.SimpleNamespace(devices=None, use_extra_bufts=True)
        with fake_injector.patched_llama_loader():
            lc.llama_model_load_from_file(b"m.gguf", params2)
        assert params2.use_extra_bufts is True
    finally:
        lc.llama_model_load_from_file = original


# --------------------------------------------------------------------- worker

def test_local_worker_rejects_missing_exe(tmp_path):
    worker = LocalRpcWorker(tmp_path / "nope.exe")
    with pytest.raises(RpcInjectionError, match="不存在"):
        worker.start(timeout=1)


def test_session_requires_some_endpoint():
    with pytest.raises(RpcInjectionError):
        RpcSession.open(endpoints=[], autostart_worker=None)


# ------------------------------------------------------------- 引擎接入（无真模型）

class _FakeLlama:
    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.closed = False
        _FakeLlama.instances.append(self)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_llama_module(monkeypatch):
    module = types.ModuleType("llama_cpp")
    module.Llama = _FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    _FakeLlama.instances = []
    return module


def test_engine_normalizes_rpc_servers():
    assert LlamaCppEngine._normalize_rpc_servers(None) == []
    assert LlamaCppEngine._normalize_rpc_servers("a:1, b:2") == ["a:1", "b:2"]
    assert LlamaCppEngine._normalize_rpc_servers(["a:1", " b:2 "]) == ["a:1", "b:2"]
    with pytest.raises(ValueError):
        LlamaCppEngine._normalize_rpc_servers(123)


def test_engine_rpc_failure_is_fail_closed(fake_llama_module, monkeypatch, tmp_path):
    """RPC 注入失败必须让 load_model 抛错，不能悄悄退回 CPU 加载。"""
    engine = LlamaCppEngine()
    model_file = tmp_path / "m.gguf"
    model_file.write_bytes(b"stub")

    def boom(*args, **kwargs):
        raise RpcInjectionError("worker 未就绪")

    monkeypatch.setattr(RpcSession, "open", staticmethod(boom))
    with pytest.raises(RpcInjectionError):
        engine.load_model(model_path=str(model_file), rpc_servers="127.0.0.1:1")
    assert engine.is_loaded is False
    assert _FakeLlama.instances == [], "注入失败时不应构造 Llama"
    assert engine._rpc_session is None


def test_engine_cpu_path_unchanged(fake_llama_module, tmp_path):
    engine = LlamaCppEngine()
    model_file = tmp_path / "m.gguf"
    model_file.write_bytes(b"stub")

    engine.load_model(model_path=str(model_file))
    try:
        assert engine.is_loaded is True
        assert _FakeLlama.instances[0].kwargs.get("n_gpu_layers") is None
        assert engine._rpc_session is None
    finally:
        engine.unload()
    assert _FakeLlama.instances[0].closed is True


def test_engine_unload_closes_rpc_session(tmp_path):
    engine = LlamaCppEngine()
    closed = []
    engine._rpc_session = types.SimpleNamespace(close=lambda: closed.append(True))
    engine.unload()
    assert closed == [True]
    assert engine._rpc_session is None


def test_patched_model_params_sets_and_restores(monkeypatch):
    from llama_rpc_device import patched_model_params

    lc_module = types.ModuleType("llama_cpp.llama_cpp")
    seen = {}

    def fake_load(path, params):
        seen["repack"] = params.use_extra_bufts
        return "model"

    lc_module.llama_model_load_from_file = fake_load
    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_cpp", lc_module)

    params = types.SimpleNamespace(use_extra_bufts=True)
    with patched_model_params(use_extra_bufts=False):
        lc_module.llama_model_load_from_file(b"m.gguf", params)
    assert seen["repack"] is False
    assert lc_module.llama_model_load_from_file is fake_load, "patch 必须恢复"


def test_engine_cpu_path_align_disables_repack(fake_llama_module, monkeypatch, tmp_path):
    """纯本机加载时 align_numerics=True 也要关 CPU_REPACK（否则只对齐了一半）。

    fake 的 ``Llama`` 不会自己触发加载，因此这里按真实 ``_internals`` 的行为在构造期间
    调用一次 ``llama_model_load_from_file``，从而验证 patch 确实生效。
    """
    seen = {}
    calls = []
    lc_module = types.ModuleType("llama_cpp.llama_cpp")

    def fake_load(path, params):
        calls.append(params.use_extra_bufts)
        return "model"

    lc_module.llama_model_load_from_file = fake_load
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_cpp", lc_module)

    class _RecordingLlama(_FakeLlama):
        def __init__(self, **kwargs):
            seen["patched"] = lc_module.llama_model_load_from_file is not fake_load
            if seen["patched"]:
                # 模拟 _internals.LlamaModel.__init__ 的调用（params 由库构造）
                lc_module.llama_model_load_from_file(
                    b"m.gguf", types.SimpleNamespace(use_extra_bufts=True)
                )
            super().__init__(**kwargs)

    fake_llama_module.Llama = _RecordingLlama
    engine = LlamaCppEngine()
    model_file = tmp_path / "m.gguf"
    model_file.write_bytes(b"stub")
    engine.load_model(model_path=str(model_file), align_numerics=True)
    try:
        assert seen["patched"] is True, "纯 CPU 路径也必须进入 patch 上下文"
        assert calls == [False], "params.use_extra_bufts 必须被改成 False"
        assert lc_module.llama_model_load_from_file is fake_load, "patch 必须恢复"
    finally:
        engine.unload()


# --------------------------------------------------------------- 分片（部分驻留）

def _cpu_plus_rpc_api(monkeypatch):
    """让 fake 枚举里同时出现本机 CPU 与 RPC0（分片场景）。"""
    monkeypatch.setattr("llama_rpc_device.ensure_dll_search_path", lambda lib_dir: None)
    monkeypatch.setattr(
        "llama_rpc_device._GgmlApi",
        lambda lib_dir=None: _FakeApi(lib_dir, devices=(
            (0x4000, "CPU", "13th Gen Intel(R) Core(TM) i9-13900H"),
            (0x5000, "RPC0", "127.0.0.1:50163"),
        )),
    )
    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
    lc_module = types.ModuleType("llama_cpp.llama_cpp")
    lc_module.llama_backend_init = lambda: None
    lc_module.llama_model_load_from_file = lambda path, params: "orig-model"
    sys.modules["llama_cpp.llama_cpp"] = lc_module
    return RpcBackendInjector(lib_dir=".")


def test_device_array_puts_local_cpu_first(monkeypatch):
    injector = _cpu_plus_rpc_api(monkeypatch)
    injector.attach(["127.0.0.1:50163"])
    assert injector.cpu_device_ptr() == 0x4000

    array = injector.device_array(include_cpu=True)
    assert array[0] == 0x4000, "本机 CPU 必须在最前（与 tensor_split 顺序一致）"
    assert array[1] == 0x5000
    assert array[2] is None


def test_patched_llama_loader_sets_tensor_split(monkeypatch):
    injector = _cpu_plus_rpc_api(monkeypatch)
    injector.attach(["127.0.0.1:50163"])
    import llama_cpp.llama_cpp as lc

    seen = {}

    def fake_load(path, params):
        seen["devices"] = params.devices
        seen["split"] = list(params.tensor_split[0:2]) if params.tensor_split else None
        return "model"

    lc.llama_model_load_from_file = fake_load
    params = types.SimpleNamespace(devices=None, tensor_split=None)
    with injector.patched_llama_loader(device_array=injector.device_array(include_cpu=True),
                                       tensor_split=[0.4, 0.6]):
        lc.llama_model_load_from_file(b"m.gguf", params)
    assert seen["devices"] is not None
    assert seen["split"] == pytest.approx([0.4, 0.6])


def test_resolve_rpc_split_variants():
    engine = LlamaCppEngine
    assert engine._resolve_rpc_split(None, 1) is None
    assert engine._resolve_rpc_split(0.25, 1) == pytest.approx([0.75, 0.25])
    assert engine._resolve_rpc_split("0.5", 1) == pytest.approx([0.5, 0.5])   # 单值字符串 == 数字
    assert engine._resolve_rpc_split("0.5,0.5", 1) == pytest.approx([0.5, 0.5])
    assert engine._resolve_rpc_split([0.3, 0.3, 0.4], 2) == pytest.approx([0.3, 0.3, 0.4])
    with pytest.raises(ValueError):
        engine._resolve_rpc_split("0.5,0.5,0.5", 1)  # 长度不匹配
    with pytest.raises(ValueError):
        engine._resolve_rpc_split(0.5, 2)            # 单个比例只允许 1 个 RPC device
    with pytest.raises(ValueError):
        engine._resolve_rpc_split([-0.5, 1.5], 1)    # 负数
    with pytest.raises(ValueError):
        engine._resolve_rpc_split(object(), 1)


def test_engine_rejects_bad_split_before_loading(monkeypatch, tmp_path):
    """分片比例非法时，load_model 必须在真正加载前失败（fail-closed）。"""
    injector = _cpu_plus_rpc_api(monkeypatch)
    stub = types.ModuleType("llama_cpp")
    stub.Llama = _FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", stub)
    engine = LlamaCppEngine()
    model_file = tmp_path / "m.gguf"
    model_file.write_bytes(b"stub")

    class _FakeSession:
        def __init__(self):
            self.injector = injector
        def close(self):
            pass

    injector.attach(["127.0.0.1:50163"])
    monkeypatch.setattr(RpcSession, "open", staticmethod(lambda **kw: _FakeSession()))
    with pytest.raises(ValueError):
        engine.load_model(model_path=str(model_file), rpc_servers="127.0.0.1:50163",
                          rpc_split="0.5,0.5,0.5")   # 需要 2 个比例，给了 3 个
    assert engine.is_loaded is False


# --------------------------------------------------------------- 真实注入（可选）

_INTEGRATION = (
    sys.platform == "win32"
    and os.environ.get("QLH_RPC_INTEGRATION") == "1"
    and DEFAULT_WORKER_EXE.exists()
)


@pytest.mark.skipif(not _INTEGRATION, reason="需要 QLH_RPC_INTEGRATION=1 与 runtime/llama-cpp/b_rpc worker")
def test_real_rpc_device_injection_roundtrip(tmp_path):
    """真实注入一回合：RPC0 出现在 device 列表，close 后 worker 被回收。"""
    session = RpcSession.open(
        autostart_worker=str(DEFAULT_WORKER_EXE),
        worker_port=50199,
        worker_threads=2,
        worker_log=tmp_path / "worker.log",
    )
    try:
        names = [dev.name for dev in session.injector.devices]
        assert names and all(name.startswith("RPC") for name in names)
        assert session.worker is not None and session.worker.proc.poll() is None
        proc = session.worker.proc
    finally:
        session.close()
    assert proc.poll() is not None, "close() 必须回收本地 worker"


def test_local_rpc_worker_stop_failure_preserves_process_for_retry():
    worker = LocalRpcWorker()
    fail = [True]

    class Process:
        returncode = None
        def poll(self):
            return self.returncode
        def terminate(self):
            if fail[0]:
                raise PermissionError("cannot terminate RPC child")
        def wait(self, timeout):
            self.returncode = 0
            return 0

    process = Process()
    worker.proc = process
    with pytest.raises(PermissionError, match="cannot terminate RPC child"):
        worker.stop()
    assert worker.proc is process and process.poll() is None
    fail[0] = False
    worker.stop()
    assert worker.proc is None and process.poll() == 0


def test_engine_rpc_close_failure_preserves_session_and_identity():
    engine = LlamaCppEngine()
    fail = [True]
    def close():
        if fail[0]:
            raise OSError("RPC child still live")
    session = types.SimpleNamespace(close=close)
    engine._rpc_session = session
    engine._pipeline_distributed_only = True
    descriptor = {"assignment_layer_range": [0, 8]}
    engine._pipeline_descriptor = descriptor
    engine._model_path = "head.gguf"
    with pytest.raises(OSError, match="RPC child still live"):
        engine.unload()
    assert engine._rpc_session is session and engine._pipeline_descriptor is descriptor
    assert engine._model_path == "head.gguf" and engine.is_pipeline_prepared
    fail[0] = False
    engine.unload()
    assert engine._rpc_session is None and not engine.is_pipeline_prepared


@pytest.mark.parametrize("failure_phase", ["start", "attach"])
def test_failed_rpc_open_exposes_owned_worker_when_cleanup_fails(monkeypatch, failure_phase):
    from llama_rpc_device import RpcCleanupError
    fail = [True]
    made = []

    class Worker:
        endpoint = "127.0.0.1:18999"
        def __init__(self, *args, **kwargs):
            made.append(self)
        def start(self):
            if failure_phase == "start":
                raise RuntimeError("startup failed")
            return self
        def stop(self):
            if fail[0]:
                raise PermissionError("child still alive")
            self.closed = True

    class Injector:
        def __init__(self, **kwargs):
            pass
        def attach(self, endpoints):
            raise RuntimeError("attach failed")
        def close(self):
            pass

    monkeypatch.setattr("llama_rpc_device.LocalRpcWorker", Worker)
    monkeypatch.setattr("llama_rpc_device.RpcBackendInjector", Injector)
    with pytest.raises(RpcCleanupError) as failed:
        RpcSession.open(autostart_worker="rpc.exe")
    session = failed.value.cleanup_owner
    assert session.worker is made[0]
    fail[0] = False
    session.close()
    assert session.worker is None and made[0].closed


def test_engine_preserves_rpc_session_from_failed_open(monkeypatch):
    from llama_rpc_device import RpcCleanupError
    session = types.SimpleNamespace(close=lambda: None)
    def fail_open(**kwargs):
        raise RpcCleanupError(session, RuntimeError("open failed"), OSError("close failed"))
    monkeypatch.setattr(RpcSession, "open", staticmethod(fail_open))
    engine = LlamaCppEngine()
    with pytest.raises(RpcCleanupError):
        engine._open_rpc_session(endpoints=[], worker_exe="rpc.exe", worker_port=18999,
                                 worker_threads=1, worker_log=None)
    assert engine._rpc_session is session
    engine.unload()
    assert engine._rpc_session is None


@pytest.mark.parametrize("entry", ["load", "metadata_prepare"])
@pytest.mark.parametrize("owner_kind", ["rpc", "keep_head", "partial_binding"])
def test_reload_retires_all_native_owners_before_replacing_identity(
        fake_llama_module, monkeypatch, tmp_path, entry, owner_kind):
    engine = LlamaCppEngine()
    old_descriptor = {"assignment_layer_range": [0, 8]}
    engine._pipeline_descriptor = old_descriptor
    engine._model_path = "old-segment.gguf"
    fail = [True]
    calls = []
    def close():
        calls.append("retire")
        if fail[0]:
            raise OSError("old native owner still alive")
    owner = types.SimpleNamespace(close=close)
    field = {"rpc": "_rpc_session", "keep_head": "_keep_head", "partial_binding": "_model"}[owner_kind]
    setattr(engine, field, owner)
    assert not engine.is_loaded
    model = tmp_path / "new.gguf"
    model.write_bytes(b"GGUF stub")
    monkeypatch.setattr(engine, "get_pipeline_descriptor", lambda: {"pipeline_runtime_supported": True})
    monkeypatch.setattr(engine, "_open_rpc_session", lambda **kwargs: pytest.fail("old owner must retire first"))
    with pytest.raises(OSError, match="old native owner still alive"):
        if entry == "load":
            engine.load_model(model_path=str(model), rpc_servers="127.0.0.1:18998")
        else:
            engine.prepare_pipeline_model(model_id="new-model", model_path=str(model))
    assert getattr(engine, field) is owner
    assert engine._pipeline_descriptor is old_descriptor and engine._model_path == "old-segment.gguf"
    assert calls == ["retire"] and _FakeLlama.instances == []
    fail[0] = False
    if entry == "load":
        engine.load_model(model_path=str(model))
        assert engine.is_loaded and len(_FakeLlama.instances) == 1
        if field == "_model":
            assert engine._model is not owner
        else:
            assert getattr(engine, field) is None
    else:
        engine.prepare_pipeline_model(model_id="new-model", model_path=str(model))
        assert engine.is_pipeline_prepared and getattr(engine, field) is None
    assert calls == ["retire", "retire"] and engine._model_path == str(model)
    engine.unload()
