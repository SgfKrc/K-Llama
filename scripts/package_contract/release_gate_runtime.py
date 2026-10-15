"""Bounded native processes and isolated runtime readiness for release gates."""
from __future__ import annotations

import ctypes
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


class Report:
    def __init__(self, platform: str) -> None:
        self.platform = platform
        self.items: list[dict[str, Any]] = []

    def add(self, name: str, status: str, detail: str = "", **evidence: Any) -> None:
        if status not in {"passed", "failed", "unverified"}:
            raise ValueError(status)
        self.items.append(dict(check=name, status=status, passed=status == "passed",
                               detail=detail, evidence=evidence))

    def ok(self, name: str, detail: str = "", **evidence: Any) -> None:
        self.add(name, "passed", detail, **evidence)

    def fail(self, name: str, detail: str, **evidence: Any) -> None:
        self.add(name, "failed", detail, **evidence)

    def unverified(self, name: str, detail: str, **evidence: Any) -> None:
        self.add(name, "unverified", detail, **evidence)

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [item for item in self.items if item["status"] != "passed"]

    def as_dict(self) -> dict[str, Any]:
        status = ("failed" if any(i["status"] == "failed" for i in self.items)
                  else "unverified" if self.failures or not self.items else "passed")
        revisions = {i["evidence"]["source_revision"] for i in self.items
                     if i["evidence"].get("revision_verified") is True and i["evidence"].get("source_revision")}
        if len(revisions) > 1:
            status = "failed"
        identity_checks = [i for i in self.items if i["check"].endswith(".revision")]
        all_identities = bool(identity_checks) and all(i["passed"] and i["evidence"].get("revision_verified") is True
                                                     for i in identity_checks)
        revision = next(iter(revisions)) if len(revisions) == 1 and all_identities else None
        return dict(platform=self.platform, status=status, passed=status == "passed", items=self.items,
                    source_revision=revision, revision_verified=revision is not None,
                    source_revision_verified=revision is not None)


def isolated_environment(work: Path) -> dict[str, str]:
    """Never inherit user QLH state, Python paths, runtime or release overrides."""
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("QLH_", "PYTHON"))}
    for key, relative in {
        "HOME": "home", "USERPROFILE": "home", "LOCALAPPDATA": "local",
        "APPDATA": "roaming", "XDG_CONFIG_HOME": "config", "XDG_DATA_HOME": "data",
        "XDG_STATE_HOME": "state", "TEMP": "tmp", "TMP": "tmp", "TMPDIR": "tmp",
        "QLH_STATE_DIR": "state", "QLH_DATA_DIR": "data", "QLH_RUNTIME_DIR": "runtime",
        "QLH_KV_COLD_CACHE_ROOT": "state/kv-cold",
    }.items():
        path = work / relative
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    for key, relative in {
        "QLH_NODE_CONFIG_PATH": "config/node_config.json",
        "QLH_NODE_SECRET_STORE_PATH": "config/node_secrets.json",
        "QLH_SQLITE_PATH": "state/local-store.sqlite3",
        "QLH_JOURNAL_REPLICATION_REPLICA_PATH": "state/journal-replica.sqlite3",
    }.items():
        env[key] = str(work / relative)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTHONUTF8="1",
               PIP_NO_INDEX="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


class OwnedProcess:
    """A private process group / Windows job; cleanup cannot target another app."""
    def __init__(self, command: list[str], *, cwd: Path, env: dict[str, str], log: Path):
        self.job = None
        log.parent.mkdir(parents=True, exist_ok=True)
        self.output = log.open("wb")
        self.log = log
        self.command = command
        self.started_at = time.time()
        options: dict[str, Any] = dict(cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                                      stdout=self.output, stderr=subprocess.STDOUT)
        if os.name == "nt":
            # Suspend until the private kill-on-close job owns the process and all children.
            from ctypes import wintypes as w
            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.CreateJobObjectW.argtypes = (ctypes.c_void_p, w.LPCWSTR)
            self.kernel.CreateJobObjectW.restype = w.HANDLE
            self.kernel.SetInformationJobObject.argtypes = (w.HANDLE, ctypes.c_int,
                                                           ctypes.c_void_p, w.DWORD)
            self.kernel.AssignProcessToJobObject.argtypes = (w.HANDLE, w.HANDLE)
            self.kernel.IsProcessInJob.argtypes = (w.HANDLE, w.HANDLE, ctypes.POINTER(w.BOOL))
            self.kernel.OpenProcess.argtypes = (w.DWORD, w.BOOL, w.DWORD)
            self.kernel.OpenProcess.restype = w.HANDLE
            self.kernel.CloseHandle.argtypes = (w.HANDLE,)

            class Basic(ctypes.Structure):
                _fields_ = [("time1", ctypes.c_int64), ("time2", ctypes.c_int64),
                            ("flags", w.DWORD), ("min", ctypes.c_size_t), ("max", ctypes.c_size_t),
                            ("active", w.DWORD), ("affinity", ctypes.c_size_t),
                            ("priority", w.DWORD), ("scheduling", w.DWORD)]

            class Extended(ctypes.Structure):
                _fields_ = [("basic", Basic), ("io", ctypes.c_uint64 * 6),
                            ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                            ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

            self.job = self.kernel.CreateJobObjectW(None, None)
            limits = Extended()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not self.job or not self.kernel.SetInformationJobObject(
                    self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                self.close()
                raise OSError("cannot create private Windows process job")
            options["creationflags"] = subprocess.CREATE_NO_WINDOW | 0x4  # CREATE_SUSPENDED
        else:
            options["start_new_session"] = True
        try:
            self.process = subprocess.Popen(command, **options)
            if self.job:
                if not self.kernel.AssignProcessToJobObject(self.job, int(self.process._handle)):
                    self.process.kill()
                    raise OSError("cannot assign suspended child to private Windows job")
                resume = ctypes.WinDLL("ntdll").NtResumeProcess
                resume.argtypes = (ctypes.c_void_p,)
                resume.restype = ctypes.c_long
                if resume(int(self.process._handle)) != 0:
                    raise OSError("cannot resume private Windows process")
        except BaseException:
            self.close()
            raise

    def owns(self, pid: int) -> bool:
        if self.job:
            from ctypes import wintypes as w
            handle = self.kernel.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            inside = w.BOOL()
            try:
                return bool(self.kernel.IsProcessInJob(handle, self.job, ctypes.byref(inside))
                            and inside.value)
            finally:
                self.kernel.CloseHandle(handle)
        try:
            return os.getpgid(pid) == self.process.pid
        except ProcessLookupError:
            return False

    def close(self) -> None:
        if self.job:
            self.kernel.CloseHandle(self.job)
            self.job = None
        elif hasattr(self, "process") and os.name != "nt":
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if hasattr(self, "process"):
            self.process.wait(timeout=15)
        self.output.close()


def read_tail(path: Path, limit: int = 2_000_000) -> str:
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode("utf-8", errors="replace")


def run_command(command: list[str], *, cwd: Path, env: dict[str, str], log: Path,
                timeout: float = 120) -> dict[str, Any]:
    evidence: dict[str, Any] = dict(command=command, log=str(log), executed=False)
    try:
        child = OwnedProcess(command, cwd=cwd, env=env, log=log)
        evidence.update(executed=True, pid=child.process.pid, started_at_unix=child.started_at)
        try:
            evidence["returncode"] = child.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            evidence.update(returncode=None, error="timeout; no readiness inferred")
        finally:
            child.close()
        evidence["output"] = read_tail(log)
    except (OSError, subprocess.SubprocessError) as exc:
        evidence.update(returncode=None, error=str(exc))
    evidence["ok"] = evidence.get("executed") is True and evidence.get("returncode") == 0
    return evidence


def json_output(result: dict[str, Any]) -> dict[str, Any]:
    if not result.get("ok"):
        raise ValueError(result.get("error") or f"command exit={result.get('returncode')}")
    try:
        value = json.loads(result["output"])
    except (KeyError, ValueError) as exc:
        raise ValueError("command did not return one JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("command JSON must be an object")
    return value


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def listener_pids(port: int) -> set[int]:
    if os.name == "nt":
        from ctypes import wintypes as w
        size = w.DWORD(0)
        api = ctypes.WinDLL("iphlpapi").GetExtendedTcpTable
        api.argtypes = (ctypes.c_void_p, ctypes.POINTER(w.DWORD), w.BOOL,
                        w.ULONG, w.ULONG, w.ULONG)
        api(None, ctypes.byref(size), False, 2, 3, 0)  # IPv4, OWNER_PID_LISTENER
        data = ctypes.create_string_buffer(size.value)
        if api(data, ctypes.byref(size), False, 2, 3, 0) != 0:
            return set()
        count = ctypes.c_uint32.from_buffer(data).value
        owners = set()
        for index in range(count):
            row = (ctypes.c_uint32 * 6).from_buffer(data, 4 + index * 24)
            if socket.ntohs(row[2] & 0xFFFF) == port:
                owners.add(int(row[5]))
        return owners
    inodes = set()
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        if table.is_file():
            for line in table.read_text().splitlines()[1:]:
                fields = line.split()
                if fields[3] == "0A" and int(fields[1].split(":")[1], 16) == port:
                    inodes.add(f"socket:[{fields[9]}]")
    owners = set()
    for process in Path("/proc").iterdir():
        if process.name.isdigit():
            try:
                if any(os.readlink(fd) in inodes for fd in (process / "fd").iterdir()):
                    owners.add(int(process.name))
            except (PermissionError, FileNotFoundError):
                continue
    return owners


def http_json(port: int, route: str) -> dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{port}{route}", timeout=2) as response:
        if response.status != 200:
            raise ValueError(f"HTTP {response.status}")
        value = json.loads(response.read(1_000_000))
    if not isinstance(value, dict):
        raise ValueError("HTTP response must be an object")
    return value


def readiness_matches(health: dict, ready: dict, status: dict, role: str) -> bool:
    return (health.get("status") == "ok" and ready.get("process_ready") is True
            and ready.get("ready") is True and ready.get("status") == "ready"
            and ready.get("error") is None
            and all(ready.get("components", {}).get(key) is True
                    for key in ("local_store", "scheduler", "device_profile"))
            and status.get("node_role") == role)


def probe_service(command: list[str], *, app: Path, work: Path, role: str,
                  timeout: float = 90, confirmed_worker: bool = False,
                  target_app: Path | None = None) -> dict[str, Any]:
    env = isolated_environment(work)
    api_port, tcp_port = free_port(), free_port()
    while api_port == tcp_port:
        tcp_port = free_port()
    env.update(QLH_API_PORT=str(api_port), QLH_SERVER_PORT=str(tcp_port),
               QLH_MASTER_PORT=str(tcp_port), QLH_CLIENT_MASTER_PORT=str(tcp_port),
               QLH_MASTER_IP="127.0.0.1", QLH_CLIENT_MASTER_IP="127.0.0.1",
               QLH_APP_HOME=str(target_app or app), QLH_CORE_ROOT=str(target_app or app), QLH_LAUNCHER_ACTIVE_SLOT="1")
    # The default role must be resolved by the artifact, never injected from the contract.
    if confirmed_worker:
        Path(env["QLH_NODE_CONFIG_PATH"]).write_text(json.dumps({
            "bootstrapped": True, "node": {"role": "worker", "role_confirmed": True},
        }), encoding="utf-8")
    log = work / "startup.log"
    result: dict[str, Any] = dict(command=command, log=str(log), executed=False,
                                  expected_role=role, api_port=api_port, ready=False)
    try:
        child = OwnedProcess(command, cwd=app, env=env, log=log)
        result.update(executed=True, pid=child.process.pid, started_at_unix=child.started_at)
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                # Launcher may exit after creating its backend, which remains in our job.
                owners = listener_pids(api_port)
                if owners and all(child.owns(pid) for pid in owners):
                    try:
                        health = http_json(api_port, "/api/health")
                        ready = http_json(api_port, "/api/ready")
                        status = http_json(api_port, "/api/status")
                        result.update(health=health, readiness=ready,
                                      observed_role=status.get("node_role"), listener_pids=sorted(owners))
                        started = ready.get("started_at")
                        current_owners = listener_pids(api_port)
                        if (current_owners and isinstance(started, (float, int)) and not isinstance(started, bool)
                                and started >= child.started_at - 1
                                and readiness_matches(health, ready, status, role)
                                and all(child.owns(pid) for pid in current_owners)):
                            result["ready"] = True
                            break
                    except (OSError, ValueError):
                        pass
                elif owners:
                    result["error"] = "API port belongs to another process; refused to reuse it"
                    break
                if child.process.poll() is not None and not owners:
                    result["error"] = f"entrypoint exited without owned ready backend: {child.process.returncode}"
                time.sleep(0.2)
            if not result["ready"]:
                result.setdefault("error", "readiness/role was not verified before timeout")
        finally:
            child.close()
        result["output_tail"] = read_tail(log, 4000)
    except (OSError, subprocess.SubprocessError) as exc:
        result["error"] = str(exc)
    return result
