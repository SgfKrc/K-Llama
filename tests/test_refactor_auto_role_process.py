"""Three independent processes must converge roles, TCP ACKs and write fences."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

pytestmark = pytest.mark.refactor_contract


def test_three_process_automatic_failover_rebinds_tcp_and_fences_returned_leader(tmp_path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cluster_control_contract import VoterSet

    reservations = [socket.socket() for _ in range(6)]
    for sock in reservations:
        sock.bind(("127.0.0.1", 0))
    ports = [sock.getsockname()[1] for sock in reservations]
    keys = [Ed25519PrivateKey.generate() for _ in range(3)]
    ids = [f"voter-{i}" for i in range(3)]
    voter_set = VoterSet(cluster_id="process-failover", voter_set_epoch=1, voters={
        name: base64.urlsafe_b64encode(key.public_key().public_bytes(serialization.Encoding.Raw,
                    serialization.PublicFormat.Raw)).decode().rstrip("=") for name, key in zip(ids, keys)})
    roots = [tmp_path / name for name in ids]
    processes, logs, environments = [], [], []
    evidence = []
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(index, path="/readout", *, post=False):
        req = urllib.request.Request(f"http://127.0.0.1:{ports[index]}" + path,
                                     data=b"{}" if post else None)
        with opener.open(req, timeout=3) as response:
            return json.load(response)

    def await_cluster(predicate, timeout=30, unavailable=()):
        until = time.monotonic() + timeout
        values = []
        while time.monotonic() < until:
            assert all(p.poll() is None for i, p in enumerate(processes) if i not in unavailable), "child exited; inspect process logs"
            try:
                values = [None if i in unavailable else request(i) for i in range(3)]
            except OSError:
                time.sleep(.1)
                continue
            evidence.append(values)
            if predicate(values):
                return values
            time.sleep(.1)
        pytest.fail("cluster did not converge: " + json.dumps(values))

    try:
        for sock in reservations:
            sock.close()
        for i, root in enumerate(roots):
            root.mkdir()
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "QLH_STATE_DIR": str(root / "state"),
                   "QLH_NODE_CONFIG_PATH": str(root / "node.json"),
                   "QLH_NODE_SECRET_STORE_PATH": str(root / "secrets.json"),
                   "QLH_SQLITE_PATH": str(root / "state/control.sqlite3"),
                   "QLH_NODE_ID": ids[i], "QLH_NODE_ROLE": "auto", "QLH_RUN_MODE": "distributed",
                   "QLH_API_PORT": str(ports[i]), "QLH_SERVER_PORT": str(ports[i+3]),
                   "QLH_CLUSTER_SECRET": "isolated-auto-role-process-cluster-secret",
                   "QLH_CONTROL_VOTER_RPC_SECRET": "isolated-auto-role-process-rpc-secret",
                   "QLH_CONTROL_VOTER_SET": json.dumps(voter_set.to_dict()),
                   "QLH_CONTROL_VOTER_ID": ids[i], "QLH_CONTROL_VOTER_LEASE_MS": "2000",
                   "QLH_CONTROL_VOTER_LEDGER": str(root / "state/voter.sqlite3"),
                   "QLH_CONTROL_VOTER_PRIVATE_KEY": base64.urlsafe_b64encode(keys[i].private_bytes(
                       serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                       serialization.NoEncryption())).decode().rstrip("="),
                   "QLH_CONTROL_VOTER_PEERS": json.dumps({name: f"http://127.0.0.1:{ports[j]}"
                                                        for j, name in enumerate(ids) if j != i})}
            environments.append(env)
            log = (root / "process.log").open("w", encoding="utf-8")
            logs.append(log)
            processes.append(subprocess.Popen([sys.executable, "-B", "-X", "utf8",
                str(Path(__file__).parent / "helpers/auto_role_process.py")], env=env,
                stdout=log, stderr=subprocess.STDOUT))
        def connected(values):
            leaders = [i for i, v in enumerate(values) if v["http_write"]]
            if len(leaders) != 1:
                return False
            leader = leaders[0]
            expected_clients = {ids[i] for i in range(3) if i != leader}
            return set(values[leader]["clients"]) == expected_clients and all(
                v["binding_ready"] and v["configured_role"] == "auto"
                and (i == leader or (v["role"]["state"] == "follower" and v["tcp_registered"]))
                for i, v in enumerate(values))
        initial = await_cluster(connected)
        previous = next(i for i, v in enumerate(initial) if v["http_write"])
        old_term = initial[previous]["role"]["certificate"]["term"]
        request(previous, "/pause", post=True)
        changed = await_cluster(lambda values: any(i != previous and v["http_write"]
            and v["role"]["certificate"]["term"] > old_term for i, v in enumerate(values)))
        assert not changed[previous]["http_write"]
        request(previous, "/resume", post=True)
        recovered = await_cluster(connected)
        assert not recovered[previous]["http_write"]
        assert recovered[previous]["role"]["state"] == "follower"
        assert recovered[previous]["tcp_registered"]
        killed = next(i for i, v in enumerate(recovered) if v["http_write"])
        killed_term = recovered[killed]["role"]["certificate"]["term"]
        processes[killed].terminate()
        processes[killed].wait(10)
        await_cluster(lambda values: sum(bool(v and v["http_write"]
            and v["role"]["certificate"]["term"] > killed_term) for v in values) == 1,
            unavailable=(killed,))
        processes[killed] = subprocess.Popen([sys.executable, "-B", "-X", "utf8",
            str(Path(__file__).parent / "helpers/auto_role_process.py")], env=environments[killed],
            stdout=logs[killed], stderr=subprocess.STDOUT)
        rejoined = await_cluster(connected)
        assert rejoined[killed]["role"]["state"] == "follower"
        assert not rejoined[killed]["http_write"]
        assert rejoined[killed]["tcp_registered"]
    finally:
        (tmp_path / "observations.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        for i, process in enumerate(processes):
            if process.poll() is None:
                try:
                    request(i, "/stop", post=True)
                except OSError:
                    pass
        for process in processes:
            try:
                process.wait(15)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(10)
        for log in logs:
            log.close()
