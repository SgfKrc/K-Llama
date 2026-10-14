"""
单元测试 — 首次连接自动部署
===========================
纯逻辑测试，不启动后端、不访问网络。
"""

import json
import os
import sys

import pytest

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_tailscale_cidr_is_trusted():
    from bootstrap import is_trusted_bootstrap_source

    assert is_trusted_bootstrap_source("100.64.0.1")
    assert is_trusted_bootstrap_source("100.127.255.254")
    assert is_trusted_bootstrap_source("127.0.0.1")
    assert not is_trusted_bootstrap_source("8.8.8.8")
    assert not is_trusted_bootstrap_source("192.168.1.10")


def test_tailscale_ipv6_ula_is_trusted():
    """Tailscale IPv6 ULA（fd7a:115c:a1e0::/48）应被信任，供纯 v6 组网发现。"""
    from bootstrap import is_trusted_bootstrap_source

    assert is_trusted_bootstrap_source("fd7a:115c:a1e0:ab12::1")
    assert is_trusted_bootstrap_source("fd7a:115c:a1e0::1")
    assert not is_trusted_bootstrap_source("fe80::1")      # 链路本地不信任
    assert not is_trusted_bootstrap_source("2001:db8::1")  # 文档段不信任


def test_tailnet_join_does_not_advertise_unroutable_lan_address():
    from bootstrap import select_advertised_master_host

    assert select_advertised_master_host("100.90.1.2", "192.168.1.20") == "100.90.1.2"
    assert select_advertised_master_host(
        "fd7a:115c:a1e0::10", "192.168.1.20"
    ) == "fd7a:115c:a1e0::10"
    assert select_advertised_master_host("master.example.ts.net", "192.168.1.20") == "master.example.ts.net"
    assert select_advertised_master_host("203.0.113.10", "192.168.1.20") == "192.168.1.20"


def test_tailnet_discovery_finds_confirmed_master(monkeypatch):
    import json
    import bootstrap

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "is_master": True,
                "master_tcp_port": 18888,
                "master_api_port": 18000,
            }).encode("utf-8")

    monkeypatch.setattr(bootstrap, "get_tailnet_peer_ips", lambda: ["100.90.1.2"])
    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", lambda url, timeout: Response())

    result = bootstrap.discover_master_via_tailnet(api_port=18000)

    assert result == {
        "found": True,
        "master_host": "100.90.1.2",
        "master_port": 18888,
        "master_api_port": 18000,
        "stale": False,
        "source": "tailnet",
    }


def test_normalize_node_id_rejects_master():
    from bootstrap import normalize_node_id

    assert normalize_node_id("master", "pc").startswith("client_")
    assert normalize_node_id("android phone/1", "android") == "android_phone_1"


def test_tailnet_ipv6_discovery_uses_brackets_only_in_url(monkeypatch):
    import json
    import bootstrap

    seen = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"is_master": True}).encode("utf-8")

    monkeypatch.setattr(
        bootstrap, "get_tailnet_peer_ips", lambda: ["fd7a:115c:a1e0::10"]
    )

    def fake_urlopen(url, timeout):
        seen.append(str(url))
        return Response()

    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", fake_urlopen)

    result = bootstrap.discover_master_via_tailnet(api_port=8000)

    assert seen == ["http://[fd7a:115c:a1e0::10]:8000/api/bootstrap/info"]
    assert result["master_host"] == "fd7a:115c:a1e0::10"


def test_first_connect_builds_valid_ipv6_url(monkeypatch):
    import json
    import bootstrap
    from bootstrap_credentials import seal_bootstrap_credential

    seen = []

    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self.body).encode("utf-8")

    def fake_urlopen(request, timeout):
        seen.append(request.full_url)
        payload = json.loads(request.data.decode("utf-8"))
        envelope = seal_bootstrap_credential(
            public_key=payload["credential_public_key"],
            request_nonce=payload["credential_request_nonce"],
            requested_at=payload["credential_requested_at"],
            cluster_id="cluster-v6",
            node_id="client-v6",
            cluster_secret="secret-v6-credential-value",
            secret_epoch=2,
            now=payload["credential_requested_at"],
        )
        return Response({
            "status": "ok",
            "cluster": {
                "cluster_id": "cluster-v6",
                "master_tcp_host": "fd7a:115c:a1e0::10",
                "master_tcp_port": 8888,
                "cluster_secret_epoch": 2,
            },
            "node": {"node_id": "client-v6", "role": "client"},
            "credential_envelope": envelope,
        })

    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(bootstrap, "persist_bootstrap_response", lambda data: None)
    monkeypatch.setattr(bootstrap, "apply_runtime_config", lambda data: None)

    bootstrap.first_connect("fd7a:115c:a1e0::10", 8000, node_id="client-v6")

    assert seen == [
        "http://[fd7a:115c:a1e0::10]:8000/api/bootstrap/first-connect"
    ]


def test_persist_bootstrap_response_writes_node_config(tmp_path, monkeypatch):
    monkeypatch.setenv("QLH_NODE_CONFIG_PATH", str(tmp_path / "node_config.json"))
    monkeypatch.setenv("QLH_CLUSTER_SECRET", "stale-local-secret")
    monkeypatch.delenv("QLH_MASTER_HOST", raising=False)
    monkeypatch.delenv("QLH_MASTER_PORT", raising=False)

    from node_config import load_node_config, persist_bootstrap_response

    response = {
        "status": "ok",
        "cluster": {
            "cluster_id": "test-cluster",
            "master_api_host": "100.64.0.10",
            "master_api_port": 8000,
            "master_tcp_host": "100.64.0.10",
            "master_tcp_port": 8888,
            "cluster_secret": "secret-123",
            "cluster_secret_epoch": 3,
        },
        "node": {
            "node_id": "client-test",
            "role": "client",
            "node_type": "pc",
            "pipeline_worker": True,
        },
    }

    path = persist_bootstrap_response(response)
    assert path.is_file()
    data = load_node_config()
    assert data["bootstrapped"] is True
    assert "cluster_secret" not in data["cluster"]
    assert data["cluster"]["cluster_secret_ref"] == "local-secret-store:v1"
    assert data["cluster"]["cluster_secret_epoch"] == 3
    assert "secret-123" not in path.read_text(encoding="utf-8")
    assert data["cluster"]["master_tcp_host"] == "100.64.0.10"
    assert data["node"]["node_id"] == "client-test"
    assert os.environ["QLH_CLUSTER_SECRET"] == "secret-123"
    assert os.environ["QLH_CLUSTER_SECRET_EPOCH"] == "3"
    assert os.environ["QLH_MASTER_HOST"] == "100.64.0.10"
    assert os.environ["QLH_MASTER_PORT"] == "8888"
    assert os.environ["QLH_MASTER_API_PORT"] == "8000"


def test_frozen_windows_config_uses_local_app_data(tmp_path, monkeypatch):
    from node_config import get_node_config_path

    install_dir = tmp_path / "Program Files" / "QLH-Edge-Inference"
    executable = install_dir / "QLH-Edge-Inference.exe"
    local_app_data = tmp_path / "LocalAppData"
    monkeypatch.delenv("QLH_NODE_CONFIG_PATH", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "executable", str(executable))

    assert get_node_config_path() == (
        local_app_data / "QLH-Edge-Inference" / "node_config.json"
    )


def test_source_checkout_uses_user_config_directory(tmp_path, monkeypatch):
    from node_config import get_node_config_path

    monkeypatch.delenv("QLH_NODE_CONFIG_PATH", raising=False)
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))

    assert get_node_config_path() == (
        tmp_path / "LocalAppData" / "QLH-Edge-Inference" / "node_config.json"
    )


def test_unconfigured_source_checkout_defaults_to_client(tmp_path, monkeypatch):
    import node_config

    monkeypatch.delenv("QLH_NODE_CONFIG_PATH", raising=False)
    monkeypatch.delenv("QLH_NODE_ROLE", raising=False)
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))

    assert node_config.resolve_initial_node_role() == "client"


def test_source_checkout_requires_explicit_master_role(tmp_path, monkeypatch):
    import node_config

    monkeypatch.delenv("QLH_NODE_CONFIG_PATH", raising=False)
    monkeypatch.setenv("QLH_NODE_ROLE", "master")
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))

    assert node_config.resolve_initial_node_role() == "master"


def test_source_checkout_rejects_unknown_role_explicitly(tmp_path, monkeypatch):
    import node_config

    monkeypatch.setenv("QLH_NODE_CONFIG_PATH", str(tmp_path / "node_config.json"))
    monkeypatch.setenv("QLH_NODE_ROLE", "typo-role")
    monkeypatch.setattr(sys, "frozen", False, raising=False)

    with pytest.raises(ValueError, match="invalid node role"):
        node_config.resolve_initial_node_role()


def test_frozen_config_migrates_legacy_exe_directory_file(tmp_path, monkeypatch):
    from node_config import get_local_cluster_secret, get_node_config_path, load_node_config

    install_dir = tmp_path / "Program Files" / "QLH-Edge-Inference"
    install_dir.mkdir(parents=True)
    executable = install_dir / "QLH-Edge-Inference.exe"
    legacy_path = install_dir / "node_config.json"
    legacy_path.write_text(
        json.dumps({
            "bootstrapped": True,
            "cluster": {"cluster_secret": "legacy-install-secret"},
            "node": {"role": "client"},
        }),
        encoding="utf-8",
    )
    local_app_data = tmp_path / "LocalAppData"
    monkeypatch.delenv("QLH_NODE_CONFIG_PATH", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.delenv("QLH_CLUSTER_SECRET", raising=False)

    data = load_node_config()

    assert data["node"]["role"] == "client"
    assert get_node_config_path().is_file()
    assert not legacy_path.exists()
    assert get_local_cluster_secret() == "legacy-install-secret"
    assert "legacy-install-secret" not in get_node_config_path().read_text(encoding="utf-8")


def test_frozen_legacy_config_is_scrubbed_when_delete_is_denied(
    tmp_path, monkeypatch,
):
    import node_config

    install_dir = tmp_path / "Program Files" / "QLH-Edge-Inference"
    install_dir.mkdir(parents=True)
    executable = install_dir / "QLH-Edge-Inference.exe"
    legacy_path = install_dir / "node_config.json"
    legacy_path.write_text(json.dumps({
        "cluster": {"cluster_secret": "legacy-locked-secret"},
        "node": {"role": "client", "role_confirmed": True},
    }), encoding="utf-8")
    monkeypatch.delenv("QLH_NODE_CONFIG_PATH", raising=False)
    monkeypatch.delenv("QLH_CLUSTER_SECRET", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "executable", str(executable))
    original_unlink = node_config.Path.unlink

    def deny_legacy_unlink(path, *args, **kwargs):
        if path == legacy_path:
            raise PermissionError("legacy config is locked")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(node_config.Path, "unlink", deny_legacy_unlink)

    data = node_config.load_node_config()

    assert data["node"]["role"] == "client"
    assert legacy_path.is_file()
    assert "legacy-locked-secret" not in legacy_path.read_text(encoding="utf-8")
    assert node_config.get_local_cluster_secret() == "legacy-locked-secret"


def test_legacy_plaintext_cluster_secret_migrates_out_of_node_config(
    tmp_path, monkeypatch,
):
    import node_config

    config_path = tmp_path / "node_config.json"
    monkeypatch.setenv("QLH_NODE_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("QLH_NODE_SECRET_STORE_PATH", str(tmp_path / "node_secrets.json"))
    monkeypatch.delenv("QLH_CLUSTER_SECRET", raising=False)
    config_path.write_text(json.dumps({
        "cluster": {"cluster_id": "legacy", "cluster_secret": "legacy-secret-value"},
        "node": {"role": "client", "role_confirmed": True},
    }), encoding="utf-8")

    data = node_config.load_node_config()

    assert "cluster_secret" not in data["cluster"]
    assert data["cluster"]["cluster_secret_ref"] == "local-secret-store:v1"
    assert node_config.get_local_cluster_secret() == "legacy-secret-value"
    assert "legacy-secret-value" not in config_path.read_text(encoding="utf-8")
    assert "legacy-secret-value" not in (
        tmp_path / "node_secrets.json"
    ).read_text(encoding="utf-8")


def test_stale_legacy_plaintext_cannot_roll_back_rotated_credential(
    tmp_path, monkeypatch,
):
    import node_config

    install_dir = tmp_path / "Program Files" / "QLH-Edge-Inference"
    install_dir.mkdir(parents=True)
    executable = install_dir / "QLH-Edge-Inference.exe"
    legacy_path = install_dir / "node_config.json"
    local_app_data = tmp_path / "LocalAppData"
    monkeypatch.delenv("QLH_NODE_CONFIG_PATH", raising=False)
    monkeypatch.setenv(
        "QLH_NODE_SECRET_STORE_PATH", str(local_app_data / "node_secrets.json"),
    )
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setenv("QLH_CLUSTER_SECRET", "initial-secret")
    monkeypatch.setenv("QLH_CLUSTER_SECRET_EPOCH", "1")
    node_config.ensure_local_cluster_secret()
    rotated_secret, rotated_epoch = node_config.rotate_local_cluster_secret()
    node_config.get_node_config_path().unlink()
    legacy_path.write_text(json.dumps({
        "cluster": {
            "cluster_secret": "stale-install-secret",
            "cluster_secret_epoch": 1,
        },
        "node": {"role": "client", "role_confirmed": True},
    }), encoding="utf-8")

    node_config.load_node_config()

    assert node_config.get_local_cluster_secret() == rotated_secret
    assert node_config.get_local_cluster_secret_epoch() == rotated_epoch


def test_primary_config_migration_failure_does_not_fallback_to_legacy(
    tmp_path, monkeypatch,
):
    import node_config

    primary = tmp_path / "user" / "node_config.json"
    primary.parent.mkdir(parents=True)
    primary.write_text(json.dumps({
        "cluster": {"cluster_secret": "primary-secret"},
        "node": {"role": "master", "role_confirmed": True},
    }), encoding="utf-8")
    install_dir = tmp_path / "install"
    install_dir.mkdir()
    executable = install_dir / "qlh.exe"
    (install_dir / "node_config.json").write_text(json.dumps({
        "cluster": {"cluster_secret": "legacy-secret"},
        "node": {"role": "client", "role_confirmed": True},
    }), encoding="utf-8")
    monkeypatch.setenv("QLH_NODE_CONFIG_PATH", str(primary))
    monkeypatch.setenv("QLH_NODE_SECRET_STORE_PATH", str(tmp_path / "secrets.json"))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setattr(
        node_config,
        "write_node_config",
        lambda _data: (_ for _ in ()).throw(OSError("migration denied")),
    )

    with pytest.raises(OSError, match="migration denied"):
        node_config.load_node_config()


def test_cluster_secret_rotation_advances_epoch_and_invalidates_old_root(
    tmp_path, monkeypatch,
):
    import node_config

    monkeypatch.setenv("QLH_NODE_CONFIG_PATH", str(tmp_path / "node_config.json"))
    monkeypatch.setenv("QLH_NODE_SECRET_STORE_PATH", str(tmp_path / "node_secrets.json"))
    monkeypatch.delenv("QLH_CLUSTER_SECRET", raising=False)
    monkeypatch.delenv("QLH_CLUSTER_SECRET_EPOCH", raising=False)
    old = node_config.ensure_local_cluster_secret()
    old_epoch = node_config.get_local_cluster_secret_epoch()

    new, new_epoch = node_config.rotate_local_cluster_secret()

    assert new != old
    assert new_epoch == old_epoch + 1
    assert node_config.get_local_cluster_secret() == new
    assert node_config.get_local_cluster_secret_epoch() == new_epoch


def test_cluster_secret_rotation_overrides_stale_environment_on_restart(
    tmp_path, monkeypatch,
):
    import node_config

    monkeypatch.setenv("QLH_NODE_CONFIG_PATH", str(tmp_path / "node_config.json"))
    monkeypatch.setenv("QLH_NODE_SECRET_STORE_PATH", str(tmp_path / "node_secrets.json"))
    monkeypatch.setenv("QLH_CLUSTER_SECRET", "legacy-dotenv-secret")
    monkeypatch.setenv("QLH_CLUSTER_SECRET_EPOCH", "1")
    old_secret = node_config.ensure_local_cluster_secret()

    new_secret, new_epoch = node_config.rotate_local_cluster_secret()
    assert new_secret != old_secret

    # Simulate python-dotenv restoring the old deployment values before the
    # normal startup call to apply_node_config_to_env().
    monkeypatch.setenv("QLH_CLUSTER_SECRET", "legacy-dotenv-secret")
    monkeypatch.setenv("QLH_CLUSTER_SECRET_EPOCH", "1")
    node_config.apply_node_config_to_env()

    assert os.environ["QLH_CLUSTER_SECRET"] == new_secret
    assert os.environ["QLH_CLUSTER_SECRET_EPOCH"] == str(new_epoch)
    assert node_config.get_local_cluster_secret() == new_secret
    assert node_config.get_local_cluster_secret_epoch() == new_epoch


def test_cluster_secret_rotation_restores_old_secret_when_config_write_fails(
    tmp_path, monkeypatch,
):
    import node_config

    monkeypatch.setenv("QLH_NODE_CONFIG_PATH", str(tmp_path / "node_config.json"))
    monkeypatch.setenv("QLH_NODE_SECRET_STORE_PATH", str(tmp_path / "node_secrets.json"))
    monkeypatch.delenv("QLH_CLUSTER_SECRET", raising=False)
    monkeypatch.delenv("QLH_CLUSTER_SECRET_EPOCH", raising=False)
    old_secret = node_config.ensure_local_cluster_secret()
    old_epoch = node_config.get_local_cluster_secret_epoch()
    monkeypatch.setattr(
        node_config,
        "write_node_config",
        lambda _data: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        node_config.rotate_local_cluster_secret()

    monkeypatch.delenv("QLH_CLUSTER_SECRET", raising=False)
    assert node_config.get_local_cluster_secret() == old_secret
    assert node_config.get_local_cluster_secret_epoch() == old_epoch


def test_bootstrap_api_port_ignores_local_api_port(monkeypatch):
    monkeypatch.delenv("QLH_BOOTSTRAP_API_PORT", raising=False)
    monkeypatch.delenv("QLH_MASTER_API_PORT", raising=False)
    monkeypatch.setenv("QLH_API_PORT", "8001")

    from scheduler import _bootstrap_api_port

    assert _bootstrap_api_port() == 8000

    monkeypatch.setenv("QLH_MASTER_API_PORT", "18000")
    assert _bootstrap_api_port() == 18000

    monkeypatch.setenv("QLH_BOOTSTRAP_API_PORT", "18001")
    assert _bootstrap_api_port() == 18001


def test_apply_runtime_config_syncs_loaded_scheduler(monkeypatch, tmp_path):
    import config as cfg
    import scheduler as scheduler_mod
    from node_config import apply_runtime_config

    monkeypatch.setattr(cfg, "NODE_ID", "old-client", raising=False)
    monkeypatch.setattr(cfg, "NODE_ROLE", "master", raising=False)
    monkeypatch.setattr(scheduler_mod, "NODE_ID", "stale-client", raising=False)
    monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "master", raising=False)
    monkeypatch.setenv("QLH_NODE_CONFIG_PATH", str(tmp_path / "node_config.json"))
    monkeypatch.setenv("QLH_NODE_SECRET_STORE_PATH", str(tmp_path / "node_secrets.json"))
    monkeypatch.setenv("QLH_CLUSTER_SECRET", "stale-runtime-secret")
    monkeypatch.setenv("QLH_NODE_ROLE", "master")

    apply_runtime_config({
        "cluster": {
            "cluster_secret": "secret-456",
            "master_tcp_host": "100.64.0.20",
            "master_tcp_port": 8889,
        },
        "node": {
            "node_id": "client-runtime",
            "role": "client",
        },
    })

    # 阶段 0.3：运行时身份写入 node_runtime（cfg 不再被写回）
    import node_runtime as node_runtime_mod
    assert node_runtime_mod.node_runtime.get_node_id() == "client-runtime"
    assert node_runtime_mod.node_runtime.get_node_role() == "client"
    assert scheduler_mod.NODE_ID == "client-runtime"
    assert scheduler_mod.NODE_ROLE == "client"
    assert os.environ["QLH_CLUSTER_SECRET"] == "secret-456"
    assert os.environ["QLH_NODE_ROLE"] == "client"


def test_tailnet_peer_ips_includes_ipv6(monkeypatch, tmp_path):
    """在线 Tailnet 对等点的 IPv6 ULA 地址应被纳入发现候选（问题 #5 回归）。"""
    import json
    from types import SimpleNamespace
    import bootstrap

    fake_exe = tmp_path / "tailscale.exe"
    fake_exe.write_bytes(b"")
    monkeypatch.setattr(bootstrap, "_find_tailscale_executable", lambda: str(fake_exe))
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0,
        stdout=json.dumps({"Peer": {
            "peer1": {"Online": True, "TailscaleIPs": ["100.90.1.2", "fd7a:115c:a1e0:ab12::1"]},
            "peer2": {"Online": True, "TailscaleIPs": ["fd7a:115c:a1e0:ab12::2"]},
            "peer3": {"Online": False, "TailscaleIPs": ["100.90.1.3"]},
        }}),
    ))

    peers = bootstrap.get_tailnet_peer_ips()

    assert "100.90.1.2" in peers
    assert "fd7a:115c:a1e0:ab12::1" in peers
    assert "fd7a:115c:a1e0:ab12::2" in peers
    assert "100.90.1.3" not in peers  # 离线对等点排除
