"""Isolated real scheduler/voter process for automatic failover integration tests."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from cluster_auto_role_runtime import AUTO_ROLE_RPC_HEADER, AUTO_ROLE_RPC_PATH
    from qlh_core.adapters.settings import SettingsStore
    from qlh_core.bootstrap.runtime import AppRuntime

    root = Path(os.environ["QLH_STATE_DIR"]).parent
    runtime = AppRuntime(SettingsStore(os.environ, dotenv_path=root / "absent.env").load())
    runtime.open_identity()
    services = runtime.services
    services.store.initialize()
    voter_runtime = services.configure_authority()
    scheduler = services.scheduler
    stopping = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, value):
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/readout":
                self.send_error(404)
                return
            role = scheduler.get_auto_role_snapshot()
            certificate = scheduler._auto_role_controller.certificate
            writable = False
            try:
                services.fence.admit(certificate, action="integration.write", source="http")
                writable = True
            except ValueError:
                pass
            client = scheduler._tcp_client
            self.reply({"role": role, "bound_role": scheduler._auto_role_bound_role,
                        "binding_ready": scheduler._auto_role_binding_ready,
                        "http_write": writable, "tcp_registered": bool(client is not None and client.is_registered),
                        "clients": scheduler._tcp_server.get_client_ids() if scheduler._tcp_server else [],
                        "configured_role": scheduler._runtime_setting("NODE_ROLE")})

        def do_POST(self):
            if self.path == AUTO_ROLE_RPC_PATH:
                if not voter_runtime.authenticate(self.headers.get(AUTO_ROLE_RPC_HEADER)):
                    self.send_error(401)
                    return
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.reply(voter_runtime.handle_rpc(payload))
            elif self.path == "/pause":
                voter_runtime.stop_lease_renewal()
                self.reply({"paused": True})
            elif self.path == "/resume":
                voter_runtime.start_lease_renewal(scheduler._auto_role_controller,
                                                 on_decision=scheduler._on_auto_role_decision)
                self.reply({"resumed": True})
            elif self.path == "/stop":
                self.reply({"stopping": True})
                stopping.set()
            else:
                self.send_error(404)

    server = ThreadingHTTPServer(("127.0.0.1", int(os.environ["QLH_API_PORT"])), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        scheduler.start(host="127.0.0.1", port=int(os.environ["QLH_SERVER_PORT"]))
        print("SCHEDULER_STARTED", flush=True)
        stopping.wait(90)
    finally:
        scheduler.stop()
        server.shutdown()
        server.server_close()
        thread.join(5)
        runtime.close()


if __name__ == "__main__":
    main()
