"""Synthetic OS/native observer fixtures for offline product gate tests only.

These fixtures do not run a model or represent physical acceptance evidence.
"""
from __future__ import annotations

import hashlib
import json


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def node_observation(identity, assignment, assignments, revision, total_layers,
                     *, captured_at=106.0, artifact_sha256="b" * 64):
    ids = [item["node_id"] for item in assignments]
    node = assignment["node_id"]
    index = ids.index(node)
    platform = "master" if node == "master" else "android" if "android" in node else "pc"
    addresses = {value: [f"10.91.0.{i+1}"] for i, value in enumerate(ids)}
    peers = ids[1:] if node == "master" else ["master"]
    pid, created = 101 + index, 10.0
    proc_identity = {"pid": pid, "created_at": created}
    cwd = "/data/user/0/com.qlh.inference.debug" if platform == "android" else "/opt/qlh"
    span = [assignment["start_layer"], assignment["end_layer"]]
    artifact_path = cwd + "/segments/segment.gguf"
    start, end = captured_at - 3.0, captured_at + 0.5
    runtime = {**identity, **proc_identity, "captured_at": captured_at,
               "engine": assignment["engine"], "layer_range": span, "total_layers": total_layers,
               "loaded": True, "full_model_loaded": False, "loader_pid": pid,
               "native_layer_count": span[1]-span[0], "package_name": "com.qlh.inference.debug",
               "artifact": {"path": artifact_path, "sha256": artifact_sha256,
                            "rehashed_sha256": artifact_sha256, "layer_range": span,
                            "decoder_layers": span[1]-span[0]}}
    mappings = [artifact_path]
    if assignment["engine"] == "pytorch":
        artifact_path = cwd + "/segments/segment.safetensors"
        runtime["artifact"]["path"] = artifact_path
        mappings = []
        names = [f"model.layers.{i}.weight" for i in range(*span)]
        runtime["materialization"] = {
            **identity, **proc_identity, "source": "safetensors_key_filtered",
            "artifact_sha256": artifact_sha256, "layer_range": span,
            "decoder_prefix": "model.layers.", "loaded_tensor_names": names,
            "tensor_metadata": {name: {"shape": [2, 2], "element_size": 2,
                                        "source_file": artifact_path} for name in names},
            "materialized_bytes": 8 * len(names)}
    process = {**proc_identity, "parent_pid": 1, "identity_after": proc_identity,
               "command": ["synthetic-observer-fixture"], "cwd": cwd,
               "children_complete": True, "child_pids": [], "maps_complete": True,
               "model_mappings": mappings, "connections_complete": True,
               "connections": [{"status": "ESTABLISHED", "local": [addresses[node][0], 8888 if node == "master" else 41000+i],
                                "remote": [addresses[peer][0], 41000+i if node == "master" else 8888]}
                               for i, peer in enumerate(peers)]}
    if platform == "android":
        info = {"schema_version": 1, "source_revision": revision, "source_dirty": False}
        path = "/data/app/com.qlh.inference.debug-fixture/base.apk"
        source = {**proc_identity, "kind": "installed_apk", "extraction": "apk_zip",
                  "package": "com.qlh.inference.debug", "path_before": path, "path_after": path,
                  "sha256_before": "e"*64, "sha256_after": "e"*64, "pulled_sha256": "e"*64,
                  "build_info_entry": "assets/build-info.json", "build_info": info,
                  "build_info_sha256": canonical_hash(info)}
    else:
        manifest = {"source_revision": revision, "source_dirty": False,
                    "files": [{"path": "src/api_server.py", "sha256": "d"*64}]}
        source = {**proc_identity, "kind": "runtime_manifest" if platform == "master" else "synced_manifest",
                  "source_root": cwd, "source_revision": revision, "source_dirty": False,
                  "manifest": manifest, "manifest_sha256": canonical_hash(manifest),
                  "runtime_manifest_sha256": canonical_hash(manifest), "rehashed_files": manifest["files"]}
    facts = {"identity_before": proc_identity, "identity_after": proc_identity,
             "runtime": runtime, "source": source,
             "process_tree": {"complete": True, "processes": [process]}}
    raw = {"schema_version": 1, "kind": "product-node-observation", "node_id": node,
           "platform": platform, "started_at": start, "finished_at": end,
           "commands": [{"command": ["synthetic-observer-fixture"], "returncode": 0,
                         "started_at": start, "finished_at": end,
                         "stdout": json.dumps(facts), "stderr": ""}], "facts": facts}
    common = {"node_id": node, **identity, "captured_at": captured_at,
              "observation_started_at": start, "observation_finished_at": end, "complete": True}
    return raw, {**common, "worker_peers": sorted(peers)}, {
        **common, "source_revision": revision, "source_dirty": False,
        "source": "native_loader_receipt", "engine": assignment["engine"],
        "layer_range": span, "full_model_loaded": False}, addresses
