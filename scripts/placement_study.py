#!/usr/bin/env python3
"""Replay an offline placement study or emit exact-range capture identities."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qlh_core.planning.capacity import PipelineCapacityError
from qlh_core.planning.placement_study import (
    PlacementStudyError, PROFILE_SCHEMA, SCOPE, _context,
    enumerate_capacity_placements, evaluate_placement_study,
)


def _read(path):
    if path.stat().st_size > 8 * 1024 * 1024:
        raise PlacementStudyError("input exceeds the 8 MiB JSON limit")
    raw = path.read_bytes()
    if len(raw) > 8 * 1024 * 1024:
        raise PlacementStudyError("input exceeds the 8 MiB JSON limit")
    return raw, json.loads(raw.decode("utf-8-sig"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--profile", type=Path)
    mode.add_argument("--template", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = {"ok": False, "schema": "qlh.placement_study_cli.v1"}
    try:
        if args.out_dir.exists() and (not args.out_dir.is_dir() or any(args.out_dir.iterdir())):
            raise PlacementStudyError("choose a fresh empty output directory")
        request_raw, request = _read(args.request)
        binding = _context(request)
        if args.template:
            search = enumerate_capacity_placements(request["descriptor"], request["nodes"], policy=request["policy"], max_candidates=request["max_candidates"])
            assignments = {a["profile_identity"]: {k: a[k] for k in ("node_id", "start_layer", "end_layer", "has_embedding", "has_lm_head", "execution_device")}
                           for candidate in search["candidates"] for a in candidate["assignments"]}
            for values in assignments.values():
                values["source_layer_range"] = [values["start_layer"] * request["workload"]["range_unit_layers"],
                                                 values["end_layer"] * request["workload"]["range_unit_layers"]]
            result = {"schema": "qlh.placement_profile_template.v1", "target_profile_schema": PROFILE_SCHEMA,
                      "scope": SCOPE, "binding": binding, "complete_search": search["complete"],
                      "reason_code": search["reason_code"], "baseline": search["baseline"],
                      "capture_assignments": [{"assignment_sha256": identity, **values} for identity, values in sorted(assignments.items())],
                      "note": "Fill a separate profile with at least three paired compute/noncompute samples for each exact stage. This template is not measurement evidence."}
            output["ok"] = search["complete"] and search["baseline"]["admitted"]
        else:
            profile_raw, profile = _read(args.profile)
            result = evaluate_placement_study(request, profile)
            output["ok"] = result["status"] == "evaluated"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "request.json").write_bytes(request_raw)
        output["input_sha256"] = {"request": hashlib.sha256(request_raw).hexdigest()}
        if args.profile is not None:
            (args.out_dir / "profile.json").write_bytes(profile_raw)
            output["input_sha256"]["profile"] = hashlib.sha256(profile_raw).hexdigest()
        (args.out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        output["result_status"] = result.get("status", "capture_template")
        output["result"] = str(args.out_dir / "result.json")
    except (PlacementStudyError, PipelineCapacityError, OSError, ValueError, TypeError, KeyError) as exc:
        output["error"] = str(exc)
    print(json.dumps(output, ensure_ascii=False, allow_nan=False))
    return 0 if output["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
