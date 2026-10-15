"""Opt-in, real Textual Route-A capture and offline DL/LL bundle combination.

The bench prepares ONE mode before each capture; this runner never loads models
or changes assignments. QLH_RUN_PHYSICAL_TESTS=1 is mandatory for capture.
Capture creates a dedicated session via Pilot /new CI-RELEASE-{mode}-{uuid};
--session resumes an existing bench session via Pilot instead.
Capture needs a clean checkout (including submodules) at --revision / HEAD.
Use an ignored output directory or a directory outside the checkout.

--topology and --native are bench-owned JSON snapshots. They may be produced
while capture is running; they are read AFTER the UI completes. Their captured_at
must be inside this capture window, or the explicitly declared bench window
--capture-started-at. Timestamps and snapshot bytes are never rewritten.
Nested raw_evidence paths are relative to each input snapshot's parent, and are
copied unchanged relative to the output bundle root. Use mode-specific paths to
avoid DL/LL conflicts. combine preserves this layout, hashes and every raw ref.

Examples (only after the bench has prepared the requested mode and two remotes)::

    python scripts/run_product_relay_gate.py capture --mode DL --master-log LOG \
        --topology BENCH/topology.json --native BENCH/native.json --output OUT/dl.json
    python scripts/run_product_relay_gate.py combine --inputs OUT/dl.json OUT/ll.json \
        --output RELEASE/bundle.json

Every invocation writes OUTPUT.result.json, or a unique failure report if that
name already exists. Synthetic unit tests are not physical product evidence.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
MESSAGE = "What is 1 + 1? Reply with only the result, no explanation."
EVIDENCE_KEYS = ("response", "intent", "handoffs", "topology", "native", "ui")
COMPLETED_PREFIX = "Route-A stage completed: "


class CaptureError(ValueError):
    """Missing or inconsistent evidence; never a successful physical run."""


def require(condition, message):
    if not condition:
        raise CaptureError(message)


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def file_ref(path, base):
    return {"path": path.relative_to(base).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def checked_file(base, ref):
    require(isinstance(ref, dict), "missing file reference")
    name = ref.get("path")
    require(isinstance(name, str) and name.strip(), "missing evidence path")
    relative = Path(name)
    require(not relative.is_absolute() and not PureWindowsPath(name).drive
            and ".." not in relative.parts, f"unconfined evidence path: {name}")
    path = (base / relative).resolve()
    require(path.is_relative_to(base.resolve()), f"evidence escapes bundle: {name}")
    require(path.is_file(), f"missing evidence: {path}")
    digest = ref.get("sha256")
    require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
            f"invalid evidence SHA-256: {name}")
    require(hashlib.sha256(path.read_bytes()).hexdigest() == digest,
            f"evidence SHA-256 mismatch: {name}")
    return path


def nested_refs(value):
    """Also reject malformed known refs instead of silently dropping them."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"screen", "raw_evidence"}:
                require(isinstance(item, dict) and "path" in item and "sha256" in item,
                        f"missing {key} file reference")
            if isinstance(item, dict) and "path" in item and "sha256" in item:
                yield item
            else:
                yield from nested_refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_refs(item)


def collect_files(base, refs, *, parse_json=True):
    """Follow refs in JSON attachments, including UI and all node raw captures."""
    found = {}
    pending = [(ref, parse_json) for ref in refs]
    while pending:
        ref, parse_attachment = pending.pop()
        path = checked_file(base, ref)
        name = Path(ref["path"]).as_posix()
        if name in found:
            require(found[name][1] == ref["sha256"], f"conflicting evidence: {name}")
            continue
        found[name] = (path, ref["sha256"])
        if parse_attachment and path.suffix.lower() == ".json":
            # Nested screen/raw files are opaque bytes, including non-UTF8 JSON
            # captured by OS tools. Only structured root attachments are parsed.
            pending.extend((child, False) for child in nested_refs(read_json(path)))
    return found


def copy_files(files, base, reserved):
    # Check all conflicts before copying any file; never overwrite bench evidence.
    targets = {}
    for name, (_, digest) in files.items():
        target = (base / name).resolve()
        require(target not in targets or targets[target] == digest, f"conflicting evidence destination: {target}")
        targets[target] = digest
        require(target.is_relative_to(base.resolve()) and target not in reserved,
                f"evidence path conflicts with output: {name}")
        if target.exists():
            require(target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == digest,
                    f"conflicting evidence destination: {target}")
    for name, (source, digest) in files.items():
        data = source.read_bytes()
        require(hashlib.sha256(data).hexdigest() == digest, f"evidence changed: {source}")
        target = base / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)


def load_validator():
    path = ROOT / "scripts" / "product_relay_gate.py"
    require(path.is_file(), f"missing product validator: {path}")
    spec = importlib.util.spec_from_file_location("_qlh_product_relay_validator", path)
    require(spec is not None and spec.loader is not None, "cannot import product validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(callable(getattr(module, "validate_bundle", None)), "missing validate_bundle interface")
    return module.validate_bundle


def validate(validator, output, revision, modes):
    result = validator(output, expected_revision=revision, required_modes=modes)
    require(isinstance(result, dict) and type(result.get("passed")) is bool
            and isinstance(result.get("errors"), list) and isinstance(result.get("runs"), list),
            "validator returned an invalid result")
    require(result["passed"] is False or not result["errors"], "validator returned contradictory results")
    return result


def git(*args):
    result = subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                            capture_output=True, text=True, timeout=15)
    return result.stdout.strip()


def source_revision(requested):
    revision = git("rev-parse", "HEAD").lower()
    require(bool(re.fullmatch(r"[0-9a-f]{40}", revision)), "HEAD is not a full source revision")
    require(requested is None or requested == revision, "--revision must equal the executing checkout HEAD")
    dirty = git("status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none")
    require(not dirty, "dirty checkout: source_revision cannot identify executing source; "
            "capture requires committed source and ignored/external output")
    return revision


def identity(value):
    return tuple(value.get(key) for key in ("operation_id", "config_id", "generation"))


def prepared_intent(intent, mode, nodes):
    require(isinstance(intent, dict) and intent.get("status") == "ready"
            and intent.get("loaded_ready") is True, "current model intent is not natively ready")
    operation, config, generation = identity(intent)
    require(isinstance(operation, str) and bool(operation) and isinstance(config, str) and bool(config)
            and type(generation) is int and generation > 0, "intent identity is missing")
    plan = intent.get("plan")
    require(isinstance(plan, dict) and plan.get("admitted") is True, "current plan is not admitted")
    assignments = plan.get("assignments")
    require(isinstance(assignments, list) and len(assignments) >= 3,
            "current plan requires master and at least two remote workers")
    require(all(isinstance(item, dict) for item in assignments), "invalid current assignments")
    ids = [item.get("node_id") for item in assignments]
    require(all(isinstance(node, str) and node for node in ids) and ids[0] == "master"
            and len(set(ids)) == len(ids), "plan requires distinct master/remote assignments")
    available = {node.get("node_id") for node in nodes
                 if isinstance(node, dict) and isinstance(node.get("node_id"), str) and node.get("node_id")
                 and node.get("is_available") is True
                 and node.get("node_id") != "master" and node.get("role") != "master"}
    require(len(available) >= 2 and set(ids[1:]) <= available,
            "at least two distinct available remote workers must already be prepared")
    total = plan.get("total_layers")
    require(type(total) is int and total >= 3, "current plan lacks total decoder layers")
    cursor = 0
    for index, assignment in enumerate(assignments):
        start, end = assignment.get("start_layer"), assignment.get("end_layer")
        require(type(start) is int and type(end) is int and start == cursor
                and 0 <= start < end <= total and end - start < total,
                "prepared segments must be partial, continuous and ordered")
        cursor = end
        expected = "pytorch" if mode == "DL" and index == 0 else "llama_cpp"
        require(assignment.get("engine") == expected, f"prepared engine does not match requested {mode}")
    require(cursor == total, "prepared segments do not cover the decoder")
    return intent


def preflight(api, mode):
    config = api.get("/cluster/config/distributed-inference")
    require(isinstance(config, dict) and config.get("enabled") is True,
            "distributed inference must already be enabled")
    status = api.get("/status")
    require(isinstance(status, dict) and status.get("node_role") == "master",
            "capture endpoint must be the actual master")
    value = api.get("/cluster/nodes")
    require(isinstance(value, dict) and isinstance(value.get("nodes"), list), "missing actual cluster nodes")
    nodes = value["nodes"]
    remote = {n.get("node_id") for n in nodes if isinstance(n, dict)
              and isinstance(n.get("node_id"), str) and n.get("is_available") is True and n.get("node_id") not in {None, "", "master"}
              and n.get("role") != "master"}
    require(len(remote) >= 2, "at least two distinct available remote workers are required; no inference sent")
    return prepared_intent(api.get("/models/intent/current"), mode, nodes)


def current_rendered_answer(before, after, message):
    # A changed prefix means a history/UI refresh raced with this request.
    # Fail instead of accepting an old answer or interpreting SSE as UI output.
    require(after.startswith(before), "UI transcript changed before the current assistant could be isolated")
    delta = after[len(before):]
    boundary = ("\n" if before else "") + f"\u4f60 {message}\nassistant "
    require(delta.startswith(boundary), "UI is missing this submitted user/assistant boundary")
    body = delta[len(boundary):]
    require("\n\u4f60 " not in body, "another user turn appeared during UI capture")
    # Restrict the last assistant block to this submission, never old history.
    return body.rsplit("\nassistant ", 1)[-1]


def widget_text(app, selector):
    # App.query_one may search the default screen after a pushed screen becomes
    # active. Read the visible screen, never a hidden/default transcript.
    from textual.widgets import Static
    rendered = app.screen.query_one(selector, Static).render()
    plain = getattr(rendered, "plain", None)
    require(isinstance(plain, str), "cannot read actual rendered Textual widget text")
    return plain


async def drive_tui(api, args, attachments, base):
    # The real consumer imports this name at tui_textual.py:87 and calls it in
    # ChatPane._stream_reply. Wrapping tui_api alone would miss that bound alias.
    import tui_api
    import tui_textual
    from textual.widgets import Input, Static

    original = tui_api.iter_chat_payloads
    require(tui_textual.iter_chat_payloads is original, "TUI stream function was already replaced")
    done_payloads = []
    ended = threading.Event()
    stream_errors = []
    stream_path = attachments / "stream.jsonl"
    response_path = attachments / "response.json"

    def observe(*positional, **options):
        # Only request budgets change; every real payload is yielded untouched.
        options["max_new_tokens"] = args.max_tokens
        options["read_timeout"] = min(args.timeout, 120.0)
        try:
            require(options.get("routing_preference") == "distributed_required", "TUI lost required routing")
            require(bool(submission) and options.get("session_id") == submission["session_id"],
                    "TUI stream left the submitted bench session")
            with stream_path.open("x", encoding="utf-8") as stream:
                for payload in original(*positional, **options):
                    stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    if payload.get("error") or payload.get("cancelled"):
                        stream_errors.append(deepcopy(payload))
                    if payload.get("done"):
                        done_payloads.append(deepcopy(payload))
                        if len(done_payloads) == 1:
                            write_json(response_path, payload)
                    yield payload
        except Exception as exc:
            stream_errors.append(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            ended.set()

    app = tui_textual.KoakumaApp(api, interval=2.0, enable_thinking=False)
    submission = {}

    def message_hook(message):
        if (isinstance(message, Input.Submitted) and message.input.id == "chat-input"
                and message.value == MESSAGE and not submission):
            require(not any(w.group == "sessions" and not w.is_finished for w in app.workers),
                    "session worker raced with message submission")
            before = widget_text(app, "#chat-log")
            submission.update(before=before, offset=len(before), captured_at=time.time(),
                              session_id=app.session_id)
            with (attachments / "transcript-before.txt").open("x", encoding="utf-8") as stream:
                stream.write(before)

    tui_textual.iter_chat_payloads = observe
    try:
        async with app.run_test(size=(120, 40), message_hook=message_hook) as pilot:
            await pilot.press("enter")  # Real splash -> main transition.
            while app.main_screen is None or app.main_screen.model_mode != "distributed" \
                    or app.main_screen.chat_model_gate():
                await pilot.pause(0.1)
            chat_input = app.screen.query_one("#chat-input", Input)
            await pilot.click("#chat-input")
            session_title = None if args.session else f"CI-RELEASE-{args.mode}-{uuid.uuid4().hex}"
            chat_input.value = f"/resume {args.session}" if args.session else f"/new {session_title}"
            await pilot.press("enter")
            while any(w.group == "sessions" and not w.is_finished for w in app.workers):
                await pilot.pause(0.05)
            await pilot.pause()  # Drain the worker's final transcript/status callbacks.
            require(isinstance(app.session_id, str) and bool(app.session_id)
                    and (app.session_id == args.session if args.session else app.session_id != "default"),
                    "Pilot could not create/resume the dedicated bench session")
            chat_input.value = "/route required"
            await pilot.press("enter")
            require(app.routing_preference == "distributed_required", "Pilot /route required failed")
            # Recheck physical readiness immediately before the inference submit.
            latest = await asyncio.to_thread(preflight, api, args.mode)
            require(identity(latest) == identity(app.main_screen.model_intent),
                    "UI has not adopted the current prepared model intent")
            chat_input.value = MESSAGE
            await pilot.press("enter")
            while not ended.is_set() or any(w.group == "chat" and not w.is_finished for w in app.workers):
                await pilot.pause(0.05)
            await pilot.pause()
            require(len(done_payloads) == 1 and not stream_errors,
                    f"real SSE did not finish with one successful done: {stream_errors}")
            response = done_payloads[0]
            require(response.get("done") is True, "real SSE done was not true")
            # Extract from the actual rendered chat widget, never from the SSE answer.
            transcript = widget_text(app, "#chat-log")
            with (attachments / "transcript-after.txt").open("x", encoding="utf-8") as stream:
                stream.write(transcript)
            require(bool(submission), "Pilot did not submit the arithmetic message")
            require(app.session_id == submission["session_id"], "session changed during the UI request")
            rendered = current_rendered_answer(submission["before"], transcript, MESSAGE)
            metrics = response.get("metrics")
            require(isinstance(metrics, dict) and bool(metrics.get("request_id")), "done lacks request_id")
            screen = attachments / "screen.svg"
            with screen.open("x", encoding="utf-8") as stream:
                stream.write(app.export_screenshot())
            ui = {"entry": "textual", "done": True, "request_id": metrics["request_id"],
                  "submitted_message": MESSAGE, "expected_response": "2",
                  "rendered_response": rendered, "status_text": widget_text(app, "#chat-status"),
                  "captured_at": time.time(), "screen": file_ref(screen, base),
                  "session_id": app.session_id, "session_title": session_title,
                  "submitted_at": submission["captured_at"], "transcript_before_offset": submission["offset"],
                  "transcript_before": file_ref(attachments / "transcript-before.txt", base),
                  "transcript_after": file_ref(attachments / "transcript-after.txt", base)}
            write_json(attachments / "ui.json", ui)
            return response
    finally:
        tui_textual.iter_chat_payloads = original


def import_observation(source, destination, base, intent, started, reserved):
    data = source.read_bytes()
    observation = json.loads(data.decode("utf-8-sig"))
    require(isinstance(observation, dict), f"invalid bench observation: {source}")
    captured = observation.get("captured_at")
    require(type(captured) in (int, float) and math.isfinite(captured)
            and started <= captured <= time.time(), f"observation outside capture window: {source}")
    require(identity(observation) == identity(intent), f"observation identity differs from current intent: {source}")
    nodes = observation.get("nodes")
    require(isinstance(nodes, list) and nodes, f"missing node observations: {source}")
    for node in nodes:
        require(isinstance(node, dict) and isinstance(node.get("raw_evidence"), dict),
                f"missing per-node raw_evidence: {source}")
    files = collect_files(source.parent, nested_refs(observation), parse_json=False)
    copy_files(files, base, reserved | {destination.resolve()})
    with destination.open("xb") as stream:
        stream.write(data)  # Preserve bench JSON bytes, timestamps and nested paths.


def capture(args):
    require(os.environ.get("QLH_RUN_PHYSICAL_TESTS") == "1", "capture requires QLH_RUN_PHYSICAL_TESTS=1")
    launched = time.time()
    started = launched if args.capture_started_at is None else args.capture_started_at
    require(math.isfinite(started) and 0 < started <= launched, "--capture-started-at must be a real, nonfuture Unix time")
    revision = source_revision(args.revision)
    validator = load_validator()  # Missing dependency fails BEFORE any inference.
    sys.path.insert(0, str(ROOT / "src"))
    from tui_api import ApiClient
    import tui_textual  # noqa: F401 - fail early if Textual is unavailable

    base = args.output.parent
    attachments = base / (args.output.stem + ".attachments") / args.mode
    attachments.mkdir(parents=True, exist_ok=False)
    api = ApiClient(args.host, args.port, timeout=min(5.0, args.timeout),
                    auth_token=os.environ.get("QLH_PRODUCT_AUTH_TOKEN", ""))
    log_stat = args.master_log.stat()
    require(args.master_log.is_file(), "missing actual master log file")
    log_offset = log_stat.st_size if args.capture_started_at is None else 0
    intent = preflight(api, args.mode)
    write_json(attachments / "intent-before.json", intent)
    asyncio.run(asyncio.wait_for(
        drive_tui(api, args, attachments, base), timeout=args.timeout))
    current = api.get("/models/intent/current")
    require(isinstance(current, dict) and identity(current) == identity(intent), "current intent changed during capture")
    write_json(attachments / "intent.json", current)
    # Allow the independent bench to finish writing its snapshots during this run.
    deadline = time.monotonic() + args.observation_wait
    while not (args.topology.is_file() and args.native.is_file()) and time.monotonic() < deadline:
        time.sleep(0.1)
    reserved = {args.output.resolve(), result_path(args.output).resolve()}
    import_observation(args.topology, attachments / "topology.json", base, current, started, reserved)
    import_observation(args.native, attachments / "native.json", base, current, started, reserved)
    after = args.master_log.stat()
    require((after.st_dev, after.st_ino) == (log_stat.st_dev, log_stat.st_ino)
            and after.st_size >= log_stat.st_size, "master log rotated/truncated during capture")
    with args.master_log.open("rb") as stream:
        stream.seek(log_offset)
        raw_log = stream.read()
    with (attachments / "handoffs.log").open("xb") as stream:
        stream.write(raw_log)
    require(COMPLETED_PREFIX in raw_log.decode("utf-8-sig"), "no actual completed Route-A handoff log")
    require(source_revision(revision) == revision, "source revision changed during capture")
    run = {"mode": args.mode, **{key: file_ref(attachments / ("handoffs.log" if key == "handoffs" else key + ".json"), base)
                                 for key in EVIDENCE_KEYS}}
    run["stream"] = file_ref(attachments / "stream.jsonl", base)
    run["intent_before"] = file_ref(attachments / "intent-before.json", base)
    bundle = {"schema_version": 1, "source_revision": revision, "physical_opt_in": True,
              "capture_started_at": started, "capture_finished_at": time.time(), "runs": [run]}
    write_json(args.output, bundle)
    return validate(validator, args.output, revision, (args.mode,))


def combine(args):
    validator = load_validator()
    bundles = [(path, read_json(path)) for path in args.inputs]
    require(bool(bundles), "no input bundles")
    revision = args.revision or bundles[0][1].get("source_revision")
    require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision), "invalid source revision")
    runs, files, windows = [], {}, []
    for path, bundle in bundles:
        require(bundle.get("source_revision") == revision, "conflicting source revisions")
        entries = bundle.get("runs")
        require(isinstance(entries, list) and len(entries) == 1 and isinstance(entries[0], dict),
                "combine inputs must be single-mode bundles")
        run = entries[0]
        mode = run.get("mode")
        require(mode in ("DL", "LL"), "invalid input mode")
        require(mode not in {r["mode"] for r in runs}, f"duplicate mode: {mode}")
        result = validate(validator, path, revision, (mode,))
        require(result["passed"], f"input bundle failed validation: {path}: {result['errors']}")
        incoming = collect_files(path.parent, [run.get(key) for key in EVIDENCE_KEYS] + list(nested_refs(run)))
        for name, source in incoming.items():
            require(name not in files or files[name][1] == source[1], f"conflicting attachment path: {name}")
            files[name] = source
        runs.append(deepcopy(run))
        # Preserve an already narrowed original window as well as ordinary
        # capture bundles. A header union must never replace a per-mode window.
        window = next((item for item in bundle.get("input_windows", []) if item["mode"] == mode), bundle)
        windows.append((window["capture_started_at"], window["capture_finished_at"]))
    require({run["mode"] for run in runs} == {"DL", "LL"}, "combine requires both DL and LL")
    reserved = {args.output.resolve(), result_path(args.output).resolve()}
    copy_files(files, args.output.parent, reserved)
    # Original per-run windows are retained as additional provenance; snapshots
    # were validated in each original window before considering this union.
    merged = {"schema_version": 1, "source_revision": revision, "physical_opt_in": True,
              "capture_started_at": min(w[0] for w in windows), "capture_finished_at": max(w[1] for w in windows),
              "runs": sorted(runs, key=lambda run: run["mode"]),
              "input_windows": [{"mode": run["mode"], "capture_started_at": window[0],
                                 "capture_finished_at": window[1]} for run, window in zip(runs, windows)]}
    write_json(args.output, merged)
    return validate(validator, args.output, revision, ("DL", "LL"))


def result_path(output):
    return output.with_name(output.name + ".result.json")


def worker(args):
    try:
        result = capture(args)
    except BaseException as exc:
        result = {"passed": False, "errors": [f"{type(exc).__name__}: {exc}"], "runs": []}
    write_json(result_path(args.output), result)


def physical_process(args):
    # A process enforces a total bound even if a Textual worker, urllib read or
    # application teardown stalls. Killing this runner does not stop the server.
    process = multiprocessing.get_context("spawn").Process(target=worker, args=(args,))
    process.start()
    try:
        process.join(args.timeout)
        if process.is_alive():
            raise CaptureError(f"capture exceeded total timeout of {args.timeout:g}s")
        require(process.exitcode == 0, f"capture worker exited abnormally: {process.exitcode}")
        return read_json(result_path(args.output))
    finally:
        if process.is_alive():
            process.terminate()
            process.join(3)
            if process.is_alive():
                process.kill()
                process.join(3)
        process.close()


def parser():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = cli.add_subparsers(dest="command", required=True)
    cap = commands.add_parser("capture", help="capture one already prepared physical mode")
    cap.add_argument("--mode", choices=("DL", "LL"), required=True)
    cap.add_argument("--host", default="127.0.0.1")
    cap.add_argument("--port", type=int, default=8000)
    cap.add_argument("--session", help="resume this existing bench session via Pilot; default creates CI-RELEASE-{mode}-{uuid}")
    cap.add_argument("--max-tokens", type=int, default=16, choices=range(1, 33), metavar="1..32")
    cap.add_argument("--timeout", type=float, default=180, help="total process timeout in seconds (1..1800)")
    cap.add_argument("--observation-wait", type=float, default=5, help="wait for bench files after UI completion")
    cap.add_argument("--capture-started-at", type=float, help="explicit real bench window start, Unix seconds")
    cap.add_argument("--master-log", type=Path, required=True, help="actual master log file, read only")
    cap.add_argument("--topology", type=Path, required=True)
    cap.add_argument("--native", type=Path, required=True)
    merge = commands.add_parser("combine", help="validate and combine DL/LL single-mode bundles offline")
    merge.add_argument("--inputs", nargs="+", type=Path, required=True)
    for command in (cap, merge):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--revision", help="full source SHA (capture: clean HEAD; combine: input SHA)")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    args.output = args.output.resolve()
    report = result_path(args.output)
    try:
        require(not args.output.exists() and not report.exists(), "output bundle/result already exists")
        require(not (args.output.parent / (args.output.stem + ".attachments")).exists(), "attachment directory already exists")
        if args.command == "capture":
            require(os.environ.get("QLH_RUN_PHYSICAL_TESTS") == "1", "capture requires QLH_RUN_PHYSICAL_TESTS=1")
            require(math.isfinite(args.timeout) and 1 <= args.timeout <= 1800, "--timeout must be 1..1800 seconds")
            require(math.isfinite(args.observation_wait) and 0 <= args.observation_wait < args.timeout,
                    "--observation-wait must be nonnegative and less than --timeout")
            require(1 <= args.port <= 65535, "invalid API port")
            result = physical_process(args)
        else:
            result = combine(args)
        if not report.exists():
            write_json(report, result)
    except BaseException as exc:
        result = {"passed": False, "errors": [f"{type(exc).__name__}: {exc}"], "runs": []}
        # Preserve all previous invocations, including failed/partial captures.
        if report.exists():
            report = args.output.with_name(args.output.name + ".failure-" + uuid.uuid4().hex + ".json")
        try:
            write_json(report, result)
        except OSError as report_error:
            print(f"cannot write failure report: {report_error}", file=sys.stderr)
    print(json.dumps({**result, "report": str(report)}, ensure_ascii=False, indent=2))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
