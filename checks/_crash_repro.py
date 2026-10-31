"""Reproduce the intermittent keep-head worker death on the speculative path.

The report is: `--spec-k 2` fails about half the time with
    RuntimeError: stage 0 (gpu): {'code': 'forward_failed',
                                 'detail': 'keep-head worker exited before responding'}
while `--spec-k 0` is stable. This script runs the real CLI, one fresh process per
run (so nothing leaks between runs), saves every stage log per run, and records the
exact failure with its timing.

Usage:
    python checks/_crash_repro.py --spec-k 2 --runs 6
    python checks/_crash_repro.py --spec-k 0 --runs 4

It never runs two instances at once (the 22 GiB constraint), and it reaps any
keep_head_worker / stage_worker left behind after each run.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOGS = ROOT / "build" / "relay-records"
OUT = HERE / "crash-logs"

MODEL = (r"H:\qlh_models\qwen3-6-35b-a3b-opus47-distill-apex-mtp"
         r"\Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-I-Quality.gguf")
SHIM_DIR = r"H:\slimarc-proto\exp-c\shim"


def _env(extra: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env["SA_MEM_SAFETY"] = "0.88"
    env["SA_BACKEND_DIR"] = SHIM_DIR
    env["SA_SPEC_K"] = "2"
    env.update(extra)
    # Keep the children on this machine's stdout in the file, not the console.
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _reap() -> None:
    """Kill anything left from this run: the 22 GiB must not stay committed."""
    try:
        import psutil
    except ImportError:
        return
    me = os.getpid()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.pid == me:
                continue
            cmd = " ".join(proc.info.get("cmdline") or [])
            if "stage_worker.py" in cmd or "keep_head_worker.py" in cmd:
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def _save_logs(tag: str) -> None:
    dest = OUT / tag
    if dest.is_dir():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for log in LOGS.glob("stage-*.log"):
        shutil.copy2(log, dest / log.name)


def run_once(spec_k: int, gen: int, tag: str, extra_env: dict[str, str],
             out_path: Path) -> dict:
    _reap()
    cmd = [sys.executable, "chain_client.py",
           "--topology", "topology-local-32-8.toml",
           "--prompt-text", "def quicksort(arr):",
           "--gen", str(gen), "--greedy", "--repeat", "1",
           "--spec-k", str(spec_k)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=_env(extra_env),
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    elapsed = time.perf_counter() - t0
    _save_logs(tag)
    record = {
        "tag": tag,
        "spec_k": spec_k,
        "rc": proc.returncode,
        "elapsed_s": round(elapsed, 1),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "ok": proc.returncode == 0 and "tokens:" in proc.stdout,
    }
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    _reap()
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-k", type=int, default=2)
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--extra-env", action="append", default=[],
                    help="KEY=VALUE applied to every run (e.g. SA_N_RS_SEQ=2)")
    args = ap.parse_args()

    extra_env: dict[str, str] = {}
    for item in args.extra_env:
        key, _, value = item.partition("=")
        extra_env[key] = value

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ok = 0
    records = []
    for i in range(1, args.runs + 1):
        tag = f"{stamp}-k{args.spec_k}-run{i}"
        path = OUT / f"{tag}.json"
        print(f"--- run {i}/{args.runs} spec_k={args.spec_k} {extra_env or ''}", flush=True)
        try:
            rec = run_once(args.spec_k, args.gen, tag, extra_env, path)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"    driver error: {exc}", flush=True)
            records.append({"tag": tag, "ok": False, "driver_error": str(exc)})
            continue
        records.append(rec)
        ok += 1 if rec["ok"] else 0
        if rec["ok"]:
            lines = [ln.strip() for ln in rec["stdout"].splitlines()
                     if "tokens:" in ln or "run 1:" in ln or "spec:" in ln]
            summary_line = " | ".join(lines).encode("ascii", "replace").decode("ascii")
            print(f"    OK {rec['elapsed_s']}s  {summary_line}", flush=True)
        else:
            print(f"    FAIL rc={rec['rc']} {rec['elapsed_s']}s", flush=True)
            for line in rec["stderr"].splitlines()[-12:]:
                print(f"      {line}", flush=True)
    summary = {"spec_k": args.spec_k, "runs": args.runs, "ok": ok,
               "extra_env": extra_env, "records": records}
    (OUT / f"{stamp}-k{args.spec_k}-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{ok}/{args.runs} succeeded  (spec_k={args.spec_k}, extra={extra_env})",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
