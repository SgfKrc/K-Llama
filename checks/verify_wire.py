"""Record and compare the hidden path across the stage chain.

Two jobs, both about the numeric path rather than throughput:

* ``--record`` walks a real topology (2B, greedy) and writes every hidden a stage
  emitted, plus the token sequence, into an ``.npz``. Running it before and after a
  transport change is what makes "float32 is bit-identical" a measurement instead of
  a claim: the comparison is on the raw bits (``view(uint32)``), which also catches a
  -0.0 / NaN-payload difference that ``max|diff|`` would hide.
* ``--compare`` puts two records side by side: bit equality for float32, and for the
  float16 record the quantization error of the *real* hidden it carried (the same
  round trip ``astype(float16) -> astype(float32)`` the wire performs) plus whether
  the greedy token sequences still agree.

The wire dtype is whatever ``SA_WIRE_DTYPE`` selects in the code under test; it is
recorded so a mixed pair cannot be compared by accident.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import protocol  # noqa: E402
from chain_client import StageClient, spawn_worker  # noqa: E402
from topology import load  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
DEFAULT_SHIM = Path(r"H:\slimarc-proto\exp-c\shim\qlh_keep_head.dll")

# Prompts chosen for shape diversity, not for output quality: the point is that the
# hidden statistics differ (norm, sign balance) between them, so a dtype that only
# works on one distribution shows up as a token mismatch on another.
PROMPTS = [
    "中国的首都是",
    "用一句话解释什么是量子纠缠。",
    "写一个 Python 函数，返回列表里第二大的数，并说明时间复杂度。",
    "The quick brown fox jumps over the lazy dog.",
    "请把下面这句话翻译成英文：层段之间的hidden state每一步都要传一次。",
]


def wire_dtype() -> str:
    """The transport dtype the code under test would use, for the record's metadata."""
    try:
        import wire  # noqa: WPS433 - optional until the transport change lands

        return wire.current_dtype()
    except Exception:  # noqa: BLE001 - pre-change tree has no wire module
        return "float32"


def _record(args: argparse.Namespace) -> int:
    stages = load(args.topology)
    shim = Path(args.shim)
    if not shim.is_file():
        raise SystemExit(f"shim not found: {shim}")
    procs = []
    clients = [StageClient(s) for s in stages]
    records: dict[str, np.ndarray] = {}
    meta = {"topology": str(args.topology), "prompts": [], "wire_dtype": wire_dtype(),
            "gen": args.gen, "stages": [s.kind for s in stages]}
    try:
        for s in stages:
            if s.is_local:
                procs.append(spawn_worker(s, shim, args.ctx))
        for c in clients:
            c.connect()
        if args.debug:
            for pi, text in enumerate(PROMPTS[: args.prompts]):
                p = clients[0].tokenize(text)
                print(f"    prompt[{pi}] {len(p)} tokens: {p}")
        for c in clients:
            c.set_sampling(temperature=0.0)
        for pi, text in enumerate(PROMPTS[: args.prompts]):
            prompt = clients[0].tokenize(text)
            if not prompt:
                raise SystemExit(f"prompt {pi} tokenized to nothing")
            for c in clients:
                c.reset()
            tokens, pre, steps = _drive(clients, prompt, args.gen)
            meta["prompts"].append({"index": pi, "text": text,
                                    "n_prompt": len(prompt),
                                    "token_ids": [int(t) for t in tokens]})
            records[f"p{pi}_tokens"] = tokens
            records[f"p{pi}_prompt"] = np.asarray(prompt, dtype=np.int64)
            for stage, arr in pre.items():
                records[f"p{pi}_pre{stage}"] = arr
            for stage, rows in steps.items():
                records[f"p{pi}_step{stage}"] = np.asarray(rows, dtype=np.float32)
            print(f"  prompt {pi}: {len(tokens)} tokens, "
                  f"unique={len(set(int(t) for t in tokens))}, "
                  f"prompt_len={len(prompt)}")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **records)
        # Metadata travels next to the arrays, not inside them: json keeps it readable
        # and out of npz's object-array path.
        out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
        print(f"wrote {out} ({len(records)} arrays), wire dtype {meta['wire_dtype']}")
    except Exception:
        for proc, log in procs:
            print(f"  !! worker pid={proc.pid} exit={proc.poll()} log={log}", file=sys.stderr)
            try:
                tail = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
                print("  " + "\n  ".join(tail), file=sys.stderr)
            except OSError:
                pass
        raise
    finally:
        for c in clients:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        for proc, _log in procs:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                proc.kill()
    return 0


def _drive(clients: list[StageClient], prompt: list[int], gen: int):
    """Prefill, then decode; return tokens and every hidden a stage handed on.

    Deliberately mirrors chain_client.forward's role dispatch instead of calling it,
    because the intermediate hidden is exactly what this script has to keep.
    """
    n_stages = len(clients)
    pre: dict[int, np.ndarray] = {}
    steps: dict[int, list[np.ndarray]] = {i: [] for i in range(n_stages - 1)}
    tokens: list[int] = []
    hidden = None
    token = -1
    for i, c in enumerate(clients):
        kind = c.stage.kind
        if kind == "gpu":
            hidden = c.forward_tokens(prompt, 0)
        elif kind == "cpu":
            hidden = c.forward_hidden(hidden, 0)
        else:
            token = c.forward_token(hidden, 0)
        if i < n_stages - 1:
            pre[i] = np.ascontiguousarray(hidden, dtype=np.float32)
    tokens.append(token)
    n_past = len(prompt)
    for _ in range(gen - 1):
        hidden = None
        for i, c in enumerate(clients):
            kind = c.stage.kind
            if kind == "gpu":
                hidden = c.forward_tokens([tokens[-1]], n_past)
            elif kind == "cpu":
                hidden = c.forward_hidden(hidden, n_past)
            else:
                token = c.forward_token(hidden, n_past)
            if i < n_stages - 1:
                steps[i].append(np.ascontiguousarray(hidden, dtype=np.float32))
        tokens.append(token)
        n_past += 1
    return np.asarray(tokens, dtype=np.int64), pre, steps


def _bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(np.ascontiguousarray(a).view(np.uint32),
                              np.ascontiguousarray(b).view(np.uint32)))


def _quant_err(a: np.ndarray) -> dict:
    """The exact error a float16 hop adds, measured on a real float32 hidden."""
    q = a.astype(np.float16).astype(np.float32)
    diff = np.abs(a - q)
    scale = np.abs(a)
    # Two denominators, because they answer different questions: per-element relative
    # error is the local one, and max|diff|/max|a| is the conservative whole-vector one.
    rel = np.divide(diff, scale, out=np.zeros_like(diff), where=scale > 0)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float(rel.max()),
        "rms_rel": float(np.sqrt(np.mean(rel ** 2))),
        "max_over_max": float(diff.max() / max(float(scale.max()), 1e-12)),
        "norm": float(np.linalg.norm(a)),
        "n": int(a.size),
    }


def _compare(args: argparse.Namespace) -> int:
    a_path, b_path = Path(args.compare[0]), Path(args.compare[1])
    a, b = np.load(a_path), np.load(b_path)
    a_meta = json.loads(a_path.with_suffix(".json").read_text(encoding="utf-8"))
    b_meta = json.loads(b_path.with_suffix(".json").read_text(encoding="utf-8"))
    print(f"A: {a_path.name}  wire={a_meta['wire_dtype']}  gen={a_meta['gen']}")
    print(f"B: {b_path.name}  wire={b_meta['wire_dtype']}  gen={b_meta['gen']}")
    if a_meta["stages"] != b_meta["stages"]:
        print("  !! stage lists differ:", a_meta["stages"], b_meta["stages"])

    ok = True
    same_tokens = 0
    common = sorted(set(a.files) & set(b.files))
    hidden_keys = [k for k in common if "_pre" in k or "_step" in k]
    token_keys = [k for k in common if k.endswith("_tokens")]
    for key in hidden_keys:
        if not _bits_equal(a[key], b[key]):
            ok = False
    for key in token_keys:
        if np.array_equal(a[key], b[key]):
            same_tokens += 1
        else:
            ok = False

    print(f"  arrays compared: {len(hidden_keys)} hidden, {len(token_keys)} token")
    print(f"  hidden bitwise identical: {'YES' if all(_bits_equal(a[k], b[k]) for k in hidden_keys) else 'NO'}")
    print(f"  token sequences identical: {same_tokens}/{len(token_keys)}")
    for pi, p in enumerate(a_meta["prompts"]):
        ta = a.get(f"p{pi}_tokens")
        tb = b.get(f"p{pi}_tokens")
        if ta is None or tb is None:
            continue
        same = np.array_equal(ta, tb)
        first_diff = next((int(i) for i in range(min(len(ta), len(tb))) if ta[i] != tb[i]), None)
        print(f"    p{pi} {'SAME' if same else 'DIFF'} at {len(ta)}/{len(tb)} tokens"
              + ("" if same else f", first divergence at step {first_diff}"))
        if not same:
            n = first_diff or 0
            print(f"        A: {[int(x) for x in ta[max(0, n - 3):n + 4]]}"
                  f"  B: {[int(x) for x in tb[max(0, n - 3):n + 4]]}")

    # Quantization error of the real float32 hidden this chain emitted. This is the
    # measured cost of a float16 hop, independent of whether the tokens survived it.
    for key in hidden_keys:
        src = a[key]
        e = _quant_err(src)
        print(f"  quant {key:<14} norm={e['norm']:8.2f} max|d|={e['max_abs']:.3e} "
              f"max_rel={e['max_rel']:.2e} rms_rel={e['rms_rel']:.2e} "
              f"max/max={e['max_over_max']:.2e}")
    print()
    print("RESULT:", "PASS" if ok else "DIFF")
    return 0 if ok else 1


def _roundtrip(args: argparse.Namespace) -> int:
    """Check that the wire encoding is exactly the dtype it claims, and nothing else.

    This is what separates "float16 is lossy" from "the float16 code path is broken":
    the bytes on the wire must equal ``arr.astype(float16).tobytes()`` and decoding
    them must reproduce that quantization bit for bit. The end-to-end token difference
    then really is the model's sensitivity, not a bug in the encoder.
    """
    import wire

    rng = np.random.default_rng(11)
    # Same shape and scale as a real relay hidden (norm ~30-100 on this model).
    arr = (rng.standard_normal((1, 2048)) * 4.0).astype(np.float32)
    ok = True
    for name in ("float32", "float16", "int16"):
        blob = wire.encode(arr, name)
        want_bytes = wire.payload_size(arr.shape, name)
        back = wire.decode(blob, name, arr.shape).astype(np.float32)
        if name == "int16":
            scale = float(np.abs(arr).max()) / 32767.0
            want = (np.clip(np.rint(arr / scale), -32767, 32767).astype(np.float32)
                    * np.float32(scale))
            exact = None
        else:
            want = arr.astype(wire.dtype_of(name)).astype(np.float32)
            exact = bytes(blob) == np.ascontiguousarray(
                arr, dtype=wire.dtype_of(name)).tobytes()
        bit_ok = bool(np.array_equal(back.view(np.uint32), want.view(np.uint32)))
        size_ok = len(blob) == want_bytes
        ok = ok and bit_ok and size_ok and exact is not False
        diff = np.abs(arr - back)
        print(f"  {name:<8} bytes={len(blob):5d} "
              f"exact-bytes={'n/a ' if exact is None else ('YES' if exact else 'NO')} "
              f"bit-exact-decode={'YES' if bit_ok else 'NO'} "
              f"max|d|={diff.max():.3e} rms_rel="
              f"{np.sqrt(np.mean((diff / np.maximum(np.abs(arr), 1e-12)) ** 2)):.2e}")
        if name == "float32" and (not exact or not bit_ok):
            print("  !! float32 wire bytes are not the raw array")
    # The frame layer must not touch the payload: encode then parse it back by hand.
    frame = protocol.encode(protocol.FORWARD, {"dtype": "float16"}, bytes(wire.encode(arr, "float16")))
    length, kind = protocol.HEADER.unpack(frame[:protocol.HEADER.size])
    payload = frame[protocol.HEADER.size:]
    (meta_len,) = protocol.META_LEN.unpack_from(payload, 0)
    payload_ok = (len(payload) == length and
                  payload[protocol.META_LEN.size + meta_len:] == bytes(wire.encode(arr, "float16")))
    ok = ok and payload_ok and kind == protocol.FORWARD
    print(f"  frame    kind={kind} payload={length} B  payload-untouched="
          f"{'YES' if payload_ok else 'NO'}")
    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="topology-2b.toml")
    ap.add_argument("--shim", default=str(DEFAULT_SHIM))
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--prompts", type=int, default=5)
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--out", default="build/wire-record.npz")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="compare two records instead of recording")
    ap.add_argument("--roundtrip", action="store_true",
                    help="check wire encode/decode/frame without loading a model")
    args = ap.parse_args()
    if args.roundtrip:
        return _roundtrip(args)
    if args.compare:
        return _compare(args)
    if not (HERE / args.topology).is_file() and not Path(args.topology).is_file():
        raise SystemExit(f"topology not found: {args.topology}")
    return _record(args)


if __name__ == "__main__":
    raise SystemExit(main())
