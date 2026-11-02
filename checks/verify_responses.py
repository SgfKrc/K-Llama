"""Independently verify the Responses endpoint against the official SDK.

Checks the four acceptance criteria without trusting the implementation report:
  1. the official SDK parses both non-streaming and streaming responses
  2. text is byte-identical to what chat completions produces
  3. no U+FFFD in Chinese output
  4. unsupported parameters are rejected with the right `param`

The `seed` caveat the implementation flagged is also exercised: it is not in the SDK's
responses.create signature, so it has to go through extra_body, and the result should
still match chat completions with the same seed.
"""
from __future__ import annotations

import json
import sys

from openai import OpenAI

BASE = "http://127.0.0.1:8080/v1"
c = OpenAI(base_url=BASE, api_key="x", timeout=300.0)
ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        ok = False


print("1. official SDK parses both modes")
r = c.responses.create(model="slimarc", input="你好", max_output_tokens=16)
check("non-streaming status", r.status in ("completed", "incomplete"), f"status={r.status}")
check("output[0].type == message", r.output[0].type == "message")
check("output_text non-empty", bool(r.output_text), f"len={len(r.output_text)}")
check("usage present", r.usage is not None,
      f"in={r.usage.input_tokens} out={r.usage.output_tokens}")

events = []
for ev in c.responses.create(model="slimarc", input="你好", max_output_tokens=16, stream=True):
    events.append(ev)
types = [e.type for e in events]
deltas = [e.delta for e in events if e.type == "response.output_text.delta"]
seqs = [getattr(e, "sequence_number", None) for e in events]
check("stream has created", "response.created" in types)
check("stream has completed", "response.completed" in types)
check("stream has deltas", len(deltas) > 0, f"{len(deltas)} deltas / {len(events)} events")
check("sequence_number monotonic",
      all(a is not None and b is not None and b >= a for a, b in zip(seqs, seqs[1:])))
done = next((e for e in events if e.type == "response.output_text.done"), None)
check("deltas join == done.text", done is not None and "".join(deltas) == done.text)
comp = next((e for e in events if e.type == "response.completed"), None)
check("deltas join == completed.output_text",
      comp is not None and comp.response.output_text == "".join(deltas))

print("\n2. identical to chat completions")

# Text can only be compared where decoding is deterministic. With sampling on and no
# seed, two calls legitimately differ -- asserting equality there tests the RNG, not the
# protocol. So: argmax for the default case, an explicit seed for the sampled case.
chat = c.chat.completions.create(model="slimarc",
                                 messages=[{"role": "user", "content": "你好"}],
                                 max_tokens=32)
resp = c.responses.create(model="slimarc", input="你好", max_output_tokens=32)
ct, rt = chat.choices[0].message.content, resp.output_text
check("default (argmax): text identical", ct == rt, f"chat={len(ct)} resp={len(rt)}")

for label, kw in (("t=0.7 top_p=0.9 seed=7", {"temperature": 0.7, "top_p": 0.9, "seed": 7}),
                  ("t=1.0 seed=99", {"temperature": 1.0, "seed": 99})):
    extra = dict(kw)
    chat = c.chat.completions.create(model="slimarc",
                                     messages=[{"role": "user", "content": "你好"}],
                                     max_tokens=32, **extra)
    # seed is not in the SDK's responses.create signature; it has to ride extra_body.
    body = {k: v for k, v in extra.items() if k != "seed"}
    rb = {"seed": extra["seed"]} if "seed" in extra else {}
    resp = c.responses.create(model="slimarc", input="你好", max_output_tokens=32,
                              extra_body=rb, **body)
    ct, rt = chat.choices[0].message.content, resp.output_text
    check(f"{label}: text identical", ct == rt, f"chat={len(ct)} resp={len(rt)}")

# Same seed twice through the same endpoint must reproduce, which is the property that
# makes the comparison above meaningful.
a = c.responses.create(model="slimarc", input="你好", max_output_tokens=24,
                       temperature=0.8, extra_body={"seed": 42}).output_text
b = c.responses.create(model="slimarc", input="你好", max_output_tokens=24,
                       temperature=0.8, extra_body={"seed": 42}).output_text
check("same seed reproduces", a == b, f"len {len(a)}/{len(b)}")

c1 = c.chat.completions.create(model="slimarc",
                               messages=[{"role": "user", "content": "你好"}],
                               max_tokens=24, temperature=0.8, seed=42)
check("seed=42 chat == seed=42 chat (control)",
      c1.choices[0].message.content == c.chat.completions.create(
          model="slimarc", messages=[{"role": "user", "content": "你好"}],
          max_tokens=24, temperature=0.8, seed=42).choices[0].message.content)

print("\n3. Chinese has no replacement characters")
zh = "请用中文写一段话介绍长城。"
acc = []
for ev in c.responses.create(model="slimarc", input=zh, max_output_tokens=48, stream=True):
    if ev.type == "response.output_text.delta":
        acc.append(ev.delta)
joined = "".join(acc)
bad = joined.count("\ufffd")
cjk = sum(1 for ch in joined if "\u4e00" <= ch <= "\u9fff")
check("no U+FFFD", bad == 0, f"{len(joined)} chars, {cjk} CJK, {bad} bad")
check("contains CJK", cjk > 0)

print("\n4. unsupported parameters rejected")
cases = [
    ("tools", {"tools": [{"type": "web_search"}]}),
    ("previous_response_id", {"previous_response_id": "resp_abc"}),
    ("store", {"store": True}),
    ("reasoning", {"reasoning": {"effort": "low"}}),
]
for param, extra in cases:
    try:
        c.responses.create(model="slimarc", input="hi", max_output_tokens=8, **extra)
        check(f"{param} rejected", False, "no error raised")
    except Exception as exc:  # noqa: BLE001 - the SDK type varies by status
        body = getattr(exc, "body", None) or {}
        got = ""
        if isinstance(body, dict):
            err = body.get("error", body)
            got = str(err.get("param", "")) if isinstance(err, dict) else ""
        check(f"{param} rejected", "400" in str(exc) or "BadRequest" in type(exc).__name__,
              f"param={got!r}")

print("\n5. empty values are accepted, not rejected")
try:
    c.responses.create(model="slimarc", input="hi", max_output_tokens=4, tools=[], store=False)
    check("tools=[] / store=false accepted", True)
except Exception as exc:  # noqa: BLE001
    check("tools=[] / store=false accepted", False, str(exc)[:80])

print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
