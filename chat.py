"""Turn OpenAI-style messages into the prompt the model expects.

Why this exists: `flatten_messages` in server.py deliberately sent plain text with role
prefixes, on the reasoning that a split model might not be a chat model. That reasoning held
for numerics but not for usefulness -- a Qwen checkpoint answered every question with
degenerate repetition ("是 是 是 ...") because it never saw `<|im_start|>`. The two facts that
settled it:

  * both models in this project carry a `tokenizer.chat_template` (8100 / 7798 chars), so they
    are chat models;
  * the shim was tokenizing with parse_special=false, which split `<|im_start|>` into ordinary
    fragments even when the caller passed it verbatim.

So: emit ChatML explicitly rather than rendering the stored Jinja template. Qwen's template
uses llama.cpp-specific filters, and this project supports the Qwen family only -- hand-writing
the format it uses is both shorter and less likely to silently render something wrong. If a
non-Qwen model is ever added, render `tokenizer.chat_template` instead and delete this.

The role of `parse_special` is the part that is easy to get wrong: the markers must become
special tokens, not text. `chat_token_env()` exists so the caller can prove the setting
reaches the shim.
"""
from __future__ import annotations

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
#: Qwen3's template opens the assistant turn with this to enable reasoning mode. Without
#: it the model answers directly and diverges from llama-server token for token
#: (measured: 16/16 different, see checks/verify_equiv.py).
THINK = "<think>\n"


def apply_chatml(messages: list[dict], system: str | None = None) -> str:
    """Render messages as ChatML, ending on an open assistant turn.

    Matches what llama.cpp's own template engine produces for these checkpoints -- that
    equality is the point, and `checks/verify_equiv.py` enforces it. Two details are
    easy to get wrong and both were:

    * No system block unless the caller passes one. The template renders nothing for a
      conversation without a system message; injecting a default changed the answer.
    * The assistant turn opens with `THINK`. Qwen3 is a reasoning model and the template
      enables it by default.

    Consecutive same-role messages stay separate blocks: merging them changes the token
    sequence, and the model was trained on the separate form.
    """
    parts: list[str] = []
    if system is not None:
        parts.append(f"{IM_START}system\n{system}{IM_END}\n")

    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, list):
            # OpenAI's multimodal shape; keep the text parts only.
            content = "".join(p.get("text", "") for p in content
                              if isinstance(p, dict) and p.get("type") == "text")
        parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")

    # Open the assistant turn: the model continues from here.
    parts.append(f"{IM_START}assistant\n{THINK}")
    return "".join(parts)


def chat_token_env() -> dict[str, str]:
    """Environment the shim needs to treat the markers as special tokens."""
    return {"SA_PARSE_SPECIAL": "1"}
