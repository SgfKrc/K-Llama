"""Exercise local GGUF template controls with real llama.cpp Jinja handlers.

Only completion and model construction are stubbed; no model weights are read.
"""
from __future__ import annotations

import pytest

from llama_engine import LlamaCppEngine
from model_config import get_model_profile_metadata


# The generation suffix is the enable_thinking branch embedded in the local
# Qwen3.5 GGUF. Keep it conditional on an explicitly defined false value.
_QWEN35_TEMPLATE = """{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}"""
_PLAIN_TEMPLATE = "{{ messages[-1].content }}"
_QUESTION = "What is 1+1? Reply with only 2."


@pytest.fixture
def local_engine(monkeypatch, tmp_path):
    llama_cpp = pytest.importorskip("llama_cpp")
    from llama_cpp import llama_chat_format

    def forbidden_construction(*_args, **_kwargs):
        raise AssertionError("local template tests must not construct a native model")

    monkeypatch.setattr(llama_cpp.Llama, "__init__", forbidden_construction)

    class Model:
        # Use the real binding's request dispatch, formatter, handler and response
        # conversion; replace the sampler with a deterministic completion stub.
        create_chat_completion = llama_cpp.Llama.create_chat_completion
        verbose = False

        def __init__(self, metadata, chat_format):
            self.metadata = metadata
            self.chat_format = chat_format
            self.chat_handler = None
            self.prompts = []
            self.handler_requests = []
            self.completion_requests = []
            self._chat_handlers = {}
            for key, template in metadata.items():
                if key == "tokenizer.chat_template":
                    name = "chat_template.default"
                elif key.startswith("tokenizer.chat_template."):
                    name = key[len("tokenizer."):]
                else:
                    continue
                handler = llama_chat_format.Jinja2ChatFormatter(
                    template=template, eos_token="<|im_end|>", bos_token="",
                    stop_token_ids=[248046],
                ).to_chat_handler()

                def recorded_handler(_handler=handler, **kwargs):
                    self.handler_requests.append(dict(kwargs))
                    return _handler(**kwargs)

                self._chat_handlers[name] = recorded_handler

        def tokenize(self, text, **kwargs):
            self.prompts.append((text.decode(), kwargs))
            return list(text)

        def create_completion(self, **kwargs):
            self.completion_requests.append(kwargs)
            completion = {
                "id": "cmpl-template-test", "created": 0, "model": "fixture",
                "choices": [{"index": 0, "text": "2", "logprobs": None,
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(kwargs["prompt"]),
                          "completion_tokens": 1, "total_tokens": len(kwargs["prompt"]) + 1},
            }
            if kwargs["stream"]:
                delta = {**completion, "choices": [{"index": 0, "text": "2",
                                                   "logprobs": None, "finish_reason": None}]}
                terminal = {**completion, "choices": [{"index": 0, "text": "",
                                                      "logprobs": None, "finish_reason": "stop"}]}
                return iter([delta, terminal])
            return completion

    def build(alias="q35-9b-local-baseline", *, template=_QWEN35_TEMPLATE,
              chat_format=None, extra_metadata=None, custom_handler=None, profile=None):
        path = tmp_path / "qwen3-5-9b-local-Q4_K_M.gguf"
        path.write_bytes(b"test model construction is intercepted")
        metadata = {"tokenizer.chat_template": template, **(extra_metadata or {})}
        captured_load_kwargs = {}

        def construct(load_kwargs, n_seq_max):
            assert n_seq_max is None
            captured_load_kwargs.update(load_kwargs)
            selected = load_kwargs.get("chat_format", "chat_template.default")
            model = Model(metadata, selected)
            model.chat_handler = custom_handler
            return model

        monkeypatch.setattr("llama_engine._new_llama_with_seq_max", construct)
        engine = LlamaCppEngine()
        engine.active_model_id = alias
        engine.load_model(
            str(path), n_ctx=128, n_threads=1, chat_format=chat_format,
            model_profile=get_model_profile_metadata(alias) if profile is None else profile,
        )
        return engine, engine._model, captured_load_kwargs

    return build


def _chat(engine, **kwargs):
    return engine.chat([{"role": "user", "content": _QUESTION}],
                       max_tokens=32, temperature=0, **kwargs)


@pytest.mark.parametrize("alias", ["q35-9b-local-baseline", "qwen3-5-9b", "unrelated-registry-alias"])
def test_unprofiled_alias_uses_selected_native_jinja_with_thinking_disabled(local_engine, alias):
    assert "template" not in get_model_profile_metadata(alias)
    engine, model, load_kwargs = local_engine(alias)
    result = _chat(engine, enable_thinking=False, show_thinking=False)
    assert "chat_format" not in load_kwargs  # Let llama.cpp select its GGUF template.
    assert model.handler_requests[-1]["enable_thinking"] is False
    assert "show_thinking" not in model.handler_requests[-1]
    assert model.prompts[-1][0] == (
        f"<|im_start|>user\n{_QUESTION}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert result["content"] == "2"  # Stub output only, not a model accuracy claim.
    assert model.completion_requests[-1]["temperature"] == 0
    assert model.completion_requests[-1]["max_tokens"] == 32
    assert "<|im_start|>" in model.completion_requests[-1]["stop"]
    caps = engine.get_capabilities()
    assert caps["chat_template"] == "chat_template.default"
    assert caps["thinking"] == "declared"
    assert caps["thinking_controlled"] is True
    assert caps["thinking_enabled"] is False


@pytest.mark.parametrize("enable,show,expected", [(False, True, False), (True, False, True),
                                                (None, False, False), (None, True, True)])
def test_explicit_generation_flag_takes_precedence_over_display_flag(local_engine, enable, show, expected):
    engine, model, _ = local_engine()
    _chat(engine, enable_thinking=enable, show_thinking=show)
    assert model.handler_requests[-1]["enable_thinking"] is expected
    assert model.prompts[-1][0].endswith(
        "<think>\n" if expected else "<think>\n\n</think>\n\n"
    )


@pytest.mark.parametrize("next_kwargs", [{}, {"enable_thinking": None, "show_thinking": None}])
def test_none_restores_native_template_default_after_explicit_false(local_engine, next_kwargs):
    engine, model, _ = local_engine()
    _chat(engine, enable_thinking=False)
    _chat(engine, **next_kwargs)
    assert "enable_thinking" not in model.handler_requests[-1]
    assert model.prompts[-1][0].endswith("<|im_start|>assistant\n<think>\n")
    assert engine.get_capabilities()["thinking_enabled"] is None


def test_streaming_reaches_same_real_jinja_false_branch(local_engine):
    engine, model, _ = local_engine()
    chunks = list(engine.chat_stream(
        [{"role": "user", "content": _QUESTION}], max_tokens=32,
        temperature=0, enable_thinking=False, show_thinking=False,
    ))
    assert chunks == ["2"]
    assert model.handler_requests[-1]["enable_thinking"] is False
    assert model.prompts[-1][0].endswith("<think>\n\n</think>\n\n")


def test_named_selected_template_is_controlled(local_engine):
    engine, model, _ = local_engine(
        template=_PLAIN_TEMPLATE, chat_format="chat_template.reasoning",
        extra_metadata={"tokenizer.chat_template.reasoning": _QWEN35_TEMPLATE},
    )
    _chat(engine, enable_thinking=False)
    assert model.prompts[-1][0].endswith("<think>\n\n</think>\n\n")
    assert engine.get_capabilities()["chat_template"] == "chat_template.reasoning"


def test_unselected_native_template_does_not_enable_control(local_engine):
    engine, model, _ = local_engine(
        chat_format="chat_template.plain",
        extra_metadata={"tokenizer.chat_template.plain": _PLAIN_TEMPLATE},
    )
    _chat(engine, enable_thinking=False)
    assert model.chat_handler is None
    assert engine.get_capabilities()["thinking_controlled"] is False
    assert "enable_thinking" not in model.handler_requests[-1]
    assert model.prompts[-1][0] == _QUESTION


def test_explicit_chatml_is_not_replaced_by_metadata_template(local_engine):
    engine, model, load_kwargs = local_engine(chat_format="chatml")
    _chat(engine, enable_thinking=False)
    assert load_kwargs["chat_format"] == "chatml"
    assert model.chat_handler is None
    assert engine.get_capabilities()["thinking_controlled"] is False
    assert "<think>" not in model.prompts[-1][0]


def test_custom_handler_is_not_claimed_as_native_jinja_control(local_engine):
    def handler(**_kwargs):
        raise AssertionError("only inspecting registration, not calling custom handler")

    engine, model, _ = local_engine(custom_handler=handler)
    assert model.chat_handler is handler
    assert engine.get_capabilities()["thinking_controlled"] is False


def test_registered_qwen3_profile_still_forwards_flag(local_engine):
    engine, model, _ = local_engine(
        "qwen3-0.6b", profile={"template": "qwen3_chat_v1", "thinking": "declared"},
    )
    _chat(engine, enable_thinking=False)
    assert model.handler_requests[-1]["enable_thinking"] is False
    assert model.prompts[-1][0].endswith("<think>\n\n</think>\n\n")
