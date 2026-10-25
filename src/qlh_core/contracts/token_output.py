"""Negotiated, position-bound tail argmax contracts; no runtime dependencies."""
from __future__ import annotations
from collections.abc import Mapping

INT32_MAX = (1 << 31) - 1
ALL_POSITIONS = "all_positions"
MULTI_POSITION_FIELDS = frozenset({"token_argmax_by_position", "positions", "n_vocab"})


class TokenOutputError(ValueError):
    def __init__(self, code, field):
        self.code, self.field = code, field
        super().__init__(f"{code}: {field}")


def _int(value, field, *, minimum=0):
    if type(value) is not int or not minimum <= value <= INT32_MAX:
        raise TokenOutputError("invalid_token_output", field)
    return value


def contiguous_positions(values, count, *, field="positions"):
    _int(count, "n_tokens", minimum=1)
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise TokenOutputError("invalid_token_output", field)
    for i, value in enumerate(values):
        _int(value, f"{field}[{i}]")
        if i and value != values[i - 1] + 1:
            raise TokenOutputError("invalid_token_output", field)
    return list(values)


def validate_token_output_request(fields, root_input):
    """True only for an explicit single-sequence, contiguous all-row request."""
    if "token_output" not in fields:
        return False
    if fields["token_output"] != ALL_POSITIONS or root_input.get("want_hidden") is not False:
        raise TokenOutputError("invalid_token_output", "token_output")
    count = (fields.get("hidden_spec") or {}).get("n_tokens")
    contiguous_positions(fields.get("positions"), count)
    if "seq_ids" in fields:
        ids = fields["seq_ids"]
        if not isinstance(ids, (list, tuple)) or len(ids) != count:
            raise TokenOutputError("invalid_token_output", "seq_ids")
        for i, seq in enumerate(ids):
            _int(seq, f"seq_ids[{i}]")
            if seq != ids[0]:
                raise TokenOutputError("invalid_token_output", "seq_ids")
    return True


def validate_token_output(output, *, n_tokens=None, positions=None, require_all=False, allow_all=True):
    """Validate the wire group and, at a receiver, bind it to its exact offer."""
    if not isinstance(output, Mapping):
        raise TokenOutputError("invalid_token_output", "output")
    present = MULTI_POSITION_FIELDS.intersection(output)
    if not present:
        if require_all:
            raise TokenOutputError("missing_token_output", "token_argmax_by_position")
        return False
    if not allow_all:
        raise TokenOutputError("unexpected_token_output", "token_argmax_by_position")
    if present != MULTI_POSITION_FIELDS or "hidden_out_f32" in output or "hidden_out_ref" in output:
        raise TokenOutputError("invalid_token_output", "output")
    count = _int(output.get("n_tokens"), "n_tokens", minimum=1)
    rows = contiguous_positions(output["positions"], count)
    vocab = _int(output["n_vocab"], "n_vocab", minimum=1)
    tokens = output["token_argmax_by_position"]
    if not isinstance(tokens, (list, tuple)) or len(tokens) != count:
        raise TokenOutputError("invalid_token_output", "token_argmax_by_position")
    for i, token in enumerate(tokens):
        if _int(token, f"token_argmax_by_position[{i}]") >= vocab:
            raise TokenOutputError("invalid_token_output", f"token_argmax_by_position[{i}]")
    scalar = _int(output.get("token_argmax"), "token_argmax")
    if scalar != tokens[-1]:
        raise TokenOutputError("invalid_token_output", "token_argmax")
    if n_tokens is not None and count != n_tokens:
        raise TokenOutputError("token_output_request_mismatch", "n_tokens")
    if positions is not None and rows != list(positions):
        raise TokenOutputError("token_output_request_mismatch", "positions")
    return True


def tail_token(value):
    """Select the final verified row for the existing serial generation loop."""
    if isinstance(value, Mapping):
        validate_token_output(value)
        return _int(value.get("token_argmax"), "token_argmax")
    return _int(value, "token_argmax")
