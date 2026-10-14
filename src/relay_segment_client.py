"""Relay **段执行器** —— 调度层与 `relay_transport` 之间唯一的接线点（★ A1 / X 档）。

为什么存在（2026-09-24）
-----------------------
A1 的实质不是"没有 Relay 传输层"，而是**传输层、合同层、生产执行体三者都在，但彼此没有接线**：
`src/` 内 `relay_transport` **零消费者**（消费者全在 `scripts/`）。本模块把 `RelayTcpClient`
收敛成调度层唯一认得的「段」抽象，作为那条接线的**唯一入口**：

* **错误只出稳定码**：任何失败都收敛成 :class:`RelaySegmentError` 的 `code`，且**必须是**
  `relay_transport` 的白名单码（`_RELAY_ERROR_CODES`）；异常类名/消息**绝不**外传
  （既有契约：`relay_transport.py` 的 "Never put exception class names or messages on the
  Relay wire"）。未在白名单 ⇒ 回落 `relay_protocol_error`，**不会**把原始字符串带上 wire。
* **默认不压缩**：`quant="none"`。X 档**不含**任何量化档（`f16`/`int8_block128`/`int4_block128`
  留给 Y 档；`int4` 已有 23/32 FAIL 证据，永不作为可选档）。
* **不自带开关**：是否启用由**调度层**按三者共同决定 —— `QLH_RELAY_ENABLED`（默认关）、
  请求级 `routing_preference`（`local_only` ⇒ 不委派）以及该 worker 在 `QLH_RELAY_SEGMENTS`
  里的映射；本模块不读环境变量（一个事实一个来源，避免"两处开关互相打架"）。
* **不 import torch / llama_cpp**：只碰 `bytes` 与 `relay_transport` ⇒ 无 torch 的环境
  （`.venv-edge`）也能导入，与 §4.7 的"本机可闭环、不需要模型"一致。

范围（X 档）：**2 段**往返（head→middle / head→tail 的"本节点把 hidden 交给远端段"这一步）；
`>2` 段拓扑、SSH 隧道编排、成本决策、TUI 新 UI **均不在此模块**。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from relay_transport import (  # noqa: PLC2701 - 白名单是**唯一定义**，重复定义才会漂移
    _RELAY_ERROR_CODES,
    RELAY_DEFAULT_MAX_TOKENS,
    RELAY_PROTOCOL_ERROR,
    RELAY_TRANSPORT_ERROR,
    REQUEST_DEADLINE_EXCEEDED,
    RelayProtocolError,
    RelayTcpClient,
)

#: X 档支持的段角色。`head` = 上游段（送 token 拿 hidden）；`middle` = 中段（hidden→hidden）；
#: `tail` = 末段（hidden→token）。
SEGMENT_ROLES = ("head", "middle", "tail")

#: X 档只允许不压缩的线上档位（见模块文档：量化档属 Y 档）。
SUPPORTED_QUANT_MODES = ("none",)


def stable_error_code(value: object, *, fallback: str = RELAY_PROTOCOL_ERROR) -> str:
    """把任意异常/字符串收敛成**白名单内**的稳定码。

    与 `relay_transport._safe_error_code` 同口径（`getattr(exc, "code", None) or str(exc)`），
    但**额外**做两件事：① `fallback` 本身也必须是白名单码（否则回落 `relay_internal_error`，
    保证返回值**永远**在白名单内）；② 非字符串/非白名单一律回落 —— 调用方因此可以无条件信任
    返回值可以上 wire。
    """
    candidate = getattr(value, "code", None) or str(value)
    resolved_fallback = fallback if fallback in _RELAY_ERROR_CODES else "relay_internal_error"
    return candidate if candidate in _RELAY_ERROR_CODES else resolved_fallback


class RelaySegmentError(RuntimeError):
    """段调用失败。`code` 保证是稳定码；`str(exc)` 由 `detail`（可选，**只给调度层读**）或 `code` 组成。

    `detail` 的用途：调度层要在 `_fallback_reason` 里写出可读原因（例如
    `relay_segment_failed:runner_failed`），而这类"带前缀的可读文本"**不是**线上的稳定码 ——
    所以它只进 `str(exc)`，**绝不**进 `code`（`code` 永远在白名单内，可安全上 wire）。
    """

    def __init__(self, code: object, *, role: str = "", endpoint: str = "",
                 detail: object = "") -> None:
        self.code = stable_error_code(code)
        self.role = str(role)
        self.endpoint = str(endpoint)
        self.detail = str(detail or "")
        label = f"{self.role}@{self.endpoint}" if self.role else self.endpoint
        message = self.detail or self.code
        super().__init__(f"{message}#{label}" if label else message)


@dataclass(frozen=True)
class RelaySegmentOutcome:
    """一次段往返的产物 + X 档要求的 5 个可观测字段（`to_metrics()`）。"""

    hidden: bytes = b""
    token: int | None = None
    n_tokens: int = 0
    role: str = ""
    endpoint: str = ""
    frames: int = 0
    payload_bytes: int = 0
    elapsed_ms: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def to_metrics(self) -> dict[str, object]:
        """X 档的 metrics 契约（调度层直接并进既有 metrics 字典）。

        字段名固定为 `relay_segment` / `relay_frames` / `relay_tokens` /
        `relay_payload_bytes` / `relay_error`，与 `scheduler_pipeline` 既有的
        `distributed_used`/`fallback_reason` 风格一致。
        """
        return {
            "relay_segment": f"{self.role}@{self.endpoint}" if self.role else self.endpoint,
            "relay_frames": int(self.frames),
            "relay_tokens": int(self.n_tokens),
            "relay_payload_bytes": int(self.payload_bytes),
            "relay_error": self.error,
        }


class RelaySegmentClient:
    """远端 relay 段执行器：**惰性连接**、会话内复用、`close()` 幂等。

    惰性连接是有意的：段不可达必须表现为一次**具名失败**（`relay_transport_error`），
    而不是"构造时就抛 OSError"或更糟的"静默返回空 hidden"。调用方据此可以走具名回退
    （`_fallback_reason="relay_segment_failed:{code}"`）而不必区分异常类型。
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        n_embd: int,
        role: str,
        timeout: float = 60.0,
        max_tokens: int = RELAY_DEFAULT_MAX_TOKENS,
        quant: str = "none",
        deadline_monotonic: float | None = None,
    ) -> None:
        if role not in SEGMENT_ROLES:
            raise ValueError(f"未知段角色 {role!r}；只允许 {SEGMENT_ROLES}")
        if int(n_embd) < 1:
            raise ValueError("n_embd 必须 >= 1")
        if quant not in SUPPORTED_QUANT_MODES:
            # X 档不含量化档：**显式拒绝**而不是静默按 none 走（静默降级会掩盖配置错误）。
            raise ValueError(
                f"X 档只允许 quant={SUPPORTED_QUANT_MODES}，实得 {quant!r}；"
                "量化档（f16/int8_block128/int4_block128）属 Y 档"
            )
        self.host = str(host)
        self.port = int(port)
        self.n_embd = int(n_embd)
        self.role = str(role)
        self.timeout = float(timeout)
        self.max_tokens = int(max_tokens)
        self.quant = str(quant)
        self._deadline_monotonic: float | None = None
        self.tighten_deadline(deadline_monotonic)
        self._client: RelayTcpClient | None = None
        self.frames = 0
        self.payload_bytes = 0

    # ---- 生命周期 ---------------------------------------------------------

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def deadline_monotonic(self) -> float | None:
        return self._deadline_monotonic

    def tighten_deadline(self, deadline_monotonic: float | None) -> None:
        if deadline_monotonic is None:
            return
        candidate = float(deadline_monotonic)
        if self._deadline_monotonic is None or candidate < self._deadline_monotonic:
            self._deadline_monotonic = candidate
        client = getattr(self, "_client", None)
        if client is not None:
            client.tighten_deadline(self._deadline_monotonic)

    def _ensure_client(
        self, deadline_monotonic: float | None = None
    ) -> RelayTcpClient:
        """建立（或复用）连接；失败一律收敛成稳定码（**绝不上抛原始异常**）。"""
        self.tighten_deadline(deadline_monotonic)
        if (
            self._deadline_monotonic is not None
            and self._deadline_monotonic <= time.monotonic()
        ):
            raise RelaySegmentError(
                REQUEST_DEADLINE_EXCEEDED,
                role=self.role,
                endpoint=self.endpoint,
            )
        if self._client is None:
            try:
                self._client = RelayTcpClient(
                    self.host, self.port, n_embd=self.n_embd,
                    timeout=self.timeout, max_tokens=self.max_tokens,
                    deadline_monotonic=self._deadline_monotonic,
                )
            except RelayProtocolError as exc:
                raise RelaySegmentError(exc, role=self.role, endpoint=self.endpoint) from None
            except OSError as exc:  # 连不上/超时 ⇒ 具名传输错误（不是"空 hidden"）
                raise RelaySegmentError(
                    RELAY_TRANSPORT_ERROR, role=self.role, endpoint=self.endpoint
                ) from exc
        else:
            self._client.tighten_deadline(self._deadline_monotonic)
        return self._client

    def _discard_client(self) -> None:
        client, self._client = self._client, None
        if client is None:
            return
        try:
            client.abort()
        except OSError:
            pass

    def close(self, *, grace_timeout: float | None = None) -> None:
        """幂等关闭；关闭失败只影响本对象后续可用性，不上抛。"""
        client, self._client = self._client, None
        if client is None:
            return
        try:
            client.close(grace_timeout=grace_timeout)
        except (RelayProtocolError, OSError):
            pass

    def __enter__(self) -> "RelaySegmentClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    # ---- 三种段角色 -------------------------------------------------------

    def forward_hidden(
        self,
        hidden: bytes,
        *,
        n_tokens: int,
        seq_meta: dict[str, object] | None = None,
        deadline_monotonic: float | None = None,
    ) -> RelaySegmentOutcome:
        """中段往返：hidden → hidden。

        `seq_meta` 给了就按 **多序列**（`HIDDEN_SEQ`，逐 token 显式 `seq_ids`/`positions`）发送；
        没给就走单序列 `HIDDEN`。两者都是**请求侧**选择，响应侧语义相同。
        """
        def _call(client: RelayTcpClient) -> bytes:
            if seq_meta is None:
                return client.request_hidden(
                    hidden,
                    n_tokens=int(n_tokens),
                    quant=self.quant,
                    deadline_monotonic=deadline_monotonic,
                )
            return client.request_hidden_seq(hidden, n_tokens=int(n_tokens), meta=seq_meta,
                                             quant=self.quant,
                                             deadline_monotonic=deadline_monotonic)

        return self._run(
            _call,
            n_tokens=int(n_tokens),
            want="hidden",
            deadline_monotonic=deadline_monotonic,
        )

    def forward_hidden_to_token(
        self, hidden: bytes, *, n_tokens: int,
        seq_meta: dict[str, object] | None = None,
        deadline_monotonic: float | None = None,
    ) -> RelaySegmentOutcome:
        """末段往返：hidden → token（远端自己跑完本段并回 argmax）。"""
        def _call(client: RelayTcpClient) -> int:
            return client.request_token(hidden, n_tokens=int(n_tokens), quant=self.quant,
                                        seq_meta=seq_meta,
                                        deadline_monotonic=deadline_monotonic)

        return self._run(
            _call,
            n_tokens=int(n_tokens),
            want="token",
            deadline_monotonic=deadline_monotonic,
        )

    def forward_tokens(
        self,
        tokens: "list[int]",
        *,
        deadline_monotonic: float | None = None,
    ) -> RelaySegmentOutcome:
        """上游段往返：token ids → hidden（远端跑它自己的 head 段）。

        ⚠️ 远端语义必须与本机 keep-head 一致（末层输出、`output_norm` **之前**）——
        否则接力首步即分叉（既有证据：裸 `forward_layers_to_hidden` 会在 `first_mismatch=2`
        分叉，故上游只能用 keep-head 语义的段）。
        """
        values = [int(t) for t in tokens]

        def _call(client: RelayTcpClient) -> bytes:
            return client.request_hidden_from_tokens(
                values, deadline_monotonic=deadline_monotonic
            )

        return self._run(
            _call,
            n_tokens=len(values),
            want="hidden",
            deadline_monotonic=deadline_monotonic,
        )

    # ---- 统一执行 + 收敛 ------------------------------------------------

    def _run(
        self,
        call,
        *,
        n_tokens: int,
        want: str,
        deadline_monotonic: float | None = None,
    ) -> RelaySegmentOutcome:
        """执行一次往返，把**所有**失败收敛成 `RelaySegmentOutcome.error`（稳定码）。"""
        started = time.perf_counter()
        try:
            client = self._ensure_client(deadline_monotonic)
            value = call(client)
        except RelaySegmentError as exc:
            return self._failure(exc.code, n_tokens=n_tokens, started=started)
        except RelayProtocolError as exc:
            return self._failure(stable_error_code(exc), n_tokens=n_tokens, started=started)
        except OSError:
            return self._failure(RELAY_TRANSPORT_ERROR, n_tokens=n_tokens, started=started)
        except Exception as exc:  # noqa: BLE001 - 边界处必须收敛；详情只进日志
            # 绝不把 str(exc) 外传：这里返回的一定是白名单码。
            return self._failure(stable_error_code(exc, fallback="relay_internal_error"),
                                 n_tokens=n_tokens, started=started)

        elapsed_ms = (time.perf_counter() - started) * 1000
        self.frames += 1
        if want == "hidden":
            payload = bytes(value)
            self.payload_bytes += len(payload)
            return RelaySegmentOutcome(
                hidden=payload, n_tokens=int(n_tokens), role=self.role,
                endpoint=self.endpoint, frames=1, payload_bytes=len(payload),
                elapsed_ms=elapsed_ms,
            )
        token = int(value)
        if token < 0:
            # 与 `relay_transport` 客户端同口径：负 token 即 runner 失败（不是"合法空输出"）。
            return self._failure("runner_failed", n_tokens=n_tokens, started=started)
        return RelaySegmentOutcome(
            token=token, n_tokens=int(n_tokens), role=self.role, endpoint=self.endpoint,
            frames=1, payload_bytes=0, elapsed_ms=elapsed_ms,
        )

    def _failure(self, code: object, *, n_tokens: int, started: float) -> RelaySegmentOutcome:
        # Any failed round-trip may have consumed or emitted a partial frame.
        # Drop the connection immediately so no later step can reuse it.
        self._discard_client()
        return RelaySegmentOutcome(
            n_tokens=int(n_tokens), role=self.role, endpoint=self.endpoint,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            error=stable_error_code(code),
        )
