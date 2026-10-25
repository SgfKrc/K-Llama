"""Production composition of frozen deployments, native ports and request ownership."""
from __future__ import annotations
from contextlib import contextmanager
import json
import threading

from qlh_core.adapters.request_repository import SqliteRequestRepository
from qlh_core.contracts.deadline import coerce_request_deadline
from qlh_core.contracts.deployment import canonical
from qlh_core.contracts.execution import ExecutionPlan, Segment, SamplingPolicy, ExecutionError, StageReceipt
from qlh_core.execution.requests import RequestService
from qlh_core.execution.linear import LinearExecutor
from qlh_core.execution.stopping import native_thinking_suppression_required


def build_requests(scheduler, store=None):
    from local_store import _connect, _write_connection
    def connect():
        if store is None:
            return _connect()
        with store.scope():
            return _connect()
    @contextmanager
    def transaction():
        if store is None:
            with _write_connection() as connection:
                yield connection
        else:
            with store.scope(), _write_connection() as connection:
                yield connection
    repository = SqliteRequestRepository(connect=connect, transaction=transaction)
    service = RequestService(repository, boot_id=scheduler._model_control.boot_id)
    return ExecutionApplication(scheduler, service)


class ExecutionApplication:
    def __init__(self, scheduler, requests):
        self.scheduler = scheduler
        self.requests = requests

    def handles(self):
        owner = self.scheduler._model_control
        state = getattr(owner, "execution_snapshot", owner.snapshot)()
        deployment = state["deployments"].get(state["active_deployment"], {})
        local = deployment.get("local_assignment") or (state.get("lifecycle") or {}).get("local_assignment") or {}
        return bool(local.get("load_spec"))

    def current(self, plan):
        scheduler = self.scheduler
        if plan.boot_id != scheduler._model_control.boot_id:
            return False
        for segment in plan.segments[1:]:
            worker = scheduler._task_worker_control.worker_snapshot(segment.node_id) or {}
            if worker.get("capabilities", {}).get("node_incarnation") != segment.incarnation:
                return False
        inputs = None
        transaction = getattr(scheduler, "_pipeline_load_transaction", None)
        if transaction is not None:
            inputs = scheduler._model_control_input_revisions(transaction)
        return scheduler._model_control.runnable(plan.deployment_id,
                    authority=scheduler._model_control_authority(),
                    connections=scheduler._model_control_connections(), input_revisions=inputs)

    def freeze(self):
        scheduler = self.scheduler
        owner = scheduler._model_control
        state = getattr(owner, "execution_snapshot", owner.snapshot)()
        key = state["active_deployment"]
        deployment = state["deployments"].get(key)
        if not deployment or deployment["status"] != "active" or not deployment.get("published"):
            raise ExecutionError("execution_deployment_unavailable")
        lifecycle = state["lifecycle"]
        local = deployment.get("local_assignment")
        if not local:
            raise ExecutionError("execution_master_prefix_missing")
        segments = [Segment.freeze(scheduler.get_effective_node_id(), local, 0, local=True, incarnation=deployment["boot_id"])]
        for node, value in deployment["assignments"].items():
            worker = scheduler._task_worker_control.worker_snapshot(node) or {}
            incarnation = worker.get("capabilities", {}).get("node_incarnation", "")
            if not incarnation:
                raise ExecutionError("execution_node_incarnation_unsupported")
            segments.append(Segment.freeze(node, value, deployment["connections"].get(node, 0), incarnation=incarnation))
        segments[1:] = sorted(segments[1:], key=lambda s: s.start)
        plan = ExecutionPlan(key, deployment["boot_id"], canonical(deployment["authority"]),
                             lifecycle["total_layers"], tuple(segments))
        if not self.current(plan):
            raise ExecutionError("execution_deployment_fenced")
        for segment in plan.segments[1:]:
            worker = scheduler._task_worker_control.worker_snapshot(segment.node_id) or {}
            if worker.get("capabilities", {}).get("stage_kv_cleanup") is not True:
                raise ExecutionError("execution_kv_cleanup_unsupported")
        return plan

    def run(self, prompt, **kwargs):
        try:
            sampling = SamplingPolicy(float(kwargs.get("temperature", 0.0)), float(kwargs.get("top_p", 1.0)))
            sampling.require_greedy()
            plan = self.freeze()
            deadline = coerce_request_deadline(kwargs.get("_request_deadline"),
                                               timeout_seconds=kwargs.get("_request_timeout_seconds", 120))
            ports = NativePorts(self, plan, prompt, kwargs)
            from qlh_core.contracts.speculation import resolve_speculative_gamma
            config = getattr(self.scheduler, "_runtime_config", None)
            if config is not None:
                environment = config.environment_snapshot()
            else:
                import os
                environment = os.environ
            gamma = resolve_speculative_gamma(kwargs.get("speculative_gamma"), environment=environment)
            if gamma > 1:
                if kwargs.get("routing_preference") == "local_only":
                    raise ExecutionError("speculative_route_unsupported")
                from qlh_core.execution.speculative import SpeculativeExecutor
                executor = SpeculativeExecutor(ports, gamma=gamma)
            else:
                executor = LinearExecutor(ports)
            stops = kwargs.get("stop") or ()
            if not isinstance(stops, (str, list, tuple)) or (not isinstance(stops, str) and any(not isinstance(s, str) for s in stops)):
                raise ExecutionError("route_a_stop_configuration_invalid:non_string_stop")
            if isinstance(stops, str):
                stops = (stops,)
            # Engine defaults include turn/role boundaries as well as caller stops.
            merge = getattr(self.scheduler._host, "_merge_stop_sequences", None)
            if callable(merge):
                stops = merge(list(stops))
            result = self.requests.run(plan=plan, deadline=deadline,
                cancel_event=kwargs.get("_cancel_event"), terminal=kwargs.get("_terminal_outcome"),
                context_revision=kwargs.get("_context_revision"), sampling=sampling,
                execute=lambda record: executor.run(plan, record,
                    max_tokens=kwargs.get("max_new_tokens", 1024), sampling=sampling,
                    stops=stops, emit=ports.emit), cleanup=ports.cleanup)
            result["execution_owner"] = "linear"
            if not result.get("error"):
                result["full_text"] = ports.model_prompt + result.get("full_text", "")
                import json
                identities = [json.loads(s.payload).get("worker_prepare_identity", {}) for s in plan.segments[1:]]
                selections = {(i.get("operation_id"), i.get("config_id"), i.get("generation")) for i in identities}
                if len(selections) == 1 and all(next(iter(selections))):
                    operation_id, config_id, generation = next(iter(selections))
                    result["metrics"].update(model_operation_id=operation_id, model_config_id=config_id, model_generation=generation)
                callbacks = self.scheduler._require_callbacks()
                text, thinking = callbacks.format_model_response(result["response"], kwargs.get("show_thinking", False),
                                                                 native_thinking_prompt=ports.native_thinking)
                result.update(response=text, thinking=thinking)
                if kwargs.get("_stream_callback"):
                    kwargs["_stream_callback"]({"done": True, **result})
            return result
        except (ExecutionError, ValueError) as exc:
            code = getattr(exc, "code", str(exc))
            from qlh_core.contracts.snapshots import reason
            return {"response": "", "error": code, "reason_code": code,
                    "reason_detail":reason("execution", code, "execution_admission", cause=getattr(exc, "reason_detail", None)), "execution_owner": "linear",
                    "metrics": {"distributed_used": False, "fallback": False}}


class NativePorts:
    def __init__(self, application, plan, prompt, options):
        self.application, self.plan, self.prompt, self.options = application, plan, prompt, options
        self.scheduler = application.scheduler
        self.host = self.scheduler._host
        self.cache = None
        self.forward_schedule = []
        self.used = []
        self.lock_held = False
        self.native_thinking = False
        self.thinking_buffer = ""
        self.prefix_started = False
        self.attempts = []
        self._receipt_guard = LinearExecutor(self)

    def current(self, plan):
        return self.application.current(plan)

    def prepare(self, plan, record):
        deadline = record["deadline"]
        if not self.host.runtime_controller.execution_lock.acquire(timeout=deadline.require_remaining()):
            raise ExecutionError("request_deadline_exceeded")
        self.lock_held = True
        if not self.current(plan):
            raise ExecutionError("execution_deployment_fenced")
        actual = getattr(self.host, "layer_range", None)
        expected = (plan.segments[0].start, plan.segments[0].end)
        if tuple(actual or ()) != expected:
            raise ExecutionError("execution_local_assignment_changed")
        self.tokenizer = self.host.tokenizer
        callbacks = self.scheduler._require_callbacks()
        thinking = self.options.get("show_thinking", False)
        self.model_prompt = callbacks.build_model_chat_prompt(self.tokenizer,
            self.options.get("messages") or [{"role": "user", "content": self.prompt}],
            system_prompt=callbacks.thinking_system_prompt if thinking else None,
            assistant_prefill="思考\n" if thinking else None,
            **({"enable_thinking": self.options["enable_thinking"]} if self.options.get("enable_thinking") is not None else {}))
        self.native_thinking = native_thinking_suppression_required(thinking, self.model_prompt)
        self.suppress_thinking = self.native_thinking
        inputs = self.tokenizer(self.model_prompt, return_tensors="pt")
        self.input_ids, self.attention = inputs["input_ids"], inputs.get("attention_mask")
        self.prompt_length = int(self.input_ids.shape[1])
        local = json.loads(plan.segments[0].payload)
        self.context_size = local["load_spec"]["context_size"]
        if self.prompt_length + self.options.get("max_new_tokens", 1024) > self.context_size:
            raise ExecutionError("execution_context_size_exceeded")

    def prepare_speculation(self, plan, record):
        """Admit the exact active owners while holding the request execution lock."""
        from qlh_core.contracts.speculation import require_speculative_plan
        from gguf_metadata_contract import read_gguf_header
        descriptor = self.host.get_pipeline_descriptor()
        require_speculative_plan(plan, descriptor, self.host.engine_type)
        try:
            metadata, tensors = read_gguf_header(descriptor.get("loaded_artifact") or descriptor["model_path"])
        except (OSError, ValueError, TypeError):
            raise ExecutionError("speculative_model_metadata_invalid") from None
        if (metadata.get("general.architecture") != "qwen2"
                or metadata.get("qwen2.full_attention_interval", 1) != 1
                or metadata.get("qwen2.nextn_predict_layers", 0) != 0
                or any("ssm" in key.lower() or "recurrent" in key.lower()
                       or ("sliding_window" in key.lower() and value != 0)
                       for key, value in metadata.items())
                or any("ssm" in name.lower() for name in tensors)):
            raise ExecutionError("speculative_attention_state_unsupported")
        prefix = plan.segments[0]
        if (metadata.get("qwen2.block_count") != prefix.end - prefix.start
                or metadata.get("qwen2.embedding_length") != descriptor.get("hidden_size")):
            raise ExecutionError("speculative_model_metadata_invalid")
        required_batch = self.prompt_length + self.options.get("max_new_tokens", 1024) - 1
        for segment in plan.segments:
            spec = json.loads(segment.payload)["load_spec"]
            if spec["batch_size"] < required_batch:
                raise ExecutionError("speculative_replay_batch_limit")
        tail = plan.segments[-1]
        worker = self.scheduler._task_worker_control.worker_snapshot(tail.node_id) or {}
        if worker.get("capabilities", {}).get("multi_position_argmax") is not True:
            raise ExecutionError("speculative_multi_position_unsupported")
        LinearExecutor._check(self._receipt_guard, plan, record)

    def prompt_tokens(self):
        return self.input_ids.reshape(-1).tolist()

    def forward_tokens(self, plan, record, ids, position, step, *, all_tokens=False):
        """Send one contiguous causal batch through every frozen segment."""
        import numpy as np
        self._receipt_guard._check(plan, record)
        if (not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids)
                or type(position) is not int or position < 0 or position + len(ids) > self.context_size):
            raise ExecutionError("speculative_input_invalid")
        if position and self.cache != position:
            raise ExecutionError("speculative_prefix_position_mismatch")
        self.prefix_started = True
        result = self.host.forward_layers(input_ids=np.array([ids], dtype=np.int64),
            attention_mask=None, past_key_values=self.cache, use_cache=True, apply_lm_head=False)
        self._receipt_guard._check(plan, record)
        self.cache = result.get("cache")
        if self.cache is None:
            self.cache = result.get("past_key_values")
        if type(self.cache) is not int or self.cache != position + len(ids):
            raise ExecutionError("speculative_prefix_position_mismatch")
        hidden = result.get("hidden_states")
        if hasattr(hidden, "detach"):
            hidden = hidden.detach().float().cpu().numpy()
        hidden = np.ascontiguousarray(hidden, dtype=np.float32)
        if hidden.ndim == 3 and hidden.shape[0] == 1:
            hidden = hidden[0]
        if hidden.ndim != 2 or hidden.shape[0] != len(ids):
            raise ExecutionError("execution_hidden_shape_invalid")
        positions = list(range(position, position + len(ids)))
        for segment in plan.segments[1:]:
            self._receipt_guard._check(plan, record)
            from qlh_core.contracts.execution import StageIdentity
            identity = StageIdentity.for_segment(plan, segment, record["request_id"], record["attempt_id"], step)
            receipt = self.stage(segment, identity, hidden, record, positions=positions,
                                 all_tokens=all_tokens and segment.head)
            self._receipt_guard._check(plan, record)
            hidden = self._receipt_guard.accept(identity, receipt)
        if all_tokens:
            from qlh_core.contracts.token_output import validate_token_output
            validate_token_output(hidden, n_tokens=len(ids), positions=positions, require_all=True)
        self.forward_schedule.append({"position": position, "ids": list(ids),
                                      "all_tokens": all_tokens})
        return hidden

    def _restore_chunks(self, ids):
        if not isinstance(ids, list) or not ids:
            raise ExecutionError("speculative_restore_schedule_invalid")
        chunks, cursor = [], 0
        for call in self.forward_schedule:
            if cursor == len(ids):
                break
            if call["position"] != cursor:
                raise ExecutionError("speculative_restore_schedule_invalid")
            count = min(len(call["ids"]), len(ids) - cursor)
            selected = call["ids"][:count]
            if selected != ids[cursor:cursor + count]:
                raise ExecutionError("speculative_restore_schedule_mismatch")
            chunks.append({"position": cursor, "ids": selected,
                           "all_tokens": call["all_tokens"]})
            cursor += count
        if cursor != len(ids):
            raise ExecutionError("speculative_restore_schedule_incomplete")
        return chunks

    def restore_operation_count(self, ids):
        return len(self._restore_chunks(ids))

    def restore_prefix(self, plan, record, ids, step, *, operation_steps):
        """Reset every owner, then replay the request's committed batch schedule."""
        self._receipt_guard._check(plan, record)
        chunks = self._restore_chunks(ids)
        if operation_steps != len(chunks):
            raise ExecutionError("speculative_restore_schedule_invalid")
        if not self._drain_attempts(record):
            raise ExecutionError("speculative_restore_cleanup_pending")
        self._reset_pipeline_kv(record, tag=f"restore:{step}", check=True)
        self._receipt_guard._check(plan, record)
        self.forward_schedule = []
        output = None
        for offset, chunk in enumerate(chunks):
            output = self.forward_tokens(plan, record, chunk["ids"], chunk["position"],
                step + offset, all_tokens=chunk["all_tokens"])
        from qlh_core.contracts.token_output import tail_token
        return tail_token(output)

    def eos_ids(self):
        value = getattr(self.tokenizer, "eos_token_id", None)
        get_eos = getattr(self.host, "_get_generation_eos_token_ids", None)
        if callable(get_eos):
            value = get_eos([])
        if type(value) is int:
            return (value,) if value >= 0 else ()
        return tuple(v for v in (value or ()) if type(v) is int and v >= 0)

    def prefix(self, segment, record, step, token):
        import numpy as np
        ids = self.input_ids
        if step:
            if hasattr(ids, "detach"):
                import torch
                ids = torch.tensor([[token]], dtype=torch.long)
            else:
                ids = np.array([[token]], dtype=np.int64)
            if self.cache is None:
                raise ExecutionError("execution_decode_missing_kv")
        self.prefix_started = True
        result = self.host.forward_layers(input_ids=ids, attention_mask=self.attention if step == 0 else None,
            past_key_values=self.cache, use_cache=True, apply_lm_head=False)
        self.cache = result.get("cache")
        if self.cache is None:
            self.cache = result.get("past_key_values")
        hidden = result.get("hidden_states")
        if hidden is None:
            raise ExecutionError("execution_prefix_missing_hidden")
        if hasattr(hidden, "detach"):
            hidden = hidden.detach().float().cpu().numpy()
        hidden = np.ascontiguousarray(hidden, dtype=np.float32)
        if hidden.ndim == 3 and hidden.shape[0] == 1:
            hidden = hidden[0]
        if hidden.ndim != 2:
            raise ExecutionError("execution_hidden_shape_invalid")
        return hidden

    def stage(self, segment, identity, hidden, record, *, positions=None, all_tokens=None):
        from execution_provider import ModelIdentity
        assignment = json.loads(segment.payload)
        frozen = assignment["worker_prepare_identity"]
        if positions is None:
            start = 0 if identity.step == 0 else self.prompt_length + identity.step - 1
            positions = list(range(start, start + len(hidden)))
        if all_tokens is None:
            all_tokens = segment.head and self.options.get("_all_position_argmax") is True
        self.used.append(segment)
        value = self.scheduler._execute_layer_stage_offer(node_id=segment.node_id, assignment=assignment,
            hidden_states=hidden, model_identity=ModelIdentity(**frozen["stage_model_identity"]),
            workflow_id="wf_" + record["request_id"], request_id=f'{record["request_id"]}:step:{identity.step}',
            stage_id=f'{segment.node_id}:step:{identity.step}', context_size=self.context_size,
            want_hidden=not segment.head,
            want_all_tokens=segment.head and all_tokens,
            middle_channel="keep_head_layer_out", positions=positions,
            seq_ids=[0] * len(hidden), cancel_event=record["cancel"], request_deadline=record["deadline"],
            execution_attempts=self.attempts)
        output = value if segment.head else value["hidden_states"]
        if segment.head:
            from qlh_core.contracts.token_output import tail_token
            tail_token(output)
            raw = json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
        else:
            # The wire decoder returns a Tensor when torch is loaded and a
            # NumPy array in edge builds. Relay one canonical f32 byte layout.
            import numpy as np
            if hasattr(output, "detach"):
                output = output.detach().float().cpu().numpy()
            output = np.ascontiguousarray(output, dtype=np.float32)
            raw = output.tobytes()
        import hashlib
        return StageReceipt(identity, hashlib.sha256(raw).hexdigest(), output)

    def head(self, hidden):
        import numpy as np
        if getattr(self.host, "engine_type", "") == "pytorch":
            import torch
            hidden = torch.from_numpy(hidden).unsqueeze(0).to(self.host.get_device())
            logits = self.host.forward_lm_head(hidden)
            return int(logits[0, -1].argmax().item())
        logits = self.host.forward_lm_head(hidden)
        return int(np.asarray(logits).reshape(-1, np.asarray(logits).shape[-1])[-1].argmax())

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def emit(self, event):
        callback = self.options.get("_stream_callback")
        if not callback:
            return
        if self.suppress_thinking:
            self.thinking_buffer += event["token"]
            marker = self.thinking_buffer.lower().find("</think>")
            if marker < 0:
                return
            event = {"token": self.thinking_buffer[marker + 8:]}
            self.thinking_buffer = ""
            self.suppress_thinking = False
        if event["token"]:
            callback(event)

    def _drain_attempts(self, record):
        for provider, attempt_id in self.attempts:
            while not provider.execution_drained(attempt_id):
                if record["deadline"].expired():
                    return False
                threading.Event().wait(min(0.02, record["deadline"].remaining()))
        return True

    def _reset_pipeline_kv(self, record, *, tag, check=False):
        import numpy as np
        from execution_provider import ModelIdentity
        errors = []
        try:
            for segment in dict.fromkeys(self.used):
                try:
                    if check:
                        self._receipt_guard._check(self.plan, record)
                    record["deadline"].require_remaining()
                    assignment = json.loads(segment.payload)
                    frozen = assignment["worker_prepare_identity"]
                    receipt = self.scheduler._execute_layer_stage_offer(node_id=segment.node_id, assignment=assignment,
                        hidden_states=np.zeros((1, frozen["hidden_size"]), dtype=np.float32),
                        model_identity=ModelIdentity(**frozen["stage_model_identity"]),
                        workflow_id="wf_" + record["request_id"], request_id=record["request_id"] + ":" + tag,
                        stage_id=segment.node_id + ":" + tag, context_size=self.context_size,
                        want_hidden=not segment.head, middle_channel="keep_head_layer_out",
                        seq_ids=[0], positions=[0], cancel_event=record["cancel"] if check else threading.Event(),
                        request_deadline=record["deadline"], cleanup_only=True,
                        execution_attempts=self.attempts)
                    if receipt.get("kv_cleared") is not True:
                        raise ExecutionError("execution_kv_cleanup_receipt_invalid")
                except Exception as exc:
                    if check:
                        raise
                    # A failed owner must not prevent healthy later owners from
                    # releasing KV while the same deadline still has budget.
                    errors.append(exc)
        finally:
            self.cache = None
            if self.prefix_started:
                try:
                    reset = getattr(self.host, "clear_pipeline_kv", None)
                    if callable(reset):
                        reset()
                    elif getattr(self.host, "engine_type", "") == "llama_cpp":
                        raise ExecutionError("execution_kv_cleanup_owner_missing")
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise errors[0]

    def cleanup(self, record):
        clean = True
        try:
            if not self._drain_attempts(record):
                clean = False
                # A still-running remote attempt cannot share KV with cleanup.
                self.cache = None
                if self.prefix_started:
                    self.host.clear_pipeline_kv()
            else:
                self._reset_pipeline_kv(record, tag="cleanup")
        except Exception:
            clean = False
        finally:
            self.cache = None
            if self.lock_held:
                self.host.runtime_controller.execution_lock.release()
                self.lock_held = False
        return clean
