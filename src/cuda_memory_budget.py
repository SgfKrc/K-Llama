"""Bounded CUDA free-memory sampling with physical-device UUID matching.

WDDM may hide other processes' allocations from torch's free-memory query.
The global sample can only reduce that budget, and only for the same UUID;
CUDA ordinals and nvidia-smi row order never establish device identity.
"""
from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
from typing import Any

TORCH_BUDGET_SOURCE = "cuda.mem_get_info"
UUID_MIN_BUDGET_SOURCE = "cuda.min_mem_get_info_nvidia_smi_uuid"
_QUERY_TIMEOUT_SECONDS = 2.0
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _canonical_uuid(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    if value.startswith("gpu-"):
        value = value[4:]
    return value if _UUID.fullmatch(value) else None


def sample_cuda_memory_budget(torch: Any, device: Any) -> dict:
    """Return torch/global minimum plus JSON-safe observations, without caching.

    A failed torch query retains its original failure semantics. The optional
    global query is best effort; unavailable or ambiguous identity always leaves
    the torch budget intact. Zero is a valid sample, never a fallback signal.
    """
    torch_free = int(torch.cuda.mem_get_info(device)[0])
    observations = {"torch_free_bytes": torch_free, "device_uuid": None,
                    "nvidia_smi_free_bytes": None, "nvidia_smi_status": "uuid_missing"}
    result = {"available_bytes": torch_free, "budget_source": TORCH_BUDGET_SOURCE,
              "budget_observations": observations}
    try:
        raw_uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        observations["nvidia_smi_status"] = "device_properties_failed"
        return result
    # Recent torch builds expose _CUuuid rather than str. Only stringify the
    # UUID property itself; the resulting value must still pass strict syntax.
    if raw_uuid is not None:
        try:
            raw_uuid = str(raw_uuid)
        except Exception:
            observations["nvidia_smi_status"] = "uuid_invalid"
            return result
    device_uuid = _canonical_uuid(raw_uuid)
    if device_uuid is None:
        if raw_uuid is not None:
            observations["nvidia_smi_status"] = "uuid_invalid"
        return result
    observations["device_uuid"] = device_uuid
    try:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            observations["nvidia_smi_status"] = "unavailable"
            return result
        completed = subprocess.run(
            [executable, "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_QUERY_TIMEOUT_SECONDS, check=False, shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        observations["nvidia_smi_status"] = "query_timeout"
        return result
    except (OSError, ValueError, subprocess.SubprocessError):
        observations["nvidia_smi_status"] = "query_failed"
        return result
    if completed.returncode != 0:
        observations["nvidia_smi_status"] = "query_failed"
        return result
    output = completed.stdout
    if not isinstance(output, str) or len(output) > 65536:
        observations["nvidia_smi_status"] = "invalid_output"
        return result
    try:
        matches = []
        for row in csv.reader(io.StringIO(output), strict=True):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 2:
                observations["nvidia_smi_status"] = "invalid_output"
                return result
            if _canonical_uuid(row[0]) == device_uuid:
                matches.append(row[1].strip())
    except csv.Error:
        observations["nvidia_smi_status"] = "invalid_output"
        return result
    if len(matches) != 1:
        observations["nvidia_smi_status"] = "uuid_not_found" if not matches else "uuid_ambiguous"
        return result
    # nvidia-smi reports integer MiB with nounits. N/A, negative, fractional or
    # unit-bearing values cannot establish a byte budget for this device.
    if re.fullmatch(r"[0-9]{1,20}", matches[0]) is None:
        observations["nvidia_smi_status"] = "invalid_free"
        return result
    global_free = int(matches[0]) * 1024**2
    observations.update(nvidia_smi_free_bytes=global_free, nvidia_smi_status="matched")
    result.update(available_bytes=min(torch_free, global_free), budget_source=UUID_MIN_BUDGET_SOURCE)
    return result
