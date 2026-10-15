#!/usr/bin/env python3
"""Collect EX-N3 Gemma judge counters from a SHA-pinned, non-executing fixture.

This is a schema bridge only.  It never starts Gemma, contacts Ollama, opens
an image, or accepts completion text/reasoning as experiment evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from experiment_core.quality import QualityEvidenceError, normalize_quality_evidence


def _read_hashed_object(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("fixture SHA-256 does not match the plan")
    value = json.loads(raw)
    if not isinstance(value, Mapping):
        raise ValueError("fixture root must be an object")
    return value


def collect_evidence(
    evidence: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    expected_model: str,
    expected_contract_id: str,
    expected_contract_sha256: str,
) -> dict[str, Any]:
    """Bind counter-only evidence to a versioned judge contract."""
    allowed_evidence_fields = {
        "model", "judge_contract_id", "judge_contract_sha256",
        "topic_hit", "key_element_coverage", "manual_review",
    }
    if set(evidence) != allowed_evidence_fields:
        raise ValueError("judge evidence contains unsupported fields")
    if contract.get("schema_version") != 1 or contract.get("id") != expected_contract_id:
        raise ValueError("judge contract identity does not match the plan")
    if contract.get("model") != expected_model or evidence.get("model") != expected_model:
        raise ValueError("judge model does not match the plan")
    if evidence.get("judge_contract_id") != expected_contract_id:
        raise ValueError("judge contract id does not match the plan")
    if evidence.get("judge_contract_sha256") != expected_contract_sha256:
        raise ValueError("judge contract SHA-256 does not match the plan")
    result = {
        "model": expected_model,
        "judge_contract_id": expected_contract_id,
        "judge_contract_sha256": expected_contract_sha256,
        "topic_hit": evidence.get("topic_hit"),
        "key_element_coverage": evidence.get("key_element_coverage"),
        "manual_review": evidence.get("manual_review"),
    }
    # The shared normalizer rejects extra fields and invalid counters before write.
    normalize_quality_evidence(
        {"gemma_judge": result},
        expected_gemma_judge={
            "model": expected_model,
            "judge_contract_id": expected_contract_id,
            "judge_contract_sha256": expected_contract_sha256,
        },
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect static Gemma quality evidence")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--expected-evidence-sha256", required=True)
    parser.add_argument("--judge-contract", required=True)
    parser.add_argument("--expected-judge-contract-sha256", required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-contract-id", required=True)
    parser.add_argument("--result-file", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        contract = _read_hashed_object(
            Path(args.judge_contract), args.expected_judge_contract_sha256,
        )
        evidence = _read_hashed_object(Path(args.evidence), args.expected_evidence_sha256)
        collected = collect_evidence(
            evidence,
            contract,
            expected_model=args.expected_model,
            expected_contract_id=args.expected_contract_id,
            expected_contract_sha256=args.expected_judge_contract_sha256,
        )
        output = Path(args.result_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "quality_completed": 1,
            "quality_evidence": {"gemma_judge": collected},
        }, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError, QualityEvidenceError):
        # No report payload or filesystem path is emitted by this static bridge.
        print("Gemma quality evidence collection failed", file=sys.stderr)
        return 2
    print(json.dumps({"status": "collected"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
