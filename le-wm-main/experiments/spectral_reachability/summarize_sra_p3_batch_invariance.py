"""Fail-closed receipt reducer for P3 CUDA batch-size invariance.

This does not make a new learned-model claim.  It checks that two executions
of the *same* fixed-tree P3 protocol differ only in CUDA floating-point
rounding, rather than in selected first action or in evaluator outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same(left: Any, right: Any, *, label: str, failures: list[str]) -> None:
    if left != right:
        failures.append(label)


def compare_reports(reference: dict[str, Any], candidate: dict[str, Any], *, atol: float) -> dict[str, Any]:
    """Compare choices/outcomes exactly and model scores up to CUDA roundoff."""

    failures: list[str] = []
    _same(reference.get("schema"), "sra_p3_fixed_tree_horizon_expansion.v1", label="reference schema", failures=failures)
    _same(candidate.get("schema"), "sra_p3_fixed_tree_horizon_expansion.v1", label="candidate schema", failures=failures)
    _same(reference.get("status"), "mechanism_ready", label="reference status", failures=failures)
    _same(candidate.get("status"), "mechanism_ready", label="candidate status", failures=failures)
    _same(reference.get("gates"), candidate.get("gates"), label="gate mismatch", failures=failures)
    if not all(reference.get("gates", {}).values()) or not all(candidate.get("gates", {}).values()):
        failures.append("a learned-model report has a failed gate")

    ref_inputs, cand_inputs = reference.get("inputs", {}), candidate.get("inputs", {})
    for key in ("roots", "horizons", "checkpoint_sha256", "physical_position_backend"):
        _same(ref_inputs.get(key), cand_inputs.get(key), label=f"input mismatch: {key}", failures=failures)
    ref_batch, cand_batch = ref_inputs.get("model_batch_size"), cand_inputs.get("model_batch_size")
    if not isinstance(ref_batch, int) or not isinstance(cand_batch, int) or ref_batch < 1 or cand_batch < 1 or ref_batch == cand_batch:
        failures.append("reports must use two distinct positive model batch sizes")

    max_model_cost_abs_error = 0.0
    ref_rows, cand_rows = reference.get("rows", []), candidate.get("rows", [])
    if len(ref_rows) != len(cand_rows):
        failures.append("row count mismatch")
    for ref_row, cand_row in zip(ref_rows, cand_rows):
        root = ref_row.get("root")
        _same(root, cand_row.get("root"), label=f"root mismatch at {root}", failures=failures)
        ref_horizons, cand_horizons = ref_row.get("horizons", {}), cand_row.get("horizons", {})
        _same(sorted(ref_horizons), sorted(cand_horizons), label=f"horizon set mismatch at root {root}", failures=failures)
        for horizon, ref_value in ref_horizons.items():
            cand_value = cand_horizons.get(horizon, {})
            for key in ("tree_sequences", "spectral_index", "immediate_index", "physical_index"):
                _same(ref_value.get(key), cand_value.get(key), label=f"{key} mismatch at root {root}, H={horizon}", failures=failures)
            for key in ("physical_first_costs", "immediate_first_costs", "spectral_regret", "immediate_regret"):
                if not np.array_equal(np.asarray(ref_value.get(key)), np.asarray(cand_value.get(key))):
                    failures.append(f"physical/outcome mismatch: {key} at root {root}, H={horizon}")
            ref_model = np.asarray(ref_value.get("model_first_costs"), dtype=np.float64)
            cand_model = np.asarray(cand_value.get("model_first_costs"), dtype=np.float64)
            if ref_model.shape != cand_model.shape or not np.isfinite(ref_model).all() or not np.isfinite(cand_model).all():
                failures.append(f"invalid model costs at root {root}, H={horizon}")
                continue
            max_model_cost_abs_error = max(max_model_cost_abs_error, float(np.max(np.abs(ref_model - cand_model))))

    for horizon, value in reference.get("aggregate_spectral_minus_immediate_regret", {}).items():
        if candidate.get("aggregate_spectral_minus_immediate_regret", {}).get(horizon) != value:
            failures.append(f"aggregate outcome mismatch at H={horizon}")
    if max_model_cost_abs_error > atol:
        failures.append(f"model cost difference {max_model_cost_abs_error} exceeds atol {atol}")
    return {
        "schema": "sra_p3_batch_invariance.v1",
        "status": "pass" if not failures else "invalid",
        "interpretation": "selection and physical outcomes are batch-invariant; this is an implementation check, not a new efficacy result" if not failures else "batch-size implementation check failed; do not interpret P3 efficacy",
        "reference_model_batch_size": ref_batch,
        "candidate_model_batch_size": cand_batch,
        "max_model_first_cost_absolute_error": max_model_cost_abs_error,
        "model_cost_atol": atol,
        "gates": {
            "same_protocol_inputs": not any(item.startswith("input mismatch") or item.startswith("reports must") for item in failures),
            "all_learned_model_gates_pass": "a learned-model report has a failed gate" not in failures,
            "same_selection_indices": not any("index mismatch" in item for item in failures),
            "same_physical_outcomes": not any(item.startswith("physical/outcome mismatch") or item.startswith("aggregate outcome mismatch") for item in failures),
            "model_scores_within_cuda_roundoff_tolerance": max_model_cost_abs_error <= atol,
        },
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1.0e-4)
    args = parser.parse_args()
    if args.output.exists() or args.atol < 0:
        raise ValueError("output must be new and atol non-negative")
    result = compare_reports(_load(args.reference), _load(args.candidate), atol=args.atol)
    result["reference_report"] = str(args.reference)
    result["candidate_report"] = str(args.candidate)
    result["reference_report_sha256"] = _sha256(args.reference)
    result["candidate_report_sha256"] = _sha256(args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise RuntimeError("P3 batch invariance receipt failed")


if __name__ == "__main__":
    main()
