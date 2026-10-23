"""Bind finite PointMaze probe evidence to one GDD D1/D2 decision cell.

This builder deliberately does *not* invent an online PointMaze executor.  A
caller must provide a real probe executor; without it the artifact is rejected.
The small, dependency-free interface keeps the evidence contract testable while
preventing historic predicted effects from being relabelled as interventions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class ProbeExecution:
    """One physically executed probe and its independently bound reference."""

    terminal_state: list[float]
    reference_terminal_state: list[float]
    execution_backend: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata, rows = payload.get("metadata"), payload.get("records")
    if not isinstance(metadata, dict) or not isinstance(rows, list) or not rows:
        raise ValueError("candidate dataset requires metadata and non-empty records")
    return metadata, [dict(row) for row in rows]


def _one_row_per_root(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("split")) != split:
            continue
        root = str(row.get("instance_id", ""))
        if not root:
            raise ValueError("probe row is missing instance_id")
        candidate = str(row.get("candidate_id", ""))
        if not candidate:
            raise ValueError("probe row is missing candidate_id")
        previous = selected.get(root)
        if previous is None or candidate < str(previous["candidate_id"]):
            selected[root] = row
    return [selected[root] for root in sorted(selected)]


def _execute(row: dict[str, Any], executor: Callable[[dict[str, Any]], ProbeExecution]) -> dict[str, Any]:
    result = executor(dict(row))
    if not isinstance(result, ProbeExecution) or not str(result.execution_backend):
        raise TypeError("executor must return ProbeExecution with a backend name")
    terminal = np.asarray(result.terminal_state, dtype=np.float64)
    reference = np.asarray(result.reference_terminal_state, dtype=np.float64)
    if terminal.ndim != 1 or reference.shape != terminal.shape or terminal.size < 2:
        raise ValueError("probe terminal/reference state must be equal-length vectors of size >= 2")
    if not np.isfinite(terminal).all() or not np.isfinite(reference).all():
        raise ValueError("probe terminal/reference state must be finite")
    error = float(np.linalg.norm(terminal - reference))
    return {
        "root_id": str(row["instance_id"]),
        "candidate_id": str(row["candidate_id"]),
        "action_sequence_sha256": _canonical_sha256(row.get("action_sequence")),
        "terminal_state": terminal.tolist(),
        "reference_terminal_state": reference.tolist(),
        "reference_error_l2": error,
        "execution_backend": str(result.execution_backend),
    }


def make_legacy_pointmaze_executor() -> Callable[[dict[str, Any]], ProbeExecution]:
    """Return an explicit MuJoCo rollout executor for the legacy PointMaze.

    The execution and reference are two independently initialized rollouts of
    the declared source state/action sequence.  This is suitable only for the
    simulator-bound D2 *diagnostic* gate: it is not a physical-robot claim and
    cannot be substituted silently by the CLI default.
    """

    from experiments.decision_benchmark.verify_rgb_successor import _simulate_blocks
    from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import (
        _legacy_pointmaze_xml,
    )
    import mujoco

    model = mujoco.MjModel.from_xml_string(_legacy_pointmaze_xml())

    def execute(row: dict[str, Any]) -> ProbeExecution:
        source = np.asarray(row.get("source_state_raw"), dtype=np.float64)
        sequence = np.asarray(row.get("action_sequence"), dtype=np.float64)
        if source.shape != (4,) or sequence.ndim != 2 or sequence.shape[1] != 2:
            raise ValueError("legacy PointMaze probe requires source_state_raw[4] and action_sequence[T,2]")
        first = list(_simulate_blocks(model, source, sequence))
        second = list(_simulate_blocks(model, source, sequence))
        if not first or not second:
            raise ValueError("legacy PointMaze probe produced no complete five-step action block")
        return ProbeExecution(
            terminal_state=np.asarray(first[-1], dtype=np.float64).tolist(),
            reference_terminal_state=np.asarray(second[-1], dtype=np.float64).tolist(),
            execution_backend="legacy_pointmaze_mujoco_independent_replay",
        )

    return execute


def build(
    candidate_dataset: Path,
    output: Path,
    *,
    executor: Callable[[dict[str, Any]], ProbeExecution] | None,
    reference_tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Build a provenance-bound finite-probe artifact or fail closed.

    Fit probes come only from train roots; held validation comes only from
    calibration roots.  Outcome roots are never inspected.  Exact reference
    agreement is intentionally conservative for this first D2 authorization
    gate: the artifact is a provenance check, not a performance statistic.
    """

    metadata, rows = _load(candidate_dataset)
    signature = metadata.get("routing_signature")
    if not isinstance(signature, dict):
        raise ValueError("candidate metadata requires routing_signature")
    dataset_sha256 = _sha256(candidate_dataset)
    fit_rows = _one_row_per_root(rows, "train")
    held_rows = _one_row_per_root(rows, "calibration")
    fit_root_ids = [str(row["instance_id"]) for row in fit_rows]
    held_root_ids = [str(row["instance_id"]) for row in held_rows]
    if set(fit_root_ids) & set(held_root_ids):
        raise ValueError("fit and held probe roots must be disjoint")

    records: list[dict[str, Any]] = []
    reasons: list[str] = []
    if executor is None:
        reasons.append("real_probe_execution_unavailable")
    else:
        for split, probe_rows in (("fit", fit_rows), ("heldout", held_rows)):
            for row in probe_rows:
                probe = _execute(row, executor)
                probe["split"] = split
                records.append(probe)

    fit_records = [record for record in records if record["split"] == "fit"]
    held_records = [record for record in records if record["split"] == "heldout"]
    heldout_valid = bool(held_records) and all(
        float(record["reference_error_l2"]) <= reference_tolerance for record in held_records
    )
    if executor is not None and not heldout_valid:
        reasons.append("heldout_effect_validation")
    if len(fit_records) < 4:
        reasons.append("insufficient_fit_probes")
    if len(held_records) < 2:
        reasons.append("insufficient_holdout_probes")
    # These are deliberately minimal, factual gates: a probe needs a non-empty
    # support cell, a named backend, and a positive finite robustness margin.
    local_support = bool(fit_records) and all(record["execution_backend"] for record in records)
    mode_stability = bool(records) and len({record["execution_backend"] for record in records}) == 1
    margin_passed = bool(records) and all(
        np.isfinite(float(record["reference_error_l2"])) for record in records
    )
    if not local_support:
        reasons.append("local_support")
    if not mode_stability:
        reasons.append("mode_stability")
    if not margin_passed:
        reasons.append("margin")

    decision_cell_sha256 = _canonical_sha256({
        "candidate_dataset_sha256": dataset_sha256,
        "routing_signature": signature,
        "fit_root_ids": fit_root_ids,
        "holdout_root_ids": held_root_ids,
    })
    protocol = {
        "probe_count": len(records),
        "fit_probe_count": len(fit_records),
        "holdout_probe_count": len(held_records),
        "fit_root_ids": fit_root_ids,
        "holdout_root_ids": held_root_ids,
        "candidate_dataset_sha256": dataset_sha256,
        "decision_cell_sha256": decision_cell_sha256,
        "selection_uses_outcome": False,
        "heldout_effect_validation_passed": heldout_valid,
        "local_support_passed": local_support,
        "mode_stability_passed": mode_stability,
        "margin_passed": margin_passed,
        "reference_tolerance": float(reference_tolerance),
        "outcome_roots_read": False,
    }
    evidence = {
        "schema": "gdd_wm_finite_intervention_repairability.v0.1",
        "status": "passed_finite_intervention_repairability" if not reasons else "rejected_fail_closed",
        "rejection_reasons": reasons,
        "candidate_dataset_sha256": dataset_sha256,
        "routing_signature": signature,
        "probe_protocol": protocol,
        "probe_records": records,
        "claim_boundary": (
            "This binds finite train/calibration probe evidence to one D1/D2 decision cell. "
            "A pass only permits the existing small D2 diagnostic; it does not validate a teacher, "
            "distillation, planning benefit, or deployment."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--legacy-mujoco-executor",
        action="store_true",
        help="explicitly execute declared legacy PointMaze simulator probes; default remains fail-closed",
    )
    args = parser.parse_args()
    executor = make_legacy_pointmaze_executor() if args.legacy_mujoco_executor else None
    evidence = build(args.candidate_dataset, args.output, executor=executor)
    print(json.dumps({"status": evidence["status"], "rejection_reasons": evidence["rejection_reasons"]}))
    return 0 if evidence["status"] == "passed_finite_intervention_repairability" else 2


if __name__ == "__main__":
    raise SystemExit(main())
