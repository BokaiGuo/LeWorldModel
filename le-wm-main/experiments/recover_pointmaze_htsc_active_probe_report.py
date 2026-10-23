"""Recover a V5 active-probe aggregate after all immutable arm reports exist.

This tool is deliberately post-execution only.  It cannot launch planning,
materialize roots, or alter an arm report.  It exists for the narrow failure
mode where a strict aggregation/validation bug aborts after a host CUDA panel
has already written every per-arm report.  The recovered aggregate records
content hashes for the immutable inputs and refuses to overwrite an output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import experiments.run_pointmaze_htsc_active_probe_strict as strict


SCHEMA = "pointmaze.htsc_active_probe.aggregate_recovery.v1"
ACTIVE_METHODS = frozenset({
    "effect_adapter_rgb_online_verified_local",
    "effect_adapter_rgb_online_affine_verified_local",
    "effect_adapter_rgb_online_affine_candidate_anchored_verified_local",
})


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arm_path(run_dir: Path, method: str) -> Path:
    path = run_dir / "held_out" / f"{method}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing immutable held-out arm report: {path}")
    return path


def _rows_for_arm(path: Path, method: str, expected_roots: list[int]) -> list[dict[str, Any]]:
    payload = _load(path)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: missing rows")
    if any(not isinstance(row, dict) or str(row.get("method")) != method for row in rows):
        raise ValueError(f"{path}: mismatched method labels")
    observed = [int(row["episode_id"]) for row in rows]
    if observed != expected_roots:
        raise ValueError(f"{path}: held-out root order/content differs from frozen protocol")
    return rows


def recover(*, run_dir: Path, output: Path, active_method: str) -> dict[str, Any]:
    """Rebuild the strict aggregate using only pre-existing immutable reports."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite recovered aggregate: {output}")
    if active_method not in ACTIVE_METHODS:
        raise ValueError(f"unsupported active method: {active_method}")
    frozen_path = run_dir / "frozen_probe_protocol.json"
    provenance_path = run_dir / "root_provenance.json"
    if not frozen_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError("recovery requires frozen protocol and root provenance")
    frozen_payload = _load(frozen_path)
    frozen = frozen_payload.get("frozen_probe")
    roots = frozen_payload.get("held_out_test_roots_read_after_freeze")
    if not isinstance(frozen, dict) or not isinstance(roots, list) or not roots:
        raise ValueError("frozen protocol lacks probe configuration or held-out roots")
    expected_roots = [int(root) for root in roots]
    if bool(frozen.get("outcome_test_used_for_selection", True)) or bool(
        frozen_payload.get("test_outcomes_read_for_selection", True)
    ):
        raise ValueError("recovery refuses a protocol that used held-out outcomes for selection")
    methods = (
        strict.VISUAL_METHOD,
        strict.ADAPTER_METHOD,
        strict.HTSC_METHOD,
        active_method,
        strict.FULL_METHOD,
    )
    if len(methods) != len(set(methods)):
        raise ValueError("active method collides with a baseline arm")
    arm_paths = {method: _arm_path(run_dir, method) for method in methods}
    reports = {
        method: {"rows": _rows_for_arm(path, method, expected_roots)}
        for method, path in arm_paths.items()
    }
    original_active, original_methods = strict.ACTIVE_METHOD, strict.DEPLOYABLE_METHODS
    strict.ACTIVE_METHOD, strict.DEPLOYABLE_METHODS = active_method, methods
    try:
        merged = {"rows": [
            row for method in methods for row in reports[method]["rows"]
        ]}
        grouped = strict.validate_active_probe_report(
            merged,
            expected_roots=expected_roots,
            total_block_budget=strict.TOTAL_BLOCK_BUDGET,
            selected_probe_directions=int(frozen["probe_directions"]),
            selected_probe_magnitude=float(frozen["probe_magnitude"]),
            selected_probe_trigger=str(frozen["active_probe_trigger"]),
            selected_probe_ambiguity_margin=float(frozen["active_probe_ambiguity_margin"]),
        )
        paired = {
            method: strict._paired_summary(
                grouped, left=method, right=strict.VISUAL_METHOD, seed=4400 + index,
            )
            for index, method in enumerate(methods)
        }
        paired["active_probe_vs_htsc_fallback"] = strict._paired_summary(
            grouped, left=active_method, right=strict.HTSC_METHOD, seed=5599,
        )
        mechanism = strict.active_probe_mechanism_diagnostics(grouped)
    finally:
        strict.ACTIVE_METHOD, strict.DEPLOYABLE_METHODS = original_active, original_methods
    active_arm_status = (
        "calibration_rejected_diagnostic_only"
        if not bool(frozen["selection_authorized"])
        else (
            "authorized_but_repair_not_exercised"
            if not bool(mechanism["repair_mechanism_exercised"])
            else "authorized_and_mechanism_exercised"
        )
    )
    report = {
        "schema": "pointmaze.htsc_active_probe.strict_rgb_closed_loop.v1",
        "status": "completed",
        "active_arm_status": active_arm_status,
        "methods": {
            strict.VISUAL_METHOD: "frozen_visual_dynamics",
            strict.ADAPTER_METHOD: "intervention_supervised_effect_adapter",
            strict.HTSC_METHOD: "HTSC_fallback",
            active_method: "calibration_frozen_short_observed_effect_probe_then_local_rerank",
            strict.FULL_METHOD: "effect_adapter_HTSC_visual_consistency",
        },
        "split_protocol": {
            "visual_calibration_roots": frozen_payload["visual_calibration_roots"],
            "gate_calibration_roots": frozen_payload["gate_calibration_roots"],
            "held_out_test_roots": expected_roots,
            "pairwise_disjoint_named_runtime_splits": True,
        },
        "budget_contract": {
            "total_physical_block_budget": strict.TOTAL_BLOCK_BUDGET,
            "baseline_planning_block_cap": strict.TOTAL_BLOCK_BUDGET - 1,
            "active_planning_block_cap_if_triggered": strict.active_planning_budget(
                strict.TOTAL_BLOCK_BUDGET, int(frozen["probe_directions"]),
            ),
            "active_planning_block_cap_if_skipped": strict.TOTAL_BLOCK_BUDGET - 1,
            "active_probe_trigger": str(frozen["active_probe_trigger"]),
            "active_probe_ambiguity_margin": float(frozen["active_probe_ambiguity_margin"]),
            "local_chart_loo_tolerance": frozen.get("local_chart_loo_tolerance"),
            "active_probe_actions_count_against_total_budget": True,
            "active_probe_geometry": strict.expected_probe_design_for_method(active_method),
            "candidate_anchored_repairability_required": bool(
                active_method == strict.CANDIDATE_ANCHORED_AFFINE_METHOD
            ),
            "local_override_visual_regret_cap": 0.0,
            "all_methods_rgb_only": True,
            "privileged_methods_included": False,
        },
        "frozen_probe": frozen,
        "aggregate": {method: strict._method_summary(grouped, method) for method in methods},
        "paired": paired,
        "active_probe_mechanism_diagnostics": mechanism,
        "artifacts": {
            "frozen_protocol": str(frozen_path.resolve()),
            "root_provenance": str(provenance_path.resolve()),
            "gpu_preflight": str((run_dir / "gpu_preflight.json").resolve()),
            "held_out_arm_reports": {method: str(path.resolve()) for method, path in arm_paths.items()},
        },
        "recovery_provenance": {
            "schema": SCHEMA,
            "post_execution_only": True,
            "frozen_protocol_sha256": _sha256(frozen_path),
            "root_provenance_sha256": _sha256(provenance_path),
            "held_out_arm_report_sha256": {method: _sha256(path) for method, path in arm_paths.items()},
            "reason": "strict aggregate validation aborted after all immutable arm reports were written",
        },
        "claim_boundary": (
            f"This is a post-execution aggregate recovery for one legacy PointMaze panel with {len(expected_roots)} held-out roots. "
            "It consumes immutable arm reports only and cannot alter selection, execution, or deployment authorization. "
            "It does not establish universal transfer, physical causality, or a universally reliable world model."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--active-method",
        default="effect_adapter_rgb_online_affine_candidate_anchored_verified_local",
    )
    args = parser.parse_args()
    report = recover(**vars(args))
    print(json.dumps({
        "output": str(args.output),
        "status": report["status"],
        "active_arm_status": report["active_arm_status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
