"""Forensically quarantine the opened V5 candidate-anchored probe panel.

V5 wrote every calibration and held-out arm before its final aggregate
validation failed.  That does *not* make the partial artifact a usable
efficacy result: the raw active trace contains an executed pulse whose
runtime goal projection was nonpositive.  This tool binds that conclusion to
the immutable raw files and emits a machine-readable no-reuse decision.  It
never reconstructs a favourable aggregate, launches a rollout, or rewrites
the legacy directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments import run_pointmaze_htsc_active_probe_strict as strict


SCHEMA = "pointmaze.htsc_candidate_anchored_affine_probe_v5_forensic_audit.v1"
ACTIVE_METHOD = "effect_adapter_rgb_online_affine_candidate_anchored_verified_local"
METHODS = (
    strict.VISUAL_METHOD,
    strict.ADAPTER_METHOD,
    strict.HTSC_METHOD,
    ACTIVE_METHOD,
    strict.FULL_METHOD,
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arm_rows(run_dir: Path, roots: list[int]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for method in METHODS:
        path = run_dir / "held_out" / f"{method}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing immutable V5 held-out arm report: {path}")
        payload = _load(path)
        arm_rows = payload.get("rows")
        if not isinstance(arm_rows, list) or not arm_rows:
            raise ValueError(f"{path}: missing held-out rows")
        if any(not isinstance(row, dict) or row.get("method") != method for row in arm_rows):
            raise ValueError(f"{path}: method label does not match its immutable arm")
        if [int(row["episode_id"]) for row in arm_rows] != roots:
            raise ValueError(f"{path}: held-out roots differ from the frozen protocol")
        rows.extend(arm_rows)
        hashes[method] = _sha256(path)
    return rows, hashes


def _executed_probe_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**event, "episode_id": int(row["episode_id"])}
        for row in rows
        if row.get("method") == ACTIVE_METHOD
        for event in row.get("active_probe_trace", [])
        if isinstance(event, dict) and "probe_index" in event and "error" not in event
    ]


def _trace_violations(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return raw event evidence for every pre-action safety violation."""

    violations: list[dict[str, Any]] = []
    for event in events:
        projection = float(event.get("action_goal_projection", float("nan")))
        progress = float(event.get("predicted_goal_progress", float("nan")))
        if not projection > 0.0 or not progress > 0.0:
            violations.append({
                "episode_id": int(event.get("episode_id", -1)),
                "probe_index": int(event.get("probe_index", -1)),
                "probe_candidate_index": event.get("probe_candidate_index"),
                "action": event.get("action"),
                "action_goal_projection": projection,
                "predicted_goal_progress": progress,
                "observed_goal_progress": event.get("observed_goal_progress"),
                "violation": (
                    "nonpositive_execution_time_goal_projection"
                    if not projection > 0.0
                    else "nonpositive_execution_time_predicted_progress"
                ),
            })
    return violations


def _strict_validation_error(rows: list[dict[str, Any]], frozen: dict[str, Any], roots: list[int]) -> str | None:
    """Run the current validator under V5's declared active arm, read-only."""

    original_active, original_methods = strict.ACTIVE_METHOD, strict.DEPLOYABLE_METHODS
    strict.ACTIVE_METHOD, strict.DEPLOYABLE_METHODS = ACTIVE_METHOD, METHODS
    try:
        strict.validate_active_probe_report(
            {"rows": rows},
            expected_roots=roots,
            total_block_budget=strict.TOTAL_BLOCK_BUDGET,
            selected_probe_directions=int(frozen["probe_directions"]),
            selected_probe_magnitude=float(frozen["probe_magnitude"]),
            selected_probe_trigger=str(frozen["active_probe_trigger"]),
            selected_probe_ambiguity_margin=float(frozen["active_probe_ambiguity_margin"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        return str(error)
    finally:
        strict.ACTIVE_METHOD, strict.DEPLOYABLE_METHODS = original_active, original_methods
    return None


def audit(run_dir: Path) -> dict[str, Any]:
    """Audit V5 in place without changing any V5 input or outcome file."""

    if (run_dir / "report.json").exists():
        raise ValueError("V5 forensic audit expects the legacy run to lack a final aggregate report")
    frozen_path = run_dir / "frozen_probe_protocol.json"
    provenance_path = run_dir / "root_provenance.json"
    gpu_path = run_dir / "gpu_preflight.json"
    for path in (frozen_path, provenance_path, gpu_path):
        if not path.is_file():
            raise FileNotFoundError(f"V5 forensic audit requires immutable input: {path}")
    frozen_payload = _load(frozen_path)
    frozen = frozen_payload.get("frozen_probe")
    roots = frozen_payload.get("held_out_test_roots_read_after_freeze")
    if not isinstance(frozen, dict) or not isinstance(roots, list) or not roots:
        raise ValueError("V5 frozen protocol lacks a frozen probe configuration or held-out roots")
    roots = [int(root) for root in roots]
    if frozen.get("outcome_test_used_for_selection") is not False or (
        frozen_payload.get("test_outcomes_read_for_selection") is not False
    ):
        raise ValueError("V5 protocol does not establish outcome-sealed selection")
    rows, arm_hashes = _arm_rows(run_dir, roots)
    events = _executed_probe_events(rows)
    violations = _trace_violations(events)
    strict_error = _strict_validation_error(rows, frozen, roots)
    if not violations:
        raise RuntimeError("V5 audit expected a raw execution-time safety violation")
    if strict_error is None:
        raise RuntimeError("V5 audit expected the current strict validator to reject the raw trace")
    return {
        "schema": SCHEMA,
        "status": "invalid_opened_outcome_protocol_contract",
        "scientific_performance_claim_allowed": False,
        "recovery_or_rerun_of_v5_allowed": False,
        "held_out_outcomes_must_not_be_used_for_tuning": True,
        "reason": "An executed V5 candidate-anchored pulse violated its execution-time task-safety contract.",
        "strict_validator_error": strict_error,
        "raw_trace_violation_count": len(violations),
        "raw_trace_violations": violations,
        "raw_probe_event_count": len(events),
        "frozen_selection": {
            "selection_authorized": frozen.get("selection_authorized"),
            "active_probe_trigger": frozen.get("active_probe_trigger"),
            "active_probe_ambiguity_margin": frozen.get("active_probe_ambiguity_margin"),
            "probe_directions": frozen.get("probe_directions"),
            "probe_magnitude": frozen.get("probe_magnitude"),
            "local_chart_loo_tolerance": frozen.get("local_chart_loo_tolerance"),
            "held_out_root_count": len(roots),
        },
        "immutable_input_sha256": {
            "frozen_probe_protocol": _sha256(frozen_path),
            "root_provenance": _sha256(provenance_path),
            "gpu_preflight": _sha256(gpu_path),
            "held_out_arm_reports": arm_hashes,
        },
        "claim_boundary": (
            "This is a protocol-forensics result, not an efficacy result. It invalidates V5 for performance claims "
            "and does not determine whether a fresh, conforming active-probe controller helps or harms."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite forensic audit: {args.output}")
    result = audit(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
