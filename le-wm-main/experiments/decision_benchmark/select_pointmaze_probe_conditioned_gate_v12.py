"""Freeze V12's probe-and-release gate from calibration-only consequences.

This selector intentionally knows nothing about the future outcome panel.  It
chooses a non-empty probe-conditioned policy only if it is non-inferior to the
RGB visual fallback on held calibration roots.  Otherwise it rejects the
intervention and leaves all outcome roots sealed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "pointmaze.probe_conditioned_gate_v12.calibration_selector.v1"
LOCAL_METHOD = "effect_adapter_rgb_online_probe_conditioned_bank_verified_local"
VISUAL_METHOD = "visual_dynamics"
MIN_RELEASED_ROOTS = 4


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value.get("rows"), list):
        raise ValueError(f"{path} has no planning rows")
    return value


def _rows(report: dict[str, Any], method: str) -> dict[int, dict[str, Any]]:
    values = [row for row in report["rows"] if row.get("method") == method]
    keyed = {int(row["episode_id"]): row for row in values}
    if not keyed or len(keyed) != len(values):
        raise ValueError(f"missing unique rows for {method}")
    return keyed


def _released_count(rows: dict[int, dict[str, Any]]) -> int:
    count = 0
    for row in rows.values():
        events = [
            event for event in row.get("active_probe_trace", [])
            if isinstance(event, dict) and event.get("probe_conditioned_bank")
        ]
        if len(events) == 0:
            # A pre-probe ambiguity rejection is a legitimate abstention: no
            # information action was charged, so no local bank can exist.
            decisions = [
                event for event in row.get("active_probe_trace", [])
                if (
                    isinstance(event, dict)
                    and event.get("probe_decision")
                    and "probe_conditioned_visual_margin" in event
                )
            ]
            if len(decisions) == 1 and not bool(decisions[0].get("probe_triggered")):
                continue
            # The ambiguity gate may authorize a probe while the independent
            # task-safety screen finds no forward information action.  This
            # is also a valid fail-closed abstention, not a missing trace.
            safe_skips = [
                event for event in row.get("active_probe_trace", [])
                if (
                    isinstance(event, dict)
                    and bool(event.get("probe_decision"))
                    and bool(event.get("probe_skipped"))
                    and event.get("reason") == "no_task_safe_probe_under_calibration_dynamics"
                )
            ]
            if len(decisions) == 1 and bool(decisions[0].get("probe_triggered")) and len(safe_skips) == 1:
                continue
            raise ValueError("local calibration root lacks an auditable probe abstention")
        if len(events) != 1:
            raise ValueError("each local calibration root needs one candidate-bank decision")
        if events[0].get("construction_status") == "constructed_released_pre_outcome":
            count += 1
    return count


def _bank_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event for event in row.get("active_probe_trace", [])
        if isinstance(event, dict) and event.get("probe_conditioned_bank")
    ]


def validate_abstain_equivalence(
    baseline: dict[int, dict[str, Any]],
    candidate: dict[int, dict[str, Any]],
) -> int:
    """Ensure an uncharged probe abstention is behaviorally RGB fallback.

    This catches the V12-r1 bug where skipped roots silently accumulated local
    fallback actions and later became a different controller.  Metadata and
    gate fields may differ, but selected physical actions and outcomes may
    not.
    """

    checked = 0
    for root in sorted(baseline):
        if _bank_events(candidate[root]):
            continue
        left, right = baseline[root], candidate[root]
        left_actions = [item.get("selected_action") for item in left.get("trace", [])]
        right_actions = [item.get("selected_action") for item in right.get("trace", [])]
        if left_actions != right_actions:
            raise ValueError("probe abstention selected a different physical action than visual fallback")
        if bool(left["success"]) != bool(right["success"]) or not np.isclose(
            float(left["final_distance"]), float(right["final_distance"]), atol=1.0e-12, rtol=0.0,
        ):
            raise ValueError("probe abstention changed the visual fallback outcome")
        checked += 1
    return checked


def summarize_candidate(
    *,
    label: str,
    baseline: dict[int, dict[str, Any]],
    candidate: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if set(baseline) != set(candidate):
        raise ValueError("calibration comparison requires exactly paired roots")
    abstain_equivalent_roots = validate_abstain_equivalence(baseline, candidate)
    root_ids = sorted(baseline)
    distance_delta = np.asarray([
        float(candidate[root]["final_distance"]) - float(baseline[root]["final_distance"])
        for root in root_ids
    ], dtype=np.float64)
    success_delta = np.asarray([
        int(bool(candidate[root]["success"])) - int(bool(baseline[root]["success"]))
        for root in root_ids
    ], dtype=np.int64)
    released = _released_count(candidate)
    return {
        "label": label,
        "root_count": len(root_ids),
        "released_root_count": released,
        "released_root_fraction": float(released / len(root_ids)),
        "abstain_equivalent_root_count": abstain_equivalent_roots,
        "success_delta": int(np.sum(success_delta)),
        "mean_final_distance_delta": float(np.mean(distance_delta)),
        "paired_distance_win_count": int(np.sum(distance_delta < 0.0)),
        "paired_distance_loss_count": int(np.sum(distance_delta > 0.0)),
        "eligible": bool(
            released >= MIN_RELEASED_ROOTS
            and int(np.sum(success_delta)) >= 0
            and float(np.mean(distance_delta)) <= 0.0
        ),
        "eligibility_rule": (
            "at least four released calibration roots AND paired success delta >= 0 "
            "AND mean local-minus-visual final distance <= 0"
        ),
    }


def select(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one calibration candidate is required")
    authorized = [candidate for candidate in candidates if bool(candidate["eligible"])]
    if not authorized:
        return {
            "deployment_authorized": False,
            "status": "rejected_calibration_no_nonempty_noninferior_gate",
            "selected_label": None,
        }
    selected = min(
        authorized,
        key=lambda row: (
            -int(row["success_delta"]),
            float(row["mean_final_distance_delta"]),
            -int(row["released_root_count"]),
            str(row["label"]),
        ),
    )
    return {
        "deployment_authorized": True,
        "status": "authorized_calibration_only",
        "selected_label": str(selected["label"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate", action="append", required=True,
        help="label:path to one calibration-only local-policy report",
    )
    parser.add_argument(
        "--local-method", default=LOCAL_METHOD,
        help="runtime local-policy method expected in each candidate report",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite immutable probe-conditioned gate selection")
    baseline_report = _read(args.baseline)
    baseline = _rows(baseline_report, VISUAL_METHOD)
    candidates: list[dict[str, Any]] = []
    source_paths: dict[str, str] = {}
    for token in args.candidate:
        if ":" not in token:
            raise ValueError("candidate must be label:path")
        label, raw_path = token.split(":", 1)
        if not label or label in source_paths:
            raise ValueError("candidate labels must be non-empty and unique")
        path = Path(raw_path)
        source_paths[label] = str(path.resolve())
        candidates.append(summarize_candidate(
            label=label, baseline=baseline, candidate=_rows(_read(path), args.local_method),
        ))
    decision = select(candidates)
    report = {
        "schema": SCHEMA,
        "baseline": str(args.baseline.resolve()),
        "candidate_reports": source_paths,
        "local_method": str(args.local_method),
        "minimum_released_roots": MIN_RELEASED_ROOTS,
        "candidates": candidates,
        "decision": decision,
        "outcome_roots_opened": False,
        "claim_boundary": (
            "Calibration-only authorization for one declared PointMaze probe-and-release "
            "gate. It is not an outcome result or a conditional reliability guarantee."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": decision["status"], "selected": decision["selected_label"]}, indent=2))


if __name__ == "__main__":
    main()
