"""Select a verified-local TwoRoom repair from calibration-only matched arms.

The selection target is *teacher contribution*, not a comparison with a
stronger RGB-reactive ceiling: every candidate is paired with an otherwise
identical frozen-LeWM hybrid using the same RGB effect verification and action
bank.  A candidate is deployable only if its teacher arm wins often enough on
unopened calibration seeds.  Outcome reports are deliberately not accepted by
this tool.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SCHEMA = "gdd_wm.tworoom_verified_local_calibration_selector.v1"
DEFAULT_PROTOCOL_ID = "tworoom_verified_local_calibrated_v1_20261023"
CALIBRATION_ROOT_COUNT = 64
CALIBRATION_SEEDS = tuple(range(20264032, 20264096))
_MATCHED_PROTOCOL_FIELDS = (
    "checkpoint", "checkpoint_frozen", "runtime_input", "runtime_state_access",
    "same_seed_pairing", "cem", "horizon", "receding_horizon", "action_block",
    "candidate_bank", "candidate_bank_definition", "candidate_bank_variant",
    "candidate_bank_iterations", "residual_support_scope", "hybrid_gate",
)


def _seed_panel(start: int, count: int) -> tuple[int, ...]:
    if count < 2:
        raise ValueError("expected root count must be at least two")
    return tuple(range(int(start), int(start) + int(count)))


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def _one_sided_sign_test(wins: int, losses: int) -> float:
    """Exact null tail P[Binomial(wins+losses,.5) >= wins]."""

    total = wins + losses
    if total == 0:
        return 1.0
    return float(sum(math.comb(total, value) for value in range(wins, total + 1)) / 2 ** total)


def _hybrid(report: dict[str, Any], path: Path) -> tuple[list[int], list[int]]:
    protocol = report.get("protocol")
    hybrid = report.get("hybrid_rgb_fallback")
    if not isinstance(protocol, dict) or not isinstance(hybrid, dict):
        raise ValueError(f"{path}: needs protocol and hybrid_rgb_fallback")
    if protocol.get("runtime_state_access") is not False or protocol.get("checkpoint_frozen") is not True:
        raise ValueError(f"{path}: hybrid must be frozen RGB-only")
    seeds, success = hybrid.get("seeds"), hybrid.get("episode_successes")
    if not isinstance(seeds, list) or not isinstance(success, list) or len(seeds) < 2 or len(seeds) != len(success):
        raise ValueError(f"{path}: hybrid paired seed/success arrays are malformed")
    if any(int(value) not in (0, 1) for value in success):
        raise ValueError(f"{path}: hybrid success values must be binary")
    return [int(value) for value in seeds], [int(value) for value in success]


def _matched_protocol(teacher: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    left, right = teacher["protocol"], baseline["protocol"]
    mismatch = {
        key: (left.get(key), right.get(key))
        for key in _MATCHED_PROTOCOL_FIELDS
        if left.get(key) != right.get(key)
    }
    if mismatch:
        raise ValueError(f"teacher/baseline protocol mismatch: {mismatch}")
    if left.get("cost_metric") != "capacity_safe_readout" or right.get("cost_metric") != "euclidean":
        raise ValueError("matched arms must compare capacity_safe_readout against frozen euclidean cost")
    return {key: left.get(key) for key in _MATCHED_PROTOCOL_FIELDS}


def evaluate_candidate(
    label: str,
    teacher_path: Path,
    baseline_path: Path,
    *,
    expected_seeds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    teacher, baseline = _load(teacher_path), _load(baseline_path)
    protocol = _matched_protocol(teacher, baseline)
    teacher_seeds, teacher_success = _hybrid(teacher, teacher_path)
    baseline_seeds, baseline_success = _hybrid(baseline, baseline_path)
    if teacher_seeds != baseline_seeds:
        raise ValueError("teacher/baseline arms do not share the exact ordered seed list")
    delta = [left - right for left, right in zip(teacher_success, baseline_success)]
    if expected_seeds is None:
        expected_seeds = CALIBRATION_SEEDS
    if tuple(teacher_seeds) != expected_seeds:
        raise ValueError(
            f"{label}: paired seed panel differs from the predeclared protocol"
        )
    if len(delta) != len(expected_seeds):
        raise ValueError(
            f"{label}: paired panel must contain exactly "
            f"{len(expected_seeds)} roots, got {len(delta)}"
        )
    wins, losses = sum(value > 0 for value in delta), sum(value < 0 for value in delta)
    return {
        "label": label,
        "teacher": str(teacher_path.resolve()),
        "baseline": str(baseline_path.resolve()),
        "root_count": len(delta),
        "teacher_hybrid_success": float(sum(teacher_success) / len(teacher_success)),
        "baseline_hybrid_success": float(sum(baseline_success) / len(baseline_success)),
        "paired_success_delta": float(sum(delta) / len(delta)),
        "teacher_only_wins": int(wins),
        "baseline_only_wins": int(losses),
        "ties": int(len(delta) - wins - losses),
        "one_sided_sign_test_pvalue": _one_sided_sign_test(wins, losses),
        "matched_protocol": protocol,
    }


def select(
    candidates: list[dict[str, Any]],
    *,
    protocol_id: str = DEFAULT_PROTOCOL_ID,
    calibration_seeds: tuple[int, ...] = CALIBRATION_SEEDS,
) -> dict[str, Any]:
    """Fail closed unless calibration gives a nonempty directional signal."""

    if len(calibration_seeds) < 2:
        raise ValueError("calibration seed panel must contain at least two roots")
    if tuple(sorted(set(calibration_seeds))) != tuple(calibration_seeds):
        raise ValueError("calibration seed panel must be strictly ordered and unique")
    eligible = [
        item for item in candidates
        if item["paired_success_delta"] > 0.0
        and item["teacher_only_wins"] >= 4
        and item["one_sided_sign_test_pvalue"] <= 0.10
    ]
    selected = min(
        eligible,
        key=lambda item: (-item["paired_success_delta"], item["one_sided_sign_test_pvalue"], item["label"]),
    ) if eligible else None
    return {
        "schema": SCHEMA,
        "protocol_id": protocol_id,
        "calibration_seed_panel": {
            "first": calibration_seeds[0],
            "last": calibration_seeds[-1],
            "count": len(calibration_seeds),
        },
        "status": "authorized_calibration_only" if selected else "rejected_calibration_no_teacher_contribution",
        "outcome_roots_opened": False,
        "selection_rule": (
            "authorize only when the teacher hybrid has positive paired success, at least four teacher-only wins, "
            "and a one-sided exact paired sign-test p-value <= 0.10 against its matched frozen-LeWM hybrid"
        ),
        "candidates": candidates,
        "selected_label": None if selected is None else selected["label"],
        "selected": selected,
        "claim_boundary": (
            "Calibration-only matched-arm authorization. It does not establish a planning benefit and must precede any outcome evaluation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True, metavar="LABEL:TEACHER:BASELINE")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-id", default=DEFAULT_PROTOCOL_ID)
    parser.add_argument("--expected-seed-start", type=int, default=CALIBRATION_SEEDS[0])
    parser.add_argument("--expected-root-count", type=int, default=CALIBRATION_ROOT_COUNT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite calibration decision: {args.output}")
    calibration_seeds = _seed_panel(args.expected_seed_start, args.expected_root_count)
    rows: list[dict[str, Any]] = []
    for value in args.candidate:
        parts = value.split(":", 2)
        if len(parts) != 3 or not all(parts):
            raise ValueError("--candidate must be LABEL:TEACHER:BASELINE")
        rows.append(evaluate_candidate(
            parts[0], Path(parts[1]), Path(parts[2]), expected_seeds=calibration_seeds,
        ))
    result = select(rows, protocol_id=args.protocol_id, calibration_seeds=calibration_seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "selected_label": result["selected_label"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
