"""Audit the sealed outcome of the calibrated TwoRoom verified-local panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.gdd_wm.select_tworoom_verified_local_calibration import (
    DEFAULT_PROTOCOL_ID,
    SCHEMA as SELECTOR_SCHEMA,
    _one_sided_sign_test,
    evaluate_candidate,
)


SCHEMA = "gdd_wm.tworoom_verified_local_calibrated_outcome_audit.v1"
OUTCOME_SEEDS = tuple(range(20264096, 20264224))


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def _seed_panel(start: int, count: int) -> tuple[int, ...]:
    if count < 2:
        raise ValueError("expected root count must be at least two")
    return tuple(range(int(start), int(start) + int(count)))


def audit(
    output_dir: Path,
    *,
    protocol_id: str = DEFAULT_PROTOCOL_ID,
    outcome_seeds: tuple[int, ...] = OUTCOME_SEEDS,
) -> dict[str, Any]:
    if len(outcome_seeds) < 2:
        raise ValueError("outcome seed panel must contain at least two roots")
    if tuple(sorted(set(outcome_seeds))) != tuple(outcome_seeds):
        raise ValueError("outcome seed panel must be strictly ordered and unique")
    selection_path = output_dir / "frozen_calibration_selection.json"
    selection = _load(selection_path)
    if selection.get("schema") != SELECTOR_SCHEMA:
        raise ValueError("unknown calibration selection schema")
    if selection.get("protocol_id", DEFAULT_PROTOCOL_ID) != protocol_id:
        raise ValueError("calibration selection belongs to a different protocol")
    if selection.get("outcome_roots_opened") is not False:
        raise ValueError("selection must record that outcomes were sealed when selected")
    selected = selection.get("selected_label")
    if selected is None:
        if (output_dir / "outcome").exists():
            raise ValueError("calibration rejected but outcome directory exists")
        return {
            "schema": SCHEMA,
            "status": "calibration_rejected_outcomes_sealed",
            "deployment_authorized": False,
            "outcome_evaluated": False,
            "selection": selection,
            "claim_boundary": "Calibration rejection only; no outcome labels were opened.",
        }
    if selection.get("status") != "authorized_calibration_only":
        raise ValueError("selected controller lacks calibration authorization")
    base = output_dir / "outcome" / "baseline" / "planning.json"
    teacher = output_dir / "outcome" / "teacher" / "planning.json"
    result = evaluate_candidate(
        str(selected), teacher, base, expected_seeds=outcome_seeds,
    )
    if any(
        Path(candidate["teacher"]).resolve() == teacher.resolve()
        for candidate in selection.get("candidates", []) if isinstance(candidate, dict)
    ):
        raise ValueError("outcome teacher report was present in calibration candidate list")
    return {
        "schema": SCHEMA,
        "protocol_id": protocol_id,
        "status": "completed_fresh_outcome",
        "deployment_authorized": True,
        "outcome_evaluated": True,
        "selection": {
            "status": selection["status"],
            "selected_label": selected,
            "selection_rule": selection["selection_rule"],
        },
        "outcome": {
            **result,
            "one_sided_sign_test_pvalue": _one_sided_sign_test(
                result["teacher_only_wins"], result["baseline_only_wins"],
            ),
        },
        "claim_boundary": (
            "One fresh TwoRoom outcome comparison of a calibration-selected teacher hybrid against its exact frozen-LeWM local-fallback control. "
            "It does not establish superiority to RGB reactive control, cross-environment transfer, or universal world-model reliability."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-id", default=DEFAULT_PROTOCOL_ID)
    parser.add_argument("--expected-seed-start", type=int, default=OUTCOME_SEEDS[0])
    parser.add_argument("--expected-root-count", type=int, default=len(OUTCOME_SEEDS))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite outcome audit: {args.output}")
    result = audit(
        args.output_dir,
        protocol_id=args.protocol_id,
        outcome_seeds=_seed_panel(args.expected_seed_start, args.expected_root_count),
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
