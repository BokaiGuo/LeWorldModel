"""Summarize the checkpoint-provenance HTSC repairability experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(root: Path, output: Path) -> dict[str, Any]:
    fixed_frozen = _load(root / "htsc_fixed_bank_alpha010.json")
    fixed_repair = _load(root / "intervention_effect_frozen_latent_action_certified.json")
    shared_frozen = _load(root / "shared_world_htsc_frozen_alpha010.json")
    shared_rgb_adapter = _load(root / "shared_world_htsc_rgb_candidate_adapter_alpha010.json")
    shared_repair = _load(root / "shared_world_intervention_effect_certified.json")

    paired = shared_repair["ensemble_task_contrast"]["paired_regret_gain_over_controls"]
    new_candidate = shared_repair["ensemble_task_contrast"]["new_candidate_control"]
    matched = shared_repair["ensemble_task_contrast"]["matched_order_pair_audit"]
    mechanism_positive = bool(
        all(float(paired[name]["lower"]) > 0.0 for name in ("bank_only", "no_dynamics", "shuffled_prediction"))
        and float(new_candidate["paired_regret_gain_over_shuffled"]["lower"]) > 0.0
        and float(matched["paired_regret_gain_over_bank_only"]["lower"]) > 0.0
    )
    operational_release_count = int(
        shared_repair["ensemble_task_contrast"]["certificate_preview"]["operational_release_count"]
    )
    report = {
        "schema": "htsc_checkpoint_provenance_repairability.v1",
        "protocol": {
            "environment": "PointMaze",
            "candidate_bank": "100 roots x 16 equal-duration candidates",
            "checkpoint_split_provenance": {"train": 54, "calibration": 14, "outcome": 32},
            "alpha": 0.1,
            "runtime_repair_input": "frozen RGB latent plus complete candidate action sequence",
            "repair_labels": "train-root-only simulator terminal XY effects",
            "outcome_used_for_fit_or_gate_selection": False,
            "candidate_modes": ["root-conditioned", "shared-world fixed bank"],
        },
        "finite_sample_coverage": {
            "root_conditioned_frozen": {
                "bound": fixed_frozen["calibration"]["bound"],
                "coverage": fixed_frozen["conformal_outcome_coverage"]["empirical_root_coverage"],
                "covered_roots": fixed_frozen["conformal_outcome_coverage"]["covered_root_count"],
                "outcome_roots": fixed_frozen["conformal_outcome_coverage"]["outcome_root_count"],
                "released_roots": fixed_frozen["selective_outcome"]["released_root_count"],
            },
            "shared_world_frozen": {
                "bound": shared_frozen["calibration"]["bound"],
                "coverage": shared_frozen["conformal_outcome_coverage"]["empirical_root_coverage"],
                "covered_roots": shared_frozen["conformal_outcome_coverage"]["covered_root_count"],
                "outcome_roots": shared_frozen["conformal_outcome_coverage"]["outcome_root_count"],
                "released_roots": shared_frozen["selective_outcome"]["released_root_count"],
            },
        },
        "shortcut_diagnosis": {
            "root_conditioned_repair_regret": fixed_repair["ensemble_task_contrast"]["outcome_selection"]["mean_regret"],
            "root_conditioned_bank_only_regret": fixed_repair["ensemble_task_contrast"]["bank_only_control"]["mean_regret"],
            "root_conditioned_gain_over_bank_only": fixed_repair["ensemble_task_contrast"]["paired_regret_gain_over_controls"]["bank_only"],
            "conclusion": "root-conditioned candidate identity leaks the goal-directed shortcut",
        },
        "shared_world_repair": {
            "outcome_regret": shared_repair["ensemble_task_contrast"]["outcome_selection"]["mean_regret"],
            "bank_only_regret": shared_repair["ensemble_task_contrast"]["bank_only_control"]["mean_regret"],
            "no_dynamics_regret": shared_repair["ensemble_task_contrast"]["no_dynamics_control"]["mean_regret"],
            "shuffled_prediction_regret": shared_repair["ensemble_task_contrast"]["shuffled_prediction_control"]["mean_regret"],
            "paired_regret_gain_over_controls": paired,
            "new_candidate_gain_over_shuffled": new_candidate["paired_regret_gain_over_shuffled"],
            "matched_order_gain_over_bank_only": matched["paired_regret_gain_over_bank_only"],
            "matched_order_left_better_fraction": matched["outcome_left_better_fraction"],
            "mechanism_positive": mechanism_positive,
        },
        "indirect_successor_repair_boundary": {
            "outcome_rgb_contrast_error_mean": shared_rgb_adapter["rgb_contrast_outcome"]["contrast_error_mean"],
            "outcome_physical_inversion_rate": shared_rgb_adapter["physical_contrast_outcome"]["inversion_rate"],
            "outcome_regret": shared_rgb_adapter["selective_outcome"]["all_root_mean_regret"],
            "released_roots": shared_rgb_adapter["selective_outcome"]["released_root_count"],
            "conclusion": "generic RGB successor residual training does not recover task ranking",
        },
        "permission": {
            "exact_winner_operational_release_count": operational_release_count,
            "full_bank_bounded_regret": shared_repair["ensemble_task_contrast"]["bounded_regret_certificate"],
            "matched_order_bounded_regret": matched["bounded_regret_certificate"],
            "status": "abstain" if operational_release_count == 0 else "permit",
        },
        "verdict": {
            "know": "validated marginal fixed-bank coverage on independent roots",
            "repair": "positive intervention-supervised shared-world action ranking with shortcut and new-candidate controls",
            "abstain": "required because exact-winner and preregistered bounded-regret permissions release zero roots",
            "claim_boundary": "one PointMaze checkpoint; not frozen-LeWM dynamics, adaptive CEM coverage, closed-loop planning, or universal transfer",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root, args.output)
    print(json.dumps({"verdict": report["verdict"], "shared_world_repair": report["shared_world_repair"]}, indent=2))


if __name__ == "__main__":
    main()
