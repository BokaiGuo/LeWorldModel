import json
from pathlib import Path

import pytest

from experiments.gdd_wm.preflight_pointmaze_shared_future_v8 import (
    EXTENSION_SEED,
    ROOT_RANGE,
    SPLIT_COUNTS,
    _extension_contract,
)
from experiments.decision_benchmark.build_pointmaze_temporal_ood_v6_matched_panel import INTERACTION_DESIGN_ROOTS
from experiments.gdd_wm.run_pointmaze_planning_shaped_control import _sha256, _validate_v8_shared_future_contract


def _action(value: float) -> list[list[float]]:
    return [[value, 0.0] for _ in range(20)]


def _row(root: int, split: str, candidate: int) -> dict:
    return {
        "instance_id": f"pointmaze_v6_episode_{root}", "start_episode_id": str(root),
        "split": split, "candidate_id": candidate, "action_sequence": _action(float(candidate)),
    }


def test_v8_extension_contract_binds_a_new_root_range(tmp_path: Path) -> None:
    (tmp_path / "extension_manifest.json").write_text(json.dumps({
        "schema": "pointmaze_independent_state_action_extension.v1", "seed": EXTENSION_SEED,
        "base_root_count": ROOT_RANGE[0], "extended_root_count": ROOT_RANGE[1],
        "new_root_id_range": [ROOT_RANGE[0], ROOT_RANGE[1] - 1],
    }))
    assert all(_extension_contract(tmp_path)["checks"].values())


def test_v8r2_root_budget_reserves_evaluation_and_disjoint_design_roots() -> None:
    """Do not reintroduce the original 512-root capacity configuration."""

    assert ROOT_RANGE[1] - ROOT_RANGE[0] >= sum(SPLIT_COUNTS.values()) + INTERACTION_DESIGN_ROOTS


def test_v8_extension_contract_rejects_v7_or_reused_roots(tmp_path: Path) -> None:
    (tmp_path / "extension_manifest.json").write_text(json.dumps({
        "schema": "pointmaze_independent_state_action_extension.v1", "seed": EXTENSION_SEED,
        "base_root_count": ROOT_RANGE[0] - 1, "extended_root_count": ROOT_RANGE[1] - 1,
        "new_root_id_range": [ROOT_RANGE[0] - 1, ROOT_RANGE[1] - 2],
    }))
    with pytest.raises(ValueError, match="violates"):
        _extension_contract(tmp_path)


def test_v8_launcher_is_gpu_only_and_uses_the_new_independent_extension():
    launcher = Path(__file__).resolve().parents[1] / "experiments" / "gdd_wm" / "run_pointmaze_shared_future_v8_cuda.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "experiments.gpu_preflight --device cuda --recurrent-smoke" in text
    assert "refusing CPU fallback" in text
    assert "point_maze_v45_shared_future_v8r2_extension_20261022" in text
    assert "--episode-count 640 --seed 202611500" in text
    assert "--root-id-min 17720 --root-id-max-exclusive 18360" in text
    assert "preflight_pointmaze_shared_future_v8" in text
    assert "--include-shared-future-control" in text
    assert "--action-representation full_ordered_20x2" in text
    assert "evaluate_pointmaze_shared_future_v8_htsc" in text
    assert "--require-v8-preflight" in text


def test_v8_runner_contract_binds_exact_roots_and_full_action_bank(tmp_path: Path) -> None:
    roots = {
        "train": list(range(1000, 1128)),
        "calibration": list(range(1128, 1320)),
        "outcome": list(range(1320, 1384)),
    }
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({
        "schema": "pointmaze_shared_future_v8_static_preflight.v1", "status": "passed_static_preflight",
        "checks": {"support": True}, "root_counts": {name: len(value) for name, value in roots.items()},
        "split_roots": roots,
        "candidate_bank": {
            "shared_candidate_in_support": True,
            "scorer_candidate_bank_protocol": "held_exact_shared_bank",
            "shared_future_action_representation": "full_ordered_20x2",
        },
    }), encoding="utf-8")
    receipt = {"schema": "pointmaze_shared_future_v8_static_preflight.v1", "status": "passed_static_preflight", "path": str(ledger), "sha256": _sha256(ledger)}
    source = {"records": [_row(root, "train", candidate) for root in roots["train"] for candidate in range(2)]}
    context = {
        "metadata": {
            "candidate_generalization_regime": "shared_candidate_in_support",
            "candidate_design": "goal_balanced_interaction_pairs",
            "scorer_candidate_bank_protocol": "held_exact_shared_bank",
            "outcome_labels_materialized_during_build": False,
            "frozen_preflight_ledger": receipt,
        },
        "records": [
            _row(root, split, candidate)
            for split in ("calibration", "outcome")
            for root in roots[split]
            for candidate in range(2)
        ],
    }
    verified = _validate_v8_shared_future_contract(source, context)
    assert verified["root_counts"] == {"train": 128, "calibration": 192, "outcome": 64}
    assert verified["candidate_count"] == 2
    context["records"][0]["action_sequence"] = _action(99.0)
    with pytest.raises(ValueError, match="action sequence changes"):
        _validate_v8_shared_future_contract(source, context)


def test_v8_runner_contract_rejects_pairwise_split_overlap_even_if_third_split_is_disjoint(tmp_path: Path) -> None:
    """Guard against checking only a three-way set intersection."""

    roots = {
        "train": list(range(1000, 1128)),
        # Deliberately overlaps train in one root but has an otherwise
        # independent outcome split.  A three-way intersection would be empty.
        "calibration": [1000, *range(1128, 1319)],
        "outcome": list(range(1320, 1384)),
    }
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({
        "schema": "pointmaze_shared_future_v8_static_preflight.v1", "status": "passed_static_preflight",
        "checks": {"support": True}, "root_counts": {name: len(value) for name, value in roots.items()},
        "split_roots": roots,
        "candidate_bank": {"shared_candidate_in_support": True, "scorer_candidate_bank_protocol": "held_exact_shared_bank", "shared_future_action_representation": "full_ordered_20x2"},
    }), encoding="utf-8")
    receipt = {"schema": "pointmaze_shared_future_v8_static_preflight.v1", "status": "passed_static_preflight", "path": str(ledger), "sha256": _sha256(ledger)}
    source = {"records": [_row(root, "train", candidate) for root in roots["train"] for candidate in range(2)]}
    context = {
        "metadata": {"candidate_generalization_regime": "shared_candidate_in_support", "candidate_design": "goal_balanced_interaction_pairs", "scorer_candidate_bank_protocol": "held_exact_shared_bank", "outcome_labels_materialized_during_build": False, "frozen_preflight_ledger": receipt},
        "records": [_row(root, split, candidate) for split in ("calibration", "outcome") for root in roots[split] for candidate in range(2)],
    }
    with pytest.raises(ValueError, match="pairwise disjoint"):
        _validate_v8_shared_future_contract(source, context)
