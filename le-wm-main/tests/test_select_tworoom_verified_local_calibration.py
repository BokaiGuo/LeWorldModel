from __future__ import annotations

import json

import pytest

from experiments.gdd_wm.select_tworoom_verified_local_calibration import (
    CALIBRATION_ROOT_COUNT,
    evaluate_candidate,
    select,
)


def _candidate(*, label: str, delta: float, wins: int, losses: int, pvalue: float) -> dict:
    return {
        "label": label,
        "paired_success_delta": delta,
        "teacher_only_wins": wins,
        "baseline_only_wins": losses,
        "one_sided_sign_test_pvalue": pvalue,
    }


def test_selector_authorizes_only_a_nonempty_paired_teacher_signal() -> None:
    result = select([
        _candidate(label="direct", delta=0.05, wins=5, losses=2, pvalue=0.2265625),
        _candidate(label="verified", delta=0.125, wins=8, losses=2, pvalue=0.0546875),
    ])
    assert result["status"] == "authorized_calibration_only"
    assert result["selected_label"] == "verified"
    assert result["outcome_roots_opened"] is False


def test_selector_fails_closed_for_a_positive_but_unresolved_small_panel() -> None:
    result = select([_candidate(label="weak", delta=0.03125, wins=3, losses=1, pvalue=0.3125)])
    assert result["status"] == "rejected_calibration_no_teacher_contribution"
    assert result["selected_label"] is None


def test_selector_binds_a_nondefault_protocol_and_seed_panel() -> None:
    seeds = tuple(range(500, 564))
    result = select(
        [_candidate(label="verified", delta=0.125, wins=8, losses=2, pvalue=0.0546875)],
        protocol_id="tworoom-v2-test",
        calibration_seeds=seeds,
    )
    assert result["protocol_id"] == "tworoom-v2-test"
    assert result["calibration_seed_panel"] == {"first": 500, "last": 563, "count": 64}


def test_selector_rejects_an_invalid_explicit_seed_panel() -> None:
    with pytest.raises(ValueError, match="ordered and unique"):
        select([], calibration_seeds=(3, 2))


def test_selector_requires_comparable_candidate_fields() -> None:
    with pytest.raises(KeyError):
        select([{"label": "incomplete"}])


def test_evaluate_candidate_rejects_a_nonprotocol_calibration_panel(tmp_path) -> None:
    def report(successes: list[int]) -> dict:
        return {
            "protocol": {
                "checkpoint_frozen": True,
                "runtime_state_access": False,
                "checkpoint": "frozen",
                "runtime_input": "rgb",
                "same_seed_pairing": True,
                "cem": {}, "horizon": 5, "receding_horizon": 5,
                "action_block": 5, "candidate_bank": True,
                "candidate_bank_definition": "fixed", "candidate_bank_variant": "vision_door16",
                "candidate_bank_iterations": 7, "residual_support_scope": "bank",
                "hybrid_gate": {}, "cost_metric": "capacity_safe_readout",
            },
            "hybrid_rgb_fallback": {"seeds": list(range(len(successes))), "episode_successes": successes},
        }

    teacher = report([1] * (CALIBRATION_ROOT_COUNT - 1))
    baseline = report([0] * (CALIBRATION_ROOT_COUNT - 1))
    baseline["protocol"]["cost_metric"] = "euclidean"
    teacher_path, baseline_path = tmp_path / "teacher.json", tmp_path / "baseline.json"
    teacher_path.write_text(json.dumps(teacher), encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    with pytest.raises(ValueError, match="seed panel differs"):
        evaluate_candidate("underpowered", teacher_path, baseline_path)


def test_evaluate_candidate_rejects_a_wrong_sized_but_plausible_seed_panel(tmp_path) -> None:
    # The protocol must reject a panel with arbitrary 64 seeds, not merely
    # check its cardinality.
    seeds = list(range(11, 11 + CALIBRATION_ROOT_COUNT))
    base = {
        "checkpoint_frozen": True, "runtime_state_access": False,
        "checkpoint": "frozen", "runtime_input": "rgb", "same_seed_pairing": True,
        "cem": {}, "horizon": 5, "receding_horizon": 5, "action_block": 5,
        "candidate_bank": True, "candidate_bank_definition": "fixed",
        "candidate_bank_variant": "vision_door16", "candidate_bank_iterations": 7,
        "residual_support_scope": "bank", "hybrid_gate": {},
    }
    teacher = {"protocol": {**base, "cost_metric": "capacity_safe_readout"},
               "hybrid_rgb_fallback": {"seeds": seeds, "episode_successes": [1] * len(seeds)}}
    baseline = {"protocol": {**base, "cost_metric": "euclidean"},
                "hybrid_rgb_fallback": {"seeds": seeds, "episode_successes": [0] * len(seeds)}}
    teacher_path, baseline_path = tmp_path / "teacher.json", tmp_path / "baseline.json"
    teacher_path.write_text(json.dumps(teacher), encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    with pytest.raises(ValueError, match="seed panel differs"):
        evaluate_candidate("wrong-seeds", teacher_path, baseline_path)
