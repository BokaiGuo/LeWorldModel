from __future__ import annotations

import numpy as np

from experiments.audit_cube_planner_ceiling import (
    _expanded_bank, _factorized_bank, _goal_conditioned_push_bank,
    _multiphase_factorized_bank, _summarize,
)
from experiments.audit_cube_adaptive_protocol_ceiling import _summarize as _summarize_adaptive


def test_expanded_cube_bank_has_diagonal_and_valid_close_sign() -> None:
    bank = _expanded_bank()
    assert bank.ndim == 3
    assert bank.shape[1:] == (40, 5)
    assert bank.shape[0] > 5
    assert np.all(np.isfinite(bank))
    assert np.max(np.abs(bank)) <= 1.0
    assert np.all(bank[1:, :, 4].max(axis=1) == 1.0)
    assert any(
        np.count_nonzero(np.abs(sequence[:, 0]) > 1e-6)
        and np.count_nonzero(np.abs(sequence[:, 1]) > 1e-6)
        for sequence in bank[1:]
    )


def test_factorized_bank_decouples_approach_and_push_directions() -> None:
    bank = _factorized_bank(length=40, direction_count=8)
    assert bank.shape == (1 + 8 * 8 * 3, 40, 5)
    assert np.all(bank[:, 14:18, 4] >= 0.0)
    approach_vectors = {tuple(np.round(row[:4, :2].mean(axis=0), 5)) for row in bank[1:]}
    push_vectors = {tuple(np.round(row[18:, :2].mean(axis=0), 5)) for row in bank[1:]}
    assert len(approach_vectors) == 8
    assert len(push_vectors) >= 8
    assert any(
        not np.allclose(row[:4, :2].mean(axis=0), row[18:, :2].mean(axis=0))
        for row in bank[1:]
    )


def test_goal_conditioned_push_bank_is_explicitly_finite_and_goal_centered() -> None:
    bank = _goal_conditioned_push_bank(np.asarray([1.0, 1.0]), length=40)
    assert bank.shape == (1 + 8 * 8 * 3, 40, 5)
    push = bank[1:, 18:, :2].reshape(-1, 2)
    assert np.max(np.abs(push)) <= 1.0
    goal = np.asarray([1.0, 1.0]) / np.sqrt(2.0)
    assert np.any(np.all(np.isclose(push, goal, atol=1.0e-6), axis=1))


def test_multiphase_bank_keeps_two_push_phases_without_reopening() -> None:
    bank = _multiphase_factorized_bank(length=40, direction_count=4)
    assert bank.shape == (1 + 4 * 4 * 4 * 2, 40, 5)
    assert np.all(bank[:, 14:, 4] >= 0.0)
    assert any(
        not np.allclose(row[18:29, :2].mean(axis=0), row[29:, :2].mean(axis=0))
        for row in bank[1:]
    )


def test_ceiling_summary_separates_reachability_from_distance() -> None:
    records = [
        {"fixed": {"reachable": False, "oracle": {"final_goal_distance_m": 0.2}}, "expanded": {"reachable": True, "oracle": {"final_goal_distance_m": 0.04}}, "reachability_recovered_by_expansion": True},
        {"fixed": {"reachable": True, "oracle": {"final_goal_distance_m": 0.03}}, "expanded": {"reachable": True, "oracle": {"final_goal_distance_m": 0.03}}, "reachability_recovered_by_expansion": False},
    ]
    summary = _summarize(records)
    assert summary["fixed_bank_reachability_rate"] == 0.5
    assert summary["expanded_bank_reachability_rate"] == 1.0
    assert summary["reachability_recovered_by_expansion_count"] == 1
    assert "not RGB online planning" in summary["claim_boundary"]


def test_ceiling_summary_reports_adaptive_protocol_when_present() -> None:
    records = [
        {
            "fixed": {"reachable": False, "oracle": {"final_goal_distance_m": 0.2}},
            "expanded": {"reachable": False, "oracle": {"final_goal_distance_m": 0.19}},
            "reachability_recovered_by_expansion": False,
            "adaptive_fixed": {"reachable": True},
            "adaptive_expanded": {"reachable": True},
            "adaptive_reachability_recovered_by_expansion": False,
        },
        {
            "fixed": {"reachable": True, "oracle": {"final_goal_distance_m": 0.04}},
            "expanded": {"reachable": True, "oracle": {"final_goal_distance_m": 0.04}},
            "reachability_recovered_by_expansion": False,
            "adaptive_fixed": {"reachable": False},
            "adaptive_expanded": {"reachable": True},
            "adaptive_reachability_recovered_by_expansion": True,
        },
    ]
    summary = _summarize(records)
    assert summary["adaptive_fixed_bank_reachability_rate"] == 0.5
    assert summary["adaptive_expanded_bank_reachability_rate"] == 1.0
    assert summary["adaptive_reachability_recovered_by_expansion_count"] == 1


def test_ceiling_summary_reports_factorized_support_when_present() -> None:
    records = [
        {
            "fixed": {"reachable": False, "oracle": {"final_goal_distance_m": 0.2}},
            "expanded": {"reachable": False, "oracle": {"final_goal_distance_m": 0.19}},
            "factorized": {"reachable": True, "oracle": {"final_goal_distance_m": 0.04}},
            "reachability_recovered_by_expansion": False,
            "reachability_recovered_by_factorized": True,
        },
    ]
    summary = _summarize(records)
    assert summary["factorized_bank_reachability_rate"] == 1.0
    assert summary["reachability_recovered_by_factorized_count"] == 1


def test_ceiling_summary_reports_multiphase_support_when_present() -> None:
    records = [
        {
            "fixed": {"reachable": False, "oracle": {"final_goal_distance_m": 0.2}},
            "expanded": {"reachable": False, "oracle": {"final_goal_distance_m": 0.19}},
            "factorized": {"reachable": False, "oracle": {"final_goal_distance_m": 0.18}},
            "multiphase": {"reachable": True, "oracle": {"final_goal_distance_m": 0.04}},
            "reachability_recovered_by_expansion": False,
            "reachability_recovered_by_factorized": False,
            "reachability_recovered_by_multiphase": True,
        },
    ]
    summary = _summarize(records)
    assert summary["multiphase_bank_reachability_rate"] == 1.0
    assert summary["reachability_recovered_by_multiphase_count"] == 1


def test_adaptive_protocol_summary_reports_root_level_ceiling() -> None:
    summary = _summarize_adaptive([
        {"adaptive": {"reachable": True, "success_count": 1,
                       "oracle": {"final_goal_distance_m": 0.02}}},
        {"adaptive": {"reachable": False, "success_count": 0,
                       "oracle": {"final_goal_distance_m": 0.20}}},
    ])
    assert summary["adaptive_fixed_bank_reachability_rate"] == 0.5
    assert summary["adaptive_fixed_bank_success_count"] == 1
    assert np.isclose(summary["adaptive_fixed_bank_oracle_distance_mean_m"], 0.11)
