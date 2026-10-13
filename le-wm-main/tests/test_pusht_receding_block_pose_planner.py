from argparse import Namespace
import sys

import numpy as np

from experiments.run_pusht_receding_block_pose_planner import (
    block_pose_cost,
    candidate_bank,
    phase_cost,
    parse_args,
)


def _args(**overrides):
    values = {
        "population": 32,
        "horizon": 8,
        "noise": 0.18,
        "disable_force_fan": False,
        "force_fan_xy_threshold": 60.0,
        "force_fan_near_goal_only": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_task_margin_cost_matches_conjunctive_termination_geometry():
    state = np.array([0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0])
    goal = np.array([0.0, 0.0, 0.0, 0.0, np.pi / 2.0, 0.0, 0.0])
    # XY is exactly at its margin, while angle is well outside it; the
    # normalized worst violation must therefore exceed one.
    assert block_pose_cost(state, goal, cost_mode="task_margin") > 1.0


def test_force_direction_fan_is_an_explicit_candidate_component():
    state = np.array([250.0, 350.0, 200.0, 318.0, 4.5, 0.0, 0.0])
    goal = np.array([100.0, 100.0, 180.0, 240.0, 1.0, 0.0, 0.0])
    with_fan = candidate_bank(np.random.default_rng(7), state, goal, _args())
    without_fan = candidate_bank(
        np.random.default_rng(7), state, goal, _args(disable_force_fan=True)
    )
    assert with_fan.shape == without_fan.shape == (32, 8, 2)
    assert np.isfinite(with_fan).all() and np.isfinite(without_fan).all()
    assert not np.array_equal(with_fan, without_fan)


def test_phase_objective_changes_priority_after_translation_closure():
    goal = np.array([0.0, 0.0, 100.0, 100.0, 0.0, 0.0, 0.0])
    translation_progress = np.array([
        [0.0, 0.0, 130.0, 100.0, 0.30, 0.0, 0.0],
        [0.0, 0.0, 105.0, 100.0, 1.00, 0.0, 0.0],
    ])
    rotation_progress = np.array([
        [0.0, 0.0, 105.0, 100.0, 0.50, 0.0, 0.0],
        [0.0, 0.0, 125.0, 100.0, 0.05, 0.0, 0.0],
    ])
    assert phase_cost(translation_progress, goal, "translate") < phase_cost(
        rotation_progress, goal, "translate"
    )
    assert phase_cost(rotation_progress, goal, "rotate") < phase_cost(
        translation_progress, goal, "rotate"
    )


def test_selected_repair_disables_force_fan_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["planner", "--output", "unused"])
    args = parse_args()
    assert args.disable_force_fan is True

    monkeypatch.setattr(
        sys, "argv", ["planner", "--output", "unused", "--enable-force-fan"]
    )
    args = parse_args()
    assert args.disable_force_fan is False
