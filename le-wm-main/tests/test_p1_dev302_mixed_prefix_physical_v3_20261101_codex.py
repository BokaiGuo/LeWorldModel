from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability import (
    p1_dev302_mixed_prefix_physical_v3_20261101_codex as subject,
)


def test_v3_contract_binds_four_roles_seed_and_planner_hashes():
    value = subject.build()
    assert value["seed"] == 301000002
    assert [row["role"] for row in value["roles"]] == list(subject.ROLES)
    assert [row["candidate_index"] for row in value["roles"]] == [5, 0, 4, 7]
    assert [row["candidate_label"] for row in value["roles"]] == [
        "rgb_goal_support_top_route",
        "zero",
        "rgb_goal_support_direct",
        "random_control",
    ]
    assert all(len(row["actions"]) == 5 for row in value["roles"])
    assert value["planner"]["prep_sha256"] == subject.PLAN_PREP_SHA
    assert value["planner"]["report_sha256"] == subject.PLAN_REPORT_SHA
    assert value["feedback_boundary"] == 5
    assert value["same_physical_horizon"] is True
    assert value["new_neural_calls_during_physical"] == 0
    assert value["P1_admission"] is False


def test_selected_and_greedy_are_rgb_constructed_and_actions_are_byte_bound():
    value = subject.build()
    rows = {row["role"]: row for row in value["roles"]}
    for role in ("selected", "same_bank_greedy"):
        assert rows[role]["constructed_from_rgb_contact_geometry"] is True
        assert rows[role]["contact_feasible_prior"] is True
    assert np.array_equal(
        np.asarray(rows["zero"]["actions"], np.float32),
        np.zeros((5, 2), np.float32),
    )
    for row in rows.values():
        assert subject._actions_sha(np.asarray(row["actions"], np.float32)) == row[
            "actions_sha256"
        ]


def test_contract_declares_rgb_only_source_and_binds_current_and_goal_images():
    value = subject.build()
    source = value["prefix_source"]
    assert source == {
        "mode": "rgb_only",
        "current_rgb_sha256": subject.sha(
            subject.source.source_directory() / "rgb_10.png"
        ),
        "goal_rgb_sha256": subject.sha(subject.source.source_directory() / "goal.png"),
        "simulator_state_used_for_prefix_generation": False,
        "physical_outcomes_used_for_selection": False,
        "contact_calibration": subject.RGB_CALIBRATION,
    }
    assert value["planner"]["physical_outcomes_used"] is False


def test_action_tamper_is_rejected_fail_closed():
    protocol, report = subject._planner_material()
    tampered = deepcopy(report)
    selected = next(row for row in tampered["rows"] if row["index"] == tampered["selected_index"])
    selected["actions"][0][0] += 0.125
    with pytest.raises(ValueError, match="selected V3 action binding"):
        subject._validated_roles(protocol, tampered)


def test_rgb_source_tamper_is_rejected_fail_closed():
    protocol, report = subject._planner_material()
    tampered = deepcopy(protocol)
    tampered["prefix_bank"]["simulator_state_used"] = True
    with pytest.raises(ValueError, match="RGB-only V3 prefix source"):
        subject._validate_rgb_only(tampered, report)


def test_role_index_alias_is_rejected_fail_closed():
    protocol, report = subject._planner_material()
    tampered = deepcopy(report)
    tampered["random_index"] = tampered["zero_index"]
    with pytest.raises(ValueError, match="four distinct V3 role indices"):
        subject._validated_roles(protocol, tampered)
