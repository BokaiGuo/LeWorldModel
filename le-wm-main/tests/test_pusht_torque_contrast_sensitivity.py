import numpy as np

from experiments.audit_pusht_torque_contrast_sensitivity import (
    RUNTIME_CONTRACT,
    SweepCell,
    assert_same_state_pairing,
    predeclared_sweep,
    triplet_specs,
)
from experiments.audit_pusht_orientation_response import matched_actions


def _actions():
    chart = {
        "contact_point_world": np.asarray([2.0, 0.0]),
        "contact_radius": np.asarray([2.0, 0.0]),
        "inward": np.asarray([1.0, 0.0]),
        "radial_tangent": np.asarray([0.0, 1.0]),
    }
    return matched_actions(chart, inward_magnitude=0.18, tangent_magnitude=0.55)


def test_triplet_contract_binds_zero_plus_minus_to_the_same_state_digest():
    state = np.asarray([1.0, 2.0, 3.0, 4.0, 0.2, 0.0, 0.0])
    specs = triplet_specs(state, _actions(), SweepCell(0.55, 2), root_id=7, root_seed=42)
    assert tuple(spec.label for spec in specs) == ("zero", "plus", "minus")
    assert len({spec.state_digest for spec in specs}) == 1
    assert len({spec.pulse_steps for spec in specs}) == 1
    assert specs[0].action == (0.0, 0.0)
    assert_same_state_pairing(specs)


def test_predeclared_sweep_is_deterministic_cartesian_and_rejects_duplicates():
    first = predeclared_sweep([0.25, 0.55], [1, 3])
    second = predeclared_sweep([0.25, 0.55], [1, 3])
    assert first == second
    assert [cell.key for cell in first] == ["t0.2500__k1", "t0.2500__k3", "t0.5500__k1", "t0.5500__k3"]
    try:
        predeclared_sweep([0.25, 0.25], [1])
    except ValueError as exc:
        assert "duplicate amplitudes" in str(exc)
    else:  # pragma: no cover - contract failure must be loud
        raise AssertionError("ambiguous duplicate sweep was accepted")


def test_artifact_contract_explicitly_forbids_runtime_or_learned_claims():
    assert RUNTIME_CONTRACT["uses_privileged_simulator_state_offline_only"]
    assert not RUNTIME_CONTRACT["runtime_action_selection"]
    assert not RUNTIME_CONTRACT["runtime_successor_verification"]
    assert not RUNTIME_CONTRACT["is_learned_ranker"]
    assert not RUNTIME_CONTRACT["is_planning_evaluation"]
