import numpy as np
import gymnasium as gym

import stable_worldmodel  # noqa: F401 -- registers PushT

from experiments.audit_pusht_phase_conditioned_candidate_bank import (
    DEFAULT_CONTRACT,
    action_bank_fingerprint,
    phase_conditioned_bank,
    phase_from_observables,
    root_fingerprint,
)
from experiments.run_pusht_mode_aware_shared_candidate import make_boundary_contact_roots


def test_phase_contract_is_total_and_uses_declared_thresholds():
    contract = DEFAULT_CONTRACT
    assert phase_from_observables(contact_gap_px=contract.contact_gap_px + 0.01, xy_error_px=0.0) == "approach"
    assert phase_from_observables(contact_gap_px=contract.contact_gap_px, xy_error_px=contract.orientation_xy_tube_px + 0.01) == "contact"
    assert phase_from_observables(contact_gap_px=contract.contact_gap_px, xy_error_px=contract.orientation_xy_tube_px) == "orientation"


def test_candidate_bank_hash_is_deterministic_and_order_sensitive():
    bank = np.asarray([[0.2, -0.1], [0.3, 0.4]], dtype=np.float64)
    assert action_bank_fingerprint(bank) == action_bank_fingerprint(bank.copy())
    assert action_bank_fingerprint(bank) != action_bank_fingerprint(bank[::-1])


def test_real_pusht_phase_bank_is_deterministic_and_has_a_stable_hash():
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        root = make_boundary_contact_roots(env, count=1, seed=202610212)[0]
        phase_a, bank_a, observables_a = phase_conditioned_bank(env, root["state"], root["goal"])
        phase_b, bank_b, observables_b = phase_conditioned_bank(env, root["state"], root["goal"])
    finally:
        env.close()
    assert phase_a == phase_b
    assert observables_a == observables_b
    assert np.array_equal(bank_a, bank_b)
    assert action_bank_fingerprint(bank_a) == action_bank_fingerprint(bank_b)
    assert np.all(np.linalg.norm(bank_a, axis=1) > 0.0)


def test_candidate_hash_rejects_no_candidate_and_wrong_action_shape():
    for invalid in (np.empty((0, 2)), np.ones((3, 3)), np.asarray([0.1, 0.2])):
        try:
            action_bank_fingerprint(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid candidate bank must fail closed")


def test_root_contract_hash_is_equal_only_for_identical_state_goal_pairs():
    root = {"state": np.arange(7, dtype=np.float64), "goal": np.arange(7, 14, dtype=np.float64)}
    clone = {"state": root["state"].copy(), "goal": root["goal"].copy()}
    changed = {"state": root["state"].copy(), "goal": root["goal"].copy()}
    changed["goal"][0] += 1.0
    assert root_fingerprint(root) == root_fingerprint(clone)
    assert root_fingerprint(root) != root_fingerprint(changed)
