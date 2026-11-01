import numpy as np

from experiments.spectral_reachability import p1_unified_constrained_cem_physical_20261101_codex as subject


def test_contract_has_two_roots_and_three_fixed_roles():
    contract = subject.build()
    assert contract["roles"] == list(subject.ROLES)
    assert len(contract["roots"]) == 2
    assert contract["new_neural_calls"] == 0
    assert contract["physical_future_used_for_selection"] is False
    for root in contract["roots"]:
        selected = np.asarray(root["candidates"]["selected"], np.float32)
        zero = np.asarray(root["candidates"]["zero_prefix"], np.float32)
        greedy = np.asarray(root["candidates"]["same_bank_greedy"], np.float32)
        assert selected.shape == zero.shape == greedy.shape == (3, 5, 2)
        assert not zero[0].any()
        np.testing.assert_array_equal(zero[1:], selected[1:])
        assert not np.array_equal(selected, greedy)
        assert root["selection"]["selected_q1"] >= root["selection"]["root_mse"]
        assert root["selection"]["selected_q3"] < root["selection"]["root_mse"]


def test_root_inputs_match_contract_identity():
    contract = subject.build()
    for row in contract["roots"]:
        _, _, expected = subject.root_inputs(row["slot"])
        assert all(expected[key] == row["expected"][key] for key in row["expected"])


def test_physical_output_name_matches_inherited_store_contract():
    output = subject.ROOT / "results" / f"{subject.NAME}_run_TEST"
    previous = subject.engine.NAME
    try:
        subject.engine.NAME = subject.NAME
        assert subject.engine.exclusive(output, "run") == output
    finally:
        subject.engine.NAME = previous
