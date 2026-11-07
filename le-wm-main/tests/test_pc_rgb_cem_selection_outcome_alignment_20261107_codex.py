from experiments.pc_rgb_cem_selection_outcome_alignment_20261107_codex import compare_signs


def test_preference_orientation_and_ties():
    assert compare_signs([5, 4], [7, 6], [-5, -6]) == {
        "predicted_preference": 1, "real_encoded_preference": 1, "reward_preference": -1}
    assert compare_signs([1, 1 + 1e-7], [2, 2], [3, 3]) == {
        "predicted_preference": 0, "real_encoded_preference": 0, "reward_preference": 0}
