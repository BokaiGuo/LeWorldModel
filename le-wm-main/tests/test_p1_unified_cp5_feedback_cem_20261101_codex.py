import numpy as np

from experiments.spectral_reachability import p1_unified_cp5_feedback_cem_20261101_codex as subject


def test_variants_and_robust_score_are_fixed():
    raw = np.zeros((2, 5, 2), np.float32)
    variants = subject.action_variants(raw)
    assert variants.shape == (5, 2, 5, 2)
    np.testing.assert_array_equal(variants[0], raw)
    score = subject.robust_scores([[1, 2, 3, 4, 5]])
    np.testing.assert_allclose(score, [7.0])


def test_bank_preserves_fixed_seed_tails():
    rng = np.random.default_rng(7)
    old = np.full((2, 5, 2), 0.2, np.float32)
    zero = np.zeros_like(old)
    bank = subject.make_bank(rng, old, np.full_like(old, 0.1), [old, zero], 0)
    np.testing.assert_array_equal(bank[0], old)
    np.testing.assert_array_equal(bank[1], zero)


def test_protocol_uses_only_cp5_inputs_and_skips_early_success():
    value = subject.protocol()
    assert value["post_cp5_physical_outcomes_used"] is False
    assert value["budget"]["rgb_encode_calls"] == 5
    assert len(value["rows"]) == 6
    skipped = [row for row in value["rows"] if row["status"] == "early_success_no_replan"]
    assert [(row["slot"], row["role"], row["stop_step"]) for row in skipped] == [
        (1, "same_bank_greedy", 4)
    ]
    for row in value["rows"]:
        if row["status"] == "replan":
            assert len(row["prefix"]) == 5
            assert len(row["old_tail"]) == 2
            assert len(row["seed_tails"]) >= 2
