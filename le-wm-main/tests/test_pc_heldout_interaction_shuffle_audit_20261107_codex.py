import numpy as np

from experiments.pc_heldout_interaction_shuffle_audit_20261107_codex import decompose, predict


def test_decomposition_preserves_margins():
    cost = np.random.default_rng(42).normal(size=(8, 8))
    additive, interaction = decompose(cost)
    np.testing.assert_allclose(additive + interaction, cost)
    np.testing.assert_allclose(interaction.mean(axis=0), 0, atol=1e-15)
    np.testing.assert_allclose(interaction.mean(axis=1), 0, atol=1e-15)


def test_additive_cost_is_invariant_to_interaction_shuffle():
    rng = np.random.default_rng(5)
    a, b = rng.normal(size=(8, 3, 2)), rng.normal(size=(8, 3, 2))
    cost = np.arange(8)[:, None] + 3 * np.arange(8)[None, :]
    np.testing.assert_allclose(predict(a, b, cost), predict(a, b, cost, 17))


def test_shuffle_is_deterministic_and_changes_interaction():
    rng = np.random.default_rng(9)
    a, b, cost = rng.normal(size=(8, 3, 2)), rng.normal(size=(8, 3, 2)), rng.normal(size=(8, 8))
    np.testing.assert_array_equal(predict(a, b, cost, 42), predict(a, b, cost, 42))
    assert not np.allclose(predict(a, b, cost), predict(a, b, cost, 42))
