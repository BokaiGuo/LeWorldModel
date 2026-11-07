import numpy as np

from experiments.pc_local_perturb_independent_replay_20261107_codex import reconstruct, recompute_choices
from experiments.pc_tworoom_local_perturb_stability_20261107_codex import perturb, choose


def test_independent_reconstruction_and_choice_parity():
    parents = np.repeat(np.random.default_rng(7).uniform(-1, 1, (8, 5, 2)).astype(np.float32), 5, axis=1)
    actual, _ = perturb(parents, 99)
    np.testing.assert_array_equal(actual, reconstruct(parents, 99))
    costs = np.random.default_rng(1).normal(size=64)
    scores = np.random.default_rng(2).normal(size=(8, 4))
    original = choose(costs, scores, 3)
    rerun = recompute_choices(costs, scores, np.asarray(original["mapping"]))
    assert all(rerun[key] == original[key] for key in rerun)


def test_direct_new_child_index():
    costs = np.arange(64, dtype=float)
    scores = np.full((8, 4), 100.)
    scores[7, 3] = -1.
    mapping = np.roll(np.arange(8), 1)
    assert recompute_choices(costs, scores, mapping)["direct"] == 95
