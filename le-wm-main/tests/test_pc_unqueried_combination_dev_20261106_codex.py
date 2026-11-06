import numpy as np

from experiments.pc_unqueried_combination_dev_20261106_codex import evaluate, predict


def test_heldout_costs_do_not_affect_predictions():
    rng = np.random.default_rng(10)
    a, b = rng.normal(size=(4, 30)), rng.normal(size=(4, 30))
    costs = rng.normal(size=(4, 4))
    mask = np.fromfunction(lambda i, j: (i + j) % 4 != 0, (4, 4), dtype=int).astype(bool)
    parent = np.arange(4.)
    first, col1 = predict(a, b, costs, mask, parent, 3)
    changed = costs.copy()
    changed[~mask] += 100000.
    second, col2 = predict(a, b, changed, mask, parent, 3)
    assert all(np.array_equal(first[k], second[k]) for k in first)
    assert np.array_equal(col1, col2)


def test_evaluation_distinguishes_prefix_and_tail_success():
    pred = {"x": np.arange(16.).reshape(4, 4)}
    success = [[10, 11, None, None]] + [[None] * 4 for _ in range(3)]
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    assert evaluate(pred, np.zeros((4, 4)), success, mask)["x"]["prefix_already_success"]
    mask[0, 0], mask[0, 1] = False, True
    assert evaluate(pred, np.zeros((4, 4)), success, mask)["x"]["tail_success"]
