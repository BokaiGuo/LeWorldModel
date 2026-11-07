import ast
import inspect

import numpy as np

from experiments import pc_tworoom_block_heldout_effect_20261107_codex as exp


def test_predictions_use_action_identity_and_but_not_b_values():
    rng = np.random.default_rng(123)
    a, b = rng.normal(size=(8, 3, 2)), rng.normal(size=(8, 3, 2))
    cost, parent = rng.normal(size=(8, 8)), rng.normal(size=8)
    output = exp.predictions(a, b, cost, parent, 5)
    for values in output.values():
        assert values.shape == (8, 8)
        assert np.isfinite(values).all()
    perm = rng.permutation(8)
    changed = exp.predictions(a[perm], b, cost[:, perm], parent, 5)
    for method in ("parent", "row_mean", "additive", "identity"):
        np.testing.assert_allclose(output[method], changed[method])
    assert not np.allclose(output["identity"], output["shuffled"])


def test_unique_prefix_selection_and_sorted_cem():
    cem = np.zeros((6, 25, 2), np.float32)
    rand = np.zeros((6, 25, 2), np.float32)
    for i in range(6):
        cem[i, :10] = i
        rand[i, :10] = i + 10
    actual = exp.select_prefixes(cem, np.arange(6)[::-1], rand)
    np.testing.assert_array_equal(actual[:, 0, 0], [5, 4, 3, 2, 10, 11, 12, 13])


def test_plan_no_truth_inputs():
    tree = ast.parse(inspect.getsource(exp.plan))
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not ({"PHYSICAL", "RESULT", "proprio"} & attrs)
    assert not ({"PHYSICAL", "RESULT"} & names)
