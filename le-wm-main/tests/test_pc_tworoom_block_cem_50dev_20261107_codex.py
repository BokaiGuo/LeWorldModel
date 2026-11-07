import ast
import inspect

import numpy as np
import pytest

from experiments import pc_tworoom_block_cem_50dev_20261107_codex as exp


@pytest.mark.parametrize("blockwise,steps", [(False, 25), (True, 5)])
def test_cem_candidates_are_bounded_reproducible_and_include_mean(blockwise, steps):
    mean = np.full((steps, 2), 0.3)
    std = np.full((steps, 2), 2.0)
    low, high = np.array([-0.8, -0.5]), np.array([0.9, 0.7])
    args = (mean, std, low, high)
    samples = exp.normalized_cem_candidates(np.random.default_rng(23), *args, blockwise=blockwise)
    repeated = exp.normalized_cem_candidates(np.random.default_rng(23), *args, blockwise=blockwise)
    assert samples.shape == (100, steps, 2)
    np.testing.assert_array_equal(samples, repeated)
    np.testing.assert_array_equal(samples[0], mean)
    assert np.all(samples >= low) and np.all(samples <= high)
    if blockwise:
        expanded = np.repeat(samples, 5, axis=1)
        for i in range(5):
            np.testing.assert_array_equal(expanded[:, i * 5:i * 5 + 5],
                                          np.repeat(samples[:, i:i + 1], 5, axis=1))


def test_plan_does_not_read_physical_outcomes_or_reference_actions():
    tree = ast.parse(inspect.getsource(exp.plan))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not ({"PHYSICAL", "RESULT"} & names)
    assert not ({"PHYSICAL", "RESULT", "proprio"} & attrs)
    data_keys = {
        node.slice.value for node in ast.walk(tree)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
        and node.value.id == "data" and isinstance(node.slice, ast.Constant)
    }
    assert data_keys == {"rows", "current", "goal"}
