import numpy as np
import pytest

from experiments.pc_shared_action_identity_posthoc_20261107_codex import predict, validate_actions


def test_common_actions_and_prefixes_are_checked():
    actions = np.arange(48, dtype=float).reshape(8, 3, 2) / 100
    chains = np.zeros((8, 8, 25, 2))
    chains[:, :, 10:] = np.repeat(actions, 5, axis=1)[None]
    validate_actions(chains, actions)
    chains[2, 3, 11, 0] += .01
    with pytest.raises(ValueError, match="tails not shared"):
        validate_actions(chains, actions)


def test_matched_is_original_row_nearest_and_additive_is_preserved():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=(2, 8, 3, 2))
    cost = rng.normal(size=(8, 8))
    additive, matched, rows, cols = predict(a, b, cost)
    nearest = np.argsort(((b[:, None] - a[None]) ** 2).sum((2, 3)), axis=1, kind="stable")[:, :3]
    np.testing.assert_allclose(matched, cost[:, nearest].mean(axis=2))
    assert rows.shape == cols.shape == (7, 8, 8)
    # Constant prefix offsets alter every matching and shuffled prediction equally.
    offsets = np.arange(8)[:, None]
    add2, match2, rows2, cols2 = predict(a, b, cost + offsets)
    np.testing.assert_allclose(add2 - additive, np.broadcast_to(offsets, (8, 8)), atol=1e-12)
    np.testing.assert_allclose(match2 - matched, np.broadcast_to(offsets, (8, 8)), atol=1e-12)
    np.testing.assert_allclose(rows2 - rows, np.broadcast_to(offsets, (7, 8, 8)), atol=1e-12)
    np.testing.assert_allclose(cols2 - cols, np.broadcast_to(offsets, (7, 8, 8)), atol=1e-12)


def test_additive_data_leaves_no_interaction():
    a = np.arange(48, dtype=float).reshape(8, 3, 2)
    cost = np.arange(8)[:, None] + np.arange(8)[None, :]
    additive, matched, rows, cols = predict(a, a, cost)
    np.testing.assert_allclose(matched, additive)
    np.testing.assert_allclose(rows, np.broadcast_to(additive, rows.shape))
    np.testing.assert_allclose(cols, np.broadcast_to(additive, cols.shape))
