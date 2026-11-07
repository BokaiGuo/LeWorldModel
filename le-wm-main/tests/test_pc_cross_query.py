import numpy as np
import pytest

from experiments.pc_cross_query import core


def observed(matrix):
    anchors = core.anchor_rows(matrix[:, :4])
    mask = core.cross_mask(anchors)
    return np.where(mask, matrix, np.nan), anchors, mask


def test_rank_two_interaction_exact_reconstruction():
    rng = np.random.default_rng(45)
    matrix = (rng.normal(size=(8, 1)) + rng.normal(size=(1, 32))
              + rng.normal(size=(8, 2)) @ rng.normal(size=(2, 32)))
    costs, anchors, mask = observed(matrix)
    predicted, diagnostic = core.interaction_prediction(costs, anchors)
    np.testing.assert_allclose(predicted, matrix, atol=1e-10)
    assert diagnostic["retained_rank"] == 2
    assert mask.sum() == 116


@pytest.mark.parametrize("constant", [True, False])
def test_additive_and_degenerate_cases(constant):
    matrix = np.ones((8, 32)) if constant else np.arange(8)[:, None] + np.arange(32)[None, :]
    costs, anchors, _ = observed(matrix)
    for prediction in (core.additive_prediction(costs, anchors),
                       core.interaction_prediction(costs, anchors)[0]):
        np.testing.assert_allclose(prediction, matrix, atol=1e-12)
    assert len(np.unique(anchors)) == 3


def test_dense_future_truth_rejected():
    matrix = np.arange(256).reshape(8, 32)
    _, anchors, _ = observed(matrix)
    with pytest.raises(ValueError, match="unknown"):
        core.predict_all(matrix, anchors, 6)


def test_shuffling_preserves_marginals_but_breaks_identity():
    costs, anchors, _ = observed(np.random.default_rng(4).normal(size=(8, 32)))
    changed = core.shuffle_identity(costs, anchors, 6)
    np.testing.assert_array_equal(changed[:, :4], costs[:, :4])
    for row in anchors:
        np.testing.assert_array_equal(np.sort(changed[row, 4:]), np.sort(costs[row, 4:]))
        assert not np.array_equal(changed[row, 4:], costs[row, 4:])


def test_all_proposals_are_unknown_and_deterministic():
    costs, anchors, mask = observed(np.arange(256).reshape(8, 32))
    values, _ = core.predict_all(costs, anchors, 9)
    for prediction in values.values():
        pairs = core.proposals(prediction, mask)
        assert pairs.shape == (12, 2)
        assert len(np.unique(pairs, axis=0)) == 12
        assert not mask[pairs[:, 0], pairs[:, 1]].any()
        np.testing.assert_array_equal(pairs, core.proposals(prediction, mask))


def test_composition_time_and_hold_contract():
    prefixes = np.full((8, 5, 2), .3)
    tails = np.zeros((32, 10, 2))
    tails[1] = -.2
    chain = core.compose(prefixes, tails, [[3, 1]])
    np.testing.assert_array_equal(chain[0, :5], prefixes[3])
    np.testing.assert_array_equal(chain[0, 5:], tails[1])
    tails[0, 0, 0] = 1.
    with pytest.raises(ValueError, match="hold"):
        core.compose(prefixes, tails, [[3, 1]])


def test_budget_and_incumbent_are_enforced():
    def model(actions):
        return actions[:, 0, 0], np.zeros((len(actions), 2)), 0.

    ledger = core.Ledger(model, 2)
    ledger.query(np.zeros((1, 15, 2)), "first")
    ledger.query(np.ones((1, 15, 2)), "last")
    assert ledger.best()["query_index"] == 0
    with pytest.raises(ValueError, match="budget"):
        ledger.query(np.zeros((1, 15, 2)), "overflow")
    assert ledger.count == 2


def test_cross_queries_count_and_no_duplicate_pairs():
    def model(actions):
        return np.square(actions).sum((1, 2)), np.zeros((len(actions), 2)), 0.

    rng = np.random.default_rng(45)
    prefixes = rng.uniform(-1, 1, (8, 5, 2))
    tails = rng.uniform(-1, 1, (32, 10, 2))
    tails[0] = 0.
    ledger = core.Ledger(model, 116)
    costs, anchors = core.acquire_cross(prefixes, tails, ledger)
    pairs = np.concatenate([x["pairs"] for x in ledger.trace])
    assert len(np.unique(pairs, axis=0)) == 116
    assert ledger.count == 116
    assert np.isfinite(costs).sum() == 116
    core.validate_observed(costs, anchors)


def test_fifteen_step_backend_matches_padding_and_independent_scalar():
    from experiments.pc_cross_query import audit, exact_run
    task = {"start": [-.5, .2, .4], "goal": [.5, -.2]}
    actions = np.random.default_rng(43).uniform(-1, 1, (16, 15, 2))
    path, _ = exact_run.trajectory(task["start"], actions)
    np.testing.assert_allclose(path[:, -1, :2], exact_run.previous.base.exact.endpoints(task, actions),
                               atol=1e-12, rtol=0)
    for chain, result in zip(actions, path, strict=True):
        np.testing.assert_allclose(audit.scalar_trajectory(task["start"], chain), result,
                                   atol=1e-12, rtol=0)


def test_evaluation_requires_sealed_audit(tmp_path, monkeypatch):
    from experiments.pc_cross_query import exact_run
    monkeypatch.setattr(exact_run, "OUT", tmp_path)
    with pytest.raises(FileNotFoundError, match="audit.json"):
        exact_run.evaluate()


def test_small_plan_preserves_incumbent_and_counts():
    from experiments.pc_cross_query import exact_run
    task = {"start": [-.5, .1, .3], "goal": [.5, -.2]}
    row = exact_run.plan_one(task, 998, 0)
    assert row["actual_runner_queries"] == 828
    assert len(row["outcomes"]) == 11
    for arm, outcome in row["outcomes"].items():
        assert outcome["query_count"] == 128
        assert outcome["dynamics_steps"] == 1920
        if outcome["uses_shared_cross"]:
            assert outcome["choice"]["score"] <= row["shared_choice"]["score"]
