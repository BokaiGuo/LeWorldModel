import numpy as np
import pytest

from experiments.pc_lineage_cross import core


def bank(n=16):
    return core.PairedBank(np.random.default_rng(438).uniform(-1, 1, (n, 15, 2)))


def test_original_program_identity_is_preserved_and_immutable():
    candidate = bank()
    np.testing.assert_array_equal(candidate.compose(candidate.lineage), candidate.nominals)
    with pytest.raises(ValueError):
        candidate.nominals[0] = 1


def test_reject_noninteger_actions_and_invalid_time():
    candidate = bank()
    with pytest.raises(ValueError, match="integer"):
        candidate.compose([[1.1, 2.]])
    with pytest.raises(ValueError, match="five-step"):
        core.PairedBank(candidate.nominals, 6)


def test_paid_witnesses_budget_and_incumbent():
    candidate = bank()

    def evaluate(actions):
        return (actions ** 2).sum((1, 2)), np.zeros((len(actions), 2)), 0.

    ledger = core.Ledger(evaluate, 128)
    observed, selected = core.acquire(candidate, ledger)
    assert 113 <= ledger.count <= 116
    np.testing.assert_array_equal(ledger.trace[0]["actions"], candidate.nominals)
    assert len(ledger.trace[0]["costs"]) == 16
    mask = core.support(candidate.shape, selected)
    predicted, _ = core.prediction("cross_interaction", observed, selected, 54)
    pairs = core.choose(predicted, mask, 128 - ledger.count)
    ledger.query(candidate.compose(pairs), "proposals", pairs)
    assert ledger.count == 128
    assert ledger.best()["score"] <= min(ledger.trace[0]["costs"])
    identities = np.concatenate([b["pairs"] for b in ledger.trace])
    assert len(np.unique(identities, axis=0)) == 128


@pytest.mark.parametrize("n", [8, 16])
def test_low_rank_recovery_and_future_truth_rejection(n):
    rng = np.random.default_rng(45)
    matrix = rng.normal(size=(n, 1)) + rng.normal(size=(1, n + 1)) + rng.normal(size=(n, 2)) @ rng.normal(size=(2, n + 1))
    selected = core.anchors(matrix[:, :4])
    mask = core.support(matrix.shape, selected)
    observed = np.where(mask, matrix, np.nan)
    recovered, diagnostic = core.interaction(observed, selected)
    np.testing.assert_allclose(recovered, matrix, atol=1e-10, rtol=0)
    assert diagnostic["retained_rank"] == 2
    with pytest.raises(ValueError, match="unknown"):
        core.prediction("additive", matrix, selected, 4)


def test_shuffling_preserves_nominal_and_A_costs():
    truth = np.random.default_rng(4).normal(size=(16, 17))
    selected = core.anchors(truth[:, :4])
    mask = core.support(truth.shape, selected)
    observed = np.where(mask, truth, np.nan)
    prediction, diagnostic = core.prediction("shuffled", observed, selected, 25)
    changed = np.asarray(diagnostic["shuffled_observed"], dtype=float)
    np.testing.assert_array_equal(changed[:, :4], observed[:, :4])
    np.testing.assert_array_equal(changed[np.arange(16), np.arange(1, 17)],
                                  observed[np.arange(16), np.arange(1, 17)])
    for i in selected:
        np.testing.assert_array_equal(np.sort(changed[i]), np.sort(observed[i]))
    np.testing.assert_array_equal(prediction[mask], observed[mask])


def test_all_arms_use_paid_nominals_at_fixed_budget():
    from experiments.pc_lineage_cross import run
    row = run.plan_one({"start": [-.5, .1, .3], "goal": [.5, -.2]}, 998, 0)
    for arm, result in row["outcomes"].items():
        batches = run.batches(row, arm)
        assert sum(len(b["costs"]) for b in batches) == 128
        np.testing.assert_array_equal(batches[0]["actions"], row["nominals"])
        assert result["choice"]["score"] <= min(batches[0]["costs"])
    assert row["actual_runner_queries"] == row["shared_queries"] + 7 * (128 - row["shared_queries"]) + 448


def test_post_selection_evaluation_requires_audit(tmp_path, monkeypatch):
    from experiments.pc_lineage_cross import run
    monkeypatch.setattr(run, "OUT", tmp_path)
    with pytest.raises(FileNotFoundError, match="audit.json"):
        run.evaluate()
