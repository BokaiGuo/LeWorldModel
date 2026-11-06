import numpy as np

from experiments.spectral_reachability import pc_staged_multistart_cem_dev as staged
from experiments.spectral_reachability import pc_v25_staged_refinement_dev as variant
from experiments import pc_v25_beam_resource_curve_dev_20261104_codex as panel
from experiments import pc_v25_staged_refinement_e0_dev_20261104_codex as experiment


def test_staged_budget_prefix_incumbent_and_determinism():
    prefix = np.full((5, 2), .2, np.float32)
    seeds = np.random.default_rng(9).uniform(-1, 1, (4, 10, 2)).astype(np.float32)

    def run():
        calls = []

        def evaluate(actions):
            calls.append(actions.copy())
            return np.square(actions).sum(axis=(1, 2))

        result = staged.optimize(evaluate, prefix, seeds, 260, seed=8)
        return result, calls

    result, calls = run()
    replay, replay_calls = run()
    assert sum(len(batch) for batch in calls) == 260
    assert sum(start["samples"] for start in result["starts"]) == 260
    assert sum(start["promoted"] for start in result["starts"]) == 2
    assert len(calls) == len(replay_calls)
    for a, b in zip(calls, replay_calls, strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[:, :5], np.broadcast_to(prefix, a[:, :5].shape))
    assert result["score"] == min(np.square(a).sum(axis=(1, 2)).min() for a in calls)
    np.testing.assert_array_equal(result["actions"], replay["actions"])


def test_staged_compiler_and_baseline_share_exact_cap():
    for arm in ("response", "cem"):
        calls = []

        def predict(actions):
            calls.append(actions.copy())
            features = actions.sum(axis=1)
            return features, np.linalg.norm(features - [1., -1.], axis=1)

        config = panel.config_for(0, "narrow", 1920)
        if arm == "response":
            result = variant.compile_program(predict, np.array([1., -1.]), config,
                                             score_scale=1., score_target=0.)
            reported = result["resource"]["prediction_tokens"]
        else:
            result = variant.plan_cem(predict, config, seed=config.seed)
            reported = result["prediction_tokens"]
        assert reported == sum(len(a) * (a.shape[1] // 5) for a in calls) <= 1920
        full = [a for a in calls if a.shape[1] == 15]
        best = min(np.linalg.norm(a.sum(axis=1) - [1., -1.], axis=1).min() for a in full)
        assert np.isclose(result["selected_score"], best)


def test_four_arms_share_global_proposal_bank():
    banks = []
    for arm in experiment.ARMS:
        calls = []

        def predict(actions):
            calls.append(actions.copy())
            features = actions.sum(axis=1)
            return features, np.linalg.norm(features - [1., -1.], axis=1)

        result = experiment.plan(predict, np.array([1., -1.]), experiment.config(1, 1920), arm)
        tokens, starts, _ = experiment.details(result, arm)
        assert tokens == sum(len(a) * (a.shape[1] // 5) for a in calls) <= 1920
        if arm.endswith("staged"):
            assert all(s.get("allocation_policy") == "staged" for s in starts)
        banks.append(calls[0])
    for bank in banks[1:]:
        np.testing.assert_array_equal(bank, banks[0])
