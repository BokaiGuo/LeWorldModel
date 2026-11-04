import numpy as np
import pytest

from experiments.spectral_reachability import pc_global_reachability_v23 as compiler
from experiments.spectral_reachability import pc_shared_multistart_cem as cem
from experiments import pc_v23_pointmaze_feedback_adapter as feedback
from experiments import pc_v23_pointmaze_live_feedback as live
from experiments.pc_equal_resource_dev_20261103_codex import Meter


def test_common_optimizer_counts_every_action_and_preserves_prefix():
    prefix = np.array([[0.2, -0.3]], np.float32)
    tails = np.array([np.zeros((3, 2)), np.full((3, 2), 0.4)], np.float32)
    trace = []

    def evaluate(actions):
        trace.append(actions.copy())
        return np.square(actions[:, 1:] - 0.25).sum(axis=(1, 2))

    result = cem.optimize(evaluate, prefix, tails, 27, seed=7, batch_size=4, elites=2)
    assert sum(len(batch) for batch in trace) == result["samples"] == 27
    assert sum(start["samples"] for start in result["starts"]) == 27
    assert all(len(start["generations"]) >= 2 for start in result["starts"])
    assert all(batch.dtype == np.float32 and np.all(batch[:, :1] == prefix) for batch in trace)
    assert all(np.abs(batch).max() <= 1 for batch in trace)
    scored = np.concatenate(trace)
    scores = np.square(scored[:, 1:] - 0.25).sum(axis=(1, 2))
    assert np.array_equal(result["actions"], scored[np.argmin(scores)])
    assert result["score"] == float(scores.min())


def test_common_optimizer_rejects_underbudget_and_invalid_scores():
    tails = np.zeros((2, 3, 2), np.float32)
    with pytest.raises(ValueError):
        cem.optimize(lambda x: np.zeros(len(x)), np.empty((0, 2)), tails, 17,
                     batch_size=4, elites=2)
    with pytest.raises(ValueError):
        cem.optimize(lambda x: np.full(len(x), np.nan), np.empty((0, 2)), tails, 18,
                     batch_size=4, elites=2)


@pytest.mark.parametrize("segments", [1, 2, 3])
def test_v23_compiler_scores_selected_chain_and_respects_budget(segments):
    nominals = (tuple((0., 0.) for _ in range(5)),
                tuple((.25, .5) for _ in range(5)))
    config = compiler.Config(action_dim=2, segment_length=5, segments=segments,
                             nominals=nominals, token_cap=1920, seed=22)
    calls = []

    def predict(actions):
        features = actions.sum(axis=1)
        scores = np.linalg.norm(features - [1., -1.], axis=1)
        calls.append((actions.copy(), scores.copy()))
        return features, scores

    result = compiler.compile_program(predict, [1., -1.], config,
                                      score_scale=1., score_target=0.)
    assert result["resource"]["prediction_tokens"] == sum(
        len(actions) * actions.shape[1] // 5 for actions, _ in calls)
    assert result["resource"]["prediction_tokens"] <= 1920
    selected = np.asarray(result["selected"], np.float32)
    assert any(np.any(np.all(actions == selected, axis=(1, 2))
                      & (scores == result["selected_score"]))
               for actions, scores in calls if actions.shape[1] == len(selected))
    assert all(start["samples"] >= 25 and len(start["generations"]) >= 2
               for start in result["child"]["refinement_starts"])


@pytest.mark.parametrize("step", [0, 1, 2])
def test_feedback_arms_share_bank_carry_and_optimizer(step):
    horizon = 15 - 5 * step
    carry = None if step == 0 else np.full((horizon, 2), .1, np.float32)
    first_batches = []
    for arm in ("response", "cem"):
        meter, batches = Meter(feedback.CAP), []

        def predict(actions):
            def raw(batch):
                batches.append(batch.copy())
                features = batch.sum(axis=1)
                return features, np.linalg.norm(features - [1., -1.], axis=1)
            return meter.predict(actions, raw)

        result = feedback.decide(arm, 0, step, np.array([1., -1.]), predict, meter, carry)
        assert np.asarray(result["selected"]["chain"]).shape == (horizon, 2)
        assert result["resource"]["prediction_tokens"] == meter.token_samples <= feedback.CAP
        if carry is not None:
            np.testing.assert_array_equal(batches[0][1], carry)
        first_batches.append(batches[0])
    np.testing.assert_array_equal(*first_batches)


def test_live_contract_rejects_stopped_root_and_unsafe_output():
    contract = {"start": [0.7, 0.7], "root_id": 2, "output_name": "pc_v23_new",
                "scope": "development", "goal_sha256": "goal"}
    _, root, name = live.validate_contract(contract, "goal")
    assert (root, name) == (2, "pc_v23_new")
    for changes in ({"start": [0.85, 0.85]}, {"output_name": "pc_v23_../escape"},
                    {"root_id": True}, {"scope": "confirm"}, {"goal_sha256": "wrong"}):
        with pytest.raises(ValueError):
            live.validate_contract(dict(contract, **changes), "goal")
