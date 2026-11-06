import numpy as np

from experiments.spectral_reachability import pc_global_reachability_v18 as v18
from experiments.spectral_reachability import pc_timed_feedback_execution_v2 as feedback
from experiments.spectral_reachability import pc_timed_program_execution_v2 as runner


def config():
    return v18.Config(action_dim=2, segment_length=5, segments=3,
                      nominals=(tuple((0., 0.) for _ in range(5)),), token_cap=1920)


def predictor(actions):
    features = actions.sum(axis=1)
    return features, np.linalg.norm(features - [1., -1.], axis=1)


def test_v18_and_cem_share_global_bank_and_meter():
    records = [runner.execute(predictor, [1., -1.], config(), arm=arm)
               for arm in ("v18", "multistart_cem")]
    np.testing.assert_array_equal(records[0]["traces"][0]["actions"], records[1]["traces"][0]["actions"])
    for result in records:
        assert result["resource"]["prediction_tokens"] <= 1920
        assert result["resource"]["prediction_tokens"] == sum(c["prediction_tokens"] for c in result["calls"])
    assert records[0]["result"]["child"]["layers"][0]["reachability"]


def test_v18_feedback_rescores_carry_at_remaining_horizons():
    decisions = []
    def prepare(context, step):
        return predictor, [1., -1.], {"observation": step}
    def seal(decision):
        decisions.append(decision)
        return {"step": decision["step"]}
    def dispatch(context, actions, receipt):
        return context + 1, receipt
    feedback.run(0, config(), arm="v18", execution_steps=5, prepare=prepare,
                 seal_decision=seal, dispatch=dispatch)
    assert [d["remaining_steps"] for d in decisions] == [15, 10, 5]
    assert all(d["carry_score"] is not None for d in decisions[1:])
