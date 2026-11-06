import numpy as np

from experiments.spectral_reachability import pc_global_reachability_v19 as v19
from experiments.spectral_reachability import pc_timed_feedback_execution_v3 as feedback
from experiments.spectral_reachability import pc_timed_program_execution_v3 as runner


def test_corrected_compiler_cem_and_feedback_share_execution_contract():
    config = v19.Config(action_dim=2, segment_length=5, segments=3,
                        nominals=(tuple((0., 0.) for _ in range(5)),), token_cap=1920)
    def predict(actions):
        features = actions.sum(axis=1)
        return features, np.linalg.norm(features - [1., -1.], axis=1)
    method = runner.execute(predict, [1., -1.], config, arm="v19")
    baseline = runner.execute(predict, [1., -1.], config, arm="multistart_cem")
    np.testing.assert_array_equal(method["traces"][0]["actions"], baseline["traces"][0]["actions"])
    assert method["resource"]["prediction_tokens"] <= config.token_cap
    decisions = []
    def prepare(context, step):
        return predict, [1., -1.], {"observation": step}
    def seal(decision):
        decisions.append(decision)
        return decision["step"]
    def dispatch(context, actions, receipt):
        return context + 1, receipt
    feedback.run(0, config, arm="v19", execution_steps=5, prepare=prepare,
                 seal_decision=seal, dispatch=dispatch)
    assert [decision["remaining_steps"] for decision in decisions] == [15, 10, 5]
    assert all(decision["carry_score"] is not None for decision in decisions[1:])
