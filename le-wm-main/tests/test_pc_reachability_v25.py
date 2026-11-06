import numpy as np
import pytest

from experiments.spectral_reachability import pc_global_reachability_v25 as compiler
from experiments.spectral_reachability import pc_recursive_reachability_v25 as core


def test_response_winner_displaces_same_bank_diversity_not_greedy():
    scores = np.array([0., 1., 2., 3., 4.])
    features = np.array([[0.], [1.], [10.], [4.], [6.]])
    reachable = [{"candidate": 0, "value": 10.}, {"candidate": 3, "value": 0.}]
    alternatives = core.retention_choices(scores, features, reachable, 3, 2, 0)
    assert alternatives["diverse"] == [0, 1, 2]
    assert alternatives["response"] == [0, 1, 3]
    assert alternatives["greedy"] == [0, 1, 2]
    assert core.retention_choices(scores, features, reachable[:1], 3, 2, 0)["response"] == alternatives["diverse"]


@pytest.mark.parametrize("policy", ["response", "greedy", "diverse", "random"])
def test_comparable_probes_and_actual_prediction_budget(policy):
    nominals = (tuple((0., 0.) for _ in range(5)), tuple((.25, .5) for _ in range(5)))
    config = compiler.Config(action_dim=2, segment_length=5, segments=3,
                             nominals=nominals, token_cap=1920, seed=2,
                             max_modes=2, retention_policy=policy)
    batches = []

    def predict(actions):
        batches.append(actions.copy())
        features = actions.sum(axis=1)
        return features, np.linalg.norm(features - [1., -1.], axis=1)

    result = compiler.compile_program(predict, [1., -1.], config,
                                      score_scale=1., score_target=0.)
    layers = result["child"]["layers"]
    assert all(layer["retained"] == layer["retention_alternatives"][policy] for layer in layers)
    assert all(len(layer["reachability"]) != 1 for layer in layers)
    assert len(layers[0]["reachability"]) >= 2
    assert layers[0]["reachability"][1]["candidate"] not in layers[0]["retention_alternatives"]["greedy"][:2]
    assert result["resource"]["prediction_tokens"] == sum(len(b) * b.shape[1] // 5 for b in batches)
    assert result["resource"]["prediction_tokens"] <= 1920
