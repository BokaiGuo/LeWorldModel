import numpy as np
import pytest

from experiments.spectral_reachability import pc_global_reachability_v22 as v22
from experiments.spectral_reachability import pc_timed_program_execution_v6 as runner


@pytest.mark.parametrize("segments", [1, 2, 3])
def test_refinement_reserve_produces_multiple_generations_at_each_horizon(segments):
    nominals = (tuple((0., 0.) for _ in range(5)), tuple((.25, .5) for _ in range(5)))
    config = v22.Config(action_dim=2, segment_length=5, segments=segments,
                        nominals=nominals, token_cap=1920, seed=22)
    def predict(actions):
        features = actions.sum(axis=1)
        return features, np.linalg.norm(features - [1., -1.], axis=1)
    result = runner.execute(predict, [1., -1.], config, arm="v22")
    child = result["result"]["child"]
    assert child["refinement_starts"]
    for index, start in enumerate(child["refinement_starts"]):
        assert start["samples"] >= 24
        assert sum(g["batch_size"] for g in child["refinement"] if g["restart"] == index) == start["samples"]
        assert sum(g["restart"] == index for g in child["refinement"]) >= 2
    assert result["resource"]["prediction_tokens"] <= 1920


def test_rejects_config_without_two_generations_per_start():
    config = v22.Config(action_dim=2, segment_length=5, segments=1,
                        nominals=(tuple((0., 0.) for _ in range(5)),),
                        min_refine_samples=12)
    def predict(actions):
        features = actions.sum(axis=1)
        return features, np.linalg.norm(features, axis=1)
    with pytest.raises(ValueError, match="invalid configuration"):
        runner.execute(predict, [0., 0.], config, arm="v22")
