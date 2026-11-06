import numpy as np

from experiments.spectral_reachability import pc_global_reachability_v20 as v20
from experiments.spectral_reachability import pc_timed_program_execution_v4 as runner


def test_candidate_trust_filters_only_retention_and_counts_all_scored_chains():
    config = v20.Config(action_dim=2, segment_length=5, segments=3,
                        nominals=(tuple((.2, .2) for _ in range(5)),), token_cap=4095,
                        trust_relative=.01, trust_absolute=.001)
    def nonlinear(actions):
        sums = actions.sum(axis=1)
        features = sums**2
        return features, np.linalg.norm(features - [4., 2.], axis=1)
    result = runner.execute(nonlinear, [4., 2.], config, arm="v20")
    layers = result["result"]["child"]["layers"]
    assert any(not item["admitted"] for layer in layers for item in layer["trust"])
    assert all(layer["trust"][index]["admitted"] for layer in layers for index in layer["retained"])
    assert result["resource"]["prediction_tokens"] == sum(c["prediction_tokens"] for c in result["calls"])
    chosen = np.asarray(result["result"]["selected"], np.float32)
    assert any(batch["actions"].shape[1:] == chosen.shape and
               np.any(np.all(batch["actions"] == chosen, axis=(1, 2))) for batch in result["traces"])
