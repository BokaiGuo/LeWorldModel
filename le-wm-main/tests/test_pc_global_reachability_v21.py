import numpy as np

from experiments.spectral_reachability import pc_global_reachability_v21 as v21
from experiments.spectral_reachability import pc_timed_program_execution_v5 as runner


def test_multichain_refinement_respects_total_budget_and_incumbent():
    config = v21.Config(action_dim=2, segment_length=5, segments=3,
                        nominals=(tuple((0., 0.) for _ in range(5)),), token_cap=4095, seed=21)
    def predict(actions):
        features = actions.sum(axis=1)
        return features, np.linalg.norm(features - [1., -1.], axis=1)
    record = runner.execute(predict, [1., -1.], config, arm="v21")
    child = record["result"]["child"]
    starts = child["refinement_starts"]
    assert 1 <= len(starts) <= config.refine_starts
    assert sum(s["samples"] for s in starts) == sum(g["batch_size"] for g in child["refinement"])
    assert max(s["samples"] for s in starts) - min(s["samples"] for s in starts) <= 1
    assert record["resource"]["prediction_tokens"] <= config.token_cap
    assert record["resource"]["prediction_tokens"] == sum(c["prediction_tokens"] for c in record["calls"])
    complete = [b["scores"] for b in record["traces"] if b["actions"].shape[1] == 15]
    assert record["result"]["selected_score"] <= min(float(scores.min()) for scores in complete) + 1e-12
