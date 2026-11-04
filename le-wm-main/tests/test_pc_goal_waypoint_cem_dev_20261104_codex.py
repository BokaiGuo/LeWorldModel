import numpy as np

from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability.pc_goal_waypoint_cem_dev_20261104_codex import plan


def test_waypoint_search_tracks_full_incumbent_and_budget():
    meter = Meter(1920)
    trace = []

    def predict(actions):
        actions = np.asarray(actions, np.float32)
        def model(batch):
            position = batch.sum(axis=1) * .1
            return position, np.linalg.norm(position - np.array([1., 0.]), axis=1)
        position, score = meter.predict(actions, model)
        trace.append((actions.copy(), score.copy()))
        return position, score

    result = plan(predict, meter, [1., 0.], 42)
    np.testing.assert_allclose(result["waypoints"][0]["waypoint"], [.5, -1.2])
    np.testing.assert_allclose(result["waypoints"][1]["waypoint"], [.5, 1.2])
    assert meter.token_samples <= 1920
    all_full = np.concatenate([s for a, s in trace if a.shape[1] == 15])
    assert result["selected_score"] == all_full.min()
    np.testing.assert_array_equal(np.asarray(result["selected"], np.float32),
                                  trace[-1][0][int(result["abstained"])])
    assert all(len(item["prefix_generations"]) > 1 for item in result["waypoints"])
    assert all(len(item["suffix_generations"]) > 1 for item in result["waypoints"])
