"""Check real-context replay token alignment against sealed planner contexts."""
import numpy as np

from experiments.spectral_reachability import p3_h3_real_context_replay_20261031 as replay


def test_all_saved_actual_history_tokens_match_planner_postcontext():
    entries, excluded, _ = replay.collect()
    roster = replay.nonlinear.load_roster()
    prior = {(index, row["index"], slot["policy"], slot["root"]): row
             for index, slot, _, horizon, row in roster if horizon == 3}
    assert len(entries) == 52 and len(excluded) == 10
    for item in entries:
        mean = np.asarray(item["task"]["action_mean"], dtype=np.float32)
        std = np.asarray(item["task"]["action_std"], dtype=np.float32)
        actual = ((item["actions"][5:15] - mean) / std).reshape(2, 10)
        recorded = np.asarray(prior[item["key"]]["postcontext"]["action_history"], dtype=np.float32)
        assert actual.tobytes() == recorded.tobytes()
