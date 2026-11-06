import numpy as np

from experiments.pc_paired_final_chain_intervention_posthoc_20261105_codex import intervene


def test_identical_controls_have_zero_gaps_and_matched_budget():
    lengths = []
    def predictor(actions):
        actions = np.asarray(actions)
        lengths.extend([actions.shape[1]] * len(actions))
        pos = actions.sum(axis=1)
        return pos, np.linalg.norm(pos - [1., 1.], axis=1)
    zero = np.zeros((15, 2), np.float32)
    result = intervene(predictor, {"start": [0, 0, 0], "goal": [1, 1]}, zero, zero, 0, 0)
    assert result["parent_comparison"]["common_improvement_delta"] == 0
    assert result["zero_comparison"]["replan_score_improvement"] == 0
    assert len(lengths) == 2 + 1 + 6 + 3*3*120
    assert result["nonprogress_001"]
