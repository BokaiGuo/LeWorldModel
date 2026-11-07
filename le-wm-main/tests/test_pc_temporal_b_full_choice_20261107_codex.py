import numpy as np

from experiments.pc_temporal_b_full_choice_20261107_codex import choose


def test_full_choice_uses_declared_time_window():
    actions = np.zeros((130, 25, 2), dtype=np.float32)
    actions[:, :, 0] = np.linspace(-0.9, 0.9, 130)[:, None]
    scores = np.full((130, 5), 10.0)
    scores[4, 0], scores[5, 2], scores[6, 4] = 0.0, 1.0, 2.0
    assert choose(actions, scores, "all_time")[0] == 4
    assert choose(actions, scores, "post_prefix")[0] == 5
    assert choose(actions, scores, "endpoint")[0] == 6
