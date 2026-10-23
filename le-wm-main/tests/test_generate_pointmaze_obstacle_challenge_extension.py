import numpy as np

from experiments.decision_benchmark.generate_pointmaze_obstacle_challenge_extension import (
    _direct_segment_blocked,
    _is_free,
    _sample_start_labels,
    obstacle_intersection_starts,
)


def test_obstacle_challenge_starts_are_free_but_direct_path_blocked() -> None:
    goal = np.asarray([1.8, 2.8])
    starts = obstacle_intersection_starts(goal)
    assert len(starts) >= 2
    assert all(_is_free(start, clearance=0.08) for start in starts)
    assert all(_direct_segment_blocked(start, goal, clearance=0.08) for start in starts)


def test_balanced_start_labels_are_fixed_before_continuous_jitter() -> None:
    labels = _sample_start_labels(episode_count=24, start_count=4, rng=np.random.default_rng(7), balanced_start_cells=True)
    assert len(labels) == 24
    assert [labels.count(index) for index in range(4)] == [6, 6, 6, 6]
