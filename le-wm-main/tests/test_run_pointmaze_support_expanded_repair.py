from __future__ import annotations

import numpy as np

from experiments.run_pointmaze_support_expanded_repair import (
    bootstrap_ci,
    recovery_decision,
)


def test_support_expanded_repair_requires_support_and_two_positive_utility_intervals() -> None:
    reference = {"joint_support_rate": 0.0, "release_rate": 0.0, "release_count": 0}
    repaired = {"joint_support_rate": 0.5, "release_rate": 0.25, "release_count": 2}
    positive = {"utility_ci_positive": True}
    decision = recovery_decision(reference, repaired, positive, positive)
    assert decision["claimed_success"] is True
    negative = recovery_decision(reference, repaired, positive, {"utility_ci_positive": False})
    assert negative["claimed_success"] is False


def test_support_expanded_bootstrap_interval_is_deterministic() -> None:
    values = np.asarray([0.1, 0.2, 0.3, 0.4])
    assert bootstrap_ci(values, seed=7, draws=500) == bootstrap_ci(values, seed=7, draws=500)
