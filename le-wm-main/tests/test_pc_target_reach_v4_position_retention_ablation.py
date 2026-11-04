import numpy as np

from experiments.spectral_reachability import pc_target_reach_feedback_v4 as v4
from experiments.spectral_reachability import pc_target_reach_v4_position_retention_ablation as ablation
from tests.test_pc_target_reach_feedback_v4 import Meter


def test_ablation_restores_v4_and_uses_same_probes():
    target = np.asarray([.4, -.2])
    before = v4.retain
    full_meter, ablation_meter = Meter(target), Meter(target)
    full = v4.decide("response", 0, 0, target, full_meter.predict, full_meter)
    other = ablation.decide("response", 0, 0, target, ablation_meter.predict, ablation_meter)
    assert v4.retain is before
    assert full["first_bank"] == other["first_bank"]
    assert full["successor_responses"] == other["successor_responses"]
    assert full_meter.token_samples == ablation_meter.token_samples == 1920
