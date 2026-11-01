import pytest
from experiments.spectral_reachability.consume_mixedmode_physical_MAIN_20261028_v1 import independent


def test_native_success_does_not_use_velocity_reward_threshold():
    row = dict(goal=[0.]*7, final_state=[0.,0.,0.,0.,0.,100.,0.],
               native_success=True, final_metrics=dict(native_distance=100.), checkpoints=[])
    assert independent(row)['native_success'] is True
    row['native_success'] = False
    with pytest.raises(ValueError):
        independent(row)
