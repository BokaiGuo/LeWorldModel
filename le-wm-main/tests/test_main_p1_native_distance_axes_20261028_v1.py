from fractions import Fraction
import pytest
from experiments.spectral_reachability import main_p1_native_distance_axes_20261028_v1 as m


def test_exact_three_axes_cover_all_seven_coordinates():
    assert m.exact_axes([1, 2, 3, 4, 5, 6, 7], [0]*7) == dict(xy=Fraction(30), angle=Fraction(25), velocity=Fraction(85))


def test_velocity_can_reverse_pose_order_without_model_error():
    a = dict(index=0, actual_Q=1., _axes=dict(xy=Fraction(1), angle=Fraction(), velocity=Fraction(100)),
             physical_metrics=dict(native_distance=101**.5))
    b = dict(index=1, actual_Q=4., _axes=dict(xy=Fraction(4), angle=Fraction(), velocity=Fraction()),
             physical_metrics=dict(native_distance=2.))
    result = m.pair(a, b)
    assert result['latent_native_reversal'] and not result['latent_pose_reversal']
    assert result['velocity_flips_pose_order'] and result['native_float64_sign_equal_exact']


def test_missing_latent_retains_physical_decomposition():
    a = dict(index=0, actual_Q=None, _axes=dict(xy=Fraction(1), angle=Fraction(), velocity=Fraction()),
             physical_metrics=dict(native_distance=1.))
    b = dict(index=1, actual_Q=0., _axes=dict(xy=Fraction(4), angle=Fraction(), velocity=Fraction()),
             physical_metrics=dict(native_distance=2.))
    result = m.pair(a, b)
    assert result['latent_native_reversal'] is None and result['squared_native_sign'] == -1


@pytest.mark.parametrize('state', [[0]*6, [0]*6+[float('nan')], [0]*6+[float('inf')]])
def test_bad_state_rejected(state):
    with pytest.raises(ValueError):
        m.exact_axes(state, [0]*7)
