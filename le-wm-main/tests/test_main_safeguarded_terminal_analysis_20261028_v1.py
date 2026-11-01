import numpy as np
import pytest
from copy import deepcopy
from experiments.spectral_reachability import main_safeguarded_terminal_analysis_20261028_v1 as m


def test_mse_dimension_scale_and_no_fit():
    z=np.zeros(192,np.float32); z[0]=2
    assert m.mse(z,np.zeros(192)) == 4/192


@pytest.mark.parametrize('a',[np.zeros(191),np.full(192,np.nan)])
def test_unknown_not_zero_or_synthetic_success(a):
    with pytest.raises(ValueError): m.mse(a,np.zeros(192))


def example():
    """Synthetic arithmetic only, never a neural/physical evidence fixture."""
    row=dict(index=0, status='encoded', root_slot=0, condition='off', arm='joint',
        goal192=[0.]*192, predicted192=[.5]*192, old_fixed_actual_Q=2.,
        old_fixed_actual_mid_predicted_Q=1., new_suffix_Q=.25)
    strict=dict(index=0, new_suffix_actual_Q=1., new_suffix_predicted_Q=.25,
        new_minus_old_actual_Q=-1., new_minus_old_predicted_Q=-.75, new_endpoint_MSE=.25)
    native=dict(index=0, root_slot=0, condition='off', arm='joint',
        incumbent_Q=1., selected_Q=.25,
        new_minus_control=dict(joint_xy_error=1., wrapped_angle_error=0.))
    return row, np.ones(192,np.float32), strict, native


def test_same_sign_latent_effect_does_not_imply_physical_benefit():
    result=m.compare(*example())
    assert result['predicted_to_actual_effect_reversal'] is False
    assert result['actual_to_jointXY_effect_reversal'] is True
    assert result['actual_to_angle_effect_reversal'] is False  # a tie is not a reversal


@pytest.mark.parametrize('field,value', [('root_slot',1),('condition','on'),('arm','conditional')])
def test_cannot_join_a_different_plan(field,value):
    row,z,strict,native=example(); native[field]=value
    with pytest.raises(ValueError): m.compare(row,z,strict,native)


def test_strict_metrics_and_compiler_objective_must_match_independent_arithmetic():
    row,z,strict,native=example()
    for target,field in [('strict','new_suffix_actual_Q'),('native','selected_Q'),('row','new_suffix_Q')]:
        r,s,n=deepcopy(row),deepcopy(strict),deepcopy(native)
        {'strict':s,'native':n,'row':r}[target][field] += .125
        with pytest.raises(ValueError): m.compare(r,z,s,n)


def test_literal_float32_terminal_required():
    row,z,strict,native=example()
    with pytest.raises(ValueError): m.compare(row,z.astype(np.float64),strict,native)
