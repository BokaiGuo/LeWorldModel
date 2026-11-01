from copy import deepcopy
import pytest
from experiments.spectral_reachability import main_p2_feedback_effect_20261028_v1 as m


def rows():
    return [dict(index=i, root_slot=i//4, condition=('off', 'on')[i//2 % 2], arm=m.ARMS[i % 2],
        original_Q=float(i % 2), feedback_Q=float(1-i % 2), actual_Q=float(1-i % 2),
        original_MSE=2., feedback_MSE=1., original_Q_error=2., feedback_Q_error=1.,
        joint_xy_error=float(1-i % 2), object_xy_error=float(i % 2), wrapped_angle_error=0.,
        native_success=False) for i in range(8)]


def test_fixed_four_pairs_and_separate_physical_axes():
    result=m.assemble(rows())
    assert result['endpoint_MSE_improved']==8 and result['absolute_Q_error_improved']==8
    counts=result['pair_order_counts']
    assert counts['original_to_actual_Q_order']['reverse']==4
    assert counts['feedback_to_actual_Q_order']['agree']==4
    assert counts['actual_Q_to_objectXY_order']['reverse']==4
    assert counts['actual_Q_to_angle_order']['unknown_or_tie']==4
    assert not result['P1_admission'] and not result['P2_admission']


@pytest.mark.parametrize('fault', ['missing', 'order', 'condition', 'arm'])
def test_denominator_and_identity_fail_closed(fault):
    value=deepcopy(rows())
    if fault=='missing':value.pop()
    elif fault=='order':value.reverse()
    elif fault=='condition':value[0]['condition']='on'
    else:value[0]['arm']='bad'
    with pytest.raises(ValueError):m.assemble(value)


def test_unknown_retained_and_ties_not_claimed_agreement():
    value=rows();value[0]['actual_Q']=None
    result=m.assemble(value)
    assert len(result['pairs'])==4 and result['pair_order_counts']['feedback_to_actual_Q_order']['unknown_or_tie']==1
    assert m.agree(0., 1.) is None and m.agree(None, -1.) is None
