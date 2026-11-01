import copy
import pytest
from experiments.spectral_reachability import p2_h15_cem_matched_pairing_20261029_v1 as m


def row(arm,n):
    return dict(arm=arm,root_slot=0,condition='off',initial_exact=True,task_id='t',root_sha256='r',
        root_state=[1],goal=[2],goal_pose=[3],goal_pixel_sha256='g',own_returned=n,
        status='native_stopped' if n<15 else 'horizon_completed',responses=[dict(step=i,
            native_metrics=dict(native_distance=float(i),object_position_error=float(i),
                wrapped_object_angle_error=0.,native_success=False),terminated=i==n and n<15,truncated=False)
            for i in range(1,n+1)])


def test_early_stop_never_pads_or_uses_last_value():
    result=m.pair(row('h15_sra',15),row('cem_matched',10))
    assert result['common_steps']==list(range(1,11))
    assert result['timeline'][9]['delta']['native_distance']==0
    assert result['timeline'][10]['cem'] is None and result['timeline'][10]['delta'] is None


@pytest.mark.parametrize('field',['task_id','root_state','goal','goal_pose','goal_pixel_sha256','root_sha256'])
def test_wrong_identity_is_rejected(field):
    b=row('cem_matched',15);b[field]='different'
    with pytest.raises(ValueError):m.pair(row('h15_sra',15),b)


def test_missing_or_out_of_order_step_rejected():
    b=row('cem_matched',15);b['responses'][3]['step']=5
    with pytest.raises(ValueError):m.pair(row('h15_sra',15),b)


def test_actual_resources_different_axes_not_fabricated():
    a={'predict_samples':132,'VJP_directions':192};b={'predict_samples':132,'VJP_directions':0}
    result=m.compare_axes(a,b)
    assert result['predict_samples']['difference_h15_minus_cem']==0
    assert result['VJP_directions']['difference_h15_minus_cem']==192
    assert result['predict_samples']['h15']==132


def test_missing_resource_is_null_not_zero():
    assert m.compare_axes({'a':1},{'b':2})['a']['cem'] is None
