import runpy
from pathlib import Path
import pytest
from experiments.spectral_reachability import p2_h15_fourway_pairing_20261029_v1 as m

fixtures=runpy.run_path(str(Path(__file__).parent/'test_p2_h15_cem_matched_pairing_20261029_v1.py'))
row=fixtures['row']


def test_fourway_partial_timeline_and_deltas():
    h=row('h15_sra',15);c=row('cem_matched',10);g=row('gn_matched',12);d=row('gradient_matched',13)
    previous=m.three.pair_three(h,c,g);value=m.pair_four(previous,h,d)
    assert value['common_steps']==list(range(1,11))
    assert value['timeline'][10]['gradient_minus_cem'] is None
    assert value['timeline'][10]['gradient_minus_gn'] is not None
    assert value['timeline'][13]['gradient'] is None and value['timeline'][13]['gradient_minus_h15'] is None


@pytest.mark.parametrize('field',['root_sha256','task_id','root_state','goal','goal_pose','goal_pixel_sha256'])
def test_reject_wrong_gradient_identity(field):
    h=row('h15_sra',15);p=m.three.pair_three(h,row('cem_matched',15),row('gn_matched',15))
    grad=row('gradient_matched',15);grad[field]='mismatch'
    with pytest.raises(ValueError):m.pair_four(p,h,grad)


def test_missing_axes_stay_null():
    assert m.extend_axes({'x':{'h15':1,'cem':0,'gn':3}},{'y':5})=={
        'x':dict(h15=1,cem=0,gn=3,gradient=None),
        'y':dict(h15=None,cem=None,gn=None,gradient=5)}


def test_mislabeled_arm_and_reordered_step_fail():
    h=row('h15_sra',15);p=m.three.pair_three(h,row('cem_matched',15),row('gn_matched',15))
    with pytest.raises(ValueError):m.pair_four(p,h,row('gn_matched',15))
    grad=row('gradient_matched',15);grad['responses'][3]['step']=6
    with pytest.raises(ValueError):m.pair_four(p,h,grad)
