import runpy
from pathlib import Path
import pytest
from experiments.spectral_reachability import p2_h15_gn_cem_threeway_pairing_20261029_v1 as m

fixture=runpy.run_path(str(Path(__file__).parent/'test_p2_h15_cem_matched_pairing_20261029_v1.py'))
row=fixture['row']


def test_threeway_stops_and_pairwise_missing():
    value=m.pair_three(row('h15_sra',15),row('cem_matched',10),row('gn_matched',12))
    assert value['common_steps']==list(range(1,11))
    assert value['timeline'][10]['cem'] is None
    assert value['timeline'][10]['h15_minus_gn'] is not None
    assert value['timeline'][12]['gn'] is None and value['timeline'][12]['gn_minus_cem'] is None


@pytest.mark.parametrize('field',['root_sha256','task_id','root_state','goal','goal_pose','goal_pixel_sha256'])
def test_gn_identity_mismatch_fails(field):
    g=row('gn_matched',15);g[field]='wrong'
    with pytest.raises(ValueError):m.pair_three(row('h15_sra',15),row('cem_matched',15),g)


def test_resource_missing_is_not_zero():
    a=m.axes(dict(h15={'x':1},cem={'x':0},gn={'y':2}))
    assert a['x']==dict(h15=1,cem=0,gn=None)
    assert a['y']==dict(h15=None,cem=None,gn=2)


def test_fixed_arm_rejects_mislabel():
    with pytest.raises(ValueError):m.pair_three(row('h15_sra',15),row('cem_matched',15),row('cem_matched',15))
