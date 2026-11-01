from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_e3_cell04_feedback_prepare_20261029_v1 as m
from experiments.spectral_reachability import agent_e3_frozen_model_task_resolver_20261029_v1 as resolver


def fixture():
    prior=dict(schema='fixed',cell=0,resolver={'sha256':'r'},control5={'prepare':{'sha256':'p'}},additional_sources={'/a':'a','/b':'b'})
    fresh=dict(schema='fixed',cell=4,resolver=deepcopy(prior['resolver']),control5=m.control5(prior),additional_sources={'/a':'a'})
    return prior,fresh


def test_fixed_roster_different_goal():
    row=resolver.roster()[m.CELL]
    assert row['cell_id']==m.TASK and (row['root'],row['goal_index'],row['horizon_packets'],row['domain'])==(0,1,1,'Dxy')
    prior,fresh=fixture();out=m.combine(prior,fresh)
    assert out['additional_sources']==prior['additional_sources'] and prior['cell']==0


@pytest.mark.parametrize('kind',['cell','resolver','prepare','report','receipt','source','new_source'])
def test_mismatch_fails_closed(kind):
    prior,fresh=fixture()
    if kind=='cell':fresh['cell']=8
    elif kind=='resolver':fresh['resolver']['sha256']='bad'
    elif kind=='prepare':fresh['control5']['prepare']['sha256']='bad'
    elif kind=='report':fresh['control5']['report']['sha256']='bad'
    elif kind=='receipt':fresh['control5']['receipt']['sha256']='bad'
    elif kind=='source':fresh['additional_sources']['/a']='bad'
    else:fresh['additional_sources']['/unknown']='bad'
    with pytest.raises(Exception):m.combine(prior,fresh)
