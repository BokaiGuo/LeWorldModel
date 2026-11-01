from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_e3_cell12_feedback_prepare_20261029_v1 as m
from experiments.spectral_reachability import agent_e3_frozen_model_task_resolver_20261029_v1 as resolver


def sample():
    prior=dict(schema='fixed',cell=0,resolver={'sha256':'r'},control5={'prepare':{'sha256':'p'}},additional_sources={'/a':'a','/b':'b'})
    fresh=dict(schema='fixed',cell=12,resolver=deepcopy(prior['resolver']),control5=m.control5(prior),additional_sources={'/a':'a'})
    return prior,fresh


def test_fixed_roster_and_preserved_source():
    v=resolver.roster()[12]
    assert (v['cell_id'],v['root'],v['goal_index'],v['horizon_packets'],v['domain'])==(m.TASK,1,1,1,'Dxy')
    a,b=sample();assert m.combine(a,b)['additional_sources']==a['additional_sources'] and a['cell']==0


@pytest.mark.parametrize('kind',['cell','resolver','prepare','report','receipt','source','new_source'])
def test_mismatch_rejected(kind):
    a,b=sample()
    if kind=='cell':b['cell']=4
    elif kind=='resolver':b['resolver']['sha256']='bad'
    elif kind=='prepare':b['control5']['prepare']['sha256']='bad'
    elif kind=='report':b['control5']['report']['sha256']='bad'
    elif kind=='receipt':b['control5']['receipt']['sha256']='bad'
    elif kind=='source':b['additional_sources']['/a']='bad'
    else:b['additional_sources']['/extra']='bad'
    with pytest.raises(Exception):m.combine(a,b)
