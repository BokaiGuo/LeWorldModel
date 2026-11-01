from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_e3_cell01_feedback_prepare_20261029_v1 as m


def sample():
    base=dict(schema='fixed',cell=0,resolver={'path':'resolver','sha256':'x'},
              control5={'prepare':{'path':'prepare','sha256':'y'}},additional_sources={'/a':'a','/b':'b'})
    fresh=dict(schema='fixed',cell=1,resolver=deepcopy(base['resolver']),control5=m.control5(base),additional_sources={'/a':'a'})
    return base,fresh


def test_exact_change_preserves_all_prior_sources():
    base,fresh=sample();value=m.combine(base,fresh)
    assert value['cell']==1 and value['additional_sources']==base['additional_sources']
    assert base['cell']==0 and value['control5']==fresh['control5']


@pytest.mark.parametrize('kind',['cell','resolver','prepare','receipt','source','new_source'])
def test_mismatches_fail_closed(kind):
    base,fresh=sample()
    if kind=='cell':fresh['cell']=2
    elif kind=='resolver':fresh['resolver']['sha256']='bad'
    elif kind=='prepare':fresh['control5']['prepare']['sha256']='bad'
    elif kind=='receipt':fresh['control5']['receipt']['sha256']='0'*64
    elif kind=='source':fresh['additional_sources']['/a']='changed'
    else:fresh['additional_sources']['/unknown']='hash'
    with pytest.raises(Exception):m.combine(base,fresh)
