from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_p3_dynamic_cp10_physical_inputs_20261029_v1 as m
from experiments.spectral_reachability import agent_p3_same_time_cp10_rgb_bridge_20261029_v1 as old


@pytest.fixture(scope='module')
def actual():
    d=old.descriptors()
    groups={prefix:{k:d[prefix+'_'+k] for k in ('prepare','report','completion','receipt','source')} for prefix in ('C','ZP')}
    rows=old.controls.read(str(old.ROOT/'results'/('agent_p3_same_time_cp10_rgb_bridge_20261029_v1_prepare_MAIN01.json')),old.META)['rows']
    return rows,groups


def test_historical_saved_actual_ABI(actual):
    rows,groups=actual;out=m.assemble(rows,groups,'historical')
    assert out['slots']==5 and not out['fresh_roots_certified'] and not out['execution_authorized']


def test_historical_actual_cannot_be_relabeled_future(actual):
    rows,groups=actual
    with pytest.raises(ValueError,match='historical root'):m.assemble(rows,groups,'future_candidate')


@pytest.mark.parametrize('kind',['order','missing','raw','source','history','root','task'])
def test_source_input_tamper_fails(actual,kind):
    rows,groups=deepcopy(actual)
    if kind=='order':rows.reverse()
    elif kind=='missing':rows.pop()
    elif kind=='raw':rows[0]['causal_raw10'][0][0]+=1
    elif kind=='source':rows[0]['source_receipt']['sha256']='0'*64
    elif kind=='history':rows[0]['history_sources'][0]['state'][0]+=1
    elif kind=='root':rows[0]['root_slot']=1
    elif kind=='task':rows[0]['cp5_task']['goal'][0]+=1
    with pytest.raises((ValueError,KeyError)):m.assemble(rows,groups,'historical')


def test_cannot_replace_source_hash(actual):
    rows,groups=deepcopy(actual);groups['C']['source']['sha256']='0'*64
    with pytest.raises(ValueError):m.assemble(rows,groups,'historical')
