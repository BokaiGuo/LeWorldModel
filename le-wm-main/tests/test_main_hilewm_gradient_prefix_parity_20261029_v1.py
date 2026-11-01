from copy import deepcopy
import pytest
from experiments.spectral_reachability import main_hilewm_gradient_prefix_parity_20261029_v1 as m
from test_agent_hilewm_transition_cuda_probe_20261029_v2 import fixture


def case():
    _,r,arrays=fixture();b=r['rows'][0];a=deepcopy(b)
    a['events'][9]['returned']=False;a['events'][0]['output_layout']['requires_grad']=False
    return a,b,deepcopy(arrays),arrays


def test_partial_only_bitwise_forward_and_gradient_return():
    a,b,aa,bb=case();v=m.compare_prefix(a,b,aa.__getitem__,bb.__getitem__)
    assert v['forward_bitwise'] and v['forward_arrays_compared']==32
    assert not v['gradient_parity'] and not v['finite_difference_verified']
    assert not v['whole_segmented_verified']


@pytest.mark.parametrize('fault',('actions','prediction','scalar','context','input','output','chronology','oldgrad','newgrad','graph'))
def test_saved_prefix_mutations_refused(fault):
    a,b,aa,bb=case()
    if fault in ('actions','prediction','scalar'):aa[a[fault]]=aa[a[fault]]+1
    elif fault=='context':aa[a['contexts'][1]['z']].flat[0]+=1
    elif fault=='input':aa[a['events'][1]['inputs'][0]].flat[0]+=1
    elif fault=='output':aa[a['events'][1]['output']].flat[0]+=1
    elif fault=='chronology':a['events'][2]['kind']='unpaid'
    elif fault=='oldgrad':a['events'][9]['returned']=True
    elif fault=='newgrad':b['events'][9]['returned']=False
    else:b['events'][0]['output_layout']['requires_grad']=False
    with pytest.raises(ValueError):m.compare_prefix(a,b,aa.__getitem__,bb.__getitem__)
