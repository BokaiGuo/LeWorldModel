from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_p2_six_target_resource_panel_20261029_v1 as m


def inputs():
    a=m.refs();return a,[m.read(a[k]) for k in ('comparison','receipt','report')]


def test_fixed_actual_saved_abi_and_no_method_selection():
    a,v=inputs();out=m.assemble(*v,a)
    assert [x['slot'] for x in out['slots']]==list(range(6))
    assert all([c['cp_public_step'] for c in x['checkpoints']]==[0,5,10,15] for x in out['slots'])
    assert len(out['accounting27']['actual'])==27 and out['accounting27']['actual']['FLOPs'] is None
    assert out['winner'] is None and not out['physical_benefit_claim']


@pytest.mark.parametrize('tamper',['order','receipt','checkpoint','axes','missing'])
def test_fail_closed_mutations(tamper):
    a,v=inputs();v=deepcopy(v);c,r,p=v
    if tamper=='order':p['rows'][0],p['rows'][1]=p['rows'][1],p['rows'][0]
    elif tamper=='receipt':r['side_axes']['physics']=1
    elif tamper=='checkpoint':c['rows'][0]['cp_public_step']=5
    elif tamper=='axes':p['accounting27']['actual']['predict_API']+=1
    else:c['rows'].pop()
    with pytest.raises(ValueError):m.assemble(c,r,p,a)


def test_exclusive_writer(tmp_path):
    p=tmp_path/(m.NAME+'_fixture.json');m.write(p,{'x':1})
    with pytest.raises(ValueError):m.write(p,{'x':2})
