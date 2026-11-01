from copy import deepcopy
import math
import pytest
from experiments.spectral_reachability import agent_p2_intersection_v2_saved_analysis_20261029_v1 as m


def inputs():
    a=m.refs();return a,[m.read(a[k]) for k in ('receipt','report','prepare')]


def test_actual_six_slots_and_independent_fsum():
    a,v=inputs();out=m.analyze(*v,a)
    assert [(r['root'],r['arm']) for r in out['rows']]==m.ORDER
    for i,r in enumerate(out['rows']):
        z=v[0]['plans'][i]['contexts'][-1]['latent_history'][-1];g=v[2]['slots'][r['root']]['packet']['goal']
        assert r['final_latent_goal_mse64']==math.fsum((float(x)-float(y))**2 for x,y in zip(z,g,strict=True))/192
        assert len(r['resource_axes27'])==27 and r['resource_axes27']['FLOPs'] is None
    assert out['winner'] is None and not out['physical_benefit']


@pytest.mark.parametrize('mutation',['report_sha','goal_shape','order','missing','axes','context'])
def test_fail_closed(mutation):
    a,v=inputs();receipt,report,prepare=deepcopy(v)
    if mutation=='report_sha':receipt['report_sha256']='0'*64
    elif mutation=='goal_shape':prepare['slots'][0]['packet']['goal'].pop()
    elif mutation=='order':receipt['accounts'][0]['arm']='grasp'
    elif mutation=='missing':receipt['plans'][0]=None
    elif mutation=='axes':receipt['accounts'][0]['used']['FLOPs']=1
    else:receipt['plans'][0]['contexts'].pop()
    with pytest.raises(ValueError):m.analyze(receipt,report,prepare,a)


def test_exclusive_output(tmp_path):
    p=tmp_path/(m.NAME+'_fixture.json');m.write(p,{'x':1})
    with pytest.raises(ValueError):m.write(p,{'x':2})
