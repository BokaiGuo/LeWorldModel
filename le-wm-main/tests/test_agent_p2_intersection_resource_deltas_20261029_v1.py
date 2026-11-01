from copy import deepcopy
import json
import pytest
from experiments.spectral_reachability import agent_p2_intersection_resource_deltas_20261029_v1 as m


def panel():
    rows=[]
    for i,(root,arm) in enumerate(m.prior.ORDER):
        axes={k:(None if k=='FLOPs' else i) for k in m.box.AXES}
        side={'Adam_step':i,'engine_backward_calls':None,'CPU_capture_seconds':float(i)}
        rows.append(dict(slot=i,root=root,arm=arm,task_id=str(root),goal_sha256=str(root),
                         resource_axes27=axes,side_axes=side,elapsed_seconds=float(i),
                         complete_account=dict(used=deepcopy(axes),side=deepcopy(side))))
    return dict(schema=m.prior.NAME+'.analysis',fixed_denominator=6,total_FLOPs=None,equal_FLOPs=False,rows=rows)


def test_pairs_fixed_signed_axes_and_unknown():
    result=m.analyze(panel());assert result['fixed_same_root_pairs']==6
    assert [(r['left_slot'],r['right_slot']) for r in result['pairs']]==[(0,1),(0,2),(1,2),(5,4),(5,3),(4,3)]
    assert result['pairs'][0]['axes_left_minus_right']['predict_API']==-1
    assert all(p['axes_left_minus_right']['FLOPs'] is None and p['side_left_minus_right']['engine_backward_calls'] is None for p in result['pairs'])
    assert result['equal_FLOPs'] is None


@pytest.mark.parametrize('mutation', ['missing','order','axes','account','goal','flops','unknown_type'])
def test_fail_closed(mutation):
    p=panel()
    if mutation=='missing':p['rows'].pop()
    elif mutation=='order':p['rows'][0]['arm']='grasp'
    elif mutation=='axes':p['rows'][0]['resource_axes27'].pop('J')
    elif mutation=='account':p['rows'][0]['complete_account']['used']['J']=99
    elif mutation=='goal':p['rows'][1]['goal_sha256']='different'
    elif mutation=='flops':p['rows'][0]['resource_axes27']['FLOPs']=0
    else:p['rows'][0]['side_axes']['engine_backward_calls']=False;p['rows'][0]['complete_account']['side']['engine_backward_calls']=False
    with pytest.raises(ValueError):m.analyze(p)


def test_tempdisk_exclusive(tmp_path):
    path=tmp_path/(m.NAME+'_fixture.json');r=m.write(path,m.analyze(panel()))
    assert m.prior.sha(path)==r['sha256'] and json.loads(path.read_bytes())['fixed_slots']==6
    with pytest.raises(ValueError):m.write(path,{})


def test_existing_authorities_saved_only():
    result=m.build();assert len(result['pairs'])==6
    assert all(p['axes_left_minus_right']['FLOPs'] is None for p in result['pairs'])
