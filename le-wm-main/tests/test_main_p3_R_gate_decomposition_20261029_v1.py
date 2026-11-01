from copy import deepcopy
import pytest
from experiments.spectral_reachability import main_p3_R_gate_decomposition_20261029_v1 as m


def row():
    return dict(postcontext=dict(latent_history=[[1.]*192]), signal=dict(nominal=[2.]*192, jacobian=[[0.]*10 for _ in range(192)]),
                solve=dict(params=[0.]*10, linear_terminal_cost=4., ridge_penalty=0., own_reach_value=4.), immediate=1., reach=4.)


def fixture():
    p=dict(tasks=[dict(goal=[0.]*192, projection=[[float(i == j) for j in range(192)] for i in range(192)])]*2)
    r=dict(status='four_groups_complete_not_admitted', errors=[], rows=[])
    for root,h in ((0,1),(0,3),(1,1),(1,3)):
        zero=row(); cs=[]
        for i in range(8):
            c=row(); c.update(index=i,eligible=False); cs.append(c)
        r['rows'].append(dict(root=root,horizon=h,result=dict(R_groups=[dict(control=zero,rows=cs,eligible_indices=[])])))
    return p,r,dict(status='all4_native_R_groups_verified_not_admitted',denominator=4)


def test_independent_components():
    s=m.scalar(row(),[0.]*192)
    assert s['nominal_drift']==3 and s['bounded_linear_gain']==0
    assert s['reach']==s['immediate']+s['nominal_drift']-s['bounded_linear_gain']


def test_all_four_and_exact_ties_not_eligible():
    out=m.explain(*fixture())
    assert out['candidates_count']==32
    assert all(x['counts']==dict(nongreedy=0,better_reach=0,eligible=0) for x in out['groups'])


@pytest.mark.parametrize('key',['immediate','reach','jacobian','penalty','nonfinite','bool'])
def test_scalar_mutations(key):
    x=row()
    if key=='immediate': x['immediate']=2.
    elif key=='reach': x['reach']=3.
    elif key=='jacobian': x['signal']['jacobian'].pop()
    elif key=='penalty': x['solve']['ridge_penalty']=.1
    elif key=='nonfinite': x['signal']['nominal'][0]=float('nan')
    else: x['solve']['params'][0]=False
    with pytest.raises(ValueError): m.scalar(x,[0.]*192)


@pytest.mark.parametrize('key',['order','candidate','gate','eligible','projection','receipt'])
def test_full_denominator_and_authority(key):
    p,r,c=deepcopy(fixture())
    if key=='order': r['rows'].reverse()
    elif key=='candidate':r['rows'][0]['result']['R_groups'][0]['rows'].pop()
    elif key=='gate':r['rows'][0]['result']['R_groups'][0]['rows'][0]['eligible']=True
    elif key=='eligible':r['rows'][0]['result']['R_groups'][0]['eligible_indices']=[0]
    elif key=='projection':p['tasks'][0]['projection'][0][0]=2.
    else:c['denominator']=3
    with pytest.raises(ValueError):m.explain(p,r,c)
