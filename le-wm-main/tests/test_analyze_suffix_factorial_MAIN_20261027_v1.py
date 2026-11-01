"""RAM fixtures only, not neural/control evidence."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import analyze_suffix_factorial_MAIN_20261027_v1 as m


def fixture():
    original=[];rows=[]
    for anchor in range(8):
        identity=dict(root_slot=anchor//4,condition='off' if anchor%4<2 else 'on',
                      arm='GN' if anchor%2==0 else 'D2P',selection_digest=str(anchor),task_id=str(anchor))
        goal=[0.]*7;start=[30.,0.,0.,0.,0.,0.,0.];final=[25.,0.,0.,0.,0.,0.,0.]
        original.append(dict(**identity,goal=goal,final_state=final,checkpoints=[dict(step=5,state=start)]))
        for v in m.physical.VARIANTS:
            rows.append(dict(**identity,index=len(rows),variant=v,status='suffix_completed',goal=goal,
                start_verified=True,start_state=start,final_state=final,native_success=False,
                responses=[dict(state=final,terminated=False)]))
    return rows,original


def test_all24_zero_success_is_retained_not_failure_of_analysis():
    rows,original=fixture();r=m.analyze_rows(rows,original)
    assert len(r['rows'])==24 and len(r['paired_comparisons'])==16
    assert all(s['native_success']==0 and s['denominator']==8 and s['observed']==8 for s in r['summaries'].values())


def test_failed_row_stays_unknown_in_fixed_denominator():
    rows,original=fixture();rows[0].update(status='engineering_failed',final_state=None,responses=[])
    r=m.analyze_rows(rows,original)
    assert r['rows'][0]['final'] is None
    assert r['summaries']['open_full']['unknown']==1
    assert r['summaries']['open_full']['denominator']==8
    assert not r['paired_comparisons'][0]['available']


@pytest.mark.parametrize('mutation',['prefix','goal','flag','identity','order'])
def test_wrong_bridge_is_rejected(mutation):
    rows,original=deepcopy(fixture())
    if mutation=='prefix':rows[0]['start_state']=[31.,0.,0.,0.,0.,0.,0.]
    elif mutation=='goal':rows[0]['goal']=[1.]*7
    elif mutation=='flag':rows[0]['native_success']=True
    elif mutation=='identity':rows[0]['task_id']='other'
    else:rows[0],rows[1]=rows[1],rows[0]
    with pytest.raises(ValueError):m.analyze_rows(rows,original)


def test_installed_joint_success_includes_agent_and_preserves_angle_diagnostic():
    g=np.zeros(7);s=np.zeros(7);s[0]=30
    assert not m.native.metrics(g,s)['native_success']
    s[0]=0;s[4]=4*np.pi
    a=m.native.metrics(g,s)
    assert a['negative_native_angle'] and a['native_success']
