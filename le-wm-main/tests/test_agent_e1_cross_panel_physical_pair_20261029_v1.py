from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_e1_cross_panel_physical_pair_20261029_v1 as m


def panels():
    def row(root,arm):
        state=[float(root),0.,0.,0.,0.,0.,0.];goal=[0.]*7
        cp=dict(step=0,state=state,snapshot={'state':state},pixel_sha256='pixel',file_sha256='file')
        plan=dict(root_sha256=f'r{root}',task_id=f't{root}',task_sha256='t',task_contract_sha256='c',blind_sha256='b',commands=[[0.,0.]]*10)
        return dict(root_slot=root,arm=arm,selected_plan=plan,checkpoints=[cp],goal=goal,status='native_stopped',own_returned=0,terminated=True,truncated=False)
    return ([row(r,a) for r,a in m.six.SLOTS],[row(r,a) for r,a in m.forced.SLOTS])


def test_fixed_denominator_and_missing_not_filled():
    a,b=panels();out=m.compare(a,b)
    assert out['denominator']==54 and out['observed']==18
    assert all(p['delta'] is None for p in out['pairs'] if p['step']>0)
    assert all(p['delta']['all_xy']==0 for p in out['pairs'] if p['step']==0)


def test_missing_root_blocks_entire_root():
    a,b=panels();b[0]['checkpoints']=[]
    out=m.compare(a,b)
    assert out['joins'][0]['initial_exact'] is False
    assert all(p['delta'] is None for p in out['pairs'] if p['root']==0)


@pytest.mark.parametrize('field',['root_sha256','task_id','task_sha256','task_contract_sha256','blind_sha256'])
def test_provenance_drift_rejected(field):
    a,b=panels();b[0]['selected_plan'][field]='changed'
    with pytest.raises(ValueError):m.compare(a,b)


@pytest.mark.parametrize('field',['state','snapshot','pixel_sha256','file_sha256'])
def test_initial_drift_rejected(field):
    a,b=panels();b[0]['checkpoints'][0][field]='changed'
    with pytest.raises(ValueError):m.compare(a,b)


def test_goal_drift_and_action_identity():
    a,b=panels();b[0]['goal']=[1.]*7
    with pytest.raises(ValueError):m.compare(a,b)
    a,b=panels();b[0]['selected_plan']['commands']=deepcopy(b[0]['selected_plan']['commands']);b[0]['selected_plan']['commands'][6]=[1.,0.]
    out=m.compare(a,b);p=out['pairs'][0]
    assert p['same_first5'] and not p['same_raw10'] and p['forced_branch']['terminated']
