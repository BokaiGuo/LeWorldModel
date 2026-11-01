from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability import main_anchor_guard_physical_factorial_20261028_v1 as m


def fixture():
    plans, reports = {}, {}
    for name, action, error in [('control', .1, 9.), ('zero', .2, 7.), ('incumbent', .3, 5.)]:
        plans[name], reports[name] = [], []
        for i in range(8):
            commands = np.full((10, 2), action, np.float32); commands[:5] = 0
            p = dict(commands=commands.tolist(), root_slot=i//4, condition='off' if i%4<2 else 'on',
                     arm=str(i%2), task_id=str(i), selection_digest=name+str(i))
            r = dict(index=i, **{k:v for k,v in p.items() if k!='commands'}, own_returned=10,
                     status='horizon_completed', final_state=[error,0.,0.,0.,0.,0.,0.],
                     goal=[0.]*7, root_state=[0.]*7, native_success=True,
                     checkpoints=[dict(step=5, state=[0.]*7, snapshot='exact', png=str(i))])
            plans[name].append(p); reports[name].append(r)
    cells = []
    for i in range(8):
        for anchor in ('zero', 'incumbent'):
            for guard in (False, True):
                source = 'control' if (anchor=='zero' and guard and i==4) else anchor
                p = plans[source][i]; raw = m.bits(p['commands'])
                cells.append(dict(branch=i, anchor=anchor, guard=guard, full10=p['commands'],
                    full10_raw_bits=raw, prefix_raw_bits=raw[:80],
                    **{k:p[k] for k in ('root_slot', 'condition', 'arm', 'task_id')},
                    selection=dict(model_Q=1.)))
    return cells, reports, plans


def test_full_fixed_factorial_and_guard_interaction():
    r = m.assemble(*fixture())
    assert len(r['cells']) == 32
    assert r['counts']['anchor_no_guard']['joint_xy'] == dict(better=8,worse=0,tie=0)
    assert r['counts']['guard_zero']['joint_xy'] == dict(better=0,worse=1,tie=7)
    assert r['contrasts'][4]['interaction']['joint_xy'] == -2.
    assert r['cells'][17]['source'] == 'control'
    assert r['new_neural_calls'] == r['new_physics'] == 0
    assert not r['P1_admission']


@pytest.mark.parametrize('bad', ['task','root','goal','action','mid','early','order','missing','success','digest','prefix'])
def test_wrong_provenance_never_reused(bad):
    cells, reports, plans = deepcopy(fixture())
    r = reports['zero'][0]
    if bad=='task': r['task_id']='other'
    if bad=='root': r['root_state'][0]=1
    if bad=='goal': r['goal'][0]=1
    if bad=='action': plans['zero'][0]['commands'][9][0]=.25
    if bad=='mid': r['checkpoints'][0]['snapshot']='different'
    if bad=='early': r['own_returned']=9
    if bad=='order': cells[0],cells[1]=cells[1],cells[0]
    if bad=='missing': cells.pop()
    if bad=='success': r['native_success']=False
    if bad=='digest': r['selection_digest']='different'
    if bad=='prefix': cells[0]['prefix_raw_bits']='wrong'
    with pytest.raises(ValueError): m.assemble(cells,reports,plans)


def test_no_float32_rounding_repair():
    a=np.zeros((10,2)); a[9,0]=.123456789
    with pytest.raises(ValueError):m.bits(a)


def test_all_error_axes_have_same_benefit_sign():
    r=m.assemble(*fixture())
    assert all(v<=0 for v in r['contrasts'][0]['anchor_no_guard'].values())
