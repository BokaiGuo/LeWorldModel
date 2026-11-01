from copy import deepcopy
import struct

import pytest

from experiments.spectral_reachability import agent_p3_settling_cp15_guard_20261029_v1 as m


def specimen(index=0):
    _,root,role,probe=m.old.ORDER[index]
    job=dict(index=index,root_slot=root,source_role=role,probe=probe,selection_digest='d'*64,
        warm10=[[.25,0.] for _ in range(10)],raw15=[[.5,0.] for _ in range(15)],
        unknown_semantics=dict(rng='unknown',contact_cache='unknown',public_state='unknown'))
    state=[100.,101.,102.,103.,.25,1.,2.]
    snap=dict(state=state,state_float64_bits=struct.pack('<7d',*state).hex(),velocity=[1.,2.])
    observations=[]
    for phase in ('warmup','plan'):
        for step in range(11 if phase=='warmup' else 16):
            for overlay in range(3):
                receipt=dict(phase=phase,step=step,overlay_index=overlay,png=f'{phase}{step}{overlay}.png',
                    state=deepcopy(state),snapshot=deepcopy(snap),file_sha256='a'*64,pixel_sha256='b'*64)
                observations.append(dict(index=len(observations),previous_sha256='a'*64,event_sha256='b'*64,
                    kind='fresh_observation',branch=index,root_slot=root,arm=probe,condition='off',
                    selection_digest=job['selection_digest'],receipt=receipt,
                    pre=dict(snapshot=deepcopy(snap),goal_state=[0,0,0,0,0,60,-180]),
                    post=dict(snapshot=deepcopy(snap),goal_state=[0,0,0,0,0,60,-180])))
    operations=[]
    for n,raw in enumerate(job['warm10']+job['raw15']):
        bit=''.join(m.old.bits([raw],1)); operations.append(dict(index=n,resource='public_step',status='returned',
            raw_bits=bit,result=dict(actual_bits=bit,actual_dtype='<f4',terminated=False,truncated=False,
                                     state=deepcopy(state),snapshot=deepcopy(snap))))
    operations.extend(dict(index=n,resource='physics_step',status='returned') for n in range(25,373))
    row=dict(job=deepcopy(job),status='horizon_completed',guards=[51,66],observations=deepcopy(observations),
             operations=deepcopy(operations)+[dict(index=373,resource='close',status='returned')],
             endpoint15=deepcopy(observations[-3]['receipt']))
    return job,dict(job=deepcopy(job),old_row=row,old_seal={}),observations,operations


@pytest.mark.parametrize('index',range(10))
def test_every_fixed_branch_guard_preserves_complete_prefix(index):
    job,source,obs,ops=specimen(index)
    proof=m.guard81(job,source,obs,ops)
    assert proof['source_verified_here'] is False
    assert len(proof['transport_mappings']) == 81
    assert proof['raw_history_float32_bits'] == ''.join(m.old.bits(job['raw15'][-10:],10))
    assert proof['unknown_semantics']['contact_cache']=='unknown'


@pytest.mark.parametrize('mutation',('state','snapshot','pixel','goal','history','receipt','sourcejob','missing'))
def test_all_authentication_dimensions_fail_closed(mutation):
    job,source,obs,ops=specimen()
    if mutation=='state': obs[-1]['receipt']['state'][0]=0.
    if mutation=='snapshot': obs[-2]['receipt']['snapshot']['velocity'][0]=0.
    if mutation=='pixel': obs[-3]['receipt']['pixel_sha256']='c'*64
    if mutation=='goal': obs[-3]['pre']['goal_state'][-1]=0.
    if mutation=='history': ops[10]['raw_bits']='00'*8
    if mutation=='receipt': ops[20]['result']['terminated']=True
    if mutation=='sourcejob': job['raw15'][0]=[0.,0.]
    if mutation=='missing': obs.pop()
    with pytest.raises(ValueError):m.guard81(job,source,obs,ops)


def test_failed_fsync_never_dispatches_step16():
    job,source,obs,ops=specimen(); calls=[]
    def failed(event):raise OSError('fsync failure')
    with pytest.raises(OSError): m.before_step16(job,source,obs,ops,failed,lambda raw:calls.append(raw))
    assert calls==[]


def test_guard_success_precedes_zero_dispatch():
    job,source,obs,ops=specimen(); events=[]
    m.before_step16(job,source,obs,ops,lambda e:events.append(e['kind']),lambda a:events.append(a))
    assert events==['cp15_guard',[0.0,0.0]]


def test_budget_formula_and_unproven_metadata_is_not_ready():
    c=m.budget()
    assert c['operations']==444 and c['events']==990
    assert c['total']['public_step']==300 and c['total']['physics_step']==3040
    assert c['total']['explicit_render']==960 and c['conservative_bytes']<c['cap_bytes']
    assert not c['metadata_structural_bound_proven'] and not c['runtime_ready']


def test_reference_requires_exact_whole_authority():
    with pytest.raises((KeyError,ValueError)):m.whole_reference({}, {}, {}, {}, fixture=False)
