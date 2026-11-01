"""RAM-only signed response arithmetic. No model/physics or disk fixtures."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import agent_mixedmode_capability_analysis_20261028_v1 as m
from test_agent_mixedmode_capability_probe_20261028_v2 import fixture as base_fixture


def fixture():
    p=base_fixture();rows=[]
    for t in p['tasks']:
        t['native']['goal']=[1.,0.,1.,0.,.5,0.,0.]
    for j in p['jobs']:
        task=p['tasks'][j['task']];native=task['native'];after=j['role']=='after'
        start=np.zeros(7);start[0]=.2 if after else 0.
        zero=start.copy();zero[1]=.3 if after else .1
        d=np.asarray(j['probe_commands'])[0]
        delta=np.zeros(7);delta[:2]=d*(2 if after else 1);delta[2:4]=d*(3 if after else 1)
        delta[4]=d[0]*.5
        end=zero+delta
        rows.append(dict(index=j['index'],task=j['task'],role=j['role'],probe_id=j['probe_id'],
            condition=task['plan']['condition'],job_digest=j['job_digest'],context_digest=j['context_digest'],
            status='completed',attempted=5,completed=5,error=None,start_verified=True,
            start_state=start.tolist(),end_state=end.tolist(),goal=native['goal'],root_state=native['root_state'],
            last_return=dict(state=end.tolist(),reward=-3.,terminated=False,truncated=False),
            frames=[dict(pre={'snapshot':{'state':start.tolist(),'bodies':{'block':{'velocity':[0.,0.]}}}})]))
    consumed=dict(status='saved_probe_responses_not_admitted',rows=rows,P1_admission=False,
        new_neural_calls=0,new_physics_calls=0,counts={'public_step':dict(attempted=700,completed=700)},errors=[])
    return p,consumed


def test_all40_and_fixed_spaces_and_original_directions():
    p,c=fixture();r=m.analyze(p,c)
    assert r['denominators']['probes']==40 and len(r['rows'])==8 and len(r['comparisons'])==4
    assert all(list(g['spaces'])==list(m.SPACES) for g in r['rows'])
    assert r['rows'][0]['spaces']['joint_xy']['original_task_direction']==r['rows'][1]['spaces']['joint_xy']['original_task_direction']
    assert not r['P1_admission'] and not r['independent_confirmation']


def test_signed_object_response_after_minus_before():
    p,c=fixture();r=m.analyze(p,c);a=float(np.float32(.1))
    assert r['rows'][0]['spaces']['object_xy']['finite_responses'][1]['signed_original_task_projection']==pytest.approx(a)
    assert r['rows'][1]['spaces']['object_xy']['finite_responses'][1]['signed_original_task_projection']==pytest.approx(3*a)
    diff=r['comparisons'][0]['spaces']['object_xy']['signed_response_changes']
    assert diff[1]['after_minus_before']==pytest.approx(2*a)
    assert diff[2]['after_minus_before']==pytest.approx(-2*a)


def test_projection_sign_not_unsigned_gain():
    p,c=fixture()
    for r in c['rows']:
        if r['role']=='after':
            z=next(q for q in c['rows'] if q['task']==r['task'] and q['role']=='after' and q['probe_id']=='zero')
            r['end_state'][2]=2*z['end_state'][2]-r['end_state'][2]
    r=m.analyze(p,c)
    assert r['comparisons'][0]['spaces']['object_xy']['signed_secant_changes'][0]==pytest.approx(-4.)
    assert r['rows'][1]['spaces']['object_xy']['unsigned_spectrum'] is None


def test_zero_probe_and_drift_separated_from_effect():
    p,c=fixture();r=m.analyze(p,c)
    assert r['rows'][0]['spaces']['agent_xy']['finite_responses'][0]['signed_original_task_projection']==0.
    assert r['rows'][1]['spaces']['agent_xy']['zero_drift']==[0.,.3]
    assert r['comparisons'][0]['time_confound']['controlled'] is False
    assert r['comparisons'][0]['causal_capability_gain']=='unknown'


def test_direction_never_uses_after_residual():
    p,c=fixture()
    for row in c['rows']:
        if row['role']=='after':
            row['start_state'][0]=2.;row['frames'][0]['pre']['snapshot']['state'][0]=2.
    r=m.analyze(p,c)
    assert r['rows'][1]['spaces']['agent_xy']['original_task_direction']==[1.,0.]


def test_object_only_not_full_task_failure():
    p,c=fixture()
    for row in c['rows']:row['end_state'][2:4]=[0.,0.]
    r=m.analyze(p,c)
    assert r['rows'][0]['spaces']['object_xy']['signed_original_task_secants']==[0.,0.]
    assert r['rows'][0]['spaces']['joint_xy']['signed_original_task_secants'][0]>0
    assert not r['rows'][0]['spaces']['object_xy']['full_task_success_inferred']


def test_missing_zero_remains40_null_signed_finite_not_fill():
    p,c=fixture();c['rows'][0].update(status='failed',end_state=None,error={'message':'failure'})
    r=m.analyze(p,c)
    assert r['denominators']['probes']==40
    assert r['rows'][0]['missing']==['zero']
    assert all(x['signed_original_task_projection'] is None for x in r['rows'][0]['spaces']['joint_xy']['finite_responses'])
    assert r['comparisons'][0]['spaces']['joint_xy']['signed_response_changes'][1]['after_minus_before'] is None


def test_zero_original_direction_undefined_not_new_direction():
    p,c=fixture()
    for t in p['tasks']:t['native']['goal']=[0.]*7
    for r in c['rows']:r['goal']=[0.]*7
    result=m.analyze(p,c)
    for space in m.SPACES:
        assert result['rows'][0]['spaces'][space]['original_task_direction'] is None
        assert all(x['signed_original_task_projection'] is None for x in result['rows'][0]['spaces'][space]['finite_responses'])


@pytest.mark.parametrize('shift',[-4*np.pi,-2*np.pi,2*np.pi,4*np.pi])
def test_wrapped_angle_direction_2pi_invariant(shift):
    p,c=fixture();first=m.analyze(p,c)
    for t in p['tasks']:t['native']['goal'][4]+=(shift)
    for r in c['rows']:r['goal']=p['tasks'][r['task']]['native']['goal']
    out=m.analyze(p,c)
    assert out['rows'][0]['spaces']['object_angle_wrapped']['original_task_direction']==first['rows'][0]['spaces']['object_angle_wrapped']['original_task_direction']


@pytest.mark.parametrize('mutation',['drop','swap','role','context','goal','start','probe'])
def test_no_cohort_or_identity_substitution(mutation):
    p,c=fixture()
    if mutation=='drop':c['rows'].pop()
    elif mutation=='swap':c['rows'][0],c['rows'][1]=c['rows'][1],c['rows'][0]
    elif mutation=='role':c['rows'][0]['role']='after'
    elif mutation=='context':c['rows'][0]['context_digest']='bad'
    elif mutation=='goal':c['rows'][0]['goal']=[99.]*7
    elif mutation=='start':c['rows'][0]['start_state'][0]+=.1
    elif mutation=='probe':p['jobs'][1]['probe_commands']=[[0.,0.]]*5
    with pytest.raises(ValueError):m.analyze(p,c)


def test_production_assemble_calls_full_physical_consumer(monkeypatch):
    p,c=fixture();called=[]
    monkeypatch.setattr(m,'sha',lambda _:m.PHYSICAL_SOURCE_SHA)
    def consume(*a):called.append(a);raise ValueError('full_consume_rejected')
    monkeypatch.setattr(m.physical,'consume',consume)
    with pytest.raises(ValueError,match='full_consume_rejected'):m.assemble('/RAM/actual',m.REPORT_SHA)
    assert len(called)==1


def test_wrong_external_report_sha_no_consume(monkeypatch):
    monkeypatch.setattr(m,'sha',lambda _:m.PHYSICAL_SOURCE_SHA)
    monkeypatch.setattr(m.physical,'consume',lambda *a:pytest.fail('must not consume another report'))
    with pytest.raises(ValueError):m.assemble('/RAM/actual','0'*64)


def test_cli_help_only(capsys):
    with pytest.raises(SystemExit) as e:m.main(['--help'])
    assert e.value.code==0 and '--physical-report-sha' in capsys.readouterr().out
