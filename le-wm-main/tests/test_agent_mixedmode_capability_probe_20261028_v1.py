"""Synthetic/RAM metadata only. No model, simulator, real prepare or results."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import pytest
from experiments.spectral_reachability import agent_mixedmode_capability_probe_20261028_v1 as m


def fixture():
    slots, plans = [], []
    for i, (s, c) in enumerate(m.ORDER):
        slot = dict(index=i, root_slot=s, condition=c, task_id=f't{i}', task_sha256='a'*64,
                    blind_sha256='b'*64, packet=dict(root_context={'history': [1,2,3]}, goal=[4]), q=[[1]])
        commands = np.zeros((10,2), np.float32).tolist()
        plan = dict(root_slot=s, condition=c, task_id=f't{i}', task_sha256='a'*64,
            blind_sha256='b'*64, commands=commands, prefix_commands=commands[:5],
            suffix_commands=commands[5:], selection_digest='c'*64, horizon=2, selected_index=i)
        slots.append(slot); plans.append(plan)
    meta = [dict(root_state=[0.]*7, goal=[1.]*7, root_sha256='d'*64,
                 goal_pose=[1.,2.,3.], renderer={'size':224}, goal_pixels={'off':'e'*64,'on':'f'*64}) for _ in range(2)]
    p = m.assemble(dict(schema='agent_mixedmode_prefix_20261028_v1.prepare', slots=slots, bindings={}),
        dict(status='all4_mixedmode_commands_verified_not_admitted', errors=[], plans=plans), meta)
    return p


def rows(p):
    out = []
    for j in p['jobs']:
        x = [0.]*7
        d = np.asarray(j['probe_commands'])[0]
        x[:2] = (d*(1 if j['role']=='before' else 2)).tolist()
        out.append(dict(job_digest=j['job_digest'], status='completed', end_state=x))
    return out


def test_fixed_denominator_and_literal_actions():
    p = fixture()
    assert len(p['jobs']) == 40
    assert [(t['plan']['root_slot'],t['plan']['condition']) for t in p['tasks']] == m.ORDER
    for j in p['jobs']:
        a=m.raw(j['probe_commands'],5)
        assert a.dtype == np.float32 and len(j['prefix_commands']) == (0 if j['role']=='before' else 5)
        assert j['job_digest'] == m.digest({k:v for k,v in j.items() if k!='job_digest'})
    assert m.contract()['public_steps_cap']==400+100+200
    assert not p['contract']['execution_ready']


def test_complete_context_changes_seal():
    p = fixture(); slot=deepcopy(p['tasks'][0]['slot'])
    slot['packet']['root_context']['history'][0]=99
    assert m.digest(slot)!=p['tasks'][0]['context_digest']


@pytest.mark.parametrize('n,bad', [(5,[[0,0]]), (1,[[float('nan'),0]]), (1,[[.1,0]])])
def test_raw_rejects_shape_nonfinite_and_nonliteral(n,bad):
    with pytest.raises(ValueError): m.raw(bad,n)


def test_empty_prefix_valid_and_negative_zero_distinct():
    assert m.raw([],0).shape==(0,2)
    assert m.raw([[0.,0.]],1).tobytes()!=m.raw([[-0.,0.]],1).tobytes()


def test_true_response_contrast_not_endpoint_improvement():
    p=fixture(); r=m.summarize(p,rows(p))
    assert r['contrasts'][1]['after_minus_before'][0] == pytest.approx(float(np.float32(.1)))
    assert r['endpoint_improvement_is_capability'] is False and not r['P1_admission']


def test_zero_failure_masks_role_not_other_role():
    p=fixture(); rr=rows(p); rr[0].update(status='failed',end_state=None)
    out=m.summarize(p,rr)
    assert all(x['before'] is None for x in out['contrasts'][:5])
    assert out['contrasts'][1]['after'] is not None and out['contrasts'][1]['after_minus_before'] is None


def test_angle_wrapped_not_2pi_capability():
    p=fixture(); rr=rows(p); rr[1]['end_state'][4]=2*np.pi
    assert abs(m.summarize(p,rr)['contrasts'][1]['before'][4])<1e-12


@pytest.mark.parametrize('change', ['drop','swap'])
def test_no_denominator_drop_or_reorder(change):
    p=fixture(); rr=rows(p)
    if change=='drop': rr.pop()
    else: rr[0],rr[1]=rr[1],rr[0]
    with pytest.raises(ValueError): m.summarize(p,rr)


def reusable():
    a={k:'a'*64 for k in ('task_sha256','blind_sha256','source_report_sha256','model_sha256',
       'context_digest','goal_digest','renderer_digest','reset_recipe_digest','warmup_digest')}
    a.update(root_slot=0,condition='off',task_id='t',role='after',
             causal_raw_commands=np.zeros((15,2),np.float32).tolist(),
             probe_commands=np.zeros((5,2),np.float32).tolist())
    old=dict(a,consumer_verified=True,old_consume_report_sha256='b'*64,old_physical_report_sha256='c'*64)
    return a,old


def test_reuse_is_eligibility_not_evidence():
    a,b=reusable(); r=m.reuse_check(a,b)
    assert r['eligible'] and r['evidence'] is None and not r['fresh']


@pytest.mark.parametrize('field', ['task_sha256','blind_sha256','source_report_sha256','model_sha256',
    'context_digest','goal_digest','renderer_digest','reset_recipe_digest','warmup_digest',
    'condition','role','consumer_verified','old_consume_report_sha256','causal_raw_commands','probe_commands'])
def test_reuse_strict_mismatch(field):
    a,b=reusable()
    b[field]=False if field=='consumer_verified' else [] if field.endswith('commands') else 'different'
    assert not m.reuse_check(a,b)['eligible']


def evidence_fixture(p):
    """Complete explicit NEW synthetic ABI; kept entirely in memory."""
    files={}; image=np.zeros((224,224,3),np.uint8); buf=io.BytesIO()
    Image.fromarray(image).save(buf,format='PNG'); files['frame.png']=buf.getvalue()
    fr=dict(file='frame.png',sha256=hashlib.sha256(files['frame.png']).hexdigest(),
            pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest())
    events=[]; rr=[]
    def emit(kind,**kw):
        e=dict(index=len(events),previous_sha256=events[-1]['event_sha256'] if events else None,kind=kind,**kw)
        e['event_sha256']=m.digest(e); events.append(e)
    for j in p['jobs']:
        meta=p['tasks'][j['task']]['native']
        operation_index=0
        def meter(resource):
            nonlocal operation_index
            for kind in ('operation_attempted','operation_returned'):
                emit(kind,job_digest=j['job_digest'],operation_index=operation_index,resource=resource)
            operation_index+=1
        row=dict(rows(p)[j['index']], attempted=5,completed=5,error=None,start_verified=True,
            context_digest=j['context_digest'],causal_prefix_commands=j['prefix_commands'],
            causal_replay_verified=True,rgb_guard_accepted=True,frames=[],
            root_state=meta['root_state'],goal=meta['goal'],root_sha256=meta['root_sha256'],
            goal_pose=meta['goal_pose'],goal_pixel_sha256=meta['goal_pixels'][p['tasks'][j['task']]['plan']['condition']],
            start_state=meta['root_state'])
        for i in range(6):
            state=row['start_state'] if i==0 else row['end_state']
            invariant=dict(snapshot={'state':state,'body_velocity':[0.,0.]},
                           renderer=meta['renderer'],goal_pose=meta['goal_pose'])
            row['frames'].append(dict(fr,step=i,pre=deepcopy(invariant),post=deepcopy(invariant),state=state))
        meter('constructor');meter('reset');meter('reset')
        for _ in range(18): meter('explicit_render')
        warm=np.repeat(np.asarray([[.1,0],[0,.1]],dtype='<f4'),5,axis=0)
        replay=np.concatenate([warm,m.raw(j['prefix_commands'],len(j['prefix_commands']))])
        for i,a in enumerate(replay):
            meter('public_step');meter('internal_physics_step')
            emit('causal_replay_returned',job_digest=j['job_digest'],step=i+1,raw_bits=a.tobytes().hex(),
                 state=meta['root_state'],terminated=False,truncated=False)
        for i,a in enumerate(m.raw(j['probe_commands'],5)):
            meter('public_step');meter('internal_physics_step')
            emit('probe_attempted',job_digest=j['job_digest'],step=i+1,raw_bits=a.tobytes().hex())
            emit('probe_returned',job_digest=j['job_digest'],step=i+1,raw_bits=a.tobytes().hex(),state=row['end_state'])
        rr.append(row)
        meter('close')
    emit('run_finished')
    files['journal.jsonl']=b''.join(m.canonical(e)+b'\n' for e in events)
    report=dict(schema=m.NAME+'.evidence',prepare_sha256='a'*64,executor_source_sha256='b'*64,
        status='complete',rows=rr,new_neural_calls=0,journal=dict(file='journal.jsonl',
        sha256=hashlib.sha256(files['journal.jsonl']).hexdigest()),journal_tail=events[-1]['event_sha256'],
        errors=[],counts={k:dict(attempted=n,completed=n) for k,n in
            [('public_step',700),('internal_physics_step',700),('constructor',40),('close',40),('reset',80),('explicit_render',720)]})
    return report,files


def ram_consume(p,r,files):
    """Filesystem reads mocked, no temporary fixtures created."""
    def fake_sha(path):
        name=Path(path).name
        if name=='executor.py': return 'b'*64
        if name=='report.json': return 'c'*64
        return hashlib.sha256(files[name]).hexdigest()
    def fake_read(path): return files[Path(path).name]
    with patch.object(m,'verify',return_value=p),patch.object(m,'sha',side_effect=fake_sha),\
         patch.object(m,'read',return_value=r),patch.object(m,'verify_bindings'),\
         patch.object(Path,'is_symlink',return_value=False),\
         patch.object(Path,'stat',return_value=type('Stat',(),{'st_size':1})()),\
         patch.object(Path,'read_bytes',fake_read),\
         patch.object(Image,'open',side_effect=lambda path: original_open(io.BytesIO(files[Path(path).name]))):
        return m.consume('prepared','a'*64,'/RAM/report.json','c'*64,'executor.py','b'*64)


original_open=Image.open


def test_consume_complete_RAM_evidence():
    p=fixture();r,files=evidence_fixture(p)
    result=ram_consume(p,r,files)
    assert result['fixed_denominator']==40 and result['counts']['public_step']['completed']==700


@pytest.mark.parametrize('mutation', ['count','end','context','goal','chain','partial','pixel','raw','role_prefix'])
def test_consume_rejects_wrong_evidence(mutation):
    p=fixture();r,files=evidence_fixture(p)
    if mutation=='count': r['counts']['public_step']['attempted']=199
    elif mutation=='end': r['rows'][0]['end_state']=[99.]*7
    elif mutation=='context': r['rows'][0]['context_digest']='f'*64
    elif mutation=='goal': r['rows'][0]['goal']=[99.]*7
    elif mutation=='chain': r['journal_tail']='f'*64
    elif mutation=='partial': r['rows'][0]['status']='failed'
    elif mutation=='pixel': r['rows'][0]['frames'][0]['pixel_sha256']='f'*64
    elif mutation=='role_prefix': r['rows'][0]['causal_prefix_commands']=[[0.,0.]]*5
    elif mutation=='raw':
        events=[json.loads(v) for v in files['journal.jsonl'].splitlines()]
        next(e for e in events if e['kind']=='probe_returned')['raw_bits']='bad'
        tail=None
        for e in events:
            e['previous_sha256']=tail;e['event_sha256']=m.digest({k:v for k,v in e.items() if k!='event_sha256'});tail=e['event_sha256']
        files['journal.jsonl']=b''.join(m.canonical(e)+b'\n' for e in events)
        r['journal']['sha256']=hashlib.sha256(files['journal.jsonl']).hexdigest();r['journal_tail']=tail
    with pytest.raises(ValueError): ram_consume(p,r,files)


def test_cli_help_no_build(capsys):
    with pytest.raises(SystemExit) as e: m.main(['--help'])
    assert e.value.code==0 and 'prepare' in capsys.readouterr().out


def test_cli_prepare_and_verify_RAM(capsys):
    p=fixture()
    with patch.object(m,'build',return_value=p) as build,patch.object(m,'write_new',return_value='a'*64) as write:
        m.main(['prepare','--output','actual','--report-sha','b'*64,'--prepared','planner.json',
                '--prepare-sha',m.PREPARE_SHA,'--source-sha',m.PRODUCER_SHA,
                '--destination',m.NAME+'_test.json'])
        assert build.call_args.kwargs['output']=='actual' and write.call_args.args[1]==p
    with patch.object(m,'verify',return_value=p) as verify:
        m.main(['verify','--prepared','probe.json','--prepare-sha','a'*64])
        verify.assert_called_once_with('probe.json','a'*64)


@pytest.mark.parametrize('mutation',['renderstate','renderinvariant','renderer','missing_axis'])
def test_full_physical_axes_and_frame_invariant(mutation):
    p=fixture();r,files=evidence_fixture(p)
    if mutation=='renderstate': r['rows'][0]['frames'][0]['state']=[99.]*7
    elif mutation=='renderinvariant': r['rows'][0]['frames'][0]['post']['snapshot']['state']=[99.]*7
    elif mutation=='renderer': r['rows'][0]['frames'][0]['pre']['renderer']={'size':32}
    elif mutation=='missing_axis': r['counts'].pop('reset')
    with pytest.raises(ValueError): ram_consume(p,r,files)
