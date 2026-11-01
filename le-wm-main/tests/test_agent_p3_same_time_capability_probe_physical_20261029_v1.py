"""RAM/synthetic PNG transport only. No native env/model is imported."""
from copy import deepcopy
import ast
import io
from pathlib import Path
import struct

import pytest
from PIL import Image
from experiments.spectral_reachability import agent_p3_same_time_capability_probe_physical_20261029_v1 as m


def png():
    b=io.BytesIO();Image.new('RGB',(224,224),(31,91,151)).save(b,format='PNG');return b.getvalue()


def observation(b,phase,step,overlay):
    data=png();state=[0.]*7;goal=[1.]*7
    history=m.f32_bits(b['warm10'][:step],step) if phase=='warmup' else m.f32_bits(b['warm10'],10)+m.f32_bits((b['raw15'] or [[0.,0.]]*15)[:step],step)
    return dict(phase=phase,step=step,overlay=overlay,png=f'{phase}_{step}_{overlay}.png',file_sha256=m.digest(data),pixel_sha256=m.png_identity(data),
        state=state,snapshot=dict(state=state,state_float64_bits=struct.pack('<7d',*state).hex(),
             rng={'fixture':[1,2]},public_state={'fixture':True},goal=goal),raw_history_bits=history,goal=goal)


def binding(job=0):
    endpoint=job//2;root,role=m.ORDER[endpoint];probe=('U','Z2')[job%2]
    b=dict(schema=m.NAME+'.fixture_binding',fixture_only=True,job=job,endpoint=endpoint,root=root,source_role=role,probe=probe,
        warm10=[[0.,0.]]*10,raw15=[[0.,0.]]*10+([[.25,-.25]]*5 if probe=='U' else [[0.,0.]]*5),reference66=[],U_dual_replay=None)
    if probe=='U':b['U_dual_replay']=dict(raw5_bits=m.f32_bits(b['raw15'][10:],5),whole_IO_sha256=m.digest(b'fixture-only IO'),segmented_IO_sha256=m.digest(b'fixture-only IO'))
    b['reference66']=[observation(b,*key) for key in m.frame_keys()[:66]]
    return b


def complete(store,*,stop_at=None,fail_resource=None):
    b=store.replay.b;store.emit('run_started',dict(binding_sha256=m.digest(m.raw(b))))
    def op(resource,phase='setup',step=None,overlay=None,action=None,result=None,failed=False):
        ident=len(store.replay.ops)
        store.emit('operation_dispatch',dict(id=ident,resource=resource,phase=phase,step=step,overlay=overlay,action=action))
        store.emit('operation_failed' if failed else 'operation_returned',dict(id=ident,result=result,error={'fixture':'failed'} if failed else None))
    op('constructor')
    for resource,n in [('reset',2),('set_state',4),('goal_render',2),('goal_bind',1),('goal_setter',3),('physics_step',4)]:
        for _ in range(n):op(resource)
    for phase,end in [('warmup',10),('plan',15)]:
        for step in range(end+1):
            if phase=='plan' and step in (6,11):
                n=51 if step==6 else 66
                store.emit('guard',dict(n=n,accepted=True,proof=store.replay.proof(n),error=None))
            if step:
                actions=b['warm10'] if phase=='warmup' else b['raw15'];ret=dict(state=[0.]*7,terminated=stop_at==(phase,step),truncated=False)
                op('public_step',phase,step,action=actions[step-1],result=ret)
                for _ in range(10):op('physics_step',phase,step)
            for overlay in range(3):
                f=observation(b,phase,step,overlay);op('explicit_render',phase,step,overlay,result={'pixel_sha256':f['pixel_sha256']})
                store.observation(f,png())
            if stop_at==(phase,step):
                op('close',phase='cleanup');closed=dict(status='native_stopped',reason='original native termination')
                store.emit('branch_closed',closed);store.emit('run_finished',dict(closed=closed));return
    op('close',phase='cleanup');closed=dict(status='completed',reason=None)
    store.emit('branch_closed',closed);store.emit('run_finished',dict(closed=closed))


def test_shape_proof_fixed_all10():
    c=m.contract();s=m.shape_census()
    assert len(c['order'])==10 and [x['probe'] for x in c['order']]==['U','Z2']*5
    assert s['operation_receipts']==sum(m.COUNTS.values())==374
    assert s['total_events']==834 and s['observation_events']==81
    assert s['conservative_total_bytes']<64*1024**2
    assert not c['production_ready'] and c['fixture_only']


@pytest.mark.parametrize('job',range(10))
def test_binding_order(job):
    b=binding(job);m.validate_binding(b)
    assert b['root']==m.ORDER[job//2][0]


@pytest.mark.parametrize('what',['future_authority','order','rawdomain','double_map','negative_zero','missing66','dual'])
def test_binding_failclosed(what):
    b=binding(1 if what=='negative_zero' else 0)
    if what=='future_authority':b['fixture_only']=False
    elif what=='order':b['source_role']='arbitrary'
    elif what=='rawdomain':b['raw15'][-1]=[2.,0.]
    elif what=='double_map':b['raw15'][-1]=[.1,0.]
    elif what=='negative_zero':b['raw15'][-1]=[-0.,0.]
    elif what=='missing66':b['reference66'].pop()
    else:b['U_dual_replay']['segmented_IO_sha256']='0'*64
    with pytest.raises(ValueError):m.validate_binding(b)


@pytest.fixture(scope='module')
def full_job(tmp_path_factory):
    folder=tmp_path_factory.mktemp('probe_transport')/(m.NAME+'_fixture')
    store=m.Store(folder,binding());complete(store);published=store.finalize()
    return folder,published


def test_complete81_allfields_census(full_job):
    folder,p=full_job;r=m.consume_fixture(folder,p['report_sha256'],p['completion_sha256'])
    assert r['PNG']==81 and r['event_count']==834
    assert r['report']['guards']==[51,66] and len(r['report']['operations'])==374
    assert r['report']['endpoint']['step']==15 and r['report']['native_metrics'] is None
    assert r['fixture_only'] and not r['production_ready']


def test_guard_durable_before_intent(tmp_path,monkeypatch):
    s=m.Store(tmp_path/(m.NAME+'_durable'),binding());s.emit('run_started',dict(binding_sha256=m.digest(m.raw(s.replay.b))))
    # A saved guard with incomplete observations cannot be faked.
    with pytest.raises(ValueError):s.emit('guard',dict(n=51,accepted=True,proof={},error=None))
    assert not s.replay.guards and s.seq==1
    # Even with a correct validator proof, failed fsync cannot activate the guard.
    monkeypatch.setattr(s.replay,'proof',lambda n:dict(test='valid-fixture-proof'))
    original=s.put
    def fail(name,data,**kw):
        if name=='journal.jsonl':raise OSError('fixture fsync failure')
        return original(name,data,**kw)
    monkeypatch.setattr(s,'put',fail)
    with pytest.raises(OSError):s.emit('guard',dict(n=51,accepted=True,proof={'test':'valid-fixture-proof'},error=None))
    assert s.replay.guards==[] and s.seq==1


@pytest.mark.parametrize('step',[6,11])
def test_no_suffix_without_guard(step):
    r=m.Replay(binding());r.feed('run_started',dict(binding_sha256=m.digest(m.raw(r.b))))
    # Arrange only prior intent count; no guard, never a native call.
    r.ops=[dict(dispatch={'resource':'public_step','phase':'plan'}) for _ in range(step-1)]
    r.counts['constructor']['completed']=1
    with pytest.raises(ValueError,match='guard BEFORE'):r.feed('operation_dispatch',dict(id=len(r.ops),resource='public_step',phase='plan',step=step,overlay=None,action=r.b['raw15'][step-1]))


def test_U_missing_blocked_no_env(tmp_path):
    b=binding();b.update(raw15=None,U_dual_replay=None)
    s=m.Store(tmp_path/(m.NAME+'_blocked'),b);s.emit('run_started',dict(binding_sha256=m.digest(m.raw(b))))
    with pytest.raises(ValueError):s.emit('operation_dispatch',dict(id=0,resource='constructor',phase='setup',step=None,overlay=None,action=None))
    closed=dict(status='blocked',reason='dual replay missing, no fallback')
    s.emit('branch_closed',closed);s.emit('run_finished',dict(closed=closed));p=s.finalize()
    result=m.consume_fixture(s.path,p['report_sha256'],p['completion_sha256'])
    assert result['PNG']==0 and result['report']['endpoint'] is None
    assert result['report']['counts']['constructor']['attempted']==0


def test_native_early_stop_missing_not_padding(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_early'),binding(1));complete(s,stop_at=('plan',12));p=s.finalize()
    r=m.consume_fixture(s.path,p['report_sha256'],p['completion_sha256'])
    assert r['PNG']==72 and r['report']['endpoint'] is None
    assert r['report']['missing_endpoint_reason']=='native_stopped'
    assert r['report']['counts']['public_step']['attempted']==22


@pytest.mark.parametrize('what',['report','event','hidden','PNG','completion'])
def test_saved_mutation_rejected(tmp_path,full_job,what):
    folder,p=full_job;out=tmp_path/'copy';import shutil;shutil.copytree(folder,out)
    rh,ch=p['report_sha256'],p['completion_sha256']
    if what=='report':
        v=m.decode((out/'report.json').read_bytes());v['native_metrics']={'fake':True};(out/'report.json').write_bytes(m.raw(v));rh=m.sha(out/'report.json')
    elif what=='event':
        lines=(out/'journal.jsonl').read_bytes().splitlines(keepends=True);lines[0]=lines[0].replace(b'run_started',b'bad_started');(out/'journal.jsonl').write_bytes(b''.join(lines))
    elif what=='hidden':(out/'hidden.json').write_bytes(b'{}')
    elif what=='PNG':next(out.glob('*.png')).write_bytes(b'not PNG')
    else:
        v=m.decode((out/'completion.json').read_bytes());v['finished']=False;(out/'completion.json').write_bytes(m.raw(v));ch=m.sha(out/'completion.json')
    with pytest.raises(ValueError):m.consume_fixture(out,rh,ch)


def test_entire_report_reconstruction_not_inventory_only(tmp_path,full_job):
    import shutil
    folder,p=full_job;out=tmp_path/'copy';shutil.copytree(folder,out)
    report=m.decode((out/'report.json').read_bytes());report['unknown']='tampered';(out/'report.json').write_bytes(m.raw(report))
    c=m.decode((out/'completion.json').read_bytes());c['inventory']['report.json']=m.sha(out/'report.json');(out/'completion.json').write_bytes(m.raw(c))
    with pytest.raises(ValueError,match='entire report'):m.consume_fixture(out,m.sha(out/'report.json'),m.sha(out/'completion.json'))


def test_cap_reserve_quotas_and_failed_final(tmp_path,monkeypatch):
    s=m.Store(tmp_path/(m.NAME+'_quota'),binding())
    with pytest.raises(ValueError):s.put('journal.jsonl',b'x'*(m.QUOTAS['journal.jsonl']+1),append=True)
    with pytest.raises(ValueError):s.put('../foreign',b'x')
    with pytest.raises(ValueError):s.put('foreign.json',b'{}')
    ticks=iter((901.,902.));s.clock=lambda:next(ticks);s.start=0
    with pytest.raises(ValueError,match='wall failed'):s.finalize()
    assert (s.path/'failure.json').exists() and (s.path/'report.json').exists()
    with pytest.raises(ValueError):m.consume_fixture(s.path,m.sha(s.path/'report.json'),m.sha(s.path/'completion.json'))


def test_production_cli_never_writes_prepare(tmp_path):
    out=tmp_path/'do_not_create'
    for command in ('prepare','run'):
        with pytest.raises(ValueError,match='unavailable'):m.main([command,'--output',str(out)])
    assert not out.exists()


def test_no_neural_physical_or_upstream_imports():
    tree=ast.parse(Path(m.__file__).read_text())
    for n in ast.walk(tree):
        if isinstance(n,ast.ImportFrom):assert not n.module.startswith(('torch','experiments','pymunk','gym'))
        if isinstance(n,ast.Import):assert not any(a.name.startswith(('torch','experiments','pymunk','gym')) for a in n.names)


def test_protocol_full_read_pin():
    assert m.sha(m.PROTOCOL)==m.PROTOCOL_SHA


def test_late_reset_and_paid_cap_rejected():
    r=m.Replay(binding());r.feed('run_started',dict(binding_sha256=m.digest(m.raw(r.b))))
    r.counts['constructor']['completed']=1;r.frames=[{'fixture':'observed'}]
    with pytest.raises(ValueError,match='poststate'):r.feed('operation_dispatch',dict(id=0,resource='reset',phase='setup',step=None,overlay=None,action=None))
    r.counts['physics_step']['attempted']=254
    with pytest.raises(ValueError,match='paid per-resource cap'):r.feed('operation_dispatch',dict(id=0,resource='physics_step',phase='plan',step=11,overlay=None,action=None))


def test_failed_operation_partial_full_reconstruction(tmp_path):
    b=binding();s=m.Store(tmp_path/(m.NAME+'_failure'),b)
    s.emit('run_started',dict(binding_sha256=m.digest(m.raw(b))))
    s.emit('operation_dispatch',dict(id=0,resource='constructor',phase='setup',step=None,overlay=None,action=None))
    s.emit('operation_failed',dict(id=0,result=None,error={'where':'constructor','reason':'fixture failure'}))
    closed=dict(status='engineering_failed',reason='constructor exception')
    s.emit('branch_closed',closed);s.emit('run_finished',dict(closed=closed));p=s.finalize()
    r=m.consume_fixture(s.path,p['report_sha256'],p['completion_sha256'])
    assert r['status']=='engineering_failed' and r['report']['endpoint'] is None
    assert r['report']['counts']['constructor']==dict(attempted=1,completed=0,failed=1)
    assert r['report']['errors']==[{'where':'constructor','reason':'fixture failure'}]


def test_failed_guard_is_not_retryable():
    r=m.Replay(binding());r.feed('run_started',dict(binding_sha256=m.digest(m.raw(r.b))))
    event=dict(n=51,accepted=False,proof=None,error={'fixture':'not equal'})
    r.feed('guard',event)
    with pytest.raises(ValueError,match='no retry'):r.feed('guard',event)
    assert not r.guards and r.guard_attempts==[51]


def test_all10_batch_blocked_slots_not_dropped(tmp_path):
    entries=[]
    for job in range(10):
        b=binding(job);s=m.Store(tmp_path/(m.NAME+'_batch_'+str(job)),b)
        s.emit('run_started',dict(binding_sha256=m.digest(m.raw(b))))
        closed=dict(status='blocked',reason='fixture: upstream not integrated')
        s.emit('branch_closed',closed);s.emit('run_finished',dict(closed=closed));p=s.finalize()
        entries.append(dict(job=job,output=str(s.path),report_sha256=p['report_sha256'],completion_sha256=p['completion_sha256']))
    r=m.consume_batch_fixture(entries)
    assert r['fixed_denominator']==r['missing_endpoints']==10
    assert r['counts']['constructor']==dict(attempted=0,completed=0,failed=0)
    with pytest.raises(ValueError):m.consume_batch_fixture(entries[:-1])
    entries[1]=dict(entries[0],job=1)
    with pytest.raises(ValueError):m.consume_batch_fixture(entries)
