"""RAM-only H1 recurrence, provenance mutations and nonrefundable meters."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
import sys
import zlib
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_observed_suffix_fd_cuda_20261027_v1 as m

A = np.sin(np.arange(1920).reshape(192,10)/37)/10


def context(z,h):
    return dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
                latent_history=np.asarray(z).tolist(),action_history=np.asarray(h).tolist())


def task_fixture():
    t = dict(environment='pusht',task_contract_sha256='a'*64,
        root_context=context(np.zeros((3,192)),np.zeros((2,10))),root_raw_action_history=np.zeros((10,2)).tolist(),
        goal=np.ones(192).tolist(),projection=np.eye(192).tolist(),model_sha256='b'*64,
        model_config_sha256='c'*64,metric_source_sha256='d'*64,goal_source_sha256='e'*64,
        representation='native_projected_cls',latent_dim=192,horizon=2,packet_steps=5,control_dim=2,
        action_mean=[.1,-.2],action_std=[.7,1.3],packet_logit_cap=float(.75*np.sqrt(5)),feedback='open_loop_model_only')
    t['root_context']['action_history'] = m.screen.core.normalized_tokens(t,np.zeros((10,2))).astype(float).tolist()
    m.screen.core.validate_task(t)
    return t


def step(task,before,raw):
    token = m.screen.core.normalized_tokens(task,raw)[0].astype(float)
    value = (.7*np.asarray(before['latent_history'][-1]) + .3*A@np.asarray(before['action_history'][-1]) + A@token).astype(np.float32)
    return m.advance(task,before,raw,value)


def protocol():
    points=[]; problems={}; matrices={}
    for k,b in enumerate(m.BRANCHES):
        t=task_fixture()
        raw=np.full((5,2),.125*(k+1),np.float32)
        before=deepcopy(t['root_context'])
        before['latent_history']=deepcopy(before['latent_history'][1:])+np.full((1,192),.1*(k+1),np.float32).astype(float).tolist()
        before['action_history']=[deepcopy(before['action_history'][-1]),m.screen.core.normalized_tokens(t,raw)[0].astype(float).tolist()]
        nominal=step(t,before,np.zeros((5,2),np.float32))
        j=A/np.tile(np.asarray(t['action_std'],np.float32),5).astype(float)
        points.append(dict(point_order=k,branch=b,source_slot=k,arm=m.base.ARMS[1],parent_task_id=m.screen.core.task_id(t),
            observed_problem_id=str(k)*64,context_sha256=m.screen.core.digest(before),at_logits=np.zeros((5,2)).tolist(),
            at_terminal=nominal['latent_history'][-1],nominal_context_sha256=m.screen.core.digest(nominal),
            logit_J_sha256=m.screen.core.digest(j.tolist()),
            saved_Jv={d['id']:(j@np.asarray(d['vector'])).tolist() for d in m.directions()}))
        problems[k]=dict(task=t,context=before); matrices[k]=j
    return dict(points=points,directions=m.directions(),epsilons=list(m.EPS),signs=list(m.SIGNS),
                numerical={},inherited_feedback_resources={}),problems,matrices


class Backend:
    def __init__(self,problem): self.problem=problem; self.active=False; self.guards=0; self.seen=[]
    @contextmanager
    def guard(self,meter):
        assert not self.active
        self.active=True; self.guards+=1
        try: yield
        finally: self.active=False
    def step(self,before,raw,meter):
        assert self.active and before==self.problem['context'] and raw.shape==(5,2) and raw.dtype==np.float32
        self.seen.append(deepcopy(before))
        for resource in ('action_encoder','predict'):
            e=meter.begin(resource); meter.end(e)
        c=step(self.problem['task'],before,raw)
        meter.emit('native_packet',mode='observed_H1',packet=0,context=c)
        return c


def workflow():
    p,problems,js=protocol(); events=[]; meter=m.Meter(events.append)
    backends={k:Backend(v) for k,v in problems.items()}
    rows=m.evaluate(p,problems,lambda k:backends[k],meter,events.append)
    return p,problems,js,events,meter,backends,rows


def test_fixed_budget_directions_and_no_H2_charge():
    b=m.budget()
    assert b['predict']==b['action_encoder']==2*(1+3*3*2)==38
    assert b['total_H1_evaluations']==38 and b['central_comparisons']==18
    assert b['AD_calls']==b['new_jacobians']==b['encoder']==b['projector']==b['physics_steps']==b['selection']==0
    assert m.EPS==(1e-3,3e-3,1e-2) and m.BRANCHES==(1,3)
    ds=m.directions(); assert ds[0]['vector'][0]==ds[1]['vector'][1]==1
    assert all(abs(np.linalg.norm(d['vector'])-1)<1e-14 for d in ds)
    assert b['equal_total_compute_claim'] is False


def test_whole_H1_workflow_all_values_retained():
    p,_,js,events,meter,backends,rows=workflow(); result=m.analyze(p,rows,js)
    assert len(rows)==38 and len(result['comparisons'])==18 and len(result['epsilon_stability'])==6
    assert meter.counts()=={k:dict(attempted=38,completed=38) for k in ('predict','action_encoder')}
    assert all(not b.active and b.guards==1 and len(b.seen)==19 for b in backends.values())
    assert sum(e['kind']=='native_packet' for e in events)==38
    assert all(len(r['context']['latent_history'])==3 and len(r['context']['action_history'])==2 for r in rows)
    assert result['scientific_pass_threshold'] is None and result['derivative_verdict']=='not_assigned'
    assert max(c['requested_direction_metrics']['relative_l2'] for c in result['comparisons'])<.004 # mock only


def test_syncflush_fullsize_RAM_receipt_capacity_and_readback():
    p,_,js,events,_,_,rows=workflow()
    comp=zlib.compressobj(6,zlib.DEFLATED,31)
    raw=b''.join(comp.compress(m.canonical(e))+comp.flush(zlib.Z_SYNC_FLUSH) for e in events)+comp.flush(zlib.Z_FINISH)
    assert [m.base.decode(x) for x in zlib.decompress(raw,31).splitlines()]==events
    analysis=zlib.compress(m.canonical(m.analyze(p,rows,js)))
    assert len(raw)+len(analysis)+m.FINAL+4096<m.CAP


@pytest.mark.parametrize('eps',m.EPS)
@pytest.mark.parametrize('sign',m.SIGNS)
def test_float32_logit5x2_then_numpy_tanh_once(eps,sign):
    v=m.directions()[2]['vector']; row=m.perturb(v,eps,sign)
    x=(sign*eps*np.asarray(v).reshape(5,2)).astype(np.float32)
    assert np.array_equal(row['logits'],x) and np.array_equal(row['commands'],np.tanh(x))


@pytest.mark.parametrize('damage',('drop','swap','nominal','history','logits','J'))
def test_all_rows_exact_anchor_and_full_J_fail_closed(damage):
    p,_,js,_,_,_,rows=workflow()
    if damage=='drop': rows.pop()
    elif damage=='swap': rows[1],rows[2]=rows[2],rows[1]
    elif damage=='nominal': rows[0]['terminal'][0]+=.1
    elif damage=='history': rows[1]['context']['action_history'][0][0]+=.1
    elif damage=='logits': rows[1]['logits'][0][0]+=.1
    else: js[0][0][0]+=.1
    with pytest.raises(ValueError): m.analyze(p,rows,js)


def test_negative_derivative_evidence_not_scientific_gate():
    p,_,js,_,_,_,rows=workflow(); js[0]*=192
    p['points'][0]['logit_J_sha256']=m.screen.core.digest(js[0].tolist())
    for d in m.directions(): p['points'][0]['saved_Jv'][d['id']]=(js[0]@np.asarray(d['vector'])).tolist()
    result=m.analyze(p,rows,js)
    assert result['comparisons'][0]['requested_direction_metrics']['relative_l2']>.9
    assert result['P1_admission'] is False and result['scientific_pass_threshold'] is None
    assert m.errors(np.zeros(192),np.zeros(192))['relative_l2'] is None


def test_nonrefund_sync_failure_and_no_second_packet():
    p,_,_=protocol(); first=m.requests(p)[0]
    def failed_sync(): raise RuntimeError('sync')
    meter=m.Meter(lambda e:None,sync=failed_sync); meter.start(first); e=meter.begin('action_encoder')
    with pytest.raises(RuntimeError): meter.end(e)
    assert meter.counts()['action_encoder']==dict(attempted=1,completed=0)
    with pytest.raises(ValueError): meter.begin('predict')
    with pytest.raises(ValueError): meter.start(first)
    _,_,_,_,meter,_,_=workflow()
    with pytest.raises(ValueError): meter.begin('action_encoder')


def test_no_grad_gate_and_disordered_calls():
    p,_,_=protocol(); first=m.requests(p)[0]
    meter=m.Meter(lambda e:None,no_grad=lambda:False); meter.start(first)
    with pytest.raises(ValueError): meter.begin('action_encoder')
    assert meter.events==[]
    meter=m.Meter(lambda e:None); meter.start(first)
    with pytest.raises(ValueError): meter.begin('predict')
    assert meter.events==[]


def test_anchor_mismatch_is_durable_before_failure_and_guard_cleanup():
    p,problems,_=protocol(); p['points'][0]['at_terminal'][0]+=.1
    events=[]; rows=[]; b=Backend(problems[0]); meter=m.Meter(events.append)
    with pytest.raises(ValueError,match='nominal EXACT'):
        m.evaluate(p,problems,lambda k:b,meter,events.append,rows)
    assert len(rows)==1 and events[-1]['kind']=='evaluation_complete' and not b.active


@pytest.mark.parametrize('damage',('nonzero','third','direction','branch','eps'))
def test_no_extra_points_directions_or_eps(damage):
    p,_,_=protocol()
    if damage=='nonzero': p['points'][0]['at_logits'][0][0]=.1
    elif damage=='third': p['points'].append(deepcopy(p['points'][0]))
    elif damage=='direction': p['directions'][0]['vector'][0]=-1
    elif damage=='branch': p['points'][0]['branch']=0
    else: p['epsilons'].append(.1)
    with pytest.raises(ValueError): m.requests(p)


def extract_fixture():
    p,problems,js=protocol(); problem=problems[0]; task=problem['task']; c=problem['context']; j=js[0]
    prefix=np.asarray(c['latent_history'][-1],np.float32); raw=np.full((5,2),.125,np.float32)
    rec=dict(branch=1,arm=m.base.ARMS[1],source_slot=0,root_slot=0,parent_task_id=m.screen.core.task_id(task),
        observed_context_sha256=m.screen.core.digest(c),observed_problem_id='a'*64,blind_sha256='b'*64,
        task_export='mock_task',task_sha256='c'*64)
    nominal=step(task,c,np.zeros((5,2),np.float32))
    result=dict(context=c,fresh_jacobian=j.tolist(),zero_nominal=nominal['latent_history'][-1],
        candidates=[dict(index=0,kind='zero_nominal',commands=np.zeros((5,2)).tolist(),context=nominal)])
    plan=dict(original_prefix_commands=raw.astype(float).tolist(),observed_context=deepcopy(c),record=rec)
    events=[dict(kind='suffix_problem_frozen',context=deepcopy(c),parent_task_id=rec['parent_task_id']),
        dict(kind='zero_nominal_saved',terminal=result['zero_nominal']),
        dict(kind='jacobian',completed=False,shape=[192,10],mathematical_vjp_directions=192),
        dict(kind='jacobian',completed=True,shape=[192,10],mathematical_vjp_directions=192),
        dict(kind='fresh_J_saved',jacobian=j.tolist(),nominal=result['zero_nominal']),
        dict(kind='ridge_frozen_before_prediction')]
    return task,rec,result,plan,events,prefix


@pytest.mark.parametrize('damage',('none','J','nominal','history','prefix','task','duplicate','chronology','shape'))
def test_original_J_zero_and_observed_context_content(damage):
    task,rec,result,plan,events,prefix=extract_fixture()
    if damage=='J': result['fresh_jacobian'][0][0]+=.1
    elif damage=='nominal': result['zero_nominal']=np.ones(192).tolist()
    elif damage=='history': result['context']['action_history'][0][0]+=.1
    elif damage=='prefix': prefix[0]+=.1
    elif damage=='task': rec['parent_task_id']='wrong'
    elif damage=='duplicate': events.append(deepcopy(events[1]))
    elif damage=='chronology': events[1],events[-1]=events[-1],events[1]
    elif damage=='shape': events[2]['shape']=[192,20]
    if damage=='none':
        point,j,problem=m.extract_point(task,rec,result,plan,events,0,prefix)
        assert j.shape==(192,10) and point['branch']==1 and problem['context']==result['context']
    else:
        with pytest.raises(ValueError): m.extract_point(task,rec,result,plan,events,0,prefix)


def test_journal_failure_preserves_attempt_no_refund():
    p,_,_=protocol()
    def fail(e):
        if e['kind']=='module_attempt': raise OSError('fsync failure')
    meter=m.Meter(fail); meter.start(m.requests(p)[0])
    with pytest.raises(OSError): meter.begin('action_encoder')
    assert meter.counts()['action_encoder']==dict(attempted=1,completed=0)
    with pytest.raises(ValueError): meter.begin('predict')


def test_native_H1_adapter_calls_exactly_one_step_from_observed_not_parent():
    p,problems,_=protocol(); problem=problems[0]; calls=[]
    # No torch import or model: exercise the real H1 adapter with array plumbing.
    backend=object.__new__(m.NativeH1); backend.task=problem['task']
    backend.torch=type('NoGrad',(),{'no_grad':staticmethod(nullcontext)})()
    backend._tensor=lambda a:np.asarray(a,np.float32)
    meter=m.Meter(lambda e:None); meter.start(m.requests(p)[0]); backend._active_meter=meter
    def single(task,z,h,raw):
        calls.append(1)
        assert task is problem['task'] and z.shape==(1,3,192) and h.shape==(1,2,10) and raw.shape==(5,2)
        assert np.array_equal(z[0],problem['context']['latent_history'])
        assert np.array_equal(h[0],problem['context']['action_history'])
        assert not np.array_equal(z[0],task['root_context']['latent_history'])
        for resource in ('action_encoder','predict'):
            e=meter.begin(resource); meter.end(e)
        c=step(task,problem['context'],raw)
        return np.asarray(c['latent_history'])[None],np.asarray(c['action_history'])[None]
    backend._step=single; backend._saved=lambda z,h:context(z[0],h[0])
    c=backend.step(problem['context'],np.zeros((5,2),np.float32),meter)
    assert calls==[1] and meter.paths==[c] and meter.packet==1
    assert meter.counts()=={k:dict(attempted=1,completed=1) for k in ('predict','action_encoder')}


def test_startup_and_cleanup_failure_preserve_first_error_RAM(monkeypatch):
    p,problems,js=protocol(); mem={}
    class RAMPath:
        def iterdir(self): return []
    class RAMStore:
        def __init__(self,output): self.output=RAMPath()
        def put(self,name,value,final=False): mem[name]=deepcopy(value)
        def event(self,e): mem.setdefault('events',[]).append(deepcopy(e))
        def close(self): raise OSError('finalize')
    monkeypatch.setattr(m,'Store',RAMStore); monkeypatch.setattr(m,'verify',lambda *a:(p,problems,js))
    monkeypatch.setitem(sys.modules,'torch',None)
    monkeypatch.setenv('HF_HUB_OFFLINE','1'); monkeypatch.setenv('TRANSFORMERS_OFFLINE','1')
    with pytest.raises(ValueError,match='partial H1 FD preserved'): m.run('unused','unused','unused')
    r=mem['report.json']; assert r['status']=='partial_or_failed' and r['first_error']['stage']=='execution'
    assert r['errors'][-1]['stage']=='journal_finalize' and r['model_hash_before'] is None
    assert r['resources']['predict']==dict(attempted=0,completed=0)


def test_actual_saved_RAM_build_no_model_or_prepare_output():
    if not m.INPUT.is_file(): pytest.skip('actual receipt unavailable')
    p,problems,js=m.collect()
    assert len(p['points'])==len(problems)==len(js)==2
    assert [p['branch'] for p in p['points']]==[1,3]
    assert p['input_report_sha256']==m.INPUT_SHA and len(m.canonical(p))<m.PREPARE_CAP
    assert all(j.shape==(192,10) for j in js.values())
    assert not any('certificate' in path or Path(path).name=='sourcepack.json' for path in p['bindings_sha256'])
    assert not any('image_03.native192.npy' in path or 'image_07.native192.npy' in path for path in p['bindings_sha256'])


def test_output_floor_and_cap_no_temp():
    with pytest.raises(ValueError): m.fd.check_space(0,1,m.FLOOR)
    with pytest.raises(ValueError): m.fd.check_space(m.CAP,1,m.FLOOR+4096)
    assert m.CAP==2*1024**2 and m.FLOOR==200*1024**2
