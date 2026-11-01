"""RAM-only NumPy recurrence/fault mocks; no CUDA, images or physics."""
from contextlib import contextmanager
from copy import deepcopy
import sys
import zlib
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_gn_fd_cuda_20261027_v1 as m

A=np.sin(np.arange(1920).reshape(192,10)/37)/10
B=np.cos(np.arange(1920).reshape(192,10)/41)/10


def context(z,h):
    return dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=z.tolist(),action_history=h.tolist())


def path(task,raw):
    z=np.asarray(task['root_context']['latent_history'],np.float32).astype(float)
    h=np.asarray(task['root_context']['action_history'],np.float32); out=[]
    for k in (0,1):
        token=(np.asarray(raw[5*k:5*k+5],np.float32).reshape(10)-np.tile(np.asarray(task['action_mean'],np.float32),5))/np.tile(np.asarray(task['action_std'],np.float32),5)
        v=A@token if k==0 else .7*z[-1]+.3*A@h[-1]+B@token
        z=np.vstack((z[1:],v.astype(np.float32).astype(float))); h=np.vstack((h[1:],token))
        out.append(context(z,h))
    return out


def protocol():
    points=[]; tasks={}; js={}
    for k in (0,1):
        tasks[k]=dict(root_context=context(np.zeros((3,192)),np.zeros((2,10))),action_mean=[.1,-.2],action_std=[.7,1.3])
        j=np.column_stack((A,B))/np.tile(tasks[k]['action_std'],10)[None,:]
        p=path(tasks[k],np.zeros((10,2)))
        points.append(dict(point_order=k,root_index=k,root_sha256=None,task_id=str(k),iteration=1,
            at_logits=np.zeros((2,10)).tolist(),at_terminal=p[-1]['latent_history'][-1],
            saved_context_sha256=[m.screen.core.digest(c) for c in p],
            saved_Jv={d['id']:(j@np.asarray(d['vector'])).tolist() for d in m.directions()}))
        js[k]=j
    return dict(points=points,directions=m.directions(),numerical={},inherited_planner_resources={}),tasks,js


class Backend:
    def __init__(self): self.active=False; self.guards=0
    @contextmanager
    def guard(self,meter):
        assert not self.active
        self.active=True; self.guards+=1
        try: yield
        finally: self.active=False
    def rollout(self,task,raw,meter,mode):
        assert self.active and mode=='whole_replay'
        result=path(task,raw)
        for k,c in enumerate(result):
            for resource in ('action_encoder','predict'):
                token=meter.begin(resource); meter.end(token)
            meter.emit('native_packet',mode=mode,packet=k,context=c)
        return result


def workflow():
    p,tasks,js=protocol(); log=[]; meter=m.Meter(log.append); backends={k:Backend() for k in tasks}
    rows=m.evaluate(p,tasks,lambda k:backends[k],meter,log.append)
    return p,tasks,js,log,meter,backends,rows


def test_exact_budget_and_fixed_not_result_selected_protocol():
    b=m.budget(); assert b['predict']==b['action_encoder']==2*(1+3*3*2)*2==76
    assert b['AD_calls']==b['new_jacobians']==b['encoder']==b['physics_steps']==b['selection']==0
    assert m.EPS==(1e-3,3e-3,1e-2)
    ds=m.directions(); assert ds[0]['vector'][0]==ds[1]['vector'][10]==1
    assert all(abs(np.linalg.norm(d['vector'])-1)<1e-14 for d in ds)
    assert b['equal_total_compute_claim'] is False


def test_complete_two_state_chains_and_all_76_calls():
    p,_,js,log,meter,backends,rows=workflow(); a=m.analyze(p,rows,js)
    assert len(rows)==38 and len(a['comparisons'])==18 and len(a['epsilon_stability'])==6
    assert a['nominal_anchors']==2 and a['scientific_pass_threshold'] is None and a['P1_admission'] is False
    assert meter.counts()=={k:dict(attempted=76,completed=76) for k in ('predict','action_encoder')}
    assert all(not b.active and b.guards==1 for b in backends.values())
    assert max(c['requested_direction_metrics']['relative_l2'] for c in a['comparisons'])<.004 # mock only
    compressed=zlib.compress(m.canonical(a)); assert len(compressed)<100*1024


def test_full_mock_stream_syncflush_cap_in_RAM():
    p,_,js,log,_,_,rows=workflow()
    comp=zlib.compressobj(6,zlib.DEFLATED,31)
    raw=b''.join(comp.compress(m.canonical(e))+comp.flush(zlib.Z_SYNC_FLUSH) for e in log)+comp.flush(zlib.Z_FINISH)
    analysis=zlib.compress(m.canonical(m.analyze(p,rows,js)))
    assert len(raw)+len(analysis)+m.FINAL<m.CAP
    plain=zlib.decompress(raw,31)
    assert [m.planner.decode(x) for x in plain.splitlines()]==log


@pytest.mark.parametrize('eps',m.EPS)
@pytest.mark.parametrize('sign',m.SIGNS)
def test_float32_once_and_effective_direction(eps,sign):
    p,_,_=protocol(); point=p['points'][0]; d=m.directions()[2]
    r=m.perturb(point,d['vector'],eps,sign)
    x=(sign*eps*np.asarray(d['vector']).reshape(2,10)).astype(np.float32)
    assert np.array_equal(r['logits'],x) and np.array_equal(r['commands'],np.tanh(x).reshape(10,2))


@pytest.mark.parametrize('damage',('drop','swap','nominal','context','logits'))
def test_all_rows_and_exact_anchor_fail_closed(damage):
    p,_,js,_,_,_,rows=workflow()
    if damage=='drop': rows.pop()
    elif damage=='swap': rows[1],rows[2]=rows[2],rows[1]
    elif damage=='nominal': rows[0]['terminal'][0]+=.1
    elif damage=='context': rows[19]['context_sha256'][0]='0'*64
    else: rows[1]['logits'][0][0]+=.1
    with pytest.raises(ValueError): m.analyze(p,rows,js)


@pytest.mark.parametrize('factor',(.7,.3,2.))
def test_wrong_saved_J_is_negative_not_rejected_by_scientific_gate(factor):
    p,_,js,_,_,_,rows=workflow(); js[0][:,:10]*=factor
    for d in m.directions(): p['points'][0]['saved_Jv'][d['id']]=(js[0]@np.asarray(d['vector'])).tolist()
    result=m.analyze(p,rows,js)
    assert max(c['requested_direction_metrics']['relative_l2'] for c in result['comparisons'])>.1
    assert result['derivative_verdict']=='not_assigned' and result['P1_admission'] is False


def test_effective_float32_grid_and_zero_reference():
    p,_,js,_,_,_,rows=workflow(); result=m.analyze(p,rows,js)
    c=result['comparisons'][6]
    assert c['effective_direction_error_l2']>0
    assert np.array_equal(c['effective_float32_Jv'],js[0]@np.asarray(c['effective_direction']))
    assert m.errors(np.zeros(192),np.zeros(192))['relative_l2'] is None


def test_failure_nonrefund_order_and_no_grad():
    p,_,_=protocol(); req=m.requests(p)[0]
    def fail(): raise RuntimeError('sync')
    meter=m.Meter(lambda e:None,sync=fail); meter.start(req); token=meter.begin('action_encoder')
    with pytest.raises(RuntimeError): meter.end(token)
    assert meter.counts()['action_encoder']==dict(attempted=1,completed=0)
    with pytest.raises(ValueError): meter.begin('predict')
    with pytest.raises(ValueError): meter.start(req)
    meter=m.Meter(lambda e:None,no_grad=lambda:False); meter.start(req)
    with pytest.raises(ValueError): meter.begin('action_encoder')
    assert meter.events==[]


def test_nominal_mismatch_retained_before_guard_failure():
    p,tasks,_=protocol(); p['points'][0]['at_terminal'][0]+=.1
    log=[]; rows=[]; b=Backend(); meter=m.Meter(log.append)
    with pytest.raises(ValueError,match='nominal EXACT'):
        m.evaluate(p,tasks,lambda k:b,meter,log.append,rows)
    assert len(rows)==1 and log[-1]['kind']=='evaluation_complete' and not b.active


def test_no_third_point_or_nonzero_point_or_modified_directions():
    for change in ('third','nonzero','direction'):
        p,_,_=protocol()
        if change=='third': p['points'].append(deepcopy(p['points'][0]))
        elif change=='nonzero': p['points'][0]['at_logits'][0][0]=.1
        else: p['directions'][0]['vector'][0]=-1
        with pytest.raises(ValueError): m.requests(p)


def test_zero_floor_and_cap_remain_strict():
    with pytest.raises(ValueError): m.fd.check_space(0,1,m.FLOOR)
    with pytest.raises(ValueError): m.fd.check_space(m.CAP,1,m.FLOOR+100)
    assert m.CAP==2*1024**2 and m.FLOOR==200*1024**2


def test_startup_failure_has_terminal_report_in_RAM(monkeypatch):
    p,tasks,js=protocol(); mem={}
    class RAMStore:
        def __init__(self,output): self.output=RAMPath()
        def put(self,name,value,final=False): mem[name]=deepcopy(value)
        def event(self,e): mem.setdefault('events',[]).append(deepcopy(e))
        def close(self): mem['closed']=True
    class RAMPath:
        def iterdir(self): return []
    monkeypatch.setattr(m,'Store',RAMStore); monkeypatch.setattr(m,'verify',lambda *a:(p,tasks,js))
    monkeypatch.setitem(sys.modules,'torch',None)
    monkeypatch.setenv('HF_HUB_OFFLINE','1'); monkeypatch.setenv('TRANSFORMERS_OFFLINE','1')
    with pytest.raises(ValueError,match='partial FD preserved'): m.run('unused','unused','unused')
    r=mem['report.json']; assert r['status']=='partial_or_failed' and r['first_error']['stage']=='execution'
    assert r['resources']['predict']==dict(attempted=0,completed=0) and mem['closed']


def test_actual_RAM_build_two_first_zero_points_no_execution():
    p,tasks,js=m.collect()
    assert len(p['points'])==len(tasks)==len(js)==2
    assert p['input_report_sha256']==m.INPUT_SHA and p['numerical']==dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False)
    assert len(m.canonical(p))<m.PREPARE_CAP
    assert all(not np.asarray(v['at_logits']).any() and v['root_sha256'] is None for v in p['points'])
    assert all(str(m.ROOT/'innerdocs'/(m.NAME+'.md')) in p['bindings_sha256'] for _ in [0])


def extract_fixture():
    p,tasks,js=protocol(); point=p['points'][0]; task=tasks[0]; j=js[0]
    group='iter_1_AD'; contexts=path(task,np.zeros((10,2)))
    slot=dict(order=0,task_id='0',task_export='task_mock.json',task_sha256='a'*64)
    d=dict(task_id='0',raw_terminal_jacobian=j.tolist(),at_logits=point['at_logits'],at_terminal=point['at_terminal'])
    events=[dict(kind='evaluation_complete',evaluation_id='initial',prediction=dict(contexts=contexts,logits=point['at_logits'])) for _ in range(3)]
    events += [dict(kind='jacobian_forward_packet',evaluation_id=group,context=c) for c in contexts]
    events += [dict(kind='jacobian_row',evaluation_id=group,row=k,raw_control_derivative=v.tolist()) for k,v in enumerate(j)]
    events += [dict(kind='joint_logit_jacobian',evaluation_id=group,jacobian=j.tolist(),terminal=point['at_terminal']),
        dict(kind='jacobian_forward_complete',evaluation_id=group,terminal=point['at_terminal'],logits=point['at_logits']),
        dict(kind='derivative_complete',evaluation_id=group,derivative=d)]
    return task,slot,events


@pytest.mark.parametrize('damage',('none','nonzero','row','task','anchor','prefix','duplicate','joint'))
def test_first_zero_raw_J_and_exact_anchors(damage):
    task,slot,events=extract_fixture()
    if damage=='nonzero': events[-1]['derivative']['at_logits'][0][0]=.01
    elif damage=='row': next(e for e in events if e['kind']=='jacobian_row')['raw_control_derivative'][0]+=.1
    elif damage=='task': slot['task_id']='wrong'
    elif damage=='anchor': events[-1]['derivative']['at_terminal']=np.ones(192).tolist()
    elif damage=='prefix': next(e for e in events if e['kind']=='jacobian_forward_packet')['context']['action_history'][0][0]+=.1
    elif damage=='duplicate': events.append(deepcopy(events[-1]))
    elif damage=='joint': next(e for e in events if e['kind']=='joint_logit_jacobian')['jacobian'][0][0]+=.1
    if damage=='none':
        point,j=m.extract_point(task,slot,events)
        assert point['root_sha256'] is None and j.shape==(192,20)
    else:
        with pytest.raises(ValueError): m.extract_point(task,slot,events)
