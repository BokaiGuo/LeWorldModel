"""RAM/NumPy protocol fixtures only. NO torch model, CUDA, simulator or tmp files."""
from contextlib import contextmanager
from copy import deepcopy
import gzip
from pathlib import Path
import sys
import zlib
import numpy as np
import pytest
from experiments.spectral_reachability import moving_history_budget_scheduler_cuda_20261027_v1 as m


class RAM:
    def __init__(self):self.events=[];self.files={}
    def event(self,e):self.events.append(deepcopy(e))
    def put(self,name,value,final=False):self.files[name]=deepcopy(value)
    def close(self):self.closed=True


def task(slot=0):
    raw=m.moving.fixed_raw();mean=[.01,-.02];std=[.03,.04]
    t=dict(action_mean=mean,action_std=std,projection=np.eye(192).tolist(),goal=np.ones(192).tolist())
    t['root_context']=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=(np.arange(576,dtype=np.float32).reshape(3,192)/1000+np.float32(slot/10)).astype(float).tolist(),
        action_history=m.core.normalized_tokens(t,raw).astype(float).tolist())
    return t


def banks():
    anchors=[]
    for v in (.05,-.05):
        logits=np.full((2,10),v,np.float32)
        anchors.append(dict(logits=logits.astype(float).tolist(),commands=np.tanh(logits).reshape(10,2).astype(float).tolist()))
    return [m.bank(s,anchors) for s in (0,1)]


def prepared():
    return dict(slots=[dict(source_slot=s,root_sha256=str(s)*64,task_id=str(s+2)*64,blind_sha256=str(s+4)*64,
        task_export=f'/mock/task_{s}.json.gz',task_sha256=str(s+6)*64,task_contract_sha256='a'*64,bank_sha256='b'*64,anchors=[]) for s in (0,1)],
        expected_model_hash='c'*64,policy='/mock/never_load',protocol=m.protocol())


class Backend:
    """Array-only shape-faithful ledger fixture, NOT predictive/AD evidence."""
    def __init__(self,slot=0,fail=None):self.slot=slot;self.fail=fail;self.n=0
    @contextmanager
    def guard(self,meter):yield
    def step(self,t,before,raw,meter):
        want=m.moving.incoming(t,before,raw);meter.emit('context_input',context=want)
        z=np.asarray(want['latent_history'],np.float32)[None];tokens=np.asarray(want['action_history'],np.float32)[None]
        embedded=np.tile(tokens.mean(axis=-1,keepdims=True),(1,1,192)).astype(np.float32)
        predicted=z.copy();predicted[:,-1]+=np.float32(np.asarray(raw,np.float32).sum()/100)
        def module(resource,inputs,out):
            e=meter.begin(resource);meter.emit('module_input',entry=e['index'],resource=resource,arrays=[m.blob(a) for a in inputs])
            meter.emit('module_output',entry=e['index'],resource=resource,arrays=m.blob(out));meter.end(e)
        module('action_encoder',[tokens],embedded)
        self.n+=1
        api=meter.begin('predict')
        if self.fail==self.n:raise ValueError('mock predict failure')
        module('predictor',[z,embedded],predicted);module('pred_proj',[predicted.reshape(3,192)],predicted.reshape(3,192))
        meter.end(api);meter.emit('predict_output',array=m.blob(predicted))
        return dict(before,latent_history=before['latent_history'][1:]+[predicted[0,-1].astype(float).tolist()],action_history=want['action_history'][-2:])
    def rollout(self,t,raw,meter,mode):
        before=deepcopy(t['root_context']);out=[]
        for k in range(2):
            before=self.step(t,before,np.asarray(raw,np.float32)[5*k:5*k+5],meter);out.append(before)
            meter.emit('native_packet',mode=mode,packet=k,context=before)
        return out
    def jacobian(self,t,incumbent,meter):
        paths=self.rollout(t,incumbent['commands'],meter,'mock_AD_forward')
        assert paths==incumbent['contexts']
        j=np.zeros((192,20),np.float64)
        for i in range(20):j[i,i]=(1+self.slot)
        for index in range(192):
            def backward():
                meter.call('raw_backward_hook',lambda:meter.emit('raw_backward',array=m.blob(j[index].reshape(10,2).astype(np.float32))))
                return j[index]
            row=meter.call('vjp_directions',lambda:meter.call('grad_api',backward))
            meter.emit('jacobian_row',row=index,array=m.blob(row))
        return np.asarray(paths[-1]['latent_history'][-1]),j


@pytest.fixture(scope='module')
def complete():
    p=prepared();ts=[task(s) for s in (0,1)];bs=banks();ram=RAM();meter=m.Meter(ram.event,m.fidelity.Limits())
    result=m.census(p);m.panel(p,ts,bs,[Backend(s) for s in (0,1)],meter,result)
    result['paired_summary']=m.summary(result)
    return p,ts,bs,ram,meter,result


def test_full_four_policy_ram_panel_and_saved_consume(complete):
    p,ts,bs,ram,meter,result=complete
    assert meter.counts()=={k:dict(attempted=v,completed=v) for k,v in m.MAXIMA.items()}
    plans=m.saved_check(p,ts,bs,result,ram.events)
    assert len(plans)==8
    assert [(x['arm'],x['source_slot']) for x in plans]==[(a,s) for a in m.METHODS for s in (0,1)]
    assert all(not x['P1_admission'] and not x['P3_admission'] for x in plans)
    assert m.protocol()['processed_action_context_tokens']==1560


@pytest.mark.parametrize('weights,expected,swapped', [([0,0],[10,10],[10,10]),([1,0],[15,5],[5,15]),([0,1],[5,15],[15,5]),([1,1],[10,10],[10,10]),([3,1],[13,7],[8,12])])
def test_geometry_rounding_and_swap(weights,expected,swapped):
    assert m.allocate(m.METHODS[1],weights)==expected
    assert m.allocate(m.METHODS[2],weights)==swapped


def test_swap_half_up_tie_exact_formula():
    assert m.allocate(m.METHODS[2],[3,1])==[8,12]


@pytest.mark.parametrize('method', [m.METHODS[0],m.METHODS[3]])
def test_uniform_cannot_receive_signal(method):
    assert m.allocate(method)==[10,10]
    with pytest.raises(ValueError):m.allocate(method,[0,0])


@pytest.mark.parametrize('w', [[-1,1],[float('nan'),0],[float('inf'),1],[1,2,3]])
def test_bad_weights_reject(w):
    with pytest.raises(ValueError):m.allocate(m.METHODS[1],w)


@pytest.mark.parametrize('b',[[5,15],[15,5],[10,10],[13,7]])
def test_schedule_floor_and_monotonic_prefix(b):
    order=m.schedule(b)
    assert order[:10]==[0,1]*5
    assert [order.count(s) for s in (0,1)]==b


def test_rng_independent_reference_and_domain():
    bb=banks();rng=np.random.Generator(np.random.PCG64(20261027))
    for index in range(3,45):
        ll=[]
        for _ in range(2):
            v=rng.standard_normal(10);u=rng.random();ll.append(v/np.sqrt(np.dot(v,v))*(.75*np.sqrt(5))*u**.1)
        ll=np.asarray(ll,np.float32)
        assert np.array_equal(bb[0][index]['logits'],ll)
        assert np.array_equal(bb[0][index]['commands'],np.tanh(ll).reshape(10,2))
    assert bb[0][3]!=bb[1][3]
    for bank in bb:
        assert len(bank)==45
        for c in bank:m.legal(c['commands'],c['logits'])


def test_anchor_literal_no_repair():
    bb=banks()[0]
    assert bb[1]['kind']==m.moving.ARMS[0] and bb[2]['kind']==m.moving.ARMS[1]
    bad=deepcopy(bb[1]);bad['logits'][0][0]=3
    with pytest.raises(ValueError):m.bank(0,[bad,bb[2]])
    bad=deepcopy(bb[1]);bad['commands'][0][0]=0
    with pytest.raises(ValueError):m.bank(0,[bad,bb[2]])


@pytest.mark.parametrize('goal,j,chi,w', [(0,0,0,0),(1,0,1,0),(1,1,172/192,20)])
def test_signal_zero_residual_rank_and_projection(goal,j,chi,w):
    t=task();t['goal']=[float(goal)]*192;jac=np.zeros((192,20));jac[:20]=np.eye(20)*j
    s=m.signal(t,np.zeros(192),jac)
    assert s['chi']==pytest.approx(chi);assert s['w']==pytest.approx(w)


def test_signal_bad_svd_no_tuning():
    def broken(*a,**k):raise np.linalg.LinAlgError('mock SVD failure')
    with pytest.raises(np.linalg.LinAlgError):m.signal(task(),np.zeros(192),np.zeros((192,20)),svd=broken)
    with pytest.raises(ValueError):m.signal(task(),np.zeros(192),np.full((192,20),np.nan))


def test_tie_stable_and_nonzero_normalized_history(complete):
    p,ts,bs,ram,meter,result=complete
    for t in ts:assert np.any(t['root_context']['action_history'])
    assert min([dict(cost=1,index=5),dict(cost=1,index=0)],key=lambda c:(c['cost'],c['index']))['index']==0
    for r in result['rows']:
        w=min(r['candidates'],key=lambda c:(c['cost'],c['index']))
        assert r['plan']['candidate_index']==w['index']


@pytest.mark.parametrize('mutation',['raw','winner','token','latent','count','module','jacobian','cost','prediction','freeze','scope'])
def test_saved_consumer_rejects_tamper(complete,mutation):
    p,ts,bs,ram,meter,base=complete;result=deepcopy(base);events=deepcopy(ram.events)
    if mutation=='raw':result['rows'][0]['candidates'][0]['commands'][0][0]=.1
    elif mutation=='winner':result['rows'][0]['winner_index']+=1
    elif mutation=='token':next(e for e in events if e['kind']=='context_input')['context']['action_history'][0][0]+=1
    elif mutation=='latent':result['rows'][0]['replays']['whole_replay'][0]['latent_history'][0][0]+=1
    elif mutation=='count':events.pop(next(i for i,e in enumerate(events) if e['kind']=='resource_return'))
    elif mutation=='module':events.pop(next(i for i,e in enumerate(events) if e['kind']=='module_output'))
    elif mutation=='jacobian':result['signals'][0]['jacobian']['raw_sha256']='0'*64
    elif mutation=='cost':result['rows'][0]['candidates'][0]['cost']+=1
    elif mutation=='prediction':
        e=next(e for e in events if e['kind']=='predict_output');a=m.unblob(e['array']);a[0,-1,0]+=np.float32(1);e['array']=m.blob(a)
    elif mutation=='freeze':next(e for e in events if e['kind']=='winner_frozen')['index']+=1
    else:next(e for e in events if e['kind']=='resource_attempt')['entry']['scope'][1]=1
    with pytest.raises((ValueError,AssertionError)):m.saved_check(p,ts,bs,result,events)


def test_attempted_predict_failure_keeps_full_denominator_and_partial_arrays():
    p=prepared();ram=RAM();meter=m.Meter(ram.event,m.fidelity.Limits());result=m.census(p)
    with pytest.raises(ValueError,match='mock predict failure'):
        m.panel(p,[task(0),task(1)],banks(),[Backend(0,fail=2),Backend(1)],meter,result)
    assert len(result['rows'])==8 and all(r['status']=='not_started' for r in result['rows'])
    assert meter.counts()['predict']==dict(attempted=2,completed=1)
    assert len([e for e in ram.events if e['kind']=='native_packet'])==1
    assert len([e for e in ram.events if e['kind']=='context_input'])==2


def test_candidate_failure_preserves_prefix_arrays_and_no_replacement():
    p=prepared();ram=RAM();meter=m.Meter(ram.event,m.fidelity.Limits());result=m.census(p)
    with pytest.raises(ValueError,match='mock predict failure'):
        m.panel(p,[task(0),task(1)],banks(),[Backend(0,fail=6),Backend(1)],meter,result)
    assert all(x['status']=='complete' for x in result['signals'])
    assert len(result['rows'])==8 and result['rows'][0]['status']=='attempted'
    assert len(result['rows'][0]['candidates'])==1 and result['rows'][0]['candidates'][0]['status']=='attempted'
    assert all(r['status']=='not_started' for r in result['rows'][1:])
    assert meter.counts()['predict']==dict(attempted=10,completed=9)
    prefixes=[e for e in ram.events if e['kind']=='native_packet' and e['scope']==[m.METHODS[0],0]]
    assert len(prefixes)==1 and prefixes[0]['packet']==0


def test_shared_math_cap_and_method_panel_cap():
    meter=m.Meter(RAM().event,m.fidelity.Limits());meter.scope=['shared',0]
    meter.call('jacobian',lambda:None)
    with pytest.raises(ValueError,match='math cap'):meter.begin('jacobian')
    meter.scope=[m.METHODS[1],0]
    for _ in range(94):meter.call('predict',lambda:None)
    meter.scope=[m.METHODS[1],1]
    for _ in range(34):meter.call('predict',lambda:None)
    with pytest.raises(ValueError,match='method cap'):meter.begin('predict')


def test_nonfinite_svd_outputs_reject():
    def nonfinite(*args,**kw):return np.zeros((192,20)),np.full(20,np.inf),np.zeros((20,20))
    with pytest.raises(ValueError):m.signal(task(),np.zeros(192),np.zeros((192,20)),svd=nonfinite)


def test_fixed_wall_and_disk_stops():
    ticks=[0.];limits=m.fidelity.Limits(clock=lambda:ticks[0]);ticks[0]=600
    with pytest.raises(ValueError,match='wall600'):limits.check()
    with pytest.raises(ValueError,match='200MiB'):m.fidelity.Limits(free=lambda:m.FLOOR-1).check()
    with pytest.raises(ValueError,match='allocated3GiB'):m.fidelity.Limits(memory=lambda:(3*1024**3+1,0)).check()


def test_global_root_method_caps_predispatch():
    ram=RAM();meter=m.Meter(ram.event,m.fidelity.Limits());meter.scope=[m.METHODS[0],0]
    for _ in range(64):meter.call('predict',lambda:None)
    with pytest.raises(ValueError,match='root cap'):meter.begin('predict')
    assert meter.counts()['predict']['attempted']==64


def test_no_math_inside_N():
    meter=m.Meter(RAM().event,m.fidelity.Limits());meter.scope=[m.METHODS[3],0]
    with pytest.raises(ValueError,match='never uses'):meter.begin('svd')


def test_blob_native_precision_three_shapes_and_tamper():
    rng=np.random.default_rng(4)
    for shape in ((192,20),(1,3,192),(1,3,10)):
        for dtype in (np.float32,np.float64):
            a=rng.standard_normal(shape).astype(dtype);b=m.blob(a)
            assert np.array_equal(a,m.unblob(b)) and m.unblob(b).dtype==dtype
            b['raw_sha256']='0'*64
            with pytest.raises(ValueError):m.unblob(b)


def test_ram_full_precision_output_bound_synthetic(complete):
    # Full independent random native traces replace the compressible mock tensors.
    # RAM gzip only; never create a tmp/prepare/results directory.
    _,_,bs,ram,_,result=complete;rng=np.random.default_rng(20261027);comp=zlib.compressobj(6,zlib.DEFLATED,31);size=0
    for event in ram.events:
        event=deepcopy(event)
        def replace(value):
            if isinstance(value,dict):
                if set(value)=={'shape','dtype','raw_sha256','bytes_b64'}:
                    return m.blob(rng.standard_normal(value['shape']).astype(value['dtype']))
                return {k:replace(v) for k,v in value.items()}
            if isinstance(value,list):return [replace(v) for v in value]
            return value
        event=replace(event)
        size+=len(comp.compress(m.canonical(event))+comp.flush(zlib.Z_SYNC_FLUSH))
    size+=len(comp.flush(zlib.Z_FINISH))
    size+=sum(len(gzip.compress(m.canonical(a),mtime=0)) for a in (result,bs))
    assert size<m.CAP-m.FINAL
    print('RAM full-precision random trace + panel/banks compressed bytes:',size)


def test_main_help_and_dispatch_RAM(monkeypatch,capsys):
    monkeypatch.setattr(sys,'argv',[m.NAME,'--help'])
    with pytest.raises(SystemExit) as e:m.main()
    assert e.value.code==0 and 'consume' in capsys.readouterr().out
    monkeypatch.setattr(m,'prepare',lambda output,**args:dict(output=output,arguments=args))
    monkeypatch.setattr(sys,'argv',[m.NAME,'prepare','--output','/RAM/never_written','--handoff-sha','a','--anchor-report-sha','b','--encoder-report-sha','c'])
    m.main();assert 'never_written' in capsys.readouterr().out


@pytest.mark.parametrize('command',['verify','run','consume'])
def test_cli_all_other_modes_RAM_dispatch(monkeypatch,capsys,command):
    monkeypatch.setattr(m,'verify',lambda *args:(dict(slots=[0,1]),[],[]))
    monkeypatch.setattr(m,'run',lambda *args:dict(status='RAM_run_not_model'))
    monkeypatch.setattr(m,'consume',lambda *args:dict(status='RAM_consume'))
    monkeypatch.setattr(sys,'argv',[m.NAME,command,'--prepared','/RAM/prepare','--prepare-sha','a'*64,'--output','/RAM/output','--report-sha','b'*64])
    m.main();assert 'status' in capsys.readouterr().out


def test_own_consume_full_RAM_artifacts(complete,monkeypatch):
    p,ts,bs,ram,meter,result=complete;p=deepcopy(p);p['bindings_sha256']={}
    report_sha='f'*64;names={'binding.json','banks.json.gz','panel.json.gz','events.jsonl.gz'}
    report=dict(schema=m.NAME+'.report',status='four_policy_commands_sealed_not_admitted',errors=[],first_error=None,
        commands_released=True,model_unchanged=True,source_verified=True,prepared='/RAM/prepare',prepare_sha256='e'*64,
        protocol=m.protocol(),model_hash_before=p['expected_model_hash'],model_hash_after=p['expected_model_hash'],
        resources=meter.counts(),paired_summary=result['paired_summary'],artifact_sha256={n:'d'*64 for n in names},
        scope_resources={g:{str(s):meter.counts(g,s) for s in (0,1)} for g in ('shared',)+m.METHODS},
        new_visual_entries=0,new_visual_attempts=[],new_physics=0,**m.FLAGS)
    values={'report.json':report,'binding.json':dict(prepared='/RAM/prepare',prepare_sha256='e'*64),'banks.json.gz':bs,'panel.json.gz':result}
    monkeypatch.setattr(m,'verify',lambda *args:(p,ts,bs))
    monkeypatch.setattr(m,'sha',lambda path:report_sha if Path(path).name=='report.json' else 'd'*64)
    monkeypatch.setattr(m,'read',lambda path,*args:deepcopy(values[Path(path).name]))
    monkeypatch.setattr(Path,'iterdir',lambda self:iter([self/n for n in names|{'report.json'}]))
    journal=gzip.compress(b''.join(m.canonical(e) for e in ram.events),mtime=0)
    monkeypatch.setattr(Path,'read_bytes',lambda self:journal)
    consumed=m.consume('/RAM/output',report_sha)
    assert len(consumed['plans'])==8 and consumed['status']=='four_policy_commands_verified_not_admitted'
    assert all(p['whole_run_report_sha256']==report_sha for p in consumed['plans'])
    report['status']='partial_or_failed'
    with pytest.raises(ValueError,match='partial release'):m.consume('/RAM/output',report_sha)


def test_run_final_errors_do_not_replace_execution_primary(monkeypatch):
    ram=RAM();p=prepared();calls=[]
    def verify(*args):
        calls.append(1)
        if len(calls)>1:raise ValueError('source disappeared')
        return p,[task(0),task(1)],banks()
    monkeypatch.setattr(m,'verify',verify);monkeypatch.setattr(m,'Store',lambda output:ram)
    # Stop at import, never instantiate/import a neural model in this test.
    monkeypatch.setitem(sys.modules,'torch',None)
    with pytest.raises(ValueError,match='partial preserved'):m.run('/RAM/prepare','a'*64,'/tmp/never_created')
    report=ram.files['report.json']
    assert report['errors'][0]['stage']=='execution' and report['errors'][1]['stage']=='final_source'
    assert report['first_error']==report['errors'][0] and not report['commands_released']
    assert len(ram.files['panel.json.gz']['rows'])==8
    assert all(r['status']=='blocked_not_dispatched' for r in ram.files['panel.json.gz']['rows'])


def test_fixed_contract_no_gate_or_total_compute_claim():
    c=m.protocol();assert c['actual_maxima']['predict']==8+4*(120+8)==520
    assert c['standalone_logical_predict'][m.METHODS[3]]==128
    assert all(c[k] is False for k in m.FLAGS)
    assert c['new_solve']==c['new_encoder']==c['new_physics']==0
