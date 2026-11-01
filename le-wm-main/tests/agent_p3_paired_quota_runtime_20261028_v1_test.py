"""RAM-only numerical wiring/fullarray tests; no model/CUDA/physics execution."""
from copy import deepcopy
from functools import lru_cache
import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_paired_quota_runtime_20261028_v1 as m

def load_test(stem):
    spec=importlib.util.spec_from_file_location('_paired_RAM_'+stem,m.ROOT/f'tests/{stem}_test.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

fixtures=load_test('agent_p3_child_program_cuda_20261028_v1')

def prepared(observed=1):
    source,_=fixtures.source_fixture(observed=observed)
    return dict(wiring=dict(tasks=source['tasks'],children=source['children'],limits=m.pair.limits()),
        profile=m.profile(),sources={},expected_model_hash='a'*64)

class RAMFactories:
    kind='RAM_FIXTURE'
    def __init__(self,geometry='short',fault=False):self.geometry=geometry;self.fault=fault;self.calls=[]
    def signal(self,task,child,sink):
        self.calls.append(('signal',child['child_id']))
        task=deepcopy(task);task['root_context']=deepcopy(child['context'])
        return RamSignal(task,sink,self.geometry,self.fault)
    def program(self,task,child,sink):
        self.calls.append(('compile',child['child_id']))
        return fixtures.RamReplay(task,sink)

@lru_cache(maxsize=3)
def panel(geometry='short'):
    p=prepared();events=[];session=m.pair.Session(p,events.append);backend=RAMFactories(geometry)
    before=dict(original_state_dict=p['expected_model_hash'],all_buffers='b'*64)
    events.extend([dict(kind='denominator',rows=m.initial_rows(p),fixed_denominator=48,child_denominator=24,
        sources_sha256=m.digest(p['sources'])),dict(kind='model_before',hashes=before)])
    for _ in range(24):session.next_pair(backend,permit_ram=True)
    events.extend([dict(kind='model_after',hashes=before),dict(kind='sources_final',sha256=m.digest(p['sources']))])
    report=dict(model_before=before,model_after=before,session=session.receipt(),journal_tail='f'*64)
    return p,report,events,backend.calls

def iter_ram(events):
    for e in events:yield e,'f'*64

class RamSignal:
    """Paid array arithmetic fixture, NOT a model or a CUDA emulation claim."""
    kind='RAM_FIXTURE'
    def __init__(self,t,sink,geometry='short',fault=False):
        self.task,self.sink,self.geometry,self.fault=t,sink,geometry,fault
    def signal(self,req,ledger):
        original=ledger.scope;ledger.scope=dict(original,horizon=req['horizon'])
        try:return self.response(req,ledger)
        finally:ledger.scope=original


    def response(self, req, ledger):
        t = self.task; h = req['horizon']; phase = step = None
        def emit(kind, operation_index=None, **arrays):
            self.sink.event(dict(kind=kind, scope=deepcopy(ledger.scope), phase=phase, step=step,
                operation_index=operation_index, arrays={k: m.base.blob(np.asarray(v, np.float32)) for k, v in arrays.items()}))
        def call(resource, fn, inputs, shape=None, directions=None):
            index = len(ledger.events)
            def actual():
                emit('io_input', operation_index=index, **{resource+':'+k: v for k, v in inputs.items()})
                out = fn(); emit('io_return', operation_index=index, **{resource+':output': out}); return out
            return ledger.call(resource, actual, shape=shape, directions=directions)
        last = np.zeros(192, np.float32)
        for phase in ('nominal', 'AD'):
            latent = np.asarray(t['root_context']['latent_history'], np.float32)[None]
            hist = np.asarray(t['root_context']['action_history'], np.float32)[None]
            mean, std = (np.asarray(t[k], np.float32) for k in ('action_mean', 'action_std'))
            emit('origin', latent=latent, history=hist, raw_flat=np.zeros(10*h, np.float32), mean=mean, std=std)
            # Nest AD forward inside the Jacobian charge, as the real adapter does.
            def forward():
                nonlocal latent, hist, step
                for step in range(h):
                    raw = np.zeros((5, 2), np.float32); token = ((raw-mean)/std).reshape(10)
                    outgoing = np.concatenate((hist, token[None, None]), axis=1)
                    emit('packet', raw=raw, token=token, outgoing=outgoing)
                    ae = call('action_encoder', lambda: np.zeros((1, 3, 192), np.float32), {'0': outgoing}, [1, 3, 10])
                    def predict():
                        pred = call('predictor', lambda: np.zeros((1, 3, 192), np.float32), {'0': latent, '1': ae}, [1, 3, 192])
                        proj = call('pred_proj', lambda: pred.reshape(3, 192), {'0': pred.reshape(3, 192)}, [3, 192])
                        return proj.reshape(1, 3, 192)
                    predicted = call('predict', predict, {'latent': latent, 'act_emb': ae}, [1, 3, 192])
                    latent = np.concatenate((latent[:, 1:], predicted[:, -1:]), axis=1); hist = outgoing[:, 1:]
                    emit('boundary', latent=latent, history=hist)
                return latent[0, -1]
            if phase == 'nominal':
                last = forward(); emit('nominal', value=last)
            else:
                def ad():
                    nonlocal step
                    forward(); step = None; emit('AD_terminal', value=last)
                    if self.fault: raise RuntimeError('paid AD forward RAM fault')
                    rows = []
                    for direction in range(192):
                        self.sink.event(dict(kind='direction', scope=deepcopy(ledger.scope), direction=direction))
                        raw_grad = np.zeros(10*h, np.float32)
                        if self.geometry!='refuse':
                            if direction==0:raw_grad[0]=1.
                            if self.geometry=='long' and h==1 and direction==1:raw_grad[0]=.5
                        elif h==3 and direction==0:raw_grad[0]=1.
                        def grad():
                            def derivative():
                                def raw_hook():
                                    emit('raw_backward', operation_index=ledger.events[-1]['index'], value=raw_grad)
                                    return raw_grad
                                return ledger.call('raw_backward', raw_hook)
                            return call('grad', derivative, {'output': last, 'raw': np.zeros(10*h, np.float32),
                                'basis': np.eye(192, dtype=np.float32)[direction]})
                        rows.append(ledger.call('vjp', grad, directions=1))
                    return np.stack(rows)
                jac = ledger.call('jacobian', ad); emit('jacobian', value=jac)
        return dict(binding_sha256=req['binding_sha256'], parent_context_sha256=req['parent_context_sha256'],
            horizon=h, nominal=last.tolist(), jacobian=jac.tolist(), producer_sha256=m.sha(m.paid.__file__))


@pytest.mark.parametrize('geometry,h,n',[('short',1,12),('long',3,4)])
def test_real_RAM_fullarray_pair_signal_barrier_generator_and_consumer(geometry,h,n):
    p,r,events,calls=panel(geometry)
    census,actual=m.validate_stream(iter_ram(events),p,r)
    assert census==dict(compiled=2,paired_paid_refusal=0,not_reached=46)
    assert calls[:4]==[('signal',p['wiring']['children'][0]['child_id'])]*2+[('compile',p['wiring']['children'][0]['child_id'])]*2
    rows=r['session']['rows'];assert rows[1]['decision']['horizon']==h
    assert rows[1]['decision']['candidates']==4 and rows[1]['execution_quota']['candidates']==n
    for j,row in enumerate(rows[:2]):
        program=row['compiler'];count=12 if j==0 else n
        assert len(program['candidates'])==len(program['bank']['actions'])==count
        assert program['bank']['fractions']==[i/(count-1) for i in range(count)]
        assert program['resource_receipt']['completed']['predict_API']==25
        assert program['resource_receipt']['completed']['metric_calls']==count
        assert program['resource_receipt']['completed']['replay_calls']==2*count+1
        assert row['signal_resources']['completed']['predict_API']==8
    assert actual['fixed']['attempted']['predict_API']==actual['adaptive']['attempted']['predict_API']==33
    assert actual['fixed']['completed']['predict_samples']==actual['adaptive']['completed']['predict_samples']==33
    if geometry=='long':assert actual['fixed']!=actual['adaptive'] # replay/metric are different units
    seals=[e for e in events if e['kind']=='work' and e['payload']['kind']=='budget_gate' and e['payload']['payload']['kind']=='pair_sealed']
    seal_index=events.index(seals[0])
    assert len([e for e in events[:seal_index] if e['kind']=='signal_return'])==2
    assert not any(e['kind']=='work' and e['payload'].get('entry',{}).get('resource')=='program_compile' for e in events[:seal_index])


def test_pair_refusal_preserves_both_independent_payments_no_subset():
    p,r,events,calls=panel('refuse');census,actual=m.validate_stream(iter_ram(events),p,r)
    assert census==dict(compiled=0,paired_paid_refusal=2,not_reached=46)
    assert len(calls)==2 and all(c[0]=='signal' for c in calls)
    fixed,adaptive=r['session']['rows'][:2]
    assert adaptive['decision']['candidates']==8 and adaptive['decision']['horizon']==3
    assert adaptive['execution_quota']['candidates']==4 and not adaptive['execution_quota']['allowed']
    assert fixed['execution_quota']['allowed'] and fixed['status']=='paired_paid_refusal'
    assert fixed['compiler'] is adaptive['compiler'] is None
    assert actual['fixed']==actual['adaptive']
    assert actual['fixed']['completed']['predict_API']==8
    assert actual['fixed']['completed']['jacobian_API']==2
    assert actual['fixed']['completed']['VJP_directions']==384


@pytest.mark.parametrize('fault',['AD'])
def test_failure_nonrefundable_failstop_before_sibling_signal(fault):
    p=prepared();events=[];session=m.pair.Session(p,events.append);f=RAMFactories(fault=True)
    with pytest.raises(RuntimeError,match='paid AD'):session.next_pair(f,permit_ram=True)
    assert session.stopped and session.gate.failed and session.gate.pending
    assert session.rows[0]['signal_resources']['attempted']['predict_API']==2
    assert len(session.rows)==48 and session.rows[1]['compiler'] is None
    before=len(f.calls)
    with pytest.raises(ValueError):session.next_pair(f,permit_ram=True)
    assert len(f.calls)==before


@pytest.mark.parametrize('axis',m.core.AXES)
def test_canonical_caps_cannot_be_lowered_or_increased(axis):
    p=prepared()['wiring'];p=deepcopy(p)
    p['limits']['common_node_cap'][axis]+=1
    with pytest.raises(ValueError,match='canonical'):m.pair.Gate(p)


@pytest.mark.parametrize('shape',[[1,3,191],[2,3,192],[1,2,192]])
def test_actual_bad_shape_predispatch_no_callable(shape):
    p=prepared();g=m.pair.Gate(p['wiring']);g.open_pair();called=[]
    sink=m.native.GateSink(g,g.key('fixed'),lambda e:None)
    ledger=m.core.Ledger(m.contract.signal_caps(1),sink=sink.event);ledger.scope={'RAM':True}
    with pytest.raises(ValueError):ledger.call('predict',lambda:called.append(1),shape=shape)
    assert not called and g.actual['fixed']['attempted']['predict_API']==0
    if shape==[1,2,192]:
        # core.delta rejects this malformed context BEFORE reaching the gate;
        # no paid gate ticket may be invented for that unentered operation.
        assert not g.failed and not g.pending
        with pytest.raises(ValueError):g.begin(g.key('fixed'),'predict',shape=shape)
    assert g.failed and g.failure_ticket
    with pytest.raises(ValueError,match='fail-stop'):g.begin(g.key('fixed'),'predict',shape=[1,3,192])


@pytest.mark.parametrize('fault',['ungated','late_gate','early_return','drop_signal','drop_seal','wrong_slot',
    'drop_pair','extra','footer','raw','J','history','selected','candidate','decision','quota','source'])
def test_full_paired_saved_negative_tamper(fault):
    p,r,source,_=panel('long');events=list(source);r=deepcopy(r)
    def change(kind,predicate=lambda e:True):
        index=next(i for i,e in enumerate(events) if e['kind']==kind and predicate(e))
        events[index]=deepcopy(events[index]);return index,events[index]
    if fault in ('ungated','late_gate','early_return'):
        i,_=change('work',lambda e:e['payload']['kind']==('resource_return' if fault=='early_return' else 'resource_attempt'))
        if fault=='ungated':del events[i-1]
        elif fault=='late_gate':events[i-1],events[i]=events[i],events[i-1]
        else:events[i],events[i+1]=events[i+1],events[i]
    elif fault=='drop_signal':events.pop(next(i for i,e in enumerate(events) if e['kind']=='signal_return'))
    elif fault=='drop_seal':events=[e for e in events if not(e['kind']=='work' and e['payload']['kind']=='budget_gate' and e['payload']['payload']['kind']=='pair_sealed')]
    elif fault=='wrong_slot':change('work')[1]['slot']=3
    elif fault=='drop_pair':events.pop(next(i for i,e in enumerate(events) if e['kind']=='pair_return'))
    elif fault=='extra':events.insert(3,dict(kind='fake'))
    elif fault=='footer':events=events[:-1]
    elif fault in ('raw','history'):
        _,e=change('work',lambda e:e['payload']['kind']=='program_packet')
        key='raw' if fault=='raw' else 'history';e['payload']['arrays'][key]=m.base.blob(np.zeros(e['payload']['arrays'][key]['shape'],np.float32))
        if fault=='raw':e['payload']['arrays'][key]=m.base.blob(np.ones((5,2),np.float32))
    elif fault=='J':change('signal_return')[1]['row']['signal']['responses'][0]['jacobian'][0][0]=3.
    elif fault=='source':change('signal_return')[1]['row']['signal']['responses'][0]['producer_sha256']='0'*64
    elif fault=='decision':change('signal_return')[1]['row']['decision']['candidates']=12
    elif fault=='quota':change('compile_return')[1]['row']['execution_quota']['candidates']=4
    elif fault=='selected':change('compile_return')[1]['row']['compiler']['selected']['index']=99
    elif fault=='candidate':change('compile_return')[1]['row']['compiler']['candidates'].pop()
    with pytest.raises((ValueError,KeyError,IndexError)):m.validate_stream(iter_ram(events),p,r)


def test_run_unauthorized_before_verify_and_torch(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('verify called'))
    with pytest.raises(ValueError,match='authorization'):m.run('/bad','0'*64,'/bad',None)


def test_real_saved_ABI_material_projection_and_exact_source(monkeypatch):
    original=m.base.read(m.UPSTREAM,64*1024**2);calls=[]
    # Exercise real saved input fields without repeating MAIN full historical consume.
    def verify(path,h):
        calls.append((path,h));assert str(path)==str(m.UPSTREAM) and h==m.UPSTREAM_SHA
        return original
    monkeypatch.setattr(m.host,'verify',verify)
    value=m.material(str(m.UPSTREAM),m.UPSTREAM_SHA)
    assert calls==[(str(m.UPSTREAM),m.UPSTREAM_SHA)] and len(value['wiring']['children'])==24
    assert value['wiring']['tasks']==original['wiring']['tasks']
    assert value['wiring']['limits']==m.pair.limits()
    assert all(str(f) in value['sources'] for f in m.own_files())
    assert value['profile']['output_bytes']==512*1024**2
    assert m.pair.envelope()['predict_API']==33
    assert m.pair.limits()['common_method_cap']['predict_API']==792
    with pytest.raises(ValueError):m.material(str(m.UPSTREAM),'0'*64)


def test_CLI_real_entrypoints_no_model_import():
    out=subprocess.run([sys.executable,'-m','experiments.spectral_reachability.'+m.NAME,'--help'],cwd=m.ROOT,capture_output=True,text=True)
    assert out.returncode==0 and all(s in out.stdout for s in ('prepare','verify','run','consume'))


def test_exclusive_output_and_bounded_writer(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    p=tmp_path/'results'/(m.NAME+'_RAM.json')
    with pytest.raises(ValueError,match='bounded'):m.write_new(str(p),{'x':'z'*100},cap=10)
    assert not p.exists()
    saved=m.write_new(str(p),{'RAM_only':True})
    assert saved['sha256']==m.sha(p)
    with pytest.raises(ValueError,match='NEW'):m.write_new(str(p),{})


def test_saved_RAM_artifacts_full_strict_report_closed_and_tamper(tmp_path):
    p,original,events,_=panel('long');r=deepcopy(original)
    out=tmp_path/'RAM_native_arrays_NOT_CUDA';store=m.paid.Store(out,p['profile'])
    store.put('binding.json',dict(prepared='/RAM/prepare.json',prepare_sha256='c'*64))
    for e in events:store.event(e)
    store.close()
    r.update(schema=m.NAME+'.report',status='complete_native_panel_not_admitted',prepared='/RAM/prepare.json',
        prepare_sha256='c'*64,errors=[],host=dict(device='cuda:0',index=0,name='RAM_FIXTURE_NOT_EVIDENCE',allocation_sum=4.,
            torch_version='RAM',torch_cuda='RAM',visible_nvidia_nodes=['RAM']),CUDA_peak_bytes=[0,0],
        sources_verified=True,cleanup_ok=True,source_manifest_sha256=m.digest(p['sources']),wall_seconds=1.,
        journal_tail=store.tail,artifact_sha256={n:m.sha(out/n) for n in ('binding.json',m.paid.JOURNAL)},**m.FLAGS)
    store.put('report.json',r);expected=m.sha(out/'report.json')
    receipt=m.validate_report(out,expected,p,r)
    assert receipt['census']==dict(compiled=2,paired_paid_refusal=0,not_reached=46)
    assert len(receipt['full_rows'])==48 and receipt['saved_verification']['signal_array_SVD_calls']==4
    assert receipt['actual_vectors_equal'] is False and receipt['FLOPs'] is None
    assert all(receipt[k] is False for k in m.FLAGS)
    for k,v in [('errors',['fake']),('status','engineering_failed'),('P3_admission',True)]:
        bad=dict(r);bad[k]=v
        with pytest.raises(ValueError):m.validate_report(out,expected,p,bad)
    with pytest.raises(ValueError,match='artifact'):m.validate_report(out,expected,p,dict(r,artifact_sha256={**r['artifact_sha256'],m.paid.JOURNAL:'0'*64}))


def test_capacity_refusal_before_neural_import(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'exclusive',lambda p:tmp_path/'uncreated')
    monkeypatch.setattr(m,'verify',lambda *a:prepared())
    monkeypatch.setattr(m.shutil,'disk_usage',lambda *a:SimpleNamespace(free=0))
    with pytest.raises(ValueError,match='reservation'):m.run('/RAM','a'*64,'/RAM',m.AUTH)
    assert not (tmp_path/'uncreated').exists()


def test_journal_fault_retains_paid_attempt_and_blocks_strategy_switch():
    p=prepared();g=m.pair.Gate(p['wiring']);g.open_pair()
    def broken(e):raise OSError('RAM disk failure')
    sink=m.native.GateSink(g,g.key('fixed'),broken)
    ledger=m.core.Ledger(m.contract.signal_caps(1),sink=sink.event);ledger.scope={'RAM':True};called=[]
    with pytest.raises(OSError):ledger.call('predict',lambda:called.append(True),shape=[1,3,192])
    assert called==[] and g.failed and g.pending
    assert g.actual['fixed']['attempted']['predict_API']==1 and g.actual['fixed']['completed']['predict_API']==0
    with pytest.raises(ValueError):g.begin(g.key('adaptive'),'predict',shape=[1,3,192])


