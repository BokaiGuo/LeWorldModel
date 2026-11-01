"""RAM-only array/kernel/wiring/failure tests. No torch, model or simulator."""
from copy import deepcopy
import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zlib
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_blind_planners_cuda_20261027_v1 as m


def helper(stem):
    spec=importlib.util.spec_from_file_location(stem,Path(__file__).with_name(stem+'.py'))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


g=helper('test_p1_line_search_gn_cuda_20261027_v1')
d=helper('test_p3_goal_patch_consistent_compiler_20261027_v2')


def inputs(seed=17):
    t=g.task(seed); t.update(environment='swm/PushT-v1',
        model_sha256=m.encode.MODELS[str(m.encode.POLICY/'weights.pt')],
        model_config_sha256=m.encode.MODELS[str(m.encode.POLICY/'config.json')],
        action_mean=m.d2p.ACTION_MEAN.astype(float).tolist(),action_std=m.d2p.ACTION_STD.astype(float).tolist(),
        goal=np.asarray(t['goal'],np.float32).astype(float).tolist())
    t['root_context']['action_history']=m.encode.core.normalized_tokens(t,t['root_raw_action_history']).astype(float).tolist()
    return dict(gn_task=t,d2p_packet=dict(root_context=deepcopy(t['root_context']),goal=t['goal']),q=t['projection'],task_id=m.encode.core.task_id(t))


class RAMStore(m.Store):
    def __init__(self):
        self.files={}; self.comp=zlib.compressobj(6,zlib.DEFLATED,31); self.closed=self.broken=False
        self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        assert append or name not in self.files
        assert sum(map(len,self.files.values()))+len(raw)<=m.CAP-(0 if final else m.FINAL)
        self.files[name]=self.files.get(name,b'')+raw


class DBackend(d.LinearBackend):
    def __init__(self,ledger): super().__init__(); self.ledger=ledger
    def step(self,c,a):
        for resource in ('action_encoder','predict'):
            e=self.ledger.begin(resource); self.ledger.end(e)
        return super().step(c,a)


def run_mock(arm,i,sink,row):
    if arm==m.ARMS[0]:
        ledger=m.gn.Meter(sink); r=m.gn.optimize(i['gn_task'],g.Backend(),ledger)
    else:
        ledger=m.D2PLedger(sink); r=m.d2p.compile_arm(i['d2p_packet'],i['q'],DBackend(ledger),ledger.journal)
    row['resources']=ledger.counts()
    # This stand-in does not execute AD. Mock expected API receipts are labelled
    # synthetic here; actual runtime uses ADTrace around real CUDA APIs.
    row['AD_API_resources']={k:dict(attempted=v,completed=v) for k,v in m.contract(1)['AD_API_per_root'][arm].items()}
    return r


@pytest.fixture(scope='module')
def full():
    i=m.interface(inputs()); results={}; events={}; rows={}
    for arm in m.ARMS:
        events[arm]=[]; rows[arm]={}; results[arm]=run_mock(arm,i,events[arm].append,rows[arm])
    return i,results,events,rows


def test_both_real_kernels_accept_identical_new_interface(full):
    i,results,events,rows=full
    for arm in m.ARMS:
        m.exact_counts(rows[arm]['resources'],m.budgets()[arm])
        chosen=m.selected(arm,i,results[arm]); assert chosen['task_id']==i['task_id']
        assert np.asarray(chosen['commands']).shape==(10,2)
    assert len(results[m.ARMS[0]]['steps'])==4
    assert len(results[m.ARMS[1]]['candidates'])==10
    assert sum(e['kind']=='jacobian_row' for e in events[m.ARMS[0]])==768


def test_real_encoder_export_public_interface_in_RAM():
    images=np.zeros((4,224,224,3),np.uint8); raw=np.zeros((10,2),np.float32)
    packet=dict(rgb_history=images[:3],goal_rgb=images[3],raw_action_history=raw)
    slot=dict(source_slot=0,blind_sha256='1'*64,pixel_sha256=[m.encode.digest(x.tobytes()) for x in images],raw_action_sha256=m.encode.digest(raw.tobytes()))
    p=dict(normalization=dict(mean=m.d2p.ACTION_MEAN.astype(float).tolist(),std=m.d2p.ACTION_STD.astype(float).tolist()),
        arguments=dict(whitelist_sha='2'*64),model_bindings=m.encode.MODELS)
    export=m.encode.task_export(p,'3'*64,slot,packet,[np.full(192,k,np.float32) for k in range(4)])
    i=m.interface(m.encode.task_interfaces(export))
    assert set(i)=={'gn_task','d2p_packet','q','task_id'}
    assert set(i['d2p_packet'])=={'root_context','goal'}
    assert i['gn_task']['root_context']['latent_history'][2]==[2.]*192


@pytest.mark.parametrize('change',[
    lambda i:i['d2p_packet']['goal'].__setitem__(0,99),
    lambda i:i.__setitem__('task_id','a'*64),
    lambda i:i['gn_task'].__setitem__('action_mean',[0.,0.]),
    lambda i:i['gn_task'].__setitem__('model_sha256','b'*64),
    lambda i:i['gn_task']['root_context'].__setitem__('action_history',np.zeros((2,10)).tolist()),
    lambda i:i['q'][0].__setitem__(0,0),
])
def test_interface_rejects_task_q_goal_normalizer_drift(change):
    i=deepcopy(inputs()); change(i)
    with pytest.raises(ValueError): m.interface(i)


def test_only_final_loader_for_arrays_no_native_identity_join(monkeypatch):
    called=[]; i=inputs()
    monkeypatch.setattr(m,'sha',lambda path:'f'*64)
    monkeypatch.setattr(m,'read',lambda *args:{})
    monkeypatch.setattr(m,'validate_export_envelope',lambda e:None)
    monkeypatch.setattr(m.encode,'load_task',lambda path,h:(called.append((str(path),h)) or i))
    result=m.load_export('/opaque/task_00.json.gz','f'*64)
    assert called==[('/opaque/task_00.json.gz','f'*64)] and set(result)==set(i)
    with pytest.raises(ValueError): m.load_export('/opaque/sourcepack.json','f'*64)
    with pytest.raises(ValueError): m.load_export('/opaque/task_00.json.gz','e'*64)


def test_no_old_material_or_source_loader_called(monkeypatch):
    def denied(*a,**kw): raise AssertionError('old/source workflow reached')
    for obj,k in ((m.gn,'material'),(m.gn,'verify'),(m.gn.saved,'material'),(m.d2p,'source_inputs'),
                  (m.encode,'build'),(m.encode,'whitelist'),(m.encode,'load_blind')):
        monkeypatch.setattr(obj,k,denied)
    assert m.interface(inputs())['task_id']


@pytest.mark.parametrize('arm',m.ARMS)
def test_selected_commands_and_scope_tamper_rejected(full,arm):
    i,results,_,_=full; r=deepcopy(results[arm])
    if arm==m.ARMS[0]: r['final_prediction']['commands'][0][0]+=.001
    else: r['winner']['commands'][0][0]+=.001
    with pytest.raises(ValueError): m.selected(arm,i,r)


def test_nonwinner_score_tamper_rejected(full):
    i,results,_,_=full; r=deepcopy(results[m.ARMS[1]])
    r['candidates'][0]['objective']+=1
    with pytest.raises(ValueError): m.selected(m.ARMS[1],i,r)


def test_math_attempts_before_dispatch_and_no_retry():
    events=[]; l=m.D2PLedger(events.append)
    e=dict(kind='jacobian',index=0,completed=False,shape=[192,10],mathematical_vjp_directions=192)
    l.journal(e); assert l.counts()['jacobian']==dict(attempted=1,completed=0)
    l.journal(dict(e,error='failed'))
    with pytest.raises(ValueError): l.journal(dict(kind='svd',index=1,completed=False))
    assert len(l.math)==1


def test_d2p_native_cap_partial_and_sync_failure():
    l=m.D2PLedger(lambda e:None)
    for _ in range(66): l.end(l.begin('predict'))
    with pytest.raises(ValueError): l.begin('predict')
    l=m.D2PLedger(lambda e:None,sync=lambda:(_ for _ in ()).throw(RuntimeError('async')))
    token=l.begin('predict')
    with pytest.raises(RuntimeError): l.end(token)
    assert l.counts()['predict']==dict(attempted=1,completed=0)
    with pytest.raises(ValueError): l.begin('predict')


def fake_torch():
    def f(*a,**kw): return 'returned'
    return SimpleNamespace(autograd=SimpleNamespace(grad=f,backward=f,functional=SimpleNamespace(**{k:f for k in ('jacobian','hessian','jvp','vjp','hvp','vhp')})),
        Tensor=SimpleNamespace(backward=f),func=SimpleNamespace(**{k:f for k in ('jvp','vjp','jacrev','jacfwd','hessian','grad','grad_and_value')}))


def test_AD_API_actual_nested_count_and_restore():
    t=fake_torch(); old=t.autograd.grad
    t.autograd.functional.jacobian=lambda *a,**kw:t.autograd.grad(is_grads_batched=True)
    ad=m.ADTrace(t,m.ARMS[1],lambda e:None,lambda:None)
    with ad:
        assert t.autograd.functional.jacobian()=='returned'
    assert t.autograd.grad is old
    assert ad.counts()=={k:dict(attempted=1,completed=1) for k in ('grad','jacobian')}
    assert ad.events[1]['is_grads_batched'] is True


@pytest.mark.parametrize('arm',m.ARMS)
def test_unbudgeted_AD_fails_closed_and_restores(arm):
    t=fake_torch(); old=t.Tensor.backward
    with pytest.raises(ValueError):
        with m.ADTrace(t,arm,lambda e:None,lambda:None): t.Tensor.backward()
    assert t.Tensor.backward is old


def test_GN_forbids_extra_functional_jacobian():
    t=fake_torch(); ad=m.ADTrace(t,m.ARMS[0],lambda e:None,lambda:None)
    with ad:
        with pytest.raises(ValueError): t.autograd.functional.jacobian()
    assert not ad.events


def test_gzip_partial_final_trailing_and_broken_store():
    store=RAMStore(); store.event(dict(kind='attempt'))
    raw=store.files['events.jsonl.gz']; plain,eof=m.inflate(raw)
    assert not eof and json.loads(plain)==dict(kind='attempt')
    store.close(); plain,eof=m.inflate(store.files['events.jsonl.gz']); assert eof
    with pytest.raises(ValueError): m.inflate(store.files['events.jsonl.gz']+b'trailing')
    with pytest.raises(ValueError): store.event({})
    store=RAMStore(); store._write=lambda *a,**kw:(_ for _ in ()).throw(OSError('disk'))
    with pytest.raises(OSError): store.event({})
    assert store.broken
    with pytest.raises(ValueError): store.close()


@pytest.mark.parametrize('raw',[b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}'])
def test_ambiguous_json_rejected(raw):
    with pytest.raises(ValueError): m.decode(raw)


def test_prepare_has_no_equal_budget_or_hidden_root_contract():
    c=m.contract(2)
    assert c['total_budget'][m.ARMS[0]]['predict']==156
    assert c['total_budget'][m.ARMS[1]]['predict']==132
    assert c['ADforward_in_predict_per_root']==dict(zip(m.ARMS,(8,11)))
    assert c['equal_budget_claim'] is c['P1_admission'] is False
    assert c['new_images']==c['new_physics']==0


def test_plan_all_opaque_slot_join_model_commands_sealed(full,monkeypatch):
    i,results,_,receipts=full; p=dict(slots=[dict(order=0,task_export='/opaque/task_00.json.gz',task_sha256='a'*64,task_id=i['task_id'])])
    monkeypatch.setattr(m,'load_export',lambda *a:deepcopy(i))
    rows=[dict(order=0,arm=a,status='not_started') for a in m.ARMS]; store=RAMStore()
    def runner(a,i,sink,row): row.update(deepcopy(receipts[a])); return deepcopy(results[a])
    m.plan_all(p,store,runner,rows); store.close()
    assert all(r['status']=='model_plan_sealed' for r in rows)
    for a in m.ARMS:
        plan=json.loads(store.files[f'root_0_{a}.plan.json'])
        identity=plan.pop('selection_sha256'); assert identity==m.encode.core.digest(plan)
        assert not {'native_root_sha256','source_slot','witness','certificate'} & set(plan)
        assert plan['status']=='selected_pending_global_closure' and plan['physical_execution'] is False
    assert sum(map(len,store.files.values()))<m.CAP


def test_first_failure_does_not_start_other_arm(monkeypatch):
    i=inputs(); monkeypatch.setattr(m,'load_export',lambda *a:i)
    p=dict(slots=[dict(order=0,task_export='/opaque/task_00.json.gz',task_sha256='a'*64,task_id=i['task_id'])])
    rows=[dict(order=0,arm=a,status='not_started') for a in m.ARMS]; calls=[]
    def fail(a,*args): calls.append(a); raise RuntimeError('first')
    with pytest.raises(RuntimeError): m.plan_all(p,RAMStore(),fail,rows)
    assert calls==[m.ARMS[0]] and rows[0]['status']=='attempted' and rows[1]['status']=='not_started'


@pytest.mark.parametrize('index,expected,task_id',[
    (0,'714f471761278fa654542d4bb0878ce18e3c304df6158005af46d2129533265a','ac82e87557962f3db47b39634410362fddc455cd85c50ca3e8f923fc29264ee8'),
    (1,'70865d56f9b66f0072b2ad7de01cc09d9a9275047925e188104a13d5182ce1f2','05ad3266de1b02051c132bae16d24e2ab41b0938d979bf363f75f0937ce9cdd7')])
def test_MAIN_actual_saved_export_only_no_model(index,expected,task_id):
    path=m.ROOT/'results/p1_constructive_blind_encode_cuda_20261027_v1_cuda_main01'/f'task_{index:02d}.json.gz'
    i=m.load_export(path,expected)
    assert i['task_id']==task_id and set(i)=={'gn_task','d2p_packet','q','task_id'}
    assert set(i['d2p_packet'])=={'root_context','goal'}
    assert i['gn_task']['root_context']['action_history'][0][0]!=0


@pytest.mark.parametrize('key,value',[('source_slot',True),('source_slot',5),('native_root_sha256','a'*64),('with_target',True)])
def test_strict_actual_envelope_rejects_identity_contract_drift(key,value):
    p=m.ROOT/'results/p1_constructive_blind_encode_cuda_20261027_v1_cuda_main01/task_00.json.gz'
    e=m.read(p); e[key]=value
    with pytest.raises(ValueError): m.validate_export_envelope(e)


def test_strict_actual_envelope_rejects_unknown_privileged_field():
    p=m.ROOT/'results/p1_constructive_blind_encode_cuda_20261027_v1_cuda_main01/task_00.json.gz'
    e=m.read(p); e['witness']=[0.]*20
    with pytest.raises(ValueError): m.validate_export_envelope(e)
