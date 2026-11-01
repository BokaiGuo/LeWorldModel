"""RAM-only analytic mocks; no actual exports, files, torch or physics."""
from copy import deepcopy
from contextlib import nullcontext
import gzip
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import zlib
import numpy as np
import pytest
from experiments.spectral_reachability import p1_moving_no_composition_baseline_cuda_20261027_v1 as m


def task(slot=0):
    moving=m.inputs.moving
    pixels=np.stack([np.full((224,224,3),k+slot,np.uint8) for k in range(4)])
    raw=moving.fixed_raw()
    packet=dict(rgb_history=pixels[:3],raw_action_history=raw,goal_rgb=pixels[3])
    s=dict(source_slot=slot,blind_sha256=str(slot+2)*64,
        pixel_sha256=[moving.digest(x.tobytes()) for x in pixels],raw_action_sha256=moving.digest(raw.tobytes()))
    p=dict(normalization=moving.normalization(),model_bindings=moving.MODELS,arguments=dict(manifest_sha='a'*64))
    z=np.random.default_rng(17+slot).normal(size=(4,192)).astype(np.float32)
    return m.inputs.interface(moving.task_interfaces(moving.task_export(p,'b'*64,s,packet,list(z))))['gn_task']


class Backend:
    """Full chain depends on both incoming histories; finite array stand-in."""
    def __init__(self,fail=None):
        self.a=np.random.default_rng(7).normal(scale=.03,size=(192,10))
        self.calls=[];self.inputs=[];self.fail=fail
    def rollout(self,t,raw,meter,mode):
        self.calls.append((meter.group,mode));before=deepcopy(t['root_context']);path=[]
        tokens=m.core.normalized_tokens(t,raw)
        for k in (0,1):
            self.inputs.append(deepcopy(before))
            for resource in ('action_encoder','predict'):
                token=meter.begin(resource)
                if resource=='predict' and len(self.inputs)==self.fail:raise RuntimeError('mock failed predict')
                meter.end(token)
            h=np.asarray(before['action_history']);z=np.asarray(before['latent_history'])
            last=(z[-1]+self.a@raw.reshape(2,10)[k]+.0001*(h[0].sum()+2*h[1].sum())).astype(np.float32)
            before=dict(before,latent_history=before['latent_history'][1:]+[last.astype(float).tolist()],
                action_history=[before['action_history'][-1],tokens[k].astype(float).tolist()])
            path.append(before);meter.emit('native_packet',mode=mode,packet=k,context=before)
        return path
    def jacobian(self,t,nominal,meter):
        # AD-forward stand-in with same entry counts, no autograd/model.
        for k in (0,1):
            for resource in ('action_encoder','predict'):meter.call(resource,lambda:None)
            meter.emit('jacobian_forward_packet',packet=k,context=nominal['contexts'][k])
        terminal=nominal['contexts'][-1]['latent_history'][-1]
        meter.emit('jacobian_forward_complete',terminal=terminal,logits=nominal['logits'])
        j=np.concatenate([self.a+.0001*2/np.asarray(t['action_std']).repeat(5).reshape(2,5).T.reshape(10)[None,:],self.a],axis=1)
        for i,row in enumerate(j):
            meter.call('vjp_directions',lambda:None)
            meter.emit('jacobian_row',row=i,raw_control_derivative=row.tolist())
        meter.emit('joint_logit_jacobian',terminal=terminal,jacobian=j.tolist())
        return np.asarray(terminal),j


@pytest.fixture(scope='module')
def complete():
    t=task();events=[];meter=m.Meter(lambda e:events.append(deepcopy(e)));b=Backend()
    r=m.optimize(t,0,b,meter)
    return t,r,events,b


def test_frozen_budget_and_denominator(complete):
    t,r,events,b=complete
    assert r['resources']=={k:dict(attempted=v,completed=v) for k,v in m.budget().items()}
    assert len(r['candidates'])==40 and len(b.calls)==43
    assert m.contract()['total_budget']['predict']==176
    assert sum(m.contract()['calls'].values())==88
    assert m.choice(t,r,0)['commands']==r['replays'][0]['commands']
    m.journal_check(t,r,events)


def test_history_rolling(complete):
    t,r,events,b=complete
    assert b.inputs[0]==t['root_context']
    assert b.inputs[1]['action_history'][0]==t['root_context']['action_history'][-1]
    assert not np.array_equal(t['root_raw_action_history'],np.zeros((10,2)))
    assert not np.array_equal(t['root_context']['action_history'][0],t['root_context']['action_history'][1])


def test_bank_semantics_damping_and_support(complete):
    t,r,events,b=complete;bank=r['bank'];original=bank['original_joint']
    assert original['labels'][:4]==['zero','dls_scale_.25','dls_scale_.5','dls_scale_1']
    expected=m.spectral.compiler.bounded_ridge(np.asarray(r['jacobian']),np.asarray(t['goal'])-np.asarray(original['nominal']),damping=.002,cap=m.spectral.CAP)
    assert original['dls']==expected
    x=np.asarray(bank['logits']);assert x.shape==(40,2,10)
    assert (np.linalg.norm(x,axis=2)<=t['packet_logit_cap']).all()
    assert np.array_equal(np.tanh(x.astype(np.float32)).reshape(40,10,2),np.asarray(bank['commands']))
    assert not np.array_equal(x.reshape(40,20),original['logits'])
    assert len([e for e in events if e['kind']=='operation_attempted' and e['event']['resource']=='svd'])==2


@pytest.mark.parametrize('field',['winner_index','jacobian','commands','cost','history','bank'])
def test_saved_tampering_rejected(complete,field):
    t,orig,events,b=complete;r=deepcopy(orig)
    if field=='winner_index':r['winner_index']=(r['winner_index']+1)%40
    elif field=='jacobian':r['jacobian'][0][0]+=1
    elif field=='commands':r['candidates'][0]['commands'][0][0]=.4
    elif field=='cost':r['candidates'][0]['cost']+=1
    elif field=='history':r['candidates'][0]['contexts'][0]['action_history'][0][0]+=1
    else:r['bank']['logits'][1][0][0]+=.1
    with pytest.raises((ValueError,AssertionError)):m.choice(t,r,0)


@pytest.mark.parametrize('kind',['jacobian_row','native_packet','operation_completed','bank_frozen','winner_frozen_before_replay'])
def test_missing_journal_evidence_rejected(complete,kind):
    t,r,events,b=complete;e=deepcopy(events);e.pop(next(i for i,v in enumerate(e) if v['kind']==kind))
    with pytest.raises((ValueError,AssertionError)):m.journal_check(t,r,e)


def test_failed_forward_no_refund_retry_or_selection():
    t=task();events=[];meter=m.Meter(events.append);backend=Backend(fail=3)
    with pytest.raises(RuntimeError,match='mock failed'):m.optimize(t,0,backend,meter)
    assert meter.counts()['predict']==dict(attempted=5,completed=4)
    assert not any(e['kind']=='winner_frozen_before_replay' for e in events)
    with pytest.raises((ValueError,AssertionError)):meter.begin('predict')


def test_predispatch_cap():
    meter=m.Meter(lambda e:None)
    for i in range(88):meter.call('predict',lambda:None)
    with pytest.raises((ValueError,AssertionError)):meter.begin('predict')
    assert meter.counts()['predict']['attempted']==88


def test_partial_svd_failure_retains_attempt(monkeypatch):
    t=task();events=[];meter=m.Meter(events.append)
    def fail(*a,**kw):raise RuntimeError('mock SVD')
    monkeypatch.setattr(m.spectral.compiler,'bounded_ridge',fail)
    with pytest.raises(RuntimeError,match='mock SVD'):m.bank(t,np.zeros((192,20)),np.zeros(192),meter)
    assert meter.counts()['svd']==dict(attempted=1,completed=0)
    assert meter.counts()['bounded_solve']==dict(attempted=1,completed=0)
    with pytest.raises((ValueError,AssertionError)):meter.begin('svd')


def test_no_scientific_or_total_compute_claim():
    c=m.contract()
    assert not any(c[k] for k in ('P1_admission','P2_admission','equal_total_compute_claim','training'))
    assert c['new_encoder']==c['new_projector']==c['new_physics']==0
    assert c['output_cap_bytes']==8*1024**2 and c['reserve_bytes']==200*1024**2


class RAMStore(m.Store):
    def __init__(self,output='/RAM_no_composition'):
        self.output=Path(output);self.files={};self.comp=zlib.compressobj(6,zlib.DEFLATED,31)
        self.broken=self.closed=False;self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        assert sum(map(len,self.files.values()))+len(raw)<=m.CAP-(0 if final else m.FINAL)
        assert append or name not in self.files
        self.files[name]=self.files.get(name,b'')+raw


def test_two_root_own_consumer_RAM(monkeypatch,complete):
    t,r,e,_=complete;t1=task(1);e1=[];r1=m.optimize(t1,1,Backend(),m.Meter(lambda v:e1.append(deepcopy(v))))
    tasks=[t,t1];store=RAMStore();slots=[];rows=[]
    for s,(t,result,events) in enumerate(zip(tasks,[r,r1],[e,e1],strict=True)):
        slot=dict(source_slot=s,task_id=m.core.task_id(t),task_export=f'task_{s}',task_sha256=str(s)*64,blind_sha256=str(s+2)*64)
        slots.append(slot);stem=f'root_{s}';selected=m.choice(t,result,s)
        plan=m.model_plan(slot,t,selected);plan['selection_sha256']=m.core.digest(plan)
        store.put(stem+'.result.json.gz',result);store.put(stem+'.plan.json',plan)
        rows.append(dict(source_slot=s,task_id=slot['task_id'],status='model_plan_sealed',result_file=stem+'.result.json.gz',plan_file=stem+'.plan.json',selection_sha256=plan['selection_sha256'],resources=result['resources']))
        for ev in events:store.event(dict(source_slot=s,task_id=slot['task_id'],event=ev))
    store.put('binding.json',dict(prepared='ram_prepare',prepare_sha256='a'*64));store.close()
    digest=lambda raw:hashlib.sha256(raw).hexdigest()
    report=dict(schema=m.NAME+'.report',status='moving_spectral_plans_sealed_not_admitted',commands_released=True,
        errors=[],first_error=None,source_verified=True,model_unchanged=True,
        model_hash_before=m.inputs.moving.low.MODEL_TENSOR_SHA,model_hash_after=m.inputs.moving.low.MODEL_TENSOR_SHA,
        prepared='ram_prepare',prepare_sha256='a'*64,contract=m.contract(),rows=rows,
        resources={k:dict(attempted=n,completed=n) for k,n in m.contract()['total_budget'].items()},artifact_sha256={n:digest(v) for n,v in store.files.items()})
    store.put('report.json',report,final=True);files=store.files;prefix=store.output;expected=digest(files['report.json'])
    monkeypatch.setattr(m,'verify',lambda *a:dict(slots=slots,bindings_sha256={}))
    monkeypatch.setattr(m,'load_export',lambda p,h:dict(gn_task=tasks[int(p[-1])]))
    monkeypatch.setattr(m,'sha',lambda p:digest(files[Path(p).name]))
    monkeypatch.setattr(m,'read',lambda p:m.base.decode(gzip.decompress(files[Path(p).name]) if str(p).endswith('.gz') else files[Path(p).name]))
    monkeypatch.setattr(Path,'iterdir',lambda p:iter(prefix/n for n in files))
    monkeypatch.setattr(Path,'read_bytes',lambda p:files[p.name])
    consumed=m.consume(prefix,expected)
    assert len(consumed['rows'])==2 and consumed['resources']==report['resources']
    assert sum(map(len,files.values()))<m.CAP
    files['root_0.plan.json']+=b' '
    with pytest.raises(ValueError,match='artifact SHA'):m.consume(prefix,expected)


def test_startup_error_preserves_final_report_RAM(monkeypatch):
    store=RAMStore();p=dict(slots=[dict(source_slot=s,task_id='a'*64) for s in (0,1)],contract=m.contract())
    monkeypatch.setattr(m,'verify',lambda *a:p);monkeypatch.setattr(m,'Store',lambda out:store)
    monkeypatch.setitem(sys.modules,'torch',NS())
    monkeypatch.setitem(sys.modules,'torch.nn.attention',NS(SDPBackend=NS(MATH=1),sdpa_kernel=lambda *a:nullcontext()))
    monkeypatch.setitem(sys.modules,'experiments.checkpoint_compat',NS(load_compatible_pretrained=lambda *a,**kw:pytest.fail('must not load model')))
    def fail(*a):raise RuntimeError('mock allocation denied')
    monkeypatch.setitem(sys.modules,'experiments.gpu_preflight',NS(require_cuda_allocation=fail))
    monkeypatch.setattr(Path,'iterdir',lambda p:iter([]))
    with pytest.raises(ValueError,match='partial retained'):m.run('ram_prepare','a'*64,str(store.output))
    r=m.base.decode(store.files['report.json'])
    assert r['first_error']['message']=='mock allocation denied'
    assert not r['commands_released'] and r['status']=='partial_or_failed'
    assert all(x['status']=='not_started' for x in r['rows'])


def test_sync_error_no_false_completion():
    def fail():raise RuntimeError('mock sync')
    meter=m.Meter(lambda e:None,fail);token=meter.begin('predict')
    with pytest.raises(RuntimeError):meter.end(token)
    assert meter.counts()['predict']==dict(attempted=1,completed=0)


def test_own_prepare_build_bridge_RAM_only(monkeypatch):
    calls=[]
    def bridge(arms):
        calls.append(arms)
        return dict(schema='old_bridge.prepare',arguments=dict(arms=list(arms)),slots=[],policy='mock',
            expected_model_hash='a'*64,contract={},bindings_sha256={})
    monkeypatch.setattr(m.inputs,'build',bridge)
    expected={str(Path(m.inputs.__file__)):'f30a1c5ac5ef618cfe3add37326aae29513278fe7c4319b5a743d9f6422300e6',
        str(Path(m.spectral.__file__)):'6e1402ac8abe3aabfffefd24aa016c341e6eb4f40d3f60e233152460ca70e45a',
        str(Path(m.spectral.compiler.__file__)):'f359e7b21d1970f95f065dbee206d2dc9359c2eaff7ddbcd7611e07686c6cf2a'}
    monkeypatch.setattr(m,'sha',lambda p:expected.get(str(Path(p)),'b'*64))
    p=m.build()
    assert calls==[('CEM',)] and p['schema']==m.NAME+'.prepare'
    assert p['contract']==m.contract() and 'arguments' not in p
    assert str(Path(m.__file__).resolve()) in p['bindings_sha256']
    assert len(m.canonical(p))<=256*1024
    assert str(Path(m.spectral.__file__).resolve()) in p['bindings_sha256']


def test_original_joint_primitive_drift_fails_before_input_bridge(monkeypatch):
    monkeypatch.setattr(m,'sha',lambda p:'0'*64)
    monkeypatch.setattr(m.inputs,'build',lambda *a:pytest.fail('must reject before bridge'))
    with pytest.raises(ValueError,match='approved immutable'):m.build()


def test_zero_only_history_rejected():
    t=task();t['root_raw_action_history']=np.zeros((10,2)).tolist()
    t['root_context']['action_history']=m.core.normalized_tokens(t,np.zeros((10,2))).astype(float).tolist()
    value=dict(gn_task=t,d2p_packet=dict(root_context=t['root_context'],goal=t['goal']),q=t['projection'],task_id=m.core.task_id(t))
    with pytest.raises(ValueError,match='own nonzero'):m.inputs.interface(value)
