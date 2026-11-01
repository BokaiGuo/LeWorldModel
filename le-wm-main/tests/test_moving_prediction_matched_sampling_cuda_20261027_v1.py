"""RAM-only callbacks/arrays, no actual prepare, files, Torch, model or physics."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import zlib
import numpy as np
import pytest
from experiments.spectral_reachability import moving_prediction_matched_sampling_cuda_20261027_v1 as m


def task(slot=0):
    moving=m.old.moving
    packet=dict(rgb_history=np.stack([np.full((224,224,3),i+slot,np.uint8) for i in range(3)]),
        raw_action_history=moving.fixed_raw(),goal_rgb=np.full((224,224,3),9+slot,np.uint8))
    s=dict(source_slot=slot,blind_sha256=str(slot+2)*64,
        pixel_sha256=[moving.digest(x.tobytes()) for x in [*packet['rgb_history'],packet['goal_rgb']]],
        raw_action_sha256=moving.digest(packet['raw_action_history'].tobytes()))
    p=dict(normalization=moving.normalization(),model_bindings=moving.MODELS,arguments=dict(manifest_sha='a'*64))
    z=np.random.default_rng(17+slot).normal(size=(4,192)).astype(np.float32)
    export=moving.task_export(p,'b'*64,s,packet,list(z))
    return m.old.interface(moving.task_interfaces(export))['gn_task']


def receipt(a):
    a=np.asarray(a,np.float32)
    return dict(shape=list(a.shape),dtype='float32',values=a.astype(float).tolist())


class Backend:
    """Array-only dynamics sensitive to both incoming action tokens."""
    def __init__(self,fail_packet=None):self.calls=0;self.fail_packet=fail_packet
    def rollout(self,t,raw,meter,mode):
        self.calls+=1;previous=deepcopy(t['root_context']);tokens=m.core.normalized_tokens(t,raw);path=[]
        for k in (0,1):
            meter.packet=k;meter.available()
            z=np.asarray(previous['latent_history'],np.float32)[None]
            outgoing=np.concatenate((previous['action_history'],tokens[k:k+1]))[None].astype(np.float32)
            a=meter.call('action_encoder',[receipt(outgoing)],lambda:outgoing.copy(),receipt)
            def api():
                if self.fail_packet==(self.calls,k):raise RuntimeError('injected API failure')
                def predict():
                    out=z.copy();out[:,-1]+=(a.sum()*.0001+raw.reshape(2,10)[k].sum()*.01)
                    return out
                pred=meter.call('predictor',[receipt(z),receipt(a)],predict,receipt)
                return meter.call('pred_proj',[receipt(pred.reshape(3,192))],lambda:pred.reshape(3,192).copy(),receipt).reshape(1,3,192)
            out=meter.call('predict_API',[receipt(z),receipt(a)],api,receipt)
            previous=dict(previous,latent_history=previous['latent_history'][1:]+[out[0,-1].astype(float).tolist()],
                action_history=[previous['action_history'][-1],tokens[k].astype(float).tolist()])
            path.append(previous);meter.emit('native_packet',mode=mode,packet=k,context=previous)
        return path


@pytest.fixture(scope='module')
def full():
    rows={}
    for s in (0,1):
        t=task(s)
        for a in m.ARMS:
            events=[];meter=m.Meter(events.append);b=Backend()
            result=m.optimize(t,s,a,b,meter)
            rows[s,a]=(t,result,events,meter.counts(),b.calls)
    return rows


def test_fixed_fourgroups_and_counts(full):
    assert len(full)==4 and m.contract()['total']==m.budget(576)
    for (s,a),(t,r,events,counts,calls) in full.items():
        assert counts==m.budget() and calls==72 and len(r['candidates'])==72
        assert m.journal_check(t,r,s,a,events)==m.choice(t,r,s,a)[0]
        assert [c['iteration'] for c in r['candidates']]==sum(([i]*24 for i in range(3)),[])
        assert [e['kind'] for e in events].count('evaluation_request')==72
        assert [e['kind'] for e in events].count('native_packet')==144
        assert events[-1]['kind']=='winner_sealed_no_replay'


def test_boundaries_and_nonclaims():
    c=m.contract()
    assert c['seeds']=={'CEM':[20261028,20261129],m.ARMS[1]:[20261029,20261130]}
    assert c['extra_replay']==c['padding']==c['nominal_evaluations']==0
    assert c['inherited_encoding_once']==dict(encoder=dict(attempted=8,completed=8),projector=dict(attempted=8,completed=8))
    assert not any(c[k] for k in ('equal_FLOPs','equal_total_compute','equal_AD_compute','P1_admission','independent_confirmation'))
    assert c['output_cap_bytes']==64*1024**2 and c['floor_bytes']==200*1024**2 and c['wall_seconds']==900


@pytest.mark.parametrize('s,a',[(s,a) for s in (0,1) for a in m.ARMS])
def test_no_final_replay_or_unscored_nominal(full,s,a):
    t,r,events,counts,calls=full[s,a];selected,_=m.choice(t,r,s,a)
    idx=min(range(72),key=lambda i:(r['candidates'][i]['cost'],i))
    assert r['winner_index']==idx
    assert selected['commands']==r['candidates'][idx]['commands']
    assert selected['prediction']==r['candidates'][idx]['prediction']['contexts'][-1]
    assert sum(e['kind']=='bank_frozen' for e in events)==3


def test_cem_elite4_ties_and_std():
    chunk=[dict(index=i,cost=1.,logits=np.full((2,10),i).tolist()) for i in range(24)]
    mean,std,d=m.update(m.ARMS[0],chunk,2.)
    assert d==dict(elite_indices=[0,1,2,3]) and np.array_equal(mean,np.full((2,10),1.5))
    assert np.array_equal(std,np.ones((2,10)))
    for c in chunk:c['logits']=np.zeros((2,10)).tolist()
    assert np.array_equal(m.update(m.ARMS[0],chunk,2.)[1],np.full((2,10),.05))


def test_mppi_weights_parameter_not_projected_logits():
    chunk=[dict(index=i,cost=float(i),parameters=np.full((2,10),.8 if i%2 else -.8).tolist(),
        logits=np.full((2,10),0.).tolist()) for i in range(24)]
    mean,std,d=m.update(m.ARMS[1],chunk,2.)
    weights=m.old.nominal.kernels.softmax_weights(np.arange(24,dtype=float),temperature=1.)
    assert np.array_equal(mean,np.tensordot(weights,np.asarray([c['parameters'] for c in chunk]),axes=(0,0)))
    assert not np.array_equal(mean,np.zeros((2,10))) and d['weights']==weights.tolist()
    assert np.array_equal(std,np.full((2,10),.5))


@pytest.mark.parametrize('field', ['seed','source_slot','arm','winner_index','updates','candidates'])
def test_corrupt_result_rejected(full,field):
    t,r,_,_,_=full[0,m.ARMS[0]];r=deepcopy(r)
    if field in ('seed','source_slot','winner_index'):r[field]=-1
    elif field=='arm':r[field]='unknown'
    elif field=='updates':r[field][0]['mean'][0][0]+=.1
    else:r[field]=r[field][:-1]
    with pytest.raises((ValueError,IndexError)):m.choice(t,r,0,m.ARMS[0])


@pytest.mark.parametrize('field',['logits','commands','cost','terminal','parameters','prediction'])
def test_corrupt_candidate_rejected(full,field):
    t,r,_,_,_=full[0,m.ARMS[1]];r=deepcopy(r);c=r['candidates'][0]
    if field in ('logits','commands','parameters'):c[field][0][0]+=.01
    elif field=='cost':c[field]+=.01
    elif field=='terminal':c[field][0]+=.01
    else:c[field]['contexts'][1]['action_history'][0][0]+=.01
    with pytest.raises(ValueError):m.choice(t,r,0,m.ARMS[1])


@pytest.mark.parametrize('change',['missing','extra','batch','pending','history','output','position'])
def test_corrupt_journal_rejected(full,change):
    t,r,ev,_,_=full[0,m.ARMS[0]];events=deepcopy(ev)
    i=next(i for i,e in enumerate(events) if e['kind']=='dispatch_attempted')
    if change=='missing':events.pop(i)
    elif change=='extra':events.insert(i,deepcopy(events[i]))
    elif change=='batch':events[i]['receipt']['batch_size']=2
    elif change=='pending':
        j=next(j for j,e in enumerate(events) if e['kind']=='dispatch_returned');events[j]['receipt']['status']='pending'
    elif change=='history':events[i]['receipt']['inputs'][0]['values'][0][0][0]+=.01
    elif change=='output':
        j=next(j for j,e in enumerate(events) if e['kind']=='dispatch_returned');events[j]['receipt']['output']['values'][0][0][0]+=.01
    else:events.insert(0,events.pop(i))
    with pytest.raises(ValueError):m.journal_check(t,r,0,m.ARMS[0],events)


def test_failure_charged_no_retry():
    events=[];meter=m.Meter(events.append);b=Backend(fail_packet=(1,0))
    with pytest.raises(RuntimeError,match='injected'):m.optimize(task(),0,m.ARMS[0],b,meter)
    c=meter.counts()
    assert c['predict_API']==dict(attempted=1,completed=0)
    assert c['action_encoder']==dict(attempted=1,completed=1)
    assert c['predictor']==dict(attempted=0,completed=0)
    assert b.calls==1 and not any(e['kind']=='winner_sealed_no_replay' for e in events)
    with pytest.raises(ValueError):meter.available()


def test_predispatch_145_refused_no_callback():
    meter=m.Meter(lambda e:None);calls=[]
    for i in range(144):meter.call('action_encoder',[],lambda:calls.append(i))
    with pytest.raises(ValueError):meter.call('action_encoder',[],lambda:calls.append(999))
    assert len(calls)==144 and meter.counts()['action_encoder']==dict(attempted=144,completed=144)


def test_parent_must_outlive_children():
    meter=m.Meter(lambda e:None);api=meter.begin('predict_API',[]);child=meter.begin('predictor',[])
    with pytest.raises(ValueError):meter.end(api,None)
    meter.end(child,None);meter.end(api,None)
    with pytest.raises(ValueError):meter.end(api,None)


def test_sink_failure_poison_before_dispatch():
    def sink(e):raise OSError('journal')
    meter=m.Meter(sink);calls=[]
    with pytest.raises(OSError):meter.call('action_encoder',[],lambda:calls.append(1))
    assert not calls and meter.counts()['action_encoder']['attempted']==1
    with pytest.raises(ValueError):meter.available()


def test_synchronize_failure_not_completed():
    def sync():raise RuntimeError('cuda synchronization fixture')
    meter=m.Meter(lambda e:None,sync)
    with pytest.raises(RuntimeError):meter.call('action_encoder',[],lambda:1)
    assert meter.counts()['action_encoder']==dict(attempted=1,completed=0)


def test_wall_prevents_dispatch():
    def check():raise ValueError('wall')
    meter=m.Meter(lambda e:None,check=check);calls=[]
    with pytest.raises(ValueError):meter.call('action_encoder',[],lambda:calls.append(1))
    assert not calls and meter.counts()==m.budget(0)


def test_literal_receipts_exact_float32():
    assert m.literal(receipt(np.ones((1,3,192))), (1,3,192)).shape==(1,3,192)
    with pytest.raises(ValueError):m.literal(receipt(np.zeros(3)),(1,3))
    v=receipt(np.zeros(3));v['values'][0]=.1
    with pytest.raises(ValueError):m.literal(v)


class RAMStore(m.Store):
    def __init__(self):
        self.output=Path('/RAM_matched_sampling');self.files={};self.started=m.time.monotonic()
        self.comp=zlib.compressobj(6,zlib.DEFLATED,31);self.closed=self.broken=False
        self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        assert sum(map(len,self.files.values()))+len(raw)<=m.CAP-(0 if final else m.FINAL)
        assert append or name not in self.files
        self.files[name]=self.files.get(name,b'')+raw


def test_store_lossless_ram_only():
    store=RAMStore();e=dict(kind='candidate',values=list(range(100)))
    store.event(e);store.put('candidate.json.gz',e);store.close()
    raw,eof=m.base.inflate(store.files['events.jsonl.gz'])
    assert eof and m.base.decode(raw)==e
    assert m.base.decode(zlib.decompress(store.files['candidate.json.gz'],31))==e


def test_plan_all_fail_keeps_four_denominators(monkeypatch):
    t=task();slots=[dict(source_slot=s,task_id=m.core.task_id(t),task_export=str(s),task_sha256=str(s)*64,
        blind_sha256='c'*64,task_contract_sha256='d'*64) for s in (0,1)]
    p=dict(slots=slots);rows=m.initial_rows(p);store=RAMStore()
    monkeypatch.setattr(m.old,'load_export',lambda path,h:dict(gn_task=t))
    def runner(t,s,a,sink,row):raise RuntimeError('arm0 fails')
    with pytest.raises(RuntimeError):m.plan_all(p,store,runner,rows)
    assert len(rows)==4 and [r['status'] for r in rows]==['attempted','not_started','not_started','not_started']


def test_build_has_new_identity_and_complete_bindings(monkeypatch):
    original=dict(slots=[dict(source_slot=i) for i in range(2)],bindings_sha256={'old':'a'*64},contract={},arguments={'arms':list(m.ARMS)})
    monkeypatch.setattr(m.old,'build',lambda:deepcopy(original))
    def hashes(path):
        if str(path)==m.old.__file__:return m.OLD_SHA
        if Path(path)==m.OFFICIAL:return m.OFFICIAL_SHA
        return 'a'*64
    monkeypatch.setattr(m,'sha',hashes)
    p=m.build()
    assert p['schema']==m.NAME+'.prepare' and p['contract']==m.contract() and p['arguments']=={}
    assert len(p['bindings_sha256'])==5 and original['contract']=={}


def test_native_backend_methods_are_own_no_globals_patch():
    assert m.NativeCuda._step is not m.gn.screen.NativeCuda._step
    assert m.NativeCuda.guard is not m.gn.screen.NativeCuda.guard
    assert m.Store._write is not m.base.Store._write
    assert m.old.CAP==8*1024**2 and m.CAP==64*1024**2


@pytest.mark.parametrize('resource,shapes',[
    ('action_encoder',[(2,3,10)]),('action_encoder',[(1,2,10)]),
    ('predictor',[(2,3,192),(2,3,10)]),('predictor',[(1,3,192),(1,2,10)]),
    ('pred_proj',[(1,3,192)]),('pred_proj',[(6,192)]),('unknown',[(3,192)])])
def test_illegal_module_shape_refused_before_dispatch(resource,shapes):
    with pytest.raises(ValueError):m.module_shapes(resource,shapes)


def test_actual_module_hook_seam_and_cleanup_ram():
    class Tensor:
        def __init__(self,a):self.a=np.asarray(a,np.float32);self.shape=self.a.shape
        def detach(self):return self
        def cpu(self):return self
        def numpy(self):return self.a
    class Module:
        def __init__(self,output):self.output=output;self.pre=[];self.post=[];self.calls=0
        def register_forward_pre_hook(self,fn):
            self.pre.append(fn);return NS(remove=lambda:self.pre.remove(fn))
        def register_forward_hook(self,fn,always_call):
            assert always_call;self.post.append(fn);return NS(remove=lambda:self.post.remove(fn))
        def __call__(self,*args):
            for f in self.pre:f(self,args)
            self.calls+=1
            for f in self.post:f(self,args,self.output)
            return self.output
    z=Tensor(np.ones((1,3,192)));a=Tensor(np.ones((1,3,10)));flat=Tensor(np.ones((3,192)))
    model=NS(action_encoder=Module(a),predictor=Module(z),pred_proj=Module(flat),encoder=Module(z),projector=Module(z))
    b=m.NativeCuda.__new__(m.NativeCuda);b.model=model
    events=[];meter=m.Meter(events.append);meter.packet=0;meter.group='candidate_0'
    with b.guard(meter):
        act=model.action_encoder(a)
        def api():model.predictor(z,act);model.pred_proj(flat);return z
        meter.call('predict_API',[m.tensor_receipt(z),m.tensor_receipt(act)],api,m.tensor_receipt)
        with pytest.raises(ValueError,match='image'):model.encoder(z)
        with pytest.raises(ValueError,match='shape'):model.action_encoder(Tensor(np.ones((2,3,10))))
    assert meter.counts()==m.budget(1) and meter.predictor_samples()==dict(attempted=1,completed=1)
    assert all(not mod.pre and not mod.post for mod in vars(model).values()) and not hasattr(b,'_active_meter')
    assert model.encoder.calls==0 and model.action_encoder.calls==1


def test_full_four_arm_saved_consume_ram(full,monkeypatch):
    slots=[];tasks={}
    for s in (0,1):
        t=full[s,m.ARMS[0]][0];tasks[s]=t
        slots.append(dict(source_slot=s,task_id=m.core.task_id(t),task_export=str(s),task_sha256=str(s)*64,
            blind_sha256=str(s+2)*64,task_contract_sha256=t['task_contract_sha256']))
    p=dict(slots=slots,contract=m.contract(),bindings_sha256={},expected_model_hash='e'*64)
    rows=m.initial_rows(p);store=RAMStore()
    monkeypatch.setattr(m.old,'load_export',lambda path,h:dict(gn_task=tasks[int(path)]))
    def runner(t,s,a,sink,row):
        r=full[s,a][1]
        for e in full[s,a][2]:sink(e)
        row['resources']=deepcopy(full[s,a][3]);row['predictor_samples']=dict(attempted=144,completed=144)
        return deepcopy(r)
    store.put('binding.json',dict(prepared='/RAM_prepare',prepare_sha256='a'*64))
    m.plan_all(p,store,runner,rows);store.close()
    assert len(rows)==4 and all(r['status']=='model_plan_sealed' for r in rows)
    # Production read() caps each expanded candidate-result to8MiB; exercise the same limit.
    for name,raw in store.files.items():
        if name.endswith('.result.json.gz'):assert len(zlib.decompress(raw,31))<=8*1024**2
    hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in store.files.items()}
    report=dict(schema=m.NAME+'.report',status='matched_sampling_plans_sealed_not_admitted',errors=[],first_error=None,
        commands_released=True,source_verified=True,model_unchanged=True,model_hash_before='e'*64,model_hash_after='e'*64,
        contract=p['contract'],inherited_encoding_once=p['contract']['inherited_encoding_once'],resources=m.budget(576),
        predictor_samples=dict(attempted=576,completed=576),prepared='/RAM_prepare',prepare_sha256='a'*64,
        artifact_sha256=hashes,rows=rows)
    store.put('report.json',report,final=True)
    def read(path):
        raw=store.files[Path(path).name]
        return m.base.decode(zlib.decompress(raw,31) if str(path).endswith('.gz') else raw)
    def sha(path):return hashlib.sha256(store.files[Path(path).name]).hexdigest()
    monkeypatch.setattr(m,'verify',lambda path,h:p);monkeypatch.setattr(m,'read',read);monkeypatch.setattr(m,'sha',sha)
    monkeypatch.setattr(Path,'iterdir',lambda path:iter(path/name for name in store.files))
    monkeypatch.setattr(Path,'read_bytes',lambda path:store.files[path.name])
    result=m.consume(store.output,sha(store.output/'report.json'))
    assert len(result['rows'])==4 and result['resources']==m.budget(576) and not result['P1_admission']
    assert all(len(plan['commands'])==10 for plan in result['rows'])
    report['rows']=report['rows'][:-1];store.files['report.json']=m.canonical(report)
    with pytest.raises(ValueError):m.consume(store.output,sha(store.output/'report.json'))


def test_partial_hook_registration_cleanup_preserves_first():
    handles=[]
    class Module:
        def register_forward_pre_hook(self,fn):
            h=NS(remove=lambda:handles.remove(h));handles.append(h);return h
        def register_forward_hook(self,fn,always_call):raise RuntimeError('registration failed')
    b=m.NativeCuda.__new__(m.NativeCuda);b.model=NS(predictor=Module(),action_encoder=Module(),pred_proj=Module())
    with pytest.raises(RuntimeError,match='registration failed'):
        with b.guard(m.Meter(lambda e:None)):pytest.fail('must not enter')
    assert handles==[] and not hasattr(b,'_active_meter')
