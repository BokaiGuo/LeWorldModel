"""Pure RAM arithmetic, fake transition and in-memory storage; no Torch forward."""
from copy import deepcopy
import hashlib
import time
import zlib
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import agent_experiment1_common_controller_20261028_v1 as m


def task(slot=0):
    moving=m.sampling.old.moving
    packet=dict(rgb_history=np.stack([np.full((224,224,3),i+slot,np.uint8) for i in range(3)]),
        raw_action_history=moving.fixed_raw(),goal_rgb=np.full((224,224,3),9+slot,np.uint8))
    s=dict(source_slot=slot,blind_sha256=str(slot+2)*64,
        pixel_sha256=[moving.digest(x.tobytes()) for x in [*packet['rgb_history'],packet['goal_rgb']]],
        raw_action_sha256=moving.digest(packet['raw_action_history'].tobytes()))
    p=dict(normalization=moving.normalization(),model_bindings=moving.MODELS,arguments=dict(manifest_sha='a'*64))
    z=np.random.default_rng(17+slot).normal(size=(4,192)).astype(np.float32)
    export=moving.task_export(p,'b'*64,s,packet,list(z))
    return m.sampling.old.interface(moving.task_interfaces(export))['gn_task']


def receipt(x):
    x=np.asarray(x,np.float32)
    return dict(shape=list(x.shape),dtype='float32',values=x.astype(float).tolist())


class Backend:
    def __init__(self,fail=None):self.calls=0;self.fail=fail
    def advance(self,t,before,raw,meter):
        self.calls+=1;previous=deepcopy(before);raw=m.beam.commands(raw);path=[]
        for k in range(len(raw)//5):
            meter.packet=k;meter.available()
            z=np.asarray(previous['latent_history'],np.float32)[None]
            outgoing=np.concatenate((np.asarray(previous['action_history'],np.float32),
                m.core.normalized_tokens(t,raw[k*5:k*5+5])))[None]
            a=meter.call('action_encoder',[receipt(outgoing)],lambda:np.resize(outgoing,(1,3,192)).astype(np.float32),receipt)
            def api():
                if self.calls==self.fail:raise RuntimeError('RAM fault')
                pred=meter.call('predictor',[receipt(z),receipt(a)],
                    lambda:(z+.002*a+np.float32(raw[k*5:k*5+5].sum()*.01)).astype(np.float32),receipt)
                return meter.call('pred_proj',[receipt(pred.reshape(3,192))],
                    lambda:pred.reshape(3,192).copy(),receipt).reshape(1,3,192)
            out=meter.call('predict_API',[receipt(z),receipt(a)],api,receipt)
            previous=m.beam.rolled(t,previous,raw[k*5:k*5+5],out[0,-1]);path.append(previous)
            meter.emit('native_packet',packet=k,context=previous)
        return path


class RAMStore(m.Store):
    def __init__(self):
        self.files={};self.started=time.monotonic();self.closed=self.broken=False
        self.meta=0;self.event_count=0;self.previous='0'*64
        self.comp=zlib.compressobj(6,zlib.DEFLATED,31)
        self._write('arrays.bin',b'');self._write('events.jsonl.gz',b'')
        self.archive=m.ArrayArchive(lambda raw:self._write('arrays.bin',raw,append=True))
    def _write(self,name,raw,append=False,final=False):
        if not append:assert name not in self.files
        self.files[name]=self.files.get(name,b'')+raw


@pytest.fixture(scope='module')
def full():
    tasks=[task(s) for s in (0,1)];slots=[];compiled=[]
    for s,t in enumerate(tasks):
        slots.append(dict(source_slot=s,task_id=m.core.task_id(t),blind_sha256=str(s+2)*64,
            task_export='RAM'+str(s),task_sha256='a'*64,task_contract_sha256='b'*64))
        compiled.append(dict(source_slot=s,target=t['root_context']['latent_history'][-2]))
    p=dict(slots=slots,compiled=compiled,contract=m.contract(),expected_model_hash={'RAM':'f'*64},bindings_sha256={})
    summaries=[]
    for s,t in enumerate(tasks):
        a=m.arbitrary(t,s)
        summaries.append(dict(source_slot=s,task_id=slots[s]['task_id'],arbitrary_sha256=m.core.digest(a),
            targets={m.ARMS[0]:a['target'],m.ARMS[1]:compiled[s]['target']}))
    p['targets']=summaries;groups={}
    for s in (0,1):
        for arm in m.ARMS:
            t=tasks[s];ev=[];meter=m.sampling.Meter(ev.append);b=Backend();d=m.Dispatch(t,b,meter,ev.append)
            r=m.optimize(t,s,arm,p['targets'][s]['targets'][arm],d,ev.append)
            groups[s,arm]=(r,ev,meter,d,b)
    return tasks,p,groups


@pytest.mark.parametrize('slot,arm',[(s,a) for s in (0,1) for a in m.ARMS])
def test_full144_strict(full,slot,arm):
    tasks,p,groups=full;r,e,meter,d,b=groups[slot,arm]
    assert meter.counts()==m.budget() and meter.predictor_samples()==dict(attempted=144,completed=144)
    assert b.calls==d.index==72 and len(r['candidates'])==72 and len(r['updates'])==3
    assert m.journal_check(tasks[slot],slot,arm,p['targets'][slot]['targets'][arm],r,e)==r
    assert r['winner_index']==min(range(72),key=lambda i:(r['candidates'][i]['cost'],i))
    assert len(r['candidates'][r['winner_index']]['commands'])==10


def test_same_first_draw_different_targets(full):
    tasks,p,groups=full
    for s in (0,1):
        one=groups[s,m.ARMS[0]][1][0];two=groups[s,m.ARMS[1]][1][0]
        assert one==two and one['kind']=='bank_frozen' and len(one['parameters'])==24
        assert p['targets'][s]['targets'][m.ARMS[0]]!=p['targets'][s]['targets'][m.ARMS[1]]
        for arm in m.ARMS:
            assert groups[s,arm][0]['task_id']==m.core.task_id(tasks[s])
    assert groups[0,m.ARMS[0]][1][0]!=groups[1,m.ARMS[0]][1][0]


def test_latent_finite_search_exact():
    t=task();a=m.arbitrary(t,0);z=np.asarray(t['root_context']['latent_history'][-1]);g=np.asarray(t['goal'])
    rng=np.random.Generator(np.random.PCG64(2028102801));u=rng.normal(size=(20,192))
    x=(z+np.linalg.norm(g-z)/np.sqrt(192)*u).astype(np.float32)
    assert a['draws']==u.tolist() and a['candidates']==x.tolist() and a['model_calls']==0
    assert a['target']==a['candidates'][min(range(20),key=lambda i:(a['costs'][i],i))]
    t['goal']=deepcopy(t['root_context']['latent_history'][-1]);a=m.arbitrary(t,0)
    assert a['scale']==0 and a['winner_index']==0 and len(a['candidates'])==20


def test_full_history_raw_domain_and_metric(full):
    tasks,_,groups=full
    for (s,arm),(r,*_) in groups.items():
        t=tasks[s];assert np.asarray(t['root_raw_action_history']).any()
        for c in r['candidates']:
            x=np.asarray(c['logits']);assert x.shape==(2,10) and all(np.linalg.norm(v)<=m.beam.CAP+1e-12 for v in x)
            assert c['commands']==np.tanh(x.astype(np.float32)).reshape(10,2).tolist()
            before=t['root_context']
            for k,cs in enumerate(c['contexts']):
                assert cs['latent_history'][:-1]==before['latent_history'][1:]
                assert cs['action_history'][0]==before['action_history'][-1]
                assert cs['action_history'][-1]==m.core.normalized_tokens(t,c['commands'][5*k:5*k+5])[0].tolist()
                before=cs
            assert c['cost']==.5*(c['prefix_target_cost']+c['model_terminal_cost'])


@pytest.mark.parametrize('change',['missing','extra','bad_history','bad_output','batch','budget','target','winner','draw','update','order','raw'])
def test_mutations_rejected(full,change):
    tasks,p,groups=full;s=0;a=m.ARMS[0];r,e,*_=groups[s,a]
    r=deepcopy(r);e=deepcopy(e);target=deepcopy(p['targets'][s]['targets'][a])
    i=next(i for i,x in enumerate(e) if x['kind']=='dispatch_attempted')
    if change=='missing':e.pop(i)
    elif change=='extra':e.insert(i,deepcopy(e[i]))
    elif change=='bad_history':e[i]['receipt']['inputs'][0]['values'][0][0][0]+=.1
    elif change=='bad_output':next(x for x in e if x['kind']=='dispatch_returned')['receipt']['output']['values'][0][0][0]+=.1
    elif change=='batch':e[i]['receipt']['batch_size']=2
    elif change=='budget':e.append(deepcopy(e[-1]))
    elif change=='target':target[0]+=1
    elif change=='winner':r['winner_index']=(r['winner_index']+1)%72
    elif change=='draw':e[0]['parameters'][0][0][0]+=.1
    elif change=='update':next(x for x in e if x['kind']=='distribution_update')['update']['mean'][0][0]+=.1
    elif change=='order':e[0],e[1]=e[1],e[0]
    elif change=='raw':next(x for x in e if x['kind']=='edge_attempt')['edge']['commands'][0][0]+=.1
    with pytest.raises(ValueError):m.journal_check(tasks[s],s,a,target,r,e)


def test_failed_call_no_refund_retry():
    t=task();ev=[];meter=m.sampling.Meter(ev.append);b=Backend(fail=1);d=m.Dispatch(t,b,meter,ev.append)
    with pytest.raises(RuntimeError,match='RAM fault'):m.optimize(t,0,m.ARMS[0],m.arbitrary(t,0)['target'],d,ev.append)
    assert meter.counts()['predict_API']==dict(attempted=1,completed=0)
    assert meter.counts()['action_encoder']==dict(attempted=1,completed=1)
    assert ev[-1]['kind']=='edge_failed'
    with pytest.raises(ValueError):d(t['root_context'],np.zeros((10,2)),'whole_H2',[0,0])
    assert b.calls==1


def test_no_extra_replay(full):
    tasks,_,groups=full;_,_,meter,d,b=groups[0,m.ARMS[0]]
    with pytest.raises(ValueError):d(tasks[0]['root_context'],np.zeros((10,2)),'replay',[0])
    calls=[]
    with pytest.raises(ValueError):meter.call('predict_API',[],lambda:calls.append(1))
    assert b.calls==72 and calls==[]


def test_partial_all_six_retained(full,monkeypatch):
    tasks,p,_=full;rows=m.initial_rows(p)
    monkeypatch.setattr(m.sampling.old,'load_export',lambda path,h:dict(gn_task=tasks[int(path[-1])]))
    store=RAMStore()
    def fail(*a):raise RuntimeError('stop')
    with pytest.raises(RuntimeError):m.plan_all(p,store,fail,rows)
    assert len(rows)==6 and rows[0]['status']=='attempted' and rows[2]['status']==rows[3]['status']=='not_started'
    assert [x['status'] for x in rows if x['arm']==m.LEARNED]==['unknown','unknown']
    assert all(x['resources'] is x['predictor_samples'] is x['commands'] is None for x in rows if x['arm']==m.LEARNED)


def test_archive_exact_signedzero_float64_and_dedup():
    sink=bytearray();ar=m.ArrayArchive(sink.extend)
    values=[-0.,float(np.nextafter(1.,2.))]+[1e-301]*10
    v=dict(one=values,two=values,ints=list(range(12)))
    packed=ar.pack(v);assert ar.size==12*8 and packed['one']==packed['two']
    rd=m.ArrayReader(bytes(sink));restored=rd.unpack(packed);rd.finish()
    assert m.canonical(v)==m.canonical(restored) and np.signbit(restored['one'][0])
    assert packed['ints']==list(range(12))


@pytest.mark.parametrize('change',['hash','range','shape','duplicate','trailing','nan','reserved','oversize'])
def test_archive_rejects(change):
    sink=bytearray();ar=m.ArrayArchive(sink.extend);v=ar.pack([float(i) for i in range(12)])
    rd=m.ArrayReader(bytes(sink))
    if change=='hash':v['__f64__']='f'*64
    elif change=='range':v['offset']=1
    elif change=='shape':v['shape']=[11]
    elif change=='duplicate':
        raw=bytes(sink)*2;rd=m.ArrayReader(raw);rd.unpack(v);v['offset']=len(sink)
    elif change=='trailing':
        rd=m.ArrayReader(bytes(sink)+b'!');rd.unpack(v)
        with pytest.raises(ValueError):rd.finish()
        return
    elif change=='nan':
        with pytest.raises(ValueError):ar.pack([float('nan')]*12)
        return
    elif change=='reserved':
        with pytest.raises(ValueError):ar.pack({'__f64__':'x'})
        return
    else:
        with pytest.raises(ValueError):ar.pack([0.]*4097)
        return
    with pytest.raises(ValueError):rd.unpack(v)


def test_stream_limits_no_concat_trailing():
    raw=b'x'*20;packed=zlib.compress(raw)
    assert m.bounded_z(packed,20)==raw
    for bad,cap in ((packed,19),(packed+b'!',20),(packed+packed,40),(packed[:-1],20)):
        with pytest.raises(ValueError):m.bounded_z(bad,cap)


def test_ram_full_saved_payload_and_strict_consumer(full,monkeypatch):
    tasks,p,groups=full
    monkeypatch.setattr(m.sampling.old,'load_export',lambda path,h:dict(gn_task=tasks[int(path[-1])]))
    rows=m.initial_rows(p);store=RAMStore();store.put('binding.json',dict(prepared='RAM',prepare_sha256='c'*64))
    tp=m.target_payload(p);store.put('targets.json',tp)
    store.event(dict(kind='ALL_targets_sealed_before_model',targets_sha256=m.core.digest(tp)))
    def runner(t,s,a,w,sink,row):
        r,ev,meter,*_=groups[s,a]
        for e in ev:sink(e)
        row['resources']=meter.counts();row['predictor_samples']=meter.predictor_samples();return r
    m.plan_all(p,store,runner,rows);store.close()
    # Exact encoded files and journal are decoded in RAM, without model replay.
    ar=m.ArrayReader(store.files['arrays.bin']);payload={}
    for name,raw in store.files.items():
        if name in ('arrays.bin','events.jsonl.gz'):continue
        if name.endswith('.z'):raw=m.bounded_z(raw,m.META_CAP)
        payload[name]=ar.unpack(m.decode(raw))
    events=[];prev='0'*64
    for i,line in enumerate(m.bounded_z(store.files['events.jsonl.gz'],m.META_CAP,31).splitlines()):
        e=m.decode(line);assert e['index']==i and e['previous']==prev
        assert e['sha256']==m.core.digest({k:e[k] for k in ('index','previous','event')})
        prev=e['sha256'];events.append(ar.unpack(e['event']))
    ar.finish();resources,samples=m.aggregate(rows)
    report=dict(schema=m.NAME+'.report',status='calibration_common_controller_plans_sealed',errors=[],first_error=None,
        commands_released=True,source_verified=True,model_unchanged=True,model_hash_before=p['expected_model_hash'],model_hash_after=p['expected_model_hash'],
        contract=p['contract'],resources=resources,predictor_samples=samples,rows=rows,prepared='RAM',prepare_sha256='c'*64,
        inherited_encoding_once=p['contract']['inherited_encoding_once'],inherited_compiled_search=p['contract']['inherited_compiled_search'],
        dispatch_denominator=4,scientific_slots=6,learned_unknown_slots=2,calibration_only=True,P1_admission=False,
        training=False,new_physics=0,equal_total_compute=False)
    plans=m.validate_saved(p,report,payload,events)
    assert len(plans)==4 and samples==dict(attempted=576,completed=576)
    assert store.archive.size<m.BLOB_CAP and store.meta<m.META_CAP
    assert sum(map(len,store.files.values()))<m.CAP-m.FINAL
    report.update(artifact_sha256={k:hashlib.sha256(v).hexdigest() for k,v in store.files.items()},
        journal_events=store.event_count,journal_final_sha256=store.previous,array_bytes=store.archive.size,
        metadata_uncompressed_bytes=store.meta)
    store.files['report.json']=m.canonical(report)
    # Exercise the real filesystem-facing consume parser with a RAM-only VFS.
    class VPath:
        def __init__(self,name='ROOT'):self.key=name.key if isinstance(name,VPath) else str(name)
        def __truediv__(self,name):return VPath(name)
        def absolute(self):return self
        @property
        def name(self):return self.key
        def iterdir(self):return iter([VPath(k) for k in store.files])
        def stat(self):return SimpleNamespace(st_size=len(store.files[self.key]))
        def is_file(self):return self.key in store.files
        def is_symlink(self):return False
        def read_bytes(self):return store.files[self.key]
    real_read=m.read
    def ram_read(path):
        if isinstance(path,VPath):return m.decode(path.read_bytes())
        return real_read(path)
    monkeypatch.setattr(m,'Path',VPath)
    monkeypatch.setattr(m,'sha',lambda path:hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(m,'read',ram_read)
    monkeypatch.setattr(m,'verify',lambda path,h:p if path=='RAM' and h=='c'*64 else pytest.fail('wrong prepare binding'))
    actual_payload,actual_events=m.saved_payloads(VPath(),report)
    assert actual_payload==payload and actual_events==events
    out=m.consume(VPath(),hashlib.sha256(store.files['report.json']).hexdigest())
    assert len(out['plans'])==4 and len(out['rows'])==6 and out['learned_unknown_slots']==2
    # Even if outer artifact SHA is re-issued, broken inner chain is rejected.
    original=store.files['events.jsonl.gz'];lines=m.bounded_z(original,m.META_CAP,31).splitlines()
    first=m.decode(lines[0]);first['previous']='f'*64
    altered=m.canonical(first)+b'\n'.join(lines[1:])+b'\n'
    comp=zlib.compressobj(wbits=31);store.files['events.jsonl.gz']=comp.compress(altered)+comp.flush()
    bad=deepcopy(report);bad['artifact_sha256']['events.jsonl.gz']=hashlib.sha256(store.files['events.jsonl.gz']).hexdigest()
    with pytest.raises(ValueError):m.saved_payloads(VPath(),bad)
    store.files['events.jsonl.gz']=original
    # Fixture storage is not a compression-ratio claim about real research data.
    for damage in ('unknown','missing','target','seal_order','native_claim'):
        r=deepcopy(report);e=deepcopy(events)
        if damage=='unknown':r['rows'][1]['resources']={}
        elif damage=='missing':r['rows'].pop()
        elif damage=='target':e[0]['targets_sha256']='f'*64
        elif damage=='seal_order':e[0],e[1]=e[1],e[0]
        else:r['P1_admission']=True
        with pytest.raises(ValueError):m.validate_saved(p,r,payload,e)


def test_metadata_cap_and_broken_latch(monkeypatch):
    store=RAMStore();store.meta=m.META_CAP
    with pytest.raises(ValueError):store.event({'kind':'over'})
    assert store.broken
    with pytest.raises(ValueError):store.event({'kind':'retry'})
    archive=m.ArrayArchive(lambda raw:None);archive.size=m.BLOB_CAP
    with pytest.raises(ValueError):archive.pack([float(i) for i in range(12)])


def test_spec_authority_contract_storage():
    c=m.contract()
    assert c['total']==m.budget(576) and c['dispatch_denominator']==4 and c['scientific_slots']==6
    assert c['seed']==[2028102811,2028102812] and c['latent_seed']==[2028102801,2028102802]
    assert m.BLOB_CAP+m.META_CAP+m.FINAL < m.CAP
    assert c['image_encoder']==c['image_projector']==c['AD']==c['physics']==c['extra_replay']==0
    assert not c['P1_admission'] and not c['equal_total_compute'] and not c['official_learned_comparison']
    assert m.sha(m.beam.__file__)==m.BEAM_SHA and m.sha(m.SPEC)==m.SPEC_SHA


def test_live_store_fsync_order_and_no_overwrite(monkeypatch):
    store=RAMStore();order=[];write=store._write
    def recorded(name,raw,append=False,final=False):
        order.append(name);return write(name,raw,append,final)
    store._write=recorded
    store.event(dict(kind='tensor',value=[float(i) for i in range(12)]))
    assert order==['arrays.bin','events.jsonl.gz']
    order.clear();store.event(dict(kind='same_tensor',value=[float(i) for i in range(12)]))
    assert order==['events.jsonl.gz']
    store.put('once.json',{'a':1})
    with pytest.raises(AssertionError):store.put('once.json',{'a':2})
    store.close()
    with pytest.raises(ValueError):store.event({'kind':'after_close'})


def test_verify_sha_before_build_and_semantic_drift(monkeypatch):
    calls=[]
    monkeypatch.setattr(m,'sha',lambda path:'a'*64)
    monkeypatch.setattr(m,'build',lambda:calls.append(1) or {'new':True})
    with pytest.raises(ValueError,match='external prepare'):m.verify('RAM','b'*64)
    assert not calls
    monkeypatch.setattr(m,'read',lambda path:{'old':True})
    with pytest.raises(ValueError,match='entire own prepare'):m.verify('RAM','a'*64)
    assert calls==[1]


def test_prepare_size_cap_no_write(monkeypatch):
    monkeypatch.setattr(m,'build',lambda:dict(oversize='x'*(256*1024)))
    with pytest.raises(ValueError,match='256KiB'):m.prepare('RAM_must_not_create.json')


def test_run_first_error_secondary_cleanup_no_neural_import(full,monkeypatch):
    _,p,_=full
    class FailedStore(RAMStore):
        output=SimpleNamespace(iterdir=lambda:[])
        def event(self,e):raise RuntimeError('secondary journal fault')
        def close(self):raise RuntimeError('secondary close fault')
    store=FailedStore()
    monkeypatch.setattr(m,'Store',lambda output:store)
    monkeypatch.setattr(m,'verify',lambda *a:p)
    def fail(p):raise ValueError('PRIMARY target failure')
    monkeypatch.setattr(m,'target_payload',fail)
    with pytest.raises(ValueError,match='partial evidence preserved'):m.run('RAM','c'*64,'RAM_OUTPUT')
    report=m.decode(store.files['report.json'])
    assert 'PRIMARY target failure' in str(report['first_error'])
    assert len(report['errors'])==3 and 'secondary journal' in str(report['errors'][1]) and 'secondary close' in str(report['errors'][2])
    assert len(report['rows'])==6 and not report['commands_released']
    assert report['status']=='partial_or_failed' and report['resources']==m.budget(0)
