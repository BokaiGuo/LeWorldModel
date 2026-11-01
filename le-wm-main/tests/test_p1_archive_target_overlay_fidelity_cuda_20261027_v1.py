"""RAM-only producer receipts and NumPy CUDA-interface mocks; no model/simulator/tmp."""
from copy import deepcopy
from io import BytesIO
import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import zlib
import numpy as np
from PIL import Image
import pytest
from experiments.spectral_reachability import p1_archive_target_overlay_fidelity_cuda_20261027_v1 as m


spec=importlib.util.spec_from_file_location('overlay_receipt_mock',Path(__file__).with_name('test_p1_archive_target_overlay_probe_20261027_v1.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


class RAMStore(m.Store):
    def __init__(self):
        self.files={};self.comp=zlib.compressobj(6,zlib.DEFLATED,31);self.closed=self.broken=False
        self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        assert append or name not in self.files
        assert sum(map(len,self.files.values()))+len(raw)<=m.CAP-(0 if final else m.FINAL)
        self.files[name]=self.files.get(name,b'')+raw


def load(store,entry):
    b=store.files[entry['file']];assert m.enc.digest(b)==entry['sha256']
    raw=gzip.decompress(b);assert m.enc.digest(raw)==entry['npy_sha256']
    a=np.load(BytesIO(raw),allow_pickle=False)
    assert m.array_id(a)==dict(shape=entry['shape'],dtype=entry['dtype'],sha256=entry['raw_sha256'])
    return a


@pytest.fixture(scope='module')
def receipts():
    # This helper executes only a deterministic in-RAM NumPy fake environment
    # from producer tests, not installed native physics or any research inputs.
    with pytest.MonkeyPatch.context() as mp:r,store,_,payload=h.execute(mp)
    p=dict(schema=m.producer.NAME+'.prepare',contract=m.producer.contract(),
        selection_fixed_before_execution=True,archive_future_state_for_evaluation_only=True)
    r['prepare_id']=m.producer.digest(p);events=deepcopy(store.events);events[0]['prepare_id']=r['prepare_id']
    previous=None
    for k,e in enumerate(events):
        e.update(index=k,previous_sha256=previous);e['event_sha256']=m.producer.digest({a:v for a,v in e.items() if a!='event_sha256'});previous=e['event_sha256']
    r['journal_chain_tail']=previous
    return p,r,events,payload,store.files


def assembled(receipts):
    p,r,events,payload,files=receipts
    def decode(f):
        b=files[f['png']];assert m.enc.digest(b)==f['png_sha256']
        with Image.open(BytesIO(b)) as im:return np.asarray(im).copy()
    return m.assemble(p,r,events,payload,decode)


def test_complete_explicit_producer_schema_triplets_and_no_replay_encodings(receipts):
    data=assembled(receipts)
    assert [d['episode_ordinal'] for d in data]==[1,6229]
    assert all(set(d['frames'])==set(m.CONDITIONS) and all(x.shape==(5,224,224,3) for x in d['frames'].values()) for d in data)
    images,rows=m.census();assert len(images)==30 and len(rows)==24
    assert m.contract()['counts']==dict(encoder=30,projector=30,predict=24,predictor=24,action_encoder=24,pred_proj=24)
    assert not any('replay' in row['condition'] for row in images)


@pytest.mark.parametrize('bad',['oldtag','partial','roworder','frameorder','step','flag','snapshot','goal','replay','oldoff','counts','raw','missing','tailframe'])
def test_assemble_rejects_partial_or_tampered_receipts(receipts,bad):
    p,r,events,payload,files=deepcopy(receipts)
    row=r['rows'][0];f=row['frames'][0]
    if bad=='oldtag':r['schema']=m.producer.prior.NAME+'.report'
    elif bad=='partial':r['status']='overlay_probe_partial_native_stop'
    elif bad=='roworder':r['rows'].reverse()
    elif bad=='frameorder':row['frames'][0],row['frames'][1]=row['frames'][1],row['frames'][0]
    elif bad=='step':f['step']=5
    elif bad=='flag':f['with_target']=True
    elif bad=='snapshot':f['post']['snapshot']['state'][0]+=1
    elif bad=='goal':f['post']['goal_pose'][0]+=1
    elif bad=='replay':row['off_replay_checks'][0]['bit_exact']=False
    elif bad=='oldoff':f['old_off_match']['file']=False
    elif bad=='counts':r['counts']['public_step']['completed']-=1
    elif bad=='raw':payload[0]['actions'][0][0]+=.001
    elif bad=='missing':row['missing_checkpoints']=[20]
    else:events=[e for e in events if not (e['kind']=='fresh_overlay_frame' and e['receipt']['step']==20)]
    def decode(frame):
        with Image.open(BytesIO(files[frame['png']])) as im:return np.asarray(im).copy()
    with pytest.raises(ValueError):m.assemble(p,r,events,payload,decode)


def test_journal_hashchain_complete_and_corruption_refused(receipts):
    _,r,es,_,_=receipts;raw=b''.join(m.canonical(e) for e in es)
    assert m.journal(raw,r['journal_chain_tail'])==es
    with pytest.raises(ValueError):m.journal(raw[:-1],r['journal_chain_tail'])
    changed=deepcopy(es);changed[10]['kind']='tampered'
    with pytest.raises(ValueError):m.journal(b''.join(m.canonical(e) for e in changed),r['journal_chain_tail'])


def test_request_chronology_same_condition_actual_teacher_force_and_no_rescale():
    zs=[np.full(192,s,np.float32) for s in (0,5,10,15,20)]
    raw=(np.arange(40,dtype=np.float32).reshape(20,2)-10)/3
    # Raw magnitudes >1 are intentionally legal expert actions, NOT planner ball actions.
    original=raw.copy();prefix=np.full(192,123,np.float32)
    requests=[m.prediction_request(1,'nativeon',kind,zs,raw,prefix) for kind in m.KINDS]
    assert all(set(r)==m.REQUEST_KEYS for r in requests)
    assert np.array_equal(requests[0]['latent_history'],requests[1]['latent_history'])
    assert np.array_equal(requests[2]['latent_history'],np.stack((zs[1],zs[2],prefix)))
    assert np.array_equal(requests[3]['latent_history'],np.stack(zs[1:4]))
    for k,r in enumerate(requests):
        expected=raw[:15] if k<2 else raw[5:20]
        assert m.bits(r['raw_packets'],expected.reshape(3,5,2))
        assert m.bits(r['tokens'],m.fidelity.normalized(expected.reshape(3,5,2)))
    assert m.bits(raw,original)


def mock_execution(data,fail=None):
    store=RAMStore();images,rows=m.census();encoded=[];predictions=[]
    def encode(rgb,meta):
        encoded.append(deepcopy(meta))
        if fail==('image',meta['index']):raise RuntimeError('encoder failure')
        # Deliberately condition-distinct stand-in representations.
        return np.full(192,meta['episode_ordinal']*.001+100*m.CONDITIONS.index(meta['condition'])+meta['step'],np.float32)
    def predict(request,index):
        predictions.append(deepcopy(request));assert set(request)==m.REQUEST_KEYS
        if fail==('predict',index):raise RuntimeError('predict failure')
        out=request['latent_history'].copy();out[-1]+=np.float32(2)
        return out
    try:analysis=m.execute(data,encode,predict,store,images,rows)
    except RuntimeError:analysis=None
    return store,images,rows,encoded,predictions,analysis


def test_all30_independent_encode_and24_predict_no_crosscondition_labels(receipts):
    data=assembled(receipts);store,images,rows,encoded,requests,analysis=mock_execution(data)
    assert len(encoded)==30 and len(requests)==24
    assert all(x['status']=='encoded' for x in images) and all(x['status']=='saved' for x in rows)
    assert len(analysis['scores'])==24 and len(analysis['paired_latent_distances'])==10
    for k,r in enumerate(requests):
        assert r['episode_ordinal']==m.EPISODES[k//12] and r['condition']==m.CONDITIONS[(k%12)//4]
        same10=np.float32(r['episode_ordinal']*.001+100*m.CONDITIONS.index(r['condition'])+10)
        assert np.array_equal(r['latent_history'][1 if k%4>=2 else 2],np.full(192,same10,np.float32))
        if k%4==2:assert np.array_equal(r['latent_history'][-1],np.full(192,same10+2,np.float32))
        if k%4==3:assert np.array_equal(r['latent_history'][-1],np.full(192,np.float32(r['episode_ordinal']*.001+100*m.CONDITIONS.index(r['condition'])+15),np.float32))
        assert m.bits(load(store,rows[k]['inputs']['tokens']),r['tokens'])
    for x in analysis['paired_latent_distances']:
        off=np.float32(x['episode_ordinal']*.001+100+x['step']);on=np.float32(x['episode_ordinal']*.001+200+x['step'])
        assert x['nativeoff_nativeon_MSE']==(float(off)-float(on))**2
    store.close();raw,eof=m.sealed.inflate(store.files['events.jsonl.gz']);assert eof
    events=[json.loads(line) for line in raw.splitlines()]
    assert sum(e['kind']=='image_saved' for e in events)==30 and sum(e['kind']=='prediction_saved' for e in events)==24
    assert events[-1]['kind']=='all24_predictions_durable_before_scoring'
    assert sum(map(len,store.files.values()))<m.CAP


@pytest.mark.parametrize('which,index',[('image',0),('image',17),('predict',0),('predict',13)])
def test_failures_keep_full_denominator_partial_input_arrays_no_retry(receipts,which,index):
    store,images,rows,encoded,requests,analysis=mock_execution(assembled(receipts),(which,index))
    assert len(images)==30 and len(rows)==24 and analysis is None
    if which=='image':
        assert len(encoded)==index+1 and not requests and images[index]['status']=='attempted'
        assert 'pixels' in images[index] and all(r['status']=='not_started' for r in rows)
    else:
        assert len(encoded)==30 and len(requests)==index+1 and rows[index]['status']=='attempted'
        assert set(rows[index]['inputs'])=={'latent_history','raw_packets','tokens'}
        assert all(r['status']=='not_started' for r in rows[index+1:])


def test_native_npy_lossless_no_dtype_or_precision_reduction():
    s=RAMStore();a=np.linspace(-123,456,192,dtype=np.float32)
    e=m.save_array(s,'native',a);assert m.bits(load(s,e),a)
    assert e['dtype']=='<f4' and e['shape']==[192]
    with pytest.raises(ValueError):m.save_array(s,'rounded',a.astype(np.float64))


class Tensor:
    """No torch import: a NumPy-backed device-interface stand-in."""
    def __init__(self,a,device='cuda'):self.a=np.asarray(a,np.float32);self.device=NS(type=device)
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.a


def test_native_trace_full_structured_encoder_and_attempt_lower_bound():
    s=RAMStore();trace=m.NativeTrace(NS(),s,lambda:None,lambda:True);trace.active=('image',0,None)
    e=trace.begin('encoder',(Tensor(np.zeros((1,3,224,224),np.float32)),),{'interpolate_pos_encoding':True})
    assert e['completed'] is False and len(trace.entries)==1
    out={'last_hidden_state':Tensor(np.zeros((1,197,192),np.float32)), 'pooler_output':Tensor(np.ones((1,192),np.float32))}
    trace.end(e,out)
    assert set(e['output'])==set(out) and load(s,e['output']['last_hidden_state']['tensor']).shape==(1,197,192)
    e2=trace.begin('projector',(Tensor(np.zeros((1,192),np.float32)),),{})
    trace.end(e2,None)
    assert trace.failed and trace.counts()['projector']==dict(attempted=1,completed=0)
    with pytest.raises(ValueError):trace.begin('projector',(),{})


def test_native_trace_refuses_cpu_wrong_history_and_unscoped_dispatch():
    trace=m.NativeTrace(NS(),RAMStore(),lambda:None,lambda:True)
    with pytest.raises(ValueError):trace.begin('encoder',(),{})
    with pytest.raises(ValueError):trace.tree(Tensor(np.zeros(3),'cpu'),'bad')
    zs=[np.zeros(192,np.float32) for _ in range(5)];r=m.prediction_request(1,'archive',m.KINDS[0],zs,np.ones((20,2),np.float32))
    trace.active=('prediction',0,r)
    with pytest.raises(ValueError):trace.begin('action_encoder',(Tensor(np.zeros((1,3,10))),),{})
    assert trace.counts()['action_encoder']==dict(attempted=1,completed=0)


def test_native_cleanup_never_covers_primary_failure():
    trace=m.NativeTrace(NS(),RAMStore(),lambda:None,lambda:True)
    def fail():raise RuntimeError('hook remove failed')
    trace.hooks=[NS(remove=fail)];primary=ValueError('original failure')
    trace.__exit__(ValueError,primary,None)
    assert str(primary)=='original failure' and 'hook remove failed' in primary.__notes__[0]
    with pytest.raises(RuntimeError):trace.__exit__(None,None,None)


def test_input_verify_before_output_or_model_import(monkeypatch):
    def fail(*args):raise ValueError('source incomplete')
    monkeypatch.setattr(m,'verify',fail);monkeypatch.setattr(m,'Store',lambda *_:pytest.fail('premature output'))
    with pytest.raises(ValueError,match='source incomplete'):m.run('/RAM/p','a'*64,'/RAM/output')


def test_material_uses_actual_encoder_hash_validator_before_input_IO(monkeypatch):
    def no_read(*args,**kwargs):pytest.fail('invalid SHA must fail before reading report')
    monkeypatch.setattr(m,'read',no_read)
    for position in range(3):
        hashes=['a'*64]*3;hashes[position]='invalid'
        with pytest.raises(ValueError):m.material('/RAM/report.json',*hashes)


def test_complete30visual24predict_full_native_arrays_synthetic_8MiB(receipts):
    """All full tensor leaves retained; compression/storage mock, NOT neural evidence.

    RGB is patterned, ViT output is random nonzero native float32. No constant
    zero-output shortcut, float16 rounding, dropped patches or omitted inputs.
    """
    data=assembled(receipts);s=RAMStore();images,rows=m.census();trace=m.NativeTrace(NS(),s,lambda:None,lambda:True)
    rng=np.random.default_rng(20261027)
    def call(kind,args,output,kwargs=None):
        e=trace.begin(kind,tuple(Tensor(a) for a in args),kwargs or {})
        trace.end(e,output);return e
    def encode(rgb,meta):
        trace.active=('image',meta['index'],None)
        x=(rgb.astype(np.float32)/np.float32(255)).transpose(2,0,1)[None].copy()
        patches=rng.normal(0,1,(1,197,192)).astype(np.float32)
        projected=patches[:,0].copy()
        call('encoder',[x],dict(last_hidden_state=Tensor(patches),pooler_output=Tensor(patches.mean(1))),dict(interpolate_pos_encoding=True))
        call('projector',[patches[:,0]],Tensor(projected))
        trace.finish('image',meta['index']);return projected[0].copy()
    def predict(r,index):
        trace.active=('prediction',index,r)
        z=r['latent_history'][None].copy();tokens=r['tokens'][None].copy()
        embedding=np.repeat(tokens.mean(-1,keepdims=True),192,axis=-1).astype(np.float32)
        call('action_encoder',[tokens],Tensor(embedding))
        outer=trace.begin('predict',(Tensor(z),Tensor(embedding)),{})
        predicted=z.copy();predicted[:,-1]+=np.float32(.125)
        call('predictor',[z,embedding],Tensor(predicted))
        call('pred_proj',[predicted.reshape(3,192)],Tensor(predicted.reshape(3,192)))
        trace.end(outer,Tensor(predicted));trace.finish('prediction',index)
        return predicted[0].copy()
    analysis=m.execute(data,encode,predict,s,images,rows)
    m.sealed.exact_counts(trace.counts(),m.contract()['counts'])
    assert len(trace.entries)==156 and all(e['completed'] for e in trace.entries)
    encoders=[e for e in trace.entries if e['resource']=='encoder']
    assert len(encoders)==30
    for e in encoders:
        assert load(s,e['inputs'][0]['tensor']).shape==(1,3,224,224)
        assert load(s,e['output']['last_hidden_state']['tensor']).shape==(1,197,192)
        assert np.any(load(s,e['output']['last_hidden_state']['tensor'])!=0)
        assert load(s,e['output']['pooler_output']['tensor']).shape==(1,192)
    s.put('analysis.json',analysis);s.close()
    s.put('report.json',dict(images=images,requests=rows,module_ledger=trace.entries,
        artifact_sha256={k:m.enc.digest(v) for k,v in s.files.items()}),final=True)
    total=sum(map(len,s.files.values()))
    raw_bytes=sum(len(gzip.decompress(v)) for k,v in s.files.items() if k.endswith('.npy.gz'))
    assert raw_bytes>20*1024**2 and total<m.CAP
    print('complete synthetic native output bytes:',total,'original NPY bytes:',raw_bytes)


def test_all_three_array_hash_layers_detect_corruption():
    store=RAMStore();a=np.arange(192,dtype=np.float32);entry=m.save_array(store,'native',a)
    compressed=store.files[entry['file']];raw=gzip.decompress(compressed)
    assert entry['sha256']==m.enc.digest(compressed)
    assert entry['npy_sha256']==m.enc.digest(raw)
    assert entry['raw_sha256']==m.enc.digest(a.tobytes())
    assert len({entry['sha256'],entry['npy_sha256'],entry['raw_sha256']})==3
    for field in ('sha256','npy_sha256','raw_sha256'):
        bad=dict(entry);bad[field]='0'*64
        with pytest.raises(AssertionError):load(store,bad)

