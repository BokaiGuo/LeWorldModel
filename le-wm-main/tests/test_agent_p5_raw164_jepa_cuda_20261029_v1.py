"""No torch/env imports: real disk codecs plus tiny hook protocol doubles."""
from copy import deepcopy
import hashlib
import json
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_raw164_jepa_cuda_20261029_v1 as m


@pytest.fixture
def work(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:NS(free=8*1024**3))
    return tmp_path


def newstore(work,name='test'):return m.Store(work/'results'/(m.NAME+'_'+name))


def test_cold_import_no_runtime():
    assert not any(k in sys.modules for k in ('torch','pymunk','pygame','cv2','stable_worldmodel'))


def test_fixed_protocol_h1_not_matched_or_rgb_only():
    p=m.protocol();assert p['requests']==164 and len(p['pairs'])==80
    assert p['context_global_frames']==[5,10,15] and p['target_global_frame']==20
    assert p['alignment_to_actual_FD164']['same_factual_H1_timing'] and not p['alignment_to_actual_FD164']['fully_matched_input']
    assert p['alignment_to_actual_FD164']['extra_agent_proprio'] and not p['RGB_only']
    assert not p['matched_cross_model_performance'] and not p['P5_admission']
    assert len(m.call_schedule())==656 and m.call_schedule()[491]==(163,'prediction','predictor')
    assert m.call_schedule()[492]==(0,'label','official_encoder')
    assert m.protocol()['caps']['predictor']==dict(api=164,batch_samples=164,frame_samples=492)


def test_real_store_journal_two_events_full_reader_chain(work):
    s=newstore(work);s.emit('one',text='escaped\nnewline');s.emit('two',value=[1,2])
    body=(s.path/'journal.jsonl').read_bytes();assert body.count(b'\n')==2 and b'\n\n' not in body
    events=list(m.journal(s.path/'journal.jsonl'));assert len(events)==2
    assert events[1]['previous_sha256']==events[0]['event_sha256'] and events[-1]['event_sha256']==s.previous
    assert s.hashes['journal.jsonl'].hexdigest()==hashlib.sha256(body).hexdigest()
    for e,line in zip(events,body.splitlines(keepends=True)):
        assert m.raw(e)==line and m.digest({k:v for k,v in e.items() if k!='event_sha256'})==e['event_sha256']


@pytest.mark.parametrize('mutation',('blank','extra_lf','missing_lf','whitespace','duplicate','chain','nonfinite'))
def test_journal_no_repair_canonical_reject(work,mutation):
    s=newstore(work);s.emit('one');s.emit('two');p=s.path/'journal.jsonl';b=p.read_bytes()
    if mutation=='blank':b=b'\n'+b
    if mutation=='extra_lf':b=b.replace(b'\n',b'\n\n',1)
    if mutation=='missing_lf':b=b[:-1]
    if mutation=='whitespace':b=b.replace(b'{',b'{ ',1)
    if mutation=='duplicate':b=b.replace(b'"sequence":0',b'"sequence":0,"sequence":0',1)
    if mutation=='chain':b=b.replace(b'"sequence":1',b'"sequence":7',1)
    if mutation=='nonfinite':b=b.replace(b'"sequence":0',b'"sequence":NaN',1)
    p.write_bytes(b)
    with pytest.raises((ValueError,KeyError)):list(m.journal(p))


def test_real_put_read_and_new_metadata_single_codec(work,monkeypatch):
    s=newstore(work);v=dict(status='fixture',z='x\ny',a=1);rec=s.put('value.json',v)
    assert m.bridge.read(s.path/'value.json',rec['sha256'])==v and (s.path/'value.json').read_bytes()==m.raw(v)
    p=work/'results'/(m.NAME+'_prepare_test.json');out=m.write_new(p,v)
    assert m.bridge.read(p,out['sha256'])==v and p.read_bytes()==m.raw(v)
    with pytest.raises(ValueError):m.write_new(p,v)


def test_real_array_disk_roundtrip_and_tree(work):
    s=newstore(work);a=np.arange(12,dtype=np.float32).reshape(3,4)
    v=({'pixels':a,'nothing':None},[a[:0],True,'literal'])
    tree=m.tree(v,s.array,lambda x:isinstance(x,np.ndarray));back=m.restore(tree,lambda rec:m.load_array(s.path,rec))
    assert np.array_equal(back[0]['pixels'],a) and back[0]['nothing'] is None
    assert back[1][0].shape==(0,4) and back[1][1:] == [True,'literal']
    assert s.tensor_bytes==a.nbytes


@pytest.mark.parametrize('mutation',('hash','shape','bytes','dtype','nan','path'))
def test_array_mutations_rejected(work,mutation):
    s=newstore(work);rec=s.array(np.ones((2,3),np.float32))
    if mutation=='hash':rec['sha256']='bad'
    if mutation=='shape':rec['shape']=[6]
    if mutation=='bytes':rec['file_bytes']+=1
    if mutation=='dtype':rec['dtype']='float64'
    if mutation=='path':rec['file']='../foreign.npy'
    if mutation=='nan':
        with (s.path/rec['file']).open('wb') as f:np.save(f,np.full((2,3),np.nan,np.float32),allow_pickle=False)
        rec['sha256']=m.sha(s.path/rec['file'])
    with pytest.raises(ValueError):m.load_array(s.path,rec)


def test_store_exclusive_caps_floor_wall_and_final(work,monkeypatch):
    s=newstore(work)
    with pytest.raises(ValueError):m.Store(s.path)
    s.put('once.json',{})
    with pytest.raises(ValueError):s.put('once.json',{})
    s.started-=m.WALL+1
    with pytest.raises(ValueError):s.emit('no_dispatch')
    s.emit('failure',final=True,error='wall')
    s.bytes=m.CAP-m.FINAL
    with pytest.raises(ValueError):s.put('no.json',{})
    s.tensor_bytes=m.protocol()['neural_tensor_io_upper_bound_bytes']
    with pytest.raises(ValueError):s.array(np.zeros(1,np.float32))
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:NS(free=m.FLOOR-1))
    with pytest.raises(ValueError):s.emit('no_final',final=True)


class Tensor:
    """Tiny protocol double, not a neural tensor or CUDA evidence."""
    def __init__(self,value):self.a=np.asarray(value,np.float32);self.is_cuda=True;self.dtype='float32';self.requires_grad=False;self.shape=self.a.shape
    def detach(self):return self
    def contiguous(self):return self
    def cpu(self):return self
    def numpy(self):return self.a.copy()


class Model:
    def __init__(self,out):self.out=out;self.pres=[];self.posts=[];self.forward_calls=0
    def register_forward_pre_hook(self,fn,with_kwargs):
        assert with_kwargs;self.pres.append(fn);return NS(remove=lambda:self.pres.remove(fn))
    def register_forward_hook(self,fn,with_kwargs):
        assert with_kwargs;self.posts.append(fn);return NS(remove=lambda:self.posts.remove(fn))
    def __call__(self,*args,**kwargs):
        for fn in self.pres:fn(self,args,kwargs)
        self.forward_calls+=1
        if isinstance(self.out,BaseException):raise self.out
        for fn in self.posts:fn(self,args,kwargs,self.out)
        return self.out


def trace_fixture(work):
    torch=NS(float32='float32',is_tensor=lambda v:isinstance(v,Tensor),is_grad_enabled=lambda:False,
        isfinite=lambda v:NS(all=lambda:bool(np.isfinite(v.a).all())),
        backends=NS(cuda=NS(matmul=NS(allow_tf32=False)),cudnn=NS(allow_tf32=False)))
    models={k:Model(Tensor(np.zeros((1,2),np.float32))) for k in m.KEYS}
    s=newstore(work);trace=m.Trace(s);trace.where=(0,'prediction');trace.install(models,torch)
    return s,trace,models,torch


def test_actual_hook_protocol_forward_once_full_io_saved(work):
    s,trace,models,torch=trace_fixture(work)
    for key,shape in zip(m.KEYS,((3,3,2,2),(1,3,4),(1,3,1,2,2,4))):models[key](Tensor(np.ones(shape,np.float32)))
    assert len(trace.calls)==3 and all(v.forward_calls==1 for v in models.values())
    events=list(m.journal(s.path/'journal.jsonl'))
    assert [v['kind'] for v in events]==['module_dispatch_intent','module_inputs_saved','module_returned','module_outputs_saved']*3
    assert trace.counts['attempted']==trace.counts['returned']
    assert trace.counts['attempted']['official_encoder']==dict(api=1,batch_samples=3,frame_samples=3)
    assert trace.counts['attempted']['predictor']==dict(api=1,batch_samples=1,frame_samples=3)
    counts=m.empty_counts();it=iter(events)
    def expect(kind):
        event=next(it);assert event['kind']==kind;return event
    for call in trace.calls:m.check_call(call,counts,expect,lambda rec:m.load_array(s.path,rec))
    assert counts==trace.counts and trace.close()==[] and all(not v.pres and not v.posts for v in models.values())


@pytest.mark.parametrize('mutation',('order','grad','tf32','cpu','requires_grad','nan'))
def test_hook_guard_before_forward(work,mutation):
    s,trace,models,torch=trace_fixture(work);x=Tensor(np.ones((3,3,2,2),np.float32));key='official_encoder'
    if mutation=='order':key='predictor'
    if mutation=='grad':torch.is_grad_enabled=lambda:True
    if mutation=='tf32':torch.backends.cuda.matmul.allow_tf32=True
    if mutation=='cpu':x.is_cuda=False
    if mutation=='requires_grad':x.requires_grad=True
    if mutation=='nan':x.a[:]=np.nan
    with pytest.raises(ValueError):models[key](x)
    assert models[key].forward_calls==0
    trace.close()


def test_failed_forward_not_refunded_or_completed(work):
    s,trace,models,torch=trace_fixture(work);models['official_encoder'].out=RuntimeError('fixture failure')
    with pytest.raises(RuntimeError):models['official_encoder'](Tensor(np.zeros((3,3,2,2),np.float32)))
    assert trace.counts['attempted']['official_encoder']['api']==1 and trace.counts['returned']['official_encoder']['api']==0
    assert trace.pending['status']=='attempted'
    assert [e['kind'] for e in m.journal(s.path/'journal.jsonl')]==['module_dispatch_intent','module_inputs_saved']
    trace.close()


def test_prediction_seal_requires_every_row_before_write(work):
    s=newstore(work);rows=[dict(index=i,prediction_status='not_started',prediction=None) for i in range(164)]
    with pytest.raises(ValueError):m.predictions_seal(s,{'bridge':{}},rows)
    assert not (s.path/'predictions_seal.json').exists()
    rec=s.array(np.zeros((1,3,256,384),np.float32))
    for row in rows:row.update(prediction_status='sealed',prediction=dict(rec,file=f"prediction_{row['index']}.npy"))
    rows[-1]['prediction_status']='attempted'
    with pytest.raises(ValueError):m.predictions_seal(s,{'bridge':{}},rows)
    rows[-1]['prediction_status']='sealed';seal=m.predictions_seal(s,{'bridge':{}},rows)
    body=m.bridge.read(s.path/'predictions_seal.json',seal['sha256'])
    assert len(body['predictions'])==164 and list(m.journal(s.path/'journal.jsonl'))[-1]['kind']=='predictions_sealed'


def test_exact_factual_mse_and_persistence_not_cross_model():
    pred=np.zeros((1,3,256,384),np.float32);context=np.zeros((3,256,384),np.float32);label=np.ones((1,256,384),np.float32)
    pred[:,-1]=2;context[-1]=3
    assert m.arithmetic(pred,context,label)==dict(successor_MSE=1.,persistence_MSE=4.)
    with pytest.raises(ValueError):m.arithmetic(pred[:,-1],context,label)


def test_full80_missing_pairs_never_removed(work):
    s=newstore(work);rows=[dict(index=i,score=None) for i in range(164)]
    result=m.pair_scores(rows,s.path)
    assert len(result)==80 and all(len(r['missing'])==2 and r['response_cosine'] is None for r in result)


def test_pair_zero_response_null_not_division_or_drop(work):
    s=newstore(work);p=s.array(np.zeros((1,3,256,384),np.float32));y=s.array(np.ones((1,256,384),np.float32))
    rows=[dict(index=i,score=dict(successor_MSE=1.),prediction=p,label_patch=y) for i in range(164)]
    result=m.pair_scores(rows,s.path)
    assert len(result)==80 and all(r['missing']==[] and r['response_cosine'] is None and r['observed_response_l2']==r['predicted_response_l2']==0 for r in result)


def test_no_authority_no_files_no_verify(work,monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('no source work'))
    with pytest.raises(ValueError):m.run('none','none',work/'results'/(m.NAME+'_run'),'bad')
    assert not any((work/'results').iterdir())


def test_real_failed_preflight_materializes164_and_blocked_receipt(work,monkeypatch):
    def fail(*a):raise ValueError('fixture source drift')
    monkeypatch.setattr(m,'verify',fail)
    out=work/'results'/(m.NAME+'_failed');result=m.run('missing','missing',out,m.AUTH)
    r=m.bridge.read(out/'report.json',result['sha256'])
    assert r['status']=='partial_or_failed' and len(r['rows'])==164 and len(r['pairs'])==80
    assert r['errors'] and not any(row['score'] is not None for row in r['rows'])
    receipt=m.consume(out,result['sha256']);assert receipt['status']=='blocked_partial_capture' and receipt['requests']==164


def test_encoder_reference_is_fixed_all_pixels():
    rgb=np.arange(3*224*224,dtype=np.uint8).reshape(1,224,224,3)
    actual=m.encoder_reference(rgb)
    assert actual.shape==(1,3,224,224) and actual.dtype==np.float32
    np.testing.assert_array_equal(actual[:,0],(rgb[:,:,:,0].astype(np.float32)*np.float32(1/255)-np.float32(.485))/np.float32(.229))


def test_full164_orchestration_and_strict_with_virtual_tensor_fixture(work,monkeypatch):
    """Full control flow only: tensor storage/load explicitly mocked with tiny
    marker files and shared RAM views, NOT genuine NPY/encoder/CUDA evidence.
    Real NPY and codec paths are independently tested above.
    """
    arrays={};mutate={'label':False}
    def virtual_array(self,value):
        name=f'tensor_{self.array_index:05d}.npy';self.array_index+=1
        rec=dict(self.write(name,b'RAM_FIXTURE_NOT_NPY'),shape=list(value.shape),dtype='float32',nbytes=value.nbytes)
        arrays[(str(self.path),name)]=value;self.tensor_bytes+=value.nbytes;return rec
    def virtual_load(directory,rec):return arrays[(str(directory),rec['file'])]
    monkeypatch.setattr(m.Store,'array',virtual_array);monkeypatch.setattr(m,'load_array',virtual_load)
    monkeypatch.setattr(Tensor,'numpy',lambda self:self.a)  # explicit shared-view fixture
    actual=m.bridge.read(m.bridge.PARENT,m.bridge.PARENT_SHA)['expected_model_receipts']
    fixture_path=__file__;bindings={fixture_path:m.sha(fixture_path)}
    requests=[dict(index=i,root_slot=m.bridge.factor.ORDER[i//41][0],condition=m.bridge.factor.ORDER[i//41][1],task_id='fixture',selection_digest='fixture') for i in range(164)]
    q=dict(requests=requests);manifest=dict(bridge=q,bindings=bindings)
    monkeypatch.setattr(m,'verify',lambda *a:manifest)
    rgb=np.zeros((3,224,224,3),np.uint8);prop=np.zeros((3,4),np.float32);acts=np.zeros((3,10),np.float32)
    items=[dict(index=i,condition=requests[i]['condition'],root_slot=requests[i]['root_slot'],rgb=rgb,normalized_proprio=prop,packets=acts) for i in range(164)]
    monkeypatch.setattr(m.bridge,'host_inputs',lambda _:iter(items))
    patch3=np.broadcast_to(np.float32(0),(3,256,384));patch1=patch3[:1]
    propout=np.broadcast_to(np.float32(0),(1,3,1,16));full=np.broadcast_to(np.float32(0),(1,3,256,384))
    tf=NS(float32='float32',is_tensor=lambda v:isinstance(v,Tensor),is_grad_enabled=lambda:False,isfinite=lambda v:NS(all=lambda:True),
        backends=NS(cuda=NS(matmul=NS(allow_tf32=False)),cudnn=NS(allow_tf32=False)))
    class Backend:
        def __init__(self,manifest,trace):
            self.trace=trace;self.before=actual;self.strict={k:dict(exact_match=True) for k in ('predictor','proprio_encoder')}
            self.sources={name:fixture_path for name in ('app.plan_common.models.AdaLN_vit','app.plan_common.models.prop_embedding',
                'dinov2.models.vision_transformer','app.plan_common.datasets.transforms')}
            self.models=dict(official_encoder=Model({'x_norm_patchtokens':Tensor(patch3)}),proprio_encoder=Model(Tensor(propout)),
                predictor=Model((Tensor(full),None,Tensor(np.broadcast_to(propout,(1,3,256,16))))))
            trace.install(self.models,tf)
        def predict(self,item):
            self.trace.where=(item['index'],'prediction')
            self.models['official_encoder'](Tensor(m.encoder_reference(rgb)),is_training=True)
            self.models['proprio_encoder'](Tensor(prop[None]))
            self.models['predictor'](Tensor(patch3.reshape(1,3,1,16,16,384)),Tensor(acts[None]),Tensor(np.broadcast_to(propout,(1,3,256,16))))
            return self.trace.calls[-1]['outputs']['$tuple'][0]['$array']
        def label(self,index,rgb):
            mutate['label']=True;self.trace.where=(index,'label');self.models['official_encoder'].out={'x_norm_patchtokens':Tensor(patch1)}
            self.models['official_encoder'](Tensor(m.encoder_reference(rgb[None])),is_training=True)
        def receipt(self):return actual
        def memory(self):return dict(cuda_peak_allocated_bytes=1,cuda_peak_reserved_bytes=2,cuda_free_bytes=3,cuda_total_bytes=4,host_peak_rss_bytes=5)  # fixture, not measurement
    monkeypatch.setattr(m,'CudaBackend',Backend)
    output=work/'results'/(m.NAME+'_full_RAM')
    def labels(q,path,sha):
        events=list(m.journal(output/'journal.jsonl'))
        assert sum(e['kind']=='module_dispatch_intent' for e in events)>=492
        assert next(i for i,e in enumerate(events) if e['kind']=='predictions_sealed')<next(i for i,e in enumerate(events) if e['kind']=='label_decode_intent')
        return [dict(index=i,rgb=rgb[0]) for i in range(164)]
    monkeypatch.setattr(m.bridge,'labels_after_seal',labels)
    result=m.run('fixture_prepared','fixture_sha',output,m.AUTH)
    assert result['status']=='fixed164_actual15_conditioned_factual_H1_recorded' and mutate['label']
    strict=m.consume(output,result['sha256'])
    assert strict['status']=='all164_factual_H1_and80_pairs_verified' and strict['counts']['returned']==m.protocol()['caps']
    r=m.bridge.read(output/'report.json',result['sha256'])
    r['rows'][8]['score']['successor_MSE']=1.
    (output/'report.json').write_bytes(m.raw(r))
    with pytest.raises(ValueError):m.consume(output,m.sha(output/'report.json'))
