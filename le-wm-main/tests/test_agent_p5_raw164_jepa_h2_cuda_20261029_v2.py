"""No torch/env imports: real disk codecs plus tiny hook protocol doubles."""
from copy import deepcopy
import hashlib
import json
import os
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_raw164_jepa_h2_cuda_20261029_v2 as m


@pytest.fixture
def work(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:NS(free=8*1024**3))
    return tmp_path


def newstore(work,name='test'):return m.Store(work/'results'/(m.NAME+'_'+name))


def test_cold_import_no_runtime():
    assert not any(k in sys.modules for k in ('torch','pymunk','pygame','cv2','stable_worldmodel'))


def test_fixed_h2_protocol_no_actual15():
    p=m.protocol();assert p['requests']==164 and len(p['pairs'])==80
    assert p['context_global_frames']==[0,5,10] and p['target_global_frames']==[15,20]
    assert not p['horizon_only_alignment']['extra_actual15_feedback'] and not p['RGB_only']
    assert not p['matched_cross_model_performance'] and not p['P5_admission']
    assert len(m.call_schedule())==820 and m.call_schedule()[655]==(163,'prediction','predictor')
    assert m.call_schedule()[656]==(0,'label','official_encoder')
    assert p['caps']['predictor']==dict(api=328,batch_samples=328,frame_samples=984)
    assert p['caps']['official_encoder']==dict(api=328,batch_samples=820,frame_samples=820)


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
    s=newstore(work);rows=[dict(index=i,prediction_status='not_started',prediction=None,prediction15=None) for i in range(164)]
    with pytest.raises(ValueError):m.predictions_seal(s,{'bridge':{}},rows)
    assert not (s.path/'predictions_seal.json').exists()
    rec=s.array(np.zeros((1,3,256,384),np.float32))
    for row in rows:
        row.update(prediction_status='sealed',prediction=rec,prediction15=rec)
    # A single output cannot masquerade as328 real dispatch returns.
    with pytest.raises(ValueError):m.predictions_seal(s,{'bridge':{}},rows)
    for row in rows:
        row['prediction']=dict(rec,file=f"pred20_{row['index']}.npy")
        row['prediction15']=dict(rec,file=f"pred15_{row['index']}.npy")
    rows[-1]['prediction_status']='attempted'
    with pytest.raises(ValueError):m.predictions_seal(s,{'bridge':{}},rows)
    rows[-1]['prediction_status']='sealed';seal=m.predictions_seal(s,{'bridge':{}},rows)
    body=m.bridge.read(s.path/'predictions_seal.json',seal['sha256'])
    assert len(body['predictions'])==328 and list(m.journal(s.path/'journal.jsonl'))[-1]['kind']=='predictions_sealed'


def test_exact_factual_mse_and_persistence_not_cross_model():
    pred=np.zeros((1,3,256,384),np.float32);context=np.zeros((3,256,384),np.float32);label=np.ones((2,256,384),np.float32)
    pred[:,-1]=2;context[-1]=3
    assert m.arithmetic(pred,context,label,pred)==dict(successor_MSE=1.,prediction15_MSE=1.,persistence_MSE=4.,persistence15_MSE=4.)
    with pytest.raises(ValueError):m.arithmetic(pred[:,-1],context,label,pred)


def test_full80_missing_pairs_never_removed(work):
    s=newstore(work);rows=[dict(index=i,score=None) for i in range(164)]
    result=m.pair_scores(rows,s.path)
    assert len(result)==80 and all(len(r['missing'])==2 and r['response_cosine'] is None for r in result)


def test_pair_zero_response_null_not_division_or_drop(work):
    s=newstore(work);p=s.array(np.zeros((1,3,256,384),np.float32));y=s.array(np.ones((2,256,384),np.float32))
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
    q=dict(schema=m.NAME+'.inputs',requests=requests);manifest=dict(bridge=q,bindings=bindings)
    monkeypatch.setattr(m,'verify',lambda *a:manifest)
    rgb=np.zeros((3,224,224,3),np.uint8);prop=np.zeros((3,4),np.float32);acts=np.zeros((4,10),np.float32)
    items=[dict(index=i,condition=requests[i]['condition'],root_slot=requests[i]['root_slot'],rgb=rgb,normalized_proprio=prop,packets=acts) for i in range(164)]
    monkeypatch.setattr(m,'host_inputs',lambda _:iter(items))
    patch3=np.broadcast_to(np.float32(0),(3,256,384));patch2=patch3[:2]
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
            self.models['predictor'](Tensor(patch3.reshape(1,3,1,16,16,384)),Tensor(acts[None,:3]),Tensor(np.broadcast_to(propout,(1,3,256,16))))
            # Explicit causal rolling fixture; not a model execution.
            vis,pr=m.rollover(patch3,propout,(full,None,np.broadcast_to(propout,(1,3,256,16))))
            self.models['predictor'](Tensor(vis),Tensor(acts[None,1:]),Tensor(pr))
            return (self.trace.calls[-2]['outputs']['$tuple'][0]['$array'],self.trace.calls[-1]['outputs']['$tuple'][0]['$array'])
        def label(self,index,rgb):
            mutate['label']=True;self.trace.where=(index,'label');self.models['official_encoder'].out={'x_norm_patchtokens':Tensor(patch2)}
            self.models['official_encoder'](Tensor(m.encoder_reference(rgb)),is_training=True)
        def receipt(self):return actual
        def memory(self):return dict(cuda_peak_allocated_bytes=1,cuda_peak_reserved_bytes=2,cuda_free_bytes=3,cuda_total_bytes=4,host_peak_rss_bytes=5)  # protocol fixture, not real measurement
    monkeypatch.setattr(m,'CudaBackend',Backend)
    output=work/'results'/(m.NAME+'_full_RAM')
    def labels(q,path,sha):
        events=list(m.journal(output/'journal.jsonl'))
        assert sum(e['kind']=='module_dispatch_intent' for e in events)>=656
        assert next(i for i,e in enumerate(events) if e['kind']=='predictions_sealed')<next(i for i,e in enumerate(events) if e['kind']=='label_decode_intent')
        return [dict(index=i,rgb=rgb[:2]) for i in range(164)]
    monkeypatch.setattr(m,'labels_after_seal',labels)
    result=m.run('fixture_prepared','fixture_sha',output,m.AUTH)
    assert result['status']=='fixed164_root10_open_loop_H2_recorded' and mutate['label']
    strict=m.consume(output,result['sha256'])
    assert strict['status']=='all164_open_loop_H2_and80_pairs_verified' and strict['counts']['returned']==m.protocol()['caps']
    r=m.bridge.read(output/'report.json',result['sha256'])
    # Independent saved-consumer rejects both visual and proprio future substitution.
    # This changes only explicit RAM fixture lookup, never any production file.
    second_input_records=m.array_records(r['calls'][3]['inputs'])
    for rec in (second_input_records[0],second_input_records[2]):
        key=(str(output),rec['file']);original=arrays[key]
        arrays[key]=np.full(original.shape,99,np.float32)
        with pytest.raises(ValueError,match='predicted15'):m.consume(output,result['sha256'])
        arrays[key]=original
    r['rows'][8]['score']['successor_MSE']=1.
    (output/'report.json').write_bytes(m.raw(r))
    with pytest.raises(ValueError):m.consume(output,m.sha(output/'report.json'))


def test_rollover_uses_predicted_visual_and_proprio_not_actual_future():
    patch=np.stack([np.full((256,384),v,np.float32) for v in (0,5,10)])
    prop=np.array([0,5,10],np.float32).reshape(1,3,1,1)*np.ones((1,3,1,16),np.float32)
    predicted=np.full((1,3,256,384),15,np.float32)
    predicted_prop=np.full((1,3,256,16),150,np.float32)
    visual,p=m.rollover(patch,prop,(predicted,None,predicted_prop))
    assert np.array_equal(visual[:,0],patch[1].reshape(1,1,16,16,384))
    assert (visual[:,2]==15).all() and (p[:,0]==5).all() and (p[:,1]==10).all() and (p[:,2]==150).all()
    with pytest.raises(ValueError):m.rollover(patch,prop,(predicted,None,None))
    with pytest.raises(ValueError):m.rollover(patch,prop,(predicted,None,predicted_prop[:,:1]))


def test_four_true_packets_no_zero_padding_no_history_rewrite():
    stats=m.bridge.stats_from_source(m.bridge.STATS)
    warm=np.asarray(m.bridge.physical.moving.warmup_commands(),np.float32)
    own=np.arange(20,dtype=np.float32).reshape(10,2)/np.float32(100)
    commands=np.concatenate([warm,own]).astype(np.float64)
    props=np.arange(12,dtype=np.float32).reshape(3,4)
    p,a=m.normalize20(props,commands,stats)
    p0,a0=m.bridge.normalized(props,commands[:15],stats)
    _,a1=m.bridge.normalized(props,commands[5:],stats)
    np.testing.assert_array_equal(p,p0);np.testing.assert_array_equal(a[:3],a0);np.testing.assert_array_equal(a[1:],a1)
    assert a.shape==(4,10) and np.any(a[:2]!=0)
    with pytest.raises(ValueError):m.normalize20(props,commands[:15],stats)


def test_input_linkage_is_bytes_not_equal_values_or_tolerance():
    p=np.array([0.,1.],np.float32);n=np.array([-0.,1.],np.float32)
    assert np.array_equal(p,n) and not m.exact_array(p,n)
    assert m.exact_array(p,p.copy()) and not m.exact_array(p,p.astype(np.float64))
    assert not m.exact_array(p,p.reshape(1,2))


def test_h2_request_selects_real_warmup0_not_postbind_plan0(tmp_path,monkeypatch):
    stats=m.bridge.stats_from_source(m.bridge.STATS)
    plan=dict(index=0,root_slot=0,condition='off',task_id='fixture',selection_digest='fixture',commands=np.zeros((10,2),np.float32).tolist())
    frames=[]
    for phase,step in [('warmup',0),('warmup',5),('warmup',10),('plan',0),('plan',5),('plan',10)]:
        state=np.array([1,2,3,4,5,6,7],np.float64)
        file=tmp_path/f'{phase}_{step}.png';file.write_bytes(b'unit fixture bytes not production PNG')
        frames.append(dict(phase=phase,step=step,overlay_index=0,condition='off',state=state.tolist(),
            snapshot=dict(state=state.tolist(),state_float64_bits=state.tobytes().hex(),
                bodies=dict(agent=dict(position=[1,2],velocity=[6,7]))),
            png=file.name,file_sha256=m.sha(file),pixel_sha256='unit-fixture'))
    row=dict(trajectory=frames)
    monkeypatch.setattr(m.bridge,'request',lambda *args:{k:plan[k] for k in ('index','root_slot','condition','task_id','selection_digest')})
    q=m.h2_request(row,plan,tmp_path,stats)
    assert [(x['phase'],x['step']) for x in q['images']]==[('warmup',0),('warmup',5),('warmup',10)]
    assert [(x['phase'],x['step']) for x in q['labels']]==[('plan',5),('plan',10)]
    assert q['raw_proprio']==[[1.,2.,6.,7.]]*3 and q['context_global_frames']==[0,5,10]
    row['trajectory']=[f for f in frames if (f['phase'],f['step'])!=('warmup',0)]
    with pytest.raises(ValueError):m.h2_request(row,plan,tmp_path,stats)
    row['trajectory']=frames+[frames[0]]
    with pytest.raises(ValueError):m.h2_request(row,plan,tmp_path,stats)


def test_host_inputs_never_decodes_actual15_or20(monkeypatch):
    stats=m.bridge.stats_from_source(m.bridge.STATS);p=np.zeros((3,4),np.float32);a=np.zeros((20,2),np.float64)
    prop,pack=m.normalize20(p,a,stats)
    req=dict(index=0,condition='off',root_slot=0,context_global_frames=[0,5,10],
        images=[dict(phase='warmup',step=k) for k in (0,5,10)],labels=[{'FORBIDDEN':15},{'FORBIDDEN':20}],
        raw_proprio=p.tolist(),commands=a.tolist(),normalized_proprio=prop.tolist(),packets=pack.tolist())
    seen=[]
    def pixels(rec):
        assert 'FORBIDDEN' not in rec;seen.append(rec['step']);return np.zeros((224,224,3),np.uint8)
    monkeypatch.setattr(m.bridge,'pixels',pixels)
    result=next(m.host_inputs(dict(schema=m.NAME+'.inputs',requests=[req]*164,stats=stats)))
    assert seen==[0,5,10] and set(result)=={'index','condition','root_slot','rgb','normalized_proprio','packets'}


def test_real328_full_npy_seal_before_first_label_decode(work,monkeypatch):
    # Identical fixture bytes hardlinked to328 distinct paths; no neural result claimed.
    directory=work/'results'/'seal_fixture';directory.mkdir()
    base=directory/'fixture.npy';np.save(base,np.zeros((1,3,256,384),np.float32),allow_pickle=False)
    h=m.sha(base);records=[]
    for i in range(328):
        path=directory/f'prediction_{i}.npy';os.link(base,path)
        records.append(dict(index=i//2,step=(15,20)[i%2],path=str(path),sha256=h))
    q=dict(requests=[dict(labels=[dict(step=15),dict(step=20)]) for i in range(164)])
    seal=dict(schema=m.NAME+'.prediction_seal',prepared_semantic_sha256=m.digest(q),predictions=records)
    path=directory/'seal.json';path.write_bytes(m.raw(seal))
    seen=[]
    monkeypatch.setattr(m.bridge,'pixels',lambda rec:seen.append(rec['step']) or np.zeros((2,2,3),np.uint8))
    values=m.labels_after_seal(q,path,m.sha(path))
    assert len(values)==164 and seen==[15,20]*164
    seen.clear();records[-1]['sha256']='bad';path.write_bytes(m.raw(seal))
    with pytest.raises(ValueError):m.labels_after_seal(q,path,m.sha(path))
    assert not seen
    records[-1]['sha256']=h;records[-1]['step']=15;path.write_bytes(m.raw(seal))
    with pytest.raises(ValueError):m.labels_after_seal(q,path,m.sha(path))
    assert not seen

