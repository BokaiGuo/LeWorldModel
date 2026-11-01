"""RAM arithmetic/hooks + real temp-file Store codec; never torch/env execution."""
from copy import deepcopy
import hashlib
import io
import json
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_latest_slot_factorial_cuda_20261029_v1 as m


@pytest.fixture
def work(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:NS(free=8*1024**3))
    return tmp_path


def store(work,suffix='fixture'):return m.Store(work/'results'/(m.NAME+'_'+suffix))


def test_no_neural_or_env_import():
    assert not any(k in sys.modules for k in ('torch','pygame','pymunk','cv2','stable_worldmodel'))


def test_fixed_scope_and_resource_arithmetic():
    p=m.protocol();assert p['branches']==164 and len(p['pairs'])==80
    assert len(m.schedule())==492 and m.schedule()[0]==(0,m.ARMS[0]) and m.schedule()[-1]==(163,m.ARMS[-1])
    assert p['alignment']['history_global_frames']==[5,10,15]
    assert p['caps']['predictor']==dict(api=492,batch_samples=492,frame_samples=1476)
    for k in ('official_encoder','proprio_encoder'):assert p['caps'][k]==dict(api=0,batch_samples=0,frame_samples=0)
    bytes_per_call=4*(np.prod(m.SHAPES[0])+np.prod(m.SHAPES[1])+np.prod(m.SHAPES[2])+3*256*384+3*256*16)
    assert bytes_per_call*492==1209198240<m.TENSOR_CAP<m.CAP
    assert not p['physical_counterfactual_certified'] and not p['P1_admission'] and not p['RGB_only']
    assert p['exploratory_post_outcome']


def test_real_store_two_events_reader_full_chain(work):
    s=store(work);s.emit('first',text='literal\nnewline');s.emit('second',value=[1,2])
    body=(s.path/'journal.jsonl').read_bytes();events=list(m.h2.journal(s.path/'journal.jsonl'))
    assert body.count(b'\n')==2 and b'\n\n' not in body
    assert len(events)==2 and events[1]['previous_sha256']==events[0]['event_sha256']==m.digest({k:v for k,v in events[0].items() if k!='event_sha256'})
    assert s.previous==events[-1]['event_sha256'] and s.hashes['journal.jsonl'].hexdigest()==hashlib.sha256(body).hexdigest()
    assert all(m.raw(e)==line for e,line in zip(events,body.splitlines(keepends=True),strict=True))


@pytest.mark.parametrize('change',('blank','extraLF','missingLF','whitespace','duplicate','chain','nonfinite'))
def test_journal_strict_no_repair(work,change):
    s=store(work);s.emit('one');s.emit('two');p=s.path/'journal.jsonl';b=p.read_bytes()
    if change=='blank':b=b'\n'+b
    elif change=='extraLF':b=b.replace(b'\n',b'\n\n',1)
    elif change=='missingLF':b=b[:-1]
    elif change=='whitespace':b=b.replace(b'{',b'{ ',1)
    elif change=='duplicate':b=b.replace(b'"sequence":0',b'"sequence":0,"sequence":0',1)
    elif change=='chain':b=b.replace(b'"sequence":1',b'"sequence":9',1)
    else:b=b.replace(b'"sequence":0',b'"sequence":NaN',1)
    p.write_bytes(b)
    with pytest.raises((ValueError,KeyError)):list(m.h2.journal(p))


def test_real_put_read_metadata_and_exclusive(work):
    s=store(work);v=dict(status='fixture',z='x\ny',a=[1,2]);rec=s.put('one.json',v)
    assert m.bridge.read(s.path/rec['file'],rec['sha256'])==v and (s.path/rec['file']).read_bytes()==m.raw(v)
    p=work/'results'/(m.NAME+'_prepare_fixture.json');out=m.write_new(p,v)
    assert m.bridge.read(p,out['sha256'])==v and p.read_bytes().count(b'\n')==1
    with pytest.raises(ValueError):m.write_new(p,v)
    with pytest.raises(ValueError):m.Store(s.path)


def test_real_array_tree_and_eof(work):
    s=store(work);a=np.arange(24,dtype=np.float32).reshape(2,3,4)
    tree=m.h2.tree(((a,a.copy()),{'none':None}),s.array,lambda x:isinstance(x,np.ndarray))
    out=m.h2.restore(tree,lambda r:m.load_array(s.path,r))
    assert m.h2.exact_array(out[0][0],a) and out[1]=={'none':None} and s.tensor_bytes==a.nbytes*2
    rec=s.array(a);p=s.path/rec['file'];p.write_bytes(p.read_bytes()+b'trailing');rec['file_bytes']=p.stat().st_size;rec['sha256']=m.sha(p)
    with pytest.raises(ValueError,match='trailing'):m.load_array(s.path,rec)


@pytest.mark.parametrize('change',('sha','shape','dtype','bytes','path','nan'))
def test_array_mutations(work,change):
    s=store(work);rec=s.array(np.ones((2,3),np.float32))
    if change=='sha':rec['sha256']='0'*64
    elif change=='shape':rec['shape']=[6]
    elif change=='dtype':rec['dtype']='float64'
    elif change=='bytes':rec['file_bytes']+=1
    elif change=='path':rec['file']='../bad.npy'
    else:
        p=s.path/rec['file'];buf=io.BytesIO();np.save(buf,np.full((2,3),np.nan,np.float32),allow_pickle=False);p.write_bytes(buf.getvalue());rec['sha256']=m.sha(p)
    with pytest.raises(ValueError):m.load_array(s.path,rec)


def test_store_caps_floor_wall_and_no_truncation(work,monkeypatch):
    s=store(work);s.started-=m.WALL+1
    with pytest.raises(ValueError):s.emit('too_late')
    s.emit('failure',final=True)
    s.started=m.time.monotonic();s.bytes=m.CAP-m.FINAL
    with pytest.raises(ValueError):s.put('cap.json',{})
    s.bytes=0;s.tensor_bytes=m.TENSOR_CAP
    with pytest.raises(ValueError):s.array(np.ones(1,np.float32))
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:NS(free=m.FLOOR))
    with pytest.raises(ValueError):s.check(1)
    with pytest.raises(ValueError):s.write('../bad',b'bad')


@pytest.fixture
def aligned(work):
    s=store(work,'upstream')
    def rec(a):return {'$array':s.array(a)}
    aargs=[rec(np.ones(shape,np.float32)) for shape in m.SHAPES]
    pargs=[rec(np.full(shape,2,np.float32)) for shape in m.SHAPES]
    pargs[1]=deepcopy(aargs[1])
    def call(args):return dict(module='predictor',phase='prediction',status='returned',branch=0,
        units=dict(api=1,batch_samples=1,frame_samples=3),inputs={'$tuple':[{'$tuple':args},{'$dict':{}}]},
        outputs={'$tuple':[rec(np.ones((1,3,256,384),np.float32)),None,rec(np.ones((1,3,256,16),np.float32))]})
    ca,cp=call(aargs),call(pargs)
    identity=dict(index=0,root_slot=0,condition='off',task_id='task',selection_digest='selection')
    label=dict(phase='plan',step=10,condition='off',overlay_index=0,file_sha256='label')
    a=dict(identity,context_global_frames=[5,10,15],successor_global_frame=20,commands=np.ones((15,2)).tolist(),
        packets=np.ones((3,10)).tolist(),label=label)
    p=dict(identity,context_global_frames=[0,5,10],successor_global_frames=[15,20],commands=np.ones((20,2)).tolist(),
        packets=np.ones((4,10)).tolist(),labels=[{'step':5},deepcopy(label)])
    ra=dict(identity,prediction=ca['outputs']['$tuple'][0]['$array'],label_patch=s.array(np.zeros((1,256,384),np.float32)))
    rp=dict(identity,prediction=cp['outputs']['$tuple'][0]['$array'],label_patch=s.array(np.zeros((2,256,384),np.float32)))
    return [0,a,p,ra,rp,ca,cp,s.path,s.path]


def test_alignment_latest_only_all_three_corners_first2_bitwise_actual(aligned):
    q=m.aligned_request(*aligned);actual,rolled=m.source_blocks(q)
    for arm,cell in zip(m.ARMS,('AP','PA','PP'),strict=True):
        args=m.mixed_inputs(q,arm)
        assert m.h2.exact_array(args[0][:,:2],actual[0][:,:2])
        assert m.h2.exact_array(args[2][:,:2],actual[2][:,:2])
        assert m.h2.exact_array(args[1],actual[1])
        assert m.h2.exact_array(args[0][:,2:3],(actual if cell[0]=='A' else rolled)[0][:,2:3])
        assert m.h2.exact_array(args[2][:,2:3],(actual if cell[1]=='A' else rolled)[2][:,2:3])
        witness=m.input_witness(q,arm,args)
        assert witness['locked_global_frames']==[5,10] and witness['crossed_global_frame']==15 and witness['latest_cell']==cell
    # New PP is NOT the old whole-history PP input.
    newpp=m.mixed_inputs(q,m.ARMS[2])
    assert not m.h2.exact_array(newpp[0],rolled[0]) and not m.h2.exact_array(newpp[2],rolled[2])


@pytest.mark.parametrize('change',('time','target','root','task','selection','condition','label','callbranch','shape','kwargs','raw','action'))
def test_cross_join_rejects(aligned,change):
    v=deepcopy(aligned);p=v[2]
    if change=='time':p['context_global_frames']=[5,10,15]
    elif change=='target':p['successor_global_frames']=[10,15]
    elif change=='root':p['root_slot']=1
    elif change=='task':p['task_id']='other'
    elif change=='selection':p['selection_digest']='other'
    elif change=='condition':p['condition']='on'
    elif change=='label':p['labels'][1]['file_sha256']='other'
    elif change=='callbranch':v[6]['branch']=1
    elif change=='shape':v[6]['inputs']['$tuple'][0]['$tuple'][0]['$array']['shape']=[1,3,256,384]
    elif change=='kwargs':v[6]['inputs']['$tuple'][1]={'$dict':{'future':True}}
    elif change=='raw':p['commands'][-1][0]=.25
    else:p['packets'][-1][0]=.25
    with pytest.raises(ValueError):m.aligned_request(*v)


def test_merge_rejects_conflicting_authority():
    d={'a':'sha'};m.merge(d,{'a':'sha','b':'other'});assert len(d)==2
    with pytest.raises(ValueError):m.merge(d,{'a':'bad'})


def test_scalar_and_vector_factorial_algebra():
    v={'AA':1.,'AP':3.,'PA':5.,'PP':11.};f=m.factorial(v)
    assert f=={'visual':6.,'proprio':4.,'interaction':4.}
    b=np.arange(6,dtype=np.float32).reshape(2,3);x={c:b+v[c] for c in m.CELLS}
    fx=m.factorial(x)
    assert np.array_equal(fx['visual'],np.full((2,3),6.))
    score=m.row_score(x,b,{'bitwise_equal':True,'maxabs':0.})
    assert score['common_label_cell_MSE']==dict(AA=1.,AP=9.,PA=25.,PP=121.)
    assert score['MSE_effects']==dict(visual=68.,proprio=52.,interaction=88.)
    with pytest.raises(ValueError):m.factorial({'AA':1.})


def test_seal_missing_reordered_or_unsealed_rejected():
    rows=m.empty_rows()
    with pytest.raises(ValueError):m.seal_value({'fixture':1},rows)
    for r in rows:r['predictions']={a:dict(file=f'{r["index"]}_{a}.npy') for a in m.ARMS}
    s=m.seal_value({'fixture':1},rows)
    assert len(s['predictions'])==492 and s['prepared_semantic_sha256']==m.digest({'fixture':1})
    rows[0],rows[1]=rows[1],rows[0]
    with pytest.raises(ValueError):m.seal_value({},rows)


def test_label_gate_checks_seal_before_any_label_read(work,monkeypatch):
    s=store(work);rows=m.empty_rows();touched=[]
    rec=s.put('predictions_seal.json',dict(schema='wrong'))
    monkeypatch.setattr(m,'load_ref',lambda r:touched.append(r))
    with pytest.raises(ValueError):m.unlock({},rows,s.path,rec)
    assert touched==[]


class Tensor:
    def __init__(self,shape,value=None):self.shape=shape;self.value=value;self.is_cuda=True;self.dtype='float32';self.requires_grad=False
    def detach(self):return self
    def contiguous(self):return self
    def cpu(self):return self
    def numpy(self):return np.zeros(self.shape,np.float32) if self.value is None else self.value


class Handle:
    def remove(self):return None


class Model:
    def register_forward_pre_hook(self,fn,with_kwargs):self.pre=fn;return Handle()
    def register_forward_hook(self,fn,with_kwargs):self.post=fn;return Handle()


def fake_torch():
    return NS(float32='float32',is_grad_enabled=lambda:False,is_tensor=lambda x:isinstance(x,Tensor),
        isfinite=lambda x:NS(all=lambda:True),backends=NS(cuda=NS(matmul=NS(allow_tf32=False)),cudnn=NS(allow_tf32=False)))


def test_native_hooks_real_store_record_all_io_and_actual_counts(work):
    s=store(work);t=m.Trace(s);models={k:Model() for k in m.h2.KEYS};t.install(models,fake_torch());t.where=m.schedule()[0]
    args=tuple(Tensor(shape) for shape in m.SHAPES);model=models['predictor']
    model.pre(model,args,{});assert t.counts['attempted']['predictor']['api']==1 and t.counts['returned']['predictor']['api']==0
    model.post(model,args,{},(Tensor((1,3,256,384)),None,Tensor((1,3,256,16))))
    assert t.counts['returned']['predictor']==dict(api=1,batch_samples=1,frame_samples=3)
    e=list(m.h2.journal(s.path/'journal.jsonl'));assert [x['kind'] for x in e]==['module_dispatch_intent','module_inputs_saved','module_returned','module_outputs_saved']
    back=m.h2.restore(t.calls[0]['inputs'],lambda r:m.load_array(s.path,r))
    assert len(back[0])==3 and back[1]=={} and s.array_index==5 and s.tensor_bytes==2457720
    assert t.close()==[]
    # A repeat without advancing the arm is refused BEFORE another forward.
    with pytest.raises(ValueError):model.pre(model,args,{})


@pytest.mark.parametrize('kind',('official_encoder','proprio_encoder','wrongarm','kwargs','grad','tf32'))
def test_hooks_reject_extra_module_and_wrong_context(work,kind):
    s=store(work);t=m.Trace(s);models={k:Model() for k in m.h2.KEYS};torch=fake_torch();t.install(models,torch);t.where=m.schedule()[0]
    if kind=='wrongarm':t.where=(0,m.ARMS[1])
    if kind=='grad':torch.is_grad_enabled=lambda:True
    if kind=='tf32':torch.backends.cuda.matmul.allow_tf32=True
    model=models[kind] if kind in models else models['predictor']
    with pytest.raises(ValueError):model.pre(model,tuple(Tensor(s) for s in m.SHAPES),{'future':True} if kind=='kwargs' else {})
    assert t.calls==[] and t.counts==m.h2.empty_counts()


def test_wrong_authorization_no_store_or_backend(work,monkeypatch):
    monkeypatch.setattr(m,'CudaBackend',lambda *args:pytest.fail('backend must not load'))
    out=work/'results'/(m.NAME+'_not_created')
    with pytest.raises(ValueError):m.run('unused','bad',out,'no')
    assert not out.exists()


def test_preflight_failure_keeps164_and80_and_partial_strict(work,monkeypatch):
    def fail(*args):raise ValueError('fixture failed authority')
    monkeypatch.setattr(m,'verify',fail)
    monkeypatch.setattr(m,'CudaBackend',lambda *args:pytest.fail('no model before verified closure'))
    out=work/'results'/(m.NAME+'_partial');answer=m.run('unused','bad',out,m.AUTH)
    assert answer['status']=='partial_or_failed'
    r=m.bridge.read(out/'report.json',answer['sha256'])
    assert len(r['rows'])==164 and len(r['pairs'])==80 and r['calls']==[] and r['counts']==m.h2.empty_counts()
    assert r['errors'][0]['type']=='ValueError'
    receipt=m.consume(out,answer['sha256']);assert receipt['status']=='blocked_partial_capture' and not receipt['P5_admission']


def test_complete492_fixture_real_writer_seal_score_strict_and_mutation(work,aligned,monkeypatch):
    """Synthetic full-size IO only, no torch/model. All actual Store/reader/strict code."""
    q=m.aligned_request(*aligned);requests=[]
    for row in m.empty_rows():requests.append(dict(deepcopy(q),**{k:row[k] for k in ('index','root_slot','condition')}))
    source=work/'fixture_source.py';source.write_text('# test fixture, not native model\n')
    models={k:dict(state_sha256='test_fixture_not_model',eval=True,frozen=True,all_cuda_finite=True) for k in m.h2.KEYS}
    manifest=dict(requests=requests,expected_model=models,expected_strict_loads={'fixture':'not model load'},bindings={str(source):m.sha(source)},
        inherited_costs={'fixture':'synthetic, not research evidence'})
    monkeypatch.setattr(m,'verify',lambda *args:manifest)
    barriers=[];real_features=m.features
    def guarded_features(req,row,directory):
        if not barriers:
            events=list(m.h2.journal(directory/'journal.jsonl'))
            assert any(e['kind']=='labels_unlocked' for e in events)
            seal=m.bridge.read(directory/'predictions_seal.json')
            assert len(seal['predictions'])==492
        barriers.append(req['index']);return real_features(req,row,directory)
    monkeypatch.setattr(m,'features',guarded_features)
    class Backend:
        def __init__(self,manifest,trace):
            self.before=deepcopy(models);self.strict=manifest['expected_strict_loads'];self.trace=trace
            self.sources={k:str(source) for k in ('app.plan_common.models.AdaLN_vit','app.plan_common.models.prop_embedding','dinov2.models.vision_transformer')}
            self.modules={k:Model() for k in m.h2.KEYS};trace.install(self.modules,fake_torch())
        def predict(self,index,arm,args):
            self.trace.where=(index,arm);model=self.modules['predictor'];ts=tuple(Tensor(x.shape,x) for x in args)
            model.pre(model,ts,{})
            model.post(model,ts,{},(Tensor((1,3,256,384)),None,Tensor((1,3,256,16))))
            return self.trace.calls[-1]['outputs']['$tuple'][0]['$array']
        def receipt(self):return deepcopy(self.before)
        def memory(self):return dict(cuda_peak_allocated_bytes=0,cuda_peak_reserved_bytes=0,cuda_free_bytes=0,cuda_total_bytes=0,host_peak_rss_bytes=0)
    monkeypatch.setattr(m,'CudaBackend',Backend)
    out=work/'results'/(m.NAME+'_synthetic_full')
    result=m.run('fixture_prepare','fixture_sha',out,m.AUTH)
    assert result['status']=='all164_latest_slot_factorial_recorded'
    assert barriers and len(barriers)==164+80*2
    receipt=m.consume(out,result['sha256'])
    assert receipt['status']=='all164_latest_slot_factorial_and80_pairs_verified'
    assert receipt['new_predictor_calls']==492 and receipt['counts']['returned']['official_encoder']['api']==0
    report=m.bridge.read(out/'report.json',result['sha256'])
    assert report['actual_tensor_io_bytes']==1209198240 and len(report['pairs'])==80
    # Keep all bytes intact; alter an actual recorded input association in report only.
    report['calls'][0]['inputs']=deepcopy(report['calls'][1]['inputs'])
    p=out/'report.json';p.write_bytes(m.raw(report))
    with pytest.raises(ValueError):m.consume(out,m.sha(p))


@pytest.mark.parametrize('change',('visual5','visual10','proprio5','proprio10','visual15','proprio15','action'))
def test_actual_prefix_or_selected15_one_bit_drift_rejected(aligned,change):
    q=m.aligned_request(*aligned);a,p=m.source_blocks(q);args=list(m.mixed_inputs(q,m.ARMS[2]))
    if change=='action':component,slot=1,2
    elif change.startswith('visual'):component,slot=0,{'5':0,'10':1,'15':2}[change[6:]]
    else:component,slot=2,{'5':0,'10':1,'15':2}[change[7:]]
    view=args[component][:,slot:slot+1]
    view.flat[0]=np.nextafter(view.flat[0],np.float32(np.inf),dtype=np.float32)
    with pytest.raises(ValueError,match='exact AA first2'):
        m.latest_witness(a,p,tuple(args),m.ARMS[2])


def test_splice_preserves_signed_zero_and_does_not_mutate_sources():
    a=tuple(np.zeros(s,np.float32) for s in m.SHAPES)
    p=tuple(np.full(s,2,np.float32) for s in m.SHAPES);p=(p[0],a[1].copy(),p[2])
    a[0][:,:2]=-np.float32(0.);a[2][:,:2]=-np.float32(0.)
    original=[x.tobytes() for x in (*a,*p)]
    args=m.splice_inputs(a,p,m.ARMS[2])
    assert m.h2.exact_array(args[0][:,:2],a[0][:,:2]) and np.signbit(args[0][:,:2]).all()
    assert m.h2.exact_array(args[2][:,:2],a[2][:,:2]) and np.signbit(args[2][:,:2]).all()
    assert original==[x.tobytes() for x in (*a,*p)]


def test_old_whole_history_arm_and_wrong_action_refused(aligned):
    q=m.aligned_request(*aligned);a,p=m.source_blocks(q)
    with pytest.raises(ValueError):m.splice_inputs(a,p,'rolled_visual_observed_proprio')
    wrong=list(p);wrong[1]=p[1].copy();wrong[1][0,-1,0]+=1
    with pytest.raises(ValueError):m.splice_inputs(a,wrong,m.ARMS[2])

