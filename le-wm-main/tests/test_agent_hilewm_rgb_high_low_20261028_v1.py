"""Non-neural RAM buffers/metadata only; no solver, torch, physics or pickle."""
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import agent_hilewm_rgb_high_low_20261028_v1 as m


def layout(x):
    return dict(shape=list(x.shape),stride=[s//x.itemsize for s in x.strides],storage_offset=0,
        contiguous=bool(x.flags.c_contiguous),device='cuda:0',dtype='torch.float32')


def fp(x,expanded=False):
    s=[v//x.itemsize for v in x.strides]
    if expanded:s[1]=0
    return dict(shape=list(x.shape),dtype=x.dtype.str,bytes=x.nbytes,sha256=m.digest(x.tobytes()),
        stride=s,device='cuda:0',requires_grad=False)


STATS=dict(mean=[.485,.456,.406],std=[.229,.224,.225])


def literal_row(index):
    """Invented RAM values only, not a neural surrogate or research asset."""
    rng=np.random.default_rng(index);arrays={};row=dict(index=index,root_slot=m.ORDER[index][0],condition=m.ORDER[index][1],status='returned',events=[])
    packet=dict(rgb_history=rng.integers(0,256,(3,224,224,3),np.uint8),goal_rgb=rng.integers(0,256,(224,224,3),np.uint8))
    def save(a):
        name=f'R{index}_{len(arrays):05d}.npy';arrays[name]=np.asarray(a).copy();return name
    def begin(kind,args,parent=None):
        e=dict(index=len(row['events']),parent=parent,kind=kind,inputs=[save(x) for x in args],layouts=[layout(x) for x in args],returned=True)
        row['events'].append(e);return e
    def end(e,a):e.update(output=save(a),output_layout=layout(a));return a
    def leaf(kind,args,out,parent):return end(begin(kind,args,parent),out)
    zs=[];row['pixels']=[]
    for role,rgb in (('root',packet['rgb_history'][-1]),('goal',packet['goal_rgb'])):
        pre=m.preprocess(rgb,STATS);row['pixels'].append(dict(role=role,preprocessed=save(pre)))
        enc=begin('encoder',[pre]);leaf('pixel.patch',[pre],np.zeros((1,192,16,16),np.float32),enc['index'])
        previous=np.full((1,257,192),.1,np.float32)
        for j in range(12):previous=leaf(f'pixel.layer{j}',[previous],np.full_like(previous,.2+j/100),enc['index'])
        hidden=leaf('pixel.finalnorm',[previous],previous+np.float32(.125),enc['index']);end(enc,hidden)
        z=leaf('projector',[hidden[:,0]],np.full((1,192),.1 if role=='root' else .8,np.float32),None);zs.append(z)
    z,g=zs;row['z_init']=save(z);row['z_goal']=save(g)
    def high(z,c,parent,samples,steps):
        history=np.repeat(z[:,None],samples,axis=1).reshape(samples,1,192);acts=c.reshape(1,samples,steps,32);pred=[]
        for step in range(steps):
            seq=np.stack([acts[:,:,k].reshape(samples,32) for k in range(step+1)],axis=1)
            cond=leaf('macro_to_condition',[seq],np.zeros((samples,step+1,192),np.float32),parent)
            h=leaf('high_predictor',[history,cond],rng.uniform(-.5,.5,(samples,step+1,192)).astype(np.float32),parent)
            p=leaf('high_pred_proj',[h.reshape(-1,192)],h.reshape(-1,192).copy(),parent)
            nxt=p.reshape(samples,step+1,192)[:,-1:];history=np.concatenate((history,nxt),axis=1);pred.append(nxt[:,0])
        return np.stack(pred,axis=1).reshape(1,samples,steps,192)
    witness=dict(original_keys=['planner_level','z_init','z_goal'],original_metadata=dict(planner_level='high'),solver_keys=['z_init','z_goal'],
        original_tensors=dict(z_init=fp(z),z_goal=fp(g)),solver_tensors=dict(z_init=fp(z),z_goal=fp(g)))
    row.update(cem=[],cem_reset=1,cem_start=1,cem_end=1,cem_history=[dict(step=i) for i in range(3)],hooks_removed=6)
    mean=np.zeros((1,2,64),np.float32);std=np.ones_like(mean);row['high_initial']=save(mean);calls=[]
    for i in range(3):
        c=rng.normal(size=(1,8,2,64)).astype(np.float32)*std[:,None]+mean[:,None];c[:,0]=mean
        ez=np.repeat(z[:,None],8,axis=1);eg=np.repeat(g[:,None],8,axis=1)
        e=begin('high_cost',[ez,eg,c]);roll=high(z,c,e['index'],8,4)
        cost=np.square(roll[:,:,-1]-eg).sum(axis=-1,dtype=np.float32);end(e,cost)
        ix=np.argsort(cost,axis=1)[:,:3].astype(np.int64);elite=c[np.arange(1)[:,None],ix];vals=cost[np.arange(1)[:,None],ix]
        newmean=elite.mean(axis=1);newstd=elite.std(axis=1,ddof=1)
        row['cem'].append(dict(step=i,arrays={k:save(v) for k,v in dict(candidates=c,costs=cost,topk_vals=vals,topk_inds=ix,
            topk_candidates=elite,mean=newmean,var=newstd,prev_mean=mean,prev_var=std).items()}))
        calls.append(dict(attempt=i+1,batch=1,candidate_rows=8,metadata='high',returned=True,candidates=fp(c),cost=fp(cost),
            expanded=dict(z_init=fp(ez,True),z_goal=fp(eg,True))))
        mean,std=newmean,newstd
    row['high_outputs']={k:save(v) for k,v in dict(actions=mean,mean=mean,var=std,costs=vals.mean(axis=1)).items()}
    macro=mean[:,:1].reshape(1,2,32);row['macro_sequence']=save(macro)
    e=begin('high_selected_rollout',[z,macro]);roll=high(z,macro,e['index'],1,2);end(e,roll)
    target=roll[:,0,0,:];row['selected_rollout']=save(roll);row['high_target']=save(target)
    row['bridge_receipt']=dict(schema=m.seam.bridge_module().TAG,backend='actual_torch_CUDA_float32',tensor_batch=1,input_witness=witness,
        cost_attempts=3,cost_returns=3,cost_failures=0,candidate_rows_attempted=24,calls=calls,predict_API_calls=None,predictor_samples=None,
        scientific_admission=False,official_unadapted_reproduction=False)
    zh=z[:,None];ah=np.zeros((1,1,10),np.float32);c=m.low_panel();e=begin('low_cost',[zh,ah,target,c])
    hist=np.repeat(zh[:,None],3,axis=1).reshape(3,1,192);a=np.zeros((3,1,10),np.float32);pred=[]
    for step in range(2):
        a=np.concatenate((a,c[:,:,step].reshape(3,1,10)),axis=1)
        emb=leaf('action_encoder',[a],np.zeros((3,step+2,192),np.float32),e['index'])
        h=leaf('low_predictor',[hist,emb[:,-(step+1):]],rng.normal(size=(3,step+1,192)).astype(np.float32),e['index'])
        p=leaf('low_pred_proj',[h.reshape(-1,192)],h.reshape(-1,192).copy(),e['index'])
        nxt=p.reshape(3,step+1,192)[:,-1:];hist=np.concatenate((hist,nxt),axis=1);pred.append(nxt[:,0])
    roll=np.stack(pred,axis=1).reshape(1,3,2,192);cost=np.square(roll[:,:,-1]-target[:,None]).sum(axis=-1,dtype=np.float32);end(e,cost)
    row['low']={k:save(v) for k,v in dict(z_hist=zh,a_hist=ah,z_subgoal=target,candidates=c,cost=cost,rollout=roll).items()}
    return row,arrays,packet


@pytest.fixture(scope='module')
def ram():return literal_row(0)


def test_full4_contract_and_fixed_model_coordinate_panel():
    p=m.protocol();assert p['order']==[[0,'off'],[0,'on'],[1,'off'],[1,'on']]
    assert p['new_calls']['high_predict']==56 and p['new_calls']['high_predict_sample_rows']==392
    assert p['new_calls']['encoder']==p['new_calls']['projector']==8
    assert not p['scientific_admission'] and not p['full_policy'] and not p['common_backbone_verified']
    a=m.low_panel();assert a.dtype==np.float32 and a.shape==(1,3,2,10)
    assert np.all(a[:,0]==0) and np.all(a[:,1]==-.25) and np.all(a[:,2]==.25)


def test_full_native_history_first_waypoint_and_cem(ram):
    row,arrays,_=ram;pixels,costs,target=m.validate_events(row,arrays,'cuda:0')
    m.validate_cem(row,arrays,costs,pixels[0][1],pixels[1][1])
    assert target.tobytes()==arrays[row['low']['z_subgoal']].tobytes()
    assert not np.array_equal(target,arrays[row['selected_rollout']][:,0,-1])


@pytest.mark.parametrize('fault',['last_waypoint','high_goal','zero_history','action_history','low_goal','low_actions',
    'low_rollout','low_shift','high_shift','high_history','macro','CLS','197','layer_chain','norm_chain','unknown','drop','parent','return','dtype','device'])
def test_native_mutations_failclosed(ram,fault):
    row,arrays,_=deepcopy(ram)
    def ev(kind):return next(e for e in row['events'] if e['kind']==kind)
    if fault=='last_waypoint':arrays[row['high_target']]=arrays[row['selected_rollout']][:,0,-1].copy()
    elif fault=='high_goal':arrays[ev('high_cost')['inputs'][1]].flat[0]+=.2
    elif fault=='zero_history':arrays[ev('low_cost')['inputs'][0]][:]=0
    elif fault=='action_history':arrays[ev('low_cost')['inputs'][1]][:]=.1
    elif fault=='low_goal':arrays[ev('low_cost')['inputs'][2]].flat[0]+=.2
    elif fault=='low_actions':arrays[ev('low_cost')['inputs'][3]][:]=0
    elif fault=='low_rollout':arrays[row['low']['rollout']].flat[0]+=.1
    elif fault=='low_shift':
        arrays[ev('low_cost')['output']]+=100;arrays[row['low']['cost']]+=100
    elif fault=='high_shift':arrays[ev('high_cost')['output']]+=100
    elif fault=='high_history':arrays[ev('high_predictor')['inputs'][0]].flat[0]+=.1
    elif fault=='macro':arrays[ev('macro_to_condition')['inputs'][0]].flat[0]+=.1
    elif fault=='CLS':arrays[ev('projector')['inputs'][0]].flat[0]+=.1
    elif fault=='197':arrays[ev('encoder')['output']]=np.zeros((1,197,192),np.float32)
    elif fault=='layer_chain':arrays[ev('pixel.layer1')['inputs'][0]].flat[0]+=.1
    elif fault=='norm_chain':arrays[ev('pixel.finalnorm')['inputs'][0]].flat[0]+=.1
    elif fault=='unknown':row['events'].append(deepcopy(row['events'][-1]));row['events'][-1]['kind']='encoder2'
    elif fault=='drop':row['events'].pop()
    elif fault=='parent':ev('low_predictor')['parent']=None
    elif fault=='return':ev('low_cost')['returned']=False
    elif fault=='dtype':arrays[ev('high_cost')['output']]=arrays[ev('high_cost')['output']].astype(np.float64)
    else:ev('high_cost')['output_layout']['device']='cpu'
    with pytest.raises((ValueError,IndexError)):m.validate_events(row,arrays,'cuda:0')


@pytest.mark.parametrize('fault',['topk','indices','std','mean_chain','callback','metadata','expanded_stride','cost_sha','initial','selected_macros','output'])
def test_cem_mutations_failclosed(ram,fault):
    row,arrays,_=deepcopy(ram);pixels,costs,_=m.validate_events(row,arrays,'cuda:0');e=row['cem'][0]['arrays']
    if fault=='topk':arrays[e['topk_vals']].flat[0]+=.1
    elif fault=='indices':arrays[e['topk_inds']].flat[0]=8
    elif fault=='std':arrays[e['var']]+=np.float32(.1)
    elif fault=='mean_chain':arrays[row['cem'][1]['arrays']['prev_mean']]+=np.float32(.1)
    elif fault=='callback':row['cem_end']=0
    elif fault=='metadata':row['bridge_receipt']['calls'][0]['metadata']='low'
    elif fault=='expanded_stride':row['bridge_receipt']['calls'][0]['expanded']['z_init']['stride'][1]=192
    elif fault=='cost_sha':row['bridge_receipt']['calls'][0]['cost']['sha256']='0'*64
    elif fault=='initial':arrays[row['high_initial']].flat[0]=1
    elif fault=='selected_macros':arrays[row['macro_sequence']].flat[0]+=.1
    else:arrays[row['high_outputs']['actions']].flat[0]+=.1
    with pytest.raises((ValueError,IndexError)):m.validate_cem(row,arrays,costs,pixels[0][1],pixels[1][1])


def test_preprocessing_original_scale_normalize_identityresize():
    rgb=np.full((224,224,3),127,np.uint8);out=m.preprocess(rgb,STATS)
    expected=(np.float32(127)*np.float32(1/255)-np.asarray(STATS['mean'],np.float32))/np.asarray(STATS['std'],np.float32)
    assert out.shape==(1,3,224,224) and out.dtype==np.float32 and np.array_equal(out[0,:,0,0],expected)
    with pytest.raises(ValueError):m.preprocess(rgb.astype(float),STATS)
    with pytest.raises(ValueError):m.preprocess(rgb[:197],STATS)


def test_actual_saved_literal_config_regression_no_model():
    # Read only the pinned 6MB inert metadata, no production build or export.
    saved=m.loader.json_read(m.loader.IR_PATH,m.loader.IR_SHA);g=m.loader.graph_spec(saved);ir=m.loader.LiteralIR(saved)
    node=next(x for x in g['modules'] if x['path']==['encoder'])['node']
    config={k:ir.literal(v) for k,v in ir.state(ir.state(node)['config']).items()}
    result=m.config_contract(saved,g,dict(encoder_config=config))
    assert result['config']['image_size']//result['config']['patch_size']==16
    assert result['original_transformers_version'] is None and not result['native_encoder_equivalence']
    config['patch_size']=16
    with pytest.raises(ValueError):m.config_contract(saved,g,dict(encoder_config=config))


def test_authority_rebuilds_both_real_receipts(monkeypatch):
    report=dict(manifest_path='/RAM/manifest.json',manifest_sha256='RAM');receipt=dict(strict=True)
    calls=[];monkeypatch.setattr(m,'read',lambda p,h:deepcopy(report if p in (m.V3_REPORT,m.CEM_REPORT) else receipt))
    monkeypatch.setattr(m,'v3',SimpleNamespace(consume=lambda *a:calls.append('v3') or deepcopy(receipt)))
    monkeypatch.setattr(m,'paid',SimpleNamespace(consume=lambda *a:calls.append('cem') or deepcopy(receipt)))
    assert len(m.authorities())==2 and calls==['v3','cem']
    monkeypatch.setattr(m.paid,'consume',lambda *a:dict(strict=False))
    with pytest.raises(ValueError):m.authorities()


def test_main_authorization_precedes_reads_and_runtime(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('unauthorized read'))
    monkeypatch.setattr(m.seam,'load_runtime',lambda:pytest.fail('unauthorized runtime'))
    with pytest.raises(ValueError):m.run_cuda(None,None,None)


def test_fixed_input_roster_cannot_filter_by_outcome(monkeypatch):
    monkeypatch.setattr(m,'sha',lambda p:m.moving.SOURCE_SHA)
    monkeypatch.setattr(m,'read',lambda *a:dict(never_passed_to_model=True))
    slots=[dict(root_slot=s,condition=c) for s,c in m.ORDER]
    monkeypatch.setattr(m.moving,'packets',lambda *a:deepcopy(slots))
    assert m.inputs()==slots
    slots.pop()
    with pytest.raises(ValueError):m.inputs()


def test_cold_import_no_new_torch_pickle_foreign():
    code='''
import numpy, builtins
original=builtins.__import__
def guarded(name,*a,**k):
    assert name.split('.')[0] not in {'torch','pickle','transformers','torchvision','stable_worldmodel','gymnasium'},name
    return original(name,*a,**k)
builtins.__import__=guarded
from experiments.spectral_reachability import agent_hilewm_rgb_high_low_20261028_v1
'''
    subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,check=True,timeout=30)


def test_source_direct_high_and_explicit_owned_not_monkeypatch():
    text=Path(m.__file__).read_text()
    assert 'lambda:root.get_cost(info,candidates)' in text and 'root.get_cost=' not in text
    assert 'roll[:,0,0,:]' in text and 'lambda:root.rollout_high(z,macro)' in text
    assert 'sdpa_kernel(SDPBackend.MATH)' in text and 'torch.load(' not in text
    assert 'Analytic' not in text and 'CPU_fallback=True' not in text


def test_failures_not_refunded(ram):
    row,_,_=deepcopy(ram);e=next(e for e in row['events'] if e['kind']=='high_cost');e['returned']=False
    actual=m.resource_counts([row])
    assert actual['modules']['high_cost']['attempted']==3 and actual['modules']['high_cost']['returned']==2
    assert actual['bridge_candidate_rows']==24 and actual['physics']==0


def test_full4_saved_consume_and_boundaries(tmp_path,monkeypatch):
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    store=m.Store(tmp_path/'results'/(m.TAG+'_RAM'),m.time.monotonic());rows=[];packets={};slots=[]
    for i in range(4):
        row,arrays,packet=literal_row(i);mapping={name:store.array(a) for name,a in arrays.items()}
        def remap(v):
            if isinstance(v,dict):return {k:remap(a) for k,a in v.items()}
            if isinstance(v,list):return [remap(a) for a in v]
            return mapping.get(v,v) if isinstance(v,str) else v
        rows.append(remap(row));path=f'/RAM/input{i}.npz';packets[path]=packet
        slots.append(dict(root_slot=m.ORDER[i][0],condition=m.ORDER[i][1],blind_path=path,blind_sha256='RAM'))
    export={k:k for k in ('export_report','export_report_sha256','export_receipt','export_receipt_sha256')}
    manifest=dict(sources=dict(RAM='literal'),export=export,statistics=STATS,slots=slots)
    monkeypatch.setattr(m,'verify',lambda *a:deepcopy(manifest))
    monkeypatch.setattr(m.moving,'load_packet',lambda p,h:deepcopy(packets[p]))
    weights=dict(RAM_tensor='RAM_hash')
    monkeypatch.setattr(m.loader,'authorities',lambda *a:(None,dict(entries=[dict(key='RAM_tensor')]),None,
        dict(rows=[dict(array=dict(value_sha256='RAM_hash'))])))
    report=dict(schema=m.TAG+'.report',manifest=str(tmp_path/'manifest.json'),manifest_sha256='RAM',protocol=m.protocol(),
        sources=manifest['sources'],completed=True,errors=[],scientific_admission=False,elapsed_seconds=1.,rows=rows,
        precision_before=m.precision_expected(),precision_after=m.precision_expected(),runtime=dict(device='cuda:0',torch='RAM'),
        model_before=weights,model_after=weights,construction=dict(device='cuda:0',torch='RAM',all_parameters_frozen=True,
            tensor_copy_attempts=425,tensor_copy_returns=425,tensor_value_hashes=weights,
            export_report_sha256=export['export_report_sha256'],export_receipt_sha256=export['export_receipt_sha256'],ir_sha256=m.loader.IR_SHA),
        arrays=store.arrays,resources=m.resource_counts(rows))
    body=m.encode(report);store.put('report.json',body,final=True);rp=store.path/'report.json';rh=m.digest(body)
    args=(tmp_path/'manifest.json','RAM',rp,rh)
    result=m.consume(*args);assert len(result['rows'])==4 and result['resources']['bridge_candidate_rows']==96
    assert all(x['target_delivered_exact'] and x['high_target_sha256']==x['low_goal_sha256'] for x in result['rows'])
    assert not result['physical_release'] and not result['common_backbone_verified']
    # Isolated in-memory report mutations exercise downstream checks against
    # the same literal saved arrays. No production authority is mocked as real.
    original_read=m.read
    for fault in ('resource','partial','model','precision','omit','preprocess'):
        changed=deepcopy(report)
        if fault=='resource':changed['resources']['bridge_candidate_rows']=95
        elif fault=='partial':changed['completed']=False
        elif fault=='model':changed['model_after']['RAM_tensor']='changed'
        elif fault=='precision':changed['precision_before']['flash_sdp']=True
        elif fault=='omit':changed['rows'].pop()
        else:
            bad=deepcopy(packets);bad['/RAM/input0.npz']['goal_rgb'][0,0,0]^=1
            monkeypatch.setattr(m.moving,'load_packet',lambda p,h:deepcopy(bad[p]))
        monkeypatch.setattr(m,'read',lambda p,h=None:deepcopy(changed) if Path(p)==rp else original_read(p,h))
        with pytest.raises(ValueError):m.consume(*args)
    monkeypatch.setattr(m,'read',original_read)
    with (store.path/next(iter(store.arrays))).open('ab') as f:f.write(b'trailing')
    with pytest.raises(ValueError,match='entire NPY SHA'):m.consume(*args)
    store.put('unexpected.npy',b'not-an-array')
    with pytest.raises(ValueError,match='whole saved census'):m.consume(*args)


class Shape:
    """Shape-only tensor mock: no numerical network evaluation."""
    def __init__(self,shape):self.shape=tuple(shape)
    def reshape(self,*shape):
        if len(shape)==1 and isinstance(shape[0],(tuple,list)):shape=shape[0]
        shape=list(shape)
        if -1 in shape:shape[shape.index(-1)]=int(np.prod(self.shape))//int(np.prod([s for s in shape if s!=-1]))
        assert np.prod(shape)==np.prod(self.shape);return Shape(shape)
    def flatten(self,start):return Shape(self.shape[:start]+(int(np.prod(self.shape[start:])),))
    def transpose(self,a,b):
        s=list(self.shape);s[a],s[b]=s[b],s[a];return Shape(s)
    def expand(self,*shape):return Shape([self.shape[i] if s==-1 else s for i,s in enumerate(shape)])
    def __getitem__(self,key):
        key=key if isinstance(key,tuple) else (key,);out=[];index=0
        for k in key:
            if k is None:out.append(1);continue
            s=self.shape[index];index+=1
            if isinstance(k,slice):out.append(len(range(*k.indices(s))))
            else:assert isinstance(k,int)
        return Shape(tuple(out)+self.shape[index:])
    def __add__(self,other):assert self.shape==other.shape or other.shape[0]==1;return Shape(self.shape)
    def __sub__(self,other):return Shape(np.broadcast_shapes(self.shape,other.shape))
    def pow(self,n):assert n==2;return Shape(self.shape)
    def sum(self,dim):return Shape(self.shape[:dim] if dim==-1 else self.shape[:dim]+self.shape[dim+1:])


class ShapeTrace:
    def __init__(self):self.calls=[];self.saved={};self.store=SimpleNamespace(array=self.save)
    def save(self,x):
        name=f'RAM_{len(self.saved)}';self.saved[name]=x;return name
    def call(self,kind,args,fn):
        self.calls.append((kind,[a.shape for a in args]));return fn()


def shape_runtime(log):
    def linear(x,w,b):log.append(('linear',x.shape,w.shape));return Shape(x.shape[:-1]+(w.shape[0],))
    def conv2(x,w,b,stride):log.append(('conv2',stride));return Shape((x.shape[0],w.shape[0],16,16))
    def conv1(x,w,b):log.append(('conv1',x.shape));return Shape((x.shape[0],w.shape[0],x.shape[-1]))
    def norm(x,s,w,b,eps):log.append(('norm',eps));return Shape(x.shape)
    def sdpa(q,k,v,dropout_p,is_causal):
        log.append(('sdpa',q.shape,dropout_p,is_causal));assert q.shape==k.shape==v.shape;return Shape(q.shape)
    def cat(xs,dim):
        s=list(xs[0].shape);s[dim]=sum(x.shape[dim] for x in xs);return Shape(s)
    def stack(xs,dim):
        s=list(xs[0].shape);s.insert(dim,len(xs));return Shape(s)
    return SimpleNamespace(nn=SimpleNamespace(functional=SimpleNamespace(linear=linear,conv2d=conv2,conv1d=conv1,
        layer_norm=norm,scaled_dot_product_attention=sdpa,gelu=lambda x,approximate:Shape(x.shape))),cat=cat,stack=stack)


def test_owned_pixel_actual_function_shape_and_sdpa_order():
    log=[];t=shape_runtime(log);trace=ShapeTrace()
    def dense(inp,out):return SimpleNamespace(weight=Shape((out,inp)),bias=Shape((out,)))
    norm=SimpleNamespace(weight=Shape((192,)),bias=Shape((192,)))
    layers={}
    for i in range(12):
        layers[str(i)]=SimpleNamespace(layernorm_before=norm,layernorm_after=norm,
            attention=SimpleNamespace(attention=SimpleNamespace(query=dense(192,192),key=dense(192,192),value=dense(192,192)),
                output=SimpleNamespace(dense=dense(192,192))),
            intermediate=SimpleNamespace(dense=dense(192,768)),output=SimpleNamespace(dense=dense(768,192)))
    root=SimpleNamespace(encoder=SimpleNamespace(embeddings=SimpleNamespace(cls_token=Shape((1,1,192)),position_embeddings=Shape((1,257,192)),
        patch_embeddings=SimpleNamespace(projection=SimpleNamespace(weight=Shape((192,3,14,14)),bias=Shape((192,))))),
        encoder=SimpleNamespace(layer=SimpleNamespace(_modules=layers)),layernorm=norm),projector=lambda x:Shape(x.shape))
    z=m.owned_pixel(t,root,Shape((1,3,224,224)),dict(layer_norm_eps=1e-12,num_hidden_layers=12),trace)
    assert z.shape==(1,192) and [k for k,_ in trace.calls]==['encoder','pixel.patch']+[f'pixel.layer{i}' for i in range(12)]+['pixel.finalnorm','projector']
    assert [e for e in log if e[0]=='sdpa']==[('sdpa',(1,3,257,64),0.,False)]*12
    # Norm precedes the three QKV linears/SDPA, not a post-norm substitute.
    first=next(i for i,e in enumerate(log) if e[0]=='norm')
    assert [x[0] for x in log[first:first+5]]==['norm','linear','linear','linear','sdpa']


def test_owned_low_actual_function_incoming_outgoing_alignment_shape_only():
    log=[];t=shape_runtime(log);trace=ShapeTrace()
    root=SimpleNamespace(action_encoder=SimpleNamespace(patch_embed=SimpleNamespace(weight=Shape((10,10,1)),bias=Shape((10,))),
        embed=lambda a:Shape(a.shape[:-1]+(192,))),low_predictor=lambda z,a:Shape(z.shape),low_pred_proj=lambda h:Shape(h.shape))
    low=m.OwnedLow(t,root,trace)
    info=dict(planner_level='low',z_hist=Shape((1,1,192)),a_hist=Shape((1,1,10)),z_subgoal=Shape((1,192)))
    cost=low.get_cost(info,Shape((1,3,2,10)));assert cost.shape==(1,3)
    assert trace.calls==[('low_cost',[(1,1,192),(1,1,10),(1,192),(1,3,2,10)]),
        ('action_encoder',[(3,2,10)]),('low_predictor',[(3,1,192),(3,1,192)]),('low_pred_proj',[(3,192)]),
        ('action_encoder',[(3,3,10)]),('low_predictor',[(3,2,192),(3,2,192)]),('low_pred_proj',[(6,192)])]
    assert trace.saved[low.rollout].shape==(1,3,2,192)
