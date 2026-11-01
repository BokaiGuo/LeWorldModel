"""Literal RAM buffers only: no neural surrogate, torch, solver, or physics."""
from copy import deepcopy
import io
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import agent_hilewm_low_cem_20261029_v1 as m


def layout(a):
    return dict(shape=list(a.shape),stride=[s//a.itemsize for s in a.strides],storage_offset=0,
        contiguous=bool(a.flags.c_contiguous),device='cuda:0',dtype='torch.float32')


def fp(a,expanded=False):
    strides=[s//a.itemsize for s in a.strides]
    if expanded:strides[1]=0
    return dict(shape=list(a.shape),dtype=a.dtype.str,bytes=a.nbytes,sha256=m.digest(a.tobytes()),
        stride=strides,device='cuda:0',requires_grad=False)


def fixture_row(index=0):
    rng=np.random.default_rng(index);arrays={}
    row=dict(index=index,root_slot=m.base.ORDER[index][0],condition=m.base.ORDER[index][1],status='returned',events=[],
        cem=[],cem_reset=1,cem_start=1,cem_end=1,cem_history=[dict(step=i) for i in range(3)],rollouts=[])
    def save(a):
        n=f'R{index}_{len(arrays):05d}.npy';arrays[n]=np.asarray(a).copy();return n
    def begin(kind,args,parent=None):
        e=dict(index=len(row['events']),parent=parent,kind=kind,returned=True,inputs=[save(a) for a in args],layouts=[layout(a) for a in args])
        row['events'].append(e);return e
    def end(e,out):e.update(output=save(out),output_layout=layout(out));return out
    def leaf(kind,args,out,parent):return end(begin(kind,args,parent),out)
    z=np.full((1,192),.15+index/50,np.float32);target=np.full((1,192),.55+index/50,np.float32)
    row['z_init']=save(z);row['target']=save(target)
    mean=np.zeros((1,2,10),np.float32);std=np.ones_like(mean);row['initial']=save(mean);calls=[]
    info=dict(z_hist=z[:,None],a_hist=np.zeros((1,1,10),np.float32),z_subgoal=target)
    for iteration in range(3):
        c=rng.normal(size=(1,8,2,10)).astype(np.float32)*std[:,None]+mean[:,None];c[:,0]=mean
        zh=np.broadcast_to(z[:,None,None],(1,8,1,192));ah=np.broadcast_to(np.zeros((1,1,1,10),np.float32),(1,8,1,10))
        g=np.broadcast_to(target[:,None],(1,8,192))
        e=begin('low_cost',[zh,ah,g,c]);hist=zh.reshape(8,1,192);act=ah.reshape(8,1,10);pred=[]
        for step in range(2):
            act=np.concatenate((act,c[:,:,step].reshape(8,1,10)),axis=1)
            emb=leaf('action_encoder',[act],np.zeros((8,step+2,192),np.float32),e['index'])
            h=leaf('low_predictor',[hist,emb[:,-(step+1):]],rng.normal(size=(8,step+1,192)).astype(np.float32),e['index'])
            p=leaf('low_pred_proj',[h.reshape(-1,192)],h.reshape(-1,192).copy(),e['index'])
            nxt=p.reshape(8,step+1,192)[:,-1:];hist=np.concatenate((hist,nxt),axis=1);pred.append(nxt[:,0])
        roll=np.stack(pred,axis=1).reshape(1,8,2,192);row['rollouts'].append(save(roll))
        value=np.square(roll[:,:,-1]-g).sum(axis=-1,dtype=np.float32);end(e,value)
        ix=np.argsort(value,axis=1)[:,:3].astype(np.int64);elite=c[np.arange(1)[:,None],ix];vals=value[np.arange(1)[:,None],ix]
        newmean=elite.mean(axis=1);newstd=elite.std(axis=1,ddof=1)
        row['cem'].append(dict(step=iteration,arrays={k:save(a) for k,a in dict(candidates=c,costs=value,topk_vals=vals,
            topk_inds=ix,topk_candidates=elite,mean=newmean,var=newstd,prev_mean=mean,prev_var=std).items()}))
        calls.append(dict(attempt=iteration+1,returned=True,metadata='low',candidate_rows=8,
            expanded={k:fp(a,True) for k,a in zip(m.INFO,(zh,ah,g),strict=True)},candidates=fp(c),cost=fp(value)))
        mean,std=newmean,newstd
    row['outputs']={k:save(a) for k,a in dict(actions=mean,mean=mean,var=std,costs=vals.mean(axis=1)).items()}
    row['primitive_normalized']=save(mean.reshape(1,10,2));row['first_packet_normalized']=save(mean[:,:1].reshape(1,5,2))
    row['bridge']=dict(schema=m.TAG+'.low_bridge',metadata='low',original={k:fp(v) for k,v in info.items()},calls=calls,
        cost_attempts=3,cost_returns=3,candidate_rows_attempted=24)
    return row,arrays,z,target


@pytest.fixture(scope='module')
def ram():return fixture_row()


def validate(ram):
    row,a,z,g=ram;costs=m.validate_native(row,a,'cuda:0',z,g);m.validate_cem(row,a,costs,z,g);return costs


def test_full_saved_native_and_cem(ram):
    assert len(validate(ram))==3
    assert m.protocol()['search']['samples']==8 and m.protocol()['search']['horizon']==2
    assert not m.protocol()['physical_release'] and not m.protocol()['final_mean_evaluated']


def test_fixed_full4_resource_axes():
    rows=[fixture_row(i)[0] for i in range(4)];r=m.resources(rows);p=m.protocol()['resources']
    assert r['bridge_candidate_rows']==p['candidate_rows']==96
    for kind in ('low_cost','action_encoder','low_predictor','low_pred_proj'):
        assert r['modules'][kind]['attempted']==r['modules'][kind]['returned']==p[kind]
    assert r['modules']['low_predictor']['input_rows']==192
    assert r['modules']['low_predictor']['context_tokens']==288
    assert r['modules']['action_encoder']['context_tokens']==480
    assert r['modules']['low_pred_proj']['input_rows']==288
    assert r['encoder']==r['projector']==r['high_predictor']==r['AD']==r['physics']==0
    assert r['FLOPs'] is None


@pytest.mark.parametrize('fault',['unknown','drop','parent','notreturned','dtype','device','history','zeroaction','target',
    'outgoing','actiontail','projinput','projshape','rollout','cost','sampleaxis','layout','swapgoal'])
def test_native_mutations(ram,fault):
    row,a,z,g=deepcopy(ram)
    def e(k):return next(x for x in row['events'] if x['kind']==k)
    if fault=='unknown':row['events'].append(deepcopy(row['events'][-1]));row['events'][-1]['kind']='encoder'
    elif fault=='drop':row['events'].pop()
    elif fault=='parent':e('low_predictor')['parent']=None
    elif fault=='notreturned':e('low_cost')['returned']=False
    elif fault=='dtype':a[e('low_cost')['output']]=a[e('low_cost')['output']].astype(np.float64)
    elif fault=='device':e('low_predictor')['layouts'][0]['device']='cpu'
    elif fault=='history':a[e('low_predictor')['inputs'][0]].flat[0]+=.1
    elif fault=='zeroaction':a[e('low_cost')['inputs'][1]].flat[0]=1
    elif fault=='target':a[e('low_cost')['inputs'][2]].flat[0]+=.1
    elif fault=='outgoing':a[e('action_encoder')['inputs'][0]][:,1]=0
    elif fault=='actiontail':a[e('low_predictor')['inputs'][1]].flat[0]=1
    elif fault=='projinput':a[e('low_pred_proj')['inputs'][0]].flat[0]+=.1
    elif fault=='projshape':a[e('low_pred_proj')['output']]=np.zeros((7,192),np.float32)
    elif fault=='rollout':a[row['rollouts'][0]].flat[0]+=.1
    elif fault=='cost':a[e('low_cost')['output']]+=100
    elif fault=='sampleaxis':a[e('low_cost')['inputs'][0]]=np.zeros((1,1,8,192),np.float32)
    elif fault=='layout':e('low_cost')['output_layout']['stride']=[1]
    else:g+=1
    with pytest.raises((ValueError,KeyError)):m.validate_native(row,a,'cuda:0',z,g)


@pytest.mark.parametrize('fault',['sampleSTD','mean','elite','vals','duplicate','prev','mean_candidate','step','lifecycle',
    'bridge','metadata','stride','witness','outputs','primitive','firstpacket','initial','fp','dtype','device','candidate_stride'])
def test_cem_mutations(ram,fault):
    row,a,z,g=deepcopy(ram);costs=m.validate_native(row,a,'cuda:0',z,g);cb=row['cem'][0]['arrays']
    if fault=='sampleSTD':a[cb['var']]=a[cb['topk_candidates']].var(axis=1,ddof=0)
    elif fault=='mean':a[cb['mean']]+=1
    elif fault=='elite':a[cb['topk_candidates']].flat[0]+=1
    elif fault=='vals':a[cb['topk_vals']]+=1
    elif fault=='duplicate':a[cb['topk_inds']][0,1]=a[cb['topk_inds']][0,0]
    elif fault=='prev':a[cb['prev_var']]+=1
    elif fault=='mean_candidate':a[cb['candidates']][:,0]+=1
    elif fault=='step':row['cem'][0]['step']=2
    elif fault=='lifecycle':row['cem_start']=2
    elif fault=='bridge':row['bridge']['candidate_rows_attempted']=23
    elif fault=='metadata':row['bridge']['calls'][0]['metadata']='high'
    elif fault=='stride':row['bridge']['calls'][0]['expanded']['z_hist']['stride'][1]=192
    elif fault=='witness':row['bridge']['original']['z_subgoal']['sha256']='0'*64
    elif fault=='outputs':a[row['outputs']['actions']]+=1
    elif fault=='primitive':a[row['primitive_normalized']]=a[row['primitive_normalized']][:,::-1].copy()
    elif fault=='firstpacket':a[row['first_packet_normalized']]=a[row['primitive_normalized']][:,5:].copy()
    elif fault=='initial':a[row['initial']]+=1
    elif fault=='fp':row['bridge']['calls'][0]['cost']['sha256']='f'*64
    elif fault=='device':row['bridge']['calls'][0]['cost']['device']='cuda:1'
    elif fault=='candidate_stride':row['bridge']['calls'][0]['candidates']['stride']=[160,20,1,2]
    else:a[cb['topk_inds']]=a[cb['topk_inds']].astype(np.float32)
    with pytest.raises(ValueError):m.validate_cem(row,a,costs,z,g)


def test_coherent_wrong_cost_rejected_by_terminal_not_flags(ram):
    row,a,z,g=deepcopy(ram)
    for i,e in enumerate(x for x in row['events'] if x['kind']=='low_cost'):
        a[e['output']]+=1000;cb=row['cem'][i]['arrays'];a[cb['costs']]+=1000;a[cb['topk_vals']]+=1000
        row['bridge']['calls'][i]['cost']=fp(a[e['output']])
    a[row['outputs']['costs']]+=1000
    with pytest.raises(ValueError,match='terminal-goal'):m.validate_native(row,a,'cuda:0',z,g)


@pytest.mark.parametrize('samples,horizon',[(1,1),(2,3),(8,2),(7,5),(64,16)])
def test_variable_owned_low_abi(samples,horizon):
    info=dict(planner_level='low',z_hist=np.zeros((1,samples,1,192),np.float32),
        a_hist=np.zeros((1,samples,1,10),np.float32),z_subgoal=np.zeros((1,samples,192),np.float32))
    assert m.low_shape(info,np.zeros((1,samples,horizon,10),np.float32))==(1,samples,horizon)


@pytest.mark.parametrize('shape',[(1,0,2,10),(1,65,2,10),(1,8,17,10),(2,8,2,10),(1,8,2,2),(1,8,10)])
def test_variable_abi_bounded(shape):
    info=dict(planner_level='low',z_hist=np.zeros((1,8,1,192)),a_hist=np.zeros((1,8,1,10)),z_subgoal=np.zeros((1,8,192)))
    with pytest.raises(ValueError):m.low_shape(info,np.zeros(shape))


@pytest.mark.parametrize('authority',[None,{},dict(certified=True,mean=[0,0],scale=[1,1]),dict(mean_=[-.0087,.0068],scale_=[.2019,.2002])])
def test_raw_inverse_never_defaults_or_self_certifies(authority):
    with pytest.raises(ValueError,match='certified frozen original'):m.inverse_transform(np.zeros((1,5,2),np.float32),authority)


def test_run_gate_before_any_io(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('I/O forbidden'))
    with pytest.raises(ValueError,match='MAIN only'):m.run_cuda('missing','x','missing')


def test_cold_import_no_neural_or_foreign_or_pickle():
    code='''import builtins, numpy
old=builtins.__import__
def guarded(name,*a,**k):
    if name.split('.')[0] in {'torch','transformers','stable_worldmodel','pickle','pickletools'}:
        raise AssertionError('forbidden import '+name)
    return old(name,*a,**k)
builtins.__import__=guarded
from experiments.spectral_reachability import agent_hilewm_low_cem_20261029_v1 as m
assert m.protocol()['resources']['encoder']==0
try:m.run_cuda('absent','x','absent')
except ValueError as e:assert 'MAIN only' in str(e)
else:raise AssertionError('gate')
'''
    p=subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,text=True,capture_output=True)
    assert p.returncode==0,p.stderr


def test_npy_exact_bytes_trailing_and_digest(tmp_path):
    a=np.arange(32,dtype=np.float32);b=io.BytesIO();np.save(b,a,allow_pickle=False);body=b.getvalue()
    spec=dict(file_bytes=len(body),file_sha256=m.digest(body),data_sha256=m.digest(a.tobytes()),shape=list(a.shape),dtype=a.dtype.str)
    (tmp_path/'x.npy').write_bytes(body);m.base.exact(m.load_array(tmp_path,'x.npy',spec),a,'RAM NPY')
    with pytest.raises(ValueError):m.load_array(tmp_path,'../x.npy',spec)
    (tmp_path/'x.npy').write_bytes(body+b'!');bad=dict(spec,file_bytes=len(body)+1,file_sha256=m.digest(body+b'!'))
    with pytest.raises(ValueError):m.load_array(tmp_path,'x.npy',bad)


def test_own_store_cap_and_no_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'scoped',lambda x:Path(x));monkeypatch.setattr(m.shutil,'disk_usage',lambda p:SimpleNamespace(free=10**10))
    s=m.Store(tmp_path/'new',m.time.monotonic());name=s.array(np.arange(10,dtype=np.float32))
    assert s.arrays[name]['file_bytes']>40
    with pytest.raises(ValueError):s.put(name,b'x')
    s.used=m.CAP-m.RESERVE
    with pytest.raises(ValueError):s.guard(1)
    s.guard(1,final=True)


def test_upstream_whole_receipt_not_subset(monkeypatch):
    monkeypatch.setattr(m.base,'consume',lambda *a:dict(status='accepted',rows=[],resources={}))
    monkeypatch.setattr(m.base,'read',lambda *a:dict(status='accepted',rows=[],resources={},extra='mutation'))
    with pytest.raises(ValueError,match='whole persisted'):m.upstream()


def test_no_old_source_edit_and_official_append_order():
    # Lightweight read-only frozen source text, never metadata producer/run.
    for p,h in m.PINS.items():assert m.base.sha(p)==h
    text=(m.base.TEXTS/'package__code__h_le_wm__models__jepa.py.source.txt').read_text()
    assert 'def rollout_low(' in text and 'def get_cost_low(' in text


@pytest.mark.parametrize('samples,horizon',[(1,1),(8,2),(2,5)])
def test_actual_owned_kernel_symbolic_no_values(samples,horizon):
    # Frozen test-only shape runtime; no tensor data, neural eval or numerical
    # model mock. Executes NEW Python indexing/ordering, not a trained network.
    path=m.ROOT/'tests/test_agent_hilewm_rgb_high_low_20261028_v1.py'
    assert m.base.sha(path)==m.PINS[str(path)]
    spec=importlib.util.spec_from_file_location('frozen_shape_helpers',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    S=helper.Shape;log=[];t=helper.shape_runtime(log);trace=helper.ShapeTrace()
    root=SimpleNamespace(action_encoder=SimpleNamespace(patch_embed=SimpleNamespace(weight=S((10,10,1)),bias=S((10,))),
        embed=lambda a:S(a.shape[:-1]+(192,))),low_predictor=lambda z,a:S(z.shape),low_pred_proj=lambda h:S(h.shape))
    low=m.OwnedLow(t,root,trace)
    info=dict(planner_level='low',z_hist=S((1,samples,1,192)),a_hist=S((1,samples,1,10)),z_subgoal=S((1,samples,192)))
    cost=low.get_cost(info,S((1,samples,horizon,10)))
    assert cost.shape==(1,samples) and len(low.rollouts)==1
    assert trace.saved[low.rollouts[0]].shape==(1,samples,horizon,192)
    assert [k for k,args in trace.calls]==['low_cost']+['action_encoder','low_predictor','low_pred_proj']*horizon
    calls=[args for kind,args in trace.calls if kind=='low_predictor']
    assert calls==[[(samples,min(3,i+1),192),(samples,min(3,i+1),192)] for i in range(horizon)]
    assert [args for kind,args in trace.calls if kind=='action_encoder']==[[(samples,i+2,10)] for i in range(horizon)]


def test_bridge_exact_expand_and_failure_never_refunded():
    class Buffer(np.ndarray):
        def expand(self,*shape):return np.broadcast_to(self,shape).view(Buffer)
    def buf(x):return x.view(Buffer)
    class Low:
        def parameters(self):return iter(())
        def get_cost(self,info,candidates):
            assert info['planner_level']=='low'
            return buf(np.zeros((1,8),np.float32))  # metadata-only ABI mock
    info=dict(z_hist=buf(np.zeros((1,1,192),np.float32)),a_hist=buf(np.zeros((1,1,10),np.float32)),z_subgoal=buf(np.ones((1,192),np.float32)))
    bridge=m.LowBatchBridge(Low(),info,SimpleNamespace(accepts=lambda x:isinstance(x,Buffer),fingerprint=fp))
    expanded={k:v[:,None].expand(1,8,*v.shape[1:]) for k,v in info.items()}
    c=buf(np.zeros((1,8,2,10),np.float32));bridge.get_cost(expanded,c)
    assert bridge.receipt()['cost_returns']==1
    bad=dict(expanded);bad['z_subgoal']=buf(np.zeros((1,8,192),np.float32))
    with pytest.raises(ValueError,match='actual expanded'):bridge.get_cost(bad,c)
    assert bridge.receipt()['cost_attempts']==2 and bridge.receipt()['cost_returns']==1
    bridge.get_cost(expanded,c)
    with pytest.raises(ValueError,match='budget'):bridge.get_cost(expanded,c)
    assert bridge.receipt()['candidate_rows_attempted']==32


def test_npy_huge_declared_size_refused_before_allocation(tmp_path):
    f=io.BytesIO();np.lib.format.write_array_header_1_0(f,dict(descr='<f4',fortran_order=False,shape=(10**15,)))
    body=f.getvalue();(tmp_path/'huge.npy').write_bytes(body)
    spec=dict(file_bytes=len(body),file_sha256=m.digest(body))
    with pytest.raises(ValueError,match='before allocation'):m.load_array(tmp_path,'huge.npy',spec)


def saved_fixture(tmp_path,monkeypatch):
    arrays={};rows=[];slots=[]
    for i in range(4):
        row,a,z,g=fixture_row(i);rows.append(row);arrays.update(a)
        slots.append(dict(index=i,root_slot=row['root_slot'],condition=row['condition'],RGB_source=dict(fixture='RAM only'),
            buffers=dict(z_init=dict(values=z.tolist()),high_target=dict(values=g.tolist()))))
    specs={}
    for n,a in arrays.items():
        f=io.BytesIO();np.save(f,a,allow_pickle=False);b=f.getvalue();(tmp_path/n).write_bytes(b)
        specs[n]=dict(file_bytes=len(b),file_sha256=m.digest(b),data_sha256=m.digest(a.tobytes()),shape=list(a.shape),dtype=a.dtype.str)
    manifest=tmp_path.parent/'manifest.json';manifest.write_bytes(b'RAM fixture only')
    mm=dict(sources={},slots=slots,inherited_resources=dict(encoder='paid 8, not new0'))
    r=dict(schema=m.TAG+'.report',manifest=str(manifest.resolve()),manifest_sha256='fixture',protocol=m.protocol(),sources={},
        inherited_resources=mm['inherited_resources'],completed=True,errors=[],physical_release=False,scientific_admission=False,
        elapsed_seconds=1,precision_before=m.base.precision_expected(),precision_after=m.base.precision_expected(),
        rows=rows,arrays=specs,resources=m.resources(rows))
    monkeypatch.setattr(m,'verify',lambda *args:mm);monkeypatch.setattr(m,'check_model',lambda *args:'cuda:0')
    rp=tmp_path/'report.json';rp.write_bytes(m.encode(r));return manifest,rp,r,mm


def test_entire_ram_saved_consume(tmp_path,monkeypatch):
    mf,rp,r,mm=saved_fixture(tmp_path,monkeypatch)
    accepted=m.consume(mf,'fixture',rp,m.base.sha(rp))
    assert len(accepted['rows'])==4 and accepted['resources']['bridge_candidate_rows']==96
    assert all(row['raw_actions'] is None and not row['final_mean_evaluated'] for row in accepted['rows'])
    assert accepted['inherited_resources']==mm['inherited_resources']


@pytest.mark.parametrize('fault',['rowmissing','roworder','resourcerefund','inherited','precision','unknownfile','unreferenced','partial','admission'])
def test_saved_full_denominator_and_census(tmp_path,monkeypatch,fault):
    mf,rp,r,mm=saved_fixture(tmp_path,monkeypatch)
    if fault=='rowmissing':r['rows'].pop()
    elif fault=='roworder':r['rows'].reverse()
    elif fault=='resourcerefund':r['resources']['bridge_candidate_rows']-=1
    elif fault=='inherited':r['inherited_resources']={}
    elif fault=='precision':r['precision_after']['tf32_cudnn']=True
    elif fault=='unknownfile':(tmp_path/'unexpected').write_bytes(b'no')
    elif fault=='unreferenced':
        n=next(iter(r['arrays']));other='extra.npy';(tmp_path/other).write_bytes((tmp_path/n).read_bytes());r['arrays'][other]=r['arrays'][n]
    elif fault=='partial':r['completed']=False
    else:r['scientific_admission']=True
    rp.write_bytes(m.encode(r))
    with pytest.raises(ValueError):m.consume(mf,'fixture',rp,m.base.sha(rp))
