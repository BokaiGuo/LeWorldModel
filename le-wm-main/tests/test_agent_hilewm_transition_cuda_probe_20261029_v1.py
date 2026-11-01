"""Pure RAM/saved fixtures, never actual model/torch/physics/full strict."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import agent_hilewm_transition_cuda_probe_20261029_v1 as m


def lay(a,grad=False):
    root=a
    while isinstance(root.base,np.ndarray):root=root.base
    return dict(shape=list(a.shape),stride=[x//a.itemsize for x in a.strides],
        storage_offset=(a.__array_interface__['data'][0]-root.__array_interface__['data'][0])//a.itemsize,
        contiguous=bool(a.flags.c_contiguous),dtype='torch.float32' if a.dtype==np.float32 else 'torch.float64',
        device='cuda:0',requires_grad=grad)


def fixture():
    """Synthetic arithmetic lineage, NOT fake HOST CUDA evidence."""
    arrays={}
    def save(x):
        name=f'array_{len(arrays):05d}.npy';arrays[name]=np.asarray(x).copy();return name
    f=m.fixed_arrays();z0=np.full((1,192),.25,np.float32)
    prep=dict(z_init=dict(values=z0.tolist()))
    rows=[]
    for i,route in enumerate(m.ROUTES):
        grad=i<3;actions=f['actions'].copy()
        if not grad:actions=actions+(1 if i==3 else -1)*m.protocol()['finite_difference']['epsilon']*f['direction']
        z=z0.reshape(1,1,1,192);ah=np.zeros((1,1,1,10),np.float32)
        row=dict(route=route,status='returned',events=[],z=save(z),a_hist=save(ah),actions=save(actions),
            contexts=[],precision_before=m.expected_precision(grad),precision_after=m.expected_precision(grad),wall_seconds=.01)
        z=z.reshape(1,1,192);a=ah.reshape(1,1,10);pred=[]
        def event(kind,ins,out,flags,outgrad):
            row['events'].append(dict(index=len(row['events']),kind=kind,attempted=True,returned=True,
                inputs=[save(x) for x in ins],layouts=[lay(x,flag) for x,flag in zip(ins,flags,strict=True)],
                output=save(out),output_layout=lay(out,outgrad),precision=m.expected_precision(grad),
                CPUcapture_seconds=.001,callback_sync_wall_seconds=.002))
        for step in range(3):
            a=np.concatenate((a,actions[:,:,step].reshape(1,1,10)),axis=1)
            emb=np.repeat(a.sum(-1,keepdims=True,dtype=np.float32),192,axis=-1)
            event('action_encoder',[a],emb,[grad],grad)
            ctx=min(3,z.shape[1],emb.shape[1]);zi,ai=z[:,-ctx:],emb[:,-ctx:]
            h=zi+ai;event('low_predictor',[zi,ai],h,[grad and step>0,grad],grad)
            p=h.reshape(-1,192).copy();event('low_pred_proj',[h.reshape(-1,192)],p,[grad],grad)
            nxt=p.reshape(1,ctx,192)[:,-1:];z=np.concatenate((z,nxt),axis=1);pred.append(nxt[:,0])
            row['contexts'].append(dict(z=save(z),a=save(a)))
        out=np.stack(pred,axis=1).reshape(1,1,3,192);row['prediction']=save(out)
        scalar=np.asarray((out[:,:,-1].astype(np.float64)*f['weights'].astype(np.float64)).sum(),np.float64)
        row['scalar']=save(scalar)
        if grad:
            g=np.full((1,1,3,10),f['weights'].sum(dtype=np.float32),np.float32)
            event('scalar_VJP',[scalar,actions],g,[True,True],False);row['gradient']=save(g)
        rows.append(row)
    r=dict(rows=rows,runtime=dict(device='cuda:0'),resources=m.resources(rows))
    return prep,r,arrays


def test_all_five_complete_synthetic_saved_routes():
    p,r,a=fixture();v=m.validate_rows(p,r,a.__getitem__)
    assert v['forward_bitwise'] and v['gradient_bitwise'] and v['directional_check']['passed']
    assert not v['official_parity'] and not v['scientific_admission']
    assert r['resources']['AD_primal_forward_calls']==9
    assert r['resources']['modules']['action_encoder']['input_tokens']==45
    assert r['resources']['modules']['low_predictor']['input_tokens']==30
    assert r['resources']['modules']['low_pred_proj']['input_rows']==30


@pytest.mark.parametrize('fault',['route','missing','extra','partial','step','kind','notreturned','precision',
    'initial','history','action','AEinput','AEshape','predinput','projinput','projection','context','prediction',
    'gradient','gradlayout','layout','scalar','finite_difference','resource','time'])
def test_strict_saved_mutations(fault):
    p,r,a=fixture();row=r['rows'][0];e=row['events']
    if fault=='route':row['route']='whole'
    elif fault=='missing':r['rows'].pop()
    elif fault=='extra':e.append(deepcopy(e[-1]))
    elif fault=='partial':row['status']='attempted'
    elif fault=='step':e[0]['index']=1
    elif fault=='kind':e[0]['kind']='encoder'
    elif fault=='notreturned':e[0]['returned']=False
    elif fault=='precision':row['precision_before']['inference']=True
    elif fault=='initial':a[row['z']].flat[0]+=1
    elif fault=='history':a[row['a_hist']].flat[0]=1
    elif fault=='action':a[row['actions']].flat[0]+=1
    elif fault=='AEinput':a[e[0]['inputs'][0]].flat[-1]+=1
    elif fault=='AEshape':a[e[0]['output']]=np.zeros((1,1,192),np.float32)
    elif fault=='predinput':a[e[1]['inputs'][0]].flat[0]+=1
    elif fault=='projinput':a[e[2]['inputs'][0]].flat[0]+=1
    elif fault=='projection':a[e[2]['output']].flat[0]+=1
    elif fault=='context':a[row['contexts'][0]['a']].flat[0]+=1
    elif fault=='prediction':a[row['prediction']].flat[0]+=1
    elif fault=='gradient':a[row['gradient']].flat[0]+=1
    elif fault=='gradlayout':e[1]['output_layout']['requires_grad']=False
    elif fault=='layout':r['rows'][1]['events'][0]['layouts'][0]['storage_offset']=1
    elif fault=='scalar':a[row['scalar']]=a[row['scalar']]+np.float64(1)
    elif fault=='finite_difference':
        # Coherent gradient mutation of both duplicates in all three routes.
        for rr in r['rows'][:3]:
            a[rr['gradient']][:]=100
            a[rr['events'][-1]['output']][:]=100
    elif fault=='resource':r['resources']['AD_primal_forward_calls']=0
    else:e[0]['CPUcapture_seconds']=-1
    with pytest.raises((ValueError,KeyError)):m.validate_rows(p,r,a.__getitem__)


def test_reference_is_independent_and_matches_ram_callback_seam():
    # Source-formula callbacks only; no neural network is constructed.
    class Ram(np.ndarray):
        def transpose(self,dim1,dim2):return self.swapaxes(dim1,dim2)
    t=SimpleNamespace(cat=lambda xs,dim:np.concatenate(xs,axis=dim).view(Ram),stack=lambda xs,dim:np.stack(xs,axis=dim).view(Ram),
        nn=SimpleNamespace(functional=SimpleNamespace(conv1d=lambda x,w,b:x)))
    root=SimpleNamespace(action_encoder=SimpleNamespace(patch_embed=SimpleNamespace(weight=None,bias=None),
        embed=lambda a:np.repeat(a.sum(-1,keepdims=True,dtype=np.float32),192,axis=-1)),
        low_predictor=lambda z,a:z+a,low_pred_proj=lambda h:h.copy())
    class Trace:
        def __init__(self):self.events=[]
        def call(self,k,args,fn):self.events.append((k,[x.shape for x in args]));return fn()
    tr=Trace();z=np.full((1,1,1,192),.25,np.float32).view(Ram)
    a=np.zeros((1,1,1,10),np.float32).view(Ram);f=m.fixed_arrays()['actions'].view(Ram)
    ref,states=m.reference(t,root,z,a,f,tr)
    def check(x):assert x.dtype==np.float32
    k=m.seam.Kernels(t.cat,t.stack,check,lambda a:root.action_encoder.embed(a),
        root.low_predictor,root.low_pred_proj,'ram_fixture')
    out=m.seam.rollout(z,a,f,kernels=k)
    assert m.comparisons(ref,out.prediction)['exact']
    assert [s[0] for s in tr.events]==list(m.KINDS)*3
    for (zz,aa),c in zip(states,out.contexts,strict=True):
        assert m.comparisons(zz,c.z)['exact'] and m.comparisons(aa,c.a)['exact']
    node=next(n for n in ast.parse(Path(m.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='reference')
    assert not any(isinstance(n,ast.Name) and n.id=='seam' for n in ast.walk(node))


def test_fixed_action_dtype_and_predetermined_fd_rule():
    f=m.fixed_arrays();assert f['actions'].shape==f['direction'].shape==(1,1,3,10)
    assert all(x.dtype==np.float32 for x in f.values())
    assert m.finite_difference_check(np.zeros(30,np.float32),np.ones(30,np.float32),0.,0.)['passed']
    assert not m.finite_difference_check(np.ones(30,np.float32),np.ones(30,np.float32),0.,0.)['passed']
    assert m.protocol()['finite_difference']==dict(epsilon=2.**-8,atol=2.e-4,rtol=2.e-2,
        rule='abs(AD_dot-FD)<=atol+rtol*max(abs(AD_dot),abs(FD))',
        scope='single fixed direction central difference of float32 neural forward; no sweep/retry')


def test_exact_comparison_rejects_signed_zero_and_one_ulp():
    a=np.zeros(2,np.float32);b=a.copy();b[0]=-0.
    assert not m.comparisons(a,b)['exact']
    b[0]=np.nextafter(np.float32(0),np.float32(1));assert not m.comparisons(a,b)['exact']


def sandbox(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    return tmp_path/'results'


def test_actual_fixture_prepare_verify_exclusive(tmp_path,monkeypatch):
    folder=sandbox(tmp_path,monkeypatch);value={'schema':m.TAG+'.prepare','fixture_only':True}
    monkeypatch.setattr(m,'build',lambda:deepcopy(value))
    p=m.prepare(folder/(m.TAG+'_prepare_fixture'));assert m.verify(p['manifest'],p['sha256'])==value
    with pytest.raises(FileExistsError):m.prepare(folder/(m.TAG+'_prepare_fixture'))
    value['tamper']=True
    with pytest.raises(ValueError):m.verify(p['manifest'],p['sha256'])


def test_npy_roundtrip_tamper_and_no_object(tmp_path,monkeypatch):
    folder=sandbox(tmp_path,monkeypatch);store=m.Store(folder/(m.TAG+'_fixture'),time.monotonic())
    for x in (np.arange(12,dtype=np.float32).reshape(3,4),np.asarray(.125,np.float64)):
        n=store.array(x);y=m.load_array(store.path,n,store.arrays[n]);assert m.comparisons(x,y)['exact']
    n=next(iter(store.arrays));d=deepcopy(store.arrays[n]);d['data_sha256']='0'*64
    with pytest.raises(ValueError):m.load_array(store.path,n,d)
    with pytest.raises(ValueError):store.array(np.asarray([object()],dtype=object))
    with pytest.raises(ValueError):m.load_array(store.path,'../escape.npy',d)


def test_authorization_before_any_io_or_runtime():
    with pytest.raises(ValueError,match='MAIN authorization'):m.run_cuda('missing','x','invalid')


def test_actual_invalid_manifest_preserves_failed_report_without_torch(tmp_path,monkeypatch):
    folder=sandbox(tmp_path,monkeypatch);monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    output=folder/(m.TAG+'_failed_fixture')
    with pytest.raises(ValueError,match='partial report'):m.run_cuda(folder/'absent.json','0'*64,output,main_authorized=True)
    r=m.read(output/'report.json');assert not r['completed'] and r['errors'] and r['model_before'] is None
    assert len(r['rows'])==5 and all(x['status']=='not_started' for x in r['rows'])
    assert r['timings']['initial_saved_verify']['returned'] is False and not r['arrays']


@pytest.fixture
def savedcase(tmp_path,monkeypatch):
    folder=sandbox(tmp_path,monkeypatch);p,r,a=fixture();store=m.Store(folder/(m.TAG+'_saved_fixture'),time.monotonic())
    for name,x in a.items():assert store.array(x)==name
    expected={'fixture.key':'f'*64};p.update(sources={'fixture-source':'f'*64},expected_model_hashes=expected,
        export=dict(export_report_sha256='e'*64,export_receipt_sha256='a'*64),
        construction_expected=dict(module_count=1,entry_count=1,state_key_count=1,ir_sha256='b'*64))
    c=dict(tensor_value_hashes=expected,export_report_sha256='e'*64,export_receipt_sha256='a'*64,
        all_parameters_frozen=True,forward_calls=0,tensor_copy_attempts=425,tensor_copy_returns=425,
        device='cuda:0',torch='fixture',own_sha256={'fixture-source':'f'*64},**p['construction_expected'])
    r.update(schema=m.TAG+'.report',protocol=m.protocol(),manifest='fixture',manifest_sha256='c'*64,
        arrays=store.arrays,sources_before=p['sources'],sources_after=p['sources'],model_before=expected,model_after=expected,
        construction=c,completed=True,errors=[],scientific_admission=False,
        timings={k:dict(attempted=True,returned=True,seconds=.01) for k in
            ('initial_saved_verify','construct','model_hash_before','dispatch_all5','model_hash_after','final_saved_verify')})
    r['runtime'].update(torch='fixture',cuda='fixture',peak_allocated_bytes=1,peak_reserved_bytes=2)
    r['diagnostics']=m.validate_rows(p,r,a.__getitem__)
    store.put('report.json',m.encode(r),final=True);monkeypatch.setattr(m,'verify',lambda path,h:deepcopy(p))
    return store,p,r


def test_saved_complete_consume_and_census_fixture(savedcase):
    store,p,r=savedcase;h=m.sha(store.path/'report.json');receipt=m.consume(store.path,h)
    assert receipt['status']=='fixed_owned_low_probe_saved_verified_not_official_parity'
    (store.path/'unknown.txt').write_text('extra')
    with pytest.raises(ValueError,match='census'):m.consume(store.path,h)


@pytest.mark.parametrize('fault',['source_after','weight_after','export_receipt','copies','construction_source',
    'construction_count','timing','diagnostics','gate','runtime','partial'])
def test_full_consume_authority_mutations(savedcase,fault):
    store,p,r=savedcase;r=deepcopy(r)
    if fault=='source_after':r['sources_after']={}
    elif fault=='weight_after':r['model_after']={}
    elif fault=='export_receipt':r['construction']['export_receipt_sha256']='0'*64
    elif fault=='copies':r['construction']['tensor_copy_returns']=424
    elif fault=='construction_source':r['construction']['own_sha256']['fixture-source']='0'*64
    elif fault=='construction_count':r['construction']['module_count']=0
    elif fault=='timing':r['timings']['initial_saved_verify']['returned']=False
    elif fault=='diagnostics':r['diagnostics']['forward_bitwise']=False
    elif fault=='gate':r['scientific_admission']=True
    elif fault=='runtime':r['runtime']['peak_allocated_bytes']=None
    else:r['completed']=False
    (store.path/'report.json').write_bytes(m.encode(r))
    with pytest.raises(ValueError):m.consume(store.path,m.sha(store.path/'report.json'))


def test_actual_saved_metadata_abi_only_no_consumer():
    r=m.read(m.REPORT,m.REPORT_SHA);c=m.read(m.RECEIPT,m.RECEIPT_SHA)
    assert Path(r['manifest'])==m.PARENT and c['report_sha256']==m.REPORT_SHA
    p=m.read(m.PARENT,r['manifest_sha256'])
    assert (r['rows'][0]['root_slot'],r['rows'][0]['condition'])==(0,'off')
    assert p['slots'][0]['buffers']['z_init']['descriptor']['shape']==[1,192]
    assert len(p['slots'][0]['buffers']['z_init']['values'][0])==192
    assert set(p['export'])=={'export_report','export_report_sha256','export_receipt','export_receipt_sha256'}


def test_source_pins_without_fullstrict_or_weights():
    assert m.sha(m.seam.__file__)==m.SEAM_SHA and m.seam.verify_sources()['source_bytes_verified']


def test_cold_import_help_no_runtime():
    code="""
import sys,runpy
class Guard:
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in {'torch','pygame','pymunk','pickle'}:
   raise RuntimeError('forbidden runtime import '+fullname)
sys.meta_path.insert(0,Guard())
sys.argv=['probe','--help']
runpy.run_module('experiments.spectral_reachability.agent_hilewm_transition_cuda_probe_20261029_v1',run_name='__main__')
"""
    # NumPy itself may use inert pickletools dependencies in some versions;
    # preload only NumPy, never a production owner/consumer/runtime.
    code='import numpy\n'+code
    p=subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,text=True,capture_output=True,check=True)
    assert 'prepare' in p.stdout and 'consume' in p.stdout
