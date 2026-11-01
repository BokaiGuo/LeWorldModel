"""RAM arithmetic callbacks only. No torch, weights, model or physics execution."""
import ast
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_hilewm_normalized_transition_20261029_v1 as m


def callbacks():
    calls=[]
    def check(x):
        if not isinstance(x,np.ndarray) or x.dtype!=np.float32 or not np.isfinite(x).all():
            raise ValueError('literal finite fixture float32')
    def action(a):
        calls.append(('action_encoder',a.copy()))
        # Explicit arithmetic labels, not a model approximation.
        return np.repeat(a.sum(axis=-1,keepdims=True,dtype=np.float32),192,axis=-1)
    def predictor(z,a):
        calls.append(('low_predictor',z.copy(),a.copy()))
        return z+a
    def project(x):
        calls.append(('low_pred_proj',x.copy()))
        return x.copy()
    k=m.Kernels(lambda xs,dim:np.concatenate(xs,axis=dim),
                lambda xs,dim:np.stack(xs,axis=dim),check,action,predictor,project,'ram_fixture')
    return k,calls


def inputs(b=1,s=2,t=3,lz=1,la=1):
    z=np.arange(b*s*lz*192,dtype=np.float32).reshape(b,s,lz,192)/np.float32(128)
    a=np.arange(b*s*la*10,dtype=np.float32).reshape(b,s,la,10)/np.float32(256)
    future=np.arange(b*s*t*10,dtype=np.float32).reshape(b,s,t,10)/np.float32(512)
    return z,a,future


def literal_reference(z,a,f):
    # Independent literal official loop, fixture callbacks algebra only.
    b,s,t,_=f.shape;zz=z.reshape(b*s,z.shape[2],192);aa=a.reshape(b*s,a.shape[2],10)
    ys=[];states=[]
    for k in range(t):
        aa=np.concatenate((aa,f[:,:,k,:].reshape(b*s,1,10)),axis=1)
        emb=np.repeat(aa.sum(axis=-1,keepdims=True,dtype=np.float32),192,axis=-1)
        ctx=min(3,zz.shape[1],emb.shape[1]);h=zz[:,-ctx:]+emb[:,-ctx:]
        p=h.reshape(-1,192).copy();y=p.reshape(b*s,ctx,192)[:,-1:]
        zz=np.concatenate((zz,y),axis=1);ys.append(y[:,0]);states.append((zz,aa))
    return np.stack(ys,axis=1).reshape(b,s,t,192),states


def exact(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


@pytest.mark.parametrize('b,s,t,lz,la',[(1,1,1,1,1),(1,8,3,1,1),(2,3,5,3,2),
    (5,2,4,3,3),(1,1,16,16,16)])
def test_whole_matches_literal_ram(b,s,t,lz,la):
    z,a,f=inputs(b,s,t,lz,la);before=[x.copy() for x in (z,a,f)];k,calls=callbacks();log=m.Ledger()
    out=m.rollout(z,a,f,kernels=k,ledger=log);ref,states=literal_reference(z,a,f)
    assert exact(out.prediction,ref) and len(out.contexts)==t
    for c,(zz,aa) in zip(out.contexts,states,strict=True):assert exact(c.z,zz) and exact(c.a,aa)
    assert all(exact(x,y) for x,y in zip((z,a,f),before,strict=True))
    assert [c[0] for c in calls]==['action_encoder','low_predictor','low_pred_proj']*t
    r=log.summary();assert not r['real_meter_authenticated'] and r['FLOPs'] is None
    assert all(v['attempts']==v['returns']==t for v in r['modules'].values())
    assert r['modules']['action_encoder']['input_rows']==b*s*t
    assert r['modules']['action_encoder']['input_tokens']==b*s*sum(la+i+1 for i in range(t))


@pytest.mark.parametrize('parts',[(1,1,1),(1,2),(2,1),(3,)])
def test_full_seams_and_all_replay_costs(parts):
    z,a,f=inputs();k,_=callbacks();log=m.Ledger()
    w=m.rollout(z,a,f,kernels=k,ledger=log)
    seg=m.segmented_rollout(z,a,f,segments=parts,kernels=k,ledger=log)
    receipt=m.check_replay(w,seg,exact=exact)
    assert receipt['all_saved_contexts_equal'] and not receipt['CUDA_numerical_equivalence']
    assert log.summary()['modules']['low_predictor']['attempts']==6
    assert seg.final.z.shape==(2,4,192) and seg.final.a.shape==(2,4,10)


def test_context_schedule_and_outgoing_not_delayed():
    z,a,f=inputs(s=1,t=5);k,calls=callbacks();m.rollout(z,a,f,kernels=k)
    for i in range(5):
        ae,pred,proj=calls[3*i:3*i+3]
        assert ae[1].shape==(1,i+2,10)
        assert exact(ae[1][:,-1],f[:,:,i].reshape(1,10))
        assert pred[1].shape==pred[2].shape==(1,min(3,i+1),192)
        assert proj[1].shape==(min(3,i+1),192)


def test_step_preserves_live_context_not_bootstrap():
    z,a,f=inputs(s=1);k,_=callbacks();c=m.FlatContext(z.reshape(1,1,192),a.reshape(1,1,10))
    c1,y=m.transition(c,f[:,:,0,:].reshape(1,1,10),kernels=k)
    c2,_=m.transition(c1,f[:,:,1,:].reshape(1,1,10),kernels=k)
    assert exact(c2.z[:,:1],c.z) and exact(c2.a[:,:1],c.a)
    assert exact(c1.z[:,-1],y) and c2.a.shape[1]==3


def test_noncontiguous_action_view_preserved():
    z,a,f=inputs(b=2,s=2,t=3);backing=np.zeros((2,2,6,10),np.float32)
    backing[:,:,::2]=f;view=backing[:,:,::2];assert not view.flags.c_contiguous
    k,_=callbacks();w=m.rollout(z,a,view,kernels=k)
    seg=m.segmented_rollout(z,a,view,segments=(1,2),kernels=k)
    m.check_replay(w,seg,exact=exact)
    ref,_=literal_reference(z,a,view);assert exact(w.prediction,ref)


@pytest.mark.parametrize('parts',[(),[],(0,3),(-1,4),(1,1),(1,3),(True,2),(1.,2),'3'])
def test_bad_partitions_no_dispatch(parts):
    k,calls=callbacks()
    with pytest.raises(ValueError):m.segmented_rollout(*inputs(),segments=parts,kernels=k)
    assert not calls


@pytest.mark.parametrize('fault',['rank','samples','width','dtype','nan','empty','long'])
def test_input_mutations(fault):
    z,a,f=inputs();k,calls=callbacks()
    if fault=='rank':z=z[:,0]
    elif fault=='samples':a=a[:,:1]
    elif fault=='width':f=f[...,:9]
    elif fault=='dtype':f=f.astype(np.float64)
    elif fault=='nan':f.flat[0]=np.nan
    elif fault=='empty':f=f[:,:,:0]
    else:f=np.zeros((1,2,17,10),np.float32)
    with pytest.raises(ValueError):m.rollout(z,a,f,kernels=k)
    assert not calls


@pytest.mark.parametrize('kind',['action_encoder','low_predictor','low_pred_proj'])
def test_bad_callback_shapes_stop_and_attempt_kept(kind):
    k,_=callbacks();bad=lambda *args:np.zeros((1,),np.float32)
    k=replace(k,**{kind:bad});log=m.Ledger()
    with pytest.raises(ValueError):m.rollout(*inputs(),kernels=k,ledger=log)
    names=['action_encoder','low_predictor','low_pred_proj'];assert len(log.events)==names.index(kind)+1
    # A returned wrong-shape array is still a returned attempt, never refunded.
    assert log.events[-1]['returned']


def test_callback_failure_not_refunded():
    def fail(*args):raise RuntimeError('fixture callback failure')
    k,_=callbacks();k=replace(k,low_predictor=fail);log=m.Ledger()
    with pytest.raises(RuntimeError):m.rollout(*inputs(),kernels=k,ledger=log)
    assert len(log.events)==2 and log.events[0]['returned'] and not log.events[1]['returned']
    assert log.summary()['modules']['low_predictor']['attempts']==1


def test_mutated_seam_rejected():
    k,_=callbacks();w=m.rollout(*inputs(),kernels=k)
    seg=m.segmented_rollout(*inputs(),segments=(1,2),kernels=k)
    seg.contexts[0].a.flat[0]+=1
    with pytest.raises(ValueError,match='context'):m.check_replay(w,seg,exact=exact)


def test_cuda_authorization_fails_before_root_or_tensor_access():
    class Forbidden:
        def __getattr__(self,key):raise AssertionError('runtime accessed')
    with pytest.raises(ValueError,match='MAIN'):m.cuda_kernels(Forbidden(),Forbidden())


def test_symbolic_input_dependencies_survive_whole_and_segmented():
    class Symbol:
        # Shape and dependency tags only: no tensor arithmetic/autograd engine.
        def __init__(self,shape,tags):self.shape=tuple(shape);self.tags=frozenset(tags)
        def reshape(self,*shape):
            dims=list(shape);size=int(np.prod(self.shape))
            if -1 in dims:dims[dims.index(-1)]=size//abs(int(np.prod(dims)))
            assert int(np.prod(dims))==size
            return Symbol(dims,self.tags)
        def __getitem__(self,key):
            return Symbol(np.empty(self.shape,dtype=np.uint8)[key].shape,self.tags)
    def check(x):assert isinstance(x,Symbol)
    def cat(xs,dim):
        shape=list(xs[0].shape);shape[dim]=sum(x.shape[dim] for x in xs)
        return Symbol(shape,set().union(*(x.tags for x in xs)))
    def stack(xs,dim):
        shape=list(xs[0].shape);shape.insert(dim,len(xs))
        return Symbol(shape,set().union(*(x.tags for x in xs)))
    k=m.Kernels(cat,stack,check,
        lambda a:Symbol((*a.shape[:2],192),a.tags),
        lambda z,a:Symbol(z.shape,z.tags|a.tags),lambda h:Symbol(h.shape,h.tags),'ram_fixture')
    z=Symbol((1,2,1,192),{'z_input'});a=Symbol((1,2,1,10),{'a_input'})
    f=Symbol((1,2,3,10),{'future_action_input'})
    whole=m.rollout(z,a,f,kernels=k)
    seg=m.segmented_rollout(z,a,f,segments=(1,2),kernels=k)
    for result in (whole,seg):
        assert result.prediction.tags=={'z_input','a_input','future_action_input'}
        assert result.final.a.tags=={'a_input','future_action_input'}
    assert f.tags=={'future_action_input'}  # no mutation of input metadata


def test_no_decorators_detach_runtime_imports_or_global_patching():
    source=Path(m.__file__).read_text();tree=ast.parse(source)
    assert all(not n.decorator_list for n in ast.walk(tree) if isinstance(n,ast.FunctionDef))
    forbidden={'detach','no_grad','inference_mode','numpy','cpu','contiguous','requires_grad_'}
    assert not any(isinstance(n,ast.Attribute) and n.attr in forbidden for n in ast.walk(tree))
    assert not any(isinstance(n,(ast.Global,ast.Nonlocal)) for n in ast.walk(tree))
    assert 'setattr(' not in source


@pytest.mark.parametrize('command',['protocol','verify-sources'])
def test_cold_cli_and_real_source_sha(command):
    code="""
import sys,runpy
class Guard:
    def find_spec(self,fullname,*args):
        if fullname.split('.')[0] in {'torch','numpy','pygame','pymunk','pickle'}:
            raise RuntimeError('forbidden runtime import '+fullname)
sys.meta_path.insert(0,Guard())
sys.argv=['seam',sys.argv[1]]
runpy.run_module('experiments.spectral_reachability.agent_hilewm_normalized_transition_20261029_v1',run_name='__main__')
"""
    p=subprocess.run([sys.executable,'-B','-c',code,command],cwd=m.ROOT,text=True,capture_output=True,check=True)
    out=json.loads(p.stdout);assert out['CUDA_numerical_equivalence'] is False
    if command=='verify-sources':assert len(out['sources'])==2


def test_real_source_hash_mutation_fails(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'PINS',{'source.txt':'0'*64})
    (tmp_path/'source.txt').write_text('tamper')
    with pytest.raises(ValueError,match='SHA'):m.verify_sources()
