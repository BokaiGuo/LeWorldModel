"""All CUDA/model objects below are explicit RAM algebra fixtures, never torch."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import io
import numpy as np
import pytest
from experiments.spectral_reachability import agent_e3_gd_host_cuda_20261029_v1 as m


DEVICE=SimpleNamespace(type='cuda')


def reduce_shape(g,shape):
    while g.ndim>len(shape):g=g.sum(axis=0,dtype=np.float32)
    for i,n in enumerate(shape):
        if n==1 and g.shape[i]!=1:g=g.sum(axis=i,keepdims=True,dtype=np.float32)
    return g.reshape(shape)


class Tensor:
    def __init__(self,value,parents=(),requires=False):
        self.a=np.asarray(value,np.float32);self.parents=parents;self.requires_grad=requires or any(p.requires_grad for p,_ in parents)
        self.grad=None;self.hooks=[];self.device=DEVICE;self.dtype=np.dtype('float32')
    @property
    def shape(self):return self.a.shape
    @property
    def ndim(self):return self.a.ndim
    @property
    def is_leaf(self):return not self.parents
    def requires_grad_(self,value):self.requires_grad=value;return self
    def detach(self):return Tensor(self.a.copy())
    def cpu(self):return self
    def numpy(self):return self.a.copy()
    def item(self):return self.a.item()
    def tolist(self):return self.a.tolist()
    def register_hook(self,f):
        self.hooks.append(f)
        return SimpleNamespace(remove=lambda:self.hooks.remove(f))
    def reshape(self,*shape):return Tensor(self.a.reshape(*shape),[(self,lambda g:g.reshape(self.shape))])
    def __getitem__(self,key):
        def backward(g):
            result=np.zeros_like(self.a);result[key]+=g;return result
        return Tensor(self.a[key],[(self,backward)])
    def mean(self):return Tensor(self.a.mean(dtype=np.float32),[(self,lambda g:np.ones_like(self.a)*g/np.float32(self.a.size))])
    def __add__(self,other):
        other=other if isinstance(other,Tensor) else Tensor(other)
        return Tensor(self.a+other.a,[(self,lambda g:reduce_shape(g,self.shape)),(other,lambda g:reduce_shape(g,other.shape))])
    def __sub__(self,other):return self+other*(-1)
    def __mul__(self,other):
        other=other if isinstance(other,Tensor) else Tensor(other)
        return Tensor(self.a*other.a,[(self,lambda g:reduce_shape(g*other.a,self.shape)),(other,lambda g:reduce_shape(g*self.a,other.shape))])
    def __truediv__(self,other):
        other=other if isinstance(other,Tensor) else Tensor(other)
        return Tensor(self.a/other.a,[(self,lambda g:reduce_shape(g/other.a,self.shape)),(other,lambda g:reduce_shape(-g*self.a/other.a**2,other.shape))])
    def __pow__(self,p):return Tensor(self.a**p,[(self,lambda g:g*np.float32(p)*self.a**(p-1))])


def grad(loss,leaf,retain_graph=False,create_graph=False):
    assert not retain_graph and not create_graph
    order=[];seen=set()
    def visit(node):
        if id(node) in seen:return
        seen.add(id(node))
        for p,_ in node.parents:visit(p)
        order.append(node)
    visit(loss);assert leaf in order,'disconnected raw leaf'
    grads={id(loss):np.ones_like(loss.a)}
    for node in reversed(order):
        value=grads.get(id(node))
        if value is None:continue
        for hook in node.hooks:value=hook(Tensor(value)).a
        for p,fn in node.parents:
            delta=fn(value);grads[id(p)]=grads.get(id(p),np.zeros_like(p.a))+delta
    return (Tensor(grads[id(leaf)]),)


def cat(items,dim=0):
    parents=[];start=0
    for item in items:
        begin,end=start,start+item.shape[dim];start=end
        def back(g,begin=begin,end=end):
            key=[slice(None)]*g.ndim;key[dim]=slice(begin,end);return g[tuple(key)]
        parents.append((item,back))
    return Tensor(np.concatenate([x.a for x in items],axis=dim),parents)


T=SimpleNamespace(float32=np.dtype('float32'),tensor=lambda a,**kw:Tensor(a),cat=cat,
    no_grad=nullcontext,enable_grad=nullcontext,autograd=SimpleNamespace(grad=grad),
    cuda=SimpleNamespace(synchronize=lambda d:None,max_memory_allocated=lambda d:1024,max_memory_reserved=lambda d:2048))


class Module:
    def __init__(self,fn):self.fn=fn;self.pre=[];self.post=[]
    def register_forward_pre_hook(self,f):self.pre.append(f);return SimpleNamespace(remove=lambda:self.pre.remove(f))
    def register_forward_hook(self,f):self.post.append(f);return SimpleNamespace(remove=lambda:self.post.remove(f))
    def __call__(self,*args):
        for f in self.pre:f(self,args)
        out=self.fn(*args)
        for f in self.post:f(self,args,out)
        return out


class Model:
    """Simple array recurrence, no neural layers, parameters or model loading."""
    def __init__(self):
        self.action_encoder=Module(lambda h:Tensor(np.ones((1,3,192),np.float32))*h.mean())
        self.predictor=Module(lambda z,e:z+e*np.float32(.001))
        self.pred_proj=Module(lambda x:x)
        self.encoder=Module(lambda x:x);self.projector=Module(lambda x:x)
    def predict(self,z,e):return self.pred_proj(self.predictor(z,e).reshape(3,192)).reshape(1,3,192)


def body(K=1,domain='Dxy'):
    context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=np.zeros((3,192),np.float32).tolist(),action_history=np.zeros((2,10),np.float32).tolist())
    return dict(schema=m.numerical.resolver.NAME+'.task',root_context=context,goal=np.ones(192).tolist(),
        projection=np.eye(192).tolist(),action_mean=np.asarray([-.0087,.0068],np.float32).tolist(),
        action_std=np.asarray([.2019,.2002],np.float32).tolist(),model_sha256=m.numerical.resolver.MODEL_SHA,
        root_sha256=m.numerical.resolver.KNOWN_ROOTS[0],horizon_packets=K,domain=m.numerical.resolver.domain_spec(domain),
        identity_sha256='0'*64,feedback_boundary=5,task_deadline_control=20)


def algebra_forward(c,a,b,leaf=None):
    z=Tensor(c['latent_history'])[None];h=Tensor(c['action_history'])[None]
    raw=Tensor(a) if leaf is None else leaf;model=Model()
    for k in range(len(a)//5):
        token=((raw[5*k:5*k+5]-Tensor(b['action_mean']))/Tensor(b['action_std'])).reshape(10)
        outgoing=cat([h,token[None,None]],1)
        predicted=model.predict(z,model.action_encoder(outgoing));z,h=cat([z[:,1:],predicted[:,-1:]],1),outgoing[:,1:]
    return z,dict(c,latent_history=z.a[0].tolist(),action_history=h.a[0].tolist())


class FrozenRAM:
    kind='RAM_FIXTURE'
    def __init__(self,b):self.b=b
    def packet(self,c,a):
        _,nc=algebra_forward(c,a,self.b)
        return nc,dict(raw=a.copy())
    def reverse(self,a,goal,primal):
        c=primal(a);leaf=Tensor(a).requires_grad_(True)
        z,_=algebra_forward(self.b['root_context'],a,self.b,leaf)
        loss=((z[0,-1]-Tensor(goal))**2).mean()
        return c,grad(loss,leaf)[0].a,np.float32(loss.item())


class CaptureTape(m.numerical.Tape):
    def __init__(self):super().__init__();self.arrays={}
    def put(self,value):
        d=super().put(value);self.arrays[d['file']]=np.asarray(value).copy();return d


@pytest.mark.parametrize('K',[1,3])
@pytest.mark.parametrize('domain',['Dxy','Dx'])
def test_host_against_frozen_all_candidates_gradients_actions_and27(tmp_path,K,domain):
    b=body(K,domain);tape=CaptureTape()
    expected=m.numerical.optimize_body(b,FrozenRAM,tape)
    store=m.Store(tmp_path/'native',m.time.monotonic(),K);row=dict(forwards=[],AD=[],result=None)
    backend=m.HostBackend(Model(),DEVICE,T,store,b,row)
    with backend.observe():result=m.solve(b,backend,store.event)
    assert result['winner']==expected['winner']
    assert [c['cost'] for c in result['candidates']]==expected['candidate_costs']
    assert result['commands']==expected['commands'] and result['context']==expected['context']
    old_candidates=[e for e in tape.events if e['kind']=='candidate' and e['mode'].startswith('iterate')]
    old_gradients=[e for e in tape.events if e['kind']=='gradient']
    assert len(old_candidates)==33 and len(old_gradients)==32
    for i,g in enumerate(result['gradients']):
        assert np.array_equal(tape.arrays[old_candidates[i]['arrays'][0]['file']],result['candidates'][i]['actions'])
        assert np.array_equal(tape.arrays[old_gradients[i]['arrays'][0]['file']],g['raw_gradient'])
        a=np.asarray(result['candidates'][i]['actions'],np.float32)
        _,direct,_=FrozenRAM(b).reverse(a,np.ones(192),lambda x:algebra_forward(b['root_context'],x,b)[1])
        assert np.array_equal(g['raw_gradient'],direct)
    assert np.array_equal(tape.arrays[old_candidates[-1]['arrays'][0]['file']],result['candidates'][-1]['actions'])
    events=m.journal(store.output,K)
    assert m.axes(events)['attempted']==m.axes(events)['completed']==m.numerical.axes27(K)
    replay=[];saved=m.SavedBackend(b,row,store.output,replay.append)
    reconstructed=m.solve(b,saved,replay.append);saved.complete()
    assert reconstructed==result and events==replay
    assert store.used==sum(p.stat().st_size for p in store.output.iterdir())
    assert len(row['forwards'])==68*K and len(row['AD'])==32
    assert all(not getattr(Model(),k).pre for k in ('encoder','projector'))


def test_K3_rawleaf_continuous_graph_and_capture_detach_not_used(tmp_path):
    b=body(3);store=m.Store(tmp_path/'graph',m.time.monotonic(),3);row=dict(forwards=[],AD=[])
    backend=m.HostBackend(Model(),DEVICE,T,store,b,row)
    with backend.observe():c,g,loss=backend.gradient(b['root_context'],np.zeros((15,2),np.float32),np.ones(192),'AD0')
    assert g.shape==(15,2) and np.isfinite(g).all() and (g!=0).all()
    # Earliest raw packet is used in all3 contexts, latest only once.
    assert np.allclose(g[:5],3*g[10:],rtol=1e-6)
    assert [e['kind'] for e in m.journal(store.output,3)].count('raw_backward')==1


def test_hook_cleanup_failure_preserves_original_error(tmp_path):
    b=body();store=m.Store(tmp_path/'error',m.time.monotonic(),1);row=dict(forwards=[],AD=[]);model=Model()
    def fail(*args):raise RuntimeError('original diagnostic with payload')
    model.predict=fail;backend=m.HostBackend(model,DEVICE,T,store,b,row)
    with pytest.raises(RuntimeError,match='original diagnostic'):
        with backend.observe():backend.forward(b['root_context'],np.zeros((5,2),np.float32),'bad')
    assert all(not module.pre and not module.post for module in (model.action_encoder,model.predictor,model.pred_proj,model.encoder,model.projector))
    assert row['forwards'][0]['completed'] is False
    counts=m.axes(m.journal(store.output,1));assert counts['attempted']['predict_API']==1 and counts['completed']['predict_API']==0


@pytest.mark.parametrize('mutation',['tail','extra','array_sha','event_move','AD_completed','scalar_vector'])
def test_saved_mutations(tmp_path,mutation):
    b=body();store=m.Store(tmp_path/'saved',m.time.monotonic(),1);row=dict(forwards=[],AD=[])
    backend=m.HostBackend(Model(),DEVICE,T,store,b,row)
    with backend.observe():row['result']=m.solve(b,backend,store.event)
    if mutation=='tail':
        d=row['AD'][0]['loss'];p=store.output/d['file'];p.write_bytes(p.read_bytes()+b'TAIL');d['sha256']=m.sha(p)
    elif mutation=='array_sha':
        p=store.output/row['forwards'][0]['inputs']['raw']['file'];p.write_bytes(b'bad')
    elif mutation=='AD_completed':row['AD'][0]['completed']=False
    elif mutation=='scalar_vector':
        row['AD'][0]['loss']=store.array(np.ones(1,np.float32))
    elif mutation=='event_move':
        es=m.journal(store.output,1);es[1],es[2]=es[2],es[1]
        replay=[];saved=m.SavedBackend(b,row,store.output,replay.append);m.solve(b,saved,replay.append)
        assert es!=replay;return
    else:
        # Standalone array loader must reject descriptor path traversal.
        d=dict(row['AD'][0]['loss']);d['file']='../outside.npy'
        with pytest.raises(ValueError):m.load_array(store.output,d)
        return
    with pytest.raises(ValueError):m.solve(b,m.SavedBackend(b,row,store.output,lambda e:None),lambda e:None)


def test_budget_exact32inventory_and_metadata_shape():
    assert m.budget(3)['total_bytes']==400042304<m.CAP
    assert m.budget(1)['total_bytes']<m.CAP
    assert m.budget(3)['reserve_bytes']==64*1024**2


def test_wall_exhausted_before_dispatch(tmp_path):
    store=m.Store(tmp_path/'expired',0,1,clock=lambda:1801.)
    with pytest.raises(ValueError,match='wall'):store.event(dict(kind='intent'))
    assert not list(store.output.iterdir())


def test_scalar_bound_no_numpy_reduction_bitparity():
    m.scalar_bound(float(np.nextafter(np.float32(1),np.float32(2))),np.zeros(192),np.ones(192))
    with pytest.raises(ValueError):m.scalar_bound(1.1,np.zeros(192),np.ones(192))


def test_signed_zero_is_bitwise_not_equal():
    assert not m.exact_f32(np.asarray([0.],np.float32),np.asarray([-0.],np.float32))
    assert m.exact_f32(np.asarray([-0.],np.float32),[-0.])


def test_full16_prepare_temporary_disk_reconstruction(tmp_path):
    bodies=[]
    for slot in m.numerical.resolver.roster():
        b=body(slot['horizon_packets'],slot['domain']);b['root_sha256']=m.numerical.resolver.KNOWN_ROOTS[slot['root']]
        bodies.append(b)
    pp=dict(expected_model_hash=m.numerical.resolver.MODEL_SHA,policy='/fixture/policy')
    def fixture_build():return m.assemble_preparation(bodies,pp,{'/fixture/source':'0'*64})
    p=tmp_path/'prepare.json';h=m.write_json(p,fixture_build())
    out=m.verify(p,h,fixture_builder=fixture_build)
    assert out['fixed_cells']==16 and out['current_wave_counts']==dict(predict=2176,scalar_reverse=512)
    assert out['closed_loop_complete'] is False and p.stat().st_size<m.META
    changed=deepcopy(out);changed['budgets']['3']['total_bytes']-=1;p.write_bytes(m.raw(changed))
    with pytest.raises(ValueError,match='reconstruction'):m.verify(p,m.sha(p),fixture_builder=fixture_build)


def test_prepare_refuses_wrong_root_horizon_and_whitelist():
    bodies=[]
    for slot in m.numerical.resolver.roster():
        b=body(slot['horizon_packets'],slot['domain']);b['root_sha256']=m.numerical.resolver.KNOWN_ROOTS[slot['root']];bodies.append(b)
    pp=dict(expected_model_hash=m.numerical.resolver.MODEL_SHA,policy='/fixture/policy')
    for key,value in (('horizon_packets',3),('root_sha256',m.numerical.resolver.KNOWN_ROOTS[1]),('evaluator',{})):
        bad=deepcopy(bodies);bad[0][key]=value
        with pytest.raises(ValueError):m.assemble_preparation(bad,pp,{})


def fake_complete_bundle(tmp_path,monkeypatch):
    """Synthetic temp fixture, NOT an actual loader/host receipt or production artifact."""
    b=body(1);store=m.Store(tmp_path/'complete',m.time.monotonic(),1)
    policy=tmp_path/'fake_policy';policy.mkdir()
    paths=[];bindings={str(Path(__file__).resolve()):m.sha(__file__)}
    for name in ('config.json','weights.pt'):
        p=policy/name;p.write_bytes(('RAM fixture '+name).encode());paths.append(p);bindings[str(p)]=m.sha(p)
    p=dict(bodies=[b]*16,expected_model_hash=m.numerical.resolver.MODEL_SHA,policy=str(policy),bindings_sha256=bindings)
    monkeypatch.setattr(m,'verify',lambda path,sha:p)
    files=[]
    for path in paths:
        h=m.sha(path);f=dict(path=str(path),sha256=h,bytes=path.stat().st_size,hash_buffer_sha256=h,deserialize_buffer_sha256=h,same_buffer=True)
        files.append(f);store.event(dict(kind='load_intent',path=str(path),expected_sha256=h))
        store.event(dict(kind='consumed_buffer_frozen',receipt=f))
    old=m.primitives.native.loader
    loader=dict(schema=old.NAME+'.loader_receipt',prepare_sha256='a'*64,adapter_source_sha256=m.sha(old.__file__),
        legacy_loader_source_sha256=old.native.SOURCES['experiments/checkpoint_compat.py'],files=files,device='cuda:0',
        remapped_keys=[],missing=[],unexpected=[],tensor_hash_before=p['expected_model_hash'],frozen_eval=True)
    store.put('loader.json',loader)
    row=dict(forwards=[],AD=[],result=None);backend=m.HostBackend(Model(),DEVICE,T,store,b,row)
    with backend.observe():row['result']=m.solve(b,backend,store.event)
    store.put('row.json',row,final=True)
    classes=[dict(name='',class_module='fixture',class_qualname='FixtureModel',path=str(Path(__file__).resolve()),sha256=m.sha(__file__))]
    report=dict(schema=m.NAME+'.report',prepared=str(tmp_path/'prepared'),prepare_sha256='a'*64,cell=0,body_sha256=m.digest(b),
        boundary=5,full_closed_loop_complete=False,model_before=p['expected_model_hash'],model_after=p['expected_model_hash'],
        classes_before=classes,classes_after=classes,precision_before=m.primitives.precision_expected(),
        precision_after=m.primitives.precision_expected(),host=dict(device='cuda:0',index=0,name='RAM_fixture',
            allocation_sum=4.,visible_nvidia_nodes=['fixture']),GPU_peak=dict(allocated=1024,reserved=2048),
        sources_verified=True,errors=[],axes=m.axes([e['payload'] for e in store.events]),artifacts=store.inventory(),
        wall_before_report_seconds=1.,**m.FLAGS)
    rh=store.put('report.json',report,final=True)
    completion=dict(schema=m.NAME+'.completion',report_sha256=rh,elapsed_seconds=2.,boundary='after_report_fsync_hash',complete=True,cell=0)
    ch=store.put('completion.json',completion,final=True)
    return store.output,rh,ch


def test_complete_disk_consumer_chain(tmp_path,monkeypatch):
    output,rh,ch=fake_complete_bundle(tmp_path,monkeypatch)
    result=m.consume(output,rh,ch)
    assert result['status']=='control5_cell_native_GD_verified_NOT_admitted'
    assert result['closed_loop_complete'] is False and result['axes']['attempted']==m.numerical.axes27(1)


@pytest.mark.parametrize('mutation',['extra_file','tail_npy','wall','host','error','source','journal'])
def test_complete_consumer_rejects_mutation(tmp_path,monkeypatch,mutation):
    output,rh,ch=fake_complete_bundle(tmp_path,monkeypatch)
    if mutation=='extra_file':(output/'unexpected').write_text('x')
    elif mutation=='tail_npy':
        f=next(output.glob('*.npy'));f.write_bytes(f.read_bytes()+b'trailer')
    elif mutation=='source':pass  # mutate bound temp policy below
    elif mutation=='journal':
        f=output/'events.jsonl';f.write_bytes(f.read_bytes()+b'\n')
    else:
        report=m.read(output/'report.json',rh)
        if mutation=='wall':report['wall_before_report_seconds']=1801.
        elif mutation=='host':report['host']['device']='cpu'
        else:report['errors']=[dict(message='failure')]
        (output/'report.json').write_bytes(m.raw(report));rh=m.sha(output/'report.json')
        comp=m.read(output/'completion.json',ch);comp['report_sha256']=rh
        (output/'completion.json').write_bytes(m.raw(comp));ch=m.sha(output/'completion.json')
    if mutation=='source':(tmp_path/'fake_policy'/'weights.pt').write_bytes(b'changed')
    with pytest.raises(ValueError):m.consume(output,rh,ch)


def test_max_full3node_metadata_RAM_shape():
    # Simulate maximum 32 array descriptors per packet with no dedup, all
    # candidate contexts and 3 nodes, plus complete 19968-file report inventory.
    d=dict(file='a999999.npy',sha256='f'*64,shape=[1,3,192],dtype='<f4')
    f=dict(index=203,tag='iterate32/2',context_sha256='f'*64,
        inputs={str(i):d for i in range(7)},modules=[dict(resource='predictor',inputs=[d,d],output=d)]*3,
        predicted=d,boundary={'latent':d,'history':d},completed=True,extra_budget_only=[d]*14)
    c=deepcopy(body()['root_context']);c['latent_history']=np.full((3,192),np.finfo(np.float32).max,np.float32).tolist()
    c['action_history']=np.full((2,10),np.finfo(np.float32).max,np.float32).tolist()
    candidate=dict(logits=np.full((3,5,2),1.).tolist(),actions=np.full((15,2),1.).tolist(),context=c,cost=1e77)
    row=dict(forwards=[f]*204,AD=[dict(index=i,tag='AD31',completed=True,loss=d,raw=d,gradient=d) for i in range(32)],
        result=dict(candidates=[candidate]*33,gradients=[dict(raw_gradient=np.full((15,2),1.).tolist(),
            objective_gradient=np.full((3,5,2),1.).tolist(),loss=1.)]*32,winner=0,commands=candidate['actions'],context=c,first5=candidate['actions'][:5]))
    inventory={f'a{i:06}.npy':'f'*64 for i in range(m.budget(3)['array_count_cap'])}
    size=3*len(m.raw(row))+len(m.raw(dict(artifacts=inventory)))+1024*1024
    assert size<m.META
