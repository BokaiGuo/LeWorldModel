from copy import deepcopy
import json
import sys
import types
from contextlib import nullcontext
import pytest
import numpy as np
from experiments.spectral_reachability import agent_p2_grasp_interleave_cuda_20261029_v1 as m
from test_agent_p2_intersection_host_cuda_20261029_v1 import test_actual_grasp_host_step_with_fake_tensor_and_saved_replay
from test_agent_p2_common_resource_grasp_cuda_20261029_v1 import task


def blob(value='00000000'):
    return dict(dtype='float32',shape=[1],stride=[1],raw_hex=value)


def events():
    out=[]
    for i in range(3):
        out.append(dict(kind='forward_attempt',index=i))
        for k in ('action_encoder','predictor','pred_proj'):
            out.extend([dict(kind='module_attempt',index=i,resource=k,inputs=[blob()]),
                        dict(kind='module_return',index=i,resource=k,output=blob())])
        out.extend([dict(kind='predict_return',index=i,output=blob()),dict(kind='forward_return',index=i)])
    return out


def test_source_binding():
    assert m.audit.inspect_sources()['cuda_parity']=='not_tested'


def test_packet_order_and_missing():
    assert len(m.packet_view(events()))==3
    e=events();e.pop(2)
    with pytest.raises(ValueError):m.packet_view(e)


def test_bitwise_includes_signedzero_stride_and_shape():
    a={'x':blob()};b=deepcopy(a);b['x']['raw_hex']='00000080'
    assert not m.compare(a,b)['x']['bytes_equal']
    b=deepcopy(a);b['x']['stride']=[2]
    assert not m.compare(a,b)['x']['metadata_equal']
    b=deepcopy(a);b['x']['shape']=[2]
    with pytest.raises(ValueError):m.compare(a,b)


def test_no_authorization_before_any_input_access():
    with pytest.raises(ValueError,match='authorization'):
        m.run('/no/such/file','0'*64,'/no/such/output','')


def test_exclusive_metadata(tmp_path):
    p=tmp_path/(m.NAME+'_fixture.json');m.write(p,{'a':1})
    with pytest.raises(ValueError):m.write(p,{'a':2})


def test_partial_journal_is_immediately_durable(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_partial'))
    s.event({'side':'original','event':{'kind':'attempt'}})
    assert (s.path/'events.jsonl').read_bytes()==m.host.box.raw({'side':'original','event':{'kind':'attempt'}})+b'\n'
    with pytest.raises((ValueError,FileNotFoundError)):
        m.consume(s.path,'0'*64)


def build_host_fixture():
    class Tensor:
        def __init__(self,x):self.a=np.asarray(x,np.float32);self.shape=self.a.shape
        def repeat(self,n):return Tensor(np.tile(self.a,n))
        def reshape(self,*shape):return Tensor(self.a.reshape(*shape))
        def __getitem__(self,key):return Tensor(self.a[key])
        def __sub__(self,other):return Tensor(self.a-other.a)
        def __truediv__(self,other):return Tensor(self.a/other.a)
    class FakeTorch:
        @staticmethod
        def cat(xs,axis):return Tensor(np.concatenate([x.a for x in xs],axis=axis))
    events=[];tape=m.host.Tape(0,'grasp',events.append,lambda:0.);assert tape.admit('joint:0')
    b=object.__new__(m.host.GraspBackend);b.t=FakeTorch;b.task=task();b.event=events.append;b.budget=tape
    b.row=dict(neural=[]);b.n=0;b.active=None;b.current=None
    b.tensor=lambda a,dtype=None:Tensor(a);b.snap=lambda x:m.host.grasp.pack(x.a)
    b.check=lambda:events.append(dict(kind='memory',allocated=1,reserved=2))
    class FakeModel:
        def action_encoder(self,x):return b.module('AE',lambda x:Tensor(np.zeros((1,3,192),np.float32)),(x,))
        def predict(self,z,a):
            tape.predict_api()
            p=b.module('predictor',lambda z,a:Tensor(np.zeros((1,3,192),np.float32)),(z,a))
            return b.module('lowproj',lambda x:Tensor(np.zeros((3,192),np.float32)),(p.reshape(3,192),)).reshape(1,3,192)
    b.model=FakeModel();t=b.task;z=Tensor(np.asarray(t['root_context']['latent_history'])[None]);h=Tensor(np.asarray(t['root_context']['action_history'])[None]);goal=np.asarray(t['goal'],np.float32)[None];v=z.a[:,-1,None]+np.asarray([1/3,2/3],np.float32).reshape(1,2,1)*(goal[:,None]-z.a[:,-1,None])
    for k in range(3):
        z,h=b.step(z,h,Tensor(np.zeros((5,2))),f'lifted/{k}')
        if k<2:z=Tensor(np.concatenate((z.a[:,:-1],v[:,k:k+1]),axis=1))
    tape.call('score',lambda:None,trajectories=1,scope='lifted_loss')
    tape.call('backward',lambda:events.append(dict(kind='chart_backward',raw_gradient=m.host.grasp.pack(np.zeros((3,5,2),np.float32)))))
    b.check()
    def desc(a):return dict(dtype=str(a.dtype),shape=list(a.shape),stride=list(np.asarray(a).strides[i]//a.itemsize for i in range(a.ndim)),raw_hex=a.tobytes().hex())
    side=dict(events=events,neural=b.row['neural'],accounting=tape.ledger.summary(),inputs=dict(
        x=desc(np.zeros(30,np.float64)),v=desc(v),z=desc(np.asarray(t['root_context']['latent_history'],np.float32)[None]),goal=desc(np.asarray(t['goal'],np.float32)[None])))
    return t,side


def test_three_actual_host_steps_and_full_saved_budget_replay():
    t,side=build_host_fixture()
    m.replay_host(t,side)
    bad=deepcopy(side);bad['accounting']['used']['backward']+=1
    with pytest.raises(ValueError):m.replay_host(t,bad)


def test_prepare_verify_complete_consume_and_tamper_tempdisk(tmp_path,monkeypatch):
    """Fixture-only authorities; real 3-packet HOST control and saved replay."""
    t,side=build_host_fixture()
    monkeypatch.setattr(m.host.common,'build',lambda p,h:dict(slots=[{'fixture':True}],bindings_sha256={},expected_model_hash='fixture'))
    monkeypatch.setattr(m.host.common,'join_tasks',lambda _: [t])
    monkeypatch.setattr(m.host.old.parent,'check_files',lambda _:None)
    monkeypatch.setattr(m.host.old.loader,'validate_loader',lambda loader,p,h:None)
    protocol=tmp_path/'protocol.json';protocol.write_bytes(b'fixture')
    prepared=tmp_path/(m.NAME+'_prepare.json');rec=m.write(prepared,m.build(protocol,m.host.sha(protocol)))
    p=m.verify(prepared,rec['sha256']);assert p['root']==0
    layout=[dict(label=f'lifted/{i}') for i in range(3)]
    metric=blob();side.update(packet_layouts=layout,loss=metric,dyn=metric,dense=metric,gx=metric,gv=metric)
    originals=[e for e in side['events'] if e.get('kind') in {'forward_attempt','module_attempt','module_return','predict_return','forward_return'}]
    original={k:deepcopy(v) for k,v in side.items() if k not in ('neural','accounting')};original['events']=originals
    out=tmp_path/(m.NAME+'_fixture');store=m.Store(out)
    loader={'files':[{'path':'fixture','sha256':'a'*64}]};store.put('loader.json',loader)
    store.event(dict(kind='load_intent',path='fixture',expected_sha256='a'*64))
    store.event(dict(kind='consumed_buffer_frozen',receipt=loader['files'][0]))
    for kind,value in [('original',original),('host',side)]:
        for e in value['events']:store.event(dict(side=kind,event=e))
        store.put(kind+'.json',value)
    guard=dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,MATH=True,FLASH=False,MEM_EFFICIENT=False,CUDNN=False)
    report=dict(schema=m.NAME+'.report',prepare=str(prepared),prepare_sha256=rec['sha256'],status='complete',errors=[],
        model_before='fixture',model_after='fixture',precision_before=guard,precision_after=guard,
        artifacts={x.name:m.host.sha(x) for x in out.iterdir()},prediction_calls=6,backward_calls=2,scientific_admission=False)
    store.put('report.json',report);sha=m.host.sha(out/'report.json')
    result=m.consume(out,sha);assert len(result['packets'])==3 and result['all_exact']
    # Rehashing the outer report cannot conceal a corrupted side/ledger artifact.
    hp=out/'host.json';bad=json.loads(hp.read_bytes());bad['accounting']['used']['backward']+=1
    hp.write_bytes(m.host.box.raw(bad));report['artifacts']['host.json']=m.host.sha(hp)
    (out/'report.json').write_bytes(m.host.box.raw(report))
    with pytest.raises(ValueError):m.consume(out,m.host.sha(out/'report.json'))
    bad_report=deepcopy(report);bad_report['status']='failed_partial';bad_report['errors']=[{'type':'fixture'}]
    (out/'report.json').write_bytes(m.host.box.raw(bad_report))
    with pytest.raises(ValueError,match='complete only'):m.consume(out,m.host.sha(out/'report.json'))


def test_run_failure_retains_prior_side_and_durable_events(tmp_path,monkeypatch):
    """Run's real finalizer with non-neural fake CUDA interfaces, no GPU."""
    fake=types.ModuleType('torch');fake.__path__=[]
    fake.cuda=types.SimpleNamespace();fake.device=lambda value:value
    fake.backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),
                                       cudnn=types.SimpleNamespace(allow_tf32=True))
    fake.enable_grad=lambda:nullcontext();fake.autocast=lambda **kw:nullcontext()
    nn=types.ModuleType('torch.nn');nn.__path__=[];attention=types.ModuleType('torch.nn.attention')
    attention.SDPBackend=types.SimpleNamespace(MATH='fixture');attention.sdpa_kernel=lambda _:nullcontext()
    monkeypatch.setitem(sys.modules,'torch',fake);monkeypatch.setitem(sys.modules,'torch.nn',nn)
    monkeypatch.setitem(sys.modules,'torch.nn.attention',attention)
    import experiments.gpu_preflight as preflight
    monkeypatch.setattr(preflight,'require_cuda_allocation',lambda _:None)
    p={'input':{'slots':[],'expected_model_hash':'fixture'}}
    monkeypatch.setattr(m,'verify',lambda *_:p)
    monkeypatch.setattr(m.host.common,'join_tasks',lambda _: [None])
    monkeypatch.setattr(m.host.old.loader,'loaded_model',lambda *args:(object(),'fixture'))
    monkeypatch.setattr(m.host.old.loader.enc,'frozen',lambda _:None)
    monkeypatch.setattr(m.host.old.loader.enc,'tensor_hash',lambda _:'fixture')
    monkeypatch.setattr(m.host.old.parent,'check_files',lambda _:None)
    monkeypatch.setattr(m.host.grasp,'precision',lambda _: {'fixture':True})
    def execute(t,model,device,task,store,kind):
        store.event(dict(side=kind,event=dict(kind='attempt')))
        if kind=='host':raise ValueError('injected failure')
        return {'fixture':True}
    monkeypatch.setattr(m,'execute_side',execute)
    output=tmp_path/(m.NAME+'_failure')
    result=m.run('fixture-prepared','fixture-sha',output,m.AUTH)
    report=json.loads((output/'report.json').read_bytes())
    assert report['status']=='failed_partial' and report['errors'][0]['message']=='injected failure'
    assert (output/'original.json').is_file() and not (output/'host.json').exists()
    assert len((output/'events.jsonl').read_bytes().splitlines())==2
    with pytest.raises(ValueError,match='complete only'):m.consume(output,result['sha256'])


def test_original_native_three_steps_real_hooks_fake_arrays():
    class Tensor:
        def __init__(self,x):self.a=np.asarray(x,np.float32);self.shape=self.a.shape
        def repeat(self,n):return Tensor(np.tile(self.a,n))
        def reshape(self,*s):return Tensor(self.a.reshape(*s))
        def __getitem__(self,k):return Tensor(self.a[k])
        def __sub__(self,o):return Tensor(self.a-o.a)
        def __truediv__(self,o):return Tensor(self.a/o.a)
    class FakeTorch:
        @staticmethod
        def cat(xs,axis):return Tensor(np.concatenate([v.a for v in xs],axis=axis))
    class Module:
        def __init__(self,fn):self.fn=fn;self.pre=[];self.post=[]
        def register_forward_pre_hook(self,fn):
            self.pre.append(fn);return types.SimpleNamespace(remove=lambda:self.pre.remove(fn))
        def register_forward_hook(self,fn):
            self.post.append(fn);return types.SimpleNamespace(remove=lambda:self.post.remove(fn))
        def __call__(self,*args):
            for f in self.pre:f(self,args)
            out=self.fn(*args)
            for f in self.post:f(self,args,out)
            return out
    class Model:
        def __init__(self):
            self.action_encoder=Module(lambda x:Tensor(np.zeros((1,3,192))))
            self.predictor=Module(lambda x,y:Tensor(np.zeros((1,3,192))))
            self.pred_proj=Module(lambda x:Tensor(np.zeros((3,192))))
            self.encoder=Module(lambda x:x);self.projector=Module(lambda x:x)
        def predict(self,z,a):return self.pred_proj(self.predictor(z,a).reshape(3,192)).reshape(1,3,192)
    events=[];b=object.__new__(m.host.grasp.Native);b.t=FakeTorch;b.task=task();b.event=events.append
    b.n=0;b.active=None;b.model=Model();b.tensor=lambda x,dtype=None:Tensor(x)
    b.snap=lambda x:m.host.grasp.pack(x.a);b.check=lambda:None
    z=Tensor(np.asarray(b.task['root_context']['latent_history'])[None]);h=Tensor(np.asarray(b.task['root_context']['action_history'])[None])
    with b.observe():
        for i in range(3):z,h=b.step(z,h,Tensor(np.zeros((5,2))),f'lifted/{i}')
    assert len(m.packet_view(events))==3 and b.n==3
    assert all(not getattr(b.model,n).pre and not getattr(b.model,n).post for n in ('action_encoder','predictor','pred_proj','encoder','projector'))
