"""Only pure arithmetic/mock tests. No torch import, GPU or physical execution."""
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import p4_feedback_root_cuda_response_20261027_v1 as m


def limits(**overrides):
    args=dict(wall_seconds=1000,allocated_bytes=1024**2,reserved_bytes=2*1024**2,grad_api_per_arm=192,engine_api_per_arm=192)
    args.update(overrides);return m.caps(**args)


@pytest.fixture
def protocol():
    ctx=dict(m.io.META,latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist())
    actual=deepcopy(ctx);actual['latent_history'][-1]=[float(np.float32(.1))]*192
    requests=[dict(arm=arm,context=c,context_sha256=m.io.digest(c),boundary_step=5,must_recompile=True,can_reuse_initial_jacobian=False)
        for arm,c in (('imagined',ctx),('actual',actual))]
    return dict(root_sha256='0'*64,source_arm='old_Q__uniform',active_metric_id='MOCK',requests=requests,
        goal=[.05]*192,q=np.eye(192).tolist(),normalization=dict(mean=[-.0087,.0068],std=[.2019,.2002],
            source='MOCK_original_public_normalizer',source_sha256='a'*64,training_provenance_verified=False),limits=limits())


class MockBackend:
    def __init__(self,p,ledger,fault=None):self.p,self.ledger,self.fault=p,ledger,fault;self.seen=[]
    def bind(self,req):self.context=deepcopy(req['context']);self.seen.append(deepcopy(req))
    def step(self,raw):
        raw=np.asarray(raw,np.float32)
        norm=self.p['normalization'];mean=np.asarray(norm['mean'],np.float32);std=np.asarray(norm['std'],np.float32)
        hist=np.asarray(self.context['action_history'],np.float32)
        after=dict(self.context,latent_history=self.context['latent_history'][1:]+[(np.asarray(self.context['latent_history'][-1])+np.pad(raw.flatten().astype(float),(0,182))).tolist()],
            action_history=[hist[-1].astype(float).tolist(),((raw-mean)/std).reshape(10).astype(float).tolist()])
        return after
    def forward(self,raw):
        self.ledger.call('action_encoder',lambda:None,shape=(1,3,10))
        return self.ledger.call('predict',lambda:self.step(raw),shape=(1,3,192))
    def nominal(self):return self.forward(np.zeros((5,2)))
    def jacobian(self):
        def jac():
            forward=self.forward(np.zeros((5,2)))
            j=np.eye(192,10)
            for i in range(192):
                self.ledger.call('grad_api',lambda:self.ledger.call('engine_api',lambda:j[i]))
            if self.fault=='bad_J':j[0,0]=np.nan
            if self.fault=='bad_AD_forward':forward['latent_history'][-1][0]+=99
            return j,forward
        return self.ledger.call('jacobian_api',jac)
    def candidate(self,commands):
        c=self.forward(commands)
        if self.fault=='bad_history':c['action_history'][0][0]=99
        if self.fault=='candidate_abort':raise RuntimeError('MOCK candidate interruption')
        return c
    def replay(self,commands):
        c=self.forward(commands)
        if self.fault=='bad_replay':c['latent_history'][-1][0]+=1
        return c


def execute_mock(tmp_path,p,fault=None):
    store=m.Store(tmp_path/'MOCK');journal=m.Journal(store);ledger=m.Ledger(p['limits'],journal)
    backend=MockBackend(p,ledger,fault);rows=[]
    try:m.compile_response(p,backend,ledger,store,rows)
    finally:journal.close()
    return store,ledger,backend,rows


def test_full_mock_two_arms_10_candidates_192_reverse_entries(tmp_path,protocol):
    store,l,b,rows=execute_mock(tmp_path,protocol)
    assert [r['status'] for r in rows]==['response_completed']*2
    assert len(b.seen)==2 and b.seen[1]['context']==protocol['requests'][1]['context']
    for arm in m.ARMS:
        assert l.counts()[arm]=={k:dict(attempted=n,completed=n) for k,n in dict(action_encoder=13,predict=13,jacobian_api=1,grad_api=192,engine_api=192).items()}
        assert len(next(r for r in rows if r['arm']==arm)['candidates'])==10
        assert (store.path/(arm+'.jacobian_forward.json')).is_file()
    assert not l.failed and not l.blocked
    events=[m.io.io.decode(line) for line in (store.path/'events.jsonl').read_bytes().splitlines()]
    assert len([e for e in events if e['event']=='attempt'])==2*(26+1+192+192)
    assert m.io.disk_bytes(store.path)<m.OUTPUT_CAP


def test_stable_tie_keeps_zero(tmp_path,protocol):
    protocol['goal']=[0.]*192
    _,_,_,rows=execute_mock(tmp_path,protocol)
    assert rows[0]['winner']['index']==0


@pytest.mark.parametrize('fault',['bad_J','bad_AD_forward','bad_history','candidate_abort','bad_replay'])
def test_mock_partial_keeps_arrays_no_second_arm(tmp_path,protocol,fault):
    store=m.Store(tmp_path/'MOCK');journal=m.Journal(store);l=m.Ledger(protocol['limits'],journal);rows=[]
    try:
        with pytest.raises((ValueError,RuntimeError)):m.compile_response(protocol,MockBackend(protocol,l,fault),l,store,rows)
        assert l.failed and len(rows)==1
        assert (store.path/'imagined.nominal.json').is_file()
        assert not (store.path/'actual.intent.json').exists()
        with pytest.raises(ValueError):l.begin_arm('actual')
    finally:journal.close()


@pytest.mark.parametrize('field,value',[
    ('wall_seconds',None),('wall_seconds',float('inf')),('wall_seconds',-1),
    ('allocated_bytes',0),('allocated_bytes',True),('reserved_bytes',1),
    ('grad_api_per_arm',191),('engine_api_per_arm',191)])
def test_required_non_guessed_caps(field,value):
    with pytest.raises(ValueError):limits(**{field:value})


@pytest.mark.parametrize('kind',['encode','encoder','projector','backward_api','unknown'])
def test_forbidden_api_fail_sticky(kind):
    events=[];l=m.Ledger(limits(),events.append);l.begin_arm('imagined');l.set_phase('nominal')
    with pytest.raises(ValueError):l.call(kind,lambda:None)
    assert l.failed and l.blocked
    with pytest.raises(ValueError):l.call('action_encoder',lambda:None,shape=(1,3,10))


@pytest.mark.parametrize('device,shape',[('cpu',(1,3,10)),('cuda',(1,1,10)),('cuda',(2,3,10))])
def test_device_and_shape(device,shape):
    l=m.Ledger(limits(),lambda e:None);l.begin_arm('imagined');l.set_phase('nominal')
    with pytest.raises(ValueError):l.call('action_encoder',lambda:None,device=device,shape=shape)
    assert l.failed


def test_attempt_charged_on_exception_and_journal_failure():
    l=m.Ledger(limits(),lambda e:None);l.begin_arm('imagined');l.set_phase('nominal')
    def abort():raise RuntimeError('MOCK error')
    with pytest.raises(RuntimeError):l.call('action_encoder',abort,shape=(1,3,10))
    assert len(l.entries)==1 and not l.entries[0]['completed'] and l.failed
    l=m.Ledger(limits(),lambda e:None);l.begin_arm('imagined');l.set_phase('nominal')
    def broken(e):raise OSError('MOCK fsync failure')
    l.emit=broken
    with pytest.raises(OSError):l.call('action_encoder',lambda:None,shape=(1,3,10))
    assert l.failed and len(l.entries)==1


def test_duplicate_extra_forward_and_wrong_order():
    l=m.Ledger(limits(),lambda e:None);l.begin_arm('imagined');l.set_phase('nominal')
    l.call('action_encoder',lambda:None,shape=(1,3,10))
    with pytest.raises(ValueError):l.call('action_encoder',lambda:None,shape=(1,3,10))
    assert l.failed
    l=m.Ledger(limits(),lambda e:None);l.begin_arm('imagined');l.set_phase('nominal')
    with pytest.raises(ValueError):l.call('predict',lambda:None,shape=(1,3,192))


@pytest.mark.parametrize('fault',['wall','allocated','reserved'])
def test_resource_boundary_caps(fault):
    tick=[0.];peak=[0,0];l=m.Ledger(limits(),lambda e:None,clock=lambda:tick[0],sample=lambda:tuple(peak))
    l.begin_arm('imagined');l.set_phase('nominal')
    def work():
        if fault=='wall':tick[0]=1001
        else:peak[0 if fault=='allocated' else 1]=3*1024**2
    with pytest.raises(ValueError):l.call('action_encoder',work,shape=(1,3,10))
    assert l.failed and len(l.entries)==1 and not l.entries[0]['completed']


def test_prepare_verify_mock_and_source_drift(tmp_path,protocol,monkeypatch):
    def mock_material(*args):return dict(protocol,schema=m.NAME+'.protocol',sourcepack='MOCK',sourcepack_prepare_sha256='MOCK')
    monkeypatch.setattr(m,'material',mock_material)
    # Real small source sealing, but inputs are explicitly mock and temporary.
    monkeypatch.setattr(m,'sources',lambda:[Path(m.__file__).resolve()])
    out=tmp_path/'MOCK_prepare'
    sha=m.prepare('MOCK','MOCK',out,protocol['limits']);assert m.verify(out,sha)['root_sha256']=='0'*64
    with pytest.raises(ValueError):m.prepare('MOCK','MOCK',out,protocol['limits'])
    with pytest.raises(ValueError):m.verify(out,'bad')
    protocol['goal'][0]=99
    with pytest.raises(ValueError):m.verify(out,sha)


def test_missing_sourcepack_writes_nothing(tmp_path):
    out=tmp_path/'never_created'
    with pytest.raises((ValueError,FileNotFoundError)):m.prepare(tmp_path/'missing','bad',out,limits())
    assert not out.exists()


def test_storage_reserve(tmp_path,monkeypatch):
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:SimpleNamespace(free=m.RESERVE-1))
    with pytest.raises(ValueError):m.prepare('MOCK','MOCK',tmp_path/'not_created',limits())
    assert not (tmp_path/'not_created').exists()


def test_no_neural_or_physics_import_and_explicit_only_run():
    assert not any(x in sys.modules for x in ('torch','stable_worldmodel','gymnasium','pymunk'))
    assert m.SOURCEPACK_SHA=='4de418b03fb7e4a664d9e5d04d8c78b7f7ac467e89f04677329b463fb6489fe2'


def test_mock_native_hook_aliases_and_restore():
    events=[];l=m.Ledger(limits(),events.append);l.begin_arm('imagined');l.set_phase('nominal')
    tensor=lambda shape:SimpleNamespace(shape=shape,device=SimpleNamespace(type='cuda'))
    model=SimpleNamespace(predict=lambda emb,act:'MOCK_STATE',encode=lambda x:x,
        action_encoder=SimpleNamespace(forward=lambda x:'MOCK_ACTION'),
        encoder=SimpleNamespace(forward=lambda x:x),projector=SimpleNamespace(forward=lambda x:x))
    engine=lambda *a,**k:'MOCK_GRAD'
    t=SimpleNamespace(autograd=SimpleNamespace(backward=lambda *a:None,_engine_run_backward=engine,
        graph=SimpleNamespace(_engine_run_backward=engine),functional=SimpleNamespace()))
    def grad(*a,**k):return t.autograd._engine_run_backward()
    def jac(*a,**k):
        assert k==dict(vectorize=False,strict=True,create_graph=False,strategy='reverse-mode')
        model.action_encoder.forward(tensor((1,3,10)));model.predict(tensor((1,3,192)),None)
        for _ in range(192):t.autograd.grad()
        return 'MOCK_J'
    t.autograd.grad=grad;t.autograd.functional.jacobian=jac
    hooks=m.io.Hooks();original=model.predict
    m.install_hooks(t,model,l,hooks)
    model.action_encoder.forward(tensor((1,3,10)));model.predict(tensor((1,3,192)),None)
    l.set_phase('jacobian')
    assert t.autograd.functional.jacobian(None,None,vectorize=False,strict=True,create_graph=False,strategy='reverse-mode')=='MOCK_J'
    assert l.counts()['imagined']['engine_api']['completed']==192
    assert l.counts()['imagined']['grad_api']['completed']==192
    assert all(e['event']!='blocked' for e in events)
    errors=[];hooks.restore(errors);assert not errors and model.predict is original
    assert t.autograd.graph._engine_run_backward is engine and t.autograd._engine_run_backward is engine


def test_public_backward_hook_denied_and_restored():
    l=m.Ledger(limits(),lambda e:None)
    model=SimpleNamespace(predict=lambda *a:None,encode=lambda *a:None,action_encoder=SimpleNamespace(forward=lambda *a:None),
        encoder=SimpleNamespace(forward=lambda *a:None),projector=SimpleNamespace(forward=lambda *a:None))
    original=lambda *a:None
    t=SimpleNamespace(autograd=SimpleNamespace(backward=original,grad=lambda *a:None,_engine_run_backward=lambda *a:None,
        graph=SimpleNamespace(_engine_run_backward=lambda *a:None),functional=SimpleNamespace(jacobian=lambda *a:None)))
    h=m.io.Hooks();m.install_hooks(t,model,l,h)
    try:
        with pytest.raises(ValueError):t.autograd.backward()
        assert l.failed and l.blocked[0]['kind']=='backward_api'
    finally:h.restore([])
    assert t.autograd.backward is original


def test_insufficient_reverse_cap_and_reentry():
    c=limits();c['engine_api_per_arm']=1;l=m.Ledger(c,lambda e:None)
    l.begin_arm('imagined');l.set_phase('nominal');l.call('action_encoder',lambda:None,shape=(1,3,10));l.call('predict',lambda:None,shape=(1,3,192));l.set_phase('jacobian')
    def jac():
        l.call('action_encoder',lambda:None,shape=(1,3,10));l.call('predict',lambda:None,shape=(1,3,192))
        for _ in range(2):l.call('grad_api',lambda:l.call('engine_api',lambda:None))
    with pytest.raises(ValueError):l.call('jacobian_api',jac)
    assert l.failed and l.counts()['imagined']['engine_api']==dict(attempted=1,completed=1)


@pytest.mark.parametrize('fault',['context_sha','boundary','reuse_J','cross_arm_history','nonfinite_Q','wrong_scope'])
def test_pack_contract_rejects_tampering(protocol,fault):
    p=dict(protocol,schema=m.source.NAME+'.protocol',status='prepared_missing_actual_response',prior_root_sha256s=['1'*64],
        response_requests=deepcopy(protocol['requests']),runtime_admissible=False,response_completed=False,independent_confirmation=False,
        P1_admission=False,P4_admission=False,model_bindings={'/MOCK/config.json':'fake','/MOCK/weights.pt':'fake'})
    if fault=='context_sha':p['response_requests'][0]['context_sha256']='bad'
    elif fault=='boundary':p['response_requests'][1]['boundary_step']=20
    elif fault=='reuse_J':p['response_requests'][1]['can_reuse_initial_jacobian']=True
    elif fault=='cross_arm_history':
        r=p['response_requests'][1];r['context']['action_history'][0][0]=99;r['context_sha256']=m.io.digest(r['context'])
    elif fault=='nonfinite_Q':p['q'][0][0]=float('nan')
    elif fault=='wrong_scope':p['runtime_admissible']=True
    with pytest.raises(ValueError):m.validate_pack(p)


@pytest.mark.parametrize('path',['transition','replay'])
def test_native_normalization_with_real_pack_metadata_layout(protocol,path):
    """Exercise Native methods with numpy tensor mocks, NOT CPU neural forwards.

    Sourcepack's normalization contains mean/std/source/source_sha256/
    training_provenance_verified. The released normalizer accepts ONLY mean/std.
    """
    class Tensor(np.ndarray):
        def detach(self):return self
        def cpu(self):return self
    tensor=lambda a,**kwargs:np.asarray(a,np.float32).view(Tensor)
    t=SimpleNamespace(tensor=tensor,float32='MOCK_float32',cat=lambda xs,dim:np.concatenate(xs,axis=dim).view(Tensor),
        isfinite=np.isfinite,no_grad=nullcontext)
    model=SimpleNamespace(action_encoder=lambda a:a,predict=lambda emb,act:emb+np.float32(.01))
    n=m.Native(t,model,SimpleNamespace(type='cuda'),protocol)
    n.bind(protocol['requests'][1]);seen=[]
    def normalizer(raw,*,mean,std):
        seen.append(dict(mean=mean,std=std))
        return ((raw-np.asarray(mean,np.float32))/np.asarray(std,np.float32)).reshape(10).view(Tensor)
    n.normalize=normalizer
    raw=tensor(np.zeros((5,2)))
    if path=='transition':result=n.export(*n.transition(raw))
    else:result=n.replay(raw.tolist())
    assert seen==[dict(mean=protocol['normalization']['mean'],std=protocol['normalization']['std'])]
    m.io.history(protocol['requests'][1]['context'],result,raw,protocol['normalization'])
    assert result['latent_history'][-1]!=protocol['requests'][1]['context']['latent_history'][-1]
