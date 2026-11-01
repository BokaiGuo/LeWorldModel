import json
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import agent_e3_gd_numerical_preparation_20261029_v1 as m


def context():return dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
    latent_history=np.zeros((3,192),np.float32).tolist(),action_history=np.zeros((2,10),np.float32).tolist())


class FakeCUDA:
    """Analytic RAM-only linear recurrence, not a neural CPU fallback."""
    kind='RAM_FIXTURE'
    def __init__(self,body=None):
        self.mean=np.asarray(body['action_mean'] if body else [.01,.01],np.float32)
        self.std=np.asarray(body['action_std'] if body else [.2,.2],np.float32)
    def packet(self,c,a):
        z=np.asarray(c['latent_history'],np.float32);h=np.asarray(c['action_history'],np.float32)
        token=(a-self.mean)/self.std
        outgoing=np.concatenate([h,token.reshape(1,10)])
        new=np.full(192,z[-1,0]+a.sum(dtype=np.float32)*np.float32(.01),np.float32)
        nz=np.concatenate([z[1:],new[None]])
        nc=deepcopy(c);nc.update(latent_history=nz.tolist(),action_history=outgoing[1:].tolist())
        # Fixture IO duplicates retained, not dedup; not a native model witness.
        return nc,dict(latent=z,history=h,raw=a,mean=np.full(2,.01,np.float32),
            std=np.full(2,.2,np.float32),token=token,outgoing=outgoing,
            predicted=np.broadcast_to(new,(1,3,192)).copy(),boundary=nz)
    def reverse(self,a,goal,primal):
        c=primal(a);delta=np.asarray(c['latent_history'][-1],np.float32)-goal.astype(np.float32)
        loss=np.mean(delta**2,dtype=np.float32)
        gradient=np.full(a.shape,np.float32(2)*np.mean(delta,dtype=np.float32)*np.float32(.01),np.float32)
        return c,gradient,loss


@pytest.mark.parametrize('K',[1,3])
@pytest.mark.parametrize('domain',['Dxy','Dx'])
def test_all32_actual_kernel(K,domain):
    t=m.Tape();result=m.optimize_node(context(),np.ones(192).tolist(),K,domain,FakeCUDA(),t)
    assert result['attempted']==result['completed']==dict(predict=68*K,reverse=32)
    assert len(result['candidate_costs'])==33
    assert result['candidate_costs'][result['winner']]<=result['candidate_costs'][0]
    assert len([e for e in t.events if e['kind']=='selection'])==1
    if domain=='Dx':
        y=np.asarray(result['commands'],np.float32)[:,1]
        assert (y==0).all() and not np.signbit(y).any()


def test_ties_first_no_optimization_outcome_selection():
    result=m.optimize_node(context(),np.zeros(192).tolist(),1,'Dxy',FakeCUDA(),m.Tape())
    assert result['winner']==0 and result['candidate_costs']==[0.]*33


def test_real_disk_kernel_reader(tmp_path):
    tape=m.Tape(tmp_path/'tape')
    m.optimize_node(context(),np.ones(192).tolist(),3,'Dx',FakeCUDA(),tape)
    assert m.read_tape(tape.path)==tape.events
    blob=(tape.path/'events.jsonl').read_bytes()
    assert blob.endswith(b'\n') and b'\n\n' not in blob
    assert tape.bytes==sum(p.stat().st_size for p in tape.path.iterdir())
    assert tape.bytes<m.transport_budget(3)['total_bytes_cap']-m.RESERVE
    a=next(tape.path.glob('*.npy'));a.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='SHA'):m.read_tape(tape.path)


def test_temp_build_verify_and_source_seal(tmp_path):
    p=tmp_path/'proof.json';value=m.build();s=m.write_new(p,value)
    assert m.verify(p,s)
    with pytest.raises(FileExistsError):m.write_new(p,value)
    value['proof']['full']['predict_API']+=1
    p.write_bytes(m.raw(value))
    with pytest.raises(ValueError,match='rebuild'):m.verify(p,m.resolver.sha(p))


def test_failure_keeps_attempt_and_no_refund():
    class Fail(FakeCUDA):
        def reverse(self,a,g,p):p(a);raise RuntimeError('AD fixture failure')
    t=m.Tape()
    with pytest.raises(RuntimeError):m.optimize_node(context(),np.ones(192).tolist(),3,'Dxy',Fail(),t)
    assert t.attempted==dict(predict=6,reverse=1)
    assert t.completed==dict(predict=6,reverse=0)
    assert t.events[-1]['kind']=='failed'


@pytest.mark.parametrize('bad',['vector_loss','wrong_point','extra_primal','wrong_gradient'])
def test_bad_AD_contract(bad):
    class Bad(FakeCUDA):
        def reverse(self,a,g,p):
            if bad=='wrong_point':return super().reverse(a+np.float32(.01),g,p)
            c,d,l=super().reverse(a,g,p)
            if bad=='extra_primal':p(a)
            if bad=='vector_loss':l=np.asarray([l],np.float32)
            if bad=='wrong_gradient':d=d[:1]
            return c,d,l
    with pytest.raises(ValueError):m.optimize_node(context(),np.ones(192).tolist(),1,'Dxy',Bad(),m.Tape())


def test_expired_stage_before_dispatch():
    t=m.Tape(clock=lambda:1801.,started=0.)
    with pytest.raises(ValueError,match='wall'):m.optimize_node(context(),np.ones(192).tolist(),1,'Dxy',FakeCUDA(),t)
    assert t.attempted==dict(predict=0,reverse=0)


@pytest.mark.parametrize('K',[True,0,2,3.])
def test_invalid_K(K):
    with pytest.raises(ValueError):m.packet_budget(K)


@pytest.mark.parametrize('K',[1,3])
def test_projection_original_cast_semantics(K):
    x=np.full((K,5,2),4.)
    expected=x.copy()
    for row in expected:row*=m.resolver.PACKET_CAP/np.linalg.norm(row)
    expected=expected.astype(np.float32)
    for row in expected:
        if np.linalg.norm(row.astype(float))>m.resolver.PACKET_CAP:row*=np.float32(1-4*np.finfo(np.float32).eps)
    assert np.array_equal(m.project(x,K,'Dxy'),expected)


def test_integer_budget_and27_axes():
    p=m.proof()
    assert p['full']['predict_API']==8*(204+612)==6528
    assert p['full']['scalar_reverse']==1536
    for K in (1,3):
        a=m.axes27(K);assert len(a)==27 and set(a)==set(m.resources.P3_AXES)
        assert a['grad_API']==a['VJP_directions']==a['raw_backward_callbacks']==32
        assert a['jacobian_API']==a['solve_calls']==a['model_SVD_calls']==0
        b=m.transport_budget(K);assert b['fits'] and b['total_bytes_cap']<512*m.MiB
    assert p['GPU']['fit_proven'] is False and p['extra_axes']['engine_backward'] is None


def test_production_backend_rejected():
    backend=FakeCUDA();backend.kind='HOST_CUDA'
    with pytest.raises(ValueError,match='not authorized'):m.optimize_node(context(),np.ones(192).tolist(),1,'Dxy',backend,m.Tape())


@pytest.mark.parametrize('value',[np.ones(769,np.float32),np.ones(1,np.float64),np.array([np.nan],np.float32)])
def test_array_caps(value):
    with pytest.raises(ValueError):m.Tape().put(value)


def test_actual_resolver_read_only_body_join():
    if not m.ACTUAL_PREPARE.is_file():pytest.skip('actual resolver output not installed')
    bodies=m.load_planner_inputs()
    assert len(bodies)==16
    for body in bodies:
        assert set(body)==set(m.resolver.PLANNER_FIELDS)
        assert len(body['root_context']['action_history'])==2
        assert 'evaluator' not in body and 'native_goal' not in body
    bad=deepcopy(bodies[0]);bad['native_goal']=[0.]*7
    with pytest.raises(ValueError,match='whitelist'):m.validate_body(bad)


def test_scalar_roundtrip_real_disk(tmp_path):
    t=m.Tape(tmp_path/'scalar');d=t.put(np.asarray(.125,np.float32));t.emit('scalar',arrays=[d])
    assert m.read_tape(t.path)==t.events and d['shape']==[]


def test_saved_event_chain_corruption(tmp_path):
    t=m.Tape(tmp_path/'chain');t.emit('one');t.emit('two')
    p=t.path/'events.jsonl';e=json.loads(p.read_bytes().splitlines()[1]);e['index']=3
    e['sha256']=m.digest({k:v for k,v in e.items() if k!='sha256'})
    p.write_bytes(m.raw(t.events[0])+b'\n'+m.raw(e)+b'\n')
    with pytest.raises(ValueError,match='chain'):m.read_tape(t.path)


def test_body_only_factory_RAM():
    body=dict(schema=m.resolver.NAME+'.task',root_context=context(),goal=np.ones(192).tolist(),
        projection=np.eye(192).tolist(),action_mean=np.asarray([-.0087,.0068],np.float32).tolist(),action_std=np.asarray([.2019,.2002],np.float32).tolist(),
        model_sha256=m.resolver.MODEL_SHA,root_sha256=m.resolver.KNOWN_ROOTS[0],horizon_packets=1,
        domain=m.resolver.domain_spec('Dx'),identity_sha256='0'*64,feedback_boundary=5,task_deadline_control=20)
    seen=[]
    def factory(t):seen.append(t);return FakeCUDA(t)
    result=m.optimize_body(body,factory,m.Tape())
    assert seen==[body] and result['attempted']['predict']==68
    assert np.asarray(result['context']['action_history']).shape==(2,10)


def test_wrong_normalized_history_refused():
    class Bad(FakeCUDA):
        def packet(self,c,a):
            nc,io=super().packet(c,a);nc['action_history'][-1]=[0.]*10;return nc,io
    with pytest.raises(ValueError,match='once-normalized'):
        m.optimize_node(context(),np.ones(192).tolist(),1,'Dx',Bad(),m.Tape())


def test_max_transport_payload_real_npy_header(tmp_path):
    t=m.Tape(tmp_path/'max');a=np.arange(768,dtype=np.float32).reshape(1,4,192)
    descs=[t.put(a.copy()) for _ in range(32)]
    t.emit('packet',keys=[str(i) for i in range(32)],arrays=descs)
    assert m.read_tape(t.path)==t.events
    assert t.bytes<32*(4096+3072)+16385


def test_all16_exact_predict_reverse_denominator_RAM():
    attempted=dict(predict=0,reverse=0);completed=dict(attempted)
    for cell in m.resolver.roster():
        for _ in range(3):
            t=m.Tape()
            m.optimize_node(context(),np.ones(192).tolist(),cell['horizon_packets'],cell['domain'],FakeCUDA(),t)
            for key in attempted:
                attempted[key]+=t.attempted[key];completed[key]+=t.completed[key]
    assert attempted==completed==dict(predict=6528,reverse=1536)


def test_coherent_descriptor_dtype_corruption(tmp_path):
    t=m.Tape(tmp_path/'dtype');d=t.put(np.asarray(1.,np.float32));t.emit('scalar',arrays=[d])
    e=deepcopy(t.events[0]);e['arrays'][0]['dtype']='<f8'
    e['sha256']=m.digest({k:v for k,v in e.items() if k!='sha256'})
    (t.path/'events.jsonl').write_bytes(m.raw(e)+b'\n')
    with pytest.raises(ValueError,match='descriptor'):m.read_tape(t.path)
