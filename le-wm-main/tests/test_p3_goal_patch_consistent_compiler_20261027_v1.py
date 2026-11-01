"""Non-neural tests: algebraic backend is not model/physics evidence."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import p3_goal_patch_consistent_compiler_20261027_v1 as m


def packet():
    return dict(root_sha256='root', goal=[1., 2.]+[0.]*190,
        root_context=dict(layout='states_with_between_state_actions_v1', context_length=3,
            latent_dim=192, action_dim=10, latent_history=np.zeros((3,192)).tolist(),
            action_history=np.zeros((2,10)).tolist()))


class LinearBackend:
    """Analytic stand-in, never imports torch or a real model."""
    def __init__(self):
        self.calls=0
        self.j=np.zeros((192,10)); self.j[0,0]=1; self.j[1,1]=2

    def step(self,context,actions):
        self.calls+=1
        c=deepcopy(context)
        z=np.asarray(c['latent_history'][-1])+self.j@np.asarray(actions).reshape(10)
        c['latent_history']=(c['latent_history'][1:]+[z.tolist()])
        c['action_history']=(c['action_history'][1:]+[np.asarray(actions).reshape(10).tolist()])
        return c

    def replay(self,context,actions):
        for raw in np.asarray(actions).reshape(-1,5,2):context=self.step(context,raw)
        return context

    def nominal_jacobian(self,context,operation):
        # AD forward charged separately from nominal, as the real backend does.
        nominal=self.step(context,np.zeros((5,2)))
        operation('jacobian',lambda: self.step(context,np.zeros((5,2))))
        return np.asarray(nominal['latent_history'][-1]),self.j.copy()


def test_q_changes_generated_actions_not_just_scores():
    p=packet(); q0=np.diag([1.,0.]+[0.]*190); q1=np.diag([0.,1.]+[0.]*190)
    a=m.compile_arm(p,q0,LinearBackend(),lambda e:None)
    b=m.compile_arm(p,q1,LinearBackend(),lambda e:None)
    assert a['bank']['params']!=b['bank']['params']
    assert a['candidates'][0]['suffix_solve']['params']!=b['candidates'][0]['suffix_solve']['params']
    assert a['q_sha256']!=b['q_sha256']


def test_all_candidates_duplicate_zero_and_independent_replay_are_kept():
    backend=LinearBackend(); q=np.eye(192)
    result=m.compile_arm(packet(),q,backend,lambda e:None)
    assert backend.calls==66
    assert len(result['candidates'])==10
    assert len(result['operations'])==23  # 11 AD + 12 SVD (bank pair + suffixes)
    assert all(o['completed'] for o in result['operations'])
    assert all(c['independent_model_replay_exact'] for c in result['candidates'])
    assert result['whole_and_segmented_exact']
    assert all(c['q_sha256']==result['q_sha256'] for c in result['candidates'])
    for c in result['candidates']:
        assert c['stage_cost']==m.cost(c['predicted_prefix'],packet()['goal'],q)
        assert c['nonlinear_terminal_cost']==m.cost(c['predicted_terminal'],packet()['goal'],q)
        assert c['objective']==c['stage_cost']+c['suffix_solve']['own_reach_value']


def test_ridge_ball_kkt_and_no_projection_fallback():
    b=LinearBackend(); q=np.eye(192)
    bank=m.make_bank(b.j,np.zeros(192),np.ones(192),q)
    assert len(bank['params'])==10
    assert np.max(np.linalg.norm(bank['params'],axis=1))<=m.compiler.CAP+1e-12
    with pytest.raises(ValueError):m.validate_q(np.zeros((192,192)))
    with pytest.raises(ValueError):m.validate_q(np.eye(192)*.5)
    with pytest.raises(ValueError):m.validate_q(np.eye(191))


def test_failed_work_is_durably_visible():
    events=[]
    class Broken(LinearBackend):
        def replay(self,*args):raise RuntimeError('replay failure')
    with pytest.raises(RuntimeError,match='replay failure'):
        m.compile_arm(packet(),np.eye(192),Broken(),events.append)
    candidates=[e for e in events if e['kind']=='candidate']
    assert candidates and candidates[-1]['completed'] is False
    assert any(e['kind']=='jacobian' and e['completed'] for e in events)


def test_matching_contract_is_h2_prerequisite_not_p3_dynamic():
    c=m.contract()
    assert c['arm_predict_cap']==66 and c['pair_predict_cap']==132
    assert c['four_root_predict_cap']==528 and c['jacobians_per_arm']==11
    assert c['svds_per_arm']==12 and c['mathematical_vjp_directions_per_arm']==2112
    assert not c['P3_admission'] and not c['dynamic_scheduler_implemented']


def test_cap_charges_failed_attempt_and_blocks_before_dispatch():
    events=[]; cap=m.NeuralCap(None,events.append); cap.role=m.ARMS[0]
    x=SimpleNamespace(device=SimpleNamespace(type='cuda'),shape=(1,3,192))
    def fail(*a):raise RuntimeError('failed')
    with pytest.raises(RuntimeError):cap.dispatch('predict',fail,(x,),{})
    assert cap.receipt()['attempted']['predict']==1
    assert cap.receipt()['completed']['predict']==0
    for _ in range(65):cap.dispatch('predict',lambda x: x,(x,),{})
    with pytest.raises(ValueError,match='cap'):cap.dispatch('predict',lambda x:x,(x,),{})
    assert cap.receipt()['attempted']['predict']==66 and cap.blocked==1


def test_fixed_bank_exactly_uses_official_local_ridge_spectral_program():
    b=LinearBackend(); g=np.arange(192,dtype=float)/100
    expected=m.compiler.spectral_bank(b.j,np.zeros(192),g,np.eye(192))
    assert m.make_bank(b.j,np.zeros(192),g,np.eye(192))==expected


def test_no_runtime_imports_at_module_import():
    import subprocess,sys
    script='import sys; from experiments.spectral_reachability import '+m.NAME+'; assert "torch" not in sys.modules; assert "pymunk" not in sys.modules; assert "gymnasium" not in sys.modules'
    subprocess.run([sys.executable,'-c',script],check=True)


def test_saved_compilation_verifier_rebuilds_bank_and_all_conditional_solves():
    p=packet();q=np.eye(192);result=m.compile_arm(p,q,LinearBackend(),lambda e:None)
    assert m.validate_compilation(p,q,result)==result['winner']


@pytest.mark.parametrize('mutation',[
    lambda r:r['candidates'][0].update(stage_cost=-1),
    lambda r:r['candidates'][0].update(objective=-1),
    lambda r:r['candidates'][0]['commands'][0].__setitem__(0,.5),
    lambda r:r['candidates'][0]['suffix_solve'].update(params=[0.]*10),
    lambda r:r['candidates'].pop(),
    lambda r:r.update(q_sha256='fake'),
    lambda r:r.update(whole_replay_context={}),
    lambda r:r['operations'][0].update(completed=False),
])
def test_saved_witness_tampering_is_rejected(mutation):
    p=packet();q=np.eye(192);r=m.compile_arm(p,q,LinearBackend(),lambda e:None)
    mutation(r)
    with pytest.raises(ValueError):m.validate_compilation(p,q,r)


def test_root_model_goal_metric_identity_are_bound():
    p=packet();q=np.eye(192)
    identity=m.metric_id(p,q,{'weights':'sha'},'metric_source')
    changed=deepcopy(p);changed['goal'][0]=5
    assert m.metric_id(changed,q,{'weights':'sha'},'metric_source')!=identity
    assert m.metric_id(p,q,{'weights':'other'},'metric_source')!=identity
    assert m.metric_id(p,q,{'weights':'sha'},'other')!=identity


def test_zero_response_and_duplicate_commands_not_removed():
    class Zero(LinearBackend):
        def __init__(self):super().__init__();self.j[:]=0
    r=m.compile_arm(packet(),np.eye(192),Zero(),lambda e:None)
    assert len(r['candidates'])==10
    assert r['candidates'][0]['commands']==r['candidates'][1]['commands']
    assert r['winner']['index']==0


def test_replay_mismatch_is_not_softened_to_endpoint_allclose():
    class Wrong(LinearBackend):
        def replay(self,*a):
            c=super().replay(*a);c['action_history'][0][0]+=1e-12;return c
    with pytest.raises(ValueError,match='full-context'):
        m.compile_arm(packet(),np.eye(192),Wrong(),lambda e:None)


def test_action_encoder_and_pair_caps_are_independent():
    cap=m.NeuralCap(None,lambda e:None)
    x=SimpleNamespace(device=SimpleNamespace(type='cuda'),shape=(1,3,10))
    for arm in m.ARMS:
        cap.role=arm
        for _ in range(66):cap.dispatch('action_encoder',lambda x:x,(x,),{})
    assert cap.receipt()['attempted']['action_encoder']==132
    assert cap.receipt()['attempted']['predict']==0
    with pytest.raises(ValueError):cap.dispatch('action_encoder',lambda x:x,(x,),{})


def fake_host(monkeypatch,tmp_path,broken=False):
    """No torch module/model: exercises real runner+caps with algebra only."""
    import sys
    from experiments import spectral_reachability as package
    module_name='experiments.spectral_reachability.run_pusht_d2p_budget_guard_cuda'
    checked=SimpleNamespace(digest=lambda p:m.io.sha(__import__('pathlib').Path(p).read_bytes()),
        tensor_state_hash=lambda model:'frozen')
    monkeypatch.setitem(sys.modules,module_name,checked)
    monkeypatch.setattr(package,'run_pusht_d2p_budget_guard_cuda',checked,raising=False)
    device=SimpleNamespace(type='cuda')
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(device=lambda _:device,cuda=SimpleNamespace(synchronize=lambda _:None)))
    class Model:
        def __init__(self):
            self.predict=lambda x:None
            self.action_encoder=SimpleNamespace(forward=lambda x:None)
            self.encoder=SimpleNamespace(forward=lambda x:None)
        def eval(self):return self
        def requires_grad_(self,_):return self
    model=Model()
    monkeypatch.setitem(sys.modules,'experiments.checkpoint_compat',SimpleNamespace(load_compatible_pretrained=lambda *a,**k:model))
    monkeypatch.setitem(sys.modules,'experiments.gpu_preflight',SimpleNamespace(require_cuda_allocation=lambda _:{'fixture':True}))
    class Backend(LinearBackend):
        def __init__(self,model,device):super().__init__();self.model=model
        def step(self,*args):
            self.model.action_encoder.forward(SimpleNamespace(device=device,shape=(1,3,10)))
            self.model.predict(SimpleNamespace(device=device,shape=(1,3,192)))
            if broken and self.calls>=7:raise RuntimeError('fixture kernel failure')
            return super().step(*args)
    monkeypatch.setattr(m,'TorchBackend',Backend)
    weight=tmp_path/'weights.pt';weight.write_bytes(b'NOT A CHECKPOINT: algebra test fixture')
    packets=[]
    for i in range(4):
        p=packet();p.update(root_sha256=str(i),source_input_sha256='old'+str(i),
            projections={a:np.eye(192).tolist() for a in m.ARMS},
            active_metric_ids={a:a+str(i) for a in m.ARMS},input_sha256='new'+str(i));packets.append(p)
    body=dict(packets=packets,roots=[str(i) for i in range(4)],model_bindings={str(weight):checked.digest(weight)},
        arm_order=[list(m.ARMS if i%2==0 else m.ARMS[::-1]) for i in range(4)])
    monkeypatch.setattr(m,'validate_prepared',lambda *a:(body,'fixture-id'))
    return body


def test_full_eight_arm_runner_and_saved_export_are_bounded_without_neural(monkeypatch,tmp_path):
    fake_host(monkeypatch,tmp_path)
    output=tmp_path/'execution'
    report=m.execute_cuda(tmp_path/'prepared','sha',output)
    assert len(report['rows'])==8
    assert report['resources']['attempted_predict_samples']==528
    assert report['resources']['completed_jacobians']==88
    assert report['resources']['completed_svds']==96
    plans=m.selected_actions(output)
    assert len(plans)==8 and all(len(p['actions'])==10 for p in plans)
    assert all(p['horizon']==2 and not p['physical_ready'] for p in plans)


def test_runner_failure_preserves_all_eight_slots_and_incomplete_work(monkeypatch,tmp_path):
    fake_host(monkeypatch,tmp_path,broken=True)
    output=tmp_path/'execution'
    with pytest.raises(RuntimeError,match='preserve all partials'):m.execute_cuda(tmp_path/'prepared','sha',output)
    r=m.io.decode((output/'report.json').read_bytes())
    assert len(r['rows'])==8 and r['rows'][0]['status']=='incomplete'
    assert sum(row['status']=='blocked' for row in r['rows'])==7
    assert r['resources']['attempted_predict_samples']>0
    assert r['resources']['completed_jacobians']>=1
    with pytest.raises(ValueError):m.selected_actions(output)


def test_real_sealed_inputs_only_bind_goal_context_q_not_old_search():
    inputs=m.source_inputs(m.PROJECT)
    assert len(inputs['packets'])==len(set(inputs['roots']))==4
    for p in inputs['packets']:
        assert set(p)=={'root_sha256','root_context','goal','source_input_sha256',
            'projections','active_metric_ids','input_sha256'}
        assert set(p['projections'])==set(m.ARMS)
        assert p['active_metric_ids'][m.ARMS[0]]!=p['active_metric_ids'][m.ARMS[1]]
        content={k:v for k,v in p.items() if k!='input_sha256'}
        assert m.digest(content)==p['input_sha256']


def test_prepare_without_model_import_and_source_byte_drift_rejected(monkeypatch,tmp_path):
    # Isolated NEW fixture files only, never modifies an actual sealed source.
    source=tmp_path/'fixture.py';source.write_text('fixture = 1\n')
    inputs=dict(roots=['fixture'],packets=[],model_bindings={},bindings={})
    monkeypatch.setattr(m,'source_inputs',lambda _:inputs)
    monkeypatch.setattr(m.old.prior.physical,'source_closure',lambda *a:[source])
    output=tmp_path/'prepare'
    m.prepare(tmp_path,output);sha=m.io.sha((output/'prepare.json').read_bytes())
    m.validate_prepared(output,sha)
    with pytest.raises(ValueError,match='exclusive'):m.prepare(tmp_path,output)
    with pytest.raises(ValueError,match='prepare SHA'):m.validate_prepared(output,'badsha')
    source.write_text('fixture = 2\n')
    with pytest.raises(ValueError,match='source drift'):m.validate_prepared(output,sha)
