"""v2 regressions: algebra/IO fault injection only, never neural or physics."""
from copy import deepcopy
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import p3_goal_patch_consistent_compiler_20261027_v2 as m


def packet():
    return dict(root_sha256='root',goal=[1.,2.]+[0.]*190,
        root_context=dict(layout='states_with_between_state_actions_v1',context_length=3,
            latent_dim=192,action_dim=10,latent_history=np.zeros((3,192)).tolist(),
            action_history=np.zeros((2,10)).tolist()))


class LinearBackend:
    def __init__(self):
        self.calls=0;self.j=np.zeros((192,10));self.j[0,0]=1;self.j[1,1]=2

    def step(self,context,actions):
        self.calls+=1;c=deepcopy(context)
        z=np.asarray(c['latent_history'][-1])+self.j@np.asarray(actions).reshape(10)
        c['latent_history']=c['latent_history'][1:]+[z.tolist()]
        c['action_history']=c['action_history'][1:]+[m.normalized_history(actions).reshape(10).tolist()]
        return c

    def replay(self,context,actions):
        for raw in np.asarray(actions).reshape(-1,5,2):context=self.step(context,raw)
        return context

    def nominal(self,context):return np.asarray(self.step(context,np.zeros((5,2)))['latent_history'][-1])

    def jacobian(self,context,operation):
        operation('jacobian',lambda:self.step(context,np.zeros((5,2))))
        return self.j.copy()


def test_nominal_retained_before_root_ad_failure():
    class Broken(LinearBackend):
        def jacobian(self,context,operation):
            def fail():raise RuntimeError('AD failed')
            return operation('jacobian',fail)
    events=[]
    with pytest.raises(RuntimeError,match='AD failed'):
        m.compile_arm(packet(),np.eye(192),Broken(),events.append)
    assert any(e['kind']=='nominal' and e['phase']=='root' and e['completed'] for e in events)


@pytest.mark.parametrize('svd_failure',[1,2,3])
def test_j_and_prefix_saved_before_later_svd_failure(monkeypatch,svd_failure):
    events=[];original=np.linalg.svd;calls=0
    def fail(*a,**k):
        nonlocal calls
        calls+=1
        if calls==svd_failure:raise RuntimeError('SVD failed')
        return original(*a,**k)
    monkeypatch.setattr(np.linalg,'svd',fail)
    with pytest.raises(RuntimeError,match='SVD failed'):
        m.compile_arm(packet(),np.eye(192),LinearBackend(),events.append)
    assert any(e['kind']=='derivative' and e['phase']=='root' for e in events)
    if svd_failure==3:
        assert any(e['kind']=='prefix' for e in events)
        assert any(e['kind']=='nominal' and e['phase']=='suffix_0' for e in events)
        assert any(e['kind']=='derivative' and e['phase']=='suffix_0' for e in events)


def test_prefix_saved_before_suffix_ad_failure():
    class Broken(LinearBackend):
        def jacobian(self,context,operation):
            if self.calls>2:
                def fail():raise RuntimeError('suffix AD failed')
                return operation('jacobian',fail)
            return super().jacobian(context,operation)
    events=[]
    with pytest.raises(RuntimeError,match='suffix AD failed'):
        m.compile_arm(packet(),np.eye(192),Broken(),events.append)
    assert any(e['kind']=='prefix' and 'context' in e for e in events)
    assert any(e['kind']=='nominal' and e['phase']=='suffix_0' for e in events)


@pytest.mark.parametrize('field,value',[
    ('layout','wrong'),('context_length',2),('latent_dim',191),('action_dim',2)])
def test_layout_tampering_rejected(field,value):
    p=packet();q=np.eye(192);r=m.compile_arm(p,q,LinearBackend(),lambda e:None)
    c=next(c for c in r['candidates'] if c['index']!=r['winner']['index'])
    c['context'][field]=value
    with pytest.raises(ValueError):m.validate_compilation(p,q,r)


def test_nonwinner_normalized_history_tampering_rejected():
    p=packet();q=np.eye(192);r=m.compile_arm(p,q,LinearBackend(),lambda e:None)
    c=next(c for c in r['candidates'] if c['index']!=r['winner']['index'])
    c['context']['action_history'][0][0]+=1e-12
    with pytest.raises(ValueError,match='history'):m.validate_compilation(p,q,r)


def test_complete_arm_has_unchanged_algebra_candidates_and_budget():
    from experiments.spectral_reachability import p3_goal_patch_consistent_compiler_20261027_v1 as v1
    class V1Backend(LinearBackend):
        def step(self,context,actions):
            c=super().step(context,actions)
            c['action_history'][-1]=np.asarray(actions).reshape(10).tolist()
            return c
        def nominal_jacobian(self,context,operation):
            return self.nominal(context),self.jacobian(context,operation)
    p=packet();q=np.eye(192);b=LinearBackend();r=m.compile_arm(p,q,b,lambda e:None)
    old=v1.compile_arm(p,q,V1Backend(),lambda e:None)
    assert b.calls==66 and len(r['candidates'])==10 and len(r['operations'])==23
    assert r['bank']==old['bank']
    for a,b in zip(r['candidates'],old['candidates'],strict=True):
        for key in ('commands','stage_cost','suffix_solve','objective','nonlinear_terminal_cost'):
            assert a[key]==b[key]
    assert r['winner']['index']==old['winner']['index']
    assert m.validate_compilation(p,q,r)==r['winner']
    assert all(m.contract()[k]==v for k,v in v1.contract().items())


def fake_host(monkeypatch,tmp_path,broken=True):
    from experiments import spectral_reachability as package
    checked=SimpleNamespace(digest=lambda p:m.io.sha(__import__('pathlib').Path(p).read_bytes()),
        tensor_state_hash=lambda model:'frozen')
    monkeypatch.setitem(sys.modules,'experiments.spectral_reachability.run_pusht_d2p_budget_guard_cuda',checked)
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
    weight=tmp_path/'weights.pt';weight.write_bytes(b'algebra fixture, not a model')
    packets=[]
    for i in range(4):
        p=packet();p.update(root_sha256=str(i),source_input_sha256='old'+str(i),
            projections={a:np.eye(192).tolist() for a in m.ARMS},
            active_metric_ids={a:a+str(i) for a in m.ARMS},input_sha256='new'+str(i));packets.append(p)
    body=dict(packets=packets,roots=[str(i) for i in range(4)],model_bindings={str(weight):checked.digest(weight)},
        arm_order=[list(m.ARMS if i%2==0 else m.ARMS[::-1]) for i in range(4)])
    monkeypatch.setattr(m,'validate_prepared',lambda *a:(body,'fixture-v2'))
    return checked


@pytest.mark.parametrize('fault',['hash','cleanup','root_write','report_write','source','journal_read'])
def test_primary_failure_and_secondary_errors_preserved(monkeypatch,tmp_path,fault):
    checked=fake_host(monkeypatch,tmp_path)
    if fault=='hash':
        original=checked.digest
        def broken(path):
            if str(path).endswith('.events.jsonl'):raise OSError('secondary hash')
            return original(path)
        monkeypatch.setattr(checked,'digest',broken)
    elif fault=='cleanup':
        original=m.NeuralCap.__exit__
        def broken(self,*exc):
            original(self,*exc);raise RuntimeError('secondary cleanup')
        monkeypatch.setattr(m.NeuralCap,'__exit__',broken)
    elif fault in ('root_write','report_write'):
        original=m.write
        def broken(path,value):
            if (fault=='root_write' and path.name=='root_0.json') or (fault=='report_write' and path.name=='report.json'):
                raise OSError('secondary write')
            return original(path,value)
        monkeypatch.setattr(m,'write',broken)
    elif fault=='source':
        original=m.validate_prepared;calls=0
        def broken(*a):
            nonlocal calls
            calls+=1
            if calls>1:raise ValueError('secondary source drift')
            return original(*a)
        monkeypatch.setattr(m,'validate_prepared',broken)
    else:
        from pathlib import Path
        original=Path.read_text
        def broken(path,*a,**k):
            if str(path).endswith('.events.jsonl'):raise OSError('secondary read')
            return original(path,*a,**k)
        monkeypatch.setattr(Path,'read_text',broken)
    out=tmp_path/'execution'
    with pytest.raises(RuntimeError) as caught:m.execute_cuda(tmp_path/'prepared','sha',out)
    report=caught.value.report
    assert report['failure']['message']=='fixture kernel failure'
    assert report['failure']['traceback']
    assert any('secondary' in f['message'] for f in report['failures'][1:])
    assert len(report['rows'])==8 and sum(r['status']=='blocked' for r in report['rows'])==7
    assert report['resources']['attempted_predict_samples']>0
    assert report['resources']['completed_jacobians']>=1
    assert (out/'report.json').exists() or (out/'failure_report.json').exists()
    with pytest.raises((ValueError,FileNotFoundError)):m.selected_actions(out)


def test_full_runner_and_export_without_neural(monkeypatch,tmp_path):
    fake_host(monkeypatch,tmp_path,broken=False)
    out=tmp_path/'execution';r=m.execute_cuda(tmp_path/'prepared','sha',out)
    assert r['resources']['attempted_predict_samples']==528
    assert r['resources']['completed_jacobians']==88 and r['resources']['completed_svds']==96
    assert len(m.selected_actions(out))==8


def test_no_runtime_imports():
    import subprocess
    script='import sys; from experiments.spectral_reachability import '+m.NAME+'; assert not any(n in sys.modules for n in ("torch","pymunk","gymnasium"))'
    subprocess.run([sys.executable,'-c',script],check=True)


def test_all_report_writes_fail_but_exception_retains_report(monkeypatch,tmp_path):
    fake_host(monkeypatch,tmp_path)
    original=m.write
    def broken(path,value):
        if path.name in ('report.json','failure_report.json'):raise OSError('disk full')
        return original(path,value)
    monkeypatch.setattr(m,'write',broken)
    with pytest.raises(m.IncompleteCompilation) as caught:m.execute_cuda(tmp_path/'prepared','sha',tmp_path/'execution')
    r=caught.value.report
    assert r['failure']['message']=='fixture kernel failure'
    assert len([e for e in r['failures'] if e['message']=='disk full'])==2
    assert any(e['kind']=='derivative' for e in r['roots'][0]['recovery_events'])


def test_journal_completion_write_does_not_mask_kernel_error(monkeypatch,tmp_path):
    fake_host(monkeypatch,tmp_path)
    original=m.os.fsync;calls=0
    def broken(fd):
        nonlocal calls
        calls+=1
        # Fail only completion records after an algebra kernel error exists.
        if sys.exception() is not None and 'fixture kernel' in str(sys.exception()):
            raise OSError('secondary fsync')
        return original(fd)
    monkeypatch.setattr(m.os,'fsync',broken)
    with pytest.raises(m.IncompleteCompilation) as caught:m.execute_cuda(tmp_path/'prepared','sha',tmp_path/'execution')
    r=caught.value.report
    assert r['failure']['message']=='fixture kernel failure'
    assert any(e['message']=='secondary fsync' for e in r['failures'])
    assert r['resources']['attempted_predict_samples']>0
    assert r['resources']['durable_journal_complete'] is False


def test_normalization_is_float32_time_major_not_div100():
    raw=np.arange(20,dtype=np.float32).reshape(10,2)/40
    expected=(raw.reshape(2,10)-np.tile(np.asarray([-.0087,.0068],np.float32),5))/np.tile(np.asarray([.2019,.2002],np.float32),5)
    assert np.array_equal(m.normalized_history(raw),expected)
    assert m.normalized_history(raw).dtype==np.float32
    assert not np.array_equal(m.normalized_history(raw/100),expected)
    with pytest.raises(ValueError):m.normalized_history(np.ones((10,2))*2)


def test_prepare_source_and_seal_coverage_drift_rejected(monkeypatch,tmp_path):
    source=tmp_path/'fixture.py';source.write_text('fixture = 1\n')
    inputs=dict(roots=['fixture'],packets=[],model_bindings={},bindings={})
    monkeypatch.setattr(m,'source_inputs',lambda _:inputs)
    monkeypatch.setattr(m,'source_files',lambda _:[source])
    out=tmp_path/'prepare';m.prepare(tmp_path,out)
    sha=m.io.sha((out/'prepare.json').read_bytes());m.validate_prepared(out,sha)
    with pytest.raises(ValueError,match='exclusive'):m.prepare(tmp_path,out)
    with pytest.raises(ValueError,match='prepare SHA'):m.validate_prepared(out,'wrong')
    receipt=m.io.decode((out/'prepare.json').read_bytes());receipt['source_seals']=[]
    (out/'prepare.json').write_text(m.io.canonical(receipt))
    with pytest.raises(ValueError,match='coverage'):m.validate_prepared(out,m.io.sha((out/'prepare.json').read_bytes()))
    # Source-byte checking must reject independently of missing seal coverage.
    source.write_text('fixture = 2\n')
    with pytest.raises(ValueError,match='source drift'):m.validate_prepared(out,m.io.sha((out/'prepare.json').read_bytes()))


def test_real_source_inputs_are_same_as_v1_and_normalizer_is_source_bound():
    from experiments.spectral_reachability import p3_goal_patch_consistent_compiler_20261027_v1 as v1
    assert m.source_inputs(m.PROJECT)==v1.source_inputs(v1.PROJECT)
    m.validate_normalizer_source(m.PROJECT)


def test_q_changes_bank_and_each_suffix_uses_own_context():
    class Conditional(LinearBackend):
        def matrix(self,c):return self.j*(1+abs(c['latent_history'][-1][0]))
        def step(self,context,actions):
            self.calls+=1;c=deepcopy(context)
            z=np.asarray(c['latent_history'][-1])+self.matrix(c)@np.asarray(actions).reshape(10)
            c['latent_history']=c['latent_history'][1:]+[z.tolist()]
            c['action_history']=c['action_history'][1:]+[m.normalized_history(actions).reshape(10).tolist()]
            return c
        def jacobian(self,context,operation):
            operation('jacobian',lambda:self.step(context,np.zeros((5,2))))
            return self.matrix(context)
    p=packet();q0=np.diag([1.,0.]+[0.]*190);q1=np.diag([0.,1.]+[0.]*190)
    b=Conditional();r=m.compile_arm(p,q0,b,lambda e:None)
    other=m.compile_arm(p,q1,Conditional(),lambda e:None)
    assert r['bank']['params']!=other['bank']['params']
    assert r['candidates'][0]['suffix_params']!=other['candidates'][0]['suffix_params']
    assert b.calls==66
    for c in r['candidates']:
        np.testing.assert_array_equal(c['jacobian'],b.j*(1+abs(c['predicted_prefix'][0])))
    assert m.validate_compilation(p,q0,r)==r['winner']


@pytest.mark.parametrize('kind,shape',[('predict',(1,3,192)),('action_encoder',(1,3,10))])
def test_caps_nonrefundable_and_no_extra_dispatch(kind,shape):
    cap=m.NeuralCap(None,lambda e:None);x=SimpleNamespace(device=SimpleNamespace(type='cuda'),shape=shape)
    def fail(x):raise RuntimeError('forward failed')
    cap.role=m.ARMS[0]
    with pytest.raises(RuntimeError):cap.dispatch(kind,fail,(x,),{})
    for _ in range(65):cap.dispatch(kind,lambda x:x,(x,),{})
    with pytest.raises(ValueError,match='cap'):cap.dispatch(kind,lambda x:x,(x,),{})
    assert cap.receipt()['attempted'][kind]==66 and cap.receipt()['completed'][kind]==65
    cap.role=m.ARMS[1]
    for _ in range(66):cap.dispatch(kind,lambda x:x,(x,),{})
    assert cap.receipt()['attempted'][kind]==132
    with pytest.raises(ValueError,match='cap'):cap.dispatch(kind,lambda x:x,(x,),{})


def test_zero_j_duplicate_actions_preserved_and_stable_tie():
    b=LinearBackend();b.j[:]=0
    r=m.compile_arm(packet(),np.eye(192),b,lambda e:None)
    assert b.calls==66 and len(r['candidates'])==10
    assert r['candidates'][0]['commands']==r['candidates'][1]['commands']
    assert r['winner']['index']==0


def test_visual_encoder_forbidden_and_all_hooks_restored():
    predict=lambda x:x;action=lambda x:x;encoder=lambda x:x
    model=SimpleNamespace(predict=predict,action_encoder=SimpleNamespace(forward=action),encoder=SimpleNamespace(forward=encoder))
    events=[]
    with pytest.raises(ValueError,match='visual encoder'):
        with m.NeuralCap(model,events.append):model.encoder.forward(None)
    assert model.predict is predict and model.action_encoder.forward is action and model.encoder.forward is encoder
    assert events==[dict(kind='forbidden_visual_encoder')]


def test_model_hash_secondary_failure_is_not_lost(monkeypatch,tmp_path):
    checked=fake_host(monkeypatch,tmp_path);calls=0
    def broken(model):
        nonlocal calls
        calls+=1
        if calls>1:raise RuntimeError('secondary model hash')
        return 'frozen'
    monkeypatch.setattr(checked,'tensor_state_hash',broken)
    with pytest.raises(m.IncompleteCompilation) as caught:m.execute_cuda(tmp_path/'prepared','sha',tmp_path/'execution')
    r=caught.value.report
    assert r['failure']['message']=='fixture kernel failure'
    assert any(e['message']=='secondary model hash' for e in r['failures'])


def test_winner_history_tamper_even_with_repeated_context_copies_is_rejected():
    p=packet();q=np.eye(192);r=m.compile_arm(p,q,LinearBackend(),lambda e:None)
    c=r['candidates'][r['winner']['index']];c['context']['action_history'][0][0]+=1e-12
    r['winner']=deepcopy(c);r['whole_replay_context']=deepcopy(c['context']);r['segmented_replay_context']=deepcopy(c['context'])
    with pytest.raises(ValueError,match='history'):m.validate_compilation(p,q,r)


def test_restore_attempts_all_hooks_and_retains_multiple_errors():
    class Forward:
        def __init__(self,label):self.label=label;self.original=lambda x:x;self.value=self.original
        @property
        def forward(self):return self.value
        @forward.setter
        def forward(self,value):
            if value is self.original:raise RuntimeError('restore '+self.label)
            self.value=value
    predict=lambda x:x
    model=SimpleNamespace(predict=predict,action_encoder=Forward('action'),encoder=Forward('visual'))
    log=m.FailureLog()
    try:
        with m.NeuralCap(model,lambda e:None):raise RuntimeError('primary kernel')
    except RuntimeError as exc:log.capture(exc,'fixture')
    assert model.predict is predict
    assert [e['message'] for e in log.entries]==['primary kernel','restore visual','restore action']


def test_partial_hook_install_restores_already_installed_hooks():
    class RejectForward:
        def __init__(self):self.original=lambda x:x
        @property
        def forward(self):return self.original
        @forward.setter
        def forward(self,value):
            if value is not self.original:raise RuntimeError('install rejected')
    predict=lambda x:x;action=lambda x:x
    model=SimpleNamespace(predict=predict,action_encoder=SimpleNamespace(forward=action),encoder=RejectForward())
    with pytest.raises(RuntimeError,match='install rejected'):
        with m.NeuralCap(model,lambda e:None):pytest.fail('must not enter body')
    assert model.predict is predict and model.action_encoder.forward is action
