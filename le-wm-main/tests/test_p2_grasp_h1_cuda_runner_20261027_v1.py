"""Offline algebra/receipt tests only; never instantiate a neural model."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

MOD = 'experiments.spectral_reachability.p2_grasp_h1_cuda_runner_20261027_v1'


def m():
    return importlib.import_module(MOD)


def test_import_is_torch_and_physics_free():
    subprocess.run([sys.executable, '-c', f'import {MOD}; import sys; assert "torch" not in sys.modules; assert "gymnasium" not in sys.modules'], check=True)


def test_fixed_budget_is_transition_not_compute_equivalence():
    c = m().contract()
    assert c['transition_samples_per_arm'] == 2703
    assert c['backward_per_arm'] == 900
    assert c['total_h1_transitions'] == 21624
    assert c['real_h3_diagnostic_transitions'] == 24
    assert c['full_pipeline_equal_compute'] is False
    assert c['initial_history'] == 'H1_last_real_full_latent'
    assert c['action_adam_steps'] == 900


@pytest.mark.parametrize('method,phase,n', [('grasp_h1','soft',2250), ('grasp_h1','sync',450), ('grasp_h1','verification',3), ('action_adam_h1','action_adam',2700)])
def test_phase_schedule(method, phase, n):
    assert m().phases(method).count(phase) == n


def test_raw_export_preserves_unbounded_actions_and_no_extra_div100():
    a = [[0.] * 10 for _ in range(3)]
    out = m().export_actions(a)
    assert len(out['commands']) == 15
    assert out['commands'][0] == pytest.approx(m().MEAN)
    a[1][0] = 10.
    out = m().export_actions(a)
    assert out['normalized_actions'] == a
    assert out['commands'][5][0] == pytest.approx(10*m().STD[0]+m().MEAN[0])
    assert out['physical_box_supported'] is False
    assert out['clipped'] is False


@pytest.mark.parametrize('bad', [[], [[0.]*10]*2, [[0.]*9]*3, [[float('nan')]*10]*3, [[True]*10]*3])
def test_raw_export_invalid(bad):
    with pytest.raises(ValueError):
        m().export_actions(bad)


def fixture_packet():
    return dict(root_sha256='a'*64, context=[[0.]*192]*3, history_actions=[[0.]*10]*2,
                goal=[1.]*192, history_rgb_sha256='b'*64, goal_rgb_sha256='c'*64)


def test_source_packet_full_latent_independent_goal_and_no_projection():
    mod=m(); root=Path(__file__).resolve().parents[1]
    c=json.loads((root/mod.COMPILATION).read_text())
    g=json.loads((root/mod.GOALS).read_text())
    f=json.loads((root/mod.FRAMES).read_text())
    ps=mod.packets(c,g,f)
    assert len(ps)==4
    assert ps[0]['context']==c['rows'][0]['compiled']['input_context']['emb'][0]
    assert ps[0]['goal']==g['rows'][0]['variants'][0]['goal_vector']
    assert len(ps[0]['goal'])==192
    assert 'projection' not in ps[0]
    g['rows'][0]['variants'][0]['goal_vector'][0] += 1
    with pytest.raises(ValueError, match='goal'):
        mod.packets(c,g,f)


@pytest.mark.parametrize('field', ['goal','context','history_actions','history_rgb_sha256'])
def test_packet_contract_rejects_bad_shapes(field):
    p=fixture_packet();p[field]=[]
    with pytest.raises(ValueError):
        m().validate_packet(p)


class Algebra:
    """No Module, weights, encoder, or learned parameters."""
    def action_encoder(self, a):
        return a
    def predict(self, s, a):
        return .9*s + a.mean(-1,keepdim=True)


@pytest.mark.parametrize('method', ['grasp_h1','action_adam_h1'])
def test_analytic_both_arms_keep_all_actions_and_independent_replay(method, tmp_path):
    import torch
    mod=m();cfg=mod.core.Config(steps=2,sync_interval=2,sync_steps=1,state_noise=0)
    events=[]
    result=mod.run_arm(Algebra(),fixture_packet(),method,events.append,analytic_fixture=True,cfg=cfg)
    assert result['status']=='analytic_fixture_completed'
    assert len(result['plan']['commands'])==15
    assert len(result['verification'])==4
    assert len(result['real_h3_diagnostic']['states'])==4
    assert result['real_h3_diagnostic']['included_in_h1_comparison'] is False
    attempted=[e for e in events if e['kind']=='transition_requested']
    assert len(attempted)==15
    assert all(len(e['normalized_action'][0])==10 for e in attempted)
    assert result['receipt']['backward_completed']==3
    assert result['transition_counts']['verification']['completed']==3
    assert result['transition_counts']['real_h3_diagnostic']['completed']==3


def test_abort_keeps_attempt_and_never_invents_final_plan():
    mod=m();events=[]
    class Broken(Algebra):
        def predict(self,s,a):
            raise RuntimeError('analytic injected failure')
    out=mod.run_arm(Broken(),fixture_packet(),'grasp_h1',events.append,
        analytic_fixture=True,cfg=mod.core.Config(steps=1,sync_interval=0))
    assert out['status']=='failed' and out['plan'] is None
    assert out['transition_counts']['soft']==dict(requested=1,completed=0,failed=1)
    assert out['receipt']['backward_completed']==0
    assert any(e['kind']=='transition_failed' for e in events)


def test_production_rejects_cpu_before_any_model_call():
    with pytest.raises(ValueError,match='CUDA'):
        m().run_arm(Algebra(),fixture_packet(),'grasp_h1',lambda e:None)


def test_durable_journal_exclusive_and_nonfinite_is_not_invalid_json(tmp_path):
    p=tmp_path/'events.jsonl'
    with m().Journal(p) as sink:
        sink(dict(kind='test',value=float('nan')))
    assert json.loads(p.read_text())['value']=='nonfinite:nan'
    with pytest.raises(FileExistsError):
        m().Journal(p)


def test_census_keeps_all_eight_unstarted_failures():
    rows=m().slots(['a','b','c','d'])
    assert len(rows)==8 and all(r['status']=='not_started' for r in rows)
    assert [(r['root_sha256'],r['method']) for r in rows[:4]]==[('a','grasp_h1'),('a','action_adam_h1'),('b','action_adam_h1'),('b','grasp_h1')]


def test_prepare_validation_rejects_bad_sha_before_loading(tmp_path):
    (tmp_path/'prepare.json').write_text('{}')
    with pytest.raises(ValueError,match='SHA'):
        m().validate_prepared(tmp_path,'0'*64)


def test_real_prepare_is_offline_exclusive_and_detects_protocol_tampering(tmp_path):
    mod=m();root=Path(__file__).resolve().parents[1];output=tmp_path/'prepared'
    subprocess.run([sys.executable,'-m',MOD,'--prepare','--root',str(root),'--output',str(output)],check=True,capture_output=True,text=True)
    r=json.loads((output/'prepare.json').read_text())
    assert r['torch_imported'] is False
    assert r['new_predictions']==r['new_physical_steps']==0
    assert r['model_loaded'] is False
    p,identity=mod.validate_prepared(output,mod.digest(output/'prepare.json'))
    assert len(p['packets'])==4 and identity==r['protocol_id']
    with pytest.raises(ValueError,match='exclusive'):
        mod.prepare(root,output)
    path=output/'protocol.json';path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='protocol SHA'):
        mod.validate_prepared(output,mod.digest(output/'prepare.json'))


def test_real_h3_failure_retains_h1_final_plan_and_verification():
    mod=m();events=[]
    class FailDiagnostic(Algebra):
        def __init__(self): self.n=0
        def predict(self,s,a):
            self.n+=1
            if self.n==7: raise RuntimeError('diagnostic failed')
            return super().predict(s,a)
    out=mod.run_arm(FailDiagnostic(),fixture_packet(),'grasp_h1',events.append,
        analytic_fixture=True,cfg=mod.core.Config(steps=1,sync_interval=0))
    assert out['status']=='failed'
    assert out['plan'] is not None and len(out['verification'])==4
    assert out['real_h3_diagnostic'] is None
    assert out['transition_counts']['real_h3_diagnostic']==dict(requested=1,completed=0,failed=1)


def test_module_hooks_count_failed_request_not_fake_completion():
    import torch
    mod=m()
    class HookTarget:
        def __init__(self): self.pre=[];self.post=[]
        def register_forward_pre_hook(self,f): self.pre.append(f);return Handle(self.pre,f)
        def register_forward_hook(self,f): self.post.append(f);return Handle(self.post,f)
        def call(self,x,fail=False):
            for f in self.pre: f(self,(x,))
            if fail: raise RuntimeError('test failure')
            for f in self.post: f(self,(x,),x)
    class Handle:
        def __init__(self,items,f): self.items=items;self.f=f
        def remove(self): self.items.remove(self.f)
    class Model: pass
    model=Model()
    for name in ('predictor','action_encoder','encoder'): setattr(model,name,HookTarget())
    work=mod.ModuleWork(model,lambda e:None)
    model.predictor.call(torch.zeros(1,3,192))
    with pytest.raises(RuntimeError): model.predictor.call(torch.zeros(1,1,192),True)
    assert work.counts['predictor']==dict(requested_calls=2,completed_calls=1,requested_samples=2,completed_samples=1,requested_context_tokens=4,completed_context_tokens=3)
    work.close()
    assert model.predictor.pre==model.predictor.post==[]


def test_preflight_failure_has_all_slots_and_durable_report(tmp_path,monkeypatch):
    mod=m();p=dict(roots=['a','b','c','d'],model_bindings={'not-a-checkpoint':'0'*64})
    monkeypatch.setattr(mod,'validate_prepared',lambda *a:(p,'identity'))
    out=mod.execute('unused','0'*64,tmp_path/'failed')
    assert out['status']=='failed'
    assert len(out['rows'])==8 and all(x['status']=='not_started' for x in out['rows'])
    assert out['failure'] is not None
    assert (tmp_path/'failed/report.json').is_file()


def test_loaded_model_override_rejected_without_neural_execution():
    mod=m()
    class Plain:
        def predict(self): return None
        def encode(self): return None
        def __init__(self):
            self.predictor=object();self.action_encoder=object();self.pred_proj=object()
    class Other(Plain): pass
    with pytest.raises(ValueError,match='class'):
        mod.reviewed_loaded_type(Other(),Plain)
    p=Plain();p.predict=lambda:None
    with pytest.raises(ValueError,match='overridden'):
        mod.reviewed_loaded_type(p,Plain)


def test_only_complete_same_root_h1_rows_enter_pair_difference():
    mod=m();rows=mod.slots(['a','b','c','d'])
    for row in rows:
        row.update(status='encoded_h1_completed_not_physical',h1_terminal_mse=2. if row['method']=='grasp_h1' else 3.)
    rows[3]['status']='failed'
    out=mod.paired_h1(rows[::-1],['a','b','c','d'])
    assert out['complete_pairs']==3 and out['incomplete_pairs']==1
    assert len(out['rows'])==4 and out['rows'][1]['difference_grasp_minus_action_adam'] is None
    assert out['mean_difference_grasp_minus_action_adam']==-1.
    assert out['metric']=='H1 full-latent terminal MSE only'
