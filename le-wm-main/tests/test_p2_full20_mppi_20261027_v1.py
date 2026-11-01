"""Mathematical callbacks/metadata only: no neural forward, GPU or physics."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p2_full20_mppi_20261027_v1 as m
from experiments.spectral_reachability.d2p_composition_witness_v2 import PROVENANCE_KEYS


def transition(context,actions):
    for start in range(0,len(actions),5):
        packet = np.asarray(actions[start:start+5]).reshape(-1).tolist()
        vector = np.asarray(context['latent_history'][-1])+sum((i+1)*v for i,v in enumerate(packet))/55
        context['latent_history'] = context['latent_history'][1:]+[vector.tolist()]
        context['action_history'] = context['action_history'][1:]+[packet]
    return context


def inputs():
    context = dict(layout='states_with_between_state_actions_v1',context_length=3,
        latent_dim=1,action_dim=10,latent_history=[[0.],[0.],[0.]],action_history=[[0.]*10,[0.]*10])
    return dict(seed=42,root=dict(root_id='a'*64,root_rgb_sha256='b'*64,
        goal_rgb_sha256='c'*64,context=context),provenance={k:'d'*64 for k in PROVENANCE_KEYS},
        action_domain=dict(name='raw',units='official PushT live command',low=[-1.,-1.],
            high=[1.,1.],action_block=5,normalization_sha256='e'*64),
        target_metric=lambda c:float((c['latent_history'][-1][0]-.5)**2),replay=transition)


def test_exact_kernel_reuse_and_all_nominal_costs_counted(monkeypatch):
    original = m.kernels.mppi
    calls = []
    def spy(cost,**kw):
        calls.append(deepcopy(kw))
        return original(cost,**kw)
    monkeypatch.setattr(m.kernels,'mppi',spy)
    monkeypatch.setattr(m.full20,'_arm',lambda **kw:pytest.fail('must not monkeypatch/use sealed CEM arm'))
    args = inputs()
    before = deepcopy(args['root'])
    events = []
    result = m.run_fixture_arm(**args,journal=events.append)
    assert args['root'] == before
    assert len(calls) == 1
    assert calls[0]['samples'] == 12 and calls[0]['iterations'] == 3
    assert calls[0]['temperature'] == 1. and calls[0]['noise_scale'] == .5
    assert result['prediction_budget_matched'] and result['fixture_only']
    assert len(events) == len(result['attempts']) == 40
    assert [e['stage'] for e in events].count('initial') == 1
    assert [e['stage'] for e in events].count('perturbation') == 36
    assert [e['stage'] for e in events].count('updated_nominal') == 3
    assert result['kernel_receipt']['cost_evaluations'] == 40
    assert result['planning_predict_samples'] == 80
    assert result['replay_predict_samples'] == 4
    assert len(result['ledger']) == 84
    assert result['witness']['independent_full_replay_exact']
    assert not result['official_MPPI'] and not result['P2_admission']
    assert not result['fair_total_budget_comparison']


def test_deterministic_and_same_as_direct_existing_kernel():
    args = inputs()
    a,b = m.run_fixture_arm(**args),m.run_fixture_arm(**args)
    assert a['attempts'] == b['attempts']
    def cost(batch):
        return np.array([args['target_metric'](transition(deepcopy(args['root']['context']),
            m.full20.raw_actions(x))) for x in batch])
    chosen, receipt = m.kernels.mppi(cost,initial=np.zeros((2,5,2)),lower=-1.,upper=1.,
        samples=12,iterations=3,temperature=1.,noise_scale=.5,seed=42)
    assert a['kernel_receipt'] == receipt
    assert m.full20.raw_actions(chosen) == a['witness']['receipt']['actions']
    for event in a['attempts']:
        assert np.asarray(event['commands']).shape == (2,5,2)
        assert np.asarray(event['actions']).shape == (10,2)
        assert np.array_equal(np.asarray(event['commands']).reshape(10,2),event['actions'])
    proposals = np.asarray([e['actions'] for e in a['attempts'] if e['stage'] == 'perturbation']).reshape(36,20)
    assert np.all(proposals.std(axis=0) > 0)
    assert not np.array_equal(proposals[0].reshape(10,2)[:5],np.repeat(proposals[0,:2][None],5,axis=0))


def test_first_tie_initial_zero_included_not_extra():
    args = inputs()
    args['target_metric'] = lambda c:1.
    result = m.run_fixture_arm(**args)
    assert result['selected_candidate_id'] == m.ARM+'_00'
    assert result['attempts'][0]['commands'] == np.zeros((2,5,2)).tolist()
    assert result['witness']['receipt']['actions'] == np.zeros((10,2)).tolist()
    assert result['kernel_receipt']['cost_evaluations'] == 40


@pytest.mark.parametrize('defect',['first_predict','mid_batch','metric','replay'])
def test_failure_has_no_padding_refund_or_deployable_winner(defect):
    args = inputs()
    count = 0
    def replay(context,actions):
        nonlocal count
        count += 1
        if count == dict(first_predict=1,mid_batch=5,metric=-1,replay=81)[defect]:
            raise RuntimeError('deliberate metadata fixture failure')
        return transition(context,actions)
    args['replay'] = replay
    if defect == 'metric': args['target_metric'] = lambda c:float('nan')
    events = []
    result = m.run_fixture_arm(**args,journal=events.append)
    assert not result['prediction_budget_matched']
    assert result['witness'] is None and result['selected_candidate_id'] is None
    assert events == result['attempts']
    assert len(result['ledger']) < 84
    if defect == 'replay':
        assert result['replay_error']
        assert result['planning_predict_samples'] == 80 and result['replay_predict_samples'] == 1
    else:
        assert result['kernel_error'] and not events[-1]['completed']
        assert result['kernel_receipt']['completed'] is False
        assert result['replay_predict_samples'] == 0


@pytest.mark.parametrize('defect',['seed','domain','context','provenance','production_callback'])
def test_metadata_rejected_before_dispatch(defect):
    args = inputs()
    args['replay'] = lambda c,a:pytest.fail('must not dispatch')
    if defect == 'seed': args['seed'] = True
    if defect == 'domain': args['action_domain']['action_block'] = 1
    if defect == 'context': args['root']['context']['action_history'] = []
    if defect == 'provenance': args['provenance'] = {}
    with pytest.raises(ValueError):
        m._arm(**args,fixture=False) if defect == 'production_callback' else m.run_fixture_arm(**args)


def test_literal_80_full_candidates_cannot_claim_matched_budget(tmp_path):
    assert 1+m.SCHEDULE['iterations']*(m.SCHEDULE['samples']+1) == 40
    assert m.SCHEDULE['candidates']*m.SCHEDULE['horizon_tokens'] == 80
    assert 80*m.SCHEDULE['horizon_tokens']+4 == 164
    with pytest.raises(ValueError,match='160\\+4'):
        m.cuda(tmp_path/'no_output')
    assert not (tmp_path/'no_output').exists()


def test_real_prepare_and_load_model_free(tmp_path,monkeypatch):
    if not (m.DEFAULT_REFERENCE/'report.json').is_file(): pytest.skip('local calibration unavailable')
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    monkeypatch.setattr(base,'require_cuda_allocation',lambda *a,**kw:pytest.fail('GPU requested'))
    monkeypatch.setattr(base,'load_compatible_pretrained',lambda *a,**kw:pytest.fail('model requested'))
    output = tmp_path/m.TAG
    m.prepare(m.DEFAULT_REFERENCE,output)
    _,protocol,prepared,source = m.load_prepared(output)
    assert len(source['rows']) == len(protocol['prepared']) == 4
    assert protocol['schedule'] == m.SCHEDULE
    assert protocol['main_thread_budget_review_required']
    assert protocol['literal_80_complete_candidates_supported'] is False
    assert prepared['new_neural_calls'] == prepared['new_physical_steps'] == 0
    assert not (output/'started.json').exists() and not (output/'report.json').exists()
    names = set()
    for seal in prepared['source_seals']:
        manifest = json.loads((Path(seal['output_dir'])/'manifest.json').read_text())
        names.update(Path(entry['source']).name for entry in manifest['files'])
    assert {m.TAG+'.py','test_'+m.TAG+'.py',m.TAG+'.md','d2p_sampling_baselines.py',
        'p2_full20_sampling_20261027_v2.py','pusht_composition_cuda_replay_v2.py',
        'd2p_adaptive_budget_v2.py'} <= names
    with pytest.raises(FileExistsError): m.prepare(m.DEFAULT_REFERENCE,output)
    # Mutate only this test's fresh manifest, never source files or old artifacts.
    changed = deepcopy(protocol)
    changed['schedule']['temperature'] = 2.
    (output/'protocol.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='protocol differs'): m.load_prepared(output)


def test_cuda_entry_static_binding_and_cpu_refusal():
    source = inspect.getsource(m.cuda)
    assert source.index('load_prepared(output)') < source.index('require_cuda_allocation')
    assert source.index("output/'started.json'") < source.index('require_cuda_allocation')
    assert 'full20.actual_entries(model,replay)' in source
    assert 'sample_limit=84' in source and 'full20.entries_match' in source
    with pytest.raises(ValueError,match='CUDA'):
        m.full20.RootAnchoredCudaReplay(object(),SimpleNamespace(type='cpu'))


def test_original_instrumentation_counts_real_entry_metadata_and_restores():
    class Model:
        def predict(self, x): return 'prediction'
        class Encoder:
            def forward(self,x): return 'action'
        action_encoder = Encoder()
    model = Model()
    replay = SimpleNamespace(_p2_scope=dict(arm=m.ARM,phase='planning',path=[0]))
    tensor = SimpleNamespace(shape=(1,3,192))
    before = model.predict
    with m.full20.actual_entries(model,replay) as entries:
        assert model.action_encoder.forward(tensor) == 'action'
        assert model.predict(tensor) == 'prediction'
    assert model.predict == before
    assert [e['kind'] for e in entries] == ['action_encoder','predict']
    assert all(e['completed'] and e['samples'] == 1 and e['context_tokens'] == 3 for e in entries)
    assert all(e['scope'] == replay._p2_scope for e in entries)


def test_failed_actual_entry_still_charged_and_wrappers_restored():
    class Encoder:
        def forward(self,x): return x
    class Model:
        action_encoder = Encoder()
        def predict(self,x): raise RuntimeError('deliberate metadata failure')
    model = Model()
    original = model.predict
    replay = SimpleNamespace(_p2_scope=dict(arm=m.ARM,phase='planning',path=[1]))
    with pytest.raises(RuntimeError):
        with m.full20.actual_entries(model,replay) as entries:
            model.predict(SimpleNamespace(shape=(1,3,192)))
    assert model.predict == original and len(entries) == 1
    assert not entries[0]['completed'] and entries[0]['samples'] == 1
    assert entries[0]['scope'] == replay._p2_scope
