"""Pure array and saved-contract fixtures; no neural forward or CUDA launch."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p2_no_composition_sra_20261027_v2 as m
from experiments.spectral_reachability.d2p_composition_witness_v2 import PROVENANCE_KEYS


def bank_inputs():
    j = np.zeros((192,20))
    j[:20] = np.diag(np.arange(20,0,-1))
    goal = np.zeros(192)
    goal[:20] = np.linspace(-2,2,20)
    return j,np.zeros(192),goal,np.eye(192)


def fixture_inputs():
    context = dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=1,
        action_dim=10,latent_history=[[0.],[0.],[0.]],action_history=[[0.]*10,[0.]*10])
    def replay(context,actions):
        for start in range(0,len(actions),5):
            packet = np.asarray(actions[start:start+5]).reshape(-1)
            value = context['latent_history'][-1][0]+float(np.dot(packet,np.arange(1,11)))/55
            context['latent_history'] = context['latent_history'][1:]+[[value]]
            context['action_history'] = context['action_history'][1:]+[packet.tolist()]
        return context
    return dict(bank=m.spectral_bank(*bank_inputs()),
        root=dict(root_id='a'*64,root_rgb_sha256='b'*64,goal_rgb_sha256='c'*64,context=context),
        provenance={k:'d'*64 for k in PROVENANCE_KEYS},
        action_domain=dict(name='raw',units='official PushT live command',low=[-1.,-1.],
            high=[1.,1.],action_block=5,normalization_sha256='e'*64),
        target_metric=lambda c:(c['latent_history'][-1][0]-.5)**2,replay=replay,fixture=True)


def test_bank_exact_scales_modes_radii_and_two_svds(monkeypatch):
    original = np.linalg.svd
    shapes = []
    def spy(matrix,**kwargs):
        shapes.append(matrix.shape)
        return original(matrix,**kwargs)
    monkeypatch.setattr(np.linalg,'svd',spy)
    j,nominal,goal,q = bank_inputs()
    bank = m.spectral_bank(j,nominal,goal,q)
    assert shapes == [(192,20),(192,20)]
    logits = np.asarray(bank['logits'])
    assert logits.shape == (40,20)
    assert np.array_equal(logits[0],np.zeros(20))
    dls = np.asarray(bank['dls']['params'])
    assert np.array_equal(logits[1:4],np.array([.25,.5,1.])[:,None]*dls)
    assert len(dls) == 20 and np.all(dls != 0)
    assert np.max(np.linalg.norm(logits,axis=1)) <= m.CAP*(1+1e-12)
    for index in range(9):
        for radius_index,radius in enumerate(m.RADII):
            start = 4+index*4+radius_index*2
            assert np.allclose(logits[start],-radius*np.asarray(bank['right_vectors'][index]))
            assert np.array_equal(logits[start],-logits[start+1])
    commands = np.asarray(bank['commands'])
    assert np.array_equal(commands,np.tanh(logits).astype(np.float32).reshape(40,2,5,2))
    assert commands.min() >= -1 and commands.max() <= 1
    assert all(v[np.argmax(np.abs(v))] >= 0 for v in np.asarray(bank['right_vectors']))


def test_dls_uses_all_twenty_and_single_joint_ball():
    j,nominal,goal,q = bank_inputs()
    goal[:20] *= 1e4
    bank = m.spectral_bank(j,nominal,goal,q)
    step = np.asarray(bank['dls']['params'])
    assert np.isclose(np.linalg.norm(step),m.CAP)
    assert np.count_nonzero(step) == 20
    # This legal joint-bank direction is NOT legal under the old per-packet cap.
    mode = np.asarray(bank['logits'][6])
    assert np.linalg.norm(mode[:10]) > m.compiler.CAP
    assert np.isclose(np.linalg.norm(mode),m.CAP)


def test_zero_spectrum_keeps_fixed_slots_and_stable_signs():
    args = list(bank_inputs())
    args[0] = np.zeros((192,20))
    first = m.spectral_bank(*args)
    assert first == m.spectral_bank(*args)
    assert len(first['commands']) == 40
    assert np.array_equal(np.asarray(first['logits'])[:4],np.zeros((4,20)))


@pytest.mark.parametrize('defect',['jac_shape','nan','goal_shape','q_shape'])
def test_invalid_arrays_rejected(defect):
    args = list(bank_inputs())
    if defect == 'jac_shape': args[0] = np.zeros((192,10))
    if defect == 'nan': args[0][0,0] = np.nan
    if defect == 'goal_shape': args[2] = np.zeros(20)
    if defect == 'q_shape': args[3] = np.eye(20)
    with pytest.raises(ValueError): m.spectral_bank(*args)


def test_complete_candidate_search_and_single_edge_double_replay():
    args = fixture_inputs()
    before = deepcopy(args['root'])
    journal = []
    result = m.run_bank(**args,journal=journal.append)
    assert args['root'] == before
    assert result['candidate_and_replay_complete']
    assert len(journal) == len(result['attempts']) == 40
    assert len(result['ledger']) == 84
    assert result['planning_predict_samples'] == 80 and result['replay_predict_samples'] == 4
    assert len(result['witness']['edges']) == 1
    assert len(result['witness']['receipt']['actions']) == 10
    best = min(result['attempts'],key=lambda e:e['cost'])
    assert best['candidate_id'] == result['selected_candidate_id']
    assert best['actions'] == result['witness']['receipt']['actions']
    assert not result['prediction_budget_matched'] and not result['fair_total_budget_comparison']
    assert not result['P2_admission']


def test_first_tie_zero_not_free_extra_candidate():
    args = fixture_inputs()
    args['target_metric'] = lambda c:1.
    result = m.run_bank(**args)
    assert result['selected_candidate_id'] == m.ARM+'_00'
    assert len(result['attempts']) == 40


@pytest.mark.parametrize('defect',['first_packet','metric','replay'])
def test_failure_retained_without_winner_refund_or_retry(defect):
    args = fixture_inputs()
    replay = args['replay']
    count = 0
    def failing(context,actions):
        nonlocal count
        count += 1
        if count == (81 if defect == 'replay' else 1) and defect != 'metric':
            raise RuntimeError('deliberate mathematical fixture failure')
        return replay(context,actions)
    args['replay'] = failing
    if defect == 'metric': args['target_metric'] = lambda c:float('nan')
    result = m.run_bank(**args)
    assert not result['candidate_and_replay_complete']
    assert result['selected_candidate_id'] is None and result['witness'] is None
    assert len(result['attempts']) == 40
    if defect == 'first_packet':
        assert len(result['ledger']) == 79
        assert not result['ledger'][0]['completed']
    if defect == 'replay': assert result['replay_predict_samples'] == 1


def test_production_callback_and_cpu_compile_refused():
    args = fixture_inputs()
    args['fixture'] = False
    with pytest.raises(ValueError,match='CUDA'): m.run_bank(**args)
    with pytest.raises(ValueError,match='CUDA'):
        m.compile_cuda(None,SimpleNamespace(type='cpu'),None,None,None,None)


def test_full_cost_arithmetic_and_static_joint_derivative():
    s = m.SCHEDULE
    assert s['total_predict_samples'] == 2+2+40*2+4 == 88
    assert 4*88 == 352
    assert s['processed_context_token_samples'] == 88*3 == 264
    assert 4*s['scalar_output_vjp_directions'] == 768
    assert 4*s['svd_calls'] == 8 and 4*s['jacobian_calls'] == 4
    source = inspect.getsource(m.compile_cuda)
    assert source.count('torch.autograd.functional.jacobian(') == 1
    assert 'vectorize=True' in source
    assert 'state,between = transition(state,between,raw[packet])' in source
    assert '.detach(' not in source[:source.index('jac = torch.autograd')]
    cuda_source = inspect.getsource(m.cuda)
    assert cuda_source.index('load_prepared(output)') < cuda_source.index('require_cuda_allocation')
    assert 'sample_limit=88' in cuda_source


@pytest.fixture
def prepared(tmp_path):
    output = tmp_path/m.TAG
    m.prepare(m.historical.DEFAULT_REFERENCE,output)
    return output


def test_prepare_binds_all_four_exact_roots_and_new_source(prepared,monkeypatch):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    monkeypatch.setattr(base,'require_cuda_allocation',lambda *a:pytest.fail('no CUDA'))
    monkeypatch.setattr(base,'load_compatible_pretrained',lambda *a,**k:pytest.fail('no model'))
    _,p,s = m.load_prepared(prepared)
    assert p['schedule'] == m.SCHEDULE and p['frozen_before_new_v2_outcomes']
    assert p['expected_four_root_totals']['predict_samples'] == 352
    assert [r['root_index'] for r in p['context']['cases']] == [0,1,2,3]
    assert s['new_neural_calls'] == 0 and s['status'] == 'prepared_not_executed'
    assert not (prepared/'report.json').exists() and not (prepared/'started.json').exists()
    assert {str(m.PROJECT/'tests'/f'test_{m.TAG}.py'),str(Path(m.__file__)),
        str(m.PROJECT/'innerdocs'/f'{m.TAG}.md')} <= set(p['source_paths'])
    with pytest.raises(FileExistsError): m.prepare(m.historical.DEFAULT_REFERENCE,prepared)
    assert m.main(['verify','--output',str(prepared)]) is None


@pytest.mark.parametrize('defect',['schedule','context','archive'])
def test_prepare_mutations_refuse(prepared,defect):
    if defect == 'archive':
        seal = m.historical.audit.read_json(prepared/'prepared.json')['source_seals'][0]
        path = Path(seal['output_dir'])/'sources/000.source'
        path.write_text('test archive only')
    else:
        path = prepared/'protocol.json'
        data = m.historical.audit.read_json(path)
        if defect == 'schedule': data['schedule']['total_predict_samples'] = 84
        else: data['context']['cases'][0]['seed'] = 1
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError): m.load_prepared(prepared)


def test_failed_svd_operation_journal_retained(monkeypatch):
    events = []
    def broken(*a,**k): raise RuntimeError('deliberate SVD failure')
    monkeypatch.setattr(np.linalg,'svd',broken)
    with pytest.raises(RuntimeError): m.spectral_bank(*bank_inputs(),journal=events.append)
    assert len(events) == 2 and all(not e['completed'] for e in events)
    assert events[0]['svd_calls_requested'] == 1


def test_existing_guard_metadata_only_compilation_search_replay_accounting():
    """No tensors or neural operations: exercise the real guard with shape metadata."""
    from experiments.spectral_reachability.d2p_adaptive_budget_v2 import Binding,PredictionBudget
    class Encoder:
        def forward(self,value): return value
    class Model:
        action_encoder = Encoder()
        def predict(self,emb,act): return 'metadata only, no tensor computation'
    model = Model()
    replay = SimpleNamespace(_p2_scope=None)
    tensor = SimpleNamespace(shape=(1,3,192),device=SimpleNamespace(type='cuda',index=0))
    result = m.run_bank(**fixture_inputs())
    with m.full20.actual_entries(model,replay) as entries, PredictionBudget(model,
            binding=Binding('a'*64,m.TAG,'b'*64,'c'*64),method=m.ARM,sample_limit=88) as guard:
        with guard.phase('planning',method=m.ARM,horizon_tokens=2):
            for stage in ('nominal','joint_jacobian'):
                replay._p2_scope = dict(arm=m.ARM,phase='compilation',path=[stage])
                for _ in range(2):
                    model.predict(tensor,model.action_encoder.forward(tensor))
        for phase,start,stop in (('planning',0,80),('evaluation',80,84)):
            with guard.phase(phase,method=m.ARM,horizon_tokens=2):
                for dispatch in result['ledger'][start:stop]:
                    replay._p2_scope = dict(arm=m.ARM,phase=dispatch['phase'],path=dispatch['path'])
                    model.predict(tensor,model.action_encoder.forward(tensor))
        actual = guard.receipt()
    assert actual['attempted_predict_samples'] == 88
    assert m.accounting_ok(entries,actual,result)
    assert sum(e['context_tokens'] for e in entries if e['kind']=='predict') == 264
    assert not m.accounting_ok(entries[2:],actual,result)
