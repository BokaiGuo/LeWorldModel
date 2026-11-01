"""Non-neural fixtures; never claims about a checkpoint or physical efficacy."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p2_matched_composition_20261027 as m
from experiments.spectral_reachability.d2p_composition_witness_v2 import PROVENANCE_KEYS, compose


def inputs():
    context = dict(layout='states_with_between_state_actions_v1', context_length=3,
                   latent_dim=1, action_dim=10, latent_history=[[0.], [0.], [0.]],
                   action_history=[[0.] * 10, [0.] * 10])
    return dict(primitive_bank=[np.full((5, 2), v) for v in (-.9, -.3, .3, .9)],
                root=dict(root_id='fixture_not_outcome', root_rgb_sha256='a'*64,
                          goal_rgb_sha256='b'*64, context=context),
                provenance={k: 'c'*64 for k in PROVENANCE_KEYS},
                action_domain=dict(name='raw', units='fixture', low=[-1., -1.], high=[1., 1.],
                                   action_block=5, normalization_sha256='d'*64),
                target_metric=lambda c: (c['latent_history'][-1][0] - .8)**2,
                replay=transition, beam_width=1, flat_seed=23)


def transition(context, actions):
    for start in range(0, len(actions), 5):
        packet = np.asarray(actions[start:start+5]).reshape(-1).tolist()
        value = context['latent_history'][-1][0] + packet[0]
        context['latent_history'] = context['latent_history'][1:] + [[value]]
        context['action_history'] = context['action_history'][1:] + [packet]
    return context


def test_equal_prediction_fixed_length_and_independent_witness():
    args = inputs()
    root = deepcopy(args['root'])
    report = m.run_fixture_pair(**args)
    assert args['root'] == root
    assert report['prediction_budget_matched']
    assert report['fixture_only']
    assert not report['fair_total_budget_comparison']
    assert not report['P1_admission'] and not report['P2_admission']
    for row in report['rows']:
        assert row['planning_predict_samples'] == 8
        assert row['replay_predict_samples'] == 4
        assert len(row['selected_path']) == 2
        witness = row['witness']
        receipt = witness['receipt']
        assert len(receipt['actions']) == 10
        assert compose(root=args['root'], provenance=args['provenance'],
                       action_domain=args['action_domain'], edges=witness['edges'],
                       replay=transition, max_edges=len(witness['edges']),
                       max_actions=10) == receipt
        assert len(witness['edges']) == (2 if row['arm'] == 'composition' else 1)


def test_default_schedule_and_deterministic_flat_paths():
    assert m.schedule()['planning_predict_samples'] == 80
    assert m.schedule()['flat_programs'] == 40
    a, b = m.run_fixture_pair(**inputs()), m.run_fixture_pair(**inputs())
    assert a['declaration_sha256'] == b['declaration_sha256']
    assert a['declaration']['flat_paths'] == b['declaration']['flat_paths']
    assert len(set(a['declaration']['flat_paths'])) == 4


def test_predict_failure_stays_in_ledger_and_invalidates_matching():
    args = inputs()
    def fail(context, actions):
        if actions[0][0] < -.5:
            raise RuntimeError('fixture predict failure')
        return transition(context, actions)
    args['replay'] = fail
    report = m.run_fixture_pair(**args)
    assert not report['prediction_budget_matched']
    assert report['status'] == 'incomplete_or_failed_pair'
    assert report['rows'][0]['failed_predict_attempts'] > 0
    assert report['rows'][0]['failed_search_attempts'] > 0
    assert any('error_type' in e for e in report['rows'][0]['ledger'])


def test_metric_failure_does_not_refund_predict_work():
    args = inputs()
    args['target_metric'] = lambda c: float('nan')
    report = m.run_fixture_pair(**args)
    for row in report['rows']:
        assert row['failed_search_attempts'] == row['attempted_expansions']
        assert row['planning_predict_samples'] > 0
        assert row['witness'] is None
    assert not report['prediction_budget_matched']


def test_replay_failure_retained_without_winner_substitution():
    args = inputs()
    calls = 0
    def failing_replay(context, actions):
        nonlocal calls
        calls += 1
        if calls == 9:
            raise RuntimeError('first verification fails')
        return transition(context, actions)
    args['replay'] = failing_replay
    report = m.run_fixture_pair(**args)
    assert report['rows'][0]['replay_error']
    assert report['rows'][0]['selected_path'] is not None
    assert not report['prediction_budget_matched']


@pytest.mark.parametrize('n,b', [(0, 1), (3, 2), (4, 5), (True, 1), (4, 0)])
def test_invalid_schedule(n, b):
    with pytest.raises(ValueError):
        m.schedule(n, b)


def test_no_cpu_neural_fallback():
    with pytest.raises(ValueError, match='CUDA'):
        m.RootAnchoredCudaReplay(object(), SimpleNamespace(type='cpu'))
    with pytest.raises(ValueError, match='RootAnchoredCudaReplay'):
        m.run_cuda_pair(**inputs())


def test_no_variable_length_primitives():
    args = inputs()
    args['primitive_bank'][0] = np.zeros((10, 2))
    with pytest.raises(ValueError, match='exactly one'):
        m.run_fixture_pair(**args)


def test_metadata_rejected_before_callback():
    args = inputs()
    args['provenance'] = {}
    args['replay'] = lambda c, a: pytest.fail('must not call model')
    with pytest.raises(ValueError):
        m.run_fixture_pair(**args)


def test_actual_entry_instrumentation_and_attribute_restoration(monkeypatch):
    # Metadata fixture, no torch model, tensor inference, or CUDA allocation.
    class Encoder:
        def forward(self, x):
            return x
    class Model:
        def __init__(self):
            self.action_encoder = Encoder()
        def predict(self, x):
            return x
    model = Model()
    replay = object.__new__(m.RootAnchoredCudaReplay)
    replay.model = model
    monkeypatch.setattr(m.RootAnchoredCudaReplay, '_check_model', lambda s: None)
    def fixture_call(self, context, actions):
        self.model.action_encoder.forward(np.zeros((1, 10)))
        self.model.predict(np.zeros((1, 3, 1)))
        return transition(context, actions)
    monkeypatch.setattr(m.RootAnchoredCudaReplay, '__call__', fixture_call)
    args = inputs()
    args['replay'] = replay
    result = m.run_cuda_pair(**args)
    assert result['actual_entries_match_dispatch']
    assert len(result['actual_model_entries']) == 48
    for row in result['rows']:
        assert row['actual_planning_predict_samples'] == 8
        assert row['actual_replay_predict_samples'] == 4
    assert 'predict' not in vars(model)
    assert 'forward' not in vars(model.action_encoder)
    args['provenance'] = {}
    with pytest.raises(ValueError):
        m.run_cuda_pair(**args)
    assert 'predict' not in vars(model)
    assert 'forward' not in vars(model.action_encoder)


def test_two_between_boundary_actions_not_three():
    args = inputs()
    args['root']['context']['action_history'].append([0.] * 10)
    with pytest.raises(ValueError):
        m.run_fixture_pair(**args)


def test_cli_real_inputs_provenance_and_source_mirror(tmp_path, monkeypatch):
    """Real sealed inputs; GPU/model execution replaced with metadata fakes."""
    import json
    from pathlib import Path
    import torch
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    compilation = Path('results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json')
    policy = Path('/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm')
    # Validate real immutable compilation/seal/input hashes, no model loading.
    source, bindings, config = checked.validate_inputs(compilation, policy)
    seen = []
    class FakeModel:
        training = False
        def eval(self): return self
        def requires_grad_(self, value): return self
        def named_buffers(self): return []
        def buffers(self): return []
        def modules(self): return [self]
        def parameters(self): return []
    monkeypatch.setattr(base, 'require_cuda_allocation', lambda d: {'fixture_only': True})
    monkeypatch.setattr(base, 'load_compatible_pretrained', lambda *a, **k: FakeModel())
    monkeypatch.setattr(base.p0, '_parameter_hash', lambda model: 'fixture_hash')
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *a: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda *a: 0)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    def fake_pair(**kwargs):
        seen.append(kwargs)
        return dict(prediction_budget_matched=True, status='metadata_fixture',
                    actual_model_entries=[dict(kind='predict', samples=168)])
    monkeypatch.setattr(m, 'run_cuda_pair', fake_pair)
    output = tmp_path / 'p2_matched_cli_fixture'
    m.run_calibration_cli(compilation, policy, output)
    protocol = json.loads((output/'p2_matched_protocol.json').read_text())
    snapshots = json.loads((output/'p2_matched_sources.json').read_text())
    report = json.loads((output/'p2_matched_report.json').read_text())
    assert len(seen) == 4 and all(report['gates'].values())
    assert protocol['bindings'] == bindings
    assert protocol['source_manifest_sha256'] == m.digest(snapshots)
    mirrored = {Path(s['path']).name for s in snapshots}
    assert {'p2_matched_composition_20261027.py', 'test_p2_matched_composition_20261027.py',
            'D2P_P2_MATCHED_COMPOSITION_20261027.md', 'checkpoint_compat.py',
            'pusht_composition_cuda_replay_v2.py', 'd2p_composition_witness_v2.py',
            'd2p_kinodynamic_program_search.py', 'run_pusht_program_search_cuda.py',
            'pusht_d2p_packet_contract.py'} <= mirrored
    import importlib
    def check_config_sources(value):
        if isinstance(value, dict):
            target = value.get('_target_', '')
            if target and not target.startswith('torch.'):
                module = importlib.import_module(target.rsplit('.', 1)[0])
                assert str(Path(module.__file__).resolve()) in {s['path'] for s in snapshots}
            for child in value.values():
                check_config_sources(child)
        elif isinstance(value, list):
            for child in value:
                check_config_sources(child)
    check_config_sources(config)
    import hashlib
    assert all(hashlib.sha256(s['source'].encode()).hexdigest() == s['sha256'] for s in snapshots)
    for passed, row in zip(seen, source['rows'], strict=True):
        assert passed['root']['root_id'] == row['root_sha256']
        assert passed['root']['context']['action_history'] == row['compiled']['input_context']['actions'][0][1:]
        assert len(passed['root']['context']['action_history']) == 2
        p = passed['provenance']
        assert set(p) == set(m.compose.__globals__['PROVENANCE_KEYS'])
        assert p['checkpoint_sha256'] == p['encoder_sha256'] == checked.OFFICIAL_SHA
        assert p['source_sha256'] == m.digest(snapshots)
        assert p['protocol_sha256'] == m.digest(protocol)
        assert p['candidate_bank_sha256'] == m.digest(passed['primitive_bank'])
        assert p['metric_sha256'] == row['compiled']['scoring']['metric']['sha256']
    assert not report['P1_admission'] and not report['P2_admission']
    assert not report['task_metric_repaired'] and not report['physical_efficacy_verified']
    with pytest.raises(FileExistsError):
        m.run_calibration_cli(compilation, policy, output)


@pytest.mark.parametrize('corruption', ['root_sha', 'history_sha', 'goal_sha', 'metric_sha', 'incoming3'])
def test_cli_bad_provenance_rejected_before_cuda(tmp_path, monkeypatch, corruption):
    import json
    from pathlib import Path
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    compilation = Path('results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json')
    policy = Path('/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm')
    source, bindings, config = checked.validate_inputs(compilation, policy)
    # Corrupt owned metadata after genuine validation to exercise CLI checks.
    upstream = []
    for path in bindings:
        if Path(path).name == 'report.json':
            value = json.loads(Path(path).read_text())
            if len(value.get('rows', [])) == 4 and all('history_rgb_sha256' in r for r in value['rows']):
                upstream.append(value)
    images = deepcopy(upstream[0])
    if corruption == 'root_sha':
        source['rows'][0]['root_sha256'] = 'bad'
    elif corruption in ('history_sha', 'goal_sha'):
        key = 'history_rgb_sha256' if corruption == 'history_sha' else 'goal_rgb_sha256'
        images['rows'][0][key] = 'bad'
    elif corruption == 'metric_sha':
        source['rows'][0]['compiled']['scoring']['metric']['sha256'] = 'bad'
    else:
        source['rows'][0]['compiled']['input_context']['actions'][0].append([0.] * 10)
    mirror_dir = tmp_path / 'p2_matched_input'
    mirror_dir.mkdir()
    image_path = mirror_dir / 'report.json'
    # New test fixture, never edit a sealed input.
    base.write_new(image_path, images)
    monkeypatch.setattr(checked, 'validate_inputs', lambda *a: (source, {str(image_path): 'fixture'}, config))
    monkeypatch.setattr(base, 'require_cuda_allocation', lambda *a: pytest.fail('CUDA must not be entered'))
    output = tmp_path / 'p2_matched_bad_cli'
    with pytest.raises(ValueError):
        m.run_calibration_cli(compilation, policy, output)
    assert not output.exists()
