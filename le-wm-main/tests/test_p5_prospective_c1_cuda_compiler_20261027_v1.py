"""Pure arithmetic/constant-array interfaces only; no torch/model/AD execution."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p5_prospective_c1_cuda_compiler_20261027_v1 as m


@pytest.fixture
def stats():
    return m.source_stats(m.STATS)


def test_native_zero_is_not_normalized_zero(stats):
    z = m.normalize_actions(np.zeros((5, 2)), stats)
    assert z.dtype == np.float32 and np.all(z != 0)
    np.testing.assert_array_equal(z[0], -np.asarray(stats['ACTION_MEAN'], np.float32)/np.asarray(stats['ACTION_STD'], np.float32))
    np.testing.assert_array_equal(m.native_ridge_packet(np.zeros(10), stats), np.zeros((5, 2)))


def test_dataset_projection_not_extra_divide100(stats):
    u = np.array([[.314159265358, -.271828182845]] * 5)
    mean, std = [np.asarray(stats[k], np.float32) for k in ('ACTION_MEAN', 'ACTION_STD')]
    expected = ((100*u).astype(np.float32)/np.float32(100)-mean)/std
    np.testing.assert_array_equal(m.normalize_actions(u, stats), expected)
    assert not np.allclose(expected, (u/100-mean)/std)


def test_centered_inverse_and_rounding_are_explicit(stats):
    delta = np.arange(10, dtype=np.float64)/10
    u = m.native_ridge_packet(delta, stats)
    assert u.dtype == np.float64
    np.testing.assert_array_equal(u, delta.reshape(5, 2)*np.asarray(stats['ACTION_STD'], np.float32).astype(np.float64))
    z0 = m.normalize_actions(np.zeros((5, 2)), stats)
    z = m.normalize_actions(u, stats)
    assert np.max(np.abs(z.astype(float)-(z0.astype(float)+delta.reshape(5, 2)))) < 1e-6


@pytest.mark.parametrize('delta', [np.ones(10)*100, np.full(10, np.nan), np.ones(9), np.full(10, np.inf)])
def test_outside_support_fails_not_clipped(stats, delta):
    with pytest.raises(ValueError):
        m.native_ridge_packet(delta, stats)


def candidates(a=.5, b=.4):
    return [dict(id='nominal', env_packet=np.zeros((5, 2)).tolist(), independent_goal_mse=a),
            dict(id='ridge', env_packet=np.full((5, 2), .2).tolist(), independent_goal_mse=b)]


def test_selection_nominal_first_tie_and_no_physical_inputs():
    assert m.choose(candidates())['id'] == 'ridge'
    assert m.choose(candidates(.4, .4))['id'] == 'nominal'
    assert m.choose(candidates(.3, .4))['id'] == 'nominal'


@pytest.mark.parametrize('fault', ['nan', 'negative', 'order', 'support'])
def test_selection_rejects_bad_contract(fault):
    c = candidates()
    if fault == 'nan': c[1]['independent_goal_mse'] = float('nan')
    if fault == 'negative': c[0]['independent_goal_mse'] = -1
    if fault == 'order': c.reverse()
    if fault == 'support': c[1]['env_packet'][0][0] = 2
    with pytest.raises(ValueError): m.choose(c)


def journal(path):
    previous = None
    rows = []
    for i, line in enumerate(path.read_text().splitlines()):
        r = json.loads(line)
        assert r['sequence'] == i and r['previous_sha256'] == previous
        assert m.digest({k: v for k, v in r.items() if k != 'entry_sha256'}) == r['entry_sha256']
        previous = r['entry_sha256']
        rows.append(r['event'])
    return rows


def test_attempt_before_return_and_partial_arrays_survive(tmp_path):
    r = {}
    ledger = m.Ledger(tmp_path, r)
    ledger.phase = 'jvp_0'
    ledger.count('attempted', 'jvp_columns')
    ledger.count('attempted', 'predictor')
    ledger.count('completed', 'predictor')
    ledger.array('predictor_partial', np.full((1, 3, 2, 2), 7, np.float32))
    ledger.count('completed', 'jvp_columns')
    ledger.array('jvp_column_0', np.ones(4, np.float32))
    ledger.phase = 'jvp_1'
    ledger.count('attempted', 'jvp_columns')
    ledger.count('attempted', 'predictor')
    ledger.event('runtime_failure', error='constant-interface mock second call failure')
    assert r['attempted']['predictor'] == 2 and r['completed']['predictor'] == 1
    assert r['attempted']['jvp_columns'] == 2 and r['completed']['jvp_columns'] == 1
    assert np.load(tmp_path/'jvp_column_0.npy', allow_pickle=False).tolist() == [1]*4
    events = journal(tmp_path/'journal.jsonl')
    assert events[-1]['phase'] == 'jvp_1'
    assert len(r['artifact_bindings']) == 2
    assert not m.check_bindings(r['artifact_bindings'])


def test_nonfinite_return_arrays_preserved_for_failure_audit(tmp_path):
    r = {}; ledger = m.Ledger(tmp_path, r)
    ledger.array('failed_column', np.array([np.nan, np.inf], np.float32))
    assert np.isnan(np.load(tmp_path/'failed_column.npy', allow_pickle=False)[0])
    assert journal(tmp_path/'journal.jsonl')[-1]['kind'] == 'array_durable'
    with pytest.raises(FileExistsError): ledger.array('failed_column', np.zeros(2))


def test_grad_instrumentation_restore_and_failure_count(tmp_path):
    calls = []
    def raw_grad(fail=False):
        calls.append(fail)
        if fail: raise RuntimeError('constant-interface failure')
        return 'no tensors'
    def raw_backward(): return 'untouched'
    fake = SimpleNamespace(autograd=SimpleNamespace(grad=raw_grad, backward=raw_backward))
    r = {}; ledger = m.Ledger(tmp_path, r)
    with pytest.raises(RuntimeError, match='constant-interface'):
        with m.grad_counts(fake, ledger):
            assert fake.autograd.grad() == 'no tensors'
            fake.autograd.grad(True)
    assert fake.autograd.grad is raw_grad and fake.autograd.backward is raw_backward
    assert r['attempted']['autograd_grad'] == 2 and r['completed']['autograd_grad'] == 1
    assert calls == [False, True]


def test_budget_exact_and_policy_fixed():
    p = m.policy()
    assert p['bridge_contract']['expected_completed'] == dict(official_encoder=1, official_rgb_frames=4,
        proprio_encoder=1, predictor=13, jvp_columns=10, autograd_grad=20)
    assert p['damping'] == .01 and not p['training'] and not p['P5_admission']
    assert p['gates']['selected_replay'] == 'torch.equal'


def test_original_error_not_overwritten_by_multiple_finalization_failures():
    r = dict(error='original runtime error', finalization_errors=[])
    def fail(): raise FileNotFoundError('source vanished')
    m.guarded_final(r, 'source', fail)
    m.guarded_final(r, 'model', fail)
    assert r['error'] == 'original runtime error'
    assert [v['check'] for v in r['finalization_errors']] == ['source', 'model']


def synthetic_bridge():
    return dict(status='ready_for_main_CUDA_compile', blockers=[], compiler_inputs_sha256='inputs',
        compiler_inputs=dict(allowed_data_files={str(i): 'sha' for i in range(7)}),
        goal=dict(rgb_sha256='goal'), runner_sources=dict(compiler=dict(path=m.__file__, sha256='src')),
        model_identity={'frozen': 'same'}, prefix_env_commands=np.zeros((10, 2)).tolist(),
        factual_reference_packet=np.full((5, 2), .1).tolist(), root_identity_sha256='root')


def test_exact_roles_shape_dtype_and_bridge_acceptance(tmp_path):
    p = synthetic_bridge()
    r = dict(schema=m.bridge.TAG+'.compiler', status='selected_packet_replayed', prepared_sha256=m.digest(p),
        compiler_inputs_sha256='inputs', compiler_data_bindings=p['compiler_inputs']['allowed_data_files'],
        goal_rgb_sha256='goal', compiler_source=p['runner_sources']['compiler'], backend='host_cuda_only',
        model_before=p['model_identity'], model_after=p['model_identity'], used_observed_frame15=False,
        physical_calls=0, selected_replay_exact=True, partial_work_unknown=False)
    ledger = m.Ledger(tmp_path, r)
    shapes = dict(response_columns=((m.DIM, 10), 'float32'), gram=((10, 10), 'float64'),
                  rhs=((10,), 'float64'), ridge_delta=((10,), 'float64'))
    for role in m.ROLES:
        shape, dtype = shapes.get(role, ((1, 256, 384), 'float32'))
        ledger.array(role, np.zeros(shape, dtype), role)
    r.update(attempted=dict(m.bridge.contract()['compiler']['expected_completed']),
             completed=dict(m.bridge.contract()['compiler']['expected_completed']), candidates=candidates())
    selected = m.choose(r['candidates'])
    r.update(selected_id=selected['id'], selected_env_packet=selected['env_packet'])
    rows = m.bridge.frozen_branches(p, r)
    assert len(rows) == 3 and rows[0]['suffix_env_packet'] == selected['env_packet']
    assert set(r['artifact_roles']) == set(m.ROLES)
    assert not m.check_bindings(r['artifact_bindings'])


def test_actual_blocked_protocol_cannot_authorize_cuda():
    p = m.read_json(m.ROOT/'results/p5_prospective_control_bridge_20261027_v1_prepare_v1/protocol.json')
    m.validate_bridge(p, m.digest(p))
    with pytest.raises(ValueError, match='blocked'):
        m.validate_bridge(p, m.digest(p), ready=True)
    with pytest.raises(ValueError, match='retained bridge digest'):
        m.validate_bridge(p, 'wrong')


def test_only_explicit_cuda_core_imports_torch_and_no_archived_data_calls():
    tree = ast.parse(Path(m.__file__).read_text())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            assert 'torch' not in ast.unparse(node)
    text = ast.unparse(tree)
    for forbidden in ('v8.prepare(', 'v8.verify_manifest(', 'factual_inputs(', 'read_arrays(', 'row_inputs('):
        assert forbidden not in text
    assert 'bridge.compiler_bundle(' in text
    assert 'torch.equal(' in text and 'SDPBackend.MATH' in text
    assert 'torch.linalg.solve(' in text and 'torch.autograd.functional.jvp(' in text


def test_frozen_loader_metadata_excludes_episode_targets():
    frozen = m.read_json(m.V8)
    loader = {k: frozen[k] for k in ('seals', 'official_weight_mapping', 'expected_dynamics_schema', 'expected_model_receipts')}
    assert 'row' not in loader and 'expected_v7_features' not in loader
    assert len(loader['seals']) > 0


def test_output_is_new_exclusive_direct_child(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    (tmp_path/'results').mkdir()
    path = tmp_path/'results'/(m.TAG+'_fixture')
    assert m.new_output(path) == path
    with pytest.raises(ValueError): m.new_output(path)
    with pytest.raises(ValueError): m.new_output(tmp_path/'results'/'wrong')


def test_source_stats_rejects_missing_or_nonliteral(tmp_path):
    path = tmp_path/'stats.py'
    path.write_text('ACTION_MEAN = torch.tensor(make_values())\n')
    with pytest.raises((ValueError, TypeError)): m.source_stats(path)

