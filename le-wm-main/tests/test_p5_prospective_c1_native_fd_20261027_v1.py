"""Non-neural arithmetic/receipt fixtures only. No CUDA, torch or simulator."""
import ast
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p5_prospective_c1_native_fd_20261027_v1 as m


def test_schedule_all_columns_eps_and_plus_then_minus():
    z = np.arange(10, dtype=np.float32)+.25
    schedule = list(m.perturbations(z))
    assert len(schedule) == 40
    assert [(e, i, s) for e, i, s, _ in schedule] == [(e, i, s) for e in (.01, .005) for i in range(10) for s in (1, -1)]
    for eps, i, sign, value in schedule:
        expected = z.copy(); expected[i] += np.float32(sign*eps)
        np.testing.assert_array_equal(value, expected)
        assert np.flatnonzero(value != z).tolist() == [i]
        assert value.dtype == np.float32


@pytest.mark.parametrize('z', [np.zeros(10), np.zeros(9, np.float32), np.full(10, np.nan, np.float32)])
def test_bad_nominal_rejected(z):
    with pytest.raises(ValueError): list(m.perturbations(z))


def test_central_float64_subtraction_declared_epsilon():
    plus, minus = np.array([1.1, -.6], np.float32), np.array([.9, -.7], np.float32)
    np.testing.assert_array_equal(m.central(plus, minus, .01), (plus.astype(float)-minus.astype(float))/.02)
    assert m.central(plus, minus, .01).dtype == np.float64


@pytest.mark.parametrize('eps', [.1, .001, 0., -.01])
def test_no_epsilon_tuning(eps):
    with pytest.raises(ValueError): m.central(np.zeros(4, np.float32), np.zeros(4, np.float32), eps)


def test_no_acceptance_gate_zero_norm_is_unknown():
    j = np.tile(np.arange(10), (3, 1)); out = m.compare(j, j)
    assert out['matrix']['error_norm'] == 0
    assert out['columns'][0]['relative_error_to_jvp'] is None and out['columns'][0]['cosine'] is None
    assert out['acceptance_threshold'] is out['accepted'] is None
    wrong = m.compare(j, -j)
    assert wrong['matrix']['relative_error_to_jvp'] == 2
    assert wrong['matrix']['cosine'] == pytest.approx(-1)


def test_fd_loop_exact_count_saved_arrays_order(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'DIM', 6)
    receipt = {}; ledger = m.compiler.Ledger(tmp_path, receipt); calls = []
    linear = np.arange(60, dtype=np.float32).reshape(6, 10)/60
    def predict(z):
        ledger.count('attempted', 'predictor'); calls.append(z.copy())
        result = (linear@z).astype(np.float32)
        ledger.count('completed', 'predictor')
        return result
    matrices = m.fd_loop(predict, np.linspace(-.2, .2, 10, dtype=np.float32), ledger)
    assert len(calls) == 40 and receipt['attempted'] == receipt['completed'] == m.policy()['completed']
    assert len(receipt['arrays']) == 82
    for eps in (.01, .005):
        assert matrices[str(eps)].shape == (6, 10)
        for i in range(10):
            stem = f'eps_{eps}_column_{i}_'
            plus = np.load(tmp_path/(stem+'plus_successor.npy'), allow_pickle=False)
            minus = np.load(tmp_path/(stem+'minus_successor.npy'), allow_pickle=False)
            np.testing.assert_array_equal(matrices[str(eps)][:, i], m.central(plus, minus, eps))
    assert not m.compiler.check_bindings(receipt['artifact_bindings'])


def test_failure_leaves_intent_partial_not_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'DIM', 3)
    r = {}; ledger = m.compiler.Ledger(tmp_path, r)
    def predict(z):
        ledger.count('attempted', 'predictor'); raise RuntimeError('non-neural fixture dispatch failure')
    with pytest.raises(RuntimeError, match='fixture'):
        m.fd_loop(predict, np.zeros(10, np.float32), ledger)
    assert r['attempted']['predictor'] == 1 and r['completed']['predictor'] == 0
    assert not list(tmp_path.glob('FD_matrix*'))
    events = [json.loads(line)['event'] for line in (tmp_path/'journal.jsonl').read_text().splitlines()]
    assert any(e['kind'] == 'FD_dispatch' for e in events)


def test_nonfinite_saved_before_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'DIM', 3)
    r = {}; ledger = m.compiler.Ledger(tmp_path, r)
    with pytest.raises(ValueError):
        m.fd_loop(lambda z: np.full(3, np.nan, np.float32), np.zeros(10, np.float32), ledger)
    saved = list(tmp_path.glob('*successor.npy'))
    assert len(saved) == 1 and np.isnan(np.load(saved[0], allow_pickle=False)).all()
    assert not list(tmp_path.glob('FD_matrix*'))


@pytest.fixture
def reference(tmp_path, monkeypatch):
    stats = m.compiler.source_stats(m.compiler.STATS)
    bundle = dict(history_env_packets=np.zeros((2, 5, 2)), raw_proprio_context=np.zeros((3, 4), np.float32))
    monkeypatch.setattr(m.compiler.bridge, 'compiler_bundle', lambda p, h: bundle)
    source = tmp_path/'manifest.json'; source.write_text('{}')
    prepared = dict(model_identity={'fixture': 'immutable'}, compiler_inputs_sha256='input',
        compiler_inputs=dict(allowed_data_files={'seven-asset-fixture': 'sha'}),
        runner_sources=dict(compiler={'path': 'compiler.py', 'sha256': 'source'}))
    manifest = dict(bridge_sha256='bridge', stats=stats)
    r = dict(status='selected_packet_replayed', runner_schema=m.compiler.TAG,
        manifest_file_sha256=m.file_sha(source), prepared_sha256='bridge', compiler_inputs_sha256='input',
        compiler_data_bindings=prepared['compiler_inputs']['allowed_data_files'],
        compiler_source=prepared['runner_sources']['compiler'], backend='host_cuda_only',
        used_observed_frame15=False, physical_calls=0, training=False,
        model_before=prepared['model_identity'], model_after=prepared['model_identity'],
        partial_work_unknown=False, counts_match=True, finalization_errors=[], error=None,
        attempted=m.compiler.bridge.contract()['compiler']['expected_completed'],
        completed=m.compiler.bridge.contract()['compiler']['expected_completed'], arrays={}, artifact_bindings={})
    for name, (shape, dtype) in m.required_arrays().items():
        a = np.zeros(shape, dtype)
        if name == 'normalized_nominal': a = m.compiler.normalize_actions(np.zeros((5, 2)), stats).reshape(10)
        elif name == 'normalized_history': a = m.compiler.normalize_actions(bundle['history_env_packets'], stats).reshape(1, 2, 10)
        elif name == 'normalized_proprio': a = (bundle['raw_proprio_context']-np.asarray(stats['PROPRIO_MEAN'], np.float32))/np.asarray(stats['PROPRIO_STD'], np.float32)
        path = tmp_path/(name+'.npy')
        with path.open('xb') as stream: np.save(stream, a, allow_pickle=False)
        r['arrays'][name] = dict(path=str(path), sha256=m.file_sha(path), shape=list(shape), dtype=dtype)
        r['artifact_bindings'][str(path)] = m.file_sha(path)
    return r, dict(path=source, manifest=manifest), prepared


def test_full_reference_non_neural_fixture(reference):
    assert set(m.reference_arrays(*reference)) == set(m.required_arrays())


@pytest.mark.parametrize('fault', ['status', 'model', 'frame15', 'source', 'inputs', 'counts', 'history', 'nominal', 'JVP', 'missing'])
def test_reference_tamper_rejected(reference, fault):
    r, c, p = deepcopy(reference)
    if fault == 'status': r['status'] = 'compiler_failed'
    elif fault == 'model': r['model_after'] = {'wrong': 'model'}
    elif fault == 'frame15': r['used_observed_frame15'] = True
    elif fault == 'source': r['compiler_source'] = {'wrong': 'source'}
    elif fault == 'inputs': r['compiler_inputs_sha256'] = 'wrong'
    elif fault == 'counts': r['counts_match'] = False
    elif fault == 'missing': del r['arrays']['jvp_column_9']
    else:
        key = dict(history='normalized_history', nominal='normalized_nominal', JVP='response_columns')[fault]
        rec = r['arrays'][key]; arr = np.load(rec['path'], allow_pickle=False); arr.flat[0] += .1
        with Path(rec['path']).open('wb') as stream: np.save(stream, arr, allow_pickle=False)
        rec['sha256'] = m.file_sha(rec['path']); r['artifact_bindings'][rec['path']] = rec['sha256']
    with pytest.raises(ValueError): m.reference_arrays(r, c, p)


def test_failed_compiler_requires_explicit_diagnostic_mode(reference):
    r, c, p = reference
    r.update(status='compiler_failed', error='retained fixture compiler failure', partial_work_unknown=True)
    with pytest.raises(ValueError): m.reference_arrays(r, c, p)
    assert len(m.reference_arrays(r, c, p, diagnostic=True)) == 27
    del r['arrays']['jvp_column_9']
    with pytest.raises(ValueError): m.reference_arrays(r, c, p, diagnostic=True)


def test_array_shape_dtype_hash_rejected(tmp_path):
    p = tmp_path/'bad.npy'
    with p.open('xb') as f: np.save(f, np.zeros(5, np.float64), allow_pickle=False)
    rec = dict(path=str(p), sha256=m.file_sha(p), shape=[5], dtype='float64')
    with pytest.raises(ValueError): m.load_array(rec, (5,), 'float32')
    with pytest.raises(ValueError): m.load_array(rec, (4,), 'float64')
    rec['sha256'] = 'wrong'
    with pytest.raises(ValueError): m.load_array(rec, (5,), 'float64')


def test_policy_counts_no_new_nominal_no_selection():
    p = m.policy()
    assert sum(p['completed'].values()) == p['completed']['predictor'] == 40
    assert p['new_nominal_predictions'] == 0 and p['selected_packet_changed'] is False
    assert p['epsilons'] == [.01, .005] and p['P5_admission'] is p['C2_established'] is False


def test_no_archived_target_reader_or_cuda_in_prepare():
    tree = ast.parse(Path(m.__file__).read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            assert n.func.attr not in {'read_arrays', 'row_inputs', 'validate_cuda', 'verify_manifest', 'step', 'reset', 'render', 'choose', 'frozen_branches'}
    prepare = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(prepare))
    for n in tree.body:
        if isinstance(n, ast.Import): assert all(x.name != 'torch' for x in n.names)
