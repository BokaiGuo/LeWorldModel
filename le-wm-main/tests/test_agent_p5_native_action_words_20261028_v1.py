"""Analytic RAM fixtures only. No model/torch/physics/network execution."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_native_action_words_20261028_v1 as m


def npy(path, a):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.save(f, a, allow_pickle=False)
    return dict(path=str(path), sha256=m.sha(path), bytes=path.stat().st_size,
                dtype=str(a.dtype), shape=list(a.shape))


@pytest.fixture
def environment(tmp_path, monkeypatch):
    """Small axes, but same saved producer/consumer file and journal layout."""
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m, 'PATCHES', 2)
    monkeypatch.setattr(m, 'FEATURES', 3)
    monkeypatch.setattr(m, 'PROP', 2)
    monkeypatch.setattr(m, 'FLOOR', 0)
    (tmp_path / 'results').mkdir()
    bindings = {}
    loader = tmp_path / 'experiments/spectral_reachability' / (m.LOADER + '.py')
    loader.parent.mkdir(parents=True)
    loader.write_text('# fixture only, never imported\n')
    monkeypatch.setattr(m, 'LOADER_SHA', m.sha(loader))
    bindings[str(loader)] = m.sha(loader)
    native = tmp_path / 'native_source.py'
    native.write_text('# native-interface fixture\n')
    monkeypatch.setattr(m, 'NATIVE', {'native_source.py': m.sha(native)})
    bindings[str(native)] = m.sha(native)
    for path in (tmp_path / 'tests' / ('test_' + m.TAG + '.py'), tmp_path / 'innerdocs' / (m.TAG + '.md')):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture source identity')
    specs, rows = [], []
    state = {'immutable_fixture': True}
    for seed in m.SEEDS:
        base = tmp_path / f'factual/seed_{seed}'
        props = np.arange(12, dtype=np.float32).reshape(3, 4)
        packets = np.arange(30, dtype=np.float32).reshape(3, 10) / 100
        patch = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
        rec = npy(base / 'patches.npy', patch)
        for name, a in (('context_proprio', props), ('normalized_packets', packets)):
            r = npy(base / 'a/consumer' / (name + '.npy'), a)
            bindings[r['path']] = r['sha256']
        goal = base / 'a/frame_10/full_public_state.json'
        goal.parent.mkdir(parents=True)
        m.write_new(goal, {'fixture_goal': seed})
        bindings[str(goal)] = m.sha(goal)
        specs.append(dict(seed=seed, root_sha256=str(seed), factual=str(base),
            stats={'ACTION_MEAN': [-.0087, .0068], 'ACTION_STD': [.2019, .2002]},
            controls={'actual': packets.tolist()}))
        rows.append(dict(seed=seed, root_sha256=str(seed), status='completed', counts_match=True,
            frames_match=True, model_after=state, artifacts={'official_patch_features': rec}))
    saved_manifest = dict(schema=m.PARENT, status='prepared_non_neural', protocol={'fixture': True},
        roots=specs, bindings=bindings, parent_manifest={'fixture': 'no real loader'})
    mp, rp = tmp_path / 'parent_manifest.json', tmp_path / 'parent_report.json'
    m.write_new(mp, saved_manifest)
    m.write_new(rp, dict(schema=m.PARENT, status='fresh_native_panel_fidelity_recorded',
        protocol=saved_manifest['protocol'], cuda_executed=True, error=None, finalization_errors=[],
        source_check_errors=[], sources_unchanged=True, model_unchanged=True,
        model_before=state, model_after=state, manifest_sha256=m.digest(saved_manifest), rows=rows))
    monkeypatch.setattr(m, 'MANIFEST', mp)
    monkeypatch.setattr(m, 'REPORT', rp)
    monkeypatch.setattr(m, 'PINS', {str(mp): m.sha(mp), str(rp): m.sha(rp)})
    return tmp_path


def make_fixture_run(prepared, directory):
    """Fabricated analytic receipt to test validation, not fake CUDA evidence."""
    store = m.Store(directory)
    rows = []
    for spec in prepared['roots']:
        seed = spec['seed']
        row = dict(seed=seed, root_sha256=spec['root_sha256'], status='completed', artifacts={}, error=None,
                   program_status=dict.fromkeys(m.WORDS, 'completed'))
        meter = m.Meter(lambda k, v: store.event(k, dict(seed=seed, meter=v)), store.guard)
        def emit(word, step, name, value):
            key = f'{word}_{step}_{name}'
            row['artifacts'][key] = store.emit(seed, word, step, name, value)
            store.event('artifact', dict(seed=seed, name=key, record=row['artifacts'][key]))
        meter.before('proprio_encoder', 1)
        prop = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
        meter.after('proprio_encoder', 1)
        emit('shared', -1, 'context_proprio_features', prop)
        def predict(v, a, p):
            meter.before('predictor', 1)
            out = (v + np.float32(a[-1].sum()), p + np.float32(.125))
            meter.after('predictor', 1)
            return out
        terminal = m.rollout(predict, m.array(spec['patches'], (4, 2, 3))[:3], prop,
            m.array(spec['packets'], (3, 10))[:2],
            {k: m.normalize(a, spec['stats']) for k, a in m.programs().items()}, emit)
        row['meter'], row['contrasts'] = meter.snapshot(), m.contrasts(terminal)
        store.event('root_terminal', row)
        rows.append(row)
    report = dict(schema=m.TAG, status='native_action_word_predictions_recorded',
        policy=m.policy(), manifest_sha256=m.digest(prepared), rows=rows,
        model_before=prepared['expected_model'], model_after=prepared['expected_model'],
        errors=[], finalization_errors=[], cuda_executed=True, scientific_admission=False,
        physical_release=False, action_witness_established=False,
        wall_seconds=.5, journal_terminal_sha256=store.previous)
    path = store.output / 'report.json'
    m.write_new(path, report)
    return path, report


def test_programs_and_units():
    p = m.programs()
    assert tuple(p) == m.WORDS
    assert all(a.shape == (2, 5, 2) and a.dtype == np.float64 for a in p.values())
    assert np.array_equal(p['xy'][::-1], p['yx'])
    assert np.array_equal(p['x_minus_x'][0], -p['x_minus_x'][1])
    assert not p['zero_zero'].any()
    stats = {'ACTION_MEAN': [-.0087, .0068], 'ACTION_STD': [.2019, .2002]}
    z = m.normalize(p['zero_zero'], stats)
    assert np.count_nonzero(z) == 20  # physical zero is not normalized zero
    expected = (((100 * p['xy']).astype(np.float32) / np.float32(100)
                 - np.array(stats['ACTION_MEAN'], np.float32)) / np.array(stats['ACTION_STD'], np.float32)).reshape(2, 10)
    assert np.array_equal(expected, m.normalize(p['xy'], stats))


@pytest.mark.parametrize('change', ['support', 'nan', 'shape', 'dtype', 'std'])
def test_normalization_rejections(change):
    a = m.programs()['xy'].copy()
    stats = {'ACTION_MEAN': [0., 0.], 'ACTION_STD': [1., 1.]}
    if change == 'support': a[0, 0, 0] = 1.001
    if change == 'nan': a[0, 0, 0] = np.nan
    if change == 'shape': a = a[:1]
    if change == 'dtype': a = a.astype(np.float32)
    if change == 'std': stats['ACTION_STD'][0] = 0
    with pytest.raises(ValueError): m.normalize(a, stats)


def test_causal_feedback_no_branch_substitution():
    visual = np.arange(18, dtype=np.float32).reshape(3, 2, 3)
    prop = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    history = np.arange(20, dtype=np.float32).reshape(2, 10)
    words = {k: a.reshape(2, 10).astype(np.float32) for k, a in m.programs().items()}
    calls = []
    def predict(v, a, p):
        calls.append((v.copy(), a.copy(), p.copy()))
        return v + np.float32(100), p + np.float32(200)
    out = m.rollout(predict, visual, prop, history, words, lambda *args: None)
    assert len(calls) == 8 and tuple(out) == m.WORDS
    for i, k in enumerate(m.WORDS):
        first, second = calls[2*i:2*i+2]
        assert np.array_equal(first[0], visual) and np.array_equal(first[2], prop)
        assert np.array_equal(first[1], np.concatenate((history, words[k][:1])))
        assert np.array_equal(second[0], np.concatenate((visual[1:], visual[-1:] + 100)))
        assert np.array_equal(second[2], np.concatenate((prop[1:], prop[-1:] + 200)))
        assert np.array_equal(second[1], np.concatenate((history[1:], words[k])))


def test_wrong_proprio_abi_fails_not_substituted():
    with pytest.raises(ValueError):
        m.rollout(lambda v, a, p: (v, p[:, :1]), np.zeros((3, 2, 3), np.float32),
            np.zeros((3, 2, 2), np.float32), np.zeros((2, 10), np.float32),
            {k: a.reshape(2, 10).astype(np.float32) for k, a in m.programs().items()}, lambda *args: None)


def test_meter_cap_failure_no_refund_and_batch():
    x = m.Meter(lambda *args: None)
    x.before('predictor', 1)  # raised model attempt: intentionally no return
    assert x.attempted['predictor'] == 1 and x.completed['predictor'] == 0
    for _ in range(7): x.before('predictor', 1)
    with pytest.raises(ValueError): x.before('predictor', 1)
    with pytest.raises(ValueError): x.before('official_encoder', 1)
    with pytest.raises(ValueError): m.Meter(lambda *args: None).before('predictor', 2)
    with pytest.raises(ValueError): m.Meter(lambda *args: None).after('predictor', 1)
    assert x.snapshot()['samples_attempted']['predictor'] == 8
    assert x.snapshot()['frames_attempted']['predictor'] == 24


def test_prepare_verify_complete_and_source_drift(environment):
    p = m.prepare()
    m.verify(p)
    assert len(p['roots']) == 3 and not p['physical_release']
    assert p['policy']['counts_total'] == dict(official_encoder=0, proprio_encoder=3, predictor=24)
    (environment / 'native_source.py').write_text('drift')
    with pytest.raises(ValueError): m.verify(p)


def test_parent_report_hash_failure(environment):
    m.REPORT.write_text('{}')
    with pytest.raises(ValueError, match='SHA'): m.prepare()


def test_saved_arrays_roundtrip_no_models(environment):
    p = m.prepare()
    path, r = make_fixture_run(p, environment / 'results' / (m.TAG + '_RAM'))
    receipt = m.consume(p, path, m.sha(path))
    assert receipt['status'] == 'all12_saved_program_predictions_verified'
    assert receipt['action_witness_established'] is False
    assert not receipt['physical_release'] and not receipt['scientific_admission']


@pytest.mark.parametrize('mutation', ['raw_array', 'report_sha', 'root', 'meter', 'word_missing', 'chain', 'step2_feedback', 'context_proprio', 'metric', 'release'])
def test_consumer_mutations_fail_closed(environment, mutation):
    p = m.prepare()
    path, r = make_fixture_run(p, environment / 'results' / (m.TAG + '_RAM'))
    row = r['rows'][0]
    if mutation == 'report_sha':
        with pytest.raises(ValueError, match='SHA'): m.consume(p, path, '0'*64)
        return
    if mutation == 'raw_array':
        Path(next(iter(row['artifacts'].values()))['path']).write_bytes(b'broken')
    elif mutation == 'root': row['root_sha256'] = 'wrong'
    elif mutation == 'meter': row['meter']['completed']['predictor'] -= 1
    elif mutation == 'word_missing': row['artifacts'].pop('xy_0_visual_out')
    elif mutation == 'chain':
        ep = path.parent / 'event_00001.json'
        ev = json.loads(ep.read_text()); ev['sequence'] += 1; ep.write_text(json.dumps(ev))
    elif mutation in ('step2_feedback', 'context_proprio'):
        key = 'xy_1_visual_in' if mutation == 'step2_feedback' else 'xy_1_proprio_in'
        rec = row['artifacts'][key]
        a = np.load(rec['path'], allow_pickle=False); a.flat[0] += 1
        with Path(rec['path']).open('wb') as f: np.save(f, a, allow_pickle=False)
        rec['sha256'] = m.sha(rec['path'])
    elif mutation == 'metric': row['contrasts']['xy_vs_yx_patch_mse'] += 1
    elif mutation == 'release': r['physical_release'] = True
    path.write_text(json.dumps(r))
    with pytest.raises((ValueError, KeyError)): m.consume(p, path, m.sha(path))


def test_recomputed_feedback_mutation_even_with_new_chain(environment):
    p = m.prepare()
    path, r = make_fixture_run(p, environment / 'results' / (m.TAG + '_RAM'))
    key = 'xy_1_proprio_in'
    rec = r['rows'][0]['artifacts'][key]
    a = np.load(rec['path'], allow_pickle=False); a[-1] += np.float32(1)
    with Path(rec['path']).open('wb') as f: np.save(f, a, allow_pickle=False)
    rec['sha256'] = m.sha(rec['path'])
    previous = None
    for event in sorted(path.parent.glob('event_*.json')):
        e = json.loads(event.read_text())
        if e['kind'] == 'artifact' and e['value']['seed'] == m.SEEDS[0] and e['value']['name'] == key:
            e['value']['record'] = rec
        if e['kind'] == 'root_terminal' and e['value']['seed'] == m.SEEDS[0]:
            e['value'] = r['rows'][0]
        e['previous_sha256'] = previous
        e['event_sha256'] = m.digest({k: v for k, v in e.items() if k != 'event_sha256'})
        previous = e['event_sha256']; event.write_text(json.dumps(e))
    r['journal_terminal_sha256'] = previous
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='causal context'): m.consume(p, path, m.sha(path))


@pytest.mark.parametrize('limit', ['wall', 'output', 'floor', 'existing'])
def test_store_resource_failures(environment, monkeypatch, limit):
    path = environment / 'results' / (m.TAG + '_cap')
    s = m.Store(path)
    if limit == 'existing':
        with pytest.raises(ValueError): m.Store(path)
        return
    if limit == 'wall': s.started -= m.WALL + 1
    if limit == 'output': s.bytes = m.CAP + 1
    if limit == 'floor': monkeypatch.setattr(m, 'FLOOR', 10**30)
    with pytest.raises(ValueError): s.guard()


@pytest.mark.parametrize('text', ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_strict_json(environment, text):
    p = environment / 'bad.json'; p.write_text(text)
    with pytest.raises(ValueError): m.read_json(p)


def test_saved_array_dtype_and_symlink(environment):
    p = environment / 'a.npy'
    r = npy(p, np.zeros((2, 3), np.float64))
    with pytest.raises(ValueError): m.array(r, (2, 3))
    link = environment / 'link.npy'; link.symlink_to(p)
    with pytest.raises(ValueError): m.sha(link)


def test_import_boundary_and_no_ad_runner():
    tree = ast.parse(Path(m.__file__).read_text())
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = [a.name for n in imports for a in n.names]
    assert not any(n.startswith(('torch', 'gym', 'pickle', 'requests')) for n in names)
    source = Path(m.__file__).read_text()
    assert 'functional.jvp' not in source and 'autograd.grad(' not in source
    assert m.policy()['AD'] == m.policy()['JVP'] == m.policy()['physics'] == 0


def test_cuda_unavailable_retains_denominator_no_fallback(environment, monkeypatch):
    p = m.prepare()
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    r = m.run_cuda(p, environment / 'results' / (m.TAG + '_FAIL'))
    assert r['status'] == 'failed' and not r['cuda_executed']
    assert len(r['rows']) == 3
    assert all(row['status'] == 'not_run' and row['program_status'] == dict.fromkeys(m.WORDS, 'not_run') for row in r['rows'])
    assert 'CUDA required' in r['errors'][0]


def test_manifest_program_mutation_rejected(environment):
    p = m.prepare()
    p['roots'][0]['plans']['xy']['raw'][0][0][0] += .1
    with pytest.raises(ValueError, match='prepared contract'): m.verify(p)


def test_native_material_size_budget_regression():
    # All five arrays/step retained, not an endpoint-only shortcut.
    per_step_elements = 2 * (3*256*384) + 2 * (3*256*16) + 30
    per_root_bytes = 4 * (3*256*16 + 8*per_step_elements) + 41*1024
    assert 3*per_root_bytes + 4*1024**2 < m.CAP
    assert 3 * 41 == 123
