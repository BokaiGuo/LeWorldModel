"""RAM-only fixtures, NumPy arithmetic; no executor, neural or physics imports."""
import ast
import base64
from copy import deepcopy
import gzip
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p1_teacher_fidelity_error_decomposition_20261027_v1 as m


def fixture():
    root = Path('/RAM/teacher')
    files = {}
    def put(path, value):
        raw = value if isinstance(value, bytes) else m.canonical(value)
        files[str(path)] = raw
        return m.sha(raw)
    def array(path, a, external=False):
        stream = BytesIO()
        np.save(stream, a, allow_pickle=False)
        h = put(path, stream.getvalue())
        if external:
            return dict(path=str(path), file_sha256=h, array_sha256=m.sha(a.tobytes()), dtype='float32', shape=list(a.shape))
        return dict(file=path.name, sha256=h, dtype='float32', shape=list(a.shape))
    def value(x):
        return np.full(192, x, np.float32)
    contract = dict(order=[list(k) for k in m.KEYS], horizon=1, packet_steps=5, latent_dim=192)
    p = dict(schema=m.TEACHER+'.prepare', contract=contract, expected_model_hash='a'*64, bindings={}, inputs=[], evaluators=[])
    rows, ledger = [], []
    resources = {k: dict(attempted=12, completed=12) for k in m.RESOURCES}
    for index, (slot, arm, mode) in enumerate(m.KEYS):
        identity = dict(index=index, episode_id=f'e{index//3:02}_{mode}', source_slot=slot, prefix_arm=arm,
                        mode=mode, parent_task_id=str(slot)*64, parent_selection_sha256=str(index//3)*64)
        prefix = value(1)
        # Pred R lower cost, actual R higher: all four layers reverse in this mock.
        pred, actual = value(1 if mode == 'R' else 2), value(3 if mode == 'R' else 2)
        z = np.stack((value(0), value(.5), prefix))
        req = dict(schema=m.TEACHER+'.forward_input', identity=identity, latents=z.tolist(),
                   history=np.zeros((2, 10), np.float32).tolist(), raw5=np.zeros((5, 2), np.float32).tolist(), mean=[0., 0.], std=[1., 1.])
        req['input_digest'] = m.digest(req)
        put(root / f'input_{index:02}.json', req)
        ev = dict(identity=identity, goal=value(0).tolist())
        for kind, step, a in (('prefix', 5, prefix), ('label', 10, actual)):
            ev[kind] = array(Path('/RAM/live') / f'{identity["episode_id"]}_encode{step}' / 'native192.npy', a, True)
            p['bindings'][ev[kind]['path']] = ev[kind]['file_sha256']
        prediction = array(root / f'prediction_{index:02}.npy', pred)
        label = array(root / f'label_{index:02}.npy', actual)
        d = m.decomposition(pred, actual, value(0))
        metrics = dict(factual_MSE=d['latent_MSE'], predicted_Q=d['predicted_Q'], actual_Q=d['actual_Q'],
                       predicted_Q_minus_actual_Q=d['predicted_Q_minus_actual_Q'],
                       actual_prefix_terminal_mse=float(np.mean((prefix.astype(float)-actual.astype(float))**2)))
        rows.append(dict(identity=identity, status='scored', prediction=prediction, label=label, metrics=metrics))
        projected = np.stack((z[1], prefix, pred))
        for j, resource in enumerate(m.RESOURCES):
            op = 4*index+j
            entry = dict(index=op, episode_index=index, resource=resource, completed=True)
            if j != 1:
                ins, output = {0: ([np.zeros((1, 3, 10), np.float32)], z[None]),
                               2: ([z[None], z[None]], projected[None]),
                               3: ([projected], projected)}[j]
                entry['inputs'] = [array(root / f'op_{op:02}_input{k}.npy', a) for k, a in enumerate(ins)]
                entry['output'] = array(root / f'op_{op:02}_output.npy', output)
            ledger.append(entry)
        p['inputs'].append(req)
        p['evaluators'].append(ev)
    prepared = '/RAM/prepare.json'
    semantic = m.canonical(p)
    envelope = dict(schema=m.TEACHER+'.prepare_envelope', codec='single-gzip+base64', semantic_bytes=len(semantic),
                    semantic_sha256=m.sha(semantic), payload=base64.b64encode(gzip.compress(semantic, mtime=0)).decode())
    prepare_sha = put(prepared, envelope)
    put(root / 'binding.json', dict(prepared=prepared, prepare_sha256=prepare_sha))
    sealed_rows = deepcopy(rows)
    for row in sealed_rows:
        row.update(status='predicted', label=None, metrics=None)
    seal_sha = put(root / 'predictions_seal.json', dict(schema=m.TEACHER+'.predictions_seal', rows=sealed_rows,
                                                     resources=resources, ledger=ledger))
    events, previous = [], None
    for index, entry in enumerate([dict(kind='all_predictions_sealed', sha256=seal_sha)] +
                                  [dict(kind='episode_scored', receipt=r) for r in rows]):
        e = dict(index=index, previous_sha256=previous, **entry)
        e['event_sha256'] = m.digest(e)
        previous = e['event_sha256']
        events.append(m.canonical(e))
    put(root / 'events.jsonl', b''.join(events))
    r = dict(schema=m.TEACHER+'.report', status='teacher_factual_recorded_not_admitted', errors=[], first_error=None,
             blocked=[], source_verified=True, model_unchanged=True, fixed_denominator=12, independent_tasks=2,
             rows=rows, host=dict(device='cuda:0'), prepared=prepared, prepare_sha256=prepare_sha,
             contract=contract, model_hash_before='a'*64, model_hash_after='a'*64, resources=resources,
             ledger=ledger, predictions_seal_sha256=seal_sha,
             artifact_sha256={Path(f).name: m.sha(raw) for f, raw in files.items() if Path(f).parent == root})
    report_path = root / 'report.json'
    report_sha = put(report_path, r)
    def saved():
        return m.Saved(reader=lambda path: files[path], listing=lambda path: {Path(f).name for f in files if Path(f).parent == path})
    return files, r, report_path, report_sha, saved


def test_fixed12_complete_recomputation_and_all_receipts():
    files, r, path, h, saved = fixture()
    result = m.analyze(path, h, saved())
    assert len(result['rows']) == 12 and result['strict_reversal_count'] == 4
    assert len(result['R_minus_O']) == 4 and not result['P1_admission']
    assert set(result['inputs']) == set(files)
    assert len(result['arrays']) == sum(f.endswith('.npy') for f in files)
    for row in result['rows']:
        assert len(row['prediction_steps']) == 3
        assert row['prediction_steps'][0]['label_scope'].startswith('saved_source')
        assert row['prediction_steps'][2]['latent_MSE'] == row['endpoint']['latent_MSE']
        assert row['endpoint']['normalization_denominator'] == 192
        assert row['endpoint']['identity_residual'] == 0


@pytest.mark.parametrize('p,a,g', [(2., 3., 0.), (0., 0., 0.), (3., 1., 2.), (1., 1., 1.)])
def test_signed_cross_exact_mean192_and_zero_goal_distance(p, a, g):
    d = m.decomposition(*(np.full(192, x, np.float32) for x in (p, a, g)))
    assert d['normalized_cross'] == 2*(a-g)*(p-a)
    assert d['normalized_squared_error'] == (p-a)**2
    assert d['predicted_Q_minus_actual_Q'] == d['normalized_cross'] + d['normalized_squared_error']
    assert np.isfinite(d['absolute_Q_error'])


def test_nonuniform_random_identity():
    rng = np.random.default_rng(31)
    for _ in range(30):
        d = m.decomposition(*(rng.normal(size=192).astype(np.float32) for _ in range(3)))
        assert abs(d['identity_residual']) <= d['identity_tolerance']


@pytest.mark.parametrize('bad', [np.zeros(191, np.float32), np.full(192, np.nan, np.float32),
                               np.full(192, .1, np.float64), np.zeros((1, 192), np.float32)])
def test_bad_vector_never_imputed(bad):
    with pytest.raises(ValueError):
        m.decomposition(bad, np.zeros(192, np.float32), np.zeros(192, np.float32))


@pytest.mark.parametrize('bad', ['missing', 'bytes', 'extra', 'report_sha', 'source_missing', 'array_sha'])
def test_missing_tampered_inputs_fail_closed(bad):
    files, r, path, h, saved = fixture()
    if bad == 'missing':
        del files[str(path.parent / 'label_00.npy')]
    elif bad == 'bytes':
        files[str(path.parent / 'prediction_00.npy')] += b'trailing'
    elif bad == 'extra':
        files[str(path.parent / 'extra')] = b'unknown'
    elif bad == 'report_sha':
        h = '0'*64
    elif bad == 'source_missing':
        del files['/RAM/live/e00_O_encode5/native192.npy']
    else:
        s = saved()
        with pytest.raises(ValueError, match='array SHA'):
            s.array(path.parent / 'label_00.npy', r['rows'][0]['label']['sha256'], [192], '0'*64)
        return
    with pytest.raises((ValueError, KeyError)):
        m.analyze(path, h, saved())


@pytest.mark.parametrize('bad', ['partial', 'metric', 'identity', 'subset', 'ledger'])
def test_semantic_corruption_rejected_even_with_rehashed_report(bad):
    files, r, path, h, saved = fixture()
    if bad == 'partial':
        r['status'] = 'partial_or_failed'
    elif bad == 'metric':
        r['rows'][0]['metrics']['factual_MSE'] += 1
    elif bad == 'identity':
        r['rows'][0]['identity']['episode_id'] = 'e01_O'
    elif bad == 'subset':
        r['rows'].pop()
    else:
        r['ledger'][0]['completed'] = False
    files[str(path)] = m.canonical(r)
    with pytest.raises(ValueError):
        m.analyze(path, m.sha(files[str(path)]), saved())


def test_pair_by_identity_not_order_and_exact_tie_is_not_reversal():
    files, r, path, h, saved = fixture()
    rows = m.analyze(path, h, saved())['rows']
    expected = m.contrasts(rows)
    assert m.contrasts(list(reversed(rows))) == expected
    rows[2]['endpoint'] = deepcopy(rows[0]['endpoint'])
    assert m.contrasts(rows)[0]['exact_tie'] and not m.contrasts(rows)[0]['strict_reversal']
    rows[2]['prefix_array_sha256'] = '0'*64
    with pytest.raises(ValueError, match='paired'):
        m.contrasts(rows)
    with pytest.raises(ValueError):
        m.contrasts(rows[:-1])


def test_strict_json_and_prepare_trailing_member():
    for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
        with pytest.raises(ValueError):
            m.parse(raw)
    files, r, _, _, _ = fixture()
    e = m.parse(files[r['prepared']])
    e['payload'] = base64.b64encode(base64.b64decode(e['payload']) + gzip.compress(b'bad')).decode()
    with pytest.raises(ValueError):
        m.decode_prepare(m.canonical(e))


def test_no_executor_or_neural_imports_and_exclusive_output(monkeypatch):
    source = Path(m.__file__).read_text()
    tree = ast.parse(source)
    modules = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    modules += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any(x and ('experiments' in x or x.startswith(('torch', 'mujoco'))) for x in modules)
    monkeypatch.setattr('sys.argv', ['analysis', '--output', m.__file__])
    monkeypatch.setattr(m, 'analyze', lambda *a: pytest.fail('existing path must fail before analysis'))
    with pytest.raises(ValueError, match='exclusive NEW'):
        m.main()


@pytest.mark.parametrize('kind', ['trailing', 'wrong_shape', 'wrong_dtype', 'nan'])
def test_array_parser_rejects_corrupt_npy_even_with_valid_file_sha(kind):
    a = np.zeros(192, np.float32)
    if kind == 'wrong_shape':
        a = a.reshape(1, 192)
    elif kind == 'wrong_dtype':
        a = a.astype(np.float64)
    elif kind == 'nan':
        a[0] = np.nan
    stream = BytesIO()
    np.save(stream, a, allow_pickle=False)
    raw = stream.getvalue() + (b'trailing' if kind == 'trailing' else b'')
    with pytest.raises(ValueError):
        m.Saved(reader=lambda path: raw).array('/RAM/array.npy', m.sha(raw), [192])


def test_terminal_file_changed_after_first_read_is_not_success():
    files, r, path, h, saved = fixture()
    s = saved()
    old_reader = s.reader
    calls = 0
    def changing_reader(p):
        nonlocal calls
        if p == str(path):
            calls += 1
            if calls > 1:
                return old_reader(p) + b' '
        return old_reader(p)
    s.reader = changing_reader
    with pytest.raises(ValueError, match='SHA mismatch'):
        m.analyze(path, h, s)
