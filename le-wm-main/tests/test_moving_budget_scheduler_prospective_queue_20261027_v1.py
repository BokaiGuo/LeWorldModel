"""All fixtures are synthetic NumPy/BytesIO data, never neural or physical."""
from copy import deepcopy
import hashlib
import io

import numpy as np
import pytest

from experiments.spectral_reachability import moving_budget_scheduler_prospective_queue_20261027_v1 as q


def h(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def manifest(mode='population'):
    panels = []
    for i in range(8 if mode == 'population' else 16):
        roots = [dict(root_sha256=h(('root', i, j)), task_id=f'task{i}_{j}',
                      blind_sha256=h(('blind', i, j)), assets={k: h(k) for k in
                      ('source', 'encoder', 'anchor', 'model', 'normalizer', 'q', 'candidate_bank_rule')})
                 for j in range(2)]
        panels.append(dict(panel_id=f'panel{i:02}', roots=roots))
    return dict(schema=q.STEM+'.manifest', mode=mode, panels=panels,
                excluded_root_sha256=[h('old0'), h('old1')],
                approval=dict(authority='MOCK_MAIN', no_signal_or_outcome_seen=True))


def fixture(mode='population', probabilities=None, failures=()):
    f = q.freeze(manifest(mode), now_ns=100)
    fh = q.sha(q.encode(f)); memory = {}

    def put(name, obj):
        b = io.BytesIO()
        if isinstance(obj, np.ndarray):
            np.save(b, obj, allow_pickle=False); data = b.getvalue()
        else:
            data = q.encode(obj)
        memory[name] = data
        return dict(file=name, sha256=q.sha(data))

    index = dict(schema=q.STEM+'.index', freeze_sha256=fh, panels=[])
    for i, panel in enumerate(f['manifest']['panels']):
        cert = dict(schema=q.STEM+'.freshness', freeze_sha256=fh, panel=panel,
                    authority='MOCK_MAIN', issued_at_ns=101, independent_roots=True,
                    not_used_for_design=True, no_outcome_or_candidate_cost_used=True)
        rows = []
        p = probabilities[i] if probabilities is not None else .5
        for j, identity in enumerate(panel['roots']):
            goal = np.zeros(192); goal[0] = np.sqrt(p if j == 0 else 1-p)
            arrays = dict(nominal=np.zeros(192), goal=goal, q=np.eye(192),
                          jacobian=np.eye(192, 20), u=np.eye(192, 20),
                          sigma=np.ones(20), vt=np.eye(20))
            rows.append(dict(identity=identity, **{k: put(f'{i}/{j}/{k}.npy', a)
                                                       for k, a in arrays.items()}))
        c = dict(attempted=dict(q.CAP), completed=dict(q.CAP))
        r = dict(schema=q.STEM+'.screen', freeze_sha256=fh, panel=panel,
                 status='screen_completed', started_at_ns=102, finished_at_ns=103,
                 counters=c, errors=[], roots=rows)
        if i in failures:
            r.update(status='screen_failed', errors=['MOCK_NUMERICAL_FAILURE'], roots=[])
            r['counters']['completed'] = {k: 0 for k in q.AXES}
        index['panels'].append(dict(panel_id=panel['panel_id'],
                                   freshness=put(f'{i}/fresh.json', cert), screen=put(f'{i}/screen.json', r)))
    return f, fh, index, memory


def run(data):
    f, fh, index, memory = data
    return q.assemble(f, fh, index, memory.__getitem__)


def change(data, i, kind, fn):
    ref = data[2]['panels'][i][kind]
    obj = q.parse(data[3][ref['file']]); fn(obj)
    data[3][ref['file']] = q.encode(obj)
    ref['sha256'] = q.sha(data[3][ref['file']])


@pytest.mark.parametrize('p,eligible', [(0, True), (.44, True), (.44001, False),
                                      (.45, False), (.5, False), (.55, False),
                                      (.55999, False), (.56, True), (1, True)])
def test_exact_threshold_and_integer_vectors(p, eligible):
    a = q.allocation([p, 1-p])
    assert a['eligible'] is eligible
    assert all(sum(v) == 20 and min(v) >= 5 and max(v) <= 15 for v in a['blocks'].values())
    if eligible:
        assert len({tuple(a['blocks'][k]) for k in ('D', 'S', 'U')}) == 3


def test_zero_weights_not_separated():
    a = q.allocation([0., 0.])
    assert a['p'] is None and not a['eligible'] and a['blocks']['D'] == [10, 10]


@pytest.mark.parametrize('weights', [[-1, 2], [float('nan'), 1], [float('inf'), 1], [1e308, 1e308]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        q.allocation(weights)


def test_population_keeps_all_even_uniform():
    r = run(fixture())
    assert r['status'] == 'saved_queue_verified_engineering_only'
    assert r['released_ids'] == [f'panel{i:02}' for i in range(8)]
    assert not any(x['eligible'] for x in r['rows'])
    assert r['planned_full_queue_cap']['predict'] == 4160
    assert r['screening_cost_lower_bound']['completed']['predict'] == 64
    assert r['actual_neural_calls'] == r['actual_physics_calls'] == 0
    assert not r['P1_admission'] and not r['P3_admission']


def test_separated_first_eight_not_largest_signal_or_index_order():
    data = fixture('signal_separated', [.5, .7, .5] + [.8]*13)
    data[2]['panels'].reverse()
    r = run(data)
    assert r['released_ids'] == ['panel01'] + [f'panel{i:02}' for i in range(3, 10)]
    assert r['denominator'] == 16 and len(r['rows']) == 16
    assert r['screening_cost_lower_bound']['completed']['predict'] == 128
    assert r['planned_full_queue_cap'] == dict(predict=4224, action_encoder=4224,
                                              jacobian=32, vjp_directions=6144, svd=32, solve=0)


def test_separated_no_replenishment_and_zero_is_terminal():
    r = run(fixture('signal_separated'))
    assert r['status'] == 'queue_empty_no_search' and r['released_ids'] == []
    assert r['denominator'] == 16 and r['planned_search_predict_upper_bound'] == 0
    r = run(fixture('signal_separated', [.7]+[.5]*15))
    assert r['released_ids'] == ['panel00']


def test_missing_actual_signals_is_blocked_not_mock_actual():
    data = fixture(); data[2]['panels'] = []
    r = run(data)
    assert r['status'] == 'queue_blocked' and len(r['blocked_ids']) == 8
    assert not r['released_ids'] and not r['screening_cost_complete']


@pytest.mark.parametrize('mode', q.MODES)
def test_missing_one_blocks_whole_queue_even_if_first_eight_eligible(mode):
    n = 8 if mode == 'population' else 16
    data = fixture(mode, [.7]*n); data[2]['panels'].pop()
    r = run(data)
    assert r['status'] == 'queue_blocked' and not r['released_ids']
    assert r['rows'][-1]['status'] == 'blocked'


@pytest.mark.parametrize('mode', q.MODES)
def test_terminal_failure_denominator_and_paid_attempts(mode):
    data = fixture(mode, [.7]*(8 if mode == 'population' else 16), failures=(0,))
    r = run(data)
    assert r['failed_ids'] == ['panel00'] and len(r['rows']) == r['denominator']
    assert r['screening_cost_lower_bound']['attempted']['predict'] == 8*r['denominator']
    if mode == 'population':
        assert r['status'] == 'queue_blocked' and not r['released_ids']
    else:
        assert r['released_ids'] == [f'panel{i:02}' for i in range(1, 9)]


@pytest.mark.parametrize('field', ['independent_roots', 'not_used_for_design', 'no_outcome_or_candidate_cost_used'])
def test_freshness_required(field):
    data = fixture()
    change(data, 0, 'freshness', lambda c: c.update({field: False}))
    assert run(data)['blocked_ids'] == ['panel00']


def test_pre_signal_chronology_and_binding():
    data = fixture()
    change(data, 0, 'screen', lambda c: c.update(started_at_ns=99))
    assert run(data)['blocked_ids'] == ['panel00']
    data = fixture()
    change(data, 0, 'freshness', lambda c: c.update(freeze_sha256='0'*64))
    assert run(data)['blocked_ids'] == ['panel00']


def test_root_history_asset_identity_not_array_position_guess():
    data = fixture()
    change(data, 0, 'screen', lambda c: c['roots'].reverse())
    assert run(data)['blocked_ids'] == ['panel00']


@pytest.mark.parametrize('field', ['outcome', 'candidate_cost', 'physics', 'winner'])
def test_forbidden_fields_rejected_not_ignored(field):
    data = fixture()
    change(data, 0, 'screen', lambda c: c.update({field: 0}))
    assert run(data)['blocked_ids'] == ['panel00']


def test_missing_array_and_corruption_block():
    data = fixture(); del data[3]['0/0/jacobian.npy']
    assert run(data)['blocked_ids'] == ['panel00']
    data = fixture(); data[3]['0/0/jacobian.npy'] += b'corruption'
    assert run(data)['blocked_ids'] == ['panel00']


def test_svd_numeric_mismatch_even_with_resealed_file():
    data = fixture()
    a = np.eye(192, 20)*2; b = io.BytesIO(); np.save(b, a, allow_pickle=False)
    data[3]['0/0/jacobian.npy'] = b.getvalue()
    change(data, 0, 'screen', lambda c: c['roots'][0]['jacobian'].update(sha256=q.sha(b.getvalue())))
    assert run(data)['blocked_ids'] == ['panel00']


def test_attempt_completed_not_fabricated():
    data = fixture()
    change(data, 0, 'screen', lambda c: c['counters']['completed'].update(predict=7))
    assert run(data)['blocked_ids'] == ['panel00']


@pytest.mark.parametrize('mutation', ['mode', 'denominator', 'duplicate', 'reuse', 'outcome'])
def test_prereg_manifest_rejections(mutation):
    m = manifest()
    if mutation == 'mode': m['mode'] = 'choose_best'
    if mutation == 'denominator': m['panels'].pop()
    if mutation == 'duplicate': m['panels'][1]['roots'][0] = deepcopy(m['panels'][0]['roots'][0])
    if mutation == 'reuse': m['excluded_root_sha256'][0] = m['panels'][0]['roots'][0]['root_sha256']
    if mutation == 'outcome': m['outcome'] = 1
    with pytest.raises(ValueError): q.freeze(m)


def test_changed_frozen_order_source_or_mode_rejected():
    data = fixture(); data[0]['manifest']['panels'].reverse()
    with pytest.raises(ValueError): run(data)
    data = fixture(); data[0]['source_sha256'] = '0'*64
    with pytest.raises(ValueError): run(data)


def test_duplicate_and_unknown_panels_rejected():
    data = fixture(); data[2]['panels'].append(deepcopy(data[2]['panels'][0]))
    with pytest.raises(ValueError): run(data)
    data = fixture(); data[2]['panels'][0]['panel_id'] = 'unregistered'
    with pytest.raises(ValueError): run(data)


def test_duplicate_json_nonfinite_rejected():
    with pytest.raises(ValueError): q.parse(b'{"a":1,"a":2}')
    with pytest.raises(ValueError): q.parse(b'{"a":NaN}')


def test_paths_no_traversal():
    load = q.file_loader('.')
    for name in ('../secret', '/etc/passwd'):
        with pytest.raises(ValueError): load(name)


def test_cli_help_and_freeze_consume_end_to_end_in_ram(monkeypatch, capsys):
    data = fixture()
    virtual = dict(data[3]); saved = {}
    manifest_bytes = q.encode(data[0]['manifest']); virtual['manifest.json'] = manifest_bytes
    monkeypatch.setattr(q.Path, 'read_bytes', lambda p: virtual[str(p)] if str(p) in virtual else original(p))
    monkeypatch.setattr(q.time, 'time_ns', lambda: 100)
    monkeypatch.setattr(q, 'file_loader', lambda _: virtual.__getitem__)
    monkeypatch.setattr(q, 'publish', lambda name, value: saved.setdefault(name, q.encode(value)) and q.sha(saved[name]))
    assert q.main(['freeze', '--manifest', 'manifest.json', '--manifest-sha', q.sha(manifest_bytes),
                   '--output', 'freeze.json']) == 0
    virtual['freeze.json'] = saved['freeze.json']
    assert virtual['freeze.json'] == q.encode(data[0])
    virtual['index.json'] = q.encode(data[2])
    args = ['consume', '--freeze', 'freeze.json', '--freeze-sha', data[1], '--index', 'index.json',
            '--index-sha', q.sha(virtual['index.json']), '--input-root', 'RAM', '--output', 'report.json']
    assert q.main(args) == 0
    assert q.parse(saved['report.json'])['denominator'] == 8
    virtual['index.json'] = q.encode(dict(data[2], panels=[]))
    args[args.index('--index-sha')+1] = q.sha(virtual['index.json'])
    args[args.index('--output')+1] = 'blocked.json'
    assert q.main(args) == 2
    assert q.parse(saved['blocked.json'])['status'] == 'queue_blocked'
    with pytest.raises(SystemExit) as e: q.main(['--help'])
    assert e.value.code == 0


original = q.Path.read_bytes


def test_publish_floor_cap_and_exclusive_without_disk(monkeypatch):
    from types import SimpleNamespace
    opened = []
    monkeypatch.setattr(q.shutil, 'disk_usage', lambda _: SimpleNamespace(free=q.FLOOR+q.OUTPUT_CAP-1))
    monkeypatch.setattr(q.Path, 'open', lambda *a, **kw: opened.append(a) or io.BytesIO())
    with pytest.raises(ValueError, match='disk floor'): q.publish('never.json', {})
    assert not opened
    monkeypatch.setattr(q.shutil, 'disk_usage', lambda _: SimpleNamespace(free=q.FLOOR+q.OUTPUT_CAP))
    with pytest.raises(ValueError, match='output cap'): q.publish('never.json', {'a': 'x'*q.OUTPUT_CAP})
    assert not opened
    def exists(path, mode):
        assert mode == 'xb'
        raise FileExistsError('preserve existing output')
    monkeypatch.setattr(q.Path, 'open', exists)
    with pytest.raises(FileExistsError): q.publish('existing.json', {})


@pytest.mark.parametrize('kind', ['dtype', 'shape', 'q', 'sigma', 'nonfinite'])
def test_invalid_saved_arrays(kind):
    data = fixture()
    role = 'sigma' if kind == 'sigma' else 'q' if kind == 'q' else 'nominal'
    a = np.ones(192)
    if kind == 'dtype': a = a.astype(np.float32)
    if kind == 'shape': a = a[:191]
    if kind == 'q': a = np.eye(192)*2
    if kind == 'sigma': a = -np.ones(20)
    if kind == 'nonfinite': a[0] = np.nan
    b = io.BytesIO(); np.save(b, a, allow_pickle=False)
    name = f'0/0/{role}.npy'; data[3][name] = b.getvalue()
    change(data, 0, 'screen', lambda c: c['roots'][0][role].update(sha256=q.sha(b.getvalue())))
    assert run(data)['blocked_ids'] == ['panel00']


def test_residual_outside_column_space_is_not_weight():
    data = fixture()
    a = np.zeros(192); a[50] = 3.
    b = io.BytesIO(); np.save(b, a, allow_pickle=False)
    data[3]['0/0/goal.npy'] = b.getvalue()
    change(data, 0, 'screen', lambda c: c['roots'][0]['goal'].update(sha256=q.sha(b.getvalue())))
    r = run(data)
    sig = r['rows'][0]['signals'][0]
    assert sig['energy'] == 9 and sig['chi'] == 1 and sig['w'] == 0
    assert r['rows'][0]['blocks']['D'] == [5, 15]
