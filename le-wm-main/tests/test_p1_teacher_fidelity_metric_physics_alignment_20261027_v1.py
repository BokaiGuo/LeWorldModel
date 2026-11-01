"""RAM-only arithmetic/identity fixtures. No model, simulator, or scientific inference."""
import ast
import base64
from copy import deepcopy
import gzip
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p1_teacher_fidelity_metric_physics_alignment_20261027_v1 as m


def fixture():
    d = dict(schema=m.dec.NAME+'.report', status='saved_array_diagnostic_not_admitted', independent_tasks=2, rows=[], arrays={})
    a = dict(schema=m.live.NAME+'.report', status='saved_descriptive_analysis_not_admitted', independent_tasks=2, rows=[], contrasts=[])
    report = dict(rows=[])
    prepared = dict(layers=[])
    tp = dict(schema=m.dec.TEACHER+'.prepare', evaluators=[])
    native_goal = np.zeros(7)
    latent_goal = np.zeros(192, np.float32)
    for li, (slot, arm) in enumerate(m.dec.LAYERS):
        layer = dict(source_slot=slot, arm=arm, task_id=str(slot)*64, selection_sha256=str(li)*64)
        prepared['layers'].append(layer)
        for mi, mode in enumerate(('O', 'I', 'R')):
            index = 3*li+mi
            ep = f'e{li:02}_{mode}'
            ident = dict(index=index, source_slot=slot, prefix_arm=arm, mode=mode, episode_id=ep,
                         parent_task_id=layer['task_id'], parent_selection_sha256=layer['selection_sha256'])
            r = dict(source_slot=slot, prefix_arm=arm, mode=mode, episode_id=ep, status='completed10',
                     full_horizon=True, first_error=None, secondary_errors=[], initial_exact=True,
                     prefix_pairing=dict(layer=li, prefix='same'), checkpoints=[])
            ar = dict(source_slot=slot, prefix_arm=arm, mode=mode, episode_id=ep, status='completed10', issues=[], endpoints={})
            ev = dict(identity=ident, goal=latent_goal.tolist())
            for step in (5, 10):
                state = np.array([3., 4., 1., 2., .1, 6., 8.])
                if step == 10:
                    state *= mi+1
                native = m.live.native_metrics(state, native_goal)
                cp = dict(phase='plan', step=step, state=state.tolist(), goal=native_goal.tolist())
                r['checkpoints'].append(cp)
                q = 2. if step == 5 else float(mi+1)
                ar['endpoints'][str(step)] = dict(checkpoint_present=True, observed_Q=q, native_recomputed=native)
                receipt = dict(path=f'/RAM/live/{ep}_encode{step}/native192.npy', array_sha256=('a' if step == 5 else 'b')*64,
                               file_sha256='c'*64, dtype='float32', shape=[192])
                r[f'encoding{step}'] = dict(latent=receipt)
                ev['prefix' if step == 5 else 'label'] = receipt
                if step == 10:
                    r['native_success'] = native['native_success']
            dr = dict(identity=ident, endpoint=dict(actual_Q=float(mi+1), predicted_Q=float(3-mi)),
                      goal_array_sha256=m.dec.sha(latent_goal.tobytes()), actual_receipt=dict(array_sha256='b'*64),
                      prefix_array_sha256='a'*64)
            d['rows'].append(dr)
            a['rows'].append(ar)
            report['rows'].append(r)
            tp['evaluators'].append(ev)
        for lhs, rhs in m.PAIRS:
            l, r = (next(x for x in a['rows'] if x['source_slot'] == slot and x['prefix_arm'] == arm and x['mode'] == mode)
                    for mode in (lhs, rhs))
            endpoints = {}
            for step in ('5', '10'):
                lv, rv = m.live.flatten(l['endpoints'][step]), m.live.flatten(r['endpoints'][step])
                endpoints[step] = {k: m.delta(lv[k], rv[k]) for k in lv}
            a['contrasts'].append(dict(source_slot=slot, prefix_arm=arm, contrast=lhs+'-'+rhs,
                                       prefix_pairing_exact=True, reason=None, endpoints=endpoints))
    return d, a, report, prepared, tp


def test_all_roots_all_modes_all_metrics_and_exact_deltas():
    rows, contrasts = m.make_alignment(*fixture())
    assert len(rows) == len(contrasts) == 12
    assert {(c['layer_index'], c['contrast']) for c in contrasts} == {(i, l+'-'+r) for i in range(4) for l, r in m.PAIRS}
    for c in contrasts:
        assert c['predicted_vs_actual_latent_Q']['strict_reversal'] is True
        for step in ('5', '10'):
            e = c['endpoints'][step]
            assert set(e['physics_metrics']) == set(m.DIRECTIONS)
            for metric in e['physics_metrics'].values():
                assert metric['delta_lhs_minus_rhs'] == m.delta(metric['lhs'], metric['rhs'])
            assert e['physics_metrics']['object_success']['actual_latent_Q_alignment']['strict_reversal'] is None
            if step == '5':
                assert e['observed_latent_Q_delta'] == 0
                assert e['physics_metrics']['native_distance7']['actual_latent_Q_alignment']['status'] == 'tie'
                assert e['physics_metrics']['native_success_receipt']['delta_lhs_minus_rhs'] is None


def test_full7_velocity_and_allxy_have_distinct_meaning():
    rows, _ = m.make_alignment(*fixture())
    e = rows[0]['endpoints']['5']
    assert e['metrics']['native_distance7'] == np.linalg.norm([3, 4, 1, 2, .1, 6, 8])
    assert e['metrics']['allxy_position_error4'] == np.linalg.norm([3, 4, 1, 2])
    assert e['metrics']['agent_position_error2'] == 5.
    assert len(e['state7']) == len(e['goal7']) == len(e['signed_native_residual7']) == 7


@pytest.mark.parametrize('q,p,direction,status,reversal', [
    (-1., -2., -1, 'same_direction', False), (-1., 1., 1, 'same_direction', False),
    (-1., -1., 1, 'opposite_direction', True), (1., -1., -1, 'opposite_direction', True),
    (0., 1., -1, 'tie', False), (1., 0., -1, 'tie', False), (0., 0., 1, 'tie', False),
    (None, 1., -1, 'unknown', None), (1., None, 1, 'unknown', None), (1., 2., None, 'unknown', None),
    (-1e-100, 1e-100, -1, 'opposite_direction', True)])
def test_direction_zero_tie_unknown_and_no_epsilon(q, p, direction, status, reversal):
    result = m.compare(q, p, direction)
    assert result['status'] == status and result['strict_reversal'] is reversal


def test_order_independent_identity_join():
    values = fixture()
    expected = m.make_alignment(*values)
    for doc in values[:3]:
        doc['rows'].reverse()
    values[3]['layers'].reverse()
    values[4]['evaluators'].reverse()
    assert m.make_alignment(*values) == expected


@pytest.mark.parametrize('bad', ['subset', 'duplicate', 'bool_slot', 'wrong_task', 'wrong_selection',
                               'wrong_receipt', 'wrong_q', 'unpaired', 'state5', 'missing_checkpoint', 'missing_q', 'native_flag'])
def test_missing_or_cross_episode_or_analysis_mismatch_is_fail_closed(bad):
    d, a, r, p, tp = fixture()
    if bad == 'subset':
        d['rows'].pop()
    elif bad == 'duplicate':
        a['rows'][1] = deepcopy(a['rows'][0])
    elif bad == 'bool_slot':
        d['rows'][0]['identity']['source_slot'] = False
    elif bad in ('wrong_task', 'wrong_selection'):
        p['layers'][0]['task_id' if bad == 'wrong_task' else 'selection_sha256'] = 'f'*64
    elif bad == 'wrong_receipt':
        tp['evaluators'][0]['label'] = deepcopy(tp['evaluators'][1]['label'])
    elif bad == 'wrong_q':
        d['rows'][0]['endpoint']['actual_Q'] += 1
    elif bad == 'unpaired':
        a['contrasts'][0]['prefix_pairing_exact'] = False
    elif bad == 'state5':
        r['rows'][0]['checkpoints'][0]['state'] = [0.]*5
    elif bad == 'missing_checkpoint':
        r['rows'][0]['checkpoints'].pop()
    elif bad == 'missing_q':
        a['rows'][0]['endpoints']['10']['observed_Q'] = None
    else:
        r['rows'][0]['native_success'] = not r['rows'][0]['native_success']
    with pytest.raises((ValueError, KeyError)):
        m.make_alignment(d, a, r, p, tp)


def test_unknown_native_success_stays_null():
    values = fixture()
    values[2]['rows'][0]['native_success'] = None
    rows, contrasts = m.make_alignment(*values)
    assert rows[0]['endpoints']['10']['metrics']['native_success_receipt'] is None
    assert contrasts[0]['endpoints']['10']['physics_metrics']['native_success_receipt']['delta_lhs_minus_rhs'] is None


def test_legacy_negative_angle_is_not_a_valid_improvement_direction():
    d, a, report, p, tp = fixture()
    row = report['rows'][0]
    cp = row['checkpoints'][1]
    cp['state'][4] = 8*np.pi
    native = m.live.native_metrics(cp['state'], cp['goal'])
    row['native_success'] = native['native_success']
    a['rows'][0]['endpoints']['10']['native_recomputed'] = native
    for c in a['contrasts']:
        if c['source_slot'] == 0 and c['prefix_arm'] == m.dec.ARMS[0]:
            lhs, rhs = c['contrast'].split('-')
            left, right = (next(r for r in a['rows'] if r['source_slot'] == 0 and r['prefix_arm'] == m.dec.ARMS[0]
                               and r['mode'] == mode) for mode in (lhs, rhs))
            lv, rv = (m.live.flatten(r['endpoints']['10']) for r in (left, right))
            c['endpoints']['10'] = {k: m.delta(lv[k], rv[k]) for k in lv}
    rows, contrasts = m.make_alignment(d, a, report, p, tp)
    assert rows[0]['endpoints']['10']['negative_native_angle'] is True
    metric = contrasts[0]['endpoints']['10']['physics_metrics']['native_angle_term']
    assert metric['delta_lhs_minus_rhs'] is not None
    assert metric['improvement_direction'] is None
    assert metric['actual_latent_Q_alignment']['strict_reversal'] is None


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), '0'])
def test_nonfinite_or_nonliteral_scalar_rejected(bad):
    with pytest.raises(ValueError):
        m.compare(1., bad, -1)


def pipeline(monkeypatch):
    d, a, report, prepared, tp = fixture()
    files = {}
    def put(path, value):
        raw = value if isinstance(value, bytes) else m.dec.canonical(value)
        files[str(path)] = raw
        return m.dec.sha(raw)
    report_path, prepared_path = '/RAM/live/report.json', '/RAM/live/prepare.json'
    ai = dict(report=report_path, report_sha256=put(report_path, report),
              prepared=prepared_path, prepare_sha256=put(prepared_path, prepared))
    a.update(inputs=ai, input_sha256={report_path: ai['report_sha256'], prepared_path: ai['prepare_sha256']},
             analysis_source_sha256={'/RAM/analysis_source': put('/RAM/analysis_source', b'original saved-only source')})
    ap = '/RAM/analysis.json'
    ah = put(ap, a)
    tp['arguments'] = dict(analysis_path=ap, analysis_sha=ah, report_path=report_path, report_sha=ai['report_sha256'],
                           live_prepared=prepared_path, live_prepare_sha=ai['prepare_sha256'])
    raw = m.dec.canonical(tp)
    envelope = dict(schema=m.dec.TEACHER+'.prepare_envelope', codec='single-gzip+base64', semantic_bytes=len(raw),
                    semantic_sha256=m.dec.sha(raw), payload=base64.b64encode(gzip.compress(raw, mtime=0)).decode())
    teacher_prepare = '/RAM/teacher_prepare.json'
    teacher = dict(prepared=teacher_prepare, prepare_sha256=put(teacher_prepare, envelope))
    d.update(input_report='/RAM/teacher_report.json', input_report_sha256=put('/RAM/teacher_report.json', teacher),
             implementation_sha256=put(m.dec.__file__, b'original saved-only decomposition source'))
    d['inputs'] = {p: dict(sha256=m.dec.sha(raw), bytes=len(raw)) for p, raw in files.items()}
    dp = '/RAM/decomposition.json'
    dh = put(dp, d)
    monkeypatch.setattr(m.dec, 'analyze', lambda *args: {k: v for k, v in d.items() if k != 'implementation_sha256'})
    monkeypatch.setattr(m.live, 'build', lambda *args: deepcopy(a))
    s = m.dec.Saved(reader=lambda path: files[path])
    return files, d, a, dp, dh, ap, ah, s


def test_saved_pipeline_binds_both_authorities_and_n_stays_two(monkeypatch):
    files, d, a, dp, dh, ap, ah, s = pipeline(monkeypatch)
    result = m.build(dp, dh, ap, ah, s)
    assert result['independent_tasks'] == 2 and not result['independent_samples']
    assert result['fixed_episode_denominator'] == 12 and result['significance'] is None
    assert len(result['contrasts']) == 12 and len(result['inputs']) == len(files)


@pytest.mark.parametrize('bad', ['missing', 'tamper', 'analysis_rebuild', 'decomposition_rebuild'])
def test_saved_pipeline_binding_failure_never_outputs_alignment(monkeypatch, bad):
    files, d, a, dp, dh, ap, ah, s = pipeline(monkeypatch)
    if bad == 'missing':
        del files[a['inputs']['report']]
    elif bad == 'tamper':
        files[ap] += b' '
    elif bad == 'analysis_rebuild':
        monkeypatch.setattr(m.live, 'build', lambda *args: {})
    else:
        monkeypatch.setattr(m.dec, 'analyze', lambda *args: {})
    with pytest.raises((ValueError, KeyError)):
        m.build(dp, dh, ap, ah, s)


def test_saved_only_imports_and_exclusive_output(monkeypatch):
    tree = ast.parse(Path(m.__file__).read_text())
    modules = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    modules += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any(x and x.startswith(('torch', 'mujoco', 'stable_worldmodel')) for x in modules)
    monkeypatch.setattr('sys.argv', ['alignment', '--output', m.__file__])
    monkeypatch.setattr(m, 'build', lambda: pytest.fail('no analysis on existing output'))
    with pytest.raises(ValueError, match='exclusive NEW'):
        m.main()
