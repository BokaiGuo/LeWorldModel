"""RAM-only corruption fixtures and one real saved B8 audit. No model imports."""
from copy import deepcopy
import gzip
import io
import json
import sys

import numpy as np
import pytest
from PIL import Image

from experiments.spectral_reachability import p1_b8_rgb_saved_audit_20261027_v1 as m


def frames():
    return [dict(order=i, root_sha256=root, arm=arm, step=step, task_id='task_'+root,
                 pixel_sha256='repeated_pixels')
        for i, (root, arm, step) in enumerate((r, a, s) for r in ('root0', 'root1') for a in m.ARMS for s in m.STEPS)]


@pytest.mark.parametrize('omitted', range(12))
def test_no_slot_can_disappear_from_denominator(omitted):
    value = frames()
    value.pop(omitted)
    with pytest.raises(ValueError, match='all12'):
        m.census(value, ['root0', 'root1'])


@pytest.mark.parametrize('damage', ('swap', 'duplicate', 'step', 'arm', 'root', 'order', 'dedup'))
def test_exact_root_arm_step_census_not_success_subset(damage):
    value = frames()
    if damage == 'swap':
        value[0], value[1] = value[1], value[0]
    elif damage == 'duplicate':
        value[1] = deepcopy(value[0])
    elif damage == 'dedup':
        value = value[:1]
    else:
        value[0][dict(step='step', arm='arm', root='root_sha256', order='order')[damage]] = 'bad'
    with pytest.raises(ValueError, match='all12'):
        m.census(value, ['root0', 'root1'])


def test_duplicate_pixels_never_reduce_valid12_denominator():
    value = frames()
    m.census(value, ['root0', 'root1'])
    assert len({f['pixel_sha256'] for f in value}) == 1 and len(value) == 12


def test_same_original_Q_goal_and_mean_over192_not_rank():
    q = np.zeros((192, 192)); q[0, 0] = 1
    goal = np.zeros(192); goal[0] = 1
    actual = np.zeros(192); actual[0] = 3; actual[1] = 100
    predicted = actual.copy(); predicted[0] = 4
    task = dict(projection=q.tolist(), goal=goal.tolist())
    before = deepcopy(task)
    out = m.metrics(task, actual, predicted)
    assert out['actual_Q_mse'] == 4/192
    assert out['predicted_Q_mse'] == 9/192
    assert out['native_prediction_mse'] == 1/192
    assert task == before
    wrong = deepcopy(task); wrong['projection'] = np.eye(192).tolist()
    assert m.metrics(wrong, actual, predicted)['actual_Q_mse'] != out['actual_Q_mse']


def raw_npy(value):
    stream = io.BytesIO(); np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def test_original_native_float32_npy_roundtrip_without_cast():
    value = np.arange(192, dtype=np.float32)
    assert np.array_equal(m.native_npy(raw_npy(value)), value)


@pytest.mark.parametrize('damage', ('float64', 'int', 'shape', 'nan', 'inf', 'trailing', 'truncated', 'large'))
def test_raw_native_corruption_rejected(damage):
    value = np.zeros(192, np.float32)
    if damage == 'float64': value = value.astype(np.float64)
    if damage == 'int': value = value.astype(np.int32)
    if damage == 'shape': value = value.reshape(1, 192)
    if damage == 'nan': value[0] = np.nan
    if damage == 'inf': value[0] = np.inf
    raw = raw_npy(value)
    if damage == 'trailing': raw += b'not_another_latent'
    if damage == 'truncated': raw = raw[:-1]
    if damage == 'large': raw += b'0'*4096
    with pytest.raises(ValueError):
        m.native_npy(raw)


def png_bytes(rgb):
    stream = io.BytesIO(); Image.fromarray(rgb).save(stream, format='PNG')
    return stream.getvalue()


def test_png_pixel_digest_and_canonical_guard_recount():
    rgb = np.zeros((224, 224, 3), np.uint8); rgb[:2, :3] = [65, 105, 225]
    assert m.png(png_bytes(rgb), m.sha(rgb.tobytes())) == 6
    with pytest.raises(ValueError, match='pixel SHA'):
        m.png(png_bytes(rgb), '0'*64)


@pytest.mark.parametrize('shape', ((223, 224, 3), (224, 224, 4)))
def test_noncanonical_png_rejected(shape):
    rgb = np.zeros(shape, np.uint8)
    with pytest.raises(ValueError, match='RGB224'):
        m.png(png_bytes(rgb), m.sha(rgb.tobytes()))


def image_fixture():
    fs = frames(); rows, entries, events = [], [], []
    for i, f in enumerate(fs):
        row = dict(latent_file=f'frame_{i:02}.native192.npy', latent_sha256='same_latent_is_not_free')
        rows.append(row)
        common = dict(frame_order=i, root_sha256=f['root_sha256'], arm=f['arm'], step=f['step'], pixel_sha256=f['pixel_sha256'])
        events.append(dict(common, kind='frame_started', task_id=f['task_id'], frame_sha256=m.digest(f)))
        for j, resource in enumerate(('encoder', 'projector')):
            pending = dict(index=2*i+j, frame_order=i, resource=resource, completed=False)
            done = dict(pending, completed=True)
            events.extend([dict(common, kind='module_attempted', entry=pending), dict(common, kind='module_completed', entry=done)])
            entries.append(done)
        events.append(dict(common, kind='native_saved', **row))
    return [events, fs, rows, entries, {k:dict(attempted=12, completed=12) for k in ('encoder', 'projector')}]


def test_all12_repeated_pixels_still_require24_calls_and72_events():
    value = image_fixture()
    m.check_image_events(*value)
    assert len(value[0]) == 72 and len(value[3]) == 24


@pytest.mark.parametrize('damage', ('drop_event', 'extra_event', 'swap_event', 'resource', 'frame', 'pending', 'entry', 'dedup', 'count'))
def test_resource_journal_corruption_rejected(damage):
    value = image_fixture(); events, fs, rows, entries, resources = value
    if damage == 'drop_event': events.pop()
    elif damage == 'extra_event': events.append(deepcopy(events[-1]))
    elif damage == 'swap_event': events[1], events[2] = events[2], events[1]
    elif damage == 'resource': events[3]['entry']['resource'] = 'encoder'
    elif damage == 'frame': events[6]['frame_order'] = 0
    elif damage == 'pending': events[2]['entry']['completed'] = False
    elif damage == 'entry': entries.pop()
    elif damage == 'dedup': rows.pop()
    else: resources['encoder']['attempted'] = 10
    with pytest.raises(ValueError):
        m.check_image_events(*value)


@pytest.mark.parametrize('raw', (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'))
def test_noncanonical_json_rejected(raw):
    with pytest.raises(ValueError):
        m.decode(raw)


@pytest.mark.parametrize('damage', ('truncated', 'trailing', 'multistream', 'limit'))
def test_partial_or_concatenated_gzip_rejected(damage):
    raw = gzip.compress(b'{"x":1}', mtime=0)
    if damage == 'truncated': raw = raw[:-1]
    if damage == 'trailing': raw += b'trailer'
    if damage == 'multistream': raw += raw
    with pytest.raises(ValueError):
        m.decompress(raw, limit=2 if damage == 'limit' else 100)


def test_complete_gzip_and_physical_hash_chain():
    assert m.decompress(gzip.compress(b'{"x":1}', mtime=0)) == {'x':1}
    event = dict(index=0, previous_sha256=None, kind='start')
    event['event_sha256'] = m.digest(event)
    raw = json.dumps(event).encode()+b'\n'
    assert m.check_chain(raw) == [event]
    with pytest.raises(ValueError, match='torn'):
        m.check_chain(raw[:-1])
    event['index'] = 1
    with pytest.raises(ValueError, match='hash chain'):
        m.check_chain(json.dumps(event).encode()+b'\n')


@pytest.fixture(scope='module')
def actual():
    return m.audit()


def test_actual_saved_all12_raw_png_original_task_Q_and_resources(actual):
    assert actual['status'] == 'all12_saved_evidence_recomputed'
    assert actual['denominator'] == dict(original_roots=2, branches=4, frames=12, unique_png_paths=12,
        unique_pixel_hashes=10, unique_raw_latent_hashes=10, counted_without_dedup=True)
    assert len(actual['rows']) == 12 and len(actual['branch_summaries']) == 4
    assert actual['bindings_checked'] == len(actual['bindings']) == 144
    assert actual['image_event_count'] == 72
    assert all(x == dict(attempted=12, completed=12) for x in actual['resources_reconciled'].values())
    assert actual['rgb_run_file_bytes'] <= actual['rgb_run_cap_bytes'] == 2*1024**2
    assert all(actual[k] == 0 for k in ('new_neural_calls', 'new_AD_calls', 'new_physical_steps', 'new_independent_samples'))
    assert actual['P1_admission'] is actual['independent_confirmation'] is actual['new_Q'] is False
    assert not any(name in sys.modules for name in ('torch', 'gymnasium', 'pymunk'))


@pytest.mark.parametrize('damage', ('drop', 'shuffle', 'wrong_Q_cost', 'physical', 'admission', 'nan'))
def test_saved_analysis_mutations_fail_independent_recomputation(actual, damage):
    a = dict(schema='p1_b8_rgb_cuda_20261027_v1.analysis', frames=12, rows=deepcopy(actual['rows']),
        P1_admission=False, training=False, new_Q=False, independent_confirmation=False)
    if damage == 'drop': a['rows'].pop()
    elif damage == 'shuffle': a['rows'][0], a['rows'][1] = a['rows'][1], a['rows'][0]
    elif damage == 'wrong_Q_cost': a['rows'][1]['actual_Q_mse'] += 0.01
    elif damage == 'physical': a['rows'][0]['physical_diagnostic'] = {}
    elif damage == 'admission': a['P1_admission'] = True
    else: a['rows'][0]['predicted_Q_mse'] = float('nan')
    with pytest.raises(ValueError):
        m.check_analysis(a, actual['rows'])


def test_actual_second_root_GN_discrepancy_is_not_a_stationarity_or_theory_verdict(actual):
    s = actual['branch_summaries'][2]
    assert s['arm'] == 'full_H2_GN'
    assert s['predicted_endpoint_minus_start'] < 0 < s['actual_endpoint_minus_start']
    assert actual['P1_admission'] is False


def test_report_hash_drift_fails_closed_without_writing(monkeypatch):
    monkeypatch.setattr(m, 'REPORT_SHA', '0'*64)
    with pytest.raises(ValueError, match='file SHA drift'):
        m.audit()
