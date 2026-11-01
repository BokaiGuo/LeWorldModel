"""Pure arrays + actual saved evidence; no tmp_path, neural or physical jobs."""
import ast
from copy import deepcopy
import io as memory_io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p1_sampling_rgb_cuda_20261027_v1 as m


def fixtures():
    q = np.zeros((192, 192))
    q[0, 0] = q[1, 1] = 1
    task = dict(environment='mock/contract-only', task_contract_sha256='a'*64,
        root_context=dict(layout='states_with_between_state_actions_v1', context_length=3, latent_dim=192,
            action_dim=10, latent_history=np.zeros((3, 192)).tolist(), action_history=np.zeros((2, 10)).tolist()),
        root_raw_action_history=np.zeros((10, 2)).tolist(), goal=np.zeros(192).tolist(), projection=q.tolist(),
        model_sha256='b'*64, model_config_sha256='c'*64, metric_source_sha256='d'*64,
        goal_source_sha256='e'*64, representation='native_projected_cls', latent_dim=192,
        horizon=2, packet_steps=5, control_dim=2, action_mean=[0., 0.], action_std=[1., 1.],
        packet_logit_cap=1., feedback='open_loop_model_only')
    tasks = {h*64: deepcopy(task) for h in ('1', '2')}
    frames = []
    pred = np.zeros(192)
    pred[0], pred[3] = 2, 4
    for root in tasks:
        for arm in m.ARMS:
            for step in m.STEPS:
                frames.append(dict(order=len(frames), root_sha256=root, arm=arm, step=step,
                    task_id=m.core.task_id(tasks[root]), predicted_native192=pred.tolist(),
                    pixel_sha256='f'*64, physical_diagnostic={'fixture_not_evidence': True}))
    values = np.zeros((12, 192))
    values[:, 0], values[:, 3] = 1, 6
    return frames, values, tasks


def test_actual_saved_complete_collect():
    protocol = m.collect()
    assert len(protocol['frames']) == 12 and len(protocol['tasks']) == 2
    assert len({f['path'] for f in protocol['frames']}) == 12
    assert protocol['encoder_cap'] == protocol['projector_cap'] == 12
    assert protocol['predict_cap'] == protocol['AD_cap'] == protocol['physical_steps_cap'] == 0
    assert len(m.canonical(protocol)) < m.CAP - m.FINAL_RESERVE
    assert protocol['new_Q'] is False and protocol['P1_admission'] is False
    assert all('q' not in f and 'goal' not in f for f in protocol['frames'])


def test_same_original_Q_not_identity_replacement():
    frames, values, tasks = fixtures()
    result = m.analyze(frames, values, tasks)
    for row in result['rows']:
        assert row['predicted_Q_mse'] == 4/192
        assert row['actual_Q_mse'] == 1/192
        assert row['native_prediction_mse'] == 5/192
        assert row['physical_diagnostic'] == {'fixture_not_evidence': True}
    assert result['new_Q'] is False and result['P1_admission'] is False


def test_repeated_pixels_are_all_retained_and_charged():
    frames, values, tasks = fixtures()
    assert len({f['pixel_sha256'] for f in frames}) == 1
    assert len(m.analyze(frames, values, tasks)['rows']) == 12


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'reorder', 'task', 'root', 'step', 'order'])
def test_frame_contract_mutations(change):
    frames, values, tasks = fixtures()
    if change == 'missing':
        frames.pop()
    elif change == 'duplicate':
        frames[1] = deepcopy(frames[0])
    elif change == 'reorder':
        frames[0], frames[1] = frames[1], frames[0]
    elif change == 'task':
        frames[0]['task_id'] = '0'*64
    elif change == 'root':
        frames[0]['root_sha256'] = '0'*64
    elif change == 'step':
        frames[0]['step'] = 1
    else:
        frames[0]['order'] = 12
    with pytest.raises(ValueError):
        m.analyze(frames, values, tasks)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_latent_rejected(bad):
    frames, values, tasks = fixtures()
    values[0, 0] = bad
    with pytest.raises(ValueError):
        m.analyze(frames, values, tasks)


@pytest.mark.parametrize('shape', [(11, 192), (12, 191), (12, 193)])
def test_latent_shape_rejected(shape):
    frames, _, tasks = fixtures()
    with pytest.raises(ValueError):
        m.analyze(frames, np.zeros(shape), tasks)


def actual_physical():
    r = m.physical.read(m.RUN/'report.json')
    p = m.physical.read(m.PREPARED/'prepare.json')
    events = m.source.read_journal(m.RUN/'journal.jsonl')
    return r, p, events


@pytest.mark.parametrize('field,value', [
    ('status', 'partial_or_failed'), ('prepare_sha256', '0'*64), ('neural_calls', 1), ('P1_admission', True)])
def test_physical_report_scope_rejected(field, value):
    r, p, events = actual_physical()
    r[field] = value
    with pytest.raises(ValueError):
        m.validate_physical(r, p, events)


def test_missing_physical_journal_operation_rejected():
    r, p, events = actual_physical()
    del events[next(i for i, e in enumerate(events) if e['kind'] == 'render_returned')]
    with pytest.raises(ValueError):
        m.validate_physical(r, p, events)


def test_physical_row_not_durable_receipt_rejected():
    r, p, events = actual_physical()
    r['rows'][0]['checkpoints'][0]['rgb_sha256'] = '0'*64
    with pytest.raises(ValueError):
        m.validate_physical(r, p, events)


def test_conflicting_binding_rejected():
    with pytest.raises(ValueError):
        m.merge_bindings({'file': 'a'*64}, {'file': 'b'*64})


def test_complete_native_accounting():
    events = []
    for _ in range(12):
        events += [dict(kind='native_image_attempted', resource=k, samples=1) for k in ('encoder', 'projector')]
        events += [dict(kind='native_image_completed', counts={'encoder': 1, 'projector': 1})]
    assert m.counts(events) == {k: dict(attempted=12, completed=12) for k in ('encoder', 'projector')}


def test_native_attempt_is_not_completion():
    assert m.counts([dict(kind='native_image_attempted', resource='encoder', samples=1)]) == {
        'encoder': dict(attempted=1, completed=0), 'projector': dict(attempted=0, completed=0)}


def test_store_start_floor_refuses_before_mkdir(monkeypatch):
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda _: SimpleNamespace(free=m.FLOOR+m.CAP-1))
    with pytest.raises(ValueError, match='reserve'):
        m.Store(m.ROOT/'results/p1_sampling_rgb_cuda_20261027_v1_never_created_mock')


def test_prepare_floor_refuses_before_write(monkeypatch):
    monkeypatch.setattr(m, 'collect', lambda: {'small': 'fixture'})
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda _: SimpleNamespace(free=m.FLOOR))
    with pytest.raises(ValueError, match='floor'):
        m.prepare(m.ROOT/'results/p1_sampling_rgb_cuda_20261027_v1_never_created_mock.json')


def test_oversize_manifest_refused_without_write(monkeypatch):
    monkeypatch.setattr(m, 'collect', lambda: {'oversize': 'x'*m.CAP})
    with pytest.raises(ValueError, match='below2MiB'):
        m.prepare(m.ROOT/'results/p1_sampling_rgb_cuda_20261027_v1_never_created_mock.json')


def test_wrong_prepare_sha_refused(monkeypatch):
    monkeypatch.setattr(m.io, 'regular', lambda *a: None)
    monkeypatch.setattr(m.io, 'sha', lambda _: 'a'*64)
    with pytest.raises(ValueError, match='external'):
        m.verify('fixture-not-opened', 'b'*64)


def test_no_top_level_neural_imports_or_call_dedup():
    tree = ast.parse(Path(m.__file__).read_text())
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [n.module or '' for n in imports if isinstance(n, ast.ImportFrom)] + [
        a.name for n in imports if isinstance(n, ast.Import) for a in n.names]
    assert not any(n.split('.')[0] in ('torch', 'gymnasium', 'stable_worldmodel', 'stable_pretraining') for n in names)
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    assert sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'encode'
               for n in ast.walk(run)) == 1
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ('unique', 'set')
                   for n in ast.walk(run))


def test_import_graph_no_neural_loaded():
    # Executed in this pure-array test process, no subprocess/temp products.
    import sys
    assert not any(n in sys.modules for n in ('torch', 'gymnasium', 'pymunk', 'stable_worldmodel'))
