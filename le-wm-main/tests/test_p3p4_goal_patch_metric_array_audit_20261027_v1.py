"""Nonneural analytical fixtures and corruption tests, never model/physics calls."""
from copy import deepcopy
import json
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import p3p4_goal_patch_metric_array_audit_20261027_v1 as a


def fixture():
    eye = np.eye(192)
    legacy = np.vstack([np.zeros(192), eye[:4]])
    vectors = np.vstack([np.zeros(192), eye[4], *(eye[4]+eye[k] for k in range(5, 9)),
                         eye[4]+eye[9], eye[4]+eye[10]])
    image = np.full((224, 224, 3), 255, np.uint8)
    image[50:54, 50:54] = [65, 105, 225]
    image[90:100, 90:94] = [119, 136, 153]
    catalog = a.goal_catalog(image)
    oldq = np.eye(192)
    oldq[:4, :4] = 0
    q = np.zeros((192, 192))
    q[9, 9] = q[10, 10] = 1
    packet = dict(root_sha256='a'*64, root_context={'identity': 'fixed_initial_context'},
                  goal=np.zeros(192).tolist(), projection=oldq.tolist())
    packet['input_sha256'] = a.digest(packet)
    candidates = []
    for h, endpoint in ((2, 2*eye[11]), (3, eye[9])):
        latents = np.zeros((3, 192))
        latents[-1] = endpoint
        context = dict(layout='states_with_between_state_actions_v1', context_length=3,
                       latent_dim=192, action_dim=10, latent_history=latents.tolist(),
                       action_history=np.zeros((2, 10)).tolist())
        candidates.append(dict(horizon=h, index=0, completed=True, kind='fixture',
                               actions=np.full((5*h, 2), .01*h).tolist(), context=context,
                               cost=float(endpoint@endpoint/192)))
    sources = [dict(root_sha256='a'*64, mode=mode, input_sha256=packet['input_sha256'],
                    status='completed', search=dict(status='completed', candidates=deepcopy(candidates)))
               for mode in ('uniform', 'dynamic')]
    rows = []
    for c in candidates:
        endpoint = np.array(c['context']['latent_history'])[-1]
        rows.append({k: deepcopy(c[k]) for k in ('horizon', 'index', 'actions', 'context')})
        rows[-1].update(old_Q_cost=c['cost'], goal_patch_Q_cost=float(np.sum((q@endpoint)**2)/192))
    selection = dict(candidates=rows, winners=dict(old_Q=deepcopy(rows[1]), goal_patch_Q=deepcopy(rows[0])),
                     common_candidates=2, cost_evaluations=4, key_comparisons=2, new_predict_calls=0,
                     physical_ready=False, independent_selected_double_replay_required=True)
    rec = a.reconstruct(vectors, legacy)
    metric = dict(projection=q.tolist(), nuisance_rank=9, task_rank=2,
                  nuisance_singular_values=rec['nuisance_singular'].tolist(),
                  object_singular_values=rec['object_singular'].tolist(), goal_anchor_max_difference=0.,
                  physical_labels_used=False, learned_parameters=0)
    goal = dict(root_index=0, root_sha256='a'*64, catalog=catalog, legacy_goal_vectors=legacy.tolist())
    row = dict(root_index=0, root_sha256='a'*64, catalog=deepcopy(catalog), goal_vectors=vectors.tolist(),
               actual_encoder_samples=len(vectors), metric=metric, selection=selection, physical_ready=False)
    legacy_row = dict(root_index=0, root_sha256='a'*64, compiled=dict(goal_vectors=legacy.tolist()))
    return row, packet, goal, legacy_row, sources, image


def test_analytical_projectors_and_ranks():
    row, packet, goal, legacy, source, image = fixture()
    audited, r = a.root_audit(row, packet, goal, legacy, source, image)
    assert audited['all_checks']
    assert (audited['old_rank'], audited['nuisance_rank'], audited['task_rank']) == (4, 9, 2)
    np.testing.assert_allclose(r['old_Q'], packet['projection'], atol=1e-14)
    np.testing.assert_allclose(r['new_Q'], row['metric']['projection'], atol=1e-14)
    assert audited['winners']['old_Q']['horizon'] == 3
    assert audited['winners']['goal_patch_Q']['horizon'] == 2


def test_roundoff_contained_object_refuses():
    row, _, goal, *_ = fixture()
    v = np.array(row['goal_vectors'])
    v[6:] = v[2:4]
    with pytest.raises(ValueError, match='zero task rank'):
        a.reconstruct(v, goal['legacy_goal_vectors'])


def test_array_svd_attempts_survive_rank_refusal():
    row, _, goal, *_ = fixture()
    v = np.array(row['goal_vectors'])
    v[6:] = v[2:4]
    events = []
    with pytest.raises(ValueError, match='zero task rank'):
        a.reconstruct(v, goal['legacy_goal_vectors'], lambda role, done: events.append((role, done)))
    assert events == [(role, done) for role in ('legacy', 'nuisance', 'object') for done in (False, True)]


def test_goal_only_mask_exact_grid_order_and_hash():
    *_, image = fixture()
    catalog = a.goal_catalog(image)
    assert [c['id'] for c in catalog['cells']] == ['object_cell_6_6', 'object_cell_7_6']
    assert catalog['object_pixels'] == 40 and catalog['expected_encoder_samples'] == 8
    changed = image.copy()
    changed[100, 100] = [119, 136, 153]
    assert a.goal_catalog(changed)['mask_sha256'] != catalog['mask_sha256']


@pytest.mark.parametrize('mode', ['missing', 'clipped', 'nonwhite', 'corner'])
def test_invalid_goal_rgb_failclosed(mode):
    *_, image = fixture()
    if mode == 'missing':
        image[90:100, 90:94] = 255
    elif mode == 'clipped':
        image[0, 50] = [65, 105, 225]
    elif mode == 'nonwhite':
        image[48, 48] = 0
    else:
        image[28, 28] = 0
    with pytest.raises(ValueError):
        a.goal_catalog(image)


@pytest.mark.parametrize('field', ['projection', 'task_rank', 'nuisance_rank', 'nuisance_singular_values',
                                   'object_singular_values', 'goal_anchor_max_difference'])
def test_metric_corruption(field):
    args = list(fixture())
    metric = args[0]['metric']
    if field == 'projection':
        metric[field][9][9] += .1
    elif field.endswith('singular_values'):
        metric[field][0] += .1
    else:
        metric[field] += 1
    assert not a.root_audit(*args)[0]['all_checks']


@pytest.mark.parametrize('field', ['old_Q_cost', 'goal_patch_Q_cost'])
def test_all_candidate_costs_recomputed(field):
    args = list(fixture())
    args[0]['selection']['candidates'][0][field] += .01
    assert not a.root_audit(*args)[0]['all_checks']


@pytest.mark.parametrize('mode', ['missing', 'duplicate', 'action', 'context', 'horizon', 'root', 'mask', 'anchor'])
def test_binding_corruption_failclosed(mode):
    args = list(fixture())
    row = args[0]
    if mode == 'missing':
        row['selection']['candidates'].pop()
    elif mode == 'duplicate':
        row['selection']['candidates'][1] = deepcopy(row['selection']['candidates'][0])
    elif mode == 'action':
        row['selection']['candidates'][0]['actions'][0][0] += .01
    elif mode == 'context':
        row['selection']['candidates'][0]['context']['action_history'][0][0] += .01
    elif mode == 'horizon':
        row['selection']['candidates'][0]['horizon'] = 3
    elif mode == 'root':
        row['root_sha256'] = 'b'*64
    elif mode == 'mask':
        row['catalog']['mask_sha256'] = '0'*64
    else:
        args[2]['legacy_goal_vectors'][0][0] += .1
    with pytest.raises(ValueError):
        a.root_audit(*args)


def test_cross_arm_same_key_requires_full_identical_witness():
    args = list(fixture())
    args[4][1]['search']['candidates'][0]['actions'][0][0] += .1
    with pytest.raises(ValueError, match='cross-arm'):
        a.root_audit(*args)


@pytest.mark.parametrize('arm', ['old_Q', 'goal_patch_Q'])
def test_winner_full_packet_not_only_identity(arm):
    args = list(fixture())
    args[0]['selection']['winners'][arm]['actions'][0][0] += .01
    assert not a.root_audit(*args)[0]['all_checks']


def test_tie_horizon_then_index_not_input_order():
    args = list(fixture())
    for source_row in args[4]:
        for c in source_row['search']['candidates']:
            c['context']['latent_history'][-1] = np.zeros(192).tolist()
            c['cost'] = 0.
    rows = args[0]['selection']['candidates']
    for c in rows:
        c['context']['latent_history'][-1] = np.zeros(192).tolist()
        c['old_Q_cost'] = c['goal_patch_Q_cost'] = 0.
    args[0]['selection']['candidates'] = rows[::-1]
    args[0]['selection']['winners'] = {k: deepcopy(rows[0]) for k in ('old_Q', 'goal_patch_Q')}
    audited, _ = a.root_audit(*args)
    assert audited['all_checks'] and audited['winners']['old_Q']['horizon'] == 2


def test_input_hash_and_partial_are_not_repaired():
    args = list(fixture())
    args[1]['goal'][0] += .1
    with pytest.raises(ValueError):
        a.root_audit(*args)
    args = list(fixture())
    args[4][0]['search']['candidates'][0]['completed'] = False
    with pytest.raises(ValueError, match='partial'):
        a.root_audit(*args)


def test_nonfinite_and_bad_shapes():
    row, _, goal, *_ = fixture()
    row['goal_vectors'][0][0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        a.reconstruct(row['goal_vectors'], goal['legacy_goal_vectors'])
    with pytest.raises(ValueError):
        a.array(np.zeros((2, 3)), (3, 2))


def test_evidence_sha_symlink_json_and_exclusive_write(tmp_path):
    p = tmp_path/'file.json'
    a.write(p, {'a': 1})
    e = a.Evidence()
    assert e.json(p)['a'] == 1
    with pytest.raises(FileExistsError):
        a.write(p, {})
    link = tmp_path/'link.json'
    link.symlink_to(p)
    with pytest.raises(ValueError):
        e.bind(link)
    with pytest.raises(ValueError, match='SHA'):
        e.bind(p, '0'*64)
    p.write_text('{"a":2}')
    with pytest.raises(ValueError, match='changed'):
        e.verify()
    p.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError, match='duplicate'):
        a.read_json(p)
    p.write_text('{"a":NaN}')
    with pytest.raises(ValueError, match='nonfinite'):
        a.read_json(p)


def test_import_has_no_neural_or_planner_modules():
    code = ("import sys; from experiments.spectral_reachability import "+a.NAME+"; "
            "assert 'torch' not in sys.modules; "
            "assert not any(k.endswith('p3p4_goal_patch_metric_20261027_v1') for k in sys.modules)")
    subprocess.run([sys.executable, '-B', '-c', code], check=True, cwd=a.ROOT)


def test_actual_saved_regression_and_exclusive_output(tmp_path):
    directory = a.ROOT/'results/p3p4_goal_patch_metric_20261027_v1_cuda_main01'
    if not (directory/'report.json').exists():
        pytest.skip('optional actual saved CUDA arrays unavailable; no neural rerun')
    sha = '7b38b91e18497b625f36e7c52d59c07e53204e6047af083c10b4c4631aa2fee6'
    prepared, output = tmp_path/'prepared', tmp_path/'audited'
    a.prepare(directory, sha, prepared)
    actual = a.run(prepared, a.file_sha(prepared/'prepare.json'), output)
    assert actual['all_checks'] and actual['failure'] is None
    assert [r['common_candidates'] for r in actual['rows']] == [31, 28, 28, 28]
    assert actual['attempted_audit_array_svds'] == actual['completed_audit_array_svds'] == 12
    assert len(actual['artifacts']) == 21
    assert len(actual['rows']) == 4 and all(actual['checks'].values())
    assert all(a.file_sha(p) == h for p, h in actual['artifacts'].items())
    assert a.file_sha(directory/'report.json') == sha
    with pytest.raises(FileExistsError):
        a.run(prepared, a.file_sha(prepared/'prepare.json'), output)
    with pytest.raises(ValueError, match='SHA'):
        a.load_inputs(directory, '0'*64)
