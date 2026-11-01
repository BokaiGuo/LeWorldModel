"""RAM mutations and read-only saved lineage; no model or simulator execution."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_midpoint_history_lineage_20261028_v1 as m


def ram_lineage():
    initial = dict(m.mid.META, latent_history=np.arange(576, dtype=np.float32).reshape(3, 192).tolist(),
                   action_history=np.zeros((2, 10), np.float32).tolist())
    raw = [[.125, -.25]]*10
    paid = deepcopy(initial)
    paid['latent_history'] = initial['latent_history'][1:] + [[7.]*192]
    paid['action_history'] = [initial['action_history'][-1], [.125, -.25]*5]
    norm = dict(mean=[0., 0.], std=[1., 1.])
    teacher, arrays = m.suffix.intervention(paid, [8.]*192, raw, norm)
    job = dict(root_slot=0, arm=m.ARMS[0], root_sha256='RAM-root', job_binding_sha256='RAM-binding',
               paid_mid_context=paid, paid_mid_context_sha256=m.digest(paid), teacher_mid_context=teacher,
               raw10=raw, raw_suffix5=raw[5:], action_normalization=norm, goal=[9.]*192,
               metric_id='RAM-Q', projection=np.eye(192).tolist())
    compiled = dict(emb=[initial['latent_history']], actions=[np.zeros((3, 10)).tolist()])
    declaration = dict(context=deepcopy(initial), root_id='RAM-root', root_rgb_sha256='RAM-history', goal_rgb_sha256='RAM-goal')
    physical = dict(root_sha256='RAM-root', method=m.ARMS[0], context_identity=dict(
        history_rgb_sha256='RAM-history', goal_rgb_sha256='RAM-goal', raw_history=[[0., 0.]]*3,
        normalized_between_history=deepcopy(initial['action_history'])))
    return [job, initial, compiled, declaration, physical, arrays['paid_latents'], arrays['teacher_latents'][None]]


def test_ram_exact_observed_source_slots_not_two_rollouts():
    args = ram_lineage()
    row = m.trace_job(*args)
    assert [r['initial_context_slot'] for r in row['older2']] == [1, 2]
    assert [r['original_warmup_primitive_step'] for r in row['older2']] == [10, 15]
    assert [r['relative_to_planning_root'] for r in row['older2']] == [-5, 0]
    assert row['older2_are_predicted_rollout_states'] is False
    assert row['per_frame_RGB_to_encoder_buffer_independently_proven'] is None
    assert all(r['original_per_frame_native_encoder_receipt'] is None for r in row['older2'])
    assert row['incoming_action_labels'] == ['A2', 'A3'] and row['outgoing_action_labels'] == ['A2', 'A3', 'A4']


@pytest.mark.parametrize('damage', ['older0', 'older1', 'initial', 'compiled_z', 'compiled_action', 'root',
                                    'history_rgb', 'goal_rgb', 'raw_history', 'between', 'suffix',
                                    'native_input', 'native_dtype', 'paid_buffer', 'teacher_older',
                                    'paid_digest', 'prefix_token', 'null', 'bool'])
def test_lineage_mutations_rejected_not_zero_filled(damage):
    args = ram_lineage()
    j, initial, compiled, declaration, physical, paid, native = args
    if damage == 'older0': j['paid_mid_context']['latent_history'][0][0] += 1
    if damage == 'older1': j['paid_mid_context']['latent_history'][1][0] += 1
    if damage == 'initial': initial['latent_history'][1][0] += 1
    if damage == 'compiled_z': compiled['emb'][0][1][0] += 1
    if damage == 'compiled_action': compiled['actions'][0][2][0] = 1.
    if damage == 'root': declaration['root_id'] = 'different-root'
    if damage == 'history_rgb': physical['context_identity']['history_rgb_sha256'] = 'different-history'
    if damage == 'goal_rgb': physical['context_identity']['goal_rgb_sha256'] = 'different-goal'
    if damage == 'raw_history': physical['context_identity']['raw_history'][0] = [1., 0.]
    if damage == 'between': physical['context_identity']['normalized_between_history'][0][0] = 1.
    if damage == 'suffix': j['raw_suffix5'] = [[.5, .5]]*5
    if damage == 'native_input': native[0, 0, 0] += 1
    if damage == 'native_dtype': args[-1] = native.astype(np.float64)
    if damage == 'paid_buffer': paid[0, 0] += 1
    if damage == 'teacher_older': j['teacher_mid_context']['latent_history'][0][0] += 1
    if damage == 'paid_digest': j['paid_mid_context_sha256'] = 'bad'
    if damage == 'prefix_token':
        j['paid_mid_context']['action_history'][1][0] += 1
        j['paid_mid_context_sha256'] = m.digest(j['paid_mid_context'])
    if damage == 'null': initial['latent_history'][1][0] = None
    if damage == 'bool': initial['latent_history'][1][0] = False
    with pytest.raises((ValueError, TypeError)):
        m.trace_job(*args)


def ram_diagnostics():
    strict, native = [], []
    for i in range(4):
        for a, arm in enumerate(m.ARMS):
            q = float(2-a)  # Q favors composition, native agent+object favors joint.
            state, goal = [float(a*30), 0., 0., 0., 0., 0., 0.], [0.]*7
            strict.append(dict(root_slot=i, arm=arm, root_sha256='RAM-'+str(i), actual_terminal_Q=q,
                               original_open_loop=dict(predicted_Q=q+.5), teacher_forced_suffix=dict(predicted_Q=q+.25),
                               latent_MSE_change=-.1))
            native.append(dict(root_slot=i, arm=arm, root_sha256='RAM-'+str(i), prediction=dict(physical_teacher_Q=q, Q_cost=q+.5),
                               observed=m.gap.decompose(state, goal), goal=goal, initial_state=[0.]*7))
    return strict, native


def test_same_Q_vs_native_different_questions_and_object_not_full_task():
    rows, pairs = m.diagnose(*ram_diagnostics())
    assert len(rows) == 8 and len(pairs) == 4
    for p in pairs:
        assert p['teacher_prediction_matches_actual_Q'] is True
        assert p['actual_Q_matches_jointXY'] is False
        assert p['actual_Q_matches_native7'] is False
        assert p['objectXY_matches_jointXY'] is False
        assert p['lower_is_better']['objectXY'] == 'tie'
        assert p['official_success'] == {m.ARMS[0]: True, m.ARMS[1]: False}


def test_unknown_native_metric_is_null_not_zero_or_win():
    s, n = ram_diagnostics()
    n[0]['observed']['objectXY'] = None
    _, pairs = m.diagnose(s, n)
    assert pairs[0]['composition_minus_joint']['objectXY'] is None
    assert pairs[0]['lower_is_better']['objectXY'] is None
    assert pairs[0]['actual_Q_matches_objectXY'] is None


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'root', 'goal', 'Q'])
def test_full_denominator_and_join_fail_closed(damage):
    s, n = ram_diagnostics()
    if damage == 'missing': s.pop()
    if damage == 'duplicate': s[-1] = deepcopy(s[0])
    if damage == 'root': n[0]['root_sha256'] = 'other'
    if damage == 'goal': n[0]['goal'][0] = 10.
    if damage == 'Q': n[0]['prediction']['physical_teacher_Q'] += 1
    with pytest.raises(ValueError): m.diagnose(s, n)


@pytest.mark.parametrize('value, expected', [(None, None), (0., 'tie'), (-1., m.ARMS[1]), (1., m.ARMS[0])])
def test_exact_sign_no_threshold(value, expected):
    assert m.winner(value) == expected


@pytest.mark.parametrize('damage', [None, 'goal', 'projection', 'metric', 'physical_metric'])
def test_original_goal_Q_identity_no_metric_substitution(damage):
    job = dict(goal=[1.]*192, projection=np.eye(192).tolist(), metric_id='RAM-original-Q')
    compiled = dict(goal_vectors=[deepcopy(job['goal'])], scoring=dict(metric=dict(
        projection=deepcopy(job['projection']), sha256=job['metric_id'])))
    physical = dict(active_metric_id=job['metric_id'])
    if damage == 'goal': job['goal'][0] += 1
    if damage == 'projection': job['projection'][0][0] = 0.
    if damage == 'metric': job['metric_id'] = 'replacement'
    if damage == 'physical_metric': physical['active_metric_id'] = 'other'
    if damage is None: m.objective_identity(job, compiled, physical)
    else:
        with pytest.raises(ValueError): m.objective_identity(job, compiled, physical)


def test_next_protocol_no_outcome_scores_used():
    jobs = [dict(root_slot=i, arm=a, root_sha256='RAM-'+str(i), job_binding_sha256='input-'+str(i)+a)
            for i in range(4) for a in m.ARMS]
    before = m.next_protocol(jobs)
    changed = deepcopy(jobs)
    for j in changed:
        j.update(native_success=True, score=-1e30, rank=1, latent_MSE=0.)
    assert m.next_protocol(changed) == before
    assert before['stage_A']['predict_API'] == 8
    assert before['stage_A']['encoder'] == before['stage_A']['physics'] == 0
    assert before['stage_B']['useful_prediction_budget']['total_predict'] == 16
    assert before['stage_B']['raw_triplet_proven_in_consumed_closure'] is None
    assert before['stage_B']['public_ABI']['n_steps'] == 0
    assert before['stage_B']['public_ABI']['loop_iterations'] == 1
    assert before['scientific_admission'] is False
    with pytest.raises(ValueError): m.next_protocol(jobs[:-1])
    with pytest.raises(ValueError): m.next_protocol(list(reversed(jobs)))


def test_read_source_without_importing_it(tmp_path):
    path = tmp_path/'sealed.source'
    path.write_text('raise RuntimeError("must never execute")\ndef known():\n    return 4\n')
    e = m.function_evidence(path, 'known')
    assert e['start_line'] == 2 and e['end_line'] == 3 and e['file_sha256'] == m.sha(path)
    with pytest.raises(ValueError): m.function_evidence(path, 'absent')


@pytest.fixture(scope='module')
def saved():
    return m.load_saved()


def test_saved_all8_lineage_and_unmodified_scope(saved):
    assert len(saved['lineage']) == len(saved['diagnostic_rows']) == 8
    assert len(saved['pairs']) == 4
    assert all(not row['older2_are_predicted_rollout_states'] for row in saved['lineage'])
    assert saved['new_neural_calls'] == saved['new_physics_steps'] == 0
    assert saved['scientific_admission'] is False
    assert saved['evidence_scope']['original_per_frame_encoder_native_receipts'] is None
    for path, expected in m.FROZEN.items(): assert m.sha(path) == expected


def test_cold_saved_analysis_no_neural_or_physics_import():
    code = '''
import sys
class Reject:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','gym','gymnasium','stable_worldmodel','stable_pretraining','pymunk'}:
            raise AssertionError('forbidden runtime import: '+fullname)
sys.meta_path.insert(0, Reject())
from experiments.spectral_reachability import agent_p2_midpoint_history_lineage_20261028_v1 as m
r=m.load_saved()
assert len(r['lineage']) == 8
assert not set(sys.modules).intersection({'torch','gym','gymnasium','pymunk'})
'''
    run = subprocess.run([sys.executable, '-B', '-c', code], cwd=m.ROOT,
                         env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1', 'OPENBLAS_NUM_THREADS':'1'},
                         text=True, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr


def test_new_output_exclusive_only(tmp_path):
    path = tmp_path/'analysis.json'
    m.mid.write(path, dict(status='RAM_only_no_research_result'))
    with pytest.raises(FileExistsError): m.mid.write(path, dict(status='must_not_replace'))
    assert m.read(path)['status'] == 'RAM_only_no_research_result'
