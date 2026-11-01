"""Synthetic RAM/disk evidence only; never analyze production receipts here."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import pytest
from experiments.spectral_reachability import agent_p2_full30_feedback_native_summary_20261029_v2 as m


def fixture(original_path='/RAM/original.json', original_sha='a'*64):
    env = str(m.ROOT/m.ENV_REL)
    def receipt(old, job=None):
        return dict(schema=(m.OLD if old else m.NEW)+'.consume',
            status='full30_four_native_recorded_NOT_admitted' if old else 'feedback_two_native_recorded_NOT_admitted',
            fixed_denominator=4 if old else 2, rows=[], plans=[], errors=[],
            own_native_saved_reconstructed_here=True, known_roots=2, fresh_roots=0,
            job_index=job, global_policy_slots=12, saved_PNG_verified=504 if old else 252,
            full_stage_wall_seconds=3.5,
            counts={k:dict(attempted=n, completed=n) for k, n in
                    (('public_step', 160 if old else 80), ('physics_step', 1616 if old else 808))},
            bindings={env:m.ENV_SHA, original_path:original_sha,
                      **{str(m.ROOT/'experiments/spectral_reachability'/k):v
                         for k, v in (m.ABI_OLD if old else m.ABI_NEW).items()}}, **{k:False for k in m.FLAGS})
    old = receipt(True); jobs = [receipt(False, i) for i in range(4)]
    for base in range(4):
        for pi, policy in enumerate(m.POLICIES):
            # Base0 has differing agent/velocity but identical object; other bases exact ties.
            def state(step):
                shift = pi if base == 0 and step > 15 else 0
                return [100.+base+step-shift, 10., 30., 40., .25, 3.+2*shift, 4.]
            commands = [[.125, .25] for _ in range(30)]
            parent = dict(schema='agent_p2_two_program_postcontext_20261029_v1.model_plan',
                task_id=f'task{base}', task_sha256=f'tasksha{base}', root_slot=base//2,
                arm=m.ARMS[base%2], model_hash='frozen_model', root_context={'frozen_history':base},
                goal={'original':base}, commands=deepcopy(commands),
                raw_float32_bits=struct.pack('<60f', *[v for r in commands for v in r]).hex())
            parent['witness_sha256'] = m.model_digest(parent, 'witness_sha256')
            if pi:
                commands[15:] = [[.125*pi, .375] for _ in range(15)]
            plan = dict(schema=(m.OLD if pi == 0 else m.NEW)+'.execution_plan',
                index=base if pi == 0 else pi-1, root_slot=base//2, arm=m.ARMS[base%2],
                condition='off', task_id=f'task{base}', native_root_sha256=f'root{base//2}',
                native_goal_pixel_sha256=f'goal{base//2}', native_goal=[0.]*7, raw_steps=30,
                commands=deepcopy(commands), planner_plan=parent,
                raw_float32_sha256=hashlib.sha256(m.raw30(commands)).hexdigest())
            if pi:
                n = dict(schema='agent_p2_full30_control15_feedback_20261029_v2.model_plan',
                    base_index=base, policy=policy, root_slot=base//2, arm=m.ARMS[base%2],
                    task_id=f'task{base}', task_sha256=f'tasksha{base}', model_hash='frozen_model',
                    parent_witness_sha256=parent['witness_sha256'], parent_raw_bits=parent['raw_float32_bits'],
                    goal=deepcopy(parent['goal']), commands=deepcopy(commands),
                    raw_float32_bits=m.raw30(commands).hex(), feedback_control=15)
                n['new_witness_sha256'] = m.model_digest(n, 'new_witness_sha256')
                plan.update(base_index=base, policy=policy, new_model_plan=n)
            plan['selection_digest'] = m.execution_digest(plan)
            row = dict(index=base if pi == 0 else pi-1, base_index=base, policy=policy,
                root_slot=base//2, arm=m.ARMS[base%2], condition='off', status='horizon_completed',
                first_error=None, secondary_errors=[], goal=[0.]*7, root_state=state(0),
                task_id=f'task{base}', root_sha256=f'root{base//2}', selection_digest=plan['selection_digest'],
                goal_pixel_sha256=f'goal{base//2}', plan=deepcopy(plan), own_returned=30, own_attempted=30,
                responses=[dict(step=b, state=state(b), native_success=False, terminated=False, truncated=False)
                           for b in range(1, 31)],
                checkpoints=[dict(step=b, state=state(b), snapshot=dict(state=state(b),
                    state_float64_bits=struct.pack('<7d', *state(b)).hex())) for b in m.CP], missing_checkpoints=[])
            dest = old if pi == 0 else jobs[base]
            dest['rows'].append(row); dest['plans'].append(plan)
    return old, jobs


def summary(old=None, jobs=None):
    if old is None:
        old, jobs = fixture()
    return m.summarize(old, jobs, '/RAM/original.json', 'a'*64)


def test_complete_all12_84_84_and_ties():
    r = summary()
    assert len(r['rows']) == 12 and len(r['panel']) == len(r['pairs']) == 84
    assert [x['policy'] for x in r['rows']] == list(m.POLICIES)*4
    assert all(x['metrics'] is not None for x in r['panel'])
    assert all(p['object_pose_exact'] is True for p in r['pairs'])
    for name, counts in r['all_pair_sign_counts'].items():
        assert sum(counts['native7_sign'].values()) == 28
        assert counts['native7_sign']['zero'] >= 21
    assert r['resources']['new_I_F']['physics_step']['completed'] == 3232


def test_independent_seven_axes_fsum_and_angle_semantics():
    x = m.metrics([3., 4., 0., 0., 0., 0., 12.], [0.]*7)
    assert x['native7_squared'] == 169 and x['native7_distance'] == 13
    assert x['joint_xy_squared'] == 25 and x['velocity_squared'] == 144
    assert x['axis_squared'] == [9., 16., 0., 0., 0., 0., 144.]
    y = m.metrics([0., 0., 0., 0., 8., 0., 0.], [0.]*7)
    assert y['negative_native_angle'] and y['native_success'] and not y['wrapped_success_diagnostic']
    # Strict boundaries, no epsilon gate or wrapped substitution.
    assert not m.metrics([20., 0., 0., 0., 0., 0., 0.], [0.]*7)['native_success']
    assert not m.metrics([0., 0., 0., 0., math.pi/9, 0., 0.], [0.]*7)['native_success']
    z = m.metrics([1e8, 1., 1., 0., 0., 0., 0.], [0.]*7)
    assert z['native7_squared'] == math.fsum([1e16, 1., 1.])
    assert z['native7_squared'] != ((1e16+1.)+1.)


def test_all_axes_pair_decomposition_and_delta_direction():
    r = summary()
    p = next(p for p in r['pairs'] if p['base'] == 0 and p['pair'] == 'O-I' and p['control_step'] == 30)
    assert p['delta_direction'] == 'I minus O'
    assert p['axis_squared_delta'] == [-259., 0., 0., 0., 0., 16., 0.]
    assert p['state_axis_delta'] == [-1., 0., 0., 0., 0., 2., 0.]
    assert p['delta']['native7_squared'] == -243.
    assert p['total_minus_axis_roundoff'] == 0


def test_producer_metric_fields_never_used():
    old, jobs = fixture()
    first = summary(old, jobs)
    for r in [old, *jobs]:
        r['metric_breakdown'] = {'malicious':1}
        for row in r['rows']:
            row['final_metrics'] = {'native7_distance':-90000}
            for response in row['responses']:
                response['native_metrics'] = {'native_success':True}
                response['reward'] = 1e300
    assert m.exact(first, summary(old, jobs))


def test_early_stop_missing_not_zero_or_terminal_padding():
    old, jobs = fixture()
    row = jobs[3]['rows'][1]
    row.update(status='native_stopped', own_returned=17, own_attempted=17,
               responses=row['responses'][:17], checkpoints=row['checkpoints'][:4], missing_checkpoints=[20,25,30])
    row['responses'][-1]['truncated'] = True
    r = summary(old, jobs)
    assert len(r['panel']) == len(r['pairs']) == 84
    missing = [p for p in r['panel'] if p['metrics'] is None]
    assert [p['control_step'] for p in missing] == [20,25,30]
    assert all(p['state7'] is None and p['missing_reason'] == 'checkpoint_not_reached' for p in missing)
    assert sum(p['comparable'] is False for p in r['pairs']) == 6
    assert r['all_pair_sign_counts']['O-F']['native7_sign']['unknown'] == 3


@pytest.mark.parametrize('fault', ['job_count','row_count','order','job_id','schema','authority','flag','ancestor',
    'task','root','goal','history','prefix','cp','bits','stop','missing','counts','condition','plan','duplicate_cp',
    'prefix_state'])
def test_resealed_semantic_tampering_rejected(fault):
    old, jobs = fixture(); j = jobs[0]; row = j['rows'][0]
    if fault == 'job_count': jobs.pop()
    elif fault == 'row_count': j['rows'].pop()
    elif fault == 'order': j['rows'].reverse(); j['plans'].reverse()
    elif fault == 'job_id': j['job_index'] = 1
    elif fault == 'schema': j['schema'] = 'old.consume'
    elif fault == 'authority': j['own_native_saved_reconstructed_here'] = False
    elif fault == 'flag': j['scientific_admission'] = True
    elif fault == 'ancestor': j['bindings']['/RAM/original.json'] = 'b'*64
    elif fault == 'task': row['task_id'] = 'wrong'
    elif fault == 'root': row['root_sha256'] = 'wrong'
    elif fault == 'goal': row['goal'][0] = 1.
    elif fault == 'history': row['plan']['planner_plan']['root_context'] = {}; j['plans'][0] = deepcopy(row['plan'])
    elif fault == 'prefix': row['plan']['commands'][0] = [0.,0.]; j['plans'][0] = deepcopy(row['plan'])
    elif fault == 'cp': row['checkpoints'][4]['state'][0] += 1.
    elif fault == 'bits': row['checkpoints'][4]['snapshot']['state_float64_bits'] = '0'*112
    elif fault == 'stop': row['responses'][0]['terminated'] = True
    elif fault == 'missing': row['missing_checkpoints'] = [30]
    elif fault == 'counts': j['counts'].pop('physics_step')
    elif fault == 'condition': row['condition'] = 'on'
    elif fault == 'plan': j['plans'][0]['native_goal'] = [1.]*7
    elif fault == 'prefix_state':
        f = row['checkpoints'][2]; f['state'][0] += 1.
        f['snapshot']['state'] = deepcopy(f['state'])
        f['snapshot']['state_float64_bits'] = struct.pack('<7d', *f['state']).hex()
        row['responses'][9]['state'] = deepcopy(f['state'])
    else: row['checkpoints'].append(deepcopy(row['checkpoints'][-1]))
    with pytest.raises(ValueError): summary(old, jobs)


@pytest.mark.parametrize('bad', [[False]*7, [float('nan')]*7, [float('inf')]*7, [0.]*6])
def test_bad_state_never_zero_filled(bad):
    with pytest.raises(ValueError): m.metrics(bad, [0.]*7)


def disk_fixture(tmp_path):
    p = tmp_path/(m.NAME+'_original.json')
    old, jobs = fixture(str(p))
    # Real files + SHA, no mock of read/build/summarize/wholeverify.
    p.write_bytes(m.encode(old)); h = m.sha(p)
    paths = []; hashes = []
    for i, job in enumerate(jobs):
        job['bindings'][str(p)] = h
        path = tmp_path/(m.NAME+f'_job{i}.json')
        path.write_bytes(m.encode(job)); paths.append(str(path)); hashes.append(m.sha(path))
    return dict(original=str(p), original_sha=h, jobs=paths, job_shas=hashes)


def test_complete_real_disk_build_write_wholeverify(tmp_path):
    args = disk_fixture(tmp_path)
    value = m.build(**args)
    out = tmp_path/(m.NAME+'_fixture.json')
    result = m.write(out, value)
    assert m.verify(str(out), result['sha256'])['verified']
    assert set(value['own_sha256']) == {str(m.ROOT/'experiments/spectral_reachability'/(m.NAME+'.py')),
        str(m.ROOT/'tests'/('test_'+m.NAME+'.py')), str(m.ROOT/'innerdocs'/(m.NAME+'.md'))}
    assert value['new_fullstrict_calls'] == value['new_neural_calls'] == value['new_physics_calls'] == 0
    with pytest.raises(FileExistsError): m.write(out, value)
    value['summary']['panel'][0]['metrics']['native7_squared'] += 1
    out.write_bytes(m.encode(value))
    with pytest.raises(ValueError, match='whole summary'): m.verify(str(out), m.sha(out))


def test_cli_complete_temp_disk_analyze_verify(tmp_path, capsys):
    args = disk_fixture(tmp_path)
    out = tmp_path/(m.NAME+'_cli.json')
    m.main(['analyze', '--original', args['original'], '--original-sha', args['original_sha'],
            '--jobs', *args['jobs'], '--job-shas', *args['job_shas'], '--output', str(out)])
    printed = json.loads(capsys.readouterr().out)
    assert printed['sha256'] == m.sha(out)
    m.main(['verify', '--summary', str(out), '--summary-sha', printed['sha256']])
    assert json.loads(capsys.readouterr().out)['verified']


def test_disk_sha_drift_and_semantics_source_binding(tmp_path):
    args = disk_fixture(tmp_path)
    p = Path(args['jobs'][0]); p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError, match='SHA256'): m.build(**args)
    args['job_shas'][0] = m.sha(p)
    value = json.loads(p.read_bytes()); value['bindings'][str(m.ROOT/m.ENV_REL)] = '0'*64
    p.write_bytes(m.encode(value)); args['job_shas'][0] = m.sha(p)
    with pytest.raises(ValueError, match='eval_state'): m.build(**args)


def test_duplicate_nonfinite_relative_and_cold_import(tmp_path):
    p = tmp_path/(m.NAME+'_bad.json')
    for b in (b'{"a":1,"a":2}', b'{"a":NaN}'):
        p.write_bytes(b)
        with pytest.raises(ValueError): m.read(str(p), hashlib.sha256(b).hexdigest())
    with pytest.raises(ValueError): m.absolute('relative.json')
    script = 'import sys; import '+m.__name__+'; assert not any(x in sys.modules for x in ("torch","numpy","pymunk","pygame"))'
    proc = subprocess.run([sys.executable, '-c', script], cwd=m.ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def reseal_execution(receipt, index=0):
    row = receipt['rows'][index]; p = row['plan']
    p['selection_digest'] = m.execution_digest(p)
    row['selection_digest'] = p['selection_digest']
    receipt['plans'][index] = deepcopy(p)


def test_distinct_correct_execution_digests_are_not_cross_policy_identity():
    old, jobs = fixture()
    for base, job in enumerate(jobs):
        assert len({old['rows'][base]['selection_digest'],
                    *[r['selection_digest'] for r in job['rows']]}) == 3
    r = summary(old, jobs)
    assert len(r['rows']) == 12
    assert len(r['panel']) == len(r['pairs']) == 84


def test_exact_noncompact_noLF_execution_and_compactLF_model_ABIs():
    value = {'z': [1., 2.], 'unicode': '目标', 'a': {'k':True}}
    expected = hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert m.execution_digest(dict(value, selection_digest='ignored')) == expected
    compactLF = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                               allow_nan=False).encode()+b'\n').hexdigest()
    assert m.model_digest(dict(value, witness_sha256='ignored'), 'witness_sha256') == compactLF
    assert expected != compactLF
    assert expected != hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()+b'\n').hexdigest()
    assert compactLF != hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


@pytest.mark.parametrize('old_policy,fault', [
    (old, fault) for old in (True, False) for fault in (
        'wrong_execution', 'row_plan_differ', 'compact_execution', 'LF_execution',
        'parent_witness', 'new_witness', 'parent_raw', 'new_raw', 'raw_SHA', 'own_scope', 'new_parent')
    if not (old and fault in ('new_witness', 'new_raw', 'new_parent'))])
def test_resealed_wrong_witnesses_rejected(old_policy, fault):
    old, jobs = fixture()
    rec = old if old_policy else jobs[0]
    row = rec['rows'][0]; p = row['plan']
    if fault == 'wrong_execution': p['selection_digest'] = row['selection_digest'] = '0'*64
    elif fault == 'row_plan_differ': row['selection_digest'] = '0'*64
    elif fault in ('compact_execution', 'LF_execution'):
        body = {k:v for k,v in p.items() if k != 'selection_digest'}
        b = m.encode(body) if fault == 'compact_execution' else json.dumps(body,sort_keys=True,allow_nan=False).encode()+b'\n'
        p['selection_digest'] = row['selection_digest'] = hashlib.sha256(b).hexdigest()
    elif fault == 'parent_witness': p['planner_plan']['witness_sha256'] = '0'*64
    elif fault == 'new_witness': p['new_model_plan']['new_witness_sha256'] = '0'*64
    elif fault == 'parent_raw': p['planner_plan']['raw_float32_bits'] = '0'*480
    elif fault == 'new_raw': p['new_model_plan']['raw_float32_bits'] = '0'*480
    elif fault == 'raw_SHA': p['raw_float32_sha256'] = '0'*64
    elif fault == 'own_scope': p['task_id'] = 'other'
    else: p['new_model_plan']['parent_witness_sha256'] = '0'*64
    if fault == 'parent_raw':
        parent = p['planner_plan']
        parent['witness_sha256'] = m.model_digest(parent, 'witness_sha256')
        if not old_policy:
            p['new_model_plan']['parent_witness_sha256'] = parent['witness_sha256']
    if not old_policy and fault in ('parent_raw', 'new_raw', 'new_parent'):
        p['new_model_plan']['new_witness_sha256'] = m.model_digest(p['new_model_plan'], 'new_witness_sha256')
    if fault not in ('wrong_execution', 'row_plan_differ', 'compact_execution', 'LF_execution'):
        reseal_execution(rec)
    else: rec['plans'][0] = deepcopy(p)
    with pytest.raises(ValueError): summary(old, jobs)


def test_swapped_parent_fully_resealed_still_rejected():
    old, jobs = fixture()
    row = jobs[0]['rows'][0]; p = row['plan']; parent = p['planner_plan']; n = p['new_model_plan']
    parent['root_context'] = {'other_parent':'resealed'}
    parent['witness_sha256'] = m.model_digest(parent, 'witness_sha256')
    n['parent_witness_sha256'] = parent['witness_sha256']
    n['new_witness_sha256'] = m.model_digest(n, 'new_witness_sha256')
    reseal_execution(jobs[0])
    # Each witness locally valid, but ancestor differs from pinned O.
    m.plan_witness(row, False)
    with pytest.raises(ValueError, match='same original full model history'):
        summary(old, jobs)


def test_resealed_new_recipe_fields_cannot_change():
    old, jobs = fixture()
    p = jobs[0]['rows'][0]['plan']
    p['native_goal_pose'] = ['not inherited']
    reseal_execution(jobs[0])
    with pytest.raises(ValueError, match='producer-declared'):
        summary(old, jobs)


@pytest.mark.parametrize('bad', [[[.1, .2]]*30, [[True, 0.]]*30, [[1.5, 0.]]*30, [[0.,0.]]*29])
def test_raw_action_no_repair(bad):
    with pytest.raises(ValueError): m.raw30(bad)


def test_actual_frozen_digest_source_bytes_only():
    # Verify exact source ABI without importing producers or processing outcomes.
    for name, h in m.ABI_NEW.items():
        assert m.sha(m.ROOT/'experiments/spectral_reachability'/name) == h


def test_disk_resealed_bad_digest_and_swapped_parent(tmp_path):
    args = disk_fixture(tmp_path)
    p = Path(args['jobs'][0]); value = json.loads(p.read_bytes())
    row = value['rows'][0]; row['plan']['selection_digest'] = row['selection_digest'] = '0'*64
    value['plans'][0] = deepcopy(row['plan'])
    p.write_bytes(m.encode(value)); args['job_shas'][0] = m.sha(p)
    with pytest.raises(ValueError, match='selection digest'): m.build(**args)
    # Restore full legitimate fixture then substitute/reseal parent with all local joins valid.
    args = disk_fixture(tmp_path)
    value = json.loads(p.read_bytes()); row = value['rows'][0]
    parent = row['plan']['planner_plan']; n = row['plan']['new_model_plan']
    parent['root_context']['extra'] = 'foreign'
    parent['witness_sha256'] = m.model_digest(parent, 'witness_sha256')
    n['parent_witness_sha256'] = parent['witness_sha256']
    n['new_witness_sha256'] = m.model_digest(n, 'new_witness_sha256')
    reseal_execution(value)
    p.write_bytes(m.encode(value)); args['job_shas'][0] = m.sha(p)
    with pytest.raises(ValueError, match='same original full model history'): m.build(**args)


def test_disk_digest_ABI_receipt_source_drift_rejected(tmp_path):
    args = disk_fixture(tmp_path)
    p = Path(args['jobs'][0]); value = json.loads(p.read_bytes())
    source_path = str(m.ROOT/'experiments/spectral_reachability'/'p1_conditional_task_admission_20261027_v1.py')
    value['bindings'][source_path] = '0'*64
    p.write_bytes(m.encode(value)); args['job_shas'][0] = m.sha(p)
    with pytest.raises(ValueError, match='digest ABI source'): m.build(**args)

