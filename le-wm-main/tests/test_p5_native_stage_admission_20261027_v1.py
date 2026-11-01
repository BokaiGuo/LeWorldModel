"""Pure arithmetic/JSON fixtures; no native physics, model, network or CUDA."""
import copy
import hashlib
import json
import math

import pytest

from experiments.spectral_reachability import p5_native_stage_admission_20261027_v1 as gate


@pytest.fixture
def sample():
    def audited(n):
        return dict(all_checks=True, check_count=n, checks={str(i): True for i in range(n)}, P5_admission=False)

    e, o, g, c, f, r, a, ready = [{} for _ in range(8)]
    data = dict(zip(('executor', 'outcome', 'geometry', 'compiler', 'fd', 'ridge', 'fd_audit', 'ready'),
                    (e, o, g, c, f, r, a, ready)))
    for d in data.values():
        d['P5_admission'] = False
    o.update(audited(542), status='c1_saved_physical_outcomes_audited', source_data_hashes_unchanged=True,
             authority=dict(compiler_receipt_file_sha256=gate.SPECS['compiler'][1], selected_env_packet=[[0., 0.]] * 5, selected_id='nominal'),
             counts=dict(close=3, control_step=45, space_step=453), protocol=dict(seed=1), rows=[])
    g.update(audited(60), status='saved_native_geometry_consistent', rows=[])
    a.update(audited(91), status='saved_fd_arithmetic_consistent', fd_receipt_sha256=gate.SPECS['fd'][1])
    r.update(status='saved_prospective_ridge_consistent', compiler_receipt_sha256=gate.SPECS['compiler'][1], arithmetic=audited(2))
    model = {k: dict(eval=True, frozen=True, all_cuda_finite=True, state_sha256='a' * 64)
             for k in ('official_encoder', 'predictor', 'proprio_encoder')}
    counts = dict(autograd_grad=20, jvp_columns=10, official_encoder=1, official_rgb_frames=4, predictor=13, proprio_encoder=1)
    for d, n in ((c, counts), (f, dict(autograd_grad=0, jvp_columns=0, official_encoder=0, official_rgb_frames=0, predictor=40, proprio_encoder=0))):
        d.update(attempted=n.copy(), completed=n.copy(), counts_match=True, error=None, finalization_errors=[],
                 partial_work_unknown=False, training=False, physical_calls=0, model_before=copy.deepcopy(model),
                 model_after=copy.deepcopy(model), used_observed_frame15=False,
                 strict_loads={'predictor': dict(exact_match=True)}, compiler_inputs_sha256='b' * 64)
    c.update(status='selected_packet_replayed', selected_replay_exact=True, source_check_errors=[], manifest_file_sha256='c' * 64,
             selected_env_packet=[[0., 0.]] * 5, selected_id='nominal')
    f.update(status='native_JVP_FD_comparison_recorded', compiler_manifest_sha256=c['manifest_file_sha256'],
             compiler_receipt_sha256=gate.SPECS['compiler'][1], comparisons={eps: dict(accepted=None,
             acceptance_threshold=None, columns=[{}] * 10, matrix=dict(relative_error_to_jvp=.001)) for eps in ('0.005', '0.01')})
    ready.update(contract=dict(compiler=dict(expected_completed=counts), decision_frame=10, terminal_frame=15, seed=1,
                              execution=dict(expected=o['counts'].copy())), goal_pose=[3., 4., 0.],
                 selected_packet=dict(packet=[[0., 0.]] * 5), branches=[])
    e.update(ready_manifest_file_sha256=gate.SPECS['ready'][1], status='native_c1_three_branch_execution_recorded',
             global_error=None, physics_executed=True, neural_executed=False, rows=[])
    frames, commands = {}, {}
    for arm in gate.ARMS:
        close = dict(call_attempted=True, call_completed=True, return_durable=True, status='completed')
        e['rows'].append(dict(arm=arm, status='completed', frames=16, prefix_gate_frames=list(range(11)), close_receipt=close, error=None))
        metric = dict(object_xy_goal_error=5., object_angle_goal_error=0., native_final_coverage=0., goal_pose=[3., 4., 0.])
        contacts = [1, 10, 10, 4] + [0] * 12
        o['rows'].append(dict(arm=arm, status='completed', frame_count=16, prefix_matches=[dict(frame=i, all5_byte_exact=True) for i in range(11)],
            interrupted_intents=[], close_failures=[], close_receipt=close.copy(), error=None,
            metrics=dict(contact_points_frames0_to15=contacts.copy(), object_pose_exactly_stationary_since_decision=True,
                object_suffix_xy=[[0., 0.]] * 6, object_suffix_angles_native=[0.] * 6,
                decision=metric.copy(), terminal=metric.copy(), agent_suffix_path_px=1.)))
        g['rows'].append(dict(arm=arm, contacts=contacts.copy(), object_suffix_stationary=True, frame_clearances=[2.] * 16,
                             decision_clearance=2., pd_position_max_abs_error=0., pd_velocity_max_abs_error=0.,
                             min_reconstructed_suffix_substep_clearance=1.))
        ready['branches'].append(dict(arm=arm, root_identity_sha256='d' * 64, prefix_env_commands=[[0., 0.]] * 10,
                                     suffix_env_packet=[[0., 0.]] * 5))
        frames[arm] = [dict(env=dict(n_contact_points=v, settings=dict(relative=True, action_scale=100), sim_hz=100,
                                    control_hz=10, seed=1), bodies=dict(block=dict(position=[0., 0.], angle=0.))) for v in contacts]
        commands[arm] = [dict(raw_env_argument=[0., 0.]) for _ in range(15)]
    o['comparison'] = dict(compiled_minus_control={arm: dict(object_xy_goal_error=0., object_angle_goal_error=0., native_final_coverage=0.) for arm in gate.ARMS[1:]})
    return data, frames, commands


def test_consistent_diagnostic_not_admission(sample):
    result = gate.evaluate(*sample)
    assert result['all_checks']
    assert len(result['checks']) >= 40
    for obs in result['observations']:
        assert obs['prefix_contact_points_sum'] == 24
        assert obs['suffix_contact_points_sum'] == 0
        assert obs['object_suffix_stationary']
    assert len(gate.blockers()) == 6


@pytest.mark.parametrize('slot,key,value', [
    ('outcome', 'all_checks', False), ('outcome', 'check_count', 541),
    ('geometry', 'status', 'inconsistent'), ('fd_audit', 'check_count', 90),
    ('executor', 'global_error', 'interrupted'), ('executor', 'neural_executed', True),
    ('executor', 'ready_manifest_file_sha256', 'wrong'), ('compiler', 'selected_replay_exact', False),
    ('compiler', 'training', True), ('compiler', 'partial_work_unknown', True),
    ('compiler', 'source_check_errors', ['changed']), ('compiler', 'P5_admission', True),
    ('fd', 'compiler_inputs_sha256', 'different_chart'), ('fd', 'compiler_receipt_sha256', 'different_run'),
    ('fd', 'compiler_manifest_sha256', 'different_manifest'), ('fd', 'physical_calls', 1),
])
def test_fail_closed_mutated_contract(sample, slot, key, value):
    data, frames, commands = sample
    data[slot][key] = value
    assert not gate.evaluate(data, frames, commands)['all_checks']


@pytest.mark.parametrize('case', ['duplicate_arm', 'missing_arm', 'missing_frame', 'negative_contact', 'bool_contact'])
def test_invalid_slots_rejected(sample, case):
    data, frames, commands = sample
    if case == 'duplicate_arm':
        data['outcome']['rows'][1]['arm'] = 'compiled'
    elif case == 'missing_arm':
        data['geometry']['rows'].pop()
    elif case == 'missing_frame':
        frames['compiled'].pop()
    else:
        frames['compiled'][11]['env']['n_contact_points'] = -1 if case == 'negative_contact' else True
    with pytest.raises(ValueError):
        gate.evaluate(data, frames, commands)


@pytest.mark.parametrize('case', ['counts', 'close', 'prefix', 'contact', 'motion', 'action', 'metric', 'FD_threshold', 'model'])
def test_arithmetic_and_identity_mutations(sample, case):
    data, frames, commands = sample
    if case == 'counts':
        data['outcome']['counts']['space_step'] -= 1
    elif case == 'close':
        data['outcome']['rows'][0]['close_receipt']['return_durable'] = False
    elif case == 'prefix':
        data['executor']['rows'][0]['prefix_gate_frames'].pop()
    elif case == 'contact':
        frames['compiled'][11]['env']['n_contact_points'] = 1
    elif case == 'motion':
        frames['compiled'][15]['bodies']['block']['position'] = [1., 0.]
    elif case == 'action':
        commands['compiled'][10]['raw_env_argument'] = [.1, 0.]
    elif case == 'metric':
        data['outcome']['rows'][0]['metrics']['terminal']['object_xy_goal_error'] = 1.
    elif case == 'FD_threshold':
        data['fd']['comparisons']['0.005']['accepted'] = True
    elif case == 'model':
        data['fd']['model_before']['predictor']['state_sha256'] = 'e' * 64
    assert not gate.evaluate(data, frames, commands)['all_checks']


def test_arm_order_not_identity(sample):
    data, frames, commands = sample
    data['outcome']['rows'].reverse()
    assert gate.evaluate(data, frames, commands)['all_checks']


@pytest.mark.parametrize('contents', ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e400}'])
def test_strict_json(tmp_path, contents):
    p = tmp_path / (gate.NAME + '.json')
    p.write_text(contents)
    with pytest.raises(ValueError):
        gate.Reader(tmp_path).read(p, hashlib.sha256(p.read_bytes()).hexdigest())


def test_reader_hash_drift_and_escape(tmp_path):
    p = tmp_path / (gate.NAME + '.json')
    p.write_text('{}')
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    reader = gate.Reader(tmp_path)
    assert reader.read(p, h) == {}
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='SHA mismatch'):
        reader.stable()
    with pytest.raises(ValueError, match='outside project'):
        reader.read(tmp_path / '..' / 'outside.json', h)


def test_reader_symlink_and_budget(tmp_path):
    p = tmp_path / (gate.NAME + '.json')
    p.write_text('{}')
    link = tmp_path / (gate.NAME + '_link.json')
    link.symlink_to(p)
    with pytest.raises(ValueError, match='symlink'):
        gate.Reader(tmp_path).read(link, hashlib.sha256(b'{}').hexdigest())
    p.write_bytes(b' ' * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match='oversized'):
        gate.Reader(tmp_path).read(p, '')


def test_missing_saved_data_report(tmp_path):
    report = gate.audit(tmp_path)
    assert report['status'] == 'invalid_evidence'
    assert not report['all_checks'] and not report['P5_admission'] and not report['theory_failure']


def test_cli_exit_codes_and_exclusive_output(tmp_path, monkeypatch):
    result = dict(all_checks=True, P5_admission=False, theory_failure=False)
    monkeypatch.setattr(gate, 'audit', lambda: result)
    assert gate.main([]) == 0
    assert gate.main(['--require-admission']) == 3
    out = tmp_path / (gate.NAME + '.json')
    assert gate.main(['--output', str(out)]) == 0
    assert json.loads(out.read_text()) == result
    with pytest.raises(ValueError, match='already exists'):
        gate.main(['--output', str(out)])
    with pytest.raises(ValueError, match='exact task stem'):
        gate.main(['--output', str(tmp_path / 'other.json')])
    monkeypatch.setattr(gate, 'audit', lambda: dict(all_checks=False, P5_admission=False))
    assert gate.main([]) == 2
