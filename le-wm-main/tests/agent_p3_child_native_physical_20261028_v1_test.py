"""RAM/saved metadata only. No native factory, model, or physical replay."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_child_native_physical_20261028_v1 as m


@pytest.fixture(scope='module')
def saved():
    cp = m.READ(m.ROOT/'results/agent_p3_child_program_cuda_20261028_v1_prepare_MAIN01.json', 64*1024**2)
    receipt = m.READ(m.STRICT, 128*1024**2)
    pp = m.old.read(m.P4_PREPARE); report = m.old.read(m.P4_OUTPUT/'report.json')
    manifest = m.old.read(pp['original_manifest'])
    data = m.old.kernel.load_source(manifest['sourcepack'], m.old.SOURCE_SHA,
                                   manifest['source_prepared'], manifest['source_prepare_sha256'])
    events = m.journal(m.P4_OUTPUT/'journal.jsonl', m.old.CAP)
    return cp, receipt, pp, report, events, data


@pytest.fixture(scope='module')
def plans(saved):
    cp, receipt, pp, report, events, data = saved
    return m.project(cp, receipt, pp, report, events, data['references'])


def test_real_saved_full_roster_and_budget(plans):
    assert len(plans) == 48
    assert [sum(p['route'] == k for p in plans) for k in ('execute', m.compiler.BLOCKED, 'not_reached')] == [41, 5, 2]
    p = m.profile(plans)
    assert p['prefix_public_cap'] == 385
    assert p['recovery_public_cap'] == 795 and p['own_public_cap'] == 205
    assert p['caps'] == dict(constructor=41, close=41, reset=123, set_state=246, goal_render=123,
                            goal_setter=164, goal_bind=41, public_step=1000, physics_step=10246, explicit_render=1082)
    assert p['neural_calls'] == 0
    assert p['caps']['explicit_render']*m.PNG_CAP+m.JOURNAL_CAP+8*1024**2 < m.CAP-m.FINAL


@pytest.mark.parametrize('field', ['source_report_sha256', 'task_binding_sha256'])
def test_real_ancestry_tamper(saved, monkeypatch, field):
    cp, receipt, pp, report, events, data = deepcopy(saved)
    cp['children'][0]['ancestry'][field] = '0'*64
    monkeypatch.setattr(m, 'png', lambda f, d: m.native_frame(f))
    with pytest.raises(ValueError): m.project(cp, receipt, pp, report, events, data['references'])


@pytest.mark.parametrize('mutation', ['prefix', 'raw_history', 'selected_context', 'missing', 'order', 'native_snapshot', 'goal'])
def test_real_projection_rejects_tamper(saved, monkeypatch, mutation):
    cp, receipt, pp, report, events, data = deepcopy(saved)
    monkeypatch.setattr(m, 'png', lambda f, d: m.native_frame(f))
    if mutation == 'prefix': report['rows'][0]['executed_raw'][0][0] = 0.
    if mutation == 'raw_history': cp['children'][0]['raw_history10'][0][1] = 0.
    if mutation == 'selected_context': receipt['rows'][0]['selected']['parent_context_sha256'] = '0'*64
    if mutation == 'missing': receipt['rows'].pop()
    if mutation == 'order': receipt['rows'][0], receipt['rows'][1] = receipt['rows'][1], receipt['rows'][0]
    if mutation in ('native_snapshot', 'goal'):
        e = next(e for e in events if e['kind'] == 'fresh_observation' and e['receipt']['phase'] == 'plan')
        if mutation == 'goal': e['receipt']['goal'][0] += 1.
        else: e['receipt']['snapshot']['bodies']['block']['unwrapped_angle'] += 2*np.pi
    with pytest.raises((ValueError, KeyError)): m.project(cp, receipt, pp, report, events, data['references'])


def test_native_modulo_saved_regression(saved):
    frame = next(e['receipt'] for e in saved[4] if e['kind'] == 'fresh_observation')
    assert frame['snapshot']['bodies']['block']['unwrapped_angle'] == -5.587802913760431
    assert frame['state'][4] == .6953823934191554
    m.native_frame(frame)
    bad = deepcopy(frame); bad['snapshot']['bodies']['block']['unwrapped_angle'] += 2*np.pi
    with pytest.raises(ValueError, match='body float64 bits'): m.native_frame(bad)
    bad = deepcopy(frame); bad['state'][4] += 2*np.pi
    with pytest.raises(ValueError, match='modulo'): m.native_frame(bad)
    bad = deepcopy(frame); bad['snapshot']['bodies']['block']['float64_bits'] = '00'*72
    with pytest.raises(ValueError): m.native_frame(bad)


@pytest.mark.parametrize('change', ['angle', 'velocity', 'force', 'goal', 'pose', 'png'])
def test_whole_snapshot_no_visual_seam_substitution(saved, change):
    f = next(e['receipt'] for e in saved[4] if e['kind'] == 'fresh_observation')
    b = deepcopy(f)
    if change == 'angle': b['snapshot']['bodies']['block']['unwrapped_angle'] += 2*np.pi
    if change == 'velocity': b['snapshot']['bodies']['block']['velocity'][0] += 1
    if change == 'force': b['snapshot']['bodies']['agent']['force'][0] += 1
    if change == 'goal': b['goal'][0] += 1
    if change == 'pose': b['goal_pose'][0] += 1
    if change == 'png': b['file_sha256'] = '0'*64
    with pytest.raises(ValueError): m.same_frame(b, f)


@pytest.mark.parametrize('value', [np.zeros((4, 2)), np.full((5, 2), 1.1), np.full((5, 2), np.nan), np.full((5, 2), .1)])
def test_raw_refuses_scale_round_nonfinite_shape(value):
    with pytest.raises(ValueError): m.raw(value, 5)


def test_native7_reward_not_native_success():
    goal = np.zeros(7); state = goal.copy(); state[5] = 1000
    result = m.metric(goal, state)
    assert result['native_success'] is True
    assert result['native7_reward'] == -1000 and result['joint_xy_error'] == 0
    state[2] = 21
    assert m.metric(goal, state)['native_success'] is False


def test_default_run_refused_before_verify_or_native(monkeypatch, tmp_path):
    def forbidden(*a, **k): pytest.fail('authorization must precede any saved verification or physics')
    monkeypatch.setattr(m, 'verify', forbidden)
    with pytest.raises(ValueError, match='authorization'): m.run('/abs', '0'*64, tmp_path/'out', '')
    assert not (tmp_path/'out').exists()


def test_source_preflight_before_consumers(monkeypatch):
    calls = []
    monkeypatch.setattr(m, 'compiler', SimpleNamespace(contract=SimpleNamespace(check_sources=lambda p: None),
                                                       consume=lambda *a: calls.append('compiler')))
    monkeypatch.setattr(m, 'sha', lambda p: 'bad')
    with pytest.raises(ValueError, match='external real'): m.material()
    assert calls == []


@pytest.mark.parametrize('which', ['compiler', 'p4', 'receipt'])
def test_mandatory_consumers_and_complete_receipt(monkeypatch, which):
    calls = []
    hashes = {str(m.STRICT): m.STRICT_SHA, str(m.COMPILER_OUTPUT/'report.json'): m.COMPILER_SHA,
              str(m.P4_OUTPUT/'report.json'): m.P4_SHA, str(m.P4_PREPARE): m.P4_PREPARE_SHA}
    monkeypatch.setattr(m, 'sha', lambda p: hashes[str(p)])
    def compileconsume(*a):
        calls.append('compiler')
        if which == 'compiler': raise ValueError('original compiler rejected')
        return {'full': ['all rows']}
    def p4consume(*a):
        calls.append('p4'); raise ValueError('original P4 rejected')
    monkeypatch.setattr(m, 'compiler', SimpleNamespace(contract=SimpleNamespace(check_sources=lambda p: None), consume=compileconsume))
    monkeypatch.setattr(m, 'p4', SimpleNamespace(consume=p4consume))
    monkeypatch.setattr(m, 'READ', lambda *a: {'summary': True} if which == 'receipt' else {'full': ['all rows']})
    with pytest.raises(ValueError): m.material()
    assert calls == (['compiler', 'p4'] if which == 'p4' else ['compiler'])


def test_real_saved_material_projection_with_RAM_consumer_responses(saved, monkeypatch):
    # Source closure and schema smoke only. No actual full consumers/experiments invoked here.
    original_compiler = m.compiler
    monkeypatch.setattr(m, 'compiler', SimpleNamespace(contract=original_compiler.contract, base=original_compiler.base,
        consume=lambda *a: saved[1], READY=original_compiler.READY, BLOCKED=original_compiler.BLOCKED, FINAL=original_compiler.FINAL))
    monkeypatch.setattr(m, 'p4', SimpleNamespace(consume=lambda *a: dict(status='saved_multiboundary_verified_NOT_scientific_pass')))
    p = m.material()
    assert p['profile']['caps']['public_step'] == 1000 and len(p['rows']) == 48
    assert p['sources'][str(m.STRICT)] == m.STRICT_SHA
    assert p['sources'][str(m.P4_PREPARE)] == m.P4_PREPARE_SHA
    assert all(p[k] is False for k in m.FLAGS)


def test_store_journal_exclusive_and_tamper(tmp_path):
    store = m.Store(tmp_path/'own'); store.emit('one', test=True); store.emit('two')
    assert len(m.journal(store.path/'journal.jsonl', m.JOURNAL_CAP)) == 2
    with pytest.raises(ValueError): m.Store(store.path)
    with pytest.raises(ValueError): store.write('../escape', b'a')
    with pytest.raises(ValueError): store.write('one.json', b'x'*(m.CAP-m.FINAL+1))
    # RAM test corruption is deliberate, never changes any sealed artifact.
    path = store.path/'journal.jsonl'; content = path.read_bytes().replace(b'"one"', b'"bad"')
    with path.open('wb') as f: f.write(content)
    with pytest.raises(ValueError, match='hash chain'): m.journal(path, m.JOURNAL_CAP)


def test_store_recovery_checks_before_next_dispatch(saved, tmp_path):
    f = deepcopy(next(e['receipt'] for e in saved[4] if e['kind'] == 'fresh_observation'))
    s = m.Store(tmp_path/'recovery'); s.recovery = [f]; s.recovered = 0
    s.emit('fresh_observation', receipt=f, episode_id='ram')
    assert s.recovered == 1
    with pytest.raises(ValueError): s.emit('fresh_observation', receipt=f, episode_id='ram')


def test_meter_attempts_not_refunded_and_unknown_axis(tmp_path):
    s = m.Store(tmp_path/'meter'); meter = m.Meter(s, dict(public_step=1))
    def fail(): raise RuntimeError('native fault')
    with pytest.raises(RuntimeError): meter.call('public_step', fail)
    assert meter.counts == dict(public_step=dict(attempted=1, completed=0))
    assert len(meter.pending) == 1
    with pytest.raises(ValueError): meter.call('public_step', lambda: None)
    with pytest.raises(ValueError): meter.call('encoder', lambda: None)


@pytest.fixture
def ram_saved(plans, monkeypatch):
    """One RAM branch returned, all other executable slots retained not_started.

    Synthetic operation ledger is test data, NEVER execution/scientific evidence.
    """
    monkeypatch.setattr(m, 'png', lambda f, d: m.native_frame(f))
    p = dict(rows=deepcopy(plans), profile=m.profile(plans)); plan = p['rows'][0]
    rows = [m.new_row(i, x) for i, x in enumerate(p['rows'])]; row = rows[0]; eid = row['episode_id']
    b = plan['child']['boundary']; events = []; serial = 0
    def emit(kind, **kw):
        e = dict(index=len(events), kind=kind, episode_id=eid, **deepcopy(kw)); events.append(e); return e
    def op(resource, region='setup'):
        nonlocal serial
        args = dict(operation_id=serial, resource=resource, units=1, region=region); serial += 1
        emit('operation_attempted', **args)
        if resource == 'public_step':
            for _ in range(10): op('physics_step', region)
        emit('operation_completed', **args)
    emit('branch_started', ordinal=0, plan_sha256=m.digest(plan))
    for k, n in dict(constructor=1, reset=3, set_state=6, goal_render=3, goal_setter=4, goal_bind=1, physics_step=6).items():
        for _ in range(n): op(k, 'goal_bind' if k == 'goal_bind' else 'setup')
    frames = []
    def capture(f, phase, j):
        f = deepcopy(f); f.update(path=f'/ram/{eid}_{phase}_{j}.png', phase=phase, step=j, observed_event_index=len(events))
        frames.append(f); emit('fresh_observation', receipt=f); op('explicit_render', 'warmup' if phase == 'warmup' else 'suffix')
        return f
    for j in range(11):
        if j:
            op('public_step', 'warmup')
        capture(plan['source_frames'][j], 'warmup', j)
        if j: emit('warmup_return', step=j, command=m.old.kernel.moving.warmup_commands()[j-1].tolist(), terminated=False, truncated=False)
    capture(plan['source_frames'][11], 'plan', 0)
    commands = np.concatenate([m.raw(plan['child']['executed_prefix'], b), m.raw(plan['selected']['raw5'], 5)])
    row.update(warmup_returned=10, prefix_returned=b, own_returned=5, executed_raw=commands.tolist(), initial_exact=True)
    for j, command in enumerate(commands):
        phase, n = ('prefix', j+1) if j < b else ('own', j-b+1)
        if j == b:
            row['boundary'] = deepcopy(frames[-1])
            emit('child_boundary_sealed', context_sha256=plan['child']['context_sha256'], raw_history10=plan['child']['raw_history10'],
                 selected_sha256=m.digest(plan['selected']), boundary=row['boundary'])
        emit('authorized_dispatch', phase=phase, step=n, command=command.tolist())
        op('public_step', 'prefix' if phase == 'prefix' else 'suffix')
        f = plan['source_frames'][12+j] if j < b else plan['source_frames'][-1]
        emit('step_return', phase=phase, step=n, command=command.tolist(), state=f['state'],
             reward=m.metric(f['goal'], f['state'])['native7_reward'], terminated=False, truncated=False)
        capture(f, phase, n)
    row.update(PNG=len(frames), last_observation=frames[-1], final_metrics=m.metric(frames[-1]['goal'], frames[-1]['state']),
               terminated=False, truncated=False, status='returned5')
    emit('cleanup_attempted'); emit('cleanup_completed')
    counts, pending = m.ledger(events, p['profile']['caps'])
    emit('branch_finished', row=row, cleanup_counts=counts['close'])
    report = dict(profile=p['profile'], fixed_denominator=48, child_denominator=24, rows=rows, resources=counts, pending=pending,
                  status='engineering_failed', errors=[dict(reason='RAM rest not_started')], secondary_errors=[], wall_seconds=1., **m.FLAGS)
    return p, report, events


def test_ram_saved_valid_branch_not_entire_physics(ram_saved):
    p, r, es = ram_saved; result = m.validate_saved(p, r, es, Path('/ram'))
    assert result['complete'] is False and len(result['rows']) == 48
    assert result['rows'][0]['own_returned'] == 5
    assert result['rows'][2]['final_metrics'] is None


@pytest.mark.parametrize('mutation', ['denominator', 'gate', 'native_metric', 'returned_raw', 'dispatch_raw', 'context',
                                    'raw_history', 'snapshot', 'goal', 'earlystop', 'resource', 'pending', 'drop', 'borrow_png', 'prefix_cost'])
def test_saved_consumer_negative_boundaries(ram_saved, mutation):
    p, r, es = ram_saved
    row = r['rows'][0]
    if mutation == 'denominator': r['fixed_denominator'] = 41
    if mutation == 'gate': r['P3_admission'] = True
    if mutation == 'native_metric': row['final_metrics']['native7_reward'] += 1.
    if mutation == 'returned_raw': next(e for e in es if e['kind'] == 'step_return')['command'][0] += .01
    if mutation == 'dispatch_raw': next(e for e in es if e['kind'] == 'authorized_dispatch')['command'][0] += .01
    if mutation == 'context': next(e for e in es if e['kind'] == 'child_boundary_sealed')['context_sha256'] = '0'*64
    if mutation == 'raw_history': next(e for e in es if e['kind'] == 'child_boundary_sealed')['raw_history10'] = [[0., 0.]]*10
    if mutation == 'snapshot': next(e for e in es if e['kind'] == 'fresh_observation')['receipt']['snapshot']['bodies']['block']['unwrapped_angle'] += 2*np.pi
    if mutation == 'goal': [e for e in es if e['kind'] == 'fresh_observation'][-1]['receipt']['goal'][0] += 1.
    if mutation == 'earlystop': next(e for e in es if e['kind'] == 'step_return')['truncated'] = True
    if mutation == 'resource': r['resources']['public_step']['completed'] -= 1
    if mutation == 'pending': r['pending'].append(dict(operation_id=-1))
    if mutation == 'drop': r['rows'].pop()
    if mutation == 'borrow_png': [e for e in es if e['kind'] == 'fresh_observation'][-1]['receipt']['path'] = [e for e in es if e['kind'] == 'fresh_observation'][0]['receipt']['path']
    if mutation == 'prefix_cost': row['prefix_returned'] -= 1
    with pytest.raises((ValueError, KeyError)): m.validate_saved(p, r, es, Path('/ram'))


def test_ram_earlystop_no_padding(ram_saved):
    p, r, es = ram_saved; row = r['rows'][0]
    # A truncated first own step is a valid native stop. Keep genuine fixture cost, no padding/fake returns.
    first = next(e for e in es if e['kind'] == 'step_return' and e['phase'] == 'own')
    stopframe = next(e for e in es if e['kind'] == 'fresh_observation' and e['receipt']['phase'] == 'own')
    cut = es.index(stopframe)+1
    remaining = [e for e in es[cut:] if e['kind'] in ('cleanup_attempted', 'cleanup_completed')]
    es[:] = es[:cut]+remaining
    # capture's explicit_render event in this RAM fixture follows observation; retain it as well.
    for kind in ('operation_attempted', 'operation_completed'):
        es.insert(cut, dict(index=0, kind=kind, episode_id=row['episode_id'], operation_id=-1,
                            resource='explicit_render', units=1, region='suffix'))
        cut += 1
    # Renumber fixture-only operation IDs and observation indices; no real artifact changed.
    mapping = {}; count = 0
    for j, e in enumerate(es):
        e['index'] = j
        if e['kind'] == 'operation_attempted': mapping[e['operation_id']] = count; e['operation_id'] = count; count += 1
        elif e['kind'] == 'operation_completed': e['operation_id'] = mapping[e['operation_id']]
        elif e['kind'] == 'fresh_observation': e['receipt']['observed_event_index'] = j
    first['truncated'] = True
    fs = [e['receipt'] for e in es if e['kind'] == 'fresh_observation']
    row.update(own_returned=1, executed_raw=row['executed_raw'][:-4], PNG=len(fs), last_observation=fs[-1], status='native_stopped', truncated=True)
    r['resources'], r['pending'] = m.ledger(es, p['profile']['caps'])
    es.append(dict(index=len(es), kind='branch_finished', episode_id=row['episode_id'], row=deepcopy(row), cleanup_counts=r['resources']['close']))
    assert m.validate_saved(p, r, es, Path('/ram'))['rows'][0]['own_returned'] == 1


@pytest.mark.parametrize('stop', [None, 2])
def test_RAM_execute_raw5_only_and_earlystop_no_H3_tail(plans, monkeypatch, stop):
    plan = deepcopy(next(p for p in plans if p['mode'] == 'adaptive' and p['route'] == 'execute'))
    row = m.new_row(0, plan); calls = []; emitted = []
    store = SimpleNamespace(recovered=0, emit=lambda kind, **kw: emitted.append((kind, kw)))
    def setup(env, ref, source_plan, row, meter, store):
        row['last_observation'] = deepcopy(plan['source_frames'][11]); store.recovered = 12
        return True
    def stub_step(env, command, row, meter, store, phase, ordinal, goal, goal_rgb):
        calls.append((phase, ordinal, command.copy()))
        f = deepcopy(plan['source_frames'][11+ordinal] if phase == 'prefix' else plan['source_frames'][-1])
        row['last_observation'] = f; row[phase+'_returned'] += 1
        if phase == 'prefix': store.recovered += 1
        return f, False, phase == 'own' and ordinal == stop
    monkeypatch.setattr(m, 'p4', SimpleNamespace(setup=setup)); monkeypatch.setattr(m, 'step', stub_step)
    m.execute(object(), plan, dict(goal=None, goal_rgb=None), {}, row, SimpleNamespace(region='setup'), store)
    own = [x for x in calls if x[0] == 'own']; count = 5 if stop is None else stop
    assert len(own) == count and row['own_returned'] == count
    assert row['status'] == ('returned5' if stop is None else 'native_stopped')
    m.exact(np.asarray([x[2] for x in own]), m.raw(plan['selected']['raw5'], 5)[:count], 'selected raw5 only')
    assert [x[0] for x in emitted] == ['child_boundary_sealed']


def test_first_native_step_fault_keeps_zero_returns(monkeypatch):
    row = dict(episode_id='ram', own_returned=0, executed_raw=[])
    class Sink:
        def __init__(self): self.events = []
        def emit(self, kind, **kw): self.events.append((kind, kw))
    store = Sink()
    def fail(command): raise RuntimeError('native step raised')
    env = SimpleNamespace(step=fail)
    with pytest.raises(RuntimeError): m.step(env, np.zeros(2, np.float32), row, None, store, 'own', 1, None, None)
    assert row['own_returned'] == 0 and row['executed_raw'] == []
    assert [x[0] for x in store.events] == ['authorized_dispatch']


def test_write_new_no_overwrite_relative(tmp_path):
    path = tmp_path/'one.json'; m.write_new(path, dict(ram=True))
    with pytest.raises(ValueError): m.write_new(path, dict(ram=False))
    with pytest.raises(ValueError): m.write_new(Path('relative.json'), {})


def test_import_does_not_load_models():
    source = Path(m.__file__).read_text()
    assert 'import torch' not in source and 'import gym' not in source
    assert 'predict(' not in source and 'encode_observation(' not in source
