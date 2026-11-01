"""RAM numeric mocks only: never native physics, prepare, neural or model calls."""
from copy import deepcopy
import hashlib
from pathlib import Path
import time
import numpy as np
import pytest
from experiments.spectral_reachability import moving_overlay_capability_probe_physical_20261027_v1 as m
from test_p1_moving_overlay_recompile_physical_20261027_v1 import (
    Env, MemoryStore as OldMemoryStore, fixture as base_fixture, fake_place, h, png)


class MemoryStore(OldMemoryStore):
    def check(self, size=4096, final=False):
        m.require(self.used+size <= m.CAP-(0 if final else m.FINAL) and self.free >= m.FLOOR+size,
                  'RAM mocked cap/floor')

    def write(self, name, value, append=False, final=False):
        self.check(len(value), final)
        assert append or name not in self.files
        if append:
            self.files.setdefault(name, bytearray()).extend(value)
        else: self.files[name] = value
        self.used += len(value)
        self.hashers.setdefault(name, hashlib.sha256()).update(value)
        return self.hashers[name].hexdigest()


def fixture():
    original, data, images = base_fixture(); anchors = []
    for i, plan in enumerate(original['plans']):
        env = Env(); root = data['roots'][plan['root_slot']]
        env.reset(seed=root['slot']['reset_seed'])
        from test_p1_constructive_moving_history_physical_20261027_v1 import REQUEST
        obs, _ = env.reset(seed=0, options=dict(state=REQUEST.copy(), goal_state=env.goal_state.copy()))
        events = []
        def capture(phase, step):
            for index, flag in enumerate((False, True, False)):
                env.with_target = flag; before = m.base.invariant(env)
                image = env.render(); snapshot = m.base.invariant(env)
                receipt = dict(phase=phase, step=step, condition='on' if flag else 'off', overlay_index=index,
                               png=f'b{i}_{phase}_{step:02}_{index}.png', file_sha256=hashlib.sha256(png(image)).hexdigest(),
                               pixel_sha256=m.base.moving.image_sha(image), state=np.asarray(obs['state']).tolist(),
                               snapshot=snapshot['snapshot'])
                events.append(dict(receipt=receipt, pre=before, post=snapshot))
            env.with_target = False
        capture('warmup', 0)
        for step, command in enumerate(m.base.moving.warmup_commands(), 1):
            obs, *_ = env.step(command.copy()); capture('warmup', step)
        m.base.moving.attach_goal(env, root['goal'], root['packets'][plan['condition']]['goal_rgb'])
        capture('plan', 0)
        for step, command in enumerate(m.base.previous.commands(plan['commands'])[:5], 1):
            obs, *_ = env.step(command.copy()); capture('plan', step)
        selected = next(e['receipt'] for e in events if e['receipt']['phase'] == 'plan' and
                        e['receipt']['step'] == 5 and e['receipt']['condition'] == plan['condition'])
        cp = {k: deepcopy(selected[k]) for k in ('step', 'png', 'file_sha256', 'pixel_sha256', 'state', 'snapshot')}
        anchors.append(dict(index=i, checkpoint=cp, trace=events))
    p = dict(schema=m.NAME+'.prepare', contract=m.contract(), plans=original['plans'], anchors=anchors,
             bindings={}, physical_report_sha256=h('synthetic physical report'),
             planner_report_sha256=original['planner_report_sha256'])
    return p, data, images


def execute(monkeypatch, *, p=None, factory=Env, started=None, store=None):
    fp, data, images = fixture(); p = fp if p is None else p
    store = MemoryStore() if store is None else store; store.free = m.FLOOR+m.CAP
    envs = []
    def make():
        e = factory(); envs.append(e); return e
    monkeypatch.setattr(m.base, 'place', fake_place)
    monkeypatch.setattr(m.base, 'validate_env', lambda _: None)
    monkeypatch.setattr(m.base, 'old_image', lambda ref: images[ref['path']].copy())
    monkeypatch.setattr(m, 'sha', lambda p: hashlib.sha256(store.files[Path(p).name]).hexdigest())
    # Any old experimental execution/model selector/witness is forbidden.
    def forbidden(*a, **kw): raise AssertionError('old execution/selection forbidden')
    monkeypatch.setattr(m.base, 'execute', forbidden)
    monkeypatch.setattr(m.base, 'episode', forbidden)
    monkeypatch.setattr(m.base.moving, 'witness', forbidden)
    report = m.execute(p, data, store, make, authorization=m.AUTH, prepare_sha256=h('synthetic prepare'), started=started)
    return report, store, envs


def test_fixed_dictionary_raw_symmetric_no_tanh_or_outcome():
    bank = m.probes()
    assert [p['name'] for p in bank] == list(m.PROBE_NAMES)
    arrays = [np.asarray(p['commands'], dtype=np.float32) for p in bank]
    assert all(a.shape == (5, 2) and a.dtype == np.float32 for a in arrays)
    assert not arrays[0].any()
    assert np.array_equal(arrays[1], -arrays[2]) and np.array_equal(arrays[3], -arrays[4])
    assert arrays[1][0, 0] == np.float32(.1) and arrays[3][0, 1] == np.float32(.1)
    assert m.contract()['fixed_denominator'] == 40 and m.contract()['independent_roots'] == 2


def test_full40_exact_replay_same_anchor_all7_responses_and_counts(monkeypatch):
    p, _, _ = fixture(); report, store, envs = execute(monkeypatch, p=p)
    assert report['status'] == 'capability_probes_recorded', report['errors']
    assert len(report['rows']) == len(envs) == 40 and report['fixed_denominator'] == 40
    assert report['counts'] == {k: dict(attempted=v, completed=v) for k, v in m.contract()['caps'].items()}
    assert len([n for n in store.files if n.endswith('.png')]) == 2640
    assert report['new_neural_calls'] == 0 and report['P1_admission'] is False
    for i, (row, env) in enumerate(zip(report['rows'], envs, strict=True)):
        assert row['start_verified'] and len(row['responses']) == 5 and row['status'] == 'probe_completed'
        assert env.closed and env.with_target is False and env.dispatch_flags == [False]*20
        assert np.array_equal(np.asarray(env.commands[10:15]), p['plans'][i//5]['commands'][:5])
        assert np.array_equal(np.asarray(env.commands[15:]), m.probes()[i % 5]['commands'])
        assert row['start_state'] == p['anchors'][i//5]['checkpoint']['state']
        for response in row['responses']:
            assert len(response['state']) == len(response['delta_native7']) == 7
            assert np.array_equal(np.asarray(response['state'])-row['start_state'], response['delta_native7'])
        assert '_set_state' not in env.__dict__ and 'reset' not in env.__dict__
    previous = None
    for event in store.events:
        assert event['previous_sha256'] == previous
        assert event['event_sha256'] == m.base.source.digest({k: v for k, v in event.items() if k != 'event_sha256'})
        previous = event['event_sha256']
    assert report['journal_chain_tail'] == previous
    assert report['journal_sha256'] == hashlib.sha256(store.files['journal.jsonl']).hexdigest()


@pytest.mark.parametrize('field', ['state', 'snapshot', 'pixel_sha256', 'file_sha256', 'pre', 'nonselected_on'])
def test_anchor_or_any_triplet_mismatch_blocks_suffix_keeps40(monkeypatch, field):
    p, _, _ = fixture()
    if field == 'state': p['anchors'][0]['checkpoint']['state'][0] += 1e-9
    elif field == 'snapshot': p['anchors'][0]['checkpoint']['snapshot']['bodies']['block']['angular_velocity'] += 1e-9
    elif field == 'pre': p['anchors'][0]['trace'][0]['pre']['goal_pose'][0] += 1e-9
    elif field == 'nonselected_on': p['anchors'][0]['trace'][1]['receipt']['pixel_sha256'] = h('wrong on')
    else: p['anchors'][0]['checkpoint'][field] = h('wrong')
    r, s, envs = execute(monkeypatch, p=p)
    assert r['status'] == 'partial_or_failed' and len(r['rows']) == 40
    assert all(len(e.commands) == 15 for e in envs[:5])
    assert all(not row['start_verified'] and row['responses'] == [] for row in r['rows'][:5])
    assert all(row['status'] == 'probe_completed' for row in r['rows'][5:])
    assert 'b0_plan_05_0.png' in s.files


@pytest.mark.parametrize('fault', ['throw', 'substitute', 'truncate', 'terminal'])
def test_suffix_failure_or_stop_keeps_actual_used_and_no_padding(monkeypatch, fault):
    r, _, envs = execute(monkeypatch, factory=lambda: Env(fault=fault, at=16))
    assert len(r['rows']) == 40
    assert all(len(e.commands) == (20 if fault == 'substitute' and i % 5 == 0 else 16)
               for i, e in enumerate(envs))
    if fault in ('truncate', 'terminal'):
        assert r['status'] == 'capability_probes_recorded', r['errors']
        assert all(row['status'] == 'native_stopped' and len(row['responses']) == 1 for row in r['rows'])
    else:
        assert r['status'] == 'partial_or_failed'
        assert all(row['status'] == ('probe_completed' if fault == 'substitute' and i % 5 == 0 else 'engineering_failed')
                   for i, row in enumerate(r['rows']))
    if fault == 'throw': assert r['counts']['public_step'] == dict(attempted=640, completed=600)


def test_expired_budget_no_constructor_and_all_missing_null(monkeypatch):
    r, _, envs = execute(monkeypatch, started=time.monotonic()-901)
    assert not envs and r['status'] == 'partial_or_failed' and len(r['rows']) == 40
    assert all(row['start_state'] is None and row['responses'] == [] for row in r['rows'])
    assert r['counts']['constructor']['attempted'] == 0


def test_unauthorized_before_verify_or_factory(monkeypatch):
    monkeypatch.setattr(m, 'verify', lambda *a: pytest.fail('unauthorized verify'))
    with pytest.raises(ValueError): m.run('unused', h('x'), '/unused', lambda: pytest.fail('factory'), authorization='wrong')


def test_meter_failed_attempt_retained_and_cap_predispatch():
    store = MemoryStore(); store.free = m.FLOOR+m.CAP; meter = m.Meter(store)
    with pytest.raises(RuntimeError): meter.call('public_step', lambda: (_ for _ in ()).throw(RuntimeError('fault')))
    assert meter.counts['public_step'] == dict(attempted=1, completed=0)
    meter.counts['constructor']['attempted'] = 40
    with pytest.raises(ValueError): meter.call('constructor', lambda: pytest.fail('cap bypass'))


@pytest.mark.parametrize('mutation', ['identity', 'checkpoint', 'history', 'count'])
def test_validate_saved_rows_missing_or_wrong_binding_failclosed(mutation):
    p, _, _ = fixture(); plans = p['plans']
    rows = []
    for i, plan in enumerate(plans):
        rows.append(dict(index=i, **{k: plan[k] for k in ('root_slot', 'condition', 'arm', 'selection_digest', 'task_id')},
                         initial_exact=True, warmup_returned=10, own_returned=10, first_error=None,
                         secondary_errors=[], checkpoints=[p['anchors'][i]['checkpoint']],
                         trajectory=[e['receipt'] for e in p['anchors'][i]['trace'] if
                                     e['receipt']['overlay_index'] == (1 if plan['condition'] == 'on' else 0)]))
    report = dict(schema=m.base.NAME+'.report', status='moving_overlay_recompile_physical_recorded', errors=[],
                  inputs_final_verified=True, fixed_denominator=8, new_neural_calls=0, rows=rows)
    m.validate_rows(report, plans)
    if mutation == 'identity': rows[0]['selection_digest'] = h('wrong')
    elif mutation == 'checkpoint': rows[0]['checkpoints'] = []
    elif mutation == 'history': rows[0]['trajectory'].pop(0)
    else: rows.pop()
    with pytest.raises(ValueError): m.validate_rows(report, plans)


@pytest.mark.parametrize('mutation', [None, 'actual', 'requested', 'stop', 'missing', 'duplicate'])
def test_original_raw_history_prefix_journal_join(mutation):
    p, _, _ = fixture(); plan = p['plans'][0]; events = []
    for phase, commands in (('warmup', m.base.moving.warmup_commands()),
                            ('plan', m.base.previous.commands(plan['commands'])[:5])):
        for step, c in enumerate(commands, 1):
            events.append(dict(kind='returned_transition', branch=0, phase=phase, step=step,
                               requested_bits=c.tobytes().hex(), actual_bits=c.tobytes().hex(),
                               actual_dtype=c.dtype.str, terminated=False, truncated=False))
    if mutation is None:
        m.validate_transitions(events, 0, plan); return
    if mutation == 'actual': events[-1]['actual_bits'] = '00'*8
    elif mutation == 'requested': events[-1]['requested_bits'] = '00'*8
    elif mutation == 'stop': events[-1]['truncated'] = True
    elif mutation == 'missing': events.pop()
    else: events.append(deepcopy(events[-1]))
    with pytest.raises(ValueError): m.validate_transitions(events, 0, plan)


@pytest.mark.parametrize('mutation', [None, 'external_sha', 'binding', 'planner', 'pixel'])
def test_material_complete_mock_metadata_closure_no_prepare_or_native(monkeypatch, mutation):
    p, data, _ = fixture(); physical = Path('/RAM/physical'); physical_sha = h('physical')
    eb = dict(prepared='/RAM/old_prepare.json', prepare_sha256=h('old prepare'))
    events = []; rows = []; images = {}
    for i, plan in enumerate(p['plans']):
        anchor = p['anchors'][i]
        row = dict(index=i, **{k: plan[k] for k in ('root_slot', 'condition', 'arm', 'selection_digest', 'task_id')},
                   initial_exact=True, warmup_returned=10, own_returned=10, first_error=None, secondary_errors=[],
                   checkpoints=[deepcopy(anchor['checkpoint'])], trajectory=[])
        for event in anchor['trace']:
            event = deepcopy(event); event.update(kind='fresh_observation', branch=i)
            events.append(event); r = event['receipt']
            images[str(physical/r['png'])] = r['pixel_sha256']
            if r['overlay_index'] == (1 if plan['condition'] == 'on' else 0): row['trajectory'].append(deepcopy(r))
        for phase, commands in (('warmup', m.base.moving.warmup_commands()),
                                ('plan', m.base.previous.commands(plan['commands'])[:5])):
            for step, c in enumerate(commands, 1):
                events.append(dict(kind='returned_transition', branch=i, phase=phase, step=step,
                                   requested_bits=c.tobytes().hex(), actual_bits=c.tobytes().hex(),
                                   actual_dtype=c.dtype.str, terminated=False, truncated=False))
        rows.append(row)
    report = dict(schema=m.base.NAME+'.report', status='moving_overlay_recompile_physical_recorded', errors=[],
                  inputs_final_verified=True, fixed_denominator=8, new_neural_calls=0, rows=rows,
                  execution_binding=deepcopy(eb), execution_binding_sha256=h('binding'),
                  prepare_sha256=eb['prepare_sha256'], planner_report_sha256=p['planner_report_sha256'],
                  source_report_sha256=m.base.SOURCE_SHA, journal_sha256=h('journal'), journal_chain_tail=h('tail'))
    original = dict(plans=p['plans'], bindings={}, planner_report_sha256=p['planner_report_sha256'],
                    source_report_sha256=m.base.SOURCE_SHA)
    if mutation == 'binding': eb['prepare_sha256'] = h('not same')
    elif mutation == 'planner': report['planner_report_sha256'] = h('not same')
    elif mutation == 'pixel': images[next(iter(images))] = h('not same')
    monkeypatch.setattr(m, 'read', lambda path: report if Path(path).name == 'report.json' else eb)
    monkeypatch.setattr(m, 'sha', lambda path: physical_sha if Path(path).name == 'report.json' else h('binding'))
    monkeypatch.setattr(m.base, 'verify', lambda *a: (original, data))
    monkeypatch.setattr(m.base.previous, 'journal', lambda *a: events)
    monkeypatch.setattr(m, 'own_files', lambda: [])
    def bind(bindings, path, expected=None):
        path = Path(path); bindings[str(path)] = expected or h(path); return path
    monkeypatch.setattr(m, 'bind', bind)
    monkeypatch.setattr(m.base, 'old_image', lambda ref: images[ref['path']])
    monkeypatch.setattr(m.base.moving, 'image_sha', lambda image: image)
    if mutation is not None:
        with pytest.raises(ValueError): m.material(physical, h('bad') if mutation == 'external_sha' else physical_sha)
    else:
        prepared, _ = m.material(physical, physical_sha)
        assert len(prepared['plans']) == len(prepared['anchors']) == 8
        assert all(len(a['trace']) == 51 for a in prepared['anchors'])
        assert str(physical/'journal.jsonl') in prepared['bindings']


def test_paired_descriptive_vectors_keep_zero_negative_and_missing_null():
    rows = []
    for slot in (0, 1):
        for c in m.base.CONDITIONS:
            for ai, arm in enumerate(m.base.ARMS):
                for probe in m.PROBE_NAMES:
                    rows.append(dict(root_slot=slot, condition=c, arm=arm, probe=probe,
                                     start_verified=True, start_state=[float(ai)]*7,
                                     responses=[dict(step=1, delta_native7=[-2. if ai else 1.]*7),
                                                dict(step=2, delta_native7=[0.]*7)]))
    paired = m.paired_responses(rows)
    assert len(paired) == 20
    assert paired[0]['D2P_minus_GN_prefix_state_native7'] == [1.]*7
    assert paired[0]['steps'][0]['D2P_minus_GN_response_delta_native7'] == [-3.]*7
    assert paired[0]['steps'][1]['D2P_minus_GN_response_delta_native7'] == [0.]*7
    assert paired[0]['steps'][2]['D2P_minus_GN_response_delta_native7'] is None
    rows[0]['start_verified'] = False
    paired = m.paired_responses(rows)
    assert paired[0]['D2P_minus_GN_prefix_state_native7'] is None
    assert paired[0]['steps'][0]['D2P_minus_GN_response_delta_native7'] is None
