"""RAM/NumPy protocol fixtures only; no model, physics or experiment evidence."""
from copy import deepcopy
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_prospective_equal_horizon_20261028_v1 as m


def sources():
    tasks = m.core.read(m.TASK_AUTHORITY)['wiring']['tasks']
    return [dict(source_id=f'known{i}', root_slot=i, task=t,
        recipe_sha256=m.digest('RAM_recipe'), prefix_sha256=m.digest('RAM_prefix'),
        authority_sha256=m.digest('RAM_authority_NOT_native_certification')) for i, t in enumerate(tasks)]


def forward(ledger):
    ledger.call('action_encoder', lambda: None, shape=[1, 3, 10])
    def predict():
        ledger.call('predictor', lambda: None, shape=[1, 3, 192])
        ledger.call('pred_proj', lambda: None, shape=[3, 192])
    ledger.call('predict', predict, shape=[1, 3, 192])


class Signal:
    def __init__(self, task, long=False, high=False, fault=None):
        self.task, self.long, self.high, self.fault = task, long, high, fault

    def signal(self, req, ledger):
        h = req['horizon']
        for _ in range(h):
            forward(ledger)
        def ad():
            for _ in range(h):
                forward(ledger)
            if self.fault == 'AD':
                raise RuntimeError('RAM paid AD failure')
            for _ in range(192):
                ledger.call('vjp', lambda: ledger.call('grad', lambda: ledger.call('raw_backward', lambda: None)), directions=1)
        ledger.call('jacobian', ad)
        j = np.zeros((192, 10*h))
        if not self.high:
            j[0, 0] = 1
            if not self.long or h == 3:
                j[1, 1] = 1
        residual = np.zeros(192)
        residual[:2] = [2, 1]
        out = dict(binding_sha256=req['binding_sha256'], parent_context_sha256=req['parent_context_sha256'],
            horizon=h, nominal=(np.asarray(self.task['goal'])-residual).tolist(),
            jacobian=j.tolist(), producer_sha256=m.digest('RAM_ARRAY_NOT_MODEL'))
        if self.fault == 'signal_parent':
            out['parent_context_sha256'] = m.digest('other')
        if self.fault == 'outcome':
            out['reward'] = 1
        return out


class Backend:
    kind = 'RAM_FIXTURE'
    def __init__(self, geometry='short', fault=None, early=None):
        self.geometry, self.fault, self.early = geometry, fault, early
        self.calls = []
        self.closes = 0

    def start(self, source, mode):
        self.s, self.mode, self.boundary = source, mode, 5
        self.history = np.zeros((10, 2), np.float32).tolist()
        self.calls.append(('start', source['source_id'], mode))
        return dict(RAM=True)

    def encode(self, frame, ledger):
        self.calls.append(('encode', self.mode, self.boundary))
        ledger.call('encoder', lambda: None, shape=[1, 3, 224, 224])
        ledger.call('projector', lambda: None, shape=[1, 192])
        c = deepcopy(self.s['task']['root_context'])
        t = self.s['task']
        c['action_history'] = ((np.asarray(self.history, np.float32)-np.asarray(t['action_mean'], np.float32))/np.asarray(t['action_std'], np.float32)).reshape(2, 10).astype(float).tolist()
        raw = deepcopy(self.history)
        if self.fault == 'history' and self.boundary > 5:
            raw[0][0] += 1
        return dict(source_sha256=m.digest(self.s), mode=self.mode, boundary=self.boundary,
            context=c, rgb_sha256=m.digest(['RAM_RGB', self.mode, self.boundary]),
            pixel_sha256=m.digest('RAM_pixels'), encoding_sha256=m.digest(c),
            raw_history10=raw, observed_index=100*self.boundary, encoded_index=100*self.boundary+1)

    def local(self, task, obs):
        self.calls.append(('signal', self.mode, self.boundary))
        long = self.geometry == 'long' or (self.geometry == 'tail' and self.boundary > 5)
        return Signal(task, long=long, high=self.geometry == 'high', fault=self.fault)

    def compile(self, req, ledger):
        self.calls.append(('compile', self.mode, self.boundary, req['decision']['horizon'], req['decision']['candidates']))
        d = req['decision']
        h, n = d['horizon'], d['candidates']
        ledger.call('program_compile', lambda: None)
        ledger.call('array_svd', lambda: None)
        ledger.call('solve', lambda: None)
        for _ in range(2*n+1):
            ledger.call('replay', lambda: [forward(ledger) for _ in range(h)])
        for _ in range(n):
            ledger.call('metric', lambda: None)
        count = 5 if self.fault == 'prefix_only' else 5*h
        out = dict(request_sha256=m.digest(req), actions=np.full((count, 2), .125, np.float32).tolist(),
            full_replay_sha256=m.digest('RAM_NO_TENSOR_EVIDENCE'))
        if self.fault == 'plan_parent':
            out['request_sha256'] = m.digest('other')
        return out

    def step(self, action, step):
        self.boundary = step
        self.history = self.history[1:]+[deepcopy(action)]
        self.calls.append(('step', self.mode, step))
        return dict(done=step == self.early, metric=dict(RAM_not_native=True), action_sha256=m.digest(action))

    def observe(self):
        return dict(RAM=True)

    def close(self):
        self.closes += 1


def execute(**kwargs):
    backend = Backend(**kwargs)
    events = []
    result = m.RamDriver(sources(), backend, events.append).execute()
    return result, backend, events


def test_full_twenty_both_policies_not_first5():
    r, b, events = execute(geometry='long')
    assert r['complete_public_horizon'] and len(r['rows']) == 4 and b.closes == 4
    assert all(x['actual_public_steps'] == 20 for x in r['rows'])
    for row in r['rows']:
        assert [a['boundary'] for a in row['rounds']] == [5, 10, 15]
        calls = row['resources']['attempted']
        assert calls['jacobian_API'] == 6 and calls['VJP_directions'] == 1152
        assert calls['encoder_calls'] == 4
        if row['mode'] == 'adaptive':
            assert row['rounds'][0]['decision']['horizon'] == 3
            assert len(row['rounds'][0]['program']['actions']) == 15
            assert [a['decision'] for a in row['rounds'][1:]] == [None, None]
            assert calls['predict_API'] == 51 and calls['schedule_decisions'] == 1
        else:
            assert calls['predict_API'] == 51 and calls['schedule_decisions'] == 3
    assert events[0]['kind'] == 'denominator'
    assert r['evidence'] == 'RAM_ONLY_NOT_SCIENCE' and not r['scientific_admission']


def test_short_feedback_and_distinct_caps():
    r, b, _ = execute()
    assert r['complete_public_horizon']
    assert m.arm_caps('fixed')['predict_API'] == 51
    assert m.arm_caps('adaptive')['predict_API'] == 75
    assert all(c[-1] == 4 for c in b.calls if c[0] == 'compile')
    assert m.program_caps(3, 8)['predict_API'] == 51  # not forced quota4


def test_tail_never_truncated_or_forced_h1():
    r, b, _ = execute(geometry='tail')
    assert not r['complete_public_horizon']
    for row in r['rows']:
        if row['mode'] == 'adaptive':
            assert row['status'] == 'refused_full_H_exceeds_remaining'
            assert row['actual_public_steps'] == 10
            assert row['rounds'][-1]['decision']['horizon'] == 3
    assert len(r['rows']) == 4


def test_uncertified_reconfiguration_not_fallback():
    r, b, _ = execute(geometry='high')
    for row in r['rows']:
        if row['mode'] == 'adaptive':
            assert row['rounds'][0]['decision']['candidates'] == 8
            assert row['status'] == 'refused_uncertified_program_type'
    assert not r['complete_public_horizon']


@pytest.mark.parametrize('fault', ['signal_parent', 'outcome', 'plan_parent', 'prefix_only', 'history', 'AD'])
def test_mutations_fail_closed_all_denominators(fault):
    r, b, _ = execute(geometry='long', fault=fault)
    assert len(r['rows']) == 4 and b.closes == 4
    if fault == 'prefix_only':
        assert all(row['status'] == ('engineering_failure' if row['mode'] == 'adaptive' else 'complete') for row in r['rows'])
    else:
        assert all(row['status'] == 'engineering_failure' for row in r['rows'])
    if fault == 'AD':
        assert all(row['resources']['attempted']['predict_API'] > 0 for row in r['rows'])
        assert all(row['resources']['completed']['jacobian_API'] == 0 for row in r['rows'])


def test_earlystop_no_padding():
    r, b, _ = execute(early=7)
    assert all(row['actual_public_steps'] == 7 and row['status'] == 'native_early_stop' for row in r['rows'])
    assert not any(c[0] == 'step' and c[-1] > 7 for c in b.calls)


def test_meter_rejects_before_callback_and_no_refund():
    caps = m.predict_axes(1)
    ledger = m.core.Ledger(caps)
    ledger.scope = dict(RAM=True)
    ledger.call('predict', lambda: None, shape=[1, 3, 192])
    called = []
    with pytest.raises(m.core.BudgetExceeded):
        ledger.call('predict', lambda: called.append(True), shape=[1, 3, 192])
    assert not called and ledger.attempted['predict_API'] == 1
    ledger = m.core.Ledger(caps)
    ledger.scope = dict(RAM=True)
    with pytest.raises(ZeroDivisionError):
        ledger.call('predict', lambda: 1/0, shape=[1, 3, 192])
    assert ledger.attempted['predict_API'] == 1 and ledger.completed['predict_API'] == 0


def test_no_production_callback_elevation():
    b = Backend()
    b.kind = 'HOST_CUDA'
    with pytest.raises(ValueError, match='RAM-only'):
        m.RamDriver(sources(), b, lambda _: None)
    with pytest.raises(ValueError, match='production blocked'):
        m.run(dict(protocol=m.protocol()), authorization='MAIN')


def test_every_five_feedback_precedes_paid_signal_no_outcome_to_compile():
    r, b, _ = execute(geometry='long')
    for row in r['rows']:
        # Compiler requests are whitelisted model-side data; native metrics live
        # in a disjoint row and never appear in the signal/compiler arguments.
        for round in row['rounds']:
            if round['request'] is not None:
                assert set(round['request']) == {'source_sha256', 'binding', 'observation', 'decision', 'signal_sha256', 'local_response'}
                assert not ({'reward', 'metric', 'native_state', 'success'} & set(round['request']['observation']))
    for i, call in enumerate(b.calls):
        if call[0] == 'signal':
            assert b.calls[i-1] == ('encode', call[1], call[2])
    assert [c for c in b.calls if c[0] == 'encode' and c[-1] == 20]


def test_entire_node_cap_before_compile_callback(monkeypatch):
    caps = m.arm_caps
    def reduced(mode):
        c = caps(mode)
        c['predict_API'] = c['predict_samples'] = 8  # only local signal fits
        return c
    monkeypatch.setattr(m, 'arm_caps', reduced)
    r, b, _ = execute()
    assert not any(c[0] == 'compile' for c in b.calls)
    assert all(row['status'] == 'engineering_failure' and
               row['resources']['attempted']['predict_API'] == 8 for row in r['rows'])


@pytest.mark.parametrize('change', ['root', 'quota', 'count', 'private'])
def test_source_roster_mutations(change):
    s = sources()
    if change == 'root':
        s[0]['task']['binding']['root_sha256'] = '0'*64
    elif change == 'quota':
        s[0]['task']['action_std'] = [1, 1]
    elif change == 'count':
        s.pop()
    else:
        s[0]['reward'] = 10
    with pytest.raises(ValueError):
        m.validate_sources(s)


def test_real_paths_prepare_verify_cli_and_tamper(tmp_path):
    roster = tmp_path/'roster.json'
    roster.write_bytes(m.canonical(sources()))
    output = tmp_path/'prepared.json'
    value = m.material(sources())
    value['roster_authority'] = dict(path=str(roster), sha256=m.sha(roster))
    output.write_bytes(m.canonical(value))
    assert m.verify(output, m.sha(output))['fixed_denominator'] == 4
    proc = subprocess.run([sys.executable, '-B', '-m', 'experiments.spectral_reachability.'+m.NAME,
        'verify', '--prepared', str(output), '--prepare-sha', m.sha(output)],
        cwd=m.ROOT, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    value['protocol']['public_horizon'] = 10
    output.write_bytes(m.canonical(value))
    with pytest.raises(ValueError, match='exact declaration'):
        m.verify(output, m.sha(output))
    roster.write_bytes(b'[]')
    with pytest.raises(ValueError, match='roster authority'):
        m.verify(output, m.sha(output))


def test_no_neural_or_physics_imports_in_fresh_interpreter():
    code = ('import sys; from experiments.spectral_reachability import '+m.NAME+'; '
            'assert not ({"torch", "pymunk", "pygame", "gymnasium"} & set(sys.modules))')
    p = subprocess.run([sys.executable, '-B', '-c', code], cwd=m.ROOT, capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr
