import numpy as np
import pytest
from types import SimpleNamespace
from experiments.spectral_reachability import agent_p1_midpoint_monotone_compiler_20261029_v1 as m


def rows():
    return [dict(iteration=i, candidate_Q=2., midpoint_Q=3.) for i in range(1, 9)]


def test_fixed_budget_and_denominators():
    c = m.contract()
    assert (c['fixed_denominator'], c['candidate_denominator'], c['midpoint_denominator']) == (4, 36, 32)
    assert (c['inherited_predict'], c['inherited_reverse'], c['added_predict']) == (72, 32, 32)
    assert not c['production_ready'] and not c['physical_release']


def test_strict_selection_and_tie():
    r = rows()
    r[1]['candidate_Q'] = 1.
    r[2]['candidate_Q'] = 1.
    assert m.select(4., r) == 2
    r[1]['midpoint_Q'] = 4.
    assert m.select(4., r) == 3
    assert m.select(2., rows()) is None


def test_all_checks_and_finite_required():
    with pytest.raises(ValueError): m.select(4., rows()[:-1])
    r = rows(); r[0]['midpoint_Q'] = float('nan')
    with pytest.raises(ValueError): m.select(4., r)


def test_midpoint_is_in_endpoint_ball_without_projection():
    a = np.arange(10, dtype=np.float64) / 10
    b = -a
    assert np.array_equal(m.midpoint_logits(a, b, 0.1), np.zeros(10))
    with pytest.raises(ValueError): m.midpoint_logits([0] * 9, b, 1.)
    with pytest.raises(ValueError): m.midpoint_logits([float('nan')] * 10, b, 1.)


def test_real_control_flow_fixture_all_eight_and_failure(monkeypatch):
    its = [dict(iteration=i, logits=[i / 20.] * 10, commands=[[i / 20.] * 2] * 5,
                Q=4. if i == 0 else 2.) for i in range(9)]
    spec = {'request': {'observed_context': {'history': [1, 2, 3]},
                        'task': {'goal': [0.] * 192}, 'incumbent': {'prefix': [[0., 0.]] * 5}}}
    base = {'numeric': {'iterates': its}, 'numeric_events': []}
    fake = SimpleNamespace(compile_row=lambda s, b, j: base, strict_numeric=lambda *a: None,
                           digest=lambda x: 'context', F=SimpleNamespace(bits=lambda a: a.tobytes().hex()),
                           core=SimpleNamespace(checked_path=lambda *a: None))
    chart = SimpleNamespace(CAP=3., chart=lambda x: np.asarray(x, dtype=np.float32).reshape(5, 2),
                            cost=lambda goal, context: 3.)
    monkeypatch.setattr(m, 'original', fake)
    monkeypatch.setattr(m, 'frozen', chart)
    class Journal:
        def __init__(self): self.events = []
        def event(self, e): self.events.append(e)
    class Backend:
        def __init__(self, fail=None): self.calls = []; self.fail = fail
        def step(self, context, action, label):
            self.calls.append((context, action.copy(), label))
            if len(self.calls) == self.fail: raise RuntimeError('injected failure')
            return {'output': 1}
    j, b = Journal(), Backend()
    r = m.compile_slot(spec, b, j)
    assert len(b.calls) == len(r['midpoint_rows']) == 8
    assert all(c[0] == spec['request']['observed_context'] for c in b.calls)
    assert r['selected_iteration'] == 1 and len(r['selected_commands']) == 10
    j, b = Journal(), Backend(fail=3)
    with pytest.raises(RuntimeError): m.compile_slot(spec, b, j)
    assert [e['kind'] for e in j.events] == ['midpoint_intent', 'midpoint_return'] * 2 + ['midpoint_intent']
