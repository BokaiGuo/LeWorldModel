import pytest
from experiments.spectral_reachability import agent_e1_same_target_candidate_intervention_20261029_v1 as m


def bank():
    return [dict(index=i, cost=1.0, commands=[[0.0, 0.0] for _ in range(10)]) for i in range(72)]


def test_both_arms_exact_account_and_replay():
    calls = []
    def dispatch(raw, phase, context):
        calls.append((phase, raw, context))
        return [dict(n=i) for i in range(len(raw)//5)]
    b = bank()
    for arm in m.ARMS:
        meter = m.Meter()
        r = m.select_and_replay(b, m.digest(b), arm, [[0.25, -0.5] for _ in range(5)], dispatch, lambda _: 0.5, meter)
        assert r['winner_index'] == 72
        assert (r['attempted'], r['returned']) == (6, 6)
        assert r['whole'] == [dict(n=0), dict(n=1)]
        assert r['selected_raw'][5:] == [[0.0, 0.0]] * 5
    assert len(calls) == 8
    assert m.PACKETS_PER_SLOT * len(m.SLOTS) == 600


def test_original_candidate_wins_ties():
    b = bank()
    r = m.select_and_replay(b, m.digest(b), 'zero_extra', [[0., 0.]]*5,
                            lambda raw, phase, ctx: [{}]*(len(raw)//5), lambda _: 1., m.Meter())
    assert r['winner_index'] == 0


def test_failure_charged_and_poisoned():
    meter = m.Meter()
    def fail(): raise RuntimeError('dispatch')
    with pytest.raises(RuntimeError): meter.invoke(2, fail)
    assert (meter.attempted, meter.returned) == (2, 0)
    with pytest.raises(ValueError): meter.invoke(1, lambda: [{}])


def test_score_exception_poisoned_after_extra_dispatch():
    b = bank()
    meter = m.Meter()
    def bad_score(_):
        raise RuntimeError('score failure')
    with pytest.raises(RuntimeError, match='score failure'):
        m.select_and_replay(b, m.digest(b), 'zero_extra', [[0., 0.]] * 5,
                            lambda raw, phase, ctx: [{}] * (len(raw) // 5), bad_score, meter)
    assert (meter.attempted, meter.returned, meter.failed) == (2, 2, True)
    with pytest.raises(ValueError):
        meter.invoke(1, lambda: [{}])


@pytest.mark.parametrize('kind', ['sha', 'index', 'domain', 'score'])
def test_bad_inputs_fail(kind):
    b = bank()
    if kind == 'index': b[0]['index'] = 1
    prefix = [[0., 0.]]*5 if kind != 'domain' else [[2., 0.]]*5
    sha = '0'*64 if kind == 'sha' else m.digest(b)
    with pytest.raises(ValueError):
        m.select_and_replay(b, sha, 'witness_extra', prefix,
                            lambda raw, phase, ctx: [{}]*(len(raw)//5),
                            lambda _: float('nan') if kind == 'score' else 0., m.Meter())
