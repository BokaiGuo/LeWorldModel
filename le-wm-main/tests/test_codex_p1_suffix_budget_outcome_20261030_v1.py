import pytest
from experiments.spectral_reachability import codex_p1_suffix_budget_outcome_20261030_v1 as m


def test_resource_axes_no_padding_or_mixing():
    events = []
    caps = dict.fromkeys(m.AXES, 1)
    budget = m.Budget(caps, events.append, wall_seconds=10)
    assert budget.call('predict', 1, lambda: 42) == 42
    with pytest.raises(ValueError, match='resource cap'):
        budget.call('predict', 1, lambda: None)
    assert budget.counts['predict'] == dict(attempted=1, returned=1, failed=0)
    assert budget.counts['encoder']['attempted'] == 0
    with pytest.raises(RuntimeError):
        budget.call('encoder', 1, lambda: (_ for _ in ()).throw(RuntimeError('actual failure')))
    assert budget.counts['encoder'] == dict(attempted=1, returned=0, failed=1)
    assert [e['kind'] for e in events] == ['attempt', 'return', 'attempt', 'failure']


def test_outcome_does_not_infer_success_or_pad_censor():
    result = m.outcome([0]*7, [0]*7, native_success=False, terminated=False, truncated=True, censored=True)
    assert result['native_success'] is False
    assert result['truncated'] and result['censored']
    assert result['object_position_error'] == result['pusher_position_error'] == 0
