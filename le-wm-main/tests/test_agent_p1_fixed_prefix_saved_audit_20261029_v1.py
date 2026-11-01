"""RAM-only native transcript fixtures, not a production authority."""
from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_p1_fixed_prefix_saved_audit_20261029_v1 as a
import test_agent_p1_fixed_prefix_refinement_20261029_v1 as fixed
import test_agent_p1_nonlinear_suffix_library_20261028_v2 as native

inputs = fixed.inputs
complete = fixed.complete
group = fixed.group


@pytest.fixture
def saved(group):
    q = deepcopy(group)
    scratch = native.Events()
    b = native.RAM(q['task'], scratch, a.core.Ledger(a.F.caps(2, 0), sink=scratch.event))
    q['expected_mid'] = b.step(q['task']['root_context'], a.F.chart(q['prefix_logits']), 'fixture_prefix')
    q['expected_end'] = b.step(q['expected_mid'], a.F.chart(q['suffix_logits']), 'fixture_suffix')
    q['baseline_Q'] = a.F.cost(q['task']['goal'], q['expected_end'])
    events = native.Events()
    ledger = a.core.Ledger(a.F.caps(19, 8), sink=events.event)
    backend = native.RAM(q['task'], events, ledger)
    result = a.method.run_group(q, backend, events.event)
    return q, result, ledger.receipt(), events.events


def test_full_native_replay_and_scalar(saved):
    q, r, ledger, events = saved
    checked = a.audit(q, r, ledger, events)
    assert checked['native_forwards'] == 19
    assert checked['scalar_objective']['checked_scalar_losses'] == 8
    assert not checked['P1_admission']


def test_scalar_independently_rejects_coherent_loss_change(saved):
    q, r, _, _ = saved
    r['gradients'][0]['native_float32_loss'] += 1
    with pytest.raises(ValueError, match='scalar objective mismatch'):
        a.scalar_audit(q['task'], r)


def test_saved_control_tamper_rejected(saved):
    q, r, ledger, events = saved
    next(x for x in events if x['kind'] == 'score_return')['row']['terminal_Q'] += 1
    with pytest.raises(ValueError):
        a.audit(q, r, ledger, events)


def test_wrong_native_budget_rejected(saved):
    q, r, ledger, events = saved
    ledger['caps'] = a.F.caps(20, 8)
    with pytest.raises(ValueError):
        a.audit(q, r, ledger, events)
