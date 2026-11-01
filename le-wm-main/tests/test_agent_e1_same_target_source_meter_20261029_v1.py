from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_e1_same_target_source_meter_20261029_v1 as m


def test_actual_saved_source_join():
    value = m.source_inputs()
    assert [r['root'] for r in value['rows']] == [0, 1]
    assert all(len(bytes.fromhex(r['target_hex'])) == 768 and len(r['witness_raw5']) == 5 for r in value['rows'])
    assert not value['production_authorized'] and not value['physical_reachability_authenticated']


def test_bank_exact_all_72_and_signed_zero():
    bank = [dict(index=i, value=0.0) for i in range(72)]
    assert m.compare_banks(bank, deepcopy(bank))['exact']
    changed = deepcopy(bank); changed[71]['value'] = -0.0
    with pytest.raises(ValueError): m.compare_banks(bank, changed)
    with pytest.raises(ValueError): m.compare_banks(bank, bank[:-1])


def packet(meter):
    for resource in ('action_encoder',):
        t = meter.begin(resource, {}); meter.end(t, {})
    api = meter.begin('predict_API', {})
    for resource in ('predictor', 'pred_proj'):
        t = meter.begin(resource, {}); meter.end(t, {})
    meter.end(api, {})


@pytest.mark.parametrize('phase,limit', [('controller',144), ('extra',2), ('replay',4)])
def test_real_hook_meter_full_phase_and_cap(phase, limit):
    events = []
    meter = m.HookMeter(phase, events.append)
    for _ in range(limit): packet(meter)
    summary = m.phase_complete(meter)
    assert summary['predictor_samples']['completed'] == limit
    assert len(events) == limit * 8
    with pytest.raises(ValueError): meter.available()


def test_failure_keeps_partial_actual_module_ticket():
    meter = m.HookMeter('extra', lambda _: None)
    with pytest.raises(RuntimeError):
        meter.call('action_encoder', {}, lambda: (_ for _ in ()).throw(RuntimeError('fail')))
    assert meter.counts()['action_encoder'] == {'attempted': 1, 'completed': 0}
    with pytest.raises(ValueError): m.phase_complete(meter)
    with pytest.raises(ValueError): meter.available()
