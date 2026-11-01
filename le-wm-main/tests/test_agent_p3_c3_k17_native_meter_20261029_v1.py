import pytest
from experiments.spectral_reachability import agent_p3_c3_k17_native_meter_20261029_v1 as m
from experiments.spectral_reachability import agent_p3_c3_k17_native_store_20261029_v1 as s
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as journal


def test_real_durable_intent_return(tmp_path):
    store = s.NativeStore(tmp_path / 'run')
    meter = m.Meter(store)
    def call():
        events = journal.read_journal(store.path / 'journal.jsonl')
        assert events[-1]['kind'] == 'operation_dispatch_intent'
        return 42
    assert meter.call('public_step', call, step=1) == 42
    events = journal.read_journal(store.path / 'journal.jsonl')
    assert [x['kind'] for x in events] == ['operation_dispatch_intent', 'operation_returned']
    assert meter.counts['public_step'] == {'attempted': 1, 'completed': 1}


def test_failed_attempt_not_refunded_and_close_allowed(tmp_path):
    store = s.NativeStore(tmp_path / 'run')
    meter = m.Meter(store)
    def fail(): raise RuntimeError('fixture')
    with pytest.raises(RuntimeError): meter.call('public_step', fail)
    assert meter.counts['public_step'] == {'attempted': 1, 'completed': 0}
    with pytest.raises(ValueError, match='fault'): meter.call('public_step', lambda: None)
    meter.call('close', lambda: None, cleanup=True)
    assert meter.counts['close']['completed'] == 1


def test_wall_and_cap_reject_before_callback(tmp_path, monkeypatch):
    store = s.NativeStore(tmp_path / 'run')
    clock = lambda: 1000
    meter = m.Meter(store, clock=clock, started=0)
    called = []
    with pytest.raises(ValueError, match='wall'): meter.call('reset', lambda: called.append(1))
    assert not called
    meter = m.Meter(store, clock=lambda: 0)
    monkeypatch.setitem(m.CAPS, 'reset', 0)
    with pytest.raises(ValueError, match='cap'): meter.call('reset', lambda: called.append(1))
    assert not called


def test_failed_intent_prevents_callback(tmp_path, monkeypatch):
    store = s.NativeStore(tmp_path / 'run')
    meter = m.Meter(store)
    monkeypatch.setattr(store, 'emit', lambda *a, **k: (_ for _ in ()).throw(OSError('fixture')))
    called = []
    with pytest.raises(OSError): meter.call('reset', lambda: called.append(1))
    assert called == [] and meter.counts['reset']['attempted'] == 0 and meter.fault
