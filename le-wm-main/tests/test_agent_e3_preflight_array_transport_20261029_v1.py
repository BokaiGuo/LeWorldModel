import numpy as np
import pytest

from experiments.spectral_reachability import agent_e3_preflight_array_transport_20261029_v1 as t
from experiments.spectral_reachability import agent_e3_encoder_cuda_preflight_20261029_v2 as preflight


def events(path):
    return [t.base.decode(line) for line in (path / 'events.jsonl').read_bytes().splitlines()]


def consume(store):
    seal = store.publish('partial')
    result = t.pub.consume(store.output, seal['report_sha256'], seal['completion_sha256'])
    assert result['complete'] is False
    return t.verify(events(store.output)), seal


def test_partial_after_array_return_has_complete_inventory(tmp_path):
    s = t.Store(tmp_path / 'out')
    descriptor = s.array('encoding', 'pixels', np.zeros((224, 224, 3), np.uint8))
    result, seal = consume(s)
    assert (result['attempted'], result['returned'], result['failed'], result['pending']) == (1, 1, 0, 0)
    assert result['saved_arrays'] == [descriptor]
    p = s.output / descriptor['file']
    data = bytearray(p.read_bytes()); data[-1] ^= 1; p.write_bytes(data)
    with pytest.raises(ValueError):
        t.pub.consume(s.output, seal['report_sha256'], seal['completion_sha256'])


@pytest.mark.parametrize('after_write', [False, True])
def test_failure_before_or_after_durable_write(tmp_path, after_write):
    class Broken(t.Store):
        def persist_array(self, name, data):
            if after_write:
                super().persist_array(name, data)
            raise RuntimeError('fixture failure')
    s = Broken(tmp_path / 'out')
    with pytest.raises(RuntimeError):
        s.array('encoding', 'pixels', np.zeros((224, 224, 3), np.uint8))
    result, _ = consume(s)
    assert (result['attempted'], result['returned'], result['failed'], result['pending']) == (1, 0, 1, 0)
    assert len(result['saved_arrays']) == int(after_write)


def test_descriptor_swap_rejected(tmp_path):
    s = t.Store(tmp_path / 'out')
    s.array('encoding', 'pixels', np.zeros((224, 224, 3), np.uint8))
    rows = events(s.output)
    rows[-1]['detail']['descriptor']['role'] = 'latent'
    with pytest.raises(ValueError, match='descriptor join'):
        t.verify(rows)


def test_reused_or_overlapping_array_intents_rejected(tmp_path):
    s = t.Store(tmp_path / 'out')
    s.array('encoding', 'pixels', np.zeros((224, 224, 3), np.uint8))
    rows = events(s.output)
    intent, terminal = rows
    reused = dict(intent, index=2)
    with pytest.raises(ValueError, match='array intent'):
        t.verify([intent, terminal, reused])
    overlapping = dict(intent, index=1, token=1)
    with pytest.raises(ValueError, match='array intent'):
        t.verify([intent, overlapping])


def test_unreferenced_file_fails_closed(tmp_path):
    s = t.Store(tmp_path / 'out')
    s.event('physical', 'record', 'fixture', 0, {})
    (s.output / 'orphan.npy').write_bytes(b'partial')
    seal = s.publish('partial')
    with pytest.raises(ValueError):
        t.pub.consume(s.output, seal['report_sha256'], seal['completion_sha256'])


def test_v2_partial_consumer_disk_chain(tmp_path, monkeypatch):
    s = t.Store(tmp_path / 'out')
    preflight.record(s, 'preflight_started', dict(schema=preflight.NAME, scientific_admission=False, physics_calls=0))
    s.array('encoding', 'pixels', np.zeros((224, 224, 3), np.uint8))
    preflight.record(s, 'preflight_failure', dict(type='RuntimeError', message='fixture', traceback='fixture'))
    seal = s.publish('partial')
    monkeypatch.setattr(preflight.host, 'authority', lambda *args: {})
    monkeypatch.setattr(preflight, 'sources', lambda binding: ({}, []))
    result = preflight.consume(str(s.output), seal['report_sha256'], seal['completion_sha256'], '/fixture', 'fixture')
    assert result['status'] == 'partial' and result['full_saved_strict'] is False
    assert result['array_transport']['returned'] == 1
    assert len(result['array_transport']['saved_arrays']) == 1
