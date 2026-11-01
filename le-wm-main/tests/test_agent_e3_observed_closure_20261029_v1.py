import types

import pytest

from experiments.spectral_reachability import agent_e3_observed_closure_20261029_v1 as mod


class Store:
    def __init__(self): self.events = []; self.index = 0
    def event(self, origin, kind, resource, token, detail):
        self.events.append(dict(origin=origin, kind=kind, resource=resource,
                                token=token, detail=detail, index=self.index))
        self.index += 1


@pytest.fixture
def sample(tmp_path, monkeypatch):
    monkeypatch.setattr(mod.census, 'ROOT', tmp_path)
    p = tmp_path / 'used.py'; p.write_text('used')
    q = tmp_path / 'unused.py'; q.write_text('unused')
    entries = [dict(module='used', path=str(p), sha256=mod.startup.file_hash(p))]
    store = Store()
    mod.census.record(store, 'before', entries)
    mod.census.record(store, 'after', entries)
    return p, q, store.events


def test_observed_only_and_drift(sample):
    p, q, events = sample
    pins, seals = mod.observed(events)
    assert set(pins) == {str(p)} and set(seals) == {'before', 'after'}
    q.write_text('independent change')
    assert mod.observed(events)[0] == pins
    p.write_text('drift')
    with pytest.raises(Exception): mod.observed(events)


def test_tampered_chunk_and_stage_order(sample):
    _, _, events = sample
    events[0]['detail']['sha256'] = '0' * 64
    with pytest.raises(Exception): mod.observed(events)


def test_disappearing_module_is_rejected(sample):
    p, _, _ = sample
    store = Store()
    first = [dict(module='used', path=str(p), sha256=mod.startup.file_hash(p))]
    second = [dict(module='replacement', path=str(p), sha256=mod.startup.file_hash(p))]
    mod.census.record(store, 'before', first)
    mod.census.record(store, 'after', second)
    with pytest.raises(Exception): mod.observed(store.events)


def test_unknown_module_remains_fail_closed(tmp_path):
    p = tmp_path / 'unsealed.py'; p.write_text('x')
    with pytest.raises(Exception):
        mod.census.snapshot({}, {'unsealed': types.SimpleNamespace(__file__=str(p))}, root=tmp_path)


def test_base_and_additional_collisions():
    binding = {'prepared': {'bindings_sha256': {'/a': 'one'}}}
    prior = {'additional_sources': {'/a': 'other'}}
    with pytest.raises(Exception): mod.project.reconstruct(binding, prior, {'/a': 'one'})


def test_exclusive_write(tmp_path):
    p = tmp_path / 'authority.json'
    mod.startup.write_new(str(p), {'x': 1})
    with pytest.raises(Exception): mod.startup.write_new(str(p), {'x': 2})
