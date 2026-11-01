"""Chunk transport tests only; no neural computation or physical execution."""
from copy import deepcopy
import pytest
from experiments.spectral_reachability import p3_matched_order_feedback_store_codex_20261029 as store
from experiments.spectral_reachability import p3_matched_order_feedback_numeric_bridge_codex_20261029 as bridge


def test_large_native_event_roundtrip_and_tamper(tmp_path):
    s = store.Store(tmp_path/'run'); b = bridge.Bridge(s,2)
    payload = dict(kind='native', value='x'*50000)
    b.event(payload); b.event(dict(kind='second'))
    path = s.output/'events.jsonl'
    events = store.inspect_journal(path,store.base.fsha(path.read_bytes()))['events']
    assert bridge.reconstruct(events,2) == [payload,dict(kind='second')]
    bad = deepcopy(events); bad[1]['detail']['index'] = 8
    with pytest.raises(ValueError,match='chunk order'):
        bridge.reconstruct(bad,2)


def test_incomplete_chunk_never_promoted(tmp_path):
    s = store.Store(tmp_path/'run'); bridge.Bridge(s,0).event(dict(value='x'*10000))
    path = s.output/'events.jsonl'
    events = store.inspect_journal(path,store.base.fsha(path.read_bytes()))['events']
    with pytest.raises(ValueError,match='complete native chunks'):
        bridge.reconstruct(events[:-1],0)
