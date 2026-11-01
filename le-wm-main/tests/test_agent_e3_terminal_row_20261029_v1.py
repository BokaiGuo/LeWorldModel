import pytest
from experiments.spectral_reachability import agent_e3_terminal_row_20261029_v1 as m
from experiments.spectral_reachability import agent_e3_transport_publication_20261029_v1 as p


@pytest.mark.parametrize('status',['completed20','native_stopped','failed'])
def test_disk_partial_row_retains_errors(status,tmp_path):
    store=p.Store(tmp_path/'job')
    row=dict(episode_id='cell00',status=status,error={'type':'fixture','traceback':'x'*13000})
    sha=m.write(store,row);seal=store.publish('partial')
    assert not p.consume(store.output,seal['report_sha256'],seal['completion_sha256'])['complete']
    events=[p.base.decode(x) for x in (store.output/'events.jsonl').read_bytes().splitlines()]
    assert m.read(events)==row and len(sha)==64
    chunks=[e for e in events if e['resource']=='terminal_row_chunk']
    chunks[0]['detail']['data']='bad!'
    with pytest.raises(ValueError):m.read(events)


def test_incomplete_terminal_row_rejected(tmp_path):
    store=p.Store(tmp_path/'job');m.write(store,dict(episode_id='x',status='failed'))
    events=[p.base.decode(x) for x in (store.output/'events.jsonl').read_bytes().splitlines()]
    with pytest.raises(ValueError):m.read(events[:-1])
