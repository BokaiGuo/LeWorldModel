import hashlib
from pathlib import Path
import time
from types import SimpleNamespace
import pytest

from experiments.spectral_reachability import agent_e3_identity_host_20261029_v1 as host
from test_agent_e3_integrated_host_saved_20261029_v1 import pinned_authority


def authority(tmp_path):
    path,sha=pinned_authority(tmp_path);value=host.launch.host.read(str(path),sha)
    for module in (host,host.identity,host.capacity):
        key=str(Path(module.__file__).resolve());value['additional_sources'][key]=host.launch.host.sha(key)
    data=host.launch.host.raw(value);path.write_bytes(data)
    return path,hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize('marker',[True,False])
def test_failure_before_loader_and_old_journal_rejected(tmp_path,marker):
    path,sha=authority(tmp_path);out=tmp_path/'job'
    def perform(binding,store):
        if marker:host.launch._record(store,'identity_host_started',dict(schema=host.NAME,
            max_records=host.MAX_IDENTITY_RECORDS,max_event_bytes=host.MAX_IDENTITY_BYTES))
        raise RuntimeError('fixture before loader')
    r=host.launch._run_lifecycle(out,time.monotonic(),lambda:host.authority(str(path),sha),perform)
    seal=r['publication'];args=(out,seal['report_sha256'],seal['completion_sha256'],str(path),sha)
    if marker:
        result=host.inspect(*args)
        assert result['identity'] is None and not result['full_saved_strict']
        assert result['base']['saved']['error']['message']=='fixture before loader'
    else:
        with pytest.raises(ValueError,match='reject old logs'):host.inspect(*args)


def test_episode_setup_failure_closes_and_durably_records(tmp_path):
    from experiments.spectral_reachability import agent_e3_terminal_row_20261029_v1 as terminal
    store=host.launch.pub.Store(tmp_path/'job');closed=[]
    env=SimpleNamespace(close=lambda:closed.append(True));row={'episode_id':'cell00','status':'started'}
    with pytest.raises(TypeError):
        host.execute_episode(env,{},None,None,SimpleNamespace(model=None),None,None,store,SimpleNamespace(),row)
    seal=store.publish('partial');assert closed==[True]
    events=[host.launch.pub.base.decode(line) for line in (store.output/'events.jsonl').read_bytes().splitlines()]
    assert terminal.read(events)['status']=='failed'
    names=[e['resource'] for e in events]
    assert names.index('native_episode_failure')<names.index('native_cleanup_intent')<names.index('native_cleanup_return')
    assert not seal['complete']


def test_budget_is_fixed_and_unauthorized_dispatch_is_absent(tmp_path):
    assert host.MAX_IDENTITY_BYTES==147465 and host.MAX_IDENTITY_RECORDS==9
    with pytest.raises(ValueError,match='authorization'):
        host.run('/nonexistent','0'*64,tmp_path/'job','fixture')
    assert not (tmp_path/'job').exists()


def test_capacity_preflight_joins_real_store_and_rejects_mutation(tmp_path):
    from test_agent_e3_encoder_identity_20261029_v1 import prepared,FakeTorch,DEVICE
    from copy import deepcopy
    store=host.launch.pub.Store(tmp_path/'job');p=prepared(tmp_path)
    _,proof=host.capacity.load_bound(p,FakeTorch,DEVICE,store,'a'*64)
    host.launch._record(store,'identity_capacity_preflight',proof)
    journal=(store.output/'events.jsonl').read_bytes()
    events=[host.launch.pub.base.decode(x) for x in journal.splitlines()]
    bind=next(e for e in events if e['resource']=='encoder_identity_bind');record=events[-1]
    assert host.verify_capacity(bind,record,journal)
    altered=deepcopy(record);altered['detail']['binding_event_bytes']+=1
    with pytest.raises(ValueError,match='exact bound event'):host.verify_capacity(bind,altered,journal)
