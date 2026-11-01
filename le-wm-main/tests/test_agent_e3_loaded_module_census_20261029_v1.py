from copy import deepcopy
import hashlib
from types import SimpleNamespace
import pytest
from experiments.spectral_reachability import agent_e3_loaded_module_census_20261029_v1 as m


def test_real_disk_census_seals_and_rejects_mutation(tmp_path):
    source=tmp_path/'one.py';source.write_text('x=1\n');h=hashlib.sha256(source.read_bytes()).hexdigest()
    bindings={str(source):h};modules={'fixture.one':SimpleNamespace(__file__=str(source))}
    before=m.snapshot(bindings,modules,root=tmp_path)
    store=m.pub.Store(tmp_path/'output');m.record(store,'before',before);m.record(store,'after',before)
    seal=store.publish('completed');assert m.pub.consume(store.output,seal['report_sha256'],seal['completion_sha256'])['complete']
    events=[m.pub.base.decode(x) for x in (store.output/'events.jsonl').read_bytes().splitlines()]
    result=m.saved(events,bindings)
    assert result['before']==result['after']==1 and not result['transient_modules_proved']
    bad=deepcopy(events);bad[-1]['detail']['sha256']='0'*64
    with pytest.raises(ValueError):m.saved(bad,bindings)
    source.write_text('x=2\n')
    with pytest.raises(ValueError):m.saved(events,bindings)


def test_unsealed_and_replaced_module_rejected(tmp_path):
    source=tmp_path/'one.py';source.write_text('x=1\n')
    modules={'x':SimpleNamespace(__file__=str(source))}
    with pytest.raises(ValueError,match='presealed'):m.snapshot({},modules,root=tmp_path)
    h=hashlib.sha256(source.read_bytes()).hexdigest();bindings={str(source):h}
    first=m.snapshot(bindings,modules,root=tmp_path);store=m.pub.Store(tmp_path/'output')
    m.record(store,'before',first);m.record(store,'after',[dict(module='y',path=str(source),sha256=h)])
    seal=store.publish('completed');assert m.pub.consume(store.output,seal['report_sha256'],seal['completion_sha256'])['complete']
    events=[m.pub.base.decode(x) for x in (store.output/'events.jsonl').read_bytes().splitlines()]
    with pytest.raises(ValueError,match='disappearing'):m.saved(events,bindings)
