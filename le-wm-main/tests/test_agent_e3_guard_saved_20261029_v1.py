import base64
import hashlib
from pathlib import Path
import pytest
from experiments.spectral_reachability import agent_e3_guard_saved_20261029_v1 as m


def test_guard_disk_consumer_and_source_tamper(tmp_path):
    source=tmp_path/'model.py';source.write_bytes(b'fixture')
    path=str(source);sha=hashlib.sha256(source.read_bytes()).hexdigest()
    cls=dict(name='',class_module='fixture',class_qualname='Model',path=path,sha256=sha)
    snap=dict(tensor_sha256=m.guard.host.numerical.resolver.MODEL_SHA,classes=[cls],
              precision=m.guard.host.primitives.precision_expected(),allocated=1,reserved=2)
    s=m.publication.Store(tmp_path/'job')
    def record(stage):
        b=m.publication.base.raw(snap);h=hashlib.sha256(b).hexdigest()
        s.event('physical','record','guard_snapshot_chunk',s.index,
                dict(stage=stage,index=0,total=1,sha256=h,data=base64.b64encode(b).decode('ascii')))
        s.event('physical','record',stage,s.index,dict(sha256=h,total=1,bytes=len(b)))
    record('guard_before');s.event('physical','record','native_reset',s.index,{})
    s.event('physical','record','native_cleanup_return',s.index,{});record('guard_after')
    seal=s.publish('completed');args=(s.output,seal['report_sha256'],seal['completion_sha256'],{path:sha},[path])
    assert m.consume(*args)['saved_model_source_guard_verified']
    source.write_bytes(b'changed')
    with pytest.raises(ValueError):m.consume(*args)
