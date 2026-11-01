import hashlib
import json

import pytest
from experiments.spectral_reachability import agent_p3_dynamic_cp10_cuda_transport_20261029_v1 as m
import test_agent_p3_same_time_cp10_rgb_bridge_20261029_v1 as fixture


def test_cuda_fail_closed_before_output_or_backend(tmp_path,monkeypatch):
    calls=[]
    def fake(*args):calls.append(args);return {'source_authenticity_independently_verified':False,'execution_authorized':False}
    monkeypatch.setattr(m.physical,'assemble',fake)
    target=tmp_path/(m.NAME+'_not_created')
    with pytest.raises(ValueError,match='BLOCKED'):m.run_cuda([],{},target)
    assert calls and not target.exists()


def test_temp_store_durable_failure_keeps_prefix(tmp_path):
    path=tmp_path/(m.NAME+'_partial')
    with pytest.raises(RuntimeError):m.record_fixture(path,[('first',b'123'),('second',b'456')],fail_at=1)
    assert (path/'first').read_bytes()==b'123' and not (path/'second').exists()
    partial=json.loads((path/'partial.json').read_bytes())
    assert partial['status']=='partial_fixture' and partial['persisted_hashes']['first']==hashlib.sha256(b'123').hexdigest()
    assert partial['execution_authorized'] is False


def test_store_refuses_existing_and_output_cap(tmp_path,monkeypatch):
    target=tmp_path/(m.NAME+'_cap');s=m.Store(target)
    monkeypatch.setattr(m,'CAP',100);monkeypatch.setattr(m,'FINAL',25)
    s.write('one',b'x'*60)
    with pytest.raises(ValueError):s.write('two',b'y'*16)
    assert not (target/'two').exists()
    with pytest.raises(FileExistsError):m.Store(target)


def test_saved_native_full_fixture_and_orphan_rejection(tmp_path,monkeypatch):
    p,s,r,_=fixture.completed_disk(tmp_path,monkeypatch)
    hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in s.path.iterdir()}
    result=m.verify_saved(p,r,s.path,hashes)
    assert result['native_validation']['event_count']==52 and not result['full_saved_consumer']
    (s.path/'unbound').write_bytes(b'extra')
    with pytest.raises(ValueError):m.verify_saved(p,r,s.path,hashes)


def test_saved_array_tampering_rejected(tmp_path,monkeypatch):
    p,s,r,_=fixture.completed_disk(tmp_path,monkeypatch)
    hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in s.path.iterdir()}
    a=next(s.path.glob('*.npy.gz'));a.write_bytes(b'changed')
    with pytest.raises(ValueError):m.verify_saved(p,r,s.path,hashes)
