import hashlib
import pytest
from experiments.spectral_reachability import p2_h15_controlled_host_20261029_v1 as m


def test_resource_guard_detects_time_memory_disk(tmp_path):
    guard=m.ResourceGuard(tmp_path,lambda:10,lambda:(0,0),lambda:10**12,start=0)
    guard.check()
    with pytest.raises(ValueError,match='wall'):
        m.ResourceGuard(tmp_path,lambda:3601,lambda:(0,0),lambda:10**12,start=0).check()
    with pytest.raises(ValueError,match='CUDA'):
        m.ResourceGuard(tmp_path,lambda:1,lambda:(m.ALLOCATED+1,0),lambda:10**12,start=0).check()
    with pytest.raises(ValueError,match='disk'):
        m.ResourceGuard(tmp_path,lambda:1,lambda:(0,0),lambda:1,start=0).check()


def test_exact_source_seal_detects_drift(tmp_path):
    path=tmp_path/'x';path.write_bytes(b'a')
    pins={str(path):hashlib.sha256(b'a').hexdigest()}
    m.check_extra_sources(pins)
    path.write_bytes(b'b')
    with pytest.raises(ValueError):m.check_extra_sources(pins)


def test_dispatch_is_conditionally_allowed():
    assert m.dispatch_ready({'authorization':m.AUTH,'preflight_verified':True,'limits_installed':True})
    for entry in ({},{'authorization':m.AUTH,'preflight_verified':True,'limits_installed':False}):
        with pytest.raises(ValueError):m.dispatch_ready(entry)


def test_prepare_rebuild_and_external_hash(tmp_path):
    path=tmp_path/(m.NAME+'_test.json')
    digest=m.prepare(path)
    assert m.verify(path,digest)['fixed_denominator']==2
    with pytest.raises(ValueError):m.verify(path,'0'*64)
    with pytest.raises(FileExistsError):m.prepare(path)
