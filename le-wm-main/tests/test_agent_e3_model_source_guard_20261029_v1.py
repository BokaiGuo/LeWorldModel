from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import pytest
from experiments.spectral_reachability import agent_e3_model_source_guard_20261029_v1 as g


def snapshot():
    return dict(tensor_sha256=g.host.numerical.resolver.MODEL_SHA,
                classes=[dict(name='',path='/fixture',sha256='a'*64)],
                precision=g.host.primitives.precision_expected(),allocated=1,reserved=2)


@pytest.mark.parametrize('field',['model','classes','precision','allocated','bool'])
def test_pair_rejects_drift(field):
    before=snapshot();after=deepcopy(before)
    if field=='model':after['tensor_sha256']='0'*64
    if field=='classes':after['classes'][0]['sha256']='b'*64
    if field=='precision':after['precision']['MATH']=False
    if field=='allocated':after['allocated']=3*1024**3+1
    if field=='bool':after['reserved']=True
    with pytest.raises(ValueError):g.validate_pair(before,after,g.host.numerical.resolver.MODEL_SHA)


def test_sources_actual_disk_and_tamper(tmp_path):
    file=tmp_path/'fixture.py';file.write_bytes(b'original\n');path=str(file)
    sha=hashlib.sha256(file.read_bytes()).hexdigest()
    g.validate_sources({path:sha},[path])
    file.write_bytes(b'changed\n')
    with pytest.raises(ValueError):g.validate_sources({path:sha},[path])


def test_actual_snapshot_path_with_non_neural_fake_model():
    class Model:
        training=False
        def modules(self):return [self]
        def parameters(self):return []
        def buffers(self):return []
        def state_dict(self):return {}
        def named_modules(self):return [('',self)]
    cuda=SimpleNamespace(synchronize=lambda device:None,max_memory_allocated=lambda device:1,
                         max_memory_reserved=lambda device:2,
                         matmul=SimpleNamespace(allow_tf32=False),math_sdp_enabled=lambda:True,
                         flash_sdp_enabled=lambda:False,mem_efficient_sdp_enabled=lambda:False)
    torch=SimpleNamespace(cuda=cuda,backends=SimpleNamespace(cuda=cuda,cudnn=SimpleNamespace(allow_tf32=False)),
                          is_autocast_enabled=lambda device:False)
    path=str(Path(__file__).resolve());sha=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    value=g.snapshot(Model(),torch,SimpleNamespace(type='cuda'),{path:sha},[path])
    assert value['tensor_sha256']==hashlib.sha256(b'').hexdigest()
    assert value['classes'][0]['path']==path and value['precision']==g.host.primitives.precision_expected()
