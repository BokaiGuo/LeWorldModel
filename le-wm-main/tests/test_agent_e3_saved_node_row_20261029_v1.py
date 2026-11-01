import hashlib
import pytest
from experiments.spectral_reachability import agent_e3_saved_node_row_20261029_v1 as m
from test_agent_e3_gd_host_cuda_20261029_v1 import body, Model, DEVICE, T


def test_real_disk_row_and_independent_replay(tmp_path):
    b=body();b['feedback_boundary']=10;s=m.bridge.transport.Store(tmp_path/'job');a=m.bridge.Bridge(s)
    row=dict(forwards=[],AD=[],result=None);backend=m.bridge.host.HostBackend(Model(),DEVICE,T,a,b,row)
    with backend.observe():row['result']=m.node.solve(b,backend,a.event)
    seal=m.persist(s,row,10);assert seal['chunks']>1
    h=hashlib.sha256((s.output/'events.jsonl').read_bytes()).hexdigest()
    out=m.consume(b,s.output,h,10)
    assert out['result']==row['result'] and out['row_sha256']==seal['row_sha256']
    assert not out['full_native_strict'] and not out['source_model_rgb_authenticated']
    with pytest.raises(ValueError):m.consume(b,s.output,h,15)


def test_wrong_row_bytes_no_synthetic_success(tmp_path):
    b=body();b['feedback_boundary']=10;s=m.bridge.transport.Store(tmp_path/'job')
    m.persist(s,dict(forwards=[],AD=[],result=None),10)
    h=hashlib.sha256((s.output/'events.jsonl').read_bytes()).hexdigest()
    with pytest.raises(ValueError):m.consume(b,s.output,h,10)
