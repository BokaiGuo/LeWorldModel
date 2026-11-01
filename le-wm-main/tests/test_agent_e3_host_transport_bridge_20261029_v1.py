import hashlib
import pytest
from experiments.spectral_reachability import agent_e3_host_transport_bridge_20261029_v1 as bridge
from experiments.spectral_reachability import agent_e3_feedback_node_20261029_v1 as node
from test_agent_e3_gd_host_cuda_20261029_v1 import body, Model, DEVICE, T


@pytest.mark.parametrize('k,domain',[(1,'Dxy'),(3,'Dx')])
def test_real_disk_host_graph_bridge_and_full_saved_replay(tmp_path,k,domain):
    b=body(k,domain);b['feedback_boundary']=10
    store=bridge.transport.Store(tmp_path/'job');adapter=bridge.Bridge(store)
    store.event('physical','intent','step',0,{})
    row=dict(forwards=[],AD=[],result=None);backend=bridge.host.HostBackend(Model(),DEVICE,T,adapter,b,row)
    with backend.observe():row['result']=node.solve(b,backend,adapter.event)
    store.event('physical','return','step',0,{})
    path=store.output/'events.jsonl';sha=hashlib.sha256(path.read_bytes()).hexdigest()
    actual=bridge.transport.consume(store.output,sha)
    native=bridge.saved_native_events(actual['events']);reconstructed=[]
    saved=bridge.host.SavedBackend(b,row,store.output,reconstructed.append)
    result=node.solve(b,saved,reconstructed.append);saved.complete()
    assert result==row['result'] and native==reconstructed
    assert actual['counts']['physical:step']==dict(attempted=1,returned=1,failed=0)


def test_forged_descriptor_rejected_before_event(tmp_path):
    store=bridge.transport.Store(tmp_path/'job');b=bridge.Bridge(store)
    import numpy as np
    desc=b.array(np.zeros(2,np.float32));desc['sha256']='0'*64
    with pytest.raises(ValueError,match='owned'):b.event(dict(kind='x',array=desc))
    assert store.index==0
