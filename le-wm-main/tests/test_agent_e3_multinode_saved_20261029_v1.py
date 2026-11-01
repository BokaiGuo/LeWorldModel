import hashlib
import pytest
from experiments.spectral_reachability import agent_e3_multinode_saved_20261029_v1 as m
from test_agent_e3_gd_host_cuda_20261029_v1 import body,Model,DEVICE,T


def test_two_real_disk_nodes_and_interleaving(tmp_path):
    s=m.bridge.transport.Store(tmp_path/'job');bodies=[]
    for boundary in (10,15):
        b=body();b['feedback_boundary']=boundary;bodies.append(b)
        s.event('physical','record','frame',s.index,{'boundary':boundary})
        adapter=m.NodeBridge(s,boundary);row=dict(forwards=[],AD=[],result=None)
        backend=m.bridge.host.HostBackend(Model(),DEVICE,T,adapter,b,row)
        with backend.observe():row['result']=m.node.solve(b,backend,adapter.event)
        adapter.finish(row)
    h=hashlib.sha256((s.output/'events.jsonl').read_bytes()).hexdigest()
    out=m.consume(bodies,s.output,h)
    assert [r['boundary'] for r in out['nodes']]==[10,15]
    assert not out['full_native_strict'] and not out['source_model_rgb_authenticated']
    with pytest.raises(ValueError):m.consume(list(reversed(bodies)),s.output,h)


def test_incomplete_node_fails_closed(tmp_path):
    s=m.bridge.transport.Store(tmp_path/'job');m.NodeBridge(s,10)
    h=hashlib.sha256((s.output/'events.jsonl').read_bytes()).hexdigest();b=body();b['feedback_boundary']=10
    with pytest.raises(ValueError,match='sealed'):m.consume([b],s.output,h)
