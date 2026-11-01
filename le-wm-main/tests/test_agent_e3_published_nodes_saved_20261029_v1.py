import pytest
from experiments.spectral_reachability import agent_e3_published_nodes_saved_20261029_v1 as c
from test_agent_e3_gd_host_cuda_20261029_v1 import body,Model,DEVICE,T


def test_two_nodes_published_tempdisk(tmp_path):
    s=c.pub.Store(tmp_path/'job');bodies=[]
    for boundary in (10,15):
        b=body();b['feedback_boundary']=boundary;bodies.append(b)
        s.event('physical','record','frame',s.index,{'boundary':boundary})
        bridge=c.m.NodeBridge(s,boundary);row=dict(forwards=[],AD=[],result=None)
        backend=c.m.bridge.host.HostBackend(Model(),DEVICE,T,bridge,b,row)
        with backend.observe():row['result']=c.m.node.solve(b,backend,bridge.event)
        bridge.finish(row)
    seal=s.publish('completed')
    out=c.consume(bodies,s.output,seal['report_sha256'],seal['completion_sha256'])
    assert [x['boundary'] for x in out['nodes']]==[10,15]
    assert not out['source_model_rgb_authenticated']
    with pytest.raises(ValueError):c.consume(list(reversed(bodies)),s.output,seal['report_sha256'],seal['completion_sha256'])
