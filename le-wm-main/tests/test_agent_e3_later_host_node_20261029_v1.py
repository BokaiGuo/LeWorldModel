import time
import pytest
from experiments.spectral_reachability import agent_e3_later_host_node_20261029_v1 as n
from test_agent_e3_gd_host_cuda_20261029_v1 import body, Model, DEVICE, T


@pytest.mark.parametrize('k,domain',[(1,'Dxy'),(1,'Dx'),(3,'Dxy'),(3,'Dx')])
def test_real_disk_fake_cuda_forward_reverse_and_saved(k,domain,tmp_path):
    b=body(k,domain);b['feedback_boundary']=10
    store=n.host.Store(tmp_path/'node',time.monotonic(),k)
    row=n.execute(b,Model(),DEVICE,T,store,lambda:{'state':'constant'})
    events=n.host.journal(store.output,k);saved=n.consume(b,row,store.output,events)
    assert saved['result']==row['result'] and not saved['authenticated_actual_feedback']
    assert len(row['forwards'])==68*k and len(row['AD'])==32


def test_environment_mutation_rejected_and_partial_written(tmp_path):
    b=body();b['feedback_boundary']=15;store=n.host.Store(tmp_path/'node',time.monotonic(),1)
    states=iter([{'x':1},{'x':2}])
    with pytest.raises(ValueError,match='mutate'):
        n.execute(b,Model(),DEVICE,T,store,lambda:next(states))
    assert (store.output/'later_node_partial.json').is_file()
