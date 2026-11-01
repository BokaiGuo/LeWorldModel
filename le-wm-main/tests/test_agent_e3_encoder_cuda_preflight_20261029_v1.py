from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import agent_e3_encoder_cuda_preflight_20261029_v1 as pre


def test_fixed_synthetic_frames_and_authorization(tmp_path):
    assert pre.fixed_frame(10).shape==(224,224,3) and pre.fixed_frame(10).dtype==np.uint8
    assert np.all(pre.fixed_frame(10)==0) and np.all(pre.fixed_frame(15)==127)
    with pytest.raises(ValueError):pre.fixed_frame(True)
    with pytest.raises(ValueError,match='authorization'):pre.run('/missing','0'*64,tmp_path/'job','wrong')
    assert not (tmp_path/'job').exists()


def test_guard_snapshot_real_store_and_corruption(tmp_path):
    store=pre.host.launch.pub.Store(tmp_path/'job');value={'model':'fixture','nested':list(range(5000))}
    pre.snapshot(store,'before',value)
    events=[pre.host.launch.pub.base.decode(x) for x in (store.output/'events.jsonl').read_bytes().splitlines()]
    result,index=pre.read_snapshot(events,'before');assert result==value and index==events[-1]['index']
    for mutation in ('hash','order','bytes','content'):
        altered=deepcopy(events)
        if mutation=='hash':altered[-1]['detail']['sha256']='0'*64
        if mutation=='order':altered[0]['detail']['index']=1
        if mutation=='bytes':altered[-1]['detail']['bytes']+=1
        if mutation=='content':altered[0]['detail']['data']='A'+altered[0]['detail']['data'][1:]
        with pytest.raises(ValueError):pre.read_snapshot(altered,'before')


def test_guard_missing_does_not_certify():
    with pytest.raises(ValueError,match='complete guard'):pre.read_snapshot([],'after')
