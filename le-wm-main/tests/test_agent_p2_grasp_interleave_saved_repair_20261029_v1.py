import json
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_grasp_interleave_saved_repair_20261029_v1 as m


def actual_inputs():
    d=m.ROOT/'results/agent_p2_grasp_interleave_cuda_20261029_v2_cuda_MAIN01'
    assert m.host.sha(d/'host.json')=='b1fe5475838e5f1dd21b17fa036684bf45804190cb7d2379f0ed8412538ed311'
    values=json.loads((d/'host.json').read_bytes())['inputs']
    def get(k):
        x=values[k];return np.frombuffer(bytes.fromhex(x['raw_hex']),dtype=x['dtype']).reshape(x['shape']).copy()
    return get('z'),get('v'),get('goal')


def test_actual_384_values_exact_and_old_fraction_fails():
    z,v,goal=actual_inputs();old=np.asarray([1/3,2/3],np.float32).reshape(1,2,1)
    ref=z[:,-1,None]+old*(goal[:,None]-z[:,-1,None])
    assert [int(np.count_nonzero(v[:,i]!=ref[:,i])) for i in range(2)]==[0,136]
    m.fixed_initialization(z,v,goal)
    altered=v.copy();altered.view(np.uint32)[0,1,0]^=1
    with pytest.raises(ValueError,match='exact fixed'):m.fixed_initialization(z,altered,goal)


def test_other_report_rejected_before_io():
    with pytest.raises(ValueError,match='pinned actual'):m.consume('/missing','0'*64)


def test_external_report_pin_and_v2_prepare_uses_original_files():
    p=m.ROOT/'results/agent_p2_grasp_interleave_cuda_20261029_v2_cuda_MAIN01/report.json'
    assert m.host.sha(p)==m.ACTUAL_REPORT
    assert m.own()[0].name=='agent_p2_grasp_interleave_cuda_20261029_v2.py'
