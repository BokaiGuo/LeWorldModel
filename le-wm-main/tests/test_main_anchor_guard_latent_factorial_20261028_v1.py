from copy import deepcopy
import gzip
import io

import numpy as np
import pytest
from experiments.spectral_reachability import main_anchor_guard_latent_factorial_20261028_v1 as m


def test_three_links_are_distinct():
    a=dict(branch=0,model_Q=2.,actual_Q=2.,metrics={k:2. for k in m.AXES})
    b=dict(branch=0,model_Q=1.,actual_Q=3.,metrics={k:1. for k in m.AXES})
    r=m.contrast(a,b)
    assert r['model_to_actual_reversal']
    assert all(r['actual_to_physical_reversal'].values())
    b['actual_Q']=1.
    assert not m.contrast(a,b)['model_to_actual_reversal']
    b['metrics']['angle']=3.
    assert m.contrast(a,b)['actual_to_physical_reversal']['angle']


def test_ties_never_count_as_strict_reversals():
    a=dict(branch=0,model_Q=1.,actual_Q=1.,metrics={k:1. for k in m.AXES})
    b=deepcopy(a);b['model_Q']=2.
    r=m.contrast(a,b)
    assert not r['model_to_actual_reversal'] and not any(r['actual_to_physical_reversal'].values())
    b['branch']=1
    with pytest.raises(ValueError):m.contrast(a,b)


@pytest.mark.parametrize('bad',['shape','dtype','nan','trailing','hash','path'])
def test_original_npy_no_repair_or_tolerance(tmp_path,bad):
    z=np.arange(192,dtype=np.float32)
    if bad=='shape':z=z[None]
    if bad=='dtype':z=z.astype(np.float64)
    if bad=='nan':z[0]=np.nan
    stream=io.BytesIO();np.save(stream,z,allow_pickle=False)
    data=stream.getvalue()+(b'extra' if bad=='trailing' else b'')
    path=tmp_path/'actual.npy.gz';path.write_bytes(gzip.compress(data))
    e=dict(file=path.name,sha256=m.sha(path),dtype='float32',shape=[192])
    if bad=='hash':e['sha256']='0'*64
    if bad=='path':e['file']='../actual.npy.gz'
    with pytest.raises(ValueError):m.latent(tmp_path,e)


def test_q_uses_whole192_float64_difference():
    z=np.arange(192,dtype=np.float32);g=np.zeros(192,dtype=np.float32)
    assert m.q(z,g)==float(np.mean(z.astype(np.float64)**2))
    with pytest.raises(ValueError):m.q(z[:-1],g[:-1])
