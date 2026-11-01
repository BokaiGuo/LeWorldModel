"""Numerical mock tests are not physical or CUDA efficacy evidence."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import p1_witness_rgb_cuda_main_20261027_v1 as m


def frames():
    return [dict(order=i,root_sha256=str(i//12),arm=str((i//3)%4),step=(0,5,10)[i%3],
        q=np.eye(192).tolist(),goal=[1.]*192,predicted_native192=[2.]*192,
        physical_diagnostic={'fixture':True}) for i in range(24)]


def test_numerical_objectives_and_census():
    a=m.analyze(frames(),np.zeros((24,192)))
    assert len(a['rows'])==24 and a['P1_admission'] is False
    assert all(r['predicted_Q_mse']==1 and r['actual_Q_mse']==1 and r['native_prediction_mse']==4 for r in a['rows'])


@pytest.mark.parametrize('shape',[(23,192),(24,191)])
def test_short_arrays(shape):
    with pytest.raises(ValueError):m.analyze(frames(),np.zeros(shape))


@pytest.mark.parametrize('field,value',[('order',5),('step',7),('q',[[1.]]),('goal',[0.])])
def test_binding_shape_and_chronology(field,value):
    fs=deepcopy(frames());fs[1][field]=value
    with pytest.raises(ValueError):m.analyze(fs,np.zeros((24,192)))
