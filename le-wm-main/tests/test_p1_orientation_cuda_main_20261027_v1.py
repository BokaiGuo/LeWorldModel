"""Array tests, not evidence of CUDA/physical efficacy."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import p1_orientation_cuda_main_20261027_v1 as m


def frames():
    out=[]
    for r in range(4):
        for k in ('original','clean'):
            out.append(dict(root_index=r,order=len(out),id=k,pixel_sha256=k))
        for angle in range(-180,180,30):
            for sampler in ('nearest','bilinear'):
                key=f'{angle}_{sampler}'
                out.append(dict(root_index=r,order=len(out),id=key,pixel_sha256=key,
                    angle_degrees=angle,sampler=sampler))
    return out


def test_all_images_and_known_mse():
    f=frames(); a=np.zeros((104,192));a[2]=2
    result=m.analyze(f,a)
    assert result['rows'][0]['contrasts'][0]['nearest_to_clean_mse']==4
    assert result['rows'][0]['contrasts'][0]['sampler_pair_mse']==4
    assert result['unique_pixel_inputs']==26
    assert result['maximum_exact_input_repeat_abs_difference']==2
    assert result['P1_admission'] is False and result['new_Q'] is False


@pytest.mark.parametrize('shape',[(103,192),(104,191),(1,104,192)])
def test_shapes_reject(shape):
    with pytest.raises(ValueError):m.analyze(frames(),np.zeros(shape))


def test_nonfinite_reject():
    a=np.zeros((104,192));a[4,2]=np.nan
    with pytest.raises(ValueError):m.analyze(frames(),a)


def test_actual_catalog_order_and_analysis():
    _, manifest=m.catalog.build()
    result=m.analyze(manifest['frames'],np.zeros((104,192)))
    assert result['unique_pixel_inputs']==27
    assert result['repeated_input_pairs']==77


@pytest.mark.parametrize('field,value',[('order',3),('root_index',9),('sampler','bicubic')])
def test_identity_reject(field,value):
    f=deepcopy(frames());f[2][field]=value
    with pytest.raises(ValueError):m.analyze(f,np.zeros((104,192)))
