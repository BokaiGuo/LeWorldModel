"""Pure-array diagnosis tests, no model/native simulation."""
import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_raster_first_stage_diagnosis_20261028_v1 as m


def test_null_first_stage_cannot_identify_encoder_invariance():
    rgb=np.zeros((224,224,3),np.uint8);features=np.zeros((256,384),np.float32)
    assert m.first_stage(m.pixels(rgb,rgb),m.compare_arrays(features,features))=='null_pixel_first_stage_no_encoder_invariance_test'


def test_changed_pixels_with_identical_features_distinguished():
    a=np.zeros((224,224,3),np.uint8);b=a.copy();b[5,9]=[144,238,144]
    f=np.zeros((256,384),np.float32)
    p=m.pixels(a,b)
    assert p['unequal_pixels']==1 and p['bbox_yxyx']==[5,9,5,9]
    assert m.first_stage(p,m.compare_arrays(f,f))=='changed_pixels_identical_features_saved_invariance'


def test_actual_feature_sensitivity_branch():
    a=np.zeros((224,224,3),np.uint8);b=a.copy();b[1,1,0]=1
    f=np.zeros((256,384),np.float32);g=f.copy();g[0,0]=.5
    assert m.first_stage(m.pixels(a,b),m.compare_arrays(f,g))=='changed_pixels_and_features_sensitive'


def test_identical_pixels_different_features_flagged():
    a=np.zeros((224,224,3),np.uint8);f=np.zeros((1,),np.float32);g=np.ones((1,),np.float32)
    assert m.first_stage(m.pixels(a,a),m.compare_arrays(f,g))=='identical_pixels_but_features_differ_requires_investigation'


def test_signed_zero_byte_identity_not_arrayequal_only():
    a=np.array([0.],np.float32);b=np.array([-0.],np.float32)
    r=m.compare_arrays(a,b)
    assert not r['bytes_equal'] and r['unequal_elements']==0 and r['mse']==0.


@pytest.mark.parametrize('bad',['shape','dtype','nan'])
def test_bad_array_abi_rejected(bad):
    a=np.zeros((2,),np.float32);b=a.copy()
    if bad=='shape':b=np.zeros((3,),np.float32)
    elif bad=='dtype':b=b.astype(np.float64)
    else:b[0]=np.nan
    with pytest.raises(ValueError):m.compare_arrays(a,b)


def test_one_ulp_is_nonidentical():
    a=np.ones((1,),np.float32);b=np.nextafter(a,np.float32(2))
    r=m.compare_arrays(a,b)
    assert not r['bytes_equal'] and r['mse']>0
