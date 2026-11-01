import numpy as np
import pytest
from experiments.spectral_reachability.pusht_clean_goal_sprite_catalog import build


def image():
    rgb=np.full((224,224,3),255,np.uint8)
    rgb[99:111,99:111]=[120,140,240]  # Anti-aliased/outline pixels, not pure blue.
    rgb[100:110,100:110]=[65,105,225]
    rgb[30:70,30:70]=[119,136,153]
    return rgb


def test_full_sprite_preserved_no_ghost():
    rgb=image();r=build(rgb);bbox=r['source_bbox'];x0,y0,x1,y1=bbox
    assert np.all(r['source_removed_rgb'][y0:y1,x0:x1]==255)
    assert r['old_pure_mask_residual_pixels']==44
    for v in r['valid']:
        assert np.sum(np.all(v['rgb']==[120,140,240],axis=-1))==44
        np.testing.assert_array_equal(v['rgb'][30:70,30:70],rgb[30:70,30:70])
    assert len(r['valid'])+len(r['invalid'])==256


def test_unknown_background_rejected():
    rgb=image();rgb[98,100]=[119,136,153]
    with pytest.raises(ValueError):build(rgb)


def test_missing_agent_rejected():
    with pytest.raises(ValueError):build(np.full((224,224,3),255,np.uint8))
