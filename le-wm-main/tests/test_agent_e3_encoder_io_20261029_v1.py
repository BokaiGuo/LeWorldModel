import io
import numpy as np
import pytest
from PIL import Image
from experiments.spectral_reachability import agent_e3_encoder_io_20261029_v1 as m


def fixture():
    image=np.zeros((224,224,3),np.uint8);b=io.BytesIO();Image.fromarray(image).save(b,format='PNG')
    prep=m.pixels.preprocess_pixels(image);hidden=np.zeros((1,257,192),np.float32)
    return b.getvalue(),dict(pixels=image,preprocessed=prep,encoder_input=prep.reshape(1,3,224,224).copy(),
        encoder_hidden=hidden,projector_input=hidden[:,0,:].copy(),projector_output=np.zeros((1,192),np.float32),latent=np.zeros(192,np.float32))


def test_saved_full_io():
    b,a=fixture();out=m.verify_arrays(b,a);assert out['saved_io_connected'] and not out['neural_parity_recomputed']


@pytest.mark.parametrize('field',['pixels','encoder_input','projector_input','latent'])
def test_bad_connection_fails(field):
    b,a=fixture();a[field]=a[field].copy();a[field].flat[0]=1
    with pytest.raises(ValueError):m.verify_arrays(b,a)


def test_hidden_wrong_shape_fails():
    b,a=fixture();a['encoder_hidden']=np.zeros((1,197,192),np.float32)
    with pytest.raises(ValueError):m.verify_arrays(b,a)
