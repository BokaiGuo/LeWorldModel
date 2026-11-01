"""Saved tensor fixtures only; no model, CUDA, or physical execution."""
import io
import numpy as np
from PIL import Image
import pytest
from experiments.spectral_reachability import p3_matched_order_feedback_encoder_codex_20261029 as enc
from experiments.spectral_reachability import p3_matched_order_feedback_encoder_saved_codex_20261029 as saved


def test_saved_encoder_io_all_matched_boundaries(tmp_path):
    root = tmp_path/'encoder_fixture'; store = enc.original.u.Store(root)
    image = np.arange(224*224*3, dtype=np.uint32).reshape(224,224,3).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(image).save(buf,format='PNG'); png = buf.getvalue()
    store.write('frame.png',png,'physical_png')
    frame = dict(file='frame.png',sha256=enc.original.u.fsha(png),pixel_sha256=enc.original.u.fsha(image.tobytes()))
    prep = enc.original.pixels.preprocess_pixels(image)
    hidden = np.zeros((1,257,192),np.float32); projector = np.zeros((1,192),np.float32)
    values = dict(pixels=image,preprocessed=prep,encoder_input=prep.reshape(1,3,224,224),
                  encoder_hidden=hidden,projector_input=hidden[:,0,:],projector_output=projector,
                  latent=projector.reshape(192))
    refs = {role:store.array('encoding',role,value) for role,value in values.items()}
    for boundary in (5,10,15,20):
        assert saved.inspect(root,frame,refs,boundary)['saved_io_connected']
    bad = dict(frame,pixel_sha256='0'*64)
    with pytest.raises(ValueError,match='pixel SHA'):
        saved.inspect(root,bad,refs,5)
    with pytest.raises(ValueError,match='matched boundary'):
        saved.inspect(root,frame,refs,25)


def test_encoder_rejects_cpu_before_model_calls():
    class CPU:
        type = 'cpu'
    with pytest.raises(ValueError,match='CUDA boundary'):
        enc.encode(None,None,CPU(),None,None,5,None)
