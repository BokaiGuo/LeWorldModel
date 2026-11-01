"""Disk transport fixture; no neural inference or physics."""
import numpy as np
import pytest
from experiments.spectral_reachability import p3_matched_order_feedback_store_codex_20261029 as mod


def test_fixed_16_encoder_payloads_and_chain(tmp_path):
    store = mod.Store(tmp_path/'run')
    for boundary in range(16):
        token = store.index
        store.event('encoding','intent','encoder',token,dict(boundary=boundary))
        for role,shape,dtype in mod.base.ENC_SHAPES:
            store.array('encoding',role,np.zeros(shape,dtype=dtype))
        store.event('encoding','return','encoder',token,dict(boundary=boundary))
    assert store.used['encoding_arrays'] >= mod.ENCODER_16_RAW_BYTES > 8*mod.M
    path = store.output/'events.jsonl'; sha = mod.base.fsha(path.read_bytes())
    receipt = mod.inspect_journal(path,sha)
    assert receipt['transport_complete'] and not receipt['physical_certified']
    with pytest.raises(ValueError,match='exclusive'):
        mod.Store(store.output)
    with pytest.raises(ValueError,match='journal hash'):
        mod.inspect_journal(path,'0'*64)


def test_unreturned_intent_is_preserved_partial(tmp_path):
    store = mod.Store(tmp_path/'partial')
    store.event('physical','intent','reset',store.index,{})
    path = store.output/'events.jsonl'
    receipt = mod.inspect_journal(path,mod.base.fsha(path.read_bytes()))
    assert not receipt['transport_complete'] and receipt['pending']


def test_write_before_category_cap(tmp_path):
    store = mod.Store(tmp_path/'limited')
    store.used['encoding_arrays'] = mod.LIMITS['encoding_arrays']
    with pytest.raises(ValueError,match='category'):
        store.array('encoding','latent',np.zeros((192,),np.float32))
    assert not (store.output/'a000000.npy').exists()
