"""Transport inventory fixtures; no model or physical execution."""
import numpy as np
import pytest
from experiments.spectral_reachability import p3_matched_order_feedback_store_codex_20261029 as store
from experiments.spectral_reachability import p3_matched_order_feedback_store_saved_codex_20261029 as saved


def test_complete_inventory_and_tamper(tmp_path):
    s = store.Store(tmp_path/'run')
    descriptor = s.array('encoding','latent',np.zeros((192,),np.float32))
    s.event('encoding','record','saved',s.index,dict(array=descriptor))
    sha = store.base.fsha((s.output/'events.jsonl').read_bytes())
    result = saved.inspect(s.output,sha)
    assert result['transport_complete'] and not result['physical_certified'] and not result['full_saved_strict']
    (s.output/descriptor['file']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='NPY SHA'):
        saved.inspect(s.output,sha)


def test_unreferenced_artifact_is_rejected(tmp_path):
    s = store.Store(tmp_path/'run')
    s.array('encoding','latent',np.zeros((192,),np.float32))
    s.event('physical','record','marker',s.index,{})
    sha = store.base.fsha((s.output/'events.jsonl').read_bytes())
    with pytest.raises(ValueError,match='all and only'):
        saved.inspect(s.output,sha)
