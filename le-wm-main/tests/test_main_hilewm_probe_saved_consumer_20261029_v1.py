from copy import deepcopy
import pytest
from experiments.spectral_reachability import main_hilewm_probe_saved_consumer_20261029_v1 as m


def test_original_loader_code_bound_metadata_scope_explicit():
    v=m.check_loader_sources({'own_sha256':deepcopy(m.LOADER)}, {'sources':{m.CODE:m.LOADER[m.CODE]}})
    assert v['executed_loader_manifest_bound'] and v['metadata_current_bytes_verified']
    assert not v['metadata_before_after_manifest_verified']
    assert len(v['nonexecuted_metadata_not_in_original_before_after_manifest'])==2


@pytest.mark.parametrize('fault',('receipt','code','contradict','extra','current'))
def test_no_missing_executed_code_or_rewritten_metadata(fault,monkeypatch):
    c={'own_sha256':deepcopy(m.LOADER)};p={'sources':{m.CODE:m.LOADER[m.CODE]}}
    if fault=='receipt':c['own_sha256'].pop(m.CODE)
    elif fault=='code':p['sources'].clear()
    elif fault=='contradict':p['sources'][next(k for k in m.LOADER if k!=m.CODE)]='0'*64
    elif fault=='extra':c['own_sha256']['unknown']='0'*64
    else:monkeypatch.setattr(m,'sha',lambda path:'0'*64)
    with pytest.raises(ValueError):m.check_loader_sources(c,p)
