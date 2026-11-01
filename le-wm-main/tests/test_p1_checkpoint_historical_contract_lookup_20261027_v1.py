"""Transport/provenance safeguards only: no transform, neural or physics fixtures."""
import io
import json
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import p1_checkpoint_historical_contract_lookup_20261027_v1 as audit


@pytest.mark.parametrize('url',[
    'https://huggingface.co/quentinll/lewm-pusht/resolve/main/weights.pt',
    'https://huggingface.co/datasets/quentinll/lewm-pusht/resolve/main/data.tar.zst',
    'https://api.github.com/repos/lucas-maes/le-wm/zipball/main',
    'http://api.github.com/repos/lucas-maes/le-wm',
    'https://example.com/README.md',
    'https://raw.githubusercontent.com/lucas-maes/le-wm/main/assets/lewm.gif',
])
def test_disallow_large_or_out_of_scope(url):
    assert not audit.allowed(url)


def test_small_public_metadata_allowed():
    assert audit.allowed('https://huggingface.co/api/models/quentinll/lewm-pusht/tree/main')
    assert audit.allowed('https://raw.githubusercontent.com/lucas-maes/le-wm/abc/config/train/data/pusht.yaml')


def test_size_header_and_stream_caps():
    with pytest.raises(ValueError,match='Content-Length'):
        audit.read_bounded(SimpleNamespace(headers={'Content-Length':'100'},read=lambda n:b''),limit=8)
    with pytest.raises(ValueError,match='stream exceeds'):
        audit.read_bounded(SimpleNamespace(headers={},read=io.BytesIO(b'123456789').read),limit=8)


def test_remote_digest_requires_pinned_revision_and_exact_lfs(tmp_path):
    path=tmp_path/'metadata.json'
    obj=dict(sha=audit.REVISION,siblings=[dict(rfilename='weights.pt',lfs=dict(sha256=audit.WEIGHTS))])
    path.write_text(json.dumps(obj))
    record=dict(label='hf_pinned_model',status='recorded',http_status=200,body_path=str(path),body_sha256=audit.sha(path))
    assert audit.hf_digest_witness([record])['status']=='verified_remote_LFS_identity'
    obj['sha']='different'
    path.write_text(json.dumps(obj))
    assert audit.hf_digest_witness([record])['status']=='unknown'
    assert audit.hf_digest_witness([])['status']=='unknown'


def test_no_redirect_following():
    assert audit.NoRedirect().redirect_request(None,None,302,'',{},'https://example.com') is None


def test_existing_output_rejected(tmp_path):
    with pytest.raises(ValueError,match='exclusive NEW'):
        audit.collect(tmp_path)
