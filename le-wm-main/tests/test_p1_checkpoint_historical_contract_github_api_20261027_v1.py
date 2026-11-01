import base64
import hashlib

import pytest

from experiments.spectral_reachability import p1_checkpoint_historical_contract_github_api_20261027_v1 as audit


def blob(content=b'public source\n'):
    sha=hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
    return dict(sha=sha,size=len(content),encoding='base64',content=base64.b64encode(content).decode()),sha


def test_exact_git_blob_digest():
    data,sha=blob()
    assert audit.decode_blob(data,sha)==b'public source\n'


@pytest.mark.parametrize('change',[{'sha':'wrong'},{'size':audit.MAX_BLOB+1},{'content':'!!!'},{'size':1},{'encoding':'raw'}])
def test_fail_closed_on_identity_size_or_encoding(change):
    data,sha=blob()
    data.update(change)
    with pytest.raises(ValueError):
        audit.decode_blob(data,sha)
