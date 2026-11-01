import hashlib

import pytest

from experiments.spectral_reachability import p1_checkpoint_historical_contract_review_20261027_v1 as audit


def test_git_blob_not_plain_file_sha1():
    assert audit.git_blob(b'abc')==hashlib.sha1(b'blob 3\0abc').hexdigest()
    assert audit.git_blob(b'abc')!=hashlib.sha1(b'abc').hexdigest()


def test_raw_response_tamper_fails(tmp_path):
    p=tmp_path/'body'
    p.write_bytes(b'original')
    record=dict(body_path=str(p),body_sha256=audit.lookup.sha(p))
    audit.verify_body(record)
    p.write_bytes(b'edited')
    with pytest.raises(ValueError,match='raw response changed'):
        audit.verify_body(record)


def test_existing_output_fails(tmp_path):
    with pytest.raises(ValueError,match='exclusive NEW'):
        audit.run(tmp_path)
