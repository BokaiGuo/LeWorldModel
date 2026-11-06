import hashlib

import pytest

from experiments.audit_pc_action_opportunity_fresh_cem_20261104_codex import verify_hashes


def test_hash_audit_rejects_modified_source(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("original")
    hashes = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    verify_hashes(hashes)
    source.write_text("modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_hashes(hashes)
