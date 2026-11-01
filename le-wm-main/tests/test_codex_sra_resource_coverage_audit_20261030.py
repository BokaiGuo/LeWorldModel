import json

import pytest

from experiments.spectral_reachability import codex_sra_resource_coverage_audit_20261030 as a


def test_real_saved_assets_read_only_and_unknowns():
    result = a.audit()
    assert result["historical_physical_report"]["per_root_predict_samples"] == 88
    assert result["historical_physical_report"]["four_root_predict_samples"] == 352
    assert result["cpu_identity_proof"]["checkpoint_tensor_count"] == 303
    assert all(value is None for value in result["unavailable"].values())
    assert result["current_asset_sizes_are_not_runtime_io"]
    assert result["equal_total_budget"] is False
    assert not result["cpu_identity_proof"]["assets"]["weights"]["current_content_reverified"]


def test_source_drift_rejected(tmp_path):
    path = tmp_path / "changed.json"
    path.write_text(json.dumps({"changed": True}))
    with pytest.raises(ValueError, match="SHA mismatch"):
        a.audit(historical=path)


def test_missing_checkpoint_remains_unknown(tmp_path):
    proof = a.anchored_json(a.PROOF, a.ANCHORS[1])
    proof["checkpoint"]["weights_path"] = str(tmp_path / "missing")
    raw = json.dumps(proof).encode()
    path = tmp_path / "proof.json"
    path.write_bytes(raw)
    import hashlib
    result = a.audit(proof=path, anchors=(a.ANCHORS[0], hashlib.sha256(raw).hexdigest()))
    assert result["cpu_identity_proof"]["assets"]["weights"]["current_file_size_bytes"] is None
