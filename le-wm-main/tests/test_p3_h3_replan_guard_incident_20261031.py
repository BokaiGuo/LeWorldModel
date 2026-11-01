import json

import pytest

from experiments.spectral_reachability import p3_h3_replan_guard_incident_20261031 as audit


def test_guard_incident_binds_manifest_and_does_not_impute():
    manifest = json.loads(audit.summary.MANIFEST.read_bytes())
    batch = audit.summary.source.prepare_batch(manifest, 16)
    folder = audit.ROOT / "results/p3_h3_real_context_physical_20261031_batch16_RUN01"
    incidents = audit.inspect_failed(folder, audit.REPORTS[16], batch, None)
    assert [item["branch"] for item in incidents] == [1, 3, 5]
    assert all(item["returned"] == 17 and not item["in_canvas"]
               and item["position"][0] < 0 and item["fixed_horizon_endpoint"] is None
               for item in incidents)
    assert incidents[0]["position"] == incidents[2]["position"]


def test_rejects_invalid_sealed_report_hash():
    manifest = json.loads(audit.summary.MANIFEST.read_bytes())
    batch = audit.summary.source.prepare_batch(manifest, 16)
    folder = audit.ROOT / "results/p3_h3_real_context_physical_20261031_batch16_RUN01"
    with pytest.raises(ValueError, match="report hash"):
        audit.inspect_failed(folder, "0" * 64, batch, None)
