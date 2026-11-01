import json

import pytest

from experiments.spectral_reachability import p3_online_pair_sealed_audit_20261102_codex_exclusive as audit


def test_real_host_artifacts_reconstruct_exactly():
    result = audit.build()
    saved = json.loads((audit.ROOT / "results" /
                        (audit.NAME + "_RESULT01.json")).read_bytes())
    assert result == saved
    assert result["batch_status"] == "partial_or_failed"
    assert len(result["slots"]) == 4
    assert result["identity"]["task_bank_reconstructed_by_host_verify_prepare"]
    assert result["identity"]["dynamic_source_closure"] == "unknown"
    assert not result["identity"]["independent_model_array_recomputation"]
    assert [x["allocations"]["signal"] for x in result["comparisons"]] == [[11, 9], [16, 14]]
    assert [x["allocations"]["reverse"] for x in result["comparisons"]] == [[9, 11], [14, 16]]
    assert all(x["classification"] == "paired_distinct_allocation"
               and x["equal_recorded_axes"] and not x["equal_total_compute"]
               for x in result["comparisons"])
    assert not result["new_model_execution"] and not result["P3_admission"]


def test_corrupt_batch_anchor_is_rejected(monkeypatch):
    monkeypatch.setattr(audit, "BATCH_SHA", "0" * 64)
    with pytest.raises(ValueError, match="batch report hash"):
        audit.build()


def test_prepare_verifier_rejects_wrong_digest():
    path = (audit.ROOT / "results" /
            "codex_p3_resolved_host_20261030_v1_RUN_WALL_P20_signal_01/prepare.json")
    with pytest.raises(ValueError, match="isolated original prepare verifier failed"):
        audit.verify_prepare_isolated(path, "0" * 64)


def test_exclusive_writer_refuses_overwrite(tmp_path):
    path = tmp_path / "result.json"
    audit.host.write_new(path, {"first": True})
    with pytest.raises(FileExistsError):
        audit.host.write_new(path, {"second": True})
    assert json.loads(path.read_bytes()) == {"first": True}
