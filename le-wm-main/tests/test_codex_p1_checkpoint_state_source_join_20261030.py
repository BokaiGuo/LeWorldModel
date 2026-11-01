import json

import pytest

from experiments.spectral_reachability import codex_p1_checkpoint_state_source_join_20261030 as audit


def test_saved_projection_and_unknowns():
    out = audit.project(audit.load_pinned())
    assert out["state_projection"]["count"] == 303
    assert set(out["state_projection"]["matches"]) == {
        "pred_proj.net.1.running_mean", "projector.net.1.running_mean"}
    assert out["source_boundaries"]["training_disjoint"] == "unknown"
    assert out["source_boundaries"]["development_disjoint"] == "unknown"
    assert not out["source_boundaries"]["P1_admitted"]


@pytest.mark.parametrize("field", ["weights_sha256", "config_sha256"])
def test_checkpoint_mismatch_rejected(field):
    blobs = audit.load_pinned()
    identity = json.loads(blobs["identity"])
    identity["checkpoint"][field] = "0" * 64
    blobs["identity"] = json.dumps(identity).encode()
    with pytest.raises(ValueError, match="boundary mismatch"):
        audit.project(blobs)


def test_pinned_source_drift_rejected(tmp_path):
    files = {"identity": ("source", "0" * 64)}
    (tmp_path / "source").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA mismatch"):
        audit.load_pinned(tmp_path, files)


def test_incomplete_state_projection_rejected():
    blobs = audit.load_pinned()
    identity = json.loads(blobs["identity"])
    identity["reference"]["persistent_state"].pop(next(iter(identity["reference"]["persistent_state"])))
    blobs["identity"] = json.dumps(identity).encode()
    with pytest.raises(ValueError, match="boundary mismatch"):
        audit.project(blobs)
