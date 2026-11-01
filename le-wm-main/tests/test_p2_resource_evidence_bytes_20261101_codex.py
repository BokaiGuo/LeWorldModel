import copy
import hashlib

import pytest

from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as gate
from experiments.spectral_reachability import p2_resource_evidence_bytes_20261101_codex as check


def sample(tmp_path):
    path = tmp_path / "evidence"
    path.write_bytes(b"saved event bytes")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {key: "a" * 64 for key in gate.IDENTITY}
    contract = {
        "schema": gate.NAME + ".contract.v1",
        "arms": ["composition", "joint"],
        "scope": "per_root_per_arm_standalone_planning_and_selected_physical_execution",
        "identity": identity,
        "axes": {key: {"unit": unit, "aggregation": "max" if key == "peak_device_bytes" else "sum",
                       "cap": 10} for key, unit in gate.AXES.items()},
    }
    contract_sha = gate.sha(gate.canonical(contract))
    rows = []
    for arm in contract["arms"]:
        axes = {}
        for key, unit in gate.AXES.items():
            axes[key] = {"unit": unit, "coverage": "complete", "evidence_sha256": digest}
            axes[key].update({"attempted": 1, "completed": 1} if key in gate.COUNTS else {"value": 1})
        rows.append({"schema": gate.NAME + ".receipt.v1", "contract_sha256": contract_sha,
                     "arm": arm, "run_id": arm, "identity": copy.deepcopy(identity),
                     "status": "complete", "axes": axes})
    return contract, contract_sha, rows, {digest: path}


def test_bytes_only_and_no_admission(tmp_path):
    c, sha, rows, paths = sample(tmp_path)
    result = check.verify(c, sha, rows, paths)
    assert result["distinct_evidence_files_verified"] == 1
    assert result["evidence_bytes_verified"]
    assert not result["runtime_evidence_verified"] and not result["P2_admission"]


def test_missing_extra_and_changed_evidence_rejected(tmp_path):
    c, sha, rows, paths = sample(tmp_path)
    with pytest.raises(ValueError, match="exactly cover"):
        check.verify(c, sha, rows, {})
    with pytest.raises(ValueError, match="exactly cover"):
        check.verify(c, sha, rows, {**paths, "b" * 64: tmp_path})
    next(iter(paths.values())).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        check.verify(c, sha, rows, paths)


def test_symlink_and_directory_rejected(tmp_path):
    c, sha, rows, paths = sample(tmp_path)
    digest, path = next(iter(paths.items()))
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        check.verify(c, sha, rows, {digest: link})
    with pytest.raises(ValueError, match="regular"):
        check.verify(c, sha, rows, {digest: tmp_path})


def test_original_strict_contract_still_enforced(tmp_path):
    c, sha, rows, paths = sample(tmp_path)
    rows[0]["axes"]["flops"]["coverage"] = "unknown"
    with pytest.raises(ValueError, match="coverage"):
        check.verify(c, sha, rows, paths)
