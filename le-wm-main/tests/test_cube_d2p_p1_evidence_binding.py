"""Synthetic integrity tests only; no learned model or efficacy experiment."""
import json

import numpy as np
import pytest

from experiments.spectral_reachability.cube_d2p_p1_evidence_binding import _array_hash, _sha256, validate_binding


@pytest.fixture
def sealed(tmp_path):
    panel = tmp_path / "panel"
    policy = tmp_path / "policy"
    panel.mkdir(); policy.mkdir()
    (policy / "weights.pt").write_bytes(b"synthetic-integrity-fixture")
    (policy / "repair_manifest.json").write_text("{}")
    arrays = {"history_rgb": np.zeros((2, 3, 2, 2, 3), np.uint8),
              "root_rgb": np.zeros((2, 2, 2, 3), np.uint8),
              "goal_rgb": np.ones((2, 2, 2, 3), np.uint8),
              "history_packets": np.zeros((2, 3, 5, 5), np.float32),
              "heldout_packets": np.full((2, 5, 5), 0.25, np.float32)}
    np.savez_compressed(panel / "model_inputs.npz", **arrays)
    roots, items = [], []
    for i in range(2):
        root = {"root_id": f"root-{i}", "history_frame_sha256": [_array_hash(x) for x in arrays["history_rgb"][i]]}
        for key, field in (("root_rgb", "root_rgb_sha256"), ("goal_rgb", "goal_rgb_sha256"),
                           ("history_packets", "history_packets_sha256"), ("heldout_packets", "heldout_next_packet_sha256")):
            root[field] = _array_hash(arrays[key][i])
        roots.append(root)
        packets = np.stack((np.zeros((5, 5), np.float32), arrays["heldout_packets"][i]))
        items.append({"root_id": root["root_id"], "proposal_goal_costs": [1.0, 0.1],
                      "proposal_raw_packets": packets.tolist(), "positive_control_index": 1,
                      "selected_proposal_index": 1, "selected_raw_packet": packets[1].tolist(),
                      "zero_goal_cost": 1.0, "selected_goal_cost": 0.1})
    report = {"schema": "cube_d2p_live_runtime_panel.v1", "status": "pass", "gates": {"fixture": True},
              "rows": roots, "model_inputs": {"sha256": _sha256(panel / "model_inputs.npz")}}
    (panel / "report.json").write_text(json.dumps(report))
    evidence = {"schema": "cube_d2p_p1_live_panel_preflight.v1", "status": "pass", "gates": {"fixture": True},
                "rows": items, "upstream_live_panel": {"report_sha256": _sha256(panel / "report.json"),
                "inputs_sha256": _sha256(panel / "model_inputs.npz")},
                "policy": {"weights_sha256": _sha256(policy / "weights.pt"),
                           "repair_manifest_sha256": _sha256(policy / "repair_manifest.json")}}
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence))
    return panel, path, policy


def test_accepts_sealed_files_and_joins_by_root_identity(sealed):
    panel, path, policy = sealed
    evidence = json.loads(path.read_text())
    evidence["rows"].reverse()
    path.write_text(json.dumps(evidence))
    bound = validate_binding(panel, path, policy)
    assert [row["root_id"] for row in bound["ordered_model_rows"]] == ["root-0", "root-1"]


@pytest.mark.parametrize("target", ["inputs", "report", "weights", "manifest"])
def test_rejects_file_drift(sealed, target):
    panel, path, policy = sealed
    changed = {"inputs": panel / "model_inputs.npz", "report": panel / "report.json",
               "weights": policy / "weights.pt", "manifest": policy / "repair_manifest.json"}[target]
    changed.write_bytes(changed.read_bytes() + b" ")
    with pytest.raises(ValueError):
        validate_binding(panel, path, policy)


@pytest.mark.parametrize("mutation", ["positive", "selected", "duplicate", "nan", "empty_gates"])
def test_rejects_corrupted_candidate_contract(sealed, mutation):
    panel, path, policy = sealed
    evidence = json.loads(path.read_text())
    row = evidence["rows"][0]
    if mutation == "positive": row["proposal_raw_packets"][1][0][0] = 0.5
    elif mutation == "selected": row["selected_proposal_index"] = 0
    elif mutation == "duplicate": evidence["rows"][1]["root_id"] = row["root_id"]
    elif mutation == "nan": row["proposal_goal_costs"][0] = float("nan")
    else: evidence["gates"] = {}
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError):
        validate_binding(panel, path, policy)


def test_rejects_tensor_root_mismatch_even_when_whole_file_hashes_are_updated(sealed):
    panel, path, policy = sealed
    with np.load(panel / "model_inputs.npz") as handle:
        arrays = {key: handle[key] for key in handle.files}
    arrays["goal_rgb"][0, 0, 0, 0] = 42
    np.savez_compressed(panel / "model_inputs.npz", **arrays)
    report = json.loads((panel / "report.json").read_text())
    report["model_inputs"]["sha256"] = _sha256(panel / "model_inputs.npz")
    (panel / "report.json").write_text(json.dumps(report))
    evidence = json.loads(path.read_text())
    evidence["upstream_live_panel"].update(report_sha256=_sha256(panel / "report.json"),
                                       inputs_sha256=_sha256(panel / "model_inputs.npz"))
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="root tensor hash"):
        validate_binding(panel, path, policy)
