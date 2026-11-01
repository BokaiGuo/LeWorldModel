"""Validate sealed Cube P1 inputs before any physical/visual interpretation.

This is an integrity check, not prediction or efficacy evidence.  Hashes bind
the actual files, and row checks bind the packets and scores to root identity.
No state is modified and no checkpoint is loaded.
"""
from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _require_passing(report: dict[str, Any], schema: str) -> None:
    gates = report.get("gates")
    if (report.get("schema") != schema or report.get("status") != "pass"
            or not isinstance(gates, dict) or not gates
            or any(value is not True for value in gates.values())):
        raise ValueError(f"inadmissible sealed {schema} report")


def validate_binding(live_panel: Path, p1_evidence: Path, policy: Path) -> dict[str, Any]:
    """Return model rows in panel order, rejecting stale/tampered evidence."""
    panel_path = live_panel / "report.json"
    inputs_path = live_panel / "model_inputs.npz"
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    evidence = json.loads(p1_evidence.read_text(encoding="utf-8"))
    _require_passing(panel, "cube_d2p_live_runtime_panel.v1")
    _require_passing(evidence, "cube_d2p_p1_live_panel_preflight.v1")
    upstream = evidence.get("upstream_live_panel", {})
    report_hash, inputs_hash = _sha256(panel_path), _sha256(inputs_path)
    if upstream.get("report_sha256") != report_hash:
        raise ValueError("model-side evidence is bound to a different panel report")
    if (panel.get("model_inputs", {}).get("sha256") != inputs_hash
            or upstream.get("inputs_sha256") != inputs_hash):
        raise ValueError("sealed model inputs hash mismatch")
    policy_meta = evidence.get("policy", {})
    weights_hash = _sha256(policy / "weights.pt")
    if policy_meta.get("weights_sha256") != weights_hash:
        raise ValueError("policy weights differ from sealed model-side scores")
    manifest_path = policy / "repair_manifest.json"
    manifest_hash = _sha256(manifest_path) if manifest_path.is_file() else None
    if policy_meta.get("repair_manifest_sha256") != manifest_hash:
        raise ValueError("policy repair manifest differs from sealed model-side evidence")

    roots, items = panel.get("rows", []), evidence.get("rows", [])
    root_ids = [root["root_id"] for root in roots]
    item_ids = [item["root_id"] for item in items]
    if (not roots or len(set(root_ids)) != len(root_ids)
            or len(set(item_ids)) != len(item_ids) or set(root_ids) != set(item_ids)):
        raise ValueError("sealed root identities differ or contain duplicates")
    by_root = {item["root_id"]: item for item in items}
    ordered = [by_root[root_id] for root_id in root_ids]
    with np.load(inputs_path, allow_pickle=False) as inputs:
        expected_arrays = {"history_rgb", "root_rgb", "goal_rgb", "history_packets", "heldout_packets"}
        if set(inputs.files) != expected_arrays or any(len(inputs[key]) != len(roots) for key in expected_arrays):
            raise ValueError("sealed model input arrays/root counts differ")
        for index, (root, item) in enumerate(zip(roots, ordered, strict=True)):
            fields = (("root_rgb", "root_rgb_sha256"), ("goal_rgb", "goal_rgb_sha256"),
                      ("history_packets", "history_packets_sha256"),
                      ("heldout_packets", "heldout_next_packet_sha256"))
            if any(_array_hash(inputs[key][index]) != root.get(field) for key, field in fields):
                raise ValueError("sealed root tensor hash mismatch")
            if [_array_hash(frame) for frame in inputs["history_rgb"][index]] != root.get("history_frame_sha256"):
                raise ValueError("sealed history frame hashes differ")
            scores = np.asarray(item.get("proposal_goal_costs"), dtype=np.float64)
            packets = np.asarray(item.get("proposal_raw_packets"), dtype=np.float32)
            if (scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all()
                    or packets.shape != (len(scores), 5, 5)
                    or not np.isfinite(packets).all() or np.any(np.abs(packets) > 1)):
                raise ValueError("sealed candidate scores/physical packets are invalid")
            positive = item.get("positive_control_index")
            if (not isinstance(positive, int) or not 0 < positive < len(scores)
                    or not np.array_equal(packets[positive], inputs["heldout_packets"][index])
                    or np.any(packets[0] != 0)):
                raise ValueError("sealed positive/zero packet identity mismatch")
            selected = int(np.argmin(scores))
            if (item.get("selected_proposal_index") != selected
                    or not np.array_equal(np.asarray(item.get("selected_raw_packet"), np.float32), packets[selected])
                    or item.get("selected_goal_cost") != scores[selected]
                    or item.get("zero_goal_cost") != scores[0]):
                raise ValueError("sealed selected packet/score differs from argmin")
    return {"panel": panel, "evidence": evidence, "ordered_model_rows": ordered,
            "hashes": {"live_panel_report_sha256": report_hash,
                       "model_inputs_sha256": inputs_hash,
                       "p1_evidence_sha256": _sha256(p1_evidence),
                       "policy_weights_sha256": weights_hash,
                       "policy_repair_manifest_sha256": manifest_hash}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-panel", type=Path, required=True)
    parser.add_argument("--p1-evidence", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("binding receipt must be new")
    binding = validate_binding(args.live_panel, args.p1_evidence, args.policy)
    result = {"schema": "cube_d2p_p1_evidence_binding.v1", "status": "pass",
              "root_ids": [row["root_id"] for row in binding["ordered_model_rows"]],
              "upstream": binding["hashes"],
              "claim_boundary": "File and row integrity only; no prediction, CUDA, or efficacy claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "root_count": len(result["root_ids"])}, sort_keys=True))


if __name__ == "__main__":
    main()
