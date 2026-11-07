"""Validate actual GPU run receipts; do not promote scientific gates."""
import json
from pathlib import Path

from experiments.pc_input_permissions import RGB, STATE, file_hash
from experiments.pc_track_registry import eligible, empty_gate_ledger

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results/pc_dual_track_smoke_20261106_codex"
PROTOCOL = ROOT.parent / "docs/PC_DUAL_TRACK_IMPLEMENTATION_PROTOCOL_20261106_CODEX.md"
TASKS = {
    RGB: ROOT / "results/pc_geometric_contact_equivalence_dev_20261106_codex_images.npz",
    STATE: ROOT / "results/pc_real_camera_effect_dev_20261106_codex.npz",
}


def main():
    output = FOLDER / "receipt_audit.json"
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for track_id in (RGB, STATE):
        path = FOLDER / track_id / "result.json"
        record = json.loads(path.read_text())
        kwargs = dict(algorithm_version="input_boundary_smoke_v1", protocol_hash=file_hash(PROTOCOL),
                      task_manifest_hash=file_hash(TASKS[track_id]))
        eligible(record, target_track=track_id, **kwargs)
        other = STATE if track_id == RGB else RGB
        try:
            eligible(record, target_track=other, **kwargs)
        except ValueError:
            rejected = True
        else:
            raise AssertionError("cross-track promotion accepted")
        ledger = empty_gate_ledger(track_id, kwargs["algorithm_version"])
        rows.append({"track_id": track_id, "source_sha256": file_hash(path),
                     "runtime_calls": len(record["input_receipts"]),
                     "max_direct_error": max(row["direct_max_error"] for row in record["rows"]),
                     "cross_track_rejected": rejected, "scientific_gates": ledger})
        print(f"{track_id}: actual receipts validated, cross-track rejected", flush=True)
    with output.open("x") as stream:
        json.dump({"rows": rows, "audit_code_sha256": file_hash(__file__),
                   "registry_code_sha256": file_hash(Path(__file__).with_name("pc_track_registry.py"))},
                  stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
