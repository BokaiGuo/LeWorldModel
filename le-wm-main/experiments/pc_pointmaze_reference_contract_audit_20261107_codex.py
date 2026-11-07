"""Apply the position-hold contract to existing state-assisted physical traces."""
import json
from pathlib import Path

import numpy as np

from experiments.pc_action_reference_contract_20261107_codex import certify_position_hold
from experiments.pc_action_witness_v2_20261107_codex import ActionWitnessV2
from experiments.pc_input_permissions import file_hash

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/pc_real_camera_effect_dev_20261106_codex"
DATA = BASE.with_suffix(".npz")
META = BASE.with_suffix(".json")
AUDIT = Path(str(BASE) + "_audit.json")
PROTOCOL = ROOT.parent / "docs/PC_ACTION_REFERENCE_CONTRACT_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_pointmaze_reference_contract_audit_20261107_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    meta, audit = json.loads(META.read_text()), json.loads(AUDIT.read_text())
    if (meta["payload_sha256"] != file_hash(DATA) or audit["report_sha256"] != file_hash(META)
            or meta["actions"][0] != [0.0, 0.0]
            or audit["status"] != "exact independent-process replay"
            or audit["archive_initial_states_match"] is not True):
        raise ValueError("physical evidence or zero-action identity mismatch")
    rows = []
    print("Check whether the existing five-step zero actions actually hold position.", flush=True)
    with np.load(DATA, allow_pickle=False) as data:
        trace = data["traces"]
        if trace.shape != (8, 3, 2, 6, 4):
            raise ValueError("unexpected trajectory dimensions")
        np.testing.assert_array_equal(trace[:, :, 0], trace[:, :, 1])
        for i in range(8):
            witness = ActionWitnessV2("zero", f"exposed_state_{i}", "5 physical steps", [[0., 0.]] * 5)
            positions = trace[i, 0, 0, :, :2]
            maximum = float(np.linalg.norm(positions - positions[0], axis=1).max())
            try:
                certify_position_hold(witness, positions, tolerance=1e-6, source_sha256=file_hash(DATA))
                accepted = True
            except ValueError:
                accepted = False
            rows.append({"state": i, "position_hold_accepted": accepted,
                         "max_position_displacement": maximum,
                         "initial_velocity_norm": float(np.linalg.norm(trace[i, 0, 0, 0, 2:])),
                         "action_fingerprint": witness.fingerprint})
    with OUTPUT.open("x") as stream:
        json.dump({"track": "official_state_assisted", "scope": "exposed physical diagnosis, not RGB-only",
                   "tolerance": 1e-6, "rows": rows, "data_sha256": file_hash(DATA),
                   "meta_sha256": file_hash(META), "audit_sha256": file_hash(AUDIT),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__)},
                  stream, allow_nan=False)
    print(f"accepted={sum(r['position_hold_accepted'] for r in rows)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
