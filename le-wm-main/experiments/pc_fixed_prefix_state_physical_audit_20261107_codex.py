"""Audit trajectory clock and the previously recorded five-step zero control."""
import json

import numpy as np

from experiments import pc_camera_multiview_dev_20261106_codex as old
from experiments import pc_fixed_prefix_state_physical_20261107_codex as replay
from experiments.pc_input_permissions import file_hash

OUT = replay.ROOT / "results/pc_fixed_prefix_state_physical_audit_20261107_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    result = json.loads(replay.OUT.read_text())
    if result["selection_sha256"] != file_hash(replay.SELECT) or result["runner_sha256"] != file_hash(replay.__file__):
        raise ValueError("replay binding drift")
    with np.load(old.PRIVATE, allow_pickle=False) as prior:
        reference = prior["traces"][0, 0, 0].copy()
    zero = np.asarray(result["rows"]["zero_control"]["trace"])
    delta = np.abs(zero[:6, :4] - reference)
    np.testing.assert_allclose(zero[:6, :4], reference, atol=1e-10, rtol=0)
    row = {}
    for name in ("selected", "zero_control"):
        trace = np.asarray(result["rows"][name]["trace"])
        np.testing.assert_allclose(np.diff(trace[:, 4]), 0.05, atol=1e-10, rtol=0)
        goal_input = np.load(old.PRIVATE, allow_pickle=False)
        try:
            goal = goal_input["traces"][1, 0, 0, 0, :2]
        finally:
            goal_input.close()
        distance = np.linalg.norm(trace[:, :2] - goal, axis=1)
        matches = np.flatnonzero(distance < .5)
        row[name] = {"first_success_step": int(matches[0]) if len(matches) else None,
                     "success_steps": int(len(matches)), "minimum_distance": float(distance.min())}
        if bool(len(matches)) != result["rows"][name]["ever_success"]:
            raise ValueError("success mismatch")
    audit = {"replay_sha256": file_hash(replay.OUT), "prior_physical_sha256": file_hash(old.PRIVATE),
             "runner_sha256": file_hash(__file__), "zero_first_five_max_error": float(delta.max()),
             "rows": row, "scope": "same MuJoCo backend and initial state; not independent physics"}
    with OUT.open("x") as f:
        json.dump(audit, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in audit.items() if k != "scope"}, indent=2))


if __name__ == "__main__":
    main()
