"""Audit dataset-vs-rendered RGB target alignment and paired probe sign changes."""
import json

import numpy as np

from experiments import pc_tworoom_goal_render_alignment_20261107_codex as run
from experiments.pc_input_permissions import file_hash
from experiments.pc_tworoom_actual_prefix_action_effect_analysis_20261107_codex import agreement, signed


def main():
    data = json.loads(run.OUT.read_text())
    if (data["source_sha256"] != file_hash(run.source.OUT) or data["input_sha256"] != file_hash(run.run.INPUT)
            or data["protocol_sha256"] != file_hash(run.PROTOCOL)
            or data["runner_sha256"] != file_hash(run.__file__) or data["feature_sha256"] != file_hash(run.FEATURES)):
        raise ValueError("artifact hash mismatch")
    rows = data["rows"]
    assert len(rows) == 40
    print("max state and goal error", max(r["state_error"] for r in rows), max(r["goal_state_error"] for r in rows))
    for key in ("current_rgb", "goal_rgb"):
        print(key, "exact", sum(r[key]["max"] == 0 for r in rows),
              "max_mae", max(r[key]["mae"] for r in rows),
              "max_pixel_difference", max(r[key]["max"] for r in rows))
    print("max goal encoding MSE", max(r["goal_feature_mse"] for r in rows))
    for arm in run.source.source.ARMS:
        data_arm = [r["arms"][arm] for r in rows if arm in r["arms"]]
        print(arm, "n", len(data_arm), "max goal reencoding error",
              max(r["goal_reencode_max_abs_error"] for r in data_arm))
        for kind in ("predicted", "encoded"):
            old, new, physical = [], [], []
            for row in data_arm:
                o, n, distance = (np.asarray(row[key]) for key in
                                  ("old_" + kind, "rendered_" + kind, "physical_distance"))
                old.extend(o[0] - o[1:5])
                new.extend(n[0] - n[1:5])
                physical.extend(distance[0] - distance[1:5])
            print(kind, "old", agreement(old, physical, 1e-6, 1e-5),
                  "rendered", agreement(new, physical, 1e-6, 1e-5),
                  "effect sign changes", int(np.sum(signed(old, 1e-6) != signed(new, 1e-6))))


if __name__ == "__main__":
    main()
