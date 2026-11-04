"""Post-hoc same-bank test of a legal RGB-anchored score hypothesis."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v22_pointmaze_actual_fd_physical_dev_20261104_codex as source
from experiments.pc_v22_pointmaze_probe_score_attribution_20261104_codex import ordering
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = source.PREFIX.with_name(source.PREFIX.name + "_rgb_anchor_score_diagnostic.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    sha = seal.digest
    manifest = json.loads(source.MANIFEST.read_text())
    result = json.loads(source.RESULT.read_text())
    audit_path = source.PREFIX.with_name(source.PREFIX.name + "_audit.json")
    audit = json.loads(audit_path.read_text())
    seal.verify(manifest["seal"])
    if (audit["result_sha256"] != sha(source.RESULT) or audit["arrays_sha256"] != sha(source.ARRAYS)
            or result["manifest_sha256"] != sha(source.MANIFEST)):
        raise ValueError("source not audited or changed")
    live = source.run
    decisions, images, _ = live.paths("response")
    decision = json.loads(decisions[2].read_text())
    camera = json.loads(live.h.CAMERA.read_text())["metadata"]
    offset = json.loads(live.h.OLD.read_text())["body_offset_from_static_scene"]
    with np.load(images, allow_pickle=False) as archive:
        frames = archive["rgb"]
        current = live.h.causal_state(frames[24], frames[25], camera, offset, .05)[:2]
    with np.load(live.h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as archive:
        goal, _ = live.h.estimate_state(archive["goal_rgb"], camera, offset)
    goal = np.asarray(goal[:2], np.float64)
    # The first saved exact zero-action five-step prediction is the fixed
    # baseline, selected without consulting any physical outcomes.
    zero = next(np.asarray(batch["positions"][i], np.float64)
                for batch in decision["trace"]
                for i, action in enumerate(batch["raw"])
                if np.asarray(action).shape == (5, 2) and not np.any(action))
    rows = []
    with np.load(source.ARRAYS, allow_pickle=False) as archive:
        for index, packet in enumerate(archive["actions"]):
            matches = [batch for batch in decision["trace"]
                       if np.array_equal(np.asarray(batch["raw"], np.float32), packet)]
            if len(matches) != 1:
                raise ValueError("probe batch not uniquely bound")
            batch = matches[0]
            predicted = np.asarray(batch["positions"], np.float64)
            reference = archive["native_states"][index, :, -1, :2]
            physical_distance = np.linalg.norm(reference - [1.8, 2.8], axis=1)
            anchored = np.asarray(current, np.float64) + predicted - zero
            new_score = np.linalg.norm(anchored - goal, axis=1)
            old_score = np.asarray(batch["scores"], np.float64)
            rows.append({"nominal": index, "original_order": ordering(old_score, physical_distance),
                         "rgb_anchor_order": ordering(new_score, physical_distance),
                         "original_scores": old_score.tolist(), "rgb_anchor_scores": new_score.tolist()})
    seal.verify(manifest["seal"])
    live.old.write_json(OUT, {"rows": rows, "rgb_current_position": np.asarray(current).tolist(),
                              "rgb_goal_position": goal.tolist(), "model_zero_position": zero.tolist(),
                              "source_result_sha256": sha(source.RESULT), "source_audit_sha256": sha(audit_path),
                              "script_sha256": sha(Path(__file__)),
                              "scope": "post-hoc fixed-bank hypothesis, no planner evaluation or gate",
                              "caveat": "zero-action future may move; current RGB anchoring ignores passive drift"})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
