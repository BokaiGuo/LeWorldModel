"""Replay sealed PointMaze actions through the RGB bridge without planner access."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v22_pointmaze_live_feedback_v2_20261104_codex as run

OUTPUT = run.PREFIX.with_name(run.PREFIX.name + "_rgb_replay.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = run.old.prior.sha
    manifest = json.loads(run.MANIFEST.read_text())
    if manifest["dependencies"] != run.dependencies():
        raise ValueError("frozen dependency mismatch")
    rows = []
    for arm in run.ARMS:
        decisions, image_path, receipt_path = run.paths(arm)
        receipt = json.loads(receipt_path.read_text())
        if receipt["images_sha256"] != sha(image_path):
            raise ValueError("image archive mismatch")
        actions = np.asarray(run.old.WARMUP.tolist() + receipt["executed_actions"], dtype=np.float32)
        if actions.shape != (30, 2) or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
            raise ValueError("invalid sealed actions")
        with np.load(image_path, allow_pickle=False) as archive:
            expected = archive["rgb"].copy()
        with run.bridge.runner.ShadowRGBBridge(np.asarray([*run.START, 0., 0.])) as env:
            replay, _ = run.old.do_actions(env, actions)
            images = [env.initial_rgb, *replay]
        if len(images) != len(expected):
            raise ValueError("replay image clock mismatch")
        matches = [run.old.digest_image(a) == run.old.digest_image(b) for a, b in zip(images, expected, strict=True)]
        rows.append({"arm": arm, "matching_frames": sum(matches), "frames": len(matches),
                     "frame_matches": matches, "receipt_sha256": sha(receipt_path),
                     "images_sha256": sha(image_path), "decision_sha256": [sha(p) for p in decisions]})
    run.old.write_json(OUTPUT, {"manifest_sha256": sha(run.MANIFEST), "rows": rows,
                                "script_sha256": sha(Path(__file__)),
                                "limitations": ["same bridge and backend; not independent replication",
                                                "frame agreement is not physical success certification"]})
    print("RGB replay:", [(r["arm"], r["matching_frames"], r["frames"]) for r in rows], flush=True)


if __name__ == "__main__":
    main()
