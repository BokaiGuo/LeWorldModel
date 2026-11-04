"""Fixed exposed-action host smoke for the RGB-only bridge."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments import pc_pointmaze_cp12_native_position_stepwise_20261103_codex as panel
from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import ShadowRGBBridge

ROOT = panel.ROOT
OUT = ROOT / "results/pc_pointmaze_shadow_rgb_bridge_smoke_20261103_CODEX.json"
JOURNAL = OUT.with_name(OUT.stem + "_events.jsonl")
EXPECTED = {
    ROOT.parent / "docs/PC_POINTMAZE_SHADOW_RGB_BRIDGE_PROTOCOL_20261103_CODEX.md":
        "ac564066c4f1a20bf7d778e2984c289b0493e96e6760802916cf4004a18a0e9c",
    ROOT.parent / "docs/PC_POINTMAZE_SHADOW_RGB_BRIDGE_SMOKE_PROTOCOL_20261103_CODEX.md":
        "c3d0dde5c242e5a92a2be3eebc0506227dd599e33876bb1a8ceb5637d5cbdd01",
    ROOT / "experiments/pc_pointmaze_shadow_rgb_bridge_20261103_codex.py":
        "97bc794e5cbb73a78e64138c0843caa4505deb983d7f802fe2a1fb717dbae33f",
    ROOT / "experiments/pc_pointmaze_shadow_rgb_bridge_worker_20261103_codex.py":
        "478d322dac6614033f30f145308c9935501ade7168e427175ccd2995bd679e9c",
    ROOT / "experiments/pc_pointmaze_shadow_renderer_worker_v4_20261103_codex.py":
        "71e43269d5577cd1dd7439763a8c660793b5ff3f2d0d147e24ab5badd679d34c",
    ROOT / "results/pc_pointmaze_shadow_renderer_audit_v4_20261103_CODEX.json":
        "7b111ab6487be59696718c59f3e4e2ebb30c36a1fcfa433e7ba690d067f9e0a0",
    ROOT / "results/pc_pointmaze_shadow_renderer_audit_v4_20261103_CODEX.npz":
        "0d254110f530af01acdb357a56f12cb41682476f00071d218c20a146649d12e7",
}


def main():
    if OUT.exists() or JOURNAL.exists():
        raise FileExistsError("refuse overwrite")
    for path, digest in {**panel.EXPECTED, **EXPECTED}.items():
        if panel.prior.sha(path) != digest:
            raise ValueError(f"source changed: {path}")
    _, _, names = panel.prior.files()
    starts, actions, _ = panel.build_inputs(names)
    with np.load(ROOT/"results/pc_pointmaze_shadow_renderer_audit_v4_20261103_CODEX.npz", allow_pickle=False) as src:
        expected = src["shadow_hashes"].copy()
    rows = []
    with JOURNAL.open("x") as journal:
        def check(index, arm, boundary, image, clock):
            if image.shape != (224, 224, 3) or image.dtype != np.uint8 or clock != boundary:
                raise ValueError("RGB shape or action clock mismatch")
            digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
            row = {"trajectory": index, "pass": arm, "boundary": boundary, "clock": clock,
                   "rgb_sha256": digest, "matches_v4": digest == expected[index, boundary]}
            rows.append(row)
            journal.write(json.dumps({"event": "frame", **row}) + "\n")
            journal.flush()
            if not row["matches_v4"]:
                raise ValueError("shadow RGB mismatch")

        try:
            for index in range(4):
                with ShadowRGBBridge(starts[index], timeout=30) as env:
                    check(index, 1, 0, env.initial_rgb, env.step_count)
                    offset = 0
                    for length in (1, 5, 5, 5, 1):
                        image = env.step(actions[index, offset:offset+length])
                        offset += length
                        check(index, 1, offset, image, env.step_count)
                    check(index, 2, 0, env.reset(), env.step_count)
                    offset = 0
                    for length in (5, 5, 5, 2):
                        image = env.step(actions[index, offset:offset+length])
                        offset += length
                        check(index, 2, offset, image, env.step_count)
        except BaseException as exc:
            journal.write(json.dumps({"event": "failed", "type": type(exc).__name__,
                                      "message": str(exc), "time_ns": time.time_ns()}) + "\n")
            journal.flush()
            raise
        journal.write(json.dumps({"event": "completed", "time_ns": time.time_ns()})+"\n")
    report = {"scope": "development smoke on four exposed trajectories, not performance evaluation",
              "passed": True, "trajectories": 4, "passes": 8, "high_level_actions": 136,
              "checked_frames": len(rows), "rows": rows, "model_calls": 0, "planner_calls": 0,
              "dependencies": {str(p): panel.prior.sha(p) for p in [*EXPECTED, Path(__file__)]},
              "journal_sha256": panel.prior.sha(JOURNAL)}
    with OUT.open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Saved {OUT}; checked {len(rows)} frames")


if __name__ == "__main__":
    main()
