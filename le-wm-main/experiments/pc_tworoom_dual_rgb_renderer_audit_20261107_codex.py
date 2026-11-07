"""Evaluation-only renderer check; physical coordinates never enter image synthesis."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments import pc_tworoom_dual_rgb_goal_reconstruction_20261107_codex as source
from experiments.pc_tworoom_standard_pixel_alignment_20261107_codex import compare

PROTOCOL = source.source.ROOT.parent / "docs/PC_TWOROOM_DUAL_RGB_RENDERER_AUDIT_PROTOCOL_20261107_CODEX.md"
OUTPUT = source.source.ROOT / "results/pc_tworoom_dual_rgb_renderer_audit_20261107_codex.json"


def diff_stats(generated, rendered):
    delta = np.abs(generated.astype(np.int16) - rendered.astype(np.int16))
    mask = np.any(delta != 0, axis=-1)
    return {"max_abs": int(delta.max()), "mean_abs": float(delta.mean()),
            "changed_pixels": int(mask.sum()), "changed_mean_abs": float(delta[mask].mean()) if mask.any() else 0.}


def run():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    record = json.loads(source.OUTPUT.read_text())
    if (record["input_sha256"] != file_hash(source.source.INPUT)
            or record["images_sha256"] != file_hash(source.IMAGES)
            or record["runner_sha256"] != file_hash(source.__file__)):
        raise ValueError("sealed source changed")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with np.load(source.source.INPUT, allow_pickle=False) as input_data, \
                np.load(source.IMAGES, allow_pickle=False) as saved, \
                h5py.File(source.source.common.DATA, "r") as truth:
            for root, row in enumerate(record["rows"]):
                images, inspected = source.make_probes(input_data["current"][root], input_data["goal"][root])
                if inspected != row["inspection"] or int(input_data["rows"][root]) != row["row"]:
                    raise ValueError("reconstruction mismatch")
                if images is None:
                    continue
                np.testing.assert_array_equal(images, saved[f"root_{root}"])
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(truth["proprio"][row["row"]])
                actor._set_goal_state(truth["proprio"][row["row"] + 25])
                original = actor._target_img.detach().cpu().numpy().transpose(1, 2, 0).copy()
                original_stats = compare(input_data["goal"][root], original)
                shifts = []
                for i, (dx, dy) in enumerate(((4, 0), (-4, 0), (0, 4), (0, -4))):
                    location = actor.target_position + torch.tensor([dx, dy], dtype=torch.float32)
                    real = actor._render_frame(location).detach().cpu().numpy().transpose(1, 2, 0).copy()
                    shifts.append({"dx": dx, "dy": dy, **diff_stats(images[i], real)})
                rows.append({"row": row["row"], "original": original_stats, "shifts": shifts})
    finally:
        env.close()
    with OUTPUT.open("x") as stream:
        json.dump({"rows": rows, "reconstruction_sha256": file_hash(source.OUTPUT),
                   "images_sha256": file_hash(source.IMAGES), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(Path(__file__)),
                   "scope": "evaluation-only renderer comparison, no legal-state or control certification"},
                  stream, indent=2, allow_nan=False)
    print(f"audited {len(rows)} roots and {len(rows)*4} renderer shifts; "
          f"max mean error={max(s['mean_abs'] for r in rows for s in r['shifts']):.6f}")


if __name__ == "__main__":
    run()
