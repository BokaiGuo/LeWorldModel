"""Compare dataset input images with state-conditioned TwoRoom renders."""
import hashlib
import inspect
import json
from pathlib import Path

import h5py
import numpy as np

from experiments.pc_standard_tworoom_reference_replay_20261105_codex import select_rows

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/gbk/.stable_worldmodel/datasets/tworoom.h5")
REFERENCE = ROOT / "results/pc_standard_tworoom_reference_replay_20261105_codex/report.json"
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_STANDARD_PIXEL_ALIGNMENT_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_tworoom_standard_pixel_alignment_20261107_codex.json"


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def centroid(image):
    rgb = np.asarray(image, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("expected HWC RGB")
    weight = np.maximum(rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2]), 0)
    total = weight.sum()
    if total < 1:
        return None
    y, x = np.indices(weight.shape)
    return [float((x * weight).sum() / total), float((y * weight).sum() / total)]


def compare(first, second):
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape or first.dtype != np.uint8 or second.dtype != np.uint8:
        raise ValueError("incompatible RGB frames")
    diff = np.abs(first.astype(np.int16) - second.astype(np.int16))
    return {"mae": float(diff.mean()), "max_abs": int(diff.max()),
            "changed_pixels": int(np.any(diff != 0, axis=-1).sum()),
            "dataset_red_centroid": centroid(first), "render_red_centroid": centroid(second),
            "dataset_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
            "render_sha256": hashlib.sha256(second.tobytes()).hexdigest()}


def run():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    reference = json.loads(REFERENCE.read_text())
    records = reference["records"]
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    output = []
    try:
        actor = env.unwrapped
        env_path = Path(inspect.getfile(type(actor)))
        with h5py.File(DATA, "r") as data:
            selected = select_rows(data)
            if selected.tolist() != [r["row"] for r in records]:
                raise ValueError("reference panel mismatch")
            for row, record in zip(selected, records, strict=True):
                row = int(row)
                positions = np.asarray(data["proprio"][[row, row + 25]])
                pos_agent = np.asarray(data["pos_agent"][[row, row + 25]])
                if not np.allclose(positions[0], record["initial_position"], atol=1e-5, rtol=0):
                    raise ValueError("reference state mismatch")
                if not np.allclose(positions[1], record["goal_position"], atol=1e-5, rtol=0):
                    raise ValueError("reference goal mismatch")
                env.reset(seed=42)
                actor._set_state(positions[0])
                actor._set_goal_state(positions[1])
                current = actor.render().copy()
                goal = actor._target_img.detach().cpu().numpy().transpose(1, 2, 0).copy()
                images = np.asarray(data["pixels"][[row, row + 25]])
                output.append({"row": row, "episode": record["episode"], "start_step": record["start_step"],
                               "current": compare(images[0], current), "goal": compare(images[1], goal),
                               "proprio_vs_pos_agent_max_abs": float(np.abs(positions - pos_agent).max()),
                               "radius": float(actor.variation_space["agent"]["radius"].value.item()),
                               "wall_axis": int(actor.wall_axis), "wall_thickness": int(actor.wall_thickness),
                               "reference_success": record["ever_success"]})
    finally:
        env.close()
    stat = DATA.stat()
    result = {"rows": output, "dataset_size": stat.st_size, "dataset_mtime_ns": stat.st_mtime_ns,
              "reference_sha256": file_hash(REFERENCE), "protocol_sha256": file_hash(PROTOCOL),
              "runner_sha256": file_hash(Path(__file__)), "environment_sha256": file_hash(env_path),
              "scope": "exposed seed-42 panel, default geometry; pixel comparison is not state equivalence"}
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"rows": len(output), "current_exact": sum(x["current"]["max_abs"] == 0 for x in output),
                      "goal_exact": sum(x["goal"]["max_abs"] == 0 for x in output),
                      "current_mean_mae": float(np.mean([x["current"]["mae"] for x in output])),
                      "goal_mean_mae": float(np.mean([x["goal"]["mae"] for x in output])),
                      "max_proprio_diff": max(x["proprio_vs_pos_agent_max_abs"] for x in output)},
                     indent=2))


if __name__ == "__main__":
    run()
