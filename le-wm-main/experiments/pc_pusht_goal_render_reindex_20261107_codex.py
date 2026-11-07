"""Evaluator-only goal-image audit with Pymunk geometry cache refreshed."""
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import h5py
import numpy as np
import stable_worldmodel  # noqa: F401
import stable_worldmodel.envs.pusht.env as pusht
import torch

from eval import img_transform
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint, native_world_pixels
from experiments.pc_pusht_goal_render_alignment_20261107_codex import compare, set_render_pose
from experiments.pc_pusht_goal_render_offset_probe_20261107_codex import image_error
from experiments import pc_pusht_first_plan_alignment_20261107_codex as source

PREFIX = source.RESULTS / "pc_pusht_goal_render_reindex_20261107_codex"
PROTOCOL = source.ROOT.parent / "docs/PC_PUSHT_GOAL_RENDER_REINDEX_PROTOCOL_20261107_CODEX.md"


def set_render_pose_reindexed(actor, state):
    set_render_pose(actor, state)
    actor.space.reindex_shapes_for_body(actor.agent)
    actor.space.reindex_shapes_for_body(actor.block)


def main():
    output, arrays = PREFIX.with_suffix(".json"), PREFIX.with_suffix(".npz")
    if output.exists() or arrays.exists():
        raise FileExistsError(PREFIX)
    record = json.loads(source.OUT.with_suffix(".json").read_text())
    if record["arrays_sha256"] != file_hash(source.OUT.with_suffix(".npz")):
        raise ValueError("source array drift")
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    fingerprint = model_fingerprint(model)
    if fingerprint != record["model_sha256"]:
        raise ValueError("model fingerprint mismatch")
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    images = {"original": [], "overlay_on": [], "overlay_off": []}
    stats = []
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
    try:
        with h5py.File(source.DATA, "r") as data:
            for row in record["rows"]:
                index = row["index"]
                target_row = row["data_row"] + 25
                target = np.asarray(data["state"][target_row], dtype=np.float64)
                original = np.asarray(data["pixels"][target_row])
                env.reset(seed=871001 + index)
                actor = env.unwrapped
                set_render_pose_reindexed(actor, target)
                error = float(np.max(np.abs(target[:5] - actor._get_obs()[:5])))
                if error > 1e-4:
                    raise ValueError(f"pose mismatch {index}: {error}")
                images["original"].append(transform(original))
                item = {"index": index, "pose_error": error}
                for key, enabled in (("overlay_on", True), ("overlay_off", False)):
                    actor.with_target = enabled
                    frame = native_world_pixels(actor.render())
                    item[key] = image_error(original, frame)
                    images[key].append(transform(frame))
                stats.append(item)
    finally:
        env.close()
    vectors = {}
    with torch.inference_mode():
        for key, stack in images.items():
            batches = []
            for offset in range(0, len(stack), 16):
                tensor = torch.stack(stack[offset:offset + 16]).to("cuda")[:, None]
                batches.append(model.encode({"pixels": tensor})["emb"][:, 0].cpu().numpy())
            vectors[key] = np.concatenate(batches)
    with np.load(source.OUT.with_suffix(".npz"), allow_pickle=False) as data:
        predicted, actual, sealed = data["predicted"], data["actual"], data["goal"]
    sealed_error = float(np.max(np.abs(vectors["original"] - sealed)))
    if sealed_error > .001 or model_fingerprint(model) != fingerprint:
        raise ValueError("goal encoding or model fingerprint mismatch")
    rows = []
    for i, old in enumerate(record["rows"]):
        labels = {"ever": [x["first_success"] is not None for x in old["arms"]],
                  "final": [x["final_success"] for x in old["arms"]]}
        scores = {}
        for kind, rep in (("predicted", predicted[i, :, -1]), ("actual_encoded", actual[i, :, -1])):
            for key, goal in vectors.items():
                cost = np.square(rep - goal[i]).sum(-1)
                scores[f"{kind}_{key}"] = {"scores": cost.tolist(),
                    **{name: compare(cost, values) for name, values in labels.items()}}
        rows.append({**stats[i], "same_chain": old["same_chain"], "scores": scores,
                     "embedding_max_abs": {key: float(np.max(np.abs(vectors[key][i] - vectors["original"][i])))
                                           for key in ("overlay_on", "overlay_off")}})
    with arrays.open("xb") as stream:
        np.savez_compressed(stream, **vectors)
    paths = (PROTOCOL, Path(__file__), source.OUT.with_suffix(".json"), source.OUT.with_suffix(".npz"),
             Path(pusht.__file__), source.ROOT / "experiments/pc_pusht_goal_render_alignment_20261107_codex.py")
    with output.open("x") as stream:
        json.dump({"rows": rows, "arrays_sha256": file_hash(arrays), "model_sha256": fingerprint,
                   "source_sha256": {str(path): file_hash(path) for path in paths},
                   "data_path": str(source.DATA), "data_size": source.DATA.stat().st_size,
                   "data_mtime_ns": source.DATA.stat().st_mtime_ns,
                   "original_goal_max_sealed_error": sealed_error,
                   "scope": "exposed tasks, evaluator-only; no gate promotion"}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"completed {len(rows)} corrected goal comparisons; sealed error {sealed_error:g}")


if __name__ == "__main__":
    main()
