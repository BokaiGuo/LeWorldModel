"""Compare sealed PushT goals with evaluator-only rerendering of goal states."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from eval import img_transform
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint, native_world_pixels
from experiments import pc_pusht_first_plan_alignment_20261107_codex as source

PREFIX = source.RESULTS / "pc_pusht_goal_render_alignment_20261107_codex"
OUTPUT = PREFIX.with_suffix(".json")
ARRAYS = PREFIX.with_suffix(".npz")
PROTOCOL = source.ROOT.parent / "docs/PC_PUSHT_GOAL_RENDER_ALIGNMENT_PROTOCOL_20261107_CODEX.md"


def compare(costs, labels, tie=0.001):
    costs, labels = np.asarray(costs), np.asarray(labels)
    if costs.shape != (2,) or labels.shape != (2,) or not np.isfinite(costs).all():
        raise ValueError("invalid candidate pair")
    delta = float(costs[0] - costs[1])
    choice = None if abs(delta) <= tie else int(delta > 0)
    defined = bool(labels[0] != labels[1])
    return {"delta": delta, "choice": choice, "physical_discordant": defined,
            "correct": (bool(labels[choice]) if defined and choice is not None else None)}


def set_render_pose(actor, state):
    """Set evaluator-only rendering pose without advancing the simulator clock."""
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("expected a finite seven-element PushT observation")
    actor.agent.position = state[:2].tolist()
    actor.block.angle = float(state[4])
    actor.block.position = state[2:4].tolist()
    actor.agent.velocity = state[5:7].tolist()


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError(PREFIX)
    record = json.loads(source.OUT.with_suffix(".json").read_text())
    if record["arrays_sha256"] != file_hash(source.OUT.with_suffix(".npz")):
        raise ValueError("source array drift")
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    fingerprint = model_fingerprint(model)
    if fingerprint != record["model_sha256"]:
        raise ValueError("model fingerprint mismatch")
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    originals, rerenders, state_hashes, image_stats = [], [], [], []
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
    try:
        with h5py.File(source.DATA, "r") as data:
            for index, row in enumerate(record["rows"]):
                if row["index"] != index:
                    raise ValueError("row mismatch")
                target = np.asarray(data["state"][row["data_row"] + 25], dtype=np.float64)
                image = np.asarray(data["pixels"][row["data_row"] + 25])
                env.reset(seed=871001 + index)
                actor = env.unwrapped
                set_render_pose(actor, target)
                actor._set_goal_state(target)
                actual = np.asarray(actor._get_obs(), dtype=np.float64)
                error = float(np.max(np.abs(target[:5] - actual[:5])))
                if error > 0.0001:
                    raise ValueError(f"state reset mismatch task={index} error={error}")
                render = native_world_pixels(actor.render())
                if image.shape != render.shape or image.dtype != np.uint8 or render.dtype != np.uint8:
                    raise ValueError("incompatible RGB observations")
                diff = np.abs(image.astype(np.int16) - render.astype(np.int16))
                image_stats.append({"state_error": error, "pixel_equal": bool(np.array_equal(image, render)),
                                    "pixel_mean_abs": float(diff.mean()), "pixel_max_abs": int(diff.max())})
                state_hashes.append(hashlib.sha256(np.ascontiguousarray(target).tobytes()).hexdigest())
                originals.append(transform(image))
                rerenders.append(transform(render))
    finally:
        env.close()
    with torch.inference_mode():
        vectors = []
        for stack in (originals, rerenders):
            batches = []
            for offset in range(0, len(stack), 16):
                t = torch.stack(stack[offset:offset + 16]).to("cuda")[:, None]
                batches.append(model.encode({"pixels": t})["emb"][:, 0].cpu().numpy())
            vectors.append(np.concatenate(batches))
    goals, rendered_goals = vectors
    with np.load(source.OUT.with_suffix(".npz"), allow_pickle=False) as data:
        predicted, actual, sealed = data["predicted"], data["actual"], data["goal"]
    max_sealed_diff = float(np.max(np.abs(goals - sealed)))
    if max_sealed_diff > 0.001:
        raise ValueError(f"original goal embedding mismatch {max_sealed_diff}")
    rows = []
    for index, item in enumerate(record["rows"]):
        measures = {}
        labels = {"ever": [x["first_success"] is not None for x in item["arms"]],
                  "final": [x["final_success"] for x in item["arms"]]}
        for kind, representation in (("predicted", predicted[index, :, -1]),
                                     ("actual_encoded", actual[index, :, -1])):
            for goal_name, goal in (("original", goals[index]), ("rerender", rendered_goals[index])):
                scores = np.square(representation - goal).sum(-1)
                measures[f"{kind}_{goal_name}"] = {
                    "scores": scores.tolist(),
                    **{label: compare(scores, truth) for label, truth in labels.items()}}
        rows.append({"index": index, "data_row": item["data_row"], "same_chain": item["same_chain"],
                     "goal_state_sha256": state_hashes[index], "image": image_stats[index],
                     "goal_embedding_max_abs": float(np.max(np.abs(goals[index] - rendered_goals[index]))),
                     "goal_embedding_mse": float(np.square(goals[index] - rendered_goals[index]).mean()),
                     "measures": measures})
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with ARRAYS.open("xb") as f:
        np.savez_compressed(f, original_goal=goals, rerender_goal=rendered_goals)
    paths = (source.OUT.with_suffix(".json"), source.OUT.with_suffix(".npz"), PROTOCOL, Path(__file__))
    with OUTPUT.open("x") as f:
        json.dump({"rows": rows, "arrays_sha256": file_hash(ARRAYS),
                   "source_sha256": {str(p): file_hash(p) for p in paths},
                   "model_sha256": fingerprint, "data_path": str(source.DATA),
                   "data_size": source.DATA.stat().st_size, "data_mtime_ns": source.DATA.stat().st_mtime_ns,
                   "original_goal_max_sealed_error": max_sealed_diff,
                   "scope": "exposed task evaluator-only diagnostic; no gate promotion"},
                  f, indent=2, allow_nan=False)
    print(f"completed {len(rows)} goal pairs; max original sealed error={max_sealed_diff:.8g}")


if __name__ == "__main__":
    main()
