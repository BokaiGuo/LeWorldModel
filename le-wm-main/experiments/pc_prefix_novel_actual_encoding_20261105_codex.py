"""Compare predicted cost, true terminal RGB encoding and physical outcomes."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler

from eval import img_transform
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def native_world_pixels(frame):
    """Match AddPixelsWrapper's PIL bilinear resize before model preprocessing."""
    image = Image.fromarray(np.asarray(frame))
    return np.array(image.resize((224, 224), Image.Resampling.BILINEAR))


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    root = Path(__file__).resolve().parents[1] / "results"
    replay = json.loads((root / "pc_prefix_novel_same_refinement_replay_seed42_20261105_codex.json").read_text())
    audit_path = root / "pc_prefix_novel_same_refinement_seed42_20261105_codex.audit.jsonl"
    audit = json.loads(audit_path.read_text().splitlines()[0])
    output = root / "pc_prefix_novel_actual_encoding_native_pixels_v2_seed42_20261105_codex.json"
    if output.exists():
        raise FileExistsError(output)
    if (replay["errors"] or len(replay["parity"]) != 50 or
            not all(row["pass"] for row in replay["parity"]) or
            len(replay["counterfactual"]) != 50 or len(audit["refinement"]) != 50):
        raise ValueError("missing prerequisite parity and complete task records")
    records, images = [], []
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        rows = panel_rows(data)
        actions = np.asarray(data["action"][:])
        scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
        for index in range(50):
            prior = replay["counterfactual"][index]
            pair = audit["refinement"][index]
            if prior["index"] != index or pair["index"] != index:
                raise ValueError("task order changed")
            if prior["no_novel_candidate"] != pair["no_novel_candidate"]:
                raise ValueError("candidate existence mismatch")
            if pair["no_novel_candidate"]:
                records.append({"index": index, "no_novel_candidate": True})
                continue
            goal = np.asarray(data["pixels"][rows[index] + 25])
            row = {"index": index, "no_novel_candidate": False, "chains": []}
            env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
            try:
                for arm, kind in enumerate(("novel", "common")):
                    physical = scaler.inverse_transform(
                        np.asarray(pair["refined_actions"][arm], dtype=np.float32).reshape(25, 2))
                    if np.any(np.abs(physical) > 1.0):
                        raise ValueError(f"illegal physical action at task {index}, arm {kind}")
                    env.reset(seed=771001 + index)
                    actor = env.unwrapped
                    actor._set_state(np.asarray(data["state"][rows[index]], dtype=np.float64))
                    actor._set_goal_state(np.asarray(data["state"][rows[index] + 25], dtype=np.float64))
                    first_success = None
                    for step, action in enumerate(physical, 1):
                        _, _, terminated, truncated, _ = actor.step(action.copy())
                        if terminated and first_success is None:
                            first_success = step
                        if truncated:
                            raise ValueError("unexpected truncation before endpoint")
                    expected = next(x for x in prior["chains"] if x["chain"] == kind)
                    if first_success != expected["first_success_step"]:
                        raise ValueError(f"physical replay success mismatch at task {index}, arm {kind}")
                    frame = actor.render()
                    if frame is None or np.asarray(frame).ndim != 3:
                        raise ValueError("missing RGB frame")
                    images.extend([goal, native_world_pixels(frame)])
                    row["chains"].append({"chain": kind, "predicted_cost": pair["refined_costs"][arm],
                                          "first_success_step": first_success})
            finally:
                env.close()
            records.append(row)
            print(f"rendered paired full horizon {index + 1}/50", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required by this declared diagnostic")
    device = torch.device("cuda")
    model = load_compatible_pretrained("pusht/lewm", device=device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    fingerprint = model_fingerprint(model)
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(images), 16):
            inputs = torch.stack([transform(frame) for frame in images[start:start + 16]]).to(device)
            embeddings.append(model.encode({"pixels": inputs[:, None]})["emb"][:, 0].cpu())
    embeddings = torch.cat(embeddings)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model parameters changed")
    cursor = 0
    for row in records:
        if row["no_novel_candidate"]:
            continue
        for chain in row["chains"]:
            chain["actual_encoded_cost"] = float((embeddings[cursor] - embeddings[cursor + 1]).square().sum())
            cursor += 2
    if cursor != len(images):
        raise ValueError("image/chain accounting mismatch")
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"records": records, "model_sha256": fingerprint,
                   "source": str(audit_path), "encoding_images": len(images)},
                  stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
