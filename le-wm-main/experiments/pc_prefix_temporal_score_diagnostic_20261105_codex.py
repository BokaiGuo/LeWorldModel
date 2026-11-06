"""Physical and encoded temporal diagnostics for existing paired refined chains."""
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from eval import img_transform
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint, native_world_pixels
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    root = Path(__file__).resolve().parents[1] / "results"
    output = root / "pc_prefix_temporal_score_diagnostic_dev_20261105_codex.json"
    if output.exists():
        raise FileExistsError(output)
    replay = json.loads((root / "pc_prefix_novel_same_refinement_replay_seed42_20261105_codex.json").read_text())
    baseline = json.loads((root / "pc_prefix_novel_actual_encoding_native_pixels_v2_seed42_20261105_codex.json").read_text())
    audit = json.loads((root / "pc_prefix_novel_same_refinement_seed42_20261105_codex.audit.jsonl").read_text().splitlines()[0])
    if replay["errors"] or len(replay["parity"]) != 50 or not all(x["pass"] for x in replay["parity"]):
        raise ValueError("physical parity prerequisite missing")
    records, images = [], []
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        rows = panel_rows(data)
        actions = np.asarray(data["action"][:])
        scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
        for index, item in enumerate(audit["refinement"]):
            prior = replay["counterfactual"][index]
            original = baseline["records"][index]
            if item["index"] != index or prior["index"] != index or original["index"] != index:
                raise ValueError("task order mismatch")
            if item["no_novel_candidate"]:
                continue
            env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
            try:
                for arm, kind in enumerate(("novel", "common")):
                    physical = scaler.inverse_transform(np.asarray(item["refined_actions"][arm], dtype=np.float32).reshape(25, 2))
                    if not np.isfinite(physical).all() or np.any(np.abs(physical) > 1):
                        raise ValueError("invalid physical action")
                    env.reset(seed=771001 + index)
                    actor = env.unwrapped
                    actor._set_state(np.asarray(data["state"][rows[index]], dtype=np.float64))
                    actor._set_goal_state(np.asarray(data["state"][rows[index] + 25], dtype=np.float64))
                    goal = np.asarray(data["pixels"][rows[index] + 25])
                    images.append(goal)
                    steps, hit, last_terminal = [], None, False
                    for step in range(26):
                        if step:
                            obs, _, terminated, truncated, _ = actor.step(physical[step - 1].copy())
                            last_terminal = bool(terminated)
                            if truncated:
                                raise ValueError("unexpected truncation")
                            if terminated and hit is None:
                                hit = step
                        else:
                            obs = actor._get_obs()
                        if step % 5 == 0:
                            images.append(native_world_pixels(actor.render()))
                            state = obs["state"] if isinstance(obs, dict) else obs
                            steps.append({"step": step, "state": np.asarray(state).tolist(),
                                          "success_this_step": last_terminal if step else None})
                    expected = next(x for x in prior["chains"] if x["chain"] == kind)
                    if hit != expected["first_success_step"]:
                        raise ValueError("first-success replay mismatch")
                    prev = next(x for x in original["chains"] if x["chain"] == kind)
                    records.append({"index": index, "chain": kind, "predicted_final_cost": item["refined_costs"][arm],
                                    "first_success_step": hit, "prior_actual_final_cost": prev["actual_encoded_cost"],
                                    "steps": steps})
            finally:
                env.close()
            print(f"temporal replay task {index + 1}/50", flush=True)
    if len(records) != 94 or not torch.cuda.is_available():
        raise ValueError("expected 47 pairs and CUDA")
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    fingerprint = model_fingerprint(model)
    if fingerprint != baseline["model_sha256"]:
        raise ValueError("checkpoint fingerprint mismatch")
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    embeds = []
    with torch.inference_mode():
        for start in range(0, len(images), 16):
            inp = torch.stack([transform(img) for img in images[start:start + 16]]).to("cuda")
            embeds.append(model.encode({"pixels": inp[:, None]})["emb"][:, 0].cpu())
    embeds = torch.cat(embeds)
    for i, record in enumerate(records):
        group = embeds[i * 7:(i + 1) * 7]
        costs = (group[1:] - group[0]).square().flatten(1).sum(1)
        for step, cost in zip(record["steps"], costs):
            step["actual_encoded_cost"] = float(cost)
        if not np.isclose(record["steps"][-1]["actual_encoded_cost"], record["prior_actual_final_cost"], rtol=1e-4, atol=1e-5):
            raise ValueError("final encoding mismatch with previous run")
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model fingerprint changed")
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"records": records, "model_sha256": fingerprint, "images": len(images)}, stream, allow_nan=False)
    print(f"wrote {len(records)} physical chain diagnostics to {output}", flush=True)


if __name__ == "__main__":
    main()
