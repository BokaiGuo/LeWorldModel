"""Replay a full frozen shared-prefix pool and encode real endpoints."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from eval import img_transform
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint, native_world_pixels
from experiments.pc_pusht_all_shared_prefix_rescore_20261107_codex import BASE, OUT as PRIOR, SOURCE
from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments.pc_pusht_shared_prefix_three_frame_20261107_codex import OUT as THREE
from experiments.pc_scored_incumbent_real_policy_parity_20261107_codex import DATA, ROOT, records

OUT = ROOT / "results/pc_pusht_shared_pool_real_endpoint_20261107_codex"
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_SHARED_POOL_REAL_ENDPOINT_PROTOCOL_20261107_CODEX.md"


def summarize(cost, ever, final):
    order = np.argsort(cost, kind="stable")
    result = {"winner": int(order[0]), "winner_ever": bool(ever[order[0]]),
              "winner_final": bool(final[order[0]])}
    for name, labels in (("ever", ever), ("final", final)):
        good, bad = cost[labels], cost[~labels]
        comparisons = (good[:, None] < bad[None, :]).sum()
        ties = (good[:, None] == bad[None, :]).sum()
        positions = np.flatnonzero(labels[order])
        result[f"{name}_count"] = int(labels.sum())
        result[f"{name}_best_rank"] = int(positions[0] + 1) if len(positions) else None
        result[f"{name}_pair_correct"] = int(comparisons)
        result[f"{name}_pair_ties"] = int(ties)
        result[f"{name}_pair_total"] = int(len(good) * len(bad))
    return result


def main(task):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if task not in range(4):
        raise ValueError("task must be 0..3")
    target = OUT.parent / f"{OUT.name}_task{task}"
    if target.with_suffix(".json").exists() or target.with_suffix(".npz").exists():
        raise FileExistsError(target)
    prior = json.loads(PRIOR.with_suffix(".json").read_text())
    previous = json.loads(SOURCE.with_suffix(".json").read_text())
    three = json.loads(THREE.with_suffix(".json").read_text())
    evaluation = json.loads((PRIOR.parent / (PRIOR.name + ".evaluation.json")).read_text())
    for path, expected in ((PRIOR.with_suffix(".npz"), prior["arrays_sha256"]),
                           (SOURCE.with_suffix(".npz"), previous["arrays_sha256"]),
                           (THREE.with_suffix(".npz"), three["arrays_sha256"])):
        if file_hash(path) != expected:
            raise ValueError(f"input changed: {path}")
    physical_path = BASE / f"physical_{task}.json"
    if file_hash(physical_path) != evaluation["physical_sha256"].get(str(physical_path)):
        raise ValueError("physical replay changed")
    physical = json.loads(physical_path.read_text())
    lookup = {x["action_sha256"]: x for x in physical["records"]}
    plan = prior["plans"][task]
    if plan["task"] != task:
        raise ValueError("task mismatch")
    trace = sorted((x for x in records(BASE / "trace.jsonl") if x["reset_seed"] == 871001 + task),
                   key=lambda x: x["step"])
    batches = [torch.load(BASE / "capture" / f"call_{task * 5 + i:03}.pt",
                          map_location="cpu", weights_only=True) for i in range(5)]
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    fingerprint = model_fingerprint(model)
    if fingerprint != prior["model_sha256"] or fingerprint != previous["model_sha256"]:
        raise ValueError("checkpoint mismatch")
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as source:
        selected = [i for i, row in enumerate(previous["rows"]) if row["task"] == task and row["selected"]]
        if len(selected) != 1:
            raise ValueError("selected source ambiguous")
        goal = torch.as_tensor(source["goal"][selected[0]], device="cuda")
    images, truth, hashes = [], [], []
    with h5py.File(DATA, "r") as data:
        action_data = data["action"][:]
        scaler = StandardScaler().fit(action_data[~np.isnan(action_data).any(axis=1)])
        row = int(physical["data_row"])
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        try:
            for index, item in enumerate(plan["candidates"]):
                action = batches[item["call"]]["actions"][0, item["slot"]].numpy()
                digest = hashlib.sha256(action.tobytes()).hexdigest()
                if digest != item["sha256"]:
                    raise ValueError("candidate identity mismatch")
                record = lookup[digest]
                env.reset(seed=871001 + task)
                actor = env.unwrapped
                actor._set_state(np.asarray(data["state"][row], np.float64))
                actor._set_goal_state(np.asarray(data["state"][row + 25], np.float64))
                np.testing.assert_allclose(snapshot(actor), trace[0]["before"], atol=1e-4, rtol=0)
                physical_actions = scaler.inverse_transform(action.reshape(25, 2))
                if not np.isfinite(physical_actions).all() or np.abs(physical_actions).max() > 1:
                    raise ValueError("illegal action")
                first, last = None, False
                for step, physical_action in enumerate(physical_actions, 1):
                    obs, _, terminated, truncated, _ = actor.step(physical_action.copy())
                    if truncated:
                        raise ValueError("unexpected truncation")
                    if terminated and first is None:
                        first = step
                    last = bool(terminated)
                    if step % 5 == 0:
                        np.testing.assert_allclose(obs["state"], record["packet_states"][step // 5 - 1],
                                                   atol=1e-4, rtol=0)
                if first != record["first_success_step"] or last != record["final_success"]:
                    raise ValueError("physical outcome mismatch")
                images.append(transform(native_world_pixels(actor.render())))
                truth.append((first is not None, last))
                hashes.append(digest)
                if (index + 1) % 100 == 0:
                    print(f"task={task}, replayed={index + 1}/{len(plan['candidates'])}", flush=True)
        finally:
            env.close()
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(images), 32):
            encoded = model.encode({"pixels": torch.stack(images[start:start + 32]).to("cuda")[:, None]})["emb"][:, 0]
            embeddings.append(encoded.cpu().numpy())
    embeddings = np.concatenate(embeddings)
    actual_cost = ((embeddings - goal.cpu().numpy()) ** 2).sum(-1)
    with np.load(PRIOR.with_suffix(".npz"), allow_pickle=False) as one, \
         np.load(THREE.with_suffix(".npz"), allow_pickle=False) as other:
        one_cost = one[f"task_{task}"].copy()
        three_cost = other[f"task_{task}"].copy()
    labels = np.asarray(truth, dtype=bool)
    if model_fingerprint(model) != fingerprint or len(hashes) != len(plan["candidates"]):
        raise ValueError("model or candidate count changed")
    metrics = {name: summarize(cost, labels[:, 0], labels[:, 1]) for name, cost in
               (("actual_encoding_oracle", actual_cost), ("one_frame", one_cost), ("three_frame", three_cost))}
    for name, cost in (("one_frame", one_cost), ("three_frame", three_cost)):
        statistic = spearmanr(cost, actual_cost).statistic
        metrics[name]["spearman_vs_real_encoding"] = float(statistic) if np.isfinite(statistic) else None
    with target.with_suffix(".npz").open("xb") as stream:
        np.savez_compressed(stream, embeddings=embeddings, actual_cost=actual_cost, ever=labels[:, 0],
                            final=labels[:, 1], one_frame=one_cost, three_frame=three_cost)
    with target.with_suffix(".json").open("x") as stream:
        json.dump({"task": task, "hashes": hashes, "model_sha256": fingerprint,
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
                   "physical_sha256": file_hash(physical_path), "prior_sha256": file_hash(PRIOR.with_suffix(".json")),
                   "three_sha256": file_hash(THREE.with_suffix(".json")),
                   "arrays_sha256": file_hash(target.with_suffix(".npz")), "metrics": metrics},
                  stream, indent=2, allow_nan=False)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    main(parser.parse_args().task)
