"""Post-hoc frozen layer audit on an existing physically replayed candidate pool."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from eval import img_transform
from experiments.checkpoint_compat import _resolve_local, load_compatible_pretrained
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint, native_world_pixels
from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments.pc_pusht_shared_pool_real_endpoint_20261107_codex import (
    BASE, DATA, OUT as REAL, PRIOR, ROOT, SOURCE, records, summarize,
)

PROTOCOL = ROOT.parent / "docs/PC_OSFAL_PUSHT_SUCCESS_POOL_POSTHOC_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_osfal_pusht_success_pool_20261107_codex"


def main(task):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if task not in (0, 1):
        raise ValueError("protocol includes tasks 0 and 1")
    target = OUT.parent / f"{OUT.name}_task{task}.json"
    if target.exists():
        raise FileExistsError(target)
    previous = json.loads(SOURCE.with_suffix(".json").read_text())
    prior = json.loads(PRIOR.with_suffix(".json").read_text())
    original_path = REAL.parent / f"{REAL.name}_task{task}.json"
    original = json.loads(original_path.read_text())
    original_arrays = original_path.with_suffix(".npz")
    for path, expected in ((SOURCE.with_suffix(".npz"), previous["arrays_sha256"]),
                           (PRIOR.with_suffix(".npz"), prior["arrays_sha256"]),
                           (original_arrays, original["arrays_sha256"])):
        if file_hash(path) != expected:
            raise ValueError(f"input hash mismatch: {path}")
    physical_path = BASE / f"physical_{task}.json"
    if file_hash(physical_path) != original["physical_sha256"]:
        raise ValueError("physical file changed")
    physical = json.loads(physical_path.read_text())
    lookup = {item["action_sha256"]: item for item in physical["records"]}
    plan = prior["plans"][task]
    if plan["task"] != task or len(plan["candidates"]) != 301:
        raise ValueError("unexpected candidate bank")
    trace = sorted((x for x in records(BASE / "trace.jsonl") if x["reset_seed"] == 871001 + task),
                   key=lambda x: x["step"])
    batches = [torch.load(BASE / "capture" / f"call_{task * 5 + i:03}.pt",
                          map_location="cpu", weights_only=True) for i in range(5)]
    weights, config = _resolve_local("pusht/lewm")
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    fingerprint = model_fingerprint(model)
    if fingerprint != original["model_sha256"]:
        raise ValueError("model fingerprint mismatch")
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    captured = {}

    def hook(index):
        def capture(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor) or value.ndim != 3:
                raise ValueError("unexpected layer output")
            captured[index] = value[:, 0].detach().cpu().numpy()
        return capture

    if len(model.encoder.layers) != 12:
        raise ValueError("unexpected layer count")
    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(model.encoder.layers)]
    try:
        with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as source:
            indices = [i for i, row in enumerate(previous["rows"]) if row["task"] == task and row["selected"]]
            if len(indices) != 1:
                raise ValueError("ambiguous source goal")
            saved_goal = source["goal"][indices[0]].copy()
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
                    if digest != item["sha256"] or digest != original["hashes"][index]:
                        raise ValueError("candidate identity mismatch")
                    record = lookup[digest]
                    env.reset(seed=871001 + task)
                    actor = env.unwrapped
                    actor._set_state(np.asarray(data["state"][row], np.float64))
                    actor._set_goal_state(np.asarray(data["state"][row + 25], np.float64))
                    np.testing.assert_allclose(snapshot(actor), trace[0]["before"], atol=1e-4, rtol=0)
                    actions = scaler.inverse_transform(action.reshape(25, 2))
                    if not np.isfinite(actions).all() or np.abs(actions).max() > 1:
                        raise ValueError("illegal action")
                    first, last = None, False
                    for step, physical_action in enumerate(actions, 1):
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
                        print(f"task={task} replayed={index + 1}", flush=True)
                # The source goal is a data-frame RGB, not the rendered simulator goal.
                # Read exactly the image used by the original goal encoder.
                goal_pixels = transform(data["pixels"][row + 25])
            finally:
                env.close()
        features = {i: [] for i in range(12)}
        official = []
        with torch.inference_mode():
            for start in range(0, len(images) + 1, 32):
                batch = (images + [goal_pixels])[start:start + 32]
                if not batch:
                    continue
                encoded = model.encode({"pixels": torch.stack(batch).to("cuda")[:, None]})["emb"][:, 0]
                official.append(encoded.cpu().numpy())
                for i in range(12):
                    features[i].append(captured[i])
                captured.clear()
        final_output = np.concatenate(official)
        np.testing.assert_allclose(final_output[-1], saved_goal, atol=1e-4, rtol=1e-4)
        labels = np.asarray(truth, dtype=bool)
        all_costs = {f"block_{i+1:02d}": np.square((values := np.concatenate(features[i]))[:-1] - values[-1]).sum(-1)
                     for i in range(12)}
        all_costs["official_visual"] = np.square(final_output[:-1] - final_output[-1]).sum(-1)
        with np.load(original_arrays, allow_pickle=False) as old:
            np.testing.assert_array_equal(labels[:, 0], old["ever"])
            np.testing.assert_array_equal(labels[:, 1], old["final"])
            np.testing.assert_allclose(all_costs["official_visual"], old["actual_cost"], atol=0.001, rtol=1e-4)
            max_cost_error = float(np.max(np.abs(all_costs["official_visual"] - old["actual_cost"])))
        if model_fingerprint(model) != fingerprint:
            raise ValueError("model changed")
        result = {"task": task, "scope": "post-hoc real-successor diagnostic, not deployable or independent",
                  "source_sha256": file_hash(original_path), "source_arrays_sha256": file_hash(original_arrays),
                  "physical_sha256": file_hash(physical_path), "protocol_sha256": file_hash(PROTOCOL),
                  "runner_sha256": file_hash(__file__), "checkpoint_sha256": file_hash(weights),
                  "config_sha256": file_hash(config), "model_fingerprint": fingerprint,
                  "max_official_cost_error": max_cost_error, "hashes": hashes,
                  "labels_ever": labels[:, 0].tolist(), "labels_final": labels[:, 1].tolist(),
                  "layers": {name: {"costs": costs.tolist(), "metrics": summarize(costs, labels[:, 0], labels[:, 1])}
                             for name, costs in all_costs.items()}}
        with target.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(json.dumps({"output": str(target), "metrics": {n: v["metrics"] for n, v in result["layers"].items()}},
                         indent=2), flush=True)
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    main(parser.parse_args().task)
