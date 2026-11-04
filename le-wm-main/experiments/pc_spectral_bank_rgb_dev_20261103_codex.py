"""Frozen-history RGB spectral bank and open-loop execution, development only."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_motion_unprivileged_dev_20261103_codex as old
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history
from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import ShadowRGBBridge
from experiments.spectral_reachability import pc_spectral_bank_planner_20261103_codex as core

h, ROOT = old.h, old.ROOT
PROTOCOL = ROOT.parent / "docs/PC_SPECTRAL_BANK_RGB_DEV_PROTOCOL_20261103_CODEX.md"
OUTPUT = ROOT / "results/pc_spectral_bank_rgb_dev_20261103_codex_decisions.json"
EXECUTION = ROOT / "results/pc_spectral_bank_rgb_dev_20261103_codex_execution.json"
IMAGES = ROOT / "results/pc_spectral_bank_rgb_dev_20261103_codex_execution.npz"


def main():
    if any(path.exists() for path in (OUTPUT, EXECUTION, IMAGES)):
        raise FileExistsError("refusing to overwrite prior run")
    source = json.loads(old.DECISIONS.read_text())
    sha = old.prior.sha
    if source["history_sha256"] != sha(old.HISTORY) or source["runner_sha256"] != sha(Path(old.__file__)):
        raise ValueError("source history or runner mismatch")
    camera, calibration = (json.loads(path.read_text()) for path in (h.CAMERA, h.OLD))
    metadata, offset = camera["metadata"], calibration["body_offset_from_static_scene"]
    rows = []
    with np.load(old.HISTORY, allow_pickle=False) as histories, \
            np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
            h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                  config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
        model, dataset = h.load_model_and_dataset(args, device)
        model.eval()
        model_hash = h.parameter_hash(model)
        if model_hash != source["model_hash"] or sha(h.DEFAULT_CHECKPOINT) != source["checkpoint_sha256"]:
            raise ValueError("checkpoint mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not np.array_equal(mean.cpu().numpy(), source["action_mean"]) or not np.array_equal(std.cpu().numpy(), source["action_std"]):
            raise ValueError("normalization mismatch")
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        goal_rgb = obstacle["goal_rgb"]
        if old.digest_image(goal_rgb) != source["goal_sha256"]:
            raise ValueError("goal mismatch")
        goal_state, _ = h.estimate_state(goal_rgb, metadata, offset)
        with torch.no_grad():
            target = h.encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
        for root in range(4):
            start = time.monotonic()
            frames = histories[f"root{root}_history_rgb"]
            if [old.digest_image(x) for x in frames] != source["rows"][root]["rgb_sha256"]:
                raise ValueError("history RGB mismatch")
            meter = core.Meter({"prediction_samples": 3100, "svd": 1, "suffix_optimizations": 15})
            encode_start = time.monotonic()
            contexts = []
            with torch.no_grad():
                for t in (5, 10, 15):
                    state = h.causal_state(frames[t-1], frames[t], metadata, offset, .05)
                    contexts.append(h.encode_with_state(model, frames[t], state, device))
                torch.cuda.synchronize()
            encode_seconds = time.monotonic() - encode_start
            ctx = {k: torch.cat([c[k] for c in contexts], dim=1) for k in ("visual", "proprio")}
            history = ((torch.as_tensor(old.WARMUP[5:], dtype=torch.float32, device=device)-mean)/std).reshape(2, 1, 10)
            initial = float(h.projected_cost(ctx["proprio"][:, -1:].reshape(1, 256, 16), target, pinv, pos_std)[0])

            def raw_predict(actions):
                actions = np.asarray(actions, dtype=np.float32)
                batch, horizon, _ = actions.shape
                normal = (torch.as_tensor(actions, device=device)-mean)/std
                packets = normal.reshape(batch, horizon//5, 10).transpose(0, 1)
                with torch.no_grad():
                    pred = unroll_with_action_history(model, ctx, history.expand(2, batch, 10), packets, 3)["proprio"][-1]
                    projected = ((pred.reshape(batch, 256, 16).mean(1) @ pinv.T)[:, :2] * pos_std)
                    score = h.projected_cost(pred.reshape(batch, 256, 16), target, pinv, pos_std)
                    torch.cuda.synchronize()
                return projected.cpu().numpy(), score.cpu().numpy()

            def response(prefix):
                return raw_predict(np.asarray(prefix)[None])[0][0]

            bank, spectrum = core.spectral_bank(response, meter, 2026110321+root)
            def prefix_score(prefix):
                return raw_predict(np.asarray(prefix)[None])[1][0]

            def cem(prefix, samples, ledger):
                if samples != 100:
                    raise ValueError("unexpected sample budget")
                rng = np.random.default_rng(2026110340+root)
                records = []
                for restart in range(2):
                    loc, scale = np.zeros((10, 2)), np.full((10, 2), 1/3)
                    for _ in range(3):
                        candidates = np.clip(loc+rng.normal(size=(16, 10, 2))*scale, -1, 1)
                        candidates[0] = loc
                        chains = np.concatenate((np.broadcast_to(prefix, (16, 5, 2)), candidates), axis=1)
                        _, values = ledger.call("prediction_samples", 16, raw_predict, chains)
                        records.extend((x.copy(), float(v)) for x, v in zip(candidates, values))
                        elites = candidates[np.argsort(values, kind="stable")[:4]]
                        loc, scale = elites.mean(0), np.maximum(elites.std(0, ddof=1), 1e-5)
                    candidates = np.stack((np.zeros((10, 2)), loc))
                    chains = np.concatenate((np.broadcast_to(prefix, (2, 5, 2)), candidates), axis=1)
                    _, values = ledger.call("prediction_samples", 2, raw_predict, chains)
                    records.extend((x.copy(), float(v)) for x, v in zip(candidates, values))
                return min(records, key=lambda x: x[1])

            result = core.plan_bank(bank, initial, prefix_score, cem, meter, 100)
            programs = result["programs"]
            lookup = {x["name"]: x for x in programs}
            nonspectral = [x for x in programs if x["source"] != "rgb_model"]
            arms = {"selected": result["selected"], "greedy": lookup[result["same_bank_greedy_name"]],
                    "zero": lookup["zero"], "random": lookup["random_0"],
                    "no_spectral": min(nonspectral, key=lambda x: x["terminal_score"])}
            rng = np.random.default_rng(2026110380+root)
            direct_records = []
            for generations in (47, 48):
                loc, scale = np.zeros((15, 2)), np.full((15, 2), 1/3)
                for _ in range(generations):
                    candidates = np.clip(loc+rng.normal(size=(16, 15, 2))*scale, -1, 1)
                    candidates[0] = loc
                    _, values = meter.call("prediction_samples", 16, raw_predict, candidates)
                    direct_records.extend((x.copy(), float(v)) for x, v in zip(candidates, values))
                    elites = candidates[np.argsort(values, kind="stable")[:4]]
                    loc, scale = elites.mean(0), np.maximum(elites.std(0, ddof=1), 1e-5)
            direct, direct_score = min(direct_records, key=lambda x: x[1])
            arms["full_horizon_cem"] = {"name": "full_horizon_cem", "prefix": direct[:5].tolist(),
                                        "suffix": direct[5:].tolist(), "terminal_score": direct_score}
            rows.append({"root": root, "initial_score": initial, "spectrum": spectrum, "plan": result,
                         "arms": arms, "counts": meter.counts, "seconds": meter.seconds,
                         "encoding_calls": 3, "encoding_seconds": encode_seconds,
                         "root_wall_seconds": time.monotonic()-start})
            print(f"root {root}: selected {arms['selected']['name']}, greedy {arms['greedy']['name']}", flush=True)
        if h.parameter_hash(model) != model_hash:
            raise ValueError("model changed")
    old.write_json(OUTPUT, {"schema": "pc_spectral_bank_rgb_dev.v1", "rows": rows,
                            "source_sha256": sha(old.DECISIONS), "history_sha256": sha(old.HISTORY),
                            "protocol_sha256": sha(PROTOCOL), "runner_sha256": sha(Path(__file__)),
                            "core_sha256": sha(Path(core.__file__)), "model_hash": model_hash,
                            "checkpoint_sha256": sha(h.DEFAULT_CHECKPOINT), "config_sha256": sha(h.DEFAULT_CONFIG),
                            "full_cost_certified": False, "full_horizon_cem_samples": 1520})
    print("all plans sealed", flush=True)
    results, arrays = [], {}
    for row in rows:
        root = row["root"]
        for role, program in row["arms"].items():
            with ShadowRGBBridge(np.asarray([*old.STARTS[root], 0., 0.])) as env:
                frames = [env.initial_rgb.copy()]
                warm, _ = old.do_actions(env, old.WARMUP)
                frames += warm
                if [old.digest_image(x) for x in frames] != source["rows"][root]["rgb_sha256"]:
                    raise RuntimeError("warmup replay mismatch")
                actions = program["prefix"] + program["suffix"]
                out, timing = old.do_actions(env, actions)
            arrays[f"root{root}_{role}"] = np.stack(out)
            results.append({"root": root, "role": role, "name": program["name"], "actions": actions,
                            "rgb_sha256": [old.digest_image(x) for x in out], "times_ns": timing})
            print(f"root {root} {role}: executed", flush=True)
    with IMAGES.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    old.write_json(EXECUTION, {"decision_sha256": sha(OUTPUT), "arrays_sha256": sha(IMAGES), "rows": results})


if __name__ == "__main__":
    main()
