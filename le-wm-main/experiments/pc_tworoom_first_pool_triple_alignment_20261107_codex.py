"""Same-chain RGB model, actual frozen encoding and physical endpoint audit."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_hold_first_pool_physical_20261107_codex as physical
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = physical.run.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_FIRST_POOL_PREDICTION_ENCODING_PHYSICAL_PROTOCOL_20261107_CODEX.md"
FRAMES = ROOT / "results/pc_tworoom_first_pool_triple_alignment_frames_20261107_codex.npz"
FRAME_META = ROOT / "results/pc_tworoom_first_pool_triple_alignment_frames_20261107_codex.json"
FEATURES = ROOT / "results/pc_tworoom_first_pool_triple_alignment_features_20261107_codex.npz"
RESULT = ROOT / "results/pc_tworoom_first_pool_triple_alignment_20261107_codex.json"


def difference(costs):
    a = np.asarray(costs)
    if a.shape != (3, 3) or not np.isfinite(a).all():
        raise ValueError("need finite 3x3 costs")
    return a[:, :1] - a[:, 1:]


def check_source():
    results = json.loads(physical.OUT.read_text())
    saved = json.loads(physical.run.OUT.read_text())
    if (results["source_sha256"] != file_hash(physical.run.OUT)
            or results["runner_sha256"] != file_hash(physical.__file__)
            or results["protocol_sha256"] != file_hash(physical.PROTOCOL)
            or saved["runner_sha256"] != file_hash(physical.run.__file__)
            or saved["search_sha256"] != file_hash(physical.run.search.__file__)):
        raise ValueError("sealed source mismatch")
    return results, saved


def frames():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if FRAMES.exists() or FRAME_META.exists():
        raise FileExistsError("frame outputs already exist")
    prior, saved = check_source()
    inputs_path = physical.run.prior.source.block.prior.inputs.INPUT
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    current, final, rows = [], [], []
    print("Replay 360 sealed RGB candidate chains and verify all physical trace steps.", flush=True)
    try:
        with (np.load(inputs_path, allow_pickle=False) as inputs,
              h5py.File(physical.run.prior.source.block.prior.common.DATA, "r") as db):
            for p, item in zip(prior["rows"], saved["rows"], strict=True):
                root, task = item["root"], item["row"]
                if (p["root"], p["row"]) != (root, task) or int(inputs["rows"][root]) != task:
                    raise ValueError("root mismatch")
                d = physical.common_first(item)["effect"]
                parents, probes, _ = physical.run.search.timed.build(d["candidate_seed"], None, 10)
                bank = np.concatenate((np.zeros_like(probes[:1]), probes))
                root_frames = []
                max_diff = 0.
                initial = None
                for i in range(3):
                    for j in range(3):
                        env.reset(seed=42)
                        actor = env.unwrapped
                        actor._set_state(db["proprio"][task])
                        actor._set_goal_state(db["proprio"][task + 25])
                        image = actor.render().copy()
                        if initial is None:
                            initial = image
                        elif not np.array_equal(image, initial):
                            raise ValueError("initial rendering differs")
                        old = p["physical"][i][j]
                        for t, action in enumerate(np.concatenate((parents[i, :10], bank[j]))):
                            _, _, term, trunc, info = actor.step(action.copy())
                            if trunc or bool(term) != old["success"][t]:
                                raise ValueError("physical flag mismatch")
                            max_diff = max(max_diff, abs(float(info["distance_to_target"]) - old["distance"][t]))
                        root_frames.append(actor.render().copy())
                if max_diff > 1e-6:
                    raise ValueError("physical distance mismatch")
                current.append(initial)
                final.append(root_frames)
                rows.append({"root": root, "row": task, "max_physical_distance_error": max_diff})
                print(f"replayed root={root}", flush=True)
    finally:
        env.close()
    with FRAMES.open("xb") as stream:
        np.savez_compressed(stream, current=np.asarray(current), final=np.asarray(final))
    with FRAME_META.open("x") as stream:
        json.dump({"rows": rows, "frames_sha256": file_hash(FRAMES), "source_sha256": file_hash(physical.OUT),
                   "input_sha256": file_hash(inputs_path), "runner_sha256": file_hash(Path(__file__)),
                   "protocol_sha256": file_hash(PROTOCOL)}, stream, allow_nan=False)


def score():
    if FEATURES.exists() or RESULT.exists():
        raise FileExistsError("feature outputs already exist")
    prior, saved = check_source()
    meta = json.loads(FRAME_META.read_text())
    if (meta["frames_sha256"] != file_hash(FRAMES) or meta["source_sha256"] != file_hash(physical.OUT)
            or meta["runner_sha256"] != file_hash(Path(__file__)) or meta["protocol_sha256"] != file_hash(PROTOCOL)):
        raise ValueError("frame provenance mismatch")
    common = physical.run.prior.source.block.prior.common
    model, scaler, transform, action_hash = common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != saved["model_sha256"] or action_hash != saved["normalizer_sha256"]:
        raise ValueError("frozen model or normalizer mismatch")
    input_path = physical.run.prior.source.block.prior.inputs.INPUT
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, FRAMES, input_path,
                                      common.CHECKPOINT / "weights.pt", common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224, TwoRoom action scaler", clock="five packets of five steps")
    predicted_features, actual_features, target_features, rows = [], [], [], []
    print("Compare frozen model rollout, actual RGB encoding and physical endpoint effects.", flush=True)
    with (np.load(FRAMES, allow_pickle=False) as images, np.load(input_path, allow_pickle=False) as inputs,
          torch.inference_mode()):
        for index, (p, item) in enumerate(zip(prior["rows"], saved["rows"], strict=True)):
            root = item["root"]
            if root != p["root"] or root != meta["rows"][index]["root"]:
                raise ValueError("root mismatch")
            d = physical.common_first(item)["effect"]
            parents, probes, _ = physical.run.search.timed.build(d["candidate_seed"], None, 10)
            bank = np.concatenate((np.zeros_like(probes[:1]), probes))
            chains = np.stack([np.concatenate((parents[i, :10], bank[j])) for i in range(3) for j in range(3)])
            normal = scaler.transform(chains.reshape(-1, 2)).reshape(1, 9, 5, 10)
            pixels = transform(images["current"][index])[None, None, None].cuda()
            goal_rgb = transform(inputs["goal"][root])
            goal = goal_rgb[None, None, None].cuda()
            wrapper = RGBLeWM(model, audit)
            predicted_cost, ctx = wrapper.score({"pixels": pixels, "goal": goal,
                                                  "action": torch.empty((1, 1, 0, 10), device="cuda")},
                                                 torch.tensor(normal, dtype=torch.float32, device="cuda"))
            predicted = ctx["predicted_emb"][0, :, -1]
            encoded_goal = model.encode({"pixels": goal_rgb[None, None].cuda()})["emb"][0, 0]
            batch = torch.stack([transform(frame) for frame in images["final"][index]]).cuda()
            actual = model.encode({"pixels": batch[:, None]})["emb"][:, 0]
            encoded_cost = (actual - encoded_goal).square().sum(-1).cpu().numpy().reshape(3, 3)
            model_cost = predicted_cost[0].cpu().numpy().reshape(3, 3)
            stored = np.asarray(d["probe_cost"])
            score_error = float(np.max(np.abs(model_cost - stored)))
            if score_error > 1e-3:
                raise ValueError("model score differs from sealed candidate pool")
            physical_cost = np.asarray([[p["physical"][i][j]["distance"][-1] for j in range(3)] for i in range(3)])
            a, b, c = difference(model_cost), difference(encoded_cost), difference(physical_cost)
            rows.append({"root": root, "model_cost": model_cost.tolist(), "encoded_cost": encoded_cost.tolist(),
                         "physical_distance": physical_cost.tolist(),
                         "model_vs_encoded": physical.sign_report(a, b),
                         "encoded_vs_physical": physical.sign_report(b, c),
                         "model_vs_physical": physical.sign_report(a, c),
                         "max_sealed_score_error": score_error,
                         "mean_feature_mse": float((predicted - actual).square().mean().cpu()),
                         "mean_abs_goal_score_error": float(np.abs(model_cost - encoded_cost).mean()),
                         "receipts": wrapper.receipts})
            predicted_features.append(predicted.cpu().numpy())
            actual_features.append(actual.cpu().numpy())
            target_features.append(encoded_goal.cpu().numpy())
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model weights changed")
    with FEATURES.open("xb") as stream:
        np.savez_compressed(stream, predicted=np.asarray(predicted_features), actual=np.asarray(actual_features),
                            goal=np.asarray(target_features))
    with RESULT.open("x") as stream:
        json.dump({"track": RGB, "scope": "same-backend exposed panel, evaluation only",
                   "model_sha256": fingerprint, "normalizer_sha256": action_hash,
                   "source_sha256": file_hash(physical.OUT), "frames_sha256": file_hash(FRAMES),
                   "features_sha256": file_hash(FEATURES), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(Path(__file__)), "rows": rows}, stream, allow_nan=False)
    print(f"saved {len(rows)} roots and predicted/actual latent features", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("frames", "score"):
        raise SystemExit("expected frames|score")
    {"frames": frames, "score": score}[sys.argv[1]]()
