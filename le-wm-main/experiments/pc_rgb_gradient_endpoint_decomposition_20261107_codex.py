"""Compare frozen-model prediction, actual RGB encoding, and physical outcomes."""
import json
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from eval import img_transform
from experiments import pc_rgb_gradient_endpoint_frames_20261107_codex as capture
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_rgb_single_frame_contract_20261107_codex import SingleFrameRGB, model_digest

ROOT = capture.ROOT
PUBLIC = capture.previous.planner.PUBLIC
PLAN = capture.previous.planner.OUT
FEATURES = ROOT / "results/pc_rgb_gradient_endpoint_features_20261107_codex.npz"
OUT = ROOT / "results/pc_rgb_gradient_endpoint_decomposition_20261107_codex.json"
CHECKPOINT = capture.previous.planner.CHECKPOINT


def pairwise(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired finite scores required")
    agree = disagree = ties = 0
    for i in range(len(a)):
        for j in range(i):
            sa, sb = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if sa == 0 or sb == 0:
                ties += 1
            elif sa == sb:
                agree += 1
            else:
                disagree += 1
    return {"agree": agree, "disagree": disagree, "ties": ties}


def main():
    if FEATURES.exists() or OUT.exists():
        raise FileExistsError("analysis artifacts already exist")
    plan = json.loads(PLAN.read_text())
    physical = json.loads(capture.OUT.read_text())
    if (physical["plan_sha256"] != file_hash(PLAN) or physical["frames_sha256"] != file_hash(capture.FRAMES)
            or physical["protocol_sha256"] != file_hash(capture.PROTOCOL)
            or plan["public_sha256"] != file_hash(PUBLIC)):
        raise ValueError("sealed input drift")
    model = load_compatible_pretrained(str(CHECKPOINT), device="cuda")
    digest = model_digest(model)
    if digest != plan["model_digest"]:
        raise ValueError("model drift")
    audit = freeze_audit(RGB, "lewm", [__file__, capture.PROTOCOL, PUBLIC, PLAN, capture.FRAMES,
                         CHECKPOINT / "weights.pt", CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; frozen normalizer", clock="25 physical steps")
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    rows, vectors = [], {}
    print("Compare predicted features, post-replay RGB encoding and physical distance.", flush=True)
    with (np.load(PUBLIC, allow_pickle=False) as public, np.load(capture.FRAMES, allow_pickle=False) as frames,
          torch.inference_mode()):
        capture.previous.planner.validate_public(public)
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_, scaler.n_features_in_ = public["action_mean"], public["action_scale"], 2
        guard = SingleFrameRGB(model, audit, expected_model_digest=digest, action_mean=scaler.mean_,
                               action_scale=scaler.scale_, packet_steps=5, physical_low=-1, physical_high=1)
        for p, truth in zip(plan["rows"], physical["rows"], strict=True):
            if p["episode"] != truth["episode"]:
                raise ValueError("episode mismatch")
            idx, = np.flatnonzero(public["episodes"] == p["episode"])
            context = {"pixels": transform(public["current"][idx])[None, None, None].cuda(),
                       "goal": transform(public["goal"][idx])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            actions = np.asarray(p["candidates"], np.float32)
            normalized = scaler.transform(actions.reshape(-1, 2)).astype(np.float32).reshape(1, 5, 5, 10)
            score, state = guard.score(context, torch.as_tensor(normalized, device="cuda"))
            predicted, target = state["predicted_emb"][0, :, -1], state["goal_emb"][0, -1]
            images = frames[truth["key"]]
            if images.shape[0] != 5:
                raise ValueError("expected five endpoint images")
            encoded = model.encode({"pixels": torch.stack([transform(image) for image in images]).cuda()[:, None]})["emb"][:, 0]
            computed = (predicted - target).square().sum(-1)
            actual_cost = (encoded - target).square().sum(-1)
            if (not torch.allclose(score[0], computed, rtol=1e-5, atol=1e-3)
                    or not np.allclose(score[0].cpu().numpy(), p["scores"], rtol=1e-5, atol=1e-3)):
                raise ValueError("predicted score does not reproduce planned objective")
            predicted_cost, actual_cost = score[0].cpu().numpy(), actual_cost.cpu().numpy()
            distance = np.asarray(truth["distance"])
            if int(np.argmin(predicted_cost)) != p["selected_index"]:
                raise ValueError("replayed selection differs")
            predicted_index, encoded_index, physical_index = int(np.argmin(predicted_cost)), int(np.argmin(actual_cost)), int(np.argmin(distance))
            key = truth["key"]
            vectors[key + "_predicted"] = predicted.cpu().numpy()
            vectors[key + "_actual"] = encoded.cpu().numpy()
            vectors[key + "_goal"] = target.cpu().numpy()
            rows.append({"episode": p["episode"], "initial_success": truth["initial_success"],
                         "predicted_cost": predicted_cost.tolist(), "actual_cost": actual_cost.tolist(),
                         "distance": distance.tolist(), "ever_success": truth["ever_success"],
                         "predicted_vs_actual": pairwise(predicted_cost, actual_cost),
                         "predicted_vs_distance": pairwise(predicted_cost, distance),
                         "actual_vs_distance": pairwise(actual_cost, distance),
                         "predicted_index": predicted_index, "actual_index_oracle": encoded_index,
                         "physical_index_oracle": physical_index,
                         "selected_distance": float(distance[predicted_index]),
                         "actual_oracle_distance": float(distance[encoded_index]),
                         "physical_best_distance": float(distance[physical_index]),
                         "latent_mse": (predicted - encoded).square().mean(-1).cpu().tolist()})
            print(f"episode={p['episode']} selection={predicted_index} physical_oracle={physical_index}", flush=True)
        guard.verify_unchanged()
    with FEATURES.open("xb") as stream:
        np.savez_compressed(stream, **vectors)
    with OUT.open("x") as stream:
        json.dump({"track": RGB, "protocol_sha256": file_hash(capture.PROTOCOL),
                   "runner_sha256": file_hash(__file__), "plan_sha256": file_hash(PLAN),
                   "frame_trace_sha256": file_hash(capture.OUT), "features_sha256": file_hash(FEATURES),
                   "model_digest": digest, "rows": rows,
                   "scope": "post-hoc endpoint diagnosis on exposed tasks; oracles are not deployable"},
                  stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
