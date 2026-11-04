"""Reference explicit-packet replay of the saved V5 PointMaze predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_v5_pointmaze_rgb_live_v2_dev_20261104_codex as configured
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

run, ROOT = configured.live, configured.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V5_POINTMAZE_PREDICTION_REPLAY_PROTOCOL_20261104_CODEX.md"
AUDIT = configured.PREFIX.with_name(configured.PREFIX.name + "_audit.json")
OUTPUT = configured.PREFIX.with_name(configured.PREFIX.name + "_prediction_replay.json")


def main():
    configured.configure()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha, h = run.old.prior.sha, run.h
    manifest, audit = json.loads(run.MANIFEST.read_text()), json.loads(AUDIT.read_text())
    if manifest["dependencies"] != run.dependencies() or audit["manifest_sha256"] != sha(run.MANIFEST):
        raise ValueError("upstream manifest mismatch")
    decisions, images, receipt_path = run.paths("response")
    receipt = json.loads(receipt_path.read_text())
    if receipt["images_sha256"] != sha(images) or receipt["manifest_sha256"] != sha(run.MANIFEST):
        raise ValueError("RGB binding mismatch")
    deps = seal.capture([Path(__file__), PROTOCOL, run.MANIFEST, AUDIT, images, receipt_path,
                         *decisions, *[Path(p) for p in manifest["dependencies"]]])
    source = json.loads(run.old.DECISIONS.read_text())
    camera, calibration = json.loads(h.CAMERA.read_text()), json.loads(h.OLD.read_text())
    metadata, offset = camera["metadata"], calibration["body_offset_from_static_scene"]
    with np.load(images, allow_pickle=False) as archive, \
            np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
            h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        frames, goal = archive["rgb"], obstacle["goal_rgb"]
        device = h.require_cuda()
        args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                  config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
        model, dataset = h.load_model_and_dataset(args, device)
        model.eval()
        model_hash = h.parameter_hash(model)
        if model_hash != source["model_hash"] or sha(h.DEFAULT_CHECKPOINT) != source["checkpoint_sha256"]:
            raise ValueError("frozen model mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not np.array_equal(mean.cpu().numpy(), source["action_mean"]) or not np.array_equal(std.cpu().numpy(), source["action_std"]):
            raise ValueError("action statistics mismatch")
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        rows = []
        with torch.no_grad():
            goal_state, _ = h.estimate_state(goal, metadata, offset)
            target = h.encode_with_state(model, goal, goal_state, device)["proprio"].reshape(256, 16)
            for step, decision in enumerate(decisions):
                data = json.loads(decision.read_text())
                binding = data["binding"]
                ticks = binding["ticks"]
                if binding["rgb_sha256"] != [run.old.digest_image(frames[t]) for t in ticks]:
                    raise ValueError("context image mismatch")
                entries = [h.encode_with_state(model, frames[t],
                           h.causal_state(frames[t-1], frames[t], metadata, offset, .05), device)
                           for t in ticks]
                visual = torch.cat([entry["visual"] for entry in entries], dim=1)
                proprio = torch.cat([entry["proprio"] for entry in entries], dim=1)
                history = ((torch.as_tensor(binding["history_actions"], dtype=torch.float32, device=device)-mean)
                           / std).reshape(2, 1, 10)
                n, tokens, max_pos, max_score = 0, 0, 0., 0.
                for batch in data["trace"]:
                    requested = np.asarray(batch["requested"])
                    raw = np.asarray(batch["raw"], np.float32)
                    np.testing.assert_array_equal(requested.astype(np.float32), raw)
                    count, length, _ = raw.shape
                    future = ((torch.as_tensor(raw, device=device)-mean)/std).reshape(count, length//5, 10)
                    video = visual.expand(count, *visual.shape[1:])
                    prop = proprio.expand(count, *proprio.shape[1:])
                    outgoing = history.expand(2, count, 10).transpose(0, 1)
                    for index in range(future.shape[1]):
                        acts = torch.cat((outgoing, future[:, index:index+1]), dim=1)
                        window = min(3, video.shape[1])
                        prediction, _, ppred = model.model.forward_pred(
                            video[:, -window:], model.model.encode_act(acts[:, -window:]), prop[:, -window:])
                        video = torch.cat((video, prediction[:, -1:]), dim=1)
                        prop = torch.cat((prop, ppred[:, -1:]), dim=1)
                        outgoing = acts
                    features = prop[:, -1].reshape(count, 256, 16)
                    position = ((features.mean(1) @ pinv.T)[:, :2] * pos_std).cpu().numpy()
                    score = h.projected_cost(features, target, pinv, pos_std).cpu().numpy()
                    max_pos = max(max_pos, float(np.max(np.abs(position-np.asarray(batch["positions"])))))
                    max_score = max(max_score, float(np.max(np.abs(score-np.asarray(batch["scores"])))))
                    n, tokens = n + count, tokens + count*length//5
                    if batch["tokens"] != count*length//5:
                        raise ValueError("token mismatch")
                if (max_pos > 2e-5 or max_score > 2e-6 or tokens != data["counts"]["prediction_token_samples"]
                        or n != data["counts"]["trajectory_samples"]):
                    raise ValueError(f"replay mismatch step={step}, pos={max_pos}, score={max_score}")
                rows.append({"step": step, "samples": n, "batches": len(data["trace"]), "tokens": tokens,
                             "max_position_error": max_pos, "max_score_error": max_score})
                print(f"step={step}: reference replayed {n} samples", flush=True)
        if h.parameter_hash(model) != model_hash:
            raise ValueError("model drift")
    seal.verify(deps)
    run.old.write_json(OUTPUT, {"dependencies": deps, "manifest_sha256": sha(run.MANIFEST),
                                "model_hash": model_hash, "rows": rows,
                                "scope": "same checkpoint and RGB, explicit packet reference"})


if __name__ == "__main__":
    main()
