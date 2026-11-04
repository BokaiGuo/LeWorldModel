"""Post-hoc predicted and observed RGB latent changes for v7 PushT dispatches."""

import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments import pc_v7_pusht_rgb_feedback_dev_20261104_codex as configured
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

PROTOCOL = configured.ROOT.parent / "docs/PC_V7_PUSHT_EXECUTED_PACKET_DEV_PROTOCOL_20261104_CODEX.md"
OUT = configured.PREFIX.with_name(configured.PREFIX.name + "_executed_packet_diag.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    configured.configure()
    run, host = configured.live, configured.live.HOST
    if os.environ.get("CUDA_VISIBLE_DEVICES") != host.UUID:
        raise ValueError("cooperative GPU UUID required")
    sha = host.bridge.sha_file
    audit_path = configured.PREFIX.with_name(configured.PREFIX.name + "_audit.json")
    audit, manifest, result = (json.loads(path.read_text()) for path in (audit_path, run.MANIFEST, run.RESULT))
    if (audit["manifest_sha256"] != sha(run.MANIFEST) or audit["result_sha256"] != sha(run.RESULT)
            or result["manifest_sha256"] != sha(run.MANIFEST) or audit["physical_steps"] != 30):
        raise ValueError("upstream replay audit mismatch")
    execution_path = run.path("response", "execution.json")
    execution = json.loads(execution_path.read_text())
    contexts = [run.path("response", f"context{i}.npz") for i in range(4)]
    decision_paths = [run.path("response", f"decision{i}.json") for i in range(3)]
    decisions = [json.loads(path.read_text()) for path in decision_paths]
    for i in range(3):
        if (decisions[i]["context_sha256"] != sha(contexts[i])
                or execution["decisions"][i]["decision_sha256"] != sha(decision_paths[i])):
            raise ValueError("context or decision mismatch")
    deps = seal.capture([Path(__file__), PROTOCOL, audit_path, run.MANIFEST, run.RESULT, execution_path,
                         *contexts, *decision_paths, *map(Path, manifest["seal"]["files"])])
    print("Recomputing v7 executed-packet model and observed RGB latent differences.", flush=True)
    with host.lock.gpu_lock(host.LOCK):
        from experiments.gpu_preflight import require_cuda_allocation
        allocation = require_cuda_allocation("cuda:0")
        if host.canonical_gpu_uuid(torch.cuda.get_device_properties(0).uuid) != host.canonical_gpu_uuid(host.UUID):
            raise ValueError("unexpected GPU")
        device = torch.device("cuda:0")
        model = run.saved.load_compatible_pretrained(run.saved.POLICY, device="cuda").eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        model_hash = run.saved.checked.tensor_state_hash(model)
        if model_hash != result["model_before"] or model_hash != result["model_after"]:
            raise ValueError("model mismatch")
        transform = run.saved.p0.image_transform()
        mean = np.tile(np.asarray(run.saved.p0.PUBLIC_ACTION_MEAN, np.float32), 5)
        std = np.tile(np.asarray(run.saved.p0.PUBLIC_ACTION_STD, np.float32), 5)
        lower, upper = (torch.as_tensor(x, device=device) for x in ((-1-mean)/std, (1-mean)/std))
        rows = []
        with torch.no_grad():
            for step in range(3):
                with np.load(contexts[step], allow_pickle=False) as a, np.load(contexts[step+1], allow_pickle=False) as b:
                    np.testing.assert_array_equal(a["goal"], b["goal"])
                    frames = run.saved.p0._frames_tensor(list(a["frames"]), transform=transform, device=device)
                    next_frames = run.saved.p0._frames_tensor(list(b["frames"]), transform=transform, device=device)
                    encoded = model.encode({"pixels": frames})["emb"]
                    current = encoded[0, -1]
                    actual = model.encode({"pixels": next_frames})["emb"][0, -1]
                    goal = run.saved.p0._goal_latent(model, a["goal"], transform=transform, device=device)
                    history = run.saved.live.adapter.raw_to_packets(a["history"][None], mean, std, device=device)
                    future = run.saved.live.adapter.raw_to_packets(
                        np.asarray(decisions[step]["dispatch"], np.float32)[None], mean, std, device=device)
                    predicted = run.saved.live.adapter.cached_terminal(model, encoded, history, future, lower, upper)[0]
                    def mse(x, y):
                        return float((x-y).square().mean().cpu())
                    current_goal, pred_goal, actual_goal = mse(current, goal), mse(predicted, goal), mse(actual, goal)
                    old, new = execution["outcomes"][step:step+2]
                    rows.append({"step": step, "prediction_actual_mse": mse(predicted, actual),
                        "persistence_actual_mse": mse(current, actual),
                        "predicted_goal_mse_change": pred_goal-current_goal,
                        "actual_goal_mse_change": actual_goal-current_goal,
                        "object_xy_change": new["pose"]["xy_distance"]-old["pose"]["xy_distance"],
                        "object_angle_change": new["pose"]["angle_distance"]-old["pose"]["angle_distance"]})
            torch.cuda.synchronize()
        if run.saved.checked.tensor_state_hash(model) != model_hash:
            raise ValueError("model drift")
    seal.verify(deps)
    host.write_new(OUT, {"dependencies": deps, "model_hash": model_hash, "cuda_allocation": allocation,
                         "measurements": rows, "scope": "post-hoc, one exposed root, same model and simulator"})
    print(f"Saved {len(rows)} executed packet comparisons.", flush=True)


if __name__ == "__main__":
    main()
