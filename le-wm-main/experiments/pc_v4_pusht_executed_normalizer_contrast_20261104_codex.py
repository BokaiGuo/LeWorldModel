"""Post-hoc fixed public/archive normalizer comparison on six actual packets."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments import pc_v4_pusht_rgb_feedback_dev_20261104_codex as run
from experiments import pc_v4_pusht_executed_packet_diag_20261104_codex as previous

ROOT = run.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V4_PUSHT_EXECUTED_NORMALIZER_CONTRAST_PROTOCOL_20261104_CODEX.md"
STATS = ROOT / "results/pusht_d2p_archive_action_statistics_v1_20261026.json"
OUT = ROOT / "results/pc_v4_pusht_executed_normalizer_contrast_20261104_codex.json"
EXPECTED_STATS_SHA = "c816c535e08a946a6f7464c659b23e98022b0375359bf5a7add4edd6e018cdda"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    host, sha = run.HOST, run.HOST.bridge.sha_file
    if os.environ.get("CUDA_VISIBLE_DEVICES") != host.UUID or sha(STATS) != EXPECTED_STATS_SHA:
        raise ValueError("GPU or archive statistics identity mismatch")
    diagnostic = json.loads(previous.OUT.read_text())
    run.seal.verify(diagnostic["dependencies"])
    stats = json.loads(STATS.read_text())["statistics"]
    conventions = {
        "public": (run.saved.p0.PUBLIC_ACTION_MEAN, run.saved.p0.PUBLIC_ACTION_STD),
        "archive_sample": (stats["mean"], stats["sample_std"]),
    }
    deps = run.seal.capture([Path(__file__), PROTOCOL, STATS, previous.OUT])
    print("Comparing fixed action statistics on the same six executed PushT packets.", flush=True)
    with host.lock.gpu_lock(host.LOCK):
        from experiments.gpu_preflight import require_cuda_allocation
        allocation = require_cuda_allocation("cuda:0")
        if host.canonical_gpu_uuid(torch.cuda.get_device_properties(0).uuid) != host.canonical_gpu_uuid(host.UUID):
            raise ValueError("unexpected GPU")
        device = torch.device("cuda:0")
        model = run.saved.load_compatible_pretrained(run.saved.POLICY, device="cuda").eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        model_hash = run.saved.checked.tensor_state_hash(model)
        if model_hash != diagnostic["model_hash"]:
            raise ValueError("model mismatch")
        transform = run.saved.p0.image_transform()
        rows = []
        with torch.no_grad():
            for prior in diagnostic["measurements"]:
                arm, step = prior["arm"], prior["step"]
                with np.load(run.path(arm, f"context{step}.npz"), allow_pickle=False) as source:
                    ctx = {key: source[key].copy() for key in ("frames", "history", "goal")}
                with np.load(run.path(arm, f"context{step+1}.npz"), allow_pickle=False) as source:
                    nxt = {key: source[key].copy() for key in ("frames", "goal")}
                np.testing.assert_array_equal(ctx["goal"], nxt["goal"])
                decision = json.loads(run.path(arm, f"decision{step}.json").read_text())
                frames = run.saved.p0._frames_tensor(list(ctx["frames"]), transform=transform, device=device)
                next_frames = run.saved.p0._frames_tensor(list(nxt["frames"]), transform=transform, device=device)
                encoded = model.encode({"pixels": frames})["emb"]
                actual = model.encode({"pixels": next_frames})["emb"][0, -1]
                current = encoded[0, -1]
                goal = run.saved.p0._goal_latent(model, ctx["goal"], transform=transform, device=device)
                predictions, arms = {}, {}
                for name, (raw_mean, raw_std) in conventions.items():
                    mean = np.tile(np.asarray(raw_mean, np.float32), 5)
                    std = np.tile(np.asarray(raw_std, np.float32), 5)
                    lower, upper = (torch.as_tensor(x, device=device) for x in ((-1-mean)/std, (1-mean)/std))
                    history = run.saved.live.adapter.raw_to_packets(ctx["history"][None], mean, std, device=device)
                    future = run.saved.live.adapter.raw_to_packets(
                        np.asarray(decision["dispatch"], np.float32)[None], mean, std, device=device)
                    predicted = run.saved.live.adapter.cached_terminal(model, encoded, history, future, lower, upper)[0]
                    predictions[name] = predicted
                    arms[name] = {"prediction_actual_mse": float((predicted-actual).square().mean().cpu()),
                        "predicted_goal_mse_change": float(((predicted-goal).square().mean() -
                                                            (current-goal).square().mean()).cpu())}
                for metric in ("prediction_actual_mse", "predicted_goal_mse_change"):
                    if abs(arms["public"][metric] - prior[metric]) > 2e-6:
                        raise ValueError(f"public branch differs from original {arm}/{step}/{metric}")
                rows.append({"arm": arm, "step": step, "statistics": arms,
                    "actual_goal_mse_change": prior["actual_goal_mse_change"],
                    "persistence_actual_mse": prior["persistence_actual_mse"],
                    "prediction_difference_l2": float(torch.linalg.vector_norm(
                        predictions["public"]-predictions["archive_sample"]).cpu())})
        if run.saved.checked.tensor_state_hash(model) != model_hash:
            raise ValueError("model drift")
    run.seal.verify(diagnostic["dependencies"])
    run.seal.verify(deps)
    host.write_new(OUT, {"dependencies": deps, "source_diagnostic_sha256": sha(previous.OUT),
        "model_hash": model_hash, "cuda_allocation": allocation, "rows": rows,
        "scope": "post-hoc fixed statistics on six observed transitions; not training provenance"})
    print("Saved all six paired normalizer comparisons.", flush=True)


if __name__ == "__main__":
    main()
