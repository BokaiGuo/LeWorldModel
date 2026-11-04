"""Re-encode saved RGB and recompute predictions using the reference rollout."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments import pc_v4_reacher_final_packet_latent_gap_20261104_codex as source_run
from experiments.pc_response_temporal_v3_reacher_prediction_replay_20261104_codex import inverse_packets
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = source_run.ROOT
PREFIX = ROOT / "results/pc_v4_reacher_final_packet_latent_gap_v2_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
RESULT = PREFIX.with_name(PREFIX.name + "_result.json")
ARRAYS = PREFIX.with_name(PREFIX.name + "_arrays.npz")
OUT = PREFIX.with_name(PREFIX.name + "_audit.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    prev, live = source_run.prev, source_run.live
    host, setup, p0, adapter = live.host, live.setup, live.p0, live.adapter
    if os.environ.get("CUDA_VISIBLE_DEVICES") != host.UUID:
        raise ValueError("GPU UUID required")
    sha = host.bridge.sha_file
    manifest, result = json.loads(MANIFEST.read_text()), json.loads(RESULT.read_text())
    seal.verify(manifest["seal"])
    if result["manifest_sha256"] != sha(MANIFEST) or result["arrays_sha256"] != sha(ARRAYS):
        raise ValueError("manifest or array mismatch")
    source = json.loads(live.run.SOURCE.read_text())
    prior = json.loads(prev.RESULT.read_text())
    with np.load(ARRAYS, allow_pickle=False) as bundle:
        data = {k: bundle[k].copy() for k in bundle.files}
    if data["frames"].shape[0] != 5 or data["actions"].shape != (5, 5, 2):
        raise ValueError("array dimensions")
    for i, branch in enumerate(prior["rows"]):
        np.testing.assert_array_equal(data["actions"][i], np.asarray(branch["raw"], np.float32))
        if live.context_lib.rgb_digest(data["frames"][i]) != result["rows"][i]["frame_sha256"]:
            raise ValueError("RGB digest mismatch")
    with host.lock.gpu_lock(host.LOCK):
        from experiments.gpu_preflight import require_cuda_allocation
        allocation = require_cuda_allocation("cuda:0")
        if host.canonical_gpu_uuid(torch.cuda.get_device_properties(0).uuid) != host.canonical_gpu_uuid(host.UUID):
            raise ValueError("unexpected GPU")
        device = torch.device("cuda:0")
        setup.install_missing_field_compat()
        stats = Path(source["action_stats_path"])
        setup.validate_training_semantic_receipt(stats)
        mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
        model = setup.load_compatible_pretrained(Path(source["weights_path"]).parent, device="cuda").eval()
        model.interpolate_pos_encoding = True
        model.requires_grad_(False)
        if setup._parameter_hash(model) != result["model_hash"]:
            raise ValueError("model mismatch")
        processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
        lo, hi = setup._normalized_action_bounds(mean, std, np.full(2, -1), np.full(2, 1))
        lo, hi = torch.as_tensor(lo, device=device), torch.as_tensor(hi, device=device)
        with np.load(prev.CONTEXT2, allow_pickle=False) as ctx:
            frames, root, goal_frame, history_raw = [ctx[k].copy() for k in ("frames", "root", "goal", "history")]
        pixels, goal, history = p0._prepare(processor, frames, root, goal_frame, history_raw, device)
        with torch.no_grad():
            ref_target = p0._goal_latent(model, goal).cpu().numpy()
            normalized = adapter.raw_to_packets(data["actions"], mean, std, device=device)
            params, action_error = inverse_packets(normalized, lo, hi)
            ref_pred = p0._terminal_batch(model, pixels, goal, history, params,
                                          action_lower=lo, action_upper=hi).cpu().numpy()
            ref_obs = []
            for frame in data["frames"]:
                _, obs, _ = p0._prepare(processor, frames, root, np.ascontiguousarray(frame), history_raw, device)
                ref_obs.append(p0._goal_latent(model, obs).cpu().numpy())
            ref_obs = np.stack(ref_obs)
        if setup._parameter_hash(model) != result["model_hash"]:
            raise ValueError("model changed")
    errors = {"target_max_abs": float(np.max(np.abs(ref_target - data["target"]))),
              "predicted_max_abs": float(np.max(np.abs(ref_pred - data["predicted"]))),
              "observed_max_abs": float(np.max(np.abs(ref_obs - data["observed"]))),
              "inverse_action_max_abs": float(action_error)}
    if max(errors.values()) > 2e-5:
        raise ValueError(f"reference mismatch {errors}")
    for i, row in enumerate(result["rows"]):
        p, o, g = data["predicted"][i], data["observed"][i], data["target"]
        expected = [float(np.mean((p-g)**2)), float(np.mean((o-g)**2)), float(np.mean((p-o)**2))]
        np.testing.assert_allclose(expected, [row["model_goal_mse"], row["observed_goal_mse"],
                                             row["prediction_observation_mse"]], rtol=0, atol=1e-7)
        if row["physical_distance"] != prior["rows"][i]["trajectory"][-1]["distance"]:
            raise ValueError("physical mismatch")
    delta_rows = []
    for i, row in enumerate(result["rows"]):
        pd, od = data["predicted"][i]-data["predicted"][2], data["observed"][i]-data["observed"][2]
        pn, on = float(np.linalg.norm(pd)), float(np.linalg.norm(od))
        delta_rows.append({"name": row["name"], "predicted_norm": pn, "observed_norm": on,
                           "difference_norm": float(np.linalg.norm(pd-od)),
                           "cosine": float(pd @ od / (pn * on)) if min(pn, on) > 1e-8 else None})
    inversions = {}
    for left, right in (("model_goal_mse", "observed_goal_mse"), ("model_goal_mse", "physical_distance"),
                        ("observed_goal_mse", "physical_distance")):
        inversions[f"{left}_vs_{right}"] = sum(
            (result["rows"][i][left]-result["rows"][j][left]) *
            (result["rows"][i][right]-result["rows"][j][right]) < 0
            for i in range(5) for j in range(i+1, 5))
    seal.verify(manifest["seal"])
    host.write_new(OUT, {"manifest_sha256": sha(MANIFEST), "result_sha256": sha(RESULT),
                         "arrays_sha256": sha(ARRAYS), "audit_script_sha256": sha(Path(__file__)),
                         "errors": errors, "zero_reference_deltas": delta_rows, "pair_inversions": inversions,
                         "cuda_allocation": allocation,
                         "scope": "same checkpoint, saved frames, reference rollout; not independent model"})
    print(f"Re-encoded 5 frames; pairwise inversions: {inversions}", flush=True)


if __name__ == "__main__":
    main()
