"""Re-encode saved frames and independently recompute model predictions."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments import pc_v4_reacher_direct_spectral_probe_v3_20261104_codex as run
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = run.PREFIX.with_name(run.PREFIX.name + "_reencode.json")
AUDIT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    host, setup, p0 = run.live.host, run.live.setup, run.live.p0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != host.UUID:
        raise ValueError("GPU UUID required")
    sha = host.bridge.sha_file
    manifest, result, audit = [json.loads(p.read_text()) for p in (run.MANIFEST, run.RESULT, AUDIT)]
    seal.verify(manifest["seal"])
    if (result["manifest_sha256"] != sha(run.MANIFEST) or result["arrays_sha256"] != sha(run.ARRAYS)
            or audit["result_sha256"] != sha(run.RESULT) or audit["arrays_sha256"] != sha(run.ARRAYS)):
        raise ValueError("source seal mismatch")
    source = json.loads(run.live.run.SOURCE.read_text())
    with np.load(run.ARRAYS, allow_pickle=False) as c:
        frames, observed, target = c["frames"].copy(), c["observed"].copy(), c["target"].copy()
        actions, predicted = c["actions"].copy(), c["predicted"].copy()
    with np.load(run.prev.CONTEXT2, allow_pickle=False) as c:
        frames0, root, goal0, history0 = [c[k].copy() for k in ("frames", "root", "goal", "history")]
    with host.lock.gpu_lock(host.LOCK):
        from experiments.gpu_preflight import require_cuda_allocation
        allocation = require_cuda_allocation("cuda:0")
        if host.canonical_gpu_uuid(torch.cuda.get_device_properties(0).uuid) != host.canonical_gpu_uuid(host.UUID):
            raise ValueError("unexpected GPU")
        setup.install_missing_field_compat()
        stats = Path(source["action_stats_path"])
        setup.validate_training_semantic_receipt(stats)
        mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
        model = setup.load_compatible_pretrained(Path(source["weights_path"]).parent, device="cuda").eval()
        model.interpolate_pos_encoding = True
        model.requires_grad_(False)
        if setup._parameter_hash(model) != result["model_hash"]:
            raise ValueError("model hash mismatch")
        processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
        pixels, goal, history = p0._prepare(processor, frames0, root, goal0, history0, torch.device("cuda:0"))
        with torch.no_grad():
            goal_latent = p0._goal_latent(model, goal).cpu().numpy()
            again = []
            for frame in frames:
                _, image, _ = p0._prepare(processor, frames0, root, np.ascontiguousarray(frame), history0,
                                           torch.device("cuda:0"))
                again.append(p0._goal_latent(model, image).cpu().numpy())
            again = np.stack(again).reshape(observed.shape)
            encoded = p0._latent_context(model, pixels, history)
            lo, hi = setup._normalized_action_bounds(mean, std, np.full(2, -1), np.full(2, 1))
            device = torch.device("cuda:0")
            future = run.live.adapter.raw_to_packets(actions, mean, std, device=device)
            recomputed = run.live.adapter.cached_terminal(model, encoded, history, future,
                torch.as_tensor(lo, device=device), torch.as_tensor(hi, device=device)).cpu().numpy()
        if setup._parameter_hash(model) != result["model_hash"]:
            raise ValueError("model changed")
    error = float(np.max(np.abs(again-observed)))
    target_error = float(np.max(np.abs(goal_latent-target)))
    prediction_error = float(np.max(np.abs(recomputed-predicted)))
    if max(error, target_error, prediction_error) > 2e-5:
        raise ValueError("re-encoding mismatch")
    seal.verify(manifest["seal"])
    host.write_new(OUT, {"manifest_sha256": sha(run.MANIFEST), "result_sha256": sha(run.RESULT),
                         "arrays_sha256": sha(run.ARRAYS), "audit_sha256": sha(AUDIT),
                         "script_sha256": sha(Path(__file__)), "frames": 18,
                         "max_abs": error, "target_max_abs": target_error, "cuda_allocation": allocation,
                         "prediction_max_abs": prediction_error,
                         "scope": "same checkpoint and saved frames, not independent model"})
    print(f"Re-encoded 18 frames and recomputed predictions, max errors {error}, {prediction_error}.", flush=True)


if __name__ == "__main__":
    main()
