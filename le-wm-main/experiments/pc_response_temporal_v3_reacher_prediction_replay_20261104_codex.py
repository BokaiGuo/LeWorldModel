"""Recompute every sealed Reacher v3 prediction via the original runner."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments import pc_response_temporal_v3_reacher_rgb_live_20261104_codex as live

OUT = live.PREFIX.with_name(live.PREFIX.name + "_prediction_replay.json")
PROTOCOL = live.ROOT.parent / "docs/PC_RESPONSE_TEMPORAL_V3_REACHER_PREDICTION_REPLAY_PROTOCOL_20261104_CODEX.md"
AUDIT = live.PREFIX.with_name(live.PREFIX.name + "_audit.json")


def inverse_packets(normalized, lower, upper):
    midpoint, radius = (lower + upper) / 2, (upper - lower) / 2
    ratio = (normalized - midpoint) / radius
    if not torch.isfinite(ratio).all() or torch.any(torch.abs(ratio) > 1 + 2e-6):
        raise ValueError("packet outside inverse parameterization")
    params = torch.atanh(ratio.clamp(-1, 1))
    reconstructed = midpoint + radius * torch.tanh(params)
    error = torch.max(torch.abs(reconstructed - normalized)).item()
    if error > 2e-6:
        raise ValueError("reference mapping changed actions")
    return params, error


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    host, setup, p0, adapter = live.host, live.setup, live.p0, live.adapter
    if os.environ.get("CUDA_VISIBLE_DEVICES") != host.UUID:
        raise ValueError("cooperative GPU UUID required")
    sha = host.bridge.sha_file
    source = json.loads(live.run.SOURCE.read_text())
    manifest = json.loads(live.MANIFEST.read_text())
    audit = json.loads(AUDIT.read_text())
    if manifest["dependencies"] != live.dependencies(source) or audit["manifest_sha256"] != sha(live.MANIFEST):
        raise ValueError("sealed dependency or audit mismatch")
    rows = []
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
        model_hash = setup._parameter_hash(model)
        if model_hash != source["model_parameter_hash_expected"]:
            raise ValueError("model hash mismatch")
        processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
        lo, hi = setup._normalized_action_bounds(mean, std, np.full(2, -1), np.full(2, 1))
        lower, upper = torch.as_tensor(lo, device=device), torch.as_tensor(hi, device=device)
        for arm in live.ARMS:
            for step in range(3):
                context_path = live.path(arm, f"context{step}.npz")
                decision_path = live.path(arm, f"decision{step}.json")
                data = json.loads(decision_path.read_text())
                if data["context_sha256"] != sha(context_path):
                    raise ValueError("context hash mismatch")
                with np.load(context_path, allow_pickle=False) as ctx:
                    pixels, goal, history = p0._prepare(processor, ctx["frames"], ctx["root"],
                                                         ctx["goal"], ctx["history"], device)
                with torch.no_grad():
                    encoded = p0._latent_context(model, pixels, history)
                    goal_latent = p0._goal_latent(model, goal)
                    chart = adapter.goal_chart(encoded, goal_latent, data["chart_seed"])
                    target = goal_latent @ chart
                    chart_error = float(np.max(np.abs(chart.cpu().numpy() - data["chart"])))
                    target_error = float(np.max(np.abs(target.cpu().numpy() - data["target"])))
                    if max(chart_error, target_error) > 2e-5:
                        raise ValueError("chart or target mismatch")
                    max_position = max_score = max_action = 0.0
                    samples = 0
                    for batch in data["trace"]:
                        raw = np.asarray(batch["raw"], dtype=np.float32)
                        normalized = adapter.raw_to_packets(raw, mean, std, device=device)
                        params, error = inverse_packets(normalized, lower, upper)
                        terminal = p0._terminal_batch(model, pixels, goal, history, params,
                                                      action_lower=lower, action_upper=upper)
                        positions = (terminal @ chart).cpu().numpy()
                        scores = (terminal - goal_latent).square().mean(dim=1).cpu().numpy()
                        if not np.isfinite(positions).all() or not np.isfinite(scores).all():
                            raise ValueError("non-finite reference output")
                        max_position = max(max_position, float(np.max(np.abs(positions - batch["positions"]))))
                        max_score = max(max_score, float(np.max(np.abs(scores - batch["scores"]))))
                        max_action = max(max_action, error)
                        samples += len(raw)
                    if max_position > 2e-5 or max_score > 2e-6:
                        raise ValueError(f"prediction mismatch {arm} {step}: {max_position}, {max_score}")
                rows.append({"arm": arm, "step": step, "batches": len(data["trace"]), "samples": samples,
                             "chart_max_abs_error": chart_error, "target_max_abs_error": target_error,
                             "action_max_abs_error": max_action, "position_max_abs_error": max_position,
                             "score_max_abs_error": max_score, "decision_sha256": sha(decision_path)})
                print(f"{arm} step={step}: recomputed {len(data['trace'])} batches, {samples} samples", flush=True)
        if setup._parameter_hash(model) != model_hash or live.dependencies(source) != manifest["dependencies"]:
            raise ValueError("model or dependency drift")
    host.write_new(OUT, {"manifest_sha256": sha(live.MANIFEST), "audit_sha256": sha(AUDIT),
                         "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
                         "model_parameter_hash": model_hash, "cuda_allocation": allocation, "rows": rows,
                         "scope": "same frozen model, preprocessing and GPU; reference rollout, not independent data"})
    print("All sealed prediction batches reproduced within the preregistered tolerances.", flush=True)


if __name__ == "__main__":
    main()
