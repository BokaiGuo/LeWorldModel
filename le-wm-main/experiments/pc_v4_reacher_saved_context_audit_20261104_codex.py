"""Recompute v4 RGB-context batches with the original reference rollout."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments import pc_v4_reacher_saved_context_20261104_codex as run
from experiments.pc_response_temporal_v3_reacher_prediction_replay_20261104_codex import inverse_packets
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    live = run.live
    host, setup, p0, adapter = live.host, live.setup, live.p0, live.adapter
    if os.environ.get("CUDA_VISIBLE_DEVICES") != host.UUID:
        raise ValueError("cooperative GPU UUID required")
    sha = host.bridge.sha_file
    manifest, data = json.loads(run.MANIFEST.read_text()), json.loads(run.RESULT.read_text())
    source = json.loads(live.run.SOURCE.read_text())
    if data["manifest_sha256"] != sha(run.MANIFEST) or data["context_sha256"] != sha(run.CONTEXT):
        raise ValueError("manifest or context mismatch")
    seal.verify(manifest["seal"])
    count = tokens = 0
    max_pos = max_score = max_action = 0.0
    all_full = []
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
        if model_hash != manifest["source_model_parameter_hash"]:
            raise ValueError("model mismatch")
        processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
        lo, hi = setup._normalized_action_bounds(mean, std, np.full(2, -1), np.full(2, 1))
        lower, upper = torch.as_tensor(lo, device=device), torch.as_tensor(hi, device=device)
        with np.load(run.CONTEXT, allow_pickle=False) as ctx:
            pixels, goal, history = p0._prepare(processor, ctx["frames"], ctx["root"], ctx["goal"],
                                                 ctx["history"], device)
        with torch.no_grad():
            encoded = p0._latent_context(model, pixels, history)
            goal_latent = p0._goal_latent(model, goal)
            chart = adapter.goal_chart(encoded, goal_latent, 2027111800)
            np.testing.assert_allclose(chart.cpu().numpy(), data["chart"], rtol=0, atol=2e-5)
            np.testing.assert_allclose((goal_latent @ chart).cpu().numpy(), data["target"], rtol=0, atol=2e-5)
            for batch in data["trace"]:
                raw = np.asarray(batch["raw"], dtype=np.float32)
                requested = np.asarray(batch["requested"])
                np.testing.assert_array_equal(raw, requested.astype(np.float32))
                if raw.ndim != 3 or raw.shape[2] != 2 or raw.shape[1] not in (5, 10, 15) or np.any(np.abs(raw) > 1 + 1e-6):
                    raise ValueError("invalid raw actions")
                tokens += len(raw) * raw.shape[1] // 5
                if batch["tokens"] != len(raw) * raw.shape[1] // 5:
                    raise ValueError("batch token mismatch")
                normalized = adapter.raw_to_packets(raw, mean, std, device=device)
                params, error = inverse_packets(normalized, lower, upper)
                terminal = p0._terminal_batch(model, pixels, goal, history, params,
                                              action_lower=lower, action_upper=upper)
                positions = (terminal @ chart).cpu().numpy()
                scores = (terminal - goal_latent).square().mean(dim=1).cpu().numpy()
                max_pos = max(max_pos, float(np.max(np.abs(positions - batch["positions"]))))
                max_score = max(max_score, float(np.max(np.abs(scores - batch["scores"]))))
                max_action = max(max_action, error)
                count += len(raw)
                if raw.shape[1] == 15:
                    all_full.extend(zip(raw, batch["scores"], strict=True))
        if max_pos > 2e-5 or max_score > 2e-6:
            raise ValueError(f"reference mismatch {max_pos} {max_score}")
        result = data["result"]
        selected = np.asarray(result["selected"]["chain"], np.float32)
        chosen_score = result["selected"]["terminal_score"]
        if not any(np.array_equal(a, selected) and s == chosen_score for a, s in all_full):
            raise ValueError("selected chain not scored")
        pre = result["precomparison"]
        if pre["terminal_score"] != min(s for _, s in all_full[:-1]):
            raise ValueError("precomparison is not global incumbent")
        comparison = np.asarray(result["comparison_chain"], np.float32)
        if (not np.array_equal(all_full[-1][0], comparison)
                or all_full[-1][1] != result["comparison_score"]
                or result["abstained"] != (result["comparison_score"] <= pre["terminal_score"])):
            raise ValueError("abstention mismatch")
        if (tokens != data["counts"]["prediction_token_samples"] or count != data["counts"]["trajectory_samples"]
                or len(data["trace"]) != data["counts"]["prediction_calls"] or tokens > 1920):
            raise ValueError("meter mismatch")
        if setup._parameter_hash(model) != model_hash:
            raise ValueError("model changed")
    seal.verify(manifest["seal"])
    host.write_new(OUT, {"manifest_sha256": sha(run.MANIFEST), "result_sha256": sha(run.RESULT),
                         "audit_script_sha256": sha(Path(__file__)), "model_hash": model_hash,
                         "batches": len(data["trace"]), "samples": count, "tokens": tokens,
                         "max_position_error": max_pos, "max_score_error": max_score,
                         "max_inverse_action_error": max_action, "cuda_allocation": allocation,
                         "scope": "same frozen model and RGB context, independent reference rollout"})
    print(f"Reference recomputed {count} samples in {len(data['trace'])} batches.", flush=True)


if __name__ == "__main__":
    main()
