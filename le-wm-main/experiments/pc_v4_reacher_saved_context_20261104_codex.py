"""V4 planner on an existing real RGB Reacher context, no physical dispatch."""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_response_temporal_v3_reacher_rgb_live_20261104_codex as live
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import d2p_local_compiler as local
from experiments.spectral_reachability import pc_target_reach_feedback_v4 as method
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = live.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V4_REACHER_SAVED_CONTEXT_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v4_reacher_saved_context_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
RESULT = PREFIX.with_name(PREFIX.name + "_result.json")
CONTEXT = live.path("response", "context0.npz")
DECISION = live.path("response", "decision0.json")
AUDIT = live.PREFIX.with_name(live.PREFIX.name + "_audit.json")
REPLAY = live.PREFIX.with_name(live.PREFIX.name + "_prediction_replay.json")


def main():
    if MANIFEST.exists() or RESULT.exists():
        raise FileExistsError("existing or partial result")
    host, setup, p0, adapter = live.host, live.setup, live.p0, live.adapter
    if os.environ.get("CUDA_VISIBLE_DEVICES") != host.UUID:
        raise ValueError("cooperative GPU UUID required")
    sha = host.bridge.sha_file
    old_manifest = json.loads(live.MANIFEST.read_text())
    old_decision = json.loads(DECISION.read_text())
    old_audit = json.loads(AUDIT.read_text())
    old_replay = json.loads(REPLAY.read_text())
    source = json.loads(live.run.SOURCE.read_text())
    if (old_decision["manifest_sha256"] != sha(live.MANIFEST)
            or old_decision["context_sha256"] != sha(CONTEXT)
            or old_audit["manifest_sha256"] != sha(live.MANIFEST)
            or old_replay["manifest_sha256"] != sha(live.MANIFEST)
            or old_replay["model_parameter_hash"] != source["model_parameter_hash_expected"]):
        raise ValueError("historical input binding mismatch")
    dependencies = seal.capture([Path(__file__), PROTOCOL, CONTEXT, DECISION, AUDIT, REPLAY,
                                 live.MANIFEST, live.run.SOURCE, Path(method.__file__), Path(local.__file__),
                                 Path(adapter.__file__)])
    host.write_new(MANIFEST, {"method": method.METHOD, "source_manifest_sha256": sha(live.MANIFEST),
                              "source_model_parameter_hash": source["model_parameter_hash_expected"],
                              "seal": dependencies, "historical_dependency_closure": False,
                              "scope": "saved real RGB, exposed root, no physical execution"})
    print("V4 on one saved real RGB context; model scores only, no action dispatch.", flush=True)
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
        if setup._parameter_hash(model) != source["model_parameter_hash_expected"]:
            raise ValueError("frozen model hash mismatch")
        processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
        lower, upper = setup._normalized_action_bounds(mean, std, np.full(2, -1), np.full(2, 1))
        lower, upper = torch.as_tensor(lower, device=device), torch.as_tensor(upper, device=device)
        with np.load(CONTEXT, allow_pickle=False) as ctx:
            pixels, goal, history = p0._prepare(processor, ctx["frames"], ctx["root"], ctx["goal"],
                                                 ctx["history"], device)
        with torch.no_grad():
            encoded = p0._latent_context(model, pixels, history)
            goal_latent = p0._goal_latent(model, goal)
            chart = adapter.goal_chart(encoded, goal_latent, 2027111800)
            target = (goal_latent @ chart).cpu().numpy()
        np.testing.assert_allclose(chart.cpu().numpy(), old_decision["chart"], rtol=0, atol=2e-5)
        np.testing.assert_allclose(target, old_decision["target"], rtol=0, atol=2e-5)
        meter, trace = Meter(1920), []

        def predict(actions):
            requested = np.asarray(actions, dtype=float)
            raw = requested.astype(np.float32)

            def evaluate(batch):
                future = adapter.raw_to_packets(batch, mean, std, device=device)
                with torch.no_grad():
                    terminal = adapter.cached_terminal(model, encoded, history, future, lower, upper)
                    positions = terminal @ chart
                    scores = (terminal - goal_latent).square().mean(dim=1)
                    torch.cuda.synchronize()
                return positions.cpu().numpy(), scores.cpu().numpy()

            positions, scores = meter.predict(raw, evaluate)
            trace.append({"requested": requested.tolist(), "raw": raw.tolist(), "positions": positions.tolist(),
                          "scores": scores.tolist(), "tokens": len(raw) * raw.shape[1] // 5})
            return positions, scores

        started = time.monotonic()
        result = method.decide("response", 0, 0, target, predict, meter)
        selected = np.asarray(result["selected"]["chain"], dtype=np.float32)
        if selected.shape != (15, 2) or np.any(np.abs(selected) > 1) or meter.token_samples > 1920:
            raise ValueError("invalid v4 output")
        if setup._parameter_hash(model) != source["model_parameter_hash_expected"]:
            raise ValueError("model changed")
        seal.verify(dependencies)
        host.write_new(RESULT, {"manifest_sha256": sha(MANIFEST), "context_sha256": sha(CONTEXT),
                                "chart": chart.cpu().tolist(), "target": target.tolist(), "result": result,
                                "trace": trace, "counts": meter.as_dict(), "planning_seconds": time.monotonic() - started,
                                "cuda_allocation": allocation, "physical_execution": False,
                                "historical_manifest_method": old_manifest["method"]})
    print(f"V4 completed {meter.calls} batches / {meter.token_samples} tokens.", flush=True)


if __name__ == "__main__":
    main()
