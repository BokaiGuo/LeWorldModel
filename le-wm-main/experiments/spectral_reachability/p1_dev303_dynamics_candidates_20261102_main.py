"""Development-only Jacobian prefixes with paired nonlinear suffix scoring.

Reads only the sealed pre-action RGB/action task, never physical outcomes.
The RGB motion guard is a development approximation, not a safety certificate.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from . import p1_dev303_v3_cuda_executor_20261102_exclusive as source
from . import p1_dev303_rgb_boundary_static_20261102_exclusive as blind
from . import p1_rgb_boundary_contact_witness_v3_20261102_exclusive as geometry
from . import p1_dynamics_prefix_kernel_20261102_main as kernel

ROOT = source.ROOT
AUTHORITY = "DEV303_DYNAMICS_PREFIX_DEVELOPMENT_CUDA"


def guard(actions, current):
    config = blind.CONFIG
    points, _ = geometry._component(current, geometry.OBJECT, config["min_component_pixels"])
    _, pos = geometry._component(current, geometry.PUSHER, config["min_component_pixels"], pusher=True)
    prev = np.zeros(2)
    gap = config["radius_pixels"] + config["clearance_pixels"]
    for action in actions:
        nxt = pos + config["current_action_pixels"] * action + config["previous_action_pixels"] * prev
        if not geometry._inside(nxt, config["radius_pixels"], current.shape[1], current.shape[0]):
            return False
        if geometry._segment_gap(pos, nxt, points) < gap - 1e-5:
            return False
        pos, prev = nxt, action
    return True


def run(output, authority):
    source.need(authority == AUTHORITY and not os.path.lexists(output), "authority/exclusive output")
    prepared = source.read_prepared()
    source.verify_bindings(prepared["model_source_bindings_declared_not_reverified"])
    source.verify_bindings(prepared["producer_sources"])
    source.need(source.sha_file(blind.NPZ) == blind.NPZ_SHA, "blind RGB/action source")
    task_path = Path(prepared["task"]["task_export"])
    source.need(source.sha_file(task_path) == prepared["task"]["task_sha256"], "task identity")
    with gzip.open(task_path, "rt") as stream:
        task = json.load(stream)["task"]
    with np.load(blind.NPZ, allow_pickle=False) as archive:
        source.need(set(archive.files) == {"rgb_history", "raw_action_history", "goal_rgb"},
                    "blind input whitelist")
        current, goal_rgb = archive["rgb_history"][-1].copy(), archive["goal_rgb"].copy()
        source.need(not archive["raw_action_history"].any(), "zero prior actions for geometry guard")
    from . import p1_dev303_goal_tangent_builder_20261102_exclusive as metric_builder

    output = Path(output)
    output.mkdir()
    sources = {str(Path(p).resolve()): source.sha_file(p) for p in
               (__file__, kernel.__file__, metric_builder.__file__, geometry.__file__, blind.__file__,
                metric_builder.tangent.__file__, metric_builder.visibility.__file__)}
    protocol = {
        "schema": "dev303_dynamics_candidates.protocol.v1", "source_sha256": sources,
        "input_sha256": source.PREP_SHA, "blind_sha256": blind.NPZ_SHA,
        "task_sha256": prepared["task"]["task_sha256"],
        "prefix_anchors": [row["actions_sha256"] for row in prepared["candidates"][:4]],
        "prefix_spectral_rank": 1, "prefix_radius": .25,
        "suffix_fd_step": .05, "suffix_scales": [0, .25, .5, 1], "ridge": .01,
        "metrics": ["latent", "goal_tangent"], "selection": "minimum nonlinear terminal cost",
        "no_forced_non_greedy_selection": True, "physical_outcomes_read": False,
        "P1_admission": False, "independent_confirmation": False,
        "classification": "post_development_model_only",
    }
    def write(name, value):
        with (output / name).open("x") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")

    write("protocol.json", protocol)
    counters, rows, handles = {}, [], []
    started = time.monotonic()
    journal = (output / "events.jsonl").open("x")
    def event(kind, **values):
        journal.write(json.dumps({"kind": kind, "elapsed": time.monotonic() - started, **values},
                                 sort_keys=True, allow_nan=False) + "\n")
        journal.flush()

    def prehook(name):
        def hook(_module, args):
            counters[name + "_attempted"] = counters.get(name + "_attempted", 0) + 1
            if name == "predictor":
                counters["predict_samples"] = counters.get("predict_samples", 0) + int(args[0].shape[0])
            event("dispatch", module=name, counts=dict(counters))
        return hook
    def posthook(name):
        def hook(_module, _args, _output):
            counters[name + "_returned"] = counters.get(name + "_returned", 0) + 1
        return hook
    try:
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from .codex_p1_blind_h2_host_cuda_20261030_v1 import model_hash

        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(prepared["model_policy"], device=device).eval().requires_grad_(False)
        before = model_hash(model)
        source.need(before["original_state_dict"] == prepared["model_sha256_expected_from_prepare"],
                    "frozen model state")
        for name, module in (("encoder", model.encoder), ("projector", model.projector),
                             ("action_encoder", model.action_encoder), ("predictor", model.predictor)):
            handles += [module.register_forward_pre_hook(prehook(name)),
                        module.register_forward_hook(posthook(name))]
        tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        z0 = tensor(task["root_context"]["latent_history"])[None]
        past0 = tensor(task["root_context"]["action_history"])[None]
        mean, std = tensor(task["action_mean"]), tensor(task["action_std"])
        source.need(bool((std > 0).all()), "positive action scales")
        goal = np.asarray(task["goal"], np.float64)
        with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
            q, metadata = metric_builder.build(model, goal_rgb, device)
            source.need(np.allclose(metadata["goal_latent"], goal, rtol=1e-5, atol=1e-6),
                        "reencoded original goal matches sealed target")
            metrics = {"latent": np.eye(len(goal)), "goal_tangent": np.asarray(q)}

            def packet(z, past, raw):
                token = ((raw - mean) / std).reshape(len(raw), 1, 10)
                history = torch.cat((past, token), dim=1)
                pred = model.predict(z, model.action_encoder(history))
                source.need(pred.shape == z.shape and bool(torch.isfinite(pred).all()), "prediction shape")
                return torch.cat((z, pred[:, -1:]), dim=1)[:, -3:], history[:, -2:]

            def rollout(prefix, tails):
                with torch.no_grad():
                    batch = len(tails)
                    z, past = packet(z0.repeat(batch, 1, 1), past0.repeat(batch, 1, 1),
                                     tensor(prefix)[None].repeat(batch, 1, 1))
                    prefix_latent = z[0, -1].cpu().numpy()
                    for block in range(2):
                        z, past = packet(z, past, tensor(tails[:, block * 5:(block + 1) * 5]))
                    return prefix_latent, z[:, -1].cpu().numpy()

            bank, rejections, spectra = [], [], []
            for anchor_index, anchor_row in enumerate(prepared["candidates"][:4]):
                anchor = np.asarray(anchor_row["actions"], np.float32)
                action = tensor(anchor).requires_grad_(True)
                z, _ = packet(z0, past0, action[None])
                vector = z[0, -1]
                derivatives = []
                for i in range(len(goal)):
                    counters["vjp_attempted"] = counters.get("vjp_attempted", 0) + 1
                    gradient = torch.autograd.grad(vector[i], action, retain_graph=i < len(goal) - 1)[0]
                    counters["vjp_returned"] = counters.get("vjp_returned", 0) + 1
                    derivatives.append(gradient.detach().flatten())
                jac = torch.stack(derivatives).cpu().numpy()
                pair, spectral = kernel.spectral_pair(anchor, jac, metrics["goal_tangent"])
                counters["svd_calls"] = counters.get("svd_calls", 0) + 1
                spectra.append({"anchor": anchor_index, "jacobian": jac.tolist(), **spectral})
                bank.append({"label": f"geometry_{anchor_index}", "actions": anchor.tolist()})
                for sign, candidate in zip((-1, 1), pair):
                    label = f"spectral_{anchor_index}_{sign:+d}"
                    if guard(candidate, current):
                        bank.append({"label": label, "actions": candidate.tolist()})
                    else:
                        rejections.append({"label": label, "actions": candidate.tolist(),
                                           "reason": "unchanged_rgb_motion_clearance_guard"})
                event("anchor_compiled", anchor=anchor_index, accepted=len(bank), rejected=len(rejections))
            bank += [{"label": item["label"], "actions": item["actions"]}
                     for item in prepared["candidates"][4:]]
            for index, candidate in enumerate(bank):
                prefix = np.asarray(candidate["actions"], np.float32)
                # Symmetric finite-amplitude response at zero suffix; no physical labels.
                eps = .05
                fd = np.concatenate((np.zeros((1, 20)), eps * np.eye(20), -eps * np.eye(20)))
                prefix_latent, ends = rollout(prefix, fd.reshape(41, 10, 2).astype(np.float32))
                response = ((ends[1:21] - ends[21:]) / (2 * eps)).T
                result = {"index": index, "label": candidate["label"], "prefix": candidate["actions"],
                          "prefix_sha256": hashlib.sha256(prefix.astype("<f4").tobytes()).hexdigest(),
                          "prefix_latent": prefix_latent.tolist(), "metrics": {}}
                for name, metric in metrics.items():
                    tails = kernel.suffix_bank(response, goal - ends[0], metric)
                    counters["svd_calls"] += 1
                    _, terminals = rollout(prefix, tails)
                    scores = kernel.costs(terminals, goal, metric)
                    chosen = kernel.select(scores)
                    result["metrics"][name] = {
                        "prefix_cost": float(kernel.costs(prefix_latent[None], goal, metric)[0]),
                        "scores": scores.tolist(), "chosen_scale_index": chosen,
                        "score": float(scores[chosen]), "tail": tails[chosen].tolist(),
                        "terminal": terminals[chosen].tolist(),
                    }
                rows.append(result)
                write(f"candidate_{index:02d}.json", result)
                event("candidate_complete", index=index, label=candidate["label"])
        torch.cuda.synchronize()
        after = model_hash(model)
        source.need(before == after, "model mutation")
        source.need(all(source.sha_file(p) == digest for p, digest in sources.items()), "source drift")
        selections = {name: kernel.select([row["metrics"][name]["score"] for row in rows])
                      for name in metrics}
        write("report.json", {
            "schema": "dev303_dynamics_candidates.report.v1", "status": "model_only_complete",
            "protocol_sha256": source.sha_file(output / "protocol.json"), "host": host,
            "model_before": before, "model_after": after, "metric": metadata,
            "spectra": spectra, "rejections": rejections, "rows": rows, "selections": selections,
            "counts": counters, "elapsed_seconds": time.monotonic() - started,
            "P1_admission": False, "independent_confirmation": False, "full_cost_certified": False})
        return selections
    except Exception as exc:
        event("failed", type=type(exc).__name__, error=str(exc))
        write("failure.json", {"error": str(exc), "type": type(exc).__name__, "counts": counters,
                               "completed_candidates": len(rows), "P1_admission": False})
        raise
    finally:
        for handle in handles:
            handle.remove()
        journal.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--authority", required=True)
    args = parser.parse_args()
    from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking
    with locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        print(json.dumps(run(args.output, args.authority)))


if __name__ == "__main__":
    main()
