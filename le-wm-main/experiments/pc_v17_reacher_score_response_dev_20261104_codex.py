"""Frozen Reacher RGB feedback: score-augmented response ablation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments import pc_v13_reacher_rgb_feedback_dev_20261104_codex as old
from experiments import pc_runtime_preflight_v1 as preflight
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_recursive_response_v17 as v17
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = old.ROOT
OUT = ROOT / "results/pc_v17_reacher_score_response_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V17_REACHER_SCORE_RESPONSE_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("v13", "v17", "global_cem")
TASK_SEED = 20263289
CAP = 1920


def decide(arm, step, target, predict, meter, carry):
    horizon = 15 - 5 * step
    nominals = tuple(tuple(tuple(float(x) for x in row) for row in nominal)
                     for nominal in old.temporal.NOMINALS)
    config = old.v13.Config(action_dim=2, segment_length=5, segments=horizon // 5,
        model_packet_length=5, nominals=nominals, token_cap=CAP, seed=2027110401 + step * 10000)
    if arm in ("v13", "v17"):
        if arm == "v13":
            result = old.v13.compile_program(predict, target, config, carry=carry)
        else:
            result = v17.compile_program(predict, target, config, score_scale=1., score_target=0., carry=carry)
        if result["resource"]["prediction_tokens"] != meter.token_samples:
            raise ValueError("compiler meter mismatch")
        return {"selected": result["selected"], "selected_score": result["selected_score"], "compiler": result}
    if arm != "global_cem":
        raise ValueError("unknown arm")
    bank = old.v13.proposal_bank(config, carry)
    _, scores = predict(bank)
    choice = int(np.argmin(scores))
    packets = horizon // 5
    count = (CAP - meter.token_samples - 2 * packets) // packets
    chain, score, generations = old.temporal.cem(bank[choice], [], count,
        np.random.default_rng(2029124000 + step), predict, float(scores[choice]), horizon)
    comparison = np.asarray(chain, np.float32).copy()
    comparison[:5] = 0
    pair = np.stack((chain, comparison)).astype(np.float32)
    _, paired = predict(pair)
    index = int(np.argmin(paired))
    return {"selected": pair[index].tolist(), "selected_score": float(paired[index]),
            "bank_choice": choice, "bank_scores": scores.tolist(), "incumbent": np.asarray(chain).tolist(),
            "incumbent_score": float(score), "pair_scores": paired.tolist(), "pair_choice": index,
            "generations": generations, "samples": count}


def save_context(path, context):
    with path.open("xb") as stream:
        np.savez_compressed(stream, frames=np.stack(context["frames"]), root=context["root_frame"],
                            goal=context["goal"], history=np.asarray(context["history_action_blocks"], np.float32))
        stream.flush()
        os.fsync(stream.fileno())


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    host, setup, base, adapter = old.host, old.setup, old.p0, old.adapter
    sha = host.bridge.sha_file
    source = json.loads(old.run.SOURCE.read_text())
    receipt = json.loads(old.PRE_RESULT.read_text())
    if (receipt["manifest_sha256"] != sha(old.PRE_MANIFEST) or receipt["context_sha256"] != sha(old.PRE_CONTEXT)
            or receipt["model_parameter_hash"] != source["model_parameter_hash_expected"]):
        raise ValueError("adapter preflight seal mismatch")
    paths = [Path(__file__), PROTOCOL, Path(old.__file__), Path(v17.__file__), Path(old.v13.__file__),
             Path(old.v12.__file__), Path(old.temporal.__file__), Path(old.context_lib.__file__),
             Path(preflight.__file__), old.PRE_MANIFEST, old.PRE_CONTEXT, old.PRE_RESULT]
    dependencies = seal.capture(paths)
    dependencies["files"].update(old.run.dependencies(source))
    if (dependencies["files"][str(Path(source["weights_path"]).resolve())] != source["weights_sha256"]
            or dependencies["files"][str(Path(source["action_stats_path"]).resolve())] != source["action_stats_sha256"]):
        raise ValueError("model or normalizer source mismatch")
    OUT.mkdir()
    host.write_new(OUT / "manifest.json", {"dependencies": dependencies, "arms": ARMS,
        "task_seed": TASK_SEED, "goal_delta": [.45, -.50], "cap": CAP,
        "scope": "single development task; full-cost and source independence unknown"})
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = {}
    with host.lock.gpu_lock(host.LOCK):
        import gymnasium as gym
        from experiments.gpu_preflight import require_cuda_allocation

        allocation = require_cuda_allocation("cuda:0")
        if host.canonical_gpu_uuid(torch.cuda.get_device_properties(0).uuid) != host.canonical_gpu_uuid(host.UUID):
            raise ValueError("unexpected GPU")
        device = torch.device("cuda:0")
        setup.install_missing_field_compat()
        stats = Path(source["action_stats_path"])
        setup.validate_training_semantic_receipt(stats)
        mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
        _, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
        model = setup.load_compatible_pretrained(Path(source["weights_path"]).parent, device="cuda").eval()
        model.interpolate_pos_encoding = True
        model.requires_grad_(False)
        model_hash = setup._parameter_hash(model)
        if model_hash != source["model_parameter_hash_expected"]:
            raise ValueError("frozen parameter mismatch")
        processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
        env = setup.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"),
                                            target_delta=(.45, -.50))
        try:
            if (not np.array_equal(env.action_space.low, np.full(2, -1))
                    or not np.array_equal(env.action_space.high, np.full(2, 1))):
                raise ValueError("action box mismatch")
            base._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
            lo, hi = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
            lower, upper = torch.as_tensor(lo, device=device), torch.as_tensor(hi, device=device)
            initial_binding = None
            for arm in ARMS:
                error, events, outcomes = None, [], []
                try:
                    packets, history_index = setup.sample_history_packets(histories, seed=TASK_SEED)
                    context = setup.collect_executed_context(env, seed=TASK_SEED, history_len=3, action_block=5,
                        warmup_packets=packets, warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
                    binding = (int(history_index), context["root_rgb_sha256"], context["goal_rgb_sha256"],
                               tuple(context["history_rgb_sha256"]),
                               np.asarray(context["history_action_blocks"], np.float32).tobytes())
                    if initial_binding is not None and binding != initial_binding:
                        raise ValueError("arms have different initial contexts")
                    initial_binding = binding
                    carry = None
                    outcomes.append(old.physical(env))
                    for step in range(3):
                        context_path = OUT / f"{arm}_context{step}.npz"
                        save_context(context_path, context)
                        start_encode = time.monotonic()
                        pixels, goal, history = base._prepare(processor, context["frames"], context["root_frame"],
                            context["goal"], np.asarray(context["history_action_blocks"], np.float32), device)
                        with torch.no_grad():
                            encoded, goal_latent = base._latent_context(model, pixels, history), base._goal_latent(model, goal)
                            chart = adapter.goal_chart(encoded, goal_latent, 2027111800 + step)
                            chart = chart / np.sqrt(goal_latent.numel())
                            target = (goal_latent @ chart).cpu().numpy()
                            torch.cuda.synchronize()
                        encoding_seconds = time.monotonic() - start_encode
                        meter, trace = Meter(CAP), []

                        def predict(actions):
                            requested = np.asarray(actions, np.float64)
                            raw = requested.astype(np.float32)
                            if (raw.ndim != 3 or raw.shape[1] % 5 or raw.shape[2] != 2
                                    or not np.isfinite(raw).all() or np.max(np.abs(raw)) > 1):
                                raise ValueError("invalid requested action")
                            def evaluate(batch):
                                future = adapter.raw_to_packets(batch, mean, std, device=device)
                                with torch.no_grad():
                                    terminal = adapter.cached_terminal(model, encoded, history, future, lower, upper)
                                    projected = terminal @ chart
                                    scores = (terminal - goal_latent).square().mean(dim=1)
                                    torch.cuda.synchronize()
                                return projected.cpu().numpy(), scores.cpu().numpy()
                            positions, scores = meter.predict(raw, evaluate)
                            trace.append({"requested": requested.tolist(), "raw": raw.tolist(),
                                "positions": positions.tolist(), "scores": scores.tolist()})
                            return positions, scores

                        started = time.monotonic()
                        decision = decide(arm, step, target, predict, meter, carry)
                        selected = np.asarray(decision["selected"], np.float32)
                        if selected.shape != (15 - 5 * step, 2) or meter.token_samples > CAP:
                            raise ValueError("invalid selected chain")
                        decision_path = OUT / f"{arm}_decision{step}.json"
                        host.write_new(decision_path, {"manifest_sha256": sha(OUT / "manifest.json"),
                            "context_sha256": sha(context_path), "carry_in": carry, "target": target.tolist(),
                            "chart": chart.cpu().tolist(), "decision": decision, "trace": trace,
                            "counts": meter.as_dict(), "dispatch": selected[:5].tolist(),
                            "encoding_seconds": encoding_seconds, "planning_seconds": time.monotonic() - started})
                        sealed_ns = time.monotonic_ns()
                        advanced = old.context_lib.advance_executed_context(env, context=context,
                            raw_action_block=selected[:5], action_block=5, verified_root_frame=context["root_frame"])
                        outcomes.append(old.physical(env))
                        events.append({"step": step, "decision_sha256": sha(decision_path),
                            "context_sha256": sha(context_path), "sealed_ns": sealed_ns,
                            "execution_return_ns": time.monotonic_ns(), "dispatch": selected[:5].tolist(),
                            "successor_root_sha256": advanced["root_rgb_sha256"],
                            "advanced_action_sha256": advanced["advanced_action_sha256"]})
                        carry, context = (selected[5:].tolist() if step < 2 else None), advanced
                        print(f"{arm} step={step} tokens={meter.token_samples}", flush=True)
                    save_context(OUT / f"{arm}_context3.npz", context)
                except Exception:
                    error = traceback.format_exc()
                rows[arm] = {"status": "completed" if error is None and len(events) == 3 else "failed",
                    "error": error, "events": events, "physical_evaluator": outcomes,
                    "model_hash": model_hash, "cuda_allocation": allocation}
                host.write_new(OUT / f"{arm}_execution.json", rows[arm])
                print(f"{arm} status={rows[arm]['status']}", flush=True)
            if setup._parameter_hash(model) != model_hash:
                raise ValueError("model changed")
            seal.verify(dependencies)
        finally:
            env.close()
    host.write_new(OUT / "summary.json", {"manifest_sha256": sha(OUT / "manifest.json"),
        "arms": {arm: {"status": row["status"], "error": row["error"],
                       "physical_evaluator": row["physical_evaluator"]} for arm, row in rows.items()},
        "scope": "single-root development; no gate conclusion"})


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    preflight.run_before_artifacts([lambda: preflight.cuda_probe(old.host.UUID),
                                    preflight.reacher_render_probe], run)


if __name__ == "__main__":
    main()
