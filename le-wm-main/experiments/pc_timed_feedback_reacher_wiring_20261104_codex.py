"""Real Reacher RGB integration of the common timed feedback executor."""
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments import pc_v17_reacher_score_response_dev_20261104_codex as existing
from experiments import pc_runtime_preflight_v1 as preflight
from experiments.spectral_reachability import pc_timed_feedback_execution as feedback
from experiments.spectral_reachability import pc_timed_program_execution as timed
from experiments.spectral_reachability import pc_global_multistart_cem as cem
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

old = existing.old
ROOT = existing.ROOT
OUT = ROOT / "results/pc_timed_feedback_reacher_wiring_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_TIMED_FEEDBACK_REACHER_WIRING_PROTOCOL_20261104_CODEX.md"
TASK_SEED = 20263371
ARMS = ("v17", "multistart_cem")


def run():
    host, setup, base, adapter = old.host, old.setup, old.p0, old.adapter
    sha = host.bridge.sha_file
    source = json.loads(old.run.SOURCE.read_text())
    receipt = json.loads(old.PRE_RESULT.read_text())
    if (receipt["manifest_sha256"] != sha(old.PRE_MANIFEST)
            or receipt["context_sha256"] != sha(old.PRE_CONTEXT)
            or receipt["model_parameter_hash"] != source["model_parameter_hash_expected"]):
        raise ValueError("adapter preflight seal mismatch")
    deps = seal.capture([Path(__file__), PROTOCOL, Path(feedback.__file__), Path(timed.__file__),
                         Path(existing.__file__), Path(old.__file__), Path(existing.v17.__file__),
                         Path(old.v13.__file__), Path(old.v12.__file__), Path(cem.__file__),
                         Path(old.temporal.__file__), Path(old.context_lib.__file__),
                         Path(preflight.__file__), old.PRE_MANIFEST, old.PRE_CONTEXT, old.PRE_RESULT])
    deps["files"].update(old.run.dependencies(source))
    if (deps["files"][str(Path(source["weights_path"]).resolve())] != source["weights_sha256"]
            or deps["files"][str(Path(source["action_stats_path"]).resolve())] != source["action_stats_sha256"]):
        raise ValueError("model or normalizer source mismatch")
    OUT.mkdir()
    host.write_new(OUT / "manifest.json", {"dependencies": deps, "arms": ARMS, "task_seed": TASK_SEED,
        "goal_delta": [.45, -.50], "cap": 1920, "scope": "one-task integration; provenance unknown"})
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
            raise ValueError("model hash mismatch")
        processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
        env = setup.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"),
                                            target_delta=(.45, -.50))
        try:
            if not (np.array_equal(env.action_space.low, np.full(2, -1))
                    and np.array_equal(env.action_space.high, np.full(2, 1))):
                raise ValueError("action box mismatch")
            base._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
            lo, hi = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
            lower, upper = torch.as_tensor(lo, device=device), torch.as_tensor(hi, device=device)
            nominals = tuple(tuple(tuple(float(x) for x in point) for point in nominal)
                             for nominal in old.temporal.NOMINALS)
            config = old.v13.Config(action_dim=2, segment_length=5, segments=3, model_packet_length=5,
                                    nominals=nominals, token_cap=1920, seed=2029321000)
            initial_binding = None
            for arm in ARMS:
                events, physical, error = [], [], None
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
                    physical.append(old.physical(env))

                    def prepare(current, step):
                        context_path = OUT / f"{arm}_context{step}.npz"
                        existing.save_context(context_path, current)
                        torch.cuda.synchronize()
                        started = time.monotonic()
                        pixels, goal, history = base._prepare(processor, current["frames"], current["root_frame"],
                            current["goal"], np.asarray(current["history_action_blocks"], np.float32), device)
                        with torch.no_grad():
                            encoded = base._latent_context(model, pixels, history)
                            goal_latent = base._goal_latent(model, goal)
                            chart = adapter.goal_chart(encoded, goal_latent, 2029322000 + step)
                            chart = chart / np.sqrt(goal_latent.numel())
                            target = (goal_latent @ chart).cpu().numpy()
                            torch.cuda.synchronize()
                        encoding_seconds = time.monotonic() - started

                        def predict(actions):
                            future = adapter.raw_to_packets(actions, mean, std, device=device)
                            with torch.no_grad():
                                terminal = adapter.cached_terminal(model, encoded, history, future, lower, upper)
                                projected = terminal @ chart
                                scores = (terminal-goal_latent).square().mean(dim=1)
                                return projected.cpu().numpy(), scores.cpu().numpy()
                        return predict, target, {"context_sha256": sha(context_path),
                            "encoding_seconds": encoding_seconds, "chart": chart.cpu().tolist(),
                            "target": target.tolist()}

                    def seal_decision(decision):
                        step = decision["step"]
                        traces = decision.pop("traces")
                        trace_path = OUT / f"{arm}_trace{step}.npz"
                        with trace_path.open("xb") as stream:
                            np.savez_compressed(stream, **{f"{name}_{i}": batch[name]
                                for i, batch in enumerate(traces) for name in ("actions", "features", "scores")})
                            stream.flush()
                            os.fsync(stream.fileno())
                        decision["trace_sha256"] = sha(trace_path)
                        decision["trace"] = trace_path.name
                        decision["manifest_sha256"] = sha(OUT / "manifest.json")
                        decision_path = OUT / f"{arm}_decision{step}.json"
                        host.write_new(decision_path, decision)
                        return {"decision_sha256": sha(decision_path), "sealed_ns": time.monotonic_ns()}

                    def dispatch(current, actions, sealed):
                        advanced = old.context_lib.advance_executed_context(env, context=current,
                            raw_action_block=actions, action_block=5, verified_root_frame=current["root_frame"])
                        physical.append(old.physical(env))
                        event = {"receipt": sealed, "execution_return_ns": time.monotonic_ns(),
                                 "dispatch": actions.tolist(), "successor_root_sha256": advanced["root_rgb_sha256"],
                                 "advanced_action_sha256": advanced["advanced_action_sha256"]}
                        events.append(event)
                        return advanced, event

                    done = feedback.run(context, config, arm=arm, execution_steps=5, prepare=prepare,
                                        seal_decision=seal_decision, dispatch=dispatch,
                                        baseline_seed=2029323000, synchronize=torch.cuda.synchronize)
                    existing.save_context(OUT / f"{arm}_context3.npz", done["final_context"])
                except Exception:
                    error = traceback.format_exc()
                rows[arm] = {"status": "completed" if error is None and len(events) == 3 else "failed",
                             "error": error, "events": events, "physical_evaluator": physical,
                             "model_hash": model_hash, "cuda_allocation": allocation}
                host.write_new(OUT / f"{arm}_execution.json", rows[arm])
                print(f"{arm}: {rows[arm]['status']}", flush=True)
            if setup._parameter_hash(model) != model_hash:
                raise ValueError("model changed")
            seal.verify(deps)
        finally:
            env.close()
    host.write_new(OUT / "summary.json", {"arms": rows, "manifest_sha256": sha(OUT / "manifest.json"),
                                         "scope": "wiring diagnostic; no gate conclusion"})


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    preflight.run_before_artifacts([lambda: preflight.cuda_probe(old.host.UUID), preflight.reacher_render_probe], run)


if __name__ == "__main__":
    main()
