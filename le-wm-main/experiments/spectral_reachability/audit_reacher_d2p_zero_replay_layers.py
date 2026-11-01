"""Host-CUDA diagnostic separating physical, RGB and encoder replay failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.spectral_reachability import audit_reacher_d2p_prefix_capability as core


def compare(left, right):
    np = core.np
    a, b = np.asarray(left["observed"]), np.asarray(right["observed"])
    return {"full_state_exact": left["snapshot"] == right["snapshot"],
            "qpos_exact": left["snapshot"]["qpos"] == right["snapshot"]["qpos"],
            "rgb_exact": left["rgb_sha256"] == right["rgb_sha256"],
            "history_rgb_exact": left["history_rgb_sha256"] == right["history_rgb_sha256"],
            "observed_exact": left["observed"] == right["observed"],
            "observed_max_error": float(np.max(np.abs(a-b)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 2:
        raise ValueError("new immutable output and at least two repeats required")
    core.verify_files(args.reference)
    ref = json.loads(args.reference.read_text())
    predictions_path = args.reference.with_suffix(".predictions.json")
    plans = {r["seed"]: r for r in json.loads(predictions_path.read_text())["rows"]}
    old = plans[args.seed]
    shared, execution, np, torch = core.shared, core.execution, core.np, core.torch
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if (source_sha != ref["sealed_protocol"]["source_sha256_receipt"]
            or core.digest(args.policy/"weights.pt") != ref["sealed_protocol"]["weights_sha256"]
            or core.digest(args.adapter_artifact) != ref["sealed_protocol"]["adapter_sha256"]):
        raise ValueError("exact original source, weights and action statistics required")
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    model = shared.load_compatible_pretrained(args.policy, device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    parameters = shared._parameter_hash(model)
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    env = core.goals.goal_wrapper(ref["sealed_protocol"]["goal_contract"])(
        core.gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(.65, -.65))
    records = []
    try:
        ctx = shared._source_state_context(env, dataset=dataset, seed=args.seed)
        ctx["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(ctx["root_frame"])
        if ctx["execution_snapshot"] != old["initial_snapshot"]:
            raise ValueError("exact original failing-root snapshot required")
        for task in core.chart.TASKS:
            for bank in old["tasks"][task]["bank"]:
                index = bank["index"]
                raw = np.asarray(bank["variants"]["cem_then_gn4"]["raw_actions"], np.float32)[:5]
                frame, goal = execution.restore_execution_snapshot(env, seed=args.seed,
                    snapshot=ctx["execution_snapshot"], root_identity="sealed_observation_exact_state")
                core.p0._verify_execution_identity(frame, goal, ctx["root_frame"], ctx["goal"])
                current = execution.advance_executed_context(env, context=ctx, raw_action_block=raw,
                    action_block=5, verified_root_frame=ctx["root_frame"])
                current["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(current["root_frame"])
                attempts = []
                for repeat in range(args.repeats):
                    frame, goal = execution.restore_execution_snapshot(env, seed=args.seed,
                        snapshot=current["execution_snapshot"], root_identity="sealed_observation_exact_state")
                    core.p0._verify_execution_identity(frame, goal, current["root_frame"], current["goal"])
                    advanced = current
                    for _ in range(2):
                        advanced = execution.advance_executed_context(env, context=advanced,
                            raw_action_block=np.zeros((5, 2), np.float32), action_block=5,
                            verified_root_frame=advanced["root_frame"])
                    pixels, _, history = core.p0._prepare(processor, advanced["frames"], advanced["root_frame"],
                        advanced["goal"], np.asarray(advanced["history_action_blocks"], np.float32), torch.device("cuda"))
                    if pixels.device.type != "cuda" or history.device.type != "cuda":
                        raise RuntimeError("CPU neural fallback forbidden")
                    with torch.no_grad():
                        observed = core.p0._latent_context(model, pixels, history)[0, -1].tolist()
                        duplicate = core.p0._latent_context(model, pixels, history)[0, -1].tolist()
                    attempts.append({"repeat": repeat, "snapshot": advanced["execution_snapshot"],
                        "rgb_sha256": advanced["root_rgb_sha256"],
                        "history_rgb_sha256": advanced["history_rgb_sha256"],
                        "observed": observed, "same_input_encoder_exact": duplicate == observed})
                comparisons = [compare(attempts[0], x) for x in attempts[1:]]
                record = {"task": task, "index": index, "attempts": attempts, "comparisons": comparisons}
                records.append(record)
                print(json.dumps({"seed": args.seed, "task": task, "index": index,
                    "comparisons": comparisons, "same_input_encoder_exact": all(x["same_input_encoder_exact"] for x in attempts)}), flush=True)
    finally:
        env.close()
    frozen = (shared._parameter_hash(model) == parameters and not any(m.training for m in model.modules())
              and all(torch.equal(v, buffers[k]) for k, v in model.named_buffers()))
    if not frozen:
        raise RuntimeError("official model changed during replay diagnosis")
    core.write_new(args.output, {"schema": "reacher_d2p_zero_replay_layers.v1", "cuda_host": host,
        "seed": args.seed, "repeats": args.repeats, "records": records,
        "bindings": {str(p.resolve()): core.digest(p) for p in (args.reference, predictions_path)},
        "implementation_sha256": core.digest(Path(__file__)), "model_unchanged": frozen,
        "P1_admission": False, "scope": "Failure-layer diagnosis, no relaxed replay acceptance or efficacy evidence."})


if __name__ == "__main__":
    main()
