"""Goal-Q-consistent version of fixed all-prefix cp5 feedback development."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch

from experiments.spectral_reachability import reacher_response_svd_cp5_cem_20261102 as base
from experiments.spectral_reachability import reacher_pc_p1_fd_cp5_cem_bridge_20261102_codexexclusive as bridge
from experiments.spectral_reachability import reacher_pc_p1_common_tail_preflight_20261102_codexexclusive as preflight
from experiments.spectral_reachability import reacher_pc_p1_common_tail_tangent_20261102_codexexclusive as common
from experiments.spectral_reachability.reacher_d2p_goal_encoder_metric import GoalEncoderMetric

ORIGINAL_PLAN = base.make_plan
PROTOCOL = Path(__file__).resolve().parents[3] / "docs/PC_REACHER_RESPONSE_SVD_QCP5_PROTOCOL_20261102.md"
AUTHORITY = "RESPONSE_SVD_ALL_PREFIX_QCP5_CEM_DEVELOPMENT"


def cem_search(evaluate, *, dimension, device, future_horizon, samples, steps, elite, seed):
    """Same sampling/update as original CEM; evaluation is an injected cost."""
    if not (dimension > 0 and future_horizon > 0 and 0 < elite < samples and steps > 0):
        raise ValueError("invalid CEM parameters")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    mean = torch.zeros((future_horizon, dimension), device=device)
    std = torch.ones_like(mean)
    trace = []
    for _ in range(steps):
        candidates = torch.randn((samples, future_horizon, dimension), generator=generator, device=device)
        candidates = candidates * std.unsqueeze(0) + mean.unsqueeze(0)
        candidates[0] = mean
        costs = evaluate(candidates)
        if costs.shape != (samples,) or not torch.isfinite(costs).all():
            raise ValueError("invalid Q CEM scores")
        values, indices = torch.topk(costs, k=elite, largest=False)
        chosen = candidates[indices]
        mean = chosen.mean(dim=0)
        std = chosen.std(dim=0).clamp_min(1.0e-4)
        trace.append(float(values.mean()))
    best = int(torch.argmin(costs))
    return candidates[best:best + 1].detach(), {
        "trace": trace, "forward_rollout_evaluations": int(samples * steps),
        "backward_vjp_rows": 0, "final_selection": "best_sample",
        "objective": "authenticated_goal_q_weighted_planning_coordinate",
    }


def make_plan(*inputs):
    plan = ORIGINAL_PLAN(*inputs)
    q_path = preflight.Q_PATH.resolve(strict=True)
    q = json.loads(q_path.read_text())
    if [int(row["seed"]) for row in q["rows"]] != list(base.source.ROOTS):
        raise ValueError("unexpected goal metric root roster")
    saved_preflight = json.loads(Path(inputs[2]).read_text())
    if bridge.sha_file(q_path) != saved_preflight["q_report_sha256"]:
        raise ValueError("goal Q source drift")
    plan["schema"] = "response_svd_all_prefix_qcp5_cem_v1"
    plan["files"].update({str(Path(__file__).resolve()): bridge.sha_file(__file__),
                          str(PROTOCOL): bridge.sha_file(PROTOCOL),
                          str(q_path): bridge.sha_file(q_path),
                          str(Path(inspect.getfile(GoalEncoderMetric)).resolve()):
                              bridge.sha_file(inspect.getfile(GoalEncoderMetric))})
    plan["q_report"] = str(q_path)
    plan["scope"] = "two exposed roots; all fixed prefixes; goal-Q CEM; P1 admission false"
    return plan


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    for key in ("source-report", "physical-plan", "preflight", "output"):
        prep.add_argument("--" + key, required=True)
    run = sub.add_parser("run")
    for key in ("plan", "sha256", "output", "authority"):
        run.add_argument("--" + key, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        base.host.write_new(args.output, make_plan(args.source_report, args.physical_plan, args.preflight))
        print(json.dumps({"plan": args.output, "sha256": bridge.sha_file(args.output)}))
        return
    base.host.require(args.authority == AUTHORITY, "Q-CEM authority")
    from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0

    plan = json.loads(Path(args.plan).read_text())
    q_by_seed = {int(row["seed"]): row["goal_metric"]["gram"]
                 for row in json.loads(Path(plan["q_report"]).read_text())["rows"]}
    previous_cem, previous_plan = p0._cem, base.make_plan

    def q_cem(model, pixels, goal, history, *, future_horizon, action_lower, action_upper,
              samples, steps, elite, seed):
        root = int(seed) - 5000
        base.host.require(root in q_by_seed, "unknown metric root")
        metric = GoalEncoderMetric(torch.as_tensor(q_by_seed[root], dtype=torch.float64, device=history.device))
        target = p0._planning_coordinate(model, p0._goal_latent(model, goal), role="goal").detach()

        def evaluate(candidates):
            terminal = p0._terminal_batch(model, pixels, goal, history, candidates,
                                          action_lower=action_lower, action_upper=action_upper)
            coordinates = p0._planning_coordinate(model, terminal, role="predicted")
            return common.q_costs(metric, coordinates, target)

        return cem_search(evaluate, dimension=history.shape[-1], device=history.device,
                          future_horizon=future_horizon, samples=samples, steps=steps, elite=elite, seed=seed)

    try:
        p0._cem = q_cem
        base.make_plan = make_plan
        print(json.dumps(base.run(args.plan, args.sha256, args.output, base.AUTHORITY)))
    finally:
        p0._cem = previous_cem
        base.make_plan = previous_plan


if __name__ == "__main__":
    main()
