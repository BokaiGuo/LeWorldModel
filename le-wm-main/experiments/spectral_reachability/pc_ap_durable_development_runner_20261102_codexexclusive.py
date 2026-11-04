"""AP development arm orchestration over a durable context adapter.

No environment creation, reset, model loading, lock acquisition, or CLI.
The host must supply authenticated context, planner and a cooperative lock.
This file provides a testable integration seam, not an authorized host run.
"""
from __future__ import annotations

import math
import numpy as np

from . import pc_ap_durable_context_adapter_20261102_codexexclusive as durable

SEEDS = (20263160, 20263161)
ARMS = ("ap_observed_target", "final_goal")


class EvaluatorEnv:
    """Keep privileged per-step distance out of planner input."""
    def __init__(self, env, journal, distance):
        self._env = env
        self._journal = journal
        self._distance = distance
        self._decision = None
        self._step = 0

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def unwrapped(self):
        return self._env.unwrapped

    @property
    def _goal(self):
        return self._env._goal

    def render(self):
        return self._env.render()

    def step(self, action):
        value = self._env.step(action)
        # A real step has returned. If evaluation/persistence fails, the
        # bridge must not advance and its intent remains durable.
        self._step += 1
        distance = float(self._distance(self._env))
        if not math.isfinite(distance):
            raise ValueError("nonfinite evaluator distance")
        self._journal.emit({"kind": "evaluator_distance", "decision": self._decision,
                            "step": self._step, "qpos_distance": distance})
        return value


def planner_view(context):
    """No simulator snapshots, qpos, evaluator distance, or environment."""
    return {
        "frames": [np.asarray(x).copy() for x in context["frames"]],
        "root_frame": np.asarray(context["root_frame"]).copy(),
        "goal": np.asarray(context["goal"]).copy(),
        "history_action_blocks": [np.asarray(x).copy() for x in context["history_action_blocks"]],
        "root_rgb_sha256": context["root_rgb_sha256"],
        "goal_rgb_sha256": context["goal_rgb_sha256"],
    }


def run_arm(env, context, planner, journal, *, seed, arm, distance):
    """Run fixed two decisions; planner(view, seed, arm, index) -> (actions, audit).

    Budget audit must be measured by the host predictor, not inferred from
    returned metadata. The adapter persists action and observation handoff.
    """
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("not an existing fixed development seed/arm")
    adapter = durable.Adapter(journal)
    wrapped = EvaluatorEnv(env, journal, distance)
    journal.emit({"kind": "arm_started", "seed": seed, "arm": arm,
                  "decisions": 2, "action_block": 5,
                  "population": 16, "iterations": 2, "elites": 4})
    for index in range(2):
        wrapped._decision, wrapped._step = index, 0
        view = planner_view(context)
        journal.emit({"kind": "planning_intent", "decision": index,
                      "seed": seed + 8100 + index, "root_rgb_sha256": view["root_rgb_sha256"]})
        try:
            actions, audit = planner(view, seed, arm, index)
            if (not isinstance(audit, dict) or audit.get("prediction_batches") != 3 or
                    audit.get("prediction_samples") != 34 or
                    audit.get("search_seed") != seed + 8100 + index):
                raise ValueError("missing fixed-budget predictor accounting")
            journal.emit({"kind": "planning_returned", "decision": index, "audit": audit})
            receipt, updated = adapter.advance(wrapped, context, actions)
        except Exception as exc:
            # This may itself fail: in that case retain the unmatched durable
            # intent and propagate; never synthesize an outcome or retry.
            journal.emit({"kind": "arm_failed", "decision": index,
                          "error_type": type(exc).__name__, "message": str(exc)})
            return {"status": "error", "decision": index, "error": str(exc)}
        if updated is None:
            journal.emit({"kind": "arm_stopped", "decision": index, "status": receipt["status"]})
            return {"status": receipt["status"], "decision": index}
        context = updated
    journal.emit({"kind": "arm_completed", "decisions": 2})
    return {"status": "complete", "decisions": 2}


def cuda_planner(model, processor, memory, mean, std, env):
    """Adapt the existing frozen host-CUDA AP kernel; called only by host.

    Caller must authenticate model, data, normalization and hold the CUDA lock.
    """
    import time
    import torch
    from . import preflight_reacher_d2p_p1_reconfiguration as setup
    from .pc_ap_official_cem_kernel import official_cem_kernel
    from .pc_ap_reacher_cuda_adapter import ReacherCudaPredictor

    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    low, high = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    low = torch.as_tensor(low, dtype=torch.float32, device="cuda").reshape(-1)
    high = torch.as_tensor(high, dtype=torch.float32, device="cuda").reshape(-1)

    def plan(view, seed, arm, index):
        pixels, goal, history = setup.p0._prepare(
            processor, view["frames"], view["root_frame"], view["goal"],
            np.asarray(view["history_action_blocks"], dtype=np.float32), torch.device("cuda"))
        embedded = setup.p0._latent_context(model, pixels, history)
        goal_embedding = setup.p0._goal_latent(model, goal)
        predictor = ReacherCudaPredictor(model, embedded, history)
        retrieval = None
        if arm == "ap_observed_target":
            item = memory.retrieve(embedded[0, -1].cpu().numpy(), goal_embedding.cpu().numpy(),
                                   25, 5 * index)
            target = item.target
            retrieval = {"episode_id": item.episode_id, "start": item.start,
                         "horizon": item.horizon, "distance": item.squared_distance}
        else:
            target = goal_embedding.cpu().numpy()

        def score(actions):
            blocks = actions[0, :, 0].detach().cpu().numpy().reshape(-1, 5, 2)
            endpoints = predictor(blocks)
            return torch.as_tensor(((endpoints - target) ** 2).sum(axis=1),
                                   device="cuda", dtype=torch.float32).reshape(1, -1)

        search_seed = seed + 8100 + index
        torch.cuda.synchronize()
        started = time.perf_counter()
        outcome = official_cem_kernel(
            score, torch.zeros((1, 1, 10), dtype=torch.float32, device="cuda"), low, high,
            torch.Generator(device="cuda").manual_seed(search_seed),
            population=16, iterations=2, elites=4)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        normalized = outcome.normalized_block.cpu().numpy().reshape(5, 2)
        raw = (normalized.reshape(-1) * std + mean).reshape(5, 2)
        clipped = np.clip(raw, env.action_space.low, env.action_space.high)
        if not np.allclose(clipped, raw, rtol=0, atol=2e-6):
            raise RuntimeError("denormalized action outside bounds")
        return np.asarray(clipped, dtype=np.float32), {
            "prediction_samples": predictor.predicted_blocks,
            "prediction_batches": predictor.prediction_batches,
            "search_seed": search_seed, "retrieval": retrieval,
            "normalized_actions": normalized.tolist(), "raw_actions": clipped.tolist(),
            "prior_cost": outcome.initial_cost, "refined_cost": outcome.refined_cost,
            "refined_selected": outcome.refined_selected,
            "planning_seconds_including_bridge": elapsed}
    return plan
