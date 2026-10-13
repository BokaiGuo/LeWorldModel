"""Held-out audit of the PushT PD-target-relative torque interface.

The live PushT action is a relative displacement that is converted inside the
environment to ``agent_position + 100 * action`` before PD control.  This
audit compares the established contact-frame feature interface with an
explicit target-relative interface on disjoint held-out contact states.

Both arms use the same direct horizon-2 endpoint labels, action fan, tree
budget, and ranking rule.  Simulator successors are used only for training
labels and held-out scoring; no simulator verification is used when ranking
the action fan.  This is a privileged physical-adapter diagnostic, not visual
LeWM planning evidence or a causal claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import stable_worldmodel  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pusht_contact_history_velocity_diagnostic import (  # noqa: E402
    collect_trajectory_data,
    mode_for_state,
)
from experiments.run_pusht_object_centric_learned_planner import (  # noqa: E402
    angle_error,
    fit_direct_contact_endpoint_model,
    make_state_only_env,
    torque_action_bank,
    torque_feature_vector,
    unit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-states", type=int, default=64)
    parser.add_argument("--held-out-states", type=int, default=48)
    parser.add_argument("--pool-transitions", type=int, default=4096)
    parser.add_argument("--directions", type=int, default=40)
    parser.add_argument("--magnitudes", nargs="+", type=float, default=[0.35, 0.55, 0.75])
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--trees", type=int, default=160)
    parser.add_argument("--seed", type=int, default=8421)
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def bootstrap(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(sample, 0.025)),
        "ci95_high": float(np.quantile(sample, 0.975)),
        "n": int(len(values)),
    }


def make_goal(state: np.ndarray, index: int) -> np.ndarray:
    """Create a fixed, held-out ranking target from the current state."""
    goal = np.asarray(state, dtype=np.float64).copy()
    offsets = np.asarray(((34.0, -27.0), (-27.0, 34.0), (41.0, 0.0), (0.0, 41.0)))
    goal[2:4] += offsets[index % len(offsets)]
    goal[4] = float((goal[4] + (0.48 if index % 2 == 0 else -0.48)) % (2.0 * np.pi))
    return goal


def actual_endpoint(env, state: np.ndarray, action: np.ndarray, horizon: int, seed: int) -> np.ndarray:
    env.reset(seed=seed, options={"state": state.copy(), "goal_state": state.copy()})
    endpoint = np.asarray(state, dtype=np.float64).copy()
    for _ in range(horizon):
        observation, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        endpoint = np.asarray(observation["state"], dtype=np.float64).copy()
        if terminated or truncated:
            break
    return endpoint


def local_target(state: np.ndarray, endpoint: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    endpoint = np.asarray(endpoint, dtype=np.float64)
    relative = state[2:4] - state[:2]
    normal = unit(relative)
    tangent = np.asarray([-normal[1], normal[0]])
    delta = endpoint[2:4] - state[2:4]
    return np.asarray([
        float(np.dot(delta, normal)),
        float(np.dot(delta, tangent)),
        float((endpoint[4] - state[4] + np.pi) % (2.0 * np.pi) - np.pi),
    ])


def local_to_world(state: np.ndarray, local: np.ndarray) -> np.ndarray:
    relative = np.asarray(state[2:4] - state[:2], dtype=np.float64)
    normal = unit(relative)
    tangent = np.asarray([-normal[1], normal[0]])
    return np.asarray([
        local[0] * normal[0] + local[1] * tangent[0],
        local[0] * normal[1] + local[1] * tangent[1],
        local[2],
    ])


def evaluate_interface(
    env,
    train_states: np.ndarray,
    held_states: list[np.ndarray],
    actions: np.ndarray,
    horizon: int,
    trees: int,
    seed: int,
    feature_interface: str,
    bootstrap_draws: int,
) -> dict[str, Any]:
    model = fit_direct_contact_endpoint_model(
        env,
        train_states,
        actions,
        horizon=horizon,
        trees=trees,
        seed=seed,
        feature_interface=feature_interface,
    )
    rows = []
    for state_index, state in enumerate(held_states):
        state = np.asarray(state, dtype=np.float64)
        goal = make_goal(state, state_index)
        features = np.asarray([
            torque_feature_vector(state, action, np.zeros(2), feature_interface)
            for action in actions
        ])
        predicted_local = np.asarray(model.predict(features), dtype=np.float64)
        predicted_effect = np.asarray([
            local_to_world(state, local) for local in predicted_local
        ])
        actual = np.asarray([
            actual_endpoint(
                env, state, action, horizon,
                seed + 500000 + state_index * 1000 + action_index,
            )
            for action_index, action in enumerate(actions)
        ])
        true_local = np.asarray([local_target(state, endpoint) for endpoint in actual])
        before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
        before_angle = angle_error(state[4], goal[4])
        predicted_xy = np.linalg.norm(
            state[2:4][None] + predicted_effect[:, :2] - goal[2:4][None], axis=1
        )
        predicted_angle = np.asarray([
            angle_error(state[4] + effect[2], goal[4]) for effect in predicted_effect
        ])
        actual_xy = np.linalg.norm(actual[:, 2:4] - goal[2:4][None], axis=1)
        actual_angle = np.asarray([
            angle_error(endpoint[4], goal[4]) for endpoint in actual
        ])
        limit = min(45.0, before_xy + 4.0)
        predicted_feasible = (predicted_angle < before_angle - 1e-5) & (predicted_xy <= limit)
        actual_feasible = (actual_angle < before_angle - 1e-5) & (actual_xy <= limit)
        score = predicted_angle + 0.02 * predicted_xy
        score[~predicted_feasible] = np.inf
        if not np.isfinite(score).any():
            score = (
                np.maximum(predicted_angle - before_angle, 0.0)
                + 0.02 * np.maximum(predicted_xy - limit, 0.0)
                + 0.02 * predicted_xy
            )
        order = np.argsort(score)
        selected = int(order[0])
        oracle_order = np.argsort(actual_angle + 0.02 * actual_xy)
        rows.append({
            "state_index": state_index,
            "mode": mode_for_state(state),
            "top1_feasible": float(actual_feasible[selected]),
            "top3_hit": float(actual_feasible[order[:3]].any()),
            "top8_hit": float(actual_feasible[order[:8]].any()),
            "oracle_any_feasible": float(actual_feasible.any()),
            "top1_angle_improvement": float(before_angle - actual_angle[selected]),
            "top1_xy": float(actual_xy[selected]),
            "oracle_top1_xy": float(actual_xy[oracle_order[0]]),
            "effect_rmse": float(np.sqrt(np.mean((predicted_local - true_local) ** 2))),
            "selected_action_index": selected,
            "predicted_feasible_count": int(predicted_feasible.sum()),
            "actual_feasible_count": int(actual_feasible.sum()),
        })
    metric_names = (
        "top1_feasible", "top3_hit", "top8_hit", "oracle_any_feasible",
        "top1_angle_improvement", "top1_xy", "effect_rmse",
    )
    metrics = {
        key: bootstrap(
            np.asarray([row[key] for row in rows]), bootstrap_draws, seed + index
        )
        for index, key in enumerate(metric_names)
    }
    return {
        "feature_interface": feature_interface,
        "feature_dim": int(len(torque_feature_vector(
            held_states[0], actions[0], np.zeros(2), feature_interface
        ))),
        "metrics": metrics,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    if args.horizon < 1 or args.train_states < 8 or args.held_out_states < 8:
        raise ValueError("horizon and state counts are too small")
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = make_state_only_env()
    try:
        train_pool = collect_trajectory_data(
            env, max(args.pool_transitions, args.train_states * 4), args.seed + 100000
        )
        held_pool = collect_trajectory_data(
            env, max(args.pool_transitions, args.held_out_states * 4), args.seed + 200000
        )
        train_states = np.asarray([
            state for state, mode in zip(train_pool["states"], train_pool["modes"])
            if mode == "contact"
        ], dtype=np.float64)[: args.train_states]
        held_states = [
            np.asarray(state, dtype=np.float64).copy()
            for state, mode in zip(held_pool["states"], held_pool["modes"])
            if mode == "contact"
        ][: args.held_out_states]
        if len(train_states) < args.train_states or len(held_states) < args.held_out_states:
            raise RuntimeError("insufficient disjoint contact states")
        actions = torque_action_bank(
            args.directions, magnitudes=tuple(args.magnitudes)
        )
        reports = [
            evaluate_interface(
                env, train_states, held_states, actions, args.horizon, args.trees,
                args.seed + 17 * index, feature_interface, args.bootstrap,
            )
            for index, feature_interface in enumerate(
                ("legacy", "target_relative", "block_target_frame")
            )
        ]
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "training": "direct constant-action horizon endpoint labels on disjoint contact states",
            "evaluation": "held-out full-fan ranking without simulator verification at selection time",
            "train_states": len(train_states),
            "held_out_states": len(held_states),
            "action_count": len(actions),
            "directions": args.directions,
            "magnitudes": [float(value) for value in args.magnitudes],
            "horizon": args.horizon,
            "trees": args.trees,
            "seed": args.seed,
            "train_heldout_disjoint_by_independent_reset_streams": True,
            "wall_seconds": time.perf_counter() - started,
        },
        "reports": reports,
        "sanity_checks": {
            "all_finite": bool(all(
                np.isfinite(value)
                for item in reports
                for metric in item["metrics"].values()
                for value in metric.values()
                if isinstance(value, (int, float))
            )),
            "same_action_count": len(actions),
            "same_states": True,
            "same_targets": True,
            "legacy_feature_dim": reports[0]["feature_dim"],
            "target_relative_feature_dim": reports[1]["feature_dim"],
            "block_target_frame_feature_dim": reports[2]["feature_dim"],
        },
        "claim_boundary": (
            "Privileged simulator-state feature-interface audit. A positive result supports the "
            "PD-target-relative adapter hypothesis for this matched PushT protocol; it does not "
            "establish frozen-LeWM latent causality, visual online planning, or task success."
        ),
    }
    (output / "target_relative_torque_feature_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    lines = [
        "# PushT target-relative torque feature audit",
        "",
        "| interface | top-1 feasible | top-3 hit | top-8 hit | effect RMSE | top-1 XY |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in reports:
        metrics = item["metrics"]
        lines.append(
            "| %s | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] |" % (
                item["feature_interface"],
                metrics["top1_feasible"]["estimate"], metrics["top1_feasible"]["ci95_low"], metrics["top1_feasible"]["ci95_high"],
                metrics["top3_hit"]["estimate"], metrics["top3_hit"]["ci95_low"], metrics["top3_hit"]["ci95_high"],
                metrics["top8_hit"]["estimate"], metrics["top8_hit"]["ci95_low"], metrics["top8_hit"]["ci95_high"],
                metrics["effect_rmse"]["estimate"], metrics["effect_rmse"]["ci95_low"], metrics["effect_rmse"]["ci95_high"],
                metrics["top1_xy"]["estimate"], metrics["top1_xy"]["ci95_low"], metrics["top1_xy"]["ci95_high"],
            )
        )
    lines += ["", report["claim_boundary"], ""]
    (output / "target_relative_torque_feature_audit.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "reports": reports, "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
