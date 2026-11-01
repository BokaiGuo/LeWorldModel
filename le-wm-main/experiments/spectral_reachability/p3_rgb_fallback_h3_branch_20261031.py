"""Twenty-step physical branch with an RGB-triggered cached-suffix fallback.

This is a branch primitive, not an experiment launcher or a saved-data
consumer. The caller owns resource caps, source authentication and lifecycle.
"""
from copy import deepcopy
import hashlib

import numpy as np

from . import p1_constructive_moving_history_physical_20261027_v1 as kernel
from . import p3_rgb_edge_preflight_20261031 as edge

require = kernel.require


def raw20(value):
    a = np.asarray(value)
    require(a.shape == (20, 2) and a.dtype.kind in "fi" and np.isfinite(a).all()
            and (np.abs(a) <= 1).all() and np.array_equal(a, a.astype(np.float32)),
            "exact finite float32 raw20")
    return a.astype(np.float32)


def capture(env, obs, phase, step, row, ops, store, index):
    image, snapshot = kernel.capture(env, obs, phase, step, row, ops, store, index)
    if phase == "plan" and step in (15, 20):
        row["checkpoints"].append(deepcopy(row["last_observation"]))
    return image, snapshot


def guard_at_15(frame, row, ops, store):
    """Record the RGB-only decision before any sixteenth action."""
    decision = edge.preflight(frame)
    row["rgb_edge_preflight"] = decision
    store.emit("rgb_edge_preflight", **ops.where, step=15,
               pixel_sha256=kernel.moving.image_sha(frame), decision=decision)
    return decision["admitted"]


def choose_suffix(frame, row, ops, store, actions, fallback):
    """Journal the decision before exposing any command from the chosen suffix."""
    require(actions.shape == fallback.shape == (20, 2)
            and actions[:15].tobytes() == fallback[:15].tobytes(), "fallback prefix identity")
    admitted = guard_at_15(frame, row, ops, store)
    suffix = actions[15:] if admitted else fallback[15:]
    selection = "planned" if admitted else "cached_fallback"
    digest = hashlib.sha256(suffix.tobytes()).hexdigest()
    store.emit("fallback_selection", **ops.where, step=15,
               selection=selection, suffix_sha256=digest)
    row["suffix_selection"] = selection
    row["chosen_suffix_sha256"] = digest
    return suffix


def branch(env, ref, source_plan, plan, row, ops, store, index):
    actions = raw20(plan["commands"])
    fallback = raw20(plan["fallback_commands"])
    require(hashlib.sha256(fallback.tobytes()).hexdigest() == plan["fallback_float32_sha256"]
            and actions[:15].tobytes() == fallback[:15].tobytes(), "cached fallback binding")
    body = env.unwrapped
    slot = ref["slot"]
    warm = kernel.raw_history(ref["raw_history"])
    history, states, dynamics, executed = [], [], [], []
    ops.where = dict(branch=index, root_slot=plan["root_slot"], arm=plan["arm"],
                     selection_digest=plan["selection_digest"])
    store.emit("branch_started", **ops.where, plan=plan, raw_warmup=warm.tolist(),
               raw_float32_bits=warm.tobytes().hex())
    _, info = ops.call("reset", lambda: env.reset(seed=slot["reset_seed"]),
                       kind_of_reset="original_seed", seed=slot["reset_seed"])
    original_goal = kernel.source.vector(info["goal_state"])
    requested = kernel.source.sample_state(original_goal, list(body.block.shapes),
                                            dict(root_seed=slot["root_seed"]), 0, 0, source_plan)
    store.emit("placement_request", **ops.where, original_goal=original_goal.tolist(),
               requested=requested.tolist(), placement_seed=0)
    obs, info = ops.call("reset", lambda: env.reset(seed=0, options=dict(
        state=requested.copy(), goal_state=original_goal.copy())),
        kind_of_reset="placement", seed=0)
    frame, snap = capture(env, obs, "warmup", 0, row, ops, store, index)
    history.append(frame); states.append(kernel.source.vector(obs["state"])); dynamics.append(snap)
    for step, command in enumerate(warm, 1):
        require(not body.eval_state(original_goal, kernel.source.vector(obs["state"]))[0],
                "native terminal before warmup")
        obs, reward, terminated, truncated, info = ops.call(
            "public_step", lambda c=command: env.step(c.copy()),
            phase="warmup", step=step, raw_action=command.tolist())
        row["warmup_returned"] += 1
        store.emit("returned_transition", **ops.where, phase="warmup", step=step,
                   state=kernel.source.vector(obs["state"]).tolist(), reward=float(reward),
                   terminated=bool(terminated), truncated=bool(truncated))
        frame, snap = capture(env, obs, "warmup", step, row, ops, store, index)
        require(kernel.moving.exact(np.asarray(body.latest_action), command), "warmup raw ACK")
        executed.append(np.asarray(body.latest_action).copy())
        store.emit("warmup_return", **ops.where, step=step, raw_action=executed[-1].tolist(),
                   raw_float32_bits=executed[-1].tobytes().hex())
        require(not terminated and not truncated, "warmup stopped")
        if step in (5, 10):
            history.append(frame); states.append(kernel.source.vector(obs["state"])); dynamics.append(snap)
    require(kernel.moving.exact(np.asarray(executed), warm)
            and kernel.moving.exact(np.asarray(history), ref["history"])
            and kernel.moving.exact(np.asarray(states), ref["states"])
            and dynamics == ref["dynamics"], "warmup source reconstruction")
    goal, goal_rgb = ref["goal"], ref["goal_rgb"]
    root = kernel.source.vector(obs["state"])
    require(kernel.source.root_hash(root, goal) == ref["root_sha256"], "root and goal identity")
    before, counts = kernel.moving.snapshot(body), deepcopy(ops.counts)
    pose = deepcopy(getattr(body, "goal_pose", None))
    ops.call("goal_bind", lambda: kernel.moving.attach_goal(body, goal, goal_rgb))
    require(kernel.moving.snapshot(body) == before
            and all(v == ops.counts[k] for k, v in counts.items() if k != "goal_bind")
            and np.array_equal(getattr(body, "goal_pose", None), pose), "goal bind side effects")
    frame, _ = capture(env, obs, "plan", 0, row, ops, store, index)
    require(kernel.moving.exact(frame, history[-1]) and np.array_equal(body.goal_state, goal)
            and np.array_equal(body._goal, goal_rgb), "goal and RGB binding")
    row.update(initial_exact=True, root_state=root.tolist(), goal=goal.tolist(),
               initial_metrics=kernel.metric(goal, root))
    require(not row["initial_metrics"]["native_success"], "root already terminal")
    store.emit("goal_attached", **ops.where, goal=goal.tolist(),
               goal_rgb_sha256=kernel.moving.image_sha(goal_rgb), snapshot=before)
    chosen_suffix = None
    for step in range(1, 21):
        if step <= 15:
            command = actions[step - 1]
        else:
            require(chosen_suffix is not None, "suffix not selected")
            command = chosen_suffix[step - 16]
        def dispatch(c=command):
            row["own_attempted"] += 1
            return env.step(c.copy())
        obs, reward, terminated, truncated, info = ops.call(
            "public_step", dispatch, phase="plan", step=step, raw_action=command.tolist())
        row["own_returned"] += 1
        state = kernel.source.vector(obs["state"])
        metric = kernel.metric(goal, state)
        success, distance = body.eval_state(goal, state)
        actual = np.asarray(body.latest_action)
        require(actual.dtype == np.float32 and actual.shape == (2,)
                and actual.tobytes() == command.tobytes(), "plan raw ACK")
        row.update(final_state=state.tolist(), final_metrics=metric, native_success=bool(success),
                   terminated=bool(terminated), truncated=bool(truncated))
        store.emit("returned_transition", **ops.where, phase="plan", step=step,
                   command=command.tolist(), command_bits=command.tobytes().hex(),
                   actual_bits=actual.tobytes().hex(), state=state.tolist(), reward=float(reward),
                   terminated=bool(terminated), truncated=bool(truncated),
                   returned_goal_state=np.asarray(info["goal_state"]).tolist(),
                   returned_goal_rgb_sha256=kernel.moving.image_sha(info["goal"]))
        frame, _ = capture(env, obs, "plan", step, row, ops, store, index)
        require(np.array_equal(info["goal_state"], goal) and np.array_equal(info["goal"], goal_rgb),
                "native goal remains bound")
        require(bool(success) == metric["native_success"] and float(distance) == metric["native_distance"]
                and bool(terminated) == bool(success) and float(reward) == -float(distance),
                "native eval reward termination")
        if terminated or truncated:
            row["status"] = "native_success" if terminated and success and not truncated else "native_stopped"
            break
        if step == 15:
            chosen_suffix = choose_suffix(frame, row, ops, store, actions, fallback)
    else:
        row["status"] = "horizon_completed"
    row["full_horizon"] = row["own_returned"] == 20 and not (row["terminated"] or row["truncated"])
    row["endpoint_scope"] = ("native_terminal" if row["terminated"] or row["truncated"]
                             else "full20" if row["full_horizon"] else
                             "partial")
