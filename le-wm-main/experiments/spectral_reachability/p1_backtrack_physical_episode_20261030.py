"""Fixed-action native episode adapter; not an execution or admission authority.

The caller must validate the sealed source join, create an exclusive bounded
store, instrument all native operations, and manage failures and the 16-row
denominator. No model call or action selection takes place here.
"""
from copy import deepcopy
import hashlib

import numpy as np

from experiments.spectral_reachability import agent_allmode_physical_20261028_v1 as native
from experiments.spectral_reachability import p1_backtrack_physical_sequence_20261030 as flow
from experiments.spectral_reachability import p1_backtrack_prefix_guard_20261030 as prefix_guard


def episode(env, root, source_plan, plan, row, meter, store):
    """Reset and execute one literal 15-step plan, retaining native early stops."""
    condition = plan["condition"]
    commands = np.frombuffer(bytes.fromhex(plan["raw15_float32_hex"]), dtype="<f4")
    native.require(commands.size == 30, "exact raw15 bytes")
    commands = commands.reshape(15, 2).copy()
    native.require(np.isfinite(commands).all() and (np.abs(commands) <= 1).all(), "bounded raw15")
    body = env.unwrapped
    packet, trace = root["packets"][condition], root["warm_trace"]
    warm = native.moving.warmup_commands()
    native.require(np.asarray(packet["raw_action_history"]).dtype == np.float32 and
                   np.asarray(packet["raw_action_history"]).tobytes() == warm.tobytes(),
                   "exact original raw warmup")
    obs, native_goal = native.place(env, source_plan, root["slot"], meter, store)
    history = [native.observe(env, obs, "warmup", 0, row, meter, store, trace)]
    for step, command in enumerate(warm, 1):
        if body.eval_state(native_goal, native.source.vector(obs["state"]))[0]:
            raise native.NativeStop("warmup already native terminal")
        obs, info, terminated, truncated, reward, actual = native.dispatch(
            env, command, "warmup", step, row, meter, store, native_goal)
        image = native.observe(env, obs, "warmup", step, row, meter, store, trace)
        native.paired.check_return(env, command, actual, info, native_goal, reward, terminated)
        if terminated or truncated:
            raise native.NativeStop("warmup native stop")
        if step in (5, 10):
            history.append(image)
    native.require(native.moving.exact(np.asarray(history), packet["rgb_history"]),
                   "exact same-condition original RGB history")
    state, goal = native.source.vector(obs["state"]), native.source.vector(root["goal"])
    native.require(native.source.root_hash(state, goal) == root["root_sha256"] ==
                   plan["physical_root_sha256"], "original native root and goal")
    native.bind_goal(body, goal, packet["goal_rgb"], meter, store)
    bound = native.invariant(body)
    row.update(initial_exact=True, root_state=state.tolist(), root_sha256=root["root_sha256"],
               goal=goal.tolist(), goal_pose=bound["goal_pose"],
               goal_pixel_sha256=native.moving.image_sha(packet["goal_rgb"]))
    if body.eval_state(goal, state)[0]:
        raise native.NativeStop("constructed root already terminal")

    def dispatch(step, command):
        result = native.dispatch(env, command, "plan", step, row, meter, store, goal)
        next_obs, info, terminated, truncated, reward, actual = result
        return dict(obs=next_obs, info=info, terminated=terminated, truncated=truncated,
                    reward=reward, actual=actual, command=command)

    def capture(step, transition):
        nonlocal obs
        if transition is not None:
            obs = transition["obs"]
        native.observe(env, obs, "plan", step, row, meter, store, trace)
        selected = deepcopy(row["trajectory"][-1])
        if step == 15:
            # The reused capture helper only records 0/5/10 as checkpoints.
            row["checkpoints"].append({k: deepcopy(selected[k]) for k in
                                      ("step", "png", "file_sha256", "pixel_sha256", "state", "snapshot")})
            row["missing_checkpoints"].remove(15)
        if transition is not None:
            success = native.paired.check_return(
                env, transition["command"], transition["actual"], transition["info"],
                goal, transition["reward"], transition["terminated"])
            now = native.invariant(body)
            native.require(native.moving.exact(np.asarray(transition["info"]["goal"]), packet["goal_rgb"])
                           and {k: v for k, v in now.items() if k != "snapshot"} ==
                           {k: v for k, v in bound.items() if k != "snapshot"},
                           "true goal and renderer invariant")
            actual_state = native.source.vector(obs["state"])
            metrics = native.base.previous.metric(goal, actual_state)
            native.require(metrics["native_success"] is bool(success) and
                           transition["reward"] == -metrics["native_distance"], "native labels")
            response = dict(step=step, state=actual_state.tolist(), reward=transition["reward"],
                            native_success=bool(success), terminated=transition["terminated"],
                            truncated=transition["truncated"], native_metrics=metrics)
            row["responses"].append(response)
            store.emit("own_response", **meter.where, response=response)
        return selected

    def guard(step, selected):
        frames = [x["receipt"] for x in row["prefix51"]["frames"]
                  if x["receipt"]["phase"] == "plan" and x["receipt"]["step"] == step]
        native.require(len(frames) == 3 and [x["overlay_index"] for x in frames] == [0, 1, 2],
                       "fresh cp5 overlay triplet")
        native.require(frames[0]["pixel_sha256"] == frames[2]["pixel_sha256"] and
                       all(x["state"] == selected["state"] and x["snapshot"] == selected["snapshot"]
                           for x in frames), "cp5 overlay state and off replay")
        prefix_hex = commands[:5].astype("<f4", copy=False).tobytes().hex()
        payload = prefix_guard.identity(task_id=plan["task_id"], condition=condition,
                                        prefix_hex=prefix_hex, state=selected["state"],
                                        snapshot=selected["snapshot"],
                                        pixels={"off": frames[0]["pixel_sha256"], "on": frames[1]["pixel_sha256"]},
                                        history_hex=warm.astype("<f4", copy=False).tobytes().hex() + prefix_hex)
        key = hashlib.sha256((plan["task_id"] + condition + prefix_hex).encode()).hexdigest()
        store.check(len(payload))
        digest = prefix_guard.establish_or_verify(store.path / ("cp5_" + key + ".json"), payload)
        store.emit("cp5_guard_verified", **meter.where, identity_sha256=digest, reference_key=key)
        return True

    result = flow.sequence(commands, dispatch, capture, guard)
    row["status"] = result["status"]
    return result
