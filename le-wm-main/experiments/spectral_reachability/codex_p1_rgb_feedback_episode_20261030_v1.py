"""Live H10 reset/prefix and causally interleaved two-stage suffix episode.

Requires a caller-owned fresh environment, H10 physical meter/store, and a
metered CUDA planning callback. Guard rejection and all other errors propagate
fail-closed; this module does not classify or silently discard censored rows.
"""
import hashlib

import numpy as np

from . import codex_p1_three_prefix_h10_runner_20261030 as h10
from . import codex_p1_two_stage_feedback_20261030_v1 as feedback
from . import codex_p1_two_stage_journal_feedback_20261030_v1 as journal_feedback
from . import codex_p1_rgb_cp10_decision_bridge_20261030_v1 as rgb_bridge

base, prior, need = h10.base, h10.prior, h10.require


def _packet(result):
    need(type(result) is dict and result.get("execute_packet") == 0, "plan result")
    raw = bytes.fromhex(result["raw_horizon_f32_hex"])
    need(len(raw) == 80, "two packet horizon")
    horizon = np.frombuffer(raw, dtype="<f4").copy().reshape(2, 5, 2)
    feedback.packet_bytes(horizon[0])
    feedback.packet_bytes(horizon[1])
    return horizon[0]


def episode(env, root, source_plan, anchor, row, meter, store, plan_stage):
    """Physically execute each returned first packet; plan only from RGB/history."""
    trace = prior.TraceStore(store)
    body = env.unwrapped
    packet = root["packets"][anchor["condition"]]
    obs, native_goal = base.place(env, source_plan, root["slot"], meter, trace)
    frames = [base.capture(env, obs, "warmup", 0, row, meter, trace, root["warm"][0])]
    for step, command in enumerate(base.moving.warmup_commands(), 1):
        need(not body.eval_state(native_goal, base.source.vector(obs["state"]))[0], "warmup terminal")
        obs, info, t, tr, reward, actual = base.dispatch(env, command, "warmup", step, row, meter, trace, native_goal)
        image = base.capture(env, obs, "warmup", step, row, meter, trace, root["warm"][step])
        base.paired.check_return(env, command, actual, info, native_goal, reward, t)
        need(not t and not tr, "warmup stopped")
        if step in (5, 10):
            frames.append(image)
    need(base.moving.exact(np.asarray(frames), packet["rgb_history"])
         and base.moving.exact(packet["raw_action_history"], base.moving.warmup_commands()), "root history")
    goal = base.source.vector(root["goal"])
    need(base.source.root_hash(base.source.vector(obs["state"]), goal) == root["root_sha256"], "root identity")
    base.bind_goal(body, goal, packet["goal_rgb"], meter, trace)
    bound = base.invariant(body)
    base.capture(env, obs, "plan", 0, row, meter, trace)
    need(not body.eval_state(goal, base.source.vector(obs["state"]))[0], "root terminal")
    prefix = np.frombuffer(bytes.fromhex(anchor["prefix5_float32_hex"]), dtype="<f4").reshape(5, 2)
    for step, command in enumerate(prefix, 1):
        obs, info, t, tr, reward, actual = base.dispatch(env, command, "plan", step, row, meter, trace, goal)
        base.capture(env, obs, "plan", step, row, meter, trace)
        base.paired.check_return(env, command, actual, info, goal, reward, t)
        need(not t and not tr, "prefix stopped")
    prior.compare_trace(row, anchor, trace, meter)
    cp5 = row["trajectory"][-1]
    path = store.path/cp5["png"]
    need(path.is_file() and not path.is_symlink() and h10.runner.physical.sha(path) == cp5["file_sha256"]
         and cp5["pixel_sha256"] == anchor["checkpoint"]["pixel_sha256"], "actual cp5 image")
    png = path.read_bytes()
    start = base.source.vector(obs["state"]).copy()
    row["start_state"], row["start_verified"] = start.tolist(), True
    start_event = [e for e in store.events if e["kind"] == "suffix_start_verified"]
    need(len(start_event) == 1, "suffix start event")
    last_prefix_event = start_event[0]["index"]
    previous_packet = None
    bound_feedback = None
    for phase in (1, 2):
        # No physical state, native metrics, or outcome is passed to planner.
        result = plan_stage(phase, png, previous_packet)
        commands = _packet(result)
        store.emit("phase_plan_returned", **meter.where, phase=phase,
                   plan_sha256=hashlib.sha256(bytes.fromhex(result["raw_horizon_f32_hex"])).hexdigest(),
                   packet_float32_hex=commands.tobytes().hex(),
                   stage_report_sha256=result["stage_report_sha256"])
        if phase == 2:
            goal_rgb = np.asarray(packet["goal_rgb"])
            goal_sha = hashlib.sha256(np.ascontiguousarray(goal_rgb).tobytes()).hexdigest()
            # The saved consumer independently obtains goal pixels from the fixed task source.
            prefix_events = [e for e in store.events if e["index"] <= bound_feedback["last_render_index"]]
            decision = rgb_bridge.decide(prefix_events, store.path, anchor["index"], anchor["condition"],
                                         previous_packet, last_prefix_event, goal_rgb, goal_sha, commands)
            store.emit("rgb_decision", **meter.where, receipt=decision)
            if decision["decision"] == "rgb_rejected":
                row["status"] = "rgb_rejected"
                return
            commands = np.frombuffer(bytes.fromhex(decision["executed_packet_f32_hex"]),
                                     dtype="<f4").copy().reshape(5, 2)
        for offset, command in enumerate(commands, 1):
            step = (phase-1)*5+offset
            obs, info, t, tr, reward, actual = base.dispatch(env, command, "probe", step, row, meter, trace, goal)
            base.capture(env, obs, "probe", step, row, meter, trace)
            base.paired.check_return(env, command, actual, info, goal, reward, t)
            now = base.invariant(body)
            need({k: v for k, v in now.items() if k != "snapshot"} ==
                 {k: v for k, v in bound.items() if k != "snapshot"}
                 and base.moving.exact(np.asarray(info["goal"]), packet["goal_rgb"]), "goal/renderer drift")
            state = base.source.vector(obs["state"])
            delta = state-start
            row["responses"].append(dict(step=step, state=state.tolist(), delta_native7=delta.tolist(),
                                         delta_angle_wrapped=float(np.arctan2(np.sin(delta[4]), np.cos(delta[4]))),
                                         terminated=t, truncated=tr))
            if t or tr:
                row["status"] = "native_stopped"
                return
        if phase == 1:
            bound_feedback = journal_feedback.from_journal(store.events, store.path, anchor["index"],
                                                          anchor["condition"], commands, after_index=last_prefix_event)
            png = bound_feedback["png"]
            previous_packet = commands.copy()
            store.emit("feedback_verified", **meter.where, phase=2,
                       pixel_sha256=bound_feedback["pixel_sha256"], png_sha256=bound_feedback["png_sha256"],
                       last_render_index=bound_feedback["last_render_index"],
                       executed_packet_sha256=hashlib.sha256(commands.tobytes()).hexdigest())
    row["status"] = "probe_completed"
