"""K17 first5 native episode seam. No CLI or full production consumer."""
from copy import deepcopy
import numpy as np
from experiments.spectral_reachability import agent_p3_h3_witness_physical_20261029_v1 as h3
from experiments.spectral_reachability import agent_p3_c3_k17_physical_gate_20261029_v1 as gate


def require(value, message):
    if not value:
        raise ValueError(message)


def episode(env, root, source_plan, item, task, reference, row, meter, store):
    """Real native operations; caller must bind authorities and bounded meter.

    This entry alone is deliberately not an authorized production runner.
    """
    body = env.unwrapped
    packet = root['packets']['off']
    trace = root['warm_trace']
    raw_warm = h3.moving.warmup_commands()
    require(h3.moving.exact(np.asarray(packet['raw_action_history']), raw_warm), 'warm history')
    obs, goal0 = h3.place(env, source_plan, root['slot'], meter, store)
    history = [h3.observe(env, obs, 'warmup', 0, row, meter, store, trace)]
    returns = []
    for step, command in enumerate(raw_warm, 1):
        if body.eval_state(goal0, h3.source.vector(obs['state']))[0]:
            raise h3.native.allmode.NativeStop('warmup native terminal')
        obs, info, terminated, truncated, reward, actual = h3.dispatch(
            env, command, 'warmup', step, row, meter, store, goal0)
        image = h3.observe(env, obs, 'warmup', step, row, meter, store, trace)
        h3.paired.check_return(env, command, actual, info, goal0, reward, terminated)
        returns.append(dict(step=step, status='returned', actual_dtype=actual.dtype.str,
                            requested_f32_hex=command.tobytes().hex(), actual_f32_hex=actual.tobytes().hex(),
                            terminated=terminated, truncated=truncated))
        if terminated or truncated:
            raise h3.native.allmode.NativeStop('warmup native early stop')
        if step in (5, 10):
            history.append(image)
    require(h3.moving.exact(np.asarray(history), packet['rgb_history']), 'RGB history')
    state, goal = h3.source.vector(obs['state']), h3.source.vector(root['goal'])
    require(h3.source.root_hash(state, goal) == root['root_sha256'], 'root hash')
    h3.bind_goal(body, goal, packet['goal_rgb'], meter, store)
    bound = h3.invariant(body)
    h3.observe(env, obs, 'plan', 0, row, meter, store, trace)
    frames = row['prefix51']['frames']
    require(len(frames) == 36, 'complete warm33 and plan0 triplet')
    live = dict(reset_identity=root['slot'], warm_frames=deepcopy(frames[:33]), warm_returns=returns,
                root_state=state.tolist(), goal_state=goal.tolist(), bound_goal=bound,
                goal_frame=deepcopy(frames[33]))
    if body.eval_state(goal, state)[0]:
        raise h3.native.allmode.NativeStop('root native terminal')
    try:
        proof = gate.capture_check(item, task, reference, live, store.path)
        require(proof['physical_dispatch_ready'] is False and proof['history']['accepted'] is True
                and proof['action']['actual_replay_tokens_exact'] is True, 'frozen guard proofs')
        require(h3.invariant(body) == bound, 'fresh bound immediately before guard event')
        store.emit('k17_guard', accepted=True, proof=proof)
    except Exception as exc:
        store.emit('k17_guard', accepted=False, error_type=type(exc).__name__)
        raise
    commands = np.frombuffer(bytes.fromhex(item['first5_raw_f32_hex']), dtype='<f4').reshape(5, 2)
    for step, command in enumerate(commands, 1):
        require(h3.invariant(body) == bound if step == 1 else True, 'goal changed before first intent')
        obs, info, terminated, truncated, reward, actual = h3.dispatch(
            env, command.copy(), 'plan', step, row, meter, store, goal)
        h3.observe(env, obs, 'plan', step, row, meter, store, trace)
        h3.paired.check_return(env, command, actual, info, goal, reward, terminated)
        require(actual.dtype.str == '<f4' and actual.tobytes() == command.tobytes(), 'actual raw action')
        if terminated or truncated:
            return {'status': 'native_stopped', 'returned': step, 'missing_steps': list(range(step + 1, 6))}
    return {'status': 'horizon_completed', 'returned': 5, 'missing_steps': []}
