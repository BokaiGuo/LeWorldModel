"""Single-render native transition adapter for E3. Not a complete launcher."""
from copy import deepcopy
import numpy as np
from experiments.spectral_reachability import p1_moving_live_feedback_physical_20261027_v1 as old


def need(ok,message):
    if not ok:raise ValueError(message)


def packet(env, commands, *, start, row, goal, goal_rgb, meter, store, ops=old):
    """Actual one-frame native packet, recording every accepted returned step.

    Caller owns source-bound setup/goal, original hooks, limits, close and partial
    report. Exceptions propagate; operation attempt/return accounting belongs to
    the native metered env. On stop the short actual prefix is returned untouched.
    """
    a=np.asarray(commands)
    need(a.dtype==np.float32 and a.shape==(5,2) and np.isfinite(a).all()
         and (abs(a)<=1).all(),'literal raw f32 packet')
    need(type(start) is int and start in (0,5,10,15),'fixed control boundary')
    body=env.unwrapped;returned=[];receipt=None;terminated=truncated=False
    for i,command in enumerate(a):
        step=start+i+1
        store.emit('authorized_dispatch',episode_id=row['episode_id'],step=step,
                   packet_boundary=start,command=command.tolist(),raw_bits=command.tobytes().hex())
        obs,reward,terminated,truncated,info=env.step(command.copy())
        row['control_returned']+=1
        actual=np.asarray(body.latest_action).copy()
        row['executed_raw'].append(actual.tolist())
        _,receipt=ops.capture(env,obs,row,store,'plan',step)
        need(actual.dtype==np.float32 and actual.shape==(2,) and actual.tobytes()==command.tobytes(),
             'actual action bits')
        need(np.array_equal(info['goal_state'],goal) and np.array_equal(info['goal'],goal_rgb),
             'actual bound goal and RGB unchanged')
        ops.native_transition(body,goal,obs,reward,terminated,truncated,row)
        returned.append(actual)
        store.emit('control_return',episode_id=row['episode_id'],step=step,
                   command=actual.tolist(),raw_bits=actual.tobytes().hex(),
                   terminated=bool(terminated),truncated=bool(truncated),reward=float(reward))
        if terminated or truncated:break
    return dict(start=start,end=start+len(returned),returned_raw=np.asarray(returned,dtype=np.float32).reshape(-1,2),
                stopped=bool(terminated or truncated),complete=len(returned)==5,observation=deepcopy(receipt))
