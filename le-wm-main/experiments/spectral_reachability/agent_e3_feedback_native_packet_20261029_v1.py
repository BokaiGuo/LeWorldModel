"""Incremental native E3 action packet primitive; not a production launcher.

Default ops are the frozen physical driver primitives. Caller owns source
authentication, env lifetime, setup, RGB encoder, durable output and consumer.
"""
from copy import deepcopy
import hashlib
import numpy as np
from experiments.spectral_reachability import agent_mixedmode_physical_20261028_v1 as native


def need(ok, message):
    if not ok:raise ValueError(message)


def root_goal_binding(env, original_goal, expected_original_root, new_goal, new_rgb,
                      expected_new_goal, expected_pixel_sha, meter, store, *, ops=native):
    """Verify old root with OLD goal before attaching independently pinned new goal."""
    body=env.unwrapped
    state=ops.source.vector(body._get_obs());old=ops.source.vector(original_goal)
    goal=ops.source.vector(new_goal);expect=ops.source.vector(expected_new_goal)
    rgb=np.asarray(new_rgb)
    need(ops.source.root_hash(state,old)==expected_original_root,'original root with original goal')
    need(goal.astype('<f8').tobytes()==expect.astype('<f8').tobytes(),'own goal native7 exact bits')
    need(rgb.dtype==np.uint8 and rgb.shape==(224,224,3) and ops.moving.image_sha(rgb)==expected_pixel_sha,
         'own goal RGB exact decoded pixels')
    pre=ops.invariant(body)
    store.emit('original_root_accepted',root_sha256=expected_original_root,
               state=state.tolist(),original_goal=old.tolist(),snapshot=deepcopy(pre['snapshot']))
    ops.bind_goal(body,goal,rgb,meter,store)
    post=ops.invariant(body)
    need(pre['snapshot']==post['snapshot'] and pre['goal_pose']==post['goal_pose']
         and pre['renderer']==post['renderer'],'goal binding changed original state/renderer/pose')
    need(np.asarray(post['goal_state'],dtype='<f8').tobytes()==goal.astype('<f8').tobytes()
         and post['goal_rgb_sha256']==expected_pixel_sha,'new goal installed exactly')
    store.emit('new_goal_accepted',goal=goal.tolist(),goal_pixel_sha256=expected_pixel_sha,
               original_root_sha256=expected_original_root,snapshot=deepcopy(post['snapshot']))
    return post


def packet(env, commands, *, start, row, goal, goal_rgb, trace, meter, store, ops=native):
    """Dispatch full packet or preserve actual partial at the first native stop.

    Every step uses original dispatch/observe/check_return; intent, return and
    actual native trajectory are written by those primitives. No next action
    after termination, truncation or exception. No CPU neural inference.
    """
    a=np.asarray(commands)
    need(a.dtype==np.float32 and a.shape==(5,2) and np.isfinite(a).all()
         and (np.abs(a)<=1).all(),'literal raw f32 packet')
    need(type(start) is int and start in (0,5,10,15),'fixed packet boundary')
    raw=[];observation=None
    for j in range(5):
        step=start+j+1
        obs,info,terminated,truncated,reward,actual=ops.dispatch(env,a[j], 'plan',step,row,meter,store,goal)
        need(actual.dtype==np.float32 and actual.shape==(2,) and actual.tobytes()==a[j].tobytes(),
             'literal returned raw bits')
        raw.append(actual.copy())
        observation=ops.allmode.observe(env,obs,'plan',step,row,meter,store,trace)
        success=ops.paired.check_return(env,a[j],actual,info,goal,reward,terminated)
        need(np.asarray(info['goal_state'],dtype='<f8').tobytes()==np.asarray(goal,dtype='<f8').tobytes()
             and ops.moving.exact(np.asarray(info['goal']),goal_rgb),'unchanged actual goal and RGB')
        store.emit('e3_native_step_accepted',step=step,requested_bits=a[j].tobytes().hex(),
            returned_bits=actual.tobytes().hex(),terminated=terminated,truncated=truncated,
            native_success=bool(success))
        if terminated or truncated:break
    return dict(start=start,end=start+len(raw),returned_raw=np.asarray(raw,dtype=np.float32).reshape(-1,2),
                complete=len(raw)==5,stopped=terminated or truncated,observation=observation)
