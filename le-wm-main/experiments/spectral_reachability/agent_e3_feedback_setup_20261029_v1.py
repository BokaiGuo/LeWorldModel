"""Native E3 source root replay and cp5 guard. Not a full launcher."""
from copy import deepcopy
import numpy as np
from experiments.spectral_reachability import p1_moving_live_feedback_physical_20261027_v1 as old
from experiments.spectral_reachability import agent_e3_feedback_single_capture_20261029_v1 as packet


def need(ok,message):
    if not ok:raise ValueError(message)


def setup(env,ref,source_plan,original_root_sha,new_goal,new_rgb,expected_goal,expected_goal_pixel,
          prefix5,expected_cp5,row,meter,store,*,ops=old,packet_ops=packet):
    """Replay original placement/warmup, bind own goal, execute exact source prefix.

    Inputs must already have been independently SHA-authorized by the caller.
    No state hydration or replacement. Returns None on an actual native stop.
    """
    body=env.unwrapped;slot=ref['slot'];meter.region='setup'
    obs,info=env.reset(seed=slot['reset_seed']);original_reset_goal=ops.source.vector(info['goal_state'])
    requested=ops.source.sample_state(original_reset_goal,list(body.block.shapes),
                                      dict(root_seed=slot['root_seed']),0,0,source_plan)
    obs,info=env.reset(seed=0,options=dict(state=requested.copy(),goal_state=original_reset_goal.copy()))
    image,rec=ops.capture(env,obs,row,store,'warmup',0)
    history=[image];states=[ops.source.vector(obs['state'])];snapshots=[rec['snapshot']];actions=[]
    meter.region='warmup'
    for step,command in enumerate(ops.kernel.raw_history(ref['raw_history']),1):
        obs,reward,terminated,truncated,info=env.step(command.copy());row['warmup_returned']+=1
        image,rec=ops.capture(env,obs,row,store,'warmup',step)
        actual=np.asarray(body.latest_action).copy();actions.append(actual)
        need(ops.kernel.moving.exact(actual,command),'actual warm action bits')
        store.emit('warmup_return',episode_id=row['episode_id'],step=step,command=actual.tolist(),
                   terminated=bool(terminated),truncated=bool(truncated))
        if terminated or truncated:
            row.update(status='native_stopped_warmup',terminated=bool(terminated),truncated=bool(truncated));return None
        if step in (5,10):history.append(image);states.append(ops.source.vector(obs['state']));snapshots.append(rec['snapshot'])
    need(ops.kernel.moving.exact(np.asarray(actions),ref['raw_history']) and
         ops.kernel.moving.exact(np.asarray(history),ref['history']) and
         ops.kernel.moving.exact(np.asarray(states),ref['states']) and snapshots==ref['dynamics'],
         'original warm history/state/snapshot exact')
    state=ops.source.vector(obs['state']);source_goal=ops.source.vector(ref['goal'])
    need(ops.source.root_hash(state,source_goal)==original_root_sha,'original root under original goal')
    goal=ops.source.vector(new_goal);expected=ops.source.vector(expected_goal);rgb=np.asarray(new_rgb)
    need(goal.astype('<f8').tobytes()==expected.astype('<f8').tobytes()
         and rgb.shape==(224,224,3) and rgb.dtype==np.uint8
         and ops.kernel.moving.image_sha(rgb)==expected_goal_pixel,'new goal/native7/RGB exact')
    before=ops.state_seal(body)
    store.emit('original_root_accepted',episode_id=row['episode_id'],root_sha256=original_root_sha,
               original_goal=source_goal.tolist(),state=state.tolist(),snapshot=deepcopy(before['snapshot']))
    meter.region='goal_bind';meter.call('goal_bind',lambda:ops.kernel.moving.attach_goal(body,goal,rgb))
    after=ops.state_seal(body)
    need(before['snapshot']==after['snapshot'] and before['goal_pose']==after['goal_pose']
         and np.asarray(after['goal'],dtype='<f8').tobytes()==goal.astype('<f8').tobytes()
         and after['goal_rgb_sha256']==expected_goal_pixel,'goal attachment/state/pose exact')
    _,root_ob=ops.capture(env,obs,row,store,'plan',0)
    store.emit('actual_root_guard',episode_id=row['episode_id'],original_root_sha256=original_root_sha,
               original_goal=source_goal.tolist(),new_goal=goal.tolist(),new_goal_pixel_sha256=expected_goal_pixel,
               history_pixels=[ops.array_sha(x) for x in history],root_observation=deepcopy(root_ob))
    first=packet_ops.packet(env,prefix5,start=0,row=row,goal=goal,goal_rgb=rgb,meter=meter,store=store,ops=ops)
    if first['stopped']:
        row.update(status='native_stopped_prefix',terminated=row.get('terminated'),truncated=row.get('truncated'));return None
    need(first['complete'] and first['end']==5,'full exact prefix')
    cp=first['observation']
    need(cp['step']==5 and cp['pixel_sha256']==expected_cp5['pixel_sha256']
         and cp['snapshot']==expected_cp5['snapshot']
         and np.asarray(cp['state'],dtype='<f8').tobytes()==np.asarray(expected_cp5['state'],dtype='<f8').tobytes(),
         'new own physical cp5 equals accepted original history')
    store.emit('actual_cp5_guard',episode_id=row['episode_id'],observation=deepcopy(cp),
               prefix_raw_bits=np.asarray(prefix5,dtype=np.float32).tobytes().hex())
    return dict(cp5_observation=deepcopy(cp),prior_raw5=np.asarray(prefix5,dtype=np.float32).copy(),goal=goal,goal_rgb=rgb)
