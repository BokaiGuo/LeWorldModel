"""Source-bound live PushT reset and observed packet adapter; no launcher.

Caller owns frozen authorities, model loading, four-slot denominator, model IO
audit, storage publication and independent saved consumption.
"""
from copy import deepcopy
import io
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.spectral_reachability import agent_e3_native_capture_transport_20261029_v1 as capture
from experiments.spectral_reachability import agent_e3_native_region_adapter_20261029_v1 as region
from experiments.spectral_reachability import agent_e3_reset_identity_20261029_v1 as reset
from experiments.spectral_reachability import p3_matched_order_feedback_encoder_v2_codex_20261029 as encoder
from experiments.spectral_reachability.p3_matched_order_feedback_native_codex_20261029 import NativeInitialStop

old = region.original.old
need = old.require


class CaptureStore(capture.CaptureStore):
    def frame(self, name, image):
        need(type(name) is str and Path(name).name == name and name.endswith('.png') and
             name not in self.frames and self.store.frames < 128, 'unique four-slot native PNG')
        a = np.asarray(image); need(a.shape == (224,224,3) and a.dtype == np.uint8, 'RGB224')
        stream = io.BytesIO(); Image.fromarray(a).save(stream, format='PNG'); data = stream.getvalue()
        pixel_sha = capture.pub.png_data(data)
        descriptor = dict(file=name, sha256=capture.pub.base.fsha(data), pixel_sha256=pixel_sha,
                          shape=[224,224,3], dtype='uint8')
        self.store.write(name, data, 'physical_png'); self.store.frames += 1
        self.frames[name] = descriptor
        self.store.event('physical','record','native_frame',self.store.index,{'frame':descriptor})
        return descriptor['sha256']


def setup(env, root, source_plan, task, row, meter, store):
    """Original reset sampling, exact warm10 and root binding, no action plan."""
    need(root['root_sha256'] == task['binding']['root_sha256'], 'same root task')
    body = env.unwrapped; slot = root['slot']; packet = root['packets']['off']
    traced_ops = reset.Ops(old, store.store, row['episode_id'])
    meter.region = 'setup'
    obs, info = env.reset(seed=slot['reset_seed'])
    reset_goal = old.source.vector(info['goal_state'])
    requested = traced_ops.source.sample_state(reset_goal, list(body.block.shapes),
                                                dict(root_seed=slot['root_seed']), 0, 0, source_plan)
    obs, _ = env.reset(seed=0, options=dict(state=requested.copy(), goal_state=reset_goal.copy()))
    trace = {(x['receipt']['step'],x['receipt']['overlay_index']):x for x in root['warm_trace']}
    def observe(step):
        image, receipt = old.capture(env, obs, row, store, 'warmup', step)
        reference = trace[(step,0)]['receipt']
        need(receipt['pixel_sha256'] == reference['pixel_sha256'] and receipt['snapshot'] == reference['snapshot'] and
             np.array_equal(np.asarray(receipt['state']), np.asarray(reference['state'])), 'original warm frame/state')
        return image
    images = [observe(0)]; meter.region = 'warmup'
    warm = np.asarray(packet['raw_action_history'])
    need(warm.dtype == np.float32 and warm.shape == (10,2), 'source actual raw10')
    for step,command in enumerate(warm,1):
        obs,reward,terminated,truncated,info = env.step(command.copy())
        actual = np.asarray(body.latest_action).copy()
        need(actual.dtype == np.float32 and actual.tobytes() == command.tobytes(), 'actual warm action')
        row['warmup_returned'] += 1
        image = observe(step)
        store.emit('warmup_return',episode_id=row['episode_id'],step=step,command=actual.tolist(),
                   terminated=bool(terminated),truncated=bool(truncated),reward=float(reward))
        if terminated or truncated:
            raise NativeInitialStop('warmup native stop')
        if step in (5,10): images.append(image)
    need(np.array_equal(np.asarray(images),packet['rgb_history']), 'actual warm RGB history')
    state = old.source.vector(obs['state']); goal = old.source.vector(root['goal'])
    need(old.source.root_hash(state,goal) == root['root_sha256'], 'actual root/goal identity')
    before = old.state_seal(body); meter.region = 'goal_bind'
    meter.call('goal_bind', lambda: old.kernel.moving.attach_goal(body,goal,packet['goal_rgb']))
    after = old.state_seal(body)
    need(before['snapshot'] == after['snapshot'] and before['goal_pose'] == after['goal_pose'] and
         np.array_equal(np.asarray(after['goal']),goal) and
         after['goal_rgb_sha256'] == old.kernel.moving.image_sha(packet['goal_rgb']), 'goal binding invariant')
    old.capture(env,obs,row,store,'plan',0)
    success,distance = body.eval_state(goal,obs['state'])
    need(np.isfinite(distance), 'native initial evaluation')
    store.emit('initial_native_evaluation',episode_id=row['episode_id'],boundary=0,
               success=bool(success),distance=float(distance))
    if success:
        raise NativeInitialStop('initial native success')
    store.emit('matched_root_accepted',episode_id=row['episode_id'],root_sha256=root['root_sha256'],
               warm10=warm.tolist(),goal=goal.tolist(),snapshot=deepcopy(after['snapshot']))
    return goal, packet['goal_rgb'], warm[-5:].copy()


class Session:
    def __init__(self, env, goal, goal_rgb, meter, store, row, model, torch, device):
        self.env,self.goal,self.goal_rgb,self.meter,self.store,self.row = env,goal,goal_rgb,meter,store,row
        self.model,self.torch,self.device = model,torch,device
        self.episode_id = row['episode_id']; self.latest = None

    def packet(self, start, raw):
        r = region.packet(self.env,raw,start=start,row=self.row,goal=self.goal,goal_rgb=self.goal_rgb,
                          meter=self.meter,store=self.store)
        self.latest = r['observation']
        data = b''
        if not r['stopped']:
            frame = self.store.descriptor(self.latest)
            data = encoder.original.u.read_bytes(self.store.path/frame['file'],256*1024)
            need(encoder.original.u.fsha(data) == frame['sha256'], 'actual current PNG bytes')
        return dict(raw=r['returned_raw'],rgb_bytes=data,end=r['end'],stopped=r['stopped'],episode_id=self.episode_id)

    def encode(self, boundary, rgb):
        frame = self.store.descriptor(self.latest)
        need(encoder.original.u.fsha(rgb) == frame['sha256'], 'same live captured bytes')
        before = old.state_seal(self.env.unwrapped); previous = self.meter.region; self.meter.region = 'encode'
        try:
            latent,_ = encoder.encode(self.model,self.torch,self.device,self.store.store,frame,boundary,
                                      lambda:self.store.store.check('encoding_arrays',0))
            need(old.state_seal(self.env.unwrapped) == before, 'encoding must not move environment')
        finally:
            self.meter.region = previous
        return dict(latent=latent,input_sha256=frame['sha256'],end=boundary,episode_id=self.episode_id)
