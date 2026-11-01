"""Instance-local K17 native guard seam; no environment or model execution."""
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
import numpy as np

NAME='agent_p3_c3_k17_live_guard_20261029_v1'
ORDER=[(s,k) for s in range(11) for k in (0,1,2)]
REF_ORDER=[(s,k) for s in range(11) for k in ((0,1,2) if s in (0,5,10) else (0,))]

def require(ok,message):
    if not ok: raise ValueError(message)

def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x): return hashlib.sha256(canonical(x)).hexdigest()
def bytehash(x): return hashlib.sha256(x).hexdigest()

def f32(x,shape,box=False):
    a=np.asarray(x)
    require(a.shape==shape and a.dtype.kind in 'fi' and np.isfinite(a).all(),'finite numeric shape')
    b=np.asarray(a,dtype='<f4')
    require(np.isfinite(b).all() and np.array_equal(a.astype(np.float64),b.astype(np.float64)),'literal float32 no conversion loss')
    if box: require(np.all(np.abs(b)<=1),'raw domain; no clip')
    return b

def root_hash(state,goal):
    a=np.asarray(state,dtype='<f8'); b=np.asarray(goal,dtype='<f8')
    require(a.shape==b.shape==(7,) and np.isfinite(a).all() and np.isfinite(b).all(),'native state7/goal7')
    return bytehash(a.tobytes()+b.tobytes())

def ref_map(trace):
    require(type(trace) is list and [(x['receipt']['step'],x['receipt']['overlay_index']) for x in trace]==REF_ORDER,
      'exact original 17-reference order')
    return {(x['receipt']['step'],x['receipt']['overlay_index']):x for x in trace}

def history_guard(*,identity,task,reference,live):
    """Caller must source-bind inputs; this validates their structure, not source provenance."""
    require(type(identity) is dict and set(identity)=={'slot','root','mode','K'} and
      (identity['slot'],identity['root'],identity['mode'],identity['K']) in
      ((3,0,'fixedH1',17),(4,0,'staticH3_progress',17),(5,0,'adaptive',17),
       (9,1,'fixedH1',17),(10,1,'staticH3_progress',17),(11,1,'adaptive',17)), 'fixed original six identities')
    require(reference['task_sha256']==digest(task) and reference['task_id']==task['binding']['task_id'] and
      reference['root_sha256']==task['binding']['root_sha256'],'same full root task')
    require(canonical(live['reset_identity'])==canonical(reference['reset_identity']),'actual reset/placement identity')
    frames=live['warm_frames']; require(type(frames) is list and len(frames)==33 and
      [(f['receipt']['step'],f['receipt']['overlay_index']) for f in frames]==ORDER,'complete ordered 33 live frames')
    refs=ref_map(reference['warm_trace']); unmatched=[]
    for item in frames:
        r=item['receipt']; s,k=r['step'],r['overlay_index']
        require(r['phase']=='warmup' and canonical(item['pre'])==canonical(item['post']),'nonmutating native capture')
        ref=refs.get((s,k))
        if ref is None and k==2: ref=refs[(s,0)]
        if ref is None:
            require(k==1 and s not in (0,5,10),'only unpaired original on frames'); unmatched.append(s); continue
        require(canonical(item['pre'])==canonical(ref['pre'])==canonical(ref['post']) and
          all(canonical(r[x])==canonical(ref['receipt'][x]) for x in
              ('phase','step','condition','file_sha256','pixel_sha256','state','snapshot')),
          'original native warm RGB/state/snapshot exact')
    require(unmatched==[1,2,3,4,6,7,8,9],'unpaired old on frames explicitly unknown')
    raw=f32(reference['warm_raw'],(10,2),box=True)
    ops=live['warm_returns']; require(type(ops) is list and len(ops)==10,'complete warm10')
    for i,op in enumerate(ops):
        require(op['step']==i+1 and op['status']=='returned' and op['actual_dtype']=='<f4' and op['requested_f32_hex']==raw[i].tobytes().hex()
          and op['actual_f32_hex']==raw[i].tobytes().hex() and op['terminated'] is False and op['truncated'] is False,
          'actual ordered raw warmup; no padding/early stop')
    t=task['root_context']; mean=f32(task['action_mean'],(2,)); std=f32(task['action_std'],(2,))
    require(np.all(std>0),'positive action std')
    tok=np.divide(np.subtract(raw,mean,dtype=np.float32),std,dtype=np.float32).reshape(2,10)
    require(np.isfinite(tok).all(),'finite normalized history')
    require(tok.tobytes()==f32(t['action_history'],(2,10)).tobytes(),'actual warm raw -> exact two task history tokens')
    require(canonical(live['root_state'])==canonical(reference['root_state']) and
      canonical(live['goal_state'])==canonical(reference['goal_state']) and
      root_hash(live['root_state'],live['goal_state'])==reference['root_sha256'],'actual native root/goal7')
    require(canonical(live['bound_goal'])==canonical(reference['bound_goal']), 'actual bound goal, pixels, pose and renderer')
    require(canonical(live['goal_frame'])==canonical(reference['goal_frame']),'bound goal off frame/state/snapshot')
    frame=live['goal_frame']; bound=live['bound_goal']
    require(canonical(frame['pre'])==canonical(frame['post'])==canonical(bound) and
      canonical(bound['goal_state'])==canonical(live['goal_state']) and
      canonical(frame['receipt']['state'])==canonical(live['root_state']) and
      canonical(frame['receipt']['snapshot'])==canonical(bound['snapshot']) and
      frame['receipt']['phase']=='plan' and frame['receipt']['step']==0 and frame['receipt']['overlay_index']==0,
      'current bound root, goal, snapshot and original off frame chain')
    return dict(schema=NAME+'.history_guard',identity=deepcopy(identity),root_task_sha256=digest(task),
      warm_trace_sha256=digest(frames),warm_raw_sha256=bytehash(raw.tobytes()),history_tokens_sha256=bytehash(tok.tobytes()),
      live_goal_sha256=digest(live['bound_goal']),live_goal_frame_sha256=digest(live['goal_frame']),
      old_unpaired_on_steps=unmatched,accepted=True,physical_release=False)

def action_guard(raw5,mean,std,saved_replay_tokens):
    """Saved replay token is required: no claim based only on recomputation."""
    raw=f32(raw5,(5,2),box=True); mu=f32(mean,(2,)); scale=f32(std,(2,))
    require(np.all(scale>0),'positive scales')
    tokens=np.divide(np.subtract(raw,mu,dtype=np.float32),scale,dtype=np.float32).reshape(10)
    require(np.isfinite(tokens).all(),'finite normalized outgoing token')
    replay=f32(saved_replay_tokens,(10,))
    require(tokens.tobytes()==replay.tobytes(),'saved actual replay action token exact; changed token requires paid replay')
    back=np.add(np.multiply(tokens.reshape(5,2),scale,dtype=np.float32),mu,dtype=np.float32)
    require(np.isfinite(back).all(),'finite inverse diagnostic')
    equal=back.tobytes()==raw.tobytes()
    return dict(schema=NAME+'.action_guard',raw_f32_hex=raw.tobytes().hex(),
      tokens_f32_hex=tokens.tobytes().hex(),inverse_f32_hex=back.tobytes().hex(),
      raw_inverse_bitexact=equal,actual_replay_tokens_exact=True,
      contract='dispatch literal original raw; inverse diagnostic is never substituted',physical_release=False)

class DurableFixtureJournal:
    """Minimal exclusive JSONL/fsync seam; production must provide bounded audited store."""
    def __init__(self,path):
        self.path=Path(path); self.stream=self.path.open('xb'); self.previous=None; self.index=0
        fd=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    def emit(self,kind,payload):
        e=dict(index=self.index,previous=self.previous,kind=kind,payload=payload)
        e['sha256']=digest(e); data=canonical(e)+b'\n'
        require(self.stream.write(data)==len(data),'journal write'); self.stream.flush(); os.fsync(self.stream.fileno())
        self.index+=1; self.previous=e['sha256']; return deepcopy(e)
    def close(self): self.stream.close()

class GuardedDispatch:
    """One-use instance guard for a future audited adapter, not a runnable adapter."""
    def __init__(self, journal): self.journal=journal; self.ready=False; self.used=False
    def arm(self, *, identity,task,reference,live,raw5,replay_tokens):
        require(not self.ready and not self.used,'single arm')
        try:
            h=history_guard(identity=identity,task=task,reference=reference,live=live)
            a=action_guard(raw5,task['action_mean'],task['action_std'],replay_tokens)
            self.journal.emit('guard_result',dict(accepted=True,history=h,action=a))
        except BaseException as exc:
            self.used=True
            self.journal.emit('guard_result',dict(accepted=False,error_type=type(exc).__name__))
            raise
        self.raw=bytes.fromhex(a['raw_f32_hex']); self.goal=canonical(live['bound_goal'])
        self.root=canonical(live['goal_frame']); self.ready=True
        return h,a
    def first(self, *, current_goal,current_frame,command,dispatch):
        require(self.ready and not self.used,'guard not armed or already used')
        self.used=True
        try:
            require(canonical(current_goal)==self.goal and canonical(current_frame)==self.root,'fresh goal/root immediately before first intent')
            raw=f32(command,(2,),box=True).tobytes(); require(raw==self.raw[:8],'literal first original action')
        except BaseException as exc:
            self.journal.emit('precommit_rejected',dict(error_type=type(exc).__name__))
            raise
        self.journal.emit('commit_intent',dict(raw_f32_hex=raw.hex(),step=1))
        result=None
        try:
            result=dispatch(np.frombuffer(raw,dtype='<f4').copy())
            canonical(result)
            require(type(result) is dict and result.get('actual_dtype')=='<f4' and
              result.get('actual_f32_hex')==raw.hex(),'actual native return action identity')
        except BaseException as exc:
            self.journal.emit('commit_failure',dict(error_type=type(exc).__name__,returned_value=result)); raise
        self.journal.emit('commit_return',dict(returned=True,receipt=result))
        return result
