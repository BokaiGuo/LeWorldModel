"""Fixed saved-only ten-branch settling diagnostic; no physical or neural execution."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

NAME = 'agent_p3_settling_tail_summary_20261029_v1'
RECEIPT_SHA = '98713b56f633a89645a6124232cfa058881c5061b85e337c25dbe4f13e799a10'
RECEIPT_SCHEMA = 'agent_p3_settling_tail_physical_20261029_v1.consume'
FLAGS = ('P1_admission','P3_admission','full_C3_demonstrated','scientific_admission',
         'physical_release','training','freshness','independent_confirmation','equal_FLOPs','equal_total_compute')
ROLES = ((0,'C'),(0,'Z'),(0,'P'),(1,'Z'),(1,'P'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encode(x):
    return (json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def vector(v):
    require(type(v) is list and len(v)==7 and all(type(x) in (int,float) and math.isfinite(x) for x in v), 'finite native7')
    return [float(x) for x in v]


def metrics(state, goal):
    s,g=vector(state),vector(goal)
    d=[a-b for a,b in zip(s,g)]
    q=[x*x for x in d]
    require(all(math.isfinite(x) for x in q), 'finite squared residuals')
    total=math.fsum(q); agent=math.fsum(q[:2]); obj=math.fsum(q[2:4]); xy=math.fsum(q[:4]); vel=math.fsum(q[5:])
    angle=abs(d[4]); native_angle=min(angle,2*math.pi-angle)
    return dict(state7=s,goal7=g,residual7=d,squared_residual7=q,native7_squared=total,
        native7_distance=math.sqrt(total),agent_xy_squared=agent,agent_xy_error=math.sqrt(agent),
        object_xy_squared=obj,object_xy_error=math.sqrt(obj),joint_xy_squared=xy,joint_xy_error=math.sqrt(xy),
        raw_angle_error=angle,raw_angle_squared=q[4],wrapped_angle_error=abs(math.atan2(math.sin(d[4]),math.cos(d[4]))),
        native_angle_term=native_angle,agent_velocity_error=math.sqrt(vel),agent_velocity_squared=vel,
        success_formula_diagnostic=math.sqrt(xy)<20 and native_angle<math.pi/9)


AXES=('native7_distance','native7_squared','agent_xy_error','agent_xy_squared','object_xy_error',
      'object_xy_squared','joint_xy_error','joint_xy_squared','raw_angle_error','raw_angle_squared',
      'wrapped_angle_error','native_angle_term','agent_velocity_error','agent_velocity_squared')


def analyze(receipt):
    require(receipt['schema']==RECEIPT_SCHEMA and receipt['fixed_denominator']==10, 'receipt identity and denominator')
    require(all(receipt[k] is False for k in FLAGS) and receipt['new_neural_calls']==0, 'scientific boundary')
    jobs=receipt['jobs']; require(type(jobs) is list and len(jobs)==10, 'all ten jobs')
    branches=[]
    for i, entry in enumerate(jobs):
        row=entry.get('row'); root,role=ROLES[i//2]; probe=('U','Z2')[i%2]
        if row is None:
            branches.append(dict(index=i,root_slot=root,role=role,probe=probe,status=entry.get('status','unpublished_failure'),
                error=entry.get('error'),goal7=None,checkpoints={'15':None,'20':None},cp20_minus_cp15=None))
            continue
        job=row['job']; require((job['index'],job['endpoint'],job['root_slot'],job['source_role'],job['probe'])==
            (i,i//2,root,role,probe), 'fixed job identity/order')
        require(row['status'] in ('horizon_completed','native_stopped','engineering_failed'), 'status')
        require(all(row[k] is False for k in FLAGS) and row['new_neural_calls']==0 and row['fixture_only'] is False, 'row boundary')
        ops=[o for o in row['operations'] if o['resource']=='public_step' and o.get('phase')=='plan' and o['status']=='returned']
        require([o['step'] for o in ops]==list(range(1,len(ops)+1)), 'returned plan sequence')
        goal=vector(ops[0]['result']['goal_state']) if ops else None
        require(all(vector(o['result']['goal_state'])==goal for o in ops), 'unchanged full goal7')
        cp={}
        for step in (15,20):
            ep=row['endpoint'+str(step)]
            returned=[o['result'] for o in ops if o['step']==step]
            listed=[x for x in row['checkpoints'] if x['step']==step]
            require(len(listed)<=1 and ep==(listed[0] if listed else None), 'checkpoint receipt identity')
            if ep is None:
                cp[str(step)]=None
                continue
            require(len(returned)==1 and goal is not None and ep['state']==returned[0]['state'] and ep['snapshot']==returned[0]['snapshot'], 'checkpoint native return identity')
            v=returned[0]; require(type(v['terminated']) is bool and type(v['truncated']) is bool, 'native flags')
            m=metrics(ep['state'],goal)
            require(math.isclose(-v['reward'],m['native7_distance'],rel_tol=1e-12,abs_tol=1e-12), 'native reward arithmetic')
            cp[str(step)]=dict(metrics=m,native_reward=v['reward'],native_success=v['terminated'],native_truncated=v['truncated'],
                png_sha256=ep['file_sha256'],pixel_sha256=ep['pixel_sha256'])
        temporal=None if cp['15'] is None or cp['20'] is None else {
            k:cp['20']['metrics'][k]-cp['15']['metrics'][k] for k in AXES}
        branches.append(dict(index=i,root_slot=root,role=role,probe=probe,status=row['status'],error=row['error'],
                             goal7=goal,checkpoints=cp,cp20_minus_cp15=temporal))
    pairs=[]
    for i,(root,role) in enumerate(ROLES):
        u,z=branches[2*i:2*i+2]
        if u['goal7'] is not None and z['goal7'] is not None: require(u['goal7']==z['goal7'], 'same pair goal7')
        for step in ('15','20'):
            a,b=u['checkpoints'][step],z['checkpoints'][step]
            diffs=None if a is None or b is None else {k:b['metrics'][k]-a['metrics'][k] for k in AXES}
            pairs.append(dict(endpoint=i,root_slot=root,role=role,step=int(step),
                direction='Z2 minus U; positive means lower U error',axis_gains=diffs,
                squared_residual7_gains=None if diffs is None else [y-x for x,y in zip(a['metrics']['squared_residual7'],b['metrics']['squared_residual7'])],
                native_success_U=None if a is None else a['native_success'],native_success_Z2=None if b is None else b['native_success']))
    return dict(schema=NAME+'.analysis',fixed_branches=10,fixed_endpoints=5,fixed_checkpoint_slots=20,fixed_pair_slots=10,
        interpretation='post-hoc time extension; not original gate or independent roots',branches=branches,pairs=pairs,
        scientific_admission=False,P1_admission=False,P3_admission=False,full_C3_demonstrated=False)


def canonical(path):
    p=Path(path); require(p.is_absolute() and str(p.resolve())==str(p), 'canonical absolute path'); return p


def build(receipt_path):
    p=canonical(receipt_path); require(p.is_file() and not p.is_symlink() and sha(p)==RECEIPT_SHA, 'pinned complete receipt SHA')
    require(p.stat().st_size<=128*1024**2, 'bounded receipt')
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'duplicate key'); out[k]=v
        return out
    with p.open('rb') as f: receipt=json.load(f,object_pairs_hook=pairs,parse_constant=lambda _:require(False,'nonfinite JSON'))
    require(sha(p)==RECEIPT_SHA, 'receipt stable while reading')
    result=analyze(receipt); result['receipt']={'path':str(p),'sha256':RECEIPT_SHA}
    root=Path(__file__).resolve().parents[2]
    paths=[Path(__file__).resolve(),root/'tests'/('test_'+NAME+'.py'),root/'innerdocs'/(NAME+'.md')]
    result['source_closure']={str(q):sha(q) for q in paths}
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=('build','verify'))
    parser.add_argument('--receipt',required=True); parser.add_argument('--output',required=True)
    a=parser.parse_args(); result=encode(build(a.receipt)); out=canonical(a.output)
    if a.command=='build':
        with out.open('xb') as f: f.write(result); f.flush(); os.fsync(f.fileno())
    else:
        require(out.is_file() and not out.is_symlink() and out.read_bytes()==result,'whole saved analysis mismatch')
    print(hashlib.sha256(result).hexdigest())


if __name__=='__main__': main()
