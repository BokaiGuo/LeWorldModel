"""Independent saved scalar arithmetic; imports no physical/model producer."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np

NAME='agent_p2_six_target_physical_scalar_summary_20261029_v1'
PRODUCER='agent_p2_six_target_physical_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
RUNNER=ROOT/'experiments/spectral_reachability'/(PRODUCER+'.py')
RUNNER_SHA='30990cedd608578679e255b867b2e1a28f12724574049ba7f415de67b90e7a32'
PROTOCOL=ROOT/'innerdocs/agent_p2_six_target_physical_comparison_protocol_20261029_v1.md'
PROTOCOL_SHA='6e45721309e9a1fe1107918c988c170daf65ebe032163d0beace73d2bcb118f0'
STATUS='six_target6_native_recorded_not_admitted'
CP=(0,5,10,15)
ARMS=('prior_FIRST','program_first_postcondition','arbitrary')
FLAGS=('P1_admission','P2_admission','P2_mechanism','scientific_admission','freshness',
       'equal_FLOPs','equal_total_compute','independent_confirmation','physical_release')
SCALARS=('native7_distance','native7_reward','native7_squared','joint_xy_error','joint_xy_squared',
         'agent_xy_error','agent_xy_squared','block_xy_error','block_xy_squared','raw_angle_error',
         'raw_angle_squared','native_angle_term','wrapped_angle_error','agent_velocity_error','agent_velocity_squared')
AXES=('joint_xy_squared','raw_angle_squared','agent_velocity_squared')
META,CAP,FLOOR=128*1024**2,16*1024**2,200*1024**2


def need(ok,label):
    if not ok:raise ValueError(label)


def encode(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def equal(a,b):return encode(a)==encode(b)


def absolute(x):
    p=Path(x);need(p.is_absolute() and str(p.resolve())==str(p),'canonical absolute path');return p


def read_bound(path,h):
    p=absolute(path)
    need(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h),'actual SHA256')
    need(p.is_file() and not p.is_symlink() and p.stat().st_size<=META,'bounded regular saved JSON')
    b=p.read_bytes();need(len(b)<=META and hashlib.sha256(b).hexdigest()==h,'external SHA256')
    def pairs(items):
        d={}
        for k,v in items:need(k not in d,'duplicate JSON key');d[k]=v
        return d
    def bad(_):raise ValueError('nonfinite JSON')
    x=json.loads(b,object_pairs_hook=pairs,parse_constant=bad);encode(x);return x


def vec(x):
    need(type(x) is list and len(x)==7 and all(type(v) in (int,float) for v in x),'literal native7')
    a=np.asarray(x,np.float64);need(np.isfinite(a).all(),'finite native7');return a


def metrics(state,goal):
    s,g=vec(state),vec(goal);d=s-g;q=d*d
    norm=lambda a:float(np.linalg.norm(a))
    xy=norm(d[:4]);angle=float(min(abs(d[4]),2*np.pi-abs(d[4])))
    wrapped=float(abs(np.arctan2(np.sin(d[4]),np.cos(d[4]))))
    success=bool(xy<20 and angle<np.pi/9)
    native=dict(native_distance=norm(d),native_success=success,native_angle_term=angle,
        negative_native_angle=bool(angle<0),object_pose=s[2:5].tolist(),object_position_error=norm(d[2:4]),
        object_success=None,wrapped_allxy_success=bool(xy<20 and wrapped<np.pi/9),wrapped_object_angle_error=wrapped)
    m=dict(native7_distance=norm(d),native7_reward=-norm(d),native7_squared=float(np.sum(q)),
        joint_xy_error=xy,joint_xy_squared=float(np.sum(q[:4])),agent_xy_error=norm(d[:2]),
        agent_xy_squared=float(np.sum(q[:2])),block_xy_error=norm(d[2:4]),block_xy_squared=float(np.sum(q[2:4])),
        raw_angle_error=float(abs(d[4])),raw_angle_squared=float(q[4]),native_angle_term=angle,
        wrapped_angle_error=wrapped,agent_velocity_error=norm(d[5:]),agent_velocity_squared=float(np.sum(q[5:])),
        native_success=success,wrapped_success_diagnostic=native['wrapped_allxy_success'],
        negative_native_angle=bool(angle<0),state=s.tolist(),goal=g.tolist(),residual7=d.tolist(),native=native)
    need(all(np.isfinite(m[k]) for k in SCALARS),'finite scalar arithmetic')
    m['squared_decomposition']=decomposition(m,m)
    return m


def decomposition(values,scale_values):
    total=float(sum(values[k] for k in AXES));residual=values['native7_squared']-total
    eps=np.finfo(np.float64).eps
    bound=float(32*eps/(1-32*eps)*sum(abs(scale_values[k]) for k in ('native7_squared',*AXES)))
    need(abs(residual)<=bound,'float64 squared decomposition rounding bound')
    return dict(total=values['native7_squared'],components={k:values[k] for k in AXES},
                sum_of_components=total,identity_residual=residual,rounding_bound=bound)


def breakdown(m):
    return None if m is None else dict(native=m['native'],native_success=m['native_success'],
        joint_xy_error=m['joint_xy_error'],angle_native_term=m['native_angle_term'],
        angle_wrapped_error=m['wrapped_angle_error'],velocity_error=m['agent_velocity_error'],
        native7_distance=m['native7_distance'],native7_reward=m['native7_reward'])


def rawbits(plan):
    x=plan['commands']
    need(type(x) is list and len(x)==15 and all(type(r) is list and len(r)==2
        and all(type(v) in (int,float) for v in r) for r in x),'literal raw15')
    a=np.asarray(x,np.float32)
    need(np.isfinite(a).all() and np.all(np.abs(a)<=1) and np.array_equal(np.asarray(x,np.float64),a.astype(np.float64))
        and a.tobytes().hex()==plan['raw_float32_bits'] and hashlib.sha256(a.tobytes()).hexdigest()==plan['raw_float32_sha256'],
        'lossless raw15 signed bits and SHA');return a


def validate(p,r,c,paths,hashes):
    """Receipt import and numeric joins, NOT a journal/PNG fullstrict reexecution."""
    pp,rp,_=paths;ph,rh,_=hashes
    need(p['schema']==PRODUCER+'.prepare' and r['schema']==PRODUCER+'.report'
        and c['schema']==PRODUCER+'.consume' and r['status']==c['status']==STATUS,'own-six actual schema and complete status')
    for x in (r,c):
        need(x['fixed_denominator']==len(x['rows'])==6 and x['executed_slots']==list(range(6)) and x['missing_slots']==[]
            and x['known_roots']==2 and x['fresh_roots']==x['inherited']==0 and x['errors']==[]
            and all(x[k] is False for k in FLAGS),'six denominator and non-admission flags')
    b=r['execution_binding']
    need(b['prepared']==str(pp) and b['prepare_sha256']==ph and b['source_report_sha256']==p['source_report_sha256']
        and equal(b['planner_authority'],p['planner_authority']) and equal(c['planner_authority'],p['planner_authority'])
        and c['physical_report_sha256']==rh and c['bindings'][str(rp)]==rh and c['bindings'][str(pp)]==ph
        and c['bindings'][str(RUNNER)]==RUNNER_SHA and c['bindings'][str(PROTOCOL)]==PROTOCOL_SHA,
        'explicit prepare/report/source/protocol/planner authority joins')
    need(r['inputs_final_verified'] is True and r['new_neural_calls']==r['new_independent_samples']==0
        and c['new_neural_calls']==c['new_physics_calls']==0 and equal(r['rows'],c['rows']) and equal(r['counts'],c['counts'])
        and equal(p['plans'],[x['plan'] for x in r['rows']]) and len(c['metric_breakdown'])==6,'complete rows/plans/resources, no status-only import')
    for i,row in enumerate(r['rows']):
        plan=p['plans'][i];root=p['source_roots'][i//3];rawbits(plan)
        need(type(row['index']) is int and row['index']==i and type(row['root_slot']) is int
            and (row['root_slot'],row['condition'],row['arm'])==(i//3,'off',ARMS[i%3])
            and (plan['index'],plan['root_slot'],plan['condition'],plan['arm'])==(i,i//3,'off',ARMS[i%3])
            and row['task_id']==plan['task_id'] and row['selection_digest']==plan['selection_digest']
            and equal(plan['native_goal'],root['goal']) and plan['native_root_sha256']==root['root_sha256'], 'fixed six task/raw/root identity')
        need(row['status'] in ('horizon_completed','native_stopped') and row['first_error'] is None and row['secondary_errors']==[], 'real branch completion or native stop')
        n,w=row['own_returned'],row['warmup_returned']
        need(type(n) is type(w) is int and 0<=n<=15 and 0<=w<=10 and row['own_attempted']==n and row['warmup_attempted']==w,
            'all actual unpadded returned counts')
        if row['initial_exact']:
            need(w==10 and equal(row['goal'],root['goal']) and equal(row['root_state'],root['root_state']), 'true native root/goal')
        if n:need(row['initial_exact'] is True and row['initial_history_guard'] is not None,'guarded actual dispatch')
        need([x['step'] for x in row['responses']]==list(range(1,n+1)),'all native responses')
        steps=[f['step'] for f in row['checkpoints']]
        need(steps==([s for s in CP if s<=n] if row['initial_exact'] else [])
            and row['missing_checkpoints']==[s for s in CP if s not in steps], 'exact checkpoints and missing positions')
        for f in row['checkpoints']:
            state=row['root_state'] if f['step']==0 else row['responses'][f['step']-1]['state']
            need(equal(f['state'],f['snapshot']['state']) and equal(f['state'],state),'cp native time/state/snapshot join')
            need(any(equal({k:t[k] for k in f},f) for t in row['trajectory'] if t['step']==f['step']), 'cp actual trajectory PNG pointer')
        for x in row['responses']:
            m=metrics(x['state'],row['goal'])
            need(equal(x['native_metrics'],m['native']) and x['reward']==m['native7_reward']
                and x['native_success'] is x['terminated'] is m['native_success'] and type(x['truncated']) is bool,'all returned native labels')
        if n:
            last=row['responses'][-1];m=metrics(row['final_state'],row['goal'])
            need(equal(row['final_state'],last['state']) and equal(row['final_metrics'],m['native'])
                and all(row[k] is last[k] for k in ('native_success','terminated','truncated'))
                and all(not x['terminated'] and not x['truncated'] for x in row['responses'][:-1])
                and ((row['status']=='native_stopped')==(last['terminated'] or last['truncated'])), 'true terminal and stop')
        else:
            m=None;need(all(row[k] is None for k in ('final_state','final_metrics','native_success','terminated','truncated')), 'no fabricated own endpoint')
        if row['status']=='horizon_completed':need(n==15,'full15 required')
        need(c['metric_breakdown'][i]==dict(root_slot=i//3,arm=ARMS[i%3],metrics=breakdown(m)), 'independent full metric breakdown')
        need(row['PNG']==3*(w+1+(n+1 if row['initial_exact'] else 0)),'actual frame census')
    counts=r['counts'];pub=sum(x['warmup_returned']+x['own_returned'] for x in r['rows'])
    goal=sum(x['initial_exact'] for x in r['rows']);png=sum(x['PNG'] for x in r['rows'])
    expected=dict(constructor=6,close=6,reset=12,set_state=24,goal_render=12,goal_setter=12+goal,
                  goal_bind=goal,public_step=pub,physics_step=24+10*pub,explicit_render=png)
    need(set(counts)==set(expected) and c['saved_PNG_verified']==png and all(counts[k]==dict(attempted=v,completed=v)
        and all(type(z) is int for z in counts[k].values()) for k,v in expected.items()),'full multi-axis lifecycle resources')


def panel(r):
    rows=[]
    for i,b in enumerate(r['rows']):
        cp={f['step']:f for f in b['checkpoints']};base=None if 0 not in cp else metrics(cp[0]['state'],b['goal'])
        for step in CP:
            f=cp.get(step);m=None if f is None else metrics(f['state'],b['goal'])
            rows.append(dict(slot=i,root_slot=i//3,arm=ARMS[i%3],step=step,metrics=m,
                missing_reason='checkpoint_not_reached' if m is None else None,checkpoint=deepcopy(f),
                terminated=None if not step or m is None else b['responses'][step-1]['terminated'],
                truncated=None if not step or m is None else b['responses'][step-1]['truncated'],
                cp0_error_reduction=None if m is None or base is None else {k:base[k]-m[k] for k in SCALARS if k!='native7_reward'},
                cp0_reward_gain=None if m is None or base is None else m['native7_reward']-base['native7_reward']))
    pairs=[]
    for root in (0,1):
        for l,t in ((0,1),(0,2),(1,2)):
            left,right=r['rows'][3*root+l],r['rows'][3*root+t]
            need(all(equal(left['plan'][k],right['plan'][k]) for k in ('task_id','task_sha256','blind_sha256','native_goal',
                'native_root_sha256','warm_raw10_bits','history_pixel_sha256','goal_pixel_sha256')), 'same-root task/history controls')
            if left['checkpoints'] and right['checkpoints']:
                need(equal(left['checkpoints'][0],right['checkpoints'][0]) or all(equal(left['checkpoints'][0][k],right['checkpoints'][0][k])
                    for k in ('step','state','snapshot','file_sha256','pixel_sha256')), 'cp0 same full snapshot/pixels control')
            for step in CP:
                a,b=(rows[i*4+CP.index(step)]['metrics'] for i in (3*root+l,3*root+t))
                delta=None if a is None or b is None else {k:b[k]-a[k] for k in SCALARS}
                scale=None if delta is None else {k:abs(a[k])+abs(b[k]) for k in ('native7_squared',*AXES)}
                pairs.append(dict(root_slot=root,step=step,left_slot=3*root+l,right_slot=3*root+t,comparable=delta is not None,
                    delta_right_minus_left=delta,directions=None if delta is None else {k:('positive' if v>0 else 'negative' if v<0 else 'tie') for k,v in delta.items()},
                    squared_margin_decomposition=None if delta is None else decomposition(delta,scale),
                    native_success_delta=None if delta is None else int(b['native_success'])-int(a['native_success']),
                    missing_slots=[i for i,m in ((3*root+l,a),(3*root+t,b)) if m is None]))
    return dict(rows=rows,pairs=pairs,denominator=dict(slots=6,checkpoints=24,pairs=24),
        comparable_pairs=sum(x['comparable'] for x in pairs),known_roots=2,fresh_roots=0,
        native_terminals=[dict(slot=i,actual_step=b['own_returned'] if b['final_state'] is not None else None,
            metrics=None if b['final_state'] is None else metrics(b['final_state'],b['goal'])) for i,b in enumerate(r['rows'])],
        physical_counts=deepcopy(r['counts']),physical_wall_seconds=r['wall_seconds'],
        difference_convention='right minus left; negative distance/error smaller, positive reward/success larger; native min-angle may be negative and is not wrapped angle',
        **{k:False for k in FLAGS})


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def analyze(prepared,prepare_sha,report,report_sha,strict,strict_sha):
    paths=list(map(absolute,(prepared,report,strict)));hashes=(prepare_sha,report_sha,strict_sha)
    p,r,c=[read_bound(f,h) for f,h in zip(paths,hashes)]
    need(sha(RUNNER)==RUNNER_SHA and sha(PROTOCOL)==PROTOCOL_SHA,'frozen native source/protocol pins')
    validate(p,r,c,paths,hashes)
    result=dict(schema=NAME+'.summary',arguments=dict(zip(('prepared','prepare_sha','report','report_sha','strict','strict_sha'),
        (str(paths[0]),prepare_sha,str(paths[1]),report_sha,str(paths[2]),strict_sha))),summary=panel(r),
        imported_planner_authority=deepcopy(p['planner_authority']),physical_wall_through_report_fsync_hash=c['wall_through_report_fsync_hash'],
        authentication='SHA-bound imported complete receipt with independent scalar/identity/resource joins; no fullstrict rerun',
        new_neural_calls=0,new_physics_calls=0,new_fullstrict_calls=0,
        bindings={**dict(zip(map(str,paths),hashes)),str(RUNNER):RUNNER_SHA,str(PROTOCOL):PROTOCOL_SHA})
    for f in own_files():result['bindings'][str(f)]=sha(f)
    for f,h in result['bindings'].items():need(sha(f)==h,'input/source unchanged')
    return result


def write_new(path,x):
    p=absolute(path);b=encode(x)
    need(p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink() and len(b)<=CAP
        and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'exclusive summary16MiB/floor200MiB')
    with p.open('xb') as f:need(f.write(b)==len(b),'complete write');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(output=str(p),sha256=sha(p))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('analyze','verify'))
    for k in ('prepared','prepare-sha','report','report-sha','strict','strict-sha','output','summary','summary-sha'):ap.add_argument('--'+k)
    a=ap.parse_args(argv)
    if a.command=='analyze':out=write_new(a.output,analyze(a.prepared,a.prepare_sha,a.report,a.report_sha,a.strict,a.strict_sha))
    else:
        saved=read_bound(a.summary,a.summary_sha);need(saved['schema']==NAME+'.summary','own summary schema')
        need(equal(saved,analyze(**saved['arguments'])),'whole saved summary reproducibility');out=dict(verified=True,sha256=a.summary_sha)
    print(encode(out).decode(),end='')


if __name__=='__main__':main()
