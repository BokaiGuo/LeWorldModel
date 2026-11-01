"""Saved-only commoncontroller native scalar comparison; no producer imports."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np

NAME='agent_p2_commoncontroller_physical_comparison_20261029_v1'
PRODUCER='agent_p2_commoncontroller_physical_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
RUNNER=ROOT/'experiments/spectral_reachability'/(PRODUCER+'.py')
RUNNER_SHA='7e715a9005a4408fbc7899184f338aa4ae6ee6b20dd414356744aa830ceff5f0'
PROTOCOL=ROOT/'innerdocs/agent_p2_commoncontroller_physical_protocol_20261029_v1.md'
PROTOCOL_SHA='42e082498484f8bb399f819d31531c2a77f73b161e0bd2e15bb8ec900f5e8de6'
ACTIVE=(0,1,3,4)
ARMS=('prior_FIRST','program_first_postcondition','arbitrary')
CP=(0,5,10,15)
TIMES=(*CP,'native_terminal')
FLAGS=('P1_admission','P2_admission','P2_mechanism','scientific_admission','freshness',
       'equal_FLOPs','equal_total_compute','independent_confirmation','physical_release')
SCALARS=('native7_distance','native7_reward','native7_squared','joint_xy_error','joint_xy_squared',
    'agent_xy_error','agent_xy_squared','object_xy_error','object_xy_squared','raw_angle_error',
    'raw_angle_squared','native_angle_term','wrapped_angle_error','velocity_error','velocity_squared')
CAP,META,FLOOR=16*1024**2,128*1024**2,200*1024**2


def need(ok,label):
    if not ok:raise ValueError(label)


def encode(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_value(x):
    need(type(x) is str and len(x)==64 and all(c in '0123456789abcdef' for c in x),'literal actual SHA256')


def absolute(x):
    p=Path(x)
    need(p.is_absolute() and str(p.resolve())==str(p),'canonical absolute path')
    return p


def decode(b):
    def pairs(items):
        out={}
        for k,v in items:need(k not in out,'duplicate JSON key');out[k]=v
        return out
    def bad(_):raise ValueError('nonfinite JSON')
    value=json.loads(b,object_pairs_hook=pairs,parse_constant=bad);encode(value);return value


def read_bound(path,h):
    p=absolute(path);hash_value(h)
    need(p.is_file() and not p.is_symlink() and p.stat().st_size<=META,'bounded regular saved JSON')
    b=p.read_bytes();need(len(b)<=META and hashlib.sha256(b).hexdigest()==h,'external exact saved SHA')
    return decode(b)


def vec7(x):
    need(type(x) is list and len(x)==7 and all(type(v) in (int,float) for v in x),'literal native7 no bool')
    a=np.asarray(x,np.float64);need(np.isfinite(a).all(),'finite native7')
    return a


def metrics(state,goal):
    """Independent float64 native7; native min-angle is NOT wrapped diagnostic."""
    s,g=vec7(state),vec7(goal);d=s-g;sq=d*d
    norm=lambda v:float(np.linalg.norm(v))
    xy=norm(d[:4]);angle=float(min(abs(d[4]),2*np.pi-abs(d[4])))
    wrapped=float(abs(np.arctan2(np.sin(d[4]),np.cos(d[4]))))
    distance=norm(d);success=bool(xy<20 and angle<np.pi/9)
    native=dict(native_distance=distance,native_success=success,native_angle_term=angle,
        negative_native_angle=bool(angle<0),object_pose=s[2:5].tolist(),object_position_error=norm(d[2:4]),
        object_success=None,wrapped_allxy_success=bool(xy<20 and wrapped<np.pi/9),wrapped_object_angle_error=wrapped)
    m=dict(native7_distance=distance,native7_reward=-distance,native7_squared=float(np.sum(sq,dtype=np.float64)),
        joint_xy_error=xy,joint_xy_squared=float(np.sum(sq[:4],dtype=np.float64)),
        agent_xy_error=norm(d[:2]),agent_xy_squared=float(np.sum(sq[:2],dtype=np.float64)),
        object_xy_error=norm(d[2:4]),object_xy_squared=float(np.sum(sq[2:4],dtype=np.float64)),
        raw_angle_error=float(abs(d[4])),raw_angle_squared=float(sq[4]),native_angle_term=angle,
        wrapped_angle_error=wrapped,velocity_error=norm(d[5:]),velocity_squared=float(np.sum(sq[5:],dtype=np.float64)),
        native_success=success,wrapped_success_diagnostic=native['wrapped_allxy_success'],
        negative_native_angle=native['negative_native_angle'],state7=s.tolist(),goal7=g.tolist(),residual7=d.tolist(),native=native)
    need(all(np.isfinite(m[k]) for k in SCALARS),'finite all scalar metrics')
    components={k:m[k] for k in ('joint_xy_squared','raw_angle_squared','velocity_squared')}
    summed=float(sum(components.values()));residual=m['native7_squared']-summed
    # Two nonnegative reductions of the SAME seven float64 squared coordinates.
    # gamma16 safely bounds both summation paths; not a scientific tolerance.
    eps=np.finfo(np.float64).eps;gamma=16*eps/(1-16*eps)
    bound=float(gamma*(abs(m['native7_squared'])+sum(abs(v) for v in components.values())))
    need(abs(residual)<=bound,'independent native7 squared axis identity')
    m['native7_squared_decomposition']=dict(native7_squared=m['native7_squared'],**components,
        sum_of_three_axes=summed,identity_residual=residual,rounding_bound=bound,
        identity='native7_squared = joint_xy_squared + raw_angle_squared + velocity_squared')
    return m


def breakdown(m):
    if m is None:return None
    return dict(native=deepcopy(m['native']),native_success=m['native_success'],joint_xy_error=m['joint_xy_error'],
        angle_native_term=m['native_angle_term'],angle_wrapped_error=m['wrapped_angle_error'],
        velocity_error=m['velocity_error'],native7_distance=m['native7_distance'],native7_reward=m['native7_reward'])


def rawbits(plan):
    value=plan['commands'];need(type(value) is list and len(value)==15
        and all(type(r) is list and len(r)==2 and all(type(v) in (int,float) for v in r) for r in value),'literal raw15')
    a=np.asarray(value,np.float32)
    need(np.isfinite(a).all() and np.all(np.abs(a)<=1)
        and all(float(x)==float(y) for x,y in zip(np.asarray(value,np.float64).flat,a.flat)),'lossless native float32 raw15')
    need(a.tobytes().hex()==plan['raw_float32_bits']
        and hashlib.sha256(a.tobytes()).hexdigest()==plan['raw_float32_sha256'],'original action bytes not reconstructed')
    return a


def validate(report,strict,report_path,report_sha,prepared,prepare_sha):
    """Import full strict authority and join all rows; NEVER rerun physical consume."""
    need(report['schema']==PRODUCER+'.report' and strict['schema']==PRODUCER+'.consume'
        and report['status']==strict['status']=='commoncontroller4_full15_native_recorded_not_admitted',
        'complete own physical report and own actual strict schema')
    for obj in (report,strict):
        need(obj['fixed_denominator']==len(obj['rows'])==6 and obj['known_roots']==2
            and obj['fresh_roots']==obj['inherited']==0 and obj['executed_slots']==list(ACTIVE)
            and obj['missing_slots']==[2,5] and obj['errors']==[]
            and all(type(i) is int for i in [*obj['executed_slots'],*obj['missing_slots']])
            and all(obj[k] is False for k in FLAGS),'full six denominator, known-only flags and errors')
    need(report['inputs_final_verified'] is True and report['new_neural_calls']==report['new_independent_samples']==0
        and strict['new_neural_calls']==strict['new_physics_calls']==0
        and strict['physical_report_sha256']==report_sha
        and encode(strict['rows'])==encode(report['rows']) and strict['counts']==report['counts']
        and encode(strict['planner_authority'])==encode(report['execution_binding']['planner_authority']),
        'entire report/strict rows, counts, planner authority and external report SHA join')
    binding=report['execution_binding']
    need(binding['prepared']==str(prepared) and binding['prepare_sha256']==prepare_sha
        and strict['bindings'][str(report_path)]==report_sha and strict['bindings'][str(prepared)]==prepare_sha
        and strict['bindings'][str(RUNNER)]==RUNNER_SHA and strict['bindings'][str(PROTOCOL)]==PROTOCOL_SHA,
        'actual source/protocol/report/prepare bindings, not status-only')
    need(len(strict['metric_breakdown'])==6,'all six metric breakdown entries')
    frames=0;public=0
    for i,row in enumerate(report['rows']):
        need(type(row['index']) is int and row['index']==i and type(row['root_slot']) is int
            and (row['root_slot'],row['condition'],row['arm'])==(i//3,'off',ARMS[i%3]),
            'literal full fixed slot order and arm identity')
        b=strict['metric_breakdown'][i]
        need(b['root_slot']==i//3 and b['arm']==ARMS[i%3],'all breakdown identities and order')
        if i not in ACTIVE:
            need(row['status']=='missing' and row['checkpoints']==[] and row['missing_checkpoints']==list(CP)
                and row['final_state'] is row['final_metrics'] is row['native_success'] is row['terminated'] is row['truncated'] is None
                and row['initial_history_guard'] is None and row['plan']['commands'] is None
                and all(row[k]==0 for k in ('own_attempted','own_returned','warmup_attempted','warmup_returned','PNG'))
                and b['metrics'] is None,'arbitrary missing no fabricated execution or endpoints')
            continue
        need(row['status'] in ('horizon_completed','native_stopped') and row['first_error'] is None
            and row['secondary_errors']==[],'complete branch or genuine native stop')
        a=rawbits(row['plan'])
        need(row['task_id']==row['plan']['task_id'] and row['selection_digest']==row['plan']['selection_digest']
            and row['root_slot']==row['plan']['root_slot'] and row['arm']==row['plan']['arm'],'full own plan/task identity')
        n=row['own_returned'];warm=row['warmup_returned']
        need(type(n) is type(warm) is int and 0<=n<=15 and 0<=warm<=10
            and n==row['own_attempted'] and warm==row['warmup_attempted'],'actual complete unpadded return counts')
        need([v['step'] for v in row['responses']]==list(range(1,n+1)),'all contiguous actual returned response steps')
        cp=row['checkpoints'];steps=[v['step'] for v in cp]
        need(steps==sorted(set(steps)) and all(k in CP for k in steps)
            and row['missing_checkpoints']==[k for k in CP if k not in steps]
            and steps==([k for k in CP if k<=n] if row['initial_exact'] else []),
            'all fixed checkpoints and missing retained, no dropped observed checkpoint')
        if n:
            need(warm==10 and row['initial_exact'] is True and row['initial_history_guard'] is not None,'actual initial guard for plan execution')
        for f in cp:
            need(f['state']==f['snapshot']['state'],'full saved checkpoint snapshot-state join')
            if f['step']==0:need(encode(f['state'])==encode(row['root_state']),'real cp0 root state')
            else:need(f['step']<=n and encode(f['state'])==encode(row['responses'][f['step']-1]['state']),'checkpoint exact same returned time')
        for response in row['responses']:
            m=metrics(response['state'],row['goal'])
            need(encode(response['native_metrics'])==encode(m['native']) and response['reward']==m['native7_reward']
                and response['native_success'] is response['terminated'] is m['native_success']
                and type(response['truncated']) is bool,'all returned native scalar labels recomputed')
        if n:
            last=row['responses'][-1];m=metrics(row['final_state'],row['goal'])
            need(encode(row['final_state'])==encode(last['state']) and encode(row['final_metrics'])==encode(m['native'])
                and all(row[k] is last[k] for k in ('native_success','terminated','truncated'))
                and all(not r['terminated'] and not r['truncated'] for r in row['responses'][:-1])
                and ((row['status']=='native_stopped')==(last['terminated'] or last['truncated'])),'real endpoint/native stop not cp15 substitution')
        else:
            need(row['final_state'] is row['final_metrics'] is row['native_success'] is None,'unobserved own endpoint stays unknown')
            m=None
        need(encode(b['metrics'])==encode(breakdown(m)),'entire six-entry metric_breakdown independently recomputed')
        if row['status']=='horizon_completed':need(n==15,'complete actual full15 not truncated history')
        frames+=row['PNG'];public+=warm+n
        need(row['PNG']==3*((warm+1)+(n+1 if row['initial_exact'] else 0)),'all actual rendered triplets accounted')
    need(strict['saved_PNG_verified']==frames==report['counts']['explicit_render']['completed']
        and report['counts']['public_step']['completed']==public,'full actual PNG and public census')
    counts=report['counts'];expected=dict(constructor=4,close=4,reset=8,set_state=16,goal_render=8,
        goal_bind=sum(report['rows'][i]['initial_exact'] for i in ACTIVE),
        public_step=public,physics_step=16+10*public,explicit_render=frames)
    expected['goal_setter']=8+expected['goal_bind']
    need(set(counts)==set(expected) and all(counts[k]==dict(attempted=v,completed=v)
        and all(type(counts[k][field]) is int for field in ('attempted','completed')) for k,v in expected.items()),
        'complete multi-axis native lifecycle imported from strict, not predict-only')
    return True


def panel(report):
    rows=[]
    for i,branch in enumerate(report['rows']):
        checkpoint={x['step']:x for x in branch['checkpoints']}
        for time in TIMES:
            terminal=time=='native_terminal';f=None if terminal else checkpoint.get(time)
            state=branch['final_state'] if terminal else (None if f is None else f['state'])
            step=branch['own_returned'] if terminal and state is not None else (None if terminal else time)
            m=None if state is None else metrics(state,branch['goal'])
            baseline=None if i not in ACTIVE or not branch['initial_exact'] else metrics(branch['root_state'],branch['goal'])
            rows.append(dict(slot=i,root_slot=i//3,arm=ARMS[i%3],time=time,actual_plan_step=step,
                branch_status=branch['status'],metrics=m,
                missing_reason=None if m is not None else ('arbitrary_missing' if i not in ACTIVE else 'native_stop_or_checkpoint_not_observed'),
                checkpoint=None if f is None else {k:deepcopy(f[k]) for k in ('step','png','file_sha256','pixel_sha256')},
                native_success=None if m is None else m['native_success'],
                terminated=branch['terminated'] if terminal else (None if not time or m is None else branch['responses'][time-1]['terminated']),
                truncated=branch['truncated'] if terminal else (None if not time or m is None else branch['responses'][time-1]['truncated']),
                error_reduction_from_cp0=None if m is None or baseline is None else
                    {k:baseline[k]-m[k] for k in SCALARS if k!='native7_reward'},
                native7_reward_gain=None if m is None or baseline is None else m['native7_reward']-baseline['native7_reward']))
    pairs=[]
    for root in (0,1):
        left,right=report['rows'][3*root],report['rows'][3*root+1]
        need(all(left['plan'][k]==right['plan'][k] for k in ('task_id','task_sha256','blind_sha256','native_goal',
            'native_root_sha256','warm_raw10_bits','history_pixel_sha256','goal_pixel_sha256')),
            'same-root pair exact task/history/original goal authority, targets may differ')
        if left['checkpoints'] and right['checkpoints'] and left['checkpoints'][0]['step']==right['checkpoints'][0]['step']==0:
            l,r=left['checkpoints'][0],right['checkpoints'][0]
            need(encode(l['state'])==encode(r['state']) and encode(l['snapshot'])==encode(r['snapshot'])
                and l['pixel_sha256']==r['pixel_sha256'] and l['file_sha256']==r['file_sha256'],'cp0 same full snapshot/state/pixels control')
        for time in TIMES:
            l,r=[next(x for x in rows if x['slot']==i and x['time']==time) for i in (3*root,3*root+1)]
            known=l['metrics'] is not None and r['metrics'] is not None
            delta=None if not known else {k:r['metrics'][k]-l['metrics'][k] for k in SCALARS}
            squared=None
            if delta is not None:
                axes={k:delta[k] for k in ('joint_xy_squared','raw_angle_squared','velocity_squared')}
                total=float(sum(axes.values()));residual=delta['native7_squared']-total
                scale=sum(abs(x['metrics'][k]) for x in (l,r)
                    for k in ('native7_squared','joint_xy_squared','raw_angle_squared','velocity_squared'))
                eps=np.finfo(np.float64).eps;bound=float(32*eps/(1-32*eps)*scale)
                need(abs(residual)<=bound,'independent paired squared margin axis identity')
                squared=dict(delta_native7_squared=delta['native7_squared'],delta_components=axes,
                    sum_of_axis_deltas=total,identity_residual=residual,rounding_bound=bound)
            pairs.append(dict(root_slot=root,time=time,prior_slot=3*root,program_slot=3*root+1,comparable=known,
                prior_actual_plan_step=l['actual_plan_step'],program_actual_plan_step=r['actual_plan_step'],
                same_control_time=known and l['actual_plan_step']==r['actual_plan_step'],
                delta_program_minus_prior=delta,
                squared_margin_decomposition=squared,
                direction_program_minus_prior=None if delta is None else
                    {k:('positive' if v>0 else 'negative' if v<0 else 'tie') for k,v in delta.items()},
                program_error_reduction_vs_prior=None if delta is None else {k:-v for k,v in delta.items() if k!='native7_reward'},
                prior_native_success=l['native_success'],program_native_success=r['native_success'],
                native_success_delta=None if not known else int(r['native_success'])-int(l['native_success']),
                missing_slots=[x['slot'] for x in (l,r) if x['metrics'] is None]))
    return dict(rows=rows,pairs=pairs,fixed_slot_denominator=6,fixed_time_denominator=5,
        fixed_scalar_row_denominator=30,fixed_pair_denominator=10,known_roots=2,fresh_roots=0,
        physical_counts=deepcopy(report['counts']),original_wall_seconds=report['wall_seconds'],
        paid_replay_receipt=deepcopy(report['execution_binding']['planner_authority']['receipt']),
        difference_convention='program minus prior: negative distance/error numerically smaller; positive reward/success larger; exact ties separate; native min-angle is not wrapped physical angle',
        terminal_convention='native own endpoint at its actual returned step; never fill missing cp15; terminal steps may differ',
        interpretation='two known roots; no fresh/timefair/equal-generation-budget/P1/P2 admission',
        **{k:False for k in FLAGS})


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def analyze(report,report_sha,receipt,receipt_sha):
    rp,sp=absolute(report),absolute(receipt)
    r,s=read_bound(rp,report_sha),read_bound(sp,receipt_sha)
    b=r['execution_binding'];pp=absolute(b['prepared']);p=read_bound(pp,b['prepare_sha256'])
    need(p['schema']==PRODUCER+'.prepare' and encode(p['plans'])==encode([row['plan'] for row in r['rows']])
        and encode(p['planner_authority'])==encode(b['planner_authority'])
        and p['source_report_sha256']==b['source_report_sha256'],'entire actual prepare/plan/authority join')
    for i in ACTIVE:
        row=r['rows'][i];root=p['source_roots'][i//3]
        need(row['plan']['native_goal']==root['goal'] and row['plan']['native_root_sha256']==root['root_sha256'],
            'actual original native goal/root identity')
        if row['initial_exact']:
            need(encode(row['root_state'])==encode(root['root_state']) and encode(row['goal'])==encode(root['goal']),
                'actual initial root/goal join to whole sealed prepare')
    validate(r,s,rp,report_sha,pp,b['prepare_sha256'])
    need(sha(RUNNER)==RUNNER_SHA and sha(PROTOCOL)==PROTOCOL_SHA,'frozen physical/protocol source bytes')
    result=dict(schema=NAME+'.summary',arguments=dict(report=str(rp),report_sha=report_sha,receipt=str(sp),receipt_sha=receipt_sha),
        authentication='imported SHA-bound complete own physical receipt; no native fullstrict reexecution',
        arithmetic='independent NumPy float64 native7 and per-axis scalar decomposition; no producer math import',
        summary=panel(r),bindings={str(rp):report_sha,str(sp):receipt_sha,str(pp):b['prepare_sha256'],
            str(RUNNER):RUNNER_SHA,str(PROTOCOL):PROTOCOL_SHA},
        new_neural_calls=0,new_physics_calls=0,new_fullstrict_calls=0)
    for f in own_files():result['bindings'][str(f)]=sha(f)
    for f,h in result['bindings'].items():need(sha(f)==h,'bound input/source unchanged after saved analysis')
    return result


def write_new(output,value):
    p=absolute(output);b=encode(value)
    need(p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink() and len(b)<=CAP
        and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'exclusive16MiB summary +200MiBfloor')
    with p.open('xb') as f:need(f.write(b)==len(b),'complete saved write');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(p),sha256=sha(p),bytes=len(b))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('analyze','verify'))
    for k in ('report','report-sha','receipt','receipt-sha','output','summary','summary-sha'):p.add_argument('--'+k)
    a=p.parse_args(argv)
    if a.command=='analyze':result=write_new(a.output,analyze(a.report,a.report_sha,a.receipt,a.receipt_sha))
    else:
        saved=read_bound(a.summary,a.summary_sha);need(encode(saved)==encode(analyze(**saved['arguments'])),'whole saved summary exact recomputation')
        result=dict(verified=True,summary_sha256=a.summary_sha)
    print(encode(result).decode(),end='')


if __name__=='__main__':main()
