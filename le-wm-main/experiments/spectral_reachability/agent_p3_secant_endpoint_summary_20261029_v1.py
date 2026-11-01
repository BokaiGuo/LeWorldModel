"""Independent saved control10/absolute20 native arithmetic; MAIN receipts trusted."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import time

NAME='agent_p3_secant_endpoint_summary_20261029_v1'
CNAME='agent_p3_secant_commit_physical_20261029_v1'
ZNAME='agent_p3_secant_controls_first5_physical_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
META,CAP,FLOOR,WALL=128*1024**2,4*1024**2,200*1024**2,120
ACTUAL={
 'C':(CNAME+'_consume_MAIN01.json','d0c5a34b14dc84ffece573d904e3950e54628df307987ee77fcb6265ca91b638'),
 'ZP':(ZNAME+'_consume_MAIN01.json','c58ca1e43439841a3e592edb1128f95e2b5b8375af8f686052ab662930e74b50')}
SOURCE_PINS={
 CNAME+'.py':'a065c8f652919d8315f7848c70e9b944ad6aec0717bc6bbe310b54b224c72aea',
 ZNAME+'.py':'e9403bd9608dcd43bde780b538e60d24711c51ca568e012083d855d47e47096c',
 'agent_mixedmode_physical_20261028_v1.py':'e4fac2479254db07c34de5d9f88fbe0d19606b8ce2b1f8d7d5e42cbad8a042ae',
 'agent_kinodynamic_matched_physical_consume_20261028_v1.py':'92f31a292a76a88e0e93bd908ac377f2430bb760b8e5b972758717d570f3241b'}
FALSE_FLAGS=('P1_admission','P3_admission','scientific_admission','full_C3_demonstrated',
 'freshness','independent_confirmation','equal_FLOPs','equal_total_compute','training','physical_release','execution_authorized')
ORDER_C=((0,'H1'),(0,'H3'),(1,'H1'),(1,'H3'))
ORDER_ZP=((0,'Z'),(0,'P'),(1,'Z'),(1,'P'))
AXES=('agent_x','agent_y','block_x','block_y','block_angle_mod2pi','agent_vx','agent_vy')
ERRORS=('native7_distance','native7_squared','joint_xy_error','joint_xy_squared','agent_xy_error',
 'block_xy_error','agent_velocity_error','raw_angle_error','raw_angle_squared','wrapped_angle_error',
 'agent_xy_squared','block_xy_squared','agent_velocity_squared')


def need(ok,msg):
    if not ok:raise ValueError(msg)


def encode(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def equal(a,b):return encode(a)==encode(b)
def sha(p):
    p=Path(p);need(p.is_file() and not p.is_symlink() and p.stat().st_size<=META,'bounded regular metadata/source')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def read(p,h):
    need(sha(p)==h,'exact external receipt/output SHA')
    data=Path(p).read_bytes();need(hashlib.sha256(data).hexdigest()==h,'stable metadata read')
    def unique(pairs):
        d={}
        for k,v in pairs:need(k not in d,'duplicate JSON key');d[k]=v
        return d
    value=json.loads(data,object_pairs_hook=unique,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    encode(value);return value


def actual_refs():return {k:dict(path=str(ROOT/'results'/p),sha256=h) for k,(p,h) in ACTUAL.items()}


def closure():
    out={}
    for filename,h in SOURCE_PINS.items():
        p=ROOT/'experiments/spectral_reachability'/filename;need(sha(p)==h,'native formula/source evidence pin');out[str(p)]=h
    for p in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):out[str(p)]=sha(p)
    return out


def vec7(v):
    need(type(v) is list and len(v)==7 and all(type(x) in (int,float) and math.isfinite(x) for x in v),'finite numeric state7/goal7, no bool')
    return [float(x) for x in v]


def metrics(state,goal):
    s,g=vec7(state),vec7(goal);d=[x-y for x,y in zip(s,g,strict=True)];sq=[x*x for x in d]
    need(all(math.isfinite(x) for x in (*d,*sq)),'finite f64 native arithmetic')
    total=math.fsum(sq);xy=math.fsum(sq[:4]);agent=math.fsum(sq[:2]);block=math.fsum(sq[2:4]);velocity=math.fsum(sq[5:])
    angle=abs(d[4]);native=min(angle,2*math.pi-angle);wrapped=abs(math.atan2(math.sin(d[4]),math.cos(d[4])))
    groups=dict(agent_xy_squared=agent,block_xy_squared=block,raw_angle_squared=sq[4],agent_velocity_squared=velocity)
    return dict(state7=s,goal7=g,axis_order=list(AXES),signed_residual7=d,absolute_residual7=[abs(x) for x in d],squared_residual7=sq,
        native7_squared=total,native7_distance=math.sqrt(total),native7_reward=-math.sqrt(total),
        joint_xy_squared=xy,joint_xy_error=math.sqrt(xy),agent_xy_squared=agent,agent_xy_error=math.sqrt(agent),
        block_xy_squared=block,block_xy_error=math.sqrt(block),agent_velocity_squared=velocity,agent_velocity_error=math.sqrt(velocity),
        raw_angle_error=angle,raw_angle_squared=sq[4],native_angle_term=native,negative_native_angle=native<0,
        wrapped_angle_error=wrapped,native_success=bool(math.sqrt(xy)<20 and native<math.pi/9),
        wrapped_success_diagnostic=bool(math.sqrt(xy)<20 and wrapped<math.pi/9),
        squared_groups=groups,group_roundoff_residual=total-math.fsum(groups.values()))


def compare(c,v,larger_is_worse=True):
    if c is None or v is None:return dict(C_minus_control=None,C_worse=None,control_worse=None,tie=None,status='null')
    need(type(c) in (int,float) and type(v) in (int,float) and math.isfinite(c) and math.isfinite(v),'finite comparison')
    delta=c-v;need(math.isfinite(delta),'finite comparison difference')
    return dict(C_minus_control=delta,C_worse=(c>v if larger_is_worse else c<v),
        control_worse=(v>c if larger_is_worse else v<c),tie=c==v,status='compared')


def endpoint(row):
    cp=[f for f in row['checkpoints'] if f['step']==10];need(len(cp)<=1,'unique actual cp10')
    if row['status']=='no_witness':
        need(row['plan']['commands'] is row['final_state'] is row['final_metrics'] is row['native_success'] is None and not row['checkpoints'] and not row['responses'] and
            row['own_attempted']==row['own_returned']==row['PNG']==0,'no_witness no borrowed endpoint')
        return None
    available=row['status'] in ('horizon_completed','native_stopped') and row['own_returned']==10 and len(cp)==1
    if not available:return None
    f=cp[0];need(equal(f['state'],f['snapshot']['state']) and
        f['snapshot']['state_float64_bits']==struct.pack('<7d',*vec7(f['state'])).hex(),'exact cp10 state bytes')
    ret=[x for x in row['responses'] if x['step']==10];need(len(ret)==1,'actual control10 return')
    need(equal(f['state'],ret[0]['state']) and equal(row['final_state'],f['state']) and
        row['native_success'] is ret[0]['native_success'],'actual same-time final/return/state')
    return f


def validate_receipt(v,which):
    name=CNAME if which=='C' else ZNAME
    need(v['schema']==name+'.consume' and v['status']==('secant_commit5_native_recorded_NOT_admitted' if which=='C' else 'all4_ZP_first5_native_recorded_NOT_admitted'),
        'exact complete receipt schema/status')
    need(v['errors']==[] and v['fixed_denominator']==len(v['rows'])==4 and v['known_roots']==2 and v['fresh_roots']==v['inherited']==0 and
        all(v[k] is False for k in FALSE_FLAGS),'original complete/science-closed known2 scope')
    need(v['completion']['status']=='published_within_cap' and v['completion']['report_sha256']==v['physical_report_sha256'],'receipt completion/report join')
    order=ORDER_C if which=='C' else ORDER_ZP
    for i,(row,key) in enumerate(zip(v['rows'],order,strict=True)):
        plan=row['plan'];need((row['index'],row['root_slot'],row['arm'],row['condition'])==(i,*key,'off') and
            (plan['index'],plan['root_slot'],plan['arm'],plan['condition'])==(i,*key,'off') and row['task_id']==plan['task_id'],'all4 fixed logical slots')
        need(plan['raw_steps']==(0 if row['status']=='no_witness' else 10) and plan['selection_uses_physics'] is False and
            plan['provenance']['actual_cp5_absolute_steps']==[5,10,15] and plan['provenance']['actual_cp5_control_steps']==[-5,0,5],
            'cp5 is control5/absolute15 history, not endpoint or baseline')
        vec7(plan['native_goal'])
        if which=='C':need(row['status']==('horizon_completed' if i==0 else 'no_witness'),'fixed C one executed/three no_witness')
        else:need(plan['control_role']==key[1] and row['status'] in ('horizon_completed','native_stopped'),'Z/P settled original role')
        if row['status']!='no_witness':
            need(equal(row['goal'],plan['native_goal']) and row['root_sha256']==plan['native_root_sha256'] and
                row['prefix_reference_exact'] is True,'native root/goal/prefix certification')


def summarize(c,z,refs):
    validate_receipt(c,'C');validate_receipt(z,'ZP')
    need(equal(z['planner_authority']['authorities']['C_receipt'],refs['C']),'ZP exact whole C authority join')
    logical=z['planner_authority']['logical_C_slots'];need(len(logical)==4 and len(z['endpoints'])==4,'whole logical C and ZP endpoint witnesses')
    ce=[endpoint(row) for row in c['rows']];ze=[endpoint(row) for row in z['rows']]
    for i,row in enumerate(c['rows']):
        need(equal(logical[i],dict(root_slot=row['root_slot'],arm=row['arm'],status=row['status'],endpoint=ce[i])),'C logical endpoint exact copy')
    for i,(row,f) in enumerate(zip(z['rows'],ze,strict=True)):
        need(equal(z['endpoints'][i],dict(index=i,root_slot=row['root_slot'],role=row['arm'],control_step=10,absolute_step=20,
            available=f is not None,checkpoint=f,missing_reason=None if f is not None else row['status'])),'ZP explicit actual time endpoint exact')
    joined=('task_id','local_task_id','source_slot','native_root_sha256','native_goal','native_goal_pose','native_goal_pixel_sha256',
        'prefix_commands','base_task','model_task','provenance')
    for root in (0,1):
        plans=[c['rows'][2*root+j]['plan'] for j in (0,1)]+[z['rows'][2*root+j]['plan'] for j in (0,1)]
        for plan in plans[1:]:need(all(equal(plan[k],plans[0][k]) for k in joined),'same original root/task/goal/context/provenance')
        # Real prefix equality only where physically observed; no synthetic C state for no_witness.
        observed=[row for row in c['rows'][2*root:2*root+2]+z['rows'][2*root:2*root+2] if row['status']!='no_witness']
        for cp in (0,5):
            frames=[[f for f in row['checkpoints'] if f['step']==cp] for row in observed]
            need(all(len(f)==1 for f in frames),'actual complete prefix checkpoints')
            first=frames[0][0]
            need(all(equal(f[0]['state'],first['state']) and f[0]['pixel_sha256']==first['pixel_sha256'] for f in frames),'same observed prefix RGB/state; not a benefit baseline')
    def item(row,f):
        m=None if f is None else metrics(f['state'],row['plan']['native_goal'])
        if m is not None:need(m['native_success'] is row['native_success'],'independent original native success rule')
        return dict(index=row['index'],root_slot=row['root_slot'],arm=row['arm'],status=row['status'],control_step=10,absolute_step=20,
            available=f is not None,missing_reason=None if f is not None else row['status'],task_id=row['task_id'],
            endpoint=f,metrics=m,source_plan_sha256=hashlib.sha256(encode(row['plan'])).hexdigest())
    cs=[item(row,f) for row,f in zip(c['rows'],ce,strict=True)];zs=[item(row,f) for row,f in zip(z['rows'],ze,strict=True)];pairs=[]
    for ci,cr in enumerate(cs):
        for role_index,role in enumerate(('Z','P')):
            zi=2*cr['root_slot']+role_index;zr=zs[zi];a,b=cr['metrics'],zr['metrics']
            available=a is not None and b is not None
            errors={k:compare(None if a is None else a[k],None if b is None else b[k]) for k in ERRORS}
            success=None if not available else dict(C=a['native_success'],control=b['native_success'],
                C_worse=(not a['native_success'] and b['native_success']),control_worse=(a['native_success'] and not b['native_success']),tie=a['native_success']==b['native_success'])
            pairs.append(dict(index=len(pairs),C_slot=ci,control_slot=zi,root_slot=cr['root_slot'],C_arm=cr['arm'],control_role=role,
                control_step=10,absolute_step=20,available=available,missing_reason=None if available else dict(C=cr['missing_reason'],control=zr['missing_reason']),
                errors=errors,reward=compare(None if a is None else a['native7_reward'],None if b is None else b['native7_reward'],False),
                native_success=success,coordinate_absolute_errors={axis:compare(None if a is None else a['absolute_residual7'][j],None if b is None else b['absolute_residual7'][j]) for j,axis in enumerate(AXES)},
                signed_state_C_minus_control=None if not available else [a['state7'][j]-b['state7'][j] for j in range(7)],
                native_angle_term_C_minus_control=None if not available else a['native_angle_term']-b['native_angle_term']))
    counts={k:{tag:sum(p['errors'][k][tag] is True for p in pairs) for tag in ('C_worse','control_worse','tie')} for k in ERRORS}
    for k in ERRORS:counts[k]['null']=sum(p['errors'][k]['status']=='null' for p in pairs)
    for key in ('reward','native_success'):
        counts[key]={tag:sum(p[key] is not None and p[key][tag] is True for p in pairs) for tag in ('C_worse','control_worse','tie')}
        counts[key]['null']=sum(p[key] is None or (key=='reward' and p[key]['status']=='null') for p in pairs)
    for axis in counts:need(sum(counts[axis].values())==8,'all8 per-axis comparison denominator')
    return dict(C_slots=cs,ZP_root_references=zs,comparisons=pairs,comparison_counts=counts,
        denominator=dict(C_logical_slots=4,C_no_witness=3,ZP_references=4,comparisons=8),
        actual_time=dict(control=10,absolute=20),baseline=None,cp5_is_baseline=False,
        winner=None,known_roots=2,fresh=False,scientific_admission=False,P1_admission=False,P3_admission=False,full_C3_demonstrated=False)


def build(refs=None,clock=time.monotonic):
    start=clock()
    def check():need(clock()-start<=WALL,'local saved whole120 wall')
    refs=actual_refs() if refs is None else refs;need(equal(refs,actual_refs()),'fixed two actual MAIN authorities')
    c,z=(read(refs[k]['path'],refs[k]['sha256']) for k in ('C','ZP'));check()
    sources={v['path']:v['sha256'] for v in refs.values()};sources.update(closure())
    result=summarize(c,z,refs)
    for path,h in sources.items():check();need(sha(path)==h,'source/receipt bytes unchanged')
    return dict(schema=NAME+'.summary',authorities=refs,local_sources=sources,**result,
        authentication='explicit MAIN complete receipts imported; NOT independently repeated fullstrict',
        arithmetic='independent scalar float64 subtraction/squares, math.fsum, sqrt; original native success rule',
        independent_fullstrict=False,new_neural_calls=0,new_physics_calls=0,new_AD_calls=0,FLOPs=None,
        inherited_accounting={k:dict(counts=v['counts'],saved_PNG_verified=v['saved_PNG_verified'],physical_report_sha256=v['physical_report_sha256']) for k,v in (('C',c),('ZP',z))})


def write(path,v):
    need(path is not None,'output required');p=Path(path).absolute();b=encode(v)
    need(p.parent.resolve()==ROOT/'results' and p.name.startswith(NAME+'_') and not p.is_symlink() and len(b)<=CAP and
        shutil.disk_usage(p.parent).free>=FLOOR+len(b),'own exclusive output4MiB/free200MiB')
    with p.open('xb') as f:need(f.write(b)==len(b),'complete write');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(p),sha256=sha(p))


def verify(path,h):
    v=read(path,h);need(equal(v,build(v['authorities'])),'entire saved endpoint summary reconstructed')
    return dict(verified=True,independent_fullstrict=False,new_neural_calls=0,new_physics_calls=0)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('build','verify'))
    p.add_argument('--output',type=Path);p.add_argument('--summary',type=Path);p.add_argument('--summary-sha256');a=p.parse_args(argv)
    start=time.monotonic();result=write(a.output,build()) if a.command=='build' else verify(a.summary,a.summary_sha256)
    need(time.monotonic()-start<=WALL,'inclusive120 read/arithmetic/fsync; failed artifact retained');print(encode(result).decode())


if __name__=='__main__':main()
