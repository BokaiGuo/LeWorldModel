"""Fixed five-pair cp15 three-layer and post-hoc cp20 native join; saved-only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

NAME='agent_p3_capability_tail_join_20261029_v1'
THREE_SHA='36b31985a27a16156afb3dea8c852b84199ea1814cfcba986c5b889db39bb01c'
TAIL_SHA='e190a357ec02243baa635b84d28cf0382310dcec31b2048c8876ab45ac40dbb6'
ORDER=((0,'C'),(0,'Z'),(0,'P'),(1,'Z'),(1,'P'))


def need(x,msg):
    if not x: raise ValueError(msg)


def read(path,h):
    p=Path(path);need(p.is_absolute() and p.is_file() and not p.is_symlink() and p.stat().st_size<=128*1024**2,'bounded absolute input')
    b=p.read_bytes();need(hashlib.sha256(b).hexdigest()==h,'input SHA mismatch');return json.loads(b)


def vec(v,n):
    need(isinstance(v,list) and len(v)==n and all(type(x) in (int,float) and math.isfinite(x) for x in v),'finite vector')
    return [float(x) for x in v]


def q(a,b):
    return math.fsum((x-y)**2 for x,y in zip(vec(a,192),vec(b,192),strict=True))/192


def metrics(point,goal):
    if point is None:return None
    m=point['metrics'];s=vec(m['state7'],7);g=vec(goal,7);need(m['goal7']==goal,'original goal7 identity')
    d=[x-y for x,y in zip(s,g,strict=True)];sq=[x*x for x in d]
    angle=min(abs(d[4]),2*math.pi-abs(d[4]));xy=math.fsum(sq[:4]);total=math.fsum(sq)
    success=math.sqrt(xy)<20 and angle<math.pi/9
    need(point['native_success'] is success,'native success identity')
    return dict(state7=s,goal7=g,residual7=d,squared_residual7=sq,joint_xy_squared=xy,
        agent_xy_squared=math.fsum(sq[:2]),object_xy_squared=math.fsum(sq[2:4]),
        raw_angle_squared=sq[4],agent_velocity_squared=math.fsum(sq[5:]),native7_squared=total,
        native7_distance=math.sqrt(total),native_angle_term=angle,
        wrapped_angle_error=abs(math.atan2(math.sin(d[4]),math.cos(d[4]))),native_success=success,
        native_reward=point['native_reward'],pixel_sha256=point['pixel_sha256'],png_sha256=point['png_sha256'])


def sign(x):return None if x is None else ('positive' if x>0 else 'negative' if x<0 else 'tie')


def analyze(three,tail):
    need(three['schema']=='agent_p3_capability_three_layer_summary_20261029_v1.summary' and
         tail['schema']=='agent_p3_settling_tail_summary_20261029_v1.analysis','actual summary schemas')
    need(three['known_roots']==2 and three['fresh_roots']==0 and three['fixture_only'] is False and
         all(three[k] is False and tail[k] is False for k in ('P1_admission','P3_admission','full_C3_demonstrated','scientific_admission')),'scientific boundary')
    need(three['fixed_denominators']==dict(C_control_pairs=8,C_logical_slots=4,C_no_witness=3,branches=10,same_context_pairs=5) and len(three['branches'])==len(tail['branches'])==10 and
         len(three['pairs'])==5 and tail['fixed_branches']==10 and tail['fixed_pair_slots']==10 and len(tail['pairs'])==10,'fixed complete rosters')
    rows=[]
    for i,(root,role) in enumerate(ORDER):
        for j,probe in enumerate(('U','Z2')):
            k=2*i+j;t=three['branches'][k];b=tail['branches'][k]
            need((t['index'],t['endpoint'],t['root_slot'],t['role'],t['probe'],t['checkpoint'])==(k,i,root,role,probe,15)
                 and (b['index'],b['root_slot'],b['role'],b['probe'])==(k,root,role,probe),'fixed branch identity')
            c15=b['checkpoints']['15'];c20=b['checkpoints']['20'];src=t['source'];nm=t['native_metrics']
            need((c15 is None)==(src is None)==(nm is None),'cp15 availability identity')
            if c15 is not None:
                need(src['state']==c15['metrics']['state7']==nm['state7'] and src['snapshot']['state']==src['state'] and
                    c15['metrics']['goal7']==nm['goal7']==b['goal7'] and src['pixel_sha256']==c15['pixel_sha256'] and
                    src['file_sha256']==c15['png_sha256'] and t['native_success'] is c15['native_success'],'same cp15 state/goal/pixel/png/success')
            pred=t['predicted_latent192'];actual=t['actual_latent192'];goal=t['goal192']
            rows.append(dict(index=k,endpoint=i,root=root,role=role,probe=probe,
                cp15_model_Q=None if pred is None else q(pred,goal),cp15_real_Q=None if actual is None else q(actual,goal),
                cp15_native=metrics(c15,b['goal7']),cp20_native=metrics(c20,b['goal7']),
                cp20_model_Q=None,cp20_real_Q=None,physical_status=b['status'],error=b['error']))
    pairs=[]
    keys=('native7_distance','native7_squared','joint_xy_squared','agent_xy_squared','object_xy_squared','raw_angle_squared','agent_velocity_squared')
    for i,(root,role) in enumerate(ORDER):
        u,z=rows[2*i:2*i+2];a=three['pairs'][i]
        need((a['index'],a['root_slot'],a['role'],a['U_job'],a['Z2_job'],a['same_parent_context'])==(i,root,role,2*i,2*i+1,True),'same-context original pair')
        mg=None if u['cp15_model_Q'] is None or z['cp15_model_Q'] is None else z['cp15_model_Q']-u['cp15_model_Q']
        rg=None if u['cp15_real_Q'] is None or z['cp15_real_Q'] is None else z['cp15_real_Q']-u['cp15_real_Q']
        need(a['model_gain']['value']==mg and a['real_latent_gain']['value']==rg,'independently reconstructed same-context gains')
        gains={}
        for cp in (15,20):
            x,y=u[f'cp{cp}_native'],z[f'cp{cp}_native']
            gains[str(cp)]=None if x is None or y is None else {k:y[k]-x[k] for k in keys}
        changes=None if gains['15'] is None or gains['20'] is None else {k:gains['20'][k]-gains['15'][k] for k in keys}
        pairs.append(dict(endpoint=i,root=root,role=role,model_gain_cp15=mg,real_gain_cp15=rg,
            native_gain=gains,change_cp20_minus_cp15=changes,model_sign=sign(mg),real_sign=sign(rg),
            native15_sign=sign(None if gains['15'] is None else gains['15']['native7_distance']),
            native20_sign=sign(None if gains['20'] is None else gains['20']['native7_distance'])))
    return dict(schema=NAME+'.analysis',rows=rows,pairs=pairs,denominator=dict(branches=10,endpoints=5,pair_times=10),
        C_logical_slots=three['original_C_dual_control_conditions'],C_control_pairs=three['original_C_control_pairs'],
        known_roots=2,fresh_roots=0,P1_admission=False,P3_admission=False,full_C3_demonstrated=False,
        scientific_admission=False,selection=False,causal_claim=False,
        interpretation='post-hoc uniform five-step zero-raw continuation; gain is Z2 minus U; cp20 has no saved model/RGB; no gate substitution')


def build(three_path,tail_path):
    r=analyze(read(three_path,THREE_SHA),read(tail_path,TAIL_SHA))
    r['sources']=dict(three=dict(path=str(three_path),sha256=THREE_SHA),tail=dict(path=str(tail_path),sha256=TAIL_SHA))
    return r


def encoded(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('build','verify'))
    for x in ('three','tail','output'):ap.add_argument('--'+x,required=True)
    ap.add_argument('--output-sha');a=ap.parse_args();data=encoded(build(a.three,a.tail));p=Path(a.output)
    need(p.is_absolute() and p.name.startswith(NAME+'_') and not p.is_symlink(),'own absolute output')
    if a.command=='build':
        need(a.output_sha is None and len(data)<=16*1024**2 and shutil.disk_usage(p.parent).free>=200*1024**2+len(data),'bounded exclusive output')
        with p.open('xb') as f:need(f.write(data)==len(data),'complete write');f.flush();os.fsync(f.fileno())
        fd=os.open(p.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
        try:os.fsync(fd)
        finally:os.close(fd)
    else:need(encoded(read(p,a.output_sha))==data,'whole saved reconstruction')
    print(a.command,hashlib.sha256(data).hexdigest())


if __name__=='__main__':main()
