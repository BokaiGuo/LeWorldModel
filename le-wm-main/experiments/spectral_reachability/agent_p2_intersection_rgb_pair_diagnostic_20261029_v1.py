"""Independent saved-only fixed18 same-root/same-time diagnostic."""
import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np

NAME='agent_p2_intersection_rgb_pair_diagnostic_20261029_v1'
RGB_RECEIPT='3797fb394ad1a693272bcf21d70ea59c9c8c344ad4aab35be3462ee298ad3087'
PHYSICAL_RECEIPT='e96462847db675dc0b0c5376067df3654edd40805d6e374ffa3f8f1d5224e6c4'
RGB_REPORT='9aa521a0f6befee544e6109b756a0193ee88a8624d172e839ecffae9fe748efa'
ORDER=((0,'joint'),(0,'composition'),(0,'grasp'),(1,'grasp'),(1,'composition'),(1,'joint'))
CP=(5,10,15)
PAIRS=(('joint','composition'),('joint','grasp'),('composition','grasp'))


def need(v,msg):
    if not v: raise ValueError(msg)


def sha(data): return hashlib.sha256(data).hexdigest()


def read(path, expected, cap=128*1024**2):
    p=Path(path);need(p.is_absolute() and p.is_file() and not p.is_symlink() and p.stat().st_size<=cap,'absolute bounded regular input')
    b=p.read_bytes();need(sha(b)==expected,'input SHA mismatch');return b


def vector(v,n):
    need(isinstance(v,list) and len(v)==n and all(type(x) in (int,float) and math.isfinite(x) for x in v),'finite vector')
    return [float(x) for x in v]


def q(a,b):
    a=vector(a,192);b=vector(b,192)
    return math.fsum((x-y)**2 for x,y in zip(a,b,strict=True))/192


def native(state,goal,response):
    s=vector(state,7);g=vector(goal,7);d=[a-b for a,b in zip(s,g,strict=True)];v=[x*x for x in d]
    xy=math.fsum(v[:4]);ang=min(abs(d[4]),2*math.pi-abs(d[4]));total=math.fsum(v)
    result=dict(residual7=d,axis_squared=v,agent_xy_squared=math.fsum(v[:2]),block_xy_squared=math.fsum(v[2:4]),
        joint_xy_squared=xy,raw_angle_squared=v[4],agent_velocity_squared=math.fsum(v[5:]),
        native7_squared=total,native7_distance=math.sqrt(total),native_angle_term=ang,
        wrapped_angle_error=abs(math.atan2(math.sin(d[4]),math.cos(d[4]))),
        native_success=math.sqrt(xy)<20 and ang<math.pi/9,returned_reward=response['reward'],
        returned_success=response['native_success'])
    need(type(response['native_success']) is bool and result['native_success'] is response['native_success'],'native success identity')
    result['distance_minus_returned_negative_reward']=result['native7_distance']+response['reward']
    return result


def sign(v): return None if v is None else (1 if v>0 else -1 if v<0 else 0)


def load_latent(folder, desc, inventory):
    need(set(desc)=={'file','sha256','array_sha256','dtype','shape'} and desc['dtype']=='float32' and desc['shape']==[192],'latent descriptor')
    name=desc['file'];need(isinstance(name,str) and Path(name).name==name and inventory.get(name)==desc['sha256'],'artifact inventory')
    compressed=read(folder/name,desc['sha256'],2*1024**2)
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as f:
        data=f.read(1024*1024+1)
    need(len(data)<=1024*1024,'bounded expanded NPY')
    stream=io.BytesIO(data);a=np.load(stream,allow_pickle=False)
    need(stream.tell()==len(data) and a.dtype==np.float32 and a.shape==(192,) and np.isfinite(a).all() and sha(a.tobytes())==desc['array_sha256'],'latent raw bytes and shape')
    return a.tolist()


def analyze(rgb,report,physical,load):
    need(rgb['schema']=='agent_p2_intersection_rgb_20261029_v1.consume' and rgb['status']=='fixed24_saved_rgb_verified_not_admitted'
        and rgb['report_sha256']==RGB_REPORT and rgb['denominator']==24 and rgb['known_roots']==2 and rgb['fresh_roots']==0
        and rgb['P1_admission'] is False and rgb['P2_admission'] is False,'RGB strict receipt')
    need(report['schema']=='agent_p2_intersection_rgb_20261029_v1.report' and report['completed'] is True and report['errors']==[]
        and report['prepare_sha256']==rgb['prepare_sha256'] and rgb['counts']==report['counts'],'RGB report identity')
    need(physical['schema']=='agent_p2_intersection_physical_20261029_v2.consume' and physical['fixed_denominator']==6
        and physical['known_roots']==2 and physical['fresh_roots']==0 and physical['errors']==[],'physical strict receipt')
    need(len(rgb['rows'])==len(report['rows'])==24 and len(physical['rows'])==len(physical['plans'])==6,'fixed roster')
    result=[]
    for i,(root,arm) in enumerate(ORDER):
        plan=physical['plans'][i];pr=physical['rows'][i]
        need((pr['index'],pr['root_slot'],pr['arm'])==(i,root,arm) and pr['plan']==plan and plan['selection_digest']==pr['selection_digest'],'physical plan identity')
        for j,cp in enumerate((0,*CP)):
            k=4*i+j;rr=rgb['rows'][k];sr=report['rows'][k]
            need((rr['index'],rr['slot'],rr['root'],rr['arm'],rr['checkpoint'])==(k,i,root,arm,cp),'RGB fixed identity')
            for field in ('index','slot','root','arm','checkpoint','predicted192','goal192','native_goal','native_response','frame','latent','prefix_bits','raw15_bits','selection_digest','prediction_role','missing_reason'):
                need(rr[field]==sr[field],'RGB report and receipt identity '+field)
            context=plan['root_context'] if cp==0 else plan['contexts'][j-1]
            need(rr['predicted192']==context['latent_history'][-1] and rr['goal192']==plan['goal'] and rr['native_goal']==pr['goal']
                and rr['raw15_bits']==plan['raw_float32_bits'] and rr['selection_digest']==plan['selection_digest']
                and rr['prefix_bits']==plan['raw_float32_bits'][:cp*16],'plan raw/context/goal identity')
            need(rr['prediction_role']==('observed_root_context_not_forecast' if cp==0 else 'same_plan_forecast'),'prediction role')
            frame=next((x for x in pr['checkpoints'] if x['step']==cp),None)
            response=next((x for x in pr['responses'] if x['step']==cp),None) if cp else None
            need(rr['native_response']==response,'same-step returned response')
            if frame is None:
                need(rr['frame'] is None and rr['latent'] is None and rr['missing_reason'] is not None,'missing remains null')
            else:
                need(rr['frame'] is not None and rr['latent'] is not None and rr['missing_reason'] is None and {key:rr['frame'][key] for key in frame}==frame and frame['state']==frame['snapshot']['state'],'frame/state identity')
                if cp: need(response is not None and response['state']==frame['state'],'same-time native7')
            model=q(rr['predicted192'],rr['goal192']);need(model==rr['model_Q'],'independent model Q')
            latent=None if rr['latent'] is None else load(rr['latent'])
            actual=None if latent is None else q(latent,rr['goal192']);error=None if latent is None else q(rr['predicted192'],latent)
            need(actual==rr['actual_Q'] and error==rr['model_actual_MSE'],'independent encoded Q')
            cross=None if latent is None else 2*math.fsum((float(x)-float(g))*(float(y)-float(x)) for x,y,g in zip(latent,rr['predicted192'],rr['goal192'],strict=True))/192
            nm=None if response is None else native(response['state'],rr['native_goal'],response)
            result.append(dict(index=k,slot=i,root=root,arm=arm,checkpoint=cp,model_Q=model,actual_Q=actual,
                prediction_actual_MSE=error,model_minus_actual_Q=None if actual is None else model-actual,
                cross_term=cross,decomposition_roundoff=None if actual is None else (model-actual)-(error+cross),
                native=nm,missing_reason=rr['missing_reason']))
    pairs=[]
    for root in (0,1):
        plans=[physical['plans'][i] for i,(r,_) in enumerate(ORDER) if r==root]
        need(all(all(p[k]==plans[0][k] for k in ('task_id','task_file','task_sha256','root_context','goal','q','warm_raw10','native_root')) for p in plans),'same root task/context/goal')
        for cp in CP:
            for a,b in PAIRS:
                left=next(x for x in result if (x['root'],x['arm'],x['checkpoint'])==(root,a,cp))
                right=next(x for x in result if (x['root'],x['arm'],x['checkpoint'])==(root,b,cp))
                md=right['model_Q']-left['model_Q']
                ad=None if left['actual_Q'] is None or right['actual_Q'] is None else right['actual_Q']-left['actual_Q']
                ln,rn=left['native'],right['native']
                nd=None if ln is None or rn is None else {key:rn[key]-ln[key] for key in ('native7_squared','native7_distance','joint_xy_squared','agent_xy_squared','block_xy_squared','raw_angle_squared','agent_velocity_squared','native_angle_term','wrapped_angle_error')}
                if nd is not None: nd['axis_squared']=[y-x for x,y in zip(ln['axis_squared'],rn['axis_squared'],strict=True)]
                pairs.append(dict(root=root,checkpoint=cp,left=a,right=b,model_delta_b_minus_a=md,actual_delta_b_minus_a=ad,native_delta_b_minus_a=nd,
                    model_sign=sign(md),actual_sign=sign(ad),native_distance_sign=sign(None if nd is None else nd['native7_distance']),
                    model_actual_sign_agreement=None if ad is None else sign(md)==sign(ad),
                    actual_native_sign_agreement=None if nd is None or ad is None else sign(ad)==sign(nd['native7_distance']),
                    native_success_left=None if ln is None else ln['native_success'],native_success_right=None if rn is None else rn['native_success'],
                    comparable_rgb=ad is not None,comparable_native=nd is not None))
    return dict(schema=NAME+'.analysis',rows=result,pairs=pairs,denominator=dict(slots=6,positions=24,pairs=18,
        rgb_comparable=sum(x['comparable_rgb'] for x in pairs),native_comparable=sum(x['comparable_native'] for x in pairs)),
        known_roots=2,fresh_roots=0,P1_admission=False,P2_admission=False,causal_claim=False,selection=False,
        interpretation='b minus a; negative means smaller error/Q; latent Q and native mixed-unit distance are distinct; ties zero, missing null; inherited strict, no rerun')


def build(rgb_path,report_path,physical_path):
    rgb=json.loads(read(rgb_path,RGB_RECEIPT));report=json.loads(read(report_path,RGB_REPORT));physical=json.loads(read(physical_path,PHYSICAL_RECEIPT))
    need(rgb['bindings'].get(str(physical_path))==PHYSICAL_RECEIPT,'RGB binds physical strict receipt')
    folder=Path(report_path).parent
    result=analyze(rgb,report,physical,lambda d:load_latent(folder,d,report['artifact_sha256']))
    result['sources']=dict(rgb_receipt=dict(path=str(rgb_path),sha256=RGB_RECEIPT),rgb_report=dict(path=str(report_path),sha256=RGB_REPORT),physical_receipt=dict(path=str(physical_path),sha256=PHYSICAL_RECEIPT))
    return result


def encoded(x): return (json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('analyze','verify'))
    for k in ('rgb-receipt','rgb-report','physical-receipt','output'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--output-sha');a=ap.parse_args()
    data=encoded(build(a.rgb_receipt,a.rgb_report,a.physical_receipt));out=Path(a.output)
    need(out.is_absolute() and out.name.startswith(NAME+'_') and not out.is_symlink(),'exclusive own absolute output')
    if a.command=='analyze':
        need(a.output_sha is None and len(data)<=16*1024**2 and shutil.disk_usage(out.parent).free>=200*1024**2+len(data),'bounded new output')
        with out.open('xb') as f:need(f.write(data)==len(data),'short write');f.flush();os.fsync(f.fileno())
        fd=os.open(out.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
        try:os.fsync(fd)
        finally:os.close(fd)
    else:need(read(out,a.output_sha)==data,'entire saved result mismatch')
    print(a.command,sha(data))


if __name__=='__main__':main()
