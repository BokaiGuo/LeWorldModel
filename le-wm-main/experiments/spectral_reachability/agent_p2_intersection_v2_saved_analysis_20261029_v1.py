"""Fixed actual intersection-v2 six-plan saved-only arithmetic analysis."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
NAME='agent_p2_intersection_v2_saved_analysis_20261029_v1'
SOURCES={
 'receipt':('results/agent_p2_intersection_host_cuda_20261029_v2_consume_MAIN01.json','bcaba6c3c4cbdd05dafc475ad45ad859930efd3587fdb8e8adba47f7da7b7432'),
 'report':('results/agent_p2_intersection_host_cuda_20261029_v2_cuda_MAIN01/report.json','be3a7998251b8367ee713a27ee50da7553c97f518c3b200201b2d6232e4098b0'),
 'prepare':('results/agent_p2_intersection_host_cuda_20261029_v2_prepare_MAIN01.json','d0c8856463b6846438655c52c59d504aa4fbfd7b8439decb7e1599610ec08fd8')}
ORDER=[(0,'joint'),(0,'composition'),(0,'grasp'),(1,'grasp'),(1,'composition'),(1,'joint')]


def need(condition,message):
    if not condition:raise ValueError(message)


def sha(p):
    p=Path(p);need(p.is_file() and not p.is_symlink(),'regular file')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def raw(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def refs():return {k:dict(path=str(ROOT/p),sha256=h) for k,(p,h) in SOURCES.items()}


def read(ref):
    p=Path(ref['path']);need(p.stat().st_size<=32*1024**2 and sha(p)==ref['sha256'],'input size/SHA')
    return json.loads(p.read_bytes())


def vector(value):
    need(isinstance(value,list) and len(value)==192 and all(type(x) in (int,float) and math.isfinite(x) for x in value),'finite 192-vector')
    return [float(x) for x in value]


def analyze(receipt,report,prepare,authority):
    need(receipt['schema']=='agent_p2_intersection_host_cuda_20261029_v2.consume' and
         receipt['status']=='all6_joint_box_saved_verified_not_scientific_admission' and receipt['fixed_denominator']==6 and
         receipt['report_sha256']==authority['report']['sha256'],'actual full strict receipt authority')
    need(report['schema']=='agent_p2_intersection_host_cuda_20261029_v2.report' and report['status']=='recorded' and
         report['errors']==[] and report['prepare_sha256']==authority['prepare']['sha256'] and
         report['prepared']==authority['prepare']['path'],'report→prepare exact binding')
    need(prepare['schema']=='agent_p2_intersection_host_cuda_20261029_v2.prepare' and
         prepare['root_arm_order']==[list(x) for x in ORDER] and len(prepare['slots'])==2 and
         len(receipt['plans'])==len(receipt['accounts'])==len(report['rows'])==6,'fixed six/root order')
    need(receipt['missing_plan_indices']==[] and receipt['partial_arm_indices']==[],'all six complete; no filling')
    rows=[]
    for i,(root,arm) in enumerate(ORDER):
        p=receipt['plans'][i];a=receipt['accounts'][i];slot=prepare['slots'][root];r=report['rows'][i]
        need(p is not None and isinstance(p['contexts'],list) and len(p['contexts'])==3 and
             slot['root_slot']==root and slot['condition']=='off' and a['root']==root and a['arm']==arm and
             (r['index'],r['root_slot'],r['arm'])==(i,root,arm),'plan/account/root identity')
        need(a['used'] is not None and len(a['used'])==27 and a['used']['FLOPs'] is None and
             a['observed_local_used']==a['used'] and not a['transport_failed'],'27 complete resource axes')
        context=p['contexts'][-1];need(context['context_length']==3 and context['latent_dim']==192 and
            len(context['latent_history'])==3,'whole H3 final context')
        z=vector(context['latent_history'][-1]);goal=vector(slot['packet']['goal'])
        mse=math.fsum((x-y)*(x-y) for x,y in zip(z,goal,strict=True))/192
        rows.append(dict(slot=i,root=root,arm=arm,task_id=slot['task_id'],goal_pointer=f'/slots/{root}/packet/goal',
            final_pointer=f'/plans/{i}/contexts/2/latent_history/2',goal_sha256=hashlib.sha256(raw(goal)).hexdigest(),
            latent_sha256=hashlib.sha256(raw(z)).hexdigest(),final_latent_goal_mse64=mse,
            plan=p,resource_axes27=a['used'],side_axes=a['side'],elapsed_seconds=a['elapsed_seconds'],
            stop=a['stop'],wall_overrun=a['wall_overrun'],committed_units=len(a['committed']),
            complete_account=a))
    return dict(schema=NAME+'.analysis',authority=authority,rows=rows,fixed_denominator=6,
        arithmetic='JSON numeric values converted to binary64; math.fsum((latent-goal)**2)/192',
        authority_scope='inherited full saved strict receipt; no re-run of full strict or independent neural/AD',
        equal_FLOPs=False,total_FLOPs=None,physical_benefit=False,scientific_admission=False,winner=None)


def build():
    a=refs();value=analyze(*(read(a[k]) for k in ('receipt','report','prepare')),a)
    need(all(sha(v['path'])==v['sha256'] for v in a.values()),'stable source after analysis');return value


def write(path,value):
    p=Path(path).absolute();b=raw(value)
    need(p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink() and len(b)<=32*1024**2,'exclusive bounded report')
    with p.open('xb') as f:need(f.write(b)==len(b),'full report write');f.flush();os.fsync(f.fileno())
    return dict(path=str(p),sha256=sha(p))


def main():
    q=argparse.ArgumentParser();q.add_argument('command',choices=('build','verify'));q.add_argument('--output',required=True);q.add_argument('--sha256');a=q.parse_args()
    if a.command=='build':v=write(a.output,build())
    else:
        need(a.sha256 and sha(a.output)==a.sha256,'external report SHA')
        data=Path(a.output).read_bytes();value=json.loads(data)
        need(data==raw(value) and value==build(),'complete arithmetic and source reconstruction');v=dict(verified=True,sha256=a.sha256)
    print(json.dumps(v,sort_keys=True))


if __name__=='__main__':main()
