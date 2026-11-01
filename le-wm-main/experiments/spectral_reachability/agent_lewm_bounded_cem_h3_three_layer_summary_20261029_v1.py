"""Independent saved CEM eight-slot three-layer arithmetic; no upstream execution."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import numpy as np

NAME='agent_lewm_bounded_cem_h3_three_layer_summary_20261029_v1'
RGB='agent_lewm_bounded_cem_h3_rgb_20261029_v1'
PHY='agent_lewm_bounded_cem_h3_physical_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
CP=(0,5,10,15)
CAP=16*1024**2
RGB_SOURCE_SHA='e47c739def9cf170513494eb4bb6244c187f9e2fe4b5ab6be1983a6c3bee9b57'
RGB_RECEIPT_SHA='0e5d8736e07ffd07417b39c84471ebd68682e4bfb448e4459517d9a5fa3234fa'
RGB_PREPARE_SHA='c40be7df2f08872bd2d698f578862d671fa9ed9fa8a550f23d32471d711dd145'
RGB_REPORT_SHA='5695f9375b7e0a10028d7149da8bf406e9702570e8fe7f8680b80647ab29e8a2'
RGB_COMPLETION_SHA='9d9a1604deff303a8d54f53256aca7de4fc587444bf8907d87103afb243f4be4'
PHYSICAL_SHA='1e69a2fe082af3c5da32ec28ac8ba10b2375b5a593fa0bf69a60367d7cbc9b86'
PHYSICAL_PREP_SHA='d17ffc349dedd7d310764f8cb6d7c4c615c9d70bc2e85ee85cd3a5991db431a8'
FLAGS=('P1_admission','P2_admission','scientific_admission','physical_release','fair_ranking',
       'official_reproduction','equal_FLOPs','equal_total_compute','freshness')
KINDS=('encoder','projector','action_encoder','predict_API','predictor','pred_proj')


def need(ok,label):
    if not ok:raise ValueError(label)


def encode(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def equal(a,b):return encode(a)==encode(b)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def absolute(p):
    p=Path(p);need(p.is_absolute() and str(p.resolve())==str(p) and not p.is_symlink(),'canonical absolute path');return p


def read(path,h):
    p=absolute(path)
    need(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h),'external SHA256')
    need(p.is_file() and p.stat().st_size<=CAP,'bounded saved JSON');body=p.read_bytes()
    need(hashlib.sha256(body).hexdigest()==h,'exact saved bytes')
    def pairs(items):
        out={}
        for k,v in items:need(k not in out,'duplicate JSON key');out[k]=v
        return out
    def bad(_):raise ValueError('nonfinite JSON')
    x=json.loads(body,object_pairs_hook=pairs,parse_constant=bad);encode(x);return x


def vector(x,n,f32=False):
    need(type(x) is list and len(x)==n and all(type(v) in (int,float) and math.isfinite(v) for v in x),'finite vector, no bool')
    if f32:
        a=np.asarray(x,np.float64);need(np.array_equal(a,a.astype(np.float32).astype(np.float64)),'literal native float32')
    return x


def mse(a,b):return math.fsum((x-y)**2 for x,y in zip(a,b,strict=True))/192
def numpy_mse(a,b):return float(np.mean((np.asarray(a,np.float64)-np.asarray(b,np.float64))**2))
def sign(x):return 'null' if x is None else 'positive' if x>0 else 'negative' if x<0 else 'tie'


def compare(x,y):
    a,b=sign(x),sign(y)
    if 'null' in (a,b):return 'null'
    if a==b:return 'both_tie' if a=='tie' else 'same'
    if a=='tie':return 'left_tie'
    if b=='tie':return 'right_tie'
    return 'reversal'


def native(state,goal):
    s,g=vector(state,7),vector(goal,7);d=[x-y for x,y in zip(s,g,strict=True)];q=[x*x for x in d]
    need(all(math.isfinite(x) for x in q),'finite native squares')
    angle=min(abs(d[4]),2*math.pi-abs(d[4]));xy=math.fsum(q[:4]);total=math.fsum(q)
    # Exact native norm route separate from the independent fsum arithmetic.
    delta=np.asarray(s,np.float64)-np.asarray(g,np.float64)
    success=bool(np.linalg.norm(delta[:4])<20 and angle<math.pi/9)
    return dict(axis_residual=d,axis_squared=q,agent_xy_squared=math.fsum(q[:2]),
        object_xy_squared=math.fsum(q[2:4]),joint_xy_squared=xy,raw_angle_squared=q[4],
        velocity_squared=math.fsum(q[5:]),native7_squared=total,agent_xy_error=math.sqrt(math.fsum(q[:2])),
        object_xy_error=math.sqrt(math.fsum(q[2:4])),joint_xy_error=math.sqrt(xy),
        native7_distance=math.sqrt(total),native7_reward=-math.sqrt(total),
        native_angle_term=angle,wrapped_angle_error=abs(math.atan2(math.sin(d[4]),math.cos(d[4]))),
        negative_native_angle=angle<0,native_success=success,
        fsum_success_diagnostic=bool(math.sqrt(xy)<20 and angle<math.pi/9),
        source_numpy_reward=-float(np.linalg.norm(delta)))


def summarize(rgb,prepared,physical,physical_prepare):
    need(rgb['schema']==RGB+'.consume' and rgb['status']=='all_available8_CEM_H3_RGB_verified_NOT_admitted'
        and rgb['fixed_denominator']==8 and rgb['trajectory_denominator']==rgb['known_roots']==2 and rgb['fresh_roots']==0
        and rgb['encoded']==7 and rgb['missing']==1 and len(rgb['rows'])==len(rgb['comparisons'])==8
        and rgb['full_upstream_rebuilt_here'] is False and rgb['own_native_encoding_saved_reconstructed_here'] is True
        and rgb['consume_new_neural_calls']==rgb['consume_new_physics']==0
        and all(rgb[k] is False for k in ('P1_admission','P2_admission','scientific_admission','physical_release','fair_ranking')),
        'MAIN complete actual7/fixed8 RGB receipt')
    need(prepared['schema']==RGB+'.prepare' and physical['schema']==PHY+'.consume'
        and physical['status']=='bounded_CEM_H3_known2_native_recorded_not_admitted'
        and physical['errors']==[] and physical['fixed_denominator']==physical['known_roots']==2
        and physical['fresh_roots']==0 and physical_prepare['schema']==PHY+'.prepare'
        and equal(physical['plans'],physical_prepare['plans']) and len(physical['rows'])==2,
        'complete known2 actual native authority')
    need(equal(rgb['inherited_physical_counts'],physical['counts'])
        and equal(rgb['inherited_model_resources'],prepared['inherited_model_resources'])
        and rgb['inherited_model_wall_seconds']==prepared['inherited_model_wall_seconds']
        and equal(rgb['inherited_model_resources'],physical_prepare['planner_authority']['resources'])
        and equal(prepared['rows'],[{k:v for k,v in r.items() if k not in ('actual192','pixels','preprocessed','modules')}
            |dict(status='not_started' if r['source'] is not None else 'missing_physical_checkpoint',latent=None) for r in rgb['rows']]),
        'whole RGB prepare/receipt identity, not source equality guessed')
    expected={k:dict(attempted=7 if k in ('encoder','projector') else 0,completed=7 if k in ('encoder','projector') else 0) for k in KINDS}
    need(rgb['counts']==expected,'actual measured7 encoder/projector; no prediction')
    rows=[]
    for i,(r,c) in enumerate(zip(rgb['rows'],rgb['comparisons'],strict=True)):
        root,step=i//4,CP[i%4];p=physical['rows'][root];plan=physical['plans'][root]
        need((r['index'],r['plan_index'],r['root_slot'],r['checkpoint'],r['absolute_step'],r['condition'],r['arm'])==
            (i,root,root,step,10+step,'off','bounded_native_CEM_H3_actualmean')
            and all(equal(r[k],plan[k]) for k in ('task_id','task_sha256','selection_digest','native_goal'))
            and p['plan']==plan and p['selection_digest']==plan['selection_digest']
            and r['actual_returned']==p['own_returned'] and r['physical_status']==p['status'], 'fixed same-time original task/action identity')
        present=not (root==0 and step==15)
        need((r['source'] is not None)==(r['actual192'] is not None)==present,'all8 including actual root0cp15 null')
        g=vector(r['goal192'],192,True);a=None if not present else vector(r['actual192'],192,True)
        pred=None if step==0 else vector(r['predicted192'],192,True)
        n=None;cp=None
        if present:
            hits=[x for x in p['checkpoints'] if x['step']==step];need(len(hits)==1,'unique actual native checkpoint');cp=hits[0]
            f=r['source'];need(equal({k:f[k] for k in cp},cp) and equal(f['state'],f['snapshot']['state']), 'PNG/state snapshot exact time join')
            n=native(f['state'],r['native_goal'])
            need(r['native_metrics']['native_success'] is n['native_success'],'independent original native success')
            if step:
                hits=[x for x in p['responses'] if x['step']==step];need(len(hits)==1 and equal(r['response'],hits[0])
                    and equal(r['response']['state'],f['state']) and r['response']['native_success'] is n['native_success']
                    and r['response']['reward']==n['source_numpy_reward'],'exact native returned reward/success independent routes')
            else:need(r['response'] is None,'anchor no invented return')
        else:need(r['native_metrics'] is r['response'] is r['causal_raw10_bits'] is None,'missing native/action-history not padded')
        aq=None if a is None else mse(a,g);pq=None if pred is None else mse(pred,g)
        err=None if aq is None or pq is None else pq-aq;pe=None if a is None or pred is None else mse(pred,a)
        oldaq=None if a is None else numpy_mse(a,g);oldpq=None if pred is None else numpy_mse(pred,g)
        olderr=None if oldaq is None or oldpq is None else oldpq-oldaq
        oldpe=None if pred is None or a is None else numpy_mse(pred,a)
        need((c['index'],c['root_slot'],c['checkpoint'],c['absolute_step'],c['task_id'])==(i,root,step,10+step,r['task_id'])
            and c['model_Q']==r['model_Q']==oldpq and c['actual_encoded_Q']==oldaq
            and c['prediction_vs_encoded_MSE']==oldpe and c['model_minus_actual_Q']==olderr
            and equal(c['native_metrics'],r['native_metrics']) and c['native_success']==(None if n is None else n['native_success'])
            and c['latent_missing'] is (not present),'source NumPy arithmetic and independent native labels')
        rows.append(dict(index=i,root=root,control_step=step,absolute_step=10+step,role='anchor' if step==0 else 'forecast',
            model_Q=pq,actual_Q=aq,model_minus_actual_Q=err,absolute_Q_error=None if err is None else abs(err),prediction_vs_actual_MSE=pe,
            producer_numpy=dict(model_Q=oldpq,actual_Q=oldaq,prediction_vs_actual_MSE=oldpe,model_minus_actual_Q=olderr),
            fsum_minus_numpy=dict(model_Q=None if pq is None else pq-oldpq,actual_Q=None if aq is None else aq-oldaq,
                prediction_vs_actual_MSE=None if pe is None else pe-oldpe,model_minus_actual_Q=None if err is None else err-olderr),
            native=n,source_native=deepcopy(r['native_metrics']),checkpoint=deepcopy(cp),missing_reason=r['missing_reason'],
            identity={k:deepcopy(r[k]) for k in ('task_id','task_sha256','selection_digest','causal_raw10_bits','planned_raw10_bits','Q_sha256','model_hash')},
            anchor_reencode=deepcopy(c['anchor_reencode'])))
    pairs=[]
    axes=('agent_xy_squared','object_xy_squared','joint_xy_squared','raw_angle_squared','velocity_squared','native7_squared',
          'agent_xy_error','object_xy_error','joint_xy_error','native7_distance','native_angle_term','wrapped_angle_error')
    for root in (0,1):
        group=rows[root*4:(root+1)*4]
        need(all(x['identity']['task_id']==group[0]['identity']['task_id'] and
                 x['identity']['selection_digest']==group[0]['identity']['selection_digest'] and
                 equal(rgb['rows'][x['index']]['goal192'],rgb['rows'][root*4]['goal192']) for x in group),'single task/goal per timeline')
        for j in range(4):
            for k in range(j+1,4):
                a,b=group[j],group[k]
                def diff(key):return None if a[key] is None or b[key] is None else b[key]-a[key]
                nd=None if a['native'] is None or b['native'] is None else {key:b['native'][key]-a['native'][key] for key in axes}
                dm,da=diff('model_Q'),diff('actual_Q')
                dirs=dict(model_to_actual=compare(dm,da))
                for axis in axes:
                    dirs['actual_to_'+axis]=compare(da,None if nd is None else nd[axis])
                    dirs['model_to_'+axis]=compare(dm,None if nd is None else nd[axis])
                pairs.append(dict(root=root,left_step=CP[j],right_step=CP[k],direction='later minus earlier, within root only',
                    delta_model_Q=dm,delta_actual_Q=da,native_delta=nd,comparisons=dirs,
                    native_success_delta=None if nd is None else int(b['native']['native_success'])-int(a['native']['native_success'])))
    keys=sorted(pairs[0]['comparisons'])
    return dict(rows=rows,pairs=pairs,denominator=dict(positions=8,anchors=2,forecasts=6,within_root_time_pairs=12,known_roots=2,fresh_roots=0),
        available=dict(actual_positions=7,anchors=2,forecasts=5,missing_positions=1),
        comparison_census={key:{v:sum(p['comparisons'][key]==v for p in pairs) for v in ('same','reversal','both_tie','left_tie','right_tie','null')} for key in keys},
        inherited_rgb_counts=deepcopy(rgb['counts']),inherited_physical_counts=deepcopy(physical['counts']),
        inherited_model_resources=deepcopy(rgb['inherited_model_resources']),inherited_model_wall_seconds=rgb['inherited_model_wall_seconds'],
        total_FLOPs=None,interpretation='post-action exploratory within-CEM diagnostic, not cross-method ranking/causal control benefit',
        **{k:False for k in FLAGS})


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def analyze(inputs,inputs_sha):
    manifest=read(inputs,inputs_sha)
    need(manifest['schema']==NAME+'.inputs' and manifest['trust']=='MAIN_complete_external_receipt_SHAs_not_rebuilt'
        and set(manifest['inputs'])=={'rgb_receipt','rgb_prepare','physical_receipt','physical_prepare'},'explicit complete authority manifest')
    specs=manifest['inputs'];need(specs['physical_receipt']['sha256']==PHYSICAL_SHA
        and specs['physical_prepare']['sha256']==PHYSICAL_PREP_SHA
        and specs['rgb_receipt']['sha256']==RGB_RECEIPT_SHA
        and specs['rgb_prepare']['sha256']==RGB_PREPARE_SHA,'fixed actual four authorities')
    data={k:read(v['path'],v['sha256']) for k,v in specs.items()};rgb=data['rgb_receipt'];prepared=data['rgb_prepare']
    bindings={str(absolute(inputs)):inputs_sha,**{v['path']:v['sha256'] for v in specs.values()}}
    need(rgb['prepare_sha256']==specs['rgb_prepare']['sha256']
        and rgb['report_sha256']==RGB_REPORT_SHA and rgb['completion_sha256']==RGB_COMPLETION_SHA
        and rgb['completion']['report_sha256']==RGB_REPORT_SHA
        and rgb['completion']['prepare_sha256']==RGB_PREPARE_SHA
        and rgb['completion']['prepared']==specs['rgb_prepare']['path']
        and rgb['source_authorities']==prepared['arguments']
        and rgb['source_authorities']['physical_receipt_sha']==PHYSICAL_SHA
        and rgb['source_authorities']['physical_prepare_sha']==PHYSICAL_PREP_SHA
        and rgb['source_authorities']['physical_receipt']==specs['physical_receipt']['path']
        and rgb['source_authorities']['physical_prepared']==specs['physical_prepare']['path'], 'exact prepare/receipt physical join')
    need(prepared['bindings'][str(ROOT/'experiments/spectral_reachability'/(RGB+'.py'))]==RGB_SOURCE_SHA,'frozen RGB owner source')
    # Re-read only the seven small final192 artifacts. Full native IO/PNG/journal
    # authentication is inherited from MAIN's complete receipt, not repeated here.
    for row in rgb['rows']:
        if row['actual192'] is None:continue
        ref=row['latent'];name=ref['file'];need(Path(name).name==name and name.endswith('.npy.gz'),'flat final latent artifact')
        paths=[p for p in rgb['bindings'] if Path(p).name==name]
        need(len(paths)==1,'unique actual latent pointer');p=absolute(paths[0]);h=rgb['bindings'][str(p)]
        need(h==ref['sha256'] and p.is_file() and p.stat().st_size<=4096 and sha(p)==h,'small full latent SHA')
        with gzip.open(p,'rb') as f:body=f.read(897)
        need(len(body)==896,'bounded exact NPY192');stream=io.BytesIO(body);arr=np.load(stream,allow_pickle=False)
        need(not stream.read() and arr.shape==(192,) and arr.dtype==np.float32 and np.isfinite(arr).all()
            and ref['shape']==[192] and ref['dtype']=='float32'
            and hashlib.sha256(arr.tobytes()).hexdigest()==ref['array_sha256']
            and arr.tobytes()==np.asarray(vector(row['actual192'],192,True),np.float32).tobytes(),'actual saved final192 bytes')
        bindings[str(p)]=h
    summary=summarize(rgb,prepared,data['physical_receipt'],data['physical_prepare'])
    for f in own_files():bindings[str(f)]=sha(f)
    need(all(sha(p)==h for p,h in bindings.items()),'all locally read inputs unchanged')
    return dict(schema=NAME+'.summary',arguments=dict(inputs=str(absolute(inputs)),inputs_sha=inputs_sha),summary=summary,
        bindings=bindings,new_neural_calls=0,new_physics_calls=0,new_fullstrict_calls=0,
        source_scope='external complete receipts trusted; own small vectors and arithmetic independently rebuilt; not entire inherited closure')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    a=sub.add_parser('analyze')
    for k in ('inputs','inputs-sha','output'):a.add_argument('--'+k,required=True)
    a=sub.add_parser('verify')
    for k in ('summary','summary-sha'):a.add_argument('--'+k,required=True)
    a=ap.parse_args(argv)
    if a.command=='verify':
        x=read(a.summary,a.summary_sha);need(x['schema']==NAME+'.summary' and equal(x,analyze(**x['arguments'])),'whole deterministic rebuild')
        out=dict(verified=True,sha256=a.summary_sha)
    else:
        x=analyze(a.inputs,a.inputs_sha);p=absolute(a.output);body=encode(x)
        need(p.name.startswith(NAME+'_') and not p.exists() and len(body)<=CAP,'new exclusive bounded output')
        with p.open('xb') as f:need(f.write(body)==len(body),'full write');f.flush();os.fsync(f.fileno())
        fd=os.open(p.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        out=dict(output=str(p),sha256=sha(p))
    print(encode(out).decode(),end='')


if __name__=='__main__':main()
