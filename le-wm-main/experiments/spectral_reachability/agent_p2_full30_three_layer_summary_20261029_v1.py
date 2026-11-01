"""Independent saved-vector fsum Q joins, no producer math or neural execution."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np

NAME='agent_p2_full30_three_layer_summary_20261029_v1'
RGB='agent_p2_two_program_full30_rgb_encode_20261029_v1'
SCALAR='main_p2_full30_same_time_summary_20261029_v1'
RGB_SHA='42164a225310c39ff83e398f55501e36e8db9876d5b52dbc697d031e31468166'
SCALAR_SHA='d9b97b5c351bcee90f1d2bde5598d2d2c6469dd7bcceb09b9a66117cad00fa16'
ROOT=Path(__file__).resolve().parents[2]
CP=(0,5,10,15,20,25,30)
ARMS=('two_program_transfer','joint_two_round')
FLAGS=('P1_admission','P2_admission','P2_mechanism','scientific_admission','freshness','equal_FLOPs','equal_total_compute','physical_release')
NATIVE_KEYS=('agent_xy_squared','object_xy_squared','joint_xy_squared','raw_angle_squared','velocity_squared',
    'native7_squared','joint_xy_error','object_xy_error','native7_distance','native_angle_term','wrapped_angle_error')
CAP=16*1024**2


def need(ok,label):
    if not ok:raise ValueError(label)


def encode(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def equal(a,b):return encode(a)==encode(b)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def absolute(path):
    p=Path(path);need(p.is_absolute() and str(p.resolve())==str(p) and not p.is_symlink(),'canonical absolute regular path');return p


def read(path,h):
    p=absolute(path);need(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h),'SHA256')
    need(p.is_file() and p.stat().st_size<=CAP,'bounded saved JSON');b=p.read_bytes()
    need(len(b)<=CAP and hashlib.sha256(b).hexdigest()==h,'exact saved SHA')
    def pairs(items):
        d={}
        for k,v in items:need(k not in d,'duplicate JSON key');d[k]=v
        return d
    def bad(_):raise ValueError('nonfinite JSON')
    x=json.loads(b,object_pairs_hook=pairs,parse_constant=bad);encode(x);return x


def vector(x):
    need(type(x) is list and len(x)==192 and all(type(v) in (int,float) and math.isfinite(v) for v in x),'finite native192 no bool')
    # Receipt vectors are literal native float32; never renormalize or fit.
    a=np.asarray(x,np.float64)
    need(np.array_equal(a,a.astype(np.float32).astype(np.float64)),'lossless float32 vector')
    return a.tolist()


def mse(a,b):
    return math.fsum((x-y)*(x-y) for x,y in zip(a,b,strict=True))/192


def original_mean(a,b):
    # Independent local reproduction ONLY to check the receipt's NumPy reduction.
    d=np.asarray(a,np.float64)-np.asarray(b,np.float64)
    return float(np.mean(d*d))


def sign(x):
    if x is None:return 'null'
    need(type(x) in (int,float) and math.isfinite(x),'finite margin')
    return 'positive' if x>0 else 'negative' if x<0 else 'tie'


def compare(x,y):
    a,b=sign(x),sign(y)
    if 'null' in (a,b):return 'null'
    if a==b:return 'both_tie' if a=='tie' else 'same'
    if a=='tie':return 'left_tie'
    if b=='tie':return 'right_tie'
    return 'reversal'  # Strict comparisons above, NEVER x*y (underflow).


def summarize(rgb,scalar):
    need(rgb['schema']==RGB+'.consume' and rgb['status']=='all28_full30_RGB_verified_NOT_admitted'
        and rgb['fixed_denominator']==28 and rgb['plan_denominator']==4 and len(rgb['rows'])==28
        and rgb['known_roots']==2 and rgb['fresh_roots']==0
        and rgb['consume_new_neural_calls']==rgb['consume_new_physics']==0
        and rgb['full_upstream_rebuilt_here'] is False and rgb['own_native_encoding_saved_reconstructed_here'] is True
        and all(rgb[k] is False for k in ('P1_admission','P2_admission','P2_mechanism','C3','equal_total_compute','physical_release')),
        'actual complete own full30 RGB receipt, no invented flags')
    need(scalar['schema']==SCALAR+'.summary' and all(scalar[k]==0 for k in ('new_neural_calls','new_physics_calls','new_fullstrict_calls')),
        'independent same-time scalar summary')
    s=scalar['summary'];need(s['denominator']==dict(branches=4,known_roots=2,fresh_roots=0,checkpoint_positions=28,pairs=14)
        and len(s['panel'])==28 and len(s['pairs'])==14 and all(s[k] is False for k in FLAGS),'fixed native panel and non-admission')
    need(rgb['physical_report_sha256']==scalar['arguments']['report_sha']
        and rgb['physical_receipt_sha256']==scalar['arguments']['receipt_sha']
        and equal(rgb['inherited_physical_counts'],s['counts']),'same actual native authority and entire resource vector')
    rows=[];nactual=0
    for i,(r,n) in enumerate(zip(rgb['rows'],s['panel'],strict=True)):
        slot,step=i//7,CP[i%7]
        need((r['index'],r['plan_index'],r['root_slot'],r['condition'],r['arm'],r['checkpoint'])==
             (i,slot,slot//2,'off',ARMS[slot%2],step)
             and (n['slot'],n['root'],n['arm'],n['control_step'],n['absolute_step'])==
             (slot,slot//2,ARMS[slot%2],step,10+step),'ordered 28 rows, original arm and time')
        need(all(r[k] is False for k in ('P1_admission','P2_admission','C3','physical_benefit_claim')),'no row benefit claim')
        goal=vector(r['goal192']);pred=None if step==0 else vector(r['predicted192'])
        actual=None if r['actual192'] is None else vector(r['actual192'])
        present=actual is not None
        need(r['latent_missing'] is (not present) and (n['metrics'] is not None)==present
            and (r['source'] is not None)==present and (n['checkpoint'] is not None)==present
            and (n['missing_reason'] is None)==present,'no fake missing/fill')
        if present:
            nactual+=1;f=r['source'];cp=n['checkpoint']
            need(set(n['metrics'])==set(NATIVE_KEYS)|{'native_success'} and type(n['metrics']['native_success']) is bool
                and all(type(n['metrics'][k]) in (int,float) and math.isfinite(n['metrics'][k]) for k in NATIVE_KEYS),'complete native axes')
            need(equal({k:f[k] for k in cp},cp) and f['state']==f['snapshot']['state'] and f['step']==step,
                 'same-time PNG/state/full snapshot join')
            absolute(f['path']);need(Path(f['path']).name==f['png'],'real PNG path pointer')
        if step==0:
            need(r['role']=='observed_anchor_reencode' and r['predicted192'] is None and r['model_Q'] is None
                and all(r[k] is None for k in ('model_minus_actual_Q','absolute_Q_error','prediction_vs_encoded_MSE','response','native_metrics'))
                and r['observed_anchor_reencode_MSE']==(0. if present else None),'anchor model null, not zero forecast')
        else:
            need(r['role']=='same_action_H6_prediction' and r['observed_anchor_reencode_MSE'] is None,'forecast role')
            if present:
                need(r['response']['step']==step and r['response']['state']==r['source']['state']
                    and r['native_metrics']['native_success'] is n['metrics']['native_success']
                    and r['response']['native_success'] is n['metrics']['native_success'],'same returned native time and success')
            else:need(r['response'] is r['native_metrics'] is None,'missing has no invented native labels')
        aq=None if actual is None else mse(actual,goal)
        pq=None if pred is None else mse(pred,goal)
        error=None if actual is None or pred is None else pq-aq
        pmse=None if actual is None or pred is None else mse(pred,actual)
        oldaq=None if actual is None else original_mean(actual,goal)
        oldpq=None if pred is None else original_mean(pred,goal)
        olderr=None if oldaq is None or oldpq is None else oldpq-oldaq
        oldmse=None if actual is None or pred is None else original_mean(pred,actual)
        need(r['actual_encoded_Q']==oldaq and r['model_Q']==oldpq and r['model_minus_actual_Q']==olderr
            and r['absolute_Q_error']==(None if olderr is None else abs(olderr)) and r['prediction_vs_encoded_MSE']==oldmse,
            'exact original NumPy route fields; fsum not forced equal or relaxed')
        rows.append(dict(index=i,plan_index=slot,root=slot//2,arm=ARMS[slot%2],control_step=step,absolute_step=10+step,
            role='anchor' if step==0 else 'forecast',model_Q=pq,actual_Q=aq,model_minus_actual_Q=error,
            absolute_Q_error=None if error is None else abs(error),prediction_vs_actual_MSE=pmse,
            producer_numpy=dict(model_Q=oldpq,actual_Q=oldaq,model_minus_actual_Q=olderr,prediction_vs_actual_MSE=oldmse),
            fsum_minus_producer_numpy=dict(model_Q=None if pq is None else pq-oldpq,actual_Q=None if aq is None else aq-oldaq,
                model_minus_actual_Q=None if error is None else error-olderr,prediction_vs_actual_MSE=None if pmse is None else pmse-oldmse),
            native=deepcopy(n['metrics']),native_source_metrics=deepcopy(r['native_metrics']),
            missing_reason=n['missing_reason'],checkpoint=deepcopy(n['checkpoint']),
            identity={k:deepcopy(r[k]) for k in ('task_id','selection_digest','causal_raw10_bits','physical_status')},
            imported_anchor_reencode_MSE=r['observed_anchor_reencode_MSE']))
    expected={k:dict(attempted=nactual if k in ('encoder','projector') else 0,completed=nactual if k in ('encoder','projector') else 0)
              for k in ('encoder','projector','action_encoder','predict_API','predictor','pred_proj')}
    need(rgb['counts']==expected,'full measured RGB resource axes, no predict-only census')
    for slot in range(4):
        base=7*slot
        need(all(rows[base]['identity']['task_id']==rows[base+j]['identity']['task_id']
            and rows[base]['identity']['selection_digest']==rows[base+j]['identity']['selection_digest']
            and equal(rgb['rows'][base]['goal192'],rgb['rows'][base+j]['goal192']) for j in range(7)),'whole plan task/goal identity')
    pairs=[]
    for i,original in enumerate(s['pairs']):
        root,step=i//7,CP[i%7];li,ri=2*root*7+CP.index(step),(2*root+1)*7+CP.index(step)
        a,b=rows[li],rows[ri]
        need((original['root'],original['control_step'],original['direction'])==(root,step,'joint_two_round minus two_program_transfer'), '14 fixed same-time pairs')
        need(a['identity']['task_id']==b['identity']['task_id'] and equal(rgb['rows'][li]['goal192'],rgb['rows'][ri]['goal192']), 'pair original task and goal coordinate')
        known=a['native'] is not None and b['native'] is not None
        native=None if not known else {k:b['native'][k]-a['native'][k] for k in a['native'] if k!='native_success'}
        need(original['comparable'] is known and equal(original['delta'],native)
            and original['native_success_delta']==(None if not known else int(b['native']['native_success'])-int(a['native']['native_success'])), 'independent native full-axis margin join')
        def diff(k):return None if a[k] is None or b[k] is None else b[k]-a[k]
        model,actual=diff('model_Q'),diff('actual_Q')
        directions=dict(model_to_actual=compare(model,actual))
        for k in NATIVE_KEYS:
            directions['actual_to_'+k]=compare(actual,None if native is None else native[k])
            directions['model_to_'+k]=compare(model,None if native is None else native[k])
        pairs.append(dict(root=root,control_step=step,role='anchor_pair' if step==0 else 'forecast_pair',left_slot=2*root,right_slot=2*root+1,
            direction=original['direction'],delta_model_Q=model,delta_actual_Q=actual,delta_model_minus_actual_Q=diff('model_minus_actual_Q'),
            native_delta=native,native_success_delta=original['native_success_delta'],comparisons=directions,
            layer_signs=dict(model=sign(model),actual=sign(actual),native=None if native is None else {k:sign(v) for k,v in native.items()}),
            missing_slots=[r['plan_index'] for r in (a,b) if r['actual_Q'] is None],
            vector_equalities=dict(actual=None if a['actual_Q'] is None or b['actual_Q'] is None else equal(rgb['rows'][li]['actual192'],rgb['rows'][ri]['actual192']),
                predicted=None if step==0 else equal(rgb['rows'][li]['predicted192'],rgb['rows'][ri]['predicted192']))))
    comparison_keys=sorted({k for p in pairs for k in p['comparisons']})
    def census(group):
        return {k:{v:sum(p['comparisons'].get(k,'null')==v for p in group) for v in ('same','reversal','both_tie','left_tie','right_tie','null')} for k in comparison_keys}
    return dict(rows=rows,pairs=pairs,denominator=dict(positions=28,anchors=4,forecasts=24,pairs=14,anchor_pairs=2,forecast_pairs=12),
        available=dict(actual_positions=nactual,anchors=sum(r['actual_Q'] is not None for r in rows if r['role']=='anchor'),
            forecasts=sum(r['actual_Q'] is not None for r in rows if r['role']=='forecast')),
        comparison_census=dict(all14=census(pairs),forecast12=census([p for p in pairs if p['role']=='forecast_pair']),anchor2=census([p for p in pairs if p['role']=='anchor_pair'])),
        inherited_rgb_counts=deepcopy(rgb['counts']),inherited_physical_counts=deepcopy(s['counts']),
        inherited_model_resources=deepcopy(rgb['inherited_model_resources']),inherited_model_wall_seconds=rgb['inherited_model_wall_seconds'],
        total_FLOPs=rgb['total_FLOPs'],known_roots=2,fresh_roots=0,
        arithmetic='float32 native vectors -> float64 coordinate subtract/square -> math.fsum /192; receipt NumPy route separate; strict sign comparisons without products',
        native_scope='entire independently authenticated native scalar table imported and source-joined, not recomputed from missing goal7',
        interpretation='post-action saved diagnostic only, no causal identification/selector/deployment/equal-budget/scientific admission',
        **{k:False for k in FLAGS})


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def analyze(rgb_receipt,rgb_sha,scalar_summary,scalar_sha):
    need(rgb_sha==RGB_SHA and scalar_sha==SCALAR_SHA,'only the two fixed actual authorities')
    paths=[absolute(rgb_receipt),absolute(scalar_summary)];hashes=[rgb_sha,scalar_sha]
    rgb,scalar=[read(str(p),h) for p,h in zip(paths,hashes)]
    result=dict(schema=NAME+'.summary',arguments=dict(rgb_receipt=str(paths[0]),rgb_sha=rgb_sha,scalar_summary=str(paths[1]),scalar_sha=scalar_sha),
        summary=summarize(rgb,scalar),source_scope='imported SHA-bound complete RGB receipt and independently verified scalar summary; no upstream fullstrict reexecution',
        new_neural_calls=0,new_physics_calls=0,new_fullstrict_calls=0,bindings=dict(zip(map(str,paths),hashes)))
    for f in own_files():result['bindings'][str(f)]=sha(f)
    for p,h in zip(paths,hashes):need(sha(p)==h,'inputs unchanged')
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('analyze','verify'))
    for k in ('rgb-receipt','rgb-sha','scalar-summary','scalar-sha','output','summary','summary-sha'):ap.add_argument('--'+k)
    a=ap.parse_args(argv)
    if a.command=='verify':
        x=read(a.summary,a.summary_sha);need(x['schema']==NAME+'.summary','own schema')
        need(equal(x,analyze(**x['arguments'])),'whole deterministic reproduction');out=dict(verified=True,sha256=a.summary_sha)
    else:
        x=analyze(a.rgb_receipt,a.rgb_sha,a.scalar_summary,a.scalar_sha);p=absolute(a.output);b=encode(x)
        need(p.name.startswith(NAME+'_') and not p.exists() and len(b)<=CAP,'exclusive bounded output')
        with p.open('xb') as f:need(f.write(b)==len(b),'complete write');f.flush();os.fsync(f.fileno())
        fd=os.open(p.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        out=dict(output=str(p),sha256=sha(p))
    print(encode(out).decode(),end='')


if __name__=='__main__':main()
