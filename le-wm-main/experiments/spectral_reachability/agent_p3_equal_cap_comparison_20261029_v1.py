"""Saved-only, receipt-bound old-parent vs equal-cap description; no driver import."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import struct

NAME='agent_p3_equal_cap_comparison_20261029_v1'
OLD='agent_p3_parent_native_cuda_20261029_v1'
NEW='agent_p3_equal_cap_parent_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
AXES=('predict_API','predict_samples','predict_context_tokens','action_encoder_calls',
 'action_encoder_samples','action_encoder_context_tokens','predictor_calls','predictor_samples',
 'predictor_context_tokens','pred_proj_calls','pred_proj_rows','encoder_calls','encoder_samples',
 'projector_calls','projector_samples','jacobian_API','VJP_directions','grad_API',
 'raw_backward_callbacks','model_SVD_calls','array_SVD_calls','solve_calls',
 'signal_compile_calls','program_compile_calls','replay_calls','metric_calls','schedule_decisions')
UNKNOWN=('scalar_backward_calls','AD_primal_predict_API','Jacobian_elements')
FLAGS=dict(P1_admission=False,P3_admission=False,full_C3_demonstrated=False,
 freshness=False,new_independent_roots=False,equal_actual_resources=False,equal_FLOPs=False,
 budget_effectiveness_identified=False)


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def unique(pairs):
    d={}
    for k,v in pairs:
        require(k not in d,'duplicate JSON field');d[k]=v
    return d


def load(ref):
    require(set(ref)=={'path','sha256'},'explicit path/SHA')
    path=Path(ref['path']);h=ref['sha256']
    require(len(h)==64 and all(c in '0123456789abcdef' for c in h),'SHA256 syntax')
    require(path.is_absolute() and path.resolve()==path and path.is_file() and not path.is_symlink(),'regular exact absolute input')
    require(path.stat().st_size<=256*1024**2 and sha(path)==h,'bounded input actual SHA')
    x=json.loads(path.read_bytes(),object_pairs_hook=unique,parse_constant=lambda x:require(False,'nonfinite JSON'))
    require(sha(path)==h,'input stable');return x


def actions(a):
    if a is None:return None
    require(type(a) is list and all(type(row) is list and len(row)==2 for row in a),'raw action Nx2')
    values=[v for row in a for v in row]
    require(all(type(v) in (int,float) and math.isfinite(v) for v in values),'finite raw')
    # Exact producer JSON and both numeric representations. Do not conflate
    # JSON number equality (notably signed zero) with actual float32 equality.
    return dict(count=len(a),json_sha256=digest(a),
        float64_sha256=hashlib.sha256(b''.join(struct.pack('<d',v) for v in values)).hexdigest(),
        float32_sha256=hashlib.sha256(b''.join(struct.pack('<f',v) for v in values)).hexdigest())


def plan(n):
    if n is None:return None
    require(type(n['depth']) is int and len(n['edges'])==len(n['indices'])==n['depth'],'node depth/edge ABI')
    return dict(depth=n['depth'],indices=n['indices'],cost=n.get('cost'),
        edge_horizons=[len(e['actions'])//5 for e in n['edges']],
        edge_types=[e['kind'] for e in n['edges']],actions=actions(n['actions']),
        context_sha256=digest(n['context']),node_sha256=digest(n))


def job(j):
    if j is None:return None
    q=j['request'];n=j['node']
    return dict(slot=j['slot'],status=j['status'],error=j['error'],
        request=None if q is None else dict(horizon=q['horizon'],candidates=q['candidates'],
            candidate_index=q['candidate_index'],program_type=q['program_type'],extension_type=q['extension_type'],
            parent_context_sha256=q['parent_context_sha256'],binding_sha256=q['binding_sha256'],
            prefix_actions=actions(q['prefix_actions']),request_sha256=digest(q)),
        node=plan(n),extension_actions=None if j['payload'] is None else actions(j['payload']['actions']))


def counters(rr):
    for phase in ('caps','attempted','completed'):
        require(set(rr[phase])==set(AXES),'all27 resource axes')
        require(all(type(v) is int and v>=0 for v in rr[phase].values()),'nonnegative integral counters')
    require(all(rr['completed'][k]<=rr['attempted'][k]<=rr['caps'][k] for k in AXES),'paid resource order')
    return {k:rr[k] for k in ('caps','attempted','completed','wall_ns','inclusive_operation_wall_ns',
        'wall_semantics','FLOPs','cuda_ns','backward_engine_calls','unknowns_are_not_zero','physics_steps')}


def timing(row):
    t=row.get('timing')
    return None if t is None else {k:t[k] for k in ('scope','cap_seconds','failed','elapsed_ns',
        'synchronization_attempts','synchronization_returned')} | dict(
        max_overrun_ns=max((p['overrun_ns'] or 0 for p in t['probes']),default=0),
        failed_probe_count=sum(p['status']=='failed' for p in t['probes']))


def arm(row,new):
    r=row['result']
    if r is None:
        return dict(slot=row['slot'],root=row['root'],mode=row['mode'],status=row['status'],
            observed=False,plan_available=False,released_plan=None,core_model_plan=None,diagnostic_best=None,
            realized_depth=None,widths=None,queues=None,resources=None,unknown=None,
            timing=timing(row),jobs=[None]*144)
    require(len(r['jobs'])==144 and [j['slot'] for j in r['jobs']]==list(range(144)),'full144 ordered request denominator')
    require(r['mode']==row['mode'],'core mode')
    release=row['released_plan'] if new else r['model_plan']
    available=row['plan_available'] if new else release is not None
    require(type(available) is bool and available==(release is not None),'release disposition')
    if new:require(release is None or digest(release)==digest(r['model_plan']),'released exact core witness')
    return dict(slot=row['slot'],root=row['root'],mode=row['mode'],status=row['status'],observed=True,
        plan_available=available,released_plan=plan(release),core_model_plan=plan(r['model_plan']),
        diagnostic_best=plan(r['diagnostic_best']),realized_depth=r['realized_progress_depth'],widths=r['realized_widths'],
        queues=[dict(status=q['status'],depth=q['depth'],parent_context_sha256=q['parent_context_sha256'],
            decision=q['decision'],request_count=len(q['requests']),used_before=q['used_before']) for q in r['queues']],
        resources=counters(r['resource_receipt']),unknown={k:r[k] for k in UNKNOWN},timing=timing(row),
        jobs=[job(j) for j in r['jobs']])


def validate_bundle(p,r,s,new,refs):
    name=NEW if new else OLD
    require([p['schema'],r['schema'],s['schema']]==[name+'.'+k for k in ('prepare','report','consume')],'real own schemas')
    require(s['status'] in (('saved_equal_cap_verified_not_admitted','saved_equal_cap_partial_verified_not_admitted') if new else
        ('saved_four_slot_engineering_verified','saved_partial_engineering_verified')),'complete strict receipt status')
    require(r['prepared']==s['prepared']==refs['prepare']['path'],'exact prepare paths')
    require(r['prepare_sha256']==s['prepare_sha256']==refs['prepare']['sha256'] and s['report_sha256']==refs['report']['sha256'],'strict report/prepare join')
    require(len(p['tasks'])==2 and len(r['rows'])==4,'known2 fixed4')
    require([(x['slot'],x['root'],x['mode']) for x in r['rows']]==[(i,i//2,('fixed','adaptive')[i%2]) for i in range(4)],'fixed slot order')
    require(s['slot_statuses']==[x['status'] for x in r['rows']],'receipt slot status join')
    require(not r['errors'] and r['sources_verified'] is True,'receipt-certified final closure')
    for k in ('P1_admission','P3_admission','freshness','equal_FLOPs'):
        require(r[k] is False and s[k] is False,'no scientific admission')
    indexed={a['slot']:a for a in s['arms']}
    require(len(indexed)==len(s['arms']) and set(indexed)=={x['slot'] for x in r['rows'] if x['result'] is not None},'receipt observed arms census')
    for row in r['rows']:
        rr=row['result']
        if rr is None:continue
        a=indexed[row['slot']]
        require((a['root'],a['mode'],a['status'])==(row['root'],row['mode'],rr['status']),'strict arm identity')
        require(digest(a['resources'])==digest(rr['resource_receipt']),'full receipt resources bound')
        require(a['realized_progress_depth']==rr['realized_progress_depth'] and a['realized_widths']==rr['realized_widths'],'receipt geometry join')
        require(a['model_plan_available']==(rr['model_plan'] is not None),'receipt core plan join')
        require(rr['resource_receipt']['caps']==p['profile']['resource_caps'][row['mode']],'prepared actual resource caps')
        require(all(a[k]==rr[k] for k in UNKNOWN),'unknown axes joined not zero-filled')
    if new:
        require(s['fixed_denominator']==dict(arms=4,requests_per_arm=144,total_requests=576),'new receipt576')
    else:require(s['denominator']==4,'old receipt4')
    return [arm(x,new) for x in r['rows']]


def compare(old,new):
    rows=[]
    for a,b in zip(old,new,strict=True):
        delta=None if a['resources'] is None or b['resources'] is None else {
            phase:{k:b['resources'][phase][k]-a['resources'][phase][k] for k in AXES}
            for phase in ('caps','attempted','completed')}
        paired=[]
        for i,(x,y) in enumerate(zip(a['jobs'],b['jobs'],strict=True)):
            comparable=x is not None and y is not None
            same_request=None if not comparable or x['request'] is None or y['request'] is None else x['request']==y['request']
            paired.append(dict(slot=i,old=x,new=y,same_request=same_request,
                same_extension_action_bits=None if not comparable or x['extension_actions'] is None or y['extension_actions'] is None
                else x['extension_actions']['float32_sha256']==y['extension_actions']['float32_sha256']))
        selected_a=a['released_plan'];selected_b=b['released_plan']
        rows.append(dict(slot=a['slot'],root=a['root'],mode=a['mode'],
            old={k:v for k,v in a.items() if k!='jobs'},new={k:v for k,v in b.items() if k!='jobs'},
            delta_new_minus_old=delta,
            same_released_action_bits=None if selected_a is None or selected_b is None else
                selected_a['actions']['float32_sha256']==selected_b['actions']['float32_sha256'],
            request_status_census={label:dict(Counter('unobserved' if x is None else x['status'] for x in side['jobs']))
                for label,side in (('old',a),('new',b))},jobs=paired))
    return rows


def analyze(inputs):
    require(set(inputs)=={'old','new'},'two explicit bundles')
    bundles={}
    for label in ('old','new'):
        require(set(inputs[label])=={'prepare','report','receipt'},'six external SHA inputs')
        bundles[label]=[load(inputs[label][k]) for k in ('prepare','report','receipt')]
    p0,r0,s0=bundles['old'];p1,r1,s1=bundles['new']
    require(digest(p0['tasks'])==digest(p1['tasks']),'identical full task/context/goal/raw binding')
    require(p0['expected_model_hash']==p1['expected_model_hash'] and r0['model_before']==r1['model_before'],'same frozen model identity')
    require(p1['profile']['resource_caps']['fixed']==p1['profile']['resource_caps']['adaptive'],'new common cap only')
    old=validate_bundle(p0,r0,s0,False,inputs['old']);new=validate_bundle(p1,r1,s1,True,inputs['new'])
    result=dict(schema=NAME+'.summary',inputs=inputs,source_sha256=sha(__file__),
        denominator=dict(known_roots=2,paired_arms=4,request_slots_per_arm=144,paired_request_slots=576),
        comparison='new_minus_old descriptive; ordinal slots are not guaranteed same-parent action witnesses',
        profiles=dict(old=p0['profile'],new=p1['profile']),
        wall=dict(old_whole=r0['wall_seconds'],new_whole=r1['wall_seconds'],
            old_arm_scope='not independently synchronized per-arm in old report; resource wall_ns is core scope',
            new_arm_scope=p1['profile']['timer_scope']),
        arms=compare(old,new),**FLAGS)
    require(all(sha(ref['path'])==ref['sha256'] for side in inputs.values() for ref in side.values()),'all six inputs stable after comparison')
    return result


def write(path,value):
    p=Path(path).absolute()
    require(p.parent==ROOT/'results' and p.name.startswith(NAME) and not p.exists() and p.resolve()==p,'new own output only')
    body=canonical(value);require(len(body)<=32*1024**2,'summary cap')
    with p.open('xb') as f:
        require(f.write(body)==len(body),'whole summary write');f.flush();os.fsync(f.fileno())
    return dict(path=str(p),sha256=sha(p))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('analyze')
    for label in ('old','new'):
        for kind in ('prepare','report','receipt'):
            a.add_argument('--'+label+'-'+kind,required=True);a.add_argument('--'+label+'-'+kind+'-sha',required=True)
    a.add_argument('--output',required=True)
    v=sub.add_parser('verify');v.add_argument('--summary',required=True);v.add_argument('--summary-sha',required=True)
    args=parser.parse_args(argv)
    if args.command=='analyze':
        inputs={label:{k:dict(path=str(Path(getattr(args,label+'_'+k)).absolute()),sha256=getattr(args,label+'_'+k+'_sha'))
            for k in ('prepare','report','receipt')} for label in ('old','new')}
        out=write(args.output,analyze(inputs))
    else:
        saved=load(dict(path=str(Path(args.summary).absolute()),sha256=args.summary_sha))
        require(canonical(saved)==canonical(analyze(saved['inputs'])),'complete independent saved summary recomputation')
        out=dict(verified=True,sha256=args.summary_sha)
    print(canonical(out).decode())


if __name__=='__main__':main()
