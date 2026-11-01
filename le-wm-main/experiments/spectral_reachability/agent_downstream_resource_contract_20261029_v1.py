"""Saved ABI and fixture-only resource vectors; no producer imports/full strict."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

NAME='agent_downstream_resource_contract_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
CAP=128*1024**2
AXES={
 'encode_calls':'API','encode_samples':'batch_samples','encode_frames':'input_frames',
 'projector_calls':'API','projector_samples':'batch_samples','projector_frames':'input_frames',
 'predict_calls':'API','predict_samples':'batch_samples','predict_frames':'context_frames',
 'action_encoder_calls':'API','action_encoder_samples':'batch_samples','action_encoder_frames':'context_frames',
 'predictor_calls':'API','predictor_samples':'batch_samples','predictor_frames':'context_frames',
 'pred_proj_calls':'API','pred_proj_rows':'flattened_rows',
 'jacobian_requests':'requests','jacobian_elements':'requested_elements','VJP_directions':'requested_output_directions',
 'grad_API':'explicit_calls','raw_backward_callbacks':'observed_callbacks','scalar_backward':'explicit_loss_backward',
 'engine_backward':'unknown_engine_invocations','JVP':'independent_requests',
 'AD_primal_packets':'primal_packet_steps','solve':'calls','model_SVD':'calls','array_SVD':'calls',
 'Adam_step':'steps','virtual_noise':'draws','public_step':'public_native_steps','physics_step':'integration_substeps',
 'RGB_feedback':'actual_encoded_boundaries','render':'calls','reset':'calls','goal_bind':'calls','training':'steps'}
P3_AXES=('predict_API','predict_samples','predict_context_tokens','action_encoder_calls','action_encoder_samples',
 'action_encoder_context_tokens','predictor_calls','predictor_samples','predictor_context_tokens','pred_proj_calls','pred_proj_rows',
 'encoder_calls','encoder_samples','projector_calls','projector_samples','jacobian_API','VJP_directions','grad_API',
 'raw_backward_callbacks','model_SVD_calls','array_SVD_calls','solve_calls','signal_compile_calls','program_compile_calls',
 'replay_calls','metric_calls','schedule_decisions')
SCALARS=dict(jacobian='jacobian_API',grad='grad_API',raw_backward='raw_backward_callbacks',model_svd='model_SVD_calls',
 array_svd='array_SVD_calls',solve='solve_calls',signal_compile='signal_compile_calls',program_compile='program_compile_calls',
 replay='replay_calls',metric='metric_calls',schedule='schedule_decisions')
P3_MAP=dict(predict_API='predict_calls',predict_samples='predict_samples',predict_context_tokens='predict_frames',
 action_encoder_calls='action_encoder_calls',action_encoder_samples='action_encoder_samples',action_encoder_context_tokens='action_encoder_frames',
 predictor_calls='predictor_calls',predictor_samples='predictor_samples',predictor_context_tokens='predictor_frames',
 pred_proj_calls='pred_proj_calls',pred_proj_rows='pred_proj_rows',encoder_calls='encode_calls',encoder_samples='encode_samples',
 projector_calls='projector_calls',projector_samples='projector_samples',jacobian_API='jacobian_requests',VJP_directions='VJP_directions',
 grad_API='grad_API',raw_backward_callbacks='raw_backward_callbacks',model_SVD_calls='model_SVD',array_SVD_calls='array_SVD',solve_calls='solve')
P2_MAP=dict(predict_API='predict_calls',predict_samples='predict_samples',predictor_context_tokens='predict_frames',
 action_encoder='action_encoder_calls',predictor='predictor_calls',pred_proj='pred_proj_calls',pred_proj_rows='pred_proj_rows',
 Jacobian='jacobian_requests',Jacobian_elements_requested='jacobian_elements',mathematical_VJP_requested='VJP_directions',
 AD_primal_packet_steps='AD_primal_packets',scalar_backward='scalar_backward',Adam_step='Adam_step',virtual_noise='virtual_noise',
 root_SVD='array_SVD',root_ridge='solve',encoder='encode_calls',projector='projector_calls',physics='physics_step',training='training')
IDENTITY=('model','task','goal','normalizer','history','action_domain')
P2='agent_p2_common_resource_grasp_cuda_20261029_v1'
P3='agent_p3_effective_schedule_20261029_v1'
SOURCE_PINS={P2:'ea763d96eee1040652a3ea02badcb5e8e37b5781530da721f73d2a63d9cb147d',
 P3:'7cef2600e9844d981f97fdd6c385bff0c3ae9bdf55713ac8c92cebd28b003c23'}


def need(ok,message):
    if not ok:raise ValueError(message)


def integer(v):need(type(v) is int and v>=0,'nonnegative literal integer, not bool')
def finite(v):need(type(v) in (int,float) and math.isfinite(v) and v>=0,'finite nonnegative wall')
def raw(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def hash_value(v):need(type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v),'explicit SHA256')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def unique(pairs):
    out={}
    for k,v in pairs:need(k not in out,'duplicate JSON field');out[k]=v
    return out


def read_bound(ref):
    need(set(ref)=={'path','sha256'},'exact external file reference');hash_value(ref['sha256'])
    p=Path(ref['path']).absolute()
    need(p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=CAP,'bounded nonsymlink saved JSON')
    body=p.read_bytes();need(hashlib.sha256(body).hexdigest()==ref['sha256'],'saved file exact SHA')
    value=json.loads(body,object_pairs_hook=unique);raw(value);return value


def empty_vector():
    return {k:dict(unit=u,attempted=None,completed=None,status='unknown',evidence='not_measured') for k,u in AXES.items()}


def put(v,k,a,c,evidence):
    need(k in AXES and type(evidence) is str and evidence,'known axis/evidence')
    integer(a);integer(c);need(c<=a,'completion cannot exceed nonrefunded attempts')
    need(v[k]['status']=='unknown','no duplicate axis or overlapping charge')
    v[k]=dict(unit=AXES[k],attempted=a,completed=c,status='known',evidence=evidence)


def validate_vector(v):
    need(type(v) is dict and set(v)==set(AXES),'complete resource axes, no silent omission')
    for k,x in v.items():
        need(set(x)=={'unit','attempted','completed','status','evidence'} and x['unit']==AXES[k] and
            type(x['evidence']) is str and x['evidence'],'axis ABI/unit/evidence')
        if x['status']=='unknown':need(x['attempted'] is x['completed'] is None,'unknown never becomes zero')
        else:
            need(x['status']=='known','explicit known/unknown');integer(x['attempted']);integer(x['completed'])
            need(x['completed']<=x['attempted'],'no refunds or invented completion')
    return v


def sum_vectors(vectors):
    need(type(vectors) is list and vectors,'nonempty explicit charge list')
    for v in vectors:validate_vector(v)
    out=empty_vector()
    for k in AXES:
        if all(v[k]['status']=='known' for v in vectors):
            put(out,k,sum(v[k]['attempted'] for v in vectors),sum(v[k]['completed'] for v in vectors),'sum_disjoint_explicit_charges')
    return out


def p3_delta(op):
    """Independent frozen P3 metadata arithmetic, not core.delta import."""
    k=op['resource'];s=op['shape'];d=op['directions'];out=dict.fromkeys(P3_AXES,0)
    if k in SCALARS:need(s is d is None,'scalar resource metadata');out[SCALARS[k]]=1
    elif k=='vjp':integer(d);need(d>0 and s is None,'VJP request metadata');out['VJP_directions']=d
    else:
        need(type(s) is list and all(type(n) is int and n>0 for n in s) and d is None,'literal native shape')
        if k in ('predict','action_encoder','predictor'):
            need(len(s)==3 and s[1]==3,'original full3 native ABI')
            names=('predict_API','predict_samples','predict_context_tokens') if k=='predict' else (k+'_calls',k+'_samples',k+'_context_tokens')
            out.update(zip(names,(1,s[0],s[0]*s[1])))
        elif k=='pred_proj':need(len(s)==2,'flattened projection');out.update(pred_proj_calls=1,pred_proj_rows=s[0])
        elif k in ('encoder','projector'):need(len(s)>=2,'native acquisition shape');out[k+'_calls']=1;out[k+'_samples']=s[0]
        else:raise ValueError('unknown P3 operator, not skipped')
    need(raw(out)==raw(op['delta']),'independent requested delta matches original ledger')
    return out


def p3_ledger(receipt):
    attempt=dict.fromkeys(P3_AXES,0);complete=dict(attempt);wall={}
    need(set(receipt['caps'])==set(P3_AXES),'original all-axis caps')
    for k,v in receipt['caps'].items():integer(v)
    failure_times=[];failed_indices=[];blocked_indices=[];last_start=-1;intervals=[]
    for i,op in enumerate(receipt['events']):
        need(op['index']==i and type(op['index']) is int,'ordered operation ENTRY indices, not return order')
        delta=p3_delta(op);integer(op['started_ns']);status=op['status']
        need(op['started_ns']>=last_start,'monotone entry times');last_start=op['started_ns']
        if status=='blocked_before_dispatch':
            need(op['ended_ns'] is None and any(attempt[k]+delta[k]>receipt['caps'][k] for k in P3_AXES),'actual predispatch refusal')
            failure_times.append(op['started_ns']);blocked_indices.append(i);continue
        need(status in ('returned','failed'),'terminal saved operation disposition')
        integer(op['ended_ns']);need(op['ended_ns']>=op['started_ns'],'same-node start/end')
        while intervals and op['started_ns']>=intervals[-1][1]:intervals.pop()
        if intervals:need(op['ended_ns']<=intervals[-1][1],'nested intervals cannot cross parent end')
        intervals.append((op['started_ns'],op['ended_ns']))
        wall[op['resource']]=wall.get(op['resource'],0)+op['ended_ns']-op['started_ns']
        for k,d in delta.items():
            attempt[k]+=d;need(attempt[k]<=receipt['caps'][k],'actual cap, no refund')
            if status=='returned':complete[k]+=d
        if status=='failed':failure_times.append(op['ended_ns']);failed_indices.append(i)
    if failure_times:
        first_failure=min(failure_times)
        # Parent failures occur on unwind and appear BEFORE their child entries
        # in this array. Children that began before the failure remain paid.
        need(all(op['started_ns']<=first_failure for op in receipt['events']),
            'no new dispatch after earliest failure/refusal boundary')
    need(raw(attempt)==raw(receipt['attempted']) and raw(complete)==raw(receipt['completed']) and
        raw(wall)==raw(receipt['inclusive_operation_wall_ns']),'independent ledger counts/wall')
    v=empty_vector()
    for src,dst in P3_MAP.items():put(v,dst,attempt[src],complete[src],'independent_saved_P3_ledger')
    return dict(vector=v,complete=not failure_times,
        disposition='failed_or_blocked_nonrefunded' if failure_times else 'returned_metadata_only',
        failed_operation_indices=failed_indices,blocked_operation_indices=blocked_indices,
        wall=dict(scope='node_inclusive_nested_operations',by_operator_ns=wall,
        additive=False,includes=['attempt_sink','body','nested_IO_and_checks'],excludes=['own_return_sink'],
        total_seconds=None,status='not_whole_wall'))


def p2_vector(resource):
    """Saved strict counters are inputs, NOT newly recertified native events."""
    need(type(resource['attempted']) is dict and type(resource['completed']) is dict,'saved actual counters')
    v=empty_vector()
    for src,dst in P2_MAP.items():
        a=resource['attempted'].get(src);c=resource['completed'].get(src)
        if a is None or c is None:need(a is c is None,'missing counts must be paired');continue
        put(v,dst,a,c,'SHA_bound_saved_receipt_counter_not_rebuilt')
    # Current fixed producer declares batch1/context3 for these callbacks.
    a=resource['attempted'];c=resource['completed']
    if 'predict_samples' in a and a['predict_samples'] is not None:
        for d in (a,c):
            need(d['predict_samples']==d['predictor'] and d['predictor_context_tokens']==3*d['predict_samples'] and
                d['pred_proj_rows']==3*d['pred_proj'],'fixed batch1/full3 counter relationships')
        put(v,'predictor_samples',a['predict_samples'],c['predict_samples'],'fixed_P2_batch1_saved_counters')
        put(v,'predictor_frames',a['predictor_context_tokens'],c['predictor_context_tokens'],'fixed_P2_full3_saved_counters')
        put(v,'action_encoder_samples',a['action_encoder'],c['action_encoder'],'fixed_P2_batch1_saved_counters')
        put(v,'action_encoder_frames',3*a['action_encoder'],3*c['action_encoder'],'fixed_P2_full3_saved_counters')
    # Explicitly unavailable engine internals cannot be mapped to VJP requests.
    need(all(resource.get(k) is None for k in ('engine_internal_backward_calls','independent_JVP_calls','total_FLOPs')),
        'new engine/FLOPs schema requires a new adapter, not a guessed conversion')
    return validate_vector(v)


def wall(scope,seconds,includes,excludes):
    need(type(scope) is str and scope and type(includes) is list and type(excludes) is list and
        not set(includes)&set(excludes),'explicit noncontradictory wall boundary')
    if seconds is not None:finite(seconds)
    return dict(scope=scope,seconds=seconds,status='unknown' if seconds is None else 'known',
        includes=includes,excludes=excludes,additive=False)


def p2_saved(report,receipt):
    need(report['schema']==P2+'.report' and receipt['schema']==P2+'.consume' and
        receipt['status']=='all6_common_cap_records_verified_not_admitted' and
        receipt['fixed_denominator']==6 and report['fixed_denominator']==6,'exact existing P2 six-slot schema')
    need(len(receipt['resources'])==len(receipt['accounts'])==len(report['rows'])==6,'all six including missing exports')
    order=[(0,'joint'),(0,'composition'),(0,'grasp'),(1,'grasp'),(1,'composition'),(1,'joint')]
    rows=[]
    for i,((root,arm),r,res,account) in enumerate(zip(order,report['rows'],receipt['resources'],receipt['accounts'],strict=True)):
        need(r['index']==i and r['root_slot']==root and r['arm']==arm,'fixed original six-slot ordering')
        rows.append(dict(slot_id=f'{root}:{arm}',root_key=str(root),strategy=arm,resources=p2_vector(res),
            wall=wall('P2_search_synchronized_450s',account['elapsed_seconds'],
                ['nominal','bank','AD','CPU_solve','capture','search_IO','export'],['loader','saved_strict','post_search_persistence']),
            identity=None,missing_identity=list(IDENTITY)))
    return dict(rows=rows,stage_wall=deepcopy(report['timings']),
        inherited_once=deepcopy(receipt['inherited_encoding_cohort']),
        inherited_allocation='unknown; never split or charge per arm',full_strict_rebuilt_here=False)


def p3_saved(report,receipt):
    need(report['schema']==P3+'.report' and receipt['schema']==P3+'.consume' and receipt['complete'] is True and
        report['fixed_denominator']==receipt['fixed_denominator']==8 and
        report['node_denominator']==receipt['node_denominator']==24,'exact eight-arm/24-node P3')
    need(raw(report['counts'])==raw(receipt['counts']) and
        raw(report['per_episode_neural'])==raw(receipt['per_episode_neural']),'P3 saved census linkage, not a cryptographic report signature')
    arms=('commit15_endpoints','commit15_midpoints','feedback5_endpoints','feedback5_midpoints')
    expected=[(s,a) for s in (0,1) for a in arms]
    need([(r['source_slot'],r['mode']) for r in report['rows']]==expected and
        [(n['episode_id'],n['boundary']) for n in report['nodes']]==[(f's{s}_{a}',b) for s,a in expected for b in (5,10,15)],'fixed8/24 no filtering')
    rows=[]
    native=receipt['physical_census']['per_episode']
    for row in report['rows']:
        eid=row['episode_id'];nodes=[n for n in report['nodes'] if n['episode_id']==eid]
        parts=[p3_ledger(n['resource_receipt']) for n in nodes if n['resource_receipt'] is not None]
        # Missing/unreached nodes are not invented paid operations. Empty traces
        # cannot establish which neural axes were physically attempted.
        v=sum_vectors([n['vector'] for n in parts]) if parts else empty_vector()
        counts=native[eid]
        mapping=dict(encoder='encode_calls',projector='projector_calls',public_step='public_step',physics_step='physics_step',
            explicit_render='render',reset='reset',goal_bind='goal_bind')
        for src,dst in mapping.items():
            pair=counts[src]
            # Neural ledger forbids acquisition; native receipt owns paid RGB.
            if v[dst]['status']=='known':need(v[dst]['attempted']==v[dst]['completed']==0,'no double-owned visual calls');v[dst]=empty_vector()[dst]
            put(v,dst,pair['attempted'],pair['completed'],'SHA_bound_native_census_not_rebuilt')
        # Zero acquisition samples in the compiler ledger say nothing about
        # the separately paid RGB acquisition's batch/frame dimensions.
        for k in ('encode_samples','encode_frames','projector_samples','projector_frames'):
            v[k]=empty_vector()[k]
        put(v,'RGB_feedback',counts['encoder']['attempted'],counts['encoder']['completed'],'one_actual_boundary_encoding_per_native_receipt')
        rows.append(dict(slot_id=eid,root_key=str(row['source_slot']),strategy=row['mode'],resources=v,
            wall=wall('per_episode_whole_wall_not_instrumented',None,[],[]),node_walls=[n['wall'] for n in parts],
            resource_trace_complete=all(n['complete'] for n in parts),
            resource_failures=[dict(node=i,failed=n['failed_operation_indices'],blocked=n['blocked_operation_indices'])
                for i,n in enumerate(parts) if not n['complete']],
            control_returned=row['control_returned'],identity=None,missing_identity=list(IDENTITY)))
    return dict(rows=rows,stage_wall=wall('P3_whole_job',report['wall_seconds'],
        ['load','source_checks','compile','RGB','physics','journal','final_checks'],['offline_saved_strict']),
        inherited_once=None,inherited_allocation='unknown',full_strict_rebuilt_here=False)


def validate_shared_panel(rows,expected_slots,compared_axes,caps,wall_scope=None,wall_cap_seconds=None):
    """Production-callable pure admission check for a NEW P3 shared panel.

    Caller supplies predeclared identities/caps and fixed slot ordering. Returns
    blockers instead of relabeling missing evidence as budget matching.
    """
    need(type(rows) is list and type(expected_slots) is list and expected_slots and len(set(expected_slots))==len(expected_slots)
        and [r['slot_id'] for r in rows]==expected_slots,'complete ordered preregistered panel')
    need(type(compared_axes) is list and compared_axes and len(set(compared_axes))==len(compared_axes) and
        set(compared_axes)<=set(AXES) and type(caps) is dict and set(caps)<=set(AXES),'explicit separate comparison axes/caps')
    blockers=[];byroot={}
    need((wall_scope is None)==(wall_cap_seconds is None),'wall contract needs scope and cap together')
    if wall_scope is not None:
        need(type(wall_scope) is str and wall_scope,'explicit common wall boundary');finite(wall_cap_seconds)
    for row in rows:
        validate_vector(row['resources']);identity=row.get('identity')
        if row.get('resource_trace_complete') is False:blockers.append(row['slot_id']+':failed_or_blocked_resource_trace')
        if identity is None:blockers.append(row['slot_id']+':missing_identity')
        else:
            need(set(identity)==set(IDENTITY),'all shared identity fields')
            for value in identity.values():hash_value(value)
            prior=byroot.setdefault(row['root_key'],identity)
            if raw(prior)!=raw(identity):blockers.append(row['slot_id']+':different_task_model_goal_normalizer_history_domain')
        for k,cap in caps.items():
            integer(cap);x=row['resources'][k]
            if x['status']=='unknown':blockers.append(row['slot_id']+':unknown_cap_axis:'+k)
            elif x['attempted']>cap:blockers.append(row['slot_id']+':cap_exceeded:'+k)
        if wall_scope is not None:
            w=row['wall']
            if w['scope']!=wall_scope:blockers.append(row['slot_id']+':wrong_wall_scope')
            if w['status']!='known' or w['seconds'] is None:blockers.append(row['slot_id']+':unknown_wall')
            else:
                finite(w['seconds'])
                if w['seconds']>wall_cap_seconds:blockers.append(row['slot_id']+':wall_cap_exceeded')
    for root in dict.fromkeys(r['root_key'] for r in rows):
        cohort=[r for r in rows if r['root_key']==root]
        need(len(cohort)>=2,'comparison needs multiple predeclared strategies per root')
        for k in compared_axes:
            xs=[r['resources'][k] for r in cohort]
            if any(x['status']=='unknown' for x in xs):blockers.append(root+':unknown_comparison_axis:'+k)
            elif len({(x['attempted'],x['completed']) for x in xs})!=1:blockers.append(root+':unequal_actual_axis:'+k)
        scopes=[r['wall']['scope'] for r in cohort]
        if len(set(scopes))!=1:blockers.append(root+':different_wall_boundaries')
    return dict(schema=NAME+'.panel_validation',status='blocked' if blockers else 'selected_resource_axes_equal_only',
        blockers=blockers,compared_axes=compared_axes,caps=caps,full_strict_rebuilt_here=False,
        wall_scope=wall_scope,wall_cap_seconds=wall_cap_seconds,equal_wall=False,
        total_FLOPs='unknown',operator_FLOPs={k:'unknown' for k in AXES},equal_FLOPs=False,equal_total_compute=False,
        P1_admission=False,P2_admission=False,P3_admission=False,scientific_admission=False)


def build_manifest(manifest):
    """Hash-bound report/receipt/source ABI, deliberately NOT full strict."""
    need(manifest['schema']==NAME+'.inputs' and manifest['kind'] in ('P2_common','P3_effective'),'supported existing resource consumers')
    report=read_bound(manifest['report']);receipt=read_bound(manifest['receipt'])
    bindings=[manifest['report'],manifest['receipt'],*manifest['sources']]
    need(manifest['sources'],'explicit source SHA authority, no source-less manifest')
    for ref in manifest['sources']:
        hash_value(ref['sha256']);need(sha(ref['path'])==ref['sha256'],'bound source unchanged')
    source=P2 if manifest['kind']=='P2_common' else P3
    source_ref=dict(path=str(ROOT/'experiments/spectral_reachability'/(source+'.py')),sha256=SOURCE_PINS[source])
    need(source_ref in manifest['sources'],'explicit frozen actual producer source')
    if 'report_sha256' in receipt:need(receipt['report_sha256']==manifest['report']['sha256'],'receipt report join')
    else:
        need(manifest['kind']=='P3_effective','no implicit missing report authority')
        # Frozen P3 receipt lacks report SHA. Bind prepare SHA through both sides;
        # retain that weaker ABI rather than fabricate a signed report link.
        need(manifest.get('prepared') is not None,'P3 requires actual prepare file binding')
        prepared=read_bound(manifest['prepared']);bindings.append(manifest['prepared'])
        need(report['prepare_sha256']==manifest['prepared']['sha256'] and prepared['schema']==P3+'.prepare','P3 actual prepare join')
    panel=p2_saved(report,receipt) if manifest['kind']=='P2_common' else p3_saved(report,receipt)
    for ref in bindings:need(sha(ref['path'])==ref['sha256'],'inputs stable after ABI arithmetic')
    return dict(schema=NAME+'.resource_panel',**panel,evidence='SHA_bound_saved_ABI_only_not_recertified',
        input_manifest=deepcopy(manifest),total_FLOPs='unknown',equal_FLOPs=False,equal_total_compute=False,
        new_neural_calls=0,new_physics_calls=0,scientific_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('saved','validate-panel'))
    p.add_argument('--input',required=True);p.add_argument('--input-sha',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();value=read_bound(dict(path=a.input,sha256=a.input_sha))
    result=build_manifest(value) if a.command=='saved' else validate_shared_panel(**value)
    path=Path(a.output).absolute();body=raw(result)
    need(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and not path.exists() and path.resolve()==path and len(body)<=CAP,'new exclusive bounded output')
    with path.open('xb') as f:need(f.write(body)==len(body),'full write');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    print(raw(dict(path=str(path),sha256=sha(path),full_strict_rebuilt_here=False)).decode())


if __name__=='__main__':main()
