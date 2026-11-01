"""Independent saved-only shared-panel contrasts; no producer arithmetic imports.

No strict replay/NN/physics is invoked. MAIN must supply the final own strict
receipt for the own shared-panel ABI before production analysis.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import numpy as np

NAME='agent_p3_shared_panel_summary_20261029_v1'
PRODUCER='agent_p3_shared_panel_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
ARMS=('fixedH1_endpoints','fixedH1_midpoints','adaptive_endpoints','adaptive_midpoints')
BOUNDARIES=(5,10,15)
CHECKPOINTS=(0,5,10,15,20)
AXES=('predict_API','predict_samples','predict_context_tokens','action_encoder_calls',
    'action_encoder_samples','action_encoder_context_tokens','predictor_calls','predictor_samples',
    'predictor_context_tokens','pred_proj_calls','pred_proj_rows','encoder_calls','encoder_samples',
    'projector_calls','projector_samples','jacobian_API','VJP_directions','grad_API',
    'raw_backward_callbacks','model_SVD_calls','array_SVD_calls','solve_calls',
    'signal_compile_calls','program_compile_calls','replay_calls','metric_calls','schedule_decisions')
NATIVE_AXES=('constructor','reset','set_state','goal_render','goal_setter','goal_bind',
    'public_step','physics_step','explicit_render','close','encoder','projector')
ERRORS=('joint_xy','agent_xy','object_xy','angle_wrapped','native_angle_term',
    'velocity_error','native7_distance','native7_reward')
META=256*1024**2
ARRAY_CAP=2*1024**2


def require(ok,message):
    if not ok:raise ValueError(message)


def raw(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(raw(x)).hexdigest()
def same(a,b,message):require(raw(a)==raw(b),message)


def absolute(value):
    require(type(value) is str and Path(value).is_absolute() and str(Path(value).resolve())==value,'canonical absolute path')
    return Path(value)


def sha(path):
    p=Path(path);require(p.is_file() and not p.is_symlink(),'regular non-symlink file')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def hash_value(h):require(type(h) is str and len(h)==64 and h!='0'*64 and all(x in '0123456789abcdef' for x in h),'actual SHA256')


def unique(items):
    out={}
    for k,v in items:require(k not in out,'duplicate JSON key');out[k]=v
    return out


def read(path,expected=None):
    p=absolute(str(path));require(p.is_file() and not p.is_symlink() and p.stat().st_size<=META,'bounded regular JSON')
    if expected is not None:hash_value(expected);require(sha(p)==expected,'external actual file SHA')
    def invalid(_):raise ValueError('nonfinite JSON literal')
    obj=json.loads(p.read_bytes(),object_pairs_hook=unique,parse_constant=invalid);raw(obj);return obj


def integer(v,lo=0,hi=None):require(type(v) is int and v>=lo and (hi is None or v<=hi),'literal bounded integer')


def finite(value,shape):
    a=np.asarray(value,np.float64);require(a.shape==shape and np.isfinite(a).all(),'exact finite numeric shape')
    require(not any(type(x) is bool for x in np.asarray(value,dtype=object).flat),'no bool numeric values')
    return a


def raw_bits(actions):
    a=finite(actions,(len(actions),2));b=a.astype(np.float32)
    require(np.isfinite(b).all() and (abs(b)<=1).all() and np.array_equal(a,b.astype(np.float64)),'literal float32 raw box without clipping')
    return b.tobytes()


def latent_q(goal,z):return float(np.mean((finite(z,(192,))-finite(goal,(192,)))**2,dtype=np.float64))


def physical_metric(goal,state):
    d=finite(state,(7,))-finite(goal,(7,));a=float(abs(d[4]));native=float(min(a,2*np.pi-a))
    xy=float(np.linalg.norm(d[:4]));distance=float(np.linalg.norm(d))
    return dict(joint_xy=xy,agent_xy=float(np.linalg.norm(d[:2])),object_xy=float(np.linalg.norm(d[2:4])),
        angle_wrapped=float(abs(np.arctan2(np.sin(d[4]),np.cos(d[4])))),native_angle_term=native,
        velocity_error=float(np.linalg.norm(d[5:])),native7_distance=distance,native7_reward=-distance,
        native7_delta=d.tolist(),native_success=bool(xy<20 and native<np.pi/9))


def error_decomposition(goal,pred,actual):
    p,a,g=(finite(v,(192,)) for v in (pred,actual,goal));e=p-a;d=a-g
    pq,aq=latent_q(g,p),latent_q(g,a);quadratic=float(np.mean(e*e));cross=float(2*np.mean(d*e))
    residual=(pq-aq)-(cross+quadratic)
    eps=np.finfo(np.float64).eps;k=16*192+64;gamma=k*eps/(1-k*eps)
    scale=float(np.mean((abs(p)+abs(g))**2+(abs(a)+abs(g))**2+(abs(p)+abs(a))**2));bound=float(gamma*scale)
    require(abs(residual)<=bound,'quadratic identity arithmetic bound, not scientific tolerance')
    return dict(model_Q=pq,real_Q=aq,model_minus_real_Q=pq-aq,cross=cross,quadratic=quadratic,
        identity_rounding_residual=residual,identity_rounding_bound=bound,normalization='mean192 not sum',scientific_tolerance_changed=False)


def sign(x):return 'missing' if x is None else 'negative' if x<0 else 'positive' if x>0 else 'zero'


def node_costs():
    out=dict.fromkeys(AXES,0)
    out.update(predict_API=44,predict_samples=44,predict_context_tokens=132,
        action_encoder_calls=44,action_encoder_samples=44,action_encoder_context_tokens=132,
        predictor_calls=44,predictor_samples=44,predictor_context_tokens=132,pred_proj_calls=44,pred_proj_rows=132,
        jacobian_API=2,VJP_directions=384,grad_API=384,raw_backward_callbacks=384,
        array_SVD_calls=4,solve_calls=2,signal_compile_calls=2,program_compile_calls=2,
        replay_calls=18,metric_calls=8,schedule_decisions=1)
    return out


def counters(axes):return {k:dict(attempted=0,completed=0) for k in axes}


def unknown_work():
    return {k:dict(attempted=None,completed=None,status='unknown',evidence='not separately instrumented in original ledger')
        for k in ('scalar_backward','AD_primal_packets','jacobian_elements','engine_backward','independent_JVP',
                  'actual_RGB_encoder_samples','actual_RGB_encoder_frames','actual_RGB_projector_samples','actual_RGB_projector_frames','total_FLOPs')}


def operation_delta(op):
    out=dict.fromkeys(AXES,0);resource=op['resource'];shape=op['shape'];directions=op['directions']
    scalar=dict(jacobian='jacobian_API',grad='grad_API',raw_backward='raw_backward_callbacks',
        model_svd='model_SVD_calls',array_svd='array_SVD_calls',solve='solve_calls',
        signal_compile='signal_compile_calls',program_compile='program_compile_calls',replay='replay_calls',
        metric='metric_calls',schedule='schedule_decisions')
    if resource in scalar:
        require(shape is directions is None,'scalar resource metadata');out[scalar[resource]]=1
    elif resource=='vjp':
        integer(directions,1);require(shape is None,'direction resource metadata');out['VJP_directions']=directions
    else:
        require(type(shape) is list and all(type(x) is int and x>0 for x in shape) and directions is None,'actual input dimensions')
        if resource in ('predict','action_encoder','predictor'):
            require(len(shape)==3 and shape[1]==3,'full3 model input')
            if resource=='predict':out.update(predict_API=1,predict_samples=shape[0],predict_context_tokens=3*shape[0])
            else:out.update({resource+'_calls':1,resource+'_samples':shape[0],resource+'_context_tokens':3*shape[0]})
        elif resource=='pred_proj':
            require(len(shape)==2,'projection input rows');out.update(pred_proj_calls=1,pred_proj_rows=shape[0])
        elif resource in ('encoder','projector'):
            require(len(shape)>=2,'visual input layout');out[resource+'_calls']=1;out[resource+'_samples']=shape[0]
        else:raise ValueError('unknown model resource, never treat missing axes as zero')
    return out


def paid_counts(report,events):
    """Full vector counts, not just predict; failed attempts remain charged."""
    ids=[r['episode_id'] for r in report['rows']]
    per={eid:dict(neural=counters(AXES),native=counters(NATIVE_AXES),inclusive_operation_wall_ns={},pending_native=[],
        unknown_work=unknown_work(),wall=dict(scope='whole_episode',seconds=None,status='unknown',additive=False),
        phases={phase:counters(AXES) for phase in ('signal_and_policy','program')}) for eid in ids}
    pending={};serial=-1
    for event in events:
        if event['kind'] not in ('operation_attempted','operation_completed'):continue
        eid,k,ident,units=(event[x] for x in ('episode_id','resource','operation_id','units'))
        require(eid in per and k in NATIVE_AXES,'known native resource/episode');integer(ident);integer(units,1)
        binding=(eid,k,units,event['region'])
        if event['kind']=='operation_attempted':
            require(ident>serial and ident not in pending,'unique increasing native attempt');serial=ident;pending[ident]=binding;a='attempted'
        else:require(pending.pop(ident,None)==binding,'native return exact attempted identity');a='completed'
        per[eid]['native'][k][a]+=units
    for ident,(eid,k,units,region) in pending.items():per[eid]['pending_native'].append(dict(operation_id=ident,resource=k,units=units,region=region))
    same({str(i):dict(operation_id=i,episode_id=eid,resource=k,units=u,region=region) for i,(eid,k,u,region) in pending.items()},
        {str(k):v for k,v in report['pending'].items()},'all unresolved native attempts, no refund')
    for node in report['nodes']:
        rr=node['resource_receipt']
        if rr is None:continue
        require(set(rr['caps'])==set(rr['attempted'])==set(rr['completed'])==set(AXES),'entire model resource vector')
        acc=counters(AXES);wall={};last=-1;intervals=[];failure_times=[]
        require([op['index'] for op in rr['events']]==list(range(len(rr['events']))),'complete ordered model operations')
        for op in rr['events']:
            require(op['status'] in ('returned','failed','blocked_before_dispatch') and set(op['delta'])==set(AXES),'actual operation disposition/full vector')
            same(op['delta'],operation_delta(op),'independent actual shape/directions-to-resource units')
            for k in AXES:integer(op['delta'][k])
            integer(op['started_ns']);require(op['started_ns']>=last,'monotone model attempt starts');last=op['started_ns']
            if op['status']=='blocked_before_dispatch':
                require(op['ended_ns'] is None and any(acc[k]['attempted']+op['delta'][k]>rr['caps'][k] for k in AXES),'real overcap predispatch refusal')
                failure_times.append(op['started_ns'])
                continue
            integer(op['started_ns']);integer(op['ended_ns']);require(op['ended_ns']>=op['started_ns'],'measured operation elapsed')
            while intervals and op['started_ns']>=intervals[-1][1]:intervals.pop()
            if intervals:require(op['ended_ns']<=intervals[-1][1],'nested operation ends within its entered parent')
            intervals.append((op['started_ns'],op['ended_ns']))
            if op['status']=='failed':failure_times.append(op['ended_ns'])
            for k in AXES:
                acc[k]['attempted']+=op['delta'][k]
                if op['status']=='returned':acc[k]['completed']+=op['delta'][k]
                phase='program' if op['scope'].get('phase') in ('program','selected') else 'signal_and_policy'
                per[node['episode_id']]['phases'][phase][k]['attempted']+=op['delta'][k]
                if op['status']=='returned':per[node['episode_id']]['phases'][phase][k]['completed']+=op['delta'][k]
            k=op['resource'];wall[k]=wall.get(k,0)+op['ended_ns']-op['started_ns']
        if failure_times:
            require(all(op['started_ns']<=min(failure_times) for op in rr['events']),'no new dispatch after earliest failure; parent unwind may precede children in entry list')
        for k in AXES:
            integer(rr['caps'][k]);require(acc[k]['completed']<=acc[k]['attempted']<=rr['caps'][k],'actual node resource cap')
            require(acc[k]['attempted']==rr['attempted'][k] and acc[k]['completed']==rr['completed'][k],'independent node vector totals')
        same(wall,rr['inclusive_operation_wall_ns'],'inclusive nested wall, not runtime sum')
        if node['status'] in ('sealed','refused'):same({k:v['completed'] for k,v in acc.items()},node_costs(),'complete two-panel natural work, no padding')
        own=per[node['episode_id']]
        for k in AXES:
            for a in ('attempted','completed'):own['neural'][k][a]+=acc[k][a]
        for k,v in wall.items():own['inclusive_operation_wall_ns'][k]=own['inclusive_operation_wall_ns'].get(k,0)+v
    same({k:{a:sum(per[e]['native'][k][a] for e in ids) for a in ('attempted','completed')} for k in NATIVE_AXES},report['counts'],'all native and visual counts')
    require(len(report['per_episode_neural'])==8,'all eight model accounts')
    for entry,eid in zip(report['per_episode_neural'],ids,strict=True):
        require(entry['episode_id']==eid,'actual producer episode-keyed model account order')
        for k in AXES:
            for a in ('attempted','completed'):require(entry[a][k]==per[eid]['neural'][k][a],'independent whole episode resource vector')
        same(entry['inclusive_operation_wall_ns'],per[eid]['inclusive_operation_wall_ns'],'episode inclusive wall')
        same(entry['phases'],{phase:{k:v['completed'] for k,v in counts.items()}
            for phase,counts in per[eid]['phases'].items()},'independent phase-specific actual vector, not just predict')
        cap={k:3*v for k,v in node_costs().items()};same(entry['caps'],cap,'original full three-node engineering cap')
        require(all(per[eid]['neural'][k]['attempted']<=cap[k] for k in AXES),'non-refundable per episode cap')
        per[eid]['neural_caps']=cap
    return per


def row_summary(row,nodes,events,base,counts,load_array):
    eid=row['episode_id'];goal=base['goal']
    require(np.array_equal(finite(base['projection'],(192,192)),np.eye(192)),'original identity192 Q, no fit')
    integer(row['control_returned'],0,20)
    require(type(row['initial_exact']) is type(row['terminated']) is type(row['truncated']) is bool,'native row flags')
    observations=[e['receipt'] for e in events if e.get('episode_id')==eid and e['kind']=='fresh_observation' and e['receipt']['phase']=='plan']
    expected=list(range(row['control_returned']+1)) if row['initial_exact'] else []
    require([o['step'] for o in observations]==expected,'every actual step including initial, no missing frame repair')
    dispatch=[e for e in events if e.get('episode_id')==eid and e['kind']=='authorized_dispatch']
    returned=[e for e in events if e.get('episode_id')==eid and e['kind']=='control_return']
    require([e['step'] for e in dispatch]==[e['step'] for e in returned]==list(range(1,row['control_returned']+1))
        and len(row['executed_raw'])==row['control_returned'],'one authorized actual return per executed raw')
    for e,f,action in zip(dispatch,returned,row['executed_raw'],strict=True):
        require(raw_bits([e['command']])==raw_bits([f['command']])==raw_bits([action]),'actual command signed bits')
        require(type(f['terminated']) is type(f['truncated']) is bool,'native boolean return flags')
    require(all(not(e['terminated'] or e['truncated']) for e in returned[:-1]),'no dispatch after native early stop')
    if returned:
        require((returned[-1]['terminated'],returned[-1]['truncated'])==(row['terminated'],row['truncated']),'last actual return flags')
    native_goal=None if not observations else observations[0]['goal']
    steps=[None]*21
    for ob in observations:
        b=ob['step'];metrics=physical_metric(ob['goal'],ob['state'])
        # Shared task identity is fixed; the goal cannot be reconstructed from outcomes.
        same(ob['goal'],native_goal,'strict-authenticated native goal unchanged across actual observations')
        if b:
            ret=returned[b-1]
            require(ret['reward']==metrics['native7_reward'] and ret['terminated'] is metrics['native_success'],'native7 reward/success independently recomputed')
        steps[b]=dict(step=b,state=deepcopy(ob['state']),metrics=metrics,pixel_sha256=ob['pixel_sha256'],
            file_sha256=ob['file_sha256'],actual_Q=None,latent=None,latent_source=None)
    if steps[0] is not None:
        z=base['root_context']['latent_history'][-1]
        steps[0].update(actual_Q=latent_q(goal,z),latent=deepcopy(z),latent_source='sealed_original_root_encoding')
    for enc in row['encodings']:
        b=enc['step'];integer(b,0,20)
        require(b in (5,10,15,20) and steps[b] is not None and steps[b]['latent'] is None,'only unique actual reached encoding slots')
        z=load_array(enc['latent']);require(z.dtype==np.float32 and z.shape==(192,) and np.isfinite(z).all(),'original saved real192 NPY')
        steps[b].update(actual_Q=latent_q(goal,z),latent=z.tolist(),latent_source='actual_RGB_this_run')
    node_rows=[];cached=deepcopy(base['root_context'])
    for node in nodes:
        b=node['boundary'];receipt=node['receipt'];panels=[]
        rr=node['resource_receipt']
        item=dict(boundary=b,status=node['status'],decision=None,executed_horizon=None,actual_context_Q=None,panels=panels,
            resources=None if rr is None else dict(attempted=deepcopy(rr['attempted']),completed=deepcopy(rr['completed']),caps=deepcopy(rr['caps'])),
            unknown_work=unknown_work(),node_wall=dict(scope='node_Ledger_lifetime',nanoseconds=None if rr is None else rr['wall_ns'],
                status='not_reached' if rr is None else 'measured',additive=False))
        if receipt is None:
            require(node['status'] not in ('sealed','refused') and row['control_returned']<=b,
                'actual execution cannot pass an absent shared node');node_rows.append(item);continue
        req=node['request'];same(req['task']['goal'],goal,'original model goal in request')
        same(req['task']['projection'],base['projection'],'same original task Q')
        require(steps[b] is not None and steps[b]['latent'] is not None,'paid real observation before model request')
        context=req['task']['root_context'];same(context['latent_history'][-1],steps[b]['latent'],'actual192 context not own-imagined')
        expected_latents=deepcopy(cached['latent_history'][1:])+[deepcopy(steps[b]['latent'])]
        same(context['latent_history'],expected_latents,'full rolling actual3, not just latest latent')
        history=finite(req['raw10'],(10,2));raw_bits(history.tolist())
        mean=finite(base['action_mean'],(2,)).astype(np.float32);std=finite(base['action_std'],(2,)).astype(np.float32)
        require((std>0).all(),'original positive normalizer')
        normalized=((history.astype(np.float32)-mean)/std).reshape(2,10)
        same(context['action_history'],normalized.tolist(),'both original float32 incoming history tokens exactly once')
        same(req['task']['action_mean'],base['action_mean'],'same original action mean')
        same(req['task']['action_std'],base['action_std'],'same original action std')
        cached=deepcopy(context)
        item['actual_context_Q']=latent_q(goal,context['latent_history'][-1])
        if node['status'] not in ('sealed','refused'):
            item['partial_receipt_present']=True;node_rows.append(item);continue
        require(receipt['status']==node['status'] and type(receipt['panels']) is list and len(receipt['panels'])==2
            and [v['horizon'] for v in receipt['panels']]==[1,3],'ABI requires full H1/H3 panel receipts, no old single panel')
        policy_record=receipt['policy'];decision=policy_record['decision']
        require(type(policy_record['supported']) is bool,'literal original supported-policy flag')
        policy,bank=row['mode'].split('_',1)
        require(policy in ('fixedH1','adaptive') and bank in ('endpoints','midpoints'),'predeclared policy/bank')
        chosen=receipt['selected_horizon']
        if node['status']=='sealed':
            require(policy_record['supported'] is True and chosen==decision['horizon'] and chosen in (1,3),'actual supported horizon')
            if policy=='fixedH1':require(chosen==1,'fixed original H1')
        else:
            require(policy_record['supported'] is False and chosen is receipt['commands'] is None,'no fallback from unsupported policy')
        item.update(decision=deepcopy(decision),executed_horizon=chosen)
        fractions=[0.,1/3,2/3,1.] if bank=='endpoints' else [1/8,3/8,5/8,7/8]
        for panel in receipt['panels']:
            h=panel['horizon'];candidates=panel['candidates'];actions=panel['bank']['actions']
            require(len(candidates)==len(actions)==4 and panel['bank']['fractions']==fractions,'all four predetermined candidates; no dropping collisions')
            costs=[];records=[]
            for i,candidate in enumerate(candidates):
                require(len(candidate['actions'])==h*5 and len(candidate['whole'])==h,'full nonlinear H-step witness')
                require(raw_bits(candidate['actions'])==raw_bits(actions[i]),'bank/candidate raw byte identity')
                cost=latent_q(goal,candidate['whole'][-1]['latent_history'][-1])
                require(cost==candidate['cost'],'independent saved candidate model MSE')
                costs.append(cost);records.append(dict(index=i,fraction=fractions[i],model_Q=cost))
            winner=panel['winner']['index'];integer(winner,0,3)
            require(winner==min(range(4),key=lambda i:(costs[i],i)),'saved winner stable(cost,index); not new selection')
            same(panel['winner'],candidates[winner],'whole original winner, not reconstructed release')
            same(panel['selected_path'],candidates[winner]['whole'],'original independent winner replay')
            endpoint=b+h*5;actual=steps[endpoint] if endpoint<=20 else None
            available=actual is not None and actual['latent'] is not None
            dec=error_decomposition(goal,candidates[winner]['whole'][-1]['latent_history'][-1],actual['latent']) if available else None
            panels.append(dict(horizon=h,selected_for_execution=h==chosen,winner_index=winner,
                all_candidate_Q=records,predicted_endpoint_step=endpoint,winner_predicted_Q=costs[winner],
                actual_Q_at_predicted_endpoint=None if not available else actual['actual_Q'],error_decomposition=dec,
                missing_actual_reason=None if available else 'beyond_executed20_lookahead' if endpoint>20 else 'not_observed_or_not_encoded',
                first5_raw=deepcopy(candidates[winner]['actions'][:5]),
                causal_endpoint_pair=bool(h==chosen and h==1),
                interpretation='H3 future feedback may change commands before its endpoint; prediction-vs-actual is descriptive, not same-action fidelity'))
        if chosen is not None:
            use=next(v for v in panels if v['selected_for_execution'])
            require(raw_bits(receipt['commands'])==raw_bits(use['first5_raw']),'literal released first5')
            count=min(5,max(0,row['control_returned']-b))
            if count:require(raw_bits(use['first5_raw'][:count])==raw_bits(row['executed_raw'][b:b+count]),'selected panel first5 only, actual suffix bits')
        node_rows.append(item)
    terminal=next((x for x in reversed(steps) if x is not None),None)
    if row['control_returned']==20:
        require(steps[20] is not None and steps[20]['actual_Q'] is not None
            and row['actual_Q20']==steps[20]['actual_Q'],'independent real terminal20 Q, not lookahead Q')
    return dict(episode_id=eid,source_slot=row['source_slot'],arm=row['mode'],status=row['status'],
        control_returned=row['control_returned'],warmup_returned=row['warmup_returned'],terminated=row['terminated'],truncated=row['truncated'],
        native_goal=deepcopy(native_goal),steps=steps,checkpoints=[dict(step=b,value=steps[b]) for b in CHECKPOINTS],nodes=node_rows,counts=counts,
        missing_steps=[i for i,x in enumerate(steps) if x is None],eventtime_terminal=terminal,
        native_success_at20=None if steps[20] is None else steps[20]['metrics']['native_success'],
        eventtime_terminal_native_success=None if terminal is None or terminal['step']==0 else terminal['metrics']['native_success'],
        no_terminal_substitution_at20=True,total_FLOPs=None)


def compare(left,right,kind):
    require(left['source_slot']==right['source_slot'],'only same known root paired')
    paired=[]
    for step,(a,b) in enumerate(zip(left['steps'],right['steps'],strict=True)):
        common=a is not None and b is not None
        delta=None if not common else {k:b['metrics'][k]-a['metrics'][k] for k in ERRORS}
        q=None if not common or a['actual_Q'] is None or b['actual_Q'] is None else b['actual_Q']-a['actual_Q']
        paired.append(dict(step=step,left_missing=a is None,right_missing=b is None,right_minus_left=delta,
            signs=None if delta is None else {k:sign(v) for k,v in delta.items()},actual_Q_delta=q,actual_Q_sign=sign(q),
            native_success=dict(left=None if a is None else a['metrics']['native_success'],right=None if b is None else b['metrics']['native_success']),
            state_bits_equal=None if not common else finite(a['state'],(7,)).tobytes()==finite(b['state'],(7,)).tobytes(),
            pixels_equal=None if not common else a['pixel_sha256']==b['pixel_sha256']))
    return dict(source_slot=left['source_slot'],contrast=kind,left=left['episode_id'],right=right['episode_id'],
        direction='right_minus_left',steps=paired,checkpoints=[paired[i] for i in CHECKPOINTS],at20=paired[20],
        eventtime_terminal_steps=dict(left=None if left['eventtime_terminal'] is None else left['eventtime_terminal']['step'],
            right=None if right['eventtime_terminal'] is None else right['eventtime_terminal']['step']),
        resource_deltas={group:{k:{a:right['counts'][group][k][a]-left['counts'][group][k][a]
            for a in ('attempted','completed')} for k in left['counts'][group]} for group in ('neural','native')},
        lower_is_better=[x for x in ERRORS if x!='native7_reward'],higher_is_better=['native7_reward'],
        equal_FLOPs=False,FLOPs=None,equal_wall=False,
        interpretation='same-bank fixed-vs-adaptive shared-panel policy' if kind=='same_bank_policy' else 'same-policy different candidate support, not pure horizon causality')


def summarize(p,r,s,events,load_array):
    require(p['schema']==PRODUCER+'.prepare' and r['schema']==PRODUCER+'.report' and s['schema']==PRODUCER+'.consume',
        'pending own shared-panel ABI only; never relabel effective_schedule')
    same(p['contract'],r['contract'],'same frozen shared-panel contract')
    require(s['complete'] is True and r['errors']==[] and r['pending']=={},'complete external strict required; no invented partial certification')
    for obj in (p['contract'],r,s):
        require(all(obj[k] is False for k in ('P1_admission','P3_admission','freshness','scientific_admission')),'no scientific admission')
    require(r['fixed_denominator']==s['fixed_denominator']==8 and r['node_denominator']==s['node_denominator']==24,'all eight episodes/24 nodes')
    expected=[(root,a) for root in (0,1) for a in ARMS]
    require([(x['source_slot'],x['mode']) for x in r['rows']]==expected
        and [x['episode_id'] for x in r['rows']]==[f's{root}_{a}' for root,a in expected],'fixed8 order no outcome/root selection')
    require([(x['episode_id'],x['boundary']) for x in r['nodes']]==[(f's{root}_{a}',b) for root,a in expected for b in BOUNDARIES],
        'all fixed24 nodes including unreached')
    require(all(n['status'] in ('sealed','refused','not_reached') for n in r['nodes']),
        'incomplete/failed node cannot inherit successful full-strict summary')
    require(len(p['tasks'])==2,'two known roots, not eight independent roots')
    paid=paid_counts(r,events);rows=[]
    for row in r['rows']:
        rows.append(row_summary(row,[n for n in r['nodes'] if n['episode_id']==row['episode_id']],events,
            p['tasks'][row['source_slot']],paid[row['episode_id']],load_array))
    for root in (0,1):
        goals=[x['native_goal'] for x in rows if x['source_slot']==root and x['native_goal'] is not None]
        require(all(raw(v)==raw(goals[0]) for v in goals),'same strict-authenticated native goal across paired strategies')
    lookup={(x['source_slot'],x['arm']):x for x in rows};pairs=[]
    for root in (0,1):
        for bank in ('endpoints','midpoints'):
            pairs.append(compare(lookup[root,'fixedH1_'+bank],lookup[root,'adaptive_'+bank],'same_bank_policy'))
        for policy in ('fixedH1','adaptive'):
            pairs.append(compare(lookup[root,policy+'_endpoints'],lookup[root,policy+'_midpoints'],'same_policy_bank'))
    return dict(schema=NAME+'.summary',status='known2_descriptive_NOT_admitted',rows=rows,pairs=pairs,
        fixed_denominator=8,node_denominator=24,panel_denominator=48,candidate_denominator=192,pair_denominator=8,
        known_roots=2,fresh_roots=0,independent_confirmation=False,independent_samples=None,
        P1_admission=False,P3_admission=False,C3=False,freshness=False,scientific_admission=False,
        equal_FLOPs=False,equal_wall=False,FLOPs=None,new_neural=0,new_physics=0,
        producer_arithmetic_imported=False,full_strict_rebuilt_here=False,
        statistics='descriptive fixed pairs only; no independent-root inference or CI',
        wall_semantics='nested inclusive operation wall is not total runtime; no actual wall equality requirement')


def load_npy(output,entry,bindings):
    path=absolute(entry['path']);base=absolute(str(output))
    require(path.is_relative_to(base) and 1<=len(path.relative_to(base).parts)<=2 and path.stat().st_size<=ARRAY_CAP,'original bounded in-output native NPY')
    hash_value(entry['file_sha256']);require(sha(path)==entry['file_sha256'],'original NPY file SHA')
    stream=io.BytesIO(path.read_bytes());a=np.load(stream,allow_pickle=False)
    require(stream.tell()==len(stream.getvalue()) and a.dtype==np.dtype('float32') and str(a.dtype)==entry['dtype']
        and list(a.shape)==entry['shape'] and np.isfinite(a).all()
        and hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==entry['array_sha256'],'entire float32 NPY layout/raw array SHA')
    require(str(path) not in bindings or bindings[str(path)]==entry['file_sha256'],'consistent original NPY binding')
    bindings[str(path)]=entry['file_sha256'];return a


def load_events(output,r,bindings):
    path=absolute(str(output))/'journal.jsonl'
    require(path.stat().st_size<=2*1024**3 and sha(path)==r['journal_sha256'],'whole saved journal SHA/cap')
    # Upstream full strict authenticates the exact native envelope/hashchain. This
    # independent consumer never reissues that certification or imports its code.
    events=[]
    allowed={'run_started','run_finished','operation_attempted','operation_completed','fresh_observation','guard_passed',
        'warmup_return','authorized_dispatch','control_return','handler_event','boundary_encoded','boundary_sealed',
        'episode_closed','invalid_render','initialization_intent'}
    previous=None
    with path.open('rb') as f:
        for i,line in enumerate(f):
            require(line.endswith(b'\n') and len(line)<=META,'complete bounded journal line')
            e=json.loads(line,object_pairs_hook=unique);raw(e)
            require(e['kind'] in allowed and type(e['index']) is int and e['index']==i and e['previous_sha256']==previous,'known complete ordered native envelope')
            hash_value(e['event_sha256']);previous=e['event_sha256'];events.append(e)
    require(events and events[0]['kind']=='run_started' and events[-1]['kind']=='run_finished'
        and previous==r['journal_chain_tail'],'terminal journal bookends')
    same(events[0]['contract'],r['contract'],'journal frozen contract')
    require(events[-1]['status']==r['status'] and events[-1]['nodes_sha256']==digest(r['nodes']),'terminal node binding')
    same(events[-1]['counts'],r['counts'],'terminal actual counts')
    bindings[str(path)]=r['journal_sha256'];return events


def analyze(prepared,prepare_sha,report,report_sha,strict,strict_sha):
    arguments=dict(prepared=prepared,prepare_sha=prepare_sha,report=report,report_sha=report_sha,strict=strict,strict_sha=strict_sha)
    p,r,s=(read(path,h) for path,h in ((prepared,prepare_sha),(report,report_sha),(strict,strict_sha)))
    require(r['prepared']==prepared and r['prepare_sha256']==prepare_sha,'actual own execution binding')
    require(s['status']=='saved_shared_panel_verified_NOT_admitted' and s['report_sha256']==report_sha
        and s['prepare_sha256']==prepare_sha,'external actual own full strict binds this report/prepare')
    same(s['per_episode_neural'],r['per_episode_neural'],'whole original strict model accounts')
    require(r['status']=='shared_panel_recorded' and r['model_before']==r['model_after']
        and r['model_before']['original_state_dict']==p['expected_model_hash'] and r['final_check']['model_unchanged'] is True,'actual clean frozen model')
    require(r['source_manifest_sha256']==digest(p['sources']),'original complete source closure')
    bindings={prepared:prepare_sha,report:report_sha,strict:strict_sha}
    source_path=ROOT/'experiments/spectral_reachability'/(PRODUCER+'.py')
    require(str(source_path) in p['sources'],'actual producer source must exist in prepared closure, no invented ABI authority')
    for path,h in p['sources'].items():
        absolute(path);hash_value(h);require(sha(path)==h,'sealed source/input/model bytes')
        require(path not in bindings or bindings[path]==h,'nonconflicting source binding');bindings[path]=h
    output=absolute(report).parent
    same(read(output/'execution_binding.json'),dict(prepared=prepared,prepare_sha256=prepare_sha),'actual saved execution binding')
    bindings[str(output/'execution_binding.json')]=sha(output/'execution_binding.json')
    events=load_events(output,r,bindings)
    result=summarize(p,r,s,events,lambda entry:load_npy(output,entry,bindings))
    for f in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):bindings[str(f)]=sha(f)
    result.update(arguments=arguments,bindings=bindings,whole_job_wall_seconds=r['wall_seconds'],
        engineering_caps=deepcopy(p['contract']),per_episode_full_path_cap={k:3*v for k,v in node_costs().items()},
        CUDA_peak_bytes=deepcopy(r['CUDA_peak_bytes']),authority=dict(external_actual_strict_sha256=strict_sha,full_strict_rebuilt_here=False))
    for path,h in bindings.items():require(sha(path)==h,'all saved inputs and sources unchanged after summary')
    return result


def write(path,obj):
    p=absolute(str(path));body=raw(obj)
    require(p.name.startswith(NAME+'_') and not p.exists() and len(body)<=128*1024**2,'new exclusive bounded own summary')
    with p.open('xb') as f:require(f.write(body)==len(body),'whole summary write');f.flush();os.fsync(f.fileno())
    return dict(path=str(p),sha256=sha(p),bytes=len(body))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('analyze','verify'))
    for k in ('prepared','prepare-sha','report','report-sha','strict','strict-sha','output','summary','summary-sha'):ap.add_argument('--'+k)
    a=ap.parse_args(argv)
    if a.command=='analyze':result=write(a.output,analyze(**{k:getattr(a,k) for k in ('prepared','prepare_sha','report','report_sha','strict','strict_sha')}))
    else:
        saved=read(a.summary,a.summary_sha);same(saved,analyze(**saved['arguments']),'complete deterministic saved arithmetic reconstruction')
        result=dict(verified=True,sha256=a.summary_sha,fixed_denominator=8,pair_denominator=8)
    print(raw(result).decode())


if __name__=='__main__':main()
