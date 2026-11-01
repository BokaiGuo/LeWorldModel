"""Independent saved numeric/paid-call recomputation. No producer imports or workers."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import numpy as np

NAME='agent_p3_commitment_numeric_recompute_20261029_v1'
PRODUCER='agent_p3_commitment_controls_20261029_v2'
ROOT=Path(__file__).resolve().parents[2]
MODES=('fixedH1','fixedH3','adaptive')
KEYS=tuple((s,m) for s in (0,1) for m in MODES)
AXES=('predict_API','predict_samples','predict_context_tokens','action_encoder_calls',
    'action_encoder_samples','action_encoder_context_tokens','predictor_calls','predictor_samples',
    'predictor_context_tokens','pred_proj_calls','pred_proj_rows','encoder_calls','encoder_samples',
    'projector_calls','projector_samples','jacobian_API','VJP_directions','grad_API',
    'raw_backward_callbacks','model_SVD_calls','array_SVD_calls','solve_calls',
    'signal_compile_calls','program_compile_calls','replay_calls','metric_calls','schedule_decisions')
PHYSICAL=('constructor','reset','set_state','goal_render','goal_setter','goal_bind',
    'public_step','physics_step','explicit_render','close','encoder','projector')
PHYSICAL_KINDS={'run_started','run_finished','operation_attempted','operation_completed',
    'initialization_intent','fresh_observation','guard_passed','warmup_return','authorized_dispatch',
    'control_return','handler_event','boundary_encoded','boundary_sealed','episode_closed','invalid_render'}
NATIVE_KINDS={'resource_attempt','resource_return','resource_failure','resource_blocked',
    'io_input','io_return','origin','packet','boundary','nominal','AD_terminal','jacobian',
    'direction','raw_backward','program_packet','program_boundary','local_signal',
    'agent_p3_commitment_controls_20261029_v1.node_final'}


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(value,*,compact=True):
    return json.dumps(value,sort_keys=True,allow_nan=False,
        **({'separators':(',',':')} if compact else {})).encode()


def digest(value,*,compact=True):return hashlib.sha256(canonical(value,compact=compact)).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def pairs_unique(items):
    out={}
    for k,v in items:
        require(k not in out,'duplicate JSON key');out[k]=v
    return out


def read(path,cap=96*1024**2,expected=None):
    path=Path(path).absolute()
    require(path.resolve()==path and path.is_file() and path.stat().st_size<=cap,
        'bounded canonical saved file')
    if expected is not None:require(sha(path)==expected,'external byte SHA')
    value=json.loads(path.read_bytes(),object_pairs_hook=pairs_unique);canonical(value)
    return value


def vector(value,n):
    a=np.asarray(value)
    require(a.shape==(n,) and a.dtype.kind in 'fi' and np.isfinite(a).all(),'finite numeric vector')
    return a.astype(np.float64)


def metrics(goal,state):
    """Independent env formula, including the original non-mod angle behavior."""
    g=vector(goal,7);s=vector(state,7);d=s-g
    xy=float(np.linalg.norm(d[:4]));angle=float(abs(d[4]));native=min(angle,2*np.pi-angle)
    wrapped=float(abs(np.arctan2(np.sin(d[4]),np.cos(d[4]))))
    original=dict(native_success=bool(xy<20 and native<np.pi/9),native_distance=float(np.linalg.norm(d)),
        native_angle_term=float(native),negative_native_angle=bool(native<0),
        wrapped_allxy_success=bool(xy<20 and wrapped<np.pi/9),wrapped_object_angle_error=wrapped,
        object_position_error=float(np.linalg.norm(d[2:4])),object_pose=s[2:5].tolist(),object_success=None)
    return dict(original=original,joint_xy=xy,agent_xy=float(np.linalg.norm(d[:2])),
        object_xy=original['object_position_error'],velocity_error=float(np.linalg.norm(d[5:7])),
        native_reward=-original['native_distance'])


def delta(op):
    """Reimplement the documented charge units, never trust a saved delta."""
    c=dict.fromkeys(AXES,0);r=op['resource'];shape=op['shape'];directions=op['directions']
    scalars=dict(jacobian='jacobian_API',grad='grad_API',raw_backward='raw_backward_callbacks',
        model_svd='model_SVD_calls',array_svd='array_SVD_calls',solve='solve_calls',
        signal_compile='signal_compile_calls',program_compile='program_compile_calls',
        replay='replay_calls',metric='metric_calls',schedule='schedule_decisions')
    if r in scalars:
        require(shape is None and directions is None,'scalar call units');c[scalars[r]]=1
    elif r=='vjp':
        require(shape is None and type(directions) is int and directions>0,'VJP units');c['VJP_directions']=directions
    else:
        require(type(shape) is list and all(type(n) is int and n>0 for n in shape) and directions is None,
            'native actual shape units')
        if r in ('predict','action_encoder','predictor'):
            require(len(shape)==3 and shape[1]==3,'full three-context predictor shape')
            prefix='predict' if r=='predict' else r
            c[prefix+'_API' if r=='predict' else prefix+'_calls']=1
            c[prefix+'_samples']=shape[0];c[prefix+'_context_tokens']=shape[0]*shape[1]
        elif r=='pred_proj':
            require(len(shape)==2,'native projection rows');c.update(pred_proj_calls=1,pred_proj_rows=shape[0])
        elif r in ('encoder','projector'):
            require(len(shape)>=2,'visual samples');c[r+'_calls']=1;c[r+'_samples']=shape[0]
        else:raise ValueError('unknown charged native resource')
    require(op['delta']==c,'saved delta agrees with independent units');return c


def journal(path,*,native=False):
    path=Path(path);require(path.resolve()==path and path.stat().st_size<=768*1024**2,'bounded journal')
    payload=path.read_bytes();require(payload.endswith(b'\n'),'complete terminal line')
    previous='0'*64 if native else None;events=[]
    for i,line in enumerate(payload.splitlines()):
        require(line and len(line)<=8*1024**2,'strict single LF, no empty record filtering')
        e=json.loads(line,object_pairs_hook=pairs_unique);canonical(e)
        require(line==canonical(e,compact=native),'exact native compact vs physical spaced line ABI')
        require(type(e['index']) is int and e['index']==i,'consecutive chain index')
        if native:
            require(set(e)=={'index','previous','payload','sha256'} and e['previous']==previous and
                e['sha256']==digest({k:v for k,v in e.items() if k!='sha256'}),'native compact hashchain')
            previous=e['sha256'];require(e['payload']['kind'] in NATIVE_KINDS,'unknown native event')
            events.append(e['payload'])
        else:
            require(e['kind'] in PHYSICAL_KINDS,'unknown physical event cannot hide paid work')
            require(e['previous_sha256']==previous and e['event_sha256']==digest(
                {k:v for k,v in e.items() if k!='event_sha256'},compact=False),'physical spaced hashchain')
            previous=e['event_sha256'];events.append(e)
    return events,previous


def physical_counts(events,report):
    total={r:dict(attempted=0,completed=0) for r in PHYSICAL}
    by={f's{s}_{m}':{r:dict(attempted=0,completed=0) for r in PHYSICAL} for s,m in KEYS}
    stack=[];pending={};serial=-1
    for e in events:
        if e['kind'] not in ('operation_attempted','operation_completed'):continue
        r=e['resource'];eid=e['episode_id'];i=e['operation_id']
        require(r in total and eid in by and type(e['units']) is int and e['units']==1 and type(i) is int,
            'one native physical/visual call, known episode')
        op={k:e[k] for k in ('operation_id','episode_id','resource','units','region')}
        if e['kind']=='operation_attempted':
            require(i==serial+1,'physical unique consecutive attempts');serial=i
            pending[i]=op;stack.append(i);axis='attempted'
        else:
            require(stack and stack.pop()==i and pending.pop(i,None)==op,'physical nested matching return')
            axis='completed'
        total[r][axis]+=1;by[eid][r][axis]+=1
    require(total==report['counts'] and {str(k):v for k,v in pending.items()}==
        {str(k):v for k,v in report['pending'].items()},'independent physical counts/pending')
    return dict(total=total,per_episode=by,pending={str(k):v for k,v in pending.items()})


def paid_native(events,receipt):
    attempted=dict.fromkeys(AXES,0);completed=dict.fromkeys(AXES,0)
    signal=dict.fromkeys(AXES,0);stack=[];ops={};ended={};blocked={}
    for e in events:
        kind=e['kind']
        if not kind.startswith('resource_'):continue
        op=e['entry'];i=op['index'];d=delta(op)
        require(type(i) is int and i>=0,'native operation id')
        if kind=='resource_blocked':
            require(i not in ops and i not in blocked and op['status']=='blocked_before_dispatch',
                'blocked call distinct from dispatched attempt');blocked[i]=op;continue
        if kind=='resource_attempt':
            require(i not in ops and op['status']=='attempted' and op['ended_ns'] is None and op['error'] is None,
                'durable actual attempt before return')
            ops[i]=op;stack.append(i)
            for k in AXES:attempted[k]+=d[k]
        else:
            require(kind in ('resource_return','resource_failure') and stack and stack.pop()==i and i in ops,
                'native nested lifecycle')
            require({**op,'status':'attempted','ended_ns':None,'error':None}==ops[i] and
                op['status']==('returned' if kind=='resource_return' else 'failed') and
                type(op['ended_ns']) is int and type(op['started_ns']) is int and op['ended_ns']>=op['started_ns'],
                'native same attempt disposition and measured duration')
            ended[i]=op
            if kind=='resource_return':
                for k in AXES:
                    completed[k]+=d[k]
                    if op['scope'].get('phase') not in ('program','selected') and 'index' not in op['scope']:
                        signal[k]+=d[k]
    require(not stack and sorted(set(ops)|set(blocked))==list(range(len(ops)+len(blocked))),
        'all complete/failed native dispositions; unclosed journals not complete numerical evidence')
    actual=[ended.get(i,blocked.get(i)) for i in range(len(ops)+len(blocked))]
    require(None not in actual and actual==receipt['events'] and attempted==receipt['attempted'] and
        completed==receipt['completed'],'journal rather than aggregate receipt drives native counts')
    for k in AXES:
        require(type(receipt['caps'][k]) is int and 0<=completed[k]<=attempted[k]<=receipt['caps'][k],
            'paid native cap; no failure refunds')
    return dict(attempted=attempted,completed=completed,signal_completed=signal,
        program_completed={k:completed[k]-signal[k] for k in AXES})


def branch(events,row):
    eid=row['episode_id'];local=[e for e in events if e.get('episode_id')==eid]
    observations=[e['receipt'] for e in local if e['kind']=='fresh_observation' and e['receipt']['phase']=='plan']
    returned=[e for e in local if e['kind']=='control_return'];dispatch=[e for e in local if e['kind']=='authorized_dispatch']
    require(len(returned)==len(dispatch)==row['control_returned']==len(row['executed_raw']),
        'all returned control/action denominator')
    result=dict(episode_id=eid,source_slot=row['source_slot'],mode=row['mode'],status=row['status'],
        actual_control_returned=row['control_returned'],full20=row['full_horizon'],
        terminal=None,prefix5=None,goal=None,native_success=None,metrics=None)
    if not observations:
        require(not returned,'no returned control without real observation');return result
    require([o['step'] for o in observations]==list(range(len(returned)+1)), 'all returned native states')
    goal=observations[0]['goal'];vector(goal,7);result['goal']=goal
    for i,(ret,dis,ob,action) in enumerate(zip(returned,dispatch,observations[1:],row['executed_raw'],strict=True),1):
        require(ret['step']==dis['step']==ob['step']==i and dis['index']<ob['observed_event_index']<ret['index'],
            'dispatch before actual observation and native return')
        require(np.asarray(ret['command'],np.float32).tobytes()==np.asarray(dis['command'],np.float32).tobytes()==
            np.asarray(action,np.float32).tobytes() and ob['goal']==goal,'exact own rawbits and unchanged task goal')
        m=metrics(goal,ob['state'])
        require(type(ret['terminated']) is bool and type(ret['truncated']) is bool and
            ret['reward']==m['native_reward'] and ret['terminated']==m['original']['native_success'],
            'native7 reward and original native termination independently recomputed')
        if i<len(returned):require(not ret['terminated'] and not ret['truncated'],'no control after native stop')
    prefix=next((o for o in observations if o['step']==5),None)
    if prefix is not None:
        result['prefix5']={k:prefix[k] for k in ('state','snapshot','goal','goal_rgb_sha256','goal_pose','pixel_sha256')}
    if returned:
        final=observations[-1];m=metrics(goal,final['state'])
        require(row['final_state']==final['state'] and row['last_observation']==final and
            row['final_metrics']==m['original'] and row['native_success']==m['original']['native_success'] and
            row['terminated']==returned[-1]['terminated'] and row['truncated']==returned[-1]['truncated'],
            'report final state/metrics/actual native return')
        result.update(terminal=final['state'],metrics=m,native_success=m['original']['native_success'])
        require(type(row['full_horizon']) is bool and
            row['full_horizon']==(len(returned)==20 and row['status']=='completed20'),
            'actual full20, not a reported intent')
        if len(returned)<20 and row['status']=='native_stopped_control':
            require(returned[-1]['terminated'] or returned[-1]['truncated'],'no false clean native stop label')
    return result


def comparisons(rows):
    out=[]
    for s in (0,1):
        for left,right in itertools.combinations(MODES,2):
            a=next(r for r in rows if (r['source_slot'],r['mode'])==(s,left))
            b=next(r for r in rows if (r['source_slot'],r['mode'])==(s,right))
            same=a['prefix5'] is not None and b['prefix5'] is not None
            if same:require(a['prefix5']==b['prefix5'] and a['goal']==b['goal'],'same-root full native prefix pairing')
            available=same and a['metrics'] is not None and b['metrics'] is not None
            out.append(dict(source_slot=s,left=left,right=right,paired_prefix_available=same,
                equal_actual_control=(a['actual_control_returned']==b['actual_control_returned']) if
                    a['actual_control_returned'] is not None and b['actual_control_returned'] is not None else None,
                delta_right_minus_left={k:b['metrics'][k]-a['metrics'][k] for k in
                    ('joint_xy','agent_xy','object_xy','velocity_error','native_reward')} if available else None,
                native_distance_delta=(b['metrics']['original']['native_distance']-
                    a['metrics']['original']['native_distance']) if available else None,
                success_pair=[a['native_success'],b['native_success']] if available else [None,None],
                causal_single_factor_claim=False,budget_match='notmatched'))
    return out


def unknown_rows(report):
    return [dict(episode_id=x['episode_id'],source_slot=x['source_slot'],mode=x['mode'],status=x['status'],
        actual_control_returned=None,metrics=None,native_success=None,prefix5=None,goal=None)
        for x in report['rows']]


def analyze(output,report_sha,prepared,prepare_sha,strict_path,strict_sha):
    output=Path(output).absolute();report_path=output/'report.json'
    p=read(prepared,16*1024**2,prepare_sha);r=read(report_path,expected=report_sha)
    strict=read(strict_path,expected=strict_sha)
    require(p['schema']==PRODUCER+'.prepare' and r['schema']==PRODUCER+'.report' and
        r['prepared']==str(Path(prepared).absolute()) and r['prepare_sha256']==prepare_sha and
        r['contract']==p['contract'],'actual v2 report/prepare, no v1 relabeling')
    require(strict['schema']==PRODUCER+'.consume' and all(strict[k] is False for k in
        ('P1_admission','P3_admission','freshness','scientific_admission')),'actual strict schema/gates')
    require([(x['source_slot'],x['mode']) for x in r['rows']]==list(KEYS) and
        [(x['episode_id'],x['boundary']) for x in r['nodes']]==
        [(f's{s}_{m}',b) for s,m in KEYS for b in (5,10,15)] and
        r['fixed_denominator']==6 and r['node_denominator']==18,'fixed complete six/eighteen denominator')
    require(all(r[k] is False for k in ('P1_admission','P3_admission','freshness','scientific_admission')),
        'no scientific admission')
    base=dict(schema=NAME+'.analysis',report_sha256=report_sha,prepare_sha256=prepare_sha,
        strict_receipt_sha256=strict_sha,fixed_denominator=6,node_denominator=18,
        budget_match='notmatched',equal_FLOPs=False,known_roots=2,independent_confirmation=False,freshness=False,
        P1_admission=False,P3_admission=False,scientific_admission=False,
        scope='numeric/native-paid-counter recomputation; not full RGB/model strict replay',
        inherited_strict_direct_report_SHA_field=False,independent_neural_replay=False)
    if 'journal_sha256' not in r:
        require(strict['complete'] is False,'startup failure never complete')
        unknown=unknown_rows(r)
        return dict(base,status='startup_partial_preserved',complete=False,rows=unknown,pairs=comparisons(unknown),paid=None)
    require(sha(output/'journal.jsonl')==r['journal_sha256'],'actual physical journal byte binding')
    events,tail=journal(output/'journal.jsonl');require(tail==r['journal_chain_tail'],'physical terminal tail')
    require(events[0]['kind']=='run_started' and events[0]['contract']==p['contract'] and
        events[-1]['kind']=='run_finished' and events[-1]['status']==r['status'] and
        events[-1]['nodes_sha256']==digest(r['nodes']) and events[-1]['counts']==r['counts'],
        'all actual terminal bookends')
    physical=physical_counts(events,r);native={};node_rows=[]
    for s,m in KEYS:
        eid=f's{s}_{m}';own=[n for n in r['nodes'] if n['episode_id']==eid]
        acc={k:dict.fromkeys(AXES,0) for k in ('attempted','completed','signal_completed','program_completed')}
        receipts=0
        for n in own:
            if 'native_journal_sha256' not in n:continue
            path=output/(eid+f"_b{n['boundary']:02}")/'events.jsonl'
            require(sha(path)==n['native_journal_sha256'],'actual native journal SHA')
            es,ntail=journal(path,native=True);require(ntail==n['native_journal_tail'],'native actual tail')
            c=paid_native(es,n['resource_receipt']);receipts+=1
            for field in acc:
                for k in AXES:acc[field][k]+=c[field][k]
            node_rows.append(dict(episode_id=eid,boundary=n['boundary'],status=n['status'],**c))
        saved=next(x for x in r['per_episode_neural'] if (x['source_slot'],x['mode'])==(s,m))
        require(all(saved[k]==acc[k] for k in acc) and saved['receipts']==receipts,
            'actual native journal counts vs report per-arm totals')
        native[eid]=dict(acc,receipts=receipts,FLOPs=None,CUDA_ns=None)
    clean=r['status']=='known_fullcommit_recorded' and not r['errors'] and not r['pending']
    if not clean:
        require(strict['complete'] is False,'partial producer/strict disagreement')
        unknown=unknown_rows(r)
        return dict(base,status='partial_paid_prefix_preserved',complete=False,rows=unknown,pairs=comparisons(unknown),
            paid=dict(physical=physical,native=native,nodes=node_rows))
    require(strict['complete'] is True and strict['counts']==physical['total'] and
        strict['fixed_denominator']==6 and strict['node_denominator']==18 and
        strict['per_episode_neural']==r['per_episode_neural'],'supplied strict receipt numeric cross-binding')
    require(r['budget_match']=='notmatched' and r['equal_FLOPs'] is False and r['equal_wall'] is False,
        'no claimed actual budget matching')
    rows=[branch(events,x) for x in r['rows']]
    require(strict['native_metrics']==[x['final_metrics'] for x in r['rows']], 'strict terminal metric array')
    for row in r['rows']:
        c=physical['per_episode'][row['episode_id']]
        require(c['public_step']['completed']==row['warmup_returned']+row['control_returned'] and
            c['physics_step']['completed']==2*c['reset']['completed']+10*c['public_step']['completed'] and
            c['constructor']==c['close']==dict(attempted=1,completed=1) and
            c['reset']==dict(attempted=2,completed=2) and c['set_state']==dict(attempted=4,completed=4),
            'independent per-arm initialization and public/substep arithmetic')
    require(sha(report_path)==report_sha and sha(prepared)==prepare_sha and sha(strict_path)==strict_sha,
        'authority bytes unchanged through numerical recomputation')
    return dict(base,status='independent_numeric_recomputed_NOT_admitted',complete=True,rows=rows,
        pairs=comparisons(rows),paid=dict(physical=physical,native=native,nodes=node_rows),
        consumer_source_sha256=sha(__file__))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('output','report-sha','prepared','prepare-sha','strict-receipt','strict-sha','receipt'):
        parser.add_argument('--'+key,required=True)
    a=parser.parse_args(argv);result=analyze(a.output,a.report_sha,a.prepared,a.prepare_sha,a.strict_receipt,a.strict_sha)
    target=Path(a.receipt).absolute()
    require(target.parent==ROOT/'results' and target.name.startswith(NAME) and target.resolve()==target and
        not target.exists(),'exclusive independent new numeric receipt')
    with target.open('xb') as f:
        b=canonical(result)+b'\n';require(f.write(b)==len(b),'complete receipt write');f.flush();os.fsync(f.fileno())
    fd=os.open(target.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    print(canonical(dict(receipt=str(target),sha256=sha(target),status=result['status'],complete=result['complete'])).decode())


if __name__=='__main__':main()
