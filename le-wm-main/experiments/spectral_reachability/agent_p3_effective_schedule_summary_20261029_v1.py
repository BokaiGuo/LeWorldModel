"""Saved-only eight-arm descriptive contrasts; MAIN runs full strict reconstruction."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p3_effective_schedule_20261029_v1 as owner

NAME='agent_p3_effective_schedule_summary_20261029_v1'
ROOT=owner.ROOT
CAP=128*1024**2
PINS=('7cef2600e9844d981f97fdd6c385bff0c3ae9bdf55713ac8c92cebd28b003c23',
      '3b20ae995f050038c5eab04e6c0e860fb3162f7ca085c2b9ae68c142df327afa',
      'bb17b92a5642a01ea3a6eb7c5659616fe39f096f6c25f09741794c1f41d1be21')
CP=(0,5,10,15,20)
ERRORS=('joint_xy','agent_xy','object_xy','angle_wrapped','native_angle_term',
        'native7_distance','native7_reward','velocity_error')


def require(value,message):
    if not value:raise ValueError(message)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def hash_value(value):
    require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'literal SHA256')


def unique(items):
    out={}
    for k,v in items:
        require(k not in out,'duplicate JSON key');out[k]=v
    return out


def read(path,expected=None):
    path=Path(path).absolute()
    require(path.resolve()==path and path.is_file() and path.stat().st_size<=CAP,'bounded canonical regular saved file')
    if expected is not None:
        hash_value(expected);require(sha(path)==expected,'external saved SHA')
    value=json.loads(path.read_bytes(),object_pairs_hook=unique);canonical(value);return value


def vector(value,size):
    require(type(value) is list and len(value)==size and
        all(type(v) in (int,float) for v in value),'literal vector, no bool')
    out=np.asarray(value,np.float64);require(np.isfinite(out).all(),'finite vector');return out


def raw_bits(value):
    require(type(value) is list,'literal raw actions')
    a=np.asarray([vector(row,2) for row in value],np.float64).reshape(-1,2)
    b=a.astype(np.float32)
    require(np.isfinite(b).all() and np.all(abs(b)<=1) and np.array_equal(a,b.astype(np.float64)),
        'lossless float32 raw box; no clipping or repair')
    return b.tobytes()


def physical_metric(goal,state):
    g=vector(goal,7);s=vector(state,7);d=s-g
    # Preserve source's native min-angle expression, including negative values
    # for >2pi. Wrapped angle is a separate descriptive axis, never a new gate.
    a=float(abs(d[4]));native=min(a,2*np.pi-a)
    return dict(joint_xy=float(np.linalg.norm(d[:4])),agent_xy=float(np.linalg.norm(d[:2])),
        object_xy=float(np.linalg.norm(d[2:4])),angle_wrapped=float(abs(np.arctan2(np.sin(d[4]),np.cos(d[4])))),
        native_angle_term=float(native),native7_distance=float(np.linalg.norm(d)),
        native7_reward=-float(np.linalg.norm(d)),velocity_error=float(np.linalg.norm(d[5:])),
        native7_delta=d.tolist(),native_success=bool(np.linalg.norm(d[:4])<20 and native<np.pi/9))


def latent_q(goal,z):
    return float(np.mean((vector(z,192)-vector(goal,192))**2))


def paid_counts(report,events):
    """Independent actual sums of native event units and ALL model ledger axes."""
    ids=[r['episode_id'] for r in report['rows']];axes=tuple(owner.core.AXES)
    native_axes=tuple(owner.contract()['physical_caps']|owner.contract()['visual_caps'])
    per={eid:dict(neural={k:dict(attempted=0,completed=0) for k in axes},
        native={k:dict(attempted=0,completed=0) for k in native_axes},inclusive_operation_wall_ns={}) for eid in ids}
    pending={};serial=-1
    for event in events:
        kind=event['kind']
        if kind not in ('operation_attempted','operation_completed'):continue
        eid=event['episode_id'];k=event['resource'];i=event['operation_id'];units=event['units']
        require(eid in per and k in native_axes and type(units) is int and units>0 and type(i) is int,'native actual units')
        binding=(eid,k,units,event['region'])
        if kind=='operation_attempted':
            require(i>serial and i not in pending,'unique native intent');serial=i;pending[i]=binding;axis='attempted'
        else:require(pending.pop(i,None)==binding,'matching actual native return');axis='completed'
        per[eid]['native'][k][axis]+=units
    require(not pending,'summary requires complete strict, not pending attempts')
    for node in report['nodes']:
        rr=node.get('resource_receipt')
        if rr is None:continue
        acc={k:dict(attempted=0,completed=0) for k in axes};wall={}
        for op in rr['events']:
            require(set(op['delta'])==set(axes) and op['status'] in ('returned','failed','blocked_before_dispatch'),
                'all model axes/attempt dispositions')
            if op['status']=='blocked_before_dispatch':continue
            for k,v in op['delta'].items():
                require(type(v) is int and v>=0,'literal resource units')
                acc[k]['attempted']+=v
                if op['status']=='returned':acc[k]['completed']+=v
            require(type(op['started_ns']) is type(op['ended_ns']) is int and op['ended_ns']>=op['started_ns'],'measured inclusive wall')
            k=op['resource'];wall[k]=wall.get(k,0)+op['ended_ns']-op['started_ns']
        require(all(acc[k][a]==rr[a][k] for k in axes for a in ('attempted','completed')) and
            wall==rr['inclusive_operation_wall_ns'],'independent model totals and inclusive operation wall')
        own=per[node['episode_id']]
        for k in axes:
            for a in ('attempted','completed'):own['neural'][k][a]+=acc[k][a]
        for k,v in wall.items():own['inclusive_operation_wall_ns'][k]=own['inclusive_operation_wall_ns'].get(k,0)+v
    total={k:{a:sum(per[eid]['native'][k][a] for eid in ids) for a in ('attempted','completed')} for k in native_axes}
    require(total==report['counts'],'independent whole native/visual census')
    for old in report['per_episode_neural']:
        own=per[f"s{old['source_slot']}_{old['mode']}"]
        require(all(own['neural'][k][a]==old[a][k] for k in axes for a in ('attempted','completed')) and
            own['inclusive_operation_wall_ns']==old['inclusive_operation_wall_ns'],'reported versus independently summed model axes')
    return per


def row_summary(row,nodes,events,base,counts,load_latent):
    eid=row['episode_id'];goal=base['goal']
    require(np.array_equal(np.asarray(base['projection']),np.eye(192)),'original identity192 Q only')
    observations=[e['receipt'] for e in events if e.get('episode_id')==eid and
        e['kind']=='fresh_observation' and e['receipt']['phase']=='plan']
    require([o['step'] for o in observations]==list(range(row['control_returned']+1)) if row['initial_exact'] else not observations,
        'every actual same-time observation; no padding')
    dispatch={e['step']:e for e in events if e.get('episode_id')==eid and e['kind']=='authorized_dispatch'}
    returned={e['step']:e for e in events if e.get('episode_id')==eid and e['kind']=='control_return'}
    require(len(returned)==len(dispatch)==len(row['executed_raw'])==row['control_returned'],'all actual raw returns')
    steps=[None]*21
    for ob in observations:
        b=ob['step'];action=None
        if b:
            action=row['executed_raw'][b-1]
            require(raw_bits([dispatch[b]['command']])==raw_bits([returned[b]['command']])==raw_bits([action]),'authorized/returned rawbits')
        steps[b]=dict(step=b,state=ob['state'],goal=ob['goal'],metrics=physical_metric(ob['goal'],ob['state']),
            rgb_pixel_sha256=ob['pixel_sha256'],png_file_sha256=ob['file_sha256'],actual_Q=None,
            actual_latent=None,Q_source=None,returned_raw=action)
    if steps[0] is not None:
        z=base['root_context']['latent_history'][-1]
        steps[0].update(actual_Q=latent_q(goal,z),actual_latent=deepcopy(z),Q_source='sealed_original_root_encoding_exact_replayed_RGB_not_new_encoder')
    for enc in row['encodings']:
        b=enc['step'];require(steps[b] is not None and steps[b]['Q_source'] is None,'encoding only at actual unique control step')
        z=load_latent(enc['latent'])
        require(z.dtype==np.float32 and z.shape==(192,) and np.isfinite(z).all(),'actual saved encoder192')
        steps[b].update(actual_Q=latent_q(goal,z.tolist()),actual_latent=z.tolist(),Q_source='this_run_actual_RGB_encoder')
    node_rows=[]
    for node in nodes:
        item=dict(boundary=node['boundary'],status=node['status'],decision=None,context_sha256=None,
            actual_context_Q=None,bank=None,selected_index=None,predicted_endpoint_step=None,
            selected_predicted_Q=None,actual_Q_at_predicted_endpoint=None,prediction_Q_error=None,
            resource_receipt=deepcopy(node.get('resource_receipt')))
        r=node.get('receipt')
        if r is not None:
            req=node['request'];item.update(context_sha256=digest(req['task']['root_context']),
                actual_context_Q=latent_q(goal,req['task']['root_context']['latent_history'][-1]),decision=r['decision'])
            require(steps[node['boundary']]['actual_Q']==item['actual_context_Q'],'actual RGB context Q independently recomputed')
            if r['status']=='sealed':
                h=r['decision']['horizon'];bank=deepcopy(r['bank']);costs=[]
                require(len(r['candidates'])==len(bank['actions'])==4,'fixed full candidate support')
                for i,candidate in enumerate(r['candidates']):
                    raw_bits(candidate['actions'])
                    q=latent_q(goal,candidate['whole'][-1]['latent_history'][-1]);require(q==candidate['cost'],'independent original terminal Q')
                    costs.append(dict(index=i,fraction=bank['fractions'][i],predicted_Q=q,raw=candidate['actions']))
                k=r['winner']['index'];endpoint=node['boundary']+h*5
                actual=None if steps[endpoint] is None else steps[endpoint]['actual_Q']
                bank['all_candidate_costs']=costs
                item.update(bank=bank,selected_index=k,predicted_endpoint_step=endpoint,selected_predicted_Q=costs[k]['predicted_Q'],
                    actual_Q_at_predicted_endpoint=actual,prediction_Q_error=None if actual is None else costs[k]['predicted_Q']-actual)
        node_rows.append(item)
    final=next((s for s in reversed(steps) if s is not None),None)
    return dict(episode_id=eid,source_slot=row['source_slot'],arm=row['mode'],status=row['status'],
        warmup_returned=row['warmup_returned'],control_returned=row['control_returned'],
        terminated=row['terminated'],truncated=row['truncated'],full_horizon=row['full_horizon'],
        steps=steps,checkpoints=[dict(step=b,value=steps[b]) for b in CP],nodes=node_rows,
        eventtime_terminal=final,counts=counts,warmup_excluded_from_task_metric=True,
        native_success=None if final is None or final['step']==0 else final['metrics']['native_success'],
        missing_steps=[i for i,v in enumerate(steps) if v is None],FLOPs=None,wall_ns=None)


def packet_support(row,boundary):
    active=[n for n in row['nodes'] if n['bank'] is not None and
        n['boundary']<=boundary<n['predicted_endpoint_step']]
    if not active:return None
    require(len(active)==1,'one original candidate bank for actual active commitment')
    node=active[0];offset=boundary-node['boundary']
    return dict(origin_boundary=node['boundary'],context_sha256=node['context_sha256'],
        horizon=node['decision']['horizon'],selected_index=node['selected_index'],
        fresh_at_boundary=node['boundary']==boundary,
        raw5=[a[offset:offset+5] for a in node['bank']['actions']],
        fractions=node['bank']['fractions'])


def compare(left,right,kind):
    require(left['source_slot']==right['source_slot'],'same known root')
    pairs=[]
    for b,(l,r) in enumerate(zip(left['steps'],right['steps'],strict=True)):
        common=l is not None and r is not None
        pairs.append(dict(step=b,left_missing=l is None,right_missing=r is None,
            right_minus_left=None if not common else {k:r['metrics'][k]-l['metrics'][k] for k in ERRORS},
            state_exact_equal=None if not common else np.asarray(l['state'],np.float64).tobytes()==np.asarray(r['state'],np.float64).tobytes(),
            RGB_pixels_equal=None if not common else l['rgb_pixel_sha256']==r['rgb_pixel_sha256'],
            PNG_bytes_equal=None if not common else l['png_file_sha256']==r['png_file_sha256'],
            actual_Q_delta=None if not common or l['actual_Q'] is None or r['actual_Q'] is None else r['actual_Q']-l['actual_Q'],
            actual_latent_bits_equal=None if not common or l['actual_latent'] is None or r['actual_latent'] is None else
                np.asarray(l['actual_latent'],np.float32).tobytes()==np.asarray(r['actual_latent'],np.float32).tobytes(),
            returned_raw_bits_equal=None if not common or b==0 else raw_bits([l['returned_raw']])==raw_bits([r['returned_raw']])))
    supports=[]
    for b in (5,10,15):
        # A future commitment buffer is not an actually reached feedback node.
        l=None if left['steps'][b] is None or not any(n['boundary']==b and n['status'] in ('sealed','observed_commitment') for n in left['nodes']) else packet_support(left,b)
        r=None if right['steps'][b] is None or not any(n['boundary']==b and n['status'] in ('sealed','observed_commitment') for n in right['nodes']) else packet_support(right,b)
        common=l is not None and r is not None
        overlap=None if not common else [dict(left_index=i,right_index=j) for i,a in enumerate(l['raw5'])
            for j,c in enumerate(r['raw5']) if raw_bits(a)==raw_bits(c)]
        supports.append(dict(boundary=b,left=l,right=r,packet_raw_support_overlap=overlap,
            compile_context_equal=None if not common else l['context_sha256']==r['context_sha256'],
            same_compile_horizon=None if not common else l['horizon']==r['horizon'],
            note='retained origin-bank slices are NOT newly evaluated candidates at actual intermediate RGB'))
    lt,rt=left['eventtime_terminal'],right['eventtime_terminal']
    same_final=lt is not None and rt is not None and lt['step']==rt['step']
    return dict(source_slot=left['source_slot'],contrast=kind,left=left['episode_id'],right=right['episode_id'],
        interpretation='same-bank scheduling contrast' if kind=='same_bank_schedule' else 'same-schedule candidate-bank intervention; NOT pure scheduling causality',
        steps=pairs,checkpoints=[pairs[b] for b in CP],support=supports,
        common_steps=[x['step'] for x in pairs if not x['left_missing'] and not x['right_missing']],
        eventtime_terminal=dict(left_step=None if lt is None else lt['step'],right_step=None if rt is None else rt['step'],
            same_time=same_final,right_minus_left=None if not same_final else {k:rt['metrics'][k]-lt['metrics'][k] for k in ERRORS}),
        resources={group:{k:{a:right['counts'][group][k][a]-left['counts'][group][k][a]
            for a in ('attempted','completed')} for k in left['counts'][group]} for group in ('neural','native')},
        budget_match='notmatched',equal_FLOPs=False,FLOPs=None)


def summarize(prepared,report,events,load_latent):
    expected=[(s,a) for s in (0,1) for a in owner.ARMS]
    require([(r['source_slot'],r['mode']) for r in report['rows']]==expected and
        [(n['episode_id'],n['boundary']) for n in report['nodes']]==[(f's{s}_{a}',b) for s,a in expected for b in (5,10,15)],
        'all eight fixed rows and24 fixed nodes, no filtering')
    paid=paid_counts(report,events);rows=[]
    for row in report['rows']:
        rows.append(row_summary(row,[n for n in report['nodes'] if n['episode_id']==row['episode_id']],
            events,prepared['tasks'][row['source_slot']],paid[row['episode_id']],load_latent))
    lookup={(r['source_slot'],r['arm']):r for r in rows};pairs=[]
    for s in (0,1):
        for bank in owner.BANKS:
            pairs.append(compare(lookup[s,'commit15_'+bank],lookup[s,'feedback5_'+bank],'same_bank_schedule'))
        for schedule in owner.SCHEDULES:
            pairs.append(compare(lookup[s,schedule+'_endpoints'],lookup[s,schedule+'_midpoints'],'same_schedule_bank'))
    return dict(rows=rows,pairs=pairs,fixed_denominator=8,node_denominator=24,pair_denominator=8,
        known_roots=2,independent_samples=None,independent_confirmation=False,P1_admission=False,P3_admission=False,C3=False,
        freshness=False,scientific_admission=False,budget_match='notmatched',equal_FLOPs=False,
        statistics='descriptive fixed pairs only; no p-values/CI or independent-eight-root inference',
        wall_semantics='inclusive nested model-operation wall is not summed into total runtime or FLOPs')


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def analyze(output,report_sha,prepared,prepare_sha,strict,strict_sha):
    output=Path(output).absolute();prepared=Path(prepared).absolute();strict=Path(strict).absolute()
    for path,h in zip(owner.own_files(),PINS,strict=True):require(sha(path)==h,'frozen owner source/tests/doc')
    report=read(output/'report.json',report_sha);p=read(prepared,prepare_sha);receipt=read(strict,strict_sha)
    bindings={str(path):h for path,h in ((output/'report.json',report_sha),(prepared,prepare_sha),(strict,strict_sha))}
    for path,h in p['sources'].items():
        require(sha(path)==h,'all upstream source/model/config bindings before rebuild');bindings[path]=h
    for path in own_files():bindings[str(path)]=sha(path)
    rebuilt=owner.consume(output,report_sha,prepared,prepare_sha)
    require(canonical(rebuilt)==canonical(receipt) and rebuilt['complete'] is True and
        rebuilt['fixed_denominator']==8 and rebuilt['node_denominator']==24,'complete OWN strict rebuilt, not status-only authority')
    events=owner.load_events(output,report)
    result=summarize(p,report,events,lambda entry:owner.load_array(output,entry))
    result.update(schema=NAME+'.summary',status='known2_descriptive_NOT_admitted',
        inputs=dict(output=str(output),report_sha256=report_sha,prepared=str(prepared),prepare_sha256=prepare_sha,
            strict=str(strict),strict_sha256=strict_sha),bindings=bindings,
        authority=dict(full_own_strict_rebuilt_here=True,independent_neural_replay=False),
        whole_job_wall_seconds=report['wall_seconds'])
    for path,h in bindings.items():require(sha(path)==h,'all sealed inputs/sources unchanged after analysis')
    return result


def write(path,value):
    path=Path(path).absolute()
    require(path.parent==ROOT/'results' and path.name.startswith(NAME) and path.resolve()==path and not path.exists(),'new own summary file')
    data=canonical(value);require(len(data)<=CAP,'128MiB summary cap')
    with path.open('xb') as f:
        require(f.write(data)==len(data),'whole saved summary');f.flush();os.fsync(f.fileno())
    owner.signal.fsync_dir(path.parent);return dict(path=str(path),sha256=sha(path),status=value['status'])


def verify(path,expected):
    value=read(path,expected);i=value['inputs']
    rebuilt=analyze(i['output'],i['report_sha256'],i['prepared'],i['prepare_sha256'],i['strict'],i['strict_sha256'])
    require(canonical(value)==canonical(rebuilt) and sha(path)==expected,'complete deterministic saved summary reconstruction')
    return dict(verified=True,status=value['status'],sha256=expected)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('analyze','verify'))
    for k in ('output','report-sha','prepared','prepare-sha','strict','strict-sha','receipt','summary','summary-sha'):
        parser.add_argument('--'+k)
    a=parser.parse_args(argv)
    if a.command=='verify':result=verify(a.summary,a.summary_sha)
    else:result=write(a.receipt,analyze(a.output,a.report_sha,a.prepared,a.prepare_sha,a.strict,a.strict_sha))
    print(canonical(result).decode())


if __name__=='__main__':main()
