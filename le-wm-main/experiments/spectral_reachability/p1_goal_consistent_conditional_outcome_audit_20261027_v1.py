"""Independent bounded saved-state/Q/journal/budget arithmetic audit.

No consumer/compiler runtime imports, SVD, neural model, physics or rerun.
Successful saved records are verified, not promoted to held-out evidence.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from io import BytesIO
import math
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import p3_quality_cost_audit_20261027_v2 as io
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal

NAME='p1_goal_consistent_conditional_outcome_audit_20261027_v1'
CONSUMER='p1_goal_consistent_conditional_physical_20261027_v1'
COMPILER='p3_goal_patch_consistent_compiler_20261027_v1'
PROJECT=Path(__file__).resolve().parents[2]
BOUND='results/'+CONSUMER+'_bound'
BOUND_SHA='1950e191ab4dad3125b81d7380c1b23d8c9002c291dec54b0e32c0443901eb0b'
PHYSICAL='results/'+CONSUMER+'_main_v1'
REPORT_SHA='445ffe9c231bb4f132049389b9e689c01c3e6f8572eebfe2845c6445f143ebed'
ARMS=('old_Q','goal_patch_Q')
require=io.require
def digest(v):return io.sha(io.canonical(v).encode())
def same(a,b):return io.canonical(a)==io.canonical(b)


def array(v,shape):
    a=np.asarray(v);require(a.dtype.kind in 'fi' and a.shape==shape and np.isfinite(a).all(),'finite aligned real array')
    return a.astype(float)


def close(a,b):
    if isinstance(b,dict):return isinstance(a,dict) and set(a)==set(b) and all(close(a[k],v) for k,v in b.items())
    if type(b) is bool:return type(a) is bool and a==b
    return type(a) in (int,float) and math.isfinite(a) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10)


def labels(state,goal):
    s,g=array(state,(7,)),array(goal,(7,));d=s-g
    norm=lambda ids:math.sqrt(math.fsum(float(d[i])**2 for i in ids))
    xy=norm((2,3));wrapped=abs(math.atan2(math.sin(d[4]),math.cos(d[4])))
    diff=abs(float(d[4]));angle=min(diff,2*math.pi-diff);pos=norm(range(4))
    return dict(pose=dict(xy_distance=xy,angle_distance=wrapped,pose_cost=xy+20*wrapped),
        official=dict(official_position_error=pos,agent_position_error=norm((0,1)),object_position_error=xy,
            official_state_distance=norm(range(7)),official_success=pos<20 and angle<math.pi/9,
            object_only_success=xy<20 and angle<math.pi/9,angle_error=angle))


def qcost(z,goal,q):
    q=array(q,(192,192));require(np.allclose(q,q.T,atol=1e-8,rtol=0) and
        np.allclose(q@q,q,atol=1e-8,rtol=0) and np.trace(q)>.5,'native nonzero orthogonal Q')
    r=q@(array(z,(192,))-array(goal,(192,)))
    return float(np.mean(r**2))


def prepare_read(e,path,expected):
    receipt=e.json(Path(path)/'prepare.json',expected)
    protocol=e.json(Path(path)/'protocol.json',receipt['protocol_sha256'])
    for seal in receipt['source_seals']:e.seal(seal)
    return receipt,protocol


def load_bundle(root):
    root=Path(root).resolve();e=io.Evidence(root)
    receipt,p=prepare_read(e,root/BOUND,BOUND_SHA)
    require(receipt['status']==p['stage']=='report_bound_ready' and p['schema']==CONSUMER+'.protocol','own bound source preparation')
    _,original=prepare_read(e,p['source_prepared'],p['source_prepare_sha256'])
    require(original['stage']=='source_prepared_awaiting_compiler_report','original source stage')
    require(all(same(p[k],v) for k,v in original.items() if k!='stage'),'bound did not alter source protocol')
    _,envelope=prepare_read(e,p['compiler_prepared'],p['compiler_prepare_sha256'])
    c=envelope['protocol'];require(digest(c)==envelope['protocol_id']==p['compiler_protocol_id'],'compiler frozen protocol')
    require(all(same(p[k],c[k]) for k in ('roots','packets','model_bindings','arm_order')),'same native input/model/root/goal/Q')
    for path,h in p['compiler_outputs_sha256'].items():e.raw(Path(path),h)
    compilation=e.json(Path(p['compiler_output'])/'report.json',p['compiler_report_sha256'])
    compile_events={r['root_sha256']:[io.decode(line) for line in e.raw(Path(r['journal_path']),r['journal_sha256']).splitlines()]
        for r in compilation['roots']}
    _,references=prepare_read(e,p['reference_prepared'],p['reference_prepare_sha256'])
    require(p['roots']==references['roots'] and p['references']==references['references'],'original independent root/frame/goal references')
    directory=root/PHYSICAL;report=e.json(directory/'report.json',REPORT_SHA)
    started=e.json(directory/'execution_protocol.json');physical=e.json(directory/'physical_report.json',report['physical_report_sha256'])
    require(physical['consumer_prepare_sha256']==BOUND_SHA and physical['compiler_report_sha256']==p['compiler_report_sha256'],'durable physical source')
    rows=io.index(report['rows'],('root_sha256','method'));encodings=io.index(report['encodings'],('root_sha256','method'))
    plans=io.index(p['plans'],('root_sha256','method'));events={};pixels={}
    from PIL import Image
    for key,plan in plans.items():
        row=rows[key];stem=f'root_{plan["root_index"]}_{plan["method"]}'
        require(same(row,e.json(directory/(stem+'.endpoint.json'))),'endpoint file/report equality')
        raw=e.json(directory/(stem+'_replay0.physical.json'))
        require(all(same(row[k],v) for k,v in raw.items()),'original durable branch/report equality')
        path=directory/(stem+'_replay0.events.jsonl');require(row['journal_path']==str(path),'fixed own journal path')
        events[key]=[io.decode(line) for line in e.raw(path,row['journal_sha256']).splitlines()]
        require(len(events[key])<=31,'bounded H2 journal');pixels[key]=[]
        for event in events[key]:
            if 'rgb_path' not in event:continue
            step=event.get('primitive_step',0);require(type(step) is int and 0<=step<=10,'saved image step')
            path=directory/f'{stem}_replay0_rgb_{step}.png';require(event['rgb_path']==str(path),'fixed image path')
            raw=e.raw(path,event['rgb_file_sha256'])
            with Image.open(BytesIO(raw)) as im:
                require(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'bounded canonical RGB PNG')
                count=int(np.count_nonzero(np.all(np.asarray(im)==[65,105,225],axis=-1)))
            pixels[key].append(dict(primitive_step=step,canonical_agent_pixels=count))
        require(same(encodings[key],e.json(directory/(stem+'.encoding.json'))),'encoding file/report equality')
    budget_events=[io.decode(line) for line in e.raw(directory/'physical_budget.events.jsonl',physical['physical_budget_journal_sha256']).splitlines()]
    encoding_events=[io.decode(line) for line in e.raw(directory/'encoding.events.jsonl').splitlines()]
    require(len(budget_events)<=1000 and len(encoding_events)<=100,'bounded side journals')
    e.recheck()
    return dict(protocol=p,compiler_protocol=c,compilation=compilation,compile_events=compile_events,
        report=report,physical=physical,started=started,rows=rows,plans=plans,encodings=encodings,
        events=events,pixels=pixels,budget_events=budget_events,encoding_events=encoding_events,
        packets={x['root_sha256']:x for x in p['packets']}),e


def inspect_branch(plan,row,events,pixels,packet,reference):
    require(plan['schema']==CONSUMER+'.plan' and row['schema']==CONSUMER+'.branch' and
        digest({k:v for k,v in plan.items() if k!='selection_sha256'})==plan['selection_sha256'],'own selected program hash/schema')
    require(all(same(row[k],plan[k]) for k in ('root_sha256','root_index','method','selection_sha256','commands',
        'planned_commands','planned_horizon_tokens','active_metric_id','export_selection_sha256','compiler_report_sha256')),'root/action/metric selection binding')
    require(type(row['planned_horizon_tokens']) is int and row['planned_horizon_tokens']==2 and
        row['planned_commands']==10 and type(row['repetition']) is int and row['repetition']==0,'fixed H2 one repetition')
    raw=array(plan['commands'],(10,2));require(np.all(np.abs(raw)<=1),'all raw commands bounded, no clip')
    a,b=(io.count(row[k]) for k in ('physical_env_step_attempts','physical_env_step_returns'))
    require(b<=a<=10 and a==row['attempted_commands'] and b==row['completed_commands'],'physical attempt/return counters')
    ctx=row['context_identity'];require(all(ctx[k]==reference[k] for k in ('history_rgb_sha256','goal_rgb_sha256')) and
        ctx['source_input_sha256']==packet['source_input_sha256'],'original root/RGB/source identity')
    require(ctx['raw_history']==[[0.,0.]]*3,'unchanged zero warmup')
    mean=np.tile(np.array([-.0087,.0068],np.float32),5);std=np.tile(np.array([.2019,.2002],np.float32),5)
    expected=((np.zeros((2,10),np.float32)-mean)/std).tolist()
    require(same(ctx['normalized_between_history'],expected) and same(expected,packet['root_context']['action_history']),'exact float32 warmup token CONTENT')
    require(events and events[0]['kind']=='initial' and same(events[0]['state'],ctx['state']),'initial decision state context')
    na=nr=0;stage='initial';visible=None;terminated=False;seen=[];observations=[];states=[]
    for index,event in enumerate(events):
        kind=event['kind']
        if kind in ('initial','observation'):
            step=event.get('primitive_step',0)
            if kind=='initial':require(index==0,'one initial image')
            else:
                require(stage=='await_observation' and step==nr and same(event['command'],plan['commands'][step-1]),'ordered returned observation')
                require(type(event['terminated']) is bool and type(event['truncated']) is bool,'typed termination flags')
                terminated=event['terminated'] or event['truncated'];observations.append(event)
                require(close(event['reward'],-labels(event['state'],ctx['goal'])['official']['official_state_distance']),'saved reward vs raw-state official distance')
            require(len(seen)<len(pixels) and pixels[len(seen)]['primitive_step']==step,'image/step census')
            visible=io.count(pixels[len(seen)]['canonical_agent_pixels']);require(visible<=224*224,'pixel count bound')
            seen.append(event);stage='ready';states.append(dict(primitive_step=step,**labels(event['state'],ctx['goal'])))
        elif kind=='attempt':
            require(stage=='ready' and visible is not None and visible>0 and not terminated,'fresh visible nonterminated dispatch only')
            na+=1;require(na<=10 and event['primitive_step']==na and same(event['command'],plan['commands'][na-1]),'raw action order')
            stage='await_return'
        elif kind=='step_returned':
            require(stage=='await_return' and event['primitive_step']==na,'return order');nr+=1;stage='await_observation'
        else:raise ValueError('unknown physical event')
    require(na==a and nr==b and len(seen)==len(pixels),'complete attempt/return/image census')
    require(same([{k:v for k,v in x.items() if k not in ('rgb_path','rgb_file_sha256')} for x in observations],row['trace']),'durable observation trace')
    require(row['remaining_commands_not_executed']==10-b and row['actual_endpoint_primitive_steps']==b,'remaining path/endpoint')
    guard=row['guard'];require(guard['root_id']==plan['root_sha256'] and guard['observations']==pixels and
        guard['primitive_commands_authorized']==a and guard['P1_admission'] is False,'independent guard pixels/state machine')
    if visible==0:require(row['status']=='observation_unavailable' and guard['abstained'] is True,'guard partial retained')
    if row['status']=='completed':require(b==10 and stage=='ready' and visible>0 and not terminated and not guard['abstained'],'complete guard contract')
    if row['status']=='environment_terminated':require(terminated,'actual termination flag')
    fresh=bool(seen and a==b and seen[-1].get('primitive_step',0)==b and row['status'] in ('completed','observation_unavailable','environment_terminated'))
    full=fresh and row['status']=='completed' and b==10
    scope=('full_H2' if full else 'partial_H2') if fresh else 'unknown_or_stale_endpoint'
    require(row['endpoint_fresh'] is fresh and row['full_horizon_completed'] is full and row['endpoint_scope']==scope,'full/partial/fresh classification')
    require(seen[-1]['rgb_path']==row['terminal_rgb_path'] and seen[-1]['rgb_file_sha256']==row['terminal_rgb_sha256'] and
        same(seen[-1]['state'],row['final_state']),'terminal state/image chronology')
    expected=labels(row['final_state'],ctx['goal'])
    require(close(row['pose'],expected['pose']) and close(row['official'],expected['official']),'independent state to pose/official')
    return states


def inspect_encoding(plan,row,encoding):
    require(encoding['schema']==CONSUMER+'.encoding' and encoding['status']=='completed' and encoding['failure'] is None and
        encoding['attempted_encode_samples']==1 and row['endpoint_fresh'],'complete fresh endpoint encoding')
    require(all(encoding[k]==row[k] for k in ('root_sha256','method','selection_sha256','endpoint_scope','active_metric_id','terminal_rgb_sha256')),'endpoint encoding exact identity')
    costs={a:qcost(encoding['terminal_vector'],plan['goal'],plan['projections'][a]) for a in ARMS}
    require(close(encoding['actual_Q_costs'],costs) and encoding['q_sha256']=={a:digest(plan['projections'][a]) for a in ARMS},'same endpoint latent BOTH Q arithmetic/source')
    return costs


def physical_budget(events,plans):
    steps={};resets={};branches={};pending=None
    for e in events:
        require(e['kind'] in ('env_step','env_reset') and e['phase'] in ('root_collection','warmup','selected') and
            type(e['completed']) is bool,'known physical budget operation')
        phase=e['phase'];key=e.get('root_sha256'),e.get('method');op=e['kind'];table=steps if op=='env_step' else resets
        c=table.setdefault(phase,dict(attempts=0,returns=0))
        if phase!='root_collection':require(key in plans and e['selection_sha256']==plans[key]['selection_sha256'],'meter source identity')
        else:require(key==(None,None) and op=='env_reset','collector has zero step budget')
        identity=(op,phase,key,e['index'])
        if not e['completed'] and 'error' not in e:
            require(pending is None and type(e['index']) is int and e['index']==c['attempts'],'meter attempt index/order')
            pending=(identity,deepcopy(e));c['attempts']+=1
            if op=='env_step':
                bc=branches.setdefault(key,{}).setdefault(phase,dict(attempts=0,returns=0));bc['attempts']+=1
                if phase=='warmup':require(e['command']==[0.,0.] and bc['attempts']<=15 and c['attempts']<=120,'warmup budget/raw commands')
                else:require(bc['attempts']<=10 and c['attempts']<=80 and same(e['command'],plans[key]['commands'][bc['attempts']-1]),'selected command budget')
        else:
            require(pending is not None and pending[0]==identity and e==dict(pending[1],completed=True),'successful exact meter completion')
            pending=None;c['returns']+=1
            if op=='env_step':branches[key][phase]['returns']+=1
    require(pending is None,'incomplete meter event preserved, cannot call complete')
    return steps,resets,branches


def encoding_budget(events,plans,encodings):
    require(events[:2]==[dict(kind='encoding_setup',completed=False),dict(kind='encoding_setup',completed=True)],'setup after durable physical report receipt')
    position=2;counts=dict(encoder_attempts=0,encoder_completed=0,projector_attempts=0,projector_completed=0,
        forbidden_predict_attempts=0,forbidden_action_encoder_attempts=0,blocked_module_entries=0)
    for key,plan in plans.items():
        group=events[position:position+6];require(len(group)==6,'one encoder+projector per endpoint');position+=6
        require([e['kind'] for e in group]==['endpoint_encode','encoder','encoder','projector','projector','endpoint_result'],'endpoint module order')
        require(all((e.get('root_sha256'),e.get('method'))==key and e['selection_sha256']==plan['selection_sha256'] for e in group),'no module cross-arm reassignment')
        for i,name in ((1,'encoder'),(3,'projector')):
            start,finish=group[i:i+2]
            require(start['completed'] is False and start['index']==counts[name+'_attempts'] and finish==dict(start,completed=True),'independent module attempt/completion receipt')
            counts[name+'_attempts']+=1;counts[name+'_completed']+=1;require(counts[name+'_attempts']<=8,'module cap')
        require(same({k:v for k,v in group[-1].items() if k!='kind'},encodings[key]),'durable encoded latent result')
        require(group[0]['attempted_encode_samples']==1 and group[0]['terminal_rgb_sha256']==encodings[key]['terminal_rgb_sha256'],'encode intent physical endpoint')
    require(position==len(events),'no extra encoder/projector/predict events')
    return counts


def compiler_budget(b):
    r=b['compilation'];p=b['protocol'];rows=io.index(r['rows'],('root_sha256','arm'))
    require(r['schema']==COMPILER+'.report' and r['status']=='cuda_completed_h2_prerequisite' and
        r['failure'] is None and r['failures']==[] and r['model_unchanged'] is True,'complete actual planner receipt')
    require(set(rows)=={(root,a) for root in p['roots'] for a in ARMS},'all eight compiler arms')
    totals=Counter()
    for record in r['roots']:
        root=record['root_sha256'];journal=b['compile_events'][root]
        receipt=record['neural_receipt'];require(receipt['blocked']==0,'no blocked compiler calls')
        entries=[{k:v for k,v in e.items() if k not in ('arm','root_sha256')} for e in journal if e['kind'] in ('predict','action_encoder')]
        require(len(entries)==2*len(receipt['events']),'all actual module entry pairs')
        for index,event in enumerate(receipt['events']):
            pair=[e for e in entries if e['index']==index]
            require(event['completed'] is True and event['context_tokens']==3 and pair==[dict(event,completed=False),event],'bound module attempt/completion')
            totals[event['kind']]+=1
        for arm in ARMS:
            row=rows[root,arm];c=row['compilation'];plan=b['plans'][root,arm]
            require(row['status']=='completed' and len(c['candidates'])==10 and c['whole_and_segmented_exact'] is True and
                same(c['winner']['commands'],plan['commands']) and same(c['winner']['context'],plan['selected_predicted_context']),'fresh own compiler selected program')
            require(c['winner']==min(c['candidates'],key=lambda x:(x['objective'],x['index'])),'sealed objective stable selection')
            for kind in ('predict','action_encoder'):require(sum(e['kind']==kind and e['role']==arm for e in receipt['events'])==66,'matched arm complete budget')
            ops=[e for e in journal if e.get('arm')==arm and e['kind'] in ('jacobian','svd')]
            require(len(ops)==46 and len(c['operations'])==23,'all AD/SVD attempts')
            for op in c['operations']:
                pair=[{k:v for k,v in e.items() if k not in ('arm','root_sha256')} for e in ops if e['index']==op['index']]
                require(op['completed'] is True and pair==[dict(op,completed=False),op],'nonduplicated AD/SVD operations')
                totals[op['kind']]+=1
                if op['kind']=='jacobian':require(op['shape']==[192,10] and op['mathematical_vjp_directions']==192,'mathematical VJP not API count')
            replays=[e for e in journal if e.get('arm')==arm and e['kind']=='replay' and e['completed']]
            require([e['phase'] for e in replays]==['candidate_'+str(i) for i in range(10)]+['winner_whole','winner_segment_0','winner_segment_1'],'all nonlinear independent replays')
            replay_calls=sum(len(e['commands'])//5 for e in replays);require(replay_calls==24,'replay calls included, never free')
            totals['replay_predict']+=replay_calls
    require(dict(totals)==dict(predict=528,action_encoder=528,jacobian=88,svd=96,replay_predict=192),'complete compiler accounting')
    resources=r['resources']
    for kind,field,n in [('predict','predict_samples',528),('action_encoder','action_encoder_samples',528),('jacobian','jacobians',88),('svd','svds',96)]:
        require(resources['attempted_'+field]==resources['completed_'+field]==n,'actual planner resource totals: '+kind)
    return dict(predict=528,action_encoder=528,jacobians=88,svds=96,replay_predict_included=192,
        ad_forward_predict_included=88,mathematical_vjp_directions=16896,backward_engine_api_calls=None,
        scope='inherited once; ADforward and replay already in528, not added again; not full-pipeline FLOPs')


def pair_rows(roots,rows,encodings):
    pairs=[]
    for root in roots:
        old,new=(rows[root,a] for a in ARMS)
        identical='context_identity' in old and 'context_identity' in new and same(old['context_identity'],new['context_identity'])
        full=identical and all(r['full_horizon_completed'] and r['endpoint_fresh'] and r['physical_env_step_returns']==10 for r in (old,new))
        partial=identical and all(r['endpoint_fresh'] for r in (old,new)) and old['physical_env_step_returns']==new['physical_env_step_returns']
        pairs.append(dict(root_index=old['root_index'],root_sha256=root,
            classification='full_H2_pair' if full else 'partial_same_steps' if partial else 'partial_or_unmatched',
            actual_steps=[old['physical_env_step_returns'],new['physical_env_step_returns']],identical_initial_context=identical,
            new_minus_old_pose_cost=new['pose']['pose_cost']-old['pose']['pose_cost'] if full else None,
            new_minus_old_official_distance=new['official']['official_state_distance']-old['official']['official_state_distance'] if full else None,
            endpoint_Q_costs={arm:encodings[root,arm]['actual_Q_costs'] for arm in ARMS},
            new_minus_old_same_Q_cost={q:encodings[root,ARMS[1]]['actual_Q_costs'][q]-encodings[root,ARMS[0]]['actual_Q_costs'][q] for q in ARMS}
                if full else None))
    return pairs


def audit(b):
    p=b['protocol'];r=b['report'];roots=p['roots']
    require(len(roots)==len(set(roots))==4 and set(b['rows'])==set(b['plans'])==set(b['encodings'])==
        {(root,a) for root in roots for a in ARMS},'four-root eight-arm full census')
    require(r['schema']==CONSUMER+'.report' and r['status']=='guarded_calibration_recorded' and r['failures']==[] and
        r['consumer_prepare_sha256']==BOUND_SHA and r['compiler_report_sha256']==p['compiler_report_sha256'] and
        r['source_verified'] is r['model_loaded'] is r['model_unchanged'] is r['model_hash_verified'] is True,'complete physical source/model claims')
    require(all(r[k] is False for k in ('P1_admission','P3_admission','P4_admission','efficacy_claim')) and r['new_predict_calls']==0,'scope/predict boundary')
    require(b['started']['prepare_sha256']==BOUND_SHA and b['started']['prepared']==str(PROJECT/BOUND) and
        b['started']['encode_terminals'] is True,'physical execution binding')
    require(same(r['rows'],b['physical']['rows']) and same(r['physical_resources'],b['physical']['physical_resources']) and
        b['physical']['failures']==[],'physical-before-CUDA durable report')
    state_rows=[]
    for key,plan in b['plans'].items():
        packet=b['packets'][key[0]];row=b['rows'][key]
        require(plan['input_sha256']==packet['input_sha256'] and plan['source_input_sha256']==packet['source_input_sha256'] and
            same(plan['predictor_context'],packet['root_context']) and same(plan['goal'],packet['goal']) and
            same(plan['projections'],packet['projections']) and plan['active_metric_id']==packet['active_metric_ids'][key[1]],'same source/model/goal/projections')
        states=inspect_branch(plan,row,b['events'][key],b['pixels'][key],packet,p['references'][key[0]])
        costs=inspect_encoding(plan,row,b['encodings'][key])
        state_rows.append(dict(root_index=plan['root_index'],root_sha256=key[0],arm=key[1],states=states,endpoint_Q_costs=costs,
            endpoint_scope=row['endpoint_scope'],actual_steps=row['physical_env_step_returns']))
    steps,resets,branches=physical_budget(b['budget_events'],b['plans'])
    require(dict(steps=steps,resets=resets)==r['physical_resources'],'independent physical phase totals')
    require(set(branches)==set(b['plans']) and all(c==dict(warmup=dict(attempts=15,returns=15),selected=dict(attempts=10,returns=10)) for c in branches.values()),'all 80 selected +120 warmup attempts/returns')
    require(steps['selected']==dict(attempts=80,returns=80) and steps['warmup']==dict(attempts=120,returns=120) and
        resets==dict(root_collection=dict(attempts=32,returns=32),warmup=dict(attempts=8,returns=8)),'no budget omission or reset conflation')
    require(r['physical_env_step_attempts']==r['physical_env_step_returns']==r['planned_selected_path_steps']==80,'selected report totals')
    enc=encoding_budget(b['encoding_events'],b['plans'],b['encodings']);require(enc==r['encoding_stats'],'encoder/projector independent counters')
    upstream=compiler_budget(b);pairs=pair_rows(roots,b['rows'],b['encodings']);saved=io.index(r['pairs'],('root_sha256',))
    for pair in pairs:
        require(all(close(saved[pair['root_sha256'],][k],pair[k]) if k.startswith('new_minus_old_') else
            same(saved[pair['root_sha256'],][k],pair[k]) for k in
            ('classification','actual_steps','identical_initial_context','new_minus_old_pose_cost','new_minus_old_official_distance')),'independent paired results')
    full=[p for p in pairs if p['classification']=='full_H2_pair']
    mean=lambda key:math.fsum(x[key] for x in full)/len(full) if full else None
    return dict(schema=NAME+'.report',status='saved_arithmetic_verified',all_checks=True,rows=state_rows,pairs=pairs,
        pair_counts=dict(Counter(p['classification'] for p in pairs)),budget=steps,reset_budget=resets,
        encoding_budget=enc,upstream_budget=upstream,
        comparison=dict(complete_root_pairs=len(full),pose_improved_roots=sum(x['new_minus_old_pose_cost']<0 for x in full),
            official_distance_improved_roots=sum(x['new_minus_old_official_distance']<0 for x in full),
            mean_new_minus_old_pose_cost=mean('new_minus_old_pose_cost'),mean_new_minus_old_official_distance=mean('new_minus_old_official_distance')),
        denominator=dict(root_pairs=4,branches=8,repetitions=1,new_independent_roots=0),
        gaps=['Saved arithmetic only, no independent simulator or encoder regeneration',
            'Original pre-warmup root state and warmup RGB frames absent from output; source fingerprints/goal/frame SHA bound, not re-rendered; goal state uses saved context, not independently recovered from goal RGB',
            'Model tensor hash claims inherited from successful executor, not independently regenerated',
            'Planner mechanism/SVD/derivative truth and normalization training provenance not re-proved here',
            'Four calibration roots only; pose and full-state official metrics differ; no held-out efficacy/admission',
            'Historical model/context/goal acquisition cost unknown; no full-pipeline equivalence'],
        new_neural_calls=0,new_physical_steps=0,new_svds=0,P1_admission=False,P3_admission=False,P4_admission=False)


def write(path,value):
    with Path(path).open('x') as stream:
        stream.write(io.canonical(value)+'\n');stream.flush();os.fsync(stream.fileno())


def run(root,output):
    root=Path(root).resolve();output=Path(output).absolute();require(not output.exists(),'exclusive NEW audit output')
    b,e=load_bundle(root);result=audit(b);e.recheck()
    sources=[Path(__file__).resolve(),root/'tests'/('test_'+NAME+'.py'),root/'innerdocs'/(NAME+'.md'),Path(io.__file__).resolve()]
    output.mkdir(exist_ok=False)
    seal=create_seal(files=sources,args_json=io.canonical(dict(physical_report_sha256=REPORT_SHA)),output_dir=output/'source_seal')
    result.update(inputs_sha256=dict(sorted(e.hashes.items())),source_seal=seal,physical_report_sha256=REPORT_SHA,consumer_prepare_sha256=BOUND_SHA)
    write(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project-root',type=Path,default=PROJECT)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    r=run(args.project_root,args.output);print(io.canonical({k:r[k] for k in ('status','all_checks','pair_counts','comparison','budget','encoding_budget')}))


if __name__=='__main__':main()
