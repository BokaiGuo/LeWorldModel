"""Bounded saved-evidence arithmetic: no NumPy/SVD, model, CUDA or simulator.

Only complete same-root H3 pairs enter quality deltas. Every partial, abort,
failed step and duplicate/tied action execution remains in the output census.
Source seals prove unchanged bytes, not independent neural replay correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from io import BytesIO
import json
import math
import os
from pathlib import Path

from experiments.spectral_reachability import p3_quality_cost_audit_20261027_v2 as io
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal

NAME='p3_action_jacobian_outcome_audit_20261027_v1'
CONSUMER='p3_action_jacobian_physical_consumer_20261027_v3'
SCHEDULER='p3_action_jacobian_budget_schedule_20261027_v3'
PHYSICAL='results/'+CONSUMER+'_main_v1/report.json'
PHYSICAL_SHA='d0f29d41ab0af04eac2c50e33d24385278868b087fb1002ab9f89a7583f044e8'
PREPARED='results/'+CONSUMER+'_prepared'
PREPARE_SHA='5cb7ef6cbebf38859dbf1b087131c3dca8a9b11cedad26cf64978c4416df2a6d'
PLANNING='results/'+SCHEDULER+'_cuda_main_v1/report.json'
PLANNING_SHA='c13dab20228fd590c3f9375e29b6be905a8db083942a547c862d97c90cf53921'
SCHEDULER_PREPARED='results/'+SCHEDULER+'_prepared'
SCHEDULER_PREPARE_SHA='27988bdb563563bd2cb0ea5ef3f69851627a2cae4d648e1e545f2e68c7d5c134'
METHODS=('uniform','dynamic')
require=io.require
same=lambda a,b:io.canonical(a)==io.canonical(b)


def number(value):
    require(type(value) in (float,int) and math.isfinite(value),'finite real scalar required')
    return float(value)


def vector(value,n):
    require(isinstance(value,list) and len(value)==n,'bounded aligned vector required')
    return [number(x) for x in value]


def close(a,b):
    if isinstance(b,dict):return isinstance(a,dict) and set(a)==set(b) and all(close(a[k],v) for k,v in b.items())
    if type(b) is bool:return type(a) is bool and a==b
    return type(a) in (int,float) and math.isfinite(a) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10)


def state_labels(state,goal):
    s,g=vector(state,7),vector(goal,7)
    norm=lambda indices:math.hypot(*(g[i]-s[i] for i in indices))
    difference=abs(g[4]-s[4]);official_angle=min(difference,2*math.pi-difference)
    angle=abs(math.atan2(math.sin(s[4]-g[4]),math.cos(s[4]-g[4])))
    xy=norm((2,3));position=norm(range(4))
    return dict(pose=dict(xy_distance=xy,angle_distance=angle,pose_cost=xy+20*angle),
        official=dict(official_position_error=position,agent_position_error=norm((0,1)),object_position_error=xy,
            official_state_distance=norm(range(7)),official_success=position<20 and official_angle<math.pi/9,
            object_only_success=xy<20 and official_angle<math.pi/9,angle_error=official_angle))


def projected_cost(terminal,packet):
    y,g=vector(terminal,192),vector(packet['goal'],192);q=packet['projection']
    require(isinstance(q,list) and len(q)==192,'bounded original Q required')
    residual=[a-b for a,b in zip(y,g)]
    projected=[sum(a*b for a,b in zip(vector(row,192),residual)) for row in q]
    cost=sum(x*x for x in projected)/192
    require(math.isfinite(cost),'finite projected cost required')
    return cost


def load_bundle(root):
    """Explicit bounded files only; pixel counting is not neural inference."""
    root=Path(root).resolve();e=io.Evidence(root)
    receipt=e.json(root/PREPARED/'prepare.json',PREPARE_SHA)
    require(receipt['status']=='prepared_not_executed' and receipt['model_loaded'] is False,'physical prepare receipt')
    require(receipt['protocol_path']==str(root/PREPARED/'protocol.json'),'physical protocol path')
    protocol=e.json(root/PREPARED/'protocol.json',receipt['protocol_sha256'])
    for path,h in protocol['inputs_sha256'].items():e.raw(Path(path),h)
    for seal in receipt['source_seals']:e.seal(seal)
    scheduler_receipt=e.json(root/SCHEDULER_PREPARED/'prepare.json',SCHEDULER_PREPARE_SHA)
    envelope=e.json(root/SCHEDULER_PREPARED/'protocol.json',scheduler_receipt['protocol_sha256'])
    scheduler=envelope['protocol'];identity=io.sha(io.canonical(scheduler).encode())
    require(identity==envelope['protocol_id']==scheduler_receipt['protocol_id']==protocol['planning_protocol_id'],
            'scheduler protocol identity')
    for seal in scheduler_receipt['source_seals']:e.seal(seal)
    planning=e.json(root/PLANNING,PLANNING_SHA);physical=e.json(root/PHYSICAL,PHYSICAL_SHA)
    directory=(root/PHYSICAL).parent
    auxiliary=dict(execution=e.json(directory/'execution_protocol.json'),physical=e.json(directory/'physical_report.json'))
    require(auxiliary['execution']['prepared']==str(root/PREPARED),'physical execution prepare path')
    disks={};events={};pixels={};encodings={};endpoints={}
    # The pinned prepared plan census, NOT outcome-array order, controls reads.
    for plan in protocol['plans']:
        key=plan['root_sha256'],plan['method'];stem=f'root_{plan["root_index"]}_{plan["method"]}_replay0'
        disks[key]=e.json(directory/(stem+'.physical.json'))
        endpoints[key]=e.json(directory/(stem+'.endpoint.json'))
        encodings[key]=e.json(directory/(stem+'.encoding.json'))
        row=io.index(physical['rows'],('root_sha256','method'))[key]
        journal=directory/(stem+'.events.jsonl')
        require(row['journal_path']==str(journal),'fixed branch journal path')
        raw=e.raw(journal,row['journal_sha256'])
        require(len(raw.splitlines())<=46,'bounded H3 journal')
        events[key]=[io.decode(line) for line in raw.splitlines()];pixels[key]=[]
        from PIL import Image
        for event in events[key]:
            if 'rgb_path' not in event:continue
            step=event.get('primitive_step',0);io.count(step);require(step<=15,'bounded RGB index')
            path=directory/f'{stem}_rgb_{step}.png'
            require(event['rgb_path']==str(path),'fixed branch RGB path')
            image=e.raw(path,event['rgb_file_sha256'])
            with Image.open(BytesIO(image)) as im:
                require(im.format=='PNG' and im.size==(224,224) and im.mode=='RGB','bounded canonical RGB PNG')
                count=sum(pixel==(65,105,225) for pixel in im.getdata())
            pixels[key].append(dict(primitive_step=step,canonical_agent_pixels=count))
    # Every model arm and its journal still matches the v3 sealed receipt.
    for row in planning['rows']:
        path=(root/PLANNING).parent/(row['root_sha256']+'_'+row['mode']+'.json')
        require(same(e.json(path),row),'planning per-arm file/report mismatch')
        path=(root/PLANNING).parent/(row['root_sha256']+'_'+row['mode']+'.events.jsonl')
        require(row['journal_path']==str(path),'planning journal path')
        journal=[io.decode(x) for x in e.raw(path,row['journal_sha256']).splitlines()]
        emitted=[]
        for event in row['search']['events']:
            emitted.extend([dict(kind='attempt',depth=event['depth'],path=event['path']),event])
            if event['depth']==1 and event['primitive_index']==15:
                emitted.append(dict(kind='allocation',signal=row['search']['signal'],allocation=row['search']['allocation']))
        require(same(emitted,journal),'planning journal event order/content')
    e.recheck()
    return dict(protocol=protocol,scheduler=scheduler,planning=planning,physical=physical,
        auxiliary=auxiliary,disks=disks,endpoints=endpoints,encoding_files=encodings,events=events,pixels=pixels),e


def inspect_branch(plan,row,events,pixels,reference,packet):
    require(row['root_sha256']==plan['root_sha256'] and row['method']==plan['method'] and
            row['selection_sha256']==plan['selection_sha256'] and row['root_index']==plan['root_index'],'root/action selection binding')
    require(type(row['repetition']) is int and row['repetition']==0,'exact one physical repetition')
    require(row['planned_commands']==plan['planned_commands']==15 and row['planned_horizon_tokens']==3,'fixed H3 horizon')
    ctx=row['context_identity']
    require(all(ctx[k]==reference[k] for k in ('history_rgb_sha256','goal_rgb_sha256')),'source RGB root/goal binding')
    require(events and events[0]['kind']=='initial' and same(events[0]['state'],ctx['state']),'initial reset state/journal binding')
    n_attempt=n_return=0;state='initial';images=[];observations=[];last_pixels=None;terminated=False
    require(len(events)<=46 and len(pixels)<=16,'bounded H3 event census')
    for index,event in enumerate(events):
        kind=event['kind']
        if kind in ('initial','observation'):
            step=event.get('primitive_step',0)
            if kind=='initial':require(index==0 and state=='initial','initial event order')
            else:
                require(state=='await_observation' and step==n_return,'observation/return order')
                require(same(event['command'],plan['commands'][step-1]),'observed command differs from raw path')
                require(type(event['terminated']) is bool and type(event['truncated']) is bool,'terminal flags')
                terminated=event['terminated'] or event['truncated'];observations.append(event)
            require(len(images)<len(pixels) and pixels[len(images)]['primitive_step']==step,'RGB/step order')
            last_pixels=io.count(pixels[len(images)]['canonical_agent_pixels']);require(last_pixels<=224*224,'pixel count bound')
            images.append(event);state='ready'
        elif kind=='attempt':
            require(state=='ready' and last_pixels is not None and last_pixels>0 and not terminated,
                    'dispatch without fresh visible nonterminated guard')
            n_attempt+=1;require(n_attempt<=15 and event['primitive_step']==n_attempt,'attempt order/bound')
            require(same(event['command'],plan['commands'][n_attempt-1]),'attempt command differs from raw bank path')
            state='await_return'
        elif kind=='step_returned':
            require(state=='await_return' and event['primitive_step']==n_attempt,'return order')
            n_return+=1;state='await_observation'
        else:raise ValueError('unknown physical journal event')
    trace=[{k:v for k,v in event.items() if k not in ('rgb_path','rgb_file_sha256')} for event in observations]
    require(len(images)==len(pixels) and same(trace,row['trace']),'physical trace/journal census')
    require(io.count(row['physical_env_step_attempts'])==io.count(row['attempted_commands'])==n_attempt and
            io.count(row['physical_env_step_returns'])==io.count(row['completed_commands'])==n_return,'physical attempt/return counters')
    require(row['actual_endpoint_primitive_steps']==n_return and row['remaining_commands_not_executed']==15-n_return,
            'endpoint step/remaining counters')
    guard=row['guard']
    require(guard['root_id']==row['root_sha256'] and same(guard['observations'],pixels) and
            io.count(guard['primitive_commands_authorized'])==n_attempt,'independent RoyalBlue guard receipt')
    if last_pixels==0:require(guard['abstained'] is True and row['status']=='observation_unavailable','guard abort not retained')
    if row['status']=='completed':
        require(state=='ready' and n_return==15 and last_pixels>0 and guard['abstained'] is False and not terminated,
                'completed H3 guard/termination unavailable')
    if row['status']=='environment_terminated':require(terminated and state=='ready','termination receipt unavailable')
    require(same(row['final_state'],images[-1]['state']),'stored final state/image snapshot binding')
    fresh=bool(images and row.get('terminal_rgb_path') and images[-1].get('primitive_step',0)==n_return and
        n_attempt==n_return and row['status'] in ('completed','observation_unavailable','environment_terminated'))
    require(images[-1]['rgb_path']==row['terminal_rgb_path'] and images[-1]['rgb_file_sha256']==row['terminal_rgb_sha256'],
            'terminal image binding')
    full=row['status']=='completed' and n_return==15 and fresh
    scope='full_H3' if full else 'guard_partial_or_terminated' if fresh else 'unknown_or_stale_endpoint'
    require(type(row['full_horizon_completed']) is bool and row['full_horizon_completed']==full and
            type(row['endpoint_fresh']) is bool and row['endpoint_fresh']==fresh and row['endpoint_scope']==scope,
            'independent full/partial/fresh classification')
    labels=state_labels(row['final_state'],ctx['goal'])
    require(close(row['pose'],labels['pose']) and close(row['official'],labels['official']),'independent state label arithmetic')
    result=dict(root_index=plan['root_index'],root_sha256=row['root_sha256'],method=row['method'],
        context_identity=deepcopy(ctx),full_H3=full,endpoint_fresh=fresh,endpoint_scope=scope,
        physical_env_step_attempts=n_attempt,physical_env_step_returns=n_return,
        guard_pixels=deepcopy(pixels),labels=labels,original_branch=deepcopy(row),primitive_indices=plan['primitive_indices'])
    return result


def classification(u,d):
    matched=same(u['context_identity'],d['context_identity'])
    if matched and u['full_H3'] and d['full_H3']:return 'full-path'
    if matched and u['endpoint_fresh'] and d['endpoint_fresh'] and u['physical_env_step_returns']==d['physical_env_step_returns']:
        return 'guard-partial'
    return 'unmatched'


def audit(b):
    protocol,scheduler,planning,physical=(b[k] for k in ('protocol','scheduler','planning','physical'))
    require(protocol['schema']==CONSUMER+'.protocol' and scheduler['schema']==SCHEDULER+'.protocol' and
            physical['schema']==CONSUMER+'.report' and planning['schema']==SCHEDULER+'.report','exact v3 schema required')
    roots=protocol['roots'];require(len(roots)==len(set(roots))==4 and roots==scheduler['roots'],'four root census')
    identity=io.sha(io.canonical(scheduler).encode())
    require(physical['consumer_prepare_sha256']==PREPARE_SHA and protocol['planning_prepare_sha256']==SCHEDULER_PREPARE_SHA and
            physical['planning_report_sha256']==protocol['planning_report_sha256']==PLANNING_SHA and
            physical['planning_protocol_id']==protocol['planning_protocol_id']==planning['protocol_id']==identity,'v3 source identity')
    require(physical['source_verified'] is True and physical['model_unchanged'] is True and
            planning['model_unchanged'] is True and physical['status']=='guarded_calibration_recorded','recorded source/model integrity')
    require(protocol['physical_repetitions']==physical['physical_repetitions']==1 and
            protocol['planned_commands_per_branch']==15 and protocol['planned_selected_path_steps']==physical['planned_selected_path_steps']==120,
            'fixed execution census/horizon')
    inherited=protocol['inherited_cost']
    require(inherited['status']=='inherited_unknown_cost' and all(inherited[k] is None for k in
        ('predict_samples','action_encoder_samples','backward_api_calls','vjp_directions','wall_time_seconds')),'unknown inherited cost must not become zero')
    for r in (physical,planning,scheduler):
        require(same(r['inherited_cost'],inherited) and r['total_pipeline_cost_equivalent'] is False and
                r['new_independent_roots']==0 and r['P1_admission'] is False and r['P3_admission'] is False,'cost/admission scope')
    require(physical['inherited_arrays_charged_once']==planning['inherited_arrays_charged_once']==4 and
            physical['new_jacobian_materializations']==planning['new_jacobian_materializations']==0 and
            physical['upstream_predict_samples_charged_once']==planning['known_predict_samples']==1176 and
            physical['new_predict_calls']==0 and planning['unknown_attempted_work'] is False,'upstream prediction/derivative accounting')
    expected={(r,m) for r in roots for m in METHODS}
    rows=io.index(physical['rows'],('root_sha256','method'));plans=io.index(protocol['plans'],('root_sha256','method'))
    modeled=io.index(planning['rows'],('root_sha256','mode'));packets=io.index(scheduler['packets'],('root_sha256',))
    encoded=io.index(physical['encodings'],('root_sha256','method'))
    for index in (rows,plans,modeled,encoded,b['events'],b['pixels'],b['disks'],b['endpoints'],b['encoding_files']):
        require(set(index)==expected,'exact eight distinct root-arm slots required')
    require(set(packets)=={(r,) for r in roots},'task packet census')
    aux=b['auxiliary'];require(aux['execution']['prepare_sha256']==PREPARE_SHA and
        aux['execution']['physical_repetitions']==1 and aux['execution']['new_predict_calls']==0 and
        aux['execution']['encode_terminals'] is True,'physical execution contract')
    durable_rows=io.index(aux['physical']['rows'],('root_sha256','method'))
    require(aux['physical']['schema']==CONSUMER+'.physical_report' and set(durable_rows)==set(rows) and
        all(same(durable_rows[k],rows[k]) for k in rows) and
        same(aux['physical']['failures'],physical['failures']),'durable physical report parity')
    branches=[];derived={};attempted_encodes=completed_encodes=0
    for root_index,root in enumerate(roots):
        packet=packets[root,]
        require(io.sha(io.canonical({k:v for k,v in packet.items() if k!='input_sha256'}).encode())==packet['input_sha256'],'task packet SHA')
        for method in METHODS:
            key=root,method;row,plan,model=rows[key],plans[key],modeled[key]
            require(plan['schema']==CONSUMER+'.plan' and plan['planning_report_sha256']==PLANNING_SHA and
                plan['root_index']==root_index and same(plan['inherited_cost'],inherited) and
                io.sha(io.canonical({k:v for k,v in plan.items() if k!='selection_sha256'}).encode())==plan['selection_sha256'],'sealed plan identity')
            binding=model['binding'];receipt=model['prediction_receipt']
            model_id=next(v for k,v in scheduler['model_bindings'].items() if Path(k).name=='weights.pt')+':'+next(
                v for k,v in scheduler['model_bindings'].items() if Path(k).name=='config.json')
            require(same(binding,dict(protocol_id=identity,root_id=root,frozen_model_id=model_id,
                task_id='pusht_fixed_h3_cached_latent_calibration')) and same(receipt['binding'],binding) and
                model['input_sha256']==plan['input_sha256']==packet['input_sha256'],'same root/task/feedback/model binding')
            require(model['status']=='completed' and model['whole_h3_replay_exact'] is True and model['failure'] is None and
                io.count(receipt['attempted_predict_samples'])==io.count(receipt['predict_sample_limit'])==147 and
                io.count(receipt['blocked_calls'])==0,'complete 147-sample planning arm')
            require(receipt['method']==SCHEDULER+'_'+method,'prediction method identity')
            phases=io.index(receipt['phases'],('phase',));require(set(phases)=={('planning',),('evaluation',)},'two compute phases')
            for phase,n in (('planning',144),('evaluation',3)):
                r=phases[phase,];require(r['completed'] is True and io.count(r['failed_predict_calls'])==0 and
                    io.count(r['predict_calls'])==io.count(r['predict_samples'])==n and r['context_lengths']=={'3':n} and
                    r['method']==SCHEDULER+'_'+method and type(r['horizon_tokens']) is int and r['horizon_tokens']==3,
                    '144+3 actual prediction samples')
            for k,n in dict(action_encoder_calls=147,action_encoder_samples=147,processed_action_context_tokens=441,
                visual_encoder_samples=0,runtime_action_backward_calls=0,runtime_action_vjp_rows=0,
                runtime_input_backward_calls=0,runtime_input_vjp_rows=0,new_jacobian_materializations=0,new_physical_steps=0).items():
                require(io.count(model[k])==n,'planning resource coordinate: '+k)
            require(same(model['inherited_cost'],inherited) and model['total_pipeline_cost_equivalent'] is False,'arm inherited cost scope')
            search=model['search'];chi=number(search['signal']['chi']);require(0<=chi<=1,'finite chi in [0,1]')
            k2=4 if method=='uniform' else 2+math.floor(4*chi+.5)
            require(same(search['allocation'],[k2,8-k2]) and same(plan['allocation'],search['allocation']) and
                search['status']=='completed' and io.count(search['planning_edge_attempts'])==len(search['events'])==144 and
                all(e['succeeded'] is True and e['replay_completed'] is True for e in search['events']),
                'original chi allocation and complete edge census')
            path=plan['primitive_indices'];require(len(path)==3 and all(type(i) is int and 0<=i<16 for i in path),'integer H3 raw bank path')
            raw=sum([packet['primitive_bank'][i] for i in path],[]);best=model['search']['best_plan']
            require(len(raw)==15 and same(raw,plan['commands']) and same(raw,best['actions']) and
                same(path,best['primitive_indices']) and best['depth']==3 and
                same(plan['selected_predicted_context'],best['context']),'exact selected raw path reconstruction')
            cost=projected_cost(best['context']['latent_history'][-1],packet)
            require(close(model['search']['best_plan']['cost'],cost) and close(plan['selected_model_cost'],cost),'selected model Q cost recomputation')
            disk=b['disks'][key]
            require(set(row)==set(disk)|{'endpoint_fresh','endpoint_scope'} and all(same(row[k],v) for k,v in disk.items()),'physical per-arm/report parity')
            endpoint=b['endpoints'][key]
            require(endpoint==dict(schema=CONSUMER+'.endpoint',planning_report_sha256=PLANNING_SHA,root_sha256=root,
                method=method,endpoint_fresh=row['endpoint_fresh'],endpoint_scope=row['endpoint_scope']),'v3 endpoint annotation parity')
            branch=inspect_branch(plan,row,b['events'][key],b['pixels'][key],protocol['references'][root],packet)
            enc=encoded[key];require(same(enc,b['encoding_files'][key]) and type(enc['repetition']) is int and
                enc['repetition']==0 and enc['endpoint_scope']==branch['endpoint_scope'],'endpoint encoding root/scope/file binding')
            attempts=io.count(enc['attempted_encode_samples']);require(attempts<=1,'at most one endpoint encode per arm')
            attempted_encodes+=attempts
            branch['original_encoding']=deepcopy(enc);branch['selected_model_q_cost']=cost
            if enc['status']=='completed':
                require(attempts==1 and branch['endpoint_fresh'] and enc['failure'] is None,'completed fresh endpoint encoding')
                branch['actual_latent_q_cost']=projected_cost(enc['terminal_vector'],packet);completed_encodes+=1
            else:branch['actual_latent_q_cost']=None
            derived[key]=branch;branches.append(branch)
    pairs=[];saved_pairs=io.index(physical['pairs'],('root_sha256',))
    require(set(saved_pairs)=={(r,) for r in roots},'all root paired receipts retained')
    for root_index,root in enumerate(roots):
        u,d=(derived[root,m] for m in METHODS);label=classification(u,d)
        require(same(modeled[root,'uniform']['search']['signal'],modeled[root,'dynamic']['search']['signal']),
                'same root/task action-Jacobian signal in both arms')
        saved=saved_pairs[root,]
        require(saved['classification']==label and saved['actual_steps']==[u['physical_env_step_returns'],d['physical_env_step_returns']] and
            saved['identical_initial_context']==same(u['context_identity'],d['context_identity']),'independent root pair classification')
        metrics=lambda b:dict(pose_cost=b['labels']['pose']['pose_cost'],
            **{k:b['labels']['official'][k] for k in ('official_state_distance','official_position_error','object_position_error',
                'angle_error','official_success','object_only_success')},
            actual_latent_q_cost=b['actual_latent_q_cost'],selected_model_q_cost=b['selected_model_q_cost'])
        um,dm=metrics(u),metrics(d);deltas=None
        if label=='full-path':deltas={k:dm[k]-um[k] if um[k] is not None and dm[k] is not None else None for k in um}
        pairs.append(dict(root_index=root_index,root_sha256=root,classification=label,
            actual_steps=[u['physical_env_step_returns'],d['physical_env_step_returns']],eligible_complete_H3=label=='full-path',
            identical_commands=same(plans[root,'uniform']['commands'],plans[root,'dynamic']['commands']),
            deltas=deltas,delta_convention='dynamic minus uniform; no partial deltas',
            endpoint_metrics={'uniform':um,'dynamic':dm},P3_admission=False))
    attempts=sum(x['physical_env_step_attempts'] for x in branches);returns=sum(x['physical_env_step_returns'] for x in branches)
    require(io.count(physical['physical_env_step_attempts'])==attempts and io.count(physical['physical_env_step_returns'])==returns,
            'aggregate physical work accounting')
    stats=physical['encoding_stats']
    require(io.count(physical['attempted_terminal_encode_samples'])==attempted_encodes and
        io.count(stats['encode_calls'])==io.count(stats['encode_samples']) and
        completed_encodes+io.count(stats['failed_encode_calls'])==stats['encode_calls']<=attempted_encodes and
        io.count(stats['forbidden_predict_attempts'])==0,'actual encoding accounting')
    eligible=[p for p in pairs if p['eligible_complete_H3']];metrics={}
    if eligible:
        for metric in eligible[0]['deltas']:
            values=[p['deltas'][metric] for p in eligible if p['deltas'][metric] is not None]
            success=metric.endswith('success')
            metrics[metric]=dict(paired_root_count=len(values),mean_dynamic_minus_uniform=sum(values)/len(values) if values else None,
                dynamic_wins=sum(v>0 if success else v<0 for v in values),ties=sum(v==0 for v in values),
                dynamic_losses=sum(v<0 if success else v>0 for v in values),higher_is_better=success)
    rc=Counter(p['classification'] for p in pairs)
    return dict(schema=NAME+'.report',status='saved_outcomes_verified',branches=branches,pairs=pairs,
        root_classifications={k:rc[k] for k in ('full-path','guard-partial','unmatched')},
        branch_classifications=dict(full_H3=sum(b['full_H3'] for b in branches),
            partial=sum(not b['full_H3'] and b['endpoint_fresh'] for b in branches),
            unmatched=sum(not b['endpoint_fresh'] for b in branches)),
        comparison=dict(eligible_root_count=len(eligible),excluded_roots=[p['root_sha256'] for p in pairs if not p['eligible_complete_H3']],
            metrics=metrics,scope='complete paired H3 only; descriptive calibration, no inferential efficacy claim'),
        accounting=dict(physical_env_step_attempts=attempts,physical_env_step_returns=returns,planned_selected_path_steps=120,
            terminal_encode_samples=stats['encode_samples'],attempted_terminal_encode_samples=attempted_encodes,
            terminal_encoding_stats=deepcopy(stats),
            upstream_predict_samples_charged_once=1176,upstream_action_encoder_samples=1176,upstream_context_tokens=3528,
            inherited_cost=deepcopy(inherited),inherited_arrays_charged_once=4,new_jacobian_materializations=0,
            total_pipeline_cost_equivalent=False,guard_abort_branches=sum(b['original_branch']['status']=='observation_unavailable' for b in branches)),
        calibration_roots=4,physical_branches=8,physical_repetitions=1,new_independent_roots=0,
        original_run_status=physical['status'],original_run_failures=deepcopy(physical['failures']),
        new_predict_samples=0,new_physical_steps=0,new_encode_samples=0,new_svd_calls=0,
        P1_admission=False,P3_admission=False,efficacy_claim=False,
        limitations=['opened calibration roots; no independent P3 admission','partial endpoints excluded from complete-H3 quality',
            'source/receipt consistency is not independent neural-output reproduction','inherited Jacobian and upstream costs unknown/excluded; no full-pipeline equivalence',
            'reset/root collection/warmup and model setup excluded from selected-path counters'])


def write(path,value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def run(root,output):
    root=Path(root).resolve();output=Path(output).absolute()
    require(not output.exists(),'exclusive NEW audit output required')
    bundle,e=load_bundle(root)
    files=[Path(__file__).resolve(),root/'tests'/('test_'+NAME+'.py'),root/'docs'/(NAME+'.md'),
        Path(io.__file__).resolve(),root/'experiments/spectral_reachability/d2p_prerun_source_seal.py']
    for path in files:e.raw(path)
    output.mkdir(exist_ok=False)
    seal=create_seal(files=files,args_json=io.canonical(dict(physical_sha256=PHYSICAL_SHA,prepare_sha256=PREPARE_SHA,
        rule='same-root complete paired H3 only; retain every partial; no runtime/neural work')),
        output_dir=output/'source_seal')
    report=audit(bundle);e.recheck()
    report.update(inputs_sha256=dict(sorted(e.hashes.items())),physical_report_sha256=PHYSICAL_SHA,
        source_seal=seal,source_verification=verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256']))
    write(output/'report.json',report)
    print(io.canonical(dict(status=report['status'],root_classifications=report['root_classifications'],
        comparison=report['comparison'],report_sha256=io.sha((output/'report.json').read_bytes()))))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run(args.project_root,args.output)
