"""Independent dense-bank saved audit. No producer import or model/simulator.

All actual paths AND external SHA values are mandatory; no pending outcome is
discovered or used as a default. Partial evidence never becomes a full-H2 pass.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_same_prefix_saved_audit_20261027_v1 as pure

NAME='p1_dense_suffix_saved_audit_20261027_v1'
ROOT=pure.ROOT
PRODUCER='p1_dense_suffix_physical_main_20261027_v1'
ROLES=('zero','bank_greedy','two_stage')
DIRECTIONS=((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1))
CAPS=dict(reset=204,step=1020,render=1122)
LIMIT=64*1024**2
require=pure.require


def bank():
    out=[dict(index=0,radius=0.,direction=None,logits=[0.]*10)]
    for radius in (.5,1.5):
        for direction in DIRECTIONS:
            unit=np.asarray(direction,dtype=float);unit=unit/np.linalg.norm(unit)
            x=(np.tile(unit,5)*radius/np.sqrt(5)).astype(np.float32)
            if np.linalg.norm(x.astype(float))>radius:
                x=x*np.float32(1-4*np.finfo(np.float32).eps)
            out.append(dict(index=len(out),radius=radius,direction=list(direction),logits=x.astype(float).tolist()))
    return out


def contract():
    return dict(roots=2,roles=list(ROLES),branches=102,suffixes=17,radii=[.5,1.5],directions=[list(x) for x in DIRECTIONS],
        support='same command direction at all5 suffix times; 10 independent coordinates remain legal',
        reset_cap=204,step_cap=1020,render_cap=1122,png_cap=306,
        order='original source slot,zero/bank_greedy/two_stage,zero then radius/direction',
        output_cap_bytes=LIMIT,floor_bytes=pure.FLOOR,neural_calls=0,training=False,original_Q_unchanged=True,
        fitted_controller=False,P1_admission=False,independent_samples=0,
        partial_policy='retain all102 planned branches; honor native termination/fresh RGB guard; no retries',
        limitation='new fixed diagnostic after sparse outcomes; not independent confirmation, not pure radius-controlled ablation, not global reachability')


def plans_check(p, parents, tasks, model_reports):
    require(p['schema']==PRODUCER+'.prepare' and p['contract']==contract(),'fixed dense protocol')
    plans=p['inputs']['plans'];require(len(plans)==102 and len(parents)==2 and {x['root_sha256'] for x in parents}==set(tasks),'all102/two distinct roots')
    roles=[]
    for i,old in enumerate(parents):
        task=tasks[old['root_sha256']];require(old['task_id']==pure.core.task_id(task),'parent task binding')
        for j,role in enumerate(pure.model_roles(task,model_reports[old['root_sha256']])):
            roles.append(dict(root_sha256=old['root_sha256'],task_id=old['task_id'],**role))
            for k,probe in enumerate(bank()):
                x=np.asarray(probe['logits'],np.float32)
                require(np.linalg.norm(x.astype(float))<=task['packet_logit_cap'],'original action ball')
                commands=np.concatenate((np.asarray(role['prefix'],np.float32),np.tanh(x).reshape(5,2)))
                expected=deepcopy(old)
                expected.update(arm=role['role']+'_dense_'+str(k),role=role['role'],bank_index=role['bank_index'],probe=probe,
                    commands=commands.astype(float).tolist(),prediction_sha256=None,origin='fixed_persistent_suffix_not_model_winner',
                    parent_selection_sha256=old['selection_sha256'],search_cost=dict(new_neural_calls=0,inherited_search_separate=True))
                expected.pop('selection_sha256');expected['selection_sha256']=pure.digest(expected)
                require(plans[51*i+17*j+k]==expected,'independent dense plan/selection SHA')
    pure.check_roles(p['inputs']['roles'],roles)


def row_check(plan,row,packet):
    require(all(row[k]==plan[k] for k in ('root_sha256','arm','selection_sha256')),'root/arm/selection join')
    require(row['status'] in ('completed','environment_terminated','failed','not_started'),'known branch status')
    if row['status']=='completed':
        pure.check_row(plan,row,packet);return
    if row['status']=='not_started':
        require(not row.get('trace') and not row.get('full_horizon') and not row.get('attempted_steps',0),'no execution hidden in not_started')
        return
    trace=row.get('trace',[]);attempted=row.get('attempted_steps',0);returned=row.get('returned_steps',0)
    require(type(attempted) is int and type(returned) is int and 0<=returned<=attempted<=10
        and len(trace)==attempted,'bounded partial dispatch/return census')
    for i,t in enumerate(trace,1):
        require(t['step']==i and t['command']==plan['commands'][i-1] and t['attempted'] is True,'partial raw command chronology')
        require(t['returned'] is (i<=returned),'partial return chronology')
        if 'actual_state' in t:
            metric=pure.metrics(packet['goal'],t['actual_state'])
            state=pure.vec(t['actual_state'])
            if 'observation' in t:require(t['observation']==dict(state=state.tolist(),proprio=state[[0,1,5,6]].tolist()),'partial returned/native observation')
            if 'official_info' in t:
                info=t['official_info']
                require(info['goal_state']==packet['goal'].tolist() and info['pos_agent']==state[:2].tolist()
                    and info['vel_agent']==state[5:].tolist() and info['block_pose'][:2]==state[2:4].tolist()
                    and info['block_pose'][2]%(2*np.pi)==state[4],'partial native info/goal/pose')
            if 'pose_diagnostics' in t:require(t['pose_diagnostics']==metric,'partial object/native arithmetic')
            if 'native_official_evaluation' in t:require(t['native_official_evaluation']==pure.official(metric),'partial native evaluation')
            if 'reward' in t:require(t['reward']==-metric['native_formula']['state_distance'],'partial native reward')
        if i<returned:require(not t.get('terminated') and not t.get('truncated'),'no dispatch after native stop')
    if row['status']=='environment_terminated':
        require(returned==attempted>0 and bool(trace[-1].get('terminated') or trace[-1].get('truncated')),'honor native termination')
    if row.get('full_horizon'):require(returned==10,'full_horizon requires ten returns, not comparable status')
    checkpoints=row.get('checkpoints',[])
    require([x['step'] for x in checkpoints]==sorted({x['step'] for x in checkpoints})
        and all(x['step'] in (0,5,10) and x['step']<=returned for x in checkpoints),'partial checkpoint census')
    for c in checkpoints:
        require(c['state']==(packet['state'].tolist() if c['step']==0 else trace[c['step']-1]['actual_state']),'partial checkpoint/native state')
        snapshot=row.get('body_dynamics',{}).get(str(c['step']))
        # Failure while saving a snapshot is a disclosed partial evidence gap,
        # not synthesized data. Complete rows require every snapshot above.
        if snapshot is not None:pure.dynamics(snapshot,c['state'])
    if row.get('initial_exact'):
        require(row['checkpoints'][0]['state']==packet['state'].tolist()
            and row['initial_pose_diagnostics']==pure.metrics(packet['goal'],packet['state']),'partial source initial binding')
    if row.get('final_state') is not None:
        require(trace and row['final_state']==trace[-1].get('actual_state'),'partial final/trace state')
        d=pure.metrics(packet['goal'],row['final_state'])
        require(row['pose_diagnostics']==d and row['official_evaluation']==pure.official(d),'partial endpoint metrics')
    g=row.get('guard_receipt')
    if g:
        require(g['root_id']==plan['root_sha256'] and g['P1_admission'] is False
            and [x['primitive_step'] for x in g['observations']]==list(range(len(g['observations'])))
            and g['primitive_commands_authorized']==attempted,'partial guard chronology')
        error=row.get('first_error') or {}
        if error.get('type')=='ObservationUnavailable' and 'canonical agent appearance absent' in error.get('message',''):
            require(g['abstained'] is True and len(g['observations'])==returned+1 and returned==attempted
                and g['observations'][-1]['canonical_agent_pixels']==0
                and all(x['canonical_agent_pixels']>0 for x in g['observations'][:-1]),'first absent RGB stops dispatch')


def full(row):return row.get('status')=='completed' and row.get('full_horizon') is True


def responses(plans,rows,goals):
    require(len(plans)==len(rows)==102,'all102 planned denominator')
    original=[];layers=[]
    for start in range(0,102,17):
        ps=plans[start:start+17];rs=rows[start:start+17];center=rs[0];root=ps[0]['root_sha256'];role=ps[0]['role']
        require([p['probe'] for p in ps]==bank() and all(p['role']==role and p['root_sha256']==root for p in ps),'dense bank and group identity')
        entries=[];partials=[];available5=[]
        for p,r in zip(ps,rs,strict=True):
            require(all(r[k]==p[k] for k in ('root_sha256','arm','selection_sha256')),'branch identity')
            cp=next((c for c in r.get('checkpoints',[]) if c['step']==5),None)
            if cp is not None and '5' in r.get('body_dynamics',{}):
                available5.append((cp,r['body_dynamics']['5']))
            entry=dict(probe_index=p['probe']['index'],status=r['status'],response=None)
            if full(r) and full(center):
                require(root in goals,'bound source goal for full comparison')
                d=pure.delta(r['final_state'],center['final_state']);goal=pure.vec(goals[root])
                entry['response']=dict(endpoint_delta_state7=d.tolist(),object_xy_delta=d[2:4].tolist(),wrapped_object_angle_delta=float(d[4]),
                    object_position_error_change=float(np.linalg.norm(pure.vec(r['final_state'])[2:4]-goal[2:4])-
                        np.linalg.norm(pure.vec(center['final_state'])[2:4]-goal[2:4])),
                    wrapped_angle_error_change=float(r['pose_diagnostics']['independent']['wrapped_angle_error']-
                        center['pose_diagnostics']['independent']['wrapped_angle_error']),native_success=bool(r['official_evaluation']['success']))
            if not full(r):
                state=r.get('final_state');partials.append(dict(probe_index=p['probe']['index'],status=r['status'],
                    attempted_steps=r.get('attempted_steps',0),returned_steps=r.get('returned_steps',0),
                    endpoint_metrics=None if state is None else pure.metrics(goals[root],state),
                    scope='partial/failed/not_started; never a full-H2 paired endpoint'))
            entries.append(entry)
        if available5:
            c0,b0=available5[0]
            require(all(c['state']==c0['state'] and c['rgb_sha256']==c0['rgb_sha256'] and b==b0 for c,b in available5),
                'step5 state/RGB/body equality including partials that reached step5')
        original.append(dict(root_sha256=root,role=role,planned=17,completed=sum(full(r) for r in rs),responses=entries))
        layers.append(dict(root_sha256=root,role=role,planned=17,full_completed=sum(full(r) for r in rs),
            partial_failed_not_started=len(partials),paired_available=sum(x['response'] is not None for x in entries),
            prefix5_records_checked=len(available5),missing_prefix5_records=sum(r.get('returned_steps',0)>=5 for r in rs)-len(available5),
            partial_layer=partials))
    return dict(groups=original,P1_admission=False,controller_selected=False,scope=contract()['limitation']),layers


def events(stream):
    previous=None
    for index,line in enumerate(stream):
        require(len(line)<=128*1024,'bounded durable journal event')
        e=json.loads(line);h=e.pop('event_sha256')
        require(e.get('index')==index and e.get('previous_sha256')==previous and pure.digest(e)==h,'journal index/chain/hash')
        previous=h
        yield {k:v for k,v in e.items() if k not in ('index','previous_sha256')}


def event_fields(e):
    kind=e['kind'];global_fields={'run_started':'prepare_sha256 contract','branch_closed':'receipt','run_finished':'status counts errors'}
    if kind in global_fields:expected={'kind',*global_fields[kind].split()}
    else:
        fields={'branch_started':'plan','reset_attempted':'attempt','reset_returned':'completed',
            'reset_observation':'seed options observation info','original_placement_request':'state goal',
            'render_attempted':'attempt','render_returned':'completed','fresh_rgb_observed':'receipt',
            'checkpoint_saved':'receipt','rgb_guard_observed':'step receipt','initial_identity':'checks',
            'initial_pose_diagnostics':'receipt','step_attempted':'receipt','step_returned':'step',
            'termination_returned':'step reward terminated truncated','returned_observation':'step observation',
            'actual_successor':'step state','official_info_returned':'step info','successor_evaluation':'step native diagnostics',
            'step_observation':'receipt','branch_failed':'first_error partial','branch_finished':'receipt'}
        require(kind in fields,'known event field schema')
        expected={'kind','selection_sha256',*fields[kind].split()}
        if kind=='fresh_rgb_observed' and e['receipt']['step'] in (0,5,10):expected.add('body_dynamics')
    require(set(e)==expected,'exact journal field schema '+kind)


def journal_check(stream,p,report,packets=None):
    """Independent event grammar, ordered operations, row/trace/body joins.

    Interrupted partial journals keep an explicit exact-count gap. Missing
    events never justify complete status or manufactured completed calls.
    """
    plans=p['inputs']['plans'];rows=report['rows'];iterator=iter(events(stream));event_count=0
    require(len(plans)==len(rows)==102 and set(report['counts'])==set(CAPS),'fixed planned/resource census')
    counts={k:dict(attempted=0,completed=0) for k in CAPS};closed=set();active=None;branch=None;stage=None
    starts=0;ended=False;last_step=0;render_step=0;failure=False;reset_no=0;placement_request=None
    recipes={}
    legal={'run_started','branch_started','reset_attempted','reset_returned','reset_observation','original_placement_request',
        'render_attempted','render_returned','fresh_rgb_observed','checkpoint_saved','rgb_guard_observed','initial_identity',
        'initial_pose_diagnostics','step_attempted','step_returned','termination_returned','returned_observation',
        'actual_successor','official_info_returned','successor_evaluation','step_observation','branch_failed','branch_finished','branch_closed','run_finished'}
    for e in iterator:
        event_count+=1;kind=e['kind'];require(kind in legal and not ended,'known event before terminal')
        event_fields(e)
        if kind=='run_started':
            require(event_count==1 and e==dict(kind=kind,prepare_sha256=report['prepare_sha256'],contract=p['contract']),'source-bound run start')
            continue
        require(event_count>1,'run_started before execution')
        if kind=='run_finished':
            require(active is None and e==dict(kind=kind,status=report['status'],counts=report['counts'],errors=report['errors']), 'terminal exact report binding')
            ended=True;continue
        if kind=='branch_started':
            require(active is None,'no interleaved branches');i=len(closed);require(i<102 and e['plan']==plans[i],'allslot ordered plan dispatch')
            active=i;branch=plans[i];stage='reset_attempted';starts+=1;last_step=0;render_step=0;failure=False;reset_no=0;placement_request=None
        elif kind=='branch_closed':
            i=len(closed);require(i<102 and e['receipt']==rows[i],'ordered exact durable branch receipt')
            if active is not None:
                require(active==i and (failure or stage=='closed'),'branch closes only after finish/failure')
            else:require(not full(rows[i]) and rows[i].get('attempted_steps',0)==0,'factory failure cannot hide physical calls')
            closed.add(i);active=None;branch=None;stage=None;continue
        else:require(active is not None,'event belongs to an active branch')
        require(e.get('selection_sha256')==branch['selection_sha256'],'per-event selection binding')
        row=rows[active]
        if kind=='branch_started':continue
        if kind in ('branch_failed','branch_finished'):
            if kind=='branch_failed':
                require(not full(row) and e['first_error']==row['first_error'],'real branch failure retained')
                require(all(row.get(k)==v for k,v in e['partial'].items()),'durable partial failure receipt')
                failure=True;stage='branch_finished'
            else:
                receipt=e['receipt'];expected={k:v for k,v in row.items() if k!='body_dynamics'}
                if full(row):require(receipt==expected and stage=='branch_finished','complete branch finished receipt/grammar')
                else:
                    require(all(receipt[k]==row[k] for k in ('root_sha256','arm','selection_sha256')),'partial finished identity')
                    # A later journal/close failure may change the final row.
                    failure=True
                stage='closed'
            continue
        require(not failure and kind==stage,'ordered event grammar '+kind+' expected '+str(stage))
        if kind.endswith('_attempted') and kind.split('_')[0] in CAPS:
            resource=kind.split('_')[0];counts[resource]['attempted']+=1
            if resource=='step':
                last_step+=1;require(last_step<=10 and e['receipt']==dict(step=last_step,command=branch['commands'][last_step-1],attempted=True,returned=False),'all10 commands ordered')
            else:require(e['attempt']==counts[resource]['attempted'],'global attempted count')
            stage=resource+'_returned'
        elif kind.endswith('_returned') and kind.split('_')[0] in CAPS:
            resource=kind.split('_')[0];counts[resource]['completed']+=1
            if resource=='step':require(e['step']==last_step,'returned command index');stage='termination_returned'
            else:
                require(e['completed']==counts[resource]['completed'],'global returned count')
                stage='reset_observation' if resource=='reset' else 'fresh_rgb_observed'
                if resource=='reset':reset_no+=1
        elif kind=='reset_observation':
            require(reset_no in (1,2),'two original recipe resets only')
            if packets is not None:require(e['info']['goal_state']==packets[branch['root_sha256']]['goal'].tolist(),'source-bound reset goal')
            # Per-branch reset number, not a changed source recipe.
            selected=branch['source_metadata']['selected_attempt'];sp=p['inputs']['source_plan']
            if reset_no==1:
                slot=next(s for s in sp['slots'] if s['candidate_id']==branch['candidate_id'])
                seed=slot['reset_seed_span'][0]+branch['source_metadata']['source']['proposal_index']*sp['reset_attempts_per_proposal']+selected['attempt_index']
                require(e['options'] is None and seed==selected['random_reset_seed'],'original source seed formula')
                require(e['seed']==selected['random_reset_seed'],'original random seed');stage='original_placement_request'
            else:
                require(e['seed']==sp['placement_reset_seed']==selected['placement_reset_seed'],'original placement seed')
                require(placement_request is not None and e['options']==dict(state=placement_request['state'],goal_state=placement_request['goal']),'exact requested placement options')
                require(e['options']['goal_state']==e['info']['goal_state'],'placement goal binding')
                if packets is not None:require(e['observation']['state']==packets[branch['root_sha256']]['state'].tolist(),'actual original placed state')
                stage='render_attempted'
        elif kind=='original_placement_request':
            placement_request={k:e[k] for k in ('state','goal')};pure.vec(e['state']);pure.vec(e['goal'])
            root=branch['root_sha256']
            if root in recipes:require(placement_request==recipes[root],'same root original placement recipe')
            recipes[root]=deepcopy(placement_request);stage='reset_attempted'
        elif kind=='fresh_rgb_observed':
            step=e['receipt']['step'];require(step==last_step,'render immediately after current command')
            if step in (0,5,10):
                c=next(x for x in row['checkpoints'] if x['step']==step)
                require(e['receipt']=={k:c[k] for k in ('step','state','rgb_sha256')}
                    and e.get('body_dynamics')==row.get('body_dynamics',{}).get(str(step)),'actual checkpoint dynamics receipt')
                stage='checkpoint_saved'
            else:
                require(e['receipt']['state']==row['trace'][step-1]['actual_state'],'fresh intermediate successor');stage='rgb_guard_observed'
            render_step=step
        elif kind=='checkpoint_saved':
            require(e['receipt']==next(x for x in row['checkpoints'] if x['step']==render_step),'exact PNG checkpoint receipt');stage='rgb_guard_observed'
        elif kind=='rgb_guard_observed':
            require(e['step']==render_step and e['receipt']['primitive_commands_authorized']==render_step
                and [x['primitive_step'] for x in e['receipt']['observations']]==list(range(render_step+1)),'fresh guard history')
            if full(row):
                g=deepcopy(row['guard_receipt']);g['observations']=g['observations'][:render_step+1];g['primitive_commands_authorized']=render_step
                require(e['receipt']==g,'exact saved guard')
            terminated=last_step>0 and bool(row['trace'][last_step-1].get('terminated') or row['trace'][last_step-1].get('truncated'))
            stage='initial_identity' if render_step==0 else 'branch_finished' if render_step==10 or terminated else 'step_attempted'
        elif kind=='initial_identity':require(e['checks']==row['initial_checks'],'initial identity checks');stage='initial_pose_diagnostics'
        elif kind=='initial_pose_diagnostics':require(e['receipt']==row['initial_pose_diagnostics'],'initial native/object pose');stage='step_attempted'
        elif kind=='termination_returned':
            t=row['trace'][last_step-1];require(all(e[k]==t[k] for k in ('step','reward','terminated','truncated')),'native stop/reward receipt');stage='returned_observation'
        elif kind=='returned_observation':require(e['observation']==row['trace'][last_step-1]['observation'] and e['step']==last_step,'actual observation');stage='actual_successor'
        elif kind=='actual_successor':require(e['state']==row['trace'][last_step-1]['actual_state'] and e['step']==last_step,'actual successor');stage='official_info_returned'
        elif kind=='official_info_returned':require(e['info']==row['trace'][last_step-1]['official_info'] and e['step']==last_step,'native info');stage='successor_evaluation'
        elif kind=='successor_evaluation':
            t=row['trace'][last_step-1];require(e['step']==last_step and e['native']==t['native_official_evaluation'] and e['diagnostics']==t['pose_diagnostics'],'native/object eval');stage='step_observation'
        elif kind=='step_observation':require(e['receipt']==row['trace'][last_step-1],'exact full step receipt');stage='render_attempted'
    for resource,cap in CAPS.items():
        top=report['counts'][resource];observed=counts[resource]
        require(all(type(top[k]) is int and observed[k]<=top[k]<=cap for k in ('attempted','completed'))
            and top['completed']<=top['attempted'],'partial counts bounded by journal and cap')
    complete=report['status']=='persistent_suffix_interventions_completed'
    if complete:
        require(ended and len(closed)==starts==102 and counts==report['counts']=={k:dict(attempted=v,completed=v) for k,v in CAPS.items()}
            and event_count==14180,'complete102 exact journal/calls')
    else:
        require(report['status']=='partial_or_failed','known partial report status')
        if active is not None:require(not full(rows[active]),'cannot lose complete branch journal')
        require(all(not full(rows[i]) for i in range(len(closed),102)),'unclosed slots cannot be full completed')
    gaps=[]
    if counts!=report['counts']:gaps.append('partial counters include operations whose durable events are missing; exact accounting not certified')
    if not ended:gaps.append('durable run_finished unavailable')
    row_steps={k:sum(r.get(k+'_steps',0) for r in rows) for k in ('attempted','returned')}
    require(row_steps['attempted']==report['counts']['step']['attempted'] and row_steps['returned']==report['counts']['step']['completed'],
        'all102 row step counters equal global step counters')
    return dict(events=event_count,observed_counts=counts,reported_counts=report['counts'],row_step_totals=row_steps,
        closed_branches=len(closed),counts_exact_from_durable_events=counts==report['counts'],evidence_gaps=gaps)


def guard_diagnosis(plan,row,pixel_counts):
    error=row.get('first_error') or {}
    if error.get('type')!='ObservationUnavailable':return None
    g=row.get('guard_receipt',{});observations=g.get('observations',[])
    require(observations and observations[-1]['canonical_agent_pixels']==0 and g['abstained'] is True,'actual missing-appearance guard')
    step=observations[-1]['primitive_step'];require(row['returned_steps']==row['attempted_steps']==step,'no blind continuation')
    x,y=pure.vec(row['final_state'])[:2];present=step in pixel_counts
    if present:require(pixel_counts[step]==0,'failed PNG confirms absent canonical appearance')
    left=bool(x<0);outside=bool(x<0 or x>512 or y<0 or y>512)
    return dict(root_sha256=plan['root_sha256'],role=plan['role'],probe_index=plan['probe']['index'],
        failure_step=step,agent_center_xy=[float(x),float(y)],guard_canonical_pixels=0,
        boundary_evidence='left of native canvas' if left else 'center outside native canvas' if outside else 'center inside; boundary-only explanation unproven',
        entire_default_agent_circle_left_of_canvas=bool(x+15<0),
        radius_basis='source default circular agent scale40 * .375 =15; not a new geometric measurement',
        failed_frame_PNG_retained=present,failed_frame_pixel_check='canonical color absent verified from saved PNG' if present else 'unknown: failed frame pixels not retained',
        occlusion_vs_color_change='unknown; not separately identifiable from receipts/state',
        scope='boundary supported by actual state, not a unique visual-cause classifier; guard unchanged')


def load(reader,prepared,prepare_sha,report_path,report_sha):
    for h in (prepare_sha,report_sha):pure.core.hash_value(h)
    p=reader.read(prepared,prepare_sha);r=reader.read(report_path,report_sha)
    require(r['schema']==PRODUCER+'.report' and r['prepare_sha256']==prepare_sha and r['contract']==p['contract'],'actual source/prepare binding')
    require(r['new_neural_calls']==r['new_independent_samples']==0 and r['P1_admission'] is False,'no neural/confirmation/admission')
    require(len(r['rows'])==102,'all102 actual/partial/not_started rows')
    for f,h in {**p['sources'],**p['inputs']['bindings'],**p['inputs']['source_plan']['source_closure']}.items():reader.bind(f,h)
    parent=reader.read(ROOT/'results/p1_witness_physical_20261027_v1_prepared01/prepare.json')
    args=parent['inputs']['arguments'];wp=Path(args['witness_prepared']);receipt=reader.read(wp/'prepare.json',args['witness_prepare_sha'])
    protocol=reader.read(wp/'protocol.json',receipt['protocol_sha256'])
    for f,h in {**protocol['sources_sha256'],**protocol['inputs']['bindings'],**protocol['inputs']['model_bindings']}.items():reader.bind(f,h)
    tasks={v['root_sha256']:v['task'] for v in protocol['inputs']['tasks']};parents=[v for v in parent['inputs']['plans'] if v['arm']=='two_stage']
    model_reports={v['root_sha256']:reader.read(v['origin']) for v in parents};plans_check(p,parents,tasks,model_reports)
    packets={}
    for old in parents:
        f=reader.bind(old['raw_source_file'],old['raw_source_sha256'])
        with np.load(f,allow_pickle=False) as z:packet={k:z[k].copy() for k in z.files}
        metadata=json.loads(str(packet.pop('metadata_json')))
        require(metadata==old['source_metadata'] and metadata['plan_id']==pure.digest(p['inputs']['source_plan']),'raw metadata/root recipe')
        import hashlib
        root=hashlib.sha256(np.ascontiguousarray(np.concatenate((pure.vec(packet['state']),pure.vec(packet['goal']))),dtype=np.float64).tobytes()).hexdigest()
        require(root==old['root_sha256'],'raw actual root');packets[root]=packet
    return p,r,packets


def audit(prepared,prepare_sha,report_path,report_sha):
    reader=pure.Reader();p,r,packets=load(reader,prepared,prepare_sha,report_path,report_sha);folder=Path(report_path).absolute().parent
    require(r['status'] in ('persistent_suffix_interventions_completed','partial_or_failed'),'known actual completion state')
    if r['status']=='persistent_suffix_interventions_completed':require(not r['errors'] and all(full(x) for x in r['rows']),'complete status requires every full row/no errors')
    pngs=set();gaps=[];guard_failures=[]
    from PIL import Image
    import hashlib
    for plan,row in zip(p['inputs']['plans'],r['rows'],strict=True):
        packet=packets[plan['root_sha256']];row_check(plan,row,packet)
        pixel_counts={}
        for c in row.get('checkpoints',[]):
            if not all(k in c for k in ('path','file_sha256','rgb_sha256')):
                require(not full(row),'complete checkpoint requires saved image');gaps.append('partial checkpoint PNG save incomplete: '+plan['arm']);continue
            name=f"{plan['candidate_id']}_{plan['arm']}_{c['step']}.png";require(Path(c['path'])==folder/name,'exact same-run PNG path');pngs.add(name)
            f=reader.bind(c['path'],c['file_sha256'])
            with Image.open(f) as image:rgb=np.asarray(image)
            require(rgb.shape==(224,224,3) and rgb.dtype==np.uint8 and hashlib.sha256(rgb.tobytes()).hexdigest()==c['rgb_sha256'],'PNG pixels bound')
            pixel_counts[c['step']]=int(np.count_nonzero(np.all(rgb==[65,105,225],axis=-1)))
            observations=row.get('guard_receipt',{}).get('observations',[])
            if c['step']<len(observations):require(pixel_counts[c['step']]==observations[c['step']]['canonical_agent_pixels'],'PNG/guard actual count')
            if c['step']==0:require(np.array_equal(rgb,packet['rgb_history'][0]),'source initial RGB exact')
        if row.get('returned_steps',0)>=5 and '5' not in row.get('body_dynamics',{}):gaps.append('partial missing step5 body snapshot: '+plan['arm'])
        diagnosed=guard_diagnosis(plan,row,pixel_counts)
        if diagnosed is not None:guard_failures.append(diagnosed)
    require(len(pngs)<=306 and sum(f.stat().st_size for f in folder.iterdir() if f.is_file())<=LIMIT,'306PNG/64MiB cap')
    extras=sorted({f.name for f in folder.iterdir()}-pngs-{'report.json','analysis.json','journal.jsonl'})
    if extras:gaps.append('unreferenced partial artifacts preserved: '+','.join(extras))
    j=reader.bind(folder/'journal.jsonl',r['journal_sha256'])
    with j.open() as f:ledger=journal_check(f,p,r,packets)
    original,layers=responses(p['inputs']['plans'],r['rows'],{root:x['goal'] for root,x in packets.items()})
    if (folder/'analysis.json').exists():require(reader.read(folder/'analysis.json')==original,'independent all102 response arithmetic')
    else:
        require(r['status']=='partial_or_failed','complete analysis mandatory');gaps.append('producer analysis absent in partial run')
    if r['status']=='persistent_suffix_interventions_completed':
        require(len(pngs)==306 and {f.name for f in folder.iterdir()}==pngs|{'report.json','analysis.json','journal.jsonl'},'complete309 artifact census')
    summaries=[]
    for g in original['groups']:
        v=[x['response'] for x in g['responses'] if x['response'] is not None]
        summaries.append(dict(root_sha256=g['root_sha256'],role=g['role'],paired_fullH2=len(v),
            object_displacement_max=None if not v else float(max(np.linalg.norm(x['object_xy_delta']) for x in v)),
            object_angle_displacement_max=None if not v else float(max(abs(x['wrapped_object_angle_delta']) for x in v)),
            native_successes=sum(x['native_success'] for x in v),responses_sha256=pure.digest(g['responses'])))
    reader.recheck()
    return dict(schema=NAME+'.report',status='saved_dense_complete_audited_not_admitted' if r['status']=='persistent_suffix_interventions_completed' else 'saved_dense_partial_audited_not_admitted',
        input_report=str(Path(report_path).absolute()),input_report_sha256=report_sha,prepare_sha256=prepare_sha,
        planned=102,full_completed=sum(full(x) for x in r['rows']),PNGs=len(pngs),PNG_cap=306,ledger=ledger,layers=layers,groups=summaries,
        guard_failures=guard_failures,
        responses_sha256=pure.digest(original),evidence_gaps=gaps+ledger['evidence_gaps'],bindings_sha256=pure.digest(reader.bindings),
        new_neural_calls=0,new_physics_steps=0,P1_admission=False,controller_selected=False,
        limitations=['fixed diagnostic after sparse outcomes, not independent confirmation or pure amplitude ablation',
            'recorded state/body/RGB, not solver cache serialization or independent physics rerun',
            'partial counters with missing durable events are explicitly not exact',
            'zero recorded finite response is not global uncontrollability; no physical winner deployed'],
        sources={str(f):pure.sha(f) for f in (Path(__file__),Path(pure.__file__),Path(pure.core.__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'))})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('prepared','prepare-sha','report','report-sha','output'):parser.add_argument('--'+key,required=True)
    a=parser.parse_args();output=Path(a.output);require(not output.exists(),'exclusive new output')
    result=audit(a.prepared,a.prepare_sha,a.report,a.report_sha);raw=(pure.core.canonical(result)+'\n').encode()
    require(len(raw)<=128*1024 and shutil.disk_usage(output.parent).free>=pure.FLOOR+len(raw),'small output plus200MiB floor')
    with output.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    print(pure.core.canonical(dict(status=result['status'],output=str(output),sha256=pure.sha(output),bytes=len(raw))))


if __name__=='__main__':main()
