"""Independent saved physical receipts/PNG audit; NumPy/Pillow, no simulator/model.

No physical producer imports and no producer verify/analyze calls. The task core
is reused ONLY for its literal task-id and saved model prediction contract.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np
from PIL import Image

from experiments.spectral_reachability import p1_conditional_task_admission_20261027_v1 as core

NAME='p1_line_search_gn_physical_saved_audit_20261027_v1'
ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'results/p1_line_search_gn_physical_20261027_v1_main01/report.json'
INPUT_SHA='3a41fe83f375e5bb3f390e9f3a0e8e17dd70c2d49a7ebe3a99079f5774de85c1'
PREPARED=ROOT/'results/p1_line_search_gn_physical_20261027_v1_prepare02.json'
PREPARE_SHA='f36f915ccfc2687da056810d1241da433190eff430bb54fa7b7b35a6af3818d5'
ARM='line_search_GN'
LIMIT=32*1024**2


def require(value,message):
    if not value:raise ValueError(message)


def canonical(value):return json.dumps(value,sort_keys=True,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'regular nonsymlink file: '+str(path))
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def raw(path):
    require(Path(path).stat().st_size<=LIMIT,'bounded input')
    data=Path(path).read_bytes()
    if str(path).endswith('.gz'):
        d=zlib.decompressobj(31);data=d.decompress(data,LIMIT+1)
        require(d.eof and not d.unused_data and not d.unconsumed_tail,'complete single gzip stream')
    require(len(data)<=LIMIT,'bounded expanded input');return data


def read(path):return json.loads(raw(path))


def vector(value):
    a=np.asarray(value)
    require(a.shape==(7,) and a.dtype.kind in 'fi' and np.isfinite(a).all(),'finite state7')
    return a.astype(float)


def root_hash(state,goal):
    return hashlib.sha256(np.ascontiguousarray(np.concatenate((vector(state),vector(goal))),dtype=np.float64).tobytes()).hexdigest()


def same(actual,expected,label):
    """Strict field/shape/boolean identity; float receipt tolerance only 1e-12."""
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),label+' fields')
        for k in expected:same(actual[k],expected[k],label+'.'+k)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),label+' length')
        for i,(a,b) in enumerate(zip(actual,expected)):same(a,b,label+f'[{i}]')
    elif type(expected) is float:
        require(type(actual) in (int,float) and np.isfinite(actual) and np.isclose(actual,expected,rtol=1e-12,atol=1e-12),label+' numeric')
    else:require(type(actual) is type(expected) and actual==expected,label+' value')


def native(goal,state):
    delta=vector(state)-vector(goal);absolute=abs(float(delta[4]))
    angle=min(absolute,2*np.pi-absolute)
    return dict(success=bool(np.linalg.norm(delta[:4])<20 and angle<np.pi/9),state_distance=float(np.linalg.norm(delta)))


def diagnostic(goal,state):
    d=vector(state)-vector(goal);absolute=float(abs(d[4]));angle=float(min(absolute,2*np.pi-absolute))
    wrapped=float(abs(np.arctan2(np.sin(d[4]),np.cos(d[4]))));allxy=float(np.linalg.norm(d[:4]))
    ns=bool(allxy<20 and angle<np.pi/9);ws=bool(allxy<20 and wrapped<np.pi/9)
    wd=d.copy();wd[4]=float(np.arctan2(np.sin(d[4]),np.cos(d[4])))
    return dict(native_formula=dict(success=ns,raw_angle_difference=absolute,angle_term=angle,
        state_distance=float(np.linalg.norm(d)),negative_angle_term=angle<0),
        independent=dict(wrapped_angle_error=wrapped,all_xy_position_error=allxy,
            object_position_error=float(np.linalg.norm(d[2:4])),agent_position_error=float(np.linalg.norm(d[:2])),
            wrapped_state_delta=wd.tolist(),wrapped_state_distance=float(np.linalg.norm(wd)),
            same_allxy_threshold_wrapped_success=ws,label='diagnostic_only_not_official_or_admission'),
        native_vs_wrapped_success_disagreement=ns!=ws)


def object_metric(goal,state):
    state=vector(state);d=state-vector(goal)
    return dict(object_position_error=float(np.linalg.norm(d[2:4])),
        wrapped_object_angle_error=float(abs(np.arctan2(np.sin(d[4]),np.cos(d[4])))),
        object_pose=state[2:5].tolist(),success=None,label='object_only_diagnostic_no_authorized_success_threshold')


def npz(path):
    # Bound decompressed arrays before NumPy allocation; pickle is never allowed.
    import zipfile
    with zipfile.ZipFile(path) as z:
        require(sum(i.file_size for i in z.infolist())<=4*1024**2,'bounded raw-source arrays')
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def load_bundle(report=INPUT,expected=INPUT_SHA,prepared=PREPARED,prepare_sha=PREPARE_SHA):
    bindings={}
    def bound(path,h):
        path=Path(path).absolute();require(sha(path)==h,'SHA mismatch: '+str(path))
        require(str(path) not in bindings or bindings[str(path)]==h,'conflicting binding')
        bindings[str(path)]=h;return path
    report=bound(report,expected);p=read(bound(prepared,prepare_sha));r=read(report)
    require(p['schema']=='p1_line_search_gn_physical_20261027_v1.prepare','physical prepare schema')
    for path,h in p['sources_sha256'].items():bound(path,h)
    a=p['arguments'];mp=read(bound(a['search_prepared'],a['search_prepare_sha']));mr=read(bound(a['search_report'],a['search_report_sha']))
    # Bind the original observed-source prepare, then its independently recorded
    # reset audit/request/sourcepack. No producer material reconstruction calls.
    op_path=next(k for k in mp['bindings_sha256'] if Path(k).name=='p1_observed_history_source_20261027_v1_prepare01.json')
    op=read(bound(op_path,mp['bindings_sha256'][op_path]));oi=op['inputs']
    pack=read(bound(oi['report'],oi['report_sha256']));request=read(bound(oi['request'],oi['request_sha256']))
    reset=read(bound(oi['reset_audit_report'],oi['reset_audit_sha256']))
    sp=pack['plan'];require(sp==request['plan'] and digest(sp)==p['source_plan_sha256'],'original frozen source plan')
    for path,h in sp['source_closure'].items():bound(path,h)
    require(len(p['plans'])==len(mp['roots'])==2,'fixed two-root cohort')
    packets=[];observed=[];tasks=[];winners=[]
    for plan in p['plans']:
        packet=npz(bound(plan['raw_source_file'],plan['raw_source_sha256']));packets.append(packet)
        tasks.append(read(bound(plan['task_file'],plan['task_file_sha256']))['task'])
        require(mp['bindings_sha256'].get(plan['task_file'])==plan['task_file_sha256'],'original search task hash')
        origin=Path(plan['origin']);require(origin.parent==Path(a['search_report']).parent and
            mr['artifact_sha256'].get(origin.name)==plan['origin_sha256'],'actual CUDA winner artifact')
        winners.append(read(bound(origin,plan['origin_sha256'])))
        observed_path=next(k for k in mp['bindings_sha256'] if '/p1_observed_history_source_20261027_v1_main01/' in k and
            Path(k).name==f'source_{plan["candidate_id"]}.npz')
        observed.append(npz(bound(observed_path,mp['bindings_sha256'][observed_path])))
    folder=report.parent;journal=raw(bound(folder/'journal.jsonl',r['journal_sha256']))
    require(journal.endswith(b'\n'),'no torn journal suffix')
    events=[json.loads(line) for line in journal.splitlines()]
    images={};files={'report.json','journal.jsonl'}
    for row in r['rows']:
        for c in row['checkpoints']:
            path=Path(c['path']);path=path if path.is_absolute() else ROOT/path
            require(path.parent.absolute()==folder.absolute(),'checkpoint in actual output directory')
            bound(path,c['file_sha256']);files.add(path.name)
            with Image.open(path) as im:
                require(im.mode=='RGB' and im.size==(224,224),'native RGB PNG format')
                images[path.name]=np.asarray(im).copy()
    require({x.name for x in folder.iterdir()}==files and len(images)==6,'exact report/journal/six PNG census')
    require(sum(x.stat().st_size for x in folder.iterdir())<=2*1024**2,'actual2MiB output cap')
    return dict(prepared=p,report=r,model_prepare=mp,model_report=mr,source_pack=pack,reset_audit=reset,
        packets=packets,observed=observed,tasks=tasks,winners=winners,events=events,images=images,
        bindings=bindings,expected_prepare_sha=prepare_sha)


class Cursor:
    def __init__(self,events):
        self.events=events;self.index=0;self.sid=None
        previous=None
        for i,event in enumerate(events):
            require(event['index']==i and event['previous_sha256']==previous,'ordered hashchain index/link')
            require(event['event_sha256']==digest({k:v for k,v in event.items() if k!='event_sha256'}),'hashchain event content')
            previous=event['event_sha256']
    def take(self,kind):
        require(self.index<len(self.events),'truncated physical journal')
        e=self.events[self.index];self.index+=1
        require(e['kind']==kind,'ordered physical event: '+kind)
        payload={k:v for k,v in e.items() if k not in ('index','previous_sha256','event_sha256')}
        if kind not in ('run_started','branch_closed','run_finished'):
            require(payload.pop('selection_sha256')==self.sid,'physical selection lineage')
        return payload
    def expect(self,kind,**values):
        e=self.take(kind);require(e==dict(kind=kind,**values),'exact physical receipt: '+kind);return e


def audit(b,*,recheck_bindings=True):
    p,r=b['prepared'],b['report'];plans=p['plans'];sp=b['source_pack']['plan']
    pack,reset=b['source_pack'],b['reset_audit']
    require(pack['status']=='completed_calibration_source_only' and pack['first_error'] is None and
        pack['secondary_errors']==[] and pack['durable_journal_complete'] is True and
        pack['physical_future_labels_read'] is False and pack['P1_admission'] is False,'completed source-only parent')
    require(reset['status']=='diagnostic_completed' and reset['all_reproduction_checks'] is True and
        reset['first_error'] is None and reset['secondary_errors']==[],'completed earlier reset diagnostic')
    require(r['schema']=='p1_line_search_gn_physical_20261027_v1.report' and r['status']=='physical_calibration_recorded' and
        r['prepare_sha256']==b['expected_prepare_sha'] and r['errors']==[],'actual completed physical report')
    require(r['contract']==p['contract'] and r['inherited']==p['inherited'],'frozen contract/inherited counts')
    require(all(r[k] is False for k in ('P1_admission','model_task_eligible','stall_certified')) and
        r['new_independent_samples']==r['neural_calls']==0 and r['global_freshness']=='unknown','calibration scope')
    require(len(plans)==len(r['rows'])==2 and len({x['root_sha256'] for x in plans})==2,'two unique branches')
    mp,mr=b['model_prepare'],b['model_report']
    require(mr['status']=='cuda_line_search_gn_completed' and mr['errors']==[] and mr['prepare_sha256']==p['arguments']['search_prepare_sha'] and
        mr['model_hash_before']==mr['model_hash_after']==mp['expected_model_hash'] and mr['source_verified'] is True and
        mr['model_unchanged'] is True,'original CUDA winner report binding only; no model reload')
    c=Cursor(b['events']);c.expect('run_started',prepare_sha256=b['expected_prepare_sha'],contract=p['contract'])
    counts={k:dict(attempted=0,completed=0) for k in ('reset','render','step')};summaries=[]
    for i,(plan,row,packet,obs,task,winner) in enumerate(zip(plans,r['rows'],b['packets'],b['observed'],b['tasks'],b['winners'],strict=True)):
        sid=plan['selection_sha256'];c.sid=sid
        require(digest({k:v for k,v in plan.items() if k!='selection_sha256'})==sid,'selected physical plan digest')
        require(plan['arm']==row['arm']==ARM and row['selection_sha256']==sid and row['root_sha256']==plan['root_sha256'],'fixed row identity')
        require(row['status']=='completed' and row['full_horizon'] is True and row['initial_exact'] is True and
            row['first_error'] is None and row['secondary_errors']==[],'complete error-free branch')
        core.validate_task(task);core.check_prediction(task,winner['final_prediction'])
        require(plan['task_id']==core.task_id(task)==winner['task_id']==mp['roots'][i]['task_id'] and
            winner['root_sha256']==plan['root_sha256']==mp['roots'][i]['root_sha256'] and winner['root_index']==i,'literal task/root/winner join')
        require(plan['commands']==winner['final_prediction']['commands'] and plan['prediction_sha256']==digest(winner['final_prediction']) and
            plan['steps_sha256']==digest(winner['steps']),'exact final CUDA winner, no physical selection')
        require(plan['goal_Q_context_sha256']==digest({k:task[k] for k in ('goal','projection','root_context','action_mean','action_std','model_sha256')}) and
            np.array_equal(task['projection'],np.eye(192)),'same goal/native192 identity Q')
        commands=np.asarray(plan['commands']);require(commands.shape==(10,2) and np.isfinite(commands).all() and
            (abs(commands)<=1).all() and np.array_equal(commands,commands.astype(np.float32)),'exact float32 raw10x2')
        meta=json.loads(str(packet['metadata_json']));goal=packet['goal'];state=packet['state']
        require(meta==plan['source_metadata'] and meta['plan_id']==digest(sp) and
            meta['root_sha256']==root_hash(state,goal)==plan['root_sha256'],'raw source identity')
        parent=pack['slots'][i]
        require(parent['slot']['candidate_id']==plan['candidate_id'] and parent['metadata']==meta and
            parent['raw_source_sha256']==plan['raw_source_sha256'] and parent['raw_source_file']==Path(plan['raw_source_file']).name,'original sourcepack row/raw binding')
        require(plan['observed_equals_parent'] is True and plan['history_semantics']=='observed_not_padding_even_when_numerically_identical','observed stationary bridge scope')
        require(np.array_equal(obs['goal'],goal) and np.array_equal(obs['state'],state) and
            np.array_equal(obs['goal_rgb'],packet['goal_rgb']) and np.array_equal(obs['rgb_history'],packet['rgb_history']) and
            np.array_equal(obs['observed_states'],np.stack([state]*3)) and np.array_equal(obs['observed_public_step_indices'],[0,5,10]) and
            np.array_equal(obs['raw_action_history'],packet['raw_action_history']) and not np.any(packet['raw_action_history']), 'actual observed0/5/10 equals parent; not new sample')
        tokens=(obs['raw_action_history'].astype(np.float32).reshape(2,10)-np.tile(np.asarray(task['action_mean'],np.float32),5))/np.tile(np.asarray(task['action_std'],np.float32),5)
        require(np.array_equal(tokens,obs['normalized_action_history']) and np.array_equal(tokens,task['root_context']['action_history']) and
            np.array_equal(obs['raw_action_history'],task['root_raw_action_history']),'observed raw/normalized task history')
        slot=sp['slots'][i];proposal=meta['source']['proposal_index'];attempt=meta['selected_attempt']['attempt_index']
        require(slot['candidate_id']==plan['candidate_id'] and slot['root_seed']==meta['source']['root_seed'] and
            0<=proposal<sp['proposal_limit_per_seed'] and 0<=attempt<sp['reset_attempts_per_proposal'],'original finite source coordinates')
        seed=slot['reset_seed_span'][0]+proposal*sp['reset_attempts_per_proposal']+attempt
        require(seed==meta['selected_attempt']['random_reset_seed'] and sp['placement_reset_seed']==meta['selected_attempt']['placement_reset_seed'],'original reset seeds')
        require(meta['source']['reset_recipe_sha256']==digest({k:sp[k] for k in ('proposal_limit_per_seed','reset_attempts_per_proposal','placement_reset_seed','recipe','domain','history','rgb_shape')}),'original placement recipe SHA')
        c.expect('branch_started',plan=plan)
        def operation(kind):
            counts[kind]['attempted']+=1;c.expect(kind+'_attempted',attempt=counts[kind]['attempted'])
            counts[kind]['completed']+=1;c.expect(kind+'_returned',completed=counts[kind]['completed'])
        operation('reset');random_reset=c.take('reset_observation')
        require(random_reset['seed']==seed and random_reset['options'] is None and random_reset['info']['goal_state']==goal.tolist(),'random reset obtains original native goal')
        placement=c.take('original_placement_request');require(placement['goal']==goal.tolist(),'same placement goal')
        references=[x for x in b['reset_audit']['rows'] if x['selected'] is True and x['candidate_id']==slot['candidate_id'] and
            x['proposal_index']==proposal and x['attempt_index']==attempt]
        require(len(references)==1 and references[0]['placement_requested_state']==placement['state'] and
            references[0]['random_reset_seed']==seed and references[0]['placement_reset_seed']==sp['placement_reset_seed'],'full pusher request matches SHA-bound earlier actual reset-audit receipt')
        rng=np.random.default_rng(np.random.SeedSequence([slot['root_seed'],proposal,attempt]));recipe=sp['recipe']
        object_xy=vector(goal)[2:4]+rng.uniform(-recipe['position_jitter'],recipe['position_jitter'],2)
        angle=float(goal[4]+rng.uniform(*recipe['angle_offset_range']))
        require(np.array_equal(object_xy,placement['state'][2:4]) and angle==placement['state'][4] and placement['state'][5:]==[0.,0.],'independent seed/recipe object pose arithmetic')
        operation('reset');placed=c.take('reset_observation')
        require(placed['seed']==sp['placement_reset_seed'] and placed['options']==dict(state=placement['state'],goal_state=goal.tolist()) and
            placed['observation']['state']==state.tolist() and placed['info']['goal_state']==goal.tolist(),'literal requested reset inputs and actual outputs; no saved-state reset')
        observed_pixels=[];png_receipts=[]
        def guard(step):
            return dict(root_id=plan['root_sha256'],observations=deepcopy(observed_pixels),primitive_commands_authorized=step,
                abstained=False,scope='canonical appearance presence only; not full observability or prevention guarantee',P1_admission=False)
        def capture(step,current):
            operation('render');fresh=c.take('fresh_rgb_observed')['receipt']
            require(fresh['step']==step and fresh['state']==current,'fresh render binds actual state')
            if step in (0,5,10):
                cp=row['checkpoints'][len(png_receipts)];c.expect('checkpoint_saved',receipt=cp)
                require(cp['step']==step and cp['state']==current and cp['rgb_sha256']==fresh['rgb_sha256'],'0/5/10 checkpoint state/RGB')
                image=b['images'][Path(cp['path']).name]
                require(image.dtype==np.uint8 and image.shape==(224,224,3) and hashlib.sha256(image.tobytes()).hexdigest()==cp['rgb_sha256'],'decoded PNG pixel SHA')
                pixel_count=int(np.count_nonzero(np.all(image==[65,105,225],axis=-1)))
                require(pixel_count>0,'checkpoint canonical pusher pixels')
                if step==0:require(np.array_equal(image,packet['rgb_history'][0]),'actual initial PNG equals original RGB')
                png_receipts.append(cp)
            else:pixel_count=row['guard_receipt']['observations'][step]['canonical_agent_pixels']
            require(type(pixel_count) is int and pixel_count>0,'positive recorded intermediate guard count (no saved PNG there)')
            observed_pixels.append(dict(primitive_step=step,canonical_agent_pixels=pixel_count))
            c.expect('rgb_guard_observed',step=step,receipt=guard(step))
        capture(0,state.tolist())
        initial_checks={k:True for k in ('state','goal','rgb','goal_rgb','root','state_at_render')}
        require(row['initial_checks']==initial_checks,'all initial identity checks')
        c.expect('initial_identity',checks=initial_checks)
        same(row['initial_pose_diagnostics'],diagnostic(goal,state),'initial diagnostics')
        c.expect('initial_pose_diagnostics',receipt=row['initial_pose_diagnostics'])
        require(len(row['trace'])==10 and row['attempted_steps']==row['returned_steps']==10,'ten commands exactly once')
        for step,entry in enumerate(row['trace'],1):
            require(entry['step']==step and entry['command']==commands[step-1].tolist() and entry['attempted'] is True and entry['returned'] is True,'exact winner raw dispatch/order')
            counts['step']['attempted']+=1;c.expect('step_attempted',receipt=dict(step=step,command=entry['command'],attempted=True,returned=False))
            counts['step']['completed']+=1;c.expect('step_returned',step=step)
            require(entry['terminated'] is False and entry['truncated'] is False,'complete branch never ignores terminal')
            c.expect('termination_returned',step=step,reward=entry['reward'],terminated=False,truncated=False)
            c.expect('returned_observation',step=step,observation=entry['observation'])
            c.expect('actual_successor',step=step,state=entry['actual_state'])
            require(entry['observation']['state']==entry['actual_state'],'returned/actual state equality')
            c.expect('official_info_returned',step=step,info=entry['official_info'])
            require(entry['official_info']['goal_state']==goal.tolist(),'actual native goal_state not static goal_pose')
            info=entry['official_info'];actual=vector(entry['actual_state'])
            require(np.array_equal(info['pos_agent'],actual[:2]) and np.array_equal(info['vel_agent'],actual[5:]) and
                np.array_equal(info['block_pose'][:2],actual[2:4]) and float(info['block_pose'][2]%(2*np.pi))==actual[4], 'native info unwrapped block pose/proprio state mapping')
            require(type(info['n_contacts']) is int and info['n_contacts']>=0,'recorded nonnegative native contact count')
            expected_native=native(goal,entry['actual_state']);expected_diag=diagnostic(goal,entry['actual_state'])
            same(entry['native_official_evaluation'],expected_native,'native successor');same(entry['pose_diagnostics'],expected_diag,'wrapped successor')
            require(np.isclose(entry['reward'],-expected_native['state_distance'],rtol=1e-12,atol=1e-12),'native negative distance reward')
            risk=dict(native_vs_wrapped_success_disagreement=expected_diag['native_vs_wrapped_success_disagreement'],
                negative_native_angle_term=expected_diag['native_formula']['negative_angle_term'],native_eval_vs_formula_disagreement=False,
                terminated_vs_eval_disagreement=bool(entry['terminated'])!=expected_native['success'],policy='honor_returned_terminated_truncated_no_override')
            same(entry['termination_protocol_risk'],risk,'native terminal risk flags')
            c.expect('successor_evaluation',step=step,native=entry['native_official_evaluation'],diagnostics=entry['pose_diagnostics'])
            c.expect('step_observation',receipt=entry);capture(step,entry['actual_state'])
        require(len(row['checkpoints'])==3 and row['final_state']==row['trace'][-1]['actual_state'] and
            row['official_info']==row['trace'][-1]['official_info'] and row['pose_diagnostics']==row['trace'][-1]['pose_diagnostics'],'final report/trace/checkpoints')
        same(row['official_evaluation'],native(goal,row['final_state']),'native final')
        wd=vector(row['final_state'])-goal;wd[4]=(wd[4]+np.pi)%(2*np.pi)-np.pi
        same(row['wrapped_state_delta'],wd.tolist(),'final wrapped state delta')
        require(row['guard_receipt']==guard(10) and row['endpoint_scope']=='full_H2' and row['endpoint_rgb_fresh'] is True,'complete guarded endpoint')
        before=deepcopy(row);before.pop('object_only');c.expect('branch_finished',receipt=before)
        metrics=[]
        for cp in row['checkpoints']:
            metrics.append(dict(step=cp['step'],state=cp['state'],rgb_sha256=cp['rgb_sha256'],file_sha256=cp['file_sha256'],
                native_official_evaluation=native(goal,cp['state']),native_wrapped_diagnostic=diagnostic(goal,cp['state']),**object_metric(goal,cp['state'])))
        same(row['object_only'],dict(checkpoints=metrics,final=object_metric(goal,row['final_state']),endpoint_scope='full_H2'),'object-only metrics')
        c.expect('object_only_result',receipt=row['object_only']);c.expect('branch_closed',receipt=row)
        initial=object_metric(goal,state);final=object_metric(goal,row['final_state'])
        summaries.append(dict(root_sha256=plan['root_sha256'],source_seed=slot['root_seed'],random_reset_seed=seed,
            placement_reset_seed=sp['placement_reset_seed'],commands=10,PNG=3,native_initial=native(goal,state),
            native_final=native(goal,row['final_state']),object_position_error_initial=initial['object_position_error'],
            object_position_error_final=final['object_position_error'],wrapped_object_angle_error_initial=initial['wrapped_object_angle_error'],
            wrapped_object_angle_error_final=final['wrapped_object_angle_error'],object_pose_exactly_stationary=bool(np.array_equal(state[2:5],np.asarray(row['final_state'])[2:5])),
            object_pose_stationary_all11=all(np.array_equal(state[2:5],np.asarray(e['actual_state'])[2:5]) for e in row['trace']),
            recorded_contact_steps=sum(e['official_info']['n_contacts']>0 for e in row['trace']),
            static_goal_pose_mismatches=sum(not np.array_equal(e['official_info']['goal_pose'],goal[2:5]) for e in row['trace']),
            negative_native_angle_terms=sum(e['pose_diagnostics']['native_formula']['negative_angle_term'] for e in row['trace']),
            native_wrapped_success_disagreements=sum(e['pose_diagnostics']['native_vs_wrapped_success_disagreement'] for e in row['trace'])))
    require(counts==r['counts']=={k:dict(attempted=n,completed=n) for k,n in dict(reset=4,step=20,render=22).items()},'independent actual operation census')
    require(r['internal_counts']==dict(reset_physics_completed_implied=8,reset_physics_attempted_upper_bound=8,
        goal_renders_completed_implied=4,actual_substeps=None) and sp['reset_semantics']['internal_physics_steps_per_reset']==2 and
        sp['reset_semantics']['internal_renders_per_reset']==1,'internal implied counts not new substep measurements')
    c.expect('run_finished',status=r['status'],counts=counts,errors=[]);require(c.index==len(c.events),'no extra physical journal events')
    if recheck_bindings:
        for path,h in b['bindings'].items():require(sha(path)==h,'post-read TOCTOU: '+path)
    return dict(schema=NAME+'.report',status='saved_physical_receipts_verified_not_admitted',all_checks=True,
        counts=counts,branches=2,PNG=6,event_count=len(c.events),roots=summaries,errors=[],new_physical_steps=0,new_neural_calls=0,
        P1_admission=False,new_independent_samples=0,global_freshness='unknown',
        scope=['independent saved arithmetic/chronology/PNG verification, not new simulation or proof of hidden solver state',
            'full pusher placement bound to earlier actual reset-audit receipt; object seed/recipe arithmetic recomputed; polygon sampler not reexecuted',
            'intermediate steps1-4/6-9 RGB hashes and guard counts cross-checked as receipts, only0/5/10 pixels saved',
            'original goalRGB/source arrays are hash-bound; new reset goalRGB equality is a producer receipt, not independently saved goal-render pixels',
            'task core reused only for task validation/identity and saved winner prediction; model/J correctness not independently rerun',
            'reset internal8 physics/4 goal renders are source-implied, not separately instrumented here'],
        input_sha256=b['bindings'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,default=INPUT)
    parser.add_argument('--report-sha',default=INPUT_SHA);parser.add_argument('--prepared',type=Path,default=PREPARED)
    parser.add_argument('--prepare-sha',default=PREPARE_SHA);a=parser.parse_args()
    result=audit(load_bundle(a.report,a.report_sha,a.prepared,a.prepare_sha))
    print(canonical({k:v for k,v in result.items() if k!='input_sha256'}))


if __name__=='__main__':main()
