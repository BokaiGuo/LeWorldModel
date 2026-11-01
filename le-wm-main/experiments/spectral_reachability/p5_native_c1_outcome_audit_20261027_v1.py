"""Independent C1 saved physical outcome auditor; no executor/env/model imports."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

NAME='p5_native_c1_outcome_audit_20261027_v1'
EXECUTOR='p5_native_c1_executor_20261027_v1'
EXECUTOR_SHA='028cbda1b5f384fafde8289537760379b3e0a54cb3bb22eaf3dbeaef1d38d6b1'
SOURCE_MANIFEST_SHA='d302060deca17ebffb14f9c617d2af8e6fd1c14113dff8e0809fedfca267884a'
BRIDGE='p5_prospective_control_bridge_20261027_v1'
HELPER='p5_jepa_fresh_fidelity_audit_20261027_v1'
HELPER_SHA='42dfe3b7df0eb973c61980b8f49c245f9c24b546eb483faf99f3ac34b965031f'
ROOT=Path(__file__).resolve().parents[2]
ARMS=('compiled','nominal','factual_reference')
NAMES=('rgb.npy','state.npy','proprio.npy','full_public_state.json','transition.json')
SEED=2026102701
EXPECTED=dict(env_construct=1,seed=2,reset=1,control_step=15,space_step=151,native_render=16,native_get_obs=16,close=1)


def require(ok,message):
    if not ok:
        raise ValueError(message)


def sha(path):
    path=Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)),'non-file/symlink rejected')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def strict_json(raw):
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'duplicate JSON key')
            out[k]=v
        return out
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
    canonical(value)
    return value


def read_json(path):
    sha(path)
    require(Path(path).stat().st_size<4*1024**2,'JSON byte bound')
    return strict_json(Path(path).read_bytes())


def numeric_helpers(root):
    path=root/f'experiments/spectral_reachability/{HELPER}.py'
    require(sha(path)==HELPER_SHA,'numeric audit helper source rejected')
    spec=importlib.util.spec_from_file_location(NAME+'_numeric_only',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arm_index(rows):
    require(len(rows)==3 and [r['arm'] for r in rows]==list(ARMS),'all three ordered arm slots required; no filtering/replacement')
    return {r['arm']:r for r in rows}


def protocol():
    return dict(schema=NAME,seed=SEED,arms=list(ARMS),scope='source/hash-bound saved arrays/JSON only, no neural or physics replication',
                prefix='all11frames0..10 x5files per arm, gate journal order before EVERY suffix intent; exact file bytes',
                authority='actual ready manifest + Ohm freeze files + canonical compiler selection binding; never substitute packet',
                counts='each complete arm151 .01s substeps,453total; failed/partial work retained without padding',
                metrics='native float64 body-origin XY and principal-angle goal errors, agent trajectory separately; decision10->15 suffix only',
                coverage='native stored coverage/reward consistency only; not independent polygon intersection; reset coverage unavailable',
                comparisons='compiled minus each of nominal/factual_reference; each quantity reported, no fitted scalar score or threshold',
                failure='retain3slots, partial frames/intents, close errors; no complete-case effectiveness summary',
                claim='protocol completion is not C1 effectiveness; agent-only motion is not object response/C2; no generalization',
                neural_calls=0,physics_calls=0,C1_effectiveness_claim=False,C2_established=False,P5_admission=False)


def source_prepare(root):
    path=root/f'results/{EXECUTOR}/manifest_source_frozen.json'
    require(sha(path)==SOURCE_MANIFEST_SHA,'executor source manifest identity')
    m=read_json(path)
    seals=dict(m['seals']);seals[str(path)]=SOURCE_MANIFEST_SHA
    require(seals[str(root/f'experiments/spectral_reachability/{EXECUTOR}.py')]==EXECUTOR_SHA,'executor identity')
    for p in (Path(__file__).resolve(),root/f'tests/test_{NAME}.py',root.parent/f'docs/{NAME}.md',root/f'experiments/spectral_reachability/{HELPER}.py'):
        seals[str(p)]=sha(p)
    require(seals[str(root/f'experiments/spectral_reachability/{HELPER}.py')]==HELPER_SHA,'helper identity')
    for p,h in seals.items():
        require(sha(p)==h,'source seal drift: '+p)
    return dict(schema=NAME,status='prepared_auditor_no_outcomes_opened',protocol=protocol(),seals=seals,
                neural_calls=0,physics_calls=0,C1_effectiveness_claim=False,C2_established=False,P5_admission=False)


def bind(root,ready_manifest,execution_dir):
    source=source_prepare(root)
    ready_manifest=Path(ready_manifest).absolute();execution_dir=Path(execution_dir).absolute()
    m=read_json(ready_manifest)
    require(m['schema']==EXECUTOR and m['status']=='ready_three_branch_native_execution_not_run','actual ready manifest required')
    arm_index(m['branches'])
    seals=dict(source['seals'])
    for p,h in m['seals'].items():
        require(p not in seals or seals[p]==h,'source/ready conflict')
        seals[p]=h
    seals[str(ready_manifest)]=sha(ready_manifest)
    require((execution_dir/'report.json').is_file(),'actual terminal physical report required; pending is not evidence')
    files=list(execution_dir.rglob('*'))
    require(len(files)<1024 and not any(p.is_symlink() for p in files),'output node/symlink bound')
    require(sum(p.stat().st_size for p in files if p.is_file())<40_000_000,'output payload bound')
    for p in files:
        if p.is_file():
            seals[str(p)]=sha(p)
    for p,h in seals.items():
        require(sha(p)==h,'input/source drift: '+p)
    return dict(schema=NAME,status='bound_saved_outcomes_not_yet_audited',protocol=protocol(),seals=seals,
                ready_manifest=str(ready_manifest),execution_dir=str(execution_dir),neural_calls=0,physics_calls=0)


def authority(ready):
    freeze=Path(ready['freeze_dir'])
    fr,all_branches,selected=(read_json(freeze/(n+'.json')) for n in ('freeze_receipt','all_branches','selected_packet'))
    require(fr['schema']==BRIDGE+'.freeze' and fr['status']=='all_three_branches_durable_not_executed' and fr['census']=={a:'planned_not_constructed' for a in ARMS},'exact three-arm bridge freeze')
    require(fr['all_branches_sha256']==sha(freeze/'all_branches.json') and fr['selected_packet_sha256']==sha(freeze/'selected_packet.json'),'freeze file hashes')
    prepared=read_json(all_branches['prepared_path']);receipt=read_json(all_branches['compiler_receipt_path'])
    require(all_branches['prepared_file_sha256']==sha(all_branches['prepared_path']) and all_branches['compiler_receipt_file_sha256']==sha(all_branches['compiler_receipt_path']),'original ready/compiler FILE hashes')
    require(prepared['schema']==BRIDGE and prepared['status']=='ready_for_main_CUDA_compile' and not prepared['blockers'] and prepared['contract']==ready['contract'],'ready original protocol')
    require(receipt['schema']==BRIDGE+'.compiler' and receipt['status']=='selected_packet_replayed' and receipt['prepared_sha256']==digest(prepared)
            and receipt['compiler_inputs_sha256']==prepared['compiler_inputs_sha256'] and receipt['used_observed_frame15'] is False
            and receipt['physical_calls']==0 and receipt['selected_replay_exact'] is True,'actual independent-goal compiler binding')
    require(receipt['attempted']==receipt['completed']==ready['contract']['compiler']['expected_completed'] and not receipt['partial_work_unknown'],'compiler accounting receipt')
    require(selected==ready['selected_packet']==dict(packet=receipt['selected_env_packet'],selection_sha256=digest(receipt)),'selected replay packet binding')
    require(all_branches['branches']==ready['branches'],'frozen branches unchanged')
    require([c['id'] for c in receipt['candidates']]==['nominal','ridge'],'fixed compiler candidates')
    scores=np.array([c['independent_goal_mse'] for c in receipt['candidates']],np.float64)
    require(scores.shape==(2,) and np.isfinite(scores).all() and np.all(scores>=0),'finite goal score receipts')
    chosen=receipt['candidates'][int(np.argmin(scores))]
    require(chosen['id']==receipt['selected_id'] and chosen['env_packet']==selected['packet'],'unchanged stable nominal-first selection')
    branches=arm_index(ready['branches'])
    packets=dict(compiled=selected['packet'],nominal=np.zeros((5,2)).tolist(),factual_reference=prepared['factual_reference_packet'])
    for arm,b in branches.items():
        require(b['seed']==SEED and b['prefix_env_commands']==prepared['prefix_env_commands'] and b['suffix_env_packet']==packets[arm]
                and b['compiler_receipt_sha256']==digest(receipt) and b['root_identity_sha256']==prepared['root_identity_sha256'],'frozen branch time/packet/root binding')
    return dict(prepared_file_sha256=sha(all_branches['prepared_path']),compiler_receipt_file_sha256=sha(all_branches['compiler_receipt_path']),
                selected_id=receipt['selected_id'],selected_env_packet=selected['packet'],derivative_arrays_recomputed=False)


def journal(path,arm,report):
    require(sha(path)==report['journal_sha256'] and Path(path).stat().st_size<4*1024**2,'journal SHA/size')
    previous=None;events=[];records={};gates=[];pending=[];operations=[];commands={};failures=[]
    for i,line in enumerate(Path(path).read_bytes().splitlines()):
        row=strict_json(line)
        require(row['sequence']==i and row['previous_sha256']==previous and row['entry_sha256']==digest({k:v for k,v in row.items() if k!='entry_sha256'}),'journal chain')
        previous=row['entry_sha256'];event=row['event'];phase=event['phase'];events.append(event)
        if phase=='intent':
            require(event['role']==arm,'cross-arm native operation')
            op=event['operation'];detail=event['details'];operations.append((op,detail));pending.append((op,detail))
            if op=='step':
                frame=detail['frame'];require(frame not in commands,'duplicate native step')
                if frame>10:
                    require(gates==list(range(11)),'suffix attempted before ALL eleven prefix byte gates')
                commands[frame]=detail
        elif phase=='returned':
            require(event['role']==arm and pending and pending[-1][0]==event['operation'],'unpaired native return')
            pending.pop()
        elif phase=='assets_durable':
            rec=event['record'];frame=rec['frame']
            require(event['arm']==arm and frame==len(records) and frame<16,'noncontiguous frame receipt')
            records[frame]=rec
        elif phase=='prefix_byte_gate_passed':
            frame=event['frame'];require(event['arm']==arm and frame==len(gates) and frame in records and frame<=10,'prefix gate sequence')
            gates.append(frame)
        elif phase=='close_failed':
            failures.append(event)
        else:
            require(phase in ('preflight_intent','terminal'),'unrecognized journal phase')
    require(events and events[-1]['phase']=='terminal' and events[-1]['status']==report['status'],'terminal journal required')
    body={k:v for k,v in report.items() if k!='journal_sha256'}
    require(events[-1]['summary_sha256']==digest(body),'terminal report canonical binding')
    require(report['frames']==len(records) and report['prefix_gate_frames']==gates,'partial frames/gates must not be padded')
    if report['status']=='completed':
        require(not pending and not failures and len(records)==16 and gates==list(range(11)),'completed arm cannot contain incomplete work')
        require([op for op,_ in operations]==['construct','seed','reset',*['step']*15,'close'],'independent complete native operation order')
    return dict(events=events,records=records,gates=gates,commands=commands,operations=operations,interrupted_intents=pending,close_failures=failures)


def pose(public):
    block=public['bodies']['block'];agent=public['bodies']['agent'];goal=np.array(public['env']['goal_pose']['values'],np.float64)
    require(goal.tolist()==[256.,256.,float(np.pi/4)],'fixed native object goal')
    theta=block['angle']-goal[2];coverage=public['env']['coverage_arr']
    return dict(agent_xy=agent['position'],agent_velocity=agent['velocity'],object_xy=block['position'],object_angle_native_rad=block['angle'],goal_pose=goal.tolist(),
                object_xy_goal_error=float(np.linalg.norm(np.array(block['position'])-goal[:2])),object_angle_goal_error=float(abs(np.arctan2(np.sin(theta),np.cos(theta)))),
                native_final_coverage=coverage[-1] if coverage else None,native_max_coverage=max(coverage) if coverage else None,C2_established=False,P5_admission=False)


def suffix_metrics(public):
    require(len(public)==16,'complete actual16frames; no endpoint padding')
    decision,terminal=pose(public[10]),pose(public[15])
    agent=np.array([f['bodies']['agent']['position'] for f in public[10:]],np.float64)
    block=np.array([f['bodies']['block']['position'] for f in public[10:]],np.float64)
    angles=np.array([f['bodies']['block']['angle'] for f in public[10:]],np.float64)
    stationary=bool(np.all(block==block[:1]) and np.all(angles==angles[:1]))
    return dict(decision=decision,terminal=terminal,agent_suffix_xy=agent.tolist(),object_suffix_xy=block.tolist(),object_suffix_angles_native=angles.tolist(),
                agent_suffix_path_px=float(np.linalg.norm(np.diff(agent,axis=0),axis=1).sum()),object_suffix_path_px=float(np.linalg.norm(np.diff(block,axis=0),axis=1).sum()),
                object_suffix_angle_path_abs_rad=float(np.abs(np.diff(angles)).sum()),object_pose_exactly_stationary_since_decision=stationary,
                agent_only_suffix_motion=bool(stationary and np.any(agent!=agent[:1])),
                object_xy_error_reduction=decision['object_xy_goal_error']-terminal['object_xy_goal_error'],
                object_angle_error_reduction=decision['object_angle_goal_error']-terminal['object_angle_goal_error'],
                coverage_gain_since_decision=terminal['native_final_coverage']-decision['native_final_coverage'],
                coverage_frames1_to15=public[-1]['env']['coverage_arr'],contact_points_frames0_to15=[f['env']['n_contact_points'] for f in public],
                C1_effectiveness_claim=False,C2_established=False,P5_admission=False)


def compare_outcomes(rows):
    indexed=arm_index(rows)
    if not all(r['status']=='completed' and r.get('metrics') is not None for r in rows):
        return dict(status='not_comparable_all_three_complete_arms_required',all3_retained=True,C1_effectiveness_claim=False,C2_established=False,P5_admission=False)
    compiled=indexed['compiled']['metrics']['terminal'];comparisons={}
    for arm in ARMS[1:]:
        control=indexed[arm]['metrics']['terminal']
        comparisons[arm]={k:compiled[k]-control[k] for k in ('object_xy_goal_error','object_angle_goal_error','native_final_coverage')}
    return dict(status='all_three_paired_observed_outcomes',compiled_minus_control=comparisons,
                signs='negative XY/angle error difference and positive coverage difference favor compiled on that quantity only; exact ties retained',
                C1_effectiveness_claim=False,C2_established=False,P5_admission=False,generalization=False)


def audit(root,frozen):
    require(frozen==bind(root,Path(frozen['ready_manifest']),Path(frozen['execution_dir'])),'audit manifest/source/data differs')
    op=numeric_helpers(root);ready=read_json(frozen['ready_manifest']);out=Path(frozen['execution_dir']);top=read_json(out/'report.json')
    auth=authority(ready);branches=arm_index(ready['branches']);indexed=arm_index(top['rows']);checks={}
    def check(name,condition):
        checks[name]=bool(condition);require(condition,'audit mismatch: '+name)
    ready_sha=sha(frozen['ready_manifest'])
    check('executor_protocol_identity',top['schema']==EXECUTOR and top['ready_manifest_file_sha256']==ready_sha and not top['neural_executed'] and not top['C1_effectiveness_claim'] and not top['C2_established'] and not top['P5_admission'])
    check('predispatch_all_packets_bound',read_json(out/'all_branches.json')==dict(branches=ready['branches'],ready_manifest_file_sha256=ready_sha) and read_json(out/'selected_packet.json')==ready['selected_packet'])
    intent=read_json(out/'dispatch_intent.json');check('predispatch_full_census',intent['ready_manifest_file_sha256']==ready_sha and intent['contract']==ready['contract'] and all(r['status']=='not_run' and r['frames']==0 for r in arm_index(intent['rows']).values()))
    all_rows=[];totals=Counter()
    for arm in ARMS:
        def ck(name,condition):
            check(arm+'_'+name,condition)
        directory=out/arm;row=indexed[arm];r=read_json(directory/'report.json')
        ck('report_identity',sha(directory/'report.json')==row['report_sha256'] and r=={k:v for k,v in row.items() if k!='report_sha256'})
        ck('status_known',r['status'] in ('completed','guard_abort','integrity_failure','environment_termination','not_run'))
        evidence=journal(directory/'journal.jsonl',arm,r)
        bank=branches[arm];commands=np.concatenate((np.asarray(bank['prefix_env_commands'],np.float64),np.asarray(bank['suffix_env_packet'],np.float64)))
        saved=directory/'frozen_env_commands.npy'
        ck('fixed_raw_commands',commands.shape==(15,2) and np.array_equal(op.load_array(saved,frozen['seals'][str(saved)],(15,2),np.float64,directory),commands))
        for operation,detail in evidence['operations']:
            if operation in ('seed','reset'):
                ck(operation+'_same_seed',detail['seed']==SEED)
            if operation=='construct':
                ck('native_constructor',detail['config']==ready['native_constructor'])
        full=[];prefix_matches=[]
        for frame,rec in evidence['records'].items():
            path=directory/f'frame_{frame:02d}'
            ck(f'{frame:02d}_asset_closure',set(rec['files'])==set(NAMES))
            for name,info in rec['files'].items():
                ck(f'{frame:02d}_{name}',sha(path/name)==info['sha256'] and (path/name).stat().st_size==info['bytes'])
            state=op.load_array(path/'state.npy',rec['files']['state.npy']['sha256'],(7,),np.float32,path)
            prop=op.load_array(path/'proprio.npy',rec['files']['proprio.npy']['sha256'],(4,),np.float32,path)
            op.load_array(path/'rgb.npy',rec['files']['rgb.npy']['sha256'],(224,224,3),np.uint8,path)
            public=read_json(path/'full_public_state.json');transition=read_json(path/'transition.json');full.append(public)
            ag,block=public['bodies']['agent'],public['bodies']['block']
            ck(f'{frame:02d}_state_projection',np.array_equal(state,np.array(ag['position']+block['position']+[block['angle']%(2*np.pi)]+ag['velocity'],np.float32)) and np.array_equal(prop,state[[0,1,5,6]]) and public['env']['seed']==SEED and state.tolist()==rec['state'] and prop.tolist()==rec['proprio'])
            if frame<=10:
                identity=ready['prefix_identity'][frame]
                matches=identity['frame']==frame and all(sha(identity['files'][name]['path'])==identity['files'][name]['sha256']==rec['files'][name]['sha256'] and Path(identity['files'][name]['path']).read_bytes()==(path/name).read_bytes() for name in NAMES)
                prefix_matches.append(dict(frame=frame,all5_byte_exact=matches))
                if frame in evidence['gates']:
                    ck(f'{frame:02d}_passed_gate_has_actual_exact_bytes',matches)
            if frame:
                cmd=read_json(directory/f'command_{frame:02d}.json');pos=np.array(full[-2]['bodies']['agent']['position'],np.float64);u=commands[frame-1]
                ck(f'{frame:02d}_native_command',cmd==evidence['commands'][frame] and cmd['frame']==frame and cmd['action_index']==frame-1 and cmd['packet_index']==(frame-1)//5 and cmd['step_within_packet']==(frame-1)%5 and np.array_equal(cmd['raw_env_argument'],u) and np.array_equal(cmd['pixel_displacement'],100*u) and np.array_equal(cmd['agent_xy_before'],pos) and np.array_equal(cmd['absolute_target'],pos+100*u) and np.array_equal(public['env']['latest_action'],pos+100*u) and cmd['pre_public_sha256']==sha(directory/f'frame_{frame-1:02d}/full_public_state.json') and cmd['dt']==.01 and cmd['expected_substeps']==10)
                info=transition['info'];history=public['env']['coverage_arr']
                ck(f'{frame:02d}_coverage_reward_history',len(history)==frame and history[:-1]==full[-2]['env']['coverage_arr'] and info['final_coverage']==history[-1] and info['max_coverage']==max(history) and 0<=history[-1]<=1 and transition['reward']==float(np.clip(history[-1]/public['env']['success_threshold'],0,1)))
        counts=(r['native_call_receipts'] or {}).get('python_call_entries',{});totals.update(counts)
        metrics=None
        if r['status']=='completed':
            ck('complete_census',all(counts.get(k)==v for k,v in EXPECTED.items()) and r['source_hashes_unchanged'] and r['error'] is None)
            steps=r['native_call_receipts']['space_step_events']
            expected_context=Counter({(arm,'reset',None):1,**{(arm,'step',i):10 for i in range(1,16)}})
            ck('151_exact_substep_contexts',len(steps)==151 and all(e['dt']==.01 for e in steps) and Counter((e['context']['role'],e['context']['operation'],e['context']['frame']) for e in steps)==expected_context)
            metrics=suffix_metrics(full)
            ck('endpoint_receipt_arithmetic',r['endpoint']==metrics['terminal'])
        all_rows.append(dict(arm=arm,status=r['status'],error=r['error'],frame_count=len(full),prefix_matches=prefix_matches,interrupted_intents=evidence['interrupted_intents'],close_failures=evidence['close_failures'],metrics=metrics))
    complete=all(r['status']=='completed' for r in all_rows)
    check('top_completion_matches_all3',top['status']==('native_c1_three_branch_execution_recorded' if complete else 'native_c1_three_branch_execution_incomplete'))
    if complete:
        check('453_total_substeps',totals['space_step']==453 and all(totals[k]==3*v for k,v in EXPECTED.items()))
    for p,h in frozen['seals'].items():
        require(sha(p)==h,'bytes changed during audit')
    return dict(schema=NAME,status='c1_saved_physical_outcomes_audited' if complete else 'c1_partial_census_audited_no_effectiveness',protocol=protocol(),checks=checks,check_count=len(checks),all_checks=all(checks.values()),
                authority=auth,rows=all_rows,comparison=compare_outcomes(all_rows),counts=dict(totals),source_data_hashes_unchanged=True,
                limitation='stored source-bound receipts, not independent physics/neural reproduction or private contact-cache authentication; one fixed calibration root',
                neural_calls=0,physics_calls=0,C1_effectiveness_claim=False,C2_established=False,P5_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','bind','audit'],nargs='?',default='prepare')
    p.add_argument('--ready-manifest',type=Path)
    p.add_argument('--execution-dir',type=Path)
    p.add_argument('--manifest',type=Path,default=ROOT/f'results/{NAME}/manifest_outcomes_frozen.json')
    args=p.parse_args()
    if args.command=='prepare':
        value=source_prepare(ROOT)
    elif args.command=='bind':
        if args.ready_manifest is None or args.execution_dir is None:
            p.error('actual --ready-manifest and --execution-dir required')
        value=bind(ROOT,args.ready_manifest,args.execution_dir)
    else:
        value=audit(ROOT,read_json(args.manifest))
    print(json.dumps(value,sort_keys=True,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
