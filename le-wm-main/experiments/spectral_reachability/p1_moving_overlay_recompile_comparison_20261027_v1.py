"""Pure saved off/on FULL-recompile comparison, never a planner or evaluator run.

Producer layout is an explicit externally hashed interface, not a guessed old
schema. Missing checkpoints stay null; no encoder/model/physics calls.
"""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

NAME = 'p1_moving_overlay_recompile_comparison_20261027_v1'
ARMS = ('line_search_GN', 'full10_conditional_D2P')
CELLS = tuple((s,c,a) for s in (0,1) for c in ('off','on') for a in ARMS)
CAP, FLOOR = 2*1024**2, 200*1024**2
SOURCEPACK_SCHEMA = 'p1_constructive_moving_history_source_20261027_v1.sourcepack'
SOURCEPACK_SHA = 'c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6'
MISSING_RGB = dict(status='missing_not_encoded', actual_latent_Q=None,
                   reason='No bound new RGB encoder artifact in this v1 input contract.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    require(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value), 'SHA256 required')
    return value


def parse(data):
    def unique(items):
        d={}
        for k,v in items:
            require(k not in d,'duplicate JSON key');d[k]=v
        return d
    return json.loads(data,object_pairs_hook=unique,
                      parse_constant=lambda _:require(False,'nonfinite JSON'))


def vector(v):
    a=np.asarray(v,dtype=np.float64)
    require(a.shape==(7,) and np.isfinite(a).all(),'finite native state7')
    return a


def metrics(goal,state):
    d=vector(state)-vector(goal)
    angle=float(abs(d[4]));native_angle=min(angle,2*math.pi-angle)
    wrapped=float(abs(np.arctan2(np.sin(d[4]),np.cos(d[4]))))
    xy=float(np.linalg.norm(d[:4]));distance=float(np.linalg.norm(d))
    native=bool(xy<20 and native_angle<math.pi/9)
    proper=bool(xy<20 and wrapped<math.pi/9)
    wd=d.copy();wd[4]=wrapped
    return dict(native_distance=distance,native_reward=-distance,native_success=native,
        native_allxy_l2=xy,native_angle_term=native_angle,negative_native_angle=native_angle<0,
        wrapped_allxy_success=proper,wrapped_angle_error=wrapped,
        object_xy_l2=float(np.linalg.norm(d[2:4])),agent_xy_l2=float(np.linalg.norm(d[:2])),
        velocity_error_l2=float(np.linalg.norm(d[5:])),wrapped_joint_state_l2=float(np.linalg.norm(wd)),
        native_wrapped_success_disagree=native!=proper,
        success_has_velocity_condition=False)


def key(row):
    s,c,a=row['root_slot'],row['condition'],row['arm']
    require(type(s) is int and (s,c,a) in CELLS,'fixed root/condition/arm identity')
    digest(row['selection_digest'])
    return s,c,a


def index_rows(rows):
    require(isinstance(rows,list) and len(rows)==8,'fixed denominator eight rows')
    out={}
    for row in rows:
        k=key(row);require(k not in out,'duplicate stable branch identity');out[k]=row
    require(set(out)==set(CELLS),'all eight cells, never favorable subset')
    return out


def chain(data,tail):
    require(len(data)<=8*1024**2 and data.endswith(b'\n'),'complete bounded journal')
    events=[parse(line) for line in data.splitlines()];last=None
    require(events,'nonempty journal')
    for i,e in enumerate(events):
        require(type(e['index']) is int and e['index']==i and e['previous_sha256']==last and
                e['event_sha256']==sha(canonical({k:v for k,v in e.items() if k!='event_sha256'})),
                'journal ordered hashchain')
        last=e['event_sha256']
    require(last==digest(tail),'journal externally bound tail')
    return events


def source_targets(pack,events):
    """Actual original sourcepack + its own construction journal, no new labels."""
    require(pack['schema']==SOURCEPACK_SCHEMA and pack['sourcepack_complete'] is True and
        pack['status']=='fixed_slots_processed' and pack['errors']==[] and len(pack['rows'])==2,
        'complete ORIGINAL moving sourcepack')
    require(events[0]['kind']=='run_started' and events[-1]['kind']=='run_finished' and
        events[-1]['status']==pack['status'] and events[-1]['counts']==pack['counts'] and
        pack['prepare_id']==sha(canonical(events[0]['prepare'])), 'original source journal boundaries/prepare identity')
    out={}
    for row in pack['rows']:
        s=row['slot']['index'];require(type(s) is int and s in (0,1) and s not in out,'source slot identity')
        actual=[e for e in events if e.get('slot')==s and e.get('phase')=='construction' and e['kind']=='native_return']
        require(actual and [e['step'] for e in actual]==list(range(1,len(actual)+1)) and len(actual)<=10,
                'original construction endpoint, no outcome goal fitting')
        goal=vector(actual[-1]['state'])
        obs=[e for e in events if e.get('slot')==s and e.get('phase')=='replay0' and
             e['kind']=='fresh_observation' and e['step']==10]
        require(len(obs)==1,'original moving initial observation')
        state=vector(obs[0]['state'])
        require(sha(np.ascontiguousarray(np.concatenate((state,goal)),dtype=np.float64).tobytes())==row['root_sha256'],
                'original physical root/goal identity')
        out[s]=dict(root_sha256=row['root_sha256'],blind_sha256=row['blind_payload_sha256'],
                    goal=goal.tolist(),initial_state=state.tolist(),initial_snapshot=obs[0]['snapshot'])
    require(set(out)=={0,1},'both original goals')
    return out


def png(data,checkpoint):
    require(len(data)<=1024**2 and sha(data)==digest(checkpoint['file_sha256']),'PNG file SHA')
    with Image.open(BytesIO(data)) as im:
        require(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'official RGB224 PNG')
        pixels=np.asarray(im).copy()
    require(pixels.dtype==np.uint8 and sha(pixels.tobytes())==digest(checkpoint['pixel_sha256']), 'PNG decoded pixel SHA')


def checkpoints(row,read_frame):
    by={}
    require(isinstance(row['checkpoints'],list) and isinstance(row['missing_checkpoints'],list),'explicit actual/missing checkpoints')
    for cp in row['checkpoints']:
        t=cp['step'];require(type(t) is int and t in (0,5,10) and t not in by,'unique actual checkpoint step')
        vector(cp['state'])
        require(cp['snapshot']['state']==cp['state'],'checkpoint native state/snapshot equality')
        png(read_frame(cp['png']),cp);by[t]=cp
    require(len(set(row['missing_checkpoints']))==len(row['missing_checkpoints']) and
            set(row['missing_checkpoints'])==set((0,5,10))-set(by),'explicit missing checkpoints exact denominator')
    return by


def difference(a,b):
    return None if a is None or b is None else a-b


def summarize(rows):
    by={key(r):r for r in rows};paired=[]
    fields=('native_distance','object_xy_l2','wrapped_angle_error','wrapped_joint_state_l2','velocity_error_l2')
    for s in (0,1):
        for arm in ARMS:
            off,on=by[(s,'off',arm)],by[(s,'on',arm)]
            final={k:difference(None if on['terminal10'] is None else on['terminal10'][k],
                                None if off['terminal10'] is None else off['terminal10'][k]) for k in fields}
            success=difference(None if on['terminal10'] is None else int(on['terminal10']['native_success']),
                               None if off['terminal10'] is None else int(off['terminal10']['native_success']))
            paired.append(dict(root_slot=s,arm=arm,root_sha256=off['root_sha256'],on_minus_off_terminal=final,
                native_success_on_minus_off=success,
                prefix_retreat_on_minus_off=difference(on['prefix_retreat_native'],off['prefix_retreat_native']),
                actual_latent_Q_on_minus_off=None))
    means={}
    for arm in ARMS:
        group=[r for r in paired if r['arm']==arm]
        means[arm]=dict(root_denominator=2,complete_terminal_pairs=sum(r['native_success_on_minus_off'] is not None for r in group),
            on_minus_off={k:None if any(r['on_minus_off_terminal'][k] is None for r in group) else
                          sum(r['on_minus_off_terminal'][k] for r in group)/2 for k in fields})
    prefix=[]
    for s in (0,1):
        for c in ('off','on'):
            a,b=by[(s,c,ARMS[1])],by[(s,c,ARMS[0])]
            prefix.append(dict(root_slot=s,condition=c,D2P_minus_GN_prefix_native_distance=difference(
                None if a['prefix5'] is None else a['prefix5']['native_distance'],
                None if b['prefix5'] is None else b['prefix5']['native_distance']),
                strict_non_greedy_vs_raw_zero=None,reason='No independent raw-zero prefix control in these eight branches.'))
    return dict(per_root=paired,by_arm=means,prefix_arm_contrasts=prefix,
                semantics='on FULL reoptimization minus off FULL reoptimization; not just rerendered old actions',
                independent_root_count=0,uncertainty='two reused known roots, descriptive only; no inferential claim')


PHYSICAL = 'p1_moving_overlay_recompile_physical_20261027_v1'
PLANNER = 'p1_moving_overlay_recompile_cuda_20261027_v1'
OVERLAY = 'p1_moving_target_overlay_source_20261027_v1'
CAPS = dict(constructor=8,close=8,reset=16,public_step=160,physics_step=1632,
            set_state=32,goal_setter=24,goal_render=16,goal_bind=8,explicit_render=528)
ROW_STATUSES = ('not_started','running','engineering_failed','horizon_completed','native_success','native_stopped')


def recount(events,report,physical,frozen):
    """Recount durable operation ledger, including warmup and failed attempts."""
    counts={k:dict(attempted=0,completed=0) for k in CAPS}
    returned={k:{p:[] for p in ('plan','warmup')} for k in CELLS}
    attempts={k:[] for k in CELLS}
    warm=np.array([[.1,0]]*5+[[0,.1]]*5,dtype=np.float32)
    observations={k:0 for k in CELLS}
    for e in events:
        kind=e['kind']
        if kind in ('operation_attempted','operation_completed','operation_failed','returned_transition','fresh_observation'):
            k=key(e);require(e['selection_digest']==frozen[k]['selection_digest'] and
                e['branch']==physical[k]['index'],'journal stable selection/branch binding')
        if kind in ('operation_attempted','operation_completed','operation_failed'):
            r=e['resource'];require(r in CAPS,'known metered resource')
            if kind!='operation_failed':counts[r]['attempted' if kind=='operation_attempted' else 'completed']+=1
            if r=='public_step':
                phase=e['phase'];t=e['step'];require(phase in ('plan','warmup') and type(t) is int and 1<=t<=10,'raw command step')
                cmd=warm[t-1] if phase=='warmup' else np.asarray(frozen[k]['commands'],dtype=np.float32)[t-1]
                require(e['raw_bits']==cmd.tobytes().hex(),'actual public command equals frozen raw float32, no clip/rescale')
                if phase=='plan' and kind=='operation_attempted':attempts[k].append(t)
        if kind=='returned_transition':
            phase=e['phase'];t=e['step'];require(phase in ('plan','warmup') and type(t) is int and 1<=t<=10,'returned step')
            cmd=warm[t-1] if phase=='warmup' else np.asarray(frozen[k]['commands'],dtype=np.float32)[t-1]
            require(e['requested_bits']==e['actual_bits']==cmd.tobytes().hex() and e['actual_dtype']=='<f4',
                    'returned raw action bits; no second /100')
            vector(e['state']);returned[k][phase].append(t)
        if kind=='fresh_observation':observations[k]+=1
    require(counts==report['counts'],'full attempted/completed journal recount including warmup/reset/internal physics')
    for r,c in counts.items():require(0<=c['completed']<=c['attempted']<=CAPS[r],'bounded actual operation count')
    for k,row in physical.items():
        for phase in ('warmup','plan'):
            steps=returned[k][phase]
            require(steps==list(range(1,len(steps)+1)),'contiguous returned steps, no retry/replacement')
        require(row['own_attempted']==len(attempts[k]) and attempts[k]==list(range(1,len(attempts[k])+1)) and
            row['own_returned']==len(returned[k]['plan']) and row['warmup_returned']==len(returned[k]['warmup']) and
            row['PNG']==observations[k], 'per-branch attempted/returned/frame denominator')
    if report['status']=='moving_overlay_recompile_physical_recorded':
        completed={k:v['completed'] for k,v in counts.items()}
        require(report['errors']==[] and report['inputs_final_verified'] is True and
            all(v['attempted']==v['completed'] for v in counts.values()) and
            completed['constructor']==completed['close']==8 and
            completed['physics_step']==2*completed['reset']+10*completed['public_step'] and
            completed['set_state']==2*completed['reset'] and completed['goal_render']==completed['reset'] and
            completed['goal_setter']==completed['reset']+completed['goal_bind'] and
            completed['public_step']==sum(r['own_returned']+r['warmup_returned'] for r in physical.values()) and
            completed['explicit_render']==sum(r['PNG'] for r in physical.values()), 'recorded full physical census identities')
    return counts


def assemble(report,prepared,events,targets,plans,read_frame):
    """NEW physical contract: checkpoints are subsets of fresh-observation receipts."""
    require(report['schema']==PHYSICAL+'.report' and report['status'] in
            ('moving_overlay_recompile_physical_recorded','partial_or_failed') and report['fixed_denominator']==8,
            'NEW fixed-eight physical schema/status')
    require(prepared['schema']==PHYSICAL+'.prepare' and prepared['contract']['caps']==CAPS,'NEW prepare/caps')
    physical=index_rows(report['rows']);frozen=index_rows(plans)
    require(sorted(r['index'] for r in physical.values())==list(range(8)),'eight unique branch indices')
    require(events[0]['kind']=='run_started' and events[-1]['kind']=='run_finished' and
            events[0]['plans']==prepared['plans'] and events[0]['contract']==prepared['contract'], 'journal frozen dispatch lifecycle')
    require(events[-1]['counts']==report['counts'] and events[-1]['errors']==report['errors'] and
            events[-1]['status']==report['status'],'journal complete census/error preservation')
    close={};observed={};triplets={}
    for e in events:
        if e['kind']=='branch_closed':
            k=key(e);row=physical[k]
            require(k not in close and e['selection_digest']==row['selection_digest'] and e['branch']==row['index'] and
                all(e[f]==row[f] for f in ('status','own_returned','first_error')), 'journal branch close shared fields')
            close[k]=e
        if e['kind']=='fresh_observation':
            cp=e['receipt'];k=key(e)
            require(cp['condition'] in ('off','on') and cp['overlay_index'] in (0,1,2) and
                cp['condition']==('on' if cp['overlay_index']==1 else 'off') and
                cp['phase'] in ('warmup','plan') and type(cp['step']) is int and 0<=cp['step']<=10,
                'actual observation phase/condition/index')
            vector(cp['state']);require(cp['snapshot']['state']==cp['state'],'observed native state/snapshot')
            png(read_frame(cp['png']),cp)
            tk=(k,cp['phase'],cp['step']);group=triplets.setdefault(tk,{})
            require(cp['overlay_index'] not in group,'unique triplet observation');group[cp['overlay_index']]=e
            if cp['phase']!='plan' or cp['overlay_index']!=(0 if k[1]=='off' else 1):continue
            require(cp['condition']==k[1],'same-condition actual label')
            ck=(k,cp['step']);require(ck not in observed,'unique chosen observation journal')
            observed[ck]=(cp,e)
    for k in CELLS:
        if k in close:continue
        row=physical[k]
        require(row['status']=='not_started' and row['checkpoints']==[] and
                set(row['missing_checkpoints'])=={0,5,10} and row['native_success'] is None,
                'missing close permitted only explicit unstarted branch, full denominator retained')
    counts=recount(events,report,physical,frozen)
    transitions={(key(e),e['phase'],e['step']):e for e in events if e['kind']=='returned_transition'}
    output=[]
    for k in CELLS:
        row=physical[k];plan=frozen[k];s,c,arm=k;target=targets[s]
        require(row['selection_digest']==plan['selection_digest'] and row['task_id']==plan['task_id'],
                'prephysical frozen selection/task identity')
        if row['initial_exact']:
            require(row['goal']==target['goal'] and row['root_state']==target['initial_state'],'original goal/root no endpoint fitting')
        cp=checkpoints(row,read_frame)
        trajectory={}
        for receipt in row['trajectory']:
            tk=(k,receipt['phase'],receipt['step']);group=triplets.get(tk,{})
            require(set(group)=={0,1,2},'accepted trajectory requires completed off/on/off triplet')
            require(all(g['pre']==g['post']==group[0]['pre'] for g in group.values()) and
                group[0]['receipt']['pixel_sha256']==group[2]['receipt']['pixel_sha256'] and
                group[0]['receipt']['file_sha256']==group[2]['receipt']['file_sha256'] and
                receipt==group[0 if c=='off' else 1]['receipt'], 'accepted unchanged fullsnapshot + exact off replay')
            if receipt['step']:
                returned=transitions.get(tk)
                require(returned is not None and returned['state']==receipt['state'] and returned['snapshot']==receipt['snapshot'],
                        'accepted label equals actual returned transition, not a different post-render state')
                if receipt['phase']=='plan':require(returned['goal_state']==target['goal'],'returned original task goal binding')
            if receipt['phase']!='plan':continue
            t=receipt['step'];require(t not in trajectory,'unique accepted plan trajectory')
            require((k,t) in observed and observed[(k,t)][0]==receipt and
                observed[(k,t)][1]['pre']==observed[(k,t)][1]['post'],'accepted full receipt/invariant journal')
            trajectory[t]=receipt
        for t,receipt in cp.items():
            require(t in trajectory and all(trajectory[t][f]==v for f,v in receipt.items()),
                    'checkpoint shared fields equal accepted trajectory receipt')
        require(set(cp)==set(trajectory)&{0,5,10},'all accepted checkpoint observations retained')
        if 0 in cp:
            require(row['initial_exact'] is True and cp[0]['state']==target['initial_state'] and cp[0]['snapshot']==target['initial_snapshot'],
                    'actual branch initial exact original moving body identity')
        if 10 in cp:require(row['own_returned']==10,'actual full horizon terminal, not early-stop substitution')
        if row['status']=='horizon_completed':require(set(cp)=={0,5,10} and row['own_returned']==10,'completed horizon checkpoints')
        values={t:metrics(target['goal'],cp[t]['state']) if t in cp else None for t in (0,5,10)}
        recorded=row['native_success'];require(recorded is None or type(recorded) is bool,'recorded native flag explicit')
        require(row['status'] in ROW_STATUSES,'explicit producer row status')
        terminal=values[10]
        output.append(dict(root_slot=s,condition=c,arm=arm,selection_digest=row['selection_digest'],
            root_sha256=target['root_sha256'],original_blind_sha256=target['blind_sha256'],
            blind_sha256=plan['blind_sha256'],goal=target['goal'],
            source_status=row['status'],initial=values[0],prefix5=values[5],terminal10=terminal,
            first_error=row['first_error'],secondary_errors=row['secondary_errors'],
            last_returned_step=row['own_returned'],selected_attempts=row['own_attempted'],warmup_returned=row['warmup_returned'],
            checkpoint_refs=[dict(step=t,png=x['png'],file_sha256=x['file_sha256'],pixel_sha256=x['pixel_sha256']) for t,x in sorted(cp.items())],
            missing_checkpoints=row['missing_checkpoints'],reported_native_success=recorded,
            native_terminal_flag_mismatch=None if terminal is None or recorded is None else terminal['native_success']!=recorded,
            prefix_retreat_native=difference(None if values[5] is None else values[5]['native_distance'],
                                             None if values[0] is None else values[0]['native_distance']),
            strict_non_greedy_vs_raw_zero=None,actual_RGB=dict(MISSING_RGB)))
    return dict(schema=NAME+'.report',status='saved_physical_comparison_not_admitted',rows=output,
        root_denominator=2,branch_denominator=8,descriptive=summarize(output),
        physical_counts=counts,physical_errors=report['errors'],
        physical_report_status=report['status'],inputs_final_verified=report['inputs_final_verified'],P1_admission=False,model_repaired=False,
        independent_confirmation=False,equal_total_FLOPs=False,actual_neural_calls=0,actual_physics_calls=0,
        counts_verification='durable_operation_attempt/completion_recount; not independent physics instrumentation')


REFS = ('physical_report','physical_prepare','physical_execution_binding','physical_journal','physical_source',
        'planner_report','planner_source','overlay_source_report','original_sourcepack','original_source_journal')


def exact(obj, fields):
    require(isinstance(obj,dict) and set(obj)==set(fields.split()),'exact explicit interface fields: '+fields)


def load_local(name):
    p=Path(name)
    require(p.is_absolute() and p.is_file() and not p.is_symlink() and p.stat().st_size<=64*1024**2,
            'actual absolute bounded artifact file')
    require(p.resolve()==p,'no symlink-parent/aliased input path')
    return p.read_bytes()


def material(binding, binding_sha, read=load_local):
    """Explicit external hashes; read-only, no producer imports or own-consume run."""
    data=read(binding);require(sha(data)==digest(binding_sha),'external comparison binding SHA')
    b=parse(data);exact(b,'schema files')
    require(b['schema']==NAME+'.binding','own binding schema, not producer schema')
    exact(b['files'],' '.join(REFS));blobs={}
    for name in REFS:
        ref=b['files'][name];exact(ref,'file sha256')
        require(isinstance(ref['file'],str) and Path(ref['file']).is_absolute(),'absolute actual artifact path')
        value=read(ref['file']);require(sha(value)==digest(ref['sha256']),'external artifact SHA: '+name)
        blobs[name]=value
    require(b['files']['original_sourcepack']['sha256']==SOURCEPACK_SHA,'fixed original moving task authority')
    report=parse(blobs['physical_report']);prepared=parse(blobs['physical_prepare'])
    refs=b['files'];ph=refs['physical_prepare']['sha256']
    require(report['prepare_sha256']==ph and report['planner_report_sha256']==refs['planner_report']['sha256'] and
        report['source_report_sha256']==refs['overlay_source_report']['sha256'],'physical report upstream external SHA binding')
    execution=parse(blobs['physical_execution_binding'])
    require(execution==dict(prepared=refs['physical_prepare']['file'],prepare_sha256=ph,
        planner_report_sha256=refs['planner_report']['sha256'],source_report_sha256=refs['overlay_source_report']['sha256'])
        and report['execution_binding']==execution and report['execution_binding_sha256']==refs['physical_execution_binding']['sha256'],
        'actual execution binding file + report equality')
    require(prepared['planner_report_sha256']==refs['planner_report']['sha256'] and
        prepared['source_report_sha256']==refs['overlay_source_report']['sha256'],'prepare upstream SHA')
    for ref in ('physical_source','planner_source','planner_report','overlay_source_report','original_sourcepack','original_source_journal'):
        require(prepared['bindings'].get(refs[ref]['file'])==refs[ref]['sha256'],'prepare frozen dependency '+ref)
    events=chain(blobs['physical_journal'],report['journal_chain_tail'])
    require(report['journal_sha256']==refs['physical_journal']['sha256'] and events[0]['prepare_sha256']==ph,
        'journal actual dispatch prepare file/SHA binding')
    pack=parse(blobs['original_sourcepack'])
    original_events=chain(blobs['original_source_journal'],pack['journal_chain_tail'])
    targets=source_targets(pack,original_events)
    plans=prepared['plans'];frozen=index_rows(plans)
    physical_rows=index_rows(report['rows'])
    overlay=parse(blobs['overlay_source_report'])
    require(overlay['schema']==OVERLAY+'.report' and overlay['status']=='paired_moving_overlay_recorded' and
        overlay['full_protocol'] is True and overlay['errors']==[],'actual paired overlay source')
    exports={}
    overlay_parent=Path(refs['overlay_source_report']['file']).parent
    for e in overlay['blind_exports']:
        s=e['source_slot'];require(type(s) is int and s in targets and s not in exports and
            e['old_root_sha256']==targets[s]['root_sha256'],'overlay to ORIGINAL root identity')
        exports[s]=e['artifacts']
    require(set(exports)=={0,1},'two paired root exports')
    planner=parse(blobs['planner_report'])
    require(planner['schema']==PLANNER+'.report' and planner['status']=='overlay_recompiled_plans_sealed_not_admitted' and
        planner['errors']==[] and planner['model_unchanged'] is True and planner['source_verified'] is True and
        planner['commands_released'] is True and planner['source_report_sha256']==refs['overlay_source_report']['sha256'],
        'actual successful NEW compiler report')
    planner_rows=index_rows(planner['rows']);planner_parent=Path(refs['planner_report']['file']).parent
    handoff=[];plan_hashes={}
    for k in CELLS:
        plan=frozen[k];s,c,a=k;e=exports[s][c]
        require(plan['blind_path']==str(overlay_parent/e['path']) and plan['blind_sha256']==e['sha256'] and
            prepared['bindings'].get(plan['blind_path'])==e['sha256'],'same-condition blind path and frozen SHA')
        require(sha(read(plan['blind_path']))==e['sha256'],'actual blind file SHA')
        if physical_rows[k]['initial_exact']:
            require(physical_rows[k]['goal_pixel_sha256']==e['goal_pixel_sha256'],
                    'same-condition attached goal RGB SHA from source, never terminal pixels')
        require(plan['source_report_sha256']==refs['overlay_source_report']['sha256'] and
            plan['planner_report_sha256']==refs['planner_report']['sha256'],'plan external source/compiler SHA')
        commands=np.asarray(plan['commands'])
        require(commands.shape==(10,2) and np.isfinite(commands).all() and np.max(np.abs(commands))<=1 and
            np.array_equal(commands,commands.astype(np.float32)),'literal full10 float32 commands, no repair')
        pr=planner_rows[k];name=pr['plan_file']
        require(Path(name).name==name and name not in ('','.','..'),'flat compiler plan artifact')
        payload=read(str(planner_parent/name));h=sha(payload);full=parse(payload)
        require(planner['artifact_sha256'][name]==h and pr['selection_digest']==plan['selection_digest']==full['selection_digest'] and
            full['schema']==PLANNER+'.plan' and sha(canonical({f:v for f,v in full.items() if f!='selection_digest'}))==full['selection_digest'],
            'actual compiler plan artifact/semantic selection seal')
        require(all(full[f]==v for f,v in plan.items() if f!='planner_report_sha256') and
            full['selection_uses_physics'] is False and full['P1_admission'] is False and full['with_target'] is (c=='on'),
            'physical commands frozen from compiler selection, no outcomes/reselection')
        handoff.append(dict(full,planner_report_sha256=refs['planner_report']['sha256']));plan_hashes[name]=h
    require(prepared['compiler_handoff_sha256']==sha(canonical(handoff)), 'full compiler consume handoff digest')
    parent=Path(refs['physical_report']['file']).parent
    frames={}
    def frame(name):
        require(isinstance(name,str) and Path(name).name==name and name not in ('','.','..'),'safe flat PNG')
        p=(parent/name).resolve();require(p.is_relative_to(parent.resolve()),'frame path containment')
        payload=read(str(p));frames[str(p)]=sha(payload);return payload
    out=assemble(report,prepared,events,targets,plans,frame)
    out['binding']=dict(file=binding,sha256=binding_sha,files=refs,frame_file_sha256=frames,
                        compiler_plan_sha256=plan_hashes,source_sha256=sha(Path(__file__).read_bytes()))
    out['planner_cost_receipt']=dict(contract=planner['contract'],per_branch=[{f:r[f] for f in
        ('root_slot','condition','arm','resources','AD_API_resources')} for r in planner['rows']])
    out['verification_scope']='source/selection/commands/physical-ledger/checkpoint binding; not a second full compiler tensor consume'
    out['cost_warning']='full reoptimization costs preserved; no equal-FLOPs/AD/physical-total claim'
    return out


def write_new(path,value):
    data=canonical(value)+b'\n';require(len(data)<=CAP,'2MiB comparison output cap')
    p=Path(path);require(shutil.disk_usage(p.parent).free>=FLOOR+CAP,'200MiB floor plus output reserve')
    with p.open('xb') as f:
        require(f.write(data)==len(data),'short output write');f.flush();os.fsync(f.fileno())
    return sha(data)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding',required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--output',required=True);a=p.parse_args(argv)
    result=material(a.binding,a.binding_sha)
    h=write_new(a.output,result)
    print(json.dumps(dict(path=a.output,sha256=h,status=result['status'],P1_admission=False)))
    return 0


if __name__=='__main__':raise SystemExit(main())
