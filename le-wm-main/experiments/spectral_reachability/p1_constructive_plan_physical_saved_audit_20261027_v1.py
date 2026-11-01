"""Independent saved-only physical protocol audit. No producer imports/calls.

No replay, neural code, simulator, witness controller or outcome-based selection.
An incomplete step has NO guaranteed post-fault frame: old observations are old.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import zlib
import numpy as np
from PIL import Image

NAME='p1_constructive_plan_physical_saved_audit_20261027_v1'
PRODUCER='p1_constructive_plan_physical_20261027_v1'
CAP=8*1024**2
KINDS=('reset','public_step','explicit_render','goal_render','set_state','physics_step','goal_bind')


def need(ok,message):
    if not ok:raise ValueError(message)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def sha(path):
    p=Path(path);need(p.is_file() and not p.is_symlink(),'regular nonsymlink file: '+str(p))
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def bound(path,expected):need(sha(path)==expected,'SHA mismatch: '+str(path));return Path(path)
def decode(data):
    def unique(pairs):
        d={}
        for k,v in pairs:need(k not in d,'duplicate JSON key');d[k]=v
        return d
    v=json.loads(data,object_pairs_hook=unique);json.dumps(v,allow_nan=False);return v
def inflate(data,limit):
    d=zlib.decompressobj(31);v=d.decompress(data,limit+1)
    need(len(v)<=limit and d.eof and not d.unused_data and not d.unconsumed_tail,'one bounded complete gzip stream');return v
def read(path,limit=CAP):
    p=Path(path);need(p.stat().st_size<=limit and not p.is_symlink(),'bounded JSON')
    b=p.read_bytes()
    if p.name.endswith('.gz'):b=inflate(b,limit)
    return decode(b)
def vector(value):
    a=np.asarray(value,dtype=np.float64);need(a.shape==(7,) and np.isfinite(a).all(),'finite state7');return a
def root_hash(state,goal):return hashlib.sha256(np.ascontiguousarray(np.r_[vector(state),vector(goal)],dtype=np.float64).tobytes()).hexdigest()
def pixel_sha(rgb):return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
def raw10(value):
    a=np.asarray(value);need(a.shape==(10,2) and a.dtype.kind in 'fi' and np.isfinite(a).all() and (abs(a)<=1).all() and np.array_equal(a,a.astype(np.float32)),'exact raw10x2');return a
def pointer(value,path):
    need(isinstance(path,str) and path.startswith('/'),'explicit pointer')
    for k in path[1:].split('/'):
        k=k.replace('~1','/').replace('~0','~');value=value[int(k)] if isinstance(value,list) else value[k]
    return value


def chain(data):
    need(len(data)<=CAP and data.endswith(b'\n'),'complete bounded journal lines')
    events=[decode(l) for l in data.splitlines()];previous=None
    for i,e in enumerate(events):
        need(e['index']==i and e['previous_sha256']==previous and e['event_sha256']==digest({k:v for k,v in e.items() if k!='event_sha256'}),'journal hashchain index '+str(i))
        previous=e['event_sha256']
    return events


def metrics(goal,state):
    d=vector(state)-vector(goal);a=float(abs(d[4]));native=min(a,2*np.pi-a)
    wrapped=float(abs(np.arctan2(np.sin(d[4]),np.cos(d[4]))));xy=float(np.linalg.norm(d[:4]))
    return dict(native_success=bool(xy<20 and native<np.pi/9),native_distance=float(np.linalg.norm(d)),native_angle_term=float(native),
        negative_native_angle=bool(native<0),wrapped_allxy_success=bool(xy<20 and wrapped<np.pi/9),wrapped_object_angle_error=wrapped,
        object_position_error=float(np.linalg.norm(d[2:4])),object_pose=vector(state)[2:5].tolist(),object_success=None)


def references(manifest):
    """Evaluator-only white-listed source fields; never read certificate actions."""
    pack_path=bound(manifest['sourcepack'],manifest['sourcepack_sha256']);pack=read(pack_path)
    prep=read(bound(manifest['source_prepared'],manifest['source_prepare_sha256']),200000)
    p=decode(inflate(base64.b64decode(prep['gzip_base64'],validate=True),2*1024**2))
    need(digest(p)==prep['plan_sha256']==pack['prepare_id'],'source prepare semantic binding')
    need(pack['sourcepack_complete'] is True and pack['errors']==[] and pack['reachable_count']==2 and len(pack['rows'])==2,'complete audited two-root source')
    ev=chain((pack_path.parent/'journal.jsonl').read_bytes())
    need(ev[-1]['event_sha256']==pack['journal_chain_tail'] and ev[-1]['kind']=='run_finished','source chain tail')
    refs=[]
    for i,row in enumerate(pack['rows']):
        need(row['slot']==p['slots'][i] and row['slot']['index']==i and row['reachable'] is True,'source slot/recipe')
        with np.load(bound(pack_path.parent/row['certificate'],row['certificate_sha256']),allow_pickle=False) as z:
            cert={k:z[k].copy() for k in ('goal','goal_rgb','observed_states','root_history')}
        with np.load(bound(pack_path.parent/row['blind_payload'],row['blind_payload_sha256']),allow_pickle=False) as z:
            need(set(z.files)=={'rgb_history','raw_action_history','goal_rgb'},'blind whitelist')
            history=z['rgb_history'].copy();goal_rgb=z['goal_rgb'].copy();actions=z['raw_action_history'].copy()
        need(history.shape==(3,224,224,3) and history.dtype==np.uint8 and actions.shape==(10,2) and not actions.any(),'actual zero full history')
        observations=[e for e in ev if e['kind']=='fresh_observation' and e['slot']==i and e['phase']=='replay0' and e['step'] in (0,5,10)]
        placement=[e for e in ev if e['kind']=='placement_recipe' and e['slot']==row['slot'] and e['phase']=='replay0']
        need(len(placement)==1 and [e['step'] for e in observations]==[0,5,10],'source replay0 reference receipts')
        need(np.array_equal(history,cert['root_history']) and np.array_equal(goal_rgb,cert['goal_rgb']) and
            np.array_equal([e['state'] for e in observations],cert['observed_states']) and root_hash(cert['observed_states'][-1],cert['goal'])==row['root_sha256'],'source state/goal/RGB identity')
        refs.append(dict(slot=row['slot'],root_sha256=row['root_sha256'],blind_sha256=row['blind_payload_sha256'],history=history,
            goal=cert['goal'],goal_rgb=goal_rgb,states=cert['observed_states'],snapshots=[e['snapshot'] for e in observations],placement=placement[0]))
    return refs


def inspect(report,prepared,events,refs,load_png):
    """Pure saved dictionaries/arrays core, no producer functions. All findings retained."""
    findings=[];summaries=[]
    def check(ok,label):
        if not ok:findings.append(label)
    plans=prepared['plans'];rows=report['rows'];complete=report['status']=='physical_calibration_recorded'
    check(len(refs)==2 and len(rows)==len(plans),'full planned denominator')
    check(report['contract']==prepared['contract'],'report/prepare contract')
    n=len(plans)
    check(prepared['contract']['caps']==dict(reset=2*n,public_step=20*n,explicit_render=22*n,goal_render=2*n,set_state=4*n,physics_step=204*n,goal_bind=n),'independently fixed caps')
    check(events[0]['kind']=='run_started' and events[0]['prepare']==prepared,'run_started full prepare')
    check(events[-1]['kind']=='run_finished','terminal journal event')
    if events[-1]['kind']=='run_finished':
        check(events[-1]['counts']==report['counts'] and events[-1]['status']==report['status'] and events[-1]['errors']==report['errors'],'terminal report/journal agreement')
    counts={k:dict(attempted=0,completed=0) for k in KINDS};stacks={}
    for e in events:
        if e['kind'] not in ('operation_attempted','operation_completed'):continue
        k=e['resource'];check(k in KINDS,'known operation resource')
        if k not in KINDS:continue
        counts[k]['attempted' if e['kind']=='operation_attempted' else 'completed']+=1
        signature={k:v for k,v in e.items() if k not in ('kind','index','previous_sha256','event_sha256')}
        stack=stacks.setdefault(e['branch'],[])
        if e['kind']=='operation_attempted':stack.append(signature)
        else:
            check(bool(stack) and stack[-1]==signature,'LIFO operation receipt branch '+str(e['branch']))
            if stack and stack[-1]==signature:stack.pop()
    check(counts==report['counts'],'journal actual attempt/completion census')
    for k,v in counts.items():check(v['completed']<=v['attempted']<=prepared['contract']['caps'][k],'resource cap '+k)
    for index,(plan,row) in enumerate(zip(plans,rows)):
        label=f'branch{index} ';slot=plan['root_slot'];ref=refs[slot];cmd=raw10(plan['commands'])
        check((row['root_slot'],row['arm'],row['selection_digest'])==(slot,plan['arm'],plan['selection_digest']),label+'fixed plan identity')
        be=[e for e in events if e.get('branch')==index]
        for e in be:check(e['root_slot']==slot and e['arm']==plan['arm'] and e['selection_digest']==plan['selection_digest'],label+'event identity')
        for pos,e in enumerate(be):
            if e['kind']!='operation_attempted' or e['resource'] not in ('reset','public_step','explicit_render'):continue
            depth=1;nested=Counter();finished=False
            for sub in be[pos+1:]:
                if sub['kind']=='operation_attempted':depth+=1;nested[sub['resource']]+=1
                elif sub['kind']=='operation_completed':
                    depth-=1
                    if depth==0:finished=True;break
            if finished:
                expected=Counter(dict(physics_step=2,set_state=2,goal_render=1)) if e['resource']=='reset' else Counter(dict(physics_step=10)) if e['resource']=='public_step' else Counter()
                check(nested==expected,label+'per-dispatch nested native counts '+e['resource'])
        starts=[e for e in be if e['kind']=='branch_started']
        check(len(starts)==(0 if row['status']=='not_started' else 1),label+'branch start denominator')
        if starts:check(starts[0]['plan']==plan,label+'own preselected commands')
        attempted=[e for e in be if e['kind']=='operation_attempted' and e['resource']=='public_step']
        transitions=[e for e in be if e['kind']=='returned_transition']
        observations=[e for e in be if e['kind']=='fresh_observation'];invalid=[e for e in be if e['kind']=='invalid_render']
        guards=[e for e in be if e['kind']=='guard_receipt'];closed=[e for e in events if e['kind']=='branch_closed' and e['receipt']['root_slot']==slot and e['receipt']['arm']==plan['arm']]
        if closed:check(len(closed)==1 and closed[0]['receipt']==row,label+'closed row exact')
        if complete:check(len(closed)==1 and not stacks.get(index),label+'complete branch closed/no open dispatch')
        for phase in ('warmup','plan'):
            ae=[e for e in attempted if e['phase']==phase];te=[e for e in transitions if e['phase']==phase]
            check([e['step'] for e in ae]==list(range(1,len(ae)+1)),label+phase+' attempted contiguous')
            check([e['step'] for e in te]==list(range(1,len(te)+1)),label+phase+' returned contiguous')
            for e in ae:check(np.array_equal(e['raw_action'],[0,0] if phase=='warmup' else cmd[e['step']-1]),label+'raw dispatched commands')
            for e in te:
                matching=[a for a in ae if a['step']==e['step'] and a['index']<e['index']]
                check(len(matching)==1,label+'returned transition own dispatch')
                if phase=='plan':check(np.array_equal(e['command'],cmd[e['step']-1]),label+'returned own commands')
                goal=ref['goal'] if phase=='plan' else ref['placement']['original_goal'];m=metrics(goal,e['state'])
                check(e['metrics']==m and float(e['reward'])==-m['native_distance'] and bool(e['terminated'])==m['native_success'],label+'native reward/metric/terminal')
                if phase=='plan':check(e['actual_native_success']==m['native_success'] and e['actual_native_distance']==m['native_distance'] and
                    np.array_equal(e['returned_goal_state'],ref['goal']) and e['returned_goal_rgb_sha256']==pixel_sha(ref['goal_rgb']),label+'actual native and goal binding')
                if e['terminated'] or e['truncated']:check(not any(a['index']>e['index'] for a in attempted),label+'no dispatch after native stop')
            check(row['warmup_returned' if phase=='warmup' else 'own_returned']==len(te),label+phase+' returned denominator')
        check(row['own_attempted']==sum(e['phase']=='plan' for e in attempted),label+'own attempted denominator')
        frames={};receipts={}
        for e in observations:
            r=e['receipt'];key=(r['phase'],r['step']);check(key not in frames,label+'unique frame step')
            rgb=load_png(r);frames[key]=rgb;receipts[key]=r
            check(rgb.shape==(224,224,3) and rgb.dtype==np.uint8 and pixel_sha(rgb)==r['pixel_sha256'],label+'PNG pixels')
            check(r['snapshot']['state']==r['state'],label+'snapshot/state')
            gs=[g for g in guards if (g['phase'],g['step'])==key]
            check(len(gs)==1 and gs[0]['index']>e['index'],label+'persisted observation before guard receipt')
            if gs:
                g=gs[0];s=vector(r['state']);pixels=int(np.count_nonzero(np.all(rgb==[65,105,225],axis=-1)))
                same=np.array_equal(s,r['observation_state']);valid=pixels>0 and bool(((s[:4]>=0)&(s[:4]<=512)).all()) and same and g['render_preserved_dynamics']
                check(g['canonical_agent_pixels']==pixels and g['returned_state_equal']==bool(same) and g['accepted']==valid and g['png']==r['png'] and g['file_sha256']==r['file_sha256'],label+'independent guard recomputation')
            if key[1]>0:
                ts=[t for t in transitions if (t['phase'],t['step'])==key]
                check(len(ts)==1 and ts[0]['index']<e['index'] and np.array_equal(ts[0]['state'],r['observation_state']),label+'transition/frame chain')
        check(row['PNG']==len(observations),label+'actual PNG count')
        for g in guards:
            if not g['accepted']:check(not any(a['index']>g['index'] for a in be if a['kind']=='operation_attempted'),label+'no dispatch after rejected guard')
        last=max(observations+invalid,key=lambda e:e['index'])['receipt'] if observations or invalid else None
        if last is not None:check(row.get('last_observation')==last,label+'last observation exact, not reconstructed')
        missing_returns=[a for a in attempted if not any(t['phase']==a['phase'] and t['step']==a['step'] and t['index']>a['index'] for t in transitions)]
        missing_frames=[t for t in transitions if not any(e['receipt']['phase']==t['phase'] and e['receipt']['step']==t['step'] and e['index']>t['index'] for e in observations+invalid)]
        scene='unknown_after_incomplete_step' if missing_returns else 'unknown_after_unobserved_transition' if missing_frames else 'saved_rejected_observation' if any(not g['accepted'] for g in guards) else 'last_recorded_observation'
        if complete:check(not missing_frames,label+'fresh scene for every completed transition')
        if row['initial_exact']:
            for step,j in ((0,0),(5,1),(10,2)):
                r=receipts.get(('warmup',step));check(r is not None,label+'blind warmup checkpoint')
                if r:check(np.array_equal(frames['warmup',step],ref['history'][j]) and np.array_equal(r['state'],ref['states'][j]) and r['snapshot']==ref['snapshots'][j],label+'exact blind state/RGB/dynamics')
            r=receipts.get(('plan',0));check(r is not None,label+'postattach root frame')
            if r:check(np.array_equal(frames['plan',0],ref['history'][-1]) and np.array_equal(r['state'],ref['states'][-1]) and r['snapshot']==ref['snapshots'][-1],label+'exact unchanged attached root')
            check(root_hash(row['root_state'],row['goal'])==ref['root_sha256'] and np.array_equal(row['goal'],ref['goal']) and row['initial_metrics']==metrics(ref['goal'],ref['states'][-1]),label+'exact root/goal/initial metric')
            attached=[e for e in be if e['kind']=='goal_attached']
            check(len(attached)==1 and np.array_equal(attached[0]['goal'],ref['goal']) and attached[0]['goal_rgb_sha256']==pixel_sha(ref['goal_rgb']) and attached[0]['snapshot']==ref['snapshots'][-1],label+'goal attachment receipt')
            binds=[e for e in be if e['kind'] in ('operation_attempted','operation_completed') and e['resource']=='goal_bind']
            check(len(binds)==2 and binds[0]['kind']=='operation_attempted' and binds[1]['kind']=='operation_completed',label+'one goal bind')
            if len(binds)==2:check(not any(e['kind']=='operation_attempted' and binds[0]['index']<e['index']<binds[1]['index'] for e in be),label+'goal attachment no logged physics/render/setter/reset')
        reset=[e for e in be if e['kind']=='operation_attempted' and e['resource']=='reset']
        if row['initial_exact'] or complete:check([(e['kind_of_reset'],e['seed']) for e in reset]==[('original_seed',ref['slot']['reset_seed']),('placement',0)],label+'original two-reset recipe')
        placement=[e for e in be if e['kind']=='placement_request']
        if placement:check(len(placement)==1 and placement[0]['original_goal']==ref['placement']['original_goal'] and placement[0]['requested']==ref['placement']['requested'] and placement[0]['placement_seed']==0,label+'original recipe requested state, no saved poststate reset')
        own=[t for t in transitions if t['phase']=='plan']
        if own:
            last_return=own[-1];m=metrics(ref['goal'],last_return['state'])
            check(row['final_state']==last_return['state'] and row['final_metrics']==m and row['native_success']==m['native_success'] and row['terminated']==last_return['terminated'] and row['truncated']==last_return['truncated'],label+'reported endpoint native/object')
        if complete:
            check(row['first_error'] is None and row['secondary_errors']==[] and row['status'] in ('horizon_completed','native_success','native_stopped') and row['initial_exact'],label+'complete row protocol')
            check(row['full_horizon']==(len(own)==10) and bool(own) and (len(own)==10 or own[-1]['terminated'] or own[-1]['truncated']),label+'honor horizon or native stop')
            expected_cp=[receipts['plan',s] for s in (0,5,10) if ('plan',s) in receipts]
            check(row['checkpoints']==expected_cp,label+'checkpoint receipts')
            if own:
                check(row['object_position_error_change']==m['object_position_error']-row['initial_metrics']['object_position_error'] and row['object_angle_error_change']==m['wrapped_object_angle_error']-row['initial_metrics']['wrapped_object_angle_error'],label+'object changes')
        endpoints={s:metrics(ref['goal'],receipts['plan',s]['state']) for s in (0,5,10) if ('plan',s) in receipts}
        summaries.append(dict(root_slot=slot,arm=plan['arm'],status=row['status'],own_returned=len(own),PNG=len(observations),
            post_step_scene=scene,last_observation_role='pre_failure_last_observation_only' if missing_returns or missing_frames else 'recorded_not_inferred',
            incomplete_steps=[dict(phase=a['phase'],step=a['step']) for a in missing_returns],checkpoint_metrics=endpoints,
            object_displacement=float(np.linalg.norm(vector(own[-1]['state'])[2:4]-vector(ref['states'][-1])[2:4])) if own else None,
            final_metrics=metrics(ref['goal'],own[-1]['state']) if own else None))
    if complete:
        n=len(rows);public=sum(r['warmup_returned']+r['own_returned'] for r in rows)
        expected=dict(reset=2*n,set_state=4*n,goal_render=2*n,goal_bind=n,public_step=public,physics_step=4*n+10*public,explicit_render=sum(r['PNG'] for r in rows))
        check(counts=={k:dict(attempted=v,completed=v) for k,v in expected.items()},'independent exact full native census')
        check(report['errors']==[] and report['native_successes']==sum(r['status']=='native_success' for r in rows),'complete report errors/success count')
    check(report['new_neural_calls']==0 and report['training'] is False and report['P1_admission'] is False and report['new_independent_samples']==0 and report['global_freshness']=='unknown_calibration_only','no scientific admission/sample inflation')
    return dict(schema=NAME+'.audit',status='saved_audit_failed' if findings else 'complete_saved_protocol_verified' if complete else 'partial_saved_no_success_claim',
        findings=findings,rows=summaries,counts=counts,new_model_calls=0,new_physics=0,P1_admission=False,
        scope='saved protocol/bindings and independent numeric recomputation, NOT simulator replay, blind planner selection math, or scientific efficacy',
        limitations=['hook receipts attest calls; offline audit cannot prove native dispatch beyond saved evidence',
            'pre-render snapshot is not saved: render_preserved_dynamics flag is an attestation, not independently replayed',
            'step internally throwing has no guaranteed fresh post-fault scene; older last_observation is not incident imagery'])


def audit(report_path,report_sha,prepared_path,prepared_sha):
    rp=bound(report_path,report_sha);pp=bound(prepared_path,prepared_sha);report=read(rp);p=read(pp,256*1024)
    need(report['schema']==PRODUCER+'.report' and p['schema']==PRODUCER+'.prepare','physical report/prepare schema')
    need(read(rp.parent/'execution_binding.json')==dict(prepared=str(pp.resolve()),prepare_sha256=prepared_sha),'actual external execution binding')
    for path,h in p['bindings'].items():bound(path,h)
    manifest=read(bound(p['manifest'],p['manifest_sha256']),256*1024);need(manifest['plans']==p['plans'],'selection manifest plans')
    refs=references(manifest)
    need(p['source_slots']==manifest['roots']==[dict(root_slot=i,root_sha256=r['root_sha256']) for i,r in enumerate(refs)] and
        report['sourcepack_sha256']==p['sourcepack_sha256']==manifest['sourcepack_sha256'],'exact evaluator source identity')
    need([(r['root_slot'],r['arm']) for r in p['plans']]==[(i,a) for i in range(2) for a in manifest['arms']],'fixed full root/arm Cartesian order')
    for plan in p['plans']:
        need(plan['selection_digest']==digest({k:v for k,v in plan.items() if k!='selection_digest'}),'own selection digest')
        h=read(bound(plan['compiler_report'],plan['compiler_report_SHA']),256*1024)
        need(h['schema']==plan['compiler_schema'] and h['status']==plan['compiler_status'] and h['errors']==[],'declared handoff release')
        need(pointer(h,plan['commands_pointer'])==plan['commands'] and pointer(h,plan['arm_pointer'])==plan['arm'] and
            type(pointer(h,plan['source_slot_pointer'])) is int and pointer(h,plan['source_slot_pointer'])==plan['root_slot'] and
            pointer(h,plan['blind_sha_pointer'])==refs[plan['root_slot']]['blind_sha256'],'SHA-bound own selected plan bridge')
        # Bind original upstream receipt without calling consume/producer verification.
        if 'planner_report_sha256' in h:
            original=bound(Path(h['planner_output'])/'report.json',h['planner_report_sha256']);pr=read(original)
            need(pr['errors']==[] and pr['commands_released'] is True and pr['model_unchanged'] is True,'upstream whole-run release')
            for path,v in h['task_bindings'].items():bound(path,v)
            for path,v in h['bridge_bindings'].items():bound(path,v)
    jp=bound(rp.parent/'journal.jsonl',report['journal_sha256']);events=chain(jp.read_bytes());used={'report.json','execution_binding.json','journal.jsonl'}
    def load_png(receipt):
        name=receipt['png'];need(Path(name).name==name,'flat PNG path');path=bound(rp.parent/name,receipt['file_sha256']);used.add(name)
        with Image.open(path) as im:need(im.format=='PNG' and im.mode=='RGB','native RGB PNG');return np.asarray(im).copy()
    for e in events:
        if e['kind']=='invalid_render' and e['receipt'].get('raw_render'):
            name=e['receipt']['raw_render'];need(Path(name).name==name,'flat raw incident artifact');bound(rp.parent/name,e['receipt']['file_sha256']);used.add(name)
    result=inspect(report,p,events,refs,load_png)
    actual={x.name for x in rp.parent.iterdir()};need(actual==used,'exact artifact census, no unexplained files/subdirectories')
    need(sum(x.stat().st_size for x in rp.parent.iterdir())<=CAP,'8MiB actual output cap')
    need(sha(rp)==report_sha and sha(pp)==prepared_sha and sha(jp)==report['journal_sha256'],'final input seal stability')
    result.update(report=str(rp.resolve()),report_sha256=report_sha,prepared=str(pp.resolve()),prepare_sha256=prepared_sha,PNG=sum(r['PNG'] for r in result['rows']))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('report','report-sha','prepared','prepare-sha'):p.add_argument('--'+k,required=True)
    p.add_argument('--output');a=p.parse_args()
    r=audit(a.report,a.report_sha,a.prepared,a.prepare_sha);b=(json.dumps(r,sort_keys=True,allow_nan=False)+'\n').encode()
    if a.output:
        out=Path(a.output);need(len(b)<=150*1024 and not out.exists() and shutil.disk_usage(out.parent).free>=200*1024**2+len(b),'NEW <=150KiB +200MiB reserve')
        need(out.parent.resolve()!=Path(a.report).parent.resolve(),'audit output must be outside sealed physical output')
        with out.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    else:print(b.decode(),end='')
    if r['findings']:raise SystemExit(2)


if __name__=='__main__':main()
