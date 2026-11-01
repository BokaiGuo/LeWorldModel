"""Independent bounded saved physical audit. No consumer/runner runtime import.

Standard library and PNG decoding only; no physics, neural, NumPy, GPU, search
or rerun. Source seals bind bytes, not independently observed simulator truth.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from io import BytesIO
import math
import os
import struct
from pathlib import Path

from experiments.spectral_reachability import p3_quality_cost_audit_20261027_v2 as io
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal

NAME='p2_grasp_h1_saved_outcome_audit_20261027_v1'
CONSUMER='p2_grasp_h1_guarded_physical_20261027_v1'
PREPARED='results/'+CONSUMER+'_prepared'
PREPARE_SHA='112cabb0684b7a16f3213a6b64759313c3711330571b2d4e6e38befa053399e8'
PHYSICAL='results/'+CONSUMER+'_main_v1/report.json'
PHYSICAL_SHA='1fa5bf810109ced7c36813d46915a842ed62b7dfe798d397eaf5fa93e60a5c62'
METHODS=('grasp_h1','action_adam_h1')
require=io.require
def same(a,b):return io.canonical(a)==io.canonical(b)


def finite(x):
    require(type(x) in (int,float) and math.isfinite(x),'finite scalar required')
    return x


def vector(x,n):
    require(isinstance(x,list) and len(x)==n,'aligned vector required')
    return [finite(v) for v in x]


def close(a,b):
    if isinstance(b,dict):return isinstance(a,dict) and set(a)==set(b) and all(close(a[k],v) for k,v in b.items())
    if type(b) is bool:return type(a) is bool and a==b
    return type(a) in (int,float) and math.isfinite(a) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10)


def labels(state,goal):
    s,g=vector(state,7),vector(goal,7)
    norm=lambda ids:math.hypot(*(s[i]-g[i] for i in ids))
    xy=norm((2,3));angle=abs(math.atan2(math.sin(s[4]-g[4]),math.cos(s[4]-g[4])))
    diff=abs(s[4]-g[4]);official_angle=min(diff,2*math.pi-diff);pos=norm(range(4))
    return dict(pose=dict(xy_distance=xy,angle_distance=angle,pose_cost=xy+20*angle),
        official=dict(official_position_error=pos,agent_position_error=norm((0,1)),object_position_error=xy,
            official_state_distance=norm(range(7)),official_success=pos<20 and official_angle<math.pi/9,
            object_only_success=xy<20 and official_angle<math.pi/9,angle_error=official_angle))


def load_bundle(root):
    root=Path(root).resolve();e=io.Evidence(root)
    receipt=e.json(root/PREPARED/'prepare.json',PREPARE_SHA)
    p=e.json(root/PREPARED/'protocol.json',receipt['protocol_sha256'])
    require(receipt['status']=='prepared_not_executed' and receipt['model_loaded'] is False,'frozen physical prepare')
    for path,h in p['inputs_sha256'].items():e.raw(Path(path),h)
    for seal in receipt['source_seals']:e.seal(seal)
    # The large inherited planning journal is streamed, not loaded/truncated.
    import hashlib
    path=Path(p['planning_journal_path']);require(path.is_relative_to(root) and not path.is_symlink(),'planning journal path')
    require(path.stat().st_size<=128*1024*1024,'bounded planning journal')
    h=hashlib.sha256();size=0
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block);size+=len(block)
    require(h.hexdigest()==p['planning_journal']['sha256'] and size==p['planning_journal']['bytes'],'inherited planning journal hash/size')
    report=e.json(root/PHYSICAL,PHYSICAL_SHA);directory=(root/PHYSICAL).parent
    started=e.json(directory/'started.json');setup=[io.decode(x) for x in e.raw(directory/'setup.events.jsonl').splitlines()]
    require(len(setup)<=10000,'bounded setup journal')
    disks={};events={};pixels={};budgets={};endpoints={}
    for plan in p['plans']:
        key=plan['root_sha256'],plan['method'];stem=f'root_{plan["root_index"]}_{plan["method"]}'
        if plan['dispatch_status']!='eligible':
            disks[key]=e.json(directory/(stem+'.rejected.json'))
            require(not any((directory/name).exists() for name in (stem+'_replay0.events.jsonl',stem+'.budget.json',stem+'.endpoint.json',stem+'_replay0.physical.json')) and
                not any(directory.glob(stem+'_replay0_rgb_*.png')),'rejected branch acquired dispatch artifacts')
            continue
        disks[key]=e.json(directory/(stem+'_replay0.physical.json'))
        endpoints[key]=e.json(directory/(stem+'.endpoint.json'));budgets[key]=e.json(directory/(stem+'.budget.json'))
        row=io.index(report['rows'],('root_sha256','method'))[key]
        journal=directory/(stem+'_replay0.events.jsonl')
        require(row['journal_path']==str(journal),'fixed physical journal path')
        events[key]=[io.decode(line) for line in e.raw(journal,row['journal_sha256']).splitlines()]
        require(len(events[key])<=46,'bounded fifteen-step journal');pixels[key]=[]
        from PIL import Image
        for event in events[key]:
            if 'rgb_path' not in event:continue
            step=event.get('primitive_step',0);require(type(step) is int and 0<=step<=15,'RGB step')
            image_path=directory/f'{stem}_replay0_rgb_{step}.png'
            require(event['rgb_path']==str(image_path),'fixed saved image path')
            raw=e.raw(image_path,event['rgb_file_sha256'])
            with Image.open(BytesIO(raw)) as im:
                require(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'bounded canonical PNG')
                count=sum(pixel==(65,105,225) for pixel in im.getdata())
            pixels[key].append(dict(primitive_step=step,canonical_agent_pixels=count))
    e.recheck()
    return dict(protocol=p,report=report,started=started,setup=setup,disks=disks,events=events,pixels=pixels,budgets=budgets,endpoints=endpoints),e


def inspect_branch(plan,row,events,pixels,packet):
    require(all(same(row[k],plan[k]) for k in ('root_sha256','root_index','method','selection_sha256','commands')),'root/action selection binding')
    require(row['planned_commands']==15 and row['planned_horizon_tokens']==3 and type(row['repetition']) is int and row['repetition']==0,'fifteen commands one repetition')
    a=io.count(row['physical_env_step_attempts']);b=io.count(row['physical_env_step_returns'])
    require(a==row['attempted_commands'] and b==row['completed_commands'] and b<=a<=15,'step attempt/return accounting')
    if plan['dispatch_status']!='eligible':
        require(row['status']=='rejected_raw_out_of_box' and a==b==0 and not row['endpoint_fresh'] and not row['full_horizon_completed'] and
            row['trace']==[] and row['terminal_rgb_path'] is None,'rejected zero dispatch/endpoint')
        return
    ctx=row['context_identity']
    require(all(ctx[k]==packet[k] for k in ('root_sha256','history_rgb_sha256','goal_rgb_sha256','input_sha256')),'source RGB/root/goal0 context binding')
    require(ctx['normalized_between_history']==packet['history_actions'] and ctx['history']==[[0.,0.]]*3,'actual between-frame action history')
    require(events and events[0]['kind']=='initial' and same(events[0]['state'],ctx['state']),'initial decision-state journal identity')
    na=nr=0;state='initial';seen=[];obs=[];visible=None;terminated=False
    for i,event in enumerate(events):
        kind=event['kind']
        if kind in ('initial','observation'):
            step=event.get('primitive_step',0)
            if kind=='initial':require(i==0,'one initial observation')
            else:
                require(state=='await_observation' and step==nr,'observation/return order')
                require(same(event['command'],plan['commands'][step-1]),'observed raw command')
                require(type(event['terminated']) is bool and type(event['truncated']) is bool,'termination flag type')
                terminated=event['terminated'] or event['truncated'];obs.append(event)
            require(len(seen)<len(pixels) and pixels[len(seen)]['primitive_step']==step,'RGB pixel journal order')
            visible=io.count(pixels[len(seen)]['canonical_agent_pixels']);require(visible<=224*224,'pixel count bound')
            seen.append(event);state='ready'
        elif kind=='attempt':
            require(state=='ready' and visible is not None and visible>0 and not terminated,'dispatch without fresh visible RGB')
            na+=1;require(na<=15 and event['primitive_step']==na and same(event['command'],plan['commands'][na-1]),'attempt order/raw path')
            state='await_return'
        elif kind=='step_returned':
            require(state=='await_return' and event['primitive_step']==na,'return order')
            nr+=1;state='await_observation'
        else:raise ValueError('unknown physical journal event')
    require(na==a and nr==b and len(seen)==len(pixels),'journal attempt/return/image census')
    require(same([{k:v for k,v in x.items() if k not in ('rgb_path','rgb_file_sha256')} for x in obs],row['trace']),'trace/journal equality')
    require(row['remaining_commands_not_executed']==15-b and row['actual_endpoint_primitive_steps']==b,'remaining/endpoint counters')
    guard=row['guard'];require(guard['root_id']==plan['root_sha256'] and same(guard['observations'],pixels) and guard['primitive_commands_authorized']==a,'independent guard receipt')
    if visible==0:require(row['status']=='observation_unavailable' and guard['abstained'] is True,'guard abort retained')
    if row['status']=='completed':require(b==15 and visible>0 and not terminated and state=='ready' and not guard['abstained'],'full completion guard')
    if row['status']=='environment_terminated':require(terminated,'termination retained')
    fresh=bool(seen and a==b and seen[-1].get('primitive_step',0)==b and row['status'] in ('completed','observation_unavailable','environment_terminated'))
    full=fresh and row['status']=='completed' and b==15
    scope='full_15' if full else 'guard_partial_or_terminated' if fresh else 'unknown_or_stale_endpoint'
    require(row['endpoint_fresh'] is fresh and row['full_horizon_completed'] is full and row['endpoint_scope']==scope,'full/partial/fresh classification')
    require(seen[-1]['rgb_path']==row['terminal_rgb_path'] and seen[-1]['rgb_file_sha256']==row['terminal_rgb_sha256'] and
        same(seen[-1]['state'],row['final_state']),'endpoint image/state identity')
    expected=labels(row['final_state'],ctx['goal'])
    require(close(row['pose'],expected['pose']) and close(row['official'],expected['official']),'independent pose/official arithmetic')


def setup_budget(events,plans):
    require(len(events)<=10000,'bounded setup events');steps={};resets={};branches={};pending=None
    by=io.index(plans,('root_sha256','method'))
    for seq,e in enumerate(events):
        require(e['sequence']==seq,'setup sequence')
        kind=e['kind'];phase=e['phase'];require(phase in ('root_collection','warmup','selected'),'known setup phase')
        require(kind in ('env_step_attempt','env_step_returned','env_reset_attempt','env_reset_returned'),'unknown/blocked setup event')
        key=(e.get('root_sha256'),e.get('method'))
        if phase!='root_collection':require(key in by and by[key]['dispatch_status']=='eligible','setup activity on rejected/foreign root-arm')
        op='step' if kind.startswith('env_step') else 'reset';table=steps if op=='step' else resets
        c=table.setdefault(phase,dict(attempts=0,returns=0))
        if kind.endswith('_attempt'):
            require(pending is None,'setup attempt before previous return');pending=(op,phase,key)
            c['attempts']+=1
            if op=='step':
                cmd=vector(e['command'],2)
                bc=branches.setdefault(key,{}).setdefault(phase,dict(attempts=0,returns=0));bc['attempts']+=1
                if phase=='warmup':require(cmd==[0.,0.],'warmup must remain raw zero')
                if phase=='selected':require(bc['attempts']<=15 and same(cmd,by[key]['commands'][bc['attempts']-1]),'meter selected raw path')
        else:
            require(pending==(op,phase,key),'setup return root/phase/order');pending=None;c['returns']+=1
            if op=='step':branches[key][phase]['returns']+=1
    return steps,resets,branches,pending


def audit(b):
    p=b['protocol'];r=b['report'];roots=p['roots']
    require(p['schema']==CONSUMER+'.protocol' and r['schema']==CONSUMER+'.report' and r['status']=='guarded_calibration_recorded','completed actual saved physical schemas')
    require(r['consumer_prepare_sha256']==PREPARE_SHA and r['planning_report_sha256']==p['planning_report_sha256'] and r['source_verified'] is True and r['failures']==[],'source/execution receipt')
    require(b['started']['prepare_sha256']==PREPARE_SHA and same(b['started']['plans'],p['plans']) and b['started']['new_predict_samples']==0,'started plan census')
    require(len(roots)==len(set(roots))==4,'four unique calibration roots')
    plans=io.index(p['plans'],('root_sha256','method'));rows=io.index(r['rows'],('root_sha256','method'));packets=io.index(p['packets'],('root_sha256',))
    require(set(plans)==set(rows)=={(root,m) for root in roots for m in METHODS},'exact eight arm census')
    eligible=0
    for key,plan in plans.items():
        require(io.sha(io.canonical({k:v for k,v in plan.items() if k!='selection_sha256'}).encode())==plan['selection_sha256'],'selection SHA')
        require(io.sha(io.canonical(plan['commands']).encode())==plan['command_sha256'],'command SHA')
        packet=packets[key[0],];require(plan['input_sha256']==packet['input_sha256'] and plan['goal_rgb_sha256']==packet['goal_rgb_sha256'],'plan goal0/context identity')
        require(io.sha(io.canonical({k:v for k,v in packet.items() if k!='input_sha256'}).encode())==packet['input_sha256'],'source packet content hash')
        supported=all(abs(finite(x))<=1 for cmd in plan['commands'] for x in vector(cmd,2))
        f32=lambda x:struct.unpack('f',struct.pack('f',x))[0]
        means=[f32(-.0087),f32(.0068)];stds=[f32(.2019),f32(.2002)]
        a=plan['normalized_actions'];require(isinstance(a,list) and len(a)==3,'three normalized tokens')
        raw=[[token[2*j+i]*stds[i]+means[i] for i in range(2)] for token in [vector(t,10) for t in a] for j in range(5)]
        require(same(raw,plan['commands']),'independent normalized-to-raw inversion')
        require(len(plan['commands'])==15 and plan['clipped'] is False and plan['physical_box_supported'] is supported and
            plan['dispatch_status']==('eligible' if supported else 'rejected_raw_out_of_box'),'raw support classification')
        row=rows[key]
        inspect_branch(plan,row,b['events'].get(key,[]),b['pixels'].get(key,[]),packet)
        if not supported:require(same(row,b['disks'][key]),'rejected durable file equality');continue
        eligible+=1
        require(same({k:v for k,v in row.items() if k!='environment_step_counts'},b['endpoints'][key]),'endpoint annotation/report equality')
        require(same({k:v for k,v in b['endpoints'][key].items() if k not in ('commands','endpoint_fresh','endpoint_scope')},b['disks'][key]),'durable physical/report equality')
    steps,resets,branches,pending=setup_budget(b['setup'],p['plans'])
    require(set(branches)=={key for key,plan in plans.items() if plan['dispatch_status']=='eligible'},'all eligible meter root-arm coverage')
    require(pending is None,'unreturned setup attempt in recorded successful report')
    require(same(steps,r['all_environment_step_counts']) and same(resets,r['all_environment_reset_counts']),'independent step/reset meter totals')
    for key,counts in branches.items():
        require(key in rows,'meter root join');row=rows[key]
        require(same(counts,row['environment_step_counts']) and same(b['budgets'][key],dict(root_sha256=key[0],method=key[1],environment_step_counts=counts)),'per-arm warmup/selected budget')
        require(counts['warmup']==dict(attempts=15,returns=15) and counts['selected']==dict(attempts=row['physical_env_step_attempts'],returns=row['physical_env_step_returns']),'per-arm independent selected/warmup counters')
    require(sum(c['attempts'] for c in steps.values())==r['total_environment_step_attempts'] and sum(c['returns'] for c in steps.values())==r['total_environment_step_returns'],'all-environment totals')
    for field,key in (('physical_env_step_attempts','attempts'),('physical_env_step_returns','returns')):
        require(sum(row[field] for row in rows.values())==r[field]==steps['selected'][key],'selected path total')
    require(steps['selected']['attempts']<=p['max_selected_path_steps']==15*eligible and steps['warmup']['attempts']<=p['max_warmup_steps']==15*eligible,'bounded physical budget')
    require(r['scheduled_arms']==8 and r['eligible_arms']==eligible and r['rejected_arms']==8-eligible,'dispatch census')
    for field,n in (('upstream_h1_transition_samples_charged_once',21624),('upstream_diagnostic_samples_charged_once',24),('upstream_backward_calls_charged_once',7200),('new_predict_samples',0),('new_neural_encodes',0),('new_model_loads',0),('new_independent_roots',0)):
        require(r[field]==n,'cost coordinate '+field)
    require(r['inherited_encoding_cost']=='unknown' and r['full_pipeline_equal_compute'] is r['P0_admission'] is r['P2_admission'] is r['efficacy_claim'] is False,'cost/admission boundary')
    pairs=[];diffs=[]
    saved_pairs=io.index(r['pairs'],('root_sha256',))
    require(set(saved_pairs)=={(root,) for root in roots},'all four saved pair records')
    for root in roots:
        a,d=(rows[root,m] for m in METHODS)
        context='context_identity' in a and 'context_identity' in d and same(a['context_identity'],d['context_identity'])
        full=context and all(x['full_horizon_completed'] and x['endpoint_fresh'] for x in (a,d))
        partial=context and all(x['endpoint_fresh'] for x in (a,d)) and a['physical_env_step_returns']==d['physical_env_step_returns'] and not full
        delta=a['pose']['pose_cost']-d['pose']['pose_cost'] if full else None
        saved=saved_pairs[root,]
        classification='full-path' if full else 'guard-partial' if partial else 'unmatched'
        require(saved['classification']==classification and saved['grasp_minus_action_adam_pose_cost']==delta and
            saved['steps']==[a['physical_env_step_returns'],d['physical_env_step_returns']],'independent physical pairing/delta')
        pairs.append(dict(root_sha256=root,classification=classification,steps=saved['steps'],grasp_minus_action_adam_pose_cost=delta))
        if full:diffs.append(delta)
    return dict(schema=NAME+'.report',status='saved_audit_pass',rows=deepcopy(r['rows']),pairs=pairs,
        branch_counts=dict(Counter(x['status'] for x in rows.values())),pair_counts=dict(Counter(x['classification'] for x in pairs)),
        budget=steps,reset_budget=resets,comparison=dict(complete_pairs=len(diffs),mean_grasp_minus_action_adam_pose_cost=sum(diffs)/len(diffs) if diffs else None),
        denominator=dict(branches=8,root_pairs=4,physical_repetitions=1,new_independent_roots=0),
        gaps=['No independent fresh physics or encoding performed','Whole warmup/source frames not saved in physical output; hashes bound to prior receipts, not independently re-rendered',
            'Original pre-warmup state/goal fingerprint not recomputed: original root state absent from branch output; root SHA provenance and goal/context joins verified',
            'Inherited encoding acquisition cost unknown; no total pipeline equivalence','Four calibration roots only; no held-out efficacy or published GRASP reproduction'],
        upstream_cost=dict(h1_transitions=21624,separate_h3_diagnostic_transitions=24,backward_calls=7200,inherited_encoding='unknown'),
        new_physical_steps=0,new_neural_calls=0,P0_admission=False,P2_admission=False)


def write(path,value):
    with Path(path).open('x') as f:f.write(io.canonical(value)+'\n');f.flush();os.fsync(f.fileno())


def run(root,output):
    root=Path(root).resolve();output=Path(output).absolute();require(not output.exists(),'exclusive NEW output required')
    b,e=load_bundle(root);result=audit(b);e.recheck()
    output.mkdir(exist_ok=False)
    files=[Path(__file__).resolve(),Path(io.__file__).resolve(),root/'experiments/spectral_reachability/d2p_prerun_source_seal.py',
        root/'tests'/('test_'+NAME+'.py'),root/'docs'/(NAME+'.md')]
    seal=create_seal(files=files,args_json=io.canonical(dict(physical_sha256=PHYSICAL_SHA,prepare_sha256=PREPARE_SHA)),output_dir=output/'source_seal')
    result.update(inputs_sha256=dict(sorted(e.hashes.items())),physical_report_sha256=PHYSICAL_SHA,consumer_prepare_sha256=PREPARE_SHA,source_seal=seal)
    result['inherited_planning_journal']=dict(path=b['protocol']['planning_journal_path'],**b['protocol']['planning_journal'])
    import hashlib
    h=hashlib.sha256()
    with Path(b['protocol']['planning_journal_path']).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    require(h.hexdigest()==b['protocol']['planning_journal']['sha256'],'inherited planning journal changed during audit')
    e.recheck();verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    write(output/'report.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();r=run(a.root,a.output);print(io.canonical(dict(status=r['status'],branch_counts=r['branch_counts'],pair_counts=r['pair_counts'],comparison=r['comparison'])))


if __name__=='__main__':main()
