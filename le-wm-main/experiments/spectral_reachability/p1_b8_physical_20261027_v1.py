"""MAIN-only B8 GN/GD physical consumer. Saved checks/prepare are array-only.

No admission, independent sample, or stall assertion. No neural work anywhere.
Original parent reset is legal ONLY after stationary observed-parent equality.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import shutil
import zlib

import numpy as np

from experiments.spectral_reachability import p1_observed_direct_cuda_20261027_v2 as direct
from experiments.spectral_reachability import p1_witness_physical_20261027_v1 as physical

source, core, screen=physical.source, physical.core, direct.screen
require, sha, read=physical.require, physical.sha, physical.read
NAME='p1_b8_physical_20261027_v1'
ROOT=Path(__file__).resolve().parents[2]
ARMS=('full_H2_GN','full_H2_gradient')
CAP, FLOOR, SMALL=4*1024**2, 200*1024**2, 300_000


def contract():
    return dict(roots=2,arms=list(ARMS),branches=4,order='original_source_slot_then_GN_GD',
        resets=8,public_steps=40,explicit_renders=44,saved_png=12,checkpoints=[0,5,10],
        reset_internal_physics_steps_implied=16,internal_goal_renders_implied=8,
        counts_note='internal physics implied, NOT instrumented; public substeps unknown',
        selection='original single zero restart strict incumbent after8 iterations; no physical selection',
        reset='original seed then sample_state then placement; never saved poststate as input',
        reset_precondition='observed0/5/10 arrays stationary and exactly equal parent, otherwise reject',
        history='observed_not_padding_even_when_numerically_identical',
        guard='sealed physical.branch fresh initial and every returned-step RGB',
        terminal='honor native terminated/truncated; retain partial, no replacement/retry',
        diagnostics='native unchanged; wrapped/negative-angle/disagreement retained separately',
        output_cap_bytes=CAP,reserve_bytes=FLOOR,neural_calls=0,new_independent_samples=0,
        P1_admission=False,model_task_eligible=False,independent_confirmation=False,
        global_freshness='unknown',equal_compute=False,stall_certified=False)


def decode_gzip(raw,limit=48*1024**2):
    """Complete ONE stream only, bounded decompression; never accept partial EOF."""
    require(len(raw)<=24*1024**2,'bounded compressed input')
    decoder=zlib.decompressobj(31);value=decoder.decompress(raw,limit+1)
    require(len(value)<=limit and decoder.eof and not decoder.unused_data and
        not decoder.unconsumed_tail,'complete bounded single gzip stream required')
    return value


def gz(path,*,lines=False):
    require(Path(path).stat().st_size<=24*1024**2,'bounded gzip input file')
    raw=decode_gzip(Path(path).read_bytes())
    return [json.loads(x) for x in raw.splitlines()] if lines else json.loads(raw)


def component(task,method,report,events):
    """Full saved trajectory +8 J receipts; no new model/SVD/solve execution."""
    c=direct.original.config(method);budget=screen.budget(c)
    require(report['status']=='direct_component_completed_not_admitted' and report['error'] is None and
        not report['secondary_errors'] and report['task_id']==core.task_id(task) and
        report['configuration']==c and report['configuration_sha256']==core.digest(c) and
        report['budget']==budget,'complete fixed same-task B4/B8 report')
    require(len(report['direct_runs'])==1 and len(report['direct_runs'][0])==9,'full zero plus8 trajectory, no thinning')
    run=report['direct_runs'][0];previous=None
    for i,step in enumerate(run):
        require(step['iteration']==i,'all nine ordered iterates')
        core.check_prediction(task,step['incumbent'])
        if i==0:
            require(step['proposal'] is None and step['derivative'] is None and step['accepted'] is False and
                step['incumbent']['logits']==c['initial_logits'][0],'original zero initial iterate')
        else:
            core.check_prediction(task,step['proposal']);core.check_derivative(task,step['derivative'],previous)
            accepted=step['proposal']['cost']<previous['cost']
            require(step['accepted'] is accepted and step['incumbent']==(step['proposal'] if accepted else previous),
                'strict original acceptance, no physical reselection')
        previous=step['incumbent']
    require(events[0]==dict(kind='direct_started',task=task,configuration=c,budget=budget) and
        events[-1]==dict(kind='direct_finished',report=report),'complete journal start/task/end/report binding')
    require([e['step'] for e in events if e['kind']=='trajectory_step']==run,'all durable trajectory steps')
    attempted=[e['event'] for e in events if e['kind']=='operation_attempted']
    completed=sorted([e['event'] for e in events if e['kind']=='operation_completed'],key=lambda e:e['index'])
    require([e['index'] for e in attempted]==list(range(len(attempted))) and
        completed==report['extended_journal'] and
        [dict(e,completed=True) for e in attempted]==completed,'all nested operation receipts complete by immutable index')
    work={k:dict(attempted=sum(e['samples'] for e in attempted if e['resource']==k),
        completed=sum(e['samples'] for e in completed if e['resource']==k and e['completed'])) for k in budget}
    require(work=={k:dict(attempted=n,completed=n) for k,n in budget.items()} and
        all(e['resource'] in budget and e['samples']==1 for e in attempted),'full exact B8 resources')
    core_events=[e for e in completed if e['resource'] in core.RESOURCES]
    require(report['journal']==[dict(e,index=i) for i,e in enumerate(core_events)], 'raw core resource journal binding')
    predictions=[run[0]['incumbent']]+[s['proposal'] for s in run[1:]]
    require([e['prediction'] for e in events if e['kind']=='prediction_complete']==predictions,'all independently evaluated iterates')
    paths=[e for e in events if e['kind']=='prediction_path']
    packets=[e for e in events if e['kind']=='native_packet']
    require(len(paths)==27 and len(packets)==54,'all full H2 nominal/whole/segmented paths')
    for i,p in enumerate(predictions):
        for j,mode in enumerate(('nominal','whole_replay','segmented_replay')):
            require(paths[3*i+j]==dict(kind='prediction_path',mode=mode,contexts=p['contexts']),'saved complete replay contexts')
            for k in range(2):
                require(packets[6*i+2*j+k]==dict(kind='native_packet',mode=mode,packet=k,context=p['contexts'][k]),'full native packet chronology')
    derivatives=[s['derivative'] for s in run[1:]]
    require([e['derivative'] for e in events if e['kind']=='derivative_complete']==derivatives and
        [e['logits'] for e in events if e['kind']=='proposal_logits']==[s['proposal']['logits'] for s in run[1:]],'full J/proposal journal')
    rows=[e for e in events if e['kind']=='jacobian_row']
    joint=[e for e in events if e['kind']=='joint_logit_jacobian']
    forward=[e for e in events if e['kind']=='jacobian_forward_complete']
    fp=[e for e in events if e['kind']=='jacobian_forward_packet']
    require((len(rows),len(joint),len(forward),len(fp))==(1536,8,8,16),'all8 full192x20 J receipts')
    for i,d in enumerate(derivatives):
        inc=run[i]['incumbent'];block=rows[192*i:192*(i+1)]
        require([e['row'] for e in block]==list(range(192)),'all raw-control derivative rows')
        raw=core.array([e['raw_control_derivative'] for e in block],(192,20))
        require(np.array_equal(raw*(1-np.asarray(inc['commands']).reshape(20)**2)[None],d['raw_terminal_jacobian']),
            'one exact raw-control to logits tanh chain')
        require(forward[i]==dict(kind='jacobian_forward_complete',terminal=d['at_terminal'],logits=d['at_logits']) and
            joint[i]==dict(kind='joint_logit_jacobian',terminal=d['at_terminal'],jacobian=d['raw_terminal_jacobian']), 'exact derivative forward and full J')
        for k in range(2):
            require(fp[2*i+k]==dict(kind='jacobian_forward_packet',packet=k,context=inc['contexts'][k]),'J fullH2 exact incumbent history')
    physical.raw_commands(previous['commands'])
    return previous,work


def stationary_parent(packet,parent_path):
    """Reject general nonstationary observed roots; never silently add warmup."""
    with np.load(parent_path,allow_pickle=False) as parent:
        require(packet['metadata']['root_sha256']==packet['metadata']['parent_root_sha256'] and
            np.array_equal(packet['state'],parent['state']) and np.array_equal(packet['goal'],parent['goal']) and
            np.array_equal(packet['goal_rgb'],parent['goal_rgb']) and
            np.array_equal(packet['rgb_history'],parent['rgb_history']) and
            np.array_equal(packet['raw_action_history'],parent['raw_action_history']) and
            all(np.array_equal(x,parent['state']) for x in packet['observed_states']) and
            all(np.array_equal(x,parent['rgb_history'][0]) for x in packet['rgb_history']),
            'original recipe only valid for this exactly stationary observed-parent bridge')


def collect(direct_prepared,direct_prepare_sha,direct_report,direct_report_sha):
    prepared=Path(direct_prepared).resolve();report_path=Path(direct_report).resolve()
    bindings={str(prepared):direct_prepare_sha,str(report_path):direct_report_sha}
    physical.bindings_check(bindings);p=direct.verify(prepared,direct_prepare_sha);r=read(report_path)
    require(r['schema']==direct.NAME+'.report' and r['status']=='observed_direct_calibration_completed' and
        not r['errors'] and r['exact_budget_verified'] is True and r['source_verified'] is True and
        r['model_unchanged'] is True and r['model_hash_before'] is not None and
        r['model_hash_before']==r['model_hash_after'] and r['prepare_sha256']==direct_prepare_sha and
        r['actual_direct_resources']=={k:dict(attempted=n,completed=n) for k,n in direct.TOTAL.items()} and
        r['actual_image_resources']=={k:dict(attempted=8,completed=8) for k in ('encoder','projector')} and
        r['encoded_tasks']==2 and r['P1_admission'] is False and r['independent_sample_increment']==0 and
        r['new_physical_steps']==0,'actual complete observed B8 source/model/resources')
    data=direct.original.material(**dict(pack=p['inputs']['pack'],expected_sha=p['inputs']['source_sha'],
        source_prepare=p['inputs']['source_prepare'],source_prepare_sha=p['inputs']['source_prepare_sha']))
    require(data['summary']==r['source_audit']==p['source_audit'],'same full observed source audit')
    op=data['source_prepare'];parent_path=Path(op['inputs']['report']);parent=read(parent_path)
    request=read(op['inputs']['request']);source_plan=request['plan']
    bindings.update(p['source_sha256']);bindings.update(p['source_bindings']);bindings.update(p['model']['bindings'])
    bindings.update(op['bindings']);bindings.update(source_plan['source_closure'])
    image_path=report_path.parent/'image.events.jsonl.gz';bindings[str(image_path)]=sha(image_path)
    images=gz(image_path,lines=True);require(len(images)==24,'eight native image receipts')
    for i in range(8):
        a,b,c=images[3*i:3*i+3]
        require(a==dict(kind='native_image_attempted',resource='encoder',samples=1) and
            b==dict(kind='native_image_attempted',resource='projector',samples=1) and
            c['kind']=='native_image_completed' and c['counts']==dict(encoder=1,projector=1),'native image accounting')
    methods={(x['root_sha256'],x['method']):x for x in r['direct']}
    require(len(r['direct'])==len(methods)==4,'unique four direct components')
    weights=next(v for f,v in p['model']['bindings'].items() if Path(f).name=='weights.pt')
    config=next(v for f,v in p['model']['bindings'].items() if Path(f).name=='config.json')
    base=direct.original.template(data,source.digest(p),weights,config)
    plans=[];roots=[]
    require(len(parent['slots'])==len(data['packets'])==2,'original two source slots')
    for i,(packet,parent_row) in enumerate(zip(data['packets'],parent['slots'])):
        root=packet['metadata']['root_sha256'];roots.append(root)
        require(parent_row['slot']['candidate_id']==packet['metadata']['candidate_id'] and
            parent_row['metadata']['root_sha256']==root,'same stationary parent/source slot')
        raw_path=parent_path.parent/parent_row['raw_source_file']
        bindings[str(raw_path)]=parent_row['raw_source_sha256'];physical.bindings_check({str(raw_path):bindings[str(raw_path)]})
        stationary_parent(packet,raw_path)
        receipts=images[12*i:12*(i+1)];cursor=iter(receipts[2::3])
        def saved_encode(rgb):
            receipt=next(cursor)
            require(receipt['rgb_sha256']==direct.observed.image_sha(rgb),'original actual native RGB/goal input binding')
            return receipt['latent']
        expected=direct.original.encode_observed(packet,base,saved_encode)
        task_path=report_path.parent/f'task_{i}.json.gz';bindings[str(task_path)]=sha(task_path)
        encoded=gz(task_path);require(encoded==expected,'full actual task/goal/identityQ/normalizer/source semantics')
        task=encoded['task'];core.validate_task(task)
        require(np.array_equal(task['projection'],np.eye(192)),'same identity native192 Q')
        for method in ARMS:
            row=methods[root,method]
            require(row['status']=='direct_component_completed_not_admitted' and row['error'] is None,'complete original arm')
            paths={}
            for field in ('report','journal'):
                path=Path(row[field+'_path']).resolve()
                require(path.parent.parent==report_path.parent and not path.is_symlink(),'original actual component artifacts')
                bindings[str(path)]=row[field+'_sha256'];paths[field]=path
            physical.bindings_check({str(v):bindings[str(v)] for v in paths.values()})
            sub=gz(paths['report']);events=gz(paths['journal'],lines=True)
            winner,work=component(task,method,sub,events)
            plan=dict(root_sha256=root,task_id=core.task_id(task),candidate_id=parent_row['slot']['candidate_id'],arm=method,
                source_metadata=parent_row['metadata'],raw_source_file=str(raw_path),raw_source_sha256=parent_row['raw_source_sha256'],
                commands=winner['commands'],prediction_sha256=source.digest(winner),trajectory_sha256=source.digest(sub['direct_runs']),
                jacobians_sha256=source.digest([s['derivative'] for s in sub['direct_runs'][0][1:]]),
                origin=str(paths['report']),origin_sha256=bindings[str(paths['report'])],task_file=str(task_path),task_file_sha256=bindings[str(task_path)],
                search_cost=work,history_semantics=encoded['history_semantics'],observed_equals_parent=True,
                goal_Q_context_sha256=source.digest({k:task[k] for k in ('goal','projection','root_context','action_mean','action_std','model_sha256')}))
            plan['selection_sha256']=source.digest(plan);plans.append(plan)
    require(len(set(roots))==2 and set(methods)=={(root,m) for root in roots for m in ARMS},'exact cohort no replacement')
    physical.bindings_check(bindings)
    return dict(arguments=dict(direct_prepared=str(prepared),direct_prepare_sha=direct_prepare_sha,
        direct_report=str(report_path),direct_report_sha=direct_report_sha),plans=plans,source_plan=source_plan,bindings=bindings,
        inherited_source=r['actual_source_acquisition_resources'],inherited_image=r['actual_image_resources'],inherited_direct=r['actual_direct_resources'])


class Store(physical.Store):
    """Reuse sealed hash-journal/PNG writer, override ONLY explicit quota methods."""
    def __init__(self,path):
        self.path=Path(path);self.used=0;self.previous=None;self.events=0
        require(shutil.disk_usage(self.path.parent).free>=FLOOR+CAP,'4MiB output plus200MiB reserve')
        self.path.mkdir(exist_ok=False)

    def check(self,size=65536):
        require(self.used+size<=CAP,'4MiB total output cap')
        require(shutil.disk_usage(self.path).free>=FLOOR+size,'200MiB write floor')


def own_sources():
    paths=set(physical.own_sources())|{str(Path(direct.__file__).resolve()),str(Path(direct.original.__file__).resolve()),
        str(Path(__file__).resolve()),str(ROOT/'tests'/('test_'+NAME+'.py')),str(ROOT/'innerdocs'/(NAME+'.md'))}
    return {path:sha(path) for path in sorted(paths)}


def build(**arguments):
    return dict(schema=NAME+'.prepare',contract=contract(),inputs=collect(**arguments),sources=own_sources())


def prepare(output,**arguments):
    p=build(**arguments);raw=(source.canonical(p)+'\n').encode();path=Path(output)
    require(len(raw)<=SMALL,'small <=300KB hash-only prepare')
    require(shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'prepare200MiB floor')
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return p


def verify(prepared,expected_sha):
    require(sha(prepared)==expected_sha,'external NEW prepare SHA')
    p=read(prepared)
    require(p['schema']==NAME+'.prepare' and p==build(**p['inputs']['arguments']),'full source/task/winner reconstruction')
    return p


def run(prepared,prepare_sha,output,factory,*,authorize_physics=False):
    require(authorize_physics is True,'MAIN explicit physics authorization')
    p=verify(prepared,prepare_sha);store=Store(output)
    rows=[dict(root_sha256=x['root_sha256'],arm=x['arm'],selection_sha256=x['selection_sha256'],status='not_started',
        first_error=None,secondary_errors=[],trace=[]) for x in p['inputs']['plans']]
    counts={k:dict(attempted=0,completed=0) for k in ('reset','render','step')}
    result=dict(schema=NAME+'.report',prepare_sha256=prepare_sha,contract=contract(),status='running',rows=rows,counts=counts,
        errors=[],neural_calls=0,P1_admission=False,model_task_eligible=False,stall_certified=False,
        global_freshness='unknown',new_independent_samples=0,inherited={k:p['inputs'][k] for k in ('inherited_source','inherited_image','inherited_direct')})
    def error(exc,phase):return dict(phase=phase,type=type(exc).__name__,message=str(exc))
    try:
        store.emit('run_started',prepare_sha256=prepare_sha,contract=contract())
        for i,plan in enumerate(p['inputs']['plans']):
            env=None
            try:
                store.check(262144);env=factory();body=env.unwrapped
                backend=inspect.getsourcefile(type(body));sp=p['inputs']['source_plan']
                require(backend is not None and sp['source_closure'].get(str(Path(backend).resolve()))==sp['environment_sha256'], 'original native environment source')
                require(body.relative is True and body.action_scale==100 and body.render_mode=='rgb_array' and body.render_size==224,'original raw mapping/render')
                rows[i]=physical.branch(plan,sp,env,store,counts)
            except (Exception,KeyboardInterrupt) as exc:
                if rows[i]['first_error'] is None:rows[i]['first_error']=error(exc,'branch_setup')
                else:rows[i]['secondary_errors'].append(error(exc,'branch_setup'))
                rows[i]['status']='failed'
            finally:
                if env is not None:
                    try:env.close()
                    except (Exception,KeyboardInterrupt) as exc:
                        if rows[i]['first_error'] is None:rows[i]['first_error']=error(exc,'close')
                        else:rows[i]['secondary_errors'].append(error(exc,'close'))
                        rows[i]['status']='failed'
                store.emit('branch_closed',receipt=rows[i])
        result['status']='physical_calibration_recorded' if all(x['status']=='completed' and x['full_horizon'] and x['initial_exact'] for x in rows) else 'partial_or_failed'
        if result['status']=='physical_calibration_recorded':
            require(counts=={k:dict(attempted=n,completed=n) for k,n in dict(reset=8,step=40,render=44).items()} and
                sum(len(x['checkpoints']) for x in rows)==12,'exact complete four-arm census')
    except (Exception,KeyboardInterrupt) as exc:
        result['errors'].append(error(exc,'execution'));result['status']='partial_or_failed'
    finally:
        try:verify(prepared,prepare_sha)
        except (Exception,KeyboardInterrupt) as exc:
            result['errors'].append(error(exc,'final_bindings'));result['status']='partial_or_failed'
        sem=p['inputs']['source_plan']['reset_semantics']
        result['internal_counts']=dict(reset_physics_completed_implied=counts['reset']['completed']*sem['internal_physics_steps_per_reset'],
            reset_physics_attempted_upper_bound=counts['reset']['attempted']*sem['internal_physics_steps_per_reset'],
            goal_renders_completed_implied=counts['reset']['completed']*sem['internal_renders_per_reset'],actual_substeps=None)
        try:
            store.emit('run_finished',status=result['status'],counts=counts,errors=result['errors'])
            result['journal_sha256']=sha(store.path/'journal.jsonl');store.json('report.json',result)
        except (Exception,KeyboardInterrupt) as exc:
            result['errors'].append(error(exc,'final_write'));result['status']='partial_or_failed'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run'))
    for name in ('direct-prepared','direct-prepare-sha','direct-report','direct-report-sha','prepared','prepare-sha','output'):
        parser.add_argument('--'+name)
    parser.add_argument('--authorize-physics',action='store_true');a=parser.parse_args()
    if a.command=='prepare':
        prepare(a.output,**{k:getattr(a,k) for k in ('direct_prepared','direct_prepare_sha','direct_report','direct_report_sha')});print(sha(a.output))
    elif a.command=='verify':print(verify(a.prepared,a.prepare_sha)['schema'])
    else:
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224)
        r=run(a.prepared,a.prepare_sha,a.output,factory,authorize_physics=a.authorize_physics);print(source.canonical(r))
        if r['status']!='physical_calibration_recorded':raise SystemExit(2)


if __name__=='__main__':main()
