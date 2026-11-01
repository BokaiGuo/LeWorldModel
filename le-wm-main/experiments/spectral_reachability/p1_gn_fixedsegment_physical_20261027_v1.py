"""MAIN-only physical consumer of the two frozen GN fixed-segment winners.

No neural/physics during collect, prepare or verify. New schema/selection IDs.
The old physical branch is imported explicitly, never globally monkey-patched.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.spectral_reachability import p1_gn_step_diagnostic_cuda_20261027_v1 as diagnostic
from experiments.spectral_reachability import p1_b8_physical_20261027_v1 as b8

physical,source,core=b8.physical,b8.source,b8.core
require,sha,read=b8.require,b8.sha,b8.read
NAME='p1_gn_fixedsegment_physical_20261027_v1'
ROOT=Path(__file__).resolve().parents[2]
ARM='GN_fixedsegment'
CAP,FLOOR,SMALL=2*1024**2,200*1024**2,100*1024


def contract():
    return dict(roots=2,arms=[ARM],order='original source-slot order; no outcome selection',
        resets=4,public_steps=20,explicit_renders=22,saved_png=6,checkpoints=[0,5,10],
        reset_internal_physics_implied=8,goal_renders_implied=4,actual_internal_substeps=None,
        reset='original random seed -> native goal -> original sample_state -> placement seed',
        reset_precondition='observed0/5/10 state/RGB/raw arrays exactly stationary and equal parent',
        commands='all10 original selected float32 raw; never clip/project/replace/replan',
        selection='frozen diagnostic incumbent + four scales stable minimum; three exact selected H2 replays',
        guard='sealed physical.branch; fresh initial and every returned-step RGB before next dispatch',
        terminal='honor native terminated/truncated; partial no replacement/retry',
        object_success_threshold=None,output_cap_bytes=CAP,reserve_bytes=FLOOR,
        P1_admission=False,model_task_eligible=False,stall_certified=False,
        global_freshness='unknown',new_independent_samples=0,neural_calls=0)


class EventCheck:
    def __init__(self,events):self.events=events;self.index=0
    def event(self,event):
        require(self.index<len(self.events) and self.events[self.index]==event,'exact diagnostic durable event order/content')
        self.index+=1


def check_diagnostic(prepared,report,selection,results,evaluations,events,tasks):
    """Reproduce saved receipts/selection only, no backend/model dispatch."""
    require(report['schema']==diagnostic.NAME+'.report' and report['status']=='cuda_fixed_segment_diagnostic_completed' and
        report['errors']==[] and report['first_error'] is None and report['source_verified'] is True and
        report['model_unchanged'] is True and report['model_hash_before']==report['model_hash_after']==prepared['expected_model_hash'] and
        report['resources']=={k:dict(attempted=32,completed=32) for k in ('predict','action_encoder')} and
        report['scales']==list(diagnostic.SCALES) and report['new_AD_calls']==report['new_image_calls']==report['new_physical_steps']==0 and
        report['training'] is False and report['new_Q'] is False and report['P1_admission'] is False and
        report['scientific_pass_threshold'] is None,'actual complete original diagnostic only')
    require(len(report['rows'])==len(prepared['roots'])==len(selection)==len(results)==len(tasks)==2 and
        len(evaluations)==16,'complete original two-root cohort and all16 H2 evaluations')
    winners=[];expected_logits=[]
    for i,(root,task) in enumerate(zip(prepared['roots'],tasks)):
        core.validate_task(task)
        require(root['root_index']==i and root['task_id']==core.task_id(task),'original root/task order')
        trials=evaluations[8*i:8*i+5]
        logits=[root['incumbent_logits']]+[c['logits'] for c in root['candidates']]
        require([c['scale'] for c in root['candidates']]==list(diagnostic.SCALES),'all original four scales')
        for c in root['candidates']:
            require(c==diagnostic.segment(root['incumbent_logits'],root['rejected_logits'],c['scale'],task['packet_logit_cap']),
                'original interpolation/rounding/raw exactly, no physical retune')
        require([e['logits'] for e in trials]==logits,'all5 exact frozen logit trials')
        require(trials[0]['cost']==root['saved_incumbent_cost'],'fresh incumbent original cost')
        winner_index=diagnostic.select(trials);winner=trials[winner_index]
        scale=None if winner_index==0 else diagnostic.SCALES[winner_index-1]
        expected_selection=dict(root_sha256=root['root_sha256'],root_index=i,evaluations=trials,winner_index=winner_index,
            selected_scale=scale,strict_improvement=winner['cost']<trials[0]['cost'])
        require(selection[i]==expected_selection,'original stable all5 argmin; never hardcode scale .25')
        replays=evaluations[8*i+5:8*i+8]
        prediction=dict(task_id=root['task_id'],logits=winner['logits'],commands=winner['commands'],contexts=replays[0]['contexts'],
            whole_replay=replays[1]['contexts'][-1],segmented_replay=replays[2]['contexts'],cost=winner['cost'])
        core.check_prediction(task,prediction)
        require(all(e['contexts']==winner['contexts'] and e['commands']==winner['commands'] and
            e['cost']==winner['cost'] and e['logits']==winner['logits'] for e in replays),'selected three independent fullH2 exact')
        expected=dict(root_sha256=root['root_sha256'],root_index=i,selected_scale=scale,incumbent_cost=trials[0]['cost'],
            scale_costs=[dict(scale=s,cost=e['cost'],improvement=trials[0]['cost']-e['cost']) for s,e in zip(diagnostic.SCALES,trials[1:])],
            selected_cost=winner['cost'],strict_improvement=expected_selection['strict_improvement'],verified_prediction=prediction,full32_required=True)
        require(results[i]==report['rows'][i]==expected,'same actual result/prediction top-level binding')
        physical.raw_commands(prediction['commands']);winners.append(prediction)
        expected_logits.extend(logits+[winner['logits']]*3)
    require(len({r['root_sha256'] for r in prepared['roots']})==2,'two original roots, no substitution')
    sink=EventCheck(events);meter=diagnostic.Meter(sink)
    for order,(value,(root,label,mode)) in enumerate(zip(evaluations,diagnostic.schedule())):
        task=tasks[root]
        require(value['root_index']==root and value['task_id']==core.task_id(task) and value['label']==label and value['mode']==mode and
            value['logits']==expected_logits[order],'complete original execution schedule')
        core.check_prediction(task,dict(task_id=value['task_id'],logits=value['logits'],commands=value['commands'],contexts=value['contexts'],
            whole_replay=value['contexts'][-1],segmented_replay=value['contexts'],cost=value['cost']))
        meter.start(root,label,mode)
        sink.event(dict(kind='evaluation_request',order=order,**{k:value[k] for k in ('root_index','task_id','label','mode','logits','commands')}))
        for packet in range(2):
            for resource in ('action_encoder','predict'):meter.end(meter.begin(resource))
            meter.emit('native_packet',packet=packet,mode=mode,context=value['contexts'][packet])
    require(sink.index==len(events) and meter.events==report['module_entries'] and meter.counts()==report['resources'],
        'all64 original module receipts, complete final packet, no extra/missing events')
    return winners


def collect(diagnostic_prepared,diagnostic_prepare_sha,diagnostic_report,diagnostic_report_sha):
    args=dict(diagnostic_prepared=str(Path(diagnostic_prepared).resolve()),diagnostic_prepare_sha=diagnostic_prepare_sha,
        diagnostic_report=str(Path(diagnostic_report).resolve()),diagnostic_report_sha=diagnostic_report_sha)
    bindings={args['diagnostic_prepared']:diagnostic_prepare_sha,args['diagnostic_report']:diagnostic_report_sha}
    physical.bindings_check(bindings)
    p=diagnostic.verify(args['diagnostic_prepared'],diagnostic_prepare_sha);r=diagnostic.read(args['diagnostic_report'])
    require(r['prepare_sha256']==diagnostic_prepare_sha,'same original diagnostic preparation')
    folder=Path(args['diagnostic_report']).parent
    names={f'eval_{i:02d}.json' for i in range(16)}|{'events.jsonl','execution_binding.json'}|{
        f'root_{i}.{kind}.json' for i in range(2) for kind in ('result','selection')}
    require(set(r['artifacts_sha256'])==names,'complete exact original artifact manifest')
    for name,h in r['artifacts_sha256'].items():bindings[str(folder/name)]=h
    physical.bindings_check(bindings)
    require(read(folder/'execution_binding.json')==dict(prepared=args['diagnostic_prepared'],prepare_sha256=diagnostic_prepare_sha),'execution binding')
    tasks=[diagnostic.read(root['task_path'])['task'] for root in p['roots']]
    winners=check_diagnostic(p,r,[read(folder/f'root_{i}.selection.json') for i in range(2)],
        [read(folder/f'root_{i}.result.json') for i in range(2)],[read(folder/f'eval_{i:02d}.json') for i in range(16)],
        [json.loads(line) for line in (folder/'events.jsonl').read_bytes().splitlines()],tasks)
    old=read(diagnostic.B8/'report.json')
    # Source/task/normalizer/RGB/full8J authentication only. NEVER read prior
    # physical results, never execute the B8 physical runner.
    data=b8.collect(str(diagnostic.B8_PREPARED),old['prepare_sha256'],str(diagnostic.B8/'report.json'),diagnostic.B8_SHA)
    gn={x['root_sha256']:x for x in data['plans'] if x['arm']=='full_H2_GN'}
    plans=[]
    for i,(root,winner) in enumerate(zip(p['roots'],winners)):
        base=gn[root['root_sha256']]
        require(base['task_id']==root['task_id'] and base['task_file']==root['task_path'],'same observed/native source task')
        plan={k:deepcopy(base[k]) for k in ('root_sha256','task_id','candidate_id','source_metadata','raw_source_file',
            'raw_source_sha256','task_file','task_file_sha256','goal_Q_context_sha256','history_semantics','observed_equals_parent')}
        plan.update(arm=ARM,commands=winner['commands'],prediction_sha256=source.digest(winner),selected_scale=r['rows'][i]['selected_scale'],
            origin=args['diagnostic_report'],origin_sha256=diagnostic_report_sha,search_cost=r['resources'],
            inherited_cost_scope='diagnostic global32/32 ONCE, not per-root free work',result_sha256=r['artifacts_sha256'][f'root_{i}.result.json'])
        plan['selection_sha256']=source.digest(plan);plans.append(plan)
    bindings.update(p['bindings_sha256']);bindings.update(data['bindings']);physical.bindings_check(bindings)
    require([x['candidate_id'] for x in plans]==[s['candidate_id'] for s in data['source_plan']['slots']], 'original source-slot order')
    return dict(arguments=args,plans=plans,source_plan=data['source_plan'],bindings_sha256=bindings,
        inherited=dict(source=data['inherited_source'],image=data['inherited_image'],direct=data['inherited_direct'],diagnostic=r['resources']))


class Store(physical.Store):
    def __init__(self,path):
        self.path=Path(path);self.used=0;self.previous=None;self.events=0
        require(shutil.disk_usage(self.path.parent).free>=FLOOR+CAP,'2MiB output plus200MiB reserve')
        self.path.mkdir(exist_ok=False)
    def check(self,size=65536):
        # Failed write/fsync may consume bytes before inherited put updates
        # self.used. Never refund those actual bytes during finalization.
        actual=sum(p.stat().st_size for p in self.path.iterdir() if p.is_file()) if self.path.exists() else 0
        require(max(self.used,actual)+size<=CAP,'2MiB TOTAL output cap')
        require(shutil.disk_usage(self.path).free>=FLOOR+size,'200MiB write floor')


def own_sources():
    paths=set(b8.own_sources())|{str(Path(__file__).resolve()),str(Path(diagnostic.__file__).resolve()),
        str(ROOT/'tests'/('test_'+NAME+'.py')),str(ROOT/'innerdocs'/(NAME+'.md'))}
    return {p:sha(p) for p in sorted(paths)}


def build(**arguments):
    data=collect(**arguments)
    return dict(schema=NAME+'.prepare',contract=contract(),arguments=data['arguments'],sources_sha256=own_sources(),
        material_sha256=source.digest(data),plans=data['plans'],inherited=data['inherited'],
        source_plan_sha256=source.digest(data['source_plan']),seal='complete upstream manifests reconstructed, digest-bound, never copied')


def prepare(output,**arguments):
    p=build(**arguments);raw=(source.canonical(p)+'\n').encode();path=Path(output)
    newfiles=[Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    require(len(raw)+sum(f.stat().st_size for f in newfiles)<=SMALL,'new source/tests/doc + one prepare <=100KiB')
    require(shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'small prepare200MiB floor')
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return p


def verify(prepared,expected_sha):
    require(sha(prepared)==expected_sha,'external own prepare SHA')
    p=read(prepared)
    require(p['schema']==NAME+'.prepare' and p==build(**p['arguments']),'source/selected action/protocol full reconstruction')
    return p


def object_results(goal,row,native_eval):
    def metric(state):
        state=source.vector(state);delta=state-source.vector(goal)
        return dict(object_position_error=float(np.linalg.norm(delta[2:4])),wrapped_object_angle_error=float(abs(np.arctan2(np.sin(delta[4]),np.cos(delta[4])))),
            object_pose=state[2:5].tolist(),success=None,label='object_only_diagnostic_no_authorized_success_threshold')
    checkpoints=[]
    for c in row.get('checkpoints',[]):
        success,distance=native_eval(np.asarray(goal),np.asarray(c['state']))
        checkpoints.append(dict(step=c['step'],state=c['state'],rgb_sha256=c['rgb_sha256'],file_sha256=c.get('file_sha256'),
            native_official_evaluation=dict(success=bool(success),state_distance=float(distance)),
            native_wrapped_diagnostic=physical.pose_diagnostics(goal,c['state']),**metric(c['state'])))
    return dict(checkpoints=checkpoints,
        final=None if row.get('final_state') is None else metric(row['final_state']),endpoint_scope=row.get('endpoint_scope','partial_or_unavailable'))


def run(prepared,prepare_sha,output,factory,*,authorize_physics=False):
    require(authorize_physics is True,'MAIN explicit physics authorization')
    p=verify(prepared,prepare_sha);data=collect(**p['arguments'])
    require(source.digest(data)==p['material_sha256'],'actual source material binding')
    store=Store(output);counts={k:dict(attempted=0,completed=0) for k in ('reset','render','step')}
    rows=[dict(root_sha256=x['root_sha256'],arm=x['arm'],selection_sha256=x['selection_sha256'],status='not_started',
        first_error=None,secondary_errors=[],trace=[]) for x in data['plans']]
    result=dict(schema=NAME+'.report',prepare_sha256=prepare_sha,contract=contract(),status='running',rows=rows,counts=counts,errors=[],
        inherited=p['inherited'],P1_admission=False,model_task_eligible=False,stall_certified=False,new_independent_samples=0,
        global_freshness='unknown',neural_calls=0)
    def error(exc,stage):return dict(stage=stage,type=type(exc).__name__,message=str(exc))
    def retain(i,exc,stage):
        if rows[i]['first_error'] is None:rows[i]['first_error']=error(exc,stage)
        else:rows[i]['secondary_errors'].append(error(exc,stage))
        rows[i]['status']='failed'
    try:
        store.emit('run_started',prepare_sha256=prepare_sha,contract=contract())
        for i,plan in enumerate(data['plans']):
            env=None
            try:
                store.check(262144);env=factory();body=env.unwrapped;sp=data['source_plan']
                path=inspect.getsourcefile(type(body))
                require(path is not None and sp['source_closure'].get(str(Path(path).resolve()))==sp['environment_sha256'],'original native constructor source')
                require(body.relative is True and body.action_scale==100 and body.render_mode=='rgb_array' and body.render_size==224,'original native raw mapping/RGB')
                rows[i]=physical.branch(plan,sp,env,store,counts)
                packet=source.load_raw_source(plan['raw_source_file'],sp)
                rows[i]['object_only']=object_results(packet['goal'],rows[i],body.eval_state)
                store.emit('object_only_result',selection_sha256=plan['selection_sha256'],receipt=rows[i]['object_only'])
            except (Exception,KeyboardInterrupt) as exc:retain(i,exc,'branch')
            finally:
                if env is not None:
                    try:env.close()
                    except (Exception,KeyboardInterrupt) as exc:retain(i,exc,'close')
                store.emit('branch_closed',receipt=rows[i])
        result['status']='physical_calibration_recorded' if all(r['status']=='completed' and r['full_horizon'] and r['initial_exact'] for r in rows) else 'partial_or_failed'
        if result['status']=='physical_calibration_recorded':
            require(counts=={k:dict(attempted=n,completed=n) for k,n in dict(reset=4,step=20,render=22).items()} and
                sum(len(r['checkpoints']) for r in rows)==6,'exact two-root complete physical census')
    except (Exception,KeyboardInterrupt) as exc:
        result['errors'].append(error(exc,'execution'));result['status']='partial_or_failed'
    finally:
        try:verify(prepared,prepare_sha)
        except (Exception,KeyboardInterrupt) as exc:
            result['errors'].append(error(exc,'final_bindings'));result['status']='partial_or_failed'
        sem=data['source_plan']['reset_semantics']
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
    for name in ('diagnostic-prepared','diagnostic-prepare-sha','diagnostic-report','diagnostic-report-sha','prepared','prepare-sha','output'):
        parser.add_argument('--'+name)
    parser.add_argument('--authorize-physics',action='store_true');a=parser.parse_args()
    if a.command=='prepare':
        prepare(a.output,**{k:getattr(a,k) for k in ('diagnostic_prepared','diagnostic_prepare_sha','diagnostic_report','diagnostic_report_sha')});print(sha(a.output))
    elif a.command=='verify':print(verify(a.prepared,a.prepare_sha)['schema'])
    else:
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224)
        r=run(a.prepared,a.prepare_sha,a.output,factory,authorize_physics=a.authorize_physics);print(source.canonical(r))
        if r['status']!='physical_calibration_recorded':raise SystemExit(2)


if __name__=='__main__':main()
