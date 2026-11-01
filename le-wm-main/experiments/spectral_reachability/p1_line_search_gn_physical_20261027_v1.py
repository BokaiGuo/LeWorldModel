"""MAIN-only two-root physical consumer of frozen line-search GN winners.

Prepare/verify are saved-array only. No model/simulator execution at import.
No old module globals are patched; no fixedsegment outcomes are read.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.spectral_reachability import p1_line_search_gn_cuda_20261027_v1 as search
from experiments.spectral_reachability import p1_gn_fixedsegment_physical_20261027_v1 as shared

b8,physical,source,core=shared.b8,shared.physical,shared.source,shared.core
require,sha,read=shared.require,shared.sha,shared.read
ROOT=Path(__file__).resolve().parents[2]
NAME='p1_line_search_gn_physical_20261027_v1'
ARM='line_search_GN'
CAP,FLOOR,SMALL=2*1024**2,200*1024**2,100*1024
Store=shared.Store


def contract():
    return dict(roots=2,arms=[ARM],order='original source-slot order',resets=4,public_steps=20,
        explicit_renders=22,saved_png=6,checkpoints=[0,5,10],output_cap_bytes=CAP,reserve_bytes=FLOOR,
        selection='zero restart,4 fixed updates,all5 scales each update,stable argmin including incumbent,independent triple replay',
        reset='original random seed -> native goal -> original sample_state -> placement seed; never saved poststate input',
        reset_precondition='observed0/5/10 state/RGB/raw exactly stationary and equal parent',
        guard='sealed physical.branch fresh initial and EVERY returned-step RGB',
        terminal='honor native terminated/truncated; partial without retry/replacement',
        reset_physics_implied=8,goal_renders_implied=4,actual_internal_substeps=None,object_success_threshold=None,
        P1_admission=False,model_task_eligible=False,stall_certified=False,global_freshness='unknown',
        new_independent_samples=0,neural_calls=0)


def check_component(task,row,events):
    """Validate all4 updates and full saved J/H2 receipts, no new SVD/model."""
    core.validate_task(task)
    require(row['task_id']==core.task_id(task) and len(row['steps'])==4 and
        [s['iteration'] for s in row['steps']]==[1,2,3,4],'complete same-task four-update trajectory')
    attempts=[e['event'] for e in events if e['kind']=='operation_attempted']
    complete=sorted([e['event'] for e in events if e['kind']=='operation_completed'],key=lambda e:e['index'])
    require([e['index'] for e in attempts]==list(range(len(attempts))) and
        [dict(e,completed=True) for e in attempts]==complete,'complete nonrefunded nested operation receipts')
    budget=search.budget()
    costs={k:dict(attempted=sum(e['samples'] for e in attempts if e['resource']==k),
        completed=sum(e['samples'] for e in complete if e['resource']==k and e['completed'])) for k in budget}
    require(row['resources']==costs=={k:dict(attempted=n,completed=n) for k,n in budget.items()} and
        all(e['resource'] in budget and e['samples']==1 for e in attempts),'exact78 forwards/4J/768VJP/4 solves per root')
    values=[e for e in events if e['kind']=='evaluation_complete']
    requests=[e for e in events if e['kind']=='evaluation_request']
    packets=[e for e in events if e['kind']=='native_packet']
    require(len(values)==len(requests)==35 and len(packets)==70,'all initial3 +4*(5trials+3replays) fullH2')
    for i,(event,request) in enumerate(zip(values,requests)):
        p=event['prediction'];require(p['task_id']==core.task_id(task),'same-Q original evaluation task')
        core.check_prediction(task,dict(p,whole_replay=p['contexts'][-1],segmented_replay=p['contexts']))
        require(event['evaluation_id']==request['evaluation_id'] and event['mode']==request['mode'] and
            p['logits']==request['logits'] and p['commands']==request['commands'],'all exact requested raw controls')
        for k in range(2):
            packet=packets[2*i+k]
            require(packet['evaluation_id']==event['evaluation_id'] and packet['mode']==event['mode'] and
                packet['packet']==k and packet['context']==p['contexts'][k],'all full native H2 histories')
    def triple(offset,group):
        es=values[offset:offset+3]
        require([e['mode'] for e in es]==['nominal','whole_replay','segmented_replay'] and
            all(e['evaluation_id']==group for e in es) and all(e['prediction']==es[0]['prediction'] for e in es),
            'exact independent triple replay')
        p=es[0]['prediction'];return dict(p,whole_replay=es[1]['prediction']['contexts'][-1],segmented_replay=es[2]['prediction']['contexts'])
    inc=triple(0,'initial');core.check_prediction(task,inc)
    initial=[e for e in events if e['kind']=='initial_verified']
    require(len(initial)==1 and initial[0]['incumbent']==inc and not np.asarray(inc['logits']).any(),'original zero restart, no inherited incumbent')
    derivatives=[e for e in events if e['kind']=='derivative_complete']
    frozen=[e for e in events if e['kind']=='line_proposals_frozen']
    selections=[e for e in events if e['kind']=='selection_complete']
    updates=[e for e in events if e['kind']=='update_verified']
    rows=[e for e in events if e['kind']=='jacobian_row'];joint=[e for e in events if e['kind']=='joint_logit_jacobian']
    forward=[e for e in events if e['kind']=='jacobian_forward_complete'];fp=[e for e in events if e['kind']=='jacobian_forward_packet']
    require((len(derivatives),len(frozen),len(selections),len(updates),len(rows),len(joint),len(forward),len(fp))==
        (4,4,4,4,768,4,4,8),'complete4 full192x20 J and selection/update receipts')
    for i in range(4):
        iteration=i+1;group=f'iter_{iteration}_AD';d=derivatives[i]['derivative'];f=frozen[i]
        core.check_derivative(task,d,inc)
        rawblock=rows[192*i:192*(i+1)]
        require([e['row'] for e in rawblock]==list(range(192)) and all(e['evaluation_id']==group for e in rawblock),'all192 raw rows at own incumbent')
        raw=core.array([e['raw_control_derivative'] for e in rawblock],(192,20))
        require(np.array_equal(raw*(1-np.asarray(inc['commands']).reshape(20)**2)[None],d['raw_terminal_jacobian']), 'exact single raw-to-logits tanh chain')
        require(joint[i]['jacobian']==d['raw_terminal_jacobian'] and joint[i]['terminal']==d['at_terminal'] and
            forward[i]['terminal']==d['at_terminal'] and forward[i]['logits']==d['at_logits'],'exact jointJ/AD nominal terminal')
        for k in range(2):require(fp[2*i+k]['packet']==k and fp[2*i+k]['context']==inc['contexts'][k],'AD wholeH2 incumbent history')
        delta=core.array(f['delta'],(20,))
        a=np.asarray(task['projection'])@np.asarray(d['raw_terminal_jacobian'])
        residual=np.asarray(task['projection'])@(np.asarray(d['at_terminal'])-np.asarray(task['goal']))
        matrix=a.T@a/192+search.DAMPING*np.eye(20);rhs=-a.T@residual/192
        require(np.allclose(matrix@delta,rhs,rtol=1e-8,atol=1e-10),'saved delta satisfies original .002 ridge normal equations; no new solve')
        endpoint=search.screen.projected_logits(np.asarray(inc['logits'])+delta.reshape(2,10),task['packet_logit_cap'])
        require(f['iteration']==iteration and f['evaluation_id']==group and f['incumbent_logits']==inc['logits'] and
            f['scales']==list(search.SCALES) and np.array_equal(f['endpoint'],endpoint),'own full-step projected endpoint')
        trials=search.line_trials(inc['logits'],f['endpoint'],task['packet_logit_cap'])
        require(f['logits']==[x.tolist() for x in trials],'all five original rounded segment proposals')
        offset=3+8*i;trial_events=values[offset:offset+5];candidates=[e['prediction'] for e in trial_events]
        require([e['evaluation_id'] for e in trial_events]==[f'iter_{iteration}_scale_{k}' for k in range(5)] and
            all(e['mode']=='whole_replay' for e in trial_events) and [p['logits'] for p in candidates]==f['logits'],'all5 trials, no skipped scales')
        index=search.select(inc,candidates);chosen=([inc]+candidates)[index]
        summary=dict(iteration=iteration,incumbent_cost=inc['cost'],trial_costs=[p['cost'] for p in candidates],
            selected_index=index,selected_scale=None if index==0 else search.SCALES[index-1],selected_cost=chosen['cost'])
        require(row['steps'][i]==summary and all(selections[i][k]==v for k,v in summary.items()),'original stable incumbent+5 selection')
        fresh=triple(offset+5,f'iter_{iteration}_selected_replay')
        require(all(fresh[k]==chosen[k] for k in ('logits','commands','contexts','cost')) and
            updates[i]['iteration']==iteration and updates[i]['prediction']==fresh,'selected fresh independent fullH2, no physical reselection')
        inc=fresh
    require(row['final_prediction']==inc,'EXACT final line-search incumbent')
    physical.raw_commands(inc['commands']);return inc


def collect(search_prepared,search_prepare_sha,search_report,search_report_sha):
    args=dict(search_prepared=str(Path(search_prepared).resolve()),search_prepare_sha=search_prepare_sha,
        search_report=str(Path(search_report).resolve()),search_report_sha=search_report_sha)
    bindings={args['search_prepared']:search_prepare_sha,args['search_report']:search_report_sha};physical.bindings_check(bindings)
    p=search.verify(args['search_prepared'],search_prepare_sha);r=read(args['search_report']);folder=Path(args['search_report']).parent
    require(r['schema']==search.NAME+'.report' and r['status']=='cuda_line_search_gn_completed' and r['errors']==[] and
        r['first_error'] is None and r['prepare_sha256']==search_prepare_sha and r['source_verified'] is True and
        r['model_unchanged'] is True and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and
        r['resources']=={k:dict(attempted=2*v,completed=2*v) for k,v in search.budget().items()} and
        r['new_images']==r['new_physics']==0 and r['training'] is False and r['new_Q'] is False and
        r['P1_admission'] is False and r['independent_sample_increment']==0 and r['scientific_pass_threshold'] is None,
        'actual full line-search completion/model/source/accounting')
    names={'binding.json','events.jsonl.gz','root_0.json.gz','root_1.json.gz'}
    require(set(r['artifact_sha256'])==names,'complete original four artifact manifest')
    bindings.update({str(folder/name):h for name,h in r['artifact_sha256'].items()});physical.bindings_check(bindings)
    require(read(folder/'binding.json')==dict(prepared=args['search_prepared'],prepare_sha256=search_prepare_sha),'actual execution binding')
    events=b8.gz(folder/'events.jsonl.gz',lines=True)
    require([e['root_index'] for e in events]==sorted(e['root_index'] for e in events) and
        set(e['root_index'] for e in events)=={0,1},'fixed complete root order')
    old=read(search.saved.B8/'report.json')
    data=b8.collect(str(search.saved.B8_PREPARED),old['prepare_sha256'],str(search.saved.B8/'report.json'),search.saved.B8_SHA)
    bases={x['root_sha256']:x for x in data['plans'] if x['arm']=='full_H2_GN'}
    require(len(r['roots'])==len(p['roots'])==2,'fixed two original roots')
    plans=[]
    for i,root in enumerate(p['roots']):
        require(root['root_index']==i,'original task order')
        task=search.read(root['task_path'])['task'];row=b8.gz(folder/f'root_{i}.json.gz')
        require(row['root_index']==i and row['root_sha256']==root['root_sha256'],'same actual model root')
        ev=[e for e in events if e['root_index']==i];starts=[e for e in ev if e['kind']=='root_started']
        require(len(starts)==1 and ev[0]==dict(kind='root_started',root_index=i,task_id=root['task_id']) and
            all(e.get('root_sha256')==root['root_sha256'] for e in ev[1:]),'root journal identity')
        winner=check_component(task,row,ev)
        require(r['roots'][i]==dict(root_sha256=root['root_sha256'],steps=row['steps'],final_cost=winner['cost']),'top-level steps/final cost exact')
        base=bases[root['root_sha256']];require(base['task_id']==root['task_id'] and base['task_file']==root['task_path'],'same observed/source task')
        plan={k:deepcopy(base[k]) for k in ('root_sha256','task_id','candidate_id','source_metadata','raw_source_file',
            'raw_source_sha256','task_file','task_file_sha256','goal_Q_context_sha256','history_semantics','observed_equals_parent')}
        plan.update(arm=ARM,commands=winner['commands'],prediction_sha256=source.digest(winner),origin=str(folder/f'root_{i}.json.gz'),
            origin_sha256=r['artifact_sha256'][f'root_{i}.json.gz'],steps_sha256=source.digest(row['steps']),
            search_cost=row['resources'],selection_method='final incumbent after4 fixed line-search GN updates')
        plan['selection_sha256']=source.digest(plan);plans.append(plan)
    require(len({p['root_sha256'] for p in plans})==2 and [p['candidate_id'] for p in plans]==[s['candidate_id'] for s in data['source_plan']['slots']], 'source-slot order no substitution')
    bindings.update(p['bindings_sha256']);bindings.update(data['bindings']);physical.bindings_check(bindings)
    return dict(arguments=args,plans=plans,source_plan=data['source_plan'],bindings_sha256=bindings,
        inherited=dict(source=data['inherited_source'],image=data['inherited_image'],B8direct=data['inherited_direct'],line_search=r['resources']))


def own_sources():
    paths=set(shared.own_sources())|{str(Path(__file__).resolve()),str(Path(search.__file__).resolve()),
        str(ROOT/'tests'/('test_'+NAME+'.py')),str(ROOT/'innerdocs'/(NAME+'.md'))}
    return {p:sha(p) for p in sorted(paths)}


def build(**arguments):
    data=collect(**arguments)
    return dict(schema=NAME+'.prepare',contract=contract(),arguments=data['arguments'],sources_sha256=own_sources(),
        material_sha256=source.digest(data),plans=data['plans'],source_plan_sha256=source.digest(data['source_plan']),inherited=data['inherited'])


def prepare(output,**arguments):
    p=build(**arguments);raw=(source.canonical(p)+'\n').encode();path=Path(output)
    files=[Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    require(len(raw)+sum(f.stat().st_size for f in files)<=SMALL,'new files plus one prepare <=100KiB')
    require(shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'small prepare200MiB floor')
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return p


def verify(prepared,expected_sha):
    require(sha(prepared)==expected_sha,'external NEW prepare SHA');p=read(prepared)
    require(p['schema']==NAME+'.prepare' and p==build(**p['arguments']),'full selected/source/closure reconstruction');return p


def run(prepared,prepare_sha,output,factory,*,authorize_physics=False):
    require(authorize_physics is True,'MAIN explicit physics authorization')
    p=verify(prepared,prepare_sha);data=collect(**p['arguments'])
    require(source.digest(data)==p['material_sha256'],'actual source material binding')
    store=Store(output);counts={k:dict(attempted=0,completed=0) for k in ('reset','render','step')}
    rows=[dict(root_sha256=x['root_sha256'],arm=x['arm'],selection_sha256=x['selection_sha256'],status='not_started',first_error=None,secondary_errors=[],trace=[]) for x in data['plans']]
    result=dict(schema=NAME+'.report',prepare_sha256=prepare_sha,contract=contract(),status='running',rows=rows,counts=counts,errors=[],
        inherited=p['inherited'],P1_admission=False,model_task_eligible=False,stall_certified=False,new_independent_samples=0,global_freshness='unknown',neural_calls=0)
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
                store.check(262144);env=factory();body=env.unwrapped;sp=data['source_plan'];path=inspect.getsourcefile(type(body))
                require(path is not None and sp['source_closure'].get(str(Path(path).resolve()))==sp['environment_sha256'],'original native constructor source')
                require(body.relative is True and body.action_scale==100 and body.render_mode=='rgb_array' and body.render_size==224,'original raw mapping/RGB')
                rows[i]=physical.branch(plan,sp,env,store,counts)
                packet=source.load_raw_source(plan['raw_source_file'],sp)
                rows[i]['object_only']=shared.object_results(packet['goal'],rows[i],body.eval_state)
                store.emit('object_only_result',selection_sha256=plan['selection_sha256'],receipt=rows[i]['object_only'])
            except (Exception,KeyboardInterrupt) as exc:retain(i,exc,'branch')
            finally:
                if env is not None:
                    try:env.close()
                    except (Exception,KeyboardInterrupt) as exc:retain(i,exc,'close')
                store.emit('branch_closed',receipt=rows[i])
        result['status']='physical_calibration_recorded' if all(r['status']=='completed' and r['full_horizon'] and r['initial_exact'] for r in rows) else 'partial_or_failed'
        if result['status']=='physical_calibration_recorded':
            require(counts=={k:dict(attempted=n,completed=n) for k,n in dict(reset=4,step=20,render=22).items()} and sum(len(r['checkpoints']) for r in rows)==6,'exact two-branch census')
    except (Exception,KeyboardInterrupt) as exc:
        result['errors'].append(error(exc,'execution'));result['status']='partial_or_failed'
    finally:
        try:verify(prepared,prepare_sha)
        except (Exception,KeyboardInterrupt) as exc:
            result['errors'].append(error(exc,'final_bindings'));result['status']='partial_or_failed'
        sem=data['source_plan']['reset_semantics']
        result['internal_counts']=dict(reset_physics_completed_implied=counts['reset']['completed']*sem['internal_physics_steps_per_reset'],
            reset_physics_attempted_upper_bound=counts['reset']['attempted']*sem['internal_physics_steps_per_reset'],goal_renders_completed_implied=counts['reset']['completed']*sem['internal_renders_per_reset'],actual_substeps=None)
        try:
            store.emit('run_finished',status=result['status'],counts=counts,errors=result['errors'])
            result['journal_sha256']=sha(store.path/'journal.jsonl');store.json('report.json',result)
        except (Exception,KeyboardInterrupt) as exc:
            result['errors'].append(error(exc,'final_write'));result['status']='partial_or_failed'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run'))
    for name in ('search-prepared','search-prepare-sha','search-report','search-report-sha','prepared','prepare-sha','output'):parser.add_argument('--'+name)
    parser.add_argument('--authorize-physics',action='store_true');a=parser.parse_args()
    if a.command=='prepare':
        prepare(a.output,**{k:getattr(a,k) for k in ('search_prepared','search_prepare_sha','search_report','search_report_sha')});print(sha(a.output))
    elif a.command=='verify':print(verify(a.prepared,a.prepare_sha)['schema'])
    else:
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224)
        r=run(a.prepared,a.prepare_sha,a.output,factory,authorize_physics=a.authorize_physics);print(source.canonical(r))
        if r['status']!='physical_calibration_recorded':raise SystemExit(2)


if __name__=='__main__':main()
