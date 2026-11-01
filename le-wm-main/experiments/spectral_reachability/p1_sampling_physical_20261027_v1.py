"""MAIN-only physical consumer of frozen CEM/IID winners; no neural work.

Saved-array prepare/verify is safe here. Actual reset/step is MAIN's authority.
This is a four-branch calibration comparison, never scientific P1 admission.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import json
from pathlib import Path
import zlib

import numpy as np

from experiments.spectral_reachability import p1_matched_sampling_cuda_20261027_v1 as sampling
from experiments.spectral_reachability import p1_witness_physical_20261027_v1 as physical

source, core = physical.source, physical.core
require, sha, read = physical.require, physical.sha, physical.read
ROOT=Path(__file__).resolve().parents[2]
NAME='p1_sampling_physical_20261027_v1'
ARMS=('cem','iid_random')
CAP, FLOOR=8*1024**2, 200*1024**2


def contract():
    return dict(roots=2, arms=list(ARMS), branches=4,
        order='original_source_slot_order_then_cem_iid', resets=8, selected_steps=40,
        explicit_renders=44, saved_png=12, checkpoints=[0,5,10],
        reset_internal_physics_steps=16, internal_goal_renders=8,
        internal_counts='completed-reset implied; NOT instrumented actual physics substeps',
        initial='original_random_seed_then_original_sample_state_then_placement_seed',
        selection='sealed_sampling_all60_stable_argmin_then_independent_exact_H2_replay',
        guard='sealed_physical.branch; fresh initial and each successor RGB; native terminal stops',
        terminal='retain partial; no override, replacement, retry, or endpoint relabel',
        diagnostic='unchanged native and independent wrapped discrepancy; neither grants admission',
        history='causal_padding_not_observed_transitions', neural_calls=0,
        artifact_cap_bytes=CAP, reserve_bytes=FLOOR, P1_admission=False,
        model_task_eligible=False, equal_compute=False, independent_confirmation=False,
        freshness='reused calibration roots; no new independent sample')


def journal(path):
    """Bounded lossless saved evidence; gzip footer required, no partial acceptance."""
    require(Path(path).stat().st_size<=CAP,'bounded compressed sampling journal')
    decoder=zlib.decompressobj(31)
    raw=decoder.decompress(Path(path).read_bytes(),16*1024**2+1)
    require(len(raw)<=16*1024**2 and decoder.eof and not decoder.unused_data and
            not decoder.unconsumed_tail, 'complete single bounded gzip journal')
    return [json.loads(line) for line in raw.splitlines()]


def selection(task, arm, report, events, seed):
    require(report['schema']==sampling.NAME+'.arm' and report['status']=='sampling_completed_not_admitted'
        and report['error'] is None and not report['secondary_errors'] and
        report['method']==arm and report['seed']==seed and report['task_id']==core.task_id(task),
        'complete original same-task sampling arm')
    candidates=[e['candidate'] for e in events if e['kind']=='candidate_complete']
    require(len(candidates)==60 and [c['index'] for c in candidates]==list(range(60)), 'all60 original ordered candidates')
    for i,c in enumerate(candidates):
        require(c['batch']==i//10 and c['slot']==i%10,'original candidate slots')
        core.check_prediction(task,dict(task_id=core.task_id(task),logits=c['logits'],commands=c['commands'],
            contexts=c['contexts'],whole_replay=c['contexts'][-1],segmented_replay=c['contexts'],cost=c['cost']))
    require(report['candidates']==[{k:c[k] for k in ('index','batch','slot','logits','cost')} for c in candidates],
        'summary and full journal candidate identity')
    require(not np.asarray(candidates[0]['logits']).any() and
        all(np.asarray(c['logits']).any() for c in candidates[1:]), 'onezero59stochastic')
    best=min(candidates,key=lambda c:(c['cost'],c['index']))
    require(report['winner']==best,'stable model-only all60 argmin, never physical reselection')
    winner=report['witness']; core.check_prediction(task,winner)
    require(all(winner[k]==best[k] for k in ('commands','logits','contexts','cost')), 'independent fullH2 exact winner')
    attempts=[e['event'] for e in events if e['kind']=='operation_attempted']
    completed=[e['event'] for e in events if e['kind']=='operation_completed']
    require(len({e['index'] for e in attempts})==len(attempts) and
        [dict(e,completed=True) for e in attempts]==completed,'every original operation completed once in order')
    costs=sampling.counts(completed,arm)
    require(report['resources']==costs and all(costs[p][k]==dict(attempted=n,completed=n)
        for p,b in sampling.budgets(arm).items() for k,n in b.items()), 'exact durable sampling/winner resources')
    physical.raw_commands(winner['commands'])
    return winner, costs


def collect(sampling_prepared, sampling_prepare_sha, sampling_report, sampling_report_sha):
    prepared=Path(sampling_prepared).resolve(); report=Path(sampling_report).resolve()
    source.hash_value(sampling_prepare_sha); source.hash_value(sampling_report_sha)
    bindings={str(prepared/'prepare.json'):sampling_prepare_sha,str(report):sampling_report_sha}
    physical.bindings_check(bindings)
    p=sampling.verify(prepared,sampling_prepare_sha); r=read(report); inputs=p['inputs']
    require(r['schema']==sampling.NAME+'.report' and r['status']=='cuda_completed_not_admitted' and
        not r['errors'] and r['source_verified'] is True and r['model_unchanged'] is True and
        r['model_hash_before'] is not None and r['model_hash_before']==r['model_hash_after'] and
        r['prepare_sha256']==sampling_prepare_sha and r['P1_admission'] is False and
        r['new_image_calls']==r['new_physical_steps']==0, 'actual sealed sampling completion, no admission')
    direct=sampling.bridge.upstream.verify(inputs['direct_prepared'],inputs['direct_prepare_sha256'])
    sp=Path(direct['sourcepack']).resolve(); sourcepack=read(sp/'sourcepack.json')
    bindings[str(sp/'sourcepack.json')]=direct['sourcepack_sha256']
    require(sourcepack['plan']==direct['source_plan'] and len(sourcepack['slots'])==2,'same original two source slots')
    tasks={x['root_sha256']:x for x in inputs['tasks']}
    rows={(x['root_sha256'],x['method']):x for x in r['arms']}
    roots=[x['metadata']['root_sha256'] for x in sourcepack['slots']]
    require(len(set(roots))==len(tasks)==2 and set(roots)==set(tasks) and len(r['arms'])==len(rows)==4 and
        set(rows)=={(root,arm) for root in roots for arm in ARMS}, 'exact root/method census no substitution')
    bindings.update(p['sources_sha256']); bindings.update(inputs['bindings']); bindings.update(inputs['model_bindings'])
    bindings.update(direct['bindings']); bindings.update(direct['sources_sha256']); bindings.update(direct['source_plan']['source_closure'])
    bindings[str(prepared/'protocol.json')]=sha(prepared/'protocol.json')
    plans=[]; totals={k:dict(attempted=0,completed=0) for k in sampling.budgets('cem')['sampling']}
    for source_row in sourcepack['slots']:
        root=source_row['metadata']['root_sha256']; item=tasks[root]; task=item['task'];core.validate_task(task)
        for arm in ARMS:
            row=rows[root,arm]
            require(row['task_id']==item['task_id']==core.task_id(task) and row['error'] is None and
                    row['status']=='sampling_completed_not_admitted','same actual arm/task')
            paths={}
            for field in ('report','journal'):
                name=row[field+'_file']; require(Path(name).name==name,'flat actual artifact')
                path=report.parent/name; paths[field]=path; bindings[str(path)]=row[field+'_sha256']
            physical.bindings_check({str(v):bindings[str(v)] for v in paths.values()})
            saved=read(paths['report']); events=journal(paths['journal'])
            seed=sampling.seed_for(next(i for i,x in enumerate(inputs['tasks']) if x['root_sha256']==root),arm)
            winner,cost=selection(task,arm,saved,events,seed)
            require(row['resources']==cost,'top-level sampling resources exact')
            for phase in cost.values():
                for k,v in phase.items():
                    for state in ('attempted','completed'):totals[k][state]+=v[state]
            raw_path=sp/source_row['raw_source_file']; bindings[str(raw_path)]=source_row['raw_source_sha256']
            plan=dict(root_sha256=root,task_id=item['task_id'],candidate_id=source_row['slot']['candidate_id'],arm=arm,
                source_metadata=source_row['metadata'],raw_source_file=str(raw_path),raw_source_sha256=source_row['raw_source_sha256'],
                commands=winner['commands'],prediction=winner,prediction_sha256=source.digest(winner),
                sampling_winner_sha256=source.digest(saved['winner']),origin=str(paths['report']),search_cost=cost,
                goal_Q_context_sha256=source.digest({k:task[k] for k in ('goal','projection','root_context','action_mean','action_std','model_sha256')}))
            plan['selection_sha256']=source.digest(plan);plans.append(plan)
    require(r['resources']==totals,'aggregate actual sampling work')
    physical.bindings_check(bindings)
    return dict(arguments=dict(sampling_prepared=str(prepared),sampling_prepare_sha=sampling_prepare_sha,
        sampling_report=str(report),sampling_report_sha=sampling_report_sha),plans=plans,
        source_plan=sourcepack['plan'],bindings=bindings,inherited_cost=inputs['inherited'],sampling_cost=totals)


def own_sources():
    paths=set(physical.own_sources())|{str(p) for p in sampling.source_files()}
    paths|={str(Path(__file__).resolve()),str(ROOT/'tests'/('test_'+NAME+'.py')),str(ROOT/'innerdocs'/(NAME+'.md'))}
    return {p:sha(p) for p in sorted(paths)}


def prepare(arguments, output):
    data=collect(**arguments)
    p=dict(schema=NAME+'.prepare',contract=contract(),inputs=data,sources=own_sources())
    physical.Store(output).json('prepare.json',p)
    return p


def verify(prepared, expected_sha):
    path=Path(prepared)/'prepare.json'
    require(sha(path)==expected_sha,'external prepare SHA')
    p=read(path)
    require(p['schema']==NAME+'.prepare' and p['contract']==contract(),'new fixed four-arm schema/contract')
    require(p['sources']==own_sources(),'complete unchanged source closure')
    physical.bindings_check(p['inputs']['bindings'])
    require(collect(**p['inputs']['arguments'])==p['inputs'],'reconstruct original sampling winners/source')
    return p


def run(prepared, prepare_sha, output, factory, *, authorize_physics=False):
    require(authorize_physics is True,'MAIN explicit physics authorization')
    p=verify(prepared,prepare_sha);store=physical.Store(output)
    counts={k:dict(attempted=0,completed=0) for k in ('reset','render','step')}
    rows=[dict(root_sha256=x['root_sha256'],arm=x['arm'],selection_sha256=x['selection_sha256'],status='not_started',
        first_error=None,secondary_errors=[],trace=[]) for x in p['inputs']['plans']]
    result=dict(schema=NAME+'.report',prepare_sha256=prepare_sha,contract=contract(),rows=rows,counts=counts,
        status='running',errors=[],neural_calls=0,P1_admission=False,model_task_eligible=False,
        independent_confirmation=False,equal_compute=False,inherited_cost=p['inputs']['inherited_cost'],
        sampling_cost=p['inputs']['sampling_cost'],new_independent_samples=0)
    def error(exc,phase):return dict(phase=phase,type=type(exc).__name__,message=str(exc))
    try:
        store.emit('run_started',prepare_sha256=prepare_sha,contract=contract())
        for i,plan in enumerate(p['inputs']['plans']):
            env=None
            try:
                store.check(262144);env=factory();body=env.unwrapped
                backend=inspect.getsourcefile(type(body))
                require(backend is not None and p['inputs']['source_plan']['source_closure'].get(str(Path(backend).resolve()))==
                    p['inputs']['source_plan']['environment_sha256'],'actual original native constructor source')
                require(body.relative is True and body.action_scale==100 and body.render_mode=='rgb_array' and
                    body.render_size==224,'original native raw mapping and render contract')
                rows[i]=physical.branch(plan,p['inputs']['source_plan'],env,store,counts)
            except (Exception,KeyboardInterrupt) as exc:
                if rows[i]['first_error'] is None:rows[i]['first_error']=error(exc,'branch_setup')
                else:rows[i]['secondary_errors'].append(error(exc,'branch_setup'))
                rows[i]['status']='failed'
            finally:
                if env is not None:
                    try:env.close()
                    except (Exception,KeyboardInterrupt) as exc:
                        err=error(exc,'close')
                        if rows[i]['first_error'] is None:rows[i]['first_error']=err
                        else:rows[i]['secondary_errors'].append(err)
                        rows[i]['status']='failed'
                store.emit('branch_closed',receipt=rows[i])
        complete=all(x['status']=='completed' and x['full_horizon'] and x['initial_exact'] for x in rows)
        result['status']='physical_calibration_recorded' if complete else 'partial_or_failed'
        if complete:
            require(counts=={k:dict(attempted=n,completed=n) for k,n in dict(reset=8,step=40,render=44).items()},'exact four-arm global census')
            require(sum(len(x['checkpoints']) for x in rows)==12,'exact12 saved0/5/10 PNGs')
    except (Exception,KeyboardInterrupt) as exc:
        result['errors'].append(error(exc,'execution'));result['status']='partial_or_failed'
    finally:
        try:verify(prepared,prepare_sha)
        except (Exception,KeyboardInterrupt) as exc:
            result['errors'].append(error(exc,'final_bindings'));result['status']='partial_or_failed'
        semantics=p['inputs']['source_plan']['reset_semantics']
        result['reset_internal_physics_steps']=dict(completed_reset_implied=counts['reset']['completed']*semantics['internal_physics_steps_per_reset'],
            attempted_upper_bound=counts['reset']['attempted']*semantics['internal_physics_steps_per_reset'],actual_observed=None)
        result['internal_goal_renders_completed_reset_implied']=counts['reset']['completed']*semantics['internal_renders_per_reset']
        result['public_step_internal_physics_steps']=dict(actual_observed=None,label='not instrumented; public steps counted separately')
        try:
            store.emit('run_finished',status=result['status'],counts=counts,errors=result['errors'])
            result['journal_sha256']=sha(store.path/'journal.jsonl');store.json('report.json',result)
        except (Exception,KeyboardInterrupt) as exc:
            result['errors'].append(error(exc,'final_write'));result['status']='partial_or_failed'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','run'))
    for name in ('sampling-prepared','sampling-prepare-sha','sampling-report','sampling-report-sha','prepared','prepare-sha','output'):
        parser.add_argument('--'+name,required=False)
    parser.add_argument('--authorize-physics',action='store_true');a=parser.parse_args()
    if a.command=='prepare':
        args={k:getattr(a,k) for k in ('sampling_prepared','sampling_prepare_sha','sampling_report','sampling_report_sha')}
        prepare(args,a.output);print(sha(Path(a.output)/'prepare.json'))
    elif a.command=='verify':print(verify(a.prepared,a.prepare_sha)['schema'])
    else:
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224)
        r=run(a.prepared,a.prepare_sha,a.output,factory,authorize_physics=a.authorize_physics)
        print(source.canonical(r))
        if r['status']!='physical_calibration_recorded':raise SystemExit(2)


if __name__=='__main__':main()
