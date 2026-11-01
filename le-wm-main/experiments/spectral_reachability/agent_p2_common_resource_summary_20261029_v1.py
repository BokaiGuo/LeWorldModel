"""Saved-only six-slot common-resource table. No model/physical executor.

Imported runtime strict authenticates its original trace; this is not an
independent gradient check. Only identity joins and float64 Q are recomputed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import agent_p2_common_resource_grasp_cuda_20261029_v1 as runtime

NAME='agent_p2_common_resource_summary_20261029_v1'
ROOT=runtime.ROOT
need=runtime.need
equal=runtime.equal
CAP,FLOOR=16*1024**2,200*1024**2
PINS={str(Path(runtime.__file__).resolve()):'ea763d96eee1040652a3ea02badcb5e8e37b5781530da721f73d2a63d9cb147d',
      str(ROOT/'tests/test_agent_p2_common_resource_grasp_cuda_20261029_v1.py'):'a8917187abdc5a6a53d4410be2adbb88c6d778b98a9153ffbcae98ca46717abb',
      str(ROOT/'innerdocs/agent_p2_common_resource_grasp_cuda_20261029_v1.md'):'436a231041a20e82911214bcadfbef91ca5a103bbdab13767b3ca88f93202f59'}
FIELDS=('packet','predict_API','predict_samples','action_encoder','predictor','pred_proj',
    'predictor_context_tokens','pred_proj_rows','Jacobian','mathematical_VJP_requested',
    'Jacobian_elements_requested','AD_primal_packet_steps','AD_primal_packet_steps_requested',
    'scalar_backward','Adam_step','virtual_noise','root_SVD','root_ridge','encoder','projector','physics','training')


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def sha_bytes(b):return hashlib.sha256(b).hexdigest()


def read_bound(path,expected,cap=CAP):
    runtime.grasp.inputs.core.hash_value(expected);p=Path(path).absolute()
    need(p.is_file() and not p.is_symlink() and p.stat().st_size<=cap,'bounded regular JSON')
    b=p.read_bytes();need(len(b)<=cap and sha_bytes(b)==expected,'actual external byte SHA')
    return runtime.grasp.inputs.parse(b)


def scalar_q64(terminal,goal):
    z=np.asarray(terminal,dtype=np.float64);g=np.asarray(goal,dtype=np.float64)
    need(z.shape==g.shape==(192,) and np.isfinite(z).all() and np.isfinite(g).all(),'finite native192 goal/terminal')
    delta=np.subtract(z,g,dtype=np.float64)
    q=float(np.mean(np.multiply(delta,delta,dtype=np.float64),dtype=np.float64))
    need(np.isfinite(q),'finite Q64');return q


def domain():
    return dict(raw_steps=15,packet_steps=5,horizon=3,context_length=3,batch=1,logit_dimension=30,
        logit_L2_cap=float(runtime.old.CAP_LOGIT),chart='numpy.tanh(float64).astype(float32)',
        raw_dtype='float32',normalization=runtime.grasp.inputs.normalization(),Q='identity192_mean_squared_no_fit')


def check_logits(value):
    x=np.asarray(value,dtype=np.float64)
    need(x.shape==(30,) and np.isfinite(x).all() and np.linalg.norm(x)<=runtime.old.CAP_LOGIT,'legal exact joint30 logits')
    return x


def trace_metadata(events):
    """Stream only fixed diagnostic metadata, never retain all neural arrays."""
    traces=[dict(event_counts=Counter(),last_complete=None,last_accepted=None,initialized=None,
        commits=0,accepted_commits=0,refusals=[],cpu_attempt=Counter(),cpu_return=Counter(),error=None) for _ in runtime.ORDER]
    index=0
    for ordinal,e in enumerate(events):
        kind=e['kind']
        if kind in ('load_intent','consumed_buffer_frozen'):
            need(index==0,'loader before arms');continue
        if kind=='arm_sealed':
            need(e['index']==index and index<6,'fixed arm seal order');index+=1;continue
        need(kind=='arm_event' and index<6 and e['index']==index,'exact six arm trace scope')
        t=traces[index];p=e['payload'];k=p['kind'];t['event_counts'][k]+=1
        if k=='native':t['event_counts']['native:'+p['payload']['kind']]+=1
        if k in ('cpu_intent','cpu_return'):t['cpu_attempt' if k=='cpu_intent' else 'cpu_return'][p['resource']]+=1
        if k=='g_native':
            n=p['payload'];t['event_counts']['native:'+n['kind']]+=1
            if n['kind']=='initialized' and p['allowed']:
                t['initialized']=dict(x=deepcopy(n['x']),journal_ordinal=ordinal)
            if not p['allowed']:
                t['refusals'].append(dict(journal_ordinal=ordinal,kind=n['kind'],resource=n.get('resource'),replay=None))
        if k=='g_admission' and not p['allowed']:
            t['refusals'].append(dict(journal_ordinal=ordinal,kind='unit_admission',resource=None,replay=p['replay']))
        if k=='g_error':
            need(t['error'] is None,'single grasp execution error')
            t['error']=dict(type=p['type'],message=p['message'],journal_ordinal=ordinal)
        if k=='g_commit':
            c=dict(x=deepcopy(p['x']),accepted=p['accepted'],journal_ordinal=ordinal,unit_ordinal=t['commits'],now=p['now'])
            t['commits']+=1;t['last_complete']=c
            if p['accepted']:t['last_accepted']=c;t['accepted_commits']+=1
    need(index==6,'all six trace seals');return traces


def snapshot(x,origin):
    a=check_logits(x)
    return dict(logits_sha256=sha_bytes(a.astype('<f8').tobytes()),logits=a.tolist(),origin=origin,
        last_complete_not_necessarily_exported=True)


def root_identity(slot,model):
    need(slot['condition']=='off','original off task')
    return dict(root_slot=slot['root_slot'],task_id=slot['task_id'],task_file=slot['task_file'],task_sha256=slot['task_sha256'],
        entire_source_slot_sha256=runtime.old.digest(slot),model_tensor_hash=model,
        root_context_sha256=runtime.old.digest(slot['packet']['root_context']),goal_sha256=runtime.old.digest(slot['packet']['goal']),
        warm_raw10_sha256=sha_bytes(np.asarray(slot['warm_raw10'],dtype='<f4').tobytes()),
        action_domain_sha256=runtime.old.digest(domain()),source_episode_independence=False)


def panel(prepared,report,strict,traces):
    """Pure saved arithmetic/join seam; production always calls full strict first."""
    need(strict['schema']==runtime.NAME+'.consume' and strict['status']=='all6_common_cap_records_verified_not_admitted'
        and strict['fixed_denominator']==6,'exact imported strict schema')
    equal(strict['contract'],runtime.contract(),'fixed original contract')
    need(len(strict['plans'])==len(strict['accounts'])==len(strict['resources'])==len(traces)==6,'six slots, no dropped denominator')
    need([(s['root_slot'],s['condition']) for s in prepared['slots']]==[(0,'off'),(1,'off')],'two known roots only')
    need(report['model_before']==report['model_after']==prepared['expected_model_hash'],'same actual frozen model')
    rows=[]
    for i,(root,arm) in enumerate(runtime.ORDER):
        slot=prepared['slots'][root];plan=strict['plans'][i];a=strict['accounts'][i];res=strict['resources'][i];t=traces[i]
        identity=root_identity(slot,prepared['expected_model_hash']);last=None;last_metric=None
        if arm=='grasp':
            need(a['completed_units']==t['commits'],'all actual complete commits preserved')
            phases=list(runtime.phases());last_unit=list(phases[a['completed_units']-1]) if a['completed_units'] else None
            chosen=t['last_accepted'] or t['initialized']
            if chosen is not None:
                x=runtime.grasp.unpack(chosen['x'],(30,),np.float64)
                last=snapshot(x,dict(kind='last_accepted_update' if t['last_accepted'] is not None else 'initialization_not_update',
                    journal_ordinal=chosen['journal_ordinal']))
            incomplete=a['incomplete_unit'];pending=deepcopy(a['pending'])
            completed_replay=a['completed']['packet']>=6 and t['event_counts']['native:export_complete']>0
        else:
            need(a['root']==root and a['arm']==arm,'strict account row/root/arm join')
            last_unit=a['finished_units'][-1] if a['finished_units'] else None
            incumbent=a['incumbent']
            if incumbent is not None:
                last=snapshot(incumbent['witness']['logits'],dict(kind='retained_model_incumbent',tag=incumbent['tag']))
                last_metric=scalar_q64(incumbent['witness']['contexts'][-1]['latent_history'][-1],slot['packet']['goal'])
                equal(last_metric,incumbent['cost'],'independent float64 incumbent Q')
            incomplete=a['active_unit'];pending=deepcopy(a['pending']);completed_replay=a['export_complete']
        model=None;missing=[]
        if plan is None:
            missing.append('no_strict_exported_plan')
            if not completed_replay:missing.append('no_complete_export_replay')
        else:
            need(plan['schema']==runtime.NAME+'.plan' and plan['root_slot']==root and plan['arm']==arm,'exact plan identity')
            for k in ('task_id','task_file','task_sha256'):equal(plan[k],slot[k],'same source '+k)
            equal(plan['root_context'],slot['packet']['root_context'],'same original root context')
            equal(plan['goal'],slot['packet']['goal'],'same exact goal')
            equal(plan['warm_raw10'],slot['warm_raw10'],'same causal warm10')
            equal(plan['model_hash'],prepared['expected_model_hash'],'same model')
            equal(plan['report_sha256'],strict['report_sha256'],'same actual producer report')
            need(last is not None and completed_replay,'export requires complete source action witness/replay')
            x=check_logits(last['logits']);raw=np.tanh(x).astype('<f4').reshape(15,2)
            need(raw.tobytes().hex()==plan['raw_float32_bits'] and np.asarray(plan['commands'],dtype='<f4').shape==(15,2)
                and raw.tobytes()==np.asarray(plan['commands'],dtype='<f4').tobytes(),'same action domain and actual raw bits')
            need(len(plan['contexts'])==3,'full H3 output contexts')
            scores=[scalar_q64(c['latent_history'][-1],slot['packet']['goal']) for c in plan['contexts']]
            equal(scores[-1],plan['model_Q'],'independent float64 exported goal Q')
            model=dict(terminal_goal_Q64=scores[-1],packet_goal_Q64=scores,reported_model_Q=plan['model_Q'],
                terminal_latent=deepcopy(plan['contexts'][-1]['latent_history'][-1]),raw_float32_sha256=sha_bytes(raw.tobytes()),
                selection_digest=plan['selection_digest'],strict_export=True)
        resources={}
        for k in FIELDS:
            values={key:res[key].get(k) for key in ('attempted','completed')}
            for v in values.values():need(v is None or type(v) is int and v>=0,'known integer resource or null')
            if all(v is not None for v in values.values()):need(values['completed']<=values['attempted'],'attempt/return order')
            resources[k]=values
        api=resources['predict_API'];samples=resources['predict_samples'];elapsed=a['elapsed_seconds']
        need(type(elapsed) in (int,float) and np.isfinite(elapsed) and elapsed>=0,'actual finite search wall')
        boxes={}
        for k,value in (('predict_API',api),('predict_samples',samples)):
            need(type(value['attempted']) is int and value['attempted']<=1902,'separate common predict cap')
            boxes[k]=dict(cap=1902,attempted=value['attempted'],completed=value['completed'],at_cap=value['attempted']==1902,
                exceeded=False,at_cap_is_not_proof_of_resource_refusal=True)
        boxes['search_wall']=dict(cap_seconds=450.,actual_seconds=elapsed,deadline_reached=elapsed>=450.,overrun=elapsed>450.,
            stopped_on_wall=a['stop']=='wall_cap',cooperative_not_preemptive=True)
        boxes['resource_cap_stop']=a['stop']=='resource_cap'
        rows.append(dict(index=i,root_slot=root,arm=arm,input_identity=identity,model=model,missing=missing,
            last_complete_unit=last_unit,last_accepted_or_incumbent=last,last_incumbent_Q64_not_export=last_metric,
            accepted_commit_count=t['accepted_commits'] if arm=='grasp' else None,
            incomplete_unit=incomplete,pending=pending,stop=a['stop'],export_missing=plan is None,
            actual_error=deepcopy(a.get('errors')) if arm!='grasp' else deepcopy(t['error']),
            resources=resources,boxes=boxes,refusals=deepcopy(t['refusals']),
            CPU_capture=dict(seconds=None,reason='included_in_search_no_separate_authenticated_timer',
                cpu_operation_attempts=dict(t['cpu_attempt']),cpu_operation_returns=dict(t['cpu_return']),
                event_counts=dict(t['event_counts']),post_search_persistence_seconds_per_arm=None),
            unknown=dict(engine_internal_backward_calls=res['engine_internal_backward_calls'],
                independent_JVP_calls=res['independent_JVP_calls'],total_FLOPs=res['total_FLOPs']),
            search_scope=runtime.contract()['search_scope'],P1_admission=False,P2_admission=False))
    pairs=[]
    for root in (0,1):
        group=[r for r in rows if r['root_slot']==root];need(len(group)==3 and {r['arm'] for r in group}=={'joint','composition','grasp'},'fixed root triple')
        for r in group:equal(r['input_identity'],group[0]['input_identity'],'same task/model/context/source/action-domain triple')
        pairs.append(dict(root_slot=root,rows=[r['index'] for r in group],arms=[r['arm'] for r in group],
            available_model_outputs=sum(r['model'] is not None for r in group),denominator=3,identity=group[0]['input_identity']))
    return dict(rows=rows,paired_roots=pairs)


def build_summary(runtime_output,report_sha,strict_receipt,strict_sha):
    for p,h in PINS.items():need(runtime.old.sha(p)==h,'frozen runtime source/test/doc')
    output=Path(runtime_output).absolute();report_path=output/'report.json'
    report=read_bound(report_path,report_sha,8*1024**2)
    external=read_bound(strict_receipt,strict_sha,64*1024**2)
    authenticated=runtime.consume(output,report_sha)  # FULL strict, never status-only.
    equal(external,authenticated,'entire persisted strict receipt equals actual full return')
    need(authenticated['report_sha256']==report_sha,'exact external report binding')
    prepared=read_bound(report['prepared'],report['prepare_sha256'],8*1024**2)
    traces=trace_metadata(runtime.journal(output))
    table=panel(prepared,report,authenticated,traces)
    bindings=deepcopy(authenticated['bindings'])
    additions={str(Path(strict_receipt).absolute()):strict_sha,str(Path(report['prepared']).absolute()):report['prepare_sha256'],**PINS}
    additions.update({str(p):runtime.old.sha(p) for p in own_files()})
    for p,h in additions.items():
        need(p not in bindings or bindings[p]==h,'no conflicting authority bindings');bindings[p]=h
    for p,h in bindings.items():need(runtime.old.sha(p)==h,'final unchanged complete saved closure')
    need(runtime.old.sha(report_path)==report_sha,'final report unchanged')
    return dict(schema=NAME+'.summary',status='six_saved_resource_model_rows_verified_not_admitted',
        runtime_output=str(output),report_sha256=report_sha,strict_receipt=str(Path(strict_receipt).absolute()),strict_sha256=strict_sha,
        fixed_denominator=6,known_roots=2,fresh_roots=0,**table,domain=domain(),
        authority=dict(full_imported_runtime_strict_rebuilt=True,independent_gradient_verification=False,
            independent_model_forward_verification=False,independent_operations=['identity joins','saved float64 goal-Q','cap and missing classification']),
        wall_scope=dict(per_arm_search_seconds=450.,stage_timings=deepcopy(report['timings']),
            per_arm_CPU_capture_seconds=None,per_arm_post_search_persistence_seconds=None,
            totals_are_not_per_arm_allocations=True,summary_or_strict_time_not_search_time=True),
        stage_cuda_peak_bytes=deepcopy(report['stage_cuda_peak_bytes']),
        inherited_encoding_cohort=deepcopy(authenticated['inherited_encoding_cohort']),bindings=bindings,
        equal_FLOPs=False,equal_total_compute=False,total_FLOPs=None,new_neural_calls=0,new_physics_calls=0,
        physical_outcomes_read=False,selected_winner=None,padding=False,P1_admission=False,P2_admission=False,fresh=False)


def write_new(path,value):
    p=Path(path).absolute();b=runtime.old.raw(value)
    need(p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink() and len(b)<=CAP,'exclusive bounded own summary')
    need(shutil.disk_usage(p.parent).free>=FLOOR+len(b),'disk floor')
    with p.open('xb') as f:need(f.write(b)==len(b),'complete write');f.flush();os.fsync(f.fileno())
    runtime.old.fsync_dir(p.parent);return dict(path=str(p),sha256=runtime.old.sha(p))


def verify(path,expected):
    value=read_bound(path,expected)
    equal(value,build_summary(value['runtime_output'],value['report_sha256'],value['strict_receipt'],value['strict_sha256']),'full deterministic saved summary rebuild')
    return dict(verified=True,summary_sha256=expected,fixed_denominator=6)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    b=sub.add_parser('build')
    for name in ('runtime-output','report-sha','strict-receipt','strict-sha','output'):b.add_argument('--'+name,required=True)
    v=sub.add_parser('verify');v.add_argument('--summary',required=True);v.add_argument('--summary-sha',required=True)
    a=p.parse_args(argv)
    if a.cmd=='build':result=write_new(a.output,build_summary(a.runtime_output,a.report_sha,a.strict_receipt,a.strict_sha))
    else:result=verify(a.summary,a.summary_sha)
    print(runtime.old.raw(result).decode(),end='');return 0


if __name__=='__main__':raise SystemExit(main())
