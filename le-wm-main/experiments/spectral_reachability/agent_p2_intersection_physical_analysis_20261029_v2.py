"""Fixed-denominator saved physical endpoint and resource analysis; no execution."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil

from experiments.spectral_reachability import agent_p2_intersection_physical_20261029_v2 as physical
from experiments.spectral_reachability import agent_p2_common_resource_physical_summary_20261029_v1 as scalar

NAME='agent_p2_intersection_physical_analysis_20261029_v2'
ROOT=physical.ROOT
require,raw,sha=physical.require,physical.raw,physical.sha
CAP,FLOOR=16*1024**2,200*1024**2
PINS={
    str(Path(physical.__file__).resolve()):'e2309903a600cc536d9d12ef746014ebfdaa0bea6a6097c326c43c46a9dbdc35',
    str(ROOT/'tests'/('test_'+physical.NAME+'.py')):'3d08bafb580000b083a8af7970b413f63f4353fb763eb1747637c9a797f9d1d5',
    str(ROOT/'innerdocs'/(physical.NAME+'.md')):'c0f92afc1dba78b2efe7f50ffeb9a7d15abc295879adabdc2fb9c0d1c91e435b',
    str(Path(scalar.__file__).resolve()):'95eb0fa6da34b7cabe1b1ea64bcba6eba98185081a5767651f973f8d1d6c1562',
}


def read_bound(path,expected,cap):
    physical.native.actual_hash(expected)
    p=Path(physical.native.absolute(str(path)))
    require(p.is_file() and not p.is_symlink() and p.stat().st_size<=cap,'bounded regular input')
    b=p.read_bytes();require(len(b)<=cap and hashlib.sha256(b).hexdigest()==expected,'external SHA mismatch')
    return physical.native.decode(b)


def analyze(report,receipt,report_sha,receipt_sha):
    require(report['schema']==physical.NAME+'.report' and report['contract']==physical.contract()
        and receipt['schema']==physical.NAME+'.consume'
        and report['status']==receipt['status'] and report['status'] in (physical.RECORDED,'partial_or_failed')
        and receipt['physical_report_sha256']==report_sha and report['fixed_denominator']==receipt['fixed_denominator']==6
        and report['known_roots']==receipt['known_roots']==2 and report['fresh_roots']==receipt['fresh_roots']==0
        and report['inherited']==receipt['inherited']==0 and report['rows']==receipt['rows']
        and report['counts']==receipt['counts'] and report['errors']==receipt['errors']
        and report['execution_binding']['planner_authority']==receipt['planner_authority'], 'exact physical strict receipt/report identity')
    flags=('P1_admission','P2_admission','P2_mechanism','scientific_admission','freshness','equal_FLOPs','equal_total_compute','physical_release','independent_confirmation')
    require(all(report[k] is False and receipt[k] is False for k in flags)
        and receipt['new_neural_calls']==receipt['new_physics_calls']==0, 'original science and consumer boundary')
    operations=report['actual_operation_receipts'];counts=report['counts'];caps=physical.contract()['caps']
    require(type(operations) is list and set(counts)==set(caps) and len(receipt['rows'])==6
        and len(receipt['planner_authority']['accounts'])==6, 'all operation axes and fixed accounts')
    per=[{k:dict(attempted=0,completed=0,failed=0) for k in caps} for _ in range(6)]
    for index,op in enumerate(operations):
        require(type(op['index']) is int and op['index']==index and type(op['branch']) is int and 0<=op['branch']<6
            and op['resource'] in caps and op['status'] in ('returned','failed'), 'actual complete resource operation identity')
        row=receipt['rows'][op['branch']]
        require(all(op[k]==row[k] for k in ('root_slot','condition','arm','selection_digest')), 'operation bound to fixed row')
        v=per[op['branch']][op['resource']];v['attempted']+=1;v['completed']+=op['status']=='returned';v['failed']+=op['status']=='failed'
    for k in caps:
        require(set(counts[k])=={'attempted','completed'} and type(counts[k]['attempted']) is type(counts[k]['completed']) is int
            and sum(x[k]['attempted'] for x in per)==counts[k]['attempted']<=caps[k]
            and sum(x[k]['completed'] for x in per)==counts[k]['completed']<=counts[k]['attempted'], 'actual all-slot resource census')
    rows=[]
    for i,(root,condition,arm) in enumerate(physical.ORDER):
        branch=receipt['rows'][i];account=receipt['planner_authority']['accounts'][i]
        require((branch['index'],branch['root_slot'],branch['condition'],branch['arm'])==(i,root,condition,arm)
            and account['root']==root and account['arm']==arm and branch['plan']==receipt['plans'][i], 'fixed ordered identity and cost account')
        responses=branch['responses'];require(type(responses) is list and [x['step'] for x in responses]==list(range(1,len(responses)+1))
            and len(responses)<=branch['own_returned']<=branch['own_attempted']<=15,'contiguous actual response prefix')
        status=branch['status'];require(status in ('horizon_completed','native_stopped','export_missing','engineering_failed','not_started'), 'known branch disposition')
        if status=='export_missing':require(branch['plan'] is None and not responses,'missing stays missing')
        if status=='horizon_completed':require(len(responses)==15 and branch['own_returned']==15,'actual full horizon')
        last=responses[-1] if responses else None
        if last is None:
            require(branch['final_state'] is None and branch['final_metrics'] is None and branch['native_success'] is None, 'missing endpoint is unknown')
            metrics=None
        else:
            require(branch['initial_exact'] is True and branch['final_state']==last['state'] and branch['native_success'] is last['native_success']
                and branch['terminated'] is last['terminated'] and branch['truncated'] is last['truncated'], 'actual last returned endpoint')
            metrics=scalar.scalars(last['state'],branch['goal'])
            require(metrics['native_success'] is last['native_success'] and metrics['native7_reward']==last['reward'], 'independent original native labels')
        checkpoints=[]
        for step in (5,10,15):
            response=next((x for x in responses if x['step']==step),None)
            cp=next((x for x in branch['checkpoints'] if x['step']==step),None)
            if cp is not None:
                require(response is not None and cp['state']==response['state'], 'same-time checkpoint/state binding')
            measured=None if response is None else scalar.scalars(response['state'],branch['goal'])
            if response is not None:
                require(measured['native_success'] is response['native_success'] and measured['native7_reward']==response['reward'], 'same-time native labels')
            checkpoints.append(dict(step=step,metrics=measured,RGB=None if cp is None else
                {k:deepcopy(cp[k]) for k in ('png','file_sha256','pixel_sha256')},
                missing=None if measured is not None else 'no_labelled_return_at_requested_step'))
        rows.append(dict(index=i,root_slot=root,condition=condition,arm=arm,status=status,
            task_id=branch['task_id'],selection_digest=branch['selection_digest'],endpoint_step=None if last is None else last['step'],
            endpoint_same_nominal_time=(last is not None and last['step']==15), endpoint_metrics=metrics,
            native_success=None if metrics is None else metrics['native_success'],
            terminated=branch['terminated'],truncated=branch['truncated'],native_stop_reason=branch.get('stop_reason'),
            warmup_attempted=branch['warmup_attempted'],warmup_returned=branch['warmup_returned'],
            own_attempted=branch['own_attempted'],own_returned=branch['own_returned'],
            missing_checkpoints=deepcopy(branch['missing_checkpoints']),first_error=deepcopy(branch['first_error']),
            secondary_errors=deepcopy(branch['secondary_errors']),native_operations=per[i],planning_account=deepcopy(account),
            checkpoints=checkpoints))
    pairs=[]
    for root in (0,1):
        for step in (5,10,15):
            for a,b in (('joint','composition'),('joint','grasp'),('composition','grasp')):
                left=next(x for x in rows if x['root_slot']==root and x['arm']==a)
                right=next(x for x in rows if x['root_slot']==root and x['arm']==b)
                require(left['task_id']==right['task_id'], 'same root/task identity for same-time comparison')
                lp=receipt['plans'][left['index']];rp=receipt['plans'][right['index']]
                if lp is not None and rp is not None:
                    require(all(lp[k]==rp[k] for k in ('task_id','task_file','task_sha256','root_context','goal','q','warm_raw10','native_root')),
                        'same full task/context/native source, not outcome-selected')
                lm=next(x['metrics'] for x in left['checkpoints'] if x['step']==step)
                rm=next(x['metrics'] for x in right['checkpoints'] if x['step']==step)
                keys=('joint_xy_error','agent_xy_error','block_xy_error','native_angle_term','wrapped_angle_error','agent_velocity_error','native7_distance','native7_reward')
                pairs.append(dict(root_slot=root,step=step,a=a,b=b,known=lm is not None and rm is not None,
                    delta_b_minus_a=None if lm is None or rm is None else {k:rm[k]-lm[k] for k in keys},
                    success_a=None if lm is None else lm['native_success'],success_b=None if rm is None else rm['native_success']))
    return dict(schema=NAME+'.analysis',report_sha256=report_sha,receipt_sha256=receipt_sha,
        fixed_denominator=6,known_roots=2,fresh_roots=0,rows=rows,same_time_pairs=pairs,
        fixed_checkpoint_denominator=18,fixed_same_time_pair_denominator=18,
        counts=deepcopy(counts),caps=deepcopy(caps),errors=deepcopy(report['errors']),
        physical_wall_seconds=report['wall_seconds'],physical_wall_scope='whole execution through pre-report measurement; final report fsync excluded',
        uninstrumented_constructor_work='unknown', per_slot_physical_wall_seconds=None, total_FLOPs=None,
        endpoint_success_counts=dict(fixed_denominator=6,known=sum(x['native_success'] is not None for x in rows),
            success=sum(x['native_success'] is True for x in rows),unknown=sum(x['native_success'] is None for x in rows)),
        interpretation=dict(endpoint='actual last returned step, not necessarily common time; early stop never padded',
            native7='original mixed-unit norm including raw angle and velocity; reward is its negative',
            angle='native threshold term and wrapped diagnostic separately',
            costs='actual physical attempted/completed/failed by slot, planner accounts separate; nested operations not additive',
            arithmetic='reuses frozen saved-scalar function; not an independent second implementation',
            authentication='inherits SHA-pinned MAIN physical strict receipt; no repeated full strict',
            selection=False,P1_admission=False,P2_mechanism=False,physical_release=False,equal_FLOPs=False,
            independent_confirmation=False))


def build(report_path,report_sha,receipt_path,receipt_sha):
    for path,digest in PINS.items():require(sha(path)==digest,'frozen source SHA: '+path)
    report=read_bound(report_path,report_sha,physical.META)
    receipt=read_bound(receipt_path,receipt_sha,physical.META)
    absolute=str(Path(report_path).absolute())
    require(receipt['bindings'].get(absolute)==report_sha,'strict receipt authenticates exact report path and SHA')
    result=analyze(report,receipt,report_sha,receipt_sha)
    result['arguments']=dict(report_path=absolute,report_sha=report_sha,receipt_path=str(Path(receipt_path).absolute()),receipt_sha=receipt_sha)
    require(sha(report_path)==report_sha and sha(receipt_path)==receipt_sha,'stable saved inputs')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('analyze','verify'))
    for name in ('report','report-sha','receipt','receipt-sha','analysis'):p.add_argument('--'+name,required=True)
    p.add_argument('--analysis-sha');a=p.parse_args(argv)
    result=build(a.report,a.report_sha,a.receipt,a.receipt_sha);out=Path(physical.native.absolute(a.analysis))
    if a.command=='analyze':
        b=raw(result);require(a.analysis_sha is None and out.parent==ROOT/'results' and out.name.startswith(NAME+'_') and not out.exists()
            and not out.is_symlink() and len(b)<=CAP and shutil.disk_usage(out.parent).free>=FLOOR+len(b),'exclusive bounded output')
        with out.open('xb') as f:require(f.write(b)==len(b),'complete write');f.flush();os.fsync(f.fileno())
        physical.paired.fsync_dir(out.parent);print(raw(dict(path=str(out),sha256=sha(out))).decode())
    else:
        saved=read_bound(out,a.analysis_sha,CAP)
        require(raw(saved)==raw(result) and sha(out)==a.analysis_sha,'whole analysis reconstruction')
        print(raw(dict(verified=True,sha256=a.analysis_sha)).decode())


if __name__=='__main__':main()

