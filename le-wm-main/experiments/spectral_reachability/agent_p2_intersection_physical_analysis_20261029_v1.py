"""Fixed-denominator saved physical endpoint and resource analysis; no execution."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil

from experiments.spectral_reachability import agent_p2_intersection_physical_20261029_v1 as physical
from experiments.spectral_reachability import agent_p2_common_resource_physical_summary_20261029_v1 as scalar

NAME='agent_p2_intersection_physical_analysis_20261029_v1'
ROOT=physical.ROOT
require,raw,sha=physical.require,physical.raw,physical.sha
CAP,FLOOR=16*1024**2,200*1024**2
PINS={
    str(Path(physical.__file__).resolve()):'c6dc8d641f679a6e17079a921d3fd101dd25bc1f59620e6925cf97ee239eab3d',
    str(ROOT/'tests'/('test_'+physical.NAME+'.py')):'a1274081cc5661a1e010b72be6b98a4ac7be47eb28b557b8a5ac5856cb08d7a9',
    str(ROOT/'innerdocs'/(physical.NAME+'.md')):'13efa1ec356f86ba4998dd68f07f6348fa4eae8519e00ce460004cf28aea326a',
    str(Path(scalar.__file__).resolve()):'95eb0fa6da34b7cabe1b1ea64bcba6eba98185081a5767651f973f8d1d6c1562',
}


def read_bound(path,expected,cap):
    physical.native.actual_hash(expected)
    p=Path(physical.native.absolute(str(path)))
    require(p.is_file() and not p.is_symlink() and p.stat().st_size<=cap,'bounded regular input')
    b=p.read_bytes();require(len(b)<=cap and hashlib.sha256(b).hexdigest()==expected,'external SHA mismatch')
    return physical.native.decode(b)


def analyze(report,receipt,report_sha,receipt_sha):
    require(report['schema']==physical.NAME+'.report' and receipt['schema']==physical.NAME+'.consume'
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
        rows.append(dict(index=i,root_slot=root,condition=condition,arm=arm,status=status,
            task_id=branch['task_id'],selection_digest=branch['selection_digest'],endpoint_step=None if last is None else last['step'],
            endpoint_same_nominal_time=(last is not None and last['step']==15), endpoint_metrics=metrics,
            native_success=None if metrics is None else metrics['native_success'],
            terminated=branch['terminated'],truncated=branch['truncated'],native_stop_reason=branch.get('stop_reason'),
            warmup_attempted=branch['warmup_attempted'],warmup_returned=branch['warmup_returned'],
            own_attempted=branch['own_attempted'],own_returned=branch['own_returned'],
            missing_checkpoints=deepcopy(branch['missing_checkpoints']),first_error=deepcopy(branch['first_error']),
            secondary_errors=deepcopy(branch['secondary_errors']),native_operations=per[i],planning_account=deepcopy(account)))
    return dict(schema=NAME+'.analysis',report_sha256=report_sha,receipt_sha256=receipt_sha,
        fixed_denominator=6,known_roots=2,fresh_roots=0,rows=rows,
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
