"""C3 fixed-panel saved budget and request scheduling, no model execution."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from experiments.spectral_reachability import agent_p3_c3_preregistered_analysis_20261029_v1 as prior
from experiments.spectral_reachability import agent_p3_c3_compact_host_cuda_20261029_v1 as host

NAME='agent_p3_c3_budget_schedule_saved_20261029_v1'
ROOT=host.ROOT
HOST=host.NAME
AXES=tuple(prior.AXES)
SLOTS=144
CAP=16*1024**2
need=prior.need


def sha(data):return hashlib.sha256(data).hexdigest()
def raw(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def identity(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def summarize_tree(slot, tree, base):
    """Recompute axes and distinguish unused job identities from refused queues."""
    need(tree.get('schema')==prior.original.NAME+'.tree' and tree.get('backend_kind')=='HOST_CUDA' and
         tree['status']==base['status'] and tree['mode']==base['mode'] and tree['K']==base['K'], 'tree/receipt identity')
    jobs=tree['jobs'];queues=tree['queues'];need(type(jobs) is list and len(jobs)==SLOTS and
        [j['slot'] for j in jobs]==list(range(SLOTS)),'all 144 fixed job identities')
    rr=tree['resource_receipt']; resources=base['resources']; axis={}
    for key in ('caps','attempted','completed'):
        need(rr[key]==resources[key] and set(rr[key])==set(AXES),'all original 27-axis values')
    for a in AXES:
        cap,attempted,completed=(resources[k][a] for k in ('caps','attempted','completed'))
        need(all(type(v) is int and v>=0 for v in (cap,attempted,completed)) and completed<=attempted<=cap,'valid integer resource vector')
        axis[a]=dict(cap=cap,attempted=attempted,completed=completed,
                     unspent_cap=cap-attempted,attempted_not_completed=attempted-completed)
    counts=Counter();assigned=[]; job_rows=[]
    for job in jobs:
        status=job['status'];request=job['request']
        need(status in ('not_requested','queued','returned_eligible','returned_ineligible','blocked_after_failure'), 'known job status')
        need((status=='not_requested')==(request is None),'requested identity has request')
        counts[status]+=1;job_rows.append(dict(slot=job['slot'],status=status))
        if request is not None:assigned.append(identity(request))
    queue_rows=[];accepted=[]
    for ordinal,q in enumerate(queues):
        req=q['requests'];decision=q['decision'];status=q['status']
        need(type(req) is list and status in ('queued','stopped_by_original_child_policy','expansion_cap_before_node_queue',
             'refused_unsupported_original_type','resource_cap_before_node_queue'),'original queue status')
        need(type(decision['candidates']) is int and type(decision['continue_expanding']) is bool and
             len(req)==(decision['candidates'] if decision['continue_expanding'] else 0),'queue request census')
        if status=='queued':accepted.extend(identity(x) for x in req)
        queue_rows.append(dict(ordinal=ordinal,status=status,generated_requests=len(req),
            horizon=decision['horizon'],extension_type=decision['extension_type'],program_type=decision['program_type'],
            candidates=decision['candidates'],continue_expanding=decision['continue_expanding'],
            minimum=q['minimum'],verification_reserve=q['verification_reserve']))
    need(Counter(assigned)==Counter(accepted),'queued requests equal assigned identities exactly')
    refused=sum(q['generated_requests'] for q in queue_rows if q['status'] not in ('queued','stopped_by_original_child_policy'))
    unused=counts['not_requested'];need(refused<=unused,'refused queue identities occupy unassigned capacity')
    return dict(slot=slot,root=base['root'],K=base['K'],mode=base['mode'],status=base['status'],
        plan_available=base['plan_available'],axis=axis,job_identity_denominator=SLOTS,job_status_counts=dict(sorted(counts.items())),
        jobs=job_rows,assigned_jobs=len(assigned),unassigned_job_identities=unused,
        queues=queue_rows,generated_queued_requests=len(accepted),generated_refused_requests=refused,
        unused_not_assigned_to_refused_requests=unused-refused,
        interpretation='refused queue requests are not assigned jobs and overlap unassigned job identity capacity',
        R_groups=base['R_groups'],eligible_counts=base['eligible_counts'],native_calls=base['native_calls'],
        AD_primal_predict_API=base['AD_primal_predict_API'],Jacobian_elements=base['Jacobian_elements'],
        wall_ns=base['wall_ns'],inclusive_operation_wall_ns=base['inclusive_operation_wall_ns'],
        FLOPs=None,backward_engine_calls=None,cuda_ns=None)


def build(receipt_path,receipt_sha,output):
    summary=prior.build(receipt_path,receipt_sha)
    receipt=prior.read_pinned(receipt_path,receipt_sha)
    p=Path(output).absolute();need(p.resolve()==p and p.is_dir(),'exact output directory')
    reader=host.Reader(p,receipt['manifest_sha256'],receipt['completion_sha256'])
    try:
        report=reader.report
        need(host.codec.file_sha(p/'report.json')==receipt['report_sha256'] and
             report['schema']==HOST+'.report' and report['status']=='twelve_slots_settled_not_admitted' and
             len(report['rows'])==12,'actual strict receipt report binding')
        result=[]
        for i,(row,base) in enumerate(zip(report['rows'],summary['rows'],strict=True)):
            need(all(row[k]==base[k] for k in ('slot','root','K','mode','status')) and row['slot']==i,'fixed twelve identities')
            result.append(summarize_tree(i,reader.result(i,row['result']),base))
    finally:reader.close()
    files=(Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'))
    return dict(schema=NAME+'.analysis',receipt_path=str(Path(receipt_path).absolute()),receipt_sha256=receipt_sha,
        output_path=str(p),report_sha256=receipt['report_sha256'],manifest_sha256=receipt['manifest_sha256'],
        completion_sha256=receipt['completion_sha256'],source_hashes={str(f):sha(f.read_bytes()) for f in files},
        denominator=12,request_identity_denominator=1728,axes=list(AXES),rows=result,
        trust='inherits external MAIN strict receipt; saved Reader verifies transport, no fullstrict/model execution',
        interpretation=dict(equal_FLOPs=False,same_actual_resources=False,scientific_admission=False,
            nested_wall='inclusive times must not be summed',unused_is_not_failed=True,
            refused_queue_overlaps_unassigned=True,unexecuted_is_not_zero_cost=True,
            fairness_or_benefit_claim=False,roots='two known roots only'))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('analyze','verify'))
    ap.add_argument('--receipt',required=True);ap.add_argument('--receipt-sha',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--analysis',required=True);ap.add_argument('--analysis-sha')
    a=ap.parse_args();value=build(a.receipt,a.receipt_sha,a.output);path=Path(a.analysis)
    if a.command=='analyze':
        need(a.analysis_sha is None and path.is_absolute() and path.resolve()==path and not path.exists(),'new absolute output')
        payload=raw(value)
        with path.open('xb') as f:need(f.write(payload)==len(payload),'short write');f.flush();os.fsync(f.fileno())
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        print(json.dumps(dict(path=str(path),sha256=sha(payload))))
    else:
        need(prior.read_pinned(path,a.analysis_sha)==value,'complete summary rebuild')
        print(json.dumps(dict(verified=True,analysis_sha256=a.analysis_sha)))


if __name__=='__main__':main()
