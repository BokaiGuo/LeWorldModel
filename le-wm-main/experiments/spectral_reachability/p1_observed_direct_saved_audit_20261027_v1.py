"""Independent saved-gzip B8 arithmetic/provenance audit, never neural/physics.

Checks all raw AD rows, joint tanh chain, H2 contexts/replays, finite direct
updates and actual entry ledgers. No threshold authority or stall inference.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import os
from pathlib import Path
import shutil
import numpy as np

from experiments.spectral_reachability import p1_observed_direct_cuda_20261027_v2 as saved
from experiments.spectral_reachability import p1_two_stage_witness_20261027_v1 as inputs

core=saved.screen.core
require=core.require
NAME='p1_observed_direct_saved_audit_20261027_v1'
ROOT=saved.ROOT
OLD=ROOT/'results/p1_direct_cuda_calibration_main_20261027_v2_cuda_main01'
OLD_PREP=ROOT/'results/p1_direct_cuda_calibration_main_20261027_v2_prepared01'
OLD_PREP_SHA='24eaf8b02873b3d93ebeeadbead9ca56c2caa079cea8c30b308357ddd8a3d2e6'
PREP_SHA='5217f0e2ccab0da8d270383885aec86b61a85788bf5b88f863ffe79ef8cb9da9'
PREP=ROOT/'results/p1_observed_direct_cuda_20261027_v2_prepare01.json'
FLOOR=200*1024**2


def one(events,kind):
    rows=[e for e in events if e['kind']==kind]
    require(len(rows)==1,'one '+kind)
    return rows[0]


def project(x,cap):
    a=np.asarray(x,dtype=float).reshape(2,10).copy()
    for row in a:
        n=np.linalg.norm(row)
        if n>cap:row*=cap/n
    a=a.astype(np.float32)
    for row in a:
        if np.linalg.norm(row.astype(float))>cap:row*=np.float32(1-4*np.finfo(np.float32).eps)
    return a.astype(float)


def strip_task_ids(value):
    if isinstance(value,dict):return {k:strip_task_ids(v) for k,v in value.items() if k!='task_id'}
    if isinstance(value,list):return [strip_task_ids(v) for v in value]
    return value


def check_ledger(events,result):
    starts=[];pending=[];groups={};current='startup'
    for outer in events:
        if outer['kind'] in ('operation_attempted','operation_completed'):
            e=outer['event'];core.fields(e,'index phase evaluation_id resource samples completed','operation')
            require(e['phase']=='direct' and e['samples']==1 and type(e['samples']) is int and e['resource'] in result['budget'],'direct operation scope')
            if outer['kind']=='operation_attempted':
                require(e['index']==len(starts) and e['completed'] is False,'ordered attempted work')
                starts.append(deepcopy(e));pending.append(deepcopy(e));current=e['evaluation_id']
            else:
                require(pending and e==dict(pending.pop(),completed=True),'LIFO completion identity')
                starts[e['index']]['completed']=True
                require(e['evaluation_id']==current,'operation group identity')
        if outer['kind']!='direct_finished':groups.setdefault(current,[]).append(outer)
    require(not pending and starts==result['extended_journal'],'complete durable extended ledger exact')
    narrow=[dict(e,index=i) for i,e in enumerate(e for e in starts if e['resource'] in core.RESOURCES)]
    require(narrow==result['journal'],'narrow module ledger exact')
    counts={r:sum(e['samples'] for e in starts if e['resource']==r and e['completed']) for r in result['budget']}
    require(counts==result['budget'],'exact complete budget, not only predicted formula')
    b=result['configuration']['B2'];method=result['configuration']['method']
    expect={'startup':{}}
    for i in range(b+1):
        expect[f'run_0_iter_{i}']=dict(predict=6,action_encoder=6,**({'selection':1} if i else {}))
        if i:expect[f'opt_0_iter_{i}']=dict(predict=2,action_encoder=2,jacobian=1,vjp_directions=192,proposal=1,
            **(dict(svd=1,linear_solve=1) if method=='full_H2_GN' else {}))
    require(set(groups)==set(expect),'all direct iteration scopes')
    for group,want in expect.items():
        actual={r:sum(e['event']['samples'] for e in groups[group] if e['kind']=='operation_attempted' and e['event']['resource']==r)
                for r in result['budget']}
        require(actual=={r:want.get(r,0) for r in result['budget']},'per-iteration budget '+group)
    return groups,counts


def check_prediction(task,p,events):
    core.check_prediction(task,p)
    require(one(events,'prediction_complete')['prediction']==p,'prediction event/report exact')
    paths=[e for e in events if e['kind']=='prediction_path']
    require([e['mode'] for e in paths]==['nominal','whole_replay','segmented_replay'] and
            all(e['contexts']==p['contexts'] for e in paths),'three full independently paid H2 paths')
    packets=[e for e in events if e['kind']=='native_packet']
    require([(e['mode'],e['packet']) for e in packets]==[(mode,k) for mode in ('nominal','whole_replay','segmented_replay') for k in range(2)] and
            all(e['context']==p['contexts'][e['packet']] for e in packets),'all six native packet arrays exact')


def check_component(task,result,events):
    core.validate_task(task);c=result['configuration'];b=c['B2']
    require(c['method'] in saved.METHODS and b in (3,8) and c['initial_logits']==np.zeros((1,2,10)).tolist() and
            c['step_scale']==1. and c['damping']==.002 and c['extra_line_search_trials']==0,'fixed finite direct configuration')
    expected=dict(predict=6+8*b,action_encoder=6+8*b,jacobian=b,vjp_directions=192*b,
                  svd=b if c['method']=='full_H2_GN' else 0,linear_solve=b if c['method']=='full_H2_GN' else 0,proposal=b,selection=b)
    require(result['budget']==expected and result['task_id']==core.task_id(task) and
            result['status']=='direct_component_completed_not_admitted' and result['error'] is None and not result['secondary_errors'],
            'complete component task/budget/status')
    require(result['P1_admission'] is False and result['model_task_eligible'] is False,'no admission')
    require(events[0]==dict(kind='direct_started',task=task,configuration=c,budget=expected) and
            events[-1]==dict(kind='direct_finished',report=result),'original full terminal/start objects')
    groups,counts=check_ledger(events,result)
    require(len(result['direct_runs'])==1 and len(result['direct_runs'][0])==b+1,'all original trajectory nodes')
    run=result['direct_runs'][0];initial=run[0]
    require(initial['iteration']==0 and initial['proposal'] is None and initial['derivative'] is None and initial['accepted'] is False and
            initial['incumbent']['logits']==np.zeros((2,10)).tolist(),'original zero incumbent')
    check_prediction(task,initial['incumbent'],groups['run_0_iter_0'])
    gradients=[];matrices=[]
    for i,step in enumerate(run):
        require(step['iteration']==i and one(groups[f'run_0_iter_{i}'],'trajectory_step')==dict(kind='trajectory_step',restart=0,step=step),
                'all trajectory nodes durable exact')
        if not i:continue
        before=run[i-1]['incumbent'];d=step['derivative'];opt=groups[f'opt_0_iter_{i}']
        core.check_derivative(task,d,before)
        require(one(opt,'derivative_complete')['derivative']==d,'full derivative durable exact')
        rows=[e for e in opt if e['kind']=='jacobian_row']
        require([e['row'] for e in rows]==list(range(192)),'all ordered 192 raw VJP rows each iteration')
        raw=core.array([e['raw_control_derivative'] for e in rows],(192,20))
        j=core.array(d['raw_terminal_jacobian'],(192,20))
        require(np.array_equal(raw*(1-np.asarray(before['commands']).reshape(20)**2)[None],j),'raw-to-logit tanh chain exactly once')
        joint=one(opt,'joint_logit_jacobian');forward=one(opt,'jacobian_forward_complete')
        require(joint['terminal']==forward['terminal']==d['at_terminal'] and joint['jacobian']==d['raw_terminal_jacobian'] and
                forward['logits']==before['logits'],'joint full-H2 AD nominal point')
        packets=[e for e in opt if e['kind']=='jacobian_forward_packet']
        require([e['packet'] for e in packets]==[0,1] and [e['context'] for e in packets]==before['contexts'],
                'both paid AD packet contexts equal incumbent full H2, not suffix-only')
        q=np.asarray(task['projection']);a=q@j;r=q@(np.asarray(d['at_terminal'])-task['goal'])
        g=2*a.T@r/192
        require(np.allclose(d['objective_gradient'],g,rtol=1e-8,atol=1e-10),'independent same-Q gradient arithmetic')
        delta=-np.linalg.solve(a.T@a/192+.002*np.eye(20),a.T@r/192) if c['method']=='full_H2_GN' else -g
        want=project(np.asarray(before['logits'])+delta.reshape(2,10),task['packet_logit_cap'])
        # Arithmetic tolerance only: independent normal solve vs original SVD,
        # followed by float32 rounding. NOT task/stall/admission thresholds.
        proposal=step['proposal'];x=core.array(proposal['logits'],(2,10))
        require(np.allclose(x,want,rtol=8*np.finfo(np.float32).eps,atol=8*np.finfo(np.float32).eps),'independent direct GN/GD proposal arithmetic')
        require(one(opt,'proposal_logits')['logits']==proposal['logits'],'proposal event exact')
        check_prediction(task,proposal,groups[f'run_0_iter_{i}'])
        accepted=proposal['cost']<before['cost']
        require(step['accepted'] is accepted and step['incumbent']==(proposal if accepted else before),'strict nonlinear incumbent selection')
        gradients.append(float(np.linalg.norm(g)));matrices.append(j)
    costs=[s['incumbent']['cost'] for s in run]
    return dict(resources=counts,costs=costs,gradient_norms=gradients,raw_J_rows=192*b,
        nonzero_J_iterations=sum(bool(np.any(j)) for j in matrices),
        changed_successive_J=sum(not np.array_equal(a,b) for a,b in zip(matrices,matrices[1:])),
        full_H2_evaluations=3*(b+1),paid_AD_H2_forwards=b)


def compare_prefix(new_task,new_result,old_task,old_result):
    provenance=('task_contract_sha256','metric_source_sha256')
    require({k:v for k,v in new_task.items() if k not in provenance}=={k:v for k,v in old_task.items() if k not in provenance},
            'same numerical Q/goal/model/root/action/history; only provenance hashes may differ')
    prefix=strip_task_ids(new_result['direct_runs'][0][:4])==strip_task_ids(old_result['direct_runs'][0])
    return dict(zero_exact=strip_task_ids(new_result['direct_runs'][0][0])==strip_task_ids(old_result['direct_runs'][0][0]),
                B0_through_B3_exact_ignoring_only_task_id=prefix,provenance_task_fields_allowed_to_differ=list(provenance))


def audit(directory,report_sha):
    directory=Path(directory).resolve();bindings={}
    def bind(path,expected=None):
        path=Path(path);h=saved.sha(path)
        if expected is not None:require(h==expected,'saved artifact SHA '+str(path))
        bindings[str(path)]=h
        return path
    top=saved.read(bind(directory/'report.json',report_sha))
    p=saved.verify(PREP,PREP_SHA);bind(PREP,PREP_SHA)
    require(top['prepare_sha256']==PREP_SHA and top['status']=='observed_direct_calibration_completed' and top['errors']==[] and
            top['source_verified'] is True and top['model_unchanged'] is True and top['model_hash_before']==top['model_hash_after'] and
            top['model_hash_before'] is not None and top['P1_admission'] is False and top['new_physical_steps']==0,'actual complete source/model status')
    old_inputs=inputs.collect(OLD_PREP,OLD_PREP_SHA,OLD)
    for f,h in {**p['source_sha256'],**p['source_bindings'],**p['model']['bindings'],**old_inputs['bindings']}.items():bind(f,h)
    images,eof=saved.recover_events(bind(directory/'image.events.jsonl.gz'))
    require(eof and len(images)==24,'all eight complete native image receipts')
    for i in range(8):
        a,b,c=images[3*i:3*i+3]
        require(a==dict(kind='native_image_attempted',resource='encoder',samples=1) and
                b==dict(kind='native_image_attempted',resource='projector',samples=1) and c['kind']=='native_image_completed' and
                c['counts']==dict(encoder=1,projector=1),'native whole encode completion receipts')
    data=saved.original.material(**dict(pack=p['inputs']['pack'],expected_sha=p['inputs']['source_sha'],
                                     source_prepare=p['inputs']['source_prepare'],source_prepare_sha=p['inputs']['source_prepare_sha']))
    old_tasks={x['root_sha256']:x['task'] for x in old_inputs['tasks']};tasks={}
    require(top['source_audit']==data['summary']==p['source_audit'],'true observed acquisition audit exact')
    for i,packet in enumerate(data['packets']):
        encoded=saved.read_gzip(bind(directory/f'task_{i}.json.gz'));t=encoded['task'];root=encoded['root_sha256']
        values=iter([images[3*(4*i+k)+2]['latent'] for k in range(4)])
        # Receipt-only callback; NO model/image forward. Independently bind RGB
        # bytes immediately below, rather than trusting reconstructed metadata.
        rebuilt=saved.original.encode_observed(packet,saved.original.template(data,saved.digest(p),t['model_sha256'],t['model_config_sha256']),
                                                lambda rgb:next(values))
        require(rebuilt==encoded and root in old_tasks and root not in tasks,'native observed task/root binding')
        for k,rgb in enumerate([*packet['rgb_history'],packet['goal_rgb']]):
            require(images[3*(4*i+k)+2]['rgb_sha256']==saved.observed.image_sha(rgb),'all native RGB identity receipts')
        tasks[root]=t
    require(len(top['direct'])==4 and len(tasks)==2,'all four actual arms')
    totals={r:0 for r in saved.TOTAL};rows=[];seen=set()
    for row in top['direct']:
        root,method=row['root_sha256'],row['method'];key=(root,method)
        require(key not in seen and root in tasks and method in saved.METHODS,'unique stable root/method arms');seen.add(key)
        i=next(i for i,x in enumerate(p['source_audit']['roots']) if x['root_sha256']==root)
        arm=directory/f'root_{i}_{method}'
        require(Path(row['report_path']).resolve()==arm/'report.json.gz' and Path(row['journal_path']).resolve()==arm/'direct_journal.jsonl.gz','exact local root/method paths')
        result=saved.read_gzip(bind(arm/'report.json.gz',row['report_sha256']))
        events,eof=saved.recover_events(bind(arm/'direct_journal.jsonl.gz',row['journal_sha256']));require(eof,'complete gzip footer')
        require(result['configuration']==saved.original.config(method),'original B4/B8 method protocol')
        receipt=check_component(tasks[root],result,events)
        old_i=next(j for j in range(2) if saved.read(OLD/f'task_{j}.json')['root_sha256']==root)
        old_arm=OLD/f'root_{old_i}_{method}'
        old_result=saved.read(bind(old_arm/'report.json'))
        old_events=saved.original.read_lines(bind(old_arm/'direct_journal.jsonl'))
        require(old_result['configuration']['B2']==3,'old B3 prefix, not new result-selected reference')
        check_component(old_tasks[root],old_result,old_events)
        comparison=compare_prefix(tasks[root],result,old_tasks[root],old_result)
        costs=receipt['costs'];receipt.update(comparison)
        receipt.update(root_sha256=root,method=method,B4=costs[4],B8=costs[8],B8_minus_B4=costs[8]-costs[4],
                       strict_improvement_B4_to_B8=costs[8]<costs[4])
        rows.append(receipt)
        for resource,n in receipt['resources'].items():totals[resource]+=n
    require(totals==saved.TOTAL and top['actual_direct_resources']=={k:dict(attempted=v,completed=v) for k,v in totals.items()},'all actual direct budgets independently reconstructed')
    require(top['actual_image_resources']=={k:dict(attempted=8,completed=8) for k in ('encoder','projector')},'successful image budget, not general return-fault accounting')
    for f,h in bindings.items():require(saved.sha(f)==h,'saved input changed during audit')
    exact=all(r['B0_through_B3_exact_ignoring_only_task_id'] for r in rows)
    return dict(schema=NAME,status='saved_arrays_audited_not_admitted' if exact else 'saved_arrays_prefix_discrepancy_not_admitted',
        bindings=bindings,rows=rows,actual_direct_resources=totals,successful_image_resources=dict(encoder=8,projector=8),
        old_B3_prefix_exact_all=exact,all_four_B4_to_B8_improve=all(r['strict_improvement_B4_to_B8'] for r in rows),
        model_unchanged=top['model_unchanged'],P1_admission=False,stall_established=False,new_neural_calls=0,new_physics_steps=0,
        limitations=['saved row/chain consistency is not independent AD/finite-difference correctness',
                    'predicted model Q only, not physical outcome or task gain','no independent threshold/witness admission',
                    'same two source families, no independent sample increment','image complete receipts do not prove general module-return fault accounting'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--report-sha',required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    try:r=audit(a.directory,a.report_sha)
    except BaseException as exc:r=dict(schema=NAME,status='audit_failed',error=dict(type=type(exc).__name__,message=str(exc)),P1_admission=False)
    r['auditor_sha256']=saved.sha(__file__)
    raw=(core.canonical(r)+'\n').encode();require(len(raw)<=96*1024,'small audit report')
    require(shutil.disk_usage(a.output.parent).free>=FLOOR+len(raw),'200MiB reserve')
    with a.output.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    print(core.canonical({k:v for k,v in r.items() if k not in ('bindings','rows')}))
    if r['status']=='audit_failed':raise SystemExit(2)


if __name__=='__main__':main()
