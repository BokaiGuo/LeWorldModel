"""Strict saved consumer of terminal runtime-guard output. No model/physics calls.

Native validators are reused, NOT its incompatible whole-report exporter.
Journal hashing alone is never enough: reconstruct budgets and join all evidence.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import math
import time
from pathlib import Path

from experiments.spectral_reachability import agent_p2_runtime_guard_20261028_v1 as producer

NAME='agent_p2_guard_saved_consume_20261028_v1'
ROOT=producer.ROOT
native=producer.native
contract=producer.contract
C=producer.C
require=C.require
GUARD_SHA='616ec94445681ede449e525a45016b4e7d849ec488181674f570e25f5ca721d3'


def eq(a,b): return producer.encode(a)==producer.encode(b)
def sha(path): return producer.file_sha(path)


def raw_bytes(path,limit=96*1024**2):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=limit,'bounded regular evidence file: '+str(path))
    data=path.read_bytes();require(len(data)<=limit,'evidence grew beyond byte limit')
    return data


def read(path,limit=96*1024**2):return contract.prior.graph.parse(raw_bytes(path,limit))


def work_forward(kind):
    require(kind in ('predict','action_encoder'),'only actual native forward modules')
    return C.vector(**({ 'predict_calls':1,'predict_samples':1,'predict_context_tokens':3} if kind=='predict' else
                     {'action_encoder_calls':1,'action_encoder_samples':1,'action_encoder_context_tokens':3}))


def work_operation(e,terminal=False):
    k=e['kind']; require(k in producer.OP_KINDS,'exact operation type')
    if k in ('jacobian','suffix_jacobian'):
        width=20 if k=='jacobian' else 10
        require(eq(e['shape'],[192,width]) and e['vectorize'] is True,'exact native derivative shape')
        if k=='jacobian':require(type(e.get('scalar_output_vjp_directions_requested')) is int and e['scalar_output_vjp_directions_requested']==192,'explicit root mathematical VJP request')
        return C.vector(**{f'jacobian_192x{width}_calls':1,f'math_vjp_192x{width}_directions':192,
                           'jacobian_scalar_entries':192*width})
    if k=='suffix_solve':
        n=e['svd_calls'] if terminal else e['svd_calls_requested']
        require(type(n) is int and n in (0,1),'literal zero/nonzero-radius operation count')
        if terminal:require(type(e['bounded_solve_calls']) is int and e['bounded_solve_calls']==n,'solve/SVD closure')
        return C.vector(svd_192x10_calls=n,bounded_solve_10_calls=n)
    require(type(e['svd_calls_requested']) is int and e['svd_calls_requested']==1,'root SVD request')
    return C.vector(svd_192x20_calls=1,bounded_solve_20_calls=int(k=='bounded_ridge_svd'))


def scope_group(scope):
    if scope['phase']=='compilation':return 'base'
    if scope['phase']=='replay':return scope['arm']+'.replay'
    require(scope['phase'] in ('planning','suffix_compilation'),'known actual scope')
    return f'{scope["arm"]}.candidate.{scope["path"][0]}'


def reconcile_packet_ledgers(row):
    """Join native packet records to actual dual-module returns, not estimates.

    Compilation/AD entries are paid separately by reconcile; packet ledgers
    describe only planning and replay, so never add them again to resource sums.
    """
    require(len(row['results'])==2,'exact two native arms')
    summary=[]
    for arm,result in zip(native.ARMS,row['results'],strict=True):
        contract.prior.graph.keys(result,('arm','attempts','complete','error','faults','fixture_only',
            'ledger','nested40_minimum','selection_evaluations','wall_time_s','winner'))
        require(result['arm']==arm,'ordered packet ledger arms')
        entries=[e for e in row['actual_model_entries'] if e['scope']['arm']==arm
                 and e['scope']['phase'] in ('planning','replay')]
        n=contract.COUNTS[arm];ledger=result['ledger']
        require(len(result['attempts'])==n and all(e['completed'] is True for e in result['attempts']),
            'fixed candidate census including every failed/pending slot; incomplete cannot export')
        require(len(ledger)==2*n+4 and len(entries)==2*len(ledger),'all candidate and replay packet returns')
        for j,packet in enumerate(ledger):
            contract.prior.graph.keys(packet,('phase','path','packet','predict_calls','predict_samples',
                'action_encoder_calls','completed','wall_time_s'))
            require(type(packet['wall_time_s']) in (int,float) and math.isfinite(packet['wall_time_s'])
                and packet['wall_time_s']>=0,'finite host packet wall time, not device cost')
            for field in ('packet','predict_calls','predict_samples','action_encoder_calls'):
                require(type(packet[field]) is int,'literal packet counts; bool forbidden')
            require(all(type(x) is int for x in packet['path']),'literal packet path')
            ae,predict=entries[2*j:2*j+2]
            expected_scope=dict(arm=arm,phase=packet['phase'],path=packet['path'])
            require(ae['kind']=='action_encoder' and predict['kind']=='predict'
                and eq(ae['scope'],expected_scope) and eq(predict['scope'],expected_scope)
                and ae['completed'] is predict['completed'] is packet['completed'] is True
                and ae['samples']==predict['samples']==packet['predict_calls']==packet['predict_samples']
                    ==packet['action_encoder_calls']==1
                and ae['context_tokens']==predict['context_tokens']==3,'packet ledger to actual module return seam')
        summary.append(dict(arm=arm,packets=len(ledger),module_returns=len(entries),
            accounting='crosscheck_only_NOT_additional_work',flops=None))
    return summary


def reconcile(records,row,ready,native_sha):
    """Independent event interpreter, NOT DispatchGuard replay on self-receipts."""
    i=ready['root_index']; root=ready['root_sha256']; resources=contract.allocation()
    require(records[0]['kind']=='header' and eq(records[0]['payload'],dict(binding=dict(root=root,
        native_prepare_sha256=native_sha,ready_sha256=C.digest(ready),wrapper_source_sha256=GUARD_SHA,
        resource_contract_sha256=C.digest(resources)),computation_observed=False)),'exact journal header bindings')
    require(records[-1]['kind']=='root_terminal' and eq(records[-1]['payload'],row['guard']),'terminal journal/root guard binding')
    expected_groups=[('base','shared',resources['graph_base_shared_upper'])]
    for a in native.ARMS:
        expected_groups += [(f'{a}.candidate.{j}',a,resources['arms'][a]['per_candidate_upper']) for j in range(contract.COUNTS[a])]
        expected_groups += [(a+'.replay',a,resources['winner_replay_upper'])]
    groups={}; reserved={a:C.vector() for a in native.ARMS}; pending={}; seen=set(); current=None
    terminals=[]; attempts=[]; operations=[]; arrays=[]; progress=[]; metrics=0; candidates=[]
    by_candidate={(r['arm'],j):e for r in row['results'] for j,e in enumerate(r['attempts'])}
    last_reserve=None; last_time=records[0]['elapsed_ns']
    payload_fields=dict(group_reserve_intent=('group','arm','upper','computation_observed'),
        operation_reserve_intent=('token','group','upper','native_evidence','computation_observed'),
        entry_boundary=('token','group','observation','actual_completed'),
        return_observed=('token','group','success','observed_completed','native_evidence','observation'),
        native_evidence=('event',),candidate_terminal=('arm','index','success','native_event_sha256'))
    for record in records[1:-1]:
        require(type(record['elapsed_ns']) is int and record['elapsed_ns']>=last_time,'monotone host event clock')
        last_time=record['elapsed_ns']; k=record['kind']; p=record['payload']
        require(k in payload_fields,'unknown/failure/duplicate-terminal event')
        contract.prior.graph.keys(p,('root_slot','root_sha256',*payload_fields[k]))
        require(type(p['root_slot']) is int and p['root_slot']==i and p['root_sha256']==root,'per-event root identity')
        if last_reserve is not None:
            require(k=='entry_boundary' and p['token']==last_reserve,'reserve must be durable immediately before entry boundary')
        if k=='group_reserve_intent':
            require(not pending and len(groups)<len(expected_groups),'no pending work at next group')
            key,a,upper=expected_groups[len(groups)]
            require(eq(p,dict(root_slot=i,root_sha256=root,group=key,arm=a,upper=upper,computation_observed=False)),
                    'fixed base/candidate/replay group order and allocation')
            for arm in (native.ARMS if a=='shared' else (a,)):
                reserved[arm]=C.add(reserved[arm],upper)
                require(all(reserved[arm][x]<=resources['common_standalone_cap'][x] for x in C.KNOWN),'standalone cap')
            groups[key]=dict(arm=a,upper=deepcopy(upper),subreserved=C.vector(),returned=C.vector());current=key
        elif k=='operation_reserve_intent':
            token=p['token'];e=p['native_evidence']
            require(current is not None and token not in seen and p['group']==current and p['computation_observed'] is False,
                    'unique paid primitive reservation')
            if token.startswith('forward.'):
                index=len(attempts);require(e['kind']=='entry_attempt' and token==f'forward.{index}','native entry attempt identity')
                expected=dict(row['actual_model_entries'][index],completed=False)
                require(eq(e['entry'],expected) and scope_group(expected['scope'])==current,'actual entry attempt/full scope seam')
                work=work_forward(expected['kind']);attempts.append(e['entry'])
            elif e.get('kind') in producer.OP_KINDS:
                require(token==e['kind']+'.'+str(e.get('index')) and e['completed'] is False,'operation request token')
                require(eq(e,row['operations'][len(operations)]),'request matches actual operations stream')
                expected_group='base' if e['kind'] in ('jacobian','bounded_ridge_svd','mode_svd') else f'{native.ARMS[1]}.candidate.{e["index"]}'
                require(current==expected_group,'derivative/solve charged to its own context group')
                work=work_operation(e);operations.append(e)
            else:
                require(eq(e,dict(kind='metric_call')) and token==f'metric.{metrics}' and '.candidate.' in current,'one ordered metric per candidate')
                require(not pending and groups[current]['subreserved']['selection_evaluations']==0,'metric after candidate forwards/AD returns')
                work=C.vector(selection_evaluations=1);metrics+=1
            require(eq(p['upper'],work),'primitive intent exact finite-axis work')
            g=groups[current];g['subreserved']=C.add(g['subreserved'],work)
            require(all(g['subreserved'][x]<=g['upper'][x] for x in C.KNOWN),'primitive work exceeds paid group')
            pending[token]=dict(group=current,upper=work,entry=False);seen.add(token);last_reserve=token
        elif k=='entry_boundary':
            token=p['token'];require(token==last_reserve and token in pending and not pending[token]['entry'],'one entry after reservation')
            require(eq(p,dict(root_slot=i,root_sha256=root,token=token,group=pending[token]['group'],
                observation='native API/operation host-call boundary; not completion',actual_completed=None)),'entry is intent, not completion')
            pending[token]['entry']=True;last_reserve=None
        elif k=='return_observed':
            token=p['token'];require(token in pending and pending[token]['entry'] and p['success'] is True,'no missing/pending/failed primitive return')
            prior=pending[token];e=p['native_evidence'];require(p['group']==prior['group'],'return paid group')
            if token.startswith('forward.'):
                idx=int(token.split('.')[1]);require(e['kind']=='entry_terminal' and idx==len(terminals),'ordered actual API returns')
                require(eq(e['entry'],row['actual_model_entries'][idx]) and eq(attempts[idx],dict(e['entry'],completed=False)),'durable terminal equals actual entries')
                work=work_forward(e['entry']['kind']);terminals.append(e['entry'])
            elif e.get('kind') in producer.OP_KINDS:
                require(e['completed'] is True and token==e['kind']+'.'+str(e.get('index')) and
                        eq(e,row['operations'][len(operations)]),'operation return matches actual stream')
                work=work_operation(e,True);operations.append(e)
            else:
                require(token.startswith('metric.') and eq(e,dict(kind='metric_return')),'metric return evidence')
                work=C.vector(selection_evaluations=1)
            require(eq(p['observed_completed'],work) and all(work[x]<=prior['upper'][x] for x in C.KNOWN),'actual work versus intent')
            require(p['observation']=='host callable returned; GPU completion requires external synchronize','host return scope')
            g=groups[prior['group']];g['returned']=C.add(g['returned'],work);del pending[token]
        elif k=='native_evidence':
            e=p['event'];require(e.get('kind')=='array_saved','no unknown/failure native event allowed')
            receipt=e['receipt'];require(eq(receipt,row['array_receipts'][len(arrays)]),'durable full array receipt')
            arrays.append(receipt);progress.append((len(terminals),len(operations)))
        elif k=='candidate_terminal':
            a,j=p['arm'],p['index'];require(type(j) is int and (a,j) in by_candidate and not pending,'terminal candidate with no unfinished work')
            require(current==f'{a}.candidate.{j}' and p['success'] is True and (a,j) not in candidates,'unique candidate terminal/group')
            require(p['native_event_sha256']==C.digest(by_candidate[a,j]),'candidate hash joins complete native witness/solve/Q event')
            require(groups[current]['returned']['selection_evaluations']==1,'candidate metric actually returned')
            candidates.append((a,j))
        else:raise ValueError('unexpected/failure/duplicate-terminal journal event: '+str(k))
    require(not pending and last_reserve is None and len(groups)==123,'all paid groups/primitive returns required')
    require(type(records[-1]['elapsed_ns']) is int and records[-1]['elapsed_ns']>=last_time,'terminal event clock')
    require(eq(terminals,row['actual_model_entries']) and len(terminals)==len(attempts)==664,'all actual dual module entries reconciled')
    require(eq(operations,row['operations']) and len(operations)==166,'all83 native request/terminal operation pairs')
    require(eq(arrays,row['array_receipts']) and eq(progress,native.expected_array_progress()),'all82 arrays persisted before next AD/solve boundary')
    require(candidates==[(a,j) for a in native.ARMS for j in range(contract.COUNTS[a])] and metrics==120,'all120 candidate/metric slots, no success-only selection')
    actual=C.add(*[g['returned'] for g in groups.values()]);shared=groups['base']['returned']
    standalone={a:C.add(shared,*[g['returned'] for key,g in groups.items() if key!='base' and g['arm']==a]) for a in native.ARMS}
    expected_candidates={a:[dict(index=j,status='completed',error=None) for j in range(contract.COUNTS[a])] for a in native.ARMS}
    expected_snapshot=dict(root_slot=i,root_sha256=root,candidates=expected_candidates,
        reservation_intent_standalone=reserved,observed_returned_lower_bound_standalone=standalone,
        observed_returned_lower_bound_unique=actual,pending_operations={},groups=groups,failed_returns=[],
        unknown_completion=False,actual_complete_totals=actual,
        physics_and_visual_endpoint='NOT_DISPATCHED; future cap room retained, not actual work',
        upstream_acquisition_cost=None,flops=None,backward_engine_api_calls=None,
        evidence_scope='native host entry/return and mathematical AD/solve counters, not synchronized kernel profiler',
        P2_admission=False,execution_authorized=False)
    require(eq(row['guard'],expected_snapshot),'independently rebuilt full guard snapshot/per-axis totals')
    # Separate direct native census, not just comparison with the guard snapshot.
    counts=native.validate_operations(operations,row['results'])
    require(actual['predict_calls']==actual['predict_samples']==actual['action_encoder_calls']==actual['action_encoder_samples']==332
        and actual['predict_context_tokens']==actual['action_encoder_context_tokens']==996
        and actual['selection_evaluations']==120 and counts['jacobian_calls']==41
        and actual['jacobian_192x20_calls']+actual['jacobian_192x10_calls']==counts['jacobian_calls']
        and actual['math_vjp_192x20_directions']+actual['math_vjp_192x10_directions']==counts['scalar_output_vjp_directions']
        and actual['svd_192x20_calls']+actual['svd_192x10_calls']==counts['svd_calls']
        and actual['bounded_solve_20_calls']+actual['bounded_solve_10_calls']==counts['bounded_solve_calls'],'direct native operation/axis census')
    return dict(root_slot=i,unique=actual,standalone=standalone,group_count=123,primitive_count=len(seen),
                module_entries=664,operation_pairs=83,metrics=120,arrays=82,candidate_count=120)


def terminal_report(report,expected_prepare,review_sha):
    contract.prior.graph.keys(report,('schema','status','wrapper_prepare_sha256','main_review_sha256',
        'root_denominator','arm_denominator','candidate_denominator','census','rows','error','secondary_errors',
        'model_unchanged','source_unchanged','model_before','model_after','new_physics_steps','P1_admission','P2_admission',
        'actual_dispatch_scope','physics_endpoint_cost','upstream_acquisition_cost'))
    require(report['schema']==producer.NAME+'.report' and report['status']=='known_root_completed'
        and report['error'] is None and report['secondary_errors']==[],'terminal successful report required; partial/failure never complete')
    require(report['wrapper_prepare_sha256']==expected_prepare and report['main_review_sha256']==review_sha,'external prepare/MAIN review binding')
    require(eq([report[k] for k in ('root_denominator','arm_denominator','candidate_denominator')],[4,8,480]),'fixed4/8/480 denominator')
    require(report['model_unchanged'] is True and report['source_unchanged'] is True and report['P1_admission'] is False
        and report['P2_admission'] is False and type(report['new_physics_steps']) is int and report['new_physics_steps']==0,'source/model/no-physics/non-admission')
    for key in ('model_before','model_after'):
        require(set(report[key])=={'parameters','buffers'},'full parameter/buffer hash declarations')
        for value in report[key].values():contract.prior.graph.digest(value)
    require(eq(report['model_before'],report['model_after']),'exact before/after model state declarations')
    require(report['physics_endpoint_cost'] is None and report['upstream_acquisition_cost'] is None
        and report['actual_dispatch_scope']=='native host calls; no FLOPs/backward-engine/kernel profiler','unmeasured costs stay unknown')
    require(len(report['rows'])==4 and len(report['census'])==8,'complete terminal root/arm census')


def consume(output,report_sha,prepared_path,prepare_sha,review_sha,*,confirmatory=False):
    if confirmatory:raise ValueError('confirmatory/physics prohibited; independent P1/freshness not admitted')
    started=time.perf_counter()
    for value in (report_sha,prepare_sha,review_sha):contract.prior.graph.digest(value)
    require(sha(producer.__file__)==GUARD_SHA,'frozen guard source bytes')
    output=Path(output).resolve();bindings={}
    def bound(path,expected=None):
        path=Path(path);data=raw_bytes(path);h=hashlib.sha256(data).hexdigest();value=contract.prior.graph.parse(data)
        require(expected is None or h==expected,'external evidence byte SHA: '+str(path))
        bindings[str(path.resolve())]=h;return value
    report=bound(output/'report.json',report_sha)
    terminal_report(report,prepare_sha,review_sha)  # No partial row reads before terminal admission.
    prepared_path=Path(prepared_path).resolve();v=bound(prepared_path,prepare_sha);producer.verify(v,prepare_sha)
    _,p,s=native.load_prepared(v['native_directory'],v['native_prepared_sha256'])
    bindings.update(v['bindings'])
    execution=bound(output/'execution.json')
    require(eq(execution,dict(wrapper_prepare_sha256=prepare_sha,main_review_sha256=review_sha,
        mode='known_root_engineering_NOT_confirmatory',P2_admission=False)),'execution envelope exact binding')
    compiled={r['root_sha256']:r['compiled'] for r in bound(p['context']['compilation'])['rows']}
    plans=[];audits=[];expected_files={'report.json','execution.json'}
    for i,ready in enumerate(p['context']['cases']):
        require(type(ready['root_index']) is int and ready['root_index']==i and ready['root_sha256']==native.ROOTS[i],'fixed protocol roots')
        receipt=report['rows'][i];path=output/f'root_{i}.json'
        require(eq(receipt,dict(root_slot=i,complete=True,file=f'root_{i}.json',sha256=sha(path))),'root row filename/order/byte seal')
        row=bound(path,receipt['sha256'])
        require(row['root_slot']==i and type(row['root_slot']) is int and row['root_sha256']==ready['root_sha256']
            and row['complete'] is True and eq(row['faults'],dict(primary=None,secondary=[]))
            and row['P2_admission'] is False and eq(row['new_physics_steps'],0),'complete error-free native root only')
        native.validate_actual(row['actual_model_entries']);native.validate_operations(row['operations'],row['results'])
        packet_audit=reconcile_packet_ledgers(row)
        saved=bound(output/f'root_{i}_bank.json');c=compiled[ready['root_sha256']];old=native.legacy()
        base=old.sra.spectral_bank(saved['base']['jacobian'],saved['base']['nominal'],c['goal_vectors'][0],c['scoring']['metric']['projection'])
        require(eq(saved['base'],base) and eq(saved['expanded'],native.candidate_bank(base,ready['root_sha256'])),'saved full root bank/goal/Q/80 regeneration')
        native.validate_array_artifacts(output,p,v['native_prepared_sha256'],ready,c,saved,row)
        for array in row['array_receipts']:
            ap=output/'arrays'/(array['name']+'.json');bound(ap,array['sha256']);expected_files.add('arrays/'+ap.name)
        jp=output/f'root_{i}_guard.jsonl'
        require(jp.is_file() and not jp.is_symlink() and jp.stat().st_size<=16*1024**2,'bounded regular durable journal')
        require(sha(jp)==row['journal_sha256'],'row/journal exact byte binding');bindings[str(jp)]=sha(jp)
        records=producer.read_journal(jp);audit=reconcile(records,row,ready,v['native_prepared_sha256'])
        audit['packet_ledgers']=packet_audit
        audit['arm_resource_comparison']=C.compare(*(audit['standalone'][arm] for arm in native.ARMS))
        require(audit['arm_resource_comparison']['predict_frontier_equal'] is True
            and audit['arm_resource_comparison']['full_resource_equal'] is False,
            'matched prediction frontier never certifies full resource/FLOPs equality')
        audits.append(audit)
        for arm,result in zip(native.ARMS,row['results'],strict=True):
            prov=native.provenance(p,s,ready,saved['expanded'],arm)
            best=native.validate_result(result,arm=arm,bank=saved['expanded'],ready=ready,prov=prov,compiled=c)
            census=report['census'][2*i+native.ARMS.index(arm)]
            require(eq(census,dict(root_slot=i,root_sha256=ready['root_sha256'],arm=arm,
                candidate_denominator=contract.COUNTS[arm],candidates=row['guard']['candidates'][arm])),'report full candidate census joins verified row')
            plans.append(dict(root_slot=i,root_sha256=ready['root_sha256'],arm=arm,commands=best['actions'],
                selection_digest=C.digest(best),native_receipt_sha256=best['witness']['receipt']['receipt_sha256'],
                candidate_id=best['candidate_id'],report_sha256=report_sha,physical_execution_authorized=False))
        expected_files.update((f'root_{i}.json',f'root_{i}_bank.json',f'root_{i}_guard.jsonl'))
    require(set(str(x.relative_to(output)) for x in output.rglob('*') if x.is_file())==expected_files,'exact actual artifact layout; no extra/foreign/partial evidence')
    require(not any(x.is_symlink() for x in output.rglob('*')),'no symlink evidence')
    total=C.add(*[a['unique'] for a in audits])
    require(len(plans)==8 and total['predict_calls']==total['action_encoder_calls']==1328
        and total['selection_evaluations']==480,'cohort actual1328/480 census')
    consumer_sources={str(path):sha(path) for path in (Path(__file__).resolve(),ROOT/'tests'/(NAME+'_tests.py'),ROOT/'innerdocs'/(NAME+'.md'))}
    for path,h in bindings.items():require(sha(path)==h,'input drift during saved consumption')
    return dict(schema=NAME+'.consume',status='terminal_saved_evidence_verified_NOT_physical_release',
        report_sha256=report_sha,prepare_sha256=prepare_sha,bindings=bindings,root_denominator=4,arm_denominator=8,
        candidate_denominator=480,root_audits=audits,unique_resources=total,selected_plans=plans,
        model_before=report['model_before'],model_after=report['model_after'],consumer_sources=consumer_sources,
        validation_wall_s=time.perf_counter()-started,new_neural_calls=0,new_physics_steps=0,
        known_root_engineering_only=True,independent_cohort=False,independent_neural_replay=False,
        full_resource_equal=False,hardware_cost_equivalence_certified=False,
        physical_execution_authorized=False,P1_admission=False,P2_admission=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input','report-sha','prepared','prepare-sha','review-sha','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--confirmatory',action='store_true');a=parser.parse_args(argv)
    if a.confirmatory:raise ValueError('confirmatory/physics prohibited')
    try:
        result=consume(a.input,a.report_sha,a.prepared,a.prepare_sha,a.review_sha);code=0
    except Exception as exc:
        result=dict(schema=NAME+'.consume',status='failed_closed',errors=[dict(type=type(exc).__name__,message=str(exc))],
            root_denominator=4,arm_denominator=8,candidate_denominator=480,selected_plans=[],
            declared_census=None,physical_execution_authorized=False,P1_admission=False,P2_admission=False)
        # Retain failed/pending declared census ONLY if the externally supplied report SHA matches.
        path=Path(a.input)/'report.json'
        if path.is_file() and sha(path)==a.report_sha:
            try:result['declared_census']=read(path).get('census')
            except Exception as diagnostic:result['errors'].append(dict(type=type(diagnostic).__name__,message=str(diagnostic)))
        code=1
    path=Path(a.output).absolute()
    require(path.name.startswith(NAME) and path.parent.resolve().is_relative_to(ROOT/'results'),'exclusive consumer output stem under results')
    native.write(path,result)
    print(producer.encode({k:v for k,v in result.items() if k in ('status','errors','report_sha256','P2_admission')}).decode())
    return code


if __name__=='__main__':raise SystemExit(main())
