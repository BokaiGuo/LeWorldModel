"""Normalize a trusted physical-v2 consume receipt, never invoke upstream consume."""
import argparse
import json
import math
import os
from pathlib import Path
from experiments.spectral_reachability import agent_p1_intersection_goal_adapter_20261029_v2 as upstream
from experiments.spectral_reachability import agent_p1_prospective_goal_semantics_20261029_v1 as semantics

NAME='agent_p1_intersection_goal_receipt_adapter_20261029_v1'
CP=(0,5,10,15)

def need(v,msg):
    if not v: raise ValueError(msg)

def normalize(spec,p,receipt,receipt_path,receipt_sha,prepare_path,prepare_sha,source_sha,report_path,report_sha):
    need(receipt['schema']==upstream.PHYSICAL+'.consume' and receipt['status'] in ('intersection6_native_recorded_not_admitted','partial_or_failed'),'physical v2 consume')
    need(receipt['fixed_denominator']==6 and receipt['known_roots']==2 and receipt['fresh_roots']==receipt['inherited']==0,'full denominator and known roots')
    need(all(receipt[k] is False for k in ('P1_admission','P2_admission','P2_mechanism','scientific_admission','freshness','equal_FLOPs','equal_total_compute','independent_confirmation','physical_release')),'science flags')
    need(receipt['physical_report_sha256']==report_sha and receipt['plans']==p['plans'] and receipt['planner_authority']==p['planner_authority'],'physical authority and exact plans')
    need(receipt['bindings'][str(prepare_path)]==prepare_sha and receipt['bindings'][str(upstream.SOURCE)]==source_sha and receipt['bindings'][str(report_path)]==report_sha,'source, prepare, report bindings')
    need(len(receipt['rows'])==len(spec['slots'])==6 and spec['checkpoints']==list(CP) and spec['rgb_pairs']==[],'roster')
    rows=[];detail=[]
    for i,(r,s,plan) in enumerate(zip(receipt['rows'],spec['slots'],p['plans'])):
        root,condition,arm=upstream.ORDER[i]
        need((r['index'],r['root_slot'],r['condition'],r['arm'])==(i,root,condition,arm) and r['plan']==plan and r['task_id']==plan['task_id'] and r['selection_digest']==plan['selection_digest'],'row identity and plan')
        need(s['slot_id']==f'{i}:{arm}' and s['root_id']==upstream.ROOTS[root] and s['goal_authority_sha256']==prepare_sha,'spec identity')
        need(r['status'] in ('not_started','engineering_failed','native_stopped','horizon_completed'),'row status')
        if r['initial_exact']:
            need(upstream.goal_bits(r['goal'])==upstream.goal_bits(s['goal7']) and r['root_sha256']==s['root_id'],'bound goal')
        checkpoints=r['checkpoints'];responses=r['responses']
        need(isinstance(checkpoints,list) and isinstance(responses,list) and [x['step'] for x in responses]==list(range(1,len(responses)+1)) and len(responses)<=r['own_returned']<=15,'ordered responses')
        steps=[x['step'] for x in checkpoints];need(steps==[x for x in CP if x in steps] and r['missing_checkpoints']==[x for x in CP if x not in steps],'checkpoint presence and order')
        need(all(x['state']==x['snapshot']['state'] for x in checkpoints),'checkpoint snapshot identity')
        need(not checkpoints or r['initial_exact'] is True,'checkpoints require goal bind')
        points=[]; labels=[]
        for cp in CP:
            item=next((x for x in checkpoints if x['step']==cp),None)
            response=next((x for x in responses if x['step']==cp),None)
            if item is None or (cp and response is None):
                why=('checkpoint_missing:'+r['status']) if item is None else 'checkpoint_observed_but_return_label_missing'
                points.append(dict(checkpoint=cp,state7=None,native_success=None,native_distance=None,missing_reason=why))
                labels.append(dict(checkpoint=cp,source='missing',observed_without_label=item is not None));continue
            state=item['state'];semantics.vec(state)
            if cp==0:
                need(state==r['root_state'],'cp0 root')
                computed=semantics.metrics(s['goal7'],state);success=computed['native_success'];distance=computed['native7_distance'];origin='derived_from_bound_cp0_not_native_return'
            else:
                need(response['state']==state and type(response['native_success']) is bool and type(response['terminated']) is bool and response['native_success'] is response['terminated'] and type(response['reward']) in (int,float) and math.isfinite(response['reward']) and response['reward']<=0,'native return/checkpoint identity')
                success=response['native_success'];distance=-response['reward'];origin='same_step_returned_native_reward_and_success'
            points.append(dict(checkpoint=cp,state7=state,native_success=success,native_distance=distance,missing_reason=None))
            labels.append(dict(checkpoint=cp,source=origin,observed_without_label=False))
        rows.append(dict(slot_id=s['slot_id'],root_id=s['root_id'],goal7=s['goal7'],goal_authority_sha256=prepare_sha,goal_rgb=None,points=points))
        detail.append(dict(slot_id=s['slot_id'],status=r['status'],first_error=r['first_error'],secondary_errors=r['secondary_errors'],own_returned=r['own_returned'],native_checkpoint_provenance=labels))
    normalized=dict(schema=semantics.NAME+'.receipt',authority=dict(path=str(receipt_path),sha256=receipt_sha),rows=rows)
    return dict(schema=NAME+'.normalized',receipt=normalized,details=detail,bindings=dict(prepare=dict(path=str(prepare_path),sha256=prepare_sha),physical_receipt=dict(path=str(receipt_path),sha256=receipt_sha),physical_report=dict(path=str(report_path),sha256=report_sha),source=dict(path=str(upstream.SOURCE),sha256=source_sha)),denominator=dict(slots=6,checkpoints=24),evidence='inherited external strict receipt; no upstream fullstrict rerun',P1_admission=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['normalize','verify']);
    for k in ('spec','spec-sha','prepared','prepare-sha','source-sha','receipt','receipt-sha','report','report-sha','output'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();prepared=Path(a.prepared);receipt_path=Path(a.receipt);report=Path(a.report);spec_path=Path(a.spec)
    spec=json.loads(upstream.read(spec_path,a.spec_sha));p=json.loads(upstream.read(prepared,a.prepare_sha));receipt=json.loads(upstream.read(receipt_path,a.receipt_sha))
    upstream.read(upstream.SOURCE,a.source_sha);upstream.read(report,a.report_sha);upstream.read(upstream.RECEIPT,upstream.RECEIPT_SHA)
    need(prepared.name.startswith(upstream.PHYSICAL+'_') and spec==upstream.project(p,prepared,a.prepare_sha,upstream.SOURCE,a.source_sha),'spec matches actual physical prepare')
    result=normalize(spec,p,receipt,receipt_path,a.receipt_sha,prepared,a.prepare_sha,a.source_sha,report,a.report_sha)
    data=upstream.encode(result);output=Path(a.output);need(output.is_absolute() and output.name.startswith(NAME+'_'),'own output')
    if a.command=='normalize':
        with output.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    else:need(output.read_bytes()==data,'saved output mismatch')
    print(a.command,upstream.digest(data))

if __name__=='__main__':main()
