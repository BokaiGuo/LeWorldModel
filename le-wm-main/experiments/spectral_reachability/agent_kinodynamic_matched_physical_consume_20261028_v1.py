"""Saved-only consumer for fixed TWO matched-domain beam physical branches.
No physics/model calls; external report SHA and frozen physical closure required.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_kinodynamic_matched_physical_20261028_v1 as physical

NAME='agent_kinodynamic_matched_physical_consume_20261028_v1'
ARM,ROOT,kernel,source=physical.ARM,physical.ROOT,physical.kernel,physical.source
require,sha,read=physical.require,physical.sha,physical.read
absolute,hash_literal,merge=physical.absolute,physical.hash_literal,physical.merge
CAP,SOURCE_SHA,REPORT_SHA=physical.CAP,physical.SOURCE_SHA,physical.REPORT_SHA
contract,verify=physical.contract,physical.verify
PHYSICAL_SOURCE_SHA='631cb4e7894b2b4da0b96cca19af627a950133089e4ff571b4a1a6f1b9403a80'
PHYSICAL_TEST_SHA='41c5b1db23123ff751f3847ba9efd2647843984403574e2e9a2e6895db3ebcbf'
PHYSICAL_DOC_SHA='7b508cf7321cef1606f6a99bd3081565604a37317db76c7ceb80da92aa64588f'


def vector(value):
    a=np.asarray(value,dtype=np.float64)
    require(a.shape==(7,) and np.isfinite(a).all(),'actual native full7 state/goal; no truncation')
    return a


def metric(goal,state):
    # Installed native semantics deliberately do NOT modulo the angle delta.
    state=vector(state);d=state-vector(goal);angle=float(abs(d[4]));native=min(angle,2*np.pi-angle)
    wrapped=float(abs(np.arctan2(np.sin(d[4]),np.cos(d[4]))));xy=float(np.linalg.norm(d[:4]))
    return dict(native_success=bool(xy<20 and native<np.pi/9),native_distance=float(np.linalg.norm(d)),
        native_angle_term=float(native),negative_native_angle=bool(native<0),
        wrapped_allxy_success=bool(xy<20 and wrapped<np.pi/9),wrapped_object_angle_error=wrapped,
        object_position_error=float(np.linalg.norm(d[2:4])),object_pose=state[2:5].tolist(),object_success=None)


def physical_seals():
    return {str(Path(physical.__file__).resolve()):PHYSICAL_SOURCE_SHA,
        str(ROOT/'tests'/('test_'+physical.NAME+'.py')):PHYSICAL_TEST_SHA,
        str(ROOT/'innerdocs'/(physical.NAME+'.md')):PHYSICAL_DOC_SHA}


def assemble(report,events,p):
    """Pure saved ABI, fixed TWO identities; no inference or simulator calls."""
    require(report['schema']==physical.NAME+'.report' and report['status'] in ('matched_domain_beam_physical_recorded_not_admitted','partial_or_failed') and
        report['contract']==p['contract']==contract(p['contract']['wall_seconds']) and
        report['sourcepack_sha256']==p['sourcepack_sha256']==SOURCE_SHA and
        report['source_prepare_id']==p['source_prepare_id'] and report['fixed_branch_denominator']==2 and
        report['planner_report_sha256']==REPORT_SHA,'own fixed two-branch authority')
    for k in ('P1_admission','P2_admission','scientific_admission','independent_confirmation','equal_total_compute','equal_FLOPs','training'):
        require(report[k] is False,'no scientific/compute upgrade')
    require(report['new_neural_calls']==report['new_independent_samples']==0,'known roots physical only')
    require(events and events[0]['kind']=='run_started' and events[0]['prepare']==p and
        events[-1]['kind']=='run_finished' and events[-1]['status']==report['status'] and
        events[-1]['counts']==report['counts'] and events[-1]['errors']==report['errors'] and
        events[-1]['event_sha256']==report['journal_chain_tail'],'durable prepare and terminal binding')
    order=[(0,ARM),(1,ARM)]
    require([(r['root_slot'],r['arm']) for r in report['rows']]==order and
        [(r['root_slot'],r['arm']) for r in p['plans']]==order,'all two rows, never success-only subset')
    counts={k:dict(attempted=0,completed=0) for k in p['contract']['caps']};intents={};terminals={};stack=[]
    for e in events:
        kind=e['kind']
        if kind not in ('operation_dispatch_intent','operation_returned','operation_failed'):continue
        r=e['receipt'];i=r['operation_index']
        require(type(i) is int and r['resource'] in counts,'literal operation index/resource')
        if kind=='operation_dispatch_intent':
            require(i==len(intents) and r['status']=='pending','sequential operation intent')
            intents[i]=r;stack.append(i)
        else:
            require(i in intents and i not in terminals and stack and stack.pop()==i and
                r['status']==('returned' if kind=='operation_returned' else 'failed'),'nested returned/failed actual receipt')
            original={k:v for k,v in r.items() if k not in ('error_type','message')}
            require(dict(original,status='pending')==intents[i],'terminal original identity')
            terminals[i]=r
    require(not stack and len(terminals)==len(intents) and
        report['actual_operation_receipts']==[terminals[i] for i in range(len(terminals))],'all actual operations durable; no guessed partial receipts')
    for r in report['actual_operation_receipts']:
        c=counts[r['resource']];c['attempted']+=1;c['completed']+=r['status']=='returned'
        i=r.get('branch');require(type(i) is int and i in (0,1),'operation fixed branch')
        plan=p['plans'][i]
        require(r['root_slot']==plan['root_slot'] and r['arm']==ARM and r['selection_digest']==plan['selection_digest'],'operation plan identity')
    require(counts==report['counts'] and all(v['attempted']<=p['contract']['caps'][k] for k,v in counts.items()),'observed counters/caps')
    closed=[e['receipt'] for e in events if e['kind']=='branch_closed']
    require(closed==[r for r in report['rows'] if r['status']!='not_started'],'durable rows, no lost denominator')
    checkpoints=[]
    for index,(row,plan) in enumerate(zip(report['rows'],p['plans'],strict=True)):
        require(type(row['root_slot']) is int and row['branch_index']==index and row['selected_plan']==plan and
            row['selection_digest']==plan['selection_digest']==source.digest({k:v for k,v in plan.items() if k!='selection_digest'}),'own full selected plan seal')
        steps=[r for r in report['actual_operation_receipts'] if r['resource']=='public_step' and r['branch']==index]
        own=[r for r in steps if r['phase']=='plan'];warm=[r for r in steps if r['phase']=='warmup']
        require(row['own_attempted']==len(own) and row['own_returned']==sum(x['status']=='returned' for x in own) and
            row['warmup_returned']==sum(x['status']=='returned' for x in warm) and row['full_horizon']==(row['own_returned']==10),'actual row counters')
        for ops,commands in ((warm,p['contract']['warmup_raw']),(own,plan['commands'])):
            require([r['step'] for r in ops]==list(range(1,len(ops)+1)) and len(ops)<=10 and
                all(r['raw_action']==commands[j] for j,r in enumerate(ops)),'literal contiguous warmup/own raw, no padding')
        observed=[e['receipt'] for e in events if e['kind']=='fresh_observation' and e.get('branch')==index]
        require(row['PNG']==len(observed),'all observed render denominator')
        cp=[c['step'] for c in row['checkpoints']]
        require(cp==sorted(set(cp)) and all(s in (0,5,10) for s in cp),'only real unique0/5/10 checkpoints')
        transitions=[e for e in events if e['kind']=='returned_transition' and e.get('branch')==index and e['phase']=='plan']
        for c in row['checkpoints']:
            require(c in observed and c['phase']=='plan' and c['step']<=row['own_returned'] and
                np.array_equal(vector(c['state']),vector(c['observation_state'])),'actual7 checkpoint observation')
            guards=[e for e in events if e['kind']=='guard_receipt' and e.get('branch')==index and e['phase']=='plan' and e['step']==c['step']]
            require(len(guards)==1 and guards[0]['accepted'] is True,'accepted actual checkpoint guard')
            if c['step']==0:require(c['state']==row['root_state'],'checkpoint root')
            else:
                ts=[e for e in transitions if e['step']==c['step']]
                require(len(ts)==1 and ts[0]['state']==c['state'],'checkpoint own command return')
            checkpoints.append(dict(branch_index=index,root_slot=row['root_slot'],arm=ARM,selection_digest=row['selection_digest'],
                selected_plan=deepcopy(plan),checkpoint=deepcopy(c)))
        if row['initial_exact']:
            require(source.root_hash(row['root_state'],row['goal'])==plan['root_sha256'] and
                row['initial_metrics']==metric(row['goal'],row['root_state']),'root and original7 native metric')
            attached=[e for e in events if e['kind']=='goal_attached' and e.get('branch')==index]
            require(len(attached)==1 and attached[0]['goal']==row['goal'],'exact offline goal')
            require(all(e['returned_goal_state']==row['goal'] and e['returned_goal_rgb_sha256']==attached[0]['goal_rgb_sha256']
                for e in transitions),'goal preserved in native returns')
        require([e['step'] for e in transitions]==list(range(1,row['own_returned']+1)),
            'every own returned public step has one saved state transition')
        placement=[e for e in events if e['kind']=='placement_request' and e.get('branch')==index]
        warm_transitions=[e for e in events if e['kind']=='returned_transition' and e.get('branch')==index and e['phase']=='warmup']
        require([e['step'] for e in warm_transitions]==list(range(1,row['warmup_returned']+1)),'all warmup native state transitions')
        for transition in transitions+warm_transitions:
            if transition['phase']=='plan':
                goal=row['goal'];require(transition['command']==plan['commands'][transition['step']-1],'own returned raw command')
            else:
                require(len(placement)==1,'warmup original native goal placement')
                goal=placement[0]['original_goal']
            native=metric(goal,transition['state'])
            require(type(transition['terminated']) is bool and type(transition['truncated']) is bool and
                transition['terminated']==native['native_success'] and
                type(transition['reward']) in (int,float) and np.isfinite(transition['reward']) and
                transition['reward']==-native['native_distance'],'unchanged native7 reward/success, not object-only success')
        if row['final_state'] is not None:
            require(transitions and row['final_state']==transitions[-1]['state'] and
                row['final_metrics']==metric(row['goal'],row['final_state']) and
                row['native_success']==row['final_metrics']['native_success'] and
                row['terminated']==transitions[-1]['terminated'] and row['truncated']==transitions[-1]['truncated'],
                'final full7 state/metrics/terminal equals last real returned transition')
        if row['status']=='horizon_completed':require(row['full_horizon'] and not row['terminated'] and not row['truncated'],'full10 nonterminal status')
        if row['status']=='native_success':require(row['native_success'] is True and row['terminated'] is True and row['truncated'] is False,'actual native success status')
    require(report['full_horizons']==sum(r['full_horizon'] for r in report['rows']) and
        report['native_successes']==sum(r['status']=='native_success' for r in report['rows']),'full/native census')
    return checkpoints


def consume(output,expected):
    for f,h in physical_seals().items():require(sha(f)==h,'frozen physical v1 source/tests/doc')
    output=Path(absolute(str(output)));hash_literal(expected);files=list(output.iterdir())
    require(all(f.is_file() and not f.is_symlink() for f in files) and sum(f.stat().st_size for f in files)<=CAP,'bounded regular output')
    require(sha(output/'report.json')==expected,'external actual physical report SHA')
    report=read(output/'report.json',CAP);p=verify(report['prepared'],report['prepare_sha256'])
    require(read(output/'execution_binding.json')==dict(prepared=report['prepared'],prepare_sha256=report['prepare_sha256']),'execution binding')
    journal=output/'journal.jsonl';require(sha(journal)==report['journal_sha256'] and journal.stat().st_size<=CAP,'actual journal SHA')
    data=journal.read_bytes();require(len(data)<=CAP and data.endswith(b'\n'),'complete bounded journal')
    import json
    events=[json.loads(line) for line in data.splitlines()];previous=None
    for i,e in enumerate(events):
        require(e['index']==i and e['previous_sha256']==previous and
            e['event_sha256']==source.digest({k:v for k,v in e.items() if k!='event_sha256'}),'event hashchain')
        previous=e['event_sha256']
    require(previous==report['journal_chain_tail'],'journal final chain')
    checkpoints=assemble(report,events,p);names={'report.json','execution_binding.json','journal.jsonl'}
    from PIL import Image
    for event in events:
        if event['kind']=='fresh_observation':
            c=event['receipt'];name=c['png'];require(Path(name).name==name and name not in ('','.','..'),'flat actual image')
            require(sha(output/name)==c['file_sha256'],'all actual PNG file SHA')
            with Image.open(output/name) as image:
                require(image.size==(224,224) and image.mode=='RGB','actual RGB224 PNG')
                frame=np.asarray(image)
            require(frame.dtype==np.uint8 and kernel.moving.image_sha(frame)==c['pixel_sha256'],'PNG pixel SHA')
            names.add(name)
        elif event['kind']=='invalid_render':
            c=event['receipt'];name=c.get('raw_render')
            if name:
                require(Path(name).name==name and name not in ('','.','..') and sha(output/name)==c['file_sha256'],'invalid render actual saved bytes')
                names.add(name)
    require({f.name for f in files}==names,'exact all artifact census, no foreign files')
    bindings={};merge(bindings,p['bindings']);merge(bindings,{report['prepared']:report['prepare_sha256']})
    merge(bindings,{str(f):sha(f) for f in files})
    require(bindings[str(output/'report.json')]==expected,'final report SHA')
    consumer_sources={str(f.resolve()):sha(f) for f in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'))}
    merge(bindings,consumer_sources)
    for f,h in bindings.items():require(sha(f)==h,'final physical/source/task closure')
    for c in checkpoints:c['png_path']=str(output/c['checkpoint']['png'])
    return dict(schema=NAME+'.consume',status=report['status'],rows=deepcopy(report['rows']),checkpoints=checkpoints,
        counts=deepcopy(report['counts']),report=report,prepared=p,bindings_sha256=bindings,physical_report_sha256=expected,
        fixed_branch_denominator=2,checkpoint_complete=all([c['step'] for c in r['checkpoints']]==[0,5,10] for r in report['rows']),
        consumer_sources=consumer_sources,scientific_admission=False,P1_admission=False,independent_confirmation=False,equal_total_compute=False,new_neural_calls=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('output','report-sha'):parser.add_argument('--'+k,required=True)
    a=parser.parse_args();r=consume(a.output,a.report_sha)
    print(source.canonical({k:v for k,v in r.items() if k not in ('rows','report','prepared','bindings_sha256','checkpoints')}),flush=True)


if __name__=='__main__':main()
