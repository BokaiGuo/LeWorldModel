"""Read-only arithmetic audit of MAIN's four sampling physical branches."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.spectral_reachability import p1_sampling_physical_20261027_v1 as consumer
from experiments.spectral_reachability import p1_witness_physical_actual_audit_20261027_v1 as independent

TAG='p1_sampling_physical_saved_audit_20261027_v1'
PROJECT=Path(__file__).resolve().parents[2]
RUN=PROJECT/'results/p1_sampling_physical_20261027_v1_main01'
PREPARED=PROJECT/'results/p1_sampling_physical_20261027_v1_prepared01'
REPORT_SHA='c3eef328bee57ce5d42b25fcf5a9935af2d2282c9de3f8caf46f2b35b53a2ef6'
PREPARE_SHA='450e620dbe3bf1ec1957c8d2d06fda9dfc05c7cc6bd50453939dc07a2133033c'
CAP=200_000
FLOOR=200*1024**2
require=consumer.require


def keyed(rows):
    result={}
    for row in rows:
        key=(row['root_sha256'],row['arm'])
        require(key not in result,'duplicate root/arm')
        result[key]=row
    return result


def delta_metrics(goal,state):
    metrics=independent.metrics(goal,state)
    delta=np.asarray(state,dtype=np.float64)-np.asarray(goal,dtype=np.float64)
    delta[4]=(delta[4]+np.pi)%(2*np.pi)-np.pi
    metrics.update(wrapped_state=float(np.linalg.norm(delta)),
        object_pose_cost=metrics['object_position']+20*metrics['object_angle'])
    return metrics,delta


def audit():
    require(consumer.sha(RUN/'report.json')==REPORT_SHA,'external actual report SHA')
    p=consumer.verify(PREPARED,PREPARE_SHA)  # source/sealed all60 selection reconstruction only
    report=consumer.read(RUN/'report.json')
    require(report['prepare_sha256']==PREPARE_SHA and report['status']=='physical_calibration_recorded'
        and not report['errors'] and report['contract']==consumer.contract() and report['neural_calls']==0,
        'completed fixed physical protocol')
    expected={k:dict(attempted=n,completed=n) for k,n in dict(reset=8,step=40,render=44).items()}
    require(report['counts']==expected,'complete attempted/returned budgets')
    require(consumer.sha(RUN/'journal.jsonl')==report['journal_sha256'],'physical journal SHA')
    events=consumer.source.read_journal(RUN/'journal.jsonl')  # verifies full hashchain
    counts=Counter(e['kind'] for e in events)
    for kind,n in dict(branch_started=4,branch_finished=4,branch_closed=4,reset_attempted=8,
        reset_returned=8,step_attempted=40,step_returned=40,step_observation=40,
        fresh_rgb_observed=44,rgb_guard_observed=44,checkpoint_saved=12,
        render_attempted=44,render_returned=44).items():
        require(counts[kind]==n,'durable census '+kind)
    require(events[-1]['kind']=='run_finished' and events[-1]['counts']==expected
        and events[-1]['status']==report['status'] and not events[-1]['errors'],'final chain receipt')
    plans=keyed(p['inputs']['plans']);rows=keyed(report['rows'])
    require(len(plans)==4 and plans.keys()==rows.keys(),'full stable four-branch census')
    oldpath=PROJECT/'results/p1_witness_physical_20261027_v1_main01/report.json'
    oldsha='48df27e54eca33159c44a834ce0be1a13723553dc9d64dde996b7b4ca9eaf747'
    require(consumer.sha(oldpath)==oldsha,'original comparator physical bytes')
    old=keyed(consumer.read(oldpath)['rows']);results=[];negative=disagreements=0;images=[]
    for key,plan in sorted(plans.items()):
        row=rows[key];sid=plan['selection_sha256']
        require(row['selection_sha256']==sid and row['status']=='completed' and row['full_horizon']
            and row['initial_exact'] and all(row['initial_checks'].values()) and row['first_error'] is None
            and not row['secondary_errors'] and row['attempted_steps']==row['returned_steps']==10,
            'full original selected branch')
        branch=[e for e in events if e.get('selection_sha256')==sid]
        of=lambda kind:[e for e in branch if e['kind']==kind]
        require(len(of('branch_started'))==len(of('branch_finished'))==1
            and of('branch_started')[0]['plan']==plan and of('branch_finished')[0]['receipt']==row,
            'frozen action plan/actual receipt')
        closed=[e for e in events if e['kind']=='branch_closed' and e['receipt']['selection_sha256']==sid]
        require(len(closed)==1 and closed[0]['receipt']==row,'close evidence preserved')
        packet=consumer.source.load_raw_source(plan['raw_source_file'],p['inputs']['source_plan'])
        goal=packet['goal'];state=packet['state']
        require(packet['metadata']==plan['source_metadata'],'original source provenance')
        reset=of('reset_observation');request=of('original_placement_request')
        require(len(reset)==2 and len(request)==1 and reset[0]['seed']==plan['source_metadata']['selected_attempt']['random_reset_seed']
            and reset[0]['options'] is None and reset[1]['seed']==plan['source_metadata']['selected_attempt']['placement_reset_seed']
            and reset[1]['options']==dict(state=request[0]['state'],goal_state=request[0]['goal'])
            and reset[1]['observation']['state']==state.tolist() and reset[1]['info']['goal_state']==goal.tolist(),
            'original recipe resets; exact saved poststate/goal')
        require(consumer.source.root_hash(state,goal)==key[0],'actual root identity')
        trace=row['trace'];observed=of('fresh_rgb_observed');guard=of('rgb_guard_observed')
        require(len(trace)==10 and len(observed)==len(guard)==11
            and [e['receipt'] for e in of('step_observation')]==trace,'all actual returns')
        for i in range(11):
            g=guard[i]['receipt'];frame=observed[i]['receipt']
            require(guard[i]['step']==frame['step']==i and g['root_id']==key[0]
                and g['primitive_commands_authorized']==i and not g['abstained']
                and [o['primitive_step'] for o in g['observations']]==list(range(i+1))
                and all(o['canonical_agent_pixels']>0 for o in g['observations']), 'fresh RGB original guard')
            current=state.tolist() if i==0 else trace[i-1]['actual_state']
            require(frame['state']==current,'render/actual chronology')
            if i<10:
                require(guard[i]['index']<of('step_attempted')[i]['index']<of('step_returned')[i]['index']<observed[i+1]['index'],
                    'fresh-before-dispatch order')
        require(row['guard_receipt']==guard[-1]['receipt'],'last guard binding')
        for i,t in enumerate(trace):
            require(t['step']==i+1 and t['attempted'] and t['returned']
                and t['command']==plan['commands'][i]==of('step_attempted')[i]['receipt']['command']
                and t['observation']['state']==t['actual_state']
                and t['official_info']['goal_state']==goal.tolist(), 'unchanged raw/actual state/goal')
            m=independent.check_metric(goal,t['actual_state'],t['pose_diagnostics'],t['native_official_evaluation'])
            independent.close(t['reward'],-m['native_state'])
            negative+=m['negative_native_angle'];disagreements+=m['native_success']!=m['wrapped_success']
            require(not t['terminated'] and not t['truncated'],'no early endpoint substitution')
        require(row['final_state']==trace[-1]['actual_state'],'actual final state')
        independent.check_metric(goal,row['final_state'],row['pose_diagnostics'],row['official_evaluation'])
        m,delta=delta_metrics(goal,row['final_state']);independent.close(row['wrapped_state_delta'],delta)
        require([c['step'] for c in row['checkpoints']]==[0,5,10]
            and [e['receipt'] for e in of('checkpoint_saved')]==row['checkpoints'],'all checkpoint slots')
        for c in row['checkpoints']:
            pixels=independent.png_array(c,RUN);step=c['step'];g=guard[step]['receipt']['observations'][-1]
            require(c['rgb_sha256']==observed[step]['receipt']['rgb_sha256'] and c['state']==observed[step]['receipt']['state']
                and int(np.count_nonzero(np.all(pixels==[65,105,225],axis=-1)))==g['canonical_agent_pixels'], 'PNG/frame/guard mapping')
            if step==0:require(np.array_equal(pixels,packet['rgb_history'][0]),'initial RGB exact')
            images.append(dict(root_sha256=key[0],arm=key[1],step=step,path=c['path'],file_sha256=c['file_sha256'],rgb_sha256=c['rgb_sha256']))
        differences={}
        for arm in ('zero','full_H2_GN','full_H2_gradient','two_stage'):
            baseline,_=delta_metrics(goal,old[key[0],arm]['final_state'])
            differences[arm]={k:m[k]-baseline[k] for k in ('object_position','object_angle','object_pose_cost','native_state','wrapped_state','all_xy')}
        results.append(dict(root_sha256=key[0],candidate_id=plan['candidate_id'],arm=key[1],selection_sha256=sid,
            predicted_Q=plan['prediction']['cost'],actual_Q=None,metrics=m,delta_vs_original_physical=differences))
    require(len(list(RUN.glob('*.png')))==12,'exact twelve original assets')
    require(consumer.sha(RUN/'report.json')==REPORT_SHA and consumer.sha(oldpath)==oldsha,'immutable actual reports')
    return dict(schema=TAG+'.report',status='saved_physical_binding_and_metric_checks_completed',branches=results,
        counts=expected,checked_steps=40,checked_rgb_receipts=44,checked_png=12,
        negative_native_angle_steps=negative,native_wrapped_success_disagreements=disagreements,
        official_successes=sum(b['metrics']['native_success'] for b in results),
        input_bindings={str(RUN/'report.json'):REPORT_SHA,str(RUN/'journal.jsonl'):report['journal_sha256'],
            str(PREPARED/'prepare.json'):PREPARE_SHA,str(oldpath):oldsha},images=images,
        actual_endpoint_encoding='pending_MAIN_CUDA; no latent-Q inferred from state or prediction',
        inherited_sampling_cost=report['sampling_cost'],inherited_earlier_cost=report['inherited_cost'],
        internal_physics=report['reset_internal_physics_steps'],public_internal_physics=report['public_step_internal_physics_steps'],
        new_neural_calls=0,new_physical_steps=0,P1_admission=False,equal_total_compute=False,
        scope='Same two calibration roots; differences are descriptive, not independent confirmation or mechanism proof.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    output=parser.parse_args().output.resolve()
    require(TAG in output.name and not output.exists(),'new exclusive output')
    result=audit();result['audit_source_sha256']=consumer.sha(__file__)
    raw=(json.dumps(result,sort_keys=True,allow_nan=False,indent=2)+'\n').encode()
    require(len(raw)<=CAP and shutil.disk_usage(output.parent).free>=FLOOR+len(raw),'200KB cap and200MiB floor')
    with output.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(output),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))))


if __name__=='__main__':main()
