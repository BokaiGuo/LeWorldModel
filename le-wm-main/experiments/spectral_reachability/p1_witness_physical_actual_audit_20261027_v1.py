"""Saved-only independent arithmetic/asset audit; no simulator/model execution."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.spectral_reachability import p1_witness_physical_20261027_v1 as physical

NAME='p1_witness_physical_actual_audit_20261027_v1'
require=physical.require


def metrics(goal,state):
    """Independent modulo arithmetic, not producer.pose_diagnostics."""
    goal=np.asarray(goal,dtype=np.float64);state=np.asarray(state,dtype=np.float64)
    require(goal.shape==state.shape==(7,) and np.isfinite(goal).all() and np.isfinite(state).all(),'finite state7')
    delta=state-goal;raw=abs(float(delta[4]));native=min(raw,2*np.pi-raw)
    wrapped=abs(float((delta[4]+np.pi)%(2*np.pi)-np.pi))
    allxy=float(np.linalg.norm(delta[:4]))
    return dict(object_position=float(np.linalg.norm(delta[2:4])),object_angle=wrapped,
                agent_position=float(np.linalg.norm(delta[:2])),all_xy=allxy,
                native_state=float(np.linalg.norm(delta)),native_angle=native,
                native_success=bool(allxy<20 and native<np.pi/9),
                wrapped_success=bool(allxy<20 and wrapped<np.pi/9),negative_native_angle=bool(native<0))


def close(a,b):
    require(np.allclose(a,b,rtol=0,atol=2e-12),'saved numeric metric mismatch')


def check_metric(goal,state,diagnostic,official):
    m=metrics(goal,state);n=diagnostic['native_formula'];w=diagnostic['independent']
    for a,b in ((m['object_position'],w['object_position_error']),(m['object_angle'],w['wrapped_angle_error']),
                (m['agent_position'],w['agent_position_error']),(m['all_xy'],w['all_xy_position_error']),
                (m['native_state'],n['state_distance']),(m['native_angle'],n['angle_term']),
                (m['native_state'],official['state_distance'])):close(a,b)
    require(m['native_success'] is n['success'] is official['success'] and
            m['wrapped_success'] is w['same_allxy_threshold_wrapped_success'] and
            m['negative_native_angle'] is n['negative_angle_term'] and
            diagnostic['native_vs_wrapped_success_disagreement'] is (m['native_success']!=m['wrapped_success']), 'saved metric flags mismatch')
    return m


def png_array(record,folder):
    path=Path(record['path']).resolve()
    require(path.parent==folder.resolve() and path.is_file() and not path.is_symlink(),'local checkpoint PNG')
    require(physical.sha(path)==record['file_sha256'],'PNG file SHA')
    with Image.open(path) as image:
        require(image.format=='PNG' and image.mode=='RGB' and image.size==(224,224),'exact PNG224 RGB')
        value=np.asarray(image).copy()
    require(value.dtype==np.uint8 and physical.core.sha(value.tobytes())==record['rgb_sha256'],'PNG pixel SHA')
    return value


def audit(prepared,prepare_sha,report,report_sha):
    report=Path(report).resolve();folder=report.parent
    require(physical.sha(report)==report_sha,'external actual physical report SHA')
    p=physical.verify(prepared,prepare_sha);r=physical.read(report)
    require(r['prepare_sha256']==prepare_sha and r['contract']==physical.contract() and
            r['status']=='physical_calibration_recorded' and not r['errors'],'complete actual physical receipt')
    require(r['counts']=={k:dict(attempted=n,completed=n) for k,n in dict(reset=16,step=80,render=88).items()},'exact physical census')
    events=physical.source.read_journal(folder/'journal.jsonl')
    require(physical.sha(folder/'journal.jsonl')==r['journal_sha256'] and events[0]['kind']=='run_started' and
            events[-1]['kind']=='run_finished' and events[-1]['counts']==r['counts'] and
            events[-1]['status']==r['status'] and not events[-1]['errors'],'complete bound physical chain')
    kinds=Counter(e['kind'] for e in events)
    for k,n in dict(branch_started=8,branch_finished=8,branch_closed=8,reset_attempted=16,reset_returned=16,
                    step_attempted=80,step_returned=80,step_observation=80,fresh_rgb_observed=88,
                    rgb_guard_observed=88,checkpoint_saved=24,render_attempted=88,render_returned=88).items():
        require(kinds[k]==n,'journal census '+k)
    plans=p['inputs']['plans'];require(len(r['rows'])==len(plans)==8,'full denominator')
    index={(x['root_sha256'],x['arm']):x for x in r['rows']}
    require(len(index)==8 and set(index)=={(x['root_sha256'],x['arm']) for x in plans},'stable root/arm census')
    branches=[];acquisition=[];negative=disagreement=termination=0
    direct_directory=Path(p['inputs']['arguments']['direct_report']).parent
    for plan in plans:
        row=index[plan['root_sha256'],plan['arm']];sid=plan['selection_sha256']
        ev=[e for e in events if e.get('selection_sha256')==sid]
        def of(kind):
            if kind=='branch_closed':
                return [e for e in events if e['kind']==kind and e['receipt']['selection_sha256']==sid]
            return [e for e in ev if e['kind']==kind]
        require(len(of('branch_started'))==len(of('branch_finished'))==len(of('branch_closed'))==1 and
                of('branch_started')[0]['plan']==plan and of('branch_finished')[0]['receipt']==row and
                of('branch_closed')[0]['receipt']==row,'bound branch receipts')
        require(row['status']=='completed' and row['full_horizon'] and row['endpoint_rgb_fresh'] and
                row['endpoint_scope']=='full_H2' and row['first_error'] is None and not row['secondary_errors'] and
                row['attempted_steps']==row['returned_steps']==10 and row['initial_exact'] and
                all(row['initial_checks'].values()),'branch completion/initial identity')
        packet=physical.source.load_raw_source(plan['raw_source_file'],p['inputs']['source_plan'])
        goal=packet['goal'];require(packet['metadata']==plan['source_metadata'],'original source identity')
        resets=of('reset_observation');require(len(resets)==2,'two resets/branch')
        require(resets[0]['seed']==plan['source_metadata']['selected_attempt']['random_reset_seed'] and resets[0]['options'] is None and
                resets[1]['seed']==plan['source_metadata']['selected_attempt']['placement_reset_seed'],'original reset seeds')
        request=of('original_placement_request');require(len(request)==1 and resets[1]['options']==dict(state=request[0]['state'],goal_state=request[0]['goal']),'original recipe request, not saved poststate input')
        require(np.array_equal(resets[1]['observation']['state'],packet['state']) and
                np.array_equal(resets[1]['info']['goal_state'],goal) and
                physical.source.root_hash(packet['state'],goal)==plan['root_sha256'],'actual initial/source state/root')
        trace=row['trace'];require(len(trace)==10 and [e['receipt'] for e in of('step_observation')]==trace,'all returned traces bound')
        observed=of('fresh_rgb_observed');guards=of('rgb_guard_observed');require(len(observed)==len(guards)==11,'fresh frame before each dispatch')
        for i in range(11):
            frame=observed[i]['receipt'];g=guards[i]['receipt'];require(frame['step']==guards[i]['step']==i,'observed chronological RGB')
            state=packet['state'] if i==0 else trace[i-1]['actual_state']
            require(np.array_equal(frame['state'],state),'render/actual state chronology')
            require(g['root_id']==plan['root_sha256'] and g['primitive_commands_authorized']==i and not g['abstained'] and
                    [o['primitive_step'] for o in g['observations']]==list(range(i+1)) and
                    all(o['canonical_agent_pixels']>0 for o in g['observations']),'original guard chronology/visibility')
            if i<10:
                attempt=of('step_attempted')[i]
                require(guards[i]['index']<attempt['index']<of('step_returned')[i]['index']<observed[i+1]['index'],'fresh dispatch event order')
        require(row['guard_receipt']==guards[-1]['receipt'],'final guard receipt')
        for i,t in enumerate(trace,1):
            require(t['step']==i and t['attempted'] and t['returned'] and
                    np.array_equal(t['command'],plan['commands'][i-1]) and
                    of('step_attempted')[i-1]['receipt']['command']==t['command'],'raw commands unchanged')
            require(np.array_equal(t['observation']['state'],t['actual_state']) and
                    of('actual_successor')[i-1]['state']==t['actual_state'] and
                    of('returned_observation')[i-1]['observation']==t['observation'] and
                    of('official_info_returned')[i-1]['info']==t['official_info'],'actual/returned state and official info')
            require(np.array_equal(t['official_info']['goal_state'],goal),'official goal unchanged')
            m=check_metric(goal,t['actual_state'],t['pose_diagnostics'],t['native_official_evaluation'])
            negative+=m['negative_native_angle'];disagreement+=m['native_success']!=m['wrapped_success'];termination+=bool(t['terminated'] or t['truncated'])
            risk=t['termination_protocol_risk']
            require(risk['negative_native_angle_term']==m['negative_native_angle'] and
                    risk['native_vs_wrapped_success_disagreement']==(m['native_success']!=m['wrapped_success']) and
                    not risk['native_eval_vs_formula_disagreement'] and
                    risk['terminated_vs_eval_disagreement']==(t['terminated']!=m['native_success']),'native risk flags')
        require(row['final_state']==trace[-1]['actual_state'] and row['official_info']==trace[-1]['official_info'],'final state/info exactly final return')
        terminal=check_metric(goal,row['final_state'],row['pose_diagnostics'],row['official_evaluation'])
        require([c['step'] for c in row['checkpoints']]==[0,5,10] and
                [e['receipt'] for e in of('checkpoint_saved')]==row['checkpoints'],'fixed checkpoint schedule')
        task_path=direct_directory/f'task_{plan["candidate_id"]}.json'
        task=physical.read(task_path)
        require(physical.core.task_id(task['task'])==task['task_id']==plan['task_id'] and task['root_sha256']==plan['root_sha256'],'encoding source task binding')
        for checkpoint in row['checkpoints']:
            step=checkpoint['step'];image=png_array(checkpoint,folder)
            require(checkpoint['rgb_sha256']==observed[step]['receipt']['rgb_sha256'] and
                    checkpoint['state']==observed[step]['receipt']['state'],'checkpoint/observed frame')
            count=int(np.count_nonzero(np.all(image==[65,105,225],axis=-1)))
            require(count==guards[step]['receipt']['observations'][-1]['canonical_agent_pixels'],'PNG independently counted guard pixels')
            if step==0:require(np.array_equal(image,packet['rgb_history'][0]),'initial RGB original source exact')
            acquisition.append(dict(root_sha256=plan['root_sha256'],task_id=plan['task_id'],arm=plan['arm'],step=step,
                 selection_sha256=sid,png=str(Path(checkpoint['path']).resolve()),file_sha256=checkpoint['file_sha256'],
                 rgb_sha256=checkpoint['rgb_sha256'],shape=[224,224,3],dtype='uint8',
                 task_path=str(task_path.resolve()),task_file_sha256=physical.sha(task_path),
                 raw_prefix_commands=plan['commands'][:step],sample_count=1))
        initial=np.asarray(packet['state']);states=np.asarray([t['actual_state'] for t in trace]);moved=np.any(states[:,2:5]!=initial[2:5],axis=1)
        branches.append(dict(root_sha256=plan['root_sha256'],candidate_id=plan['candidate_id'],arm=plan['arm'],metrics=terminal,
              object_state_changed_steps=(np.flatnonzero(moved)+1).tolist(),
              positive_official_contact_steps=[t['step'] for t in trace if t['official_info']['n_contacts']>0]))
    require(len(list(folder.glob('*.png')))==24,'exact24 PNG assets, no borrowed/extra checkpoint')
    paired=[]
    for cid in (0,1):
        subset={b['arm']:b for b in branches if b['candidate_id']==cid};zero=subset['zero']['metrics'];two=subset['two_stage']['metrics']
        for arm in physical.ARMS:
            current=subset[arm]['metrics']
            paired.append(dict(candidate_id=cid,root_sha256=subset[arm]['root_sha256'],arm=arm,
                metrics=current,delta_vs_zero={k:current[k]-zero[k] for k in ('object_position','object_angle','agent_position','all_xy','native_state')},
                two_stage_minus_this_arm={k:two[k]-current[k] for k in ('object_position','object_angle','agent_position','all_xy','native_state')}))
    require(physical.sha(report)==report_sha,'physical report changed during audit')
    result=dict(schema=NAME+'.report',physical_report_sha256=report_sha,journal_sha256=r['journal_sha256'],
                status='saved_audit_passed',branches=branches,paired=paired,checked_steps=80,checked_frames=88,checked_png=24,
                negative_native_angle_steps=negative,native_wrapped_success_disagreements=disagreement,termination_steps=termination,
                official_successes=sum(b['metrics']['native_success'] for b in branches),P1_admission=False,equal_compute=False)
    manifest=dict(schema=NAME+'.acquisition',physical_report=str(report),physical_report_sha256=report_sha,
                  physical_prepare=str(Path(prepared).resolve()),physical_prepare_sha256=prepare_sha,
                  journal_sha256=r['journal_sha256'],images=acquisition,
                  new_native_samples=dict(encoder=24,projector=24,predict=0,action_encoder=0),
                  selection='all24_fixed_no_dedup_or_outcome_selection',execute_neural=False,P1_admission=False)
    return result,manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('prepared','prepare-sha','report','report-sha','output'):parser.add_argument('--'+k,required=True)
    a=parser.parse_args();r,m=audit(a.prepared,a.prepare_sha,a.report,a.report_sha)
    store=physical.Store(a.output);store.json('report.json',r);store.json('acquisition.json',m)
    print(physical.source.canonical(dict(status=r['status'],report_sha256=physical.sha(store.path/'report.json'),acquisition_sha256=physical.sha(store.path/'acquisition.json'))))


if __name__=='__main__':main()
