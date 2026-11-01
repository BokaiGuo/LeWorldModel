"""Independent saved audit; no tmp/cache/result writes, neural or physics.

Direct execution prints a compact recomputed JSON summary, no output file.
"""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import p1_b8_physical_20261027_v1 as producer

physical=producer.physical
ROOT=producer.ROOT
NEW=ROOT/'results/p1_b8_physical_20261027_v1_main01'
OLD=ROOT/'results/p1_witness_physical_20261027_v1_main01'
PREP=ROOT/'results/p1_b8_physical_20261027_v1_prepare01.json'
OLDPREP=ROOT/'results/p1_witness_physical_20261027_v1_prepared01'
PREPSHA='e872fdbfef99ef47a00d48ea9263a590cc3f792dd9b9e3184cebdbb56aaad706'
NEWSHA='e94a6cbf55dd9b408d350e690d17bfdb9ca693b2ef360a12ab904024f272cd7a'
OLDSHA='48df27e54eca33159c44a834ce0be1a13723553dc9d64dde996b7b4ca9eaf747'
OLDPREPSHA='e4fc7b7adeac3a23036196adaadfbbe69d8873f1e8649ebbf5fe9e8fb1eca0ff'


def metrics(goal,state):
    delta=np.asarray(state,float)-np.asarray(goal,float)
    assert delta.shape==(7,) and np.isfinite(delta).all()
    raw=abs(float(delta[4]));native=min(raw,2*np.pi-raw)
    wrapped=abs(float((delta[4]+np.pi)%(2*np.pi)-np.pi))
    xy=float(np.linalg.norm(delta[:4]))
    return dict(object_position=float(np.linalg.norm(delta[2:4])),object_angle=wrapped,
        agent_position=float(np.linalg.norm(delta[:2])),all_xy=xy,native_state=float(np.linalg.norm(delta)),
        native_angle=native,negative_native_angle=native<0,
        native_success=bool(xy<20 and native<np.pi/9),wrapped_success=bool(xy<20 and wrapped<np.pi/9))


def check_metrics(goal,state,diagnostic,official):
    value=metrics(goal,state);native=diagnostic['native_formula'];wrapped=diagnostic['independent']
    for key,saved in [('object_position',wrapped['object_position_error']),('object_angle',wrapped['wrapped_angle_error']),
        ('agent_position',wrapped['agent_position_error']),('all_xy',wrapped['all_xy_position_error']),
        ('native_state',native['state_distance']),('native_state',official['state_distance']),('native_angle',native['angle_term'])]:
        assert np.isclose(value[key],saved,rtol=0,atol=2e-12),(key,value[key],saved)
    assert value['native_success'] is native['success'] is official['success']
    assert value['wrapped_success'] is wrapped['same_allxy_threshold_wrapped_success']
    assert value['negative_native_angle'] is native['negative_angle_term']
    assert diagnostic['native_vs_wrapped_success_disagreement'] is (value['native_success']!=value['wrapped_success'])
    return value


def png(record,folder):
    path=Path(record['path'])
    if not path.is_absolute():path=ROOT/path
    assert path.resolve().parent==folder.resolve() and not path.is_symlink()
    assert physical.sha(path)==record['file_sha256']
    with Image.open(path) as image:
        assert image.format=='PNG' and image.mode=='RGB' and image.size==(224,224)
        array=np.asarray(image).copy()
    assert array.dtype==np.uint8 and physical.core.sha(array.tobytes())==record['rgb_sha256']
    return array


def bound_run(folder,expected_sha,prepare_sha,branches):
    assert physical.sha(folder/'report.json')==expected_sha
    report=physical.read(folder/'report.json')
    assert report['status']=='physical_calibration_recorded' and report['errors']==[]
    assert report['prepare_sha256']==prepare_sha and len(report['rows'])==branches
    counts={k:dict(attempted=n,completed=n) for k,n in dict(reset=2*branches,step=10*branches,render=11*branches).items()}
    assert report['counts']==counts
    assert physical.sha(folder/'journal.jsonl')==report['journal_sha256']
    events=physical.source.read_journal(folder/'journal.jsonl')
    previous=None
    for i,event in enumerate(events):
        body=dict(event);claim=body.pop('event_sha256')
        assert body['index']==i and body['previous_sha256']==previous and physical.source.digest(body)==claim
        previous=claim
    assert events[0]['kind']=='run_started' and events[0]['prepare_sha256']==prepare_sha
    assert events[-1]['kind']=='run_finished' and events[-1]['status']==report['status']
    assert events[-1]['counts']==counts and not events[-1]['errors']
    census=Counter(e['kind'] for e in events)
    for key,n in dict(branch_started=branches,branch_finished=branches,branch_closed=branches,
        reset_attempted=2*branches,reset_returned=2*branches,reset_observation=2*branches,
        step_attempted=10*branches,step_returned=10*branches,step_observation=10*branches,
        render_attempted=11*branches,render_returned=11*branches,fresh_rgb_observed=11*branches,
        rgb_guard_observed=11*branches,checkpoint_saved=3*branches).items():assert census[key]==n,key
    return report,events


def branch(row,plan,events,folder,source_plan):
    sid=plan['selection_sha256'];ev=[e for e in events if e.get('selection_sha256')==sid]
    def of(kind):
        if kind=='branch_closed':return [e for e in events if e['kind']==kind and e['receipt']['selection_sha256']==sid]
        return [e for e in ev if e['kind']==kind]
    assert row['root_sha256']==plan['root_sha256'] and row['arm']==plan['arm']
    assert row['status']=='completed' and row['full_horizon'] and row['endpoint_rgb_fresh']
    assert row['endpoint_scope']=='full_H2' and row['first_error'] is None and not row['secondary_errors']
    assert row['initial_exact'] and all(row['initial_checks'].values())
    assert row['attempted_steps']==row['returned_steps']==10
    assert len(of('branch_started'))==len(of('branch_finished'))==len(of('branch_closed'))==1
    assert of('branch_started')[0]['plan']==plan
    assert of('branch_finished')[0]['receipt']==of('branch_closed')[0]['receipt']==row
    packet=physical.source.load_raw_source(plan['raw_source_file'],source_plan)
    assert packet['metadata']==plan['source_metadata']
    goal=packet['goal'];initial=packet['state']
    assert physical.source.root_hash(initial,goal)==plan['root_sha256']
    resets=of('reset_observation');request=of('original_placement_request')
    assert len(resets)==2 and len(request)==1
    assert resets[0]['seed']==packet['metadata']['selected_attempt']['random_reset_seed'] and resets[0]['options'] is None
    assert resets[1]['seed']==source_plan['placement_reset_seed']
    assert resets[1]['options']==dict(state=request[0]['state'],goal_state=request[0]['goal'])
    assert np.array_equal(resets[0]['info']['goal_state'],goal)
    assert np.array_equal(resets[1]['observation']['state'],initial) and np.array_equal(resets[1]['info']['goal_state'],goal)
    assert [e['receipt'] for e in of('step_observation')]==row['trace']
    assert len(row['trace'])==10
    observed=of('fresh_rgb_observed');guards=of('rgb_guard_observed')
    assert len(observed)==len(guards)==11
    for i in range(11):
        frame=observed[i]['receipt'];g=guards[i]['receipt']
        assert frame['step']==guards[i]['step']==i
        assert np.array_equal(frame['state'],initial if i==0 else row['trace'][i-1]['actual_state'])
        assert g['root_id']==plan['root_sha256'] and g['primitive_commands_authorized']==i and not g['abstained']
        assert [o['primitive_step'] for o in g['observations']]==list(range(i+1))
        assert all(o['canonical_agent_pixels']>0 for o in g['observations'])
        if i<10:assert guards[i]['index']<of('step_attempted')[i]['index']<of('step_returned')[i]['index']<observed[i+1]['index']
    assert row['guard_receipt']==guards[-1]['receipt']
    initial_metric=check_metrics(goal,initial,row['initial_pose_diagnostics'],
        dict(success=metrics(goal,initial)['native_success'],state_distance=metrics(goal,initial)['native_state']))
    negative=disagree=terminal=0
    for i,t in enumerate(row['trace'],1):
        assert t['step']==i and t['attempted'] and t['returned']
        assert t['command']==plan['commands'][i-1]==of('step_attempted')[i-1]['receipt']['command']
        assert t['observation']['state']==t['actual_state']==of('actual_successor')[i-1]['state']
        assert of('returned_observation')[i-1]['observation']==t['observation']
        assert of('official_info_returned')[i-1]['info']==t['official_info']
        assert np.array_equal(t['official_info']['goal_state'],goal)
        term=of('termination_returned')[i-1]
        assert (term['reward'],term['terminated'],term['truncated'])==(t['reward'],t['terminated'],t['truncated'])
        value=check_metrics(goal,t['actual_state'],t['pose_diagnostics'],t['native_official_evaluation'])
        negative+=value['negative_native_angle'];disagree+=value['native_success']!=value['wrapped_success']
        terminal+=bool(t['terminated'] or t['truncated'])
        risk=t['termination_protocol_risk']
        assert risk['negative_native_angle_term']==value['negative_native_angle']
        assert risk['native_vs_wrapped_success_disagreement']==(value['native_success']!=value['wrapped_success'])
        assert not risk['native_eval_vs_formula_disagreement']
        assert risk['terminated_vs_eval_disagreement']==(t['terminated']!=value['native_success'])
    assert row['final_state']==row['trace'][-1]['actual_state'] and row['official_info']==row['trace'][-1]['official_info']
    final=check_metrics(goal,row['final_state'],row['pose_diagnostics'],row['official_evaluation'])
    assert [c['step'] for c in row['checkpoints']]==[0,5,10]
    assert [e['receipt'] for e in of('checkpoint_saved')]==row['checkpoints']
    for c in row['checkpoints']:
        image=png(c,folder);i=c['step']
        assert c['rgb_sha256']==observed[i]['receipt']['rgb_sha256'] and c['state']==observed[i]['receipt']['state']
        assert int(np.count_nonzero(np.all(image==[65,105,225],axis=-1)))==guards[i]['receipt']['observations'][-1]['canonical_agent_pixels']
        if i==0:assert np.array_equal(image,packet['rgb_history'][0])
    moved=[t['step'] for t in row['trace'] if not np.array_equal(t['actual_state'][2:5],initial[2:5])]
    return dict(initial=initial_metric,final=final,object_pose_changed_steps=moved,
        fresh_rgb_hashes=[e['receipt']['rgb_sha256'] for e in observed],
        contact_steps=[t['step'] for t in row['trace'] if t['official_info']['n_contacts']>0],
        negative_native_angle_steps=int(negative),native_wrapped_disagreements=int(disagree),terminal_steps=int(terminal))


def audit():
    p=producer.verify(PREP,PREPSHA);oldp=physical.verify(OLDPREP,OLDPREPSHA)
    new,events=bound_run(NEW,NEWSHA,PREPSHA,4);old,oldevents=bound_run(OLD,OLDSHA,OLDPREPSHA,8)
    assert new['schema']==producer.NAME+'.report' and new['contract']==producer.contract()
    assert not new['P1_admission'] and not new['model_task_eligible'] and not new['stall_certified']
    assert new['new_independent_samples']==new['neural_calls']==0 and new['global_freshness']=='unknown'
    assert new['internal_counts']==dict(reset_physics_completed_implied=16,reset_physics_attempted_upper_bound=16,
        goal_renders_completed_implied=8,actual_substeps=None)
    assert len(list(NEW.glob('*.png')))==12
    index={(r['root_sha256'],r['arm']):r for r in new['rows']}
    oldindex={(r['root_sha256'],r['arm']):r for r in old['rows']}
    oldplans={(r['root_sha256'],r['arm']):r for r in oldp['inputs']['plans']}
    assert len(index)==4 and set(index)=={(x['root_sha256'],x['arm']) for x in p['inputs']['plans']}
    assert p['inputs']['source_plan']==oldp['inputs']['source_plan']
    rows=[]
    for plan in p['inputs']['plans']:
        key=plan['root_sha256'],plan['arm'];row=index[key];oldrow=oldindex[key];oldplan=oldplans[key]
        current=branch(row,plan,events,NEW,p['inputs']['source_plan'])
        prior=branch(oldrow,oldplan,oldevents,OLD,oldp['inputs']['source_plan'])
        assert plan['raw_source_sha256']==oldplan['raw_source_sha256'] and plan['source_metadata']==oldplan['source_metadata']
        task=producer.gz(plan['task_file'])['task']
        old_task_path=Path(oldp['inputs']['arguments']['direct_report']).parent/f'task_{plan["candidate_id"]}.json'
        oldtask=physical.read(old_task_path)['task']
        for k in ('goal','projection','root_context','action_mean','action_std','root_raw_action_history','model_sha256','model_config_sha256','packet_logit_cap'):
            assert task[k]==oldtask[k],k
        model=producer.gz(plan['origin']);trajectory=model['direct_runs'][0]
        assert plan['prediction_sha256']==physical.source.digest(trajectory[-1]['incumbent'])
        assert plan['commands']==trajectory[-1]['incumbent']['commands']
        assert physical.sha(plan['origin'])==plan['origin_sha256']
        assert physical.sha(plan['task_file'])==plan['task_file_sha256']
        oldmodel=physical.read(oldplan['origin']);oldB=oldmodel['configuration']['B2']
        assert oldplan['commands']==oldmodel['direct_runs'][0][-1]['incumbent']['commands']
        oldzero=oldindex[plan['root_sha256'],'zero']
        goal=physical.source.load_raw_source(plan['raw_source_file'],p['inputs']['source_plan'])['goal']
        zero=metrics(goal,oldzero['final_state'])
        fields=('object_position','object_angle','agent_position','all_xy','native_state')
        exact=dict(commands=plan['commands']==oldplan['commands'],full_trace=row['trace']==oldrow['trace'],
            all11_fresh_rgb_hashes=current['fresh_rgb_hashes']==prior['fresh_rgb_hashes'],
            final_state=row['final_state']==oldrow['final_state'],guard=row['guard_receipt']==oldrow['guard_receipt'],
            checkpoint_states=[c['state'] for c in row['checkpoints']]==[c['state'] for c in oldrow['checkpoints']],
            checkpoint_pixel_hashes=[c['rgb_sha256'] for c in row['checkpoints']]==[c['rgb_sha256'] for c in oldrow['checkpoints']],
            checkpoint_file_hashes=[c['file_sha256'] for c in row['checkpoints']]==[c['file_sha256'] for c in oldrow['checkpoints']])
        rows.append(dict(root=plan['candidate_id'],root_sha256=plan['root_sha256'],arm=plan['arm'],old_physical_B2=oldB,
            exact_vs_old=exact,initial=current['initial'],final=current['final'],old_final=prior['final'],
            physical_delta_vs_old={k:current['final'][k]-prior['final'][k] for k in fields},
            physical_delta_vs_zero={k:current['final'][k]-zero[k] for k in fields},
            object_pose_changed_steps=current['object_pose_changed_steps'],contact_steps=current['contact_steps'],
            model_B4=trajectory[4]['incumbent']['cost'],model_B8=trajectory[8]['incumbent']['cost'],
            negative_native_angle_steps=current['negative_native_angle_steps'],
            native_wrapped_disagreements=current['native_wrapped_disagreements'],terminal_steps=current['terminal_steps']))
    assert physical.sha(NEW/'report.json')==NEWSHA and physical.sha(OLD/'report.json')==OLDSHA
    return dict(status='saved_audit_passed',new_report_sha256=NEWSHA,old_report_sha256=OLDSHA,
        checked_new_steps=40,checked_new_frames=44,checked_new_png=12,checked_old_comparison_steps=40,
        official_successes=sum(r['final']['native_success'] for r in rows),
        wrapped_successes=sum(r['final']['wrapped_success'] for r in rows),
        rows=rows,P1_admission=False,new_independent_samples=0,neural_calls=0,physical_calls=0)


@pytest.fixture(scope='module')
def actual():return audit()


def test_saved_bindings_assets_and_protocol(actual):
    assert actual['status']=='saved_audit_passed'
    assert actual['checked_new_steps']==40 and actual['checked_new_png']==12


@pytest.mark.parametrize('root',[0,1])
def test_GN_entire_actual_trajectory_identical_to_old(actual,root):
    row=next(r for r in actual['rows'] if r['root']==root and r['arm']=='full_H2_GN')
    assert all(row['exact_vs_old'].values()),row['exact_vs_old']
    assert row['old_physical_B2']==3 and row['model_B4']==row['model_B8']


@pytest.mark.parametrize('root',[0,1])
def test_GD8_improves_agent_not_object_pose(actual,root):
    row=next(r for r in actual['rows'] if r['root']==root and r['arm']=='full_H2_gradient')
    assert not row['exact_vs_old']['commands'] and not row['exact_vs_old']['full_trace']
    assert row['model_B8']<row['model_B4']
    assert row['physical_delta_vs_old']['all_xy']<0 and row['physical_delta_vs_old']['agent_position']<0
    assert row['physical_delta_vs_old']['object_position']==row['physical_delta_vs_old']['object_angle']==0
    assert row['physical_delta_vs_zero']['object_position']==row['physical_delta_vs_zero']['object_angle']==0
    assert row['object_pose_changed_steps']==row['contact_steps']==[]


def test_no_success_or_disagreement_or_admission(actual):
    assert actual['official_successes']==actual['wrapped_successes']==0
    assert sum(r['negative_native_angle_steps'] for r in actual['rows'])==0
    assert sum(r['native_wrapped_disagreements'] for r in actual['rows'])==0
    assert sum(r['terminal_steps'] for r in actual['rows'])==0
    assert not actual['P1_admission'] and actual['new_independent_samples']==0


if __name__=='__main__':print(physical.source.canonical(audit()))
