"""RAM only: no filesystem fixtures, model imports, physics, network or process launch."""
from copy import deepcopy
import json
import math
import pytest
from experiments.spectral_reachability import agent_baseline_protocol_bridge_20261028_v1 as m


def export(slot=0):
    return dict(source_slot=slot,with_target=False,P1_admission=False,task_id=str(slot)*64,blind_sha256='b'*64,
        task_contract=dict(source_slot=slot,with_target=False,metric='identity_native192_no_fit',observed_steps=[0,5,10],
            normalization=dict(mean=[-.0087,.0068],std=[.2019,.2002],training_provenance_verified=False)),
        task=dict(environment='swm/PushT-v1',horizon=2,packet_steps=5,control_dim=2,latent_dim=192,
            feedback='open_loop_model_only',packet_logit_cap=.75*math.sqrt(5),goal=[0.]*192,
            projection=[[int(i==j) for j in range(192)] for i in range(192)],
            root_raw_action_history=[[0.10000000149011612,0.]]*5+[[0.,0.10000000149011612]]*5,
            action_mean=[-.0087,.0068],action_std=[.2019,.2002],root_context=dict(layout='states_with_between_state_actions_v1',
                context_length=3,latent_dim=192,action_dim=10,latent_history=[[0.]*192 for _ in range(3)],action_history=[[.1]*10,[.2]*10])))


def fixture():
    protocols=[m.task_protocol(export(i),i,str(i)*64) for i in (0,1)]
    p=[];r=[];fw=[]
    for w in m.readiness.WORKFLOWS:
        c=dict(with_target=False,training=False,P1_admission=False)
        if w.key=='moving_gradient':c.update(logits_shape=[2,10],raw_commands_shape=[10,2],feedback_MPC=False,arm='moving_gradient_open_loop')
        else:c.update(horizon=2,arms=['GRASP_H3_bounded'] if w.key=='moving_grasp' else ['CEM','kernel_faithful_mppi_nominal'])
        slots=[{k:x[k] for k in ('source_slot','task_sha256','task_id','blind_sha256')} for x in protocols]
        p.append(dict(expected_model_hash=m.readiness.MODEL_SHA,contract=c,slots=slots))
        r.append(dict(contract=deepcopy(c),resources={'predict':{'attempted':3,'completed':3}}))
        fw.append(dict(workflow=w.key,status='ready_for_MAIN_dry_preflight',prepared='/fixture/prepared.json',saved_report='/fixture/report.json',
            main_dry_preflight=[dict(argv=['python','-m',w.module,'consume'],executed=False)]))
    return dict(shared_errors=[],local_workflows=fw),p,r,protocols,[dict(id=k,record={'value':42.},source={'table':'fixture'}) for k in m.CELLS]


def test_actual_identity_full_protocol_and_author_separation():
    out=m.assemble(*fixture())
    assert len(out['local_workflows'])==3 and out['all_methods_denominator']==4
    assert len(out['shared_tasks'])==2 and len(out['missing_asset_requests'])==4
    assert out['same_saved_task_identity'] and not out['equal_predict_budget_GRASP_vs_LeWM']
    assert out['forward_matched_pair']==['CEM','kernel_faithful_mppi_nominal']
    assert out['author_reported_context_only'][0]['record']['value']==42.
    assert all(x['local_result_values'] is None for x in out['local_workflows'])
    assert not out['P1_admission'] and not out['launch_authorized']


@pytest.mark.parametrize('case',range(10))
def test_task_mutations_failclosed(case):
    e=export();t=e['task'];c=e['task_contract']
    if case==0:e['source_slot']=False
    elif case==1:e['with_target']=True
    elif case==2:t['horizon']=1
    elif case==3:t['root_context']['latent_history'].pop()
    elif case==4:t['projection'][0][1]=.01
    elif case==5:t['root_raw_action_history']=[[0.,0.]]*10
    elif case==6:t['action_std'][0]=0.
    elif case==7:t['packet_logit_cap']=4.
    elif case==8:t['goal'][0]=float('nan')
    else:c['metric']='fitted_readout'
    with pytest.raises(ValueError):m.task_protocol(e,0,'a'*64)


@pytest.mark.parametrize('case',range(7))
def test_fixed_manifest_no_eligibility_by_outcome(case):
    f=list(fixture());pre,ps,rs,tasks,authors=f
    if case==0:pre['shared_errors'].append({'missing':'authority'})
    elif case==1:pre['local_workflows'][0]['status']='blocked'
    elif case==2:ps[0]['slots'].pop()
    elif case==3:ps[1]['slots'][0]['task_sha256']='f'*64
    elif case==4:ps[2]['contract']['feedback_MPC']=True;rs[2]['contract']['feedback_MPC']=True
    elif case==5:authors.reverse()
    else:ps[0]['expected_model_hash']='f'*64
    with pytest.raises(ValueError):m.assemble(*f)


def test_negative_local_outcomes_never_used_to_select_method():
    f=list(fixture());rs=f[2]
    for r in rs:r.update(native_success=False,Q=100000.)
    a=m.assemble(*f)
    for r in rs:r.update(native_success=True,Q=-999.)
    assert m.assemble(*f)==a


@pytest.mark.parametrize('method',m.METHODS)
def test_intake_transport_cannot_authorize_runtime(method):
    records=[dict(role=r,path='/fixture/'+r,sha256='a'*64) for r in m.ROLES[method]]
    report=m.validate_records(method,records,lambda p:'a'*64)
    assert report['status']=='materials_verified_not_runtime_ready'
    assert report['official_protocol_ready'] is report['inference_authorized'] is report['semantic_review_complete'] is False
    assert report['local_result'] is None


@pytest.mark.parametrize('case',range(8))
def test_unknown_missing_corrupt_intake(case):
    method='Hi-LeWM';records=[dict(role=r,path='/fixture/'+r,sha256='a'*64) for r in m.ROLES[method]]
    if case==0:records.pop()
    elif case==1:records.append(deepcopy(records[0]))
    elif case==2:records[0]['role']='fake'
    elif case==3:records[0]['externally_reviewed_complete']=True
    elif case==4:records[0]['path']='relative'
    elif case==5:records[0]['path']='/fixture/incomplete.part'
    elif case==6:records[0]['sha256']=None
    else:records[0]['sha256']='f'*64
    if case<4:
        with pytest.raises(ValueError):m.validate_records(method,records,lambda p:'a'*64)
    else:
        r=m.validate_records(method,records,lambda p:'a'*64)
        assert r['status']=='blocked' and len(r['errors'])==1


def test_template_unknown_fails_without_any_file_read():
    r=m.validate_records('PlanWAM',m.request('PlanWAM')['records'],lambda p:pytest.fail('unknown must not read'))
    assert r['status']=='blocked' and len(r['errors'])==len(m.ROLES['PlanWAM'])


def test_hi_base_does_not_incorrectly_require_C_bank():
    r=m.request('Hi-LeWM');assert '-C' in r['variant'] and all('bank' not in x for x in r['required_roles'])
    assert 'additional' in r['notes']


@pytest.mark.parametrize('raw',[b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":Infinity}'])
def test_strict_json(raw):
    with pytest.raises(ValueError):m.decode(raw)


def test_external_manifest_sha_before_build(monkeypatch):
    monkeypatch.setattr(m,'hash_material',lambda p:'a'*64)
    monkeypatch.setattr(m,'build',lambda:pytest.fail('bad SHA before rebuild'))
    with pytest.raises(ValueError):m.verify('/fixture/manifest','f'*64)


def test_check_cli_no_write_no_dispatch(monkeypatch,capsys):
    data={'schema':m.NAME+'.protocol_manifest','launch_authorized':False}
    monkeypatch.setattr(m,'build',lambda:data)
    monkeypatch.setattr(m,'write_new',lambda *a:pytest.fail('no output requested'))
    assert m.main(['check'])==0
    assert json.loads(capsys.readouterr().out)==data
