"""MOCK contracts only: not source provenance or real CUDA/physics evidence."""
from copy import deepcopy
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import p4_feedback_root_sourcepack_20261027_v1 as m


@pytest.fixture
def bundle():
    records=[];frames=[];plans=[];roots=[];tasks=[];metrics={}
    for i in range(4):
        root=str(i)*64
        initial=dict(m.saved.META,latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist())
        packet=dict(root_sha256=root,root_context=initial,goal=np.zeros(192).tolist(),projection=np.eye(192).tolist())
        packet['input_sha256']=m.saved.digest(packet)
        metric=dict(projection=packet['projection'],active_metric_id='mock_metric')
        record=dict(packet=packet,metrics={'old_Q':metric,'goal_patch_Q':metric});records.append(record)
        metrics[root]=dict(goal=packet['goal'],metrics=record['metrics'])
        predicted=deepcopy(initial);predicted['latent_history'][-1]=[1.]*192
        observed=deepcopy(predicted);observed['latent_history'][-1]=[2.]*192
        arms={}
        for arm in m.saved.ARMS:
            plans.append(dict(root_sha256=root,method=arm,selection_sha256='mock_selection',predictor_context=initial,commands=[[0.,0.]]*10))
            arms[arm]=dict(segments=[predicted],winner=dict(actions=[[0.,0.]]*10))
            frames.append(dict(root_sha256=root,arm=arm,primitive_step=5,selection_sha256='mock_selection',source_input_sha256=packet['input_sha256'],
                active_metric_id='mock_metric',predicted_vector=[1.]*192,actual_vector=[2.]*192,rgb_path='MOCK_ONLY',frame_id='MOCK_'+root+arm))
            tasks.append(dict(root_sha256=root,arm=arm,boundary_step=5,context=observed))
        roots.append(dict(root_sha256=root,arms=arms))
    return dict(records=records,frames=frames,plans=plans,roots=roots,teacher_tasks=tasks,root_metrics=metrics,
        normalization=dict(mean=[0.,0.],std=[1.,1.]),bindings={},model_bindings={},
        compilation_refs={str(i)*64:dict(root_sha256=str(i)*64,may_reuse_as_observed_prefix_response=False) for i in range(4)},
        prior_roots=['1'*64],
        census=dict(known_roots=4,physical_branches=16,real_prefix_frames=17,independent_new_roots=0))


def build(b,root='0'*64,arm='old_Q__uniform',metric='old_Q'):
    return m.reconstruct(b,root,arm,metric,['1'*64])


@pytest.mark.parametrize('root',['0'*64,'2'*64,'3'*64])
@pytest.mark.parametrize('metric',['old_Q','goal_patch_Q'])
def test_parameterized_mock_roots(bundle,root,metric):
    p=build(bundle,root,metric+'__dynamic',metric)
    assert p['status']=='prepared_missing_actual_response'
    assert not p['runtime_admissible'] and not p['response_completed']
    assert not p['independent_confirmation'] and not p['P1_admission'] and not p['P4_admission']
    assert all(v==0 for v in p['actual_new_calls'].values())
    imagined,actual=p['response_requests']
    assert imagined['context']['latent_history'][-1]==[1.]*192
    assert actual['context']['latent_history'][-1]==[2.]*192
    assert actual['context']['action_history']==imagined['context']['action_history']
    assert actual['must_recompile'] and not actual['can_reuse_initial_jacobian']


@pytest.mark.parametrize('mutation',[
    'duplicate_record','wrong_packet_hash','wrong_prefix_source','wrong_selection','wrong_action',
    'wrong_predicted_latent','teacher_future_leak','wrong_metric','missing_frame','duplicate_frame',
    'wrong_goal','wrong_context_history'])
def test_reconstruction_fail_closed(bundle,mutation):
    if mutation=='duplicate_record':bundle['records'][3]=deepcopy(bundle['records'][0])
    elif mutation=='wrong_packet_hash':bundle['records'][0]['packet']['input_sha256']='bad'
    elif mutation=='wrong_prefix_source':bundle['frames'][0]['source_input_sha256']='bad'
    elif mutation=='wrong_selection':bundle['frames'][0]['selection_sha256']='bad'
    elif mutation=='wrong_action':bundle['plans'][0]['commands'][0]=[.1,.2]
    elif mutation=='wrong_predicted_latent':bundle['frames'][0]['predicted_vector']=[99.]*192
    elif mutation=='teacher_future_leak':bundle['teacher_tasks'][0]['context']['latent_history'][-1][0]=99
    elif mutation=='wrong_metric':bundle['frames'][0]['active_metric_id']='bad'
    elif mutation=='missing_frame':bundle['frames'].pop(0)
    elif mutation=='duplicate_frame':bundle['frames'].append(deepcopy(bundle['frames'][0]))
    elif mutation=='wrong_goal':bundle['root_metrics']['0'*64]=dict(goal=[99.]*192)
    else:bundle['roots'][0]['arms']['old_Q__uniform']['segments'][0]['action_history'][0][0]=99
    with pytest.raises((ValueError,KeyError)):build(bundle)


@pytest.mark.parametrize('priors',[[],['0'*64],['1'*64,'1'*64],['9'*64],['2'*64]])
def test_prior_roster(bundle,priors):
    with pytest.raises(ValueError):m.reconstruct(bundle,'0'*64,'old_Q__uniform','old_Q',priors)


def test_no_metric_switch(bundle):
    with pytest.raises(ValueError):build(bundle,arm='old_Q__uniform',metric='goal_patch_Q')


def test_budget_is_future_not_actual():
    b=m.budget(2);assert (b['predict_samples_cap'],b['mathematical_VJP_directions_cap'])==(26,384)
    r=m.runtime_budget();assert (r['predict_samples_cap'],r['mathematical_VJP_directions_cap'],r['physics_steps_cap'])==(78,1152,40)
    assert r['encoded_frames_cap']==20 and r['new_encoder_calls_cap']==8
    assert r['backward_engine_entries'] is None and r['launch_blockers']
    assert r['terminal_purpose']=='evaluation_only_never_feedback'


def test_exclusive_prepare_and_verify_mock(bundle,tmp_path,monkeypatch):
    monkeypatch.setattr(m,'collect',lambda **kwargs:deepcopy(bundle))
    output=tmp_path/'MOCK_ONLY'
    m.prepare(output,{},'0'*64,'old_Q__uniform','old_Q',['1'*64])
    sha=m.saved.sha(output/'prepare.json');p=m.verify(output,sha)
    assert p['runtime_admissible'] is False
    with pytest.raises(ValueError):m.prepare(output,{},'0'*64,'old_Q__uniform','old_Q',['1'*64])
    with pytest.raises(ValueError):m.verify(output,'0'*64)
    assert m.saved.disk_bytes(output)<m.OUTPUT_CAP
    bundle['frames'][0]['actual_vector'][0]=99
    with pytest.raises(ValueError):m.verify(output,sha)


def test_missing_saved_source_writes_nothing(tmp_path):
    output=tmp_path/'never_written'
    with pytest.raises((FileNotFoundError,ValueError)):
        m.prepare(output,dict(prefix_dir=tmp_path/'missing',prefix_sha='0'*64,teacher_prepared='',teacher_prepare_sha='',teacher_dir='',teacher_report_sha=''),
            '0'*64,'old_Q__uniform','old_Q',['1'*64])
    assert not output.exists()


def test_no_neural_or_environment_import():
    assert not any(x in sys.modules for x in ('torch','gymnasium','pymunk','stable_worldmodel'))


@pytest.mark.parametrize('fault',['hash','status','errors','model','counts','scope','entries'])
def test_mock_receipt_rejects_before_using_frames(tmp_path,monkeypatch,fault):
    p=dict(frames=[dict(frame_id=str(i)) for i in range(17)],model_bindings={},physical_report_sha256='p',planner_report_sha256='c',goal_q_sha256='q')
    r=dict(status='all_17_prefix_latents_saved',errors=[],blocked=[],journal_errors=[],source_verified=True,model_unchanged=True,
        model_hash_before='model',model_hash_after='model',model_bindings={},prepare_sha256='prepare',
        counts={k:dict(attempted=17,completed=17) for k in ('encoder','existing_projector')},
        schema=m.saved.NAME+'.derived_receipt',first_error=None,physical_report_sha256='p',planner_report_sha256='c',goal_q_sha256='q',
        new_predict_calls=0,new_AD_calls=0,new_physical_steps=0,training=False,P1_admission=False,P3_admission=False,efficacy_claim=False,
        rows=[{}]*17,entries=[dict(frame_id=str(i),kind=k,completed=True) for i in range(17) for k in ('encoder','existing_projector')])
    if fault=='status':r['status']='partial'
    elif fault=='errors':r['errors']=['mock_failed']
    elif fault=='model':r['model_hash_after']='drift'
    elif fault=='counts':r['counts']['encoder']['attempted']=18
    elif fault=='scope':r['new_AD_calls']=1
    elif fault=='entries':r['entries'].pop()
    monkeypatch.setattr(m.saved,'sha',lambda p:'mock_sha')
    monkeypatch.setattr(m.saved,'read',lambda path:r if str(path).endswith('receipt.json') else dict(prepared='mock',prepare_sha256='prepare'))
    monkeypatch.setattr(m.saved,'verify',lambda *args:p)
    with pytest.raises(ValueError):
        m.collect(tmp_path,'bad_sha' if fault=='hash' else 'mock_sha','mock','mock','mock','mock')
