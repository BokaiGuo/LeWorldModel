"""RAM JSON/arithmetic/witness mutations and saved-only legacy bridge."""
from copy import deepcopy
from pathlib import Path
import pytest
from experiments.spectral_reachability import agent_subgoal_gap_contract_20261028_v1 as m
from experiments.spectral_reachability.d2p_composition_witness_v2 import PROVENANCE_KEYS
from experiments.spectral_reachability.p2_composition_paired_20261027_v2 import saved_witness


def fixture():
    h=m.digest({'RAM_only':True})
    c=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=2,
        action_dim=10,latent_history=[[0.,0.]]*3,action_history=[[0.]*10]*2)
    root=dict(root_id='RAM-only',root_rgb_sha256=h,goal_rgb_sha256=h,context=c)
    domain=dict(name='raw',units='RAM',low=[-1.,-1.],high=[1.,1.],action_block=5,normalization_sha256=h)
    mid=deepcopy(c);mid['latent_history'][-1]=[1.,0.]
    end=deepcopy(c);end['latent_history'][-1]=[2.,0.]
    w=saved_witness(root,{k:h for k in PROVENANCE_KEYS},domain,[c,mid,end],[[[.1,0.]]*5,[[.2,0.]]*5],'RAM')
    shared={k:h for k in m.SHARED};shared.update(root=m.digest(root),history=m.digest(c))
    wh=m.digest(w);native=dict(success=False,evaluator_sha256=h,evidence_sha256=h,task_goal_sha256=h)
    def row(arm):
        return dict(arm=arm,shared_sha256=m.digest(shared),status='completed',
            proposal=dict(source_sha256=h,generator_receipt_sha256=None,action_witness_sha256=None),
            target=[[1.,0.],[2.,0.]],predicted=[[1.,0.],[3.,0.]],observed=[[0.,0.],[2.,0.]],
            native=deepcopy(native),resources={'flops':None,'predict_samples':4},error=None)
    rows=[row(a) for a in m.ARMS]
    rows[1].update(status='unknown',proposal=None,target=None,predicted=None,observed=None,native=None,
        resources={'flops':None,'predict_samples':None},error='paired HiLeWM asset unavailable')
    rows[2]['proposal']['action_witness_sha256']=wh
    spec=dict(schema=m.NAME+'.input',latent_dim=2,checkpoints=[5,10],trial_denominator=1,
        trials=[dict(trial_id='fixed-root-0',shared=shared,arms=rows)],
        evidence_sha256=sorted(set(shared.values())|{wh}))
    return spec,{wh:w}


def test_three_arms_keep_unknown_native_false_and_tracking_errors_separate():
    spec,artifacts=fixture();out=m.score(spec,artifacts)
    a,prior,compiled=out['rows']
    assert a['deviation']==[dict(l2=1.,mse=.5),dict(l2=0.,mse=0.)]
    assert a['prediction_deviation'][-1]==dict(l2=1.,mse=.5)
    assert a['endpoint_deviation']==dict(l2=0.,mse=0.)
    assert a['endpoint_prediction_deviation']==dict(l2=1.,mse=.5)
    assert compiled['native_success'] is False  # zero endpoint latent error is NOT success.
    assert prior['deviation'] is prior['native_success'] is None
    assert out['arm_denominator']==3 and out['summary'][m.ARMS[1]]['unknown']==1
    assert out['physical_execution_authorized'] is out['scientific_pass'] is False


@pytest.mark.parametrize('damage',['missing_arm','missing_trial','shared_controller','fill_unknown0',
    'invent_prior','bad_native_goal','native_from_distance','bool_vector','missing_checkpoint','waypoint_swap',
    'broken_witness','bool_denominator'])
def test_contract_mutations_failclosed(damage):
    spec,artifacts=fixture();rows=spec['trials'][0]['arms']
    if damage=='missing_arm':rows.pop()
    elif damage=='missing_trial':spec['trials']=[]
    elif damage=='shared_controller':rows[0]['shared_sha256']='b'*64
    elif damage=='fill_unknown0':rows[1]['resources']['flops']=0
    elif damage=='invent_prior':rows[1]=deepcopy(rows[0]);rows[1]['arm']=m.ARMS[1]
    elif damage=='bad_native_goal':rows[0]['native']['task_goal_sha256']='b'*64
    elif damage=='native_from_distance':rows[0]['native']['success']=0.
    elif damage=='bool_vector':rows[0]['observed'][0][0]=False
    elif damage=='missing_checkpoint':rows[0]['observed'].pop()
    elif damage=='waypoint_swap':rows[2]['target'][0][0]=99.
    elif damage=='broken_witness':next(iter(artifacts.values()))['edges'][1]['input_context']['latent_history'][-1]=[99.,0.]
    else:spec['trial_denominator']=True
    with pytest.raises(ValueError):m.score(spec,artifacts)


def learned(spec,artifacts):
    row=spec['trials'][0]['arms'][1];shared=spec['trials'][0]['shared']
    row.update(deepcopy(spec['trials'][0]['arms'][0]));row['arm']=m.ARMS[1]
    r=dict(schema=m.NAME+'.learned_pairing',method='HiLeWM_hierarchical_adaptation',
        prior_checkpoint_sha256=shared['model'],targets_sha256=m.digest(row['target']),
        lowlevel_targets_sha256=m.digest(row['target']),generator_attempted=1,generator_completed=1,fallback=False)
    for key in ('model','encoder','normalizer','observation','history','task_goal'):r[key+'_sha256']=shared[key]
    h=m.digest(r);row['proposal']['generator_receipt_sha256']=h;artifacts[h]=r;spec['evidence_sha256'].append(h)
    return r


@pytest.mark.parametrize('damage',[None,'pairing','fallback','not_consumed','not_called'])
def test_real_learned_receipt_required_not_random_prior_or_final_goal(damage):
    spec,artifacts=fixture();r=learned(spec,artifacts)
    if damage=='pairing':r['encoder_sha256']='b'*64
    elif damage=='fallback':r['fallback']=True
    elif damage=='not_consumed':r['lowlevel_targets_sha256']='b'*64
    elif damage=='not_called':r['generator_completed']=0
    if damage is None:assert m.score(spec,artifacts)['summary'][m.ARMS[1]]['native_known']==1
    else:
        with pytest.raises(ValueError):m.score(spec,artifacts)


def test_partial_failure_keeps_trace_and_complete_denominator():
    spec,artifacts=fixture();row=spec['trials'][0]['arms'][0]
    row.update(status='failed',error='controller stopped',native=None);row['observed'][1]=None
    result=m.score(spec,artifacts)
    assert result['rows'][0]['deviation'][0]['l2']==1 and result['rows'][0]['deviation'][1] is None
    assert result['rows'][0]['endpoint_deviation'] is None
    assert result['summary'][m.ARMS[0]]['failed']==1 and result['arm_denominator']==3


@pytest.mark.parametrize('raw',[b'{"x":0,"x":1}',b'{"x":NaN}',b'{"x":1e999}'])
def test_nonportable_json_rejected(raw):
    with pytest.raises(ValueError):m.parse(raw)


def test_actual_byte_bound_score_CLI_and_changed_evidence_refused(tmp_path,capsys):
    spec,artifacts=fixture();w=next(iter(artifacts.values()))
    bodies=[{'RAM_only':True},w['receipt']['root'],w['receipt']['root']['context'],w]
    table={}
    for index,body in enumerate(bodies):
        path=tmp_path/f'evidence_{index}.json';path.write_bytes(m.encode(body));table[str(path)]=m.digest(body)
    inp=tmp_path/'input.json';inp.write_bytes(m.encode(spec))
    bindings=tmp_path/'bindings.json';bindings.write_bytes(m.encode(table))
    args=['score','--input',str(inp),'--sha256',m.digest(spec),'--bindings',str(bindings),'--bindings-sha256',m.digest(table)]
    assert m.main(args)==0
    out=m.parse(capsys.readouterr().out);assert out['arm_denominator']==3 and out['scientific_pass'] is False
    Path(next(iter(table))).write_bytes(b'changed RAM evidence')
    with pytest.raises(ValueError,match='bound evidence changed'):m.main(args)


def test_real_legacy_saved_reuses_original_validator_not_three_arm_evidence():
    root=Path(__file__).resolve().parents[1]
    result=m.legacy_waypoints(root/'results/reacher_d2p_waypoint_tracking8_20261026_cuda.json',
        '035493fc45022e0971a526ec04c22b6f471bbea7b1294ba8f4a858cd6b81a750')
    assert len(result['rows'])==64 and result['root_count']==8 and result['checkpoint_denominator']==384
    assert result['existing_learned_prior'] is result['arbitrary_latent_waypoint'] is result['native_success'] is None
    assert all(r['native_success'] is None and len(r['observed_to_frozen_waypoint_l2'])==6 for r in result['rows'])
