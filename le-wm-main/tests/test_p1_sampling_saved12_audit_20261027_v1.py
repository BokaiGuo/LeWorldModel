"""Saved arrays / lightweight tamper tests. No tmp, neural calls or physics."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import p1_sampling_saved12_audit_20261027_v1 as a


@pytest.fixture(scope='module')
def saved():
    p=a.read(a.PREPARED)
    pp=a.read(a.producer.PREPARED/'prepare.json')
    sp=a.read(a.Path(pp['inputs']['arguments']['sampling_prepared'])/'protocol.json')
    tasks={x['root_sha256']:x['task'] for x in sp['inputs']['tasks']}
    events=[a.json.loads(line) for line in (a.RUN/'image.events.jsonl').read_text().splitlines()]
    return p['frames'],events,tasks


def test_actual_saved_raw_latents_same_original_Q(saved):
    rows,initial,_=a.recompute(*saved)
    reference=a.keyed(a.read(a.RUN/'analysis.json')['rows'])
    assert len(rows)==12 and all(x['initial_latents_bitwise_identical'] for x in initial)
    for row in rows:
        old=reference[row['root_sha256'],row['arm'],row['step']]
        for field in ('predicted_Q_mse','actual_Q_mse','native_prediction_mse'):
            assert row[field]==pytest.approx(old[field],rel=0,abs=2e-12)


@pytest.mark.parametrize('damage',['missing','extra','pixel','scope','task','count','attempt','raw_copy','nonfinite','shape','duplicate'])
def test_tamper_rejected(saved,damage):
    frames,events,tasks=deepcopy(saved)
    if damage=='missing':events.pop()
    if damage=='extra':events.append(events[0])
    if damage=='pixel':events[3]['rgb_sha256']='wrong'
    if damage=='scope':events[1]['root_sha256']='wrong'
    if damage=='task':events[0]['task_id']='wrong'
    if damage=='count':events[3]['counts']['encoder']=0
    if damage=='attempt':events[2]['samples']=2
    if damage=='raw_copy':events[4]['raw_native192'][0]+=.01
    if damage=='nonfinite':events[3]['latent'][0]=float('nan')
    if damage=='shape':events[3]['latent'].pop()
    if damage=='duplicate':frames[-1]=deepcopy(frames[0])
    with pytest.raises(ValueError):a.recompute(frames,events,tasks)


def test_original_Q_not_implicitly_identity(saved):
    frames,events,tasks=deepcopy(saved)
    original,_,_=a.recompute(*saved)
    # Rebind task receipts for a synthetic arithmetic intervention, not actual evidence.
    q=np.zeros((192,192));q[0,0]=1
    for task in tasks.values():task['projection']=q.tolist()
    for frame in frames:
        taskid=a.producer.core.task_id(tasks[frame['root_sha256']]);frame['task_id']=taskid
        events[5*frame['order']]['task_id']=taskid
    changed,_,_=a.recompute(frames,events,tasks)
    for i,(old,new) in enumerate(zip(original,changed)):
        goal=tasks[frames[i]['root_sha256']]['goal'][0]
        assert new['actual_Q_mse']==pytest.approx((events[5*i+3]['latent'][0]-goal)**2/192)
        assert new['predicted_Q_mse']==pytest.approx((frames[i]['predicted_native192'][0]-goal)**2/192)
        assert new['native_prediction_mse']==old['native_prediction_mse']
    assert any(new['actual_Q_mse']!=old['actual_Q_mse'] for old,new in zip(original,changed))


def test_velocity_changes_native_reward_not_success():
    goal=np.zeros(7);stationary=np.zeros(7);moving=stationary.copy();moving[5]=100
    left=a.native_decomposition(goal,stationary);right=a.native_decomposition(goal,moving)
    assert left['metrics']['native_success'] and right['metrics']['native_success']
    assert left['metrics']['wrapped_success'] and right['metrics']['wrapped_success']
    assert left['native_reward']==0 and right['native_reward']==-100
    assert right['reward_distance_squared_terms']['agent_velocity_squared']==10000
    assert right['success_has_velocity_condition'] is False


def test_wrapped_angle_not_native_silent_repair():
    goal=np.zeros(7);state=np.zeros(7);state[4]=2*np.pi+.4
    row=a.native_decomposition(goal,state)
    assert row['metrics']['negative_native_angle']
    assert row['metrics']['native_success'] and not row['metrics']['wrapped_success']


def test_stable_analysis_join_and_duplicate_rejection(saved):
    rows,_,_=a.recompute(*saved)
    assert a.keyed(rows)==a.keyed(rows[::-1])
    with pytest.raises(ValueError):a.keyed(rows+[rows[0]])
