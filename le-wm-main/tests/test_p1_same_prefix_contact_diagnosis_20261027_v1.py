"""Saved-only mechanism checks; all mutations remain in memory."""
from copy import deepcopy
import json
import numpy as np
import pytest
from experiments.spectral_reachability import p1_same_prefix_contact_diagnosis_20261027_v1 as m


@pytest.fixture(scope='module')
def data():
    p=json.loads(m.saved.PREPARED.read_text());r=json.loads((m.saved.ACTUAL/'report.json').read_text())
    protocol=json.loads((m.ROOT/'results/p1_two_stage_witness_20261027_v1_prepared01/protocol.json').read_text())
    return p,r,{v['root_sha256']:v['task'] for v in protocol['inputs']['tasks']}


def one(data,index):
    p,r,t=data;role=p['inputs']['roles'][index];plans=p['inputs']['plans'][41*index:41*(index+1)];rows=r['rows'][41*index:41*(index+1)]
    return role,plans,rows,rows[0]['official_info']['goal_state'],t[role['root_sha256']]


@pytest.mark.parametrize('index',[1,2,4,5])
def test_actual_selected_suffix_still_but_agent_moves(data,index):
    args=one(data,index);v=m.group(*args,zero_q5=0.)
    assert v['suffix_sampled_object_displacement_from_step5_max']==0.
    assert v['suffix_sampled_object_wrapped_angle_from_step5_max']==0.
    assert v['endpoint_centered_object_displacement_max']==0.
    assert v['suffix_steps_with_positive_native_contact_receipt']==0
    assert v['nonzero_probes_with_positive_signed_agent_response']==40
    assert v['off_axis_agent_response_max']==0.
    assert v['no_agent_state_change_before_scheduled_probe']
    assert v['signed_probed_axis_agent_response_range'][0]>7.


@pytest.mark.parametrize('index,position,angle,positive',[(0,17.61681135295363,.07875210232882335,127),(3,20.70187928784818,.14142692818030067,36)])
def test_zero_prefix_has_object_response(data,index,position,angle,positive):
    v=m.group(*one(data,index),zero_q5=0.)
    assert v['suffix_sampled_object_displacement_from_step5_max']==pytest.approx(position)
    assert v['suffix_sampled_object_wrapped_angle_from_step5_max']==pytest.approx(angle)
    assert v['suffix_steps_with_positive_native_contact_receipt']==positive


def test_prefix_already_moved_object_not_confused_with_suffix(data):
    v=m.group(*one(data,2),zero_q5=0.)
    assert v['prefix_object_displacement']==pytest.approx(7.536121976312143)
    assert v['prefix_object_wrapped_angle_delta']==pytest.approx(.047099261528567915)
    assert v['suffix_sampled_object_wrapped_angle_from_step5_max']==0.


def test_root1_duplicate_does_not_create_independent_root(data):
    p,r,_=data;v=m.repetition(p['inputs']['plans'],r['rows'])
    assert v['root_prefix_groups']==5 and v['unique_root_command_sequences']==205
    assert v['independent_new_roots']==0 and len(v['repeated_role_pairs'])==1
    assert v['repeated_role_pairs'][0]['identical_actual_traces']


def test_repeated_command_does_not_assert_actual_trace_identity(data):
    p,r,_=data;rows=deepcopy(r['rows']);rows[205]['trace'][0]['actual_state'][0]+=.1
    v=m.repetition(p['inputs']['plans'],rows)
    assert not v['repeated_role_pairs'][0]['identical_actual_traces']


@pytest.mark.parametrize('defect',['axis','time','normalizer','center','prefix','domain','partial','identity','dynamics'])
def test_coordinate_center_and_actual_dispatch_tamper(data,defect):
    role,ps,rs,goal,task=one(data,1);plans=deepcopy(ps);rows=deepcopy(rs)
    if defect=='axis':plans[1]['commands'][5]=list(reversed(plans[1]['commands'][5]))
    elif defect=='time':plans[1]['commands'][5],plans[1]['commands'][6]=plans[1]['commands'][6],plans[1]['commands'][5]
    elif defect=='normalizer':plans[1]['commands'][5][0]=(plans[1]['commands'][5][0]-task['action_mean'][0])/task['action_std'][0]
    elif defect=='center':plans[1]['commands'][6][0]=.1
    elif defect=='prefix':plans[1]['commands'][0][0]=.1
    elif defect=='domain':task=deepcopy(task);task['packet_logit_cap']=.1
    elif defect=='partial':rows[1]['full_horizon']=False
    elif defect=='identity':plans[1]['root_sha256']='other'
    elif defect=='dynamics':rows[1]['body_dynamics']['5']['block']['velocity'][0]=.1
    with pytest.raises(ValueError):m.group(role,plans,rows,goal,task,0.)


def test_physical_agent_share_not_a_latent_q_attribution():
    e=m.goal_errors([10.,0.,1.,0.,0.,0.,0.],[0.]*7)
    assert e['agent_share_native_position_squared_error']==pytest.approx(100/101)
    assert 'NOT' in e['caveat']
    assert m.goal_errors([0.]*7,[0.]*7)['agent_share_native_position_squared_error'] is None


def test_angle_wrap_preserves_separate_physical_measurement():
    e=m.goal_errors([0.,0.,0.,0.,2*np.pi-.1,0.,0.],[0.]*7)
    assert e['wrapped_object_angle']==pytest.approx(.1)
    assert e['native_state7_distance']>6


def test_agent_pixel_evidence_no_fitted_readout():
    image=np.zeros((224,224,3),np.uint8);image[10,20]=[65,105,225]
    v=m.agent_pixels(image)
    assert v['canonical_RoyalBlue_pixels']==1 and v['centroid_xy']==[20.,10.]
    assert m.agent_pixels(np.zeros_like(image))['centroid_xy'] is None


def test_absolute_radii_not_cap_fraction():
    cap=.75*np.sqrt(5)
    assert .5/cap==pytest.approx(.29814239699997197)
    assert 100*np.tanh(np.float32(.5))==pytest.approx(46.21172)
    assert all(np.linalg.norm(p['logits'])<=.5 for p in m.saved.probes())
