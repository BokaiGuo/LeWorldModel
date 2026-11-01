"""Saved actual arrays plus in-memory tampering; no tmp, GPU, or physics."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import p1_observed_direct_saved_audit_20261027_v1 as m

DIRECTORY=m.ROOT/'results/p1_observed_direct_cuda_20261027_v2_cuda_main01'


@pytest.fixture(scope='module')
def arrays():
    t=m.saved.read_gzip(DIRECTORY/'task_0.json.gz')['task']
    arm=DIRECTORY/'root_0_full_H2_GN'
    result=m.saved.read_gzip(arm/'report.json.gz')
    events,eof=m.saved.recover_events(arm/'direct_journal.jsonl.gz')
    assert eof
    return t,result,events


def test_actual_one_arm_full_saved_check(arrays):
    t,r,e=arrays;a=m.check_component(t,r,e)
    assert a['resources']['predict']==70 and a['raw_J_rows']==1536
    assert a['full_H2_evaluations']==27 and a['paid_AD_H2_forwards']==8
    assert len(a['costs'])==9 and len(a['gradient_norms'])==8


@pytest.mark.parametrize('defect',['raw_row','row_order','joint_J','forward_point','AD_packet','packet_token',
    'missing_prediction','operation_identity','missing_return','changed_report','trajectory','configuration'])
def test_in_memory_corruption_rejected(arrays,defect):
    t,r,e=deepcopy(arrays)
    def first(kind):return next(x for x in e if x['kind']==kind)
    if defect=='raw_row':first('jacobian_row')['raw_control_derivative'][0]+=.1
    elif defect=='row_order':first('jacobian_row')['row']=191
    elif defect=='joint_J':first('joint_logit_jacobian')['jacobian'][0][0]+=.1
    elif defect=='forward_point':first('jacobian_forward_complete')['logits'][0][0]=.1
    elif defect=='AD_packet':first('jacobian_forward_packet')['context']['latent_history'][0][0]+=.1
    elif defect=='packet_token':first('native_packet')['context']['action_history'][0][0]+=.1
    elif defect=='missing_prediction':e.remove(first('prediction_path'))
    elif defect=='operation_identity':first('operation_completed')['event']['resource']='svd'
    elif defect=='missing_return':e.remove(first('operation_completed'))
    elif defect=='changed_report':r['direct_runs'][0][8]['incumbent']['cost']+=1
    elif defect=='trajectory':first('trajectory_step')['step']['iteration']=17
    else:r['configuration']['damping']=.02
    with pytest.raises(ValueError):m.check_component(t,r,e)


def test_ledger_count_false_completion_rejected(arrays):
    t,r,e=deepcopy(arrays)
    r['extended_journal'][0]['completed']=False
    with pytest.raises(ValueError):m.check_component(t,r,e)


def test_prefix_only_ignores_provenance_task_ids(arrays):
    t,r,_=arrays
    old=dict(t,task_contract_sha256='a'*64,metric_source_sha256='b'*64)
    prefix=dict(direct_runs=[deepcopy(r['direct_runs'][0][:4])])
    prefix['direct_runs'][0][0]['incumbent']['task_id']='c'*64
    a=m.compare_prefix(t,r,old,prefix)
    assert a['zero_exact'] and a['B0_through_B3_exact_ignoring_only_task_id']
    prefix['direct_runs'][0][3]['incumbent']['cost']+=1
    assert not m.compare_prefix(t,r,old,prefix)['B0_through_B3_exact_ignoring_only_task_id']


@pytest.mark.parametrize('field',['goal','action_mean','root_context','projection'])
def test_prefix_same_root_label_not_enough(arrays,field):
    t,r,_=arrays;bad=deepcopy(t)
    if field=='root_context':bad[field]['action_history'][0][0]+=.1
    elif field=='projection':bad[field][0][0]+=.1
    else:bad[field][0]+=.1
    with pytest.raises(ValueError):m.compare_prefix(t,r,bad,dict(direct_runs=[r['direct_runs'][0][:4]]))


def test_projection_independent_matches_float32_contract():
    for seed in range(4):
        x=np.random.default_rng(seed).normal(size=(2,10))*3
        actual=m.project(x,float(.75*np.sqrt(5)))
        assert np.array_equal(actual,m.saved.screen.projected_logits(x,float(.75*np.sqrt(5))))
