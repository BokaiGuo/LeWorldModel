"""Non-neural command/receipt fixtures; no environment allocation or models."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import p2_gradient_mpc_physical_20261027_v1 as subject


def ready():
    return dict(root_index=0, root_sha256='fixed', root=dict(root_id='fixed', context={}),
        action_domain=dict(name='raw', low=[-1.,-1.], high=[1.,1.], action_block=5))


def test_plan_exact_fixed_horizon():
    plan = subject.plan_for(ready(), np.zeros((10,2),np.float32), 'candidate', 'receipt')
    subject.check_plan(plan)
    assert plan['planned_commands']==10 and plan['planned_horizon_tokens']==2
    assert plan['method']==subject.planner.ARM


@pytest.mark.parametrize('commands', [np.zeros((5,2)),np.full((10,2),np.nan),np.full((10,2),1.001)])
def test_bad_commands_rejected(commands):
    with pytest.raises(ValueError):
        subject.plan_for(ready(), commands, 'candidate', 'receipt')


def test_wrong_root_rejected():
    row = ready(); row['root']['root_id']='other'
    with pytest.raises(ValueError):
        subject.plan_for(row, np.zeros((10,2)), 'candidate', 'receipt')


@pytest.mark.parametrize('field,value', [('commands', [[1.,0.]]*10),('method','other'),('planned_commands',5)])
def test_modified_selection_rejected(field,value):
    plan = subject.plan_for(ready(),np.zeros((10,2)),'candidate','receipt')
    plan[field]=value
    with pytest.raises(ValueError): subject.check_plan(plan)


def receipt_fixture():
    rows=[dict(root_sha256='a',method='m',full_horizon_completed=True,completed_commands=10),
          dict(root_sha256='b',method='m',full_horizon_completed=False,completed_commands=6)]
    enc=[dict(root_sha256='a',method='m',status='completed',attempted_encode_samples=1),
         dict(root_sha256='b',method='m',status='blocked_fixed_horizon',attempted_encode_samples=0)]
    return rows,enc


def test_partial_must_not_be_encoded_as_fixed_horizon():
    rows,enc=receipt_fixture()
    assert subject.encoding_contract(rows,enc)
    enc[1].update(status='completed',attempted_encode_samples=1)
    assert not subject.encoding_contract(rows,enc)


def test_full_endpoint_must_not_silently_skip_encoding():
    rows,enc=receipt_fixture()
    enc[0].update(status='blocked_fixed_horizon',attempted_encode_samples=0)
    assert not subject.encoding_contract(rows,enc)


def test_duplicate_missing_receipts_rejected():
    rows,enc=receipt_fixture()
    assert not subject.encoding_contract(rows,enc[:1])
    assert not subject.encoding_contract(rows,enc+[deepcopy(enc[0])])


def test_no_invented_zero_derivative_cost():
    assert subject.ACCOUNTING['autograd_grad_calls']==156
    assert subject.ACCOUNTING['total_predict_samples']==336
    assert subject.ACCOUNTING['fair_total_budget_comparison'] is False
