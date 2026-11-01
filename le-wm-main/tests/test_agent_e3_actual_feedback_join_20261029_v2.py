import numpy as np
import pytest
from experiments.spectral_reachability import agent_e3_actual_feedback_join_20261029_v2 as v2
from test_agent_e3_actual_feedback_join_20261029_v1 import args


def valid():
    a=args();a['prior_raw5']=np.arange(10,dtype=np.float32).reshape(5,2)/np.float32(20)
    a['context']['action_history'][-1]=((a['prior_raw5']-a['mean'])/a['std']).reshape(10)
    return a


def test_nonzero_exact_history_and_next_boundary():
    a=valid();c=v2.joined_context(**a)
    a.update(context=c,prior_raw5=a['returned_raw5'],start=10,end=15,physical_end=15,encoding_boundary=15)
    v2.joined_context(**a)


def test_wrong_prior_fails():
    a=valid();a['prior_raw5']=a['prior_raw5'].copy();a['prior_raw5'][0,0]=np.float32(.3)
    with pytest.raises(ValueError,match='prior raw5'):v2.joined_context(**a)


def test_signed_zero_bits_are_not_equal():
    a=valid();a['mean'][:]=0;a['std'][:]=1;a['prior_raw5'][:]=0
    a['context']['action_history'][-1]=0
    a['context']['action_history'][-1,0]=np.float32(-0.)
    with pytest.raises(ValueError,match='prior raw5'):v2.joined_context(**a)
