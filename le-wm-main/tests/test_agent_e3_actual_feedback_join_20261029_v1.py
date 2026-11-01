import copy
import hashlib
import numpy as np
import pytest
from experiments.spectral_reachability import agent_e3_actual_feedback_join_20261029_v1 as j


def args():
    return dict(context={'latent_history':np.zeros((3,192),np.float32),
                         'action_history':np.zeros((2,10),np.float32)},
                prior_raw5=np.zeros((5,2),np.float32), submitted_raw5=np.ones((5,2),np.float32)*.2,
                returned_raw5=np.ones((5,2),np.float32)*.2, observed_latent=np.ones(192,np.float32),
                mean=np.array([.1,.2],np.float32),std=np.array([.2,.4],np.float32),
                expected_rgb_sha=hashlib.sha256(b'frame').hexdigest(),rgb_bytes=b'frame',
                encoding_input_sha=hashlib.sha256(b'frame').hexdigest(),episode_id='cell0',
                physical_episode_id='cell0',encoder_episode_id='cell0',start=5,end=10,
                physical_end=10,encoding_boundary=10,stopped=False)


def test_actual_two_boundaries():
    a=args(); c=j.joined_context(**a)
    np.testing.assert_array_equal(c['latent_history'][-1],a['observed_latent'])
    np.testing.assert_array_equal(c['action_history'],((np.concatenate((a['prior_raw5'],a['returned_raw5']))-a['mean'])/a['std']).reshape(2,10))
    a.update(context=c, prior_raw5=a['returned_raw5'],start=10,end=15,physical_end=15,encoding_boundary=15)
    j.joined_context(**a)


@pytest.mark.parametrize('field,value', [('encoder_episode_id','other'),('physical_episode_id','other'),
 ('physical_end',9),('encoding_boundary',15),('stopped',True),('stopped',1),('start',True),
 ('rgb_bytes',b'other'),('encoding_input_sha','0'*64)])
def test_reject_wrong_join(field,value):
    a=args();a[field]=value
    with pytest.raises(ValueError):j.joined_context(**a)


def test_raw_bits_and_dtype():
    a=args();a['returned_raw5']=a['returned_raw5'].copy();a['returned_raw5'][0,0]=np.nextafter(a['returned_raw5'][0,0],np.float32(1))
    with pytest.raises(ValueError):j.joined_context(**a)
    a=args();a['observed_latent']=a['observed_latent'].astype(np.float64)
    with pytest.raises(ValueError):j.joined_context(**a)


def test_no_mutation_and_budget():
    a=args();old=copy.deepcopy(a['context']);j.joined_context(**a)
    np.testing.assert_array_equal(old['latent_history'],a['context']['latent_history'])
    assert j.full_budget()['total_transport_proven'] is False
