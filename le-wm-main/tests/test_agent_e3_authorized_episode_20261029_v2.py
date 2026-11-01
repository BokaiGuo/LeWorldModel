import pytest
from experiments.spectral_reachability import agent_e3_authorized_episode_20261029_v2 as m


def test_rejects_missing_main_authorization_before_io():
    with pytest.raises(ValueError,match='authorized'):
        m.run((),None,None,None,None,None,None,{}, {},'not-authorized')


def test_rejects_bad_authority_tuple_before_io():
    with pytest.raises(ValueError,match='authorized'):
        m.run((),None,None,None,None,None,None,{'episode_id':'cell00'}, {},m.AUTH)
