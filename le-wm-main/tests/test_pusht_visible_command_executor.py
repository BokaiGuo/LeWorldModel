import numpy as np
import pytest
from experiments.spectral_reachability.pusht_visible_command_executor import execute


class Env:
    def __init__(self, disappear=100, fail=False): self.count=0; self.disappear=disappear; self.fail=fail; self.commands=[]
    def render(self):
        rgb=np.full((224,224,3),255,np.uint8)
        if self.count < self.disappear: rgb[20,20]=[65,105,225]
        return rgb
    def step(self, command):
        self.commands.append(command.copy());self.count+=1
        if self.fail: raise RuntimeError('fixture')
        return {},0.,False,False,{}


def run(env, actions=None):
    return execute(env, [[.1,0.]]*10 if actions is None else actions, root_id='r',agent_rgb=(65,105,225))


def test_stops_immediately_after_first_missing_observation():
    env=Env(disappear=2);r=run(env)
    assert r['status']=='observation_unavailable'
    assert env.count==r['completed_commands']==2
    assert r['remaining_commands_not_executed']==8
    assert len(env.commands)==2


def test_no_blind_first_command():
    env=Env(disappear=0);r=run(env)
    assert r['status']=='observation_unavailable' and not env.commands


def test_complete_program_preserves_all_raw_commands():
    env=Env();r=run(env)
    assert r['status']=='completed' and r['completed_commands']==10
    np.testing.assert_array_equal(env.commands, [[.1,0.]]*10)


def test_execution_error_not_retried_or_padded():
    env=Env(fail=True);r=run(env)
    assert r['status']=='execution_failed'
    assert r['attempted_commands']==1 and r['completed_commands']==0 and len(env.commands)==1


@pytest.mark.parametrize('value', [[], [[2.,0.]], [[True,0.]], [[float('nan'),0.]], [[0.,0.]]*11])
def test_bad_program_rejected(value):
    with pytest.raises(ValueError): run(Env(), value)
