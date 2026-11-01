"""Real meter/dispatch interfaces exercised against fixture callbacks, not NN."""
import pytest
import test_agent_experiment1_common_controller_20261028_v1 as fixture
from experiments.spectral_reachability import agent_e1_six_slot_phase_meter_20261029_v1 as m


def test_replay_live_hooks_and_saved_consumer():
    task=fixture.task(0);events=[]
    meter=m.PhaseMeter('replay',events.append)
    raw=[[0.1,0.0]]*5+[[0.0,0.1]]*5
    # Literal source values must be exactly f32.
    raw=[[float(__import__('numpy').float32(x)) for x in row] for row in raw]
    result=m.replay(task,raw,fixture.Backend(),meter,events.append)
    assert result==m.verify_saved_replay(task,raw,events)
    assert meter.counts()==m.beam.budget(4)
    with pytest.raises(ValueError):meter.available()
    events[-1]['contexts'][0]['latent_history'][-1][0]+=1
    with pytest.raises(ValueError):m.verify_saved_replay(task,raw,events)


def test_fresh_phases_cannot_borrow_and_failed_attempt_not_refunded():
    events=[];a=m.PhaseMeter('controller',events.append);b=m.PhaseMeter('replay',events.append)
    a.group='edge_0';a.packet=0
    def fail():raise RuntimeError('fixture failure')
    with pytest.raises(RuntimeError):a.call('action_encoder',[],fail)
    assert a.counts()['action_encoder']==dict(attempted=1,completed=0)
    assert b.counts()==m.beam.budget(0)
    with pytest.raises(ValueError):a.available()


def test_reject_other_phase():
    with pytest.raises(ValueError):m.PhaseMeter('unknown',lambda _:None)
