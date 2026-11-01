import pytest
from experiments.spectral_reachability import agent_e3_authorized_episode_20261029_v1 as m


def test_auth_snapshot_env_execution_and_postcheck_order():
    calls=[]
    def load():calls.append('authority');return {'body':'fixture'}
    def snap():calls.append('snapshot');return {'frozen':True}
    def create():calls.append('create');return object()
    def execute(env,b):calls.append('execute_and_cleanup');return b
    def validate(a,b):calls.append('validate');assert a==b
    def record(kind,value):calls.append(kind)
    assert m._sequence(load,snap,create,execute,validate,record)=={'body':'fixture'}
    assert calls==['authority','snapshot','validate','guard_before','create','execute_and_cleanup',
                   'snapshot','validate','guard_after']


@pytest.mark.parametrize('failure',['authority','before','execute','after'])
def test_failure_order_and_postcheck(failure):
    calls=[];count=0
    def load():
        calls.append('authority')
        if failure=='authority':raise ValueError('authority')
        return {}
    def snap():
        nonlocal count
        count+=1;calls.append('snapshot')
        if (failure=='before' and count==1) or (failure=='after' and count==2):raise ValueError(failure)
        return {}
    def create():calls.append('create');return object()
    def execute(env,b):
        calls.append('execute_and_cleanup')
        if failure=='execute':raise ValueError('execute')
    with pytest.raises(ValueError,match=failure):
        m._sequence(load,snap,create,execute,lambda a,b:None,lambda k,v:calls.append(k))
    if failure in ('authority','before'):assert 'create' not in calls
    else:assert calls.count('snapshot')==2
    if failure=='after':assert 'guard_failure' in calls
