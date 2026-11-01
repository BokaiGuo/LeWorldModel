import pytest
from experiments.spectral_reachability import agent_e3_native_meter_20261029_v1 as m


class Space:
    def __init__(self,fail=False):self.fail=fail
    def step(self):
        if self.fail:raise RuntimeError('physics payload')


class Base:
    def __init__(self,fail=False):self.fail=fail;self.space=Space(fail);self.closed=0
    def _setup(self):self.space=Space(self.fail)
    def _set_state(self,value):return value
    def _set_goal_state(self,value):return value
    def reset(self):
        self._setup();self._set_state(1);self._set_goal_state(1);self.render()
    def render(self):return 'image'
    def step(self,action):self.space.step();return action
    def close(self):self.closed+=1


def fixture(tmp_path,fail=False,stop=False):
    store=m.pub.Store(tmp_path/'job');meter=m.Meter(store,'cell00')
    cls=m.native.metered_class(Base,meter)
    env=m.Env(meter.call('constructor',lambda:cls(fail)),meter)
    env.reset();meter.region='warmup'
    try:
        env.step(1)
        store.event('physical','record','native_warmup_return',store.index,
                    dict(terminated=stop,truncated=False))
    except RuntimeError:
        pass
    env.close()
    result=store.publish('partial' if fail else 'completed')
    return store,meter,env,result


@pytest.mark.parametrize('stop',[False,True])
def test_real_metered_class_nested_and_early_stop(tmp_path,stop):
    s,meter,env,seal=fixture(tmp_path,stop=stop)
    result=m.consume(s.output,seal['report_sha256'],seal['completion_sha256'],'cell00')
    assert result['ledger_complete'] and result['native_stop_observed'] is stop
    assert result['counts']==meter.counts and result['counts']['physics_step']['completed']==1
    assert result['counts']['close']['completed']==1 and env._env.closed==1
    assert not result['full_native_strict']


def test_nested_failure_retained_and_cleanup(tmp_path):
    s,meter,env,seal=fixture(tmp_path,fail=True)
    result=m.consume(s.output,seal['report_sha256'],seal['completion_sha256'],'cell00')
    assert not result['ledger_complete'] and result['counts']==meter.counts
    assert result['counts']['physics_step']['failed']==result['counts']['public_step']['failed']==1
    assert result['counts']['close']['completed']==1 and env._env.closed==1


def test_deadline_cleanup_executes_even_if_record_disallowed(tmp_path):
    now=[0.];s=m.pub.Store(tmp_path/'job',clock=lambda:now[0]);meter=m.Meter(s,'cell00');called=[]
    now[0]=1801.
    with pytest.raises(ValueError,match='wall'):meter.cleanup(lambda:called.append('closed'))
    assert called==['closed'] and meter.counts['close']['attempted']==0


def test_no_new_public_after_stop_in_saved_ledger(tmp_path):
    s=m.pub.Store(tmp_path/'job');meter=m.Meter(s,'cell00')
    meter.call('constructor',lambda:None);meter.region='suffix'
    s.event('physical','record','native_control_return',s.index,dict(terminated=True,truncated=False))
    meter.call('public_step',lambda:None);meter.cleanup(lambda:None);seal=s.publish('completed')
    with pytest.raises(ValueError,match='after stop'):m.consume(s.output,seal['report_sha256'],seal['completion_sha256'],'cell00')


def test_write_before_cap_and_bool_units(tmp_path):
    s=m.pub.Store(tmp_path/'job');meter=m.Meter(s,'cell00');meter.region='suffix'
    with pytest.raises(ValueError):meter.begin('public_step',True)
    for _ in range(30):meter.call('public_step',lambda:None)
    with pytest.raises(ValueError,match='predispatch'):meter.call('public_step',lambda:pytest.fail('must not run'))


def test_cleanup_failure_is_a_failed_native_operation(tmp_path):
    s=m.pub.Store(tmp_path/'job');meter=m.Meter(s,'cell00')
    meter.call('constructor',lambda:None)
    def failed_close():raise RuntimeError('actual close failure')
    with pytest.raises(RuntimeError,match='actual close failure'):meter.cleanup(failed_close)
    seal=s.publish('partial');result=m.consume(s.output,seal['report_sha256'],seal['completion_sha256'],'cell00')
    assert result['counts']['close']==dict(attempted=1,completed=0,failed=1)
    assert not result['ledger_complete']


def test_native_phase_gate_rejects_physics_in_compile(tmp_path):
    s=m.pub.Store(tmp_path/'job');meter=m.Meter(s,'cell00');meter.region='compile'
    with pytest.raises(ValueError,match='region'):meter.call('physics_step',lambda:pytest.fail('must not run'))


def test_constructor_failure_is_recorded_without_false_close(tmp_path):
    s=m.pub.Store(tmp_path/'job');meter=m.Meter(s,'cell00')
    def fail():raise RuntimeError('constructor payload')
    with pytest.raises(RuntimeError,match='constructor payload'):meter.call('constructor',fail)
    seal=s.publish('partial');out=m.consume(s.output,seal['report_sha256'],seal['completion_sha256'],'cell00')
    assert out['counts']['constructor']==dict(attempted=1,completed=0,failed=1)
    assert out['counts']['close']['attempted']==0 and not out['ledger_complete']
