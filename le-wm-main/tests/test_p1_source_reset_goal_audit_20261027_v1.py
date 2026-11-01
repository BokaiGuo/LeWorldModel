"""Pure array/interface fakes only: no native environment or neural imports."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import p1_source_reset_goal_audit_20261027_v1 as m


class Block:
    shapes = []
    velocity = (0., 0.)
    angular_velocity = 0.


class Fake:
    def __init__(self, fail=None, close_fail=False):
        self.unwrapped = self
        self.block = Block()
        self.state = np.zeros(7)
        self.goal_state = np.zeros(7)
        self.calls = []
        self.fail = fail
        self.close_fail = close_fail
        self.closed = False

    def _get_obs(self): return self.state.copy()

    def _set_state(self, value):
        self.state = np.asarray(value).copy()
        self.block.velocity = (2., 3.)
        self.block.angular_velocity = .5

    def render(self): return np.full((224,224,3), 19, np.uint8)

    def reset(self, seed, options=None):
        self.calls.append((seed, deepcopy(options)))
        if len(self.calls) == self.fail: raise RuntimeError('reset primary')
        self.goal_state = np.array([200.,200.,250.,250.,0.,0.,0.]) if options is None else options['goal_state'].copy()
        self._set_state(self.goal_state)
        goal_rgb = self.render()
        state = self.goal_state.copy() if options is None else options['state'].copy()
        self._set_state(state)
        return {'state': self._get_obs()}, {'goal_state':self.goal_state.copy(),'goal':goal_rgb}

    def step(self, *args): raise AssertionError('never step')

    def close(self):
        self.closed = True
        if self.close_fail: raise RuntimeError('close secondary')


def fixture(monkeypatch, tmp_path):
    goal = np.array([200.,200.,250.,250.,0.,0.,0.])
    requested = goal.copy(); requested[0] += 10
    plan = dict(slots=[dict(candidate_id=i,root_seed=10+i,reset_seed_span=[100+4*i,104+4*i]) for i in range(2)],
                placement_reset_seed=0,rgb_shape=[224,224,3],
                environment_sha256=m.sha(__file__),source_closure={str(Path(__file__).resolve()):m.sha(__file__)})
    image = np.full((224,224,3),19,np.uint8)
    operations = {}
    packets = {}
    for cid in range(2):
        packets[cid] = dict(state=requested.copy(),goal=goal.copy(),rgb_history=np.stack([image]*3),goal_rgb=image,
                            metadata=dict(source=dict(proposal_index=0),selected_attempt=dict(attempt_index=0),
                                          root_sha256=m.source.root_hash(requested,goal)))
        for i in range(2):
            for j in range(2):
                for op,state in (('random_reset',goal),('placement_reset',requested)):
                    operations[cid,i,j,op] = dict(state=state.tolist(),goal=goal.tolist(),goal_rgb_sha256=m.image_sha(image))
                operations[cid,i,j,'render'] = dict(rgb_sha256=m.image_sha(image))
    monkeypatch.setattr(m,'material',lambda *args: (dict(plan=plan),{},operations,packets))
    monkeypatch.setattr(m.source,'sample_state',lambda *args: requested.copy())
    prepared = dict(schema=m.NAME+'.prepare',driver_sha256=m.sha(m.__file__),source_sha256=m.sha(m.source.__file__),
                    request=m.__file__,report=m.__file__,request_sha256=m.sha(m.__file__),report_sha256=m.sha(m.__file__))
    return prepared,packets,operations


def test_all_eight_original_requests_exact_hook_counts_and_selected(monkeypatch,tmp_path):
    prepared,_,_ = fixture(monkeypatch,tmp_path); env = Fake()
    result = m.execute(prepared,tmp_path/'run',lambda:env,authorization=m.AUTH)
    assert result['status'] == 'diagnostic_completed'
    assert result['all_reproduction_checks']
    assert result['counts'] == {k:dict(attempted=v,completed=v) for k,v in dict(reset=16,explicit_render=8,set_state=32,goal_render=16).items()}
    assert [seed for seed,opt in env.calls if opt is None] == list(range(100,108))
    assert all(opt['state'][0] == 210 for seed,opt in env.calls if opt is not None)
    assert sum(r['selected'] for r in result['rows']) == 2
    assert env.closed and '_set_state' not in env.__dict__ and 'render' not in env.__dict__
    events = m.source.read_journal(tmp_path/'run'/'journal.jsonl')
    assert sum(e['kind']=='set_state_before' for e in events) == 32
    assert sum(e['kind']=='set_state_after' for e in events) == 32
    assert all(r['placement_reset']['renders'][0]['before']['block_velocity']==[2.,3.] for r in result['rows'])
    assert result['public_steps'] == result['neural_calls'] == 0 and not result['P1_admission']


def test_saved_post_state_is_comparison_only_not_placement_input(monkeypatch,tmp_path):
    prepared,packets,_ = fixture(monkeypatch,tmp_path)
    packets[0]['state'][0] = 999.
    env = Fake(); result = m.execute(prepared,tmp_path/'run',lambda:env,authorization=m.AUTH)
    assert result['status']=='diagnostic_completed' and not result['all_reproduction_checks']
    assert result['counts']['reset']['completed']==16
    assert env.calls[1][1]['state'][0]==210.
    assert not result['rows'][0]['checks']['selected_state_exact']


def test_reset_failure_keeps_denominator_attempt_and_original_error(monkeypatch,tmp_path):
    prepared,_,_ = fixture(monkeypatch,tmp_path); env = Fake(fail=2,close_fail=True)
    result = m.execute(prepared,tmp_path/'run',lambda:env,authorization=m.AUTH)
    assert result['status']=='failed' and len(result['rows'])==8
    assert result['rows'][0]['status']=='failed' and result['rows'][1]['status']=='not_started'
    assert result['first_error']['message']=='reset primary'
    assert result['secondary_errors'][0]['phase']=='close'
    assert result['counts']['reset']==dict(attempted=2,completed=1)
    assert (tmp_path/'run'/'report.json').is_file()


def test_hook_failure_retains_before_arrays_and_attempt(monkeypatch,tmp_path):
    prepared,_,_ = fixture(monkeypatch,tmp_path)
    class Broken(Fake):
        def _set_state(self,value): raise RuntimeError('setter failed')
    result = m.execute(prepared,tmp_path/'run',Broken,authorization=m.AUTH)
    assert result['counts']['set_state']==dict(attempted=1,completed=0)
    assert result['rows'][0]['random_reset']['set_state'][0]['requested']
    assert result['rows'][0]['random_reset']['set_state'][0]['after_failure']['state7']
    assert result['first_error']['message']=='setter failed'


def test_goal_actual_drift_is_recorded_without_skipping_attempts(monkeypatch,tmp_path):
    prepared,_,_ = fixture(monkeypatch,tmp_path)
    class Drift(Fake):
        def _set_state(self,value):
            super()._set_state(value)
            self.state[0] += .25
    result = m.execute(prepared,tmp_path/'run',Drift,authorization=m.AUTH)
    assert result['status']=='diagnostic_completed' and not result['all_reproduction_checks']
    assert result['counts']['reset']['completed']==16
    for row in result['rows']:
        for phase in ('random_reset','placement_reset'):
            capture=row[phase]['renders'][0]
            assert capture['goal_actual_difference']['raw_delta'][0]==.25
            assert not capture['goal_actual_difference']['exact']


def test_render_failure_keeps_attempt_and_snapshot(monkeypatch,tmp_path):
    prepared,_,_ = fixture(monkeypatch,tmp_path)
    class BrokenRender(Fake):
        def render(self): raise RuntimeError('render primary')
    result = m.execute(prepared,tmp_path/'run',BrokenRender,authorization=m.AUTH)
    assert result['counts']['goal_render']==dict(attempted=1,completed=0)
    capture=result['rows'][0]['random_reset']['renders'][0]
    assert capture['before']['state7'] and not capture['completed']
    assert result['first_error']['message']=='render primary'


def test_output_exclusive_and_authorization_before_factory(monkeypatch,tmp_path):
    prepared,_,_ = fixture(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match='authorization'):
        m.execute(prepared,tmp_path/'run',lambda:pytest.fail('factory'),authorization='NO')
    (tmp_path/'run').mkdir()
    with pytest.raises(FileExistsError):
        m.execute(prepared,tmp_path/'run',lambda:pytest.fail('factory'),authorization=m.AUTH)


def test_final_sha_failure_does_not_replace_execution_error(monkeypatch,tmp_path):
    prepared,_,_ = fixture(monkeypatch,tmp_path)
    original=m.sha
    class Gone(Fake):
        def close(self):
            super().close()
            monkeypatch.setattr(m,'sha',lambda path: (_ for _ in ()).throw(OSError('gone')))
    result=m.execute(prepared,tmp_path/'run',lambda:Gone(fail=1),authorization=m.AUTH)
    assert result['first_error']['message']=='reset primary'
    assert any(e['phase']=='final_binding' for e in result['secondary_errors'])
    monkeypatch.setattr(m,'sha',original)


def test_storage_gate_before_factory(monkeypatch,tmp_path):
    prepared,_,_ = fixture(monkeypatch,tmp_path)
    from collections import namedtuple
    usage=namedtuple('Usage','total used free')
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:usage(100,99,1))
    with pytest.raises(ValueError,match='reserve'):
        m.execute(prepared,tmp_path/'run',lambda:pytest.fail('factory called'),authorization=m.AUTH)
    assert not (tmp_path/'run').exists()


def test_external_sha_drift_before_parsing(tmp_path):
    with pytest.raises(ValueError,match='SHA drift'):
        m.material(__file__,__file__,'0'*64,'0'*64)


def test_difference_keeps_raw_and_wrapped_angle():
    a=np.zeros(7); b=np.zeros(7); a[4]=2*np.pi
    value=m.difference(a,b)
    assert not value['exact'] and value['raw_delta'][4]==2*np.pi
    assert value['angle_wrapped_delta'][4]==0


def test_import_is_non_neural_non_environment():
    subprocess.run([sys.executable,'-c','import sys; from experiments.spectral_reachability import '+m.NAME+'; assert not any(x in sys.modules for x in ("torch","gymnasium","pymunk","stable_worldmodel"))'],check=True)
