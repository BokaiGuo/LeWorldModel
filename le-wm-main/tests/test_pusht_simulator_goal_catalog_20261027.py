"""Real official simulator render tests; NO neural inference or GPU allocation."""
import os
from copy import deepcopy
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
import numpy as np
import pytest

from experiments.spectral_reachability import pusht_simulator_goal_catalog_20261027 as m


@pytest.fixture
def env():
    import gymnasium as gym
    import stable_worldmodel.envs  # register official environment
    e = gym.make('swm/PushT-v1', render_mode='rgb_array')
    yield e
    e.close()


def spec(x=80., y=80.):
    return np.asarray([x,y,256.,256.,float(np.float32(np.pi/4)),0.,0.])


def test_real_goal_render_not_restored_observation(env):
    target = spec()
    before = target.copy()
    result = m.render_goal_view(env, target)
    assert result['valid'] and not result['penetrations']
    np.testing.assert_array_equal(result['actual_goal_render_state'], target)
    np.testing.assert_array_equal(before, target)
    assert result['rgb'].shape == (224,224,3)
    assert result['rgb_sha256'] == m.pixels_sha(result['rgb'])
    assert 'render' not in vars(env.unwrapped)


def test_overlap_rejected_even_if_first_render_state_unchanged(env):
    result = m.render_goal_view(env, spec(256,256))
    assert not result['valid']
    assert result['penetrations']
    assert 'agent_penetrates_object_or_wall' in result['invalid_reasons']


def test_grid_all_slots_retained_no_pixel_editing(env):
    entries = m.build_catalog(env, spec())
    assert len(entries) == len({v['id'] for v in entries}) == 256
    assert any(e['valid'] for e in entries) and any(not e['valid'] for e in entries)
    for e in entries:
        np.testing.assert_array_equal(e['requested_state'][2:5], spec()[2:5])
        if e['valid']:
            np.testing.assert_allclose(e['actual_goal_render_state'][2:5], spec()[2:5], rtol=0,atol=1e-9)
            assert not e['penetrations']
    e = next(e for e in entries if e['valid'])
    again = m.render_goal_view(env, e['requested_state'])
    assert again['rgb_sha256'] == e['rgb_sha256']


def test_state_drift_at_goal_render_rejected(env, monkeypatch):
    u = env.unwrapped
    original = u._set_state
    def changed(value):
        original(value)
        u.block.position = (u.block.position.x + 1, u.block.position.y)
    monkeypatch.setattr(u, '_set_state', changed)
    result = m.render_goal_view(env, spec())
    assert not result['valid']
    assert 'goal_object_drift_during_reset' in result['invalid_reasons']


def test_capture_exception_restores_renderer(env, monkeypatch):
    original = env.unwrapped.render
    def fail(*args,**kwargs):
        raise RuntimeError('fixture reset failure')
    monkeypatch.setattr(env, 'reset', fail)
    with pytest.raises(RuntimeError):
        m.render_goal_view(env, spec())
    assert env.unwrapped.render == original
    assert 'render' not in vars(env.unwrapped)


@pytest.mark.parametrize('value', [[0]*6,[0]*6+[float('nan')],[0]*5+[1,0]])
def test_bad_task_specs(value):
    with pytest.raises(ValueError):
        m.goal_state(value)


def test_diagnose_preserves_all_branches_and_exact_min():
    catalogs = {str(i):dict(vectors=[[0.,0.],[1.,1.]]) for i in range(4)}
    physical = dict(status='diagnostic_completed',gates={'complete':True},rows=[])
    control = dict(status='calibration_completed',gates={'complete':True},rows=[])
    for i in range(4):
        probes=[dict(vector=[.2,.2],state=[0.]*7,status='completed') for _ in range(5)]
        physical['rows'].append(dict(root_sha256=str(i),contexts=[dict(candidate_id='selected',probes=probes)]))
        control['rows'].append(dict(root_sha256=str(i),branches=[dict(candidate_id='aborted',
            final_vector=[.2,.2],status='observation_unavailable',pose={'pose_cost':3.})]))
    snapshot=deepcopy((physical,control))
    result=m.diagnose(catalogs,physical,control)
    assert snapshot==(physical,control)
    assert len(result['control_diagnostics'])==4
    for row in result['control_diagnostics']:
        assert row['branches'][0]['status']=='observation_unavailable'
        assert row['branches'][0]['goal_set_cost']==pytest.approx(.04)
    control['rows'].pop()
    with pytest.raises(ValueError):
        m.diagnose(catalogs,physical,control)


def test_exclusive_write(tmp_path):
    path=tmp_path/'pusht_simulator_goal_catalog_20261027.json'
    m.write_new(path,{'fixture':True})
    with pytest.raises(FileExistsError):
        m.write_new(path,{'fixture':False})


def test_common_ids_exclude_original_goals_and_keep_coverage():
    sim={str(i):dict(ids=['agent_16_16','agent_29_16'],vectors=[[0.,0.],[1.,1.]]) for i in range(4)}
    clean={str(i):dict(ids=['source_goal_0','agent_16_16','agent_42_16'],
                       vectors=[[9.,9.],[2.,2.],[3.,3.]]) for i in range(4)}
    s,c,coverage=m.common_id_catalogs(sim,clean)
    for i in range(4):
        assert s[str(i)]['ids']==c[str(i)]['ids']==['agent_16_16']
        assert s[str(i)]['vectors']==[[0.,0.]] and c[str(i)]['vectors']==[[2.,2.]]
        assert coverage[i]['clean_original_goals_excluded']==['source_goal_0']
        assert coverage[i]['simulator_only_ids']==['agent_29_16']
        assert coverage[i]['clean_only_ids']==['agent_42_16']
    clean['0']['ids'][1]='agent_42_42'
    with pytest.raises(ValueError,match='empty common'):
        m.common_id_catalogs(sim,clean)
