import ast
from copy import deepcopy

import numpy as np
import pymunk
import pytest

from experiments.spectral_reachability import p4_saved_contact_geometry_20261027_v1 as m


@pytest.fixture
def geometry():
    return m.contract(m.saved.Inputs())


def state(x,y,angle=0,bx=200,by=200):
    return [x,y,bx,by,angle,0,0]


def test_official_contract(geometry):
    assert geometry['pusher_radius'] == 15
    assert geometry['collision_slop'] == .1
    assert geometry['polygons'][0][0] == [-60,30]


@pytest.mark.parametrize('point,expected',[
    ((200,185),0), ((200,184),1), ((200,180),5),
    ((200,200),-15), ((250,250),5), ((200,330),-5),
    ((200,335),0), ((200,340),5), ((280,215),5),
])
def test_non_center_circle_surface_clearance(geometry,point,expected):
    result = m.metrics(state(*point),geometry)
    assert result['signed_surface_clearance'] == pytest.approx(expected)


def test_union_internal_seam_not_false_boundary(geometry):
    boundary = m.world_boundary(state(200,230),geometry)
    assert m.signed_polygon_distance([200,230],boundary) == pytest.approx(-15)


@pytest.mark.parametrize('angle',[0,.3,1.57,3.14,5.4])
def test_official_transform_and_distance(geometry,angle):
    s = state(400,400,angle)
    body = pymunk.Body(1,1)
    parts = [pymunk.Poly(body,v) for v in geometry['polygons']]
    body.center_of_gravity = (parts[0].center_of_gravity+parts[1].center_of_gravity)/2
    body.angle = angle
    body.position = (s[2],s[3])
    for part in parts:
        part.cache_bb()  # Geometry cache only: no Space or physics step.
    expected = min(p.point_query(tuple(s[:2])).distance for p in parts)-15
    assert m.metrics(s,geometry)['signed_surface_clearance'] == pytest.approx(expected)
    np.testing.assert_allclose(m.world_boundary(s,geometry)[0],body.local_to_world((-60,0)))


def test_rigid_transform_invariance(geometry):
    a = state(280,215)
    theta = .71
    rot = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    shift = np.array([13,27])
    agent,block = rot@a[:2]+shift,rot@a[2:4]+shift
    b = [*agent,*block,theta,0,0]
    assert m.metrics(a,geometry)['signed_surface_clearance'] == pytest.approx(m.metrics(b,geometry)['signed_surface_clearance'])


def test_displacements_and_flags(geometry):
    previous = state(200,170)
    item = m.label(state(200,185),previous,geometry)
    assert item['agent_delta'] == [0,15]
    assert item['flags']['approaching'] is True
    assert item['near_contact'] is True
    assert item['flags']['block_sub_slop_motion'] is True
    assert item['flags']['near_contact_with_block_motion'] is False
    assert item['dynamic_contact'] is None
    assert item['force_contact'] is None
    assert item['flags']['task_closure'] is None


def test_unknown_initial_not_zero_motion(geometry):
    result = m.label(state(200,185),None,geometry)
    assert result['agent_displacement'] is None
    assert result['flags']['approaching'] is None


def test_rotation_counts_object_motion(geometry):
    result = m.label(state(200,185,.01),state(200,185),geometry)
    assert result['block_displacement'] == 0
    assert result['block_max_vertex_displacement'] > .1
    assert result['flags']['block_sub_slop_motion'] is False


@pytest.mark.parametrize('bad',[[0]*5,[0]*6,[0]*8,[0,0,0,0,float('nan'),0,0]])
def test_invalid_state_rejected(geometry,bad):
    with pytest.raises(ValueError):
        m.metrics(bad,geometry)


def test_changed_official_source_fails_closed(monkeypatch,tmp_path):
    path = tmp_path/'env.py'
    path.write_text('class PushT: pass')
    monkeypatch.setattr(m,'ENV',path)
    with pytest.raises(ValueError,match='unsupported official geometry'):
        m.contract(m.saved.Inputs())


def test_no_simulator_or_neural_calls():
    source = m.Path(m.__file__).read_text()
    tree = ast.parse(source)
    forbidden = {'Space','step','reset','predict','make','eval_state'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            name = node.func.attr if isinstance(node.func,ast.Attribute) else getattr(node.func,'id','')
            assert name not in forbidden
    assert 'import torch' not in source
    assert 'import gym' not in source


def test_actual_saved_censuses_and_determinism(geometry):
    inputs = m.saved.Inputs()
    a,_ = m.p4_rows(inputs,m.P4,geometry)
    b,_ = m.capability_rows(inputs,m.CAP,geometry)
    assert sum(len(r['steps']) for r in a) == 80
    assert len(b) == 123
    assert sum(len(r['steps'])-1 for r in b) == 1230
    assert sum(s['rgb']['file_verified'] for r in a for s in r['steps']) == 16
    assert all(r['steps'][0]['agent_displacement'] is None for r in a)
    selected = [r for r in b if r['method'].startswith('selected_')]
    assert len(selected) == 41
    for row in selected:
        for item in row['steps'][1:]:
            repeat = m.label(item['state'],row['steps'][item['primitive_step']-1]['state'],geometry)
            assert repeat['signed_surface_clearance'] == item['signed_surface_clearance']
            assert repeat['flags'] == item['flags']
    assert inputs.stable()


def test_census_corruption_rejected(geometry):
    class Broken(m.saved.Inputs):
        def json(self,path):
            result = super().json(path)
            if m.Path(path) == m.P4:
                result = deepcopy(result)
                result['rows'].pop()
            return result
    with pytest.raises(ValueError,match='census'):
        m.p4_rows(Broken(),m.P4,geometry)


def test_exclusive_output(tmp_path):
    output = tmp_path/(m.TAG+'_existing')
    output.mkdir()
    with pytest.raises(FileExistsError):
        m.run(output)


def test_failclosed_saved_output(tmp_path,monkeypatch):
    def bad(_):
        raise ValueError('unsupported geometry test')
    monkeypatch.setattr(m,'contract',bad)
    result = m.run(tmp_path/(m.TAG+'_failure'))
    assert result['status'] == 'integrity_fail'
    assert result['P1_admission'] is result['P4_admission'] is False
    assert result['new_physics_steps'] == result['new_neural_calls'] == 0
