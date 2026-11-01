"""Nonneural geometry/rigid-motion fixtures and actual saved-trace regression."""
import ast
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import p4_feedback_motion_alignment_20261027_v1 as m


@pytest.fixture
def g():
    return m.geometry.contract(m.saved.Inputs())


def state(x=200.,y=200.,angle=0.,ax=200.,ay=185.):
    return [ax,ay,x,y,angle,0.,0.]


@pytest.mark.parametrize('d,e,cosine,reduction,projection',[
    ([2,0],[10,0],1,2,2),([-2,0],[10,0],-1,-2,-2),
    ([3,0],[1,0],1,-1,3),([0,2],[10,0],0,10-np.sqrt(104),0),
    ([0,0],[10,0],None,0,0),([2,0],[0,0],None,-2,None),
    ([0,0],[0,0],None,0,None),
])
def test_projection_cosine_and_exact_reduction(d,e,cosine,reduction,projection):
    r = m.direction(d,e)
    assert r['cosine'] == cosine
    assert r['exact_goal_distance_reduction'] == pytest.approx(reduction)
    assert r['signed_goal_axis_displacement'] == projection


@pytest.mark.parametrize('d,e',[([1],[1,2]),([float('nan')],[0]),([[1,2]],[[3,4]]),([],[])])
def test_invalid_vectors_rejected(d,e):
    with pytest.raises(ValueError): m.direction(d,e)


def test_official_origin_and_nonzero_cog(g):
    f = m.body_frame(state(angle=np.pi/2),g)
    np.testing.assert_allclose(f['cog_local'],[0,45],atol=1e-12)
    np.testing.assert_allclose(f['origin'],[200,200],atol=1e-12)
    np.testing.assert_allclose(f['cog'],[155,200],atol=1e-12)
    np.testing.assert_allclose(f['vertices'],m.geometry.world_boundary(state(angle=np.pi/2),g))


def test_translation_toward_fixed_goal(g):
    r = m.motion(state(),state(x=202,ax=202),state(x=210),g)
    assert r['origin_vs_fixed_goal']['exact_goal_distance_reduction'] == 2
    assert r['origin_vs_fixed_goal']['cosine'] == 1
    assert r['pusher_vs_object_origin_motion']['cosine'] == 1
    assert r['corresponding_vertices']['exact_goal_rms_reduction'] == pytest.approx(2)
    assert r['original_pose']['cost_change'] == -2
    assert r['angular']['principal_displacement'] == 0
    a = r['contact_geometry']['before_anchor']
    np.testing.assert_allclose(a['material_displacement_world'],[2,0])
    np.testing.assert_allclose(a['pusher_minus_material_displacement_world'],[0,0])
    assert a['pusher_inward_displacement'] == pytest.approx(0)
    assert a['force_direction'] is None
    assert a['contact_point_measured'] is False


def test_stationary_object_alignment_not_fabricated(g):
    r = m.motion(state(),state(ax=205),state(x=210),g)
    assert r['origin_vs_fixed_goal']['cosine'] is None
    assert r['pusher_vs_object_origin_motion']['cosine'] is None
    assert r['contact_geometry']['before_anchor']['pusher_vs_material_motion']['cosine'] is None
    assert r['origin_vs_fixed_goal']['exact_goal_distance_reduction'] == 0
    assert 'exact_goal_distance_reduction' not in r['pusher_vs_object_origin_motion']


def test_pure_rotation_preserves_origin_but_moves_cog(g):
    r = m.motion(state(),state(angle=.1),state(angle=.2),g)
    assert r['origin_vs_fixed_goal']['displacement_norm'] == 0
    assert r['cog_vs_fixed_goal']['displacement_norm'] > 0
    assert r['angular']['absolute_error_reduction'] == pytest.approx(.1)
    assert r['original_pose']['angle_cost_change'] == pytest.approx(-2)
    assert r['corresponding_vertices']['exact_goal_rms_reduction'] > 0
    assert r['corresponding_vertices']['reconstruction_max_abs_error_origin'] < 1e-12
    assert r['corresponding_vertices']['reconstruction_max_abs_error_cog'] < 1e-12


def test_stationary_pusher_body_coordinate_change_not_displacement(g):
    r = m.motion(state(),state(angle=.1),state(angle=.2),g)
    assert r['pusher_displacement_world'] == [0,0]
    assert r['pusher_displacement_in_before_body_axes'] == [0,0]
    assert r['before']['pusher_body_xy'] != r['after']['pusher_body_xy']


def test_rotation_wrap_and_half_turn_ambiguity(g):
    r = m.motion(state(angle=2*np.pi-.05),state(angle=.05),state(angle=.2),g)
    assert r['angular']['principal_displacement'] == pytest.approx(.1)
    assert r['angular']['absolute_error_reduction'] == pytest.approx(.1)
    ambiguous = m.motion(state(),state(angle=.1),state(angle=np.pi),g)
    assert ambiguous['angular']['half_turn_goal_direction_ambiguous']


def transform(s,theta,shift):
    s = np.asarray(s).copy()
    c,t = np.cos(theta),np.sin(theta)
    rot = np.array([[c,-t],[t,c]])
    s[:2] = rot@s[:2]+shift
    s[2:4] = rot@s[2:4]+shift
    s[4] += theta
    return s


def test_global_rigid_frame_invariance(g):
    before,after,goal = state(),state(x=203,y=201,angle=.05,ax=202,ay=186),state(x=220,y=210,angle=.3)
    a = m.motion(before,after,goal,g)
    b = m.motion(*(transform(s,.7,np.array([10,17])) for s in (before,after,goal)),g)
    for field in ('origin_vs_fixed_goal','cog_vs_fixed_goal'):
        for key in ('cosine','exact_goal_distance_reduction','signed_goal_axis_displacement'):
            assert a[field][key] == pytest.approx(b[field][key],abs=1e-10)
    for key in ('cosine','dot','signed_reference_axis_displacement'):
        assert a['pusher_vs_object_origin_motion'][key] == pytest.approx(b['pusher_vs_object_origin_motion'][key],abs=1e-10)
    np.testing.assert_allclose(a['pusher_displacement_in_before_body_axes'],b['pusher_displacement_in_before_body_axes'],atol=1e-10)
    for endpoint in ('before_anchor','after_anchor'):
        for key in ('pusher_inward_displacement','pusher_tangential_displacement','pusher_relative_inward_displacement'):
            assert a['contact_geometry'][endpoint][key] == pytest.approx(b['contact_geometry'][endpoint][key],abs=1e-10)
        np.testing.assert_allclose(a['contact_geometry'][endpoint]['material_point_body'],b['contact_geometry'][endpoint]['material_point_body'],atol=1e-10)


def test_nearest_corner_normal_ambiguity_retained(g):
    r = m.motion(state(ax=270,ay=240),state(ax=271,ay=240),state(x=210),g)
    a = r['contact_geometry']['before_anchor']
    assert a['normal_nonunique_at_corner'] is True
    assert len(a['tied_edge_indices']) >= 2


def test_origin_translation_goal_is_not_cog_translation_goal(g):
    r = m.motion(state(),state(angle=.1),state(angle=.2),g)
    assert r['origin_vs_fixed_goal']['goal_error_before'] == 0
    assert r['cog_vs_fixed_goal']['goal_error_before'] > 0


def test_goal_change_does_not_relabel_contact_geometry(g):
    a = m.motion(state(),state(x=202,ax=202),state(x=210),g)
    b = m.motion(state(),state(x=202,ax=202),state(x=150,y=150,angle=1),g)
    assert a['origin_vs_fixed_goal'] != b['origin_vs_fixed_goal']
    for key in ('before_near','after_near','before_clearance','after_clearance','either_endpoint_near'):
        assert a['contact_geometry'][key] == b['contact_geometry'][key]
    for key in ('edge_index','edge_fraction','material_point_body','material_displacement_world','pusher_inward_displacement'):
        assert a['contact_geometry']['before_anchor'][key] == b['contact_geometry']['before_anchor'][key]


@pytest.mark.parametrize('bad',[[0.]*6,[0.]*8,[0.,0.,float('inf'),0.,0.,0.,0.]])
def test_invalid_goal_state_rejected(g,bad):
    with pytest.raises(ValueError): m.motion(state(),state(x=201),bad,g)


def steps(g):
    result = []
    previous = None
    for k in range(11):
        s = state(x=200+k*.2,ax=200+k*.2)
        item = m.geometry.label(s,previous,g)
        item.update(primitive_step=k,command=None if k == 0 else [0.,0.])
        result.append(item); previous = s
    return result


def test_complete_all_intervals_and_telescope(g):
    r = m.analyze_steps(steps(g),state(x=210),g)
    assert len(r['intervals']) == 10
    assert len([v for v in r['intervals'] if v['phase']=='suffix']) == 5
    assert r['summary']['origin_goal_distance_reduction'] == pytest.approx(1)
    assert r['summary']['suffix_origin_path_length'] == pytest.approx(1)
    assert r['summary']['suffix_either_endpoint_near_intervals'] == 5
    assert r['summary']['near_interval_pusher_vs_origin_cosine_defined_count'] == 5
    assert r['summary']['near_interval_pusher_vs_origin_cosine_mean'] == pytest.approx(1)
    gain = sum(v['origin_vs_fixed_goal']['exact_goal_distance_reduction'] for v in r['intervals'][5:])
    assert gain == pytest.approx(r['summary']['origin_goal_distance_reduction'])


@pytest.mark.parametrize('mutation',['missing','order','near','clearance','nonfinite'])
def test_corrupt_snapshot_or_geometry_rejected(g,mutation):
    s = steps(g)
    if mutation == 'missing': s.pop()
    elif mutation == 'order': s[2]['primitive_step'] = 1
    elif mutation == 'near': s[2]['near_contact'] = not s[2]['near_contact']
    elif mutation == 'clearance': s[2]['signed_surface_clearance'] += 1
    elif mutation == 'nonfinite': s[2]['state'][2] = float('nan')
    with pytest.raises(ValueError): m.analyze_steps(s,state(x=210),g)


def test_exclusive_output(tmp_path):
    output = tmp_path/(m.TAG+'_existing'); output.mkdir()
    with pytest.raises(FileExistsError): m.run(m.DEFAULT_RECOVERY,output)


def test_failclosed_report_retained(tmp_path,monkeypatch):
    def bad(_): raise ValueError('bad source fixture')
    monkeypatch.setattr(m,'audit',bad)
    output = tmp_path/(m.TAG+'_failure')
    r = m.run(m.DEFAULT_RECOVERY,output)
    assert r['status'] == 'integrity_fail'
    assert r['P1_admission'] is r['P4_admission'] is False
    assert r['new_physics_steps'] == r['new_neural_calls'] == 0
    assert (output/'report.json').is_file() and (output/'report.md').is_file()


def test_neural_and_simulator_imports_not_loaded():
    code = ('import sys; from experiments.spectral_reachability import p4_feedback_motion_alignment_20261027_v1; '
            'assert not any(x in sys.modules for x in ("torch","gymnasium","stable_worldmodel"))')
    subprocess.run([sys.executable,'-B','-c',code],check=True,cwd=m.REPO)
    tree = ast.parse(Path(m.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr not in ('step','reset','predict','make','Space','load_compatible_pretrained')


def test_saved_recovery_tampering_rejected(monkeypatch):
    base = m.saved.Inputs
    class Changed(base):
        def json(self,path):
            result = super().json(path)
            if Path(path) == m.DEFAULT_RECOVERY:
                result = deepcopy(result); result['rows'][0]['steps'][6]['state'][2] += 1
            return result
    monkeypatch.setattr(m.saved,'Inputs',Changed)
    with pytest.raises(ValueError,match='re-audit differs'):
        m.audit(m.DEFAULT_RECOVERY)


def test_actual_saved_six_branch_audit():
    r = m.audit(m.DEFAULT_RECOVERY)
    assert r['status'] == 'saved_motion_alignment_recorded'
    assert r['census'] == dict(branches=6,snapshot_receipts=66,primitive_intervals=60,suffix_intervals=30)
    assert r['all_inputs_unchanged']
    assert {(row['role'],row['arm']) for row in r['rows']} == {(role,arm) for role in m.recovery.physical.ROLES for arm in m.recovery.physical.ARMS}
    assert all(len(row['intervals']) == 10 for row in r['rows'])
    assert all(row['fixed_goal_state'] == r['fixed_goal_state'] for row in r['rows'])
    for row in r['rows']:
        for interval in row['intervals']:
            assert interval['contact_geometry']['actual_contact_during_interval'] is None
            assert interval['contact_geometry']['force_causality'] is None
        summary = row['summary']
        expected_cost = -summary['origin_goal_distance_reduction']-20*summary['absolute_angle_error_reduction']
        assert summary['original_pose_cost_change'] == pytest.approx(expected_cost,abs=1e-10)
