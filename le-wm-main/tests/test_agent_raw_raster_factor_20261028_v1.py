"""Model/environment-free tests. No production raster analysis is run here."""
from copy import deepcopy
from functools import lru_cache
import sys
from pathlib import Path
from types import SimpleNamespace
import math

import numpy as np
import pytest

from experiments.spectral_reachability import agent_raw_raster_factor_20261028_v1 as m


def test_source_extracted_circle_uses_python_round_and_two_fill_disks():
    draw = m.draw_api(m.UTILS.read_text())
    x = draw.circle([10.5, 11.5], 15, [65, 105, 225])
    assert x['commands'] == [
        dict(op='circle', color=[65, 105, 225, 255], center=[10, 12], radius=15, width=0),
        dict(op='circle', color=[78, 126, 255, 255], center=[10, 12], radius=11, width=0)]


def test_polygon_records_every_edge_and_integer_outline_primitive():
    draw = m.draw_api(m.UTILS.read_text())
    x = draw.polygon([[0.1, 0.2], [10.1, 0.2], [10.1, 10.2], [0.1, 10.2]], 0, [119,136,153])
    assert x['integer_points'] == [[0,0],[10,0],[10,10],[0,10]]
    assert len(x['commands']) == 17  # initial polygon, 4*(line+quad+2caps)
    assert x['commands'][0]['points'][-1] == [0,0]
    assert x['commands'][1]['width'] == 4  # source overrides radius=2


def token(x=10., y=20.):
    return dict(status='known', bodies=dict(agent=dict(
        shapes=[m.draw_api(m.UTILS.read_text()).circle([x,y],15,[65,105,225])]),
        block=dict(shapes=[])), scene_constants=dict(render_size=224))


def test_float_geometry_can_change_without_integer_drawing_change():
    r=m.factor(token(10.1),token(10.2),True)
    assert r['classification']=='same_draw_inputs_sameRGB'
    assert r['float_geometry_changed'] and not r['integer_draw_inputs_changed']
    assert r['rounding_erased_shape_change']
    assert r['full_scene_sufficient_cause_certified'] is False


@pytest.mark.parametrize('same,expected',[(True,'changed_draw_inputs_sameRGB'),(False,'changedRGB')])
def test_changed_draw_inputs_do_not_imply_changed_rgb(same,expected):
    assert m.factor(token(10),token(11),same)['classification']==expected


def test_same_reconstructed_inputs_changed_rgb_is_retained_as_unexplained():
    r=m.factor(token(10.1),token(10.2),False)
    assert r['classification']=='changedRGB' and r['same_draw_inputs_changedRGB']
    assert not r['full_scene_sufficient_cause_certified']


def test_unknown_geometry_retains_pair_and_does_not_fake_tokens():
    r=m.factor(dict(status='unknown',reason='unsupported shape'),token(),True)
    assert r['classification']=='unknown_draw_inputs_sameRGB'
    assert r['integer_draw_inputs_changed'] is None


def test_cold_import_has_no_torch_gym_pygame_or_pymunk():
    assert not any(k in sys.modules for k in ('torch','gymnasium','pygame','pymunk','stable_worldmodel'))


def test_schedule_is_all80_in_original_minus_plus_order():
    s=m.schedule()
    assert len(s)==80 and len({(x['slot'],x['h'],x['coord']) for x in s})==80
    assert s[0]==dict(slot=0,root_slot=0,condition='off',h=.01,coord=0,minus_index=1,plus_index=2)
    assert s[-1]==dict(slot=3,root_slot=1,condition='on',h=.001,coord=9,minus_index=162,plus_index=163)


def test_draw_source_change_fails_before_ast_execution():
    with pytest.raises(ValueError,match='renderer source'):
        m.draw_api(m.UTILS.read_text().replace('radius = 2','radius = 3'))


class FakeBody:
    """RAM geometry test double only; no Chipmunk/native physics imports."""
    KINEMATIC=1
    def __init__(self,*args,**kwargs):
        self.position=[0.,0.];self.angle=0.;self.center_of_gravity=np.zeros(2)
    def local_to_world(self,v):
        x,y=v;c,s=math.cos(self.angle),math.sin(self.angle)
        return np.array([c*x-s*y+self.position[0],s*x+c*y+self.position[1]])


class FakePoly:
    def __init__(self,body,vertices):
        self.body=body;self.vertices=vertices;self.radius=0.
        self.center_of_gravity=np.mean(vertices,axis=0)
    def get_vertices(self):return self.vertices


class FakeCircle:
    def __init__(self,body,radius):self.body=body;self.radius=radius;self.offset=(0.,0.)


def fake_pm():
    return SimpleNamespace(Body=FakeBody,Poly=FakePoly,Circle=FakeCircle,
        ShapeFilter=type('Filter',(),{'ALL_MASKS':staticmethod(lambda:2**32-1),
            '__init__':lambda self,**kw:None}),
        moment_for_poly=lambda *a,**kw:1.,moment_for_box=lambda *a,**kw:1.)


def test_geometry_dispatch_comes_from_actual_source_and_forced_scale_is_not_reset_scale():
    api=m.geometry_api(m.ENV.read_text(),fake_pm())
    result,meta=api.body(dict(shape=2,start_position=[0.,0.],angle=0.,scale=40,color=[119,136,153]))
    assert meta['shape_name']=='T' and meta['requested_scale']==40
    assert len(result[1])==2
    assert result[1][0].get_vertices()==[(-60.,30),(60.,30),(60.,0),(-60.,0)]
    # Distinct source shape, not a universally assumed T.
    result,meta=api.body(dict(shape=4,start_position=[0.,0.],angle=0.,scale=40,color=[1,2,3]))
    assert meta['shape_name']=='square' and len(result[1])==1


def test_unknown_geometry_is_explicit_and_original_source_change_rejected():
    api=m.geometry_api(m.ENV.read_text(),fake_pm())
    result,meta=api.body(dict(shape=100))
    assert result is None and meta['status']=='unknown'
    with pytest.raises(ValueError,match='geometry source'):
        m.geometry_api(m.ENV.read_text().replace('scale = 30','scale = 40'),fake_pm())


@lru_cache(maxsize=1)
def saved():
    return {k:m.read(m.ROOT/'results'/n,h) for k,(n,h) in m.PINS.items()}


def test_actual_saved_164_abi_whole_roster_step10_and_fixed80():
    d=saved()
    m.validate_rows(*(d[k] for k in ('physical_prepare','physical_report','physical_receipt','observability')))


@pytest.mark.parametrize('field',['index','task_id','commands','snapshot'])
def test_actual_shape_roster_tampering_is_not_silently_dropped(field):
    d=saved();p=d['physical_prepare'];r=d['physical_report'];c=d['physical_receipt'];a=d['observability']
    rr=dict(r,rows=list(r['rows']));rr['rows'][-1]=deepcopy(rr['rows'][-1])
    row=rr['rows'][-1]
    if field=='index':row['index']=162
    elif field=='task_id':row['task_id']='wrong'
    elif field=='commands':row['plan']['commands'][9][1]+=.001
    else:row['checkpoints'][-1]['snapshot']['bodies']['block']['unwrapped_angle']+=2*math.pi
    cc=dict(c,rows=rr['rows'])  # even a self-consistent report/receipt mutation must reject
    with pytest.raises(ValueError):m.validate_rows(p,rr,cc,a)


def test_all164_ram_geometry_has_all_source_polygons_and_unwrapped_poses():
    d=saved();p=d['physical_prepare'];r=d['physical_report']
    draw=m.draw_api(m.UTILS.read_text());geometry=m.geometry_api(m.ENV.read_text(),fake_pm())
    tokens=[]
    for i,row in enumerate(r['rows']):
        snap=row['checkpoints'][-1]['snapshot'];root=p['source_roots'][row['root_slot']]
        t=m.reconstruct(root['renderer'],snap,root['goal_pose'],row['condition'],draw,geometry)
        assert t['status']=='known'
        assert t['bodies']['agent']['shapes'][0]['radius']==15
        assert len(t['bodies']['block']['shapes'])==2
        assert sum(len(s['world']) for s in t['bodies']['block']['shapes'])==8
        assert t['bodies']['block']['unwrapped_angle']==snap['bodies']['block']['unwrapped_angle']
        assert bool(t['scene_constants']['goal_commands'])==(row['condition']=='on')
        tokens.append(t)
    # Test fixture math is NOT a native reconstruction result/scientific count.
    out=[m.factor(tokens[s['minus_index']],tokens[s['plus_index']],
        a['label']=='unobservable_RGB') for s,a in zip(m.schedule(),d['observability']['rows'],strict=True)]
    assert len(out)==80 and all(x['classification'] in m.LABELS for x in out)


def test_actual_terminal_event_projection_and_all492_saved_pixels():
    d=saved();r=d['physical_report'];p=d['physical_prepare']
    output=m.ROOT/'results'/Path(m.PINS['physical_report'][0]).parent
    events=m.terminal_events(output/'journal.jsonl',r,p)
    assert len(events)==164
    for es in events:
        pixels=[m.png(output/e['receipt']['png'],e['receipt']) for e in es]
        assert np.array_equal(pixels[0],pixels[2])


def test_snapshot_wrapped_angle_cannot_substitute_unwrapped_angle():
    row=saved()['physical_report']['rows'][0]
    s=deepcopy(row['checkpoints'][-1]['snapshot']);s['bodies']['block']['unwrapped_angle']=s['state'][4]
    with pytest.raises(ValueError,match='body float64'):m.check_snapshot(s)


@pytest.mark.parametrize('bit',['file_sha256','pixel_sha256'])
def test_actual_png_tamper_rejected(bit):
    row=saved()['physical_report']['rows'][0];f=deepcopy(row['checkpoints'][-1]);f[bit]='0'*64
    output=m.ROOT/'results'/Path(m.PINS['physical_report'][0]).parent
    with pytest.raises(ValueError):m.png(output/f['png'],f)


def test_changed_goal_or_renderer_cannot_be_named_rounding():
    a=token();b=deepcopy(a);b['scene_constants']['render_size']=512
    with pytest.raises(ValueError,match='same fixed renderer'):m.factor(a,b,True)


def test_same_agent_round_center_does_not_hide_changed_block_vertices():
    a=token(10.1);b=token(10.2);d=m.draw_api(m.UTILS.read_text())
    a['bodies']['block']['shapes']=[d.polygon([[0,0],[10,0],[10,10],[0,10]],0,[119,136,153])]
    b['bodies']['block']['shapes']=[d.polygon([[1,0],[11,0],[11,10],[1,10]],0,[119,136,153])]
    r=m.factor(a,b,False)
    assert not r['components']['agent']['integer_commands_changed']
    assert r['components']['block']['integer_commands_changed']
    assert r['components']['block']['integer_vertex_or_center_changes']==4
    assert r['classification']=='changedRGB' and not r['rounding_erased_shape_change']


def test_unknown_draw_inputs_changed_rgb_is_a_retained_category():
    assert m.factor(dict(status='unknown',reason='unsealed geometry'),token(),False)[
        'classification']=='unknown_draw_inputs_changedRGB'


def test_velocity_only_changes_are_not_misnamed_rounding():
    r=m.factor(token(),token(),True)
    assert not r['float_geometry_changed'] and not r['rounding_erased_shape_change']


def test_native_geometry_origin_restoration_error_fails_not_rounded_away():
    d=saved();r=d['physical_report']['rows'][0];root=d['physical_prepare']['source_roots'][0]
    pm=fake_pm()
    class WrongOrigin(FakeBody):
        def local_to_world(self,v):return super().local_to_world(v)+np.array([1e-6,0.])
    pm.Body=WrongOrigin
    with pytest.raises(ValueError,match='exact exported local0'):
        m.reconstruct(root['renderer'],r['checkpoints'][-1]['snapshot'],root['goal_pose'],
            r['condition'],m.draw_api(m.UTILS.read_text()),m.geometry_api(m.ENV.read_text(),pm))


def test_main_authorization_precedes_saved_load_or_native_import():
    with pytest.raises(ValueError,match='authorization'):m.build('no')
