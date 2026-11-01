"""RAM-only trace fixtures. No native environment, Body, Space or physics import."""
from collections import Counter
from types import SimpleNamespace as NS
import copy
import sys

import pytest
from shapely.geometry import Polygon, MultiPolygon

from experiments.spectral_reachability import agent_p5_native_geometry_observer_20261028_v1 as m


class MemoryStore:
    def __init__(self): self.saved={};self.events=[]
    def json(self,name,value):
        self.saved[name]=copy.deepcopy(value)
    def event(self,value): self.events.append(copy.deepcopy(value))


class FixtureBody:
    position=(0.,0.)
    def local_to_world(self,v):
        # Fixture lookup, not a substitute for native transformations.
        return self.lookup[tuple(v)]


class FixturePoly:
    def __init__(self,vertices,body): self.vertices=vertices;self.body=body
    def get_vertices(self): return self.vertices


def fixture_conversion(body,shapes):
    geoms=[]
    for shape in shapes:
        verts=[body.local_to_world(v) for v in shape.get_vertices()]
        verts += [verts[0]]
        geoms.append(Polygon(verts))
    geom=MultiPolygon(geoms)
    return geom


def setup_fixture(reverse=False):
    block=FixtureBody();goal=FixtureBody()
    a=[(0.,0.),(2.,0.),(2.,1.),(0.,1.)]
    b=[(0.,2.),(1.,2.),(1.,3.),(0.,3.)]
    block.lookup={v:v for v in a+b}
    goal.lookup={v:(v[0]+.25,v[1]+.25) for v in a+b}
    shapes=[FixturePoly(v,block) for v in (a,b)]
    if reverse: shapes.reverse()
    observer=m.Observer.__new__(m.Observer)
    observer.geometry_code=fixture_conversion.__code__
    observer.vertices_code=FixturePoly.get_vertices.__code__
    observer.world_code=FixtureBody.local_to_world.__code__
    observer.codes={observer.geometry_code:'geometry'}
    observer.context=dict(state_id='xy',operation='step',frame=13)
    observer.entries=Counter();observer.returns=Counter();observer.errors=[]
    observer.store=MemoryStore();observer.active={};observer.observations=[]
    observer.envs={'xy':NS(block=block)}
    observer.base=NS(BODY_FIELDS=('position',),numeric_json=lambda x:list(x),
        public_state=lambda env:dict(env=dict(success_threshold=.95)))
    return observer,block,goal,shapes


def trace(observer,body,shapes):
    old=sys.getprofile()
    try:
        sys.setprofile(observer.profile)
        return fixture_conversion(body,shapes)
    finally: sys.setprofile(old)


def pair():
    o,b,g,s=setup_fixture()
    trace(o,g,s);trace(o,b,s)
    return o.observations


@pytest.mark.parametrize('reverse',[False,True])
def test_actual_returns_vertices_order_wkb(reverse):
    o,b,g,s=setup_fixture(reverse)
    returned=trace(o,g,s)
    assert not o.errors
    r=o.observations[0]
    assert r['wkb_hex']==returned.wkb_hex
    assert r['shapes']==[id(x) for x in s]
    assert r['shape_body_ids']==[id(b)]*2
    assert r['local_vertices']==[[list(v) for v in x.vertices] for x in s]
    assert r['world_vertices']==[list(map(list,x.exterior.coords)) for x in returned.geoms]
    assert r['body_role']=='goal'
    assert len(r['world_calls'])==8
    assert o.store.saved['geometry/return_0.json']==r
    assert o.entries==o.returns=={'geometry':1}


def test_only_target_frame_is_captured():
    o,b,g,s=setup_fixture();o.context['frame']=12
    trace(o,b,s)
    assert not o.observations and o.entries['geometry']==1


def test_no_observer_function_or_callback_replacement():
    original=fixture_conversion.__code__;o,b,g,s=setup_fixture()
    trace(o,b,s)
    assert fixture_conversion.__code__ is original
    assert sys.getprofile() is None


def test_malformed_trace_persisted_before_guard():
    o,b,g,s=setup_fixture()
    s[0].body=g
    trace(o,g,s)
    assert len(o.observations)==1 and len(o.store.saved)==1
    assert 'ownership' in o.errors[0]


@pytest.mark.parametrize('field',['world_vertices','local_vertices','world_calls','shapes','shape_body_ids','body_id'])
def test_trace_mutations_rejected(field):
    r=copy.deepcopy(pair()[0])
    if field=='body_id': r[field]+=1
    elif field in ('shapes','shape_body_ids'): r[field][0]+=1
    elif field=='world_calls': r[field][0]['world'][0]+=.01
    elif field=='world_vertices': r[field][0][0][0]+=.01
    else: r[field][0][0][0]+=.01
    if field=='shapes':
        # An arbitrary pointer change cannot be authenticated saved-only;
        # duplicates can, and are rejected.
        r[field][0]=r[field][1]
    with pytest.raises(ValueError): m.validate_observation(r)


def test_enumeration_uses_observed_components_not_transform_reconstruction():
    r=pair()
    result=m.enumerate_geometry(r,.95,dict(reward=0.,info=dict(final_coverage=0.)))
    v=result['rows'][0]['values']
    tr=dict(reward=v['reward'],info=dict(final_coverage=v['coverage']))
    result=m.enumerate_geometry(r,.95,tr)
    assert result['actual_order_matches']
    assert len(result['rows'])==4
    assert result['historical_membership'] is None
    assert result['renderer_release'] is False
    assert result['rows'][0]['bits']['coverage']==m.bits(v['coverage'])


def test_one_ulp_not_allclose():
    import numpy as np
    r=pair();base=m.enumerate_geometry(r,.95,dict(reward=0.,info=dict(final_coverage=0.)))['rows'][0]['values']
    result=m.enumerate_geometry(r,.95,dict(reward=base['reward'],info=dict(final_coverage=np.nextafter(base['coverage'],float('inf')))))
    assert not result['actual_order_matches']


def test_signed_zero_bits_differ(): assert m.bits(-0.)!=m.bits(0.)


@pytest.mark.parametrize('mutation',['missing','role','world'])
def test_saved_geometry_join_rejects(mutation):
    r=pair()
    if mutation=='missing': r.pop()
    if mutation=='role': r[0]['body_role']='block'
    if mutation=='world': r[0]['world_vertices'][0][0][0]+=1
    with pytest.raises(ValueError): m.enumerate_geometry(r,.95,dict(reward=0.,info=dict(final_coverage=0.)))


def test_fixed_budget_not_original805_or_fullsuffix():
    p=m.protocol()
    assert p['physics_steps']==4+43*10==434
    assert sum(map(len,p['frames'].values()))==47
    assert p['frames']['xy']==list(range(14))
    assert all(p['frames'][a]==list(range(11)) for a in m.ARMS[1:])
    assert m.EXPECTED['geometry']==86
    assert p['renderer_release'] is False


def test_run_requires_main_authorization_before_runtime():
    with pytest.raises(ValueError,match='authorization'): m.run(None,None,None,False)


def test_profile_io_error_is_recorded_not_raised_into_native():
    o,b,g,s=setup_fixture()
    def broken(*args): raise OSError('fixture disk failure')
    o.store.json=broken
    trace(o,b,s)
    assert any('fixture disk failure' in e for e in o.errors)


def test_empty_exceptional_geometry_is_observer_failure():
    o,b,g,s=setup_fixture();g.lookup={}
    with pytest.raises(KeyError): trace(o,g,s)
    assert o.errors


def test_inherited_helper_and_budget_untouched():
    assert m.h.EXPECTED['space_step']==805
    assert m.EXPECTED['space_step']==434
