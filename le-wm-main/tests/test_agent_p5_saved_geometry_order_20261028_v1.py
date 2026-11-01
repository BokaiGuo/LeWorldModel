"""Saved-array/geometry tests only. No environment or native physics imports."""
from copy import deepcopy
import math
import sys

import pytest

from experiments.spectral_reachability import agent_p5_saved_geometry_order_20261028_v1 as m


def snapshot():
    return dict(bodies={'block':dict(position=[10.,20.],angle=0.,center_of_gravity=[0.,45.])},
        shapes=[dict(body='block',type='Poly',radius=0.,vertices=[[0.,0.],[1.,0.],[1.,1.],[0.,1.]]),
                dict(body='block',type='Poly',radius=0.,vertices=[[1.,0.],[2.,0.],[2.,1.],[1.,1.]])],
        env=dict(goal_pose={'dtype':'<f8','shape':[3],'values':[10.,20.,0.]},
                 success_threshold=.95,coverage_arr=[0.]*12+[1.]))


def sc(angle): return math.sin(angle),math.cos(angle)


def test_four_full_order_cells_no_filter():
    r = m.permutations(snapshot(),sc)
    assert [(x['goal_order'],x['block_order']) for x in r['permutations']]==[
        ([0,1],[0,1]),([0,1],[1,0]),([1,0],[0,1]),([1,0],[1,0])]
    assert all(x['values']['coverage']==1. and x['values']['reward']==1. for x in r['permutations'])


def test_do_not_union_or_repair_touching_native_components():
    r = m.permutations(snapshot(),sc)
    assert all(not x['native_input_validity']['goal'] for x in r['permutations'])
    assert all(x['goal_area']==2. for x in r['permutations'])


def test_translation_is_local_origin_not_cog_world():
    r = m.permutations(snapshot(),sc)
    assert r['block_world_vertices'][0][0]==[10.,20.]
    # subtracting snapshot CoG again would incorrectly produce [10,-25].
    assert r['block_world_vertices'][0][0] != [10.,-25.]


def test_scalar_operation_order_and_bits():
    points,t = m.transform([[3.,4.]],[10.,20.],0.,lambda a:(0.,1.))
    assert points==[[13.,24.]]
    assert t['values']==[1.,0.,-0.,1.,10.,20.]
    assert t['bits'][2]==m.bits(-0.) and m.bits(0.)!=m.bits(-0.)


def test_bit_match_never_becomes_native_equivalence():
    rows=m.permutations(snapshot(),sc)['permutations']
    result=m.compare(rows,dict(values=rows[0]['values']))
    assert result['conditional_exact_member'] and len(result['conditional_exact_matches'])==4
    assert result['legal_native_permutation_member'] is None
    assert not result['native_transform_certified']


def test_one_ulp_is_not_accepted():
    rows=m.permutations(snapshot(),sc)['permutations']
    values=deepcopy(rows[0]['values']); values['coverage']=math.nextafter(1.,0.)
    assert not m.compare(rows,dict(values=values))['conditional_exact_member']


@pytest.mark.parametrize('which',('shapes','length','radius','angle','translation','history','goal'))
def test_bad_snapshot_rejected(which):
    s=snapshot()
    if which=='shapes': s['shapes'].pop()
    if which=='length': s['shapes'][0]['vertices'][0].append(1.)
    if which=='radius': s['shapes'][0]['radius']=1.
    if which=='angle': s['bodies']['block']['angle']=float('nan')
    if which=='translation': s['bodies']['block']['position'][0]=0.
    if which=='history': s['env']['coverage_arr'].pop()
    if which=='goal': s['env']['goal_pose']['shape']=[7]
    with pytest.raises(ValueError): m.permutations(s,sc)


def test_recorded_requires_exact_transition_and_snapshot_join():
    s=snapshot(); t=dict(kind='step',done=False,reward=1.,info=dict(final_coverage=1.,max_coverage=1.))
    assert m.recorded(s,t)['values']['coverage']==1.
    t['info']['final_coverage']=math.nextafter(1.,0.)
    with pytest.raises(ValueError): m.recorded(s,t)


def test_current_sincos_does_not_claim_history_attestation():
    fn,receipt=m.sincos_runtime()
    assert fn(0.)==(0.,1.) and receipt['historic_loaded_bytes_attested'] is False
    assert len(receipt['sha256'])==64


def test_never_imports_env_or_constructs_native_object():
    assert not any(k=='pymunk' or k.startswith('pymunk.') for k in sys.modules)
    assert m.SYMBOLS[0]=='cpBodyLocalToWorld'
    assert m.FAILED_SHA=='1b02ed4a2ca136b4dc7f8549073ff030620df8587d654eb2a07ca86036cad0c6'


def test_only_frame13_history_max_recomputed():
    s=snapshot(); s['env']['coverage_arr'][3]=2.
    rows=m.permutations(s,sc)['permutations']
    assert all(r['values']['max_coverage']==2. for r in rows)
    assert all(r['values']['final_coverage']==1. for r in rows)
