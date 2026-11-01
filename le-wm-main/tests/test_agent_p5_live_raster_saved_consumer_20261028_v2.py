"""RAM receipts only; goal-constructor RNG seam, no native environment."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from test_agent_p5_live_raster_intervention_20261028_v1 import ram
from experiments.spectral_reachability import agent_p5_live_raster_saved_consumer_20261028_v2 as m


@pytest.fixture
def delayed_goal(ram,monkeypatch):
    cls=ram[2]()[0];original=cls.__init__;rng=np.random.get_state()
    def ctor(self,role):
        original(self,role)
        if role=='goal':np.random.randint(0,25536)  # original seed(None) operation
    monkeypatch.setattr(cls,'__init__',ctor)
    try:yield ram
    finally:np.random.set_state(rng)


def execute(ram):return m.renderer.execute(ram[0],'fixture-prepare','/ram-raster',ram[2])


def consume():return m.consume('/unused','fixture-prepare','/ram-raster/report.json',m.sha('/ram-raster/report.json'))


def test_original_v1_rejects_seam_v2_preserves_full_diff(delayed_goal):
    r=execute(delayed_goal);assert not r['errors']
    with pytest.raises(ValueError,match='color only invariant'):
        m.renderer.consume('/unused','fixture-prepare','/ram-raster/report.json',m.sha('/ram-raster/report.json'))
    c=consume()
    assert c['status']=='strict_live_raster_complete_v2'
    assert len(c['encoder_inputs'])==10 and len(c['source_comparisons'])==65
    assert len(c['baseline_to_current_anchor'])==5
    for x in c['baseline_to_current_anchor'][:4]:
        assert x['exact_differences']
        assert all(d['pointer'].startswith('/global_rng/numpy/') for d in x['exact_differences'])
        assert x['baseline_snapshot_digest']!=x['current_anchor_digest']
        assert x['constructor_rng']['full_state_exact']
    assert c['baseline_to_current_anchor'][4]['exact_differences']==[]
    assert not c['old_renderer_release'] and c['historical_full_public_equivalence'] is None


def test_word_no_interphase_draw_is_rejected(ram):
    assert not execute(ram)['errors']
    with pytest.raises(ValueError,match='ctor randint'):consume()


def rehash(memory,r,path,value):
    b=m.canonical(value)+b'\n';memory['/ram-raster/'+path]=b
    rec=r['artifacts'][path];rec.update(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
    # All receipt references share objects when created by execute. Serialization
    # below records the modified bytes consistently; non-SHA semantic gates tested.


@pytest.mark.parametrize('target',['public_state','topology','goal_color'])
def test_crossbaseline_nonrng_change_rejected(delayed_goal,target):
    r=execute(delayed_goal);memory=delayed_goal[1]
    cell=r['rows'][0]['cells'][0];path=cell['anchor']['path'];anchor=m.read('/ram-raster/'+path)
    if target=='public_state':anchor[target]['frame']+=1
    elif target=='topology':anchor[target][0]+=1
    else:anchor[target]['object_id']+=1
    rehash(memory,r,path,anchor);memory['/ram-raster/report.json']=m.canonical(r)+b'\n'
    with pytest.raises(ValueError,match='baseline/current'):consume()


@pytest.mark.parametrize('shot_name',['baseline','visible','visible_restored','white','white_restored'])
def test_every_shot_full_rng_before_after_stays_strict(delayed_goal,shot_name):
    r=execute(delayed_goal);memory=delayed_goal[1]
    shot=next(s for s in r['rows'][0]['shots'] if s['shot']==shot_name)
    path=shot['files']['after']['path'];after=m.read('/ram-raster/'+path)
    after['global_rng']['numpy'][2]+=1
    rehash(memory,r,path,after);memory['/ram-raster/report.json']=m.canonical(r)+b'\n'
    with pytest.raises(ValueError,match='before/after'):consume()


def test_both_color_anchors_must_be_full_equal(delayed_goal):
    r=execute(delayed_goal);memory=delayed_goal[1]
    path=r['rows'][0]['cells'][1]['anchor']['path'];anchor=m.read('/ram-raster/'+path)
    anchor['global_rng']['numpy'][2]+=1
    rehash(memory,r,path,anchor);memory['/ram-raster/report.json']=m.canonical(r)+b'\n'
    with pytest.raises(ValueError,match='intervention anchor'):consume()


def test_restoration_keeps_current_rng_not_baseline_rng(delayed_goal):
    r=execute(delayed_goal);memory=delayed_goal[1];row=r['rows'][0]
    baseline=m.read('/ram-raster/'+row['shots'][0]['files']['after']['path'])
    path=row['cells'][0]['restoration']['path'];restored=m.read('/ram-raster/'+path)
    restored['snapshot']['global_rng']=baseline['global_rng']
    rehash(memory,r,path,restored);memory['/ram-raster/report.json']=m.canonical(r)+b'\n'
    with pytest.raises(ValueError,match='restoration receipt'):consume()


@pytest.mark.parametrize('mutation',['missing_row','missing_cell','png','command','native_count','old_failure'])
def test_inherited_checks_not_relaxed(delayed_goal,mutation):
    r=execute(delayed_goal);memory=delayed_goal[1]
    if mutation=='missing_row':r['rows'].pop()
    elif mutation=='missing_cell':r['rows'][0]['cells'].pop()
    elif mutation=='native_count':r['meter']['entries']['space_step']-=1
    elif mutation=='old_failure':r['old_failure']['sha256']='0'*64
    else:memory['/ram-raster/xy/'+('white.png' if mutation=='png' else 'command_11.json')]+=b' '
    memory['/ram-raster/report.json']=m.canonical(r)+b'\n'
    with pytest.raises((ValueError,IndexError)):consume()


def test_anchor_schema_no_unknown_or_missing():
    a=dict(public_state={},global_rng={'python':[],'numpy':[1]},topology=[],goal_color={})
    b=deepcopy(a)
    assert m.validate_anchor(a,b,'goal')['exact_differences']==[]
    b['other']=0
    with pytest.raises(ValueError):m.validate_anchor(a,b,'goal')


def serialize(rng):
    s=rng.get_state()
    return [s[0],dict(dtype=s[1].dtype.str,shape=list(s[1].shape),values=s[1].tolist()),int(s[2]),int(s[3]),float(s[4])]


@pytest.mark.parametrize('mutation',['words','position','gaussian_flag','gaussian_cache','two_draws','float_draw','none'])
def test_full_rng_not_just_cursor(mutation):
    rng=np.random.RandomState(124);before=serialize(rng);draw=int(rng.randint(0,25536));after=serialize(rng)
    proof=m.constructor_rng_receipt(before,after)
    assert proof['draw']==draw and proof['full_state_exact']
    if mutation=='words':after[1]['values'][417]^=1
    elif mutation=='position':after[2]+=1
    elif mutation=='gaussian_flag':after[3]=1-after[3]
    elif mutation=='gaussian_cache':after[4]=.125
    elif mutation=='two_draws':rng.randint(0,25536);after=serialize(rng)
    elif mutation=='float_draw':rng.set_state(('MT19937',np.array(before[1]['values'],np.uint32),before[2],before[3],before[4]));rng.random();after=serialize(rng)
    else:after=before
    with pytest.raises(ValueError):m.constructor_rng_receipt(before,after)


def test_local_randomstate_does_not_mutate_process_global_rng():
    before=np.random.get_state();rng=np.random.RandomState(77);a=serialize(rng);rng.randint(0,25536)
    m.constructor_rng_receipt(a,serialize(rng));after=np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]


def test_no_run_or_physics_api():
    assert not hasattr(m,'run') and not hasattr(m,'execute')
    assert m.EXPECTED['space_step']==605
