"""Pure RAM geometry and array-publication ABI, no native imports/experiments."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_contact_source_contract_20261028_v1 as m


@pytest.fixture
def owners():
    return m.load(m.BASE),m.load(m.GEOMETRY)


def source_shape():
    return m.read_json(m.ROOT/'results/p5_jepa_factual_panel_20261027_v3/factual_main_v1/seed_2026102702/a/frame_00/full_public_state.json')


def test_candidate_uses_actual_native_shape_only(owners):
    base,geo = owners
    saved = source_shape()
    candidate = m.geometry_candidate(geo,saved)
    assert candidate['initial_circle_polygon_gap']==1.
    assert candidate['first_target_static_geometry_gap']==-9.
    assert candidate['status']=='static_contact_candidate_not_observed'
    assert candidate['actual_contacts'] is None
    # Old agent/body placement is not reused or used to select the new pose.
    changed = copy.deepcopy(saved)
    changed['bodies']['block']['position']=[999.,999.]
    changed['bodies']['block']['angle']=999.
    changed['bodies']['agent']['position']=[888.,888.]
    assert m.geometry_candidate(geo,changed)==candidate
    assert base.CTOR['reset_to_state'] is None


def test_native_constructor_delta_is_explicit_ood(owners):
    base,_ = owners
    old = copy.deepcopy(base.CTOR)
    new = m.constructor(base.CTOR)
    assert base.CTOR==old
    assert [k for k in old if old[k]!=new[k]]==['reset_to_state']
    assert new['reset_to_state']==[180.,271.,256.,256.,0.,0.,0.]
    assert new['relative'] and new['action_scale']==100
    assert new['with_target'] and new['with_velocity']
    assert new['legacy'] is False


def test_fixed_raw64_plan_no_selection_or_native_calls(owners):
    base,geo = owners
    q = m.reference_queues()
    assert tuple(q)==m.ARMS
    assert all(a.dtype==np.float64 and a.shape==(10,2) for a in q.values())
    assert np.array_equal(q['push_x'],np.tile([.1,0.],(10,1)))
    assert np.array_equal(q['raw_zero'],np.zeros((10,2)))
    candidate = m.geometry_candidate(geo,source_shape())
    c = m.contract(base.CTOR,candidate,dict(ACTION_MEAN=[-.0087,.0068],ACTION_STD=[.2019,.2002]),'a'*64)
    assert len(c['generator_private']['plans'])==2
    assert c['generator_private']['actual_attempts']==0
    assert c['generator_private']['expected_native_call_entries']['space_step']==402
    assert c['generator_private']['expected_native_call_entries']['native_render']==42
    assert not c['source_witness_to_planner'] and not c['physics_executed']
    assert not c['action_witness_established'] and not c['scientific_admission']
    assert c['task_success'] is None
    assert c['deployment']==m.blind_schema()
    assert 'OOD' in c['calibration']


@pytest.mark.parametrize('change',['radius','filter','sensor','interior'])
def test_geometry_mutations_fail(owners,change):
    _,geo = owners
    saved = source_shape()
    agent = next(s for s in saved['shapes'] if s['body']=='agent')
    if change=='radius':
        agent['radius']=16.
    elif change=='filter':
        agent['filter']['mask']=0
    elif change=='sensor':
        agent['sensor']=True
    else:
        poly = np.array([[0.,0.],[4.,0.],[4.,4.],[0.,4.]])
        with pytest.raises(ValueError,match='interior'):
            geo.polygon_gap([2.,2.],poly,15.)
        return
    with pytest.raises(ValueError):
        m.geometry_candidate(geo,saved)


def test_blind_schema_contains_no_privileged_plan():
    schema = m.blind_schema()
    assert not schema['ready']
    assert set(schema['files'])=={'context_rgb_00','context_rgb_05','context_rgb_10',
                                 'context_proprio','history_raw_u','history_normalized','goal_rgb'}
    assert not any(k in schema for k in ('seed','STATE','constructor','reference_suffix','source_witness','geometry'))


@pytest.fixture
def publication(tmp_path):
    public = tmp_path/'deployment_public'
    public.mkdir()
    stats = dict(ACTION_MEAN=[-.0087,.0068],ACTION_STD=[.2019,.2002])
    producer = m.load(m.PRODUCER)
    for key,(dtype,shape) in m.blind_schema()['files'].items():
        a = np.zeros(shape,np.dtype(dtype))
        if key=='history_normalized':
            a = producer.normalize(np.zeros((2,5,2),np.float64),stats)
        with (public/(key+'.npy')).open('wb') as f:
            np.save(f,a,allow_pickle=False)
    return public,stats,m.sha(public/'goal_rgb.npy')


def test_project_allowlist_never_releases(publication):
    public,stats,goal = publication
    result = m.project_blind(public,stats,goal)
    assert result['status']=='unreleased_numeric_projection'
    assert result['ready'] is False
    assert not result['source_witness_included']
    assert set(result['inputs'])==set(m.blind_schema()['files'])
    assert all(Path(r['path']).parent==public for r in result['inputs'].values())


@pytest.mark.parametrize('name',['reference_suffix.json','witness.json','source_contract.json','contacts.npy','subdir'])
def test_witness_leakage_rejected(publication,name):
    public,stats,goal = publication
    (public/name).write_text('{}')
    with pytest.raises(ValueError,match='allowlist'):
        m.project_blind(public,stats,goal)


@pytest.mark.parametrize('change',['history_raw','history_normalized','rgb_dtype','proprio_shape','object_array','goal_hash'])
def test_blind_array_mutations_fail(publication,change):
    public,stats,goal = publication
    key = dict(history_raw='history_raw_u',history_normalized='history_normalized',
               rgb_dtype='context_rgb_00',proprio_shape='context_proprio',object_array='context_proprio').get(change)
    if change=='goal_hash':
        goal='f'*64
    else:
        p = public/(key+'.npy')
        a = np.load(p,allow_pickle=False)
        if change=='history_raw':
            a[0,0]=.1
        elif change=='history_normalized':
            a[:]=0
        elif change=='rgb_dtype':
            a=a.astype(np.float32)
        elif change=='proprio_shape':
            a=a[:2]
        else:
            a=np.array([dict(witness=True)],object)
        with p.open('wb') as f:
            np.save(f,a,allow_pickle=True)
    with pytest.raises(ValueError):
        m.project_blind(public,stats,goal)


def test_missing_publication_no_zero_fill(publication):
    public,stats,goal = publication
    missing = public/'context_proprio.npy'
    # Move fixture instead of deleting; preserves negative evidence.
    missing.rename(public.parent/'missing_proprio.npy')
    with pytest.raises(ValueError,match='allowlist'):
        m.project_blind(public,stats,goal)


def test_sources_pinned_read_only_and_imports_safe():
    for rel,h in m.PINS.items():
        assert m.sha(m.ROOT/rel)==h
    before = set(sys.modules)
    m.load(m.BASE)
    m.load(m.GEOMETRY)
    m.load(m.PRODUCER)
    assert not {x for x in set(sys.modules)-before if x.split('.')[0] in ('torch','pymunk','pygame','gym')}


def test_saved_prepare_missing_authority_fails_before_import(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m,'load',lambda name:pytest.fail('premature import'))
    with pytest.raises(ValueError,match='nonsymlink'):
        m.prepare()


def test_json_output_exclusive_and_cli_verify(monkeypatch,tmp_path):
    value = dict(schema=m.TAG,status='RAM_fixture_not_native',ready=False)
    monkeypatch.setattr(m,'prepare',lambda:value)
    p = tmp_path/'contract.json'
    assert m.main(['prepare','--contract',str(p)])==0
    assert m.main(['verify','--contract',str(p),'--contract-sha256',m.sha(p)])==0
    with pytest.raises(ValueError,match='new contract'):
        m.main(['prepare','--contract',str(p)])
    with pytest.raises(ValueError,match='FILE SHA'):
        m.main(['verify','--contract',str(p),'--contract-sha256','0'*64])
    p.write_text('{"tampered":true}')
    with pytest.raises(ValueError,match='drift'):
        m.main(['verify','--contract',str(p),'--contract-sha256',m.sha(p)])


def test_json_duplicates_nan_symlinks(tmp_path):
    p = tmp_path/'bad.json'
    p.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError,match='duplicate'):
        m.read_json(p)
    p.write_text('{"x":NaN}')
    with pytest.raises(ValueError,match='nonfinite'):
        m.read_json(p)
    link = tmp_path/'link'
    link.symlink_to(p)
    with pytest.raises(ValueError,match='nonsymlink'):
        m.sha(link)
