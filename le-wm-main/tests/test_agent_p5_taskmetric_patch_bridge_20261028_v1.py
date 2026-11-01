"""Pure saved/RAM tests; fixtures do not establish a task-weighted metric."""
from copy import deepcopy
from fractions import Fraction
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from experiments.spectral_reachability import agent_p5_taskmetric_patch_bridge_20261028_v1 as m


PALETTE={'visible':[144,238,144,255],'white':[255,255,255,255]}


def rational(record):return Fraction(int(record['numerator']),int(record['denominator']))


def assert_conserves(term,vector):
    assert sum((Fraction.from_float(float(v)) for v in vector),Fraction())+rational(term['exact_reduction_residual'])==Fraction.from_float(term['reference_float64'])


def fixture_data():
    rng=np.random.default_rng(20261028)
    f={k:rng.standard_normal(m.SHAPE).astype(np.float32) for k in m.v4.KEYS}
    p={w:rng.standard_normal(m.SHAPE).astype(np.float32) for w in m.WORDS}
    native=[]
    for i,w in enumerate(m.WORDS):
        pose=[250.+i,260.-i,.2*i];goal=[256.,256.,.75]
        state=[100.,200.,*np.asarray(pose,np.float32).tolist(),3.,4.]
        block=dict(position=pose[:2],angle=pose[2],velocity=[0.,0.],angular_velocity=0.)
        transition=dict(info=dict(block_pose=dict(values=pose),goal_pose=dict(values=goal)))
        native.append(dict(word=w,frame=20,task_success=None,public_state=dict(bodies=dict(block=block,agent=dict(position=[100.,200.])),env=dict(goal_pose=dict(values=goal))),
            state=state,proprio=[100.,200.,3.,4.],transition=transition,coverage=.1*i,reward=.2*i,done=False))
    scalars=[]
    for w in m.WORDS:
        goal=f['goal__native'];obs=f[w+'__frame20'];pred=p[w]
        scalar=dict(word=w)
        for name,a,b in zip(m.FIELDS,(obs,pred,pred),(goal,goal,obs),strict=True):
            scalar[name]=float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
        scalars.append(scalar)
    mask=m.mask_decision(np.full((224,224,3),255,np.uint8),PALETTE)
    return f,p,native,scalars,mask


def test_no_new_metric_or_science_gate():
    p=m.protocol()
    assert p['word_denominator']==4 and p['patch_denominator']==256 and p['pair_denominator']==12
    assert p['new_Q'] is p['goal_region_aggregate'] is p['other_region_aggregate'] is None
    assert p['predict']==p['encoder']==p['AD']==p['physics']==0
    assert p['scientific_admission'] is p['C2_established'] is False


def test_mask_refusal_is_goal_only_even_if_palette_visible():
    assert list(inspect.signature(m.mask_decision).parameters)==['goal_rgb','palette']
    rgb=np.full((224,224,3),255,np.uint8)
    rgb[14:28,28:42]=PALETTE['visible'][:3]
    result=m.mask_decision(rgb,deepcopy(PALETTE))
    assert result['exact_palette_pixel_counts']['visible']==196
    assert result['status']=='mask_refused_all256_fallback'
    assert all(result[k] is None for k in ('mask','goal_region_patches','other_region_patches','threshold','new_weighted_metric','encoder_invariance'))


def test_null_firststage_never_means_encoder_invariance():
    r=m.mask_decision(np.full((224,224,3),255,np.uint8),PALETTE)
    assert r['exact_palette_pixel_counts']['visible']==0 and r['encoder_invariance'] is None


def test_grid_is_row_major_no_best_patch_order():
    grid=m.layout()
    assert [r['index'] for r in grid]==list(range(256))
    assert grid[15]['rgb_bounds_yxyx_exclusive']==[0,210,14,224]
    assert grid[16]['rgb_bounds_yxyx_exclusive']==[14,0,28,14]
    assert grid[-1]['rgb_bounds_yxyx_exclusive']==[210,210,224,224]


def test_spatial_coordinate_and_conservation():
    a=np.zeros(m.SHAPE,np.float32);b=a.copy();a[19,:]=2.
    r=m.spatial(a,b)
    assert r['patch_mse'][19]==4.
    assert r['reference_float64']==4./256
    assert sum(v!=0 for v in r['uniform_full_mean_contribution'])==1
    assert_conserves(r,r['uniform_full_mean_contribution'])


def test_rounding_residual_is_explicit_exact_not_tolerance():
    r=m.conservation(1.,[.1]*10)
    assert rational(r['exact_reduction_residual'])!=0
    assert_conserves(r,[.1]*10)


def test_all_words_all_pairs_all_spatial_terms():
    f,p,n,s,mask=fixture_data();result=m.decompose(f,p,n,s,mask)
    assert len(result['words'])==4 and len(result['pairs'])==12
    for row in result['words']:
        assert set(row['spatial'])==set(m.METRICS)
        assert len(row['native']['native_state7'])==7
        for term in row['spatial'].values():
            assert len(term['patch_mse'])==len(term['uniform_full_mean_contribution'])==256
            assert_conserves(term,term['uniform_full_mean_contribution'])
    pairs={(r['left'],r['right']):r for r in result['pairs']}
    for (a,b),row in pairs.items():
        reverse=pairs[b,a]
        assert row['object_pose_delta']==[-v for v in reverse['object_pose_delta']]
        for key,term in row['spatial'].items():
            assert len(term['signed_patch_contribution'])==256
            assert np.array_equal(np.asarray(term['signed_patch_contribution']),-np.asarray(reverse['spatial'][key]['signed_patch_contribution']))
            assert_conserves(term,term['signed_patch_contribution'])


def test_native_outcome_cannot_select_spatial_terms_or_mask():
    f,p,n,s,mask=fixture_data();before=m.decompose(f,p,n,s,mask)
    changed=deepcopy(n)
    for i,row in enumerate(changed):row['coverage']=100.-i;row['reward']=-100.+i
    after=m.decompose(f,p,changed,s,mask)
    assert before['mask']==after['mask']
    assert before['layout']==after['layout']
    assert [w['spatial'] for w in before['words']]==[w['spatial'] for w in after['words']]
    assert [r['spatial'] for r in before['pairs']]==[r['spatial'] for r in after['pairs']]


def test_native_double_axes_not_truncated_or_float32_substituted():
    _,_,native,_,_=fixture_data();row=native[0]
    row['public_state']['bodies']['block']['position'][0]=250.0000000001
    row['transition']['info']['block_pose']['values'][0]=250.0000000001
    result=m.native_axes(row)
    assert result['object_pose_public_double'][0]!=result['native_object_pose_float32'][0]
    assert result['object_error_x_y_theta_raw'][0]==250.0000000001-256.
    assert result['agent_goal'] is result['task_success'] is None
    assert result['native_state7'][-2:]==[3.,4.]


@pytest.mark.parametrize('kind',['missing_word','wrong_order','missing_prediction','wrong_scalar','wrong_frame','truncated7','invented_mask','bad_native_pose'])
def test_fixed_join_mutations_rejected(kind):
    f,p,n,s,mask=fixture_data()
    if kind=='missing_word':f.pop(m.v4.KEYS[0])
    elif kind=='wrong_order':n.reverse()
    elif kind=='missing_prediction':p.pop(m.WORDS[-1])
    elif kind=='wrong_scalar':s[0][m.FIELDS[0]]=np.nextafter(s[0][m.FIELDS[0]],np.inf)
    elif kind=='wrong_frame':n[0]['frame']=15
    elif kind=='truncated7':n[0]['state']=n[0]['state'][:5]
    elif kind=='invented_mask':mask['mask']=[1]*256
    else:n[0]['transition']['info']['block_pose']['values']=[0.,0.,0.]
    with pytest.raises(ValueError):m.decompose(f,p,n,s,mask)


@pytest.mark.parametrize('kind',['shape','dtype','nan'])
def test_patch_abi_rejects_invalid(kind):
    a=np.zeros(m.SHAPE,np.float32);b=a.copy()
    if kind=='shape':a=a[:192]
    elif kind=='dtype':a=a.astype(np.float64)
    else:a[1,1]=np.nan
    with pytest.raises(ValueError):m.spatial(a,b)


@pytest.mark.parametrize('kind',['rgb_shape','rgb_dtype','missing_palette','float_palette'])
def test_mask_abi_no_guessed_defaults(kind):
    rgb=np.zeros((224,224,3),np.uint8);palette=deepcopy(PALETTE)
    if kind=='rgb_shape':rgb=rgb[:200]
    elif kind=='rgb_dtype':rgb=rgb.astype(np.float32)
    elif kind=='missing_palette':palette.pop('white')
    else:palette['visible'][0]=144.
    with pytest.raises(ValueError):m.mask_decision(rgb,palette)


def test_fixed_authority_no_arbitrary_dataset(monkeypatch):
    monkeypatch.setattr(m,'sha',lambda *a:m.V4_SHA)
    with pytest.raises(ValueError):m.analyze('a','0'*64,'b',m.REPORT_SHA,'c',m.STRICT_SHA)


def test_verify_recomputes_whole_report(tmp_path,monkeypatch):
    value=dict(authority={k:dict(path=k,sha256=k) for k in ('prepare','report','strict')},words=[1,2,3,4])
    path=tmp_path/'saved.json';path.write_bytes(m.canonical(value))
    monkeypatch.setattr(m,'analyze',lambda *a:deepcopy(value))
    assert m.verify(path,m.sha(path))['status']=='all_patch_saved_bridge_verified'
    mutated=deepcopy(value);mutated['words'][0]=99;path.write_bytes(m.canonical(mutated))
    with pytest.raises(ValueError):m.verify(path,m.sha(path))


@pytest.mark.parametrize('payload',[b'{"a":1,"a":2}',b'{"x":NaN}'])
def test_duplicate_or_nonfinite_saved_rejected(tmp_path,payload):
    p=tmp_path/'bad.json';p.write_bytes(payload)
    with pytest.raises(ValueError):m.read_saved(p,m.sha(p))


def test_exclusive_capped_writer(tmp_path,monkeypatch):
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    path=tmp_path/'results'/(m.TAG+'_fixture.json')
    m.write_output(path,{'fixture':'RAM'})
    with pytest.raises(ValueError):m.write_output(path,{'fixture':'overwrite'})
    with pytest.raises(ValueError):m.write_output(tmp_path/'results'/'MAIN.json',{})
    monkeypatch.setattr(m,'CAP',4)
    with pytest.raises(ValueError):m.write_output(tmp_path/'results'/(m.TAG+'_oversize.json'),{'large':'abc'})
