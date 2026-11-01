"""RAM goal-palette/support fixtures only; no physics/model import or production scoring."""
from copy import deepcopy
import base64
import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from experiments.spectral_reachability import agent_p5_goal_object_interior_contract_20261028_v1 as m


def shapes():
    return [dict(body='block',type='Poly',color=[119,136,153,255]) for _ in range(2)]+[
        dict(body='agent',type='Circle',color=[65,105,225,255]),
        dict(body='static',type='Segment',color=[211,211,211,255])]


def colors():return m.palette(shapes(),[231,76,60,255],[144,238,144,255])


def unpack(result):
    p=result['pixel_support']
    return np.unpackbits(np.frombuffer(base64.b64decode(p['packed_base64']),np.uint8),bitorder=p['bitorder']).reshape(224,224).astype(bool)


def test_actual_native_interior_not_raw_shape_color_or_truncation():
    assert m.interior_rgba([119,136,153,255])==[143,163,184,255]
    assert m.interior_rgba([65,105,225,255])==[78,126,255,255]
    assert m.interior_rgba([211,211,211,255])==[253,253,253,255]
    assert colors()['polygon_edge_rgba']==[119,136,153,255]


def test_source_color_parsers_are_literal_only():
    c=m.named_colors('THECOLORS={"lightgreen":(144,238,144,255),"lightslategray":(119,136,153,255),"white":(255,255,255,255)}')
    assert c['lightgreen']==[144,238,144,255]
    assert m.collision_palette('self.collision_point_color = SpaceDebugColor(231,76,60,255)')==[231,76,60,255]
    with pytest.raises(ValueError):m.collision_palette('self.collision_point_color = make_dynamic_color()')
    with pytest.raises(ValueError):m.collision_palette('a=1')


def test_contract_has_no_feature_or_outcome_input():
    assert list(inspect.signature(m.support).parameters)==['goal_rgb','colors']
    p=m.protocol()
    assert p['production_allowed'] is p['scientific_admission'] is False
    assert p['Q'] is p['weights'] is p['normalization'] is None
    assert p['old_mask_refusal_unchanged'] is True
    assert p['neural_calls']==p['physics']==p['AD']==0


def test_fixed_all196_rule_no_any_pixel_fallback():
    rgb=np.full((224,224,3),255,np.uint8)
    fill=colors()['polygon_interior_rgba'][:3]
    rgb[14:28,28:42]=fill
    rgb[0,0]=fill
    result=m.support(rgb,colors())
    assert result['full_patch_count']==1
    assert result['all256_full_patch_support'][18] is True
    assert result['all256_pixel_counts'][0]==1
    assert result['all256_full_patch_support'][0] is False
    mask=unpack(result)
    assert int(mask.sum())==197 and mask[0,0]
    rgb[14,28]=[119,136,153]
    reduced=m.support(rgb,colors())
    assert reduced['full_patch_count']==0
    assert reduced['all256_pixel_counts'][18]==195
    assert len(reduced['all256_full_patch_support'])==256
    assert reduced['weights'] is reduced['weighted_Q'] is None


def test_empty_support_is_not_zero_Q_or_relaxed_mask():
    result=m.support(np.full((224,224,3),255,np.uint8),colors())
    assert result['pixel_support']['count']==result['full_patch_count']==0
    assert result['metric_release'] is result['production_allowed'] is False
    assert result['overlay_mask'] is result['geometric_interior_certified'] is None


def test_raw_edge_and_interpolated_near_match_are_excluded():
    rgb=np.zeros((224,224,3),np.uint8)
    rgb[0,0]=[119,136,153]
    rgb[0,1]=[143,163,183]
    rgb[0,2]=[143,163,184]
    result=m.support(rgb,colors())
    assert result['pixel_support']['count']==1
    assert unpack(result)[0,2] and not unpack(result)[0,1]


def test_palette_collision_is_blocked_not_ignored():
    c=colors();c['exact_primitive_color_aliases']=[dict(role='agent',color=c['polygon_interior_rgba'])]
    with pytest.raises(ValueError):m.support(np.full((224,224,3),255,np.uint8),c)
    s=shapes();s[-1]['color']=c['polygon_interior_rgba']
    assert m.palette(s,[231,76,60,255],[144,238,144,255])['exact_primitive_color_aliases']


@pytest.mark.parametrize('kind',['different_block_color','missing_block','wrong_type','unknown_other','transparent'])
def test_unknown_palette_semantics_rejected(kind):
    s=shapes()
    if kind=='different_block_color':s[0]['color']=[0,0,0,255]
    elif kind=='missing_block':s.pop(0)
    elif kind=='wrong_type':s[0]['type']='Circle'
    elif kind=='unknown_other':s[-1]['type']='Unknown'
    else:s[0]['color'][3]=0
    with pytest.raises(ValueError):m.palette(s,[231,76,60,255],[144,238,144,255])


@pytest.mark.parametrize('kind',['shape','float_rgb','wrong_rgba','float_rgba'])
def test_exact_input_abi(kind):
    rgb=np.zeros((224,224,3),np.uint8);c=colors()
    if kind=='shape':rgb=rgb[:200]
    elif kind=='float_rgb':rgb=rgb.astype(np.float32)
    elif kind=='wrong_rgba':c['polygon_interior_rgba']=[143,163,184]
    else:c['polygon_interior_rgba']=[143.,163.,184.,255.]
    with pytest.raises(ValueError):m.support(rgb,c)


def test_prepare_runtime_has_no_hidden_candidate_or_import_path():
    src=inspect.getsource(m.prepare)
    tree=ast.parse(src)
    indexed_keys={n.slice.value for n in ast.walk(tree) if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant)}
    assert not indexed_keys.intersection({'token_artifacts','features','coverage','reward','candidate','state','goal_pose','position'})
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Name) and n.id in ('v4','owner','load_runtime','torch','pymunk','pygame') for n in ast.walk(tree))


def test_saved_contract_verify_recomputes_and_rejects_changed_rule(tmp_path,monkeypatch):
    original=dict(protocol=m.protocol(),support=m.support(np.zeros((224,224,3),np.uint8),colors()))
    path=tmp_path/'prepare.json';path.write_bytes(m.canonical(original))
    monkeypatch.setattr(m,'prepare',lambda:deepcopy(original))
    assert m.verify(path,m.sha(path))['production_allowed'] is False
    altered=deepcopy(original);altered['support']['patch_rule_all_pixels']=1
    path.write_bytes(m.canonical(altered))
    with pytest.raises(ValueError):m.verify(path,m.sha(path))


@pytest.mark.parametrize('data',[b'{"x":1,"x":2}',b'{"x":NaN}'])
def test_saved_json_strict(tmp_path,data):
    p=tmp_path/'bad.json';p.write_bytes(data)
    with pytest.raises(ValueError):m.read_json(p,m.sha(p))


def test_planning_artifact_never_overwritten_or_shared(tmp_path,monkeypatch):
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    p=tmp_path/'results'/(m.TAG+'_fixture.json');m.write_new(p,{'production_allowed':False})
    with pytest.raises(ValueError):m.write_new(p,{})
    with pytest.raises(ValueError):m.write_new(tmp_path/'results'/'MAIN.json',{})
