"""Prospective goal-only exact-fill support. No feature scoring or production release."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

TAG='agent_p5_goal_object_interior_contract_20261028_v1'
ROOT=Path(__file__).resolve().parents[2]
GOAL=ROOT/'results/p5_native_goal_scene_20261027_v1_scene_main_v1'
GOAL_REPORT_SHA='eb3776d2d0e6851eec0e40b56c0a9bed3e33bdebdddca8d1fbe3d2ef866e76e2'
RGB_SHA='15ceb8699ec97cefff3fe72d831428dc5c7f1ee0172a25229fdec05e9a94354f'
PUBLIC_SHA='915e3415b56d65be217d16b6ac97bc7ee3922a03d08a80fc2dcc7d8eb61d466c'
ENV=ROOT/'external/jepa-wms/evals/simu_env_planning/envs/pusht_env/pusht_env.py'
ENV_SHA='6ba34790b6c2ce017a76322c2f1c205ef97d4d3b08a620f844a57075c608a723'
SDK=ROOT/'.jepa-physics-site/pymunk/space_debug_draw_options.py'
SDK_SHA='0693f0fb6460fba13e5cc94463108ee83bbd64ce01e115433eb974abe4a89c6a'
COLORS=ROOT/'.venv/lib/python3.12/site-packages/pygame/colordict.py'
COLORS_SHA='c52fa6c13c5ab6855adc7a30a69e77a2d44d677a7e6f5333642865bf2d1cba27'
CAP,FLOOR=2*1024**2,200*1024**2


def require(ok,msg):
    if not ok:raise ValueError(msg)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(path):
    path=Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)),'regular nonsymlink file')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path,expected):
    require(sha(path)==expected and Path(path).stat().st_size<=CAP,'bounded explicit SHA JSON')
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'duplicate JSON key');out[k]=v
        return out
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,parse_constant=lambda x:require(False,'nonfinite JSON'))


def protocol():
    return dict(schema=TAG,status='prospective_preparation_only',allowed_inputs=['original goal RGB','goal-scene shape color/type/role metadata','sealed native renderer/debug color source'],
        forbidden_inputs=['candidate RGB','candidate features','candidate native state','coverage','reward','candidate ranking'],
        pixel_rule='exact RGB equality to native polygon interior light_color(shape.color).as_int()[:3]',
        patch_rule='row-major 16x16 grid; support iff ALL 196 pixels in fixed 14x14 cell match; no any-pixel fallback',
        empty_rule='retain all256 cells and empty support, block a future support-mean metric; no threshold/erosion/center/nearest fallback',
        patch_denominator=256,mask_meaning='observed exact-fill-color support, not certified geometric interior or full object/overlay segmentation',
        weights=None,normalization=None,Q=None,fitting=False,threshold_tuning=False,
        production_allowed=False,scientific_admission=False,C2_established=False,
        old_mask_refusal_unchanged=True,full_overlay_identified=False,geometric_interior_certified=None,
        neural_calls=0,physics=0,AD=0,output_cap_bytes=CAP,free_floor_bytes=FLOOR)


def rgba(value):
    require(isinstance(value,(list,tuple)) and len(value)==4 and
            all(type(x) is int and 0<=x<=255 for x in value),'exact integer RGBA')
    return list(value)


def interior_rgba(value):
    """Literal native light_color then SDK as_int math; no env/pygame/Pymunk import."""
    values=rgba(value)
    light=np.minimum(1.2*np.float32(values),np.float32([255]))
    return [round(x) for x in light]


def collision_palette(source):
    """Read the one bound SDK literal assignment, not a runtime or default guess."""
    tree=ast.parse(source);values=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='collision_point_color' for t in node.targets):
            require(isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
                    and node.value.func.id=='SpaceDebugColor' and not node.value.keywords,'fixed SDK collision color expression')
            values.append(rgba([ast.literal_eval(x) for x in node.value.args]))
    require(len(values)==1,'unique explicit collision point color')
    return values[0]


def named_colors(source):
    tree=ast.parse(source);found=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='THECOLORS' for t in node.targets):
            found.append(ast.literal_eval(node.value))
    require(len(found)==1,'one literal native named-color table')
    return {k:rgba(found[0][k]) for k in ('lightgreen','lightslategray','white')}


def palette(shapes,collision_color,goal_overlay_rgba):
    require(isinstance(shapes,list) and shapes,'goal scene shape inventory required')
    blocks=[s for s in shapes if s['body']=='block']
    require(len(blocks)==2 and all(s['type']=='Poly' for s in blocks),'fixed original T two-polygons only')
    base=rgba(blocks[0]['color'])
    require(base[3]==255,'opaque native body fill only')
    require(all(rgba(s['color'])==base for s in blocks),'uniform goal-body color, no guessed palette')
    fill=interior_rgba(base)
    nuisance=[dict(role='background',color=[255,255,255,255]),
              dict(role='declared_native_goal_overlay_named_color',color=rgba(goal_overlay_rgba)),
              dict(role='collision_points',color=rgba(collision_color))]
    for s in shapes:
        if s['body']=='block':continue
        require(s['body'] in ('agent','static') and s['type'] in ('Circle','Segment'),'no unmodeled native primitive')
        c=rgba(s['color']);nuisance.append(dict(role=s['body']+'_'+s['type'],color=c))
        if s['type']=='Circle':nuisance.append(dict(role='agent_circle_light_fill',color=interior_rgba(c)))
    aliases=[x for x in nuisance if x['color'][:3]==fill[:3]]
    return dict(shape_base_rgba=base,polygon_interior_rgba=fill,polygon_edge_rgba=base,
        known_other_primitive_colors=nuisance,exact_primitive_color_aliases=aliases,
        resize_mixture_aliases_identified=False,
        caveat='Distinct known primitive palettes do not prove no bilinear-resize color coincidences.')


def support(goal_rgb,colors):
    """No features, native outcomes, configurable threshold or candidate state accepted."""
    require(goal_rgb.shape==(224,224,3) and goal_rgb.dtype==np.uint8,'original RGB224 uint8 only')
    fill=rgba(colors['polygon_interior_rgba'])
    require(not colors['exact_primitive_color_aliases'],'blocked: fill color shared with another known primitive')
    mask=np.all(goal_rgb==np.array(fill[:3],np.uint8),axis=2)
    counts=mask.reshape(16,14,16,14).sum(axis=(1,3)).reshape(-1)
    full=counts==196
    count=int(np.count_nonzero(full))
    return dict(status='operational_support_defined_not_geometric_certification' if count else 'operational_pixel_support_defined_but_full_patch_support_empty',
        pixel_support=dict(shape=[224,224],dtype='bool',bitorder='little',order='C row-major',
            packed_base64=base64.b64encode(np.packbits(mask.reshape(-1),bitorder='little').tobytes()).decode(),
            unpacked_uint8_sha256=hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest(),count=int(mask.sum())),
        all256_pixel_counts=counts.tolist(),all256_full_patch_support=full.tolist(),full_patch_count=count,
        pixel_rule_color=fill[:3],patch_rule_all_pixels=196,
        geometric_interior_certified=None,overlay_mask=None,weights=None,weighted_Q=None,
        production_allowed=False,metric_release=False)


def prepare():
    """Goal-only saved feasibility; deliberately does not read v4/candidate reports or features."""
    bindings={}
    def bind(path,h):
        require(sha(path)==h,'source/goal byte identity');bindings[str(path)]=h
    bind(ENV,ENV_SHA);bind(SDK,SDK_SHA);bind(COLORS,COLORS_SHA);bind(GOAL/'report.json',GOAL_REPORT_SHA)
    report=read_json(GOAL/'report.json',GOAL_REPORT_SHA)
    require(report['schema']=='p5_native_goal_scene_20261027_v1' and
            report['status']=='native_goal_scene_recorded' and report['errors']==[],'original independent goal source complete')
    ctor=report['protocol']['constructor']
    require(ctor['shape']=='T' and ctor['color']=='LightSlateGray' and ctor['with_target'] is True and
            ctor['render_action'] is False and ctor['render_size']==224,'fixed goal renderer settings')
    record=report['assets']['record'];require(record['frame']==0,'goal scene reset asset, not any candidate endpoint')
    refs={}
    for name,expected in (('rgb.npy',RGB_SHA),('full_public_state.json',PUBLIC_SHA)):
        rec=record['files'][name];path=GOAL/'frame_00'/name
        require(rec['sha256']==expected and path.stat().st_size==rec['bytes'],'explicit original goal capture')
        bind(path,expected);refs[name]=dict(path=str(path),**rec)
    public=read_json(refs['full_public_state.json']['path'],PUBLIC_SHA)
    # Only shape role/type/color fields enter palette; no goal/body pose, coverage, native metric or geometry witness.
    shapes=[{k:s[k] for k in ('body','type','color')} for s in public['shapes']]
    names=named_colors(COLORS.read_text())
    colors=palette(shapes,collision_palette(SDK.read_text()),names['lightgreen'])
    require(colors['shape_base_rgba']==names['lightslategray'],'saved body palette agrees with declared named-color semantics')
    colors['named_color_provenance_scope']='current read-only bound text table, not certification of historical compiled pygame loaded bytes'
    rgb=np.load(refs['rgb.npy']['path'],allow_pickle=False,max_header_size=1024)
    candidate=support(rgb,colors)
    for path in (Path(__file__),ROOT/'tests'/('test_'+TAG+'.py'),ROOT/'innerdocs'/(TAG+'.md')):bind(path,sha(path))
    require(all(sha(p)==h for p,h in bindings.items()),'final stable goal-only input bytes')
    return dict(schema=TAG+'.prepare',status='goal_only_prospective_contract_not_production',protocol=protocol(),
        goal_refs=refs,bindings=bindings,palette=colors,support=candidate,
        causes_and_limits=dict(raw_shape_color_not_interior_fill=True,goal_drawn_before_body=True,
            interior_fill_operator='native float32 light_color then SDK round, not truncation',
            source_pixel_geometric_identity=None,resize_mixture_identity=None,occluded_overlay_recovery=False),
        next_authority_required='MAIN must approve a separate metric/normalization/empty-support contract; no current candidate scoring authorized',
        production_allowed=False,old_mask_refusal_unchanged=True,scientific_admission=False)


def verify(path,expected):
    saved=read_json(path,expected)
    require(canonical(saved)==canonical(prepare()),'complete prospective goal-only preparation recomputation')
    return dict(status='prospective_goal_only_contract_verified_not_released',production_allowed=False)


def write_new(path,value):
    path=Path(path).absolute();data=canonical(value)+b'\n'
    require(path.parent==ROOT/'results' and path.name.startswith(TAG+'_') and not path.exists() and
            not any(p.is_symlink() for p in (path,*path.parents)),'exclusive new own planning artifact')
    require(len(data)<=CAP and shutil.disk_usage(path.parent).free>=FLOOR+len(data),'2MiB cap/200MiB floor')
    with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','verify'));p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--manifest-sha256');a=p.parse_args(argv)
    if a.command=='prepare':result=prepare();write_new(a.manifest,result)
    else:
        require(a.manifest_sha256,'actual prospective manifest SHA required');result=verify(a.manifest,a.manifest_sha256)
    print(json.dumps(dict(status=result['status'],production_allowed=False)))
    return 0


if __name__=='__main__':raise SystemExit(main())
