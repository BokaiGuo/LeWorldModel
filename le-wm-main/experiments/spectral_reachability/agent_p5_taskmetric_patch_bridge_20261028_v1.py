"""Saved-only all-patch P5 bridge. No fitted metric, selected patch, model or physics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from fractions import Fraction

import numpy as np
from experiments.spectral_reachability import agent_p5_same_time_frame20_20261028_v4 as v4

TAG='agent_p5_taskmetric_patch_bridge_20261028_v1'
ROOT=v4.ROOT
WORDS=v4.WORDS
SHAPE=(256,384)
V4_SHA='b9139d497cea2aaaaf680790d2bb4d8174c0df39c10f39d7cb2c30db6f8ac059'
PREP_SHA='88e6403b2fd828dcfe08c803cc878fc948b112b2106aa95b4d4e64acf2f1475a'
REPORT_SHA='96c369c9b51301d902a489f0063754cce63007ca98aa5227fa54fc52ffc644e5'
STRICT_SHA='542f2717e3860903d9ea87577edef2725a1fd7ae919c5aa216b45b75c0a29684'
METRICS=('observed_goal','predicted_goal','predicted_observed')
FIELDS=('full_patch_goal_mse','predicted_goal_mse','prediction_to_real_feature_mse')
CAP,FLOOR,WALL=16*1024**2,200*1024**2,120
require,sha=v4.require,v4.sha


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def read_saved(path,expected):
    require(sha(path)==expected and Path(path).stat().st_size<=CAP,'bounded actual saved JSON')
    def pairs(items):
        result={}
        for k,v in items:
            require(k not in result,'duplicate saved JSON key');result[k]=v
        return result
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,
                      parse_constant=lambda x:require(False,'nonfinite JSON'))


def protocol():
    return dict(schema=TAG,words=list(WORDS),frame=20,word_denominator=4,
        patch_denominator=256,channels=384,pair_denominator=12,unordered_pairs=6,
        metrics=list(METRICS),mask_policy='refuse_goal_foreground_partition_use_all256_unselected',
        goal_region_aggregate=None,other_region_aggregate=None,weights_fitted=False,
        thresholds_tuned=False,patch_selection=False,topk=None,new_Q=None,
        reduction='float64 squared differences; mean over 384 channels per patch; uniform contribution=patch_MSE/256',
        exact_accounting='reference float64 reduction plus exact rational reduction residual; no allclose or hidden tolerance',
        spatial_warning='token grid origin, NOT causal locality: transformer tokens have global self-attention',
        native_warning='object axes descriptive; retain native7/agent/coverage/reward; not full PushT success',
        predict=0,encoder=0,AD=0,physics=0,training=False,scientific_admission=False,
        C2_established=False,task_success=None,renderer_release=False,
        output_cap_bytes=CAP,free_floor_bytes=FLOOR,wall_seconds=WALL)


def layout():
    return [dict(index=i,row=i//16,col=i%16,
                 rgb_bounds_yxyx_exclusive=[(i//16)*14,(i%16)*14,(i//16+1)*14,(i%16+1)*14])
            for i in range(256)]


def mask_decision(goal_rgb,palette):
    """Only goal RGB and captured native palette enter this function, never candidate/labels."""
    require(goal_rgb.shape==(224,224,3) and goal_rgb.dtype==np.uint8,'original goal RGB224')
    require(set(palette)=={'visible','white'},'both actual captured goal colors')
    counts={}
    for color,rgba in palette.items():
        require(isinstance(rgba,list) and len(rgba)==4 and all(type(x) is int and 0<=x<=255 for x in rgba),'native RGBA bytes')
        counts[color]=int(np.count_nonzero(np.all(goal_rgb==np.array(rgba[:3],np.uint8),axis=2)))
    return dict(status='mask_refused_all256_fallback',palette=palette,exact_palette_pixel_counts=counts,
        threshold=None,mask=None,goal_region_patches=None,other_region_patches=None,
        goal_region_scores=None,other_region_scores=None,
        reason='Occluded goal overlay and resized boundary pixels cannot be uniquely segmented from goal RGB by palette; gray body pixels cannot be relabeled as a certified goal mask.',
        encoder_invariance=None,new_weighted_metric=None)


def fraction_record(value):
    return dict(numerator=str(value.numerator),denominator=str(value.denominator))


def conservation(reference,values):
    """Exact identity for stored IEEE numbers, not a claim of bitwise equal reduction orders."""
    require(math.isfinite(reference) and all(math.isfinite(x) for x in values),'finite spatial contributions')
    residual=Fraction.from_float(float(reference))-sum((Fraction.from_float(float(x)) for x in values),Fraction())
    return dict(reference_float64=float(reference),contribution_sum_float64=math.fsum(values),
                exact_reduction_residual=fraction_record(residual))


def spatial(a,b):
    require(a.shape==b.shape==SHAPE and a.dtype==b.dtype==np.float32 and
            np.isfinite(a).all() and np.isfinite(b).all(),'full finite float32[256,384]')
    squared=(a.astype(np.float64)-b.astype(np.float64))**2
    require(np.isfinite(squared).all(),'finite squared differences')
    patch=np.mean(squared,axis=1,dtype=np.float64)
    contribution=(patch/256.).tolist()
    full=float(np.mean(squared,dtype=np.float64))
    return dict(patch_mse=patch.tolist(),uniform_full_mean_contribution=contribution,
                **conservation(full,contribution))


def native_axes(row):
    require(row['frame']==20 and row['task_success'] is None,'same-time native row, no success reinterpretation')
    public=row['public_state'];block=public['bodies']['block'];goal=public['env']['goal_pose']['values']
    pose=[*block['position'],block['angle']]
    state=row['state'];proprio=row['proprio']
    require(len(pose)==len(goal)==3 and len(state)==7 and len(proprio)==4,'full public/object/native7 ABI')
    require(row['transition']['info']['block_pose']['values']==pose and
            row['transition']['info']['goal_pose']['values']==goal,'actual native double pose identity')
    require(all(math.isfinite(x) for x in [*pose,*goal,*state,*proprio]),'finite native axes')
    error=[pose[i]-goal[i] for i in range(3)]
    wrapped=math.atan2(math.sin(error[2]),math.cos(error[2]))
    return dict(object_pose_public_double=pose,goal_pose_public_double=goal,
        object_error_x_y_theta_raw=error,object_error_theta_wrapped=wrapped,
        object_velocity=list(block['velocity']),object_angular_velocity=block['angular_velocity'],
        native_state7=state,native_object_pose_float32=state[2:5],native_proprio4=proprio,
        agent_public=public['bodies']['agent'],agent_goal=None,
        coverage=row['coverage'],reward=row['reward'],done=row['done'],task_success=None,
        full_native_transition=row['transition'])


def decompose(features,predictions,native,scalar_rows,mask):
    require(tuple(features)==v4.KEYS and set(predictions)==set(WORDS),'fixed five features and four predictions')
    require([r['word'] for r in native]==[r['word'] for r in scalar_rows]==list(WORDS),'full word denominator/order')
    require(mask['status']=='mask_refused_all256_fallback' and mask['mask'] is None,'no data-selected mask')
    words=[];goal=features['goal__native']
    for w,n,scalar in zip(WORDS,native,scalar_rows,strict=True):
        observed=features[w+'__frame20'];predicted=predictions[w]
        terms=dict(zip(METRICS,(spatial(observed,goal),spatial(predicted,goal),spatial(predicted,observed)),strict=True))
        require(all(terms[k]['reference_float64']==scalar[field] for k,field in zip(METRICS,FIELDS,strict=True)),
                'all three original full-patch costs must reproduce exactly')
        words.append(dict(word=w,frame=20,spatial=terms,native=native_axes(n)))
    pairs=[]
    for left in words:
        for right in words:
            if left['word']==right['word']:continue
            terms={}
            for metric in METRICS:
                a,b=left['spatial'][metric],right['spatial'][metric]
                vector=(np.asarray(a['uniform_full_mean_contribution'],np.float64)-
                        np.asarray(b['uniform_full_mean_contribution'],np.float64)).tolist()
                reference=a['reference_float64']-b['reference_float64']
                terms[metric]=dict(signed_patch_contribution=vector,**conservation(reference,vector))
            a,b=left['native'],right['native']
            pairs.append(dict(left=left['word'],right=right['word'],spatial=terms,
                object_pose_delta=[x-y for x,y in zip(a['object_pose_public_double'],b['object_pose_public_double'],strict=True)],
                object_error_axis_delta=[x-y for x,y in zip(a['object_error_x_y_theta_raw'],b['object_error_x_y_theta_raw'],strict=True)],
                wrapped_object_error_delta=a['object_error_theta_wrapped']-b['object_error_theta_wrapped'],
                coverage_delta=a['coverage']-b['coverage'],reward_delta=a['reward']-b['reward']))
    return dict(words=words,pairs=pairs,layout=layout(),mask=mask)


def analyze(prepare_path,prepare_sha256,report_path,report_sha256,strict_path,strict_sha256):
    started=time.monotonic()
    require(sha(v4.__file__)==V4_SHA,'frozen v4 source, no edits')
    require((prepare_sha256,report_sha256,strict_sha256)==(PREP_SHA,REPORT_SHA,STRICT_SHA),'fixed actual v4 three external SHA')
    authority={}
    for name,path,h in (('prepare',prepare_path,prepare_sha256),('report',report_path,report_sha256),('strict',strict_path,strict_sha256)):
        require(sha(path)==h,'actual explicit authority SHA')
        authority[name]=dict(path=str(Path(path).absolute()),sha256=h)
    m=v4.read_json(prepare_path,prepare_sha256);r=v4.read_json(report_path,report_sha256)
    strict=v4.read_json(strict_path,strict_sha256)
    require(digest(strict)==digest(v4.consume(m,report_path,report_sha256)),'full authoritative saved v4 strict receipt')
    bindings=dict(m['bindings'])
    def bind(path,h):
        p=str(Path(path).absolute());require(sha(p)==h,'saved source/array SHA')
        require(p not in bindings or bindings[p]==h,'binding collision');bindings[p]=h
    for ref in authority.values():bind(ref['path'],ref['sha256'])
    # Read only bound sources as bytes, never import renderer/encoder/environment libraries.
    relative_sources={
        'native_render':'external/jepa-wms/evals/simu_env_planning/envs/pusht_env/pusht_env.py',
        'patch_embed':'external/dinov2_official_7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/layers/patch_embed.py',
        'token_order':'external/dinov2_official_7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/models/vision_transformer.py',
        'loader':'experiments/spectral_reachability/p5_jepa_official_action_fidelity_20261027_v7.py',
        'preprocessing':'external/jepa-wms/app/plan_common/datasets/transforms.py',
        'crop':'external/jepa-wms/src/datasets/utils/video/transforms.py'}
    geometry_sources={}
    for key,rel in relative_sources.items():
        path=str(ROOT/rel);require(path in bindings,'upstream sealed grid/palette/preprocessing source')
        bind(path,bindings[path]);geometry_sources[key]=dict(path=path,sha256=bindings[path])
    rgb=v4.owner.array(m['inputs'][-1]['rgb'],(224,224,3),np.uint8)
    require(m['inputs'][-1]['image_key']=='goal__native','original goal input')
    ref=m['authority']['model_prepare'];old=v4.read_json(ref['path'],ref['sha256'])
    palette={}
    for color in ('visible','white'):
        cell=next(x for x in old['inputs'] if x['state_id']=='goal' and x['raster']==color)
        record=cell['after'];bind(record['path'],record['sha256'])
        snapshot=v4.read_json(record['path'],record['sha256'])
        palette[color]=snapshot['goal_color']['rgba']
        require(v4.owner.array(cell['rgb'],(224,224,3),np.uint8).tobytes()==rgb.tobytes(),'goal null pixel first-stage retained')
    # Freeze fallback from goal-only inputs before any candidate spatial/native decomposition.
    mask=mask_decision(rgb,palette)
    features={}
    for key in v4.KEYS:
        rec=r['token_artifacts'][key];bind(rec['path'],rec['sha256'])
        features[key]=v4.owner.array(rec,SHAPE)
    predictions={w:v4.owner.array(m['predictions'][w],(3,*SHAPE))[-1] for w in WORDS}
    output=decompose(features,predictions,m['legacy_native_coverage'],r['cross_condition'],mask)
    for path in (Path(__file__),ROOT/'tests'/('test_'+TAG+'.py'),ROOT/'innerdocs'/(TAG+'.md')):bind(path,sha(path))
    require(all(sha(p)==h for p,h in bindings.items()),'final bound bytes stable')
    require(time.monotonic()-started<WALL,'bounded saved analysis wall')
    return dict(schema=TAG+'.report',status='all4_words_all256_patches_and12_pairs_decomposed',
        protocol=protocol(),authority=authority,bindings=bindings,geometry_sources=geometry_sources,**output,
        scientific_admission=False,C2_established=False,task_success=None,renderer_release=False,
        interpretation='descriptive all-patch taskmetric gap localization, not patch selection, causal attribution, weighted-Q validation or task success')


def verify(path,expected):
    value=read_saved(path,expected)
    a=value['authority']
    exact=analyze(a['prepare']['path'],a['prepare']['sha256'],a['report']['path'],a['report']['sha256'],a['strict']['path'],a['strict']['sha256'])
    require(digest(value)==digest(exact),'full independent saved decomposition recomputation')
    return dict(status='all_patch_saved_bridge_verified',report_sha256=expected,scientific_admission=False)


def write_output(path,value):
    path=Path(path).absolute()
    require(path.parent==ROOT/'results' and path.name.startswith(TAG+'_') and
            not path.exists() and not any(p.is_symlink() for p in (path,*path.parents)),'exclusive own output only')
    payload=canonical(value)+b'\n'
    require(len(payload)<=CAP and shutil.disk_usage(path.parent).free>=FLOOR+len(payload),'16MiB cap and 200MiB floor')
    with path.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_DIRECTORY|os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('analyze','verify'));p.add_argument('--output',type=Path)
    for key in ('prepare','report','strict','saved'):
        p.add_argument('--'+key,type=Path);p.add_argument('--'+key+'-sha256')
    a=p.parse_args(argv)
    if a.command=='analyze':
        require(all(getattr(a,k) is not None for k in ('prepare','prepare_sha256','report','report_sha256','strict','strict_sha256','output')),'explicit three authorities/SHA and new output')
        value=analyze(a.prepare,a.prepare_sha256,a.report,a.report_sha256,a.strict,a.strict_sha256)
        write_output(a.output,value)
    else:
        require(a.saved and a.saved_sha256,'explicit own saved report/SHA')
        value=verify(a.saved,a.saved_sha256)
    print(json.dumps(dict(status=value['status'],scientific_admission=False)))
    return 0


if __name__=='__main__':raise SystemExit(main())
