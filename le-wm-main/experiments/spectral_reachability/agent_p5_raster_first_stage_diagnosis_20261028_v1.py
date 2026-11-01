"""Targeted saved-only first-stage/endpoint diagnosis, not another consumer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

NAME='agent_p5_raster_first_stage_diagnosis_20261028_v1'
ROOT=Path(__file__).resolve().parents[2]
WORDS=('xy','yx','x_minus_x','zero_zero')
COLORS=('visible','white')
NATIVE=ROOT/'external/jepa-wms/evals/simu_env_planning/envs/pusht_env/pusht_env.py'
NATIVE_SHA='6ba34790b6c2ce017a76322c2f1c205ef97d4d3b08a620f844a57075c608a723'
ORIGINAL=ROOT/'results/agent_p5_contact_action_words_physical_20261028_v1_physics_MAIN01/report.json'
ORIGINAL_SHA='a07c2583c5d5dd553324d0e5c0b740ffbde3fa7a806798b3a8dcb0fe77ab6633'


def require(ok,msg):
    if not ok:raise ValueError(msg)


def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(path):
    p=Path(path);require(p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)),'regular nonsymlink file')
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compare_arrays(a,b):
    require(a.shape==b.shape and a.dtype==b.dtype and np.isfinite(a).all() and np.isfinite(b).all(),'same finite array ABI')
    delta=a.astype(np.float64)-b.astype(np.float64)
    return dict(shape=list(a.shape),dtype=a.dtype.str,bytes_equal=a.tobytes()==b.tobytes(),
        unequal_elements=int(np.count_nonzero(a!=b)),mse=float(np.mean(delta**2)),maxabs=float(np.max(np.abs(delta))))


def pixels(a,b):
    require(a.shape==b.shape==(224,224,3) and a.dtype==b.dtype==np.uint8,'RGB224')
    out=compare_arrays(a,b);positions=np.argwhere(np.any(a!=b,axis=2))
    out['unequal_pixels']=int(len(positions))
    out['bbox_yxyx']=None if not len(positions) else [int(positions[:,0].min()),int(positions[:,1].min()),int(positions[:,0].max()),int(positions[:,1].max())]
    return out


def first_stage(pixel_comparison,feature_comparison):
    if pixel_comparison['bytes_equal']:
        return 'null_pixel_first_stage_no_encoder_invariance_test' if feature_comparison['bytes_equal'] else 'identical_pixels_but_features_differ_requires_investigation'
    return 'changed_pixels_identical_features_saved_invariance' if feature_comparison['bytes_equal'] else 'changed_pixels_and_features_sensitive'


class Saved:
    def __init__(self):self.bindings={}
    def bind(self,path,expected):
        p=Path(path).absolute();require(sha(p)==expected,'external/artifact SHA')
        require(str(p) not in self.bindings or self.bindings[str(p)]==expected,'binding collision')
        self.bindings[str(p)]=expected;return p
    def json(self,path,expected):
        p=self.bind(path,expected);require(p.stat().st_size<=16*1024**2,'bounded JSON')
        def pairs(xs):
            out={}
            for k,v in xs:require(k not in out,'duplicate JSON key');out[k]=v
            return out
        return json.loads(p.read_bytes(),object_pairs_hook=pairs,parse_constant=lambda x:require(False,'nonfinite'))
    def array(self,rec,base=None):
        p=Path(rec['path']);p=p if p.is_absolute() else Path(base)/p
        p=self.bind(p,rec['sha256']);require(p.stat().st_size==rec['bytes'],'array bytes')
        a=np.load(p,allow_pickle=False,max_header_size=1024)
        require(a.dtype.kind in 'biuf' and np.isfinite(a).all(),'finite numeric array')
        return a
    def image(self,rec,base):
        a=self.array(rec['npy'],base);p=self.bind(Path(base)/rec['png']['path'],rec['png']['sha256'])
        require(p.stat().st_size==rec['png']['bytes'],'PNG file bytes')
        with Image.open(p) as im:require(im.mode=='RGB' and np.array_equal(np.asarray(im),a),'PNG/NPY exact pixels')
        require(hashlib.sha256(a.tobytes()).hexdigest()==rec['pixel_sha256'],'native pixel SHA')
        return a
    def stable(self):require(all(sha(p)==h for p,h in self.bindings.items()),'final saved bytes unchanged')


def diagnose(analysis_path,analysis_sha,strict_path,strict_sha):
    saved=Saved();ar=saved.json(analysis_path,analysis_sha);strict=saved.json(strict_path,strict_sha)
    require(ar['status']=='all16_saved_raster_algebra_recorded' and strict['status']=='saved_algebra_verified'
        and strict['report_sha256']==analysis_sha,'actual MAIN analysis/strict association')
    authority=ar['authority'];ep=saved.json(authority['encoder_prepare']['path'],authority['encoder_prepare']['sha256'])
    er=saved.json(authority['report']['path'],authority['report']['sha256']);ebase=Path(authority['report']['path']).parent
    a=ep['render_authority'];rp=saved.json(a['render_manifest'],a['render_manifest_sha256']);rr=saved.json(a['render_report'],a['render_report_sha256']);rbase=Path(a['render_report']).parent
    require([(x['state_id'],x['frame']) for x in rp['targets']]==[(w,15) for w in WORDS]+[('goal',0)],'predeclared actual raster endpoints')
    require([x['state_id'] for x in rr['rows']]==[*WORDS,'goal'],'all five original rows')
    saved.bind(NATIVE,NATIVE_SHA)
    require(rp['bindings'][str(ORIGINAL)]==ORIGINAL_SHA,'original report identity from source closure')
    original=saved.json(ORIGINAL,ORIGINAL_SHA)
    inputs={x['image_key']:x for x in ep['inputs']};rgb={};features={};diagnostics=[]
    for row in rr['rows']:
        s=row['state_id'];local={}
        for color in COLORS:
            key=s+'__'+color;shot=next(x for x in row['shots'] if x['shot']==color)
            rec=shot['files']['rgb_image'];rgb[key]=saved.image(rec,rbase)
            require(Path(inputs[key]['rgb']['path'])==(rbase/rec['npy']['path']).absolute()
                and inputs[key]['rgb']['sha256']==rec['npy']['sha256'],'encoder actually selected this native raster')
            feature=er['token_artifacts'][key]
            require(Path(feature['path'])==(ebase/(key+'_patch.npy')).absolute(),'own saved feature identity')
            features[key]=saved.array(feature)
            require(features[key].shape==(256,384) and features[key].dtype==np.float32,'full patch ABI')
            local[color]=saved.json(rbase/shot['files']['after']['path'],shot['files']['after']['sha256'])
        pc=pixels(rgb[s+'__visible'],rgb[s+'__white']);fc=compare_arrays(features[s+'__visible'],features[s+'__white'])
        diagnostics.append(dict(state_id=s,pixels=pc,features=fc,classification=first_stage(pc,fc),
            actual_colors={c:local[c]['goal_color']['rgba'] for c in COLORS},
            public_state_equal=canonical(local['visible']['public_state'])==canonical(local['white']['public_state'])))
    goal=next(x for x in rr['rows'] if x['state_id']=='goal')
    gs=saved.json(rbase/goal['shots'][1]['files']['after']['path'],goal['shots'][1]['files']['after']['sha256'])['public_state']
    goal_pose=gs['env']['goal_pose']['values'];block_pose=[*gs['bodies']['block']['position'],gs['bodies']['block']['angle']]
    plans={x['arm']:x for x in rp['plans']};x=plans['xy'];mx=plans['x_minus_x']
    acommands=x['prefix_raw_u']+x['reference_suffix_raw_u'];bcommands=mx['prefix_raw_u']+mx['reference_suffix_raw_u']
    first_difference=next((i for i,(u,v) in enumerate(zip(acommands,bcommands),1) if canonical(u)!=canonical(v)),None)
    pair_frames=[]
    rows={row['state_id']:row for row in rr['rows']}
    for n in range(16):
        comparisons={};command_comparisons={}
        for name in ('rgb.npy','state.npy','proprio.npy','transition.json','full_public_state.json'):
            refs=[next(f for f in rows[w]['records'] if f['frame']==n)['files'][name] for w in ('xy','x_minus_x')]
            paths=[saved.bind(rbase/f['path'],f['sha256']) for f in refs]
            comparisons[name]=paths[0].read_bytes()==paths[1].read_bytes()
            # Original reference frame selected by saved prepare, not a filename guess.
            source=[rp['frames'][w][n][name] for w in ('xy','x_minus_x')]
            originals=[saved.bind(f['path'],f['sha256']) for f in source]
            command_comparisons[name]=dict(original_pair_exact=originals[0].read_bytes()==originals[1].read_bytes(),
                each_live_matches_original=[p.read_bytes()==q.read_bytes() for p,q in zip(paths,originals)])
        if n:
            raw=[]
            for w,commands in (('xy',acommands),('x_minus_x',bcommands)):
                cpath=rbase/w/f'command_{n:02d}.json';c=saved.json(cpath,rr['artifacts'][str(cpath.relative_to(rbase))]['sha256'])
                require(c['frame']==n and canonical(c['raw_env_argument'])==canonical(commands[n-1]),'actual own command/frame')
                raw.append(c['raw_env_argument'])
            require(canonical(raw[0])==canonical(raw[1]),'same actually executed raw prefix through15')
        pair_frames.append(dict(frame=n,live_pair_exact=comparisons,source_frame_checks=command_comparisons))
    original_late=[]
    orows={row['arm']:row for row in original['rows']}
    for n in (15,16,20):
        arrs={}
        for w in ('xy','x_minus_x'):
            rec=next(f for f in orows[w]['records'] if f['frame']==n)
            arrs[w]={}
            for name in ('rgb.npy','state.npy'):
                path=ORIGINAL.parent/w/f'frame_{n:02d}'/name
                require(str(path) in rp['bindings'] and rp['bindings'][str(path)]==rec['files'][name]['sha256'],'original late frame source closure')
                arrs[w][name]=saved.array(dict(path=str(path),**rec['files'][name]))
        original_late.append(dict(frame=n,rgb=compare_arrays(arrs['xy']['rgb.npy'],arrs['x_minus_x']['rgb.npy']),
            state=compare_arrays(arrs['xy']['state.npy'],arrs['x_minus_x']['state.npy'])))
    pair_rasters={c:dict(pixels=pixels(rgb['xy__'+c],rgb['x_minus_x__'+c]),
        features=compare_arrays(features['xy__'+c],features['x_minus_x__'+c])) for c in COLORS}
    declared_counts=dict(controls=rr['meter']['entries']['control_step'],physics=rr['meter']['entries']['space_step'])
    conclusions=dict(goal_pixel_first_stage=diagnostics[-1]['classification'],
        word_endpoint='same_executed_prefix_at15_before_declared_word_divergence' if first_difference==16 else 'requires_investigation',
        full_word_terminal_measured=False,coverage_frame=20,raster_frame=15,native_vs_raster_rank_agreement=None,
        encoder_goal_invariance_identified=False,C2_established=False,scientific_admission=False,
        old_renderer_release=False,historical_full_public_equivalence=None)
    saved.stable()
    return dict(schema=NAME+'.report',status='targeted_saved_diagnosis_recorded',analysis_sha256=analysis_sha,
        analysis_strict_sha256=strict_sha,raster_first_stage=diagnostics,goal_geometry=dict(goal_pose=goal_pose,block_pose=block_pose,
            exact_pose_alignment=canonical(goal_pose)==canonical(block_pose),native_draw_order='goal polygons first; live bodies debug_draw afterward'),
        xy_x_minus_x=dict(same_constructor=canonical(x['constructor'])==canonical(mx['constructor']),same_seed=x['seed']==mx['seed'],
            first_different_planned_frame=first_difference,actual_frames_0_15=pair_frames,original_late_frames=original_late,
            endpoint_rasters=pair_rasters,declared_measured_counts=declared_counts),
        conclusions=conclusions,bindings=saved.bindings,neural=0,physics=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);p.add_argument('--analysis-sha',required=True)
    p.add_argument('--strict',required=True);p.add_argument('--strict-sha',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=diagnose(a.analysis,a.analysis_sha,a.strict,a.strict_sha);data=canonical(r)+b'\n'
    require(len(data)<2*1024**2,'bounded diagnosis');out=Path(a.output).absolute()
    require(not any(x.is_symlink() for x in (out,*out.parents)),'nonsymlink output')
    with out.open('xb') as f:
        import os
        f.write(data);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(status=r['status'],conclusions=r['conclusions']),sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
