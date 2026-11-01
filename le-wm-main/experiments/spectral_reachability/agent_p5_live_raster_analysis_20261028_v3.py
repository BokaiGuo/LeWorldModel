"""Saved-only complete raster factorial algebra. Sensitivity, never C2 evidence."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib
from itertools import combinations, permutations
import json
import math
from pathlib import Path

import numpy as np

NAME='agent_p5_live_raster_analysis_20261028_v3'
ROOT=Path(__file__).resolve().parents[2]
ENCODER='main_p5_live_raster_encoder_20261028_v2'
ENCODER_SHA='4f3b1d42a56097e56863c85bc4fe47a0c22e7d8b242b9bc252b4967a70a57a7e'
WORDS=('xy','yx','x_minus_x','zero_zero')
COLORS=('visible','white')
CONDITIONS=tuple((t,g) for t in COLORS for g in COLORS)
SHAPE=(256,384)
CAP=2*1024**2


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(x):return hashlib.sha256(canonical(x)).hexdigest()


def sha(path):
    p=Path(path)
    require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'regular nonsymlink file')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path,expected):
    require(sha(path)==expected and Path(path).stat().st_size<=16*1024**2,'explicit actual SHA/bounded JSON')
    def pairs(items):
        result={}
        for k,v in items:
            require(k not in result,'duplicate JSON key');result[k]=v
        return result
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,
        parse_constant=lambda x:require(False,'nonfinite JSON'))


def encoder():
    p=ROOT/'experiments/spectral_reachability'/f'{ENCODER}.py'
    require(sha(p)==ENCODER_SHA,'fixed reviewed encoder source')
    return importlib.import_module('experiments.spectral_reachability.'+ENCODER)


def protocol():
    return dict(schema=NAME,words=list(WORDS),conditions=[dict(terminal=t,goal=g) for t,g in CONDITIONS],
        score_denominator=16,word_pairs_unordered=6,word_pairs_directed=12,
        directed_pair_condition_rows=48,directed_condition_comparisons=72,
        factorial='mu=mean4; T=(S10+S11-S00-S01)/2; G=(S01+S11-S00-S10)/2; X=S11-S10-S01+S00',
        coding='terminal first, goal second; 0=visible,1=white; lower cost is closer',
        algebra='exact rational arithmetic over saved binary64 score atoms; no epsilon/tolerance',
        ties='exact sign zero, neither strict reversal nor strict agreement',
        scope='all four fixed words/all four color conditions, no ranking-based subset',
        raster_frame=15,legacy_coverage_frame=20,legacy_pairing=False,
        legacy='frame20 native coverage/reward separate column only, not frame15 truth',
        independent_samples=False,significance=None,C2_established=False,scientific_admission=False,
        neural=0,physics=0,fitting=False,selection=False,output_cap_bytes=CAP)


def fraction(value):
    require(type(value) in (float,int) and math.isfinite(value),'finite scalar, not bool/null')
    return Fraction.from_float(float(value))


def atom(x):
    x=Fraction(x)
    try:approx=float(x)
    except OverflowError:approx=None
    if approx is not None and not math.isfinite(approx):approx=None
    return dict(numerator=str(x.numerator),denominator=str(x.denominator),binary64_display=approx)


def sign(x):return int(x>0)-int(x<0)


def validate_scores(rows):
    wanted=[(w,t,g) for w in WORDS for t,g in CONDITIONS]
    require(isinstance(rows,list) and len(rows)==16,'all16 scores, no partial/subset')
    require([(x['word'],x['terminal_raster'],x['goal_raster']) for x in rows]==wanted,'fixed complete ordered16 identities')
    values={}
    for row,key in zip(rows,wanted):
        require(row['status']=='computed','computed score required')
        value=fraction(row['full_patch_goal_mse']);require(value>=0,'nonnegative MSE')
        values[key]=value
    return values


def factorial(scores):
    a,b,c,d=scores  # 00,01,10,11
    mu=(a+b+c+d)/4;terminal=(c+d-a-b)/2;goal=(b+d-a-c)/2;interaction=d-c-b+a
    reconstructed=[mu+terminal*st/2+goal*sg/2+interaction*st*sg/4 for st,sg in ((-1,-1),(-1,1),(1,-1),(1,1))]
    require(reconstructed==list(scores),'exact factorial reconstruction')
    return dict(mean=atom(mu),terminal_main_effect=atom(terminal),goal_main_effect=atom(goal),interaction=atom(interaction),
        terminal_effect_goal_visible=atom(c-a),terminal_effect_goal_white=atom(d-b),
        goal_effect_terminal_visible=atom(b-a),goal_effect_terminal_white=atom(d-c),
        joint_white_minus_joint_visible=atom(d-a),reconstructed=[atom(x) for x in reconstructed],exact_reconstruction=True)


def analyze_scores(rows,coverage):
    values=validate_scores(rows)
    require(isinstance(coverage,list) and [r['word'] for r in coverage]==list(WORDS),'all4 ordered legacy rows')
    for r in coverage:
        fraction(r['coverage']);fraction(r['reward'])
        require(r['done'] is False and r['task_success'] is None and isinstance(r['source_transition_sha256'],str)
            and len(r['source_transition_sha256'])==64,'legacy provenance/native status')
    per_word=[dict(word=w,**factorial([values[w,t,g] for t,g in CONDITIONS])) for w in WORDS]
    mean_by_condition=[sum((values[w,t,g] for w in WORDS),Fraction())/4 for t,g in CONDITIONS]
    pairwise=[];reversals=[]
    for a,b in permutations(WORDS,2):
        deltas={c:values[a,*c]-values[b,*c] for c in CONDITIONS}
        for t,g in CONDITIONS:
            delta=deltas[t,g]
            pairwise.append(dict(a=a,b=b,terminal=t,goal=g,cost_a_minus_b=atom(delta),sign=sign(delta),
                preference='tie' if delta==0 else ('a' if delta<0 else 'b')))
        for old,new in combinations(CONDITIONS,2):
            left,right=deltas[old],deltas[new]
            reversals.append(dict(a=a,b=b,from_condition=dict(terminal=old[0],goal=old[1]),
                to_condition=dict(terminal=new[0],goal=new[1]),from_sign=sign(left),to_sign=sign(right),
                intervention_axis='goal_only' if old[0]==new[0] else 'terminal_only' if old[1]==new[1] else 'both',
                from_cost_a_minus_b=atom(left),to_cost_a_minus_b=atom(right),delta_change=atom(right-left),
                strict_reversal=left*right<0,tie_involved=left==0 or right==0,
                direction=('a_to_b' if left<0<right else 'b_to_a' if right<0<left else None)))
    # Unique-pair counts are descriptive enumerations, not replicate counts.
    unique=[x for x in reversals if WORDS.index(x['a'])<WORDS.index(x['b'])]
    return dict(cross_scores=rows,per_word_factorial=per_word,
        fixed_four_word_average_factorial=factorial(mean_by_condition),
        directed_rankings=pairwise,directed_reversals=reversals,
        reversal_census=dict(unordered_word_pairs=6,condition_pairs=6,comparisons=36,
            strict_reversals=sum(x['strict_reversal'] for x in unique),tie_involved=sum(x['tie_involved'] for x in unique),
            independent_sample_count=None),
        legacy_native_coverage_separate=coverage,legacy_coverage_frame=20,raster_frame=15,
        native_vs_raster_rank_agreement=None,C2_established=False,scientific_admission=False)


def sources():
    return {str(p):sha(p) for p in (Path(__file__).resolve(),ROOT/f'tests/test_{NAME}.py',ROOT/f'innerdocs/{NAME}.md')}


def prepare():
    e=encoder()
    require(e.WORDS==WORDS and e.RASTERS==COLORS and e.SHAPE==SHAPE,'fixed actual encoder ABI')
    return dict(schema=NAME+'.prepare',protocol=protocol(),bindings={**sources(),str(Path(e.__file__).resolve()):ENCODER_SHA},
        status='analysis_protocol_only_no_actual_results',actual_report=None)


def verify(path,expected):
    p=read(path,expected);require(p==prepare(),'exact frozen analysis protocol');return p


def analyze(path,expected,encoder_prepare,encoder_prepare_sha,report_path,report_sha,receipt_path,receipt_sha):
    p=verify(path,expected);e=encoder();ep=read(encoder_prepare,encoder_prepare_sha)
    receipt=e.consume(ep,report_path,report_sha)
    require(receipt==read(receipt_path,receipt_sha),'full encoder saved consume equality')
    r=read(report_path,report_sha)
    # Independent arithmetic on every original saved patch; no encoder/model call.
    arrays={}
    for key in e.KEYS:
        rec=r['token_artifacts'][key];f=Path(report_path).absolute().parent/(key+'_patch.npy')
        require(Path(rec['path'])==f and sha(f)==rec['sha256'] and f.stat().st_size==rec['bytes'],'exact original patch bytes/identity')
        a=np.load(f,allow_pickle=False)
        require(a.shape==SHAPE and a.dtype==np.float32 and np.isfinite(a).all(),'original full finite patch shape')
        arrays[key]=a
    computed=[]
    for w in WORDS:
        for t,g in CONDITIONS:
            d=arrays[w+'__'+t].astype(np.float64)-arrays['goal__'+g].astype(np.float64)
            score=float(np.mean(d*d,dtype=np.float64))
            computed.append(dict(word=w,terminal_raster=t,goal_raster=g,status='computed',full_patch_goal_mse=score))
    require(canonical(computed)==canonical(r['cross_condition'])==canonical(receipt['cross_condition']),'independent all16 binary64 arithmetic')
    result=analyze_scores(computed,receipt['legacy_native_coverage'])
    # Loaded renderer manifest confirms new endpoint15; legacy frame20 is pinned
    # by the reviewed encoder->v5 source contract, not inferred from filenames.
    render_manifest=read(ep['render_authority']['render_manifest'],ep['render_authority']['render_manifest_sha256'])
    require([(x['state_id'],x['frame']) for x in render_manifest['targets']]==[(w,15) for w in WORDS]+[('goal',0)],'new raster source frames')
    raster_receipt=ep['render_receipt']
    require(raster_receipt['status']=='strict_live_raster_complete_v2'
        and [x['state_id'] for x in raster_receipt['baseline_to_current_anchor']]==[*WORDS,'goal']
        and len(raster_receipt['source_comparisons'])==65,'v2 complete time/difference evidence')
    authority=dict(analysis_prepare=dict(path=str(Path(path).absolute()),sha256=expected),
        encoder_prepare=dict(path=str(Path(encoder_prepare).absolute()),sha256=encoder_prepare_sha),
        report=dict(path=str(Path(report_path).absolute()),sha256=report_sha),
        receipt=dict(path=str(Path(receipt_path).absolute()),sha256=receipt_sha))
    require(read(report_path,report_sha)==r and verify(path,expected)==p,'final saved/source drift')
    return dict(schema=NAME+'.report',status='all16_saved_raster_algebra_recorded',protocol=protocol(),authority=authority,
        source_bindings=p['bindings'],analysis=result,within_state_raster_difference=receipt['fixed_within_state_raster_difference'],
        raster_time_diagnostics=dict(baseline_to_current_anchor=raster_receipt['baseline_to_current_anchor'],
            source_comparisons=raster_receipt['source_comparisons'],
            artifact_base=str(Path(ep['render_authority']['render_report']).absolute().parent)),
        old_failure=render_manifest['old_failure'],old_renderer_release=False,historical_full_public_equivalence=None,
        neural=0,physics=0,C2_established=False,scientific_admission=False,errors=[])


def consume(path,expected):
    r=read(path,expected);a=r['authority']
    result=analyze(a['analysis_prepare']['path'],a['analysis_prepare']['sha256'],a['encoder_prepare']['path'],
        a['encoder_prepare']['sha256'],a['report']['path'],a['report']['sha256'],a['receipt']['path'],a['receipt']['sha256'])
    require(canonical(r)==canonical(result),'complete deterministic saved algebra equality')
    return dict(schema=NAME+'.consume',status='saved_algebra_verified',report_sha256=expected,
        score_denominator=16,C2_established=False,scientific_admission=False)


def write_new(path,value):
    import os
    import shutil
    p=Path(path).absolute();data=canonical(value)+b'\n'
    require(len(data)<=CAP and p.parent.is_dir() and not any(q.is_symlink() for q in (p,*p.parents)),'exclusive bounded nonsymlink output')
    require(shutil.disk_usage(p.parent).free>=200*1024**2+len(data),'200MiB free floor')
    with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('prepare','verify','analyze','consume'))
    for key in ('prepare','prepare-sha','encoder-prepare','encoder-prepare-sha','report','report-sha','receipt','receipt-sha','output'):ap.add_argument('--'+key)
    a=ap.parse_args()
    if a.mode=='prepare':r=prepare();write_new(a.output,r)
    elif a.mode=='verify':verify(a.prepare,a.prepare_sha);r=dict(status='verified')
    elif a.mode=='analyze':
        r=analyze(a.prepare,a.prepare_sha,a.encoder_prepare,a.encoder_prepare_sha,a.report,a.report_sha,a.receipt,a.receipt_sha);write_new(a.output,r)
    else:
        r=consume(a.report,a.report_sha)
        if a.output:write_new(a.output,r)
    print(json.dumps(dict(status=r['status'],C2_established=False,scientific_admission=False)))
    return 0


if __name__=='__main__':raise SystemExit(main())
