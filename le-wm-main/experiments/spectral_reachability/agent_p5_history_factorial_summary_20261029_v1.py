"""Saved-only fixed164/80 history factorial summary; exploratory, no neural imports."""
import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import agent_p5_history_factorial_cuda_20261029_v1 as owner

NAME='agent_p5_history_factorial_summary_20261029_v1'
ROOT=owner.ROOT
OWNER_SHA='2b012657c44f2c469a79abec590b14f384a20b444f251772a6d7624247554b7d'
ZERO_SHA='35b4c01e4ce7ad846af740878b5399ef17247425f22cba44ceb330a55755c3be'
ZERO_SCHEMA='agent_p5_jepa_callback_zero_sets_20261029_v1.analysis'
CELLS=('AA','AP','PA','PP')
SIGNS=('negative','positive','zero','null')
CAP,FLOOR,WALL=32*1024**2,200*1024**2,600
require,sha,raw,digest=owner.require,owner.sha,owner.raw,owner.digest


def protocol():
    return dict(schema=NAME+'.protocol',branches=164,pairs=80,original_nonzero_pairs=56,original_zero_pairs=24,baseline_branches=4,
        pair_schedule=owner.protocol()['pairs'],cells=list(CELLS),
        strata=['all','root','condition','root_condition'],branch_cohorts=['all164','baseline4','fixed56_endpoints112','fixed24_endpoints48'],
        pair_cohorts=['all80','fixed56','fixed24'],cosine_signs=list(SIGNS),
        transitions='all12 ordered distinct cell pairs, all16 exact-sign bins each, no threshold or selected direction',
        effects='visual=((PA-AA)+(PP-AP))/2; proprio=((AP-AA)+(PP-PA))/2; interaction=PP-PA-AP+AA',
        scalar_arithmetic='independent float64 C-order common-H1-label MSE, MSE effects, feature and pair-response effect L2',
        quantiles=[0,.5,1],selection='fixed original SHA-bound56/24 membership; never new cosine/sign/amplitude selection',
        inference='descriptive only; paired finite differences share baselines/roots, no independent164 or significance claim',
        label_batch_seam='perbranch exact bitwise and float64 maxabs H1 actual20 versus H2 actual20; all four cells use H1 label',
        new_neural_calls=0,new_physics_calls=0,new_training=0,RGB_only=False,extra_agent_position_velocity=True,
        physically_reachable_mixed_context=False,causal_control_benefit=False,exploratory_post_outcome=True,
        P1_admission=False,P5_admission=False,output_cap_bytes=CAP,disk_floor_bytes=FLOOR,wall_seconds=WALL)


def read(path,expected,used):
    path=Path(path).absolute()
    require(type(expected) is str and len(expected)==64 and all(c in '0123456789abcdef' for c in expected),'explicit external SHA256')
    value=owner.bridge.read(path,expected,CAP)
    owner.merge(used,{str(path):expected});return value


def load_ref(ref,used):
    directory=Path(ref['directory']);r=ref['array'];path=directory/r['file']
    require(directory.is_absolute() and Path(r['file']).name==r['file'] and path.resolve()==path and not path.is_symlink() and
        path.stat().st_size==r['file_bytes']<=4*1024**2 and sha(path)==r['sha256'],'whole bounded original array SHA/bytes/path')
    body=path.read_bytes();buf=io.BytesIO(body);a=np.load(buf,allow_pickle=False)
    require(buf.tell()==len(body) and a.dtype==np.float32 and r['dtype']=='float32' and list(a.shape)==r['shape'] and
        a.nbytes==r['nbytes'] and np.isfinite(a).all(),'single original finite full float32 NPY, EOF')
    owner.merge(used,{str(path):r['sha256']});return a


def sign(v):
    require(v is None or type(v) in (float,int) and np.isfinite(v),'finite cosine or explicit null')
    return 'null' if v is None else 'negative' if v<0 else 'positive' if v>0 else 'zero'


def describe(values):
    values=list(values);known=[float(x) for x in values if x is not None]
    require(all(np.isfinite(x) for x in known),'finite descriptive scalars, no silent dropping')
    return dict(total=len(values),available=len(known),null=len(values)-len(known),mean=None if not known else float(np.mean(known,dtype=np.float64)),
        min_median_max=None if not known else [float(x) for x in np.quantile(known,[0,.5,1])])


def effects(values):
    require(set(values)==set(CELLS),'complete fixed four cells')
    a,b,c,d=(np.asarray(values[k],dtype=np.float64) for k in CELLS)
    require(a.shape==b.shape==c.shape==d.shape and all(np.isfinite(x).all() for x in (a,b,c,d)),'same finite factorial support')
    return dict(visual=((c-a)+(d-b))/2,proprio=((b-a)+(d-c))/2,interaction=d-c-b+a)


def original_cohorts(z):
    schedule=protocol()['pair_schedule']
    require(z['schema']==ZERO_SCHEMA and z['status']=='all80_exact_identity_joined' and len(z['rows'])==80 and
        z['P1_admission'] is z['P5_admission'] is False and z['historical_callback_membership'] is None and
        z['sufficient_original_cause_certified'] is False,'original established join, not fresh/causal certification')
    mapping={i:dict(cohort='baseline4',pair_index=None,sign=None) for i in (0,41,82,123)}
    cohort=[]
    for i,(s,r) in enumerate(zip(schedule,z['rows'],strict=True)):
        require(all(type(r[k]) is type(s[k]) and r[k]==v for k,v in s.items()) and not r['missing'] and
            type(r['observed_response_zero']) is bool,'exact original80 schedule identity, no ambiguous membership')
        zero=r['observed_response_zero'];cohort.append('fixed24' if zero else 'fixed56')
        for which in ('minus','plus'):
            idx=s[which+'_index'];require(idx not in mapping,'no duplicate endpoint/baseline assignment')
            mapping[idx]=dict(cohort='fixed24_endpoints48' if zero else 'fixed56_endpoints112',pair_index=i,sign=which)
    require(cohort.count('fixed56')==56 and cohort.count('fixed24')==24 and set(mapping)==set(range(164)),'exact original56/24 plus4 baseline denominator')
    return cohort,mapping


def certified_bundle(output,report_sha,strict_receipt,strict_sha,used):
    require(sha(owner.__file__)==OWNER_SHA,'frozen actual factorial runtime source')
    owner.merge(used,{str(Path(owner.__file__).resolve()):OWNER_SHA})
    supplied=read(strict_receipt,strict_sha,used)
    # Actual saved-only full IO/hashchain/closure consumer, not trusting a status string.
    rebuilt=owner.consume(output,report_sha)
    require(supplied==rebuilt and rebuilt['schema']==owner.NAME+'.consume' and
        rebuilt['status']=='all164_history_factorial_and80_pairs_verified' and rebuilt['report_sha256']==report_sha and
        rebuilt['branches']==164 and rebuilt['pairs']==80 and rebuilt['new_predictor_calls']==328 and
        rebuilt['P1_admission'] is rebuilt['P5_admission'] is False,'external strict receipt exactly rebuilt, complete only')
    directory=Path(output).absolute();r=read(directory/'report.json',report_sha,used)
    require(r['schema']==owner.NAME+'.report' and r['status']=='all164_history_factorial_recorded' and r['protocol']==owner.protocol() and
        len(r['rows'])==164 and len(r['pairs'])==80 and r['errors']==r['source_errors']==[],'complete source report')
    b=r['execution_binding'];m=read(b['prepared'],b['prepare_sha256'],used)
    require(m['schema']==owner.NAME+'.prepare' and m['protocol']==r['protocol'] and len(m['requests'])==164 and
        m['bindings'].get(str(Path(owner.__file__).resolve()))==OWNER_SHA,'actual prepared source binding')
    owner.merge(used,m['bindings'])
    for file,rec in r['artifacts'].items():owner.merge(used,{str(directory/file):rec['sha256']})
    return directory,r,m


def independent_branch(req,row,directory,used):
    require(all(row[k]==req[k] for k in ('index','root_slot','condition')) and set(row['predictions'])==set(owner.ARMS),'unchanged branch and both new corners')
    arrays={'AA':load_ref(req['AA'],used),'PP':load_ref(req['PP'],used)}
    for arm,cell in zip(owner.ARMS,('AP','PA'),strict=True):arrays[cell]=load_ref(dict(directory=str(directory),array=row['predictions'][arm]),used)
    require(all(x.shape==(1,3,256,384) for x in arrays.values()),'all original full3 outputs')
    cells={k:v[0,-1].astype(np.float64) for k,v in arrays.items()}
    label=load_ref(req['common_label'],used);other=load_ref(req['H2_label'],used)
    require(label.shape==(1,256,384) and other.shape==(2,256,384),'original H1/H2 label batch shapes')
    target=label[0].astype(np.float64)
    scores={k:float(np.mean((v-target)**2,dtype=np.float64)) for k,v in cells.items()}
    result=dict(common_label_cell_MSE=scores,MSE_effects={k:float(v) for k,v in effects(scores).items()},
        feature_effect_L2={k:float(np.linalg.norm(v)) for k,v in effects(cells).items()},
        label_H1_H2_numeric_seam=dict(bitwise_equal=label[0].tobytes()==other[1].tobytes(),maxabs=float(np.max(np.abs(target-other[1].astype(np.float64))))))
    require(result==row['score'],'independently reproduced all4 MSE/effects/label seam exactly, no tolerance')
    return cells,target,result


def independent_pairs(r,m,directory,cohorts,branch_rows,used,check):
    out=[]
    for i,s in enumerate(protocol()['pair_schedule']):
        check();lo,hi=s['minus_index'],s['plus_index']
        a,la,_=independent_branch(m['requests'][lo],r['rows'][lo],directory,used)
        b,lb,_=independent_branch(m['requests'][hi],r['rows'][hi],directory,used)
        observed=lb-la;on=float(np.linalg.norm(observed))
        require((on==0)==(cohorts[i]=='fixed24'),'same original zero membership, never redefine cohort from new prediction')
        responses={c:b[c]-a[c] for c in CELLS}
        item=dict(s,observed_response_L2=on,cells={},response_effect_L2={k:float(np.linalg.norm(v)) for k,v in effects(responses).items()})
        for c,dp in responses.items():
            pn=float(np.linalg.norm(dp));item['cells'][c]=dict(predicted_response_L2=pn,
                response_cosine=None if pn==0 or on==0 else float(np.dot(dp.ravel(),observed.ravel())/(pn*on)),
                plus_minus_MSE=branch_rows[hi]['common_label_cell_MSE'][c]-branch_rows[lo]['common_label_cell_MSE'][c])
        require(item==r['pairs'][i],'independent complete80 four-cell response/cosine/effect arithmetic exactly')
        out.append(dict(item,cohort=cohorts[i]))
    return out


def branch_summary(rows):
    return dict(branches=len(rows),cells={c:describe(r['common_label_cell_MSE'][c] for r in rows) for c in CELLS},
        MSE_effects={k:describe(r['MSE_effects'][k] for r in rows) for k in ('visual','proprio','interaction')},
        feature_effect_L2={k:describe(r['feature_effect_L2'][k] for r in rows) for k in ('visual','proprio','interaction')},
        label_batch_seam=dict(bitwise_equal=sum(r['label_H1_H2_numeric_seam']['bitwise_equal'] for r in rows),
            bitwise_different=sum(not r['label_H1_H2_numeric_seam']['bitwise_equal'] for r in rows),
            maxabs=describe(r['label_H1_H2_numeric_seam']['maxabs'] for r in rows)))


def pair_summary(rows):
    return dict(pairs=len(rows),observed_response_L2=describe(r['observed_response_L2'] for r in rows),
        cells={c:dict(signs={s:sum(sign(r['cells'][c]['response_cosine'])==s for r in rows) for s in SIGNS},
            cosine=describe(r['cells'][c]['response_cosine'] for r in rows),
            predicted_response_L2=describe(r['cells'][c]['predicted_response_L2'] for r in rows),
            plus_minus_MSE=describe(r['cells'][c]['plus_minus_MSE'] for r in rows)) for c in CELLS},
        response_effect_L2={k:describe(r['response_effect_L2'][k] for r in rows) for k in ('visual','proprio','interaction')},
        transitions={a+'_to_'+b:{sa+'_to_'+sb:sum(sign(r['cells'][a]['response_cosine'])==sa and sign(r['cells'][b]['response_cosine'])==sb for r in rows)
            for sa in SIGNS for sb in SIGNS} for a in CELLS for b in CELLS if a!=b})


def stratify(branches,pairs):
    require(len(branches)==164 and len(pairs)==80,'full denominators before any reporting strata')
    levels=[('all',{})]+[('root',dict(root_slot=k)) for k in (0,1)]+[('condition',dict(condition=c)) for c in ('off','on')]+[
        ('root_condition',dict(root_slot=k,condition=c)) for k in (0,1) for c in ('off','on')]
    tables=[]
    for factor,level in levels:
        bs=[r for r in branches if all(r[k]==v for k,v in level.items())]
        ps=[r for r in pairs if all(r[k]==v for k,v in level.items())]
        tables.append(dict(factor=factor,level=level,branches={c:branch_summary(bs if c=='all164' else [r for r in bs if r['cohort']==c])
            for c in protocol()['branch_cohorts']},pairs={c:pair_summary(ps if c=='all80' else [r for r in ps if r['cohort']==c])
            for c in protocol()['pair_cohorts']}))
    return tables


def analyze(output,report_sha,strict_receipt,strict_sha,zero_analysis,zero_sha):
    started=time.monotonic();used={}
    arguments=dict(output=str(Path(output).absolute()),report_sha=report_sha,strict_receipt=str(Path(strict_receipt).absolute()),strict_sha=strict_sha,
        zero_analysis=str(Path(zero_analysis).absolute()),zero_sha=zero_sha)
    def check():require(time.monotonic()-started<WALL,'whole saved summary600s cap including strict/final hashes')
    require(zero_sha==ZERO_SHA,'fixed original zero-cohort authority, no replacement membership')
    directory,r,m=certified_bundle(output,report_sha,strict_receipt,strict_sha,used);check()
    z=read(zero_analysis,zero_sha,used)
    require(z['authority']['jepa_report_sha256']==owner.PINS['H1'][1] and z['authority']['jepa_receipt_sha256']==owner.PINS['H1_strict'][1],
        'cohort is exactly inherited H1, not a new renderer/model subset')
    owner.merge(used,z['bindings']);cohorts,mapping=original_cohorts(z)
    branches=[]
    for i,(req,row) in enumerate(zip(m['requests'],r['rows'],strict=True)):
        check();require(req['index']==row['index']==i,'fixed original branch order')
        _,_,score=independent_branch(req,row,directory,used)
        branches.append(dict(index=i,root_slot=req['root_slot'],condition=req['condition'],**mapping[i],**score))
    pairs=independent_pairs(r,m,directory,cohorts,branches,used,check)
    tables=stratify(branches,pairs);check()
    for p in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):owner.merge(used,{str(p):sha(p)})
    for path,h in used.items():check();require(sha(path)==h,'full used source/model/array evidence unchanged')
    check()
    return dict(schema=NAME+'.analysis',status='all164_80_saved_factorial_summary_verified',protocol=protocol(),arguments=arguments,
        authority=dict(report_sha256=report_sha,strict_sha256=strict_sha,zero_cohort_sha256=zero_sha,owner_source_sha256=OWNER_SHA),
        branches=branches,pairs=pairs,tables=tables,bindings=used,source_strict_receipt_rebuilt_here=True,
        new_neural_calls=0,new_physics_calls=0,P1_admission=False,P5_admission=False,
        limitations=['Exploratory after prior outcomes; no preregistered confirmation or significance.',
            'Complete history block intervention, not isolated frame15 or pure horizon.',
            'Mixed contexts may be unreachable; model-internal sensitivity is not a physical counterfactual or control benefit.',
            'MSE factorial interaction is metric-space arithmetic; feature interaction is separately reported.',
            'Known roots and shared finite-difference baselines, not164 independent roots or fresh evidence.',
            'Extra agent position/velocity permission; not RGB-only or fair cross-model input ranking.'])


def write_new(value,output):
    p=Path(output).absolute();body=raw(value)
    require(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and not p.exists() and not any(x.is_symlink() for x in (p,*p.parents)) and
        len(body)<=CAP and shutil.disk_usage(p.parent).free>=FLOOR+len(body),'new own bounded summary/floor, no overwrite')
    with p.open('xb') as f:require(f.write(body)==len(body),'complete write');f.flush();os.fsync(f.fileno())
    owner.bridge.physical.paired.fsync_dir(p.parent)
    return dict(status=value['status'],path=str(p),sha256=sha(p))


def verify(analysis,expected):
    old=read(analysis,expected,{})
    require(old['schema']==NAME+'.analysis' and old['protocol']==protocol() and old==analyze(**old['arguments']),
        'full source strict, all4cell arithmetic, cohort/strata/transition summary deterministically rebuilt')
    return dict(status='saved_factorial_summary_recomputed',analysis_sha256=expected,new_neural_calls=0,new_physics_calls=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('analyze')
    for key in ('factorial-output','report-sha','strict-receipt','strict-sha','zero-analysis','zero-sha','output'):a.add_argument('--'+key,required=True)
    a=sub.add_parser('verify');a.add_argument('--analysis',required=True);a.add_argument('--analysis-sha',required=True)
    q=p.parse_args()
    if q.command=='analyze':result=write_new(analyze(q.factorial_output,q.report_sha,q.strict_receipt,q.strict_sha,q.zero_analysis,q.zero_sha),q.output)
    else:result=verify(q.analysis,q.analysis_sha)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
