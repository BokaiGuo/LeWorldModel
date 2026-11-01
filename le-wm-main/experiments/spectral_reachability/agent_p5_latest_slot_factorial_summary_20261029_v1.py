"""Saved-only latest-frame15 factorial arithmetic; never launches a neural worker."""
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
from experiments.spectral_reachability import agent_p5_latest_slot_factorial_cuda_20261029_v1 as owner

NAME='agent_p5_latest_slot_factorial_summary_20261029_v1'
ROOT=owner.ROOT
OWNER_SHA='f5436a9ae922e717920c7662087f8380960c2c6a343503e2b8a7bc750b090ead'
CELLS=('AA','AP','PA','PP')
EFFECTS=('visual','proprio','interaction')
SIGNS=('negative','zero','positive','null')
CAP,FLOOR,WALL=32*1024**2,200*1024**2,900
require,sha=owner.require,owner.sha


def raw(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def protocol():
    return dict(schema=NAME+'.protocol',branches=164,pairs=80,new_predictor_calls=492,
        cells=list(CELLS),pair_schedule=owner.protocol()['pairs'],locked_global_frames=[5,10],crossed_global_frame=15,
        AA='original H1 actual history, same common H1 actual20 label for all cells',
        AP_PA_PP='NEW latest15-only interventions; H2 output is provenance, not PP',
        effects='visual=((PA-AA)+(PP-AP))/2; proprio=((AP-AA)+(PP-PA))/2; interaction=PP-PA-AP+AA',
        arithmetic='float64 C-order MSE and full feature/response effects; exact saved score checks, no tolerance',
        zero_sets='exact zero norms on all80, no epsilon, no assumed24 membership',
        inference='descriptive two known roots with shared baselines, not164 independent samples',
        whole_history_comparison_included=False,physical_counterfactual_certified=False,
        RGB_only=False,extra_actual_agent_position_velocity=True,exploratory_post_outcome=True,
        new_neural_calls=0,new_physics_calls=0,P1_admission=False,P5_admission=False,
        output_cap_bytes=CAP,disk_floor_bytes=FLOOR,wall_seconds=WALL)


def merge(used,more):
    for p,h in more.items():
        require(p not in used or used[p]==h,'no conflicting evidence binding');used[p]=h


def read(path,expected,used):
    require(type(expected) is str and len(expected)==64 and all(c in '0123456789abcdef' for c in expected),'explicit external SHA')
    p=Path(path).absolute();value=owner.bridge.read(p,expected,CAP)
    merge(used,{str(p):expected});return value


def load_ref(ref,used):
    directory=Path(ref['directory']);rec=ref['array'];p=directory/rec['file']
    require(directory.is_absolute() and Path(rec['file']).name==rec['file'] and p.resolve()==p and
        not p.is_symlink() and p.stat().st_size==rec['file_bytes']<=4*1024**2 and sha(p)==rec['sha256'],
        'single bounded original array path/bytes/SHA')
    body=p.read_bytes();f=io.BytesIO(body);a=np.load(f,allow_pickle=False)
    require(f.tell()==len(body) and a.dtype==np.float32 and rec['dtype']=='float32' and
        list(a.shape)==rec['shape'] and a.nbytes==rec['nbytes'] and np.isfinite(a).all(),'finite complete float32 NPY EOF')
    merge(used,{str(p):rec['sha256']});return a


def effects(values):
    require(set(values)==set(CELLS),'all four fixed corners')
    aa,ap,pa,pp=(np.asarray(values[c],np.float64) for c in CELLS)
    require(aa.shape==ap.shape==pa.shape==pp.shape and all(np.isfinite(v).all() for v in (aa,ap,pa,pp)),
        'same finite four-cell support')
    return dict(visual=((pa-aa)+(pp-ap))/2,proprio=((ap-aa)+(pp-pa))/2,interaction=pp-pa-ap+aa)


def cell_scores(cells,label,seam):
    require(set(cells)==set(CELLS) and all(v.shape==label.shape for v in cells.values()),'common target support')
    target=label.astype(np.float64)
    scores={c:float(np.mean((v.astype(np.float64)-target)**2,dtype=np.float64)) for c,v in cells.items()}
    return dict(common_label_cell_MSE=scores,MSE_effects={k:float(v) for k,v in effects(scores).items()},
        feature_effect_L2={k:float(np.linalg.norm(v)) for k,v in effects(cells).items()},label_H1_H2_numeric_seam=seam)


def branch(req,row,directory,used):
    require(all(type(req[k]) is type(row[k]) and req[k]==row[k] for k in ('index','root_slot','condition')) and
        set(row['predictions'])==set(owner.ARMS),'exact original branch and all three NEW corners')
    arrays={'AA':load_ref(req['AA'],used)}
    for arm,cell in zip(owner.ARMS,('AP','PA','PP'),strict=True):
        arrays[cell]=load_ref(dict(directory=str(directory),array=row['predictions'][arm]),used)
    require(all(v.shape==(1,3,256,384) for v in arrays.values()),'full native3 outputs; terminal row only for scores')
    cells={c:v[0,-1].astype(np.float64) for c,v in arrays.items()}
    label=load_ref(req['common_label'],used);other=load_ref(req['H2_label'],used)
    require(label.shape==(1,256,384) and other.shape==(2,256,384),'original H1/H2 actual20 batch shapes')
    target=label[0].astype(np.float64)
    seam=dict(bitwise_equal=label[0].tobytes()==other[1].tobytes(),maxabs=float(np.max(np.abs(target-other[1].astype(np.float64)))))
    score=cell_scores(cells,target,seam)
    require(raw(score)==raw(row['score']),'independent four MSE/three effects/feature norms/seam exactly match runtime')
    return cells,target,score


def pair_record(spec,a,b,la,lb,score_a,score_b):
    observed=lb-la;on=float(np.linalg.norm(observed))
    responses={c:b[c]-a[c] for c in CELLS}
    item=dict(spec,observed_response_L2=on,cells={},
        response_effect_L2={k:float(np.linalg.norm(v)) for k,v in effects(responses).items()})
    for c,delta in responses.items():
        pn=float(np.linalg.norm(delta))
        item['cells'][c]=dict(predicted_response_L2=pn,
            response_cosine=None if pn==0 or on==0 else float(np.dot(delta.ravel(),observed.ravel())/(pn*on)),
            plus_minus_MSE=score_b['common_label_cell_MSE'][c]-score_a['common_label_cell_MSE'][c])
    return item


def sign(v):
    require(v is None or type(v) in (int,float) and np.isfinite(v),'finite sign or null')
    return 'null' if v is None else 'negative' if v<0 else 'positive' if v>0 else 'zero'


def describe(values):
    values=list(values);known=[v for v in values if v is not None]
    require(all(type(v) in (int,float) and np.isfinite(v) for v in known),'no silent nonfinite removal')
    return dict(total=len(values),available=len(known),null=len(values)-len(known),
        mean=None if not known else float(np.mean(known,dtype=np.float64)),
        min_median_max=None if not known else [float(v) for v in np.quantile(known,[0,.5,1])])


def table(branches,pairs):
    return dict(branches=len(branches),pairs=len(pairs),
        cell_MSE={c:describe(r['common_label_cell_MSE'][c] for r in branches) for c in CELLS},
        MSE_effects={e:describe(r['MSE_effects'][e] for r in branches) for e in EFFECTS},
        feature_effect_L2={e:describe(r['feature_effect_L2'][e] for r in branches) for e in EFFECTS},
        response_effect_L2={e:describe(r['response_effect_L2'][e] for r in pairs) for e in EFFECTS},
        observed_response=describe(r['observed_response_L2'] for r in pairs),
        cells={c:dict(cosine=describe(r['cells'][c]['response_cosine'] for r in pairs),
            signs={s:sum(sign(r['cells'][c]['response_cosine'])==s for r in pairs) for s in SIGNS},
            predicted_response=describe(r['cells'][c]['predicted_response_L2'] for r in pairs),
            plus_minus_MSE=describe(r['cells'][c]['plus_minus_MSE'] for r in pairs)) for c in CELLS},
        transitions={a+'_to_'+b:{sa+'_to_'+sb:sum(sign(r['cells'][a]['response_cosine'])==sa and
            sign(r['cells'][b]['response_cosine'])==sb for r in pairs) for sa in SIGNS for sb in SIGNS}
            for a in CELLS for b in CELLS if a!=b})


def summarize(m,r,directory,used,check=lambda:None):
    require(len(m['requests'])==len(r['rows'])==164 and len(r['pairs'])==80,'whole164/80 no subset')
    rows=[]
    for i,(req,row) in enumerate(zip(m['requests'],r['rows'],strict=True)):
        check();require(type(req['index']) is type(row['index']) is int and req['index']==row['index']==i and
            (req['root_slot'],req['condition'])==owner.bridge.factor.ORDER[i//41],'fixed original164 identity/order')
        _,_,score=branch(req,row,directory,used)
        rows.append(dict(index=i,root_slot=req['root_slot'],condition=req['condition'],
            task_id=req['task_id'],selection_digest=req['selection_digest'],
            baseline=i%41==0,**score))
    pairs=[];endpoints=set()
    for i,s in enumerate(protocol()['pair_schedule']):
        check();lo,hi=s['minus_index'],s['plus_index']
        require(lo not in endpoints and hi not in endpoints and lo!=hi,'one fixed pair per endpoint')
        endpoints.update((lo,hi))
        a,la,_=branch(m['requests'][lo],r['rows'][lo],directory,used)
        b,lb,_=branch(m['requests'][hi],r['rows'][hi],directory,used)
        item=pair_record(s,a,b,la,lb,rows[lo],rows[hi])
        require(raw(item)==raw(r['pairs'][i]),'independent all80 response/cosine/effects exact identity and values')
        pairs.append(item)
    require(endpoints==set(range(164))-{0,41,82,123},'all160 endpoints and four unpaired baselines')
    zero=[i for i,rp in enumerate(pairs) if rp['observed_response_L2']==0]
    predzero={c:[i for i,rp in enumerate(pairs) if rp['cells'][c]['predicted_response_L2']==0] for c in CELLS}
    levels=[('all',{})]+[('root',dict(root_slot=i)) for i in (0,1)]+[('condition',dict(condition=c)) for c in ('off','on')]+[
        ('root_condition',dict(root_slot=i,condition=c)) for i in (0,1) for c in ('off','on')]
    tables=[dict(factor=factor,level=level,summary=table([b for b in rows if all(b[k]==v for k,v in level.items())],
        [p for p in pairs if all(p[k]==v for k,v in level.items())])) for factor,level in levels]
    return dict(branches=rows,pairs=pairs,tables=tables,
        zero_sets=dict(observed_pair_indices=zero,predicted_pair_indices=predzero,
            predicted_equals_observed={c:predzero[c]==zero for c in CELLS},
            identity_basis='indices into complete original minus/plus/root/condition/h/coord pair schedule; not count equality'))


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def analyze(output,report_sha,prepared,prepare_sha,strict_receipt,strict_sha):
    started=time.monotonic();used={}
    def check():require(time.monotonic()-started<WALL,'whole saved analysis900s including strict and final hashes')
    require(sha(owner.__file__)==OWNER_SHA,'frozen latest-slot runtime source')
    merge(used,{str(Path(owner.__file__).resolve()):OWNER_SHA})
    r=read(Path(output)/'report.json',report_sha,used);m=read(prepared,prepare_sha,used)
    supplied=read(strict_receipt,strict_sha,used)
    require(r['execution_binding']['prepared']==str(Path(prepared).absolute()) and
        r['execution_binding']['prepare_sha256']==prepare_sha,'explicit actual prepare equals runtime execution binding')
    require(r['schema']==owner.NAME+'.report' and m['schema']==owner.NAME+'.prepare' and
        r['protocol']==m['protocol']==owner.protocol(),'latest-slot schemas only, never whole-history')
    for p in own_files():merge(used,{str(p):sha(p)})
    rebuilt=owner.consume(output,report_sha);check()
    require(raw(rebuilt)==raw(supplied) and rebuilt['schema']==owner.NAME+'.consume' and
        rebuilt['status']=='all164_latest_slot_factorial_and80_pairs_verified' and rebuilt['report_sha256']==report_sha and
        rebuilt['branches']==164 and rebuilt['pairs']==80 and rebuilt['new_predictor_calls']==492 and
        rebuilt['actual_native_AA_prefix_bitwise_verified'] is True and
        rebuilt['P1_admission'] is rebuilt['P5_admission'] is False,'external receipt fully rebuilt, complete latest-slot authority')
    merge(used,m['bindings'])
    for name,rec in r['artifacts'].items():merge(used,{str(Path(output).absolute()/name):rec['sha256']})
    result=summarize(m,r,Path(output).absolute(),used,check)
    for path,h in used.items():check();require(sha(path)==h,'all used closure unchanged after numerical analysis')
    check()
    return dict(schema=NAME+'.analysis',status='all164_80_latest_slot_independently_recomputed',protocol=protocol(),
        arguments=dict(output=str(Path(output).absolute()),report_sha=report_sha,prepared=str(Path(prepared).absolute()),
            prepare_sha=prepare_sha,strict_receipt=str(Path(strict_receipt).absolute()),strict_sha=strict_sha),
        authority=dict(report_sha256=report_sha,prepare_sha256=prepare_sha,strict_sha256=strict_sha,
            owner_source_sha256=OWNER_SHA,full_source_strict_rebuilt_here=True,
            original_H1=deepcopy(owner.PINS['H1']),original_H1_strict=deepcopy(owner.PINS['H1_strict'])),
        bindings=used,**result,actual_counts=r['counts'],inherited_costs=r['inherited_costs'],
        whole_history_comparison=None,new_neural_calls=0,new_physics_calls=0,P1_admission=False,P5_admission=False,
        limitations=['Only15 crossed; selected15 features retain original numeric encoding/rollover history.',
            'Model-internal mixed contexts may be physically unreachable, not control benefit or physical counterfactual.',
            'Known2 roots, shared baselines and post-outcome exploration; no confidence interval/independent-sample inference.',
            'H1 common label only; H2 second output never substituted as PP or label.',
            'No whole-history comparison; such comparison needs separate actual source/report/strict SHA authority.'])


def write_new(output,value):
    p=Path(output).absolute();body=raw(value)
    require(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and not p.exists() and
        not any(x.is_symlink() for x in (p,*p.parents)) and len(body)<=CAP and
        shutil.disk_usage(p.parent).free>=FLOOR+len(body),'new own32MiB output/floor, no overwrite')
    with p.open('xb') as f:
        require(f.write(body)==len(body),'whole saved report');f.flush();os.fsync(f.fileno())
    owner.bridge.physical.paired.fsync_dir(p.parent)
    return dict(status=value['status'],path=str(p),sha256=sha(p))


def verify(analysis,expected):
    value=read(analysis,expected,{})
    require(value['schema']==NAME+'.analysis' and raw(value)==raw(analyze(**value['arguments'])),'full strict and numeric summary rebuilt')
    return dict(verified=True,sha256=expected,new_neural_calls=0,new_physics_calls=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('analyze')
    for k in ('factorial-output','report-sha','prepared','prepare-sha','strict-receipt','strict-sha','output'):a.add_argument('--'+k,required=True)
    a=sub.add_parser('verify');a.add_argument('--analysis',required=True);a.add_argument('--analysis-sha',required=True)
    q=parser.parse_args()
    if q.command=='verify':result=verify(q.analysis,q.analysis_sha)
    else:result=write_new(q.output,analyze(q.factorial_output,q.report_sha,q.prepared,q.prepare_sha,q.strict_receipt,q.strict_sha))
    print(raw(result).decode())


if __name__=='__main__':main()
