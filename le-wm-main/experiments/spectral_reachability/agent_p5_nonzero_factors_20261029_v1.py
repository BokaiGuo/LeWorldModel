"""Saved-only all80/established56 response diagnostics; no neural/physics work."""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

from experiments.spectral_reachability import agent_p5_jepa_callback_zero_sets_20261029_v1 as saved

NAME='agent_p5_nonzero_factors_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
H1='agent_p5_raw164_jepa_cuda_20261029_v1'
H2='agent_p5_raw164_jepa_h2_cuda_20261029_v2'
PHASES=('H1_20','H2_15','H2_20')
SOURCE_PINS={H1:'610523392e1923e4371556606109def7f882a4a90741e2e14b63af68880e2180',
    H2:'b661a7c27181ef29f8277e154006f9238ec1c94487c311c1065e4d42fcfc2241',
    saved.NAME:'38a3749f0defbae1e38771ef216034752926061d451f372332d43923513b844f'}
PINS={'fd_prepare':('agent_observed_raw_coordinate_fd_20261028_v1_prepare_MAIN01.json',
    'a1a87327d230af7e02e2379909305c9fbbfb8a05768ba85eb4fcbdac85f95ef5'),
    'physical_report':('agent_observed_raw_coordinate_physical_20261028_v1_MAIN01/report.json',
    'c4610c7f495137cbda7d70b147c74670db6f85339561da611ce5bee152d7bb10')}
require,sha=saved.require,saved.sha
WALL=600


def protocol():
    return dict(fixed_pairs=80,fixed_branches=164,established_H1_nonzero_cohort=56,
        phases=list(PHASES),scales=[.01,.001],coordinates=list(range(10)),
        raw_steps_global=list(range(16,21)),axes=['x','y'],conditions=['off','on'],roots=[0,1],
        subset='all80 retained; primary cohort only original H1 exact nonzero membership, never cosine/outcome selection',
        low_signal='continuous pixel/feature amplitude and float32 ULP envelope ratios; no tuned epsilon or low/high cutoff',
        nonlinearity='both fixed h secants, nonuniform one-sided slopes and curvature; not infinitesimal Jacobian certification',
        prediction_error='endpoint/common/differential errors and analytic response projection; no fitted metric or readout',
        temporal='H2_15 negative control before raw16..20; H1/H2 conditioning differs, not isolated causal horizon effect',
        cross_model_MSE_ranking=False,causal_attribution=False,training=False,
        new_neural_calls=0,new_physics_calls=0,P1_admission=False,P5_admission=False,
        wall_seconds=WALL,output_cap_bytes=32*1024**2,disk_floor_bytes=200*1024**2)


def vector(v):
    a=np.asarray(v,np.float64).reshape(-1)
    require(a.size>0 and np.isfinite(a).all(),'finite nonempty vector');return a


def cosine(a,b):
    a,b=vector(a),vector(b);require(a.shape==b.shape,'same vector support')
    an,bn=float(np.linalg.norm(a)),float(np.linalg.norm(b))
    return None if an==0 or bn==0 else float(a@b/(an*bn))


def ratio(a,b):return None if b==0 else float(a/b)


def response(pm,pp,am,ap):
    pm,pp,am,ap=map(vector,(pm,pp,am,ap));require(pm.shape==pp.shape==am.shape==ap.shape,'same response feature space')
    dp,da=pp-pm,ap-am;em,ep=pm-am,pp-ap;de=ep-em
    pn,an,en=map(lambda a:float(np.linalg.norm(a)),(dp,da,de))
    gain=None if an==0 else float((dp@da)/(da@da))
    orth=None if gain is None else float(np.linalg.norm(dp-gain*da))
    # Algebra only, not a fitted controller/metric; keep finite arithmetic residual visible.
    ulp=float(np.linalg.norm(np.spacing(np.abs(am).astype(np.float32)).astype(np.float64))+
        np.linalg.norm(np.spacing(np.abs(ap).astype(np.float32)).astype(np.float64)))
    common=(em+ep)/2
    return dict(observed_L2=an,predicted_L2=pn,cosine=cosine(dp,da),observed_exact_zero=bool(an==0),
        predicted_exact_zero=bool(pn==0),response_error_L2=en,parallel_gain=gain,orthogonal_predicted_L2=orth,
        response_error_over_signal=ratio(en,an),response_over_endpoint_L2=ratio(an,(float(np.linalg.norm(am))+float(np.linalg.norm(ap)))/2),
        float32_one_ULP_endpoint_envelope_L2=ulp,response_over_ULP_envelope=ratio(an,ulp),
        minus_MSE=float(np.mean(em*em)),plus_MSE=float(np.mean(ep*ep)),
        common_error_MSE=float(np.mean(common*common)),differential_error_MSE=float(np.mean(de*de)),
        endpoint_MSE_decomposition_residual=float((np.mean(em*em)+np.mean(ep*ep))/2-np.mean(common*common)-np.mean(de*de)/4),
        response_error_identity_residual_L2=float(np.linalg.norm((dp-da)-de)))


def local(pm,pp,pb,am,ap,ab,minus_delta,plus_delta):
    require(np.isfinite([minus_delta,plus_delta]).all() and minus_delta>0 and plus_delta>0,'actual positive float32 perturbation distances')
    out={}
    for key,(m,p,b) in dict(predicted=(pm,pp,pb),observed=(am,ap,ab)).items():
        m,p,b=map(vector,(m,p,b));require(m.shape==p.shape==b.shape,'same baseline feature support')
        forward=(p-b)/plus_delta;backward=(b-m)/minus_delta
        curv=2*(forward-backward)/(plus_delta+minus_delta)
        out[key]=dict(one_sided_cosine=cosine(forward,backward),
            forward_L2=float(np.linalg.norm(forward)),backward_L2=float(np.linalg.norm(backward)),
            second_secant_L2=float(np.linalg.norm(curv)),
            one_sided_disagreement_ratio=ratio(float(np.linalg.norm(forward-backward)),float(np.linalg.norm(forward))+float(np.linalg.norm(backward))))
    return out


def scale_compare(large,small):
    large,small=vector(large),vector(small);require(large.shape==small.shape,'same coordinate across both fixed scales')
    ln,sn=float(np.linalg.norm(large)),float(np.linalg.norm(small))
    return dict(large_L2=ln,small_L2=sn,cosine=cosine(large,small),
        small_over_large=ratio(sn,ln),relative_difference_to_large=ratio(float(np.linalg.norm(small-large)),ln),
        large_exact_zero=ln==0,small_exact_zero=sn==0)


def pixel_response(a,b):
    require(a.dtype==b.dtype==np.uint8 and a.shape==b.shape==(224,224,3),'original same RGB224 support')
    d=b.astype(np.int16)-a.astype(np.int16)
    return dict(exact_equal=bool(np.array_equal(a,b)),changed_pixels=int(np.count_nonzero(np.any(d!=0,axis=-1))),
        changed_channels=int(np.count_nonzero(d)),RMS=float(np.sqrt(np.mean(d.astype(np.float64)**2))),
        max_integer_change=int(np.max(np.abs(d))))


def native_response(a,b):
    a,b=vector(a),vector(b);require(a.shape==b.shape==(7,),'whole native7, never truncate task success')
    d=b-a;wrapped=float(np.arctan2(np.sin(d[4]),np.cos(d[4])))
    return dict(delta_native7=d.tolist(),object_delta_xytheta=[float(d[2]),float(d[3]),wrapped],
        object_angle_unwrapped=float(d[4]),agent_delta_xy=d[:2].tolist(),agent_delta_velocity=d[5:].tolist(),
        interpretation='axes in native units, not aggregate task failure or object-only success')


def load_array(directory,rec,shape,inventory,used):
    require(rec['shape']==list(shape) and rec['dtype']=='float32' and Path(rec['file']).name==rec['file'] and
        inventory.get(rec['file'])=={k:rec[k] for k in ('sha256','file_bytes')},'exact original artifact descriptor/inventory')
    p=saved.checked(Path(directory)/rec['file'],rec['sha256'],used)
    require(p.stat().st_size==rec['file_bytes']<=4*1024**2,'original NPY byte bound')
    buf=io.BytesIO(p.read_bytes());a=np.load(buf,allow_pickle=False)
    require(buf.tell()==p.stat().st_size and a.shape==shape and a.dtype==np.float32 and
        a.nbytes==rec['nbytes'] and np.isfinite(a).all(),'full finite original NPY and EOF')
    return a


def load_rgb(rec,used):
    from PIL import Image
    p=saved.checked(rec['path'],rec['file_sha256'],used)
    with Image.open(p) as im:
        require(im.mode=='RGB' and im.size==(224,224),'no color conversion or resize');a=np.asarray(im).copy()
    require(hashlib.sha256(a.tobytes()).hexdigest()==rec['pixel_sha256'],'original pixel SHA');return a


def bundle(output,report_sha,receipt,receipt_sha,kind,used):
    r=saved.read(Path(output)/'report.json',report_sha,used);a=saved.read(receipt,receipt_sha,used)
    stem=H1 if kind=='H1' else H2
    require(r['schema']==stem+'.report' and a['schema']==stem+'.consume' and a['report_sha256']==report_sha and
        a['status']==('all164_factual_H1_and80_pairs_verified' if kind=='H1' else 'all164_open_loop_H2_and80_pairs_verified') and
        r['status']==('fixed164_actual15_conditioned_factual_H1_recorded' if kind=='H1' else 'fixed164_root10_open_loop_H2_recorded') and
        a['requests']==164 and a['pairs']==80 and len(r['rows'])==164 and len(r['pairs'])==80 and
        r['errors']==r['source_errors']==[] and r['model_unchanged'] is True and a['counts']==r['counts'] and
        r['counts']['attempted']==r['counts']['returned']==r['protocol']['caps'] and
        r['model_before']==r['model_after_predictions']==r['model_after'] and
        r['P1_admission'] is r['P5_admission'] is a['P1_admission'] is a['P5_admission'] is False,'complete sealed H1/H2, external strict receipt')
    b=r['execution_binding'];p=saved.read(b['prepared'],b['prepare_sha256'],used)
    source=ROOT/'experiments/spectral_reachability'/(stem+'.py')
    require(p['schema']==stem+'.prepare' and p['protocol']==r['protocol'] and
        p['bindings'].get(str(source))==SOURCE_PINS[stem],'original runtime source binding')
    saved.checked(source,SOURCE_PINS[stem],used)
    require(p['protocol']['pairs']==saved.schedule(),'fixed identical80 schedule')
    require([saved.identity(x) for x in r['pairs']]==[saved.identity(x) for x in saved.schedule()] and
        all(x['missing']==[] for x in r['pairs']),'no missing/reordered/duplicated pair')
    return r,p


def branch_identity(i,job,h1,h2,p1,p2,native):
    req1,req2=p1['bridge']['requests'][i],p2['bridge']['requests'][i]
    rows=[h1['rows'][i],h2['rows'][i],req1,req2,native['rows'][i]]
    require(all(x['index']==i and x['root_slot']==job['root_slot'] and x['condition']==job['condition'] for x in rows),'same full branch/root/condition')
    for k in ('task_id','selection_digest'):require(len({x[k] for x in rows})==1,'same task/selection identity')
    require(saved.rawbits(req1['commands'][-10:],10)==saved.rawbits(req2['commands'][-10:],10)==saved.rawbits(job['full10'],10)==
        saved.rawbits(native['rows'][i]['plan']['commands'],10),'literal original raw10, no outcome-selected join')
    require(native['rows'][i]['plan']['producer_job_sha256']==job['job_sha256'],'original job authority')
    require(req1['context_global_frames']==[5,10,15] and req1['successor_global_frame']==20 and
        req2['context_global_frames']==[0,5,10] and req2['successor_global_frames']==[15,20],'separate H1/H2 time contracts')
    l1,l2=req1['label'],req2['labels'][1]
    require(l1==l2 and (l1['phase'],l1['step'],l1['condition'],l1['overlay_index'])==
        ('plan',10,job['condition'],int(job['condition']=='on')),'same exact terminal PNG identity')
    return req1,req2


def collect_group(slot,h1,h2,p1,p2,jobs,native,d1,d2,used):
    out=[]
    for i in range(41*slot,41*(slot+1)):
        q1,q2=branch_identity(i,jobs[i],h1,h2,p1,p2,native)
        r1,r2=h1['rows'][i],h2['rows'][i]
        get=lambda directory,r,key,shape:load_array(directory,r[key],shape,(h1 if directory==d1 else h2)['artifacts'],used)
        y1=get(d1,r1,'prediction',(1,3,256,384))[0,-1].copy()
        a1=get(d1,r1,'label_patch',(1,256,384))[0].copy()
        y15=get(d2,r2,'prediction15',(1,3,256,384))[0,-1].copy()
        y20=get(d2,r2,'prediction',(1,3,256,384))[0,-1].copy()
        labels=get(d2,r2,'label_patch',(2,256,384))
        mse=lambda a,b:float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2,dtype=np.float64))
        require(mse(y1,a1)==r1['score']['successor_MSE'] and mse(y15,labels[0])==r2['score']['prediction15_MSE'] and
            mse(y20,labels[1])==r2['score']['successor_MSE'],'independent original saved endpoint MSE exactly reproduced')
        out.append(dict(phases=dict(H1_20=(y1,a1),H2_15=(y15,labels[0].copy()),H2_20=(y20,labels[1].copy())),
            rgb=load_rgb(q1['label'],used),state=native['rows'][i]['final_state'],
            label20_H1_H2=dict(bitwise_equal=a1.tobytes()==labels[1].tobytes(),maxabs=float(np.max(np.abs(a1.astype(float)-labels[1].astype(float))))),
            first_H2_inputs=h2['calls'][4*i+2]['inputs'],
            first_H2_frame15=q2['labels'][0]))
    return out


def pair_record(spec,g,jobs,z):
    i,j=spec['minus_index'],spec['plus_index'];offset=41*spec['slot'];lo,hi,base=g[i-offset],g[j-offset],g[0]
    coord=spec['coord'];row,axis=coord//2,coord%2
    um,up,ub=[np.asarray(jobs[k]['raw_suffix5'],np.float64).reshape(-1) for k in (i,j,offset)]
    dm,dp=float(ub[coord]-um[coord]),float(up[coord]-ub[coord]);span=dm+dp
    mask=np.arange(10)!=coord
    require(np.array_equal(um[mask],ub[mask]) and np.array_equal(up[mask],ub[mask]) and
        span==jobs[i]['actual_pair_spacing']==jobs[j]['actual_pair_spacing'] and dm>0 and dp>0,'single declared coordinate/actual float32 span')
    require(saved.identity(spec)==saved.identity(z) and not z['missing'] and z['replay_to_original224_exact'] is True,'original exact zero/raster join identity')
    rows={}
    for phase in PHASES:
        pm,am=lo['phases'][phase];pp,ap=hi['phases'][phase];pb,ab=base['phases'][phase]
        rows[phase]=dict(response=response(pm,pp,am,ap),local=local(pm,pp,pb,am,ap,ab,dm,dp))
    require(rows['H1_20']['response']['observed_exact_zero'] is z['observed_response_zero'],'same established exact-zero membership')
    pixels=pixel_response(lo['rgb'],hi['rgb'])
    require(pixels['exact_equal'] is z['rgb224_equal'],'same original224 equality as certified zero-set join')
    # Descriptor paths differ even when input tensor bytes are identical; compare their content SHA trees instead.
    def content(tree):
        if not isinstance(tree,dict):return tree
        if '$array' in tree:return {k:tree['$array'][k] for k in ('sha256','shape','dtype','nbytes')}
        return {k:[content(v) for v in x] if isinstance(x,list) else content(x) for k,x in tree.items()}
    input_equal=saved.canonical(content(lo['first_H2_inputs']))==saved.canonical(content(hi['first_H2_inputs']))
    return dict(spec,pair_id=z['pair_id'],suffix_raw_step=row+1,global_raw_step=16+row,axis=('x','y')[axis],
        actual_minus_delta=dm,actual_plus_delta=dp,actual_span=span,primary_H1_nonzero=not z['observed_response_zero'],
        raster={k:z[k] for k in ('integer_primitives_equal','surface512_equal','rgb224_equal','historical_callback_membership')},
        pixels=pixels,native=native_response(lo['state'],hi['state']),phases=rows,
        H2_before_perturbation_control=dict(first_predictor_inputs_content_equal=input_equal,
            observed15_pixels_identity_equal=lo['first_H2_frame15']['pixel_sha256']==hi['first_H2_frame15']['pixel_sha256'],
            predicted15_exact_zero=rows['H2_15']['response']['predicted_exact_zero'],observed15_exact_zero=rows['H2_15']['response']['observed_exact_zero']),
        label20_batch_layout_comparison=[lo['label20_H1_H2'],hi['label20_H1_H2']],
        H2_minus_H1_20_mean_endpoint_MSE=(rows['H2_20']['response']['minus_MSE']+rows['H2_20']['response']['plus_MSE']-
            rows['H1_20']['response']['minus_MSE']-rows['H1_20']['response']['plus_MSE'])/2)


def summary(rows,phase):
    vals=[r['phases'][phase]['response'] for r in rows]
    def describe(key):
        v=[r[key] for r in vals if r[key] is not None]
        return dict(available=len(v),missing=len(vals)-len(v),min_median_max=None if not v else [float(x) for x in np.quantile(v,[0,.5,1])])
    return dict(pairs=len(vals),observed_zero=sum(v['observed_exact_zero'] for v in vals),
        predicted_zero=sum(v['predicted_exact_zero'] for v in vals),
        cosine_signs={key:sum((v['cosine'] is None) if key=='null' else (v['cosine'] is not None and
            (v['cosine']>0 if key=='positive' else v['cosine']<0 if key=='negative' else v['cosine']==0)) for v in vals)
            for key in ('positive','negative','zero','null')},
        continuous={k:describe(k) for k in ('observed_L2','response_over_endpoint_L2','response_over_ULP_envelope',
            'response_error_over_signal','common_error_MSE','differential_error_MSE')})


def factors(rows):
    specs=[('all',()),('root',('root_slot',)),('condition',('condition',)),('step_size',('h',)),
        ('coordinate',('coord',)),('raw_step',('global_raw_step',)),('axis',('axis',)),
        ('root_condition_step_size',('root_slot','condition','h'))]
    result=[]
    for name,fields in specs:
        groups=defaultdict(list)
        for r in rows:groups[tuple(r[k] for k in fields)].append(r)
        for key,group in sorted(groups.items()):
            primary=[r for r in group if r['primary_H1_nonzero']]
            result.append(dict(factor=name,level=dict(zip(fields,key)),all80_support=len(group),primary_nonzero_support=len(primary),
                phases={p:dict(all_pairs=summary(group,p),fixed_nonzero_cohort=summary(primary,p)) for p in PHASES}))
    return result


def sign(value):return 'null' if value is None else 'positive' if value>0 else 'negative' if value<0 else 'zero'


def paired_directions(rows):
    return {cohort:{f'{a}_to_{b}':sum(sign(r['phases']['H1_20']['response']['cosine'])==a and
        sign(r['phases']['H2_20']['response']['cosine'])==b for r in selected)
        for a in ('positive','negative','zero','null') for b in ('positive','negative','zero','null')}
        for cohort,selected in [('all80',rows),('fixed56',[r for r in rows if r['primary_H1_nonzero']])]}


def analyze(h1_output,h1_report_sha,h1_receipt,h1_receipt_sha,h2_output,h2_report_sha,h2_receipt,h2_receipt_sha,zero_analysis,zero_sha):
    started=time.monotonic();used={};arguments=deepcopy(locals());arguments.pop('started');arguments.pop('used')
    def check():require(time.monotonic()-started<WALL,'bounded saved analysis600s, no partial scientific conclusion')
    for stem,h in SOURCE_PINS.items():saved.checked(ROOT/'experiments/spectral_reachability'/(stem+'.py'),h,used)
    r1,p1=bundle(h1_output,h1_report_sha,h1_receipt,h1_receipt_sha,'H1',used)
    r2,p2=bundle(h2_output,h2_report_sha,h2_receipt,h2_receipt_sha,'H2',used)
    require(r1['model_before']==r2['model_before'],'same frozen JEPA loaded model state, not cross-architecture MSE comparison')
    z=saved.read(zero_analysis,zero_sha,used)
    require(z['schema']==saved.NAME+'.analysis' and z['status']=='all80_exact_identity_joined' and len(z['rows'])==80 and
        z['authority']['jepa_report_sha256']==h1_report_sha and z['authority']['jepa_receipt_sha256']==h1_receipt_sha and
        z['P1_admission'] is z['P5_admission'] is False and z['sufficient_original_cause_certified'] is False and
        z['historical_callback_membership'] is None,'existing complete exact zero/raster identity join; no historical causality upgrade')
    # Recheck the byte bindings used by the established join rather than restating its producer flags as a new consumer.
    for f,h in z['bindings'].items():check();saved.checked(f,h,used)
    assets={k:saved.read(ROOT/'results'/name,h,used) for k,(name,h) in PINS.items()}
    jobs=assets['fd_prepare']['jobs'];native=assets['physical_report']
    require(len(jobs)==len(native['rows'])==164 and native['errors']==[],'whole original164 physical/source cohort')
    require([saved.identity(x) for x in z['rows']]==[saved.identity(x) for x in saved.schedule()],'exact established all80 membership/order')
    rows=[];scales=[]
    for slot in range(4):
        check();g=collect_group(slot,r1,r2,p1,p2,jobs,native,Path(h1_output),Path(h2_output),used)
        for k,spec in enumerate(saved.schedule()):
            if spec['slot']==slot:rows.append(pair_record(spec,g,jobs,z['rows'][k]))
        for coord in range(10):
            ks=[1+20*hi+2*coord for hi in range(2)]
            comps={};native_axes=[]
            for k in ks:
                delta=native_response(g[k]['state'],g[k+1]['state']);span=jobs[41*slot+k]['actual_pair_spacing']
                native_axes.append(dict(native7_secants=(np.asarray(delta['delta_native7'])/span).tolist(),
                    object_xytheta_secants=(np.asarray(delta['object_delta_xytheta'])/span).tolist()))
            for phase in PHASES:
                values={'predicted':[],'observed':[]}
                for k in ks:
                    span=jobs[41*slot+k]['actual_pair_spacing']
                    for axis,key in enumerate(('predicted','observed')):
                        values[key].append((g[k+1]['phases'][phase][axis].astype(float)-g[k]['phases'][phase][axis].astype(float)).ravel()/span)
                comps[phase]={key:scale_compare(*v) for key,v in values.items()}
            scales.append(dict(slot=slot,root_slot=saved.ORDER[slot][0],condition=saved.ORDER[slot][1],coord=coord,
                global_raw_step=16+coord//2,axis=('x','y')[coord%2],phases=comps,
                native_axes_large_small=native_axes,native_axes_note='per-axis native units, no combined norm or task-success inference'))
        del g
    require(len(rows)==80 and len(scales)==40 and sum(r['primary_H1_nonzero'] for r in rows)==56,'fixed all80/40 and established56, never redefine cohort')
    check()
    for f,h in tuple(used.items()):check();require(sha(f)==h,'all actually consumed saved evidence unchanged')
    for f in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):used[str(f)]=sha(f)
    return dict(schema=NAME+'.analysis',status='saved_factor_diagnostics_complete_not_causal',protocol=protocol(),arguments=arguments,
        rows=rows,cross_scale=scales,factor_tables=factors(rows),paired_direction_signs=paired_directions(rows),bindings=used,
        source_strict_receipts_rebuilt_here=False,historical_callback_membership=None,sufficient_original_cause_certified=False,
        remaining_identification_gaps=['No physical counterfactual for model-predicted features.',
            'Finite h curvature does not isolate raster quantization from dynamics/encoder nonlinearity.',
            'H1/H2 changes actual versus predicted intermediate visual/proprio context; not a pure causal horizon intervention.',
            'Low continuous signal or error correlation is not a causal assignment or a tuned selection rule.'],
        P1_admission=False,P5_admission=False,new_neural_calls=0,new_physics_calls=0)


def write_new(value,output):
    require(Path(output).parent==ROOT/'results','own results directory only')
    require(Path(output).name.startswith(NAME+'_'),'own new result stem')
    # Reuse checked canonical/fsync writer with this module's explicit prefix; never patch another module global.
    p=Path(output);body=saved.canonical(value)+b'\n'
    import os
    import shutil
    require(p.is_absolute() and p.parent.resolve()==p.parent and not p.exists() and not p.is_symlink() and
        len(body)<=protocol()['output_cap_bytes'] and shutil.disk_usage(p.parent).free>=protocol()['disk_floor_bytes']+len(body),'exclusive bounded output/floor')
    with p.open('xb') as f:require(f.write(body)==len(body),'complete write');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(status=value['status'],path=str(p),sha256=sha(p))


def main():
    ap=argparse.ArgumentParser(description=__doc__);sp=ap.add_subparsers(dest='command',required=True)
    a=sp.add_parser('analyze')
    args=('h1-output','h1-report-sha','h1-receipt','h1-receipt-sha','h2-output','h2-report-sha','h2-receipt','h2-receipt-sha','zero-analysis','zero-sha')
    for key in (*args,'output'):a.add_argument('--'+key,required=True)
    v=sp.add_parser('verify');v.add_argument('--analysis',required=True);v.add_argument('--analysis-sha',required=True)
    q=ap.parse_args()
    if q.command=='analyze':result=write_new(analyze(*(getattr(q,k.replace('-','_')) for k in args)),q.output)
    else:
        old=saved.read(q.analysis,q.analysis_sha,{})
        require(old==analyze(**old['arguments']),'complete deterministic factor report independently recomputed')
        result=dict(status='saved_factor_diagnostics_recomputed',analysis_sha256=q.analysis_sha,new_neural_calls=0,new_physics_calls=0)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
