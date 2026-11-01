"""Paired raw-coordinate model/RGB/native contrasts; never an oracle selector."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import main_observed_fd_scalar_cause_20261028_v1 as algebra
from experiments.spectral_reachability import main_observed_raw_fd_recompute_20261028_v1 as model
from experiments.spectral_reachability import main_terminal_witness_physical_contrast_20261028_v1 as native

NAME='main_raw_coordinate_response_20261028_v1'
ROOT=model.ROOT
require=algebra.require


def native_controls(output):
    """Independent native directions while waiting for RGB; no outcomes select actions."""
    new=ROOT/'results/agent_observed_raw_coordinate_physical_20261028_v1_MAIN01/report.json'
    new_sha='c4610c7f495137cbda7d70b147c74670db6f85339561da611ce5bee152d7bb10'
    old=ROOT/'results/agent_nonlinear_suffix_physical_20261028_v1_MAIN01/report.json'
    old_sha='30f5131d6f70abdbe669ce0aef86a430c1ba6593e40149c2402801b6d0c37c8a'
    receipt=ROOT/'results/agent_observed_raw_coordinate_physical_20261028_v1_consume_MAIN01.json'
    receipt_sha='8df105747cf6835056d41ca1b1fd80f7f5b4170f7f527ec538213addc4c9d0bb'
    for f,h in ((new,new_sha),(old,old_sha),(receipt,receipt_sha),model.PINS[0]):
        require(algebra.sha(f)==h,'exact original/native/full physical strict bytes')
    p,r,o,u=map(algebra.read,(model.PREP,new,old,receipt))
    require(len(r['rows'])==164 and r['rows']==u['rows'] and r['errors']==u['errors']==[]
            and u['physical_report_sha256']==new_sha,'complete real164 actual physical receipt')
    metrics=[];baselines=[]
    for i,(row,j) in enumerate(zip(r['rows'],p['jobs'],strict=True)):
        require(row['index']==i and row['plan']['producer_job_sha256']==j['job_sha256']
                and row['plan']['commands']==j['full10'] and row['prefix_reference_exact'] is True
                and row['status']=='horizon_completed' and row['own_returned']==10,'exact all164 real raw/time/cp5')
        metrics.append(native.metrics(row['final_state'],row['goal']))
        if j['local_index']==0:
            slot=j['slot'];prior=o['rows'][slot]
            a=[cp for cp in row['checkpoints'] if cp['step']==10]
            b=[cp for cp in prior['checkpoints'] if cp['step']==10]
            require(len(a)==len(b)==1 and row['plan']['commands']==prior['plan']['commands']
                    and row['final_state']==prior['final_state']
                    and all(a[0][k]==b[0][k] for k in ('state','snapshot','file_sha256','pixel_sha256')),
                    'four baseline complete original state/snapshot/PNG exact replay')
            baselines.append(dict(slot=slot,exact_original_cp10=True,metrics=metrics[-1]))
    panels=[]
    for slot in range(4):
        j=p['jobs'][41*slot];ad=np.asarray(j['original_AD_raw_gradient']).reshape(10)
        raw=np.asarray(j['raw_suffix5']).reshape(10);steps=[]
        for hi,h in enumerate((.01,.001)):
            derivatives={k:[] for k in metrics[0] if k!='native_success'}
            for coord in range(10):
                idx=41*slot+1+20*hi+2*coord;span=p['jobs'][idx]['actual_pair_spacing']
                for key in derivatives:derivatives[key].append((metrics[idx+1][key]-metrics[idx][key])/span)
            steps.append(dict(h=h,native_coordinate_secants=derivatives,
                raw_model_descent_native_secants={k:float(-ad@np.asarray(v)) for k,v in derivatives.items()},
                logit_chain_model_descent_native_secants={k:float(-(ad*(1-raw*raw)**2)@np.asarray(v)) for k,v in derivatives.items()}))
        panels.append(dict(slot=slot,root_slot=j['root_slot'],condition=j['condition'],steps=steps))
    value=dict(schema=NAME+'.native_controls',baselines=baselines,panels=panels,rows=164,
        native_success=sum(v['native_success'] for v in metrics),known_roots=2,fresh_roots=0,
        bindings={str(f):h for f,h in ((new,new_sha),(old,old_sha),(receipt,receipt_sha),model.PINS[0])},
        new_physics_calls=0,new_neural_calls=0,P1_admission=False,full_upstream_rebuilt_here=False,
        scope='All164 real fixed probes; local secants along model-descent directions are diagnostics, not executed policy steps or theoretical guarantees.')
    out=Path(output).absolute()
    require(out.resolve()==out and out.parent==ROOT/'results' and out.name.startswith(NAME+'_'),'exclusive native analysis')
    with out.open('xb') as f:f.write(json.dumps(value,sort_keys=True,allow_nan=False).encode()+b'\n')
    return dict(output=str(out),sha256=algebra.sha(out),baseline_exact=len(baselines),native_success=value['native_success'])


def pair(pm,pp,am,ap,goal,span):
    """Each scale remains separate. Secants do not assume smooth real rendering."""
    result=algebra.scalar_cause(pm,pp,am,ap,goal,np.eye(192).tolist(),span)
    jm=(np.asarray(pp)-np.asarray(pm))/span
    ja=(np.asarray(ap)-np.asarray(am))/span
    den=float(np.linalg.norm(jm)*np.linalg.norm(ja))
    result.update(vector_cosine=float(jm@ja)/den if den else None,
                  response_error_norm=float(np.linalg.norm(jm-ja)))
    return result


def anchored_terms(pm,pp,am,ap,goal,span,pbase,abase):
    pm,pp,am,ap,g,pb,ab=[np.asarray(v,float) for v in (pm,pp,am,ap,goal,pbase,abase)]
    require(all(v.shape==(192,) and np.isfinite(v).all() for v in (pm,pp,am,ap,g,pb,ab))
            and span>0,'whole192 finite anchor-relative vectors')
    e0=pb-ab;em=(pm-am)-e0;ep=(pp-ap)-e0
    common=2*float(np.mean((ap-am)*e0))/span
    response=float(np.mean(2*(ap-g)*ep-2*(am-g)*em+2*e0*(ep-em)+ep*ep-em*em))/span
    gap=float(np.mean((pp-g)**2-(pm-g)**2-(ap-g)**2+(am-g)**2))/span
    residual=gap-common-response
    require(abs(residual)<=1e-9*(1+abs(gap)+abs(common)+abs(response)),'fixed-baseline error identity')
    return dict(common_anchor_term=common,action_response_term=response,identity_residual=residual,
        terms_opposite_sign=bool(common*response<0),constant_offset_for_both_probes=bool(np.all(em==0) and np.all(ep==0)),
        minus_error_change_MSE=float(np.mean(em*em)),plus_error_change_MSE=float(np.mean(ep*ep)))


def build(output,report_sha,receipt,receipt_sha):
    from experiments.spectral_reachability import agent_observed_raw_coordinate_rgb_20261028_v1 as rgb
    for f,h in (*model.PINS,(Path(output)/'report.json',report_sha),(Path(receipt),receipt_sha)):
        require(algebra.sha(f)==h,'exact actual model/RGB/strict bytes')
    mp,mr,_=[algebra.read(f) for f,_ in model.PINS]
    rr=algebra.read(Path(output)/'report.json');u=algebra.read(receipt)
    require(u['schema']==rgb.NAME+'.consume' and u['status']=='raw164_RGB_verified_not_admitted'
            and u['report_sha256']==report_sha and u['fixed_denominator']==164 and len(u['rows'])==164
            and rr['model_unchanged'] is rr['source_verified'] is True and rr['errors']==[], 'complete actual RGB receipt')
    require(algebra.sha(rr['prepared'])==rr['prepare_sha256'],'actual RGB prepare bytes')
    rp=algebra.read(rr['prepared'])
    old_path=ROOT/'results/agent_nonlinear_suffix_physical_20261028_v1_MAIN01/report.json'
    old_sha='30f5131d6f70abdbe669ce0aef86a430c1ba6593e40149c2402801b6d0c37c8a'
    require(algebra.sha(old_path)==old_sha,'original exact baseline physical report')
    old=algebra.read(old_path)
    actual=[];baselines=[]
    for i,(spec,row,accepted,job) in enumerate(zip(rp['rows'],rr['rows'],u['rows'],mp['jobs'],strict=True)):
        require(spec['index']==row['index']==accepted['index']==job['index']==i
                and spec['probe_id']==accepted['probe_id']==job['id']
                and spec['producer_job_sha256']==accepted['producer_job_sha256']==job['job_sha256']
                and spec['original_task']==job['task'] and spec['full10']==job['full10']
                and accepted['goal192']==job['task']['goal'], 'same all164 raw/task/goal/probe identities')
        if row['latent'] is None:
            require(accepted['actual192'] is None and accepted['latent_missing'] is True,'missing stays explicit')
            actual.append(None);continue
        z=rgb.bridge.rgb.load_array(row['latent'],lambda n:(Path(output)/n).read_bytes())
        require(z.dtype==np.float32 and z.shape==(192,) and z.tolist()==accepted['actual192'], 'independent original latent NPY')
        require(float(np.mean((z.astype(float)-np.asarray(job['task']['goal']))**2))==accepted['actual_Q'],'independent actual goalQ')
        actual.append(z.astype(float))
        if job['local_index']==0:
            slot=job['slot'];prior=old['rows'][slot]
            cp=[v for v in prior['checkpoints'] if v['step']==10]
            require(len(cp)==1 and prior['plan']['commands']==job['full10']
                    and spec['source']['state']==prior['final_state']
                    and all(spec['source'][k]==cp[0][k] for k in ('state','snapshot','file_sha256','pixel_sha256')),
                    'baseline actual full10 state/snapshot/PNG reproduces original complete branch')
            baselines.append(dict(slot=slot,exact_original_cp10=True))
    panels=[]
    for slot in range(4):
        jobs=mp['jobs'][41*slot:41*(slot+1)];values=mr['rows'][slot]['values'];steps=[]
        for hi,h in enumerate((.01,.001)):
            comparisons=[]
            for coord in range(10):
                k=1+20*hi+2*coord;idx=41*slot+k
                if actual[idx] is None or actual[idx+1] is None:
                    comparisons.append(dict(coord=coord,available=False,reason='missing actual step10, never padded'));continue
                pm,pp=[v['context']['latent_history'][-1] for v in values[k:k+2]]
                c=pair(pm,pp,actual[idx].tolist(),actual[idx+1].tolist(),jobs[k]['task']['goal'],jobs[k]['actual_pair_spacing'])
                c['anchored']=anchored_terms(pm,pp,actual[idx],actual[idx+1],jobs[k]['task']['goal'],
                    jobs[k]['actual_pair_spacing'],values[0]['context']['latent_history'][-1],actual[41*slot])
                s0,s1=[u['rows'][i]['evaluation']['final_state'] for i in (idx,idx+1)]
                metrics=native.pair(s0,s1,rp['rows'][idx]['native_goal'])
                pixels_same=rp['rows'][idx]['source']['pixel_sha256']==rp['rows'][idx+1]['source']['pixel_sha256']
                states_same=s0==s1
                latents_same=np.array_equal(actual[idx],actual[idx+1])
                c.update(coord=coord,available=True,native=metrics,
                    native_states_identical=states_same,RGB_pixels_identical=pixels_same,
                    actual_latents_identical=bool(latents_same),
                    state_change_not_visible_in_RGB=bool(not states_same and pixels_same),
                    identical_RGB_but_different_latents=bool(pixels_same and not latents_same),
                    different_RGB_but_identical_latents=bool(not pixels_same and latents_same))
                comparisons.append(c)
            available=[c for c in comparisons if c['available']]
            if len(available)==10:
                gm=np.array([c['model_scalar_fd'] for c in available]);ga=np.array([c['actual_scalar_fd'] for c in available])
                ad=np.asarray(jobs[0]['original_AD_raw_gradient']).reshape(10)
                a=np.asarray(jobs[0]['raw_suffix5']).reshape(10)
                # These are diagnostic directional secants, not real policy steps.
                slope_raw=float(-ad@ga);slope_logit=float(-(ad*(1-a*a)**2)@ga)
                model_actual_cos=model.vector_metrics(gm,ga)['cosine']
            else:slope_raw=slope_logit=model_actual_cos=None
            steps.append(dict(h=h,comparisons=comparisons,available_pairs=len(available),
                model_actual_scalar_cosine=model_actual_cos,
                raw_descent_actual_directional_secant=slope_raw,
                logit_chain_descent_actual_directional_secant=slope_logit))
        panels.append(dict(slot=slot,root_slot=jobs[0]['root_slot'],condition=jobs[0]['condition'],steps=steps))
    require(len(baselines)==4,'all4 original baseline controls available')
    allpairs=[c for p in panels for s in p['steps'] for c in s['comparisons'] if c['available']]
    return dict(schema=NAME,panels=panels,baseline_controls=baselines,fixed_requests=164,fixed_pairs=80,
        available_pairs=len(allpairs),strict_direction_reversals=sum(c['strict_direction_reversal'] for c in allpairs),
        opposing_term_pairs=sum(c['opposing_terms'] for c in allpairs),known_roots=2,fresh_roots=0,
        opposing_fixed_anchor_terms=sum(c['anchored']['terms_opposite_sign'] for c in allpairs),
        nonconstant_anchor_offset_pairs=sum(not c['anchored']['constant_offset_for_both_probes'] for c in allpairs),
        report_sha256=report_sha,receipt_sha256=receipt_sha,original_physical_sha256=old_sha,
        new_neural_calls=0,new_physics_calls=0,P1_admission=False,full_upstream_rebuilt_here=False,
        scope='Pinned full strict receipts plus independent NPY/arithmetic/baseline check. Descriptive ambient rawbox diagnosis; not a deployment correction, independent confirmation or unique causal proof.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('rgb-output','report-sha','receipt','receipt-sha','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.output).absolute()
    require(out.resolve()==out and out.parent==ROOT/'results' and out.name.startswith(NAME+'_'),'exclusive own result')
    value=build(a.rgb_output,a.report_sha,a.receipt,a.receipt_sha)
    with out.open('xb') as f:f.write(json.dumps(value,sort_keys=True,allow_nan=False).encode()+b'\n')
    print(json.dumps(dict(output=str(out),sha256=algebra.sha(out),available=value['available_pairs'],reversals=value['strict_direction_reversals'])))


if __name__=='__main__':main()
