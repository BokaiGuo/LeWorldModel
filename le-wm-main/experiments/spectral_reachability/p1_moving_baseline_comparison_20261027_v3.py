"""Saved moving2 six-baseline assembler; no producer/model/physics execution.

Frozen v2 handles existing disjoint physical/RGB cohorts. New providers use
their OWN extended saved adapter; null downstream is never zero success.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_moving_baseline_comparison_20261027_v2 as prior

base=prior.base
NAME='p1_moving_baseline_comparison_20261027_v3'
ROOT,CAP,FLOOR=base.ROOT,prior.CAP,prior.FLOOR
require,fields,sha,canonical=base.require,base.fields,base.sha,base.canonical
actual,missing=base.actual,base.missing
PRIOR_SHA='2464d72928056358d860ad68f7bdb855884a860efa8ab39f50c8773a3a3974aa'
EXT_ADAPTER='p1_moving_extended_baseline_rgb_cuda_20261027_v1'
EXT_ADAPTER_SHA='da384d180254858238a50c0ecc602313204947060041b45597d43e6195f4ef9a'
EXT_PHYSICAL_SHA='a0f7348ebe8c6e061b5d63a1ba4a8ecb064a26e7f7a1ef4be57fe8b07270d799'
PROVIDERS={
 'grasp':dict(module='p1_moving_grasp_baseline_cuda_20261027_v1',arm='GRASP_H3_bounded',
    sha='52d3157574e4e2726f6d88b43bd7e5dc804ce79730b65512888e63e0dd7c7106',
    report_sha='7593940c1503416f42ce87a5d8135e8512763121e027c0ef98dbe614bfb31f91',
    status='moving_grasp_model_plans_sealed_not_admitted',
    budget=dict(predict=1802,action_encoder=1802,backward=900),ad_forward=1800,reverse=900),
 'no_composition':dict(module='p1_moving_no_composition_baseline_cuda_20261027_v1',arm='joint_H2_spectral_packet_mapped',
    sha='8c709bed26d88e093b0293482be07cfc34702fb095b2fe23a22edadbbb985963',
    report_sha='ec39956ca0c90cc5dbf955b1a65bf300b68d166627a748a82314a2176a481c34',
    status='moving_spectral_plans_sealed_not_admitted',
    budget=dict(predict=88,action_encoder=88,jacobian=1,vjp_directions=192,svd=2,bounded_solve=1),ad_forward=2,reverse=192)}
EXTRA_ARMS=tuple(s['arm'] for s in PROVIDERS.values())
BASELINES=prior.COHORT+(prior.KINO,)+EXTRA_ARMS
RESOURCE_KEYS=('predict','action_encoder','backward','grad_API','jacobian','jacobian_API','svd','pred_projector','bounded_solve')


def check_sources(entries):
    require(isinstance(entries,list) and [e.get('provider') for e in entries]==list(PROVIDERS),'both new providers in predeclared order')
    for e in entries:
        require(e.get('state') in ('complete','partial','missing'),'explicit model state')
        keys='provider state arms reason' if e['state']=='missing' else 'provider state arms report consume' if e['state']=='complete' else 'provider state arms report reason'
        fields(e,keys,'new model source');spec=PROVIDERS[e['provider']]
        require(e['arms']==[spec['arm']],'own arm only, no duplicate/result filtering')
        if e['state']=='complete':require(e['report']['sha256']==spec['report_sha'],'fixed declared actual model invocation')
        else:require(type(e['reason']) is str and 0<len(e['reason'])<=500,'explicit bounded missing/partial reason')


def manifest_check(m):
    fields(m,'schema purpose existing additional_sources new_downstream','v3 manifest')
    require(m['schema']==NAME+'.manifest' and m['purpose']=='fixed_known_moving2_six_baselines','own v3 purpose')
    prior.manifest_check(m['existing'])
    require(m['existing']['cohort'] in [list(prior.COHORT+(prior.KINO,)+old) for old in ((),prior.LEGACY)],'all original four baseline arms predeclared; legacy pair optional')
    check_sources(m['additional_sources'])
    check_deferred(m['new_downstream'])


def deferred():
    return dict(arms=list(EXTRA_ARMS),physical=dict(state='missing',reason='extended physical not supplied'),
        rgb=dict(state='missing',reason='extended RGB not supplied'))


def check_deferred(entries):
    fields(entries,'arms physical rgb','extended BOTH-arm downstream cohort')
    require(entries['arms']==list(EXTRA_ARMS),'both new arms in predeclared order; no outcome subset')
    for k in ('physical','rgb'):
        e=entries[k];require(e.get('state') in ('complete','partial','missing'),'explicit extended state')
        keys='state reason' if e['state']=='missing' else 'state report prepare_sha256'+(' source_sha256' if k=='physical' else '')+(' reason' if e['state']=='partial' else '')
        fields(e,keys,'extended '+k)
        if e['state']!='complete':require(type(e['reason']) is str and 0<len(e['reason'])<=500,'explicit missing/partial reason')
        if e['state']!='missing':
            base.hash_value(e['prepare_sha256'])
            if k=='physical':require(e['source_sha256']==EXT_PHYSICAL_SHA,'own extended physical source, NOT old provider')
    if entries['rgb']['state']=='complete':require(entries['physical']['state']=='complete','extended RGB requires same physical chain')


def extended_adapter():
    path=ROOT/'experiments/spectral_reachability'/(EXT_ADAPTER+'.py')
    require(sha(path)==EXT_ADAPTER_SHA,'frozen EXTENDED adapter, never widen old RGB authority')
    module=importlib.import_module(base.PREFIX+EXT_ADAPTER)
    require(Path(module.__file__).resolve()==path.resolve(),'extended module identity');return module


def load_extended_physical(e):
    if e['state']=='missing':return dict(state='missing',entry=deepcopy(e))
    r=base.load_ref(e['report'])
    if e['state']=='partial':return dict(state='partial',entry=deepcopy(e),report=r)
    module=extended_adapter();p=module.build(physical_report=e['report']['path'],physical_sha=e['report']['sha256'],
        physical_prepare_sha=e['prepare_sha256'],physical_source_sha=e['source_sha256'],arms=list(EXTRA_ARMS))
    require(r['schema']=='p1_moving_extended_baseline_physical_20261027_v1.report' and r['status']=='moving_extended_baseline_physical_recorded' and
        p['arguments']['physical_sha']==e['report']['sha256'] and p['arguments']['physical_prepare_sha']==e['prepare_sha256'],'own actual physical chain')
    return dict(state='complete',entry=deepcopy(e),report=r,prepared=p,images=p['images'])


def extended_rgb_receipts(r,p,events,arrays,pixels,compare):
    images=p['images'];n=len(images)
    require(n==8 and len(r['rows'])==len(arrays)==len(pixels)==n and len(events)==7*n,'ALL8 extended images and7 receipts each')
    require(r['resources']=={k:dict(attempted=n,completed=n) for k in ('encoder','projector')},'actual8 encoder/projector')
    out=[];prefix={}
    for i,(item,row,z,a) in enumerate(zip(images,r['rows'],arrays,pixels,strict=True)):
        require(row['index']==i and row['status']=='encoded' and row['provenance']==item,'own actual image mapping')
        for k in ('branch','root_slot','source_slot','arm','role','task_id','task_export','task_sha256','blind_sha256'):require(row[k]==item[k],'exact extended RGB '+k)
        cp=item['checkpoint'];meta=dict(file=f'image_{i:02d}.rgb.png',sha256=cp['file_sha256'],pixel_sha256=cp['pixel_sha256'],dtype='uint8',shape=[224,224,3])
        require(row['rawpixel']==meta and a.dtype==np.uint8 and a.shape==(224,224,3),'own raw PNG metadata')
        group=events[7*i:7*i+7]
        expected=[dict(kind='image_attempt',provenance=item),dict(kind='raw_image_saved',index=i,receipt=meta)]
        for j,k in enumerate(('encoder','projector')):
            for phase,completed in (('module_attempt',False),('module_return',True)):
                expected.append(dict(kind=phase,entry=dict(index=2*i+j,image=i,resource=k,completed=completed)))
        expected.append(dict(kind='image_saved',receipt=row));require(group==expected,'durable raw PNG BEFORE exact module dispatch')
        metrics=compare(item,z);key=item['source_slot'],item['arm'],item['selection_digest']
        if item['role']=='prefix5':
            require(key not in prefix,'one own prefix');prefix[key]=(a,metrics['actual_Q']);pixel_mse=0.;change=0.
        else:
            require(item['role']=='terminal10' and key in prefix,'terminal after own prefix')
            pixel_mse=float(np.mean((a.astype(np.float64)-prefix[key][0].astype(np.float64))**2));change=metrics['actual_Q']-prefix[key][1]
        metrics.update(rawpixel_vs_prefix5_mse=pixel_mse,actual_Q_change_from_prefix5=change,rawpixel_mean=float(np.mean(a.astype(np.float64))))
        require(row['metrics']==metrics,'raw NPY AND PNG metrics recomputed')
        out.append(dict(item=item,metrics=metrics,latent=deepcopy(row['latent'])))
    return out


def load_extended_rgb(e,physical):
    if e['state']=='missing':return dict(state='missing',entry=deepcopy(e))
    r=base.load_ref(e['report'])
    if e['state']=='partial':return dict(state='partial',entry=deepcopy(e),report=r)
    require(physical['state']=='complete','own extended physical required');module=extended_adapter()
    require(r['schema']==EXT_ADAPTER+'.report' and r['status']=='moving_extended_baseline_rgb_encoded_not_admitted' and
        r['errors']==[] and r['first_error'] is None and r['source_verified'] is True and r['model_unchanged'] is True and
        r['P1_admission'] is r['P2_admission'] is False,'complete own extended RGB')
    require(r['prepare_sha256']==e['prepare_sha256'] and sha(r['prepared'])==e['prepare_sha256'],'actual RGB prepare external SHA')
    p=module.verify(r['prepared'],e['prepare_sha256'])
    require(p['arguments']==r['inputs']==physical['prepared']['arguments'] and p['images']==physical['images'] and p['contract']==r['contract'],'same exact extended physical/PNG/source scope')
    require(r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and r['fixed_image_denominator']==8 and r['fixed_branch_denominator']==4 and
        r['new_predict']==r['new_AD']==r['new_physics']==0 and r['training'] is False,'actual frozen encoding-only budget')
    output=Path(e['report']['path']).parent;names={'binding.json','events.jsonl'}|{f'image_{i:02d}.{suffix}' for i in range(8) for suffix in ('rgb.png','native192.npy')}
    require(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'},'exact extended18 artifact census')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'actual extended artifact SHA')
    require(base.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=e['prepare_sha256']),'own execution binding')
    raw=(output/'events.jsonl').read_bytes();require(len(raw)<=8*1024**2 and raw.endswith(b'\n'),'bounded complete extended event journal')
    events=[base.json.loads(line) for line in raw.splitlines()];canonical(events)
    arrays=[prior.latent_file(output,row['latent']) for row in r['rows']];pixels=[]
    for i,row in enumerate(r['rows']):
        require(row['rawpixel']['file']==f'image_{i:02d}.rgb.png','own flat raw PNG file')
        path=output/row['rawpixel']['file'];require(path.stat().st_size<=1024**2,'bounded original PNG')
        pixels.append(module.rgb.old.png(path,row['rawpixel']['sha256'],row['rawpixel']['pixel_sha256']))
    receipts=extended_rgb_receipts(r,p,events,arrays,pixels,module.compare)
    for path,h in p['bindings'].items():require(sha(path)==h,'final extended saved seal')
    return dict(state='complete',entry=deepcopy(e),report=r,prepared=p,receipts=receipts)


def provider_bindings(report):
    ref=dict(path=report['prepared'],sha256=report['prepare_sha256']);p=base.load_ref(ref)
    bindings=deepcopy(p['bindings_sha256']);bindings[ref['path']]=ref['sha256']
    return bindings


def load_new_sources(entries):
    check_sources(entries);out=[]
    for entry in entries:
        b=dict(entry=deepcopy(entry));spec=PROVIDERS[entry['provider']]
        if entry['state']!='missing':b['report']=base.load_ref(entry['report'])
        if entry['state']=='complete':
            path=ROOT/'experiments/spectral_reachability'/(spec['module']+'.py')
            require(sha(path)==spec['sha'],'frozen ownconsume source')
            module=importlib.import_module(base.PREFIX+spec['module'])
            require(Path(module.__file__).resolve()==path.resolve(),'ownconsume module identity')
            c=base.load_ref(entry['consume']);saved=module.consume(Path(entry['report']['path']).parent,entry['report']['sha256'])
            require(c==saved,'exact original saved ownconsume JSON, no adapter metadata invention');b['consume']=c
            b['bindings']=provider_bindings(b['report'])
            for name,h in b['report'].get('artifact_sha256',{}).items():
                require(Path(name).name==name and name not in ('','.','..'),'flat ownconsume artifact')
                path=str(Path(entry['report']['path']).parent/name)
                require(path not in b['bindings'] or b['bindings'][path]==h,'nonconflicting model artifact seal');b['bindings'][path]=h
        out.append(b)
    return out


def exact_budget(raw,budget):
    require(isinstance(raw,dict) and set(raw)==set(budget),'exact native resource keys')
    for k,n in budget.items():require(base.counter(raw[k])==dict(attempted=n,completed=n),'complete actual resource '+k)


def new_resources(provider,row):
    spec=PROVIDERS[provider];raw=row['resources'];exact_budget(raw,spec['budget'])
    out={k:base.count_cell(raw,k,'actual ownconsume-verified operation receipts') for k in RESOURCE_KEYS}
    zero=('jacobian','jacobian_API','svd','bounded_solve') if provider=='grasp' else ('jacobian_API',)
    for k in zero:out[k]=actual(dict(attempted=0,completed=0),'frozen algorithm/API absence contract; not separately metered')
    out['forward_samples']=deepcopy(out['predict'])
    out['AD_forward_in_predict']=actual(spec['ad_forward'],'ownconsumer verified fixed schedule; subset ALREADY in predict')
    out['mathematical_reverse_directions']=actual(spec['reverse'],'scalar loss reverse sweeps for GRASP; scalar output VJPs for spectral, not equal work')
    out['raw_resources']=deepcopy(raw);out['raw_predict_API_resources']=deepcopy(row.get('predict_API_resources',{}))
    out['raw_AD_API_resources']=deepcopy(row.get('AD_API_resources',{}))
    out['additive_warning']='backward/VJP/grad/Jacobian overlap; ADforward already in predict; missing hooks are NOT zero'
    if provider=='grasp':
        exact_budget(row['predict_API_resources'],dict(predict=1802))
    return out


def validate_new_plan(task,spec,plan,report_sha):
    require(plan['schema']==spec['module']+'.model_plan' and plan['arm']==spec['arm'] and type(plan['source_slot']) is int and
        plan['source_slot']==task['source_slot'],'own plan schema/slot/arm')
    for k in ('task_id','task_export','task_sha256','task_contract_sha256','blind_sha256'):
        require(plan[k]==task[k],'exact original nonzero moving task '+k)
    require(plan['with_target'] is False and plan['physical_execution'] is False and plan['P1_admission'] is False and
        plan['whole_run_report_sha256']==report_sha,'model-only approved invocation')
    if spec is PROVIDERS['grasp']:require(plan['selection_uses_physics'] is False,'GRASP blind choice')
    else:
        require(plan['encoder_report_sha256']==base.ENCODER_SHA and
            plan['source_authority_sha256']=='c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6','spectral original authority')
    raw=np.asarray(plan['commands']);require(raw.shape==(10,2) and raw.dtype.kind in 'fi' and np.isfinite(raw).all() and np.abs(raw).max()<=1,'literal raw10x2, no padding')
    prediction=plan['prediction'];require(prediction['layout']==task['root_context']['layout'],'same prediction layout')
    z=np.asarray(prediction['latent_history'][-1]);require(z.shape==(192,) and np.isfinite(z).all(),'finite native192 terminal')
    q=float(np.mean((z-np.asarray(task['goal']))**2))
    require(base.number(plan['model_terminal_cost']) and q==plan['model_terminal_cost'],'saved model Q arithmetic, not native success')
    base.hash_value(plan['selection_sha256']);return q,base.digest(plan['commands'])


def append_new(result,tasks,bundles):
    check_sources([b['entry'] for b in bundles]);require(len(tasks)==2,'both original roots')
    for b in bundles:
        e=b['entry'];provider=e['provider'];spec=PROVIDERS[provider];arm=spec['arm'];state=e['state'];run_id=base.digest(e)
        group=[base.row_blank(t,arm,dict(physical=dict(state='missing',reason='new provider physical not connected'),rgb=dict(state='missing',reason='new provider RGB not connected'))) for t in tasks]
        run=dict(source_id=run_id,provider=provider,arms=[arm],state=state,provenance=deepcopy(e),contract=None,
            latency_seconds=missing('whole invocation elapsed time not metered','unmeasured'))
        if state!='missing':
            r=b['report'];require(r['schema']==spec['module']+'.report','own new source report schema');run['contract']=deepcopy(r['contract'])
            require(r['contract'].get('with_target',False) is False,'no target-on new provider contract')
            require(len(r['rows'])==2 and [(x['source_slot'],x['task_id']) for x in r['rows']]==[(s,tasks[s]['task_id']) for s in (0,1)] and
                all(type(x['source_slot']) is int for x in r['rows']),'all fixed roots, no replacement/shrunk denominator')
            elapsed=r.get('limits',{}).get('elapsed_seconds')
            if elapsed is not None:
                require(base.number(elapsed) and elapsed>=0,'finite invocation latency');run['latency_seconds']=actual(elapsed,'WHOLE original model invocation; do not divide among roots/arms')
            if state=='complete':
                require(r['status']==spec['status'] and r['commands_released'] is True and r['errors']==[] and r['first_error'] is None and
                    r['source_verified'] is True and r['model_unchanged'] is True and r['model_hash_before']==r['model_hash_after'] and r['P1_admission'] is False,'actual own complete CUDA closure')
                base.hash_value(r['model_hash_before'])
                require(r['contract']['per_root_budget']==spec['budget'] and r['contract']['total_budget']=={k:2*v for k,v in spec['budget'].items()},'same frozen model budget contract')
                exact_budget(r['resources'],{k:2*v for k,v in spec['budget'].items()})
                c=b['consume'];require(c['schema']==spec['module']+'.consume' and c['P1_admission'] is False and
                    len(c['rows'])==2 and [x['source_slot'] for x in c['rows']]==[0,1],'ownconsume fixed denominator')
            else:require(r['status']!=spec['status'] or r.get('errors') or r.get('commands_released') is not True,'complete release cannot be relabeled partial')
        for s,row in enumerate(group):
            row.update(source_id=run_id,contract=deepcopy(run['contract']),physical_status='missing',rgb_status='missing',
                full_horizon=missing('no connected actual physical receipt'))
            if state=='complete':
                raw=b['report']['rows'][s];require(raw['status']=='model_plan_sealed' and raw.get('arm',arm)==arm,'actual model row/own arm')
                q,h=validate_new_plan(tasks[s],spec,b['consume']['rows'][s],e['report']['sha256'])
                row.update(model_status='actual',model_Q=actual(q,'own selected native192 fullH2 model prediction, NOT actualQ/native'),
                    commands_sha256=actual(h,'canonical selected literal commands digest'),resources=actual(new_resources(provider,raw),'own native resource ledger'),source_row_status=raw['status'])
            elif state=='partial':
                raw=b['report']['rows'][s];counters=raw.get('resources',{})
                for v in counters.values():base.counter(v)
                row.update(model_status='partial',source_row_status=raw['status'],model_Q=missing(e['reason'],'partial'),
                    resources=dict(status='partial',value=None,reason='unreleased counters, not complete budget',reported_counters=deepcopy(counters)))
            else:row['model_Q']=missing(e['reason'])
        complete=all(row['model_status']=='actual' for row in group)
        enc=result['inherited_encoding_shared_once']['value']
        summary=dict(arm=arm,expected_roots=2,actual_model_roots=2 if complete else 0,partial_model_roots=2 if state=='partial' else 0,
            mean_model_Q=actual(sum(x['model_Q']['value'] for x in group)/2,'both original fixed roots') if complete else missing('both-root mean unavailable'),
            native_successes=missing('physical not connected, NOT 0 failures'),native_denominator=2,actual_physical_roots=0,actual_RGB_roots=0,
            full_horizons=missing('physical not connected'),evaluation_encoding=missing('no actual new provider RGB'),
            standalone_inherited_encoding=actual(enc,'original8/8 per standalone TWO-ROOT algorithm; do not add to shared ledger'),
            resources={k:base.aggregate_counts(group,k) for k in RESOURCE_KEYS})
        for k in ('realQ_root','realQ_prefix5','realQ_terminal10'):summary['mean_'+k]=missing('no connected physical/RGB evidence')
        result['rows'].extend(group);result['summary'].append(summary);result['runs'].append(run)
    return result


def assemble(existing,tasks,bundles,extended=None):
    require(existing['schema']==prior.NAME+'.comparison' and existing['cohort'] in
        [list(prior.COHORT+(prior.KINO,)+old) for old in ((),prior.LEGACY)],'own four-baseline v2 RAM result, optional core pair')
    base.validate_tasks(tasks);r=append_new(deepcopy(existing),tasks,bundles)
    if extended is None:extended={k:dict(state='missing',entry=e) for k,e in deferred().items() if k!='arms'}
    models={(p['source_slot'],p['arm']):dict(plan=p,report_sha256=b['entry']['report']['sha256']) for b in bundles if b['entry']['state']=='complete' for p in b['consume']['rows']}
    prior.attach_physical(r,tasks,models,extended['physical'],EXTRA_ARMS);prior.attach_rgb(r,extended['rgb'],EXTRA_ARMS)
    r['new_downstream']=dict(arms=list(EXTRA_ARMS),**{k:deepcopy(p['entry']) for k,p in extended.items()})
    for summary in r['summary']:
        if summary['arm'] not in EXTRA_ARMS:continue
        group=[x for x in r['rows'] if x['arm']==summary['arm']];ap=[x for x in group if x['physical_status']=='actual'];ar=[x for x in group if x['rgb_status']=='actual']
        summary.update(actual_physical_roots=len(ap),actual_RGB_roots=len(ar),
            native_successes=actual(sum(x['native_success']['value'] for x in group),'both fixed native outcomes') if len(ap)==2 else missing('both native outcomes unavailable'),
            full_horizons=actual(sum(x['full_horizon']['value'] for x in group),'both fixed horizons') if len(ap)==2 else missing('both physical horizons unavailable'),
            evaluation_encoding=actual({k:dict(attempted=4,completed=4) for k in ('encoder','projector')},'actual two endpoints x two roots, separately paid') if len(ar)==2 else missing('actual extended RGB incomplete'))
        for k in ('realQ_root','realQ_prefix5','realQ_terminal10'):
            summary['mean_'+k]=actual(sum(x[k]['value'] for x in group)/2,'BOTH original roots') if all(x[k]['value'] is not None for x in group) else missing('both fixed actual Q unavailable')
    for p in extended.values():
        if p['state']=='partial':r['unreleased_downstream'].append(dict(entry=p['entry'],reported_status=p.get('report',{}).get('status'),reported_resources=deepcopy(p.get('report',{}).get('counts',p.get('report',{}).get('resources')))))
    r.update(schema=NAME+'.comparison',cohort=existing['cohort']+list(EXTRA_ARMS),fixed_baselines=list(BASELINES))
    n=len(r['cohort']);r['denominator'].update(algorithms=n,expected_model_rows=2*n,expected_physical_rows=2*n,expected_RGB_images=4*n,
        actual_model_rows=sum(x['model_status']=='actual' for x in r['rows']),actual_physical_rows=sum(x['physical_status']=='actual' for x in r['rows']),
        actual_RGB_images=2*sum(x['rgb_status']=='actual' for x in r['rows']))
    r['model_layer_complete']=all(x['model_status']=='actual' for x in r['rows'])
    complete=all(x['model_status']==x['physical_status']==x['rgb_status']=='actual' for x in r['rows'])
    r['status']='comparison_recorded_not_admitted' if complete else 'model_complete_downstream_partial_not_admitted' if r['model_layer_complete'] else 'partial_comparison_not_admitted'
    r['warnings']+=['GRASP bounded moving extension NOT official reproduction; H3=3 history states, future H2',
        'spectral joint40 mapped packet balls changes joint-domain support; unnormalized ridge .002, mean192 equivalent .002/192',
        'extended physical/RGB only from OWN frozen adapter; absent/partial evidence stays null, modelQ never actualQ']
    for s in r['summary']:
        group=[x for x in r['rows'] if x['arm']==s['arm']]
        for k in ('AD_forward_in_predict','mathematical_reverse_directions'):
            vals=[x['resources']['value'][k]['value'] if x['resources']['value'] is not None else None for x in group]
            s[k]=actual(sum(vals),'both fixed roots, descriptive overlapping schedule NOT added compute') if all(v is not None for v in vals) else missing('complete derivative schedule unavailable')
    canonical(r);return r


def template():
    e=prior.template();e['cohort'].append(prior.KINO)
    e['sources'].append(next(x for x in base.template()['sources'] if x['provider']=='kino'))
    e['additional_cohorts']=[dict(arms=[prior.KINO],
        physical=dict(state='complete',report=dict(path=str(ROOT/'results/p1_moving_baseline_physical_20261027_v1_kino_main01/report.json'),sha256='2db7370214bf62810c21ad8778c1d58bc397dbe8f08c9f5ef1db435468abc655'),
            prepare_sha256='4f03963d6f3b5915072cecf3464d7c94cdf9d58ac76f9bfc8b0ecae717b21fbf',source_sha256=prior.PHYSICAL_SOURCE_SHA),
        rgb=dict(state='complete',report=dict(path=str(ROOT/'results/p1_moving_baseline_rgb_cuda_20261027_v1_kino_cuda_main01/report.json'),sha256='71dafeeebd1d2bd2567028e65800f4f27f479c72b066da3d2ace6a35ef8e7869'),
            prepare_sha256='12ad9280e68625e9f8a9850a72fa1e769fe3f1014bd541b1934d0dcfe592767e'))]
    down=deferred();down['physical']=dict(state='complete',report=dict(path=str(ROOT/'results/p1_moving_extended_baseline_physical_20261027_v1_main01/report.json'),
        sha256='f05ce7e2d773bb046c6c8ba905c9e86c367c7e19ad09174d4509e1823a297038'),
        prepare_sha256='9819cd6857e043492d27a239728fb3eee6d6ed5542546796467bf0ae83425fd7',source_sha256=EXT_PHYSICAL_SHA)
    return dict(schema=NAME+'.manifest',purpose='fixed_known_moving2_six_baselines',existing=e,new_downstream=down,
        additional_sources=[dict(provider=p,state='missing',arms=[s['arm']],reason='MAIN填实际report及原样ownconsume绝对路径/外部SHA') for p,s in PROVIDERS.items()])


def markdown(r):
    text=prior.markdown(r)
    lines=['','六基线固定原两根；GRASP/no-composition使用独立extended cohort，物理/RGB missing不作0。','',
        '| 算法 | predict完成/尝试 | backward完成/尝试 | J完成/尝试 | SVD完成/尝试 | ADforward子集 | reverse方向/标量次数（不可横比） |',
        '|---|---:|---|---|---|---:|---:|']
    c=base.md_cell
    for s in r['summary']:
        lines.append('| '+' | '.join([s['arm']]+[c(s['resources'][k]) for k in ('predict','backward','jacobian','svd')]+[c(s[k]) for k in ('AD_forward_in_predict','mathematical_reverse_directions')])+' |')
    lines+=['','GRASP非官方等价；谱bank映射改变支持域。共享编码8/8一次，评价编码另列；不等域、不等预算、不据此排名。','']
    return text+'\n'.join(lines)


def run(manifest,manifest_sha,output):
    require(sha(prior.__file__)==PRIOR_SHA and sha(base.__file__)==prior.BASE_SHA,'frozen v1/v2 sources')
    ref=dict(path=str(Path(manifest).absolute()),sha256=manifest_sha);m=base.load_ref(ref);manifest_check(m);e=m['existing']
    tasks=base.task_inputs();encoder=base.load_ref(e['encoder']);old=base.load_sources(e);new=load_new_sources(m['additional_sources'])
    ph=prior.load_physical(e['physical']);rgb=prior.load_rgb(e['rgb'],ph);lp=lr=None;extras=[]
    if e['legacy']['state']=='complete':
        lp=prior.load_physical(dict(state='complete',**e['legacy']['physical']),True);lr=prior.load_rgb(e['legacy']['rgb'],lp,True)
    for x in e['additional_cohorts']:
        p=prior.load_physical(x['physical'],arms=x['arms']);z=prior.load_rgb(x['rgb'],p)
        extras.append(dict(arms=x['arms'],physical=p,rgb=z))
    ep=load_extended_physical(m['new_downstream']['physical']);er=load_extended_rgb(m['new_downstream']['rgb'],ep)
    r=assemble(prior.assemble(tasks,encoder,old,e['cohort'],ph,rgb,lp,lr,extras),tasks,new,dict(physical=ep,rgb=er))
    bindings={str(Path(prior.__file__).resolve()):PRIOR_SHA,str(Path(base.__file__).resolve()):prior.BASE_SHA,
        str(ROOT/'experiments/spectral_reachability'/(base.LOADER+'.py')):base.LOADER_SHA}
    bindings.update({t['task_export']:t['task_sha256'] for t in tasks})
    for path in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):bindings[str(path.resolve())]=sha(path)
    for bundles,specs in ((old,base.PROVIDERS),(new,PROVIDERS)):
        for b in bundles:
            if b['entry']['state']=='complete':
                spec=specs[b['entry']['provider']]
                bindings[str(ROOT/'experiments/spectral_reachability'/(spec['module']+'.py'))]=spec['sha']
                for path,h in b.get('bindings',{}).items():
                    require(path not in bindings or bindings[path]==h,'nonconflicting model closure');bindings[path]=h
    packages=[ph,rgb,lp,lr,ep,er]+[p for x in extras for p in (x['physical'],x['rgb'])]
    for p in packages:
        if p and p['state']=='complete':
            for path,h in p['prepared']['bindings'].items():
                require(path not in bindings or bindings[path]==h,'nonconflicting saved bindings');bindings[path]=h
    r.update(manifest=ref,bindings_sha256=bindings)
    data=canonical(r);md=markdown(r).encode();require(len(data)+len(md)<=CAP,'<=1MiB combined output')
    base.reference(ref);base.reference(e['encoder'])
    for b in old+new:
        for k in ('report','consume'):
            if k in b['entry']:base.reference(b['entry'][k])
    for p in packages:
        if p and p['state']!='missing':base.reference(p['entry']['report'])
    for path,h in bindings.items():require(sha(path)==h,'final saved source/task seal')
    path=Path(output).absolute();require(not path.exists() and shutil.disk_usage(path.parent).free>=FLOOR+len(data)+len(md),'NEW output +200MiB floor')
    path.mkdir()
    for name,raw in (('comparison.json',data),('comparison.md',md)):
        require(shutil.disk_usage(path).free>=FLOOR+len(raw),'per-write reserve')
        with (path/name).open('xb') as f:require(f.write(raw)==len(raw),'short write');f.flush();os.fsync(f.fileno())
    return dict(schema=NAME+'.output',status=r['status'],files={str(path/n):sha(path/n) for n in ('comparison.json','comparison.md')},P1_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('template','assemble'))
    for k in ('manifest','manifest-sha','output'):p.add_argument('--'+k)
    a=p.parse_args();print(canonical(template() if a.command=='template' else run(a.manifest,a.manifest_sha,a.output)).decode(),end='')


if __name__=='__main__':main()
