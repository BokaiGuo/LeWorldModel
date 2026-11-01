"""Moving2 predeclared disjoint-cohort model/physical/RGB result assembler.

Saved-only Jason build/verify reuse; no producer execution or neural fallback.
v1 stays frozen. Missing is not zero; native success is not model/real Q.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_moving_baseline_comparison_20261027_v1 as base

NAME='p1_moving_baseline_comparison_20261027_v2'
ROOT=base.ROOT
require,fields,sha,read,canonical=base.require,base.fields,base.sha,base.read,base.canonical
actual,missing=base.actual,base.missing
COHORT=('moving_gradient_open_loop','CEM','kernel_faithful_mppi_nominal')
KINO='fixed_H2_kinodynamic_beam16x4'
LEGACY=base.ARMS[:2]
CAP,FLOOR=1024**2,200*1024**2
BASE_SHA='3a1141d4e33fecb3f3235c5526481c4c76e5ba1f0a9e5d5a3ba95a5fd8f486e3'
ADAPTER='p1_moving_baseline_rgb_cuda_20261027_v1'
ADAPTER_SHA='bd174c810c22579862fd2707e24de58ed9fafd45327543a1c1f7da24397f0865'
OLD_ADAPTER='p1_constructive_moving_history_rgb_cuda_20261027_v1'
OLD_ADAPTER_SHA='ed9b80bebb212434d60ddd55d3a3cba01e5cc55306170a0ea6f9a167bb156469'
PHYSICAL_SOURCE_SHA='19b367debdfcc47032de8828dd0ab7124171c553d6c3f7b0a225ca86820cc3e0'
PHYSICAL_REPORT=ROOT/'results/p1_moving_baseline_physical_20261027_v1_main01/report.json'
PHYSICAL_SHA='6a9f4cf602a328cf5cb0fbe82296fc989a109a6859287c2efb703bebf44511db'
PHYSICAL_PREPARE_SHA='2e7304f11420ed65b6f3f039dd49522fee042e30bf6e97c82db791addd3fa50e'
RGB_REPORT=ROOT/'results/p1_moving_baseline_rgb_cuda_20261027_v1_cuda_main01/report.json'
RGB_SHA='ab549f1d1a75c88b64a1c8c8697b004f8e6bf84d603b6e350406cf439fa305cd'
RGB_PREPARE_SHA='24a20378230ce64052f81c72cd91755309e426f7fa8bf2cd5f390761c826a0f0'


def adapter(legacy=False):
    name,h=(OLD_ADAPTER,OLD_ADAPTER_SHA) if legacy else (ADAPTER,ADAPTER_SHA)
    path=ROOT/'experiments/spectral_reachability'/(name+'.py')
    require(sha(path)==h,'frozen saved adapter source');module=importlib.import_module(base.PREFIX+name)
    require(Path(module.__file__).resolve()==path.resolve(),'own adapter module identity');return module


def pending():return {k:dict(state='schema_unfrozen',reason='model-layer reuse; v2 downstream attached separately') for k in ('physical','rgb')}


def checked_cohort(cohort):
    require(cohort in [list(COHORT+extra) for extra in ((),(KINO,),LEGACY,(KINO,)+LEGACY)],'predeclared3 plus optional independent kino and/or both legacy arms')


def check_entry(e,kind):
    require(isinstance(e,dict) and e.get('state') in ('complete','partial','missing'),'explicit '+kind+' state')
    if e['state']=='missing':fields(e,'state reason',kind);require(type(e['reason']) is str and bool(e['reason']),'missing reason');return
    keys='state report prepare_sha256'
    if kind=='physical':keys+=' source_sha256'
    if e['state']=='partial':keys+=' reason'
    fields(e,keys,kind);base.hash_value(e['prepare_sha256'])
    if kind=='physical':require(e['source_sha256']==PHYSICAL_SOURCE_SHA,'new physical frozen source')


def manifest_check(m):
    fields(m,'schema purpose cohort encoder sources physical rgb legacy additional_cohorts','own v2 manifest')
    require(m['schema']==NAME+'.manifest' and m['purpose']=='fixed_known_moving2_physical_RGB_comparison','own v2 purpose')
    checked_cohort(m['cohort'])
    model=dict(schema=base.NAME+'.manifest',purpose='fixed_known_moving2_model_comparison',encoder=m['encoder'],sources=m['sources'],downstream=pending())
    base.manifest_check(model)
    require(all(a in m['cohort'] for e in m['sources'] for a in e['arms']),'model source arms must belong to declared cohort')
    check_entry(m['physical'],'physical');check_entry(m['rgb'],'rgb')
    if m['rgb']['state']=='complete':require(m['physical']['state']=='complete','actual RGB requires its actual physical chain')
    old=m['legacy'];require(isinstance(old,dict) and old.get('state') in ('missing','complete'),'legacy state')
    if old['state']=='missing':
        fields(old,'state reason','legacy');require(not any(a in m['cohort'] for a in LEGACY),'no legacy inclusion without saved source/task/plan proof')
    else:
        fields(old,'state physical rgb','legacy');require(all(a in m['cohort'] for a in LEGACY),'legacy cohort must be declared BEFORE assembly')
        fields(old['physical'],'report prepare_sha256','legacy physical');base.hash_value(old['physical']['prepare_sha256'])
        check_entry(old['rgb'],'rgb')
        require(any(e['provider']=='moving' and e['state']=='complete' and e['arms']==list(LEGACY) for e in m['sources']),'legacy requires own actual GN/D2P model authority')
    require(isinstance(m['additional_cohorts'],list) and len(m['additional_cohorts'])<=1,'bounded explicitly predeclared additional cohorts')
    declared=[]
    for extra in m['additional_cohorts']:
        fields(extra,'arms physical rgb','additional physical/RGB cohort')
        require(extra['arms']==[KINO] and KINO in m['cohort'] and not declared,'kino only once, never duplicate original three arms')
        check_entry(extra['physical'],'physical');check_entry(extra['rgb'],'rgb')
        if extra['rgb']['state']=='complete':require(extra['physical']['state']=='complete','extra RGB requires same physical cohort')
        declared.extend(extra['arms'])
    require((KINO in m['cohort'])==(declared==[KINO]),'explicit kino physical/RGB state even when missing; no implicit drop')


def load_physical(entry,legacy=False,arms=None):
    """Jason saved build validates physical source, literal plans, journal/PNG.

    Does NOT call physical.verify/material/execute (which would load sources
    beyond the saved adapter), and does not create a prepare file.
    """
    if entry['state']=='missing':return dict(state='missing',entry=deepcopy(entry))
    ref=entry['report'];r=base.load_ref(ref)
    if entry['state']=='partial':return dict(state='partial',entry=deepcopy(entry),report=r)
    module=adapter(legacy)
    if legacy:
        p=module.build(physical_report=ref['path'],physical_sha=ref['sha256'],physical_prepare_sha=entry['prepare_sha256'])
    else:
        p=module.build(physical_report=ref['path'],physical_sha=ref['sha256'],physical_prepare_sha=entry['prepare_sha256'],
            physical_source_sha=entry['source_sha256'],arms=list(COHORT if arms is None else arms))
    require(p['arguments']['physical_sha']==ref['sha256'] and p['arguments']['physical_prepare_sha']==entry['prepare_sha256'],'same external physical chain')
    return dict(state='complete',entry=deepcopy(entry),report=r,prepared=p,images=p['images'],legacy=legacy)


def latent_file(output,meta):
    fields(meta,'file sha256 dtype shape','saved native latent')
    require(type(meta['file']) is str and Path(meta['file']).name==meta['file'] and meta['dtype']=='float32' and meta['shape']==[192],'flat native192 latent file')
    p=output/meta['file'];require(p.stat().st_size<=4096 and sha(p)==meta['sha256'],'bounded latent SHA')
    with p.open('rb') as f:z=np.load(f,allow_pickle=False)
    require(z.dtype==np.float32 and z.shape==(192,) and np.isfinite(z).all(),'native finite raw float32 NPY')
    return z


def rgb_receipts(r,p,events,arrays,compare):
    """Pure saved receipt mapping and arithmetic; no encoder execution."""
    images=p['images'];n=len(images)
    require(len(r['rows'])==n==4*len(p['contract']['arms']) and len(arrays)==n,'all requested RGB images, no smaller denominator')
    require(r['resources']=={k:dict(attempted=n,completed=n) for k in ('encoder','projector')},'actual encoder/projector full denominator')
    require(len(events)==6*n,'image-attempt + four native module receipts + saved image')
    out=[]
    for index,(item,row,z) in enumerate(zip(images,r['rows'],arrays,strict=True)):
        require(row['status']=='encoded' and row['provenance']==item and row['index']==index,'actual prepared image identity')
        for k in ('branch','root_slot','source_slot','arm','role','task_id','task_export','task_sha256','blind_sha256'):
            require(row[k]==item[k],'same physical/RGB/task scope '+k)
        group=events[6*index:6*index+6]
        require(group[0]==dict(kind='image_attempt',provenance=item) and group[-1]==dict(kind='image_saved',receipt=row),'durable image lifecycle')
        expected=[]
        for j,k in enumerate(('encoder','projector')):
            for phase,completed in (('module_attempt',False),('module_return',True)):
                expected.append(dict(kind=phase,entry=dict(index=2*index+j,image=index,resource=k,completed=completed)))
        require(group[1:5]==expected,'actual module receipts, no estimate/dedup')
        metrics=compare(item,z);require(row['metrics']==metrics,'actual saved NPY metrics recomputed')
        out.append(dict(item=item,metrics=metrics,latent=deepcopy(row['latent'])))
    return out


def load_rgb(entry,physical,legacy=False):
    if entry['state']=='missing':return dict(state='missing',entry=deepcopy(entry))
    r=base.load_ref(entry['report'])
    if entry['state']=='partial':return dict(state='partial',entry=deepcopy(entry),report=r)
    require(physical['state']=='complete','RGB cannot float without physical')
    module=adapter(legacy);require(r['schema']==module.NAME+'.report' and r['errors']==[] and r['first_error'] is None and
        r['source_verified'] is True and r['model_unchanged'] is True and r['P1_admission'] is False,'complete actual RGB report')
    expected_status='moving_selected_rgb_encoded_not_admitted' if legacy else 'moving_baseline_rgb_encoded_not_admitted'
    require(r['status']==expected_status,'own RGB completion status')
    require(r['prepare_sha256']==entry['prepare_sha256'] and sha(r['prepared'])==entry['prepare_sha256'],'external actual RGB prepare SHA')
    p=module.verify(r['prepared'],entry['prepare_sha256'])  # PURE saved chain only.
    require(p['arguments']==r['inputs']==physical['prepared']['arguments'] and p['images']==physical['images'],'exact RGB/physical inputs, checkpoint12 and arm cohort')
    require(r['contract']==p['contract'] and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'],'same frozen model/config')
    require(r['new_predict']==r['new_AD']==r['new_physics']==0 and r['training'] is False,'encoding only, not new rollout')
    if not legacy:require(r['fixed_image_denominator']==len(p['images']) and r['fixed_branch_denominator']==2*len(p['contract']['arms']),'fixed actual baseline image/branch denominator')
    output=Path(entry['report']['path']).parent;names={'binding.json','events.jsonl'}|{f'image_{i:02d}.native192.npy' for i in range(len(p['images']))}
    require(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'},'own exact RGB artifact census')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'RGB artifact SHA')
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=entry['prepare_sha256']),'own RGB execution binding')
    raw=(output/'events.jsonl').read_bytes();require(len(raw)<=8*1024**2 and raw.endswith(b'\n'),'bounded complete RGB receipt JSONL')
    events=[json.loads(line) for line in raw.splitlines()];canonical(events)
    arrays=[latent_file(output,row['latent']) for row in r['rows']]
    receipts=rgb_receipts(r,p,events,arrays,module.compare)
    for f,h in p['bindings'].items():require(sha(f)==h,'final saved RGB input seal')
    require(sha(entry['report']['path'])==entry['report']['sha256'],'actual RGB report final SHA')
    return dict(state='complete',entry=deepcopy(entry),report=r,receipts=receipts,prepared=p)


def selected_models(bundles):
    selected={}
    for b in bundles:
        if b['entry']['state']!='complete':continue
        spec=base.PROVIDERS[b['entry']['provider']]
        for plan in b['consume'][spec['consume_rows']]:
            key=plan['source_slot'],plan['arm'];require(key not in selected,'unique model selection')
            selected[key]=dict(plan=plan,report_sha256=b['entry']['report']['sha256'])
    return selected


def attach_physical(result,tasks,models,package,allowed):
    if package['state']!='complete':
        reason=package['entry'].get('reason','unreleased partial physical record')
        for row in result['rows']:
            if row['arm'] in allowed:
                row['physical_status']=package['state'];row['native_success']=missing(reason, 'partial' if package['state']=='partial' else 'missing')
        return
    report=package['report'];images=package['images'];expected=[(s,a) for s in (0,1) for a in allowed]
    require(report['contract']['arms']==list(allowed) and report['contract']['with_target'] is False and report['errors']==[] and
        report['sourcepack_sha256']=='c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6','same original moving source/no-target physical')
    require([(r['root_slot'],r['arm']) for r in report['rows']]==expected and len(images)==2*len(expected),'complete physical declared denominator')
    lookup={(x['source_slot'],x['arm']):x for x in result['rows']}
    require([(i['root_slot'],i['arm'],i['role']) for i in images]==[(s,a,role) for s,a in expected for role in ('prefix5','terminal10')],'prefix5/terminal10, no substituted images')
    source_id=base.digest(package['entry'])
    for index,physical_row in enumerate(report['rows']):
        s,a=expected[index];row=lookup[s,a];require((s,a) in models and row['model_status']=='actual','physical must match a released model selection')
        chosen=models[s,a];plan=chosen['plan'];task=tasks[s];pair=images[2*index:2*index+2]
        require(type(physical_row['native_success']) is bool and type(physical_row['full_horizon']) is bool and physical_row['initial_exact'] is True,'actual native boolean/full horizon receipt')
        require(physical_row['status'] in ('horizon_completed','native_success') and physical_row['full_horizon'] is True and
            physical_row['own_returned']==physical_row['own_attempted']==10 and physical_row['first_error'] is None,'complete10, not successful merely by horizon')
        require((physical_row['status']=='native_success')==physical_row['native_success'],'native status/boolean agree')
        for item in pair:
            require(item['task_id']==task['task_id'] and item['task_export']==task['task_export'] and item['task_sha256']==task['task_sha256'] and
                item['blind_sha256']==task['blind_sha256'] and item['selection_digest']==physical_row['selection_digest'] and item['Q']=='identity192' and
                item['root']==task['root_context']['latent_history'][-1] and item['goal']==task['goal'],'exact same task/source/metric/root/goal join')
            if package.get('legacy',False):
                require(item['planner_selection_sha256']==plan['selection_sha256'] and package['prepared']['arguments']['planner_sha']==chosen['report_sha256'],'old GN/D2P exact selected same-command authority')
            else:
                # SAME default-spaces digest as physical.source.digest; NOT
                # base.digest's compact/newline canonical representation.
                own=base.hashlib.sha256(json.dumps(plan,sort_keys=True,allow_nan=False).encode()).hexdigest()
                require(item['compiler_report_sha256']==chosen['report_sha256'] and item['own_plan_sha256']==own,'physical exact ownconsume plan incl literal commands, not just same arm/task')
        row.update(physical_status='actual',physical_source_id=source_id,native_success=actual(physical_row['native_success'],'actual native receipt; NOT Q threshold'),
            full_horizon=actual(physical_row['full_horizon'],'own returned10 and literal commands via saved adapter'),
            physical_returned_steps=actual(physical_row['own_returned'],'actual own public-step returns'),
            physical_selection_digest=actual(physical_row['selection_digest'],'actual predeclared physical plan seal'),
            realQ_root=actual(float(np.mean((np.asarray(pair[0]['root'])-np.asarray(pair[0]['goal']))**2)), 'original OBSERVED task root/goal encoder latents; physical root RGB exact matched; no new encode'))
    require(report['native_successes']==sum(r['native_success'] for r in report['rows']) and report['full_horizons']==len(expected),'whole physical native/full-horizon census')
    result['physical_runs'].append(dict(source_id=source_id,provenance=package['entry'],arms=list(allowed),counts=report['counts'],
        latency_seconds=actual(report['wall_seconds'],'WHOLE physical run incl reset/internal physics/render; never divide among arms'),
        expected_branches=len(expected),actual_branches=len(expected),native_successes=report['native_successes'],full_horizons=report['full_horizons']))


def attach_rgb(result,package,allowed):
    rows={(r['source_slot'],r['arm']):r for r in result['rows']}
    if package['state']!='complete':
        for row in rows.values():
            if row['arm'] in allowed:
                row['rgb_status']=package['state']
                for k in ('realQ_prefix5','realQ_terminal10'):row[k]=missing(package['entry'].get('reason','partial RGB not released'),'partial' if package['state']=='partial' else 'missing')
        return
    receipts=package['receipts'];expected=[(s,a,role) for s in (0,1) for a in allowed for role in ('prefix5','terminal10')]
    require([(x['item']['source_slot'],x['item']['arm'],x['item']['role']) for x in receipts]==expected,'ALL RGB rows with original denominator/order')
    for e in receipts:
        item,metrics=e['item'],e['metrics'];row=rows[item['source_slot'],item['arm']]
        require(row['physical_status']=='actual' and item['selection_digest']==row['physical_selection_digest']['value'],'RGB belongs to actual same physical plan')
        require(metrics['root_Q']==row['realQ_root']['value'] and base.number(metrics['actual_Q']) and metrics['actual_Q']>=0,'same observed root Q/actual finite RGB Q')
        row['realQ_'+item['role']]=actual(metrics['actual_Q'],'actual native192 NPY re-evaluation; own view, no predicted-Q substitution')
        row['rgb_status']='actual';row.setdefault('RGB_latents',{})[item['role']]=e['latent']
    result['RGB_runs'].append(dict(provenance=package['entry'],arms=list(allowed),resources=package['report']['resources'],
        image_denominator=len(expected),actual_images=len(receipts),new_predict=0,new_AD=0,new_physics=0,
        latency_seconds=missing('not separately metered by RGB producer','unmeasured')))


def assemble(tasks,encoder,bundles,cohort,physical,rgb,legacy_physical=None,legacy_rgb=None,additional=None):
    checked_cohort(list(cohort));additional=[] if additional is None else additional
    r=base.assemble(tasks,encoder,bundles,pending());r['rows']=[x for a in cohort for x in r['rows'] if x['arm']==a]
    r['summary']=[x for a in cohort for x in r['summary'] if x['arm']==a];r.update(schema=NAME+'.comparison',cohort=list(cohort),physical_runs=[],RGB_runs=[])
    for row in r['rows']:row.update(physical_status='missing',rgb_status='missing',full_horizon=missing('no physical receipt'))
    models=selected_models(bundles);attach_physical(r,tasks,models,physical,COHORT);attach_rgb(r,rgb,COHORT)
    seen=set(COHORT)
    for extra in additional:
        require(extra['arms']==[KINO] and KINO in cohort and KINO not in seen,'no duplicate/cohort overlap/original three rerun')
        seen.add(KINO);attach_physical(r,tasks,models,extra['physical'],extra['arms']);attach_rgb(r,extra['rgb'],extra['arms'])
    require((KINO in cohort)==(KINO in seen),'kino declared with explicit complete/partial/missing cohort, not result-filtered')
    if all(a in cohort for a in LEGACY):
        require(legacy_physical is not None and legacy_physical['state']=='complete' and legacy_rgb is not None,'legacy cannot enter without strict source/task adapter')
        attach_physical(r,tasks,models,legacy_physical,LEGACY);attach_rgb(r,legacy_rgb,LEGACY)
    for summary in r['summary']:
        group=[x for x in r['rows'] if x['arm']==summary['arm']]
        actual_ph=[x for x in group if x['physical_status']=='actual'];actual_rgb=[x for x in group if x['rgb_status']=='actual']
        summary.update(actual_physical_roots=len(actual_ph),actual_RGB_roots=len(actual_rgb),native_denominator=2,
            native_successes=actual(sum(x['native_success']['value'] for x in group),'both fixed native outcomes') if len(actual_ph)==2 else missing('not all two native outcomes available'),
            full_horizons=actual(sum(x['full_horizon']['value'] for x in group),'both fixed physical horizons') if len(actual_ph)==2 else missing('not all horizons available'))
        for key in ('realQ_root','realQ_prefix5','realQ_terminal10'):
            summary['mean_'+key]=actual(sum(x[key]['value'] for x in group)/2,'arithmetic mean over BOTH original fixed roots') if all(x[key]['value'] is not None for x in group) else missing('two-root actual RGB metric unavailable; no denominator shrinking')
        summary['evaluation_encoding']=actual({k:dict(attempted=4,completed=4) for k in ('encoder','projector')},'two endpoints x two roots actual module receipts, separate from inherited8/8') if len(actual_rgb)==2 else missing('whole algorithm evaluation encoding not complete')
    r['denominator']=dict(algorithms=len(cohort),roots_per_algorithm=2,expected_model_rows=2*len(cohort),
        actual_model_rows=sum(x['model_status']=='actual' for x in r['rows']),expected_physical_rows=2*len(cohort),
        actual_physical_rows=sum(x['physical_status']=='actual' for x in r['rows']),expected_RGB_images=4*len(cohort),
        actual_RGB_images=2*sum(x['rgb_status']=='actual' for x in r['rows']))
    r['status']='comparison_recorded_not_admitted' if all(x['model_status']==x['physical_status']==x['rgb_status']=='actual' for x in r['rows']) else 'partial_comparison_not_admitted'
    r['downstream']=dict(physical=physical['entry'],rgb=rgb['entry']);r['warnings']=[
        'fixed known moving2, P1/P2 false; no independent confirmation/theory failure conclusion',
        'native success, full horizon, modelQ and realQ are DIFFERENT metrics; no Q/native threshold substitution',
        'nominal MPPI is local nonofficial; budgets/action domains differ; forward count NOT equal total compute',
        'shared inherited8/8 ONCE; endpoint evaluation encoding separately paid; standalone per-algorithm8/8 not summed',
        'all latencies retain invocation scope; do not divide or duplicate among roots/arms',
        'missing/partial remain unknown, not zero success or zero resource']
    r['unreleased_downstream']=[dict(entry=p['entry'],reported_status=p.get('report',{}).get('status'),
        reported_resources=deepcopy(p.get('report',{}).get('counts',p.get('report',{}).get('resources'))))
        for p in [physical,rgb,legacy_physical,legacy_rgb]+[p for e in additional for p in (e['physical'],e['rgb'])] if p and p['state']=='partial']
    r['additional_cohorts']=[dict(arms=e['arms'],physical=e['physical']['entry'],rgb=e['rgb']['entry']) for e in additional]
    return r


def markdown(r):
    c=base.md_cell;lines=['# Moving 固定两根基线比较 v2','',
        '预声明互斥cohort、各含固定两根；已知根，非独立确认，P1/P2未通过。不等总算力；本地nominal MPPI非官方。','',
        '| 算法 | model完成/2 | 平均modelQ | forward完成/尝试 | native成功/2 | 完整10步/2 | RGB根数/2 | actualQ root→5→10 |',
        '|---|---:|---:|---:|---|---|---:|---|']
    for s in r['summary']:
        q=' → '.join(c(s['mean_'+k]) for k in ('realQ_root','realQ_prefix5','realQ_terminal10'))
        lines.append('| '+' | '.join([s['arm'],str(s['actual_model_roots'])+'/2',c(s['mean_model_Q']),c(s['resources']['predict']),
            c(s['native_successes'])+'/2',c(s['full_horizons'])+'/2',str(s['actual_RGB_roots'])+'/2',q])+' |')
    lines+=['','native=0是实际失败计数，仅在两根实际结果齐全时显示；missing不作0。完整10步不等于成功，Q改善不等于成功。','',
        '每根数据、反向/Jacobian/SVD、完整算法合同和来源SHA见comparison.json。共享原编码8/8只记一次；新端点评价编码另计。','',
        '| 物理运行 | 实际分支 | native成功 | public/physics/render完成 | 整次秒数（不均摊） |', '|---|---:|---:|---|---:|']
    for p in r['physical_runs']:
        counts=p['counts'];numbers='/'.join(str(counts[k]['completed']) for k in ('public_step','physics_step','explicit_render'))
        lines.append(f"| {p['source_id'][:12]} | {p['actual_branches']} | {p['native_successes']} | {numbers} | {c(p['latency_seconds'])} |")
    lines+=['','无SOTA/官方复现/等预算宣称；不能由当前结果推导理论失败。',''];return '\n'.join(lines)


def template():
    b=base.template();sources=[]
    for e in b['sources']:
        if e['provider'] in ('sampling','gradient'):sources.append(e)
    return dict(schema=NAME+'.manifest',purpose='fixed_known_moving2_physical_RGB_comparison',cohort=list(COHORT),encoder=b['encoder'],sources=sources,
        physical=dict(state='complete',report=dict(path=str(PHYSICAL_REPORT),sha256=PHYSICAL_SHA),prepare_sha256=PHYSICAL_PREPARE_SHA,source_sha256=PHYSICAL_SOURCE_SHA),
        rgb=dict(state='complete',report=dict(path=str(RGB_REPORT),sha256=RGB_SHA),prepare_sha256=RGB_PREPARE_SHA),
        legacy=dict(state='missing',reason='default cohort excludes old GN/D2P; only add both with strict saved adapter proof'),additional_cohorts=[])


def run(manifest,manifest_sha,output):
    require(sha(base.__file__)==BASE_SHA,'frozen v1 reuse, no edits')
    ref=dict(path=str(Path(manifest).absolute()),sha256=manifest_sha);m=base.load_ref(ref);manifest_check(m)
    tasks=base.task_inputs();encoder=base.load_ref(m['encoder']);bundles=base.load_sources(m)
    physical=load_physical(m['physical']);rgb=load_rgb(m['rgb'],physical)
    lp=lr=None
    if m['legacy']['state']=='complete':
        lp=load_physical(dict(state='complete',**m['legacy']['physical']),True);lr=load_rgb(m['legacy']['rgb'],lp,True)
    additional=[]
    for extra in m['additional_cohorts']:
        ep=load_physical(extra['physical'],arms=extra['arms']);er=load_rgb(extra['rgb'],ep)
        additional.append(dict(arms=extra['arms'],physical=ep,rgb=er))
    result=assemble(tasks,encoder,bundles,m['cohort'],physical,rgb,lp,lr,additional)
    bindings={str(Path(base.__file__).resolve()):BASE_SHA,
        str(ROOT/'experiments/spectral_reachability'/(base.LOADER+'.py')):base.LOADER_SHA}
    bindings.update({t['task_export']:t['task_sha256'] for t in tasks})
    for b in bundles:
        if b['entry']['state']=='complete':
            spec=base.PROVIDERS[b['entry']['provider']]
            bindings[str(ROOT/'experiments/spectral_reachability'/(spec['module']+'.py'))]=spec['sha']
    for p in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):bindings[str(p.resolve())]=sha(p)
    for package in [physical,rgb,lp,lr]+[p for e in additional for p in (e['physical'],e['rgb'])]:
        if package and package['state']=='complete':
            for p,h in package['prepared']['bindings'].items():
                require(p not in bindings or bindings[p]==h,'nonconflicting saved provenance');bindings[p]=h
    # Model ownconsumers already bound their entire source/task/weight closure.
    result.update(manifest=ref,bindings_sha256=bindings)
    data=canonical(result);md=markdown(result).encode();require(len(data)+len(md)<=CAP,'<=1MiB result output')
    base.reference(ref);base.reference(m['encoder'])
    for b in bundles:
        for k in ('report','consume'):
            if k in b['entry']:base.reference(b['entry'][k])
    entries=[m['physical'],m['rgb']]+[p for e in m['additional_cohorts'] for p in (e['physical'],e['rgb'])]
    if m['legacy']['state']=='complete':entries+=[dict(state='complete',**m['legacy']['physical']),m['legacy']['rgb']]
    for e in entries:
        if e['state']!='missing':base.reference(e['report'])
    for p,h in bindings.items():require(sha(p)==h,'final saved/source SHA')
    path=Path(output).absolute();require(not path.exists() and shutil.disk_usage(path.parent).free>=FLOOR+len(data)+len(md),'NEW directory +200MiB floor')
    path.mkdir()
    for n,raw in (('comparison.json',data),('comparison.md',md)):
        require(shutil.disk_usage(path).free>=FLOOR+len(raw),'per-write reserve')
        with (path/n).open('xb') as f:require(f.write(raw)==len(raw),'short output');f.flush();os.fsync(f.fileno())
    return dict(schema=NAME+'.output',status=result['status'],files={str(path/n):sha(path/n) for n in ('comparison.json','comparison.md')},P1_admission=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('template','assemble'))
    for k in ('manifest','manifest-sha','output'):parser.add_argument('--'+k)
    a=parser.parse_args();r=template() if a.command=='template' else run(a.manifest,a.manifest_sha,a.output)
    print(canonical(r).decode(),end='')


if __name__=='__main__':main()
