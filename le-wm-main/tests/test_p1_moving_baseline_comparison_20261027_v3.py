"""RAM fixtures only; no production assemble/consumer/prepare/neural/physics."""
from copy import deepcopy
from pathlib import Path
import json
import runpy
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import p1_moving_baseline_comparison_20261027_v3 as m


def fixture():
    defs=runpy.run_path(str(Path(__file__).with_name('test_p1_moving_baseline_comparison_20261027_v2.py')))
    f,extra=defs['kino_fixture']();existing=m.prior.assemble(*f,additional=[extra]);tasks=f[0];bundles=[]
    for provider,spec in m.PROVIDERS.items():
        entry=dict(provider=provider,state='complete',arms=[spec['arm']],report=dict(path=f'/ram/{provider}/report.json',sha256=spec['report_sha']),
            consume=dict(path=f'/ram/{provider}/consume.json',sha256='b'*64))
        rows=[];plans=[]
        for s,task in enumerate(tasks):
            row=dict(source_slot=s,task_id=task['task_id'],status='model_plan_sealed',resources={k:dict(attempted=n,completed=n) for k,n in spec['budget'].items()})
            if provider=='grasp':row.update(arm=spec['arm'],predict_API_resources=dict(predict=dict(attempted=1802,completed=1802)))
            rows.append(row)
            p={k:task[k] for k in ('source_slot','task_id','task_export','task_sha256','task_contract_sha256','blind_sha256')}
            p.update(schema=spec['module']+'.model_plan',arm=spec['arm'],commands=[[.1,.1]]*10,prediction=deepcopy(task['root_context']),
                with_target=False,physical_execution=False,P1_admission=False,whole_run_report_sha256=spec['report_sha'],selection_sha256='c'*64)
            p['prediction']['latent_history'][-1]=[.5]*192;p['model_terminal_cost']=.25
            if provider=='grasp':p['selection_uses_physics']=False
            else:p.update(encoder_report_sha256=m.base.ENCODER_SHA,source_authority_sha256='c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6')
            plans.append(p)
        contract=dict(per_root_budget=spec['budget'],total_budget={k:2*v for k,v in spec['budget'].items()},with_target=False,horizon=2)
        report=dict(schema=spec['module']+'.report',status=spec['status'],rows=rows,contract=contract,commands_released=True,
            errors=[],first_error=None,source_verified=True,model_unchanged=True,model_hash_before='d'*64,model_hash_after='d'*64,P1_admission=False,
            resources={k:dict(attempted=2*n,completed=2*n) for k,n in spec['budget'].items()})
        if provider=='grasp':report['limits']=dict(elapsed_seconds=63.25)
        bundles.append(dict(entry=entry,report=report,consume=dict(schema=spec['module']+'.consume',P1_admission=False,rows=plans)))
    return existing,tasks,bundles


def test_six_actual_models_new_downstream_missing_not_native_zero():
    r=m.assemble(*fixture());assert len(r['rows'])==12 and len(r['summary'])==6
    assert r['fixed_baselines']==list(m.BASELINES) and r['model_layer_complete'] is True
    assert r['status']=='model_complete_downstream_partial_not_admitted'
    assert r['denominator']==dict(algorithms=6,roots_per_algorithm=2,expected_model_rows=12,actual_model_rows=12,
        expected_physical_rows=12,actual_physical_rows=8,expected_RGB_images=24,actual_RGB_images=16)
    assert len(r['physical_runs'])==len(r['RGB_runs'])==2
    for s in r['summary']:
        assert s['actual_model_roots']==2
        if s['arm'] in m.EXTRA_ARMS:
            assert s['native_successes']['value'] is None and s['native_successes']['status']=='missing'
            assert s['mean_realQ_root']['value'] is s['mean_realQ_terminal10']['value'] is s['full_horizons']['value'] is None
        else:assert s['native_successes']['value']==0
    assert json.loads(m.canonical(r))==r
    assert all(r[k] is False for k in ('P1_admission','P2_admission','independent_confirmation','official_reproduction','equal_total_compute','equal_search_domain'))


def test_all_six_budgets_no_AD_double_count_no_compute_equivalence():
    r=m.assemble(*fixture());s={x['arm']:x for x in r['summary']}
    assert {a:v['resources']['predict']['value']['completed'] for a,v in s.items()}==dict(zip(m.BASELINES,(272,268,144,168,3604,176)))
    grasp=s[m.EXTRA_ARMS[0]];spectral=s[m.EXTRA_ARMS[1]]
    assert grasp['resources']['backward']['value']['completed']==1800
    assert grasp['AD_forward_in_predict']['value']==3600 and grasp['mathematical_reverse_directions']['value']==1800
    assert grasp['resources']['jacobian']['value']==dict(attempted=0,completed=0)
    assert spectral['AD_forward_in_predict']['value']==4 and spectral['mathematical_reverse_directions']['value']==384
    assert spectral['resources']['jacobian']['value']['completed']==2 and spectral['resources']['svd']['value']['completed']==4
    assert spectral['resources']['bounded_solve']['value']['completed']==2
    assert spectral['resources']['backward']['value'] is spectral['resources']['grad_API']['value'] is None
    assert all(x['resources']['value']['forward_samples']==x['resources']['value']['predict'] for x in r['rows'])


def test_shared_encoding_once_standalone_and_latency_not_copied():
    r=m.assemble(*fixture())
    assert r['inherited_encoding_shared_once']['value']['encoder']['completed']==8
    assert all(s['standalone_inherited_encoding']['value']['encoder']['completed']==8 for s in r['summary'])
    assert [p['resources']['encoder']['completed'] for p in r['RGB_runs']]==[12,4]
    runs={x['provider']:x for x in r['runs']}
    assert runs['grasp']['latency_seconds']['value']==63.25 and runs['no_composition']['latency_seconds']['value'] is None
    assert all(x['latency_seconds']['value'] is None for x in r['rows'])


@pytest.mark.parametrize('provider',[0,1])
@pytest.mark.parametrize('state',['missing','partial'])
def test_explicit_new_model_missing_partial_no_Q_or_completed_budget(provider,state):
    f=list(fixture());b=f[2][provider];e=b['entry'];e.pop('consume');e.update(state=state,reason='not released')
    if state=='missing':e.pop('report')
    else:
        b['report'].update(status='partial_or_failed',commands_released=False,errors=['interrupted'])
        b['report']['rows'][0]['resources']['predict']=dict(attempted=3,completed=2)
    r=m.assemble(*f);assert not r['model_layer_complete'] and r['denominator']['actual_model_rows']==10
    group=[x for x in r['rows'] if x['arm']==m.EXTRA_ARMS[provider]]
    assert all(x['model_Q']['value'] is None and x['resources']['value'] is None and x['native_success']['value'] is None for x in group)
    if state=='partial':assert group[0]['resources']['reported_counters']['predict']['completed']==2


@pytest.mark.parametrize('edit',['task_id','task_export','task_sha256','task_contract_sha256','blind_sha256','source_slot','arm','whole_run_report_sha256',
    'goal_cost','terminal','with_target','physical_execution','denominator','budget','top_budget','model_hash','status','old_schema','invocation'])
def test_wrong_new_binding_or_budget_rejected(edit):
    f=list(fixture());b=f[2][1];p=b['consume']['rows'][0]
    if edit in ('task_id','task_sha256','task_contract_sha256','blind_sha256','whole_run_report_sha256'):p[edit]='e'*64
    elif edit in ('arm','task_export'):p[edit]='wrong'
    elif edit=='source_slot':p[edit]=True
    elif edit=='goal_cost':p['model_terminal_cost']=0.
    elif edit=='terminal':p['prediction']['latent_history'][-1]=[.2]*191
    elif edit in ('with_target','physical_execution'):p[edit]=True
    elif edit=='denominator':b['consume']['rows'].pop()
    elif edit=='budget':b['report']['rows'][0]['resources']['predict']['completed']=87
    elif edit=='top_budget':b['report']['resources']['predict']['completed']=175
    elif edit=='model_hash':b['report']['model_hash_after']='e'*64
    elif edit=='status':b['report']['commands_released']=False
    elif edit=='old_schema':b['report']['schema']='old_zero.report'
    else:b['entry']['report']['sha256']='e'*64
    with pytest.raises(ValueError):m.assemble(*f)


@pytest.mark.parametrize('edit',['missing_root','replaced_root','false_grasp_arm','grasp_predict_API','nocomp_authority','command_padding','nan'])
def test_more_denominator_and_blind_failures(edit):
    f=list(fixture());b=f[2][0];p=b['consume']['rows'][0]
    if edit=='missing_root':b['report']['rows'].pop()
    elif edit=='replaced_root':b['report']['rows'][1]['task_id']=b['report']['rows'][0]['task_id']
    elif edit=='false_grasp_arm':b['report']['rows'][0]['arm']='CEM'
    elif edit=='grasp_predict_API':b['report']['rows'][0]['predict_API_resources']['predict']['completed']=1801
    elif edit=='nocomp_authority':f[2][1]['consume']['rows'][0]['source_authority_sha256']='e'*64
    elif edit=='command_padding':p['commands']=p['commands'][:5]
    else:p['prediction']['latent_history'][-1][0]=float('nan')
    with pytest.raises(ValueError):m.assemble(*f)


def test_template_full_cohorts_and_new_providers_no_outcome_selection():
    v=m.template();m.manifest_check(v)
    assert v['existing']['cohort']==list(m.prior.COHORT+(m.prior.KINO,))
    assert [e['state'] for e in v['additional_sources']]==['missing','missing']
    assert v['existing']['additional_cohorts'][0]['rgb']['report']['sha256']=='71dafeeebd1d2bd2567028e65800f4f27f479c72b066da3d2ace6a35ef8e7869'
    v['additional_sources'].reverse()
    with pytest.raises(ValueError,match='predeclared'):m.manifest_check(v)


def test_saved_ownconsumer_loader_no_producer_entry(monkeypatch):
    _,_,bundles=fixture();entries=[b['entry'] for b in bundles];calls=[]
    refs={b['entry'][k]['path']:b['report' if k=='report' else 'consume'] for b in bundles for k in ('report','consume')}
    monkeypatch.setattr(m.base,'load_ref',lambda ref:deepcopy(refs[ref['path']]))
    monkeypatch.setattr(m,'provider_bindings',lambda report:{})
    monkeypatch.setattr(m,'sha',lambda p:next(s['sha'] for s in m.PROVIDERS.values() if str(p).endswith(s['module']+'.py')))
    def imported(name):
        spec=next(s for s in m.PROVIDERS.values() if name==m.base.PREFIX+s['module'])
        b=next(b for b in bundles if b['entry']['arms']==[spec['arm']])
        def consume(output,expected):calls.append((str(output),expected));return deepcopy(b['consume'])
        return SimpleNamespace(__file__=str(m.ROOT/'experiments/spectral_reachability'/(spec['module']+'.py')),consume=consume)
    monkeypatch.setattr(m.importlib,'import_module',imported)
    assert len(m.load_new_sources(entries))==2 and len(calls)==2
    assert calls[0][1]==m.PROVIDERS['grasp']['report_sha']


def test_partial_loader_never_imports_or_replays_ownconsumer(monkeypatch):
    v=m.template()['additional_sources'];v[0].update(state='partial',report=dict(path='/ram/partial.json',sha256='a'*64))
    monkeypatch.setattr(m.base,'load_ref',lambda ref:dict(status='partial_or_failed'))
    def forbidden(*a):raise AssertionError('no ownconsumer on partial')
    monkeypatch.setattr(m.importlib,'import_module',forbidden)
    out=m.load_new_sources(v);assert out[0]['report']['status']=='partial_or_failed' and 'consume' not in out[0]


def test_existing_input_not_mutated_frozen_v2_and_small_JSON_MD():
    f=fixture();before=deepcopy(f);r=m.assemble(*f);assert f==before
    text=m.markdown(r);assert 'GRASP非官方' in text and 'missing不作0' in text and '3604/3604' in text and '176/176' in text
    assert len(m.canonical(r))+len(text.encode())<m.CAP
    assert m.sha(m.prior.__file__)==m.PRIOR_SHA and m.sha(m.base.__file__)==m.prior.BASE_SHA


@pytest.mark.parametrize('edit',['old_physical','wrong_arm','single_arm','RGB_without_physical'])
def test_no_invented_future_physical_RGB_contract(edit):
    v=m.template();d=v['new_downstream']
    if edit=='old_physical':d['physical']['source_sha256']=m.prior.PHYSICAL_SOURCE_SHA
    elif edit=='wrong_arm':d['arms']=['CEM','GRASP_H3_bounded']
    elif edit=='single_arm':d['arms'].pop()
    else:
        d['physical']=dict(state='missing',reason='not present')
        d['rgb']=dict(state='complete',report=dict(path='/ram/rgb.json',sha256='e'*64),prepare_sha256='e'*64)
    with pytest.raises(ValueError):m.manifest_check(v)


def test_explicit_deferred_reason_propagated_without_model_Q_substitution():
    d=m.deferred();d['physical']['reason']='MAIN physical not yet available'
    r=m.assemble(*fixture(),extended={k:dict(state=e['state'],entry=e) for k,e in d.items() if k!='arms'})
    assert r['new_downstream']==d
    rows=[x for x in r['rows'] if x['arm']==m.EXTRA_ARMS[0]]
    assert all(x['native_success']==m.missing('MAIN physical not yet available') for x in rows)


def test_full_model_prepare_bindings_preserved_without_neural(monkeypatch):
    refs=[]
    def load(ref):refs.append(ref);return dict(bindings_sha256={'/ram/weights.pt':'a'*64})
    monkeypatch.setattr(m.base,'load_ref',load)
    out=m.provider_bindings(dict(prepared='/ram/prepare.json',prepare_sha256='b'*64))
    assert out=={'/ram/weights.pt':'a'*64,'/ram/prepare.json':'b'*64} and len(refs)==1


def extended_fixture():
    f=fixture();tasks,bundles=f[1:];rows=[];images=[]
    for s,t in enumerate(tasks):
        for b in bundles:
            plan=b['consume']['rows'][s];arm=plan['arm'];ident=str(len(rows)+1)*64
            rows.append(dict(root_slot=s,arm=arm,selection_digest=ident,status='horizon_completed',native_success=False,
                full_horizon=True,initial_exact=True,own_returned=10,own_attempted=10,first_error=None))
            for role in ('prefix5','terminal10'):
                images.append(dict(index=len(images),branch=len(rows)-1,source_slot=s,root_slot=s,arm=arm,role=role,
                    task_id=t['task_id'],task_export=t['task_export'],task_sha256=t['task_sha256'],blind_sha256=t['blind_sha256'],
                    root=t['root_context']['latent_history'][-1],goal=t['goal'],Q='identity192',selection_digest=ident,
                    compiler_report_sha256=b['entry']['report']['sha256'],own_plan_sha256=m.base.hashlib.sha256(json.dumps(plan,sort_keys=True,allow_nan=False).encode()).hexdigest(),
                    checkpoint=dict(file_sha256='f'*64,pixel_sha256='a'*64)))
    counts={k:dict(attempted=n,completed=n) for k,n in dict(environment_construct=4,environment_close=4,reset=8,set_state=16,
        goal_bind=4,goal_render=8,public_step=80,physics_step=816,explicit_render=88).items()}
    report=dict(rows=rows,contract=dict(arms=list(m.EXTRA_ARMS),with_target=False),errors=[],native_successes=0,full_horizons=4,
        sourcepack_sha256='c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6',counts=counts,wall_seconds=12.25,
        schema='p1_moving_extended_baseline_physical_20261027_v1.report',status='moving_extended_baseline_physical_recorded')
    p=dict(arguments=dict(physical_sha='f'*64,physical_prepare_sha='e'*64),images=images,bindings={},contract=dict(arms=list(m.EXTRA_ARMS)))
    package=dict(state='complete',entry=dict(state='complete',report=dict(path='/ram/extended/report.json',sha256='f'*64),prepare_sha256='e'*64,
        source_sha256=m.EXT_PHYSICAL_SHA),report=report,prepared=p,images=images)
    return f,dict(physical=package,rgb=dict(state='missing',entry=dict(state='missing',reason='actual RGB pending')))


def test_extended_physical_actual_not_old_authority_and_no_RGB_Q_invention():
    f,e=extended_fixture();r=m.assemble(*f,extended=e)
    assert r['denominator']['actual_physical_rows']==12 and r['denominator']['actual_RGB_images']==16
    assert [p['actual_branches'] for p in r['physical_runs']]==[6,2,4]
    assert [p['counts']['public_step']['completed'] for p in r['physical_runs']]==[120,40,80]
    for s in r['summary']:
        assert s['native_successes']['value']==0 and s['full_horizons']['value']==2
        if s['arm'] in m.EXTRA_ARMS:assert s['mean_realQ_terminal10']['value'] is None and s['actual_RGB_roots']==0


def raw_extended_fixture():
    f,e=extended_fixture();p=e['physical']['prepared'];rows=[];events=[];arrays=[];pixels=[]
    def compare(item,z):
        return dict(actual_Q=float(np.mean((z.astype(float)-np.asarray(item['goal']))**2)),root_Q=float(np.mean((np.asarray(item['root'])-np.asarray(item['goal']))**2)))
    for i,item in enumerate(p['images']):
        z=np.full(192,.5,np.float32);a=np.full((224,224,3),i,np.uint8);arrays.append(z);pixels.append(a)
        meta=dict(file=f'image_{i:02d}.rgb.png',sha256='f'*64,pixel_sha256='a'*64,dtype='uint8',shape=[224,224,3])
        metrics=compare(item,z);metrics.update(rawpixel_mean=float(i),rawpixel_vs_prefix5_mse=float(i%2),actual_Q_change_from_prefix5=0.)
        row=dict(**{k:item[k] for k in ('index','branch','root_slot','source_slot','arm','role','task_id','task_export','task_sha256','blind_sha256')},
            status='encoded',provenance=item,rawpixel=meta,latent=dict(file=f'image_{i:02d}.native192.npy'),metrics=metrics)
        rows.append(row);events+=[dict(kind='image_attempt',provenance=item),dict(kind='raw_image_saved',index=i,receipt=meta)]
        for j,k in enumerate(('encoder','projector')):
            for phase,completed in (('module_attempt',False),('module_return',True)):
                events.append(dict(kind=phase,entry=dict(index=2*i+j,image=i,resource=k,completed=completed)))
        events.append(dict(kind='image_saved',receipt=row))
    report=dict(rows=rows,resources={k:dict(attempted=8,completed=8) for k in ('encoder','projector')})
    return f,e,report,p,events,arrays,pixels,compare


def test_extended8_RGB_saved_arrays_complete24_without_new_neural():
    f,e,r,p,events,arrays,pixels,compare=raw_extended_fixture()
    receipts=m.extended_rgb_receipts(r,p,events,arrays,pixels,compare)
    e['rgb']=dict(state='complete',entry=dict(state='complete',report=dict(path='/ram/extendedRGB/report.json',sha256='a'*64),prepare_sha256='b'*64),
        report=r,receipts=receipts)
    out=m.assemble(*f,extended=e)
    assert out['status']=='comparison_recorded_not_admitted' and out['denominator']['actual_RGB_images']==24
    assert [r['resources']['encoder']['completed'] for r in out['RGB_runs']]==[12,4,8]
    assert all(s['native_successes']['value']==0 and s['actual_RGB_roots']==2 for s in out['summary'])


@pytest.mark.parametrize('edit',['old6_events','one_root','resource','PNG_mse','PNG_mean','raw_before_dispatch','metadata','NPY_Q','wrong_pair'])
def test_extended_rawPNG_receipt_and_metric_faults(edit):
    _,_,r,p,events,arrays,pixels,compare=raw_extended_fixture()
    if edit=='old6_events':events.pop(1)
    elif edit=='one_root':arrays.pop()
    elif edit=='resource':r['resources']['encoder']['completed']=7
    elif edit=='PNG_mse':pixels[1]=np.full((224,224,3),20,np.uint8)
    elif edit=='PNG_mean':r['rows'][0]['metrics']['rawpixel_mean']=1.
    elif edit=='raw_before_dispatch':events[1],events[2]=events[2],events[1]
    elif edit=='metadata':r['rows'][0]['rawpixel']['dtype']='float32'
    elif edit=='NPY_Q':arrays[0]=np.full(192,.25,np.float32)
    else:p['images'][1]['selection_digest']='e'*64
    with pytest.raises(ValueError):m.extended_rgb_receipts(r,p,events,arrays,pixels,compare)


def test_extended_saved_build_only_own_module(monkeypatch):
    f,e=extended_fixture();ph=e['physical'];calls=[]
    monkeypatch.setattr(m.base,'load_ref',lambda ref:ph['report'])
    def build(**kwargs):calls.append(kwargs);return ph['prepared']
    monkeypatch.setattr(m,'extended_adapter',lambda:SimpleNamespace(build=build))
    got=m.load_extended_physical(ph['entry'])
    assert got['images']==ph['images'] and calls[0]['arms']==list(m.EXTRA_ARMS) and calls[0]['physical_source_sha']==m.EXT_PHYSICAL_SHA


def test_partial_extended_keeps_unknown_and_never_invokes_adapter(monkeypatch):
    monkeypatch.setattr(m.base,'load_ref',lambda ref:dict(status='partial_or_failed',counts={}))
    def forbidden():raise AssertionError('no complete saved verifier')
    monkeypatch.setattr(m,'extended_adapter',forbidden)
    entry=dict(state='partial',report={},reason='interrupted')
    assert m.load_extended_physical(entry)['state']=='partial' and m.load_extended_rgb(entry,{})['state']=='partial'
