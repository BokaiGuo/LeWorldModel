"""RAM fixtures only: no production assemble/verify, neural or physical run."""
from copy import deepcopy
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import p1_moving_baseline_comparison_20261027_v2 as m


def models():
    # Reuse already frozen v1 arithmetic fixture, NOT any saved result.
    f=runpy.run_path(str(Path(__file__).with_name('test_p1_moving_baseline_comparison_20261027_v1.py')))['fixture']()
    tasks,encoder,bundles,_=f
    return tasks,encoder,[b for b in bundles if b['entry']['provider'] in ('sampling','gradient')]


def physical_fixture(tasks,bundles,arms=None,legacy=False):
    arms=list(m.COHORT if arms is None else arms);selected=m.selected_models(bundles);rows=[];images=[]
    for s in (0,1):
        for a in arms:
            chosen=selected[s,a];plan=chosen['plan'];task=tasks[s];ident=str(len(rows)+1)*64
            rows.append(dict(root_slot=s,arm=a,selection_digest=ident,status='horizon_completed',native_success=False,
                full_horizon=True,initial_exact=True,own_returned=10,own_attempted=10,first_error=None))
            for role in ('prefix5','terminal10'):
                item=dict(index=len(images),branch=len(rows)-1,root_slot=s,source_slot=s,arm=a,role=role,
                    task_id=task['task_id'],task_export=task['task_export'],task_sha256=task['task_sha256'],blind_sha256=task['blind_sha256'],
                    Q='identity192',root=task['root_context']['latent_history'][-1],goal=task['goal'],selection_digest=ident,
                    compiler_report_sha256=chosen['report_sha256'],own_plan_sha256=m.base.hashlib.sha256(json.dumps(plan,sort_keys=True,allow_nan=False).encode()).hexdigest(),
                    checkpoint=dict(pixel_sha256='d'*64),png_path=f'/ram/image_{len(images)}.png',predicted=None)
                if legacy:item['planner_selection_sha256']=plan['selection_sha256']
                images.append(item)
    n=len(rows);counts={k:dict(attempted=v,completed=v) for k,v in dict(reset=2*n,set_state=4*n,goal_bind=n,goal_render=2*n,
        public_step=20*n,physics_step=204*n,explicit_render=22*n).items()}
    r=dict(rows=rows,contract=dict(arms=arms,with_target=False),errors=[],sourcepack_sha256='c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6',
        native_successes=0,full_horizons=n,counts=counts,wall_seconds=9.125)
    p=dict(arguments=dict(physical_sha='a'*64,physical_prepare_sha='b'*64),images=images,bindings={},contract=dict(arms=arms))
    if legacy:p['arguments']['planner_sha']=selected[0,arms[0]]['report_sha256']
    return dict(state='complete',entry=dict(state='complete',report=dict(path='/ram/physical.json',sha256='a'*64),prepare_sha256='b'*64),
        report=r,prepared=p,images=images,legacy=legacy)


def compare(item,z):
    real=z.astype(float);root=np.asarray(item['root']);goal=np.asarray(item['goal'])
    return dict(actual_Q=float(np.mean((real-goal)**2)),root_Q=float(np.mean((root-goal)**2)))


def rgb_fixture(physical):
    receipts=[]
    for i,item in enumerate(physical['images']):
        z=np.full(192,.5 if item['role']=='prefix5' else .75,np.float32)
        receipts.append(dict(item=deepcopy(item),metrics=compare(item,z),latent=dict(file=f'image_{i:02d}.native192.npy',sha256='c'*64,dtype='float32',shape=[192])))
    n=len(receipts)
    return dict(state='complete',entry=dict(state='complete',report=dict(path='/ram/rgb.json',sha256='c'*64),prepare_sha256='d'*64),
        receipts=receipts,report=dict(resources={k:dict(attempted=n,completed=n) for k in ('encoder','projector')}))


def fixture():
    t,e,b=models();p=physical_fixture(t,b);rgb=rgb_fixture(p);return t,e,b,list(m.COHORT),p,rgb


def test_actual_three_arms_no_kino_dependency_native_not_Q():
    r=m.assemble(*fixture())
    assert r['status']=='comparison_recorded_not_admitted' and len(r['rows'])==6 and len(r['summary'])==3
    assert r['denominator']==dict(algorithms=3,roots_per_algorithm=2,expected_model_rows=6,actual_model_rows=6,
        expected_physical_rows=6,actual_physical_rows=6,expected_RGB_images=12,actual_RGB_images=12)
    assert all(s['native_successes']['value']==0 and s['full_horizons']['value']==2 and s['actual_physical_roots']==2 for s in r['summary'])
    assert all(s['mean_realQ_terminal10']['value']==.0625 and s['mean_model_Q']['value']==.25 for s in r['summary'])
    assert all(row['native_success']['value'] is False for row in r['rows'])
    assert r['physical_runs'][0]['native_successes']==0 and r['physical_runs'][0]['full_horizons']==6
    assert json.loads(m.canonical(r))==r
    assert r['P1_admission'] is r['P2_admission'] is r['equal_total_compute'] is r['official_reproduction'] is False


def test_actual_resource_ledgers_are_separate_and_latency_not_divided():
    r=m.assemble(*fixture());p=r['physical_runs'][0]
    assert p['counts']['public_step']['completed']==120 and p['counts']['physics_step']['completed']==1224
    assert p['latency_seconds']['value']==9.125
    assert r['RGB_runs'][0]['resources']['encoder']['completed']==12
    assert r['inherited_encoding_shared_once']['value']['encoder']['completed']==8
    assert all(s['standalone_inherited_encoding']['value']['encoder']['completed']==8 for s in r['summary'])
    assert all(s['evaluation_encoding']['value']['encoder']['completed']==4 for s in r['summary'])
    assert {s['arm']:s['resources']['predict']['value']['completed'] for s in r['summary']}==dict(moving_gradient_open_loop=272,CEM=268,kernel_faithful_mppi_nominal=144)
    assert all(row['latency_seconds']['value'] is None for row in r['rows'])


@pytest.mark.parametrize('which',['physical','rgb'])
@pytest.mark.parametrize('state',['missing','partial'])
def test_missing_partial_not_native_zero_or_RGB_zero(which,state):
    f=list(fixture());i=4 if which=='physical' else 5
    f[i]=dict(state=state,entry=dict(state=state,reason='not available'),report=dict(status='partial_or_failed',counts=dict(predict=3)))
    if which=='physical':f[5]=dict(state='missing',entry=dict(state='missing',reason='no parent'))
    r=m.assemble(*f);assert r['status']=='partial_comparison_not_admitted'
    for s in r['summary']:
        assert s['mean_realQ_terminal10']['value'] is None
        if which=='physical':assert s['native_successes']['value'] is None and s['actual_physical_roots']==0
        else:assert s['native_successes']['value']==0 and s['actual_physical_roots']==2


@pytest.mark.parametrize('edit',['report_sha','own_plan','task_id','goal','root','blind','task_sha','selection','role_order','native_bool','horizon','source','missing_model'])
def test_exact_model_physical_RGB_join_rejects_wrong_chain(edit):
    f=list(fixture());p=f[4];item=p['images'][0]
    if edit=='report_sha':item['compiler_report_sha256']='e'*64
    elif edit=='own_plan':item['own_plan_sha256']='e'*64
    elif edit=='task_id':item['task_id']='e'*64
    elif edit=='goal':item['goal']=[0.]*192
    elif edit=='root':item['root']=[0.]*192
    elif edit=='blind':item['blind_sha256']='e'*64
    elif edit=='task_sha':item['task_sha256']='e'*64
    elif edit=='selection':item['selection_digest']='e'*64
    elif edit=='role_order':p['images'].reverse()
    elif edit=='native_bool':p['report']['rows'][0]['native_success']=0
    elif edit=='horizon':p['report']['rows'][0]['own_returned']=9
    elif edit=='source':p['report']['sourcepack_sha256']='e'*64
    else:f[2]=[b for b in f[2] if b['entry']['provider']!='gradient']
    with pytest.raises(ValueError):m.assemble(*f)


def raw_rgb_fixture():
    *_,p,rgb=fixture();images=p['images'];rows=[];arrays=[];events=[]
    for i,image in enumerate(images):
        z=np.full(192,.5,np.float32);arrays.append(z)
        row=dict(index=i,status='encoded',provenance=image,latent=dict(file=f'image_{i:02d}.native192.npy'),metrics=compare(image,z),
            **{k:image[k] for k in ('branch','root_slot','source_slot','arm','role','task_id','task_export','task_sha256','blind_sha256')})
        rows.append(row);events.append(dict(kind='image_attempt',provenance=image))
        for j,resource in enumerate(('encoder','projector')):
            for kind,completed in (('module_attempt',False),('module_return',True)):
                events.append(dict(kind=kind,entry=dict(index=2*i+j,image=i,resource=resource,completed=completed)))
        events.append(dict(kind='image_saved',receipt=row))
    report=dict(rows=rows,resources={k:dict(attempted=12,completed=12) for k in ('encoder','projector')})
    return report,dict(images=images,contract=dict(arms=list(m.COHORT))),events,arrays


def test_actual_RGB_receipt_mapping_and_NPY_math_only():
    r,p,e,z=raw_rgb_fixture();out=m.rgb_receipts(r,p,e,z,compare)
    assert len(out)==12 and all(x['metrics']['actual_Q']==.25 for x in out)


@pytest.mark.parametrize('edit',['missing_image','resource','module_return','saved_receipt','metrics','provenance'])
def test_RGB_receipt_faults_not_silently_zero(edit):
    r,p,e,z=raw_rgb_fixture()
    if edit=='missing_image':z.pop()
    elif edit=='resource':r['resources']['encoder']['completed']=11
    elif edit=='module_return':e[2]['entry']['completed']=False
    elif edit=='saved_receipt':e[-1]=dict(kind='image_saved',receipt={})
    elif edit=='metrics':r['rows'][0]['metrics']=dict(actual_Q=0.,root_Q=0.)
    else:r['rows'][0]['provenance']=dict(p['images'][0],task_id='e'*64)
    with pytest.raises(ValueError):m.rgb_receipts(r,p,e,z,compare)


def test_template_actual_seals_and_optional_legacy_gate():
    v=m.template();m.manifest_check(v)
    assert v['physical']['report']['sha256']==m.PHYSICAL_SHA and v['rgb']['report']['sha256']==m.RGB_SHA
    assert len(v['cohort'])==3 and 'kino' not in [s['provider'] for s in v['sources']]
    v['cohort']+=list(m.LEGACY)
    with pytest.raises(ValueError,match='legacy'):m.manifest_check(v)


def test_legacy_same_task_binding_supported_only_after_full_saved_adapter():
    f=list(fixture());old=runpy.run_path(str(Path(__file__).with_name('test_p1_moving_baseline_comparison_20261027_v1.py')))['fixture']()[2][0]
    for plan in old['consume']['plans']:plan['selection_sha256']='f'*64
    f[2].append(old);f[3]+=list(m.LEGACY)
    lp=physical_fixture(f[0],f[2],m.LEGACY,True);lr=rgb_fixture(lp)
    r=m.assemble(*f,lp,lr);assert len(r['summary'])==5 and r['denominator']['actual_physical_rows']==10
    lp['images'][0]['planner_selection_sha256']='e'*64
    with pytest.raises(ValueError,match='GN/D2P'):m.assemble(*f,lp,lr)


def test_saved_build_and_verify_only_no_neural_physics(monkeypatch):
    t,e,b,cohort,p,rgb=fixture();entry=dict(state='complete',report=dict(path='/ram/p/report.json',sha256='a'*64),prepare_sha256='b'*64,source_sha256=m.PHYSICAL_SOURCE_SHA)
    calls=[]
    def build(**kwargs):calls.append(kwargs);return p['prepared']
    monkeypatch.setattr(m.base,'load_ref',lambda ref:deepcopy(p['report']))
    monkeypatch.setattr(m,'adapter',lambda legacy:SimpleNamespace(build=build))
    result=m.load_physical(entry)
    assert result['images']==p['images'] and calls[0]['arms']==list(m.COHORT)
    assert 'execute' not in calls[0] and 'run' not in calls[0]


def test_partial_loads_never_invoke_saved_full_verifier(monkeypatch):
    monkeypatch.setattr(m.base,'load_ref',lambda ref:dict(status='partial_or_failed'))
    def forbidden(*a):raise AssertionError('partial must stay partial')
    monkeypatch.setattr(m,'adapter',forbidden)
    entry=dict(state='partial',report={},reason='interrupted')
    assert m.load_physical(entry)['state']=='partial' and m.load_rgb(entry,dict(state='missing'))['state']=='partial'


def test_chinese_table_and_missing_not_success_conflation():
    r=m.assemble(*fixture());text=m.markdown(r)
    assert '0/2' in text and '2/2' in text and '完整10步不等于成功' in text and '1224' in text
    assert '非官方' in text and '不等总算力' in text and '理论失败' in text


def test_payload_bounded_and_v1_not_changed():
    r=m.assemble(*fixture());assert len(m.canonical(r))+len(m.markdown(r).encode())<m.CAP
    assert m.sha(m.base.__file__)==m.BASE_SHA


def kino_fixture():
    f=list(fixture())
    kino=runpy.run_path(str(Path(__file__).with_name('test_p1_moving_baseline_comparison_20261027_v1.py')))['fixture']()[2][-1]
    assert kino['entry']['provider']=='kino'
    f[2].append(kino);f[3].append(m.KINO)
    p=physical_fixture(f[0],f[2],[m.KINO]);p['entry']['report']['path']='/ram/kino/physical.json'
    p['entry']['report']['sha256']='e'*64
    rgb=rgb_fixture(p);rgb['entry']['report']['path']='/ram/kino/rgb.json'
    return f,dict(arms=[m.KINO],physical=p,rgb=rgb)


def test_disjoint_actual_kino_cohort_no_original_three_duplication():
    f,extra=kino_fixture();r=m.assemble(*f,additional=[extra])
    assert len(r['rows'])==8 and len(r['summary'])==4
    assert r['denominator']['actual_physical_rows']==8 and r['denominator']['actual_RGB_images']==16
    assert [p['arms'] for p in r['physical_runs']]==[list(m.COHORT),[m.KINO]]
    assert [p['counts']['public_step']['completed'] for p in r['physical_runs']]==[120,40]
    assert [p['resources']['encoder']['completed'] for p in r['RGB_runs']]==[12,4]
    assert r['inherited_encoding_shared_once']['value']['encoder']['completed']==8
    kino=next(s for s in r['summary'] if s['arm']==m.KINO)
    assert kino['resources']['predict']['value']['completed']==168 and kino['native_successes']['value']==0
    assert len(m.canonical(r))+len(m.markdown(r).encode())<m.CAP


@pytest.mark.parametrize('state',['missing','partial'])
def test_pending_kino_cohort_keeps_original_three_actual_and_fixed_denominators(state):
    f,extra=kino_fixture()
    for k in ('physical','rgb'):
        extra[k]=dict(state=state,entry=dict(state=state,reason='independent kino cohort pending'),report=dict(status='partial_or_failed'))
    r=m.assemble(*f,additional=[extra]);assert r['status']=='partial_comparison_not_admitted'
    assert r['denominator']['expected_physical_rows']==8 and r['denominator']['actual_physical_rows']==6
    assert r['denominator']['expected_RGB_images']==16 and r['denominator']['actual_RGB_images']==12
    assert len(r['physical_runs'])==1 and len(r['RGB_runs'])==1
    for s in r['summary']:
        assert s['actual_model_roots']==2
        if s['arm']==m.KINO:assert s['native_successes']['value'] is None and s['mean_realQ_terminal10']['value'] is None
        else:assert s['actual_physical_roots']==2 and s['native_successes']['value']==0


@pytest.mark.parametrize('edit',['duplicate','original_overlap','implicit_drop','single_root'])
def test_additional_cohort_no_duplicate_no_root_filter(edit):
    f,extra=kino_fixture();additional=[extra]
    if edit=='duplicate':additional.append(extra)
    elif edit=='original_overlap':extra['arms']=[m.COHORT[0]]
    elif edit=='implicit_drop':additional=[]
    else:extra['physical']['report']['rows'].pop()
    with pytest.raises(ValueError):m.assemble(*f,additional=additional)


def test_manifest_kino_requires_explicit_independent_cohort_not_repeated_original():
    v=m.template();v['cohort'].append(m.KINO)
    kino=next(s for s in m.base.template()['sources'] if s['provider']=='kino')
    v['sources'].append(kino)
    with pytest.raises(ValueError,match='explicit kino'):m.manifest_check(v)
    extra=dict(arms=[m.KINO],physical=dict(state='missing',reason='MAIN independent cohort not completed'),
        rgb=dict(state='missing',reason='requires same cohort physical'))
    v['additional_cohorts']=[extra];m.manifest_check(v)
    extra['arms']=[m.COHORT[0]]
    with pytest.raises(ValueError,match='kino only once'):m.manifest_check(v)
