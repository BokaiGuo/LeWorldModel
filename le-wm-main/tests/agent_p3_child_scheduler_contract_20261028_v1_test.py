"""RAM callbacks and saved task metadata only. No real prepare/model/physics."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_child_scheduler_contract_20261028_v1 as m


def task(slot=0):
    c=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=np.zeros((3,192)).tolist(),action_history=np.full((2,10),.25).tolist())
    t=dict(root_context=c,goal=[1.]+[0.]*191,projection=np.eye(192).tolist(),action_mean=[0.,0.],action_std=[1.,1.])
    t['binding']={k:m.digest([slot,k]) for k in m.core.BINDINGS}
    t['binding'].update(context_sha256=m.digest(c),goal_sha256=m.digest(t['goal']),projection_sha256=m.digest(t['projection']),
        normalizer_sha256=m.digest([t['action_mean'],t['action_std']]),action_domain_sha256=m.digest(m.core.action_domain()))
    return m.core.validate_task(t)


def child(eid='e00_R',boundary=5):
    t=task(m.EPISODES.index(eid)//4); prefix=np.full((boundary,2),np.float32(.125))
    raw=np.concatenate((np.full((10,2),np.float32(.25)),prefix))[-10:]
    c=deepcopy(t['root_context']);c['latent_history'][-1][1]=.5;c['action_history']=m.rolling.normalize(raw,t)
    index=m.EPISODES.index(eid)
    return dict(child_id=eid+f'_b{boundary:02}',episode_id=eid,root_slot=index//4,prefix_arm=m.rolling.ARMS[(index//2)%2],
        source_mode=('I','R')[index%2],boundary=boundary,status='observed_child',context=c,context_sha256=m.digest(c),
        raw_history10=raw.tolist(),executed_prefix=prefix.tolist(),observation=dict(boundary=boundary,
            png_sha256=m.digest('png'),pixel_sha256=m.digest('pixels'),encoding_sha256=m.digest('encode'),
            observed_index=1,encoded_index=2,observed_latent_sha256=m.rolling.packet_sha(c['latent_history'][-1])),
        original_compiler_context_sha256=m.digest(c),ancestry=dict(task_binding_sha256=m.digest(t['binding']),
            source_report_sha256=m.digest('P4 actual metadata test ONLY'),original_request_sha256=m.digest('request')))


class Backend:
    kind='RAM_FIXTURE'
    source_sha256=m.digest('RAM callback, NOT scientific evidence')
    def __init__(self,geometry='progress',fault=None): self.geometry=geometry;self.fault=fault;self.requests=[]
    def forward(self,ledger):
        ledger.call('action_encoder',lambda:None,shape=[1,3,10])
        def predict():
            ledger.call('predictor',lambda:None,shape=[1,3,192])
            ledger.call('pred_proj',lambda:None,shape=[3,192])
        ledger.call('predict',predict,shape=[1,3,192])
    def signal(self,request,ledger):
        self.requests.append(deepcopy(request)); h=request['horizon']
        if self.fault=='raise':raise RuntimeError('RAM signal failure')
        for _ in range(h): self.forward(ledger)
        def jacobian():
            for _ in range(h):self.forward(ledger)
            def gradients():
                for _ in range(192):ledger.call('grad',lambda:ledger.call('raw_backward',lambda:None))
            ledger.call('vjp',gradients,directions=192)
        ledger.call('jacobian',jacobian)
        j=np.zeros((192,10*h),np.float32)
        if self.geometry=='progress' or (self.geometry=='long' and h==3): j[0,0]=1
        response=dict(binding_sha256=request['binding_sha256'],parent_context_sha256=request['parent_context_sha256'],
            horizon=h,nominal=[0.]*192,jacobian=j.tolist(),producer_sha256=self.source_sha256)
        if self.fault=='stale':response['parent_context_sha256']=m.digest(task()['root_context'])
        if self.fault=='oracle':response['native_success']=True
        if self.fault=='source':response['producer_sha256']=m.digest('wrong adapter')
        if self.fault=='missing':response.pop('nominal')
        if self.fault=='extra_forward':self.forward(ledger)
        return response


def run(geometry='progress',sharing='independent',fault=None):
    backends=[]
    def factory(c,group):
        b=Backend(geometry,fault);backends.append(b);return b
    result=m.plan_child(child(),[task(),task(1)],factory,sharing=sharing,permit_ram=True)
    return result,backends


def test_independent_child_signals_fully_paid_and_exact_parent():
    c=child();result,backends=run()
    assert result['fixed_denominator']==2 and result['status']=='ram_planned_not_evidence'
    assert len(backends)==len(result['acquisitions'])==2
    for b,a,row in zip(backends,result['acquisitions'],result['rows'],strict=True):
        assert [q['horizon'] for q in b.requests]==[1,3]
        assert all(q['parent_context']==c['context'] and q['parent_context_sha256']==c['context_sha256'] for q in b.requests)
        receipt=a['resource_receipt'];assert receipt['attempted']==receipt['completed']==m.signal_caps(1)
        assert receipt['completed']['predict_API']==receipt['completed']['predict_samples']==8
        assert receipt['completed']['jacobian_API']==2 and receipt['completed']['VJP_directions']==384
        assert receipt['FLOPs'] is receipt['cuda_ns'] is receipt['backward_engine_calls'] is None
        assert row['decision']['candidates']==len(row['queue'])==4
        assert row['signal_charge']==dict(numerator=1,denominator=1)
        assert row['commands'] is None and row['generation_dispatch_allowed'] is False
        assert all(q['prefix_edges']==q['prefix_actions']==[] and q['actual_executed_prefix']==c['executed_prefix'] for q in row['queue'])
    assert all(result[k] is False for k in m.FLAGS)


def test_shared_equal_actual_once_rational_half_no_free_or_double_payment():
    result,backends=run(sharing='shared_equal')
    assert len(backends)==len(result['acquisitions'])==1
    a=result['acquisitions'][0];assert a['resource_receipt']['completed']==m.signal_caps(2)
    assert sum(r['signal_charge']['numerator']/r['signal_charge']['denominator'] for r in result['rows'])==1
    assert result['rows'][0]['signal_group']==result['rows'][1]['signal_group']==a['group']
    assert all(r['signal_charge']==dict(numerator=1,denominator=2) for r in result['rows'])


@pytest.mark.parametrize('geometry,kind,h,count',[('progress','progress',1,4),('short','reconfiguration',1,8),('long','reconfiguration',3,8)])
def test_original_choose_quota_type_depth_not_rewritten_for_budget(geometry,kind,h,count):
    result,_=run(geometry)
    row=result['rows'][1];decision=row['decision']
    assert decision['extension_type']==kind and decision['horizon']==h and decision['candidates']==count
    assert len(row['queue'])==count and row['reserved_search_predict_API']==2*h*count
    if 2*h*count>24:assert row['status']=='blocked_whole_node_search_budget'
    else:assert row['status']=='pending_reviewed_child_compiler'
    assert row['decision']['inherited_engineering_thresholds']==dict(chi=.5,gain=.05)
    assert row['decision']['program_type']!='composition' and row['composition_witness'] is None


@pytest.mark.parametrize('fault',['raise','stale','oracle','source','missing','extra_forward'])
@pytest.mark.parametrize('sharing',['independent','shared_equal'])
def test_failed_or_stale_signal_no_plan_no_retry_no_zero_fill(fault,sharing):
    result,backends=run(fault=fault,sharing=sharing)
    assert len(backends)==(2 if sharing=='independent' else 1)
    assert all(r['status']=='failed_child_signal_or_schedule' and r['queue'] is None and r['decision'] is None for r in result['rows'])
    for a in result['acquisitions']:
        receipt=a['resource_receipt'];m.core.audit_ledger(receipt)
        assert receipt['attempted']['signal_compile_calls']>=1
        if fault=='raise':assert receipt['attempted']['signal_compile_calls']>receipt['completed']['signal_compile_calls']


@pytest.mark.parametrize('fault',['outcome','boolslot','boundary20','history_scale','history_order','normalizer','current_latent','causal','binding','context','raw_prefix'])
def test_child_whitelist_raw_causal_identity_failclosed(fault):
    c=child();tasks=[task(),task(1)]
    if fault=='outcome':c['native_success']=True
    elif fault=='boolslot':c['root_slot']=False
    elif fault=='boundary20':c['boundary']=20
    elif fault=='history_scale':c['raw_history10'][0][0]/=100
    elif fault=='history_order':c['raw_history10'].reverse()
    elif fault=='normalizer':tasks[0]['action_std']=[2.,2.]
    elif fault=='current_latent':c['observation']['observed_latent_sha256']=m.digest('otherlatent')
    elif fault=='causal':c['observation']['encoded_index']=c['observation']['observed_index']
    elif fault=='binding':c['ancestry']['task_binding_sha256']=m.digest('otherroot')
    elif fault=='context':c['context_sha256']=m.digest('othercontext')
    else:c['executed_prefix'][0][0]=.5
    calls=[]
    with pytest.raises(ValueError):m.plan_child(c,tasks,lambda *a:calls.append(a),permit_ram=True)
    assert calls==[]


def test_unreached_keeps_two_method_slots_unknown_and_no_backend():
    c=child(boundary=15);c['status']='not_reached'
    for k in ('context','context_sha256','raw_history10','executed_prefix','observation','original_compiler_context_sha256'):c[k]=None
    c['ancestry']['original_request_sha256']=None
    calls=[];r=m.plan_child(c,[task(),task(1)],lambda *a:calls.append(a),permit_ram=True)
    assert calls==r['acquisitions']==[] and r['fixed_denominator']==2
    assert all(x['queue'] is x['commands'] is None and x['status']=='not_reached' for x in r['rows'])


def test_default_refuses_adapter_execution_and_no_cpu_fallback():
    result=m.plan_child(child(),[task(),task(1)],lambda *a:Backend())
    assert all(a['status']=='failed' and a['resource_receipt'] is None for a in result['acquisitions'])
    b=Backend();b.kind='CPU'
    result=m.plan_child(child(),[task(),task(1)],lambda *a:b,permit_ram=True)
    assert not b.requests and all(r['queue'] is None for r in result['rows'])


def test_plan_exact48_denominator_missing_actual_signals_not_default_chi():
    children=[child(e,b) for e in m.EPISODES for b in m.STEPS]
    p=dict(schema=m.NAME+'.prepare',rule=m.rule(),tasks=[task(),task(1)],children=children,
        inputs=dict(sharing='shared_equal'),jobs=m.jobs(children,'shared_equal'),missing_dependencies=['child adapter'],**m.FLAGS)
    r=m.plan(p);assert len(r['jobs'])==r['fixed_denominator']==48 and len(r['signal_requests'])==48
    assert len({q['group'] for q in r['signal_requests']})==24
    assert all(j['queue'] is j['commands'] is None for j in r['jobs'])
    p['children'].pop()
    with pytest.raises(ValueError):m.plan(p)


def test_source_conflicts_literal_caps_and_unknown_units():
    with pytest.raises(ValueError):m.merge({'/x':'a'*64},{'/x':'b'*64})
    with pytest.raises(ValueError):m.signal_caps(True)
    assert m.signal_caps()['encoder_calls']==m.signal_caps()['projector_calls']==0
    assert m.rule()['search_predict_cap']==24 and not m.rule()['full_C3_tree']


def test_new_output_bounded_exclusive_cli_cannot_run(tmp_path,monkeypatch):
    root=tmp_path/'repo';(root/'results').mkdir(parents=True);monkeypatch.setattr(m,'ROOT',root)
    out=root/'results'/(m.NAME+'_RAM.json');r=m.write_new(str(out),{'RAM':'only'})
    assert r['sha256']==m.sha(out)
    with pytest.raises(ValueError):m.write_new(str(out),{'overwrite':True})
    monkeypatch.setattr(m,'CAP',1)
    with pytest.raises(ValueError):m.write_new(str(out.with_name(m.NAME+'_too_big.json')),{'RAM':'only'})
    with pytest.raises(SystemExit):m.main(['run','--input',str(out),'--sha256',m.sha(out)])


def test_cold_import_no_torch_gym_or_simulator():
    code='import sys; from experiments.spectral_reachability import '+m.NAME+'; assert not any(x in sys.modules for x in ("torch","gym","gymnasium","pymunk"))'
    subprocess.run([sys.executable,'-B','-c',code],cwd=m.core.ROOT,check=True,capture_output=True,timeout=15)


def test_real_saved_export_task_contract_keeps_original_codec():
    path=m.core.ROOT/'results/agent_p4_moving_multiboundary_20261028_v1_prepare_MAIN02.json'
    p=m.rolling.Reader().json(path,'8528e8d7f9b688b1c93be154305ccc35b8ed6218de63e736b1ae567691c432a2')
    reader=m.rolling.Reader();manifest=reader.json(p['original_manifest'],p['bindings'][p['original_manifest']])
    pack=reader.json(manifest['sourcepack'],manifest['sourcepack_sha256'])
    for layer in (p['layers'][0],p['layers'][2]):
        exported=m.rolling.Reader().json(layer['task_export'],layer['task_sha256'])
        source=pack['rows'][layer['source_slot']]
        t=m.task_from_export(exported,layer,source)
        assert t['binding']['task_contract_sha256']==m.rolling.object_sha(exported['task_contract'])
        assert exported['native_root_sha256'] is None and t['binding']['root_sha256']==source['root_sha256']
        assert m.digest(exported['task_contract'])!=t['binding']['task_contract_sha256']
        altered=deepcopy(exported);altered['task_contract']['with_target']=True
        with pytest.raises(ValueError):m.task_from_export(altered,layer,source)
        wrong=deepcopy(source);wrong['blind_payload_sha256']='0'*64
        with pytest.raises(ValueError):m.task_from_export(exported,layer,wrong)


def saved_boundary():
    reader=m.rolling.Reader();root=m.core.ROOT
    folder=root/'results/agent_p4_moving_multiboundary_20261028_v1_MAIN02'
    h='60489b7db3d4eff70d1be46d47d92209e6adcf333ae1d0a0e389b259ec1f6084'
    p=reader.json(root/'results/agent_p4_moving_multiboundary_20261028_v1_prepare_MAIN02.json',
        '8528e8d7f9b688b1c93be154305ccc35b8ed6218de63e736b1ae567691c432a2')
    report=reader.json(folder/'report.json',h)
    receipt=reader.json(root/'results/consume_zero_and_rolling_MAIN_20261028_v1_rolling_MAIN01.json',
        '182d5ce14a6363b287607c511d3bc4dde85769618e8c8082029f090fb19c1232')
    return reader,folder,p,report,receipt,h


def test_saved_actual_RGB_actions_full24_children_without_running_consumer():
    reader,folder,p,report,receipt,h=saved_boundary()
    tasks,children=m.collect_children(reader,folder,p,report,receipt,h)
    assert len(children)==24 and len(tasks)==2
    assert sum(c['status']=='observed_child' for c in children)==23
    absent=[c for c in children if c['status']=='not_reached']
    assert [c['child_id'] for c in absent]==['e01_R_b15'] and absent[0]['context'] is None
    imagined=[c for c in children if c['source_mode']=='I']
    assert any(c['context_sha256']!=c['original_compiler_context_sha256'] for c in imagined)
    actual=[c for c in children if c['source_mode']=='R' and c['status']=='observed_child']
    assert all(c['context_sha256']==c['original_compiler_context_sha256'] for c in actual)
    prepared=dict(schema=m.NAME+'.prepare',rule=m.rule(),tasks=tasks,children=children,
        inputs=dict(sharing='independent'),jobs=m.jobs(children,'independent'),missing_dependencies=['actual child J'],**m.FLAGS)
    plan=m.plan(prepared)
    assert plan['refusal_census']==dict(not_reached=2,missing_child_signal=46,measured_search_budget_refusals=None,
        all_observed_children_budget_refused=None,reason='no actual child signals/decisions yet; cannot infer quota refusal from root or old outcome')


def test_saved_future_actions_outcomes_not_scheduler_inputs():
    reader,folder,p,report,receipt,h=saved_boundary()
    tasks,children=m.collect_children(reader,folder,p,report,receipt,h)
    altered=deepcopy(report)
    for row in altered['rows']:
        # b5 is unaffected by anything beyond b5. Full later saved artifacts
        # are authenticated elsewhere; this tests projection-only causality.
        row['native_success']=not row['native_success'];row['final_state']=['outcome withheld']
    reader2=m.rolling.Reader()
    tasks2,children2=m.collect_children(reader2,folder,p,altered,receipt,h)
    assert tasks2==tasks and children2==children


def test_material_paid_root_reference_never_released_as_child_signal(tmp_path,monkeypatch):
    reader,folder,p,report,receipt,h=saved_boundary();calls=[]
    inputs=dict(p4=dict(output=str(folder),report_sha256=h,
        prepared=str(m.core.ROOT/'results/agent_p4_moving_multiboundary_20261028_v1_prepare_MAIN02.json'),
        prepare_sha256='8528e8d7f9b688b1c93be154305ccc35b8ed6218de63e736b1ae567691c432a2'),p3={},sharing='independent')
    panels={};prepared={};paid={}
    for mode in m.MODES:
        caps={k:2*v for k,v in m.signal_caps(2).items()};ledger=m.core.Ledger(caps)
        for slot in (0,1):
            t=task(slot);node=dict(context=t['root_context'],depth=0);ledger.scope=dict(slot=slot)
            m.core.local_signal(t,node,Backend(),ledger,m.core.policy(t,node))
            for _ in range(2):ledger.call('schedule',lambda:None)
        d=tmp_path/mode;d.mkdir();prep=d/'prepare.json';prep.write_text('{}')
        for name in ('report.json','events.jsonl','binding.json'):(d/name).write_text('{}')
        panels[mode]=dict(resource_receipt=ledger.receipt(),prepared=str(prep))
        prepared[mode]=dict(sources={});paid[mode]=dict(RAM='mock original full consumer seam ONLY')
        inputs['p3'][mode]=dict(output=str(d),report_sha256=m.sha(d/'report.json'))
    monkeypatch.setattr(m,'consume_boundary',lambda *args:(calls.append(args) or receipt))
    monkeypatch.setattr(m,'load_paid_panels',lambda args:(prepared,panels,paid))
    result=m.material(inputs)
    assert len(calls)==1 and len(result['children'])==24 and len(result['jobs'])==48
    assert result['new_child_signals'] is None and all(j['queue'] is None for j in result['jobs'])
    assert all(not r['child_signal_reused'] for r in result['paid_root_signal_reference'].values())
    assert all(str(path) in result['sources'] for path in m.own_files())
    result['sources'][str(m.own_files()[0])]='0'*64
    with pytest.raises(ValueError):m.check_sources(result['sources'])


def test_cli_existing_output_refused_before_any_material(tmp_path,monkeypatch):
    root=tmp_path/'repo';(root/'results').mkdir(parents=True);monkeypatch.setattr(m,'ROOT',root)
    output=root/'results'/(m.NAME+'_existing.json');output.write_text('preserve RAM fixture')
    calls=[];monkeypatch.setattr(m,'material',lambda *a:calls.append(a))
    with pytest.raises(ValueError):m.main(['prepare','--input',str(tmp_path/'unused'),
        '--sha256','a'*64,'--output',str(output)])
    assert calls==[] and output.read_text()=='preserve RAM fixture'
