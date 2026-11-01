"""RAM metadata fixtures; never run producer/consume, load models or read live results."""
from copy import deepcopy
import runpy
import subprocess
import sys
import pytest
from experiments.spectral_reachability import agent_p3_parent_tree_summary_20261029_v1 as m


@pytest.fixture(scope='module')
def data():
    # Reuse only the frozen pure-core numpy callbacks. Not the production runner.
    old=runpy.run_path(str(m.ROOT/'tests/test_agent_p3_parent_native_adapter_20261029_v1.py'))
    fixed,_,_=old['run']('fixed');adaptive,_,_=old['run']()
    t=old['task']();t['goal']=[1.]*192;t['binding']['goal_sha256']=m.digest(t['goal']);sink=old['Sink']()
    refused=old['m'].run_core(t,lambda t,s:old['RAM'](t,s),sink,mode='adaptive',permit_ram=True)
    # root1 fixed uses its own same-goal authority, matching its adaptive arm.
    other=old['m'].run_core(t,lambda t,s:old['RAM'](t,s),sink,mode='fixed',permit_ram=True)
    return pair([fixed,adaptive,other,refused])


def pair(results):
    rows=[dict(**identity,status=result['status'],result=result) for identity,result in zip(m.roster(),results,strict=True)]
    report=dict(schema=m.PRODUCER+'.report',status='four_slots_settled_not_admitted',prepared='/saved/own_prepare.json',
        prepare_sha256='1'*64,rows=rows,errors=[],model_before={},model_after={},classes_before=[],classes_after=[],
        precision=[],host={},sources_verified=True,source_manifest_sha256='2'*64,wall_seconds=1.,CUDA_peak_bytes=[0,0],
        journal_tail='3'*64,artifact_sha256={},**m.FLAGS)
    receipt=accepted(report)
    return report,receipt


def accepted(report):
    arms=[]
    for row in report['rows']:
        r=row['result']
        if r is None:continue
        arms.append(dict(slot=row['slot'],root=row['root'],mode=row['mode'],status=row['status'],native_calls=1,complete_native_calls=1,
            model_plan_available=r['model_plan'] is not None,realized_progress_depth=r['realized_progress_depth'],realized_widths=r['realized_widths'],
            resources=deepcopy(r['resource_receipt']),scalar_backward_calls=None,AD_primal_predict_API=None,Jacobian_elements=None))
    settled=all(row['result'] is not None and not row['result']['errors'] for row in report['rows'])
    report['status']='four_slots_settled_not_admitted' if settled else 'partial_or_failed'
    return dict(schema=m.PRODUCER+'.consume',status='saved_four_slot_engineering_verified' if settled else 'saved_partial_engineering_verified',
        prepared=report['prepared'],prepare_sha256=report['prepare_sha256'],report_sha256='4'*64,denominator=4,arms=arms,
        slot_statuses=[x['status'] for x in report['rows']],historical_acquisition=dict(note='fixture-only; no current physics'),
        saved_verification='original queue/core/ridge recomputed; full native IO algebra; no neural parity recomputation',total_FLOPs=None,**m.FLAGS)


def test_all_four_original144_and_actual_tree_dimensions(data):
    r,c=data;s=m.summarize(r,c)
    assert s['denominator']==dict(arms=4,expansion_slots_per_arm=144,total_expansion_slots=576)
    assert s['plan_available_count']==3 and [a['observed_job_rows'] for a in s['arms']]==[144]*4
    fixed,adaptive,_,refused=s['arms']
    assert fixed['realized_progress_depth']==3 and fixed['returned_node_depth_counts']=={'1':4,'2':16,'3':16}
    assert adaptive['realized_progress_depth']==1 and adaptive['original_child_stops']==4
    assert adaptive['returned_extension_H_counts']=={'3':4}
    assert refused['refusal_nodes']==1 and refused['queue_nodes'][0]['requested_width']==8
    assert refused['job_status_counts']=={'not_requested':136,'refused_unsupported_original_type':8} and not refused['plan_available']
    assert all(s[k] is False for k in m.FLAGS)
    assert s['resource_totals']['attempted']['all_four_total']==s['resource_totals']['attempted']['known_sum']


def test_parent_J_not_root_and_complete_identity_fields(data):
    s=m.summarize(*data);arm=s['arms'][1];maps=arm['signal_parent_identities']
    assert len(maps)==10 and arm['mapped_child_contexts']>0
    root=next(x for x in maps if x['is_original_root_context'] and x['horizon']==1)
    child=next(x for x in maps if not x['is_original_root_context'] and x['horizon']==1)
    assert root['jacobian_semantic_json_sha256']!=child['jacobian_semantic_json_sha256']
    assert root['native_binding_sha256']==root['outer_binding_sha256']
    assert child['native_binding_sha256']!=child['outer_binding_sha256'] and child['jacobian_shape']==[192,10]
    assert arm['J_API_completed']==arm['J_API_attempted']==arm['signal_mappings_returned']==10


def test_failure_blocked_arms_unknown_not_zero(data):
    r=deepcopy(data[0]);old=runpy.run_path(str(m.ROOT/'tests/test_agent_p3_parent_native_adapter_20261029_v1.py'))
    failure=old['m'].run_core(old['task'](),lambda t,s:old['RAM'](t,s,fail=True),old['Sink'](),mode='fixed',permit_ram=True)
    r['rows'][0].update(status=failure['status'],result=failure)
    for row in r['rows'][1:]:row.update(status='blocked_after_runtime_failure',result=None)
    c=accepted(r);s=m.summarize(r,c)
    assert s['unobserved_resource_arms']==3 and s['resource_totals']['attempted']['all_four_total'] is None
    assert s['resource_totals']['attempted']['known_sum']['predict_API']==1
    assert all(a['resources'] is None and a['job_status_counts'] is None and a['unobserved_expansion_slots']==144 for a in s['arms'][1:])


def test_cap_status_not_relabelled_as_plan_or_full_search(data):
    r=deepcopy(data[0]);old=runpy.run_path(str(m.ROOT/'tests/test_agent_p3_parent_native_adapter_20261029_v1.py'))
    caps=old['m'].default_caps('adaptive');caps['predict_API']=19
    cap,_,_=old['run'](caps=caps)
    # Shape fixture for the original cap-disposition schema, not new execution evidence.
    cap['resource_receipt']['caps']['predict_API']=562
    r['rows'][1].update(status=cap['status'],result=cap)
    s=m.summarize(r,accepted(r));arm=s['arms'][1]
    assert arm['cap_nodes']==1 and not arm['plan_available'] and arm['queue_nodes'][0]['allocated_jobs']==0
    assert arm['job_status_counts']=={'not_requested':144}


def test_child_R_after_progress_preserves_diagnostic_not_plan(data):
    # Saved schema fixture, not a policy re-experiment: a paid child requests R8.
    r=deepcopy(data[0]);runtime=r['rows'][1]['result'];queue=runtime['queues'][1]
    d=queue['decision'];d.update(extension_type='reconfiguration',candidates=8,program_depth_limit=3,continue_expanding=True)
    template=deepcopy(runtime['jobs'][0]['request']);parent_node=next(x['node'] for x in runtime['jobs'] if x['node'] is not None and m.digest(x['node']['context'])==queue['parent_context_sha256'])
    reqs=[]
    for i in range(8):
        req=deepcopy(template);req.update(d,candidate_index=i,parent_context=parent_node['context'],
            parent_context_sha256=queue['parent_context_sha256'],prefix_edges=parent_node['edges'],prefix_actions=parent_node['actions'],
            signal_sha256=m.digest(runtime['signals'][1]),local_response=runtime['signals'][1]['responses'])
        reqs.append(req)
    queue.update(status='refused_unsupported_original_type',requests=reqs)
    for i,req in enumerate(reqs,4):runtime['jobs'][i].update(status='refused_unsupported_original_type',request=req)
    runtime.update(status='refused_unsupported_original_type',queues=runtime['queues'][:2],model_plan=None)
    r['rows'][1]['status']=runtime['status'];s=m.summarize(r,accepted(r));arm=s['arms'][1]
    assert arm['diagnostic_best_available'] is True and arm['plan_available'] is False
    assert arm['job_status_counts']=={'not_requested':132,'refused_unsupported_original_type':8,'returned':4}
    assert arm['realized_progress_depth']==1 and arm['queue_nodes'][-1]['allocated_jobs']==8


@pytest.mark.parametrize('damage',('denominator','order','receiptstatus','resources','plan','flag','schema','prepared','partialreceipt','unknownzero'))
def test_failclosed_input_contract_mutations(data,damage):
    r,c=deepcopy(data)
    if damage=='denominator':r['rows'][0]['result']['jobs'].pop()
    elif damage=='order':r['rows'].reverse()
    elif damage=='receiptstatus':c['slot_statuses'][0]='madeup'
    elif damage=='resources':c['arms'][0]['resources']['attempted']['predict_API']+=1
    elif damage=='plan':c['arms'][3]['model_plan_available']=True
    elif damage=='flag':c['P1_admission']=True
    elif damage=='schema':c['schema']='old.consume'
    elif damage=='prepared':c['prepare_sha256']='5'*64
    elif damage=='partialreceipt':c.pop('saved_verification')
    else:c['arms'][0]['scalar_backward_calls']=0
    with pytest.raises(ValueError):m.summarize(r,c)


@pytest.mark.parametrize('damage',('root','parent','J','raw','older','rebasedbinding'))
def test_signal_parent_lineage_mutations(data,damage):
    r=deepcopy(data[0]);mapping=r['rows'][1]['result']['signal_mappings'][2]
    if damage=='root':mapping['root_task_sha256']='f'*64
    elif damage=='parent':mapping['native_response']['parent_context_sha256']='f'*64
    elif damage=='J':mapping['native_response']['jacobian'].pop()
    elif damage=='raw':mapping['outer_request']['parent_context']['action_history'][0][0]+=.25
    elif damage=='older':mapping['outer_request']['parent_context']['latent_history'][0][0]+=.25
    else:mapping['parent_task']['binding']['model_sha256']='f'*64
    mapping['mapping_sha256']=m.digest({k:v for k,v in mapping.items() if k!='mapping_sha256'})
    with pytest.raises(ValueError):m.summarize(r,accepted(r))


def write_fixture(tmp_path,data):
    r,c=deepcopy(data);rp=tmp_path/'report.json';rp.write_bytes(m.canonical(r));rh=m.sha(rp)
    c['report_sha256']=rh;cp=tmp_path/'receipt.json';cp.write_bytes(m.canonical(c))
    return rp,rh,cp,m.sha(cp)


def test_saved_build_reads_only_two_explicit_inputs(tmp_path,data):
    args=write_fixture(tmp_path,data);s=m.build(*args)
    assert s['inputs']['report']['sha256']==args[1] and s['inputs']['consume']['sha256']==args[3]
    assert s['inputs']['original_prepare']['read_or_rebuilt_here'] is False
    assert not (tmp_path/'events.jsonl').exists()
    # The referenced prepare path does not exist; no producer/fullstrict may be called.
    assert s['plan_available_count']==3


def test_external_SHA_and_cross_report_receipt_rejected(tmp_path,data):
    rp,rh,cp,ch=write_fixture(tmp_path,data)
    with pytest.raises(ValueError,match='SHA mismatch'):m.build(rp,'f'*64,cp,ch)
    c=m.json.loads(cp.read_bytes());c['report_sha256']='f'*64;cp.write_bytes(m.canonical(c))
    with pytest.raises(ValueError,match='binds actual report'):m.build(rp,rh,cp,m.sha(cp))


def test_duplicate_json_and_symlink_refused(tmp_path):
    p=tmp_path/'x.json';p.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError,match='duplicate'):m.load(p,m.sha(p),1024)
    link=tmp_path/'link.json';link.symlink_to(p)
    with pytest.raises(ValueError,match='regular'):m.load(link,m.sha(p),1024)


def test_frozen_producer_pins():assert all(m.sha(p)==h for p,h in m.PINS.items())


def test_MAIN_actual_report_and_completed_receipt_defaults():
    assert m.REPORT_SHA=='75f4c162aaa75223dfd8523381e7614c8a35637afd52278df8a80d6affa34e40'
    assert m.REPORT.name=='report.json'
    assert m.CONSUME_SHA=='622d1cdaf8d1ed69c4e78f8d1f5c7c6114285db6121f0021fbbaf6573a8a76ad'
    assert m.CONSUME.name=='agent_p3_parent_native_cuda_20261029_v1_consume_MAIN01.json'


def test_cold_stdlib_no_producer_numpy_torch_or_physics_import():
    code=f'import sys; from experiments.spectral_reachability import {m.NAME} as m; assert not any(x in sys.modules for x in ("torch","numpy","pymunk","experiments.spectral_reachability.{m.PRODUCER}")); m.main(["--help"])'
    p=subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert 'analyze' in p.stdout and 'verify' in p.stdout
