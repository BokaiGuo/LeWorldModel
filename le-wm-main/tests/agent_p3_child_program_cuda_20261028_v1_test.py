"""Explicit NumPy RAM fixtures, never model/CUDA/physical evidence."""
from copy import deepcopy
from functools import lru_cache
import builtins
import gzip
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_child_program_cuda_20261028_v1 as m


def load_test(stem):
    spec=importlib.util.spec_from_file_location('_RAM_'+stem,m.ROOT/f'tests/{stem}_test.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


c=load_test('agent_p3_child_scheduler_contract_20261028_v1')
old=load_test('agent_p3_native_program_20261028_v1')


def source_fixture(geometry='short',observed=1,sharing='independent'):
    tasks=[c.task(),c.task(1)];children=[];rows=[]
    for index,(eid,boundary) in enumerate((e,b) for e in m.contract.EPISODES for b in m.contract.STEPS):
        child=c.child(eid,boundary);children.append(child)
        if index>=observed:
            child['status']='not_reached'
            for key in ('context','context_sha256','raw_history10','executed_prefix','observation','original_compiler_context_sha256'):child[key]=None
            child['ancestry']['original_request_sha256']=None
            rows.append(dict(child_id=child['child_id'],status='not_reached',result=None,error=None));continue
        task=tasks[child['root_slot']];acquisitions=[];methods=[]
        groups=[('shared',m.contract.MODES)] if sharing=='shared_equal' else [(mode,(mode,)) for mode in m.contract.MODES]
        for group,modes in groups:
            responses=[]
            for h in (1,3):
                j=np.zeros((192,10*h),np.float32);nominal=np.zeros(192,np.float32)
                if geometry=='short':j[0,0]=1
                elif geometry=='long':
                    nominal[:2]=-1.;j[0,0]=1
                    if h==3:j[1,1]=1
                elif geometry!='refuse':raise AssertionError('RAM fixture geometry')
                responses.append(dict(binding_sha256=m.digest(task['binding']),parent_context_sha256=child['context_sha256'],
                    horizon=h,nominal=nominal.tolist(),jacobian=j.tolist(),producer_sha256=m.sha(m.paid.__file__)))
            signals,signal=m.paid.recompute_signal(task,child,responses)
            acquisition=dict(group=child['child_id']+':'+group,status='returned',signal=signal,
                resource_receipt=dict(RAM_fixture_only=True),error=None,backend_kind='RAM_FIXTURE')
            acquisitions.append(acquisition)
            for mode in modes:
                d=m.core.choose(task,dict(context=child['context'],depth=0),signals,mode);h,n=d['horizon'],d['candidates']
                queue=[m.core.request(task,dict(context=child['context']),operation='compile_child_program_extension',**d,
                    child_id=child['child_id'],child_sha256=m.digest(child),candidate_index=k,prefix_actions=[],prefix_edges=[],
                    actual_executed_prefix=child['executed_prefix'],signal_sha256=m.digest(signal),local_response=responses) for k in range(n)]
                methods.append(dict(mode=mode,status=m.BLOCKED if 2*h*n>24 else 'pending_reviewed_child_compiler',decision=d,queue=queue,
                    signal_group=acquisition['group'],signal_charge=dict(numerator=1,denominator=len(modes)),
                    reserved_search_predict_API=2*h*n,search_predict_cap=24,selected_prefix_predict_reserve=1))
        rows.append(dict(child_id=child['child_id'],status='returned',result=dict(acquisitions=acquisitions,rows=methods),error=None))
    contract=dict(tasks=tasks,children=children,inputs=dict(sharing=sharing))
    receipt=dict(schema=m.paid.NAME+'.consume',fixed_denominator=48,child_denominator=24,rows=rows,
        resources=dict(RAM_historical_only=True),saved_verification=dict(array_SVD_calls=4*observed),**m.FLAGS)
    return contract,receipt


def prepared_fixture(geometry='short',observed=1,sharing='independent'):
    contract,receipt=source_fixture(geometry,observed,sharing);branches=m.branches_from(contract,receipt)
    return dict(schema=m.NAME+'.prepare',tasks=contract['tasks'],children=contract['children'],branches=branches,
        rule=m.rule(),profile=m.profile(branches),sources={m.__file__:m.sha(m.__file__)},
        expected_model_hash=m.digest('RAM_only_not_GPU'),historical_signal_resources=receipt['resources'],
        paid_saved_consumer_array_SVD_calls=receipt['saved_verification']['array_SVD_calls'],**m.FLAGS)


class Sink:
    def __init__(self):self.events=[]
    def event(self,e):self.events.append(deepcopy(e))


class RamReplay:
    kind='RAM_FIXTURE'
    def __init__(self,task,sink,fault=None):self.task,self.sink,self.fault=task,sink,fault;self.event_sink=sink.event
    def replay(self,parent,actions,phase,ledger):
        if self.fault=='raise':raise RuntimeError('RAM paid replay fault')
        # Old NumPy-only fixture does not enforce/root-rebind a task. Relabel
        # fixture phases locally; no producer or model globals are modified.
        class PhaseSink:
            def event(inner,e):
                value=deepcopy(e)
                if 'phase' in value:value['phase']=phase
                self.sink.event(value)
        path=old.RamReplay(self.task,PhaseSink()).replay(parent,actions,phase=='segmented',ledger)
        if self.fault=='prefix' and phase=='selected_prefix':path[0]['latent_history'][-1][0]+=1
        if self.fault=='segmented' and phase=='segmented':path[-1]['latent_history'][-1][0]+=1
        return path


@lru_cache(maxsize=4)
def complete(geometry='short'):
    p=prepared_fixture(geometry);rows=m.initial_rows(p['branches']);traces={}
    for i,(branch,row) in enumerate(zip(p['branches'],rows,strict=True)):
        if branch['status']!=m.READY:continue
        sink=Sink();child=p['children'][i//2]
        m.execute_branch(p['tasks'][child['root_slot']],child,branch,row,
            lambda child,branch,sink:RamReplay(p['tasks'][child['root_slot']],sink),sink,permit_ram=True)
        traces[i]=sink.events
    return p,rows,traces


@pytest.mark.parametrize('geometry,expected_h',[('short',1),('long',3)])
def test_full_child_compile_and_saved_native_closure(geometry,expected_h):
    p,rows,traces=complete(geometry)
    assert len(rows)==48 and sum(r['status']=='not_reached' for r in rows)==46
    for i,events in traces.items():
        b=p['branches'][i];ch=p['children'][i//2];r=rows[i]
        m.validate_branch(p['tasks'][ch['root_slot']],ch,b,r,events)
        assert b['decision']['horizon']==(1 if b['mode']=='fixed' else expected_h)
        assert len(r['candidates'])==b['decision']['candidates']==4
        resources=r['resource_receipt']['completed'];h=b['decision']['horizon']
        assert resources['predict_API']==resources['predict_samples']==2*h*4+1
        assert resources['replay_calls']==9 and resources['metric_calls']==4
        assert resources['jacobian_API']==resources['VJP_directions']==resources['encoder_calls']==resources['projector_calls']==0
        assert r['selected']['prefix']==r['candidates'][r['selected']['index']]['whole'][:1]
        assert not r['selected']['physical_release']


@pytest.mark.parametrize('h,n,status',[(1,4,m.READY),(1,8,m.READY),(3,4,m.READY),(3,8,m.BLOCKED)])
def test_exact_original_quota_budget_gate(h,n,status):
    d=dict(horizon=h,candidates=n,program_type='progress',extension_type='progress',continue_expanding=True)
    assert m.eligibility(d)==status
    if status==m.READY:assert m.caps(dict(status=status,decision=d))['predict_API']==2*h*n+1
    else:assert not any(m.caps(dict(status=status,decision=d)).values())


def test_refusal_all48_no_selection_of_eligible_subset_and_no_backend_calls():
    p=prepared_fixture('refuse',23);value=m.plan(p)
    assert sum(value['counts'].values())==48 and value['counts']['not_reached']==2
    assert value['counts'][m.READY]+value['counts'][m.BLOCKED]+value['counts'][m.UNSUPPORTED]==46
    assert value['counts'][m.BLOCKED]+value['counts'][m.UNSUPPORTED]>0
    # Fixed remains progress; adaptive refusal never deletes its method slot.
    assert len([b for b in value['branches'] if b['mode']=='fixed'])==24
    for b in p['branches']:
        if b['status']==m.READY:continue
        with pytest.raises(ValueError):m.execute_branch(p['tasks'][b['root_slot']],p['children'][0],b,
            m.initial_rows([b])[0],lambda *a:pytest.fail('refused factory called'),Sink(),permit_ram=True)


def test_undefined_reconfiguration_not_substituted_with_progress():
    d=dict(horizon=1,candidates=8,program_type='reconfiguration',extension_type='reconfiguration',continue_expanding=True)
    assert m.eligibility(d)==m.UNSUPPORTED
    d['horizon']=3;assert m.eligibility(d)==m.BLOCKED


@pytest.mark.parametrize('sharing',['independent','shared_equal'])
def test_child_local_paid_projection_and_rational_signal_charge(sharing):
    contract,receipt=source_fixture('long',2,sharing);branches=m.branches_from(contract,receipt)
    assert len(branches)==48
    for i in range(4):
        b=branches[i];child=contract['children'][i//2]
        assert b['response']['parent_context_sha256']==child['context_sha256']
        assert b['response']['parent_context_sha256']!=m.digest(contract['tasks'][0]['root_context'])
        assert b['signal_charge']==dict(numerator=1,denominator=2 if sharing=='shared_equal' else 1)


@pytest.mark.parametrize('fault',['drop_child','drop_method','root_J','cross_child','quota','queue','producer','missing_paid'])
def test_saved_paid_projection_tampering_rejected(fault):
    contract,receipt=source_fixture('long');row=receipt['rows'][0];result=row['result']
    if fault=='drop_child':receipt['rows'].pop()
    elif fault=='drop_method':result['rows'].pop()
    elif fault=='root_J':result['acquisitions'][0]['signal']['responses'][0]['parent_context_sha256']=m.digest(contract['tasks'][0]['root_context'])
    elif fault=='cross_child':row['child_id']='e99_wrong'
    elif fault=='quota':result['rows'][0]['decision']['candidates']=8
    elif fault=='queue':result['rows'][0]['queue'].pop()
    elif fault=='producer':result['acquisitions'][0]['signal']['responses'][0]['producer_sha256']='0'*64
    else:row['status']='failed'
    with pytest.raises((ValueError,IndexError,KeyError)):m.branches_from(contract,receipt)


@pytest.mark.parametrize('fault',['raw','history','normalizer','module','shape','basis_extra','cost','selected','prefix',
    'selected_index','drop_candidate','drop_io','resource','scope','nesting','queue_bank','unknown','reorder'])
def test_saved_compiler_tampering_rejected(fault):
    p,rows,traces=deepcopy(complete('long'));i=1;b=p['branches'][i];ch=p['children'][i//2];row=rows[i];events=traces[i]
    def mutate(kind,key):
        e=next(e for e in events if e['kind']==kind and key in e.get('arrays',{}));a=m.base.unblob(e['arrays'][key]).copy()
        a.flat[0]+=1;e['arrays'][key]=m.base.blob(a)
    if fault=='raw':mutate('program_packet','raw')
    elif fault=='history':mutate('program_packet','history')
    elif fault=='normalizer':mutate('program_packet','std')
    elif fault=='module':mutate('io_input','predictor:0')
    elif fault=='shape':next(op for op in row['resource_receipt']['events'] if op['resource']=='predict')['shape'][0]=2
    elif fault=='basis_extra':
        e=next(e for e in events if e['kind']=='program_packet');e['arrays']['basis']=m.base.blob(np.ones(1,np.float32))
    elif fault=='cost':row['candidates'][0]['cost']+=1
    elif fault=='selected':row['selected']['selection_digest']='0'*64
    elif fault=='prefix':row['selected']['prefix'][0]['latent_history'][-1][0]+=1
    elif fault=='selected_index':row['selected']['index']=99
    elif fault=='drop_candidate':row['candidates'].pop()
    elif fault=='drop_io':events.pop(next(i for i,e in enumerate(events) if e['kind']=='io_return'))
    elif fault=='resource':row['resource_receipt']['attempted']['predict_API']-=1
    elif fault=='scope':next(e for e in events if e['kind']=='io_input')['scope']['child_id']='other'
    elif fault=='nesting':events[0],events[1]=events[1],events[0]
    elif fault=='queue_bank':row['bank']['fractions'][-1]=.5
    elif fault=='unknown':events.append(dict(kind='success_oracle'))
    else:
        j=next(i for i,e in enumerate(events) if e['kind']=='io_input');events[j],events[j+1]=events[j+1],events[j]
    with pytest.raises((ValueError,KeyError,IndexError)):m.validate_branch(p['tasks'][ch['root_slot']],ch,b,row,events)


@pytest.mark.parametrize('fault',['raise','prefix','segmented'])
def test_failed_replay_keeps_attempts_and_partial_candidates_no_winner(fault):
    p=prepared_fixture();b=p['branches'][0];ch=p['children'][0];row=m.initial_rows([b])[0];sink=Sink()
    with pytest.raises((RuntimeError,ValueError)):
        m.execute_branch(p['tasks'][0],ch,b,row,lambda c,b,s:RamReplay(p['tasks'][0],s,fault),sink,permit_ram=True)
    assert row['status']=='failed' and row['selected'] is None
    receipt=row['resource_receipt'];m.core.audit_ledger(receipt)
    assert receipt['attempted']['replay_calls']>0
    if fault=='raise':assert receipt['attempted']['replay_calls']==1 and receipt['completed']['replay_calls']==0


def test_authorization_and_invalid_native_request_before_torch_or_factory(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(m,'verify',lambda *a:calls.append('verify'))
    with pytest.raises(ValueError):m.run('/not_read','0'*64,str(tmp_path/'not_created'),None)
    assert calls==[] and not (tmp_path/'not_created').exists()
    p=prepared_fixture();b=p['branches'][0];ch=p['children'][0]
    with pytest.raises(ValueError):m.execute_branch(p['tasks'][0],ch,b,m.initial_rows([b])[0],lambda *a:calls.append('factory'),Sink())
    assert calls==[]
    backend=object.__new__(m.NativeProgram);backend.device=SimpleNamespace(type='cuda');backend.child=ch
    with pytest.raises(ValueError):backend.replay(p['tasks'][0]['root_context'],[[0.,0.]]*5,'whole',None)


def test_import_and_cli_help_have_no_neural_or_physics():
    code='import sys; from experiments.spectral_reachability import '+m.NAME+'; assert not any(x in sys.modules for x in ("torch","gym","gymnasium","pymunk"))'
    subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,check=True,capture_output=True,timeout=15)
    result=subprocess.run([sys.executable,'-B','-m','experiments.spectral_reachability.'+m.NAME,'--help'],cwd=m.ROOT,
        check=True,capture_output=True,timeout=15)
    assert b'prepare' in result.stdout and b'consume' in result.stdout


def test_explicit_eight_candidate_fraction_wrapper_keeps_quota_without_common_budget_override():
    p=prepared_fixture();b=deepcopy(p['branches'][0]);b['decision']['candidates']=8
    ledger=m.core.Ledger(m.caps(b));ledger.scope=m.scope(b,None,'compile',1)
    bank=ledger.call('program_compile',lambda:m.generate(p['tasks'][0],p['children'][0],b,ledger))
    assert len(bank['actions'])==8 and bank['fractions']==[i/7 for i in range(8)]
    assert ledger.completed['solve_calls']==ledger.completed['array_SVD_calls']==1
    assert ledger.completed['predict_API']==0  # Saved-array generation only, no fake replay.


def saved_ram_panel(tmp_path,all_refused=False):
    if all_refused:
        p=prepared_fixture(observed=0);rows=m.initial_rows(p['branches']);traces={}
    else:p,rows,traces=deepcopy(complete('long'))
    output=tmp_path/'RAM_saved_not_GPU';store=m.paid.Store(output,p['profile'])
    binding=dict(prepared=str(tmp_path/'RAM_prepare.json'),prepare_sha256=m.digest('RAM_only_prepare'))
    store.put('binding.json',binding)
    store.event(dict(kind='denominator',rows=m.initial_rows(p['branches']),fixed_denominator=48,child_denominator=24))
    active=not all_refused;hashes=dict(original_state_dict=p['expected_model_hash'],all_buffers=m.digest('RAM_only_buffers')) if active else None
    store.event(dict(kind='model_before',hashes=hashes,native_dispatch=active))
    for i,b in enumerate(p['branches']):
        if b['status']!=m.READY:continue
        for e in traces[i]:store.event(dict(kind='compiler_trace',group=b['child_id']+':'+b['mode'],payload=e))
        store.event(dict(kind='branch_return',row=rows[i]))
    store.event(dict(kind='model_after',hashes=hashes,native_dispatch=active))
    store.event(dict(kind='sources_final',sha256=m.digest(p['sources'])));store.close()
    r=dict(schema=m.NAME+'.report',status='child_panel_saved_not_admitted',**binding,rows=rows,resources=m.totals(rows),errors=[],
        model_before=hashes,model_after=hashes,native_dispatch=active,
        host=dict(device='cuda:0',index=0,name='RAM_NOT_GPU',allocation_sum=4.,torch_version='RAM',torch_cuda='RAM',
            visible_nvidia_nodes=['RAM_NOT_GPU']) if active else None,sources_verified=True,cleanup_ok=True,
        source_manifest_sha256=m.digest(p['sources']),fixed_denominator=48,child_denominator=24,wall_seconds=1.,
        CUDA_peak_bytes=[0,0] if active else None,journal_tail=store.tail,
        artifact_sha256={n:m.sha(output/n) for n in ('binding.json',m.paid.JOURNAL)},**m.FLAGS)
    store.put('report.json',r);return p,output,m.sha(output/'report.json')


@pytest.mark.parametrize('all_refused',[False,True])
def test_strict_full_saved_ram_panel_has_no_model_calls(tmp_path,monkeypatch,all_refused):
    p,output,h=saved_ram_panel(tmp_path,all_refused)
    calls=[];monkeypatch.setattr(m,'verify',lambda *a:(calls.append(a) or p));monkeypatch.setattr(m,'pins',lambda:{})
    result=m.consume(output,h)
    assert len(calls)==1 and result['fixed_denominator']==48 and len(result['rows'])==48
    assert result['new_resources']['completed']['predict_API']==(0 if all_refused else 34)
    assert result['saved_verification']['neural_calls']==result['saved_verification']['physics_calls']==0
    assert result['physical_benefit'] is None and not result['full_tree'] and all(result[k] is False for k in m.FLAGS)


@pytest.mark.parametrize('fault',['external_SHA','gate','unknown_field','drop_method','model','wall','resources','extra_file','gzip','guard','branch_order','drop_return','cross_child'])
def test_full_saved_boundary_and_semantic_rechain_tampering(tmp_path,monkeypatch,fault):
    p,output,h=saved_ram_panel(tmp_path);r=m.base.read(output/'report.json',256*1024**2)
    calls=[];monkeypatch.setattr(m,'verify',lambda *a:(calls.append(a) or p));monkeypatch.setattr(m,'pins',lambda:{})
    if fault=='external_SHA':
        with pytest.raises(ValueError):m.consume(output,'0'*64)
        assert calls==[];return
    if fault=='gate':r['P3_admission']=True
    elif fault=='unknown_field':r['native_success']=True
    elif fault=='drop_method':r['rows'].pop()
    elif fault=='model':r['model_after']['original_state_dict']='0'*64
    elif fault=='wall':r['wall_seconds']=901
    elif fault=='resources':r['resources']['completed']['predict_API']-=1
    elif fault=='extra_file':(output/'unexpected').write_bytes(b'RAM')
    elif fault=='gzip':
        path=output/m.paid.JOURNAL;path.write_bytes(path.read_bytes()[:-1]);r['artifact_sha256'][m.paid.JOURNAL]=m.sha(path)
    else:
        events,_=m.paid.journal(output/m.paid.JOURNAL)
        if fault=='guard':events.insert(2,deepcopy(events[1]))
        elif fault=='drop_return':events=[e for e in events if e['kind']!='branch_return']
        elif fault=='branch_order':
            returns=[i for i,e in enumerate(events) if e['kind']=='branch_return'];a,b=returns
            events[a],events[b]=events[b],events[a]
        else:next(e for e in events if e['kind']=='compiler_trace')['group']='other:fixed'
        tail='0'*64;encoded=[]
        for i,e in enumerate(events):
            record=dict(index=i,previous=tail,payload=e);record['sha256']=m.digest(record);tail=record['sha256']
            encoded.append(m.canonical(record)+b'\n')
        (output/m.paid.JOURNAL).write_bytes(gzip.compress(b''.join(encoded),mtime=0))
        r['journal_tail']=tail;r['artifact_sha256'][m.paid.JOURNAL]=m.sha(output/m.paid.JOURNAL)
    (output/'report.json').write_bytes(m.canonical(r))
    with pytest.raises((ValueError,KeyError,IndexError)):m.consume(output,m.sha(output/'report.json'))


def paid_saved_boundary(tmp_path):
    contract,receipt=source_fixture('long');folder=tmp_path/'RAM_paid';folder.mkdir()
    prepared=tmp_path/'RAM_paid_prepare.json';saved=tmp_path/'RAM_full_receipt.json'
    p=dict(schema=m.paid.NAME+'.prepare',child_contract=contract,sources={m.__file__:m.sha(m.__file__)},
        policy='/RAM_policy_not_loaded',expected_model_hash=m.digest('RAM_model'))
    prepared.write_bytes(m.canonical(p));saved.write_bytes(m.canonical(receipt))
    report=dict(prepared=str(prepared),prepare_sha256=m.sha(prepared),sharing='independent')
    (folder/'report.json').write_bytes(m.canonical(report));(folder/'binding.json').write_bytes(b'{}')
    (folder/m.paid.JOURNAL).write_bytes(b'RAM_ONLY_not_a_CUDA_journal')
    inputs=dict(signal_output=str(folder),report_sha256=m.sha(folder/'report.json'),signal_receipt=str(saved),receipt_sha256=m.sha(saved))
    return inputs,receipt


def test_material_calls_complete_real_consume_interface_before_projection(tmp_path,monkeypatch):
    inputs,receipt=paid_saved_boundary(tmp_path);calls=[]
    monkeypatch.setattr(m,'inputs_template',lambda:deepcopy(inputs))
    def consumer(output,expected):calls.append((output,expected));return deepcopy(receipt)
    monkeypatch.setattr(m.paid,'consume',consumer)
    value=m.material(inputs)
    assert calls==[(inputs['signal_output'],inputs['report_sha256'])]
    assert len(value['branches'])==48 and value['fixed_denominator']==48
    assert str(Path(inputs['signal_receipt'])) in value['sources']
    assert all(str(f) in value['sources'] for f in m.own_files())
    assert value['historical_signal_resources']==receipt['resources']


@pytest.mark.parametrize('fault',['receipt_SHA','source_consume','summary','outcome_manifest','missing_receipt'])
def test_upstream_failure_never_bypasses_consumer_or_projects_partial(tmp_path,monkeypatch,fault):
    inputs,receipt=paid_saved_boundary(tmp_path);calls=[]
    monkeypatch.setattr(m,'inputs_template',lambda:deepcopy(inputs))
    monkeypatch.setattr(m,'branches_from',lambda *a:pytest.fail('must not project unverified paid rows'))
    def consumer(*a):
        calls.append(a)
        if fault=='source_consume':raise ValueError('real producer refused')
        return receipt
    monkeypatch.setattr(m.paid,'consume',consumer)
    if fault=='receipt_SHA':inputs['receipt_sha256']='0'*64
    elif fault=='summary':
        path=Path(inputs['signal_receipt']);path.write_bytes(m.canonical(dict(status='not_the_full_receipt')));inputs['receipt_sha256']=m.sha(path)
    elif fault=='outcome_manifest':inputs['native_success']=True
    elif fault=='missing_receipt':inputs['signal_receipt']=str(tmp_path/'future_receipt_does_not_exist')
    with pytest.raises((ValueError,FileNotFoundError)):m.material(inputs)
    assert len(calls)==(1 if fault in ('source_consume','summary') else 0)


def test_prepare_verify_rebuild_manifest_and_cli_plan_exclusive(tmp_path,monkeypatch,capsys):
    root=tmp_path/'repo';(root/'results').mkdir(parents=True);monkeypatch.setattr(m,'ROOT',root)
    p=prepared_fixture();manifest=tmp_path/'RAM_inputs.json';manifest.write_bytes(m.canonical(dict(RAM_only=True)))
    monkeypatch.setattr(m,'material',lambda inputs:deepcopy(p))
    prepared=root/'results'/(m.NAME+'_RAM_prepare.json')
    assert m.main(['prepare','--input',str(manifest),'--sha256',m.sha(manifest),'--output',str(prepared)])==0
    h=m.sha(prepared);actual=m.verify(prepared,h)
    assert actual['sources'][str(manifest)]==m.sha(manifest)
    output=root/'results'/(m.NAME+'_RAM_plan.json')
    assert m.main(['plan','--prepared',str(prepared),'--sha256',h,'--output',str(output)])==0
    assert m.base.read(output,256*1024**2)['fixed_denominator']==48
    with pytest.raises(ValueError):m.main(['plan','--prepared',str(prepared),'--sha256',h,'--output',str(output)])
    manifest.write_bytes(m.canonical(dict(RAM_changed=True)))
    with pytest.raises(ValueError):m.verify(prepared,h)


def test_real_saved_MAIN02_metadata_projection_keeps41_5_2_without_producer_or_model_execution():
    assert m.sha(m.SIGNAL_OUTPUT/'report.json')==m.SIGNAL_REPORT_SHA
    assert m.sha(m.SIGNAL_RECEIPT)==m.SIGNAL_RECEIPT_SHA and m.SIGNAL_RECEIPT.stat().st_size==94023655
    report=m.base.read(m.SIGNAL_OUTPUT/'report.json',256*1024**2)
    receipt=m.base.read(m.SIGNAL_RECEIPT,256*1024**2)
    assert m.sha(report['prepared'])==report['prepare_sha256']
    paid_prepare=m.base.read(report['prepared'],64*1024**2)
    branches=m.branches_from(paid_prepare['child_contract'],receipt)
    assert len(branches)==48
    assert sum(b['status']==m.READY for b in branches)==41
    assert sum(b['status']==m.BLOCKED for b in branches)==5
    assert sum(b['status']=='not_reached' for b in branches)==2
    assert not any(b['status']==m.UNSUPPORTED for b in branches)
    profile=m.profile(branches);assert profile['resource_caps']['predict_API']==657
    assert sum(m.caps(b)['predict_API'] for b in branches if b['mode']=='fixed')==207
    assert sum(m.caps(b)['predict_API'] for b in branches if b['mode']=='adaptive')==450
    assert profile['resource_caps']['array_SVD_calls']==41
    assert profile['resource_caps']['replay_calls']==369 and profile['resource_caps']['metric_calls']==164
    assert profile['resource_caps']['jacobian_API']==profile['resource_caps']['VJP_directions']==0
    assert m.sha(m.SIGNAL_RECEIPT)==m.SIGNAL_RECEIPT_SHA  # untouched


def test_fixed_actual_MAIN02_authority_cannot_be_replaced_before_consumption(monkeypatch):
    inputs=m.inputs_template();inputs['report_sha256']='0'*64
    monkeypatch.setattr(m.paid,'consume',lambda *a:pytest.fail('wrong cohort must fail first'))
    with pytest.raises(ValueError,match='fixed actual MAIN02'):m.material(inputs)


def test_inputs_cli_exact_actual_hash_and_no_overwrite(tmp_path,monkeypatch):
    root=tmp_path/'repo';(root/'results').mkdir(parents=True);monkeypatch.setattr(m,'ROOT',root)
    path=root/'results'/(m.NAME+'_RAM_inputs.json')
    assert m.main(['inputs','--output',str(path)])==0
    assert m.sha(path)=='f7455c34f7b1035fc8043b7500e4d70ba23b441ab21454cd3cdbded378a90ad2'
    assert m.base.read(path)==m.inputs_template()
    with pytest.raises(ValueError):m.main(['inputs','--output',str(path)])


def test_eight_candidates_execute_all16_search_plus_one_prefix_without_dedup():
    p=prepared_fixture();b=deepcopy(p['branches'][0]);b['decision']['candidates']=8
    child=p['children'][0];row=m.initial_rows([b])[0];sink=Sink()
    m.execute_branch(p['tasks'][0],child,b,row,lambda c,b,s:RamReplay(p['tasks'][0],s),sink,permit_ram=True)
    m.validate_branch(p['tasks'][0],child,b,row,sink.events)
    assert len(row['candidates'])==8 and row['resource_receipt']['completed']['predict_API']==17
    assert row['resource_receipt']['completed']['replay_calls']==17


def test_exact_tie_keeps_original_index_and_still_replays_every_duplicate():
    p=prepared_fixture();b=deepcopy(p['branches'][0]);b['response']['jacobian']=np.zeros((192,10),np.float32).tolist()
    child=p['children'][0];row=m.initial_rows([b])[0];sink=Sink()
    m.execute_branch(p['tasks'][0],child,b,row,lambda c,b,s:RamReplay(p['tasks'][0],s),sink,permit_ram=True)
    m.validate_branch(p['tasks'][0],child,b,row,sink.events)
    assert len({m.digest(c['actions']) for c in row['candidates']})==1
    assert row['selected']['index']==0 and row['resource_receipt']['completed']['predict_API']==9
