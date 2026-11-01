"""NumPy/mock only: no neural forward, CUDA allocation, or physics."""
from copy import deepcopy
from types import SimpleNamespace as NS
import json
import numpy as np
import pytest
from experiments.spectral_reachability import p2_composition_predict_frontier_cuda_20261027_v1 as m
from test_p2_composition_predict_frontier_20261027_v1 import base,legacy_inputs


@pytest.mark.parametrize('arm,n,suffix_count',[(m.ARMS[0],80,0),(m.ARMS[1],40,80)])
def test_real_new_runner_dispatches_every_slot_and_double_replay(arm,n,suffix_count):
    old,args,calls,suffix_calls=legacy_inputs();args['bank']=m.candidate_bank(base(),m.ROOTS[0]);events=[]
    r=m.run_arm(arm=arm,journal=events.append,**args)
    assert r['complete'] and r['error'] is None and len(r['attempts'])==n
    assert len(calls)==164 and len(suffix_calls)==suffix_count
    assert len(r['ledger'])==2*n+4 and r['selection_evaluations']==n
    assert all(e['completed'] and len(e['actions'])==10 for e in r['attempts'])
    assert r['winner']['cost']==min(e['cost'] for e in r['attempts'])
    assert r['winner']['whole_context']==r['winner']['segmented_contexts'][-1]
    assert r['nested40_minimum']==min(e['cost'] for e in r['attempts'][:40])
    assert r['ledger'][-1]['path']==[n+2] and r['ledger'][-1]['packet']==1
    assert len([e for e in events if e.get('kind')=='candidate_request'])==n


def test_stop_first_error_no_refund_or_replay():
    old,args,calls,_=legacy_inputs();args['bank']=m.candidate_bank(base(),m.ROOTS[0]);original=args['replay']
    def failing(c,a):
        if len(calls)==3:raise RuntimeError('intentional')
        return original(c,a)
    args['replay']=failing;events=[];r=m.run_arm(arm=m.ARMS[0],journal=events.append,**args)
    assert not r['complete'] and r['winner'] is None and len(r['attempts'])==2
    assert len(calls)==3 and not r['ledger'][-1]['completed']
    assert events[-1]['kind']=='arm_failure'


def test_production_no_mock_replay():
    _,args,calls,_=legacy_inputs();args.update(fixture=False,bank=m.candidate_bank(base(),m.ROOTS[0]))
    with pytest.raises(ValueError):m.run_arm(arm=m.ARMS[0],**args)
    assert not calls


def fake_model():
    calls=[]
    model=NS(predict=lambda x,*a:calls.append('predict'),action_encoder=NS(forward=lambda x:calls.append('action_encoder')),
        encoder=NS(forward=lambda x:calls.append('visual')))
    replay=NS(_p2_scope=None)
    return model,replay,calls


def tensor(kind,shape=None,device='cuda'):
    return NS(shape=shape or (1,3,192 if kind=='predict' else 10),device=NS(type=device))


def dispatch(model,kind,x):
    return model.predict(x) if kind=='predict' else model.action_encoder.forward(x)


def test_predispatch_dual_budget_full332_not336():
    model,replay,calls=fake_model();events=[]
    with m.guarded_entries(model,replay,events.append) as es:
        for spec,scope in zip(m.expected_entries(80),m.expected_scopes()):
            replay._p2_scope=scope;dispatch(model,spec['kind'],tensor(spec['kind']))
        assert m.validate_actual(es)['standalone_predict']==[168,168]
        with pytest.raises(ValueError):model.predict(tensor('predict'))
    assert len(calls)==664 and calls.count('predict')==332 and calls.count('action_encoder')==332
    assert len([e for e in events if e['kind']=='entry_attempt'])==664
    assert events[-1]['kind']=='blocked_entry'


@pytest.mark.parametrize('damage',['scope','batch','context','device','order','visual','hidden'])
def test_guard_blocks_before_forward_and_poison(damage):
    model,replay,calls=fake_model();journal=[];replay._p2_scope=m.expected_scopes()[0]
    with m.guarded_entries(model,replay,journal.append):
        x=tensor('action_encoder');kind='action_encoder'
        if damage=='scope':replay._p2_scope=None
        elif damage=='batch':x.shape=(2,3,10)
        elif damage=='context':x.shape=(1,2,10)
        elif damage=='device':x.device.type='cpu'
        elif damage=='order':kind='predict';x=tensor(kind)
        with pytest.raises(ValueError):
            if damage=='visual':model.encoder.forward(x)
            elif damage=='hidden':model.predict(x)
            else:dispatch(model,kind,x)
        replay._p2_scope=m.expected_scopes()[0]
        with pytest.raises(ValueError):model.action_encoder.forward(tensor('action_encoder'))
    assert calls==[] and len(journal)==2


def test_attempt_and_failed_terminal_durable_and_methods_restored():
    model,replay,calls=fake_model();replay._p2_scope=m.expected_scopes()[0];journal=[]
    def fail(x):raise RuntimeError('kernel failed')
    model.action_encoder.forward=fail
    with m.guarded_entries(model,replay,journal.append) as es:
        with pytest.raises(RuntimeError):model.action_encoder.forward(tensor('action_encoder'))
        with pytest.raises(ValueError):model.predict(tensor('predict'))
    assert model.action_encoder.forward is fail and len(es)==1 and not es[0]['completed']
    assert [e['kind'] for e in journal]==['entry_attempt','entry_terminal','blocked_entry']


@pytest.mark.parametrize('damage',['delete','failed','batch','scope','index'])
def test_actual_census_not_formula(damage):
    specs=m.expected_entries(80);es=[dict(kind=s['kind'],scope=c,index=i,samples=1,context_tokens=3,completed=True)
        for i,(s,c) in enumerate(zip(specs,m.expected_scopes()))]
    if damage=='delete':es.pop()
    elif damage=='failed':es[200]['completed']=False
    elif damage=='batch':es[200]['samples']=2
    elif damage=='scope':es[200]['scope']=None
    else:es[200]['index']=201
    with pytest.raises(ValueError):m.validate_actual(es)


def joint_result(monkeypatch):
    old,args,_,_=legacy_inputs();args['bank']=m.candidate_bank(base(),m.ROOTS[0])
    r=m.run_arm(arm=m.ARMS[0],**args);r['fixture_only']=False
    monkeypatch.setattr(m,'_normalized_token',lambda a:np.asarray(a).reshape(10).tolist())
    compiled=dict(goal_vectors=[np.zeros(192).tolist()],scoring=dict(metric=dict(projection=np.eye(192).tolist())))
    # Existing target metric has the exact same original mean-square formula.
    ready=dict(root=args['root'],action_domain=args['action_domain'])
    return r,dict(arm=m.ARMS[0],bank=args['bank'],ready=ready,prov=args['provenance'],compiled=compiled)


def test_allslot_validation_good_joint(monkeypatch):
    r,kw=joint_result(monkeypatch);best=m.validate_result(r,**kw)
    assert best['candidate_id']==r['winner']['candidate_id']


@pytest.mark.parametrize('damage',[None,'suffix_context','suffix_solve','suffix_cost','prefix_witness'])
def test_conditional_allslot_saved_array_validation(monkeypatch,damage):
    old,args,_,_=legacy_inputs();args['bank']=m.candidate_bank(base(),m.ROOTS[0])
    compiled=dict(goal_vectors=[np.zeros(192).tolist()],scoring=dict(metric=dict(projection=np.eye(192).tolist())))
    def suffix(context,prefix,index):
        out=old.suffix_solution(prefix,np.zeros((192,10)),np.zeros(192),compiled['goal_vectors'][0],np.eye(192))
        out['input_context_sha256']=old.digest(context);return out
    args['suffix']=suffix;r=m.run_arm(arm=m.ARMS[1],**args);assert r['complete'];r['fixture_only']=False
    monkeypatch.setattr(m,'_normalized_token',lambda a:np.asarray(a).reshape(10).tolist())
    kw=dict(arm=m.ARMS[1],bank=args['bank'],ready=dict(root=args['root'],action_domain=args['action_domain']),prov=args['provenance'],compiled=compiled)
    if damage=='suffix_context':r['attempts'][39]['suffix_compilation']['input_context_sha256']='f'*64
    elif damage=='suffix_solve':r['attempts'][39]['suffix_compilation']['suffix_logits'][0]=.5
    elif damage=='suffix_cost':r['attempts'][39]['cost']+=1
    elif damage=='prefix_witness':r['attempts'][39]['witness']['edges'][0]['output_context']['latent_history'][-1][0]+=.1
    if damage:
        with pytest.raises(ValueError):m.validate_result(r,**kw)
    else:m.validate_result(r,**kw)


def operation_fixture():
    operations=[];locals_=[]
    def add(kind,index=None,**kw):
        request=dict(kind=kind,completed=False,**kw)
        if index is not None:request['index']=index
        terminal=dict(request,completed=True)
        if kind=='suffix_solve':terminal.update(svd_calls=1,bounded_solve_calls=1)
        operations.extend([request,terminal])
    add('jacobian',shape=[192,20],vectorize=True)
    add('bounded_ridge_svd');add('mode_svd')
    for i in range(40):
        add('suffix_jacobian',i,shape=[192,10],vectorize=True);add('suffix_solve',i,svd_calls_requested=1)
        locals_.append(dict(suffix_compilation=dict(svd_calls=1,bounded_solve_calls=1)))
    return operations,[dict(arm=m.ARMS[1],attempts=locals_)]


@pytest.mark.parametrize('damage',[None,'missing','wrongAD','request','svd','solve','index'])
def test_AD_SVD_solve_dispatch_receipts(damage):
    operations,results=operation_fixture()
    if damage=='missing':operations.pop()
    elif damage=='wrongAD':operations[0]['shape']=[192,10]
    elif damage=='request':operations[0]['completed']=True
    elif damage=='svd':operations[-1]['svd_calls']=0
    elif damage=='solve':operations[-1]['bounded_solve_calls']=0
    elif damage=='index':operations[-1]['index']=0
    if damage:
        with pytest.raises(ValueError):m.validate_operations(operations,results)
    else:assert m.validate_operations(operations,results)['svd_calls']==42


def test_public_normalization_is_model_free():
    from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
    raw=np.linspace(-1,1,10,dtype=np.float32)
    mean=np.asarray(p0.PUBLIC_ACTION_MEAN,dtype=np.float32).reshape(-1)
    std=np.asarray(p0.PUBLIC_ACTION_STD,dtype=np.float32).reshape(-1)
    if len(mean)==2:mean=np.tile(mean,5)
    if len(std)==2:std=np.tile(std,5)
    assert m._normalized_token(raw.reshape(5,2))==((raw-mean)/std).tolist()


@pytest.mark.parametrize('damage',['tail_action','cost','root','winner','replay','ledger','nested','missing','fixture','provenance'])
def test_allslot_validation_rejects_tampering(monkeypatch,damage):
    r,kw=joint_result(monkeypatch)
    if damage=='tail_action':r['attempts'][79]['actions'][0][0]+=.01
    elif damage=='cost':r['attempts'][79]['cost']+=1
    elif damage=='root':kw['ready']['root']['root_id']='foreign'
    elif damage=='winner':r['winner']['candidate_id']='fake'
    elif damage=='replay':r['winner']['whole_context']['latent_history'][-1][0]+=1
    elif damage=='ledger':r['ledger'][-1]['packet']=0
    elif damage=='nested':r['nested40_minimum']+=1
    elif damage=='missing':r['attempts'].pop()
    elif damage=='fixture':r['fixture_only']=True
    else:kw['prov']['source_sha256']='f'*64
    with pytest.raises(ValueError):m.validate_result(r,**kw)


def test_durable_exclusive_json_and_duplicates(tmp_path):
    p=tmp_path/'one.json';m.write(p,dict(ok=True));assert m.read(p)==dict(ok=True)
    with pytest.raises(FileExistsError):m.write(p,dict(ok=False))
    duplicate=tmp_path/'dup.json';duplicate.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError):m.read(duplicate)
    with pytest.raises(ValueError):m.read(p,1)


def test_refuse_incomplete_export(tmp_path):
    m.write(tmp_path/'report.json',dict(schema=m.NAME+'.report',status='incomplete_or_failed'))
    m.write(tmp_path/'execution.json',{})
    with pytest.raises(ValueError):m.selected_actions(tmp_path)


def test_saved_prepare_verify_without_cuda(tmp_path,monkeypatch):
    # Real inherited source/input seals, no report/outcome or model execution.
    output=tmp_path/(m.NAME+'_prepared')
    reference=m.PROJECT/'results/p2_composition_paired_20261027_v2_prepared01'
    r=m.prepare(reference,output);_,p,s=m.load_prepared(output,r['prepared_sha256'])
    assert p['contract']['budget_points'][1]['standalone_predict']==[168,168]
    assert s['new_neural_calls']==0 and not (output/'started.json').exists()
    with pytest.raises(ValueError):m.load_prepared(output,'a'*64)
    with pytest.raises(ValueError):m.prepare(reference,output)
    p['context']['cases'][0]['root_sha256']='a'*64
    # Mutate only the temporary new fixture preparation, never any source.
    (output/'protocol.json').write_text(json.dumps(p))
    with pytest.raises(ValueError):m.load_prepared(output,r['prepared_sha256'])


def test_complete_eight_arm_exporter_saved_mock_pipeline(tmp_path,monkeypatch):
    """Complete fake artifact chain, not a neural/efficacy result."""
    old,template,_,_=legacy_inputs()
    monkeypatch.setattr(m,'_normalized_token',lambda a:np.asarray(a).reshape(10).tolist())
    compiled=dict(goal_vectors=[np.zeros(192).tolist()],scoring=dict(metric=dict(projection=np.eye(192).tolist())))
    cases=[dict(root_sha256=sha,root_index=i,root=deepcopy(template['root']),action_domain=template['action_domain'],metric_sha256='a'*64) for i,sha in enumerate(m.ROOTS)]
    source=tmp_path/'compilation.json';m.write(source,dict(rows=[dict(root_sha256=sha,compiled=compiled) for sha in m.ROOTS]))
    p=dict(context=dict(compilation=str(source),cases=cases));s=dict(source_seals=[dict(manifest_sha256='b'*64)])
    prepared=tmp_path/'mock_prepared';binding=dict(prepared=str(prepared),prepared_sha256=m.digest(s),protocol_sha256=m.digest(p))
    monkeypatch.setattr(m,'load_prepared',lambda *a:(prepared,p,s))
    output=tmp_path/'mock_run';output.mkdir();m.write(output/'execution.json',binding)
    receipts=[];operation_totals=[]
    jac=np.zeros((192,20));jac[:20]=np.diag(np.linspace(2,1,20))
    for ready in cases:
        bank0=old.sra.spectral_bank(jac,np.zeros(192),compiled['goal_vectors'][0],np.eye(192));bank=m.candidate_bank(bank0,ready['root_sha256'])
        results=[]
        for arm in m.ARMS:
            _,args,_,_=legacy_inputs();args.update(bank=bank,root=ready['root'],provenance=m.provenance(p,s,ready,bank,arm))
            def suffix(c,u,i):
                value=old.suffix_solution(u,np.zeros((192,10)),np.zeros(192),compiled['goal_vectors'][0],np.eye(192))
                value['input_context_sha256']=old.digest(c);return value
            args['suffix']=suffix;r=m.run_arm(arm=arm,**args);assert r['complete'];r['fixture_only']=False;results.append(r)
        entries=[dict(kind=e['kind'],scope=scope,index=i,samples=1,context_tokens=3,completed=True)
            for i,(e,scope) in enumerate(zip(m.expected_entries(80),m.expected_scopes()))]
        operations,_=operation_fixture()
        for i,e in enumerate(results[1]['attempts']):
            local=e['suffix_compilation'];operations[6+4*i+2]['svd_calls_requested']=local['svd_calls']
            operations[6+4*i+3].update(svd_calls=local['svd_calls'],bounded_solve_calls=local['bounded_solve_calls'])
        index=ready['root_index'];bankpath=output/f'root_{index}_bank.json';m.write(bankpath,dict(base=bank0,expanded=bank))
        journal=output/f'root_{index}_journal.jsonl'
        with journal.open('x') as stream:
            for entry in entries:
                stream.write(json.dumps(dict(kind='entry_attempt',entry=dict(entry,completed=False)))+'\n')
                stream.write(json.dumps(dict(kind='entry_terminal',entry=entry))+'\n')
            for op in operations:stream.write(json.dumps(op)+'\n')
            for r in results:
                for e in r['attempts']:stream.write(json.dumps(e)+'\n')
        counts=m.validate_operations(operations,results);operation_totals.append(counts)
        row=dict(root_sha256=ready['root_sha256'],root_index=index,results=results,actual_model_entries=entries,
            operations=operations,completed_operations=counts,error=None,complete=True,
            resource_receipt=old.resource_receipt(entries,results,operations),journal_sha256=m.file_sha(journal),bank_sha256=m.file_sha(bankpath))
        m.write(output/f'root_{index}.json',row);receipts.append(dict(root_sha256=ready['root_sha256'],root_index=index,row_sha256=m.digest(row),complete=True))
    report=dict(schema=m.NAME+'.report',status='cuda_completed',model_unchanged=True,frozen_eval=True,P1_admission=False,P2_admission=False,total_resource_matched=False,
        rows=receipts,actual_unique_predict_calls=1328,actual_unique_action_encoder_calls=1328,completed_selection_evaluations=480,
        completed_operations={k:sum(c[k] for c in operation_totals) for k in operation_totals[0]},**binding)
    m.write(output/'report.json',report);out=m.selected_actions(output)
    assert len(out['plans'])==8 and {p['root_sha256'] for p in out['plans']}==set(m.ROOTS)
    assert all(len(p['raw_commands'])==10 and p['horizon']==2 for p in out['plans'])
    assert not out['P2_admission'] and not out['independent_neural_replay_performed_by_exporter']
    with journal.open('a') as stream:stream.write('{}\n')
    with pytest.raises(ValueError):m.selected_actions(output)
