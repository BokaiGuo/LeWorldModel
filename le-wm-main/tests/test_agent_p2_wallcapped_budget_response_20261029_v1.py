"""RAM-only actual-schema records; not model or physical evidence."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_wallcapped_budget_response_20261029_v1 as m


def context(z=0.):
    return dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
                latent_history=np.full((3,192),z).tolist(),action_history=np.zeros((2,10)).tolist())


def fixture():
    slots = [dict(root_slot=k,condition='off',task_id=f'task{k}',task_sha256=str(k)*64,warm_raw10=np.zeros((10,2)).tolist(),
                  packet=dict(root_context=context(),goal=np.zeros(192).tolist())) for k in (0,1)]
    p = dict(schema=m.RUNTIME+'.prepare',root_arm_order=[list(v) for v in m.ORDER],slots=slots,expected_model_hash='f'*64,
             resource_caps_per_root_arm=deepcopy(m.CAPS),wall_cap_seconds_per_root_arm=450.)
    rows=[]; commands=[]; accounts=[]; plans=[]; payloads=[]
    for i,(root,arm) in enumerate(m.ORDER):
        start=1000.*(i+1); slot=slots[root]; elapsed=455. if i==2 else 400.
        finished=['nominal','bank','seed:0']; events=[dict(kind='unit_start',unit='nominal',elapsed=0.)]
        cs=[]; inc=None; own=[]
        payloads.append(dict(kind='arm_event',index=i,payload=dict(kind='ledger_init',now=start,root=root,arm=arm)))
        # Improve; exact tie keeps earlier; fourth late improvement cannot win.
        for tag,z,t in [('nominal',2.,1.),('seed:0',1.+i/8,20.),('seed:1',1.+i/8,30.)]+([('H1:0',.25,451.)] if i==2 else []):
            w=dict(logits=np.zeros(30).tolist(),contexts=[context(z)]*3)
            q=float(z*z); accepted=t<=450 and (inc is None or q<inc['cost'])
            c=dict(tag=tag,cost=q,accepted=accepted,late=t>450,elapsed=t);cs.append(c)
            if accepted:inc=dict(tag=tag,cost=q,witness=deepcopy(w),raw_float32_bits=np.zeros(30,np.float32).tobytes().hex(),elapsed=t)
            e=dict(kind='ledger_intent',method='commit',args=[w,slot['packet']['goal'],tag],kwargs={},now=start+t)
            ret=dict(kind='ledger_return',method='commit',result_sha256=m.digest(accepted))
            own.append(dict(command=e,result=ret,ledger_start=start))
            for v in (e,ret):payloads.append(dict(kind='arm_event',index=i,payload=v))
        # Fixed missing export for one arm; all rows retained.
        export=deepcopy(inc['witness']) if i!=1 else None
        count=dict.fromkeys(m.FIELDS,0);count.update(predict=9+i,action_encoder=9+i,predictor=9+i,pred_proj=9+i,
            jacobian=1,mathematical_VJP=192,jacobian_elements=5760,AD_rollout_steps=3,root_SVD=2,root_ridge=1)
        completed=deepcopy(count)
        if i==1:completed['predict']-=1
        account=dict(schema=m.ACCOUNT,root=root,arm=arm,binding=deepcopy(slot),elapsed_seconds=elapsed,wall_cap_seconds=450.,
            wall_overrun=elapsed>450,attempted=count,completed=completed,caps=deepcopy(m.CAPS),stop='wall_cap' if i==2 else 'natural_completion',
            errors=[dict(operation=9,resource='predict',elapsed=10.)] if i==1 else [],units_completed=len(finished),finished_units=finished,
            active_unit=None,pending=[],incumbent=inc,commits=cs,export_complete=export is not None,export_missing=export is None,events=events,
            sealed=True,peak_allocated_bytes=None if i==1 else 1024,peak_reserved_bytes=None if i==1 else 2048,
            accounting_only=True,model_witness_authenticated=False,total_FLOPs=None,engine_internal_backward_calls=None,equal_FLOPs=False,P2_admission=False)
        row=dict(index=i,root_slot=root,arm=arm,accounting=account,result=dict(export=export,aborted_unit='replay' if i==1 else None))
        summary={k:deepcopy(v) for k,v in account.items() if k!='events'};summary['ledger_events_sha256']=m.digest(events)
        plan=None
        if export is not None:
            plan=dict(schema=m.RUNTIME+'.plan',root_slot=root,arm=arm,task_id=slot['task_id'],task_sha256=slot['task_sha256'],
                root_context=slot['packet']['root_context'],goal=slot['packet']['goal'],warm_raw10=slot['warm_raw10'],
                commands=np.zeros((15,2),np.float32).tolist(),raw_float32_bits=inc['raw_float32_bits'],contexts=export['contexts'],
                model_hash=p['expected_model_hash'],model_Q=inc['cost'],report_sha256='fixture',P1_admission=False,P2_admission=False,selection_uses_physics=False)
            plan['selection_digest']=m.digest(plan)
        rows.append(row);commands.append(own);accounts.append(summary);plans.append(plan)
        payloads.append(dict(kind='arm_sealed',index=i,row_sha256=m.digest(row)))
    report=dict(schema=m.RUNTIME+'.report',status='recorded_not_admitted',errors=[],fixed_denominator=4,
        model_unchanged=True,model_before=p['expected_model_hash'],model_after=p['expected_model_hash'],wall_seconds=1700.,
        arm_stops=[r['accounting']['stop'] for r in rows],
        rows=[dict(index=i,root_slot=root,arm=arm,file=f'arm{i}.json.gz') for i,(root,arm) in enumerate(m.ORDER)])
    receipt=dict(schema=m.RUNTIME+'.consume',status='wallcapped_records_verified_not_admitted',fixed_denominator=4,accounts=accounts,plans=plans)
    return p,report,receipt,rows,commands,payloads


def analyze(f):return m.assemble(*f[:5])


def test_full_four_and_two_pairs_all_arithmetic_and_missing():
    r=analyze(fixture());assert len(r['rows'])==4 and len(r['pairs'])==2
    assert r['export_pairs_available']==1 and r['export_pairs_fixed_denominator']==2
    a=r['rows'][2];assert a['wall_overrun'] and a['overrun_seconds']==5
    assert len(a['commits'])==4 and a['commits'][2]['accepted'] is False and a['commits'][3]['late']
    assert a['Q_capped_incumbent']==1.5625 and a['Q_export']==1.5625 and a['model_Q_gain']==2.4375
    assert r['rows'][1]['export_missing'] and r['rows'][1]['Q_export'] is None
    assert r['rows'][1]['peak_allocated_bytes'] is None
    assert r['rows'][1]['resources']['predict']['failed_or_unreturned_requested']==1
    assert a['resources']['encoder']['utilization'] is None
    assert r['pairs'][0]['values']['Q_export']['ordering']=='missing'
    assert r['pairs'][0]['values']['Q_capped_incumbent']['delta']==1.265625-1.
    assert r['pairs'][1]['values']['Q_capped_incumbent']['delta']==1.5625-1.890625
    assert r['cap_deviation_roots']==1 and r['paired_ordering_counts']['Q_capped_incumbent']==dict(better=1,tie=0,worse=1,missing=0)
    assert r['rows'][2]['commits'][-1]['gain_from_nominal']==3.9375
    assert r['total_FLOPs'] is None and not r['P2_admission']


def test_objective_is_float64_mean192_not_float32_sum():
    w=dict(logits=[0.]*30,contexts=[context() for _ in range(3)])
    w['contexts'][-1]['latent_history'][-1]=np.linspace(-.123,1.23,192,dtype=np.float32).tolist()
    g=np.full(192,.125).tolist(); expected=float(np.mean((np.array(w['contexts'][-1]['latent_history'][-1],np.float64)-g)**2))
    assert m.objective(w,g)==expected
    assert expected!=float(np.mean((np.array(w['contexts'][-1]['latent_history'][-1],np.float32)-np.float32(.125))**2))


@pytest.mark.parametrize('mutation',['Q','accepted','late','clock','goal','incumbent','export','planQ','raw','caps','counts','overrun','row','missing','digest','receipt'])
def test_resealed_semantic_mutations_reject(mutation):
    f=fixture();p,r,s,rows,cmd,_=f;a=rows[0]['accounting']
    if mutation=='Q':a['commits'][0]['cost']+=1
    elif mutation=='accepted':a['commits'][2]['accepted']=True
    elif mutation=='late':a['commits'][0]['late']=True
    elif mutation=='clock':cmd[0][0]['command']['now']+=1
    elif mutation=='goal':cmd[0][0]['command']['args'][1]=[1.]*192
    elif mutation=='incumbent':a['incumbent']['cost']+=1
    elif mutation=='export':rows[0]['result']['export']['contexts'][-1]['latent_history'][-1][0]+=1
    elif mutation=='planQ':s['plans'][0]['model_Q']+=1
    elif mutation=='raw':s['plans'][0]['commands'][0][0]=.1
    elif mutation=='caps':a['caps']['predict']+=1
    elif mutation=='counts':a['completed']['predict']+=1
    elif mutation=='overrun':a['wall_overrun']=True
    elif mutation=='row':rows.reverse()
    elif mutation=='missing':s['plans'][1]=deepcopy(s['plans'][0])
    elif mutation=='digest':s['plans'][0]['selection_digest']='a'*64
    else:s['status']='failed'
    # Reseal summary to test independent arithmetic rather than only copy comparison.
    if mutation!='row':
        s['accounts'][0]={k:deepcopy(v) for k,v in a.items() if k!='events'}
        s['accounts'][0]['ledger_events_sha256']=m.digest(a['events'])
    with pytest.raises(ValueError):analyze(f)


def test_all_missing_incumbents_and_exports_do_not_become_zero():
    f=fixture();p,r,s,rows,commands,payloads=f
    for i,row in enumerate(rows):
        a=row['accounting'];a.update(incumbent=None,commits=[],export_complete=False,export_missing=True)
        row['result']['export']=None;s['plans'][i]=None;commands[i]=[]
        s['accounts'][i]={k:deepcopy(v) for k,v in a.items() if k!='events'};s['accounts'][i]['ledger_events_sha256']=m.digest(a['events'])
    result=analyze(f)
    assert all(x['model_Q_gain'] is None for x in result['rows'])
    assert result['paired_ordering_counts']['Q_capped_incumbent']['missing']==2
    assert result['export_pairs_available']==0


def chained(path,payloads):
    prev='0'*64;lines=[]
    for i,p in enumerate(payloads):
        e=dict(index=i,previous_sha256=prev,payload=p);e['event_sha256']=m.digest(e);prev=e['event_sha256'];lines.append(m.raw(e))
    path.write_bytes(gzip.compress(b''.join(lines)))


def test_real_writer_reader_hashchain_and_commit_extraction(tmp_path):
    f=fixture();chained(tmp_path/'events.jsonl.gz',f[5])
    commits,sealed,diagnostics=m.journal_commits(tmp_path)
    assert commits==f[4] and sealed==[m.digest(r) for r in f[3]]
    assert diagnostics==[dict(arm_exceptions=[],ledger_failures=[]) for _ in range(4)]
    p=deepcopy(f[5]);p.append(dict(kind='unknown'))
    chained(tmp_path/'events.jsonl.gz',p)
    with pytest.raises(ValueError):m.journal_commits(tmp_path)


@pytest.mark.parametrize('change',['chain','unclosed','order','rowhash','terminal','strictsha','reportsha','binding','status'])
def test_disk_consume_authorities_and_mutations(tmp_path,change):
    out,strict,rsha,ssha=on_disk(tmp_path)
    if change=='strictsha':ssha='a'*64
    elif change=='reportsha':rsha='a'*64
    elif change in ('binding','status'):
        s=json.loads(strict.read_bytes())
        if change=='binding':s['bindings'][str(out/'report.json')]='b'*64
        else:s['status']='failed'
        strict.write_bytes(m.raw(s));ssha=m.sha(strict)
    else:
        f=fixture();ps=f[5]
        if change=='unclosed':ps.pop()
        elif change=='order':ps[0]['index']=2
        elif change=='rowhash':ps[-1]['row_sha256']='b'*64
        elif change=='terminal':ps.append(dict(kind='extra'))
        chained(out/'events.jsonl.gz',ps)
        if change=='chain':
            plain=gzip.decompress((out/'events.jsonl.gz').read_bytes()).replace(b'ledger_init',b'ledger_inix',1)
            (out/'events.jsonl.gz').write_bytes(gzip.compress(plain))
        # Bind changed bytes through report+receipt: semantic validation must still reject.
        r=json.loads((out/'report.json').read_bytes());r['artifact_sha256']['events.jsonl.gz']=m.sha(out/'events.jsonl.gz')
        (out/'report.json').write_bytes(m.raw(r));rsha=m.sha(out/'report.json')
        s=json.loads(strict.read_bytes());s['report_sha256']=rsha
        for plan in s['plans']:
            if plan is not None:
                plan['report_sha256']=rsha
                plan['selection_digest']=m.digest({k:v for k,v in plan.items() if k!='selection_digest'})
        s['bindings'][str(out/'report.json')]=rsha;s['bindings'][str(out/'events.jsonl.gz')]=m.sha(out/'events.jsonl.gz')
        strict.write_bytes(m.raw(s));ssha=m.sha(strict)
    with pytest.raises(ValueError):m.consume(out,rsha,strict,ssha)


def on_disk(tmp_path):
    p,r,s,rows,commands,payloads=fixture();out=tmp_path/'output';out.mkdir()
    prepared=tmp_path/'prepare.json';prepared.write_bytes(m.raw(p));r.update(prepared=str(prepared),prepare_sha256=m.sha(prepared))
    for i,row in enumerate(rows):(out/f'arm{i}.json.gz').write_bytes(gzip.compress(m.raw(row)))
    chained(out/'events.jsonl.gz',payloads)
    r['artifact_sha256']={f.name:m.sha(f) for f in out.iterdir()}
    (out/'report.json').write_bytes(m.raw(r));rh=m.sha(out/'report.json');s['report_sha256']=rh
    for plan in s['plans']:
        if plan is not None:
            plan['report_sha256']=rh
            plan['selection_digest']=m.digest({k:v for k,v in plan.items() if k!='selection_digest'})
    runtime=m.ROOT/'experiments/spectral_reachability'/f'{m.RUNTIME}.py'
    s['bindings']={str(f):m.sha(f) for f in [prepared,runtime,*out.iterdir()]}
    strict=tmp_path/'strict.json';strict.write_bytes(m.raw(s))
    return out,strict,rh,m.sha(strict)


def test_full_disk_consume_analyze_verify_no_driver_import(tmp_path,monkeypatch,capsys):
    out,strict,rh,sh=on_disk(tmp_path);r=m.consume(out,rh,strict,sh)
    assert len(r['rows'])==4 and r['authority']['strict_sha256']==sh
    derivative=tmp_path/(m.NAME+'_RAM.json');info=m.write_new(derivative,r)
    monkeypatch.setattr(sys,'argv',['saved','verify','--output',str(out),'--report-sha',rh,'--strict',str(strict),
                                  '--strict-sha',sh,'--analysis',str(derivative),'--analysis-sha',info['sha256']])
    m.main();assert json.loads(capsys.readouterr().out)['verified'] is True
    with pytest.raises(ValueError):m.write_new(derivative,r)
    source=Path(m.__file__).read_text()
    assert 'import torch' not in source and 'import agent_p2_wallcapped_cuda' not in source


def test_classification_exact_ties_no_epsilon():
    assert [m.classify(v) for v in (None,0.,-1e-300,1e-300)]==['missing','tie','better','worse']


def test_all_four_exports_complete_and_public_exact_ties():
    f=fixture();p,r,s,rows,cmd,_=f
    for i,row in enumerate(rows):
        a=row['accounting'];a['export_complete']=True;a['export_missing']=False
        row['result']['export']=deepcopy(a['incumbent']['witness'])
    s['plans'][1]=deepcopy(s['plans'][0]);plan=s['plans'][1];a=rows[1]['accounting']
    plan.update(arm='composition',contexts=deepcopy(a['incumbent']['witness']['contexts']),model_Q=a['incumbent']['cost'])
    plan['selection_digest']=m.digest({k:v for k,v in plan.items() if k!='selection_digest'})
    for i,row in enumerate(rows):
        a=row['accounting'];s['accounts'][i]={k:deepcopy(v) for k,v in a.items() if k!='events'}
        s['accounts'][i]['ledger_events_sha256']=m.digest(a['events'])
    x=analyze(f)
    assert x['export_pairs_available']==2 and all(not a['export_missing'] for a in x['rows'])
    assert x['paired_ordering_counts']['Q_export']==dict(better=1,tie=0,worse=1,missing=0)
    # Exact paired tie is separate; not hidden in better/worse.
    panels=deepcopy(x['rows']);panels[1]['Q_export']=panels[0]['Q_export']
    assert m.paired(panels)[0]['values']['Q_export']['ordering']=='tie'


def test_failed_commit_and_arm_exception_retained(tmp_path):
    f=fixture();p,r,s,rows,cmd,payloads=f
    e=deepcopy(cmd[0][-1]['command']);e['now']=1050.
    failure=dict(kind='ledger_failure',method='commit',error='RAM error')
    cmd[0].append(dict(command=e,result=failure,ledger_start=1000.))
    end=next(i for i,p in enumerate(payloads) if p['kind']=='arm_sealed')
    extra=[dict(kind='arm_event',index=0,payload=q) for q in
           (e,failure,dict(kind='arm_exception',type='ValueError',message='RAM error'))]
    payloads[end:end]=extra
    chained(tmp_path/'events.jsonl.gz',payloads)
    cs,_,diagnostics=m.journal_commits(tmp_path)
    x=m.assemble(p,r,s,rows,cs,diagnostics)
    assert x['rows'][0]['commit_failures']==[dict(tag='seed:1',Q=1.,error='RAM error',host_now=1050.)]
    assert x['rows'][0]['journal_failures']['arm_exceptions'][0]['message']=='RAM error'
    assert len(x['rows'][0]['commits'])==3


def test_codec_newline_is_not_default_json():
    assert m.raw([1,2])==b'[1,2]\n'
    assert m.digest([1,2])!=__import__('hashlib').sha256(json.dumps([1,2]).encode()).hexdigest()
