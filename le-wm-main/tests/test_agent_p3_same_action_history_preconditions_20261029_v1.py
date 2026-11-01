"""Nonneural fixture and temporary saved ABI tests, no upstream strict invocation."""
import base64
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_same_action_history_preconditions_20261029_v1 as m
import test_agent_p3_same_action_error_decomposition_20261029_v1 as old_fixture


def blob(a):
    a=np.asarray(a,np.float32)
    return dict(shape=list(a.shape),dtype='<f4',sha256=m.sha(a.tobytes()),
                data=base64.b64encode(a.tobytes()).decode())


def make_stream(task,request,winner,scope):
    state,history=m.context(request['task']['root_context']);selected=[];ops=[]
    commands=np.asarray(winner['actions'],np.float32)
    for step in range(3):
        raw=commands[step*5:step*5+5];mean=np.asarray(task['action_mean'],np.float32)
        std=np.asarray(task['action_std'],np.float32);token=((raw-mean)/std).reshape(10)
        out=np.concatenate((history,token[None]),axis=0)[None]
        encoded=np.full((1,3,192),step*.125,np.float32)
        full=np.full((1,3,192),.125,np.float32)
        full[0,-1]=winner['whole'][step]['latent_history'][-1]
        hidden=full+.125
        newstate=np.concatenate((state[1:],full[0,-1:]),axis=0)
        groups=[dict(latent=state[None],history=history[None],raw=raw,mean=mean,std=std,token=token,outgoing=out),
            {'action_encoder:0':out},{'action_encoder:output':encoded},
            {'predict:latent':state[None],'predict:act_emb':encoded},
            {'predictor:0':state[None],'predictor:1':encoded},{'predictor:output':hidden},
            {'pred_proj:0':hidden.reshape(3,192)},{'pred_proj:output':full.reshape(3,192)},
            {'predict:output':full},dict(latent=newstate[None],history=out[:,1:])]
        ids=[None,step*4,step*4,step*4+1,step*4+2,step*4+2,step*4+3,step*4+3,step*4+1,None]
        for n,((kind,keys),group,opid) in enumerate(zip(m.SEQUENCE,groups,ids,strict=True)):
            selected.append(dict(record_index=len(selected),payload=dict(kind=kind,scope=deepcopy(scope),
                phase='whole',step=step,operation_index=opid,arrays={k:blob(group[k]) for k in keys})))
        for resource,offset,shape in [('action_encoder',0,[1,3,10]),('predict',1,[1,3,192]),
                                      ('predictor',2,[1,3,192]),('pred_proj',3,[3,192])]:
            ops.append(dict(index=step*4+offset,resource=resource,scope=deepcopy(scope),
                            status='returned',error=None,shape=shape))
        state=newstate;history=out[0,1:]
    return selected,ops


@lru_cache(maxsize=1)
def base():
    p,r,receipt,arrays=old_fixture.fixture();pp=dict(schema=m.PLANNER+'.prepare',
        expected_model_hash=p['expected_model_hash'],tasks=[])
    pr=dict(schema=m.PLANNER+'.report',nodes=[],per_episode_neural=[])
    pc=dict(schema=m.PLANNER+'.consume',complete=True,status='saved_shared_panel_verified_NOT_admitted',
            P1_admission=False,P3_admission=False,C3=False)
    streams={};receipt['actual_context_chains']=[]
    for b,plan in enumerate(p['plans']):
        w=plan['source_witness'];slot=b//2;bank=('endpoints','midpoints')[b%2];task=w['task']
        task.update(action_mean=[0.,0.],action_std=[1.,1.])
        raw10=np.asarray(w['request']['raw10'],np.float32);commands=np.asarray(plan['commands'],np.float32)
        c=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
               latent_history=[old_fixture.v(.25) for _ in range(3)],action_history=m.normalization(task,raw10).tolist())
        request=dict(task=dict(task,root_context=deepcopy(c)),raw10=raw10.tolist())
        w['request']=request;predctx=deepcopy(c);items=[];actual=deepcopy(c)
        for j,cp in enumerate((5,10,15,20)):
            spec=p['rows'][4*b+j];hist=np.concatenate((raw10,commands[5:cp]))[-10:]
            spec['normalized_action_history']=m.normalization(task,hist).tolist()
            r['rows'][4*b+j]['normalized_action_history']=spec['normalized_action_history']
            if j:
                predctx=dict(predctx,latent_history=predctx['latent_history'][1:]+[spec['predicted192']],
                             action_history=spec['normalized_action_history'])
                w['winner']['whole'][j-1]=deepcopy(predctx)
                actual=dict(actual,latent_history=actual['latent_history'][1:]+[arrays[4*b+j]],
                            action_history=spec['normalized_action_history'])
            items.append(dict(step=cp,context=deepcopy(actual),context_missing=False))
        receipt['actual_context_chains'].append(dict(plan_index=b,root_slot=slot,bank=bank,checkpoints=items))
        scope=dict(slot=slot,mode='adaptive_'+bank,horizon=3,index=0,phase='selected')
        w['winner']['index']=0
        stream,ops=make_stream(task,request,w['winner'],scope)
        result=dict(status='sealed',error=None,selected_horizon=3,request_sha256=m.digest(request),
                    resource_receipt=dict(events=ops),panels=[dict(horizon=3,winner=deepcopy(w['winner']),
                        selected_path=deepcopy(w['winner']['whole']))])
        w['result']=result
        node=dict(status='sealed',error=None,source_slot=slot,mode='adaptive_'+bank,boundary=5,
                  episode_id=f's{slot}_adaptive_{bank}',request=deepcopy(request),receipt=deepcopy(result),
                  resource_receipt=deepcopy(result['resource_receipt']),native_journal_sha256='e'*64,
                  native_journal_tail='f'*64)
        pr['nodes'].append(node);streams[node['episode_id']]=stream
        if b%2==0:pp['tasks'].append(deepcopy(task))
    return p,r,receipt,pp,pr,pc,arrays,streams


def fixture():return deepcopy(base())


def run_fixture(f):
    p,r,rc,pp,pr,pc,arrays,streams=f
    return m.resolve(p,r,rc,pp,pr,pc,lambda d:arrays[d['id']],
        lambda n,s:(streams[n['episode_id']],dict(path='RAM',sha256=n['native_journal_sha256'])))


def test_all8_literal_contexts_and_budget():
    v=run_fixture(fixture())
    assert v['fixed_denominator']==v['ready']==8 and v['blocked']==0
    assert [(j['start'],j['end']) for j in v['jobs']]==[(10,15),(15,20)]*4
    assert all(j['R_reuse_verified'] and j['new_R_calls']==0 for j in v['jobs'])
    assert v['IO_budget']['projected_bounded_output_bytes']<64*1024**2
    assert all(j['R_proof']['native_tensor_bytes']==25840 for j in v['jobs'])


@pytest.mark.parametrize('mutation',['completed','order','unknown','actemb','raw','token','latent',
    'output','operation','duplicate_op','phase','blobsha','record_step','winner','record_index'])
def test_R_mutations_block_only_source_branch(mutation):
    f=fixture();node=f[4]['nodes'][0];stream=f[7][node['episode_id']]
    if mutation=='completed':node['resource_receipt']['events'][0]['status']='failed'
    elif mutation=='order':stream[11],stream[12]=stream[12],stream[11]
    elif mutation=='unknown':stream.append(deepcopy(stream[-1]))
    elif mutation=='actemb':stream[13]['payload']['arrays']['predict:act_emb']=blob(np.ones((1,3,192),np.float32))
    elif mutation=='raw':stream[10]['payload']['arrays']['raw']=blob(np.zeros((5,2),np.float32))
    elif mutation=='token':stream[10]['payload']['arrays']['token']=blob(np.zeros(10,np.float32))
    elif mutation=='latent':stream[13]['payload']['arrays']['predict:latent']=blob(np.zeros((1,3,192),np.float32))
    elif mutation=='output':stream[18]['payload']['arrays']['predict:output']=blob(np.zeros((1,3,192),np.float32))
    elif mutation=='operation':stream[18]['payload']['operation_index']=1000
    elif mutation=='duplicate_op':stream[11]['payload']['operation_index']=stream[12]['payload']['operation_index']=0
    elif mutation=='phase':stream[10]['payload']['phase']='segmented'
    elif mutation=='blobsha':stream[10]['payload']['arrays']['raw']['sha256']='0'*64
    elif mutation=='record_step':stream[10]['payload']['step']=True
    elif mutation=='winner':node['receipt']['panels'][0]['winner']['index']=1
    elif mutation=='record_index':stream[10]['record_index']=stream[9]['record_index']
    v=run_fixture(f)
    assert v['blocked']==2 and v['ready']==6
    assert all(not j['R_reuse_verified'] for j in v['jobs'][:2])
    assert all(j['new_R_calls']==0 for j in v['jobs'])


def test_no_journal_or_node_keeps_denominator():
    f=fixture();f[4]['nodes'].pop(0)
    v=run_fixture(f)
    assert v['blocked']==2 and len(v['jobs'])==8


@pytest.mark.parametrize('mutation',['schema','task','A_context','A_missing_flag','normalizer','bool'])
def test_bad_A_or_authority_rejected(mutation):
    f=fixture()
    if mutation=='schema':f[5]['schema']='old.consume'
    elif mutation=='task':f[3]['tasks'][0]['goal']=old_fixture.v(9)
    elif mutation=='A_context':f[2]['actual_context_chains'][0]['checkpoints'][1]['context']['latent_history'][0][0]=9.
    elif mutation=='A_missing_flag':f[2]['actual_context_chains'][0]['checkpoints'][1]['context_missing']=True
    elif mutation=='normalizer':f[0]['plans'][0]['source_witness']['task']['action_std']=[0.,1.]
    elif mutation=='bool':f[0]['plans'][0]['source_witness']['request']['task']['root_context']['latent_history'][0][0]=True
    with pytest.raises(ValueError):run_fixture(f)


def test_missing_actual_fullhistory_blocks_not_padded():
    f=fixture();p,r,rc=f[:3];i=2
    p['rows'][i]['source']=r['rows'][i]['source']=rc['rows'][i]['source']=None
    r['rows'][i]['latent']=None;r['rows'][i]['status']='missing_physical_checkpoint'
    rc['rows'][i].update(actual192=None,latent_missing=True,actual_encoded_Q=None,
        model_minus_actual_Q=None,absolute_Q_error=None,prediction_vs_encoded_MSE=None)
    for j in (2,3):rc['actual_context_chains'][0]['checkpoints'][j].update(context=None,context_missing=True)
    v=run_fixture(f)
    assert v['blocked']==2 and len(v['jobs'])==8
    assert 'actual_endpoint_label_missing' in v['jobs'][0]['blocked_reasons']
    assert 'A_full_actual_context_missing' in v['jobs'][1]['blocked_reasons']


def write_journal(path,stream):
    tail='0'*64;lines=[]
    for i,rec in enumerate(stream):
        body=dict(index=i,previous=tail,payload=rec['payload']);tail=m.digest(body)
        lines.append(m.canonical(dict(body,sha256=tail))+b'\n')
    data=b''.join(lines);path.write_bytes(data)
    return m.sha(data),tail


def test_real_native_journal_chain_and_fullIO(tmp_path):
    f=fixture();node=f[4]['nodes'][0];stream=f[7][node['episode_id']]
    path=tmp_path/'events.jsonl';h,tail=write_journal(path,stream)
    got,ref=m.read_selected_journal(path,h,tail,stream[0]['payload']['scope'])
    assert got==stream and ref['records']==30
    cert=m.certify_R(node,f[0]['plans'][0]['source_witness'],got)
    assert len(cert)==3 and all(x['native_tensor_bytes']==25840 for x in cert)


@pytest.mark.parametrize('bad',['doubleLF','tail','sha','unknown','missingLF','duplicatekey'])
def test_journal_mutation_rejected(tmp_path,bad):
    f=fixture();stream=next(iter(f[7].values()));path=tmp_path/'events.jsonl'
    h,tail=write_journal(path,stream)
    if bad=='doubleLF':path.write_bytes(path.read_bytes().replace(b'\n',b'\n\n'))
    elif bad=='tail':tail='a'*64
    elif bad=='sha':h='a'*64
    elif bad=='unknown':path.write_bytes(path.read_bytes()+b'{}\n')
    elif bad=='missingLF':path.write_bytes(path.read_bytes()[:-1])
    elif bad=='duplicatekey':path.write_bytes(b'{"index":0,"index":0}\n')
    with pytest.raises((ValueError,KeyError)):
        m.read_selected_journal(path,h,tail,stream[0]['payload']['scope'])


def test_exact_zero_and_negative_zero_preserved():
    a=np.zeros((3,192),np.float32);b=a.copy();b[0,0]=-0.
    with pytest.raises(ValueError):m.same(a,b,'negative zero differs')


@pytest.mark.parametrize('bad',[True,float('inf'),float('nan'),'.1'])
def test_strict_numeric(bad):
    with pytest.raises(ValueError):m.f32([bad,0],(2,))


def save_inputs(tmp_path):
    f=fixture();p,r,rc,pp,pr,pc,arrays,streams=f
    folder=tmp_path/'planner';folder.mkdir()
    for n in pr['nodes']:
        sub=folder/(n['episode_id']+'_b05');sub.mkdir()
        h,tail=write_journal(sub/'events.jsonl',streams[n['episode_id']])
        n.update(native_journal_sha256=h,native_journal_tail=tail)
    # Encode the real native gzNPY descriptor ABI in temporary fixtures.
    import gzip
    from io import BytesIO
    enc=tmp_path/'encoder';enc.mkdir()
    for row in r['rows']:
        i=row['index'];a=np.asarray(arrays[i],np.float32);buf=BytesIO();np.save(buf,a,allow_pickle=False)
        data=gzip.compress(buf.getvalue(),mtime=0);name=f'{i}.npy.gz';(enc/name).write_bytes(data)
        row['latent']=dict(file=name,sha256=m.sha(data),array_sha256=m.sha(a.tobytes()),shape=[192],dtype='float32')
        r['artifact_sha256'][name]=m.sha(data)
    args={}
    def persist(key,obj,path):
        path.write_bytes(m.canonical(obj));args[key]=str(path);args[key+'_sha']=m.sha(path.read_bytes())
    persist('planner_prepared',pp,folder/'prepare.json')
    pr.update(prepared=args['planner_prepared'],prepare_sha256=args['planner_prepared_sha'])
    persist('planner_report',pr,folder/'report.json')
    pc.update(prepared=args['planner_prepared'],prepare_sha256=args['planner_prepared_sha'],report_sha256=args['planner_report_sha'])
    persist('planner_receipt',pc,folder/'receipt.json')
    p['bindings']={args[k]:args[k+'_sha'] for k in ('planner_prepared','planner_report','planner_receipt')}
    persist('encoder_prepared',p,enc/'prepare.json')
    r.update(prepared=args['encoder_prepared'],prepare_sha256=args['encoder_prepared_sha'])
    persist('encoder_report',r,enc/'report.json')
    rc['report_sha256']=args['encoder_report_sha'];persist('encoder_receipt',rc,enc/'receipt.json')
    return args


def test_complete_saved_prepare_inspect_and_writer(tmp_path):
    args=save_inputs(tmp_path);p=m.build(**args)
    assert p['ready']==8 and p['evidence_scope']['full_upstream_strict_rebuilt_here'] is False
    target=tmp_path/(m.NAME+'_prepare_fixture.json');w=m.write(target,p)
    result=m.inspect(str(target),w['sha256'])
    assert result['ready']==8 and result['production_execution'] is False
    with pytest.raises(FileExistsError):m.write(target,p)


@pytest.mark.parametrize('key',m.KEYS)
def test_all_six_external_SHA_fail_closed(tmp_path,key):
    args=save_inputs(tmp_path);args[key+'_sha']='0'*64
    with pytest.raises(ValueError):m.build(**args)


def test_saved_missing_journal_not_fake_zero_cost(tmp_path):
    args=save_inputs(tmp_path)
    (tmp_path/'planner/s0_adaptive_endpoints_b05/events.jsonl').unlink()
    p=m.build(**args)
    assert p['blocked']==2 and all(not x['R_reuse_verified'] for x in p['jobs'][:2])


def test_own_source_binding_tamper_rejected(tmp_path):
    args=save_inputs(tmp_path);p=m.build(**args)
    p['bindings'][str(Path(m.__file__))]='0'*64
    target=tmp_path/(m.NAME+'_fixture.json');w=m.write(target,p)
    with pytest.raises(ValueError):m.inspect(str(target),w['sha256'])


def test_pinned_source_mismatch_rejected(tmp_path,monkeypatch):
    args=save_inputs(tmp_path)
    monkeypatch.setattr(m,'PINS',{Path(m.saved.__file__):'0'*64})
    with pytest.raises(ValueError):m.build(**args)


def test_cross_authority_join_rejected_with_recomputed_SHA(tmp_path):
    args=save_inputs(tmp_path);path=Path(args['encoder_receipt']);rc=m.saved.decode(path.read_bytes())
    rc['report_sha256']='0'*64;path.write_bytes(m.canonical(rc));args['encoder_receipt_sha']=m.sha(path.read_bytes())
    with pytest.raises(ValueError):m.build(**args)


def test_actual_byte_budgets_base64_and_metadata():
    f=fixture();stream=next(iter(f[7].values()))[:10]
    arrays=[blob for e in stream for blob in e['payload']['arrays'].values()]
    assert len(arrays)==19
    assert sum(len(base64.b64decode(b['data'])) for b in arrays)==25840
    assert sum(len(b['data']) for b in arrays)*8==275712
    assert 275712<=run_fixture(f)['IO_budget']['base64_data_bytes_upper_bound']
