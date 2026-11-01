"""Saved synthetic array contracts only; never import producer, torch or gym."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import p4_feedback_root_cuda_response_saved_audit_20261027_v1 as a


@pytest.fixture
def material():
    """Analytic diagonal MOCK response, not any actual learned-system result."""
    root='0'*64;q=np.eye(192);goal=np.full(192,.04);j=np.zeros((192,10));j[:10]=np.diag(np.arange(10,0,-1))
    normal=dict(mean=[-.0087,.0068],std=[.2019,.2002],source='MOCK',source_sha256='b'*64,training_provenance_verified=False)
    p=dict(root_sha256=root,active_metric_id='MOCK_Q',goal=goal.tolist(),q=q.tolist(),normalization=normal,requests=[])
    rows=[];files={}
    for arm in a.ARMS:
        latent=np.zeros((3,192),np.float32)
        if arm=='actual':latent[-1]=.125
        c=dict(a.META,latent_history=latent.astype(float).tolist(),action_history=np.zeros((2,10)).tolist())
        req=dict(arm=arm,context=c,context_sha256=a.source_digest(c),boundary_step=5,must_recompile=True,can_reuse_initial_jacobian=False)
        p['requests'].append(req)
        def successor(raw):
            raw=np.asarray(raw,np.float32);mean=np.asarray(normal['mean'],np.float32);std=np.asarray(normal['std'],np.float32)
            last=(latent[-1].astype(float)+j@raw.flatten()).astype(np.float32).astype(float)
            return dict(c,latent_history=c['latent_history'][1:]+[last.tolist()],
                action_history=[c['action_history'][-1],((raw-mean)/std).reshape(10).astype(float).tolist()])
        nominal=successor(np.zeros((5,2)));nom=np.asarray(nominal['latent_history'])[-1]
        singular=np.arange(10,0,-1,dtype=float);progress=singular*(goal[:10]-nom[:10])/(singular**2+.002)
        values=[np.zeros(10),progress];kinds=['zero','root_dls_progress']
        for radius in (np.sqrt(5)*.25,np.sqrt(5)*.75):
            for mode in range(2):
                for sign in (-1,1):
                    values.append(sign*radius*np.eye(10)[mode]);kinds.append(f'spectral_mode_{mode}_radius_{radius:.8f}_sign_{sign}')
        bank=dict(params=np.stack(values).tolist(),kinds=kinds,singular_values=singular.tolist(),right_vectors=np.eye(10).tolist())
        files[arm+'.intent.json']=dict(request=req,root_sha256=root,active_metric_id='MOCK_Q')
        files[arm+'.nominal.json']=nominal;files[arm+'.jacobian_forward.json']=deepcopy(nominal)
        files[arm+'.jacobian.json']=j.tolist();files[arm+'.bank.json']=bank;candidates=[]
        for i,(u,kind) in enumerate(zip(values,kinds)):
            commands=np.tanh(u.astype(np.float32)).reshape(5,2).tolist();ctx=successor(commands)
            cost=float(np.mean((np.asarray(ctx['latent_history'])[-1]-goal)**2))
            candidate=dict(index=i,kind=kind,logits=u.tolist(),commands=commands,context=ctx,cost=cost)
            files[arm+f'.candidate_{i}.json']=candidate;files[arm+f'.candidate_{i}.intent.json']=dict(index=i,commands=commands,kind=kind);candidates.append(candidate)
        winner=min(candidates,key=lambda x:(x['cost'],x['index']))
        row=dict(arm=arm,root_sha256=root,boundary_step=5,context_sha256=req['context_sha256'],status='response_completed',context_specific_AD=True,
            independent_replay_exact=True,initial_J_reused=False,actual_physics=False,candidates=candidates,winner=deepcopy(winner))
        files[arm+'.response.json']=row;files[arm+'.replay.json']=deepcopy(winner['context']);rows.append(row)
    return p,rows,files


def test_analytic_mock_bank_history_selection(material):
    p,rows,files=material
    for request,row in zip(p['requests'],rows):
        result=a.arm_check(p,request,row,files)
        assert result['J_shape']==[192,10] and result['bank_and_ridge_recomputed']
        assert result['AD_derivative_numerically_rederived'] is False


@pytest.mark.parametrize('fault',['J_shape','J_nan','J_changed','J_zero','AD_forward','bank_ridge','bank_mode','bank_sign','logits','commands','history','latent_chronology','cost','winner','replay','candidate_identity','root_identity'])
def test_saved_arrays_tamper_rejected(material,fault):
    p,rows,f=material;row=rows[0];pre='imagined.'
    if fault=='J_shape':f[pre+'jacobian.json']=[[0.]*10]
    elif fault=='J_nan':f[pre+'jacobian.json'][0][0]=float('nan')
    elif fault=='J_changed':f[pre+'jacobian.json'][0][0]+=1
    elif fault=='J_zero':f[pre+'jacobian.json']=np.zeros((192,10)).tolist()
    elif fault=='AD_forward':f[pre+'jacobian_forward.json']['latent_history'][-1][0]=99
    elif fault=='bank_ridge':f[pre+'bank.json']['params'][1][0]+=1
    elif fault=='bank_mode':f[pre+'bank.json']['right_vectors'][0][1]=.1
    elif fault=='bank_sign':f[pre+'bank.json']['right_vectors'][0][0]=-1
    elif fault=='logits':row['candidates'][0]['logits'][0]=.1
    elif fault=='commands':row['candidates'][0]['commands'][0][0]=.1
    elif fault=='history':row['candidates'][0]['context']['action_history'][0][0]=99
    elif fault=='latent_chronology':row['candidates'][0]['context']['latent_history'][0][0]=99
    elif fault=='cost':row['candidates'][0]['cost']+=.01
    elif fault=='winner':row['winner']=deepcopy(row['candidates'][9])
    elif fault=='replay':f[pre+'replay.json']['latent_history'][-1][0]=99
    elif fault=='candidate_identity':row['candidates'][1]['index']=0
    elif fault=='root_identity':row['root_sha256']='1'*64
    with pytest.raises(ValueError):a.arm_check(p,p['requests'][0],row,f)


def test_independent_ridge_active_ball():
    k=np.eye(192,10);r=np.full(192,100.);x=a.ridge(k,r,.002,1.)
    assert abs(np.linalg.norm(x)-1)<1e-12 and np.allclose(x,np.full(10,1/np.sqrt(10)))


def receipt_entries():
    events,entries=a.expected_journal()
    counts={arm:{k:dict(attempted=n,completed=n) for k,n in a.COUNTS.items()} for arm in a.ARMS}
    return events,dict(entries=entries,counts=counts)


def test_full_nested_384_each_not_total384():
    events,r=receipt_entries();work=a.work_check(r,events)
    assert work['total_metered_entries']==822 and work['journal_events']==1673
    assert work['grad_API_calls']==work['engine_API_entries']==384
    assert work['jacobian_API_calls']==2 and work['predict_samples']==26


@pytest.mark.parametrize('fault',['drop_event','duplicate_event','swap_nested_return','cross_arm','failed_entry','wrong_index','counts','append_extra'])
def test_journal_identity_and_count_tamper(fault):
    events,r=receipt_entries()
    if fault=='drop_event':events.pop(12)
    elif fault=='duplicate_event':events.insert(12,deepcopy(events[12]))
    elif fault=='swap_nested_return':events[14],events[15]=events[15],events[14]
    elif fault=='cross_arm':events[12]['arm']='actual'
    elif fault=='failed_entry':r['entries'][5]['completed']=False
    elif fault=='wrong_index':r['entries'][5]['index']=4
    elif fault=='counts':r['counts']['imagined']['grad_api']['attempted']=191
    elif fault=='append_extra':events.append(dict(event='attempt',kind='backward_api'))
    with pytest.raises(ValueError):a.work_check(r,events)


@pytest.fixture
def mock_run(tmp_path,material):
    """Full mock disk pack; NEVER passes provenance or scientific admission."""
    p,rows,files=material;folder=tmp_path/'MOCK_run';prep=tmp_path/'MOCK_prepared';folder.mkdir();prep.mkdir()
    def write(path,v):path.write_text(json.dumps(v,sort_keys=True,allow_nan=False))
    p.update(schema='p4_feedback_root_cuda_response_20261027_v1.protocol',source_arm='MOCK',sourcepack_prepare_sha256='MOCK',model_bindings={},known_roots_only=True,
        P1_admission=False,P4_admission=False,runtime_admissible=False,independent_confirmation=False,
        limits=dict(wall_seconds=600,allocated_bytes=1000000,reserved_bytes=2000000,grad_api_per_arm=192,engine_api_per_arm=192,jacobian_api_per_arm=1,predict_per_arm=13,action_encoder_per_arm=13))
    write(prep/'protocol.json',p);write(prep/'prepare.json',dict(status='prepared_not_run',protocol_sha256=a.sha(prep/'protocol.json')))
    binding=dict(prepared=str(prep),prepare_sha256=a.sha(prep/'prepare.json'),sourcepack_prepare_sha256='MOCK')
    files['execution_binding.json']=binding
    for name,value in files.items():write(folder/name,value)
    events,rr=receipt_entries();(folder/'events.jsonl').write_text('\n'.join(json.dumps(x) for x in events)+'\n')
    r=dict(rr,rows=rows,schema='p4_feedback_root_cuda_response_20261027_v1.receipt',status='two_prefix_specific_responses_saved',
        errors=[],blocked=[],source_verified=True,model_unchanged=True,model_hash_before='MOCK',model_hash_after='MOCK',
        root_sha256=p['root_sha256'],source_arm='MOCK',active_metric_id=p['active_metric_id'],model_bindings={},limits=p['limits'],
        prepare_sha256=binding['prepare_sha256'],sourcepack_prepare_sha256='MOCK',
        P1_admission=False,P4_admission=False,runtime_admissible=False,independent_confirmation=False,known_roots_only=True,training=False,
        new_encoder_calls=0,new_projector_calls=0,new_physical_steps=0,journal_sha256=a.sha(folder/'events.jsonl'),
        wall_seconds=1.,total_wall_including_preservation=2.,peaks=dict(allocated_bytes=100,reserved_bytes=200),
        output_files_sha256={name:a.sha(folder/name) for name in a.file_names()})
    write(folder/'receipt.json',r)
    return folder,r,write


def test_mock_full_saved_pack_without_provenance(mock_run):
    folder,_,_=mock_run;r=a.audit(folder,a.sha(folder/'receipt.json'),provenance=False)
    assert r['status']=='mock_arrays_verified_without_provenance' and not r['provenance_verified']
    assert r['P1_admission'] is r['P4_admission'] is False


@pytest.mark.parametrize('fault',['file_sha','receipt_pin','walltime','memory','inventory','admission'])
def test_mock_pack_fail_closed(mock_run,fault):
    folder,r,write=mock_run
    if fault=='file_sha':(folder/'actual.jacobian.json').write_text('[]')
    elif fault=='walltime':r['wall_seconds']=601
    elif fault=='memory':r['peaks']['allocated_bytes']=3000000
    elif fault=='inventory':r['output_files_sha256']['../escape.json']='bad'
    elif fault=='admission':r['P4_admission']=True
    write(folder/'receipt.json',r)
    with pytest.raises(ValueError):a.audit(folder,'bad' if fault=='receipt_pin' else a.sha(folder/'receipt.json'),provenance=False)


def test_live_interface_is_not_driver_admission():
    d=a.live_interface();assert not d['old_runtime_admitted'] and not d['saved_response_dispatch_allowed']
    assert d['boundaries']==[5,10,15] and d['terminal_evaluation_boundary']==20
    b=d['prospective_budget'];assert b['total_predict']==80 and b['prefix_predict_recomputed']==2
    assert b['grad_API']==b['engine_API']==1152 and b['encoded_frames']==20


def test_no_producer_or_neural_imports():
    assert not any(x in sys.modules for x in ('torch','gymnasium','pymunk','stable_worldmodel'))
    # Auditor imports no response producer or compiler even for arithmetic.
    text=Path(a.__file__).read_text();assert 'import pusht_full_packet_compiler' not in text


def test_report_storage_guard(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(a.shutil,'disk_usage',lambda p:SimpleNamespace(free=a.RESERVE))
    with pytest.raises(ValueError):a.write_report(tmp_path/'no_write',{})
    assert not (tmp_path/'no_write').exists()
