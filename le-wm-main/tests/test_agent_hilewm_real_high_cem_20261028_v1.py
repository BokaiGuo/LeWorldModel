"""RAM literal buffers only. No torch/CEM/model/pickle execution in worker."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_hilewm_real_high_cem_20261028_v1 as m


def fp(a,expanded=False):
    stride=[s//a.itemsize for s in a.strides]
    if expanded:stride[1]=0
    return dict(shape=list(a.shape),dtype=a.dtype.str,bytes=a.nbytes,sha256=m.digest(a.tobytes()),
        device='cuda:0',requires_grad=False,stride=stride)


def layout(a):
    return dict(shape=list(a.shape),stride=[s//a.itemsize for s in a.strides],storage_offset=0,
        contiguous=bool(a.flags.c_contiguous),dtype='torch.float32',device='cuda:0')


def case_fixture(batch):
    arrays={};rng=np.random.default_rng(1729+batch)
    def save(name,a):
        name=f'B{batch}_{name}.npy';assert name not in arrays;arrays[name]=np.array(a,copy=True);return name
    original=m.inputs(batch)
    witness=dict(original_keys=['planner_level','z_init','z_goal'],original_metadata=dict(planner_level='high'),
        solver_keys=['z_init','z_goal'],original_tensors={k:fp(a) for k,a in original.items()},
        solver_tensors={k:fp(a) for k,a in original.items()})
    c=dict(batch=batch,status='returned',events=[],cost_attempts=[],native=[],resets=1,starts=(batch+1)//2,
        ends=1,blocked=[],hooks_removed=10,callback_output=[])
    c['initial']=save('initial',np.zeros((batch,2,64),np.float32));calls=[];means=[];stds=[];costs=[]
    for chunk,offset in enumerate(range(0,batch,2)):
        n=min(2,batch-offset);bs=n*8;mean=np.zeros((n,2,64),np.float32);std=np.ones_like(mean)
        expanded={k:np.repeat(a[offset:offset+n,None],8,axis=1) for k,a in original.items()}
        for iteration in range(3):
            index=len(c['events'])
            candidates=rng.normal(size=(n,8,2,64)).astype(np.float32)*std[:,None]+mean[:,None];candidates[:,0]=mean
            history=expanded['z_init'].reshape(bs,1,192);actions=candidates.reshape(n,8,4,32)
            for step in range(4):
                sequence=np.stack([actions[:,:,k].reshape(bs,32) for k in range(step+1)],axis=1)
                # Literal RAM buffers, explicitly NOT a surrogate neural model.
                projected=np.zeros((bs,step+1,192),np.float32)
                high=(np.arange(bs*(step+1)*192,dtype=np.float32).reshape(bs,step+1,192)%31)/np.float32(128)
                final=high.reshape(-1,192).copy()
                for kind,ins,out in (('macro_to_condition',[sequence],projected),
                    ('high_predictor',[history,projected],high),('high_pred_proj',[high.reshape(-1,192)],final)):
                    event=dict(phase='attempt',kind=kind,cost_index=index,step=step,
                        batch=bs*(step+1) if kind=='high_pred_proj' else bs,
                        context=1 if kind=='high_pred_proj' else step+1,
                        inputs=[save(f'n{len(c["native"])}_in{j}',v) for j,v in enumerate(ins)],
                        input_layouts=[layout(v) for v in ins])
                    c['native'].append(event)
                    c['native'].append(dict(event,phase='return',output=save(f'n{len(c["native"])}_out',out),output_layout=layout(out)))
                history=np.concatenate((history,final.reshape(bs,step+1,192)[:,-1:]),axis=1)
            terminal=history[:,-1].reshape(n,8,192)
            cost=np.square(terminal-expanded['z_goal']).sum(axis=-1,dtype=np.float32)
            inds=np.argsort(cost,axis=1)[:,:3].astype(np.int64);ix=np.arange(n)[:,None]
            elites=candidates[ix,inds];vals=cost[ix,inds]
            newmean=elites.mean(axis=1);newstd=elites.std(axis=1,ddof=1)
            state=dict(candidates=candidates,costs=cost,topk_vals=vals,topk_inds=inds,topk_candidates=elites,
                mean=newmean,var=newstd,prev_mean=mean,prev_var=std)
            c['events'].append(dict(step=iteration,chunk=chunk,attempt=index,
                arrays={k:save(f'e{index}_{k}',v) for k,v in state.items()},cuda_exact_checks={k:True for k in m.CHECKS}))
            c['cost_attempts'].append(dict(index=index,chunk=chunk,returned=True,candidate_rows=n*8,
                **{k:save(f'cost{index}_{k}',v) for k,v in expanded.items()},
                candidates=save(f'cost{index}_candidates',candidates),cost=save(f'cost{index}_value',cost),native_cost_reduction_exact=True))
            calls.append(dict(attempt=index+1,batch=n,candidate_rows=n*8,metadata='high',returned=True,
                candidates=fp(candidates),cost=fp(cost),expanded={k:fp(a,True) for k,a in expanded.items()}))
            c['callback_output'].append(dict(chunk=chunk,step=iteration));mean,std=newmean,newstd
        means.append(mean);stds.append(std);costs.append(vals.mean(axis=1))
    c['outputs']={k:save('output_'+k,v) for k,v in (('actions',np.concatenate(means)),('mean',np.concatenate(means)),
        ('var',np.concatenate(stds)),('costs',np.concatenate(costs)))}
    c['bridge_receipt']=dict(schema=m.seam.bridge_module().TAG,backend='actual_torch_CUDA_float32',tensor_batch=batch,
        input_witness=witness,cost_attempts=len(calls),cost_returns=len(calls),cost_failures=0,
        candidate_rows_attempted=batch*8*3,calls=calls,predict_API_calls=None,predictor_samples=None,
        scientific_admission=False,official_unadapted_reproduction=False)
    return c,arrays


@pytest.fixture(scope='module')
def ram():
    cases=[];arrays={}
    for b in (5,2):
        c,a=case_fixture(b);cases.append(c);arrays.update(a)
    return cases,arrays


def test_fixed_real_cost_contract_not_analytic():
    p=m.protocol();assert p['horizon']==2 and p['action_block']*p['macro_dim']==p['dim']==64
    assert (p['cost_attempts'],p['candidate_rows'],p['high_predictor_calls'],p['high_predictor_sample_rows'])==(12,168,48,672)
    assert p['pixel_encoder']==p['macro_encoder']==p['AD']==p['physics']==0
    assert p['bounds']==[-1.,1.] and not p['baseline_runnable'] and not p['scientific_admission']
    for b in (5,2):
        expected=((np.arange(b*192)%31-15+1729+b).astype(np.float32)/64).reshape(b,192)
        assert m.inputs(b)['z_init'].tobytes()==expected.tobytes()


def test_all_tail_native_rows_and_complete_cem_algebra(ram):
    cases,arrays=ram
    for c in cases:m.validate_case(c,arrays,'cuda:0')
    r=m.resources(cases)
    assert r['cost']==dict(attempted=12,returned=12) and r['bridge_candidate_rows']==168
    assert r['modules']['high_predictor']==dict(attempted=48,returned=48,sample_rows=672,context_tokens=1680)
    assert cases[0]['bridge_receipt']['calls'][-1]['batch']==1
    assert any(np.max(np.abs(a))>1 for name,a in arrays.items() if name.endswith('_candidates.npy'))  # no silent clamp


@pytest.mark.parametrize('fault',['tail','metadata','source','fingerprint','stride','drop','topk','indices','std','chain',
    'mean0','output','history','macro','proj','native_order','native_extra','resource','native_flag','precision_dtype'])
def test_mutations_rejected(ram,fault):
    cases,arrays=deepcopy(ram);c=cases[0];x=c['events'][0]['arrays'];call=c['bridge_receipt']['calls'][0]
    if fault=='tail':c['bridge_receipt']['calls'][-1]['batch']=2
    elif fault=='metadata':call['metadata']='low'
    elif fault=='source':arrays[c['cost_attempts'][0]['z_init']][0,0,0]+=.1
    elif fault=='fingerprint':call['candidates']['sha256']='0'*64
    elif fault=='stride':call['expanded']['z_init']['stride'][1]=192
    elif fault=='drop':c['events'].pop()
    elif fault=='topk':arrays[x['topk_vals']][0,0]+=1
    elif fault=='indices':arrays[x['topk_inds']][0,0]=8
    elif fault=='std':arrays[x['var']]+=np.float32(.1)
    elif fault=='chain':arrays[c['events'][1]['arrays']['prev_mean']]+=np.float32(.1)
    elif fault=='mean0':arrays[x['candidates']][:,0]+=np.float32(.1)
    elif fault=='output':arrays[c['outputs']['actions']]+=np.float32(.1)
    elif fault in ('history','macro','proj'):
        kind={'history':'high_predictor','macro':'macro_to_condition','proj':'high_pred_proj'}[fault]
        e=next(e for e in c['native'] if e['kind']==kind and e['phase']=='attempt')
        arrays[e['inputs'][0]].flat[0]+=np.float32(.1)
    elif fault=='native_order':c['native'][0],c['native'][2]=c['native'][2],c['native'][0]
    elif fault=='native_extra':c['native'].append(deepcopy(c['native'][-1]))
    elif fault=='resource':c['native'][0]['batch']-=1
    elif fault=='native_flag':c['cost_attempts'][0]['native_cost_reduction_exact']=False
    else:arrays[x['costs']]=arrays[x['costs']].astype(np.float64)
    with pytest.raises((ValueError,IndexError)):m.validate_case(c,arrays,'cuda:0')


def test_coherently_wrong_cost_rejected_not_only_flags(ram):
    cases,arrays=deepcopy(ram);c=cases[0]
    # Shift every cost/callback/elite/final logging value consistently, preserve
    # order/topk/distribution and all claimed CUDA flags. Only native terminal
    # arithmetic can reject this coherent false cost.
    for e,a,call in zip(c['events'],c['cost_attempts'],c['bridge_receipt']['calls'],strict=True):
        for f in (e['arrays']['costs'],e['arrays']['topk_vals'],a['cost']):arrays[f]+=np.float32(100)
        call['cost']=fp(arrays[a['cost']])
    arrays[c['outputs']['costs']]+=np.float32(100)
    with pytest.raises(ValueError,match='native cost arithmetic bound'):m.validate_case(c,arrays,'cuda:0')


def test_cost_arithmetic_bound_separate_from_cem_tolerance():
    z=np.full((2,8,192),np.float32(27));g=np.linspace(-1,1,192,dtype=np.float32)[None,None]+np.zeros_like(z)
    actual=np.square(z-g).sum(axis=-1,dtype=np.float32);m.check_cost(actual,z,g)
    # Conservative gamma bound permits a few ULP of scalar reduction, not a
    # coherent +100 cost or any tolerance on neural outputs.
    m.check_cost(np.nextafter(actual,np.float32(np.inf)),z,g)
    with pytest.raises(ValueError):m.check_cost(actual+100,z,g)
    m.check_cost(np.zeros((2,8),np.float32),z,z)
    with pytest.raises(ValueError):m.check_cost(np.full((2,8),1e-40,np.float32),z,z)


def test_failed_native_call_and_bridge_rows_never_refunded(ram):
    cases,_=deepcopy(ram);c=cases[0];c['cost_attempts'][0]['returned']=False
    c['native']=[e for e in c['native'] if not(e['cost_index']==0 and e['phase']=='return')]
    r=m.resources(cases);assert r['cost']==dict(attempted=12,returned=11)
    assert r['bridge_candidate_rows']==168 and r['modules']['high_predictor']['attempted']==48
    assert r['modules']['high_predictor']['returned']==44


@pytest.mark.parametrize('main,v3',[(False,False),(True,False),(False,True)])
def test_no_execution_without_both_main_flags(monkeypatch,main,v3):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('no reads before approvals'))
    monkeypatch.setattr(m.seam,'load_runtime',lambda:pytest.fail('no torch/solver'))
    with pytest.raises(ValueError):m.run_cuda(None,None,None,main_authorized=main,main_v3_allfour_approved=v3)


def test_authority_entire_v2_receipt_rebuilt_not_fullbase(monkeypatch):
    result=dict(status='RAM_DIAGNOSTIC',scientific_admission=False)
    parent={k:k for k in ('export_report','export_report_sha256','export_receipt','export_receipt_sha256')}
    calls=[];monkeypatch.setattr(m.diag,'consume',lambda *a:calls.append(a) or deepcopy(result))
    monkeypatch.setattr(m,'read',lambda *a:deepcopy(result));monkeypatch.setattr(m.diag,'upstream',lambda:(parent,[]))
    a=m.authority();assert a['diagnostic_only'] and calls==[(m.V2,m.V2_SHA,m.V2_REPORT,m.V2_REPORT_SHA)]
    monkeypatch.setattr(m,'read',lambda *a:dict(result,unexpected='not equal'))
    with pytest.raises(ValueError):m.authority()


def test_manifest_whole_rebuild_without_production_prepare(monkeypatch):
    value=dict(schema=m.TAG+'.prepare',RAM_ONLY=True);body=m.encode(value)
    monkeypatch.setattr(m,'read',lambda *a:deepcopy(value));monkeypatch.setattr(m,'build',lambda:deepcopy(value))
    monkeypatch.setattr(Path,'read_bytes',lambda path:body)
    assert m.verify('/RAM/manifest.json',m.digest(body))==value
    monkeypatch.setattr(m,'build',lambda:dict(value,RAM_ONLY=False))
    with pytest.raises(ValueError):m.verify('/RAM/manifest.json',m.digest(body))


def test_cold_import_no_torch_foreign_pickle_or_mutable_v3():
    code="""
import builtins
original=builtins.__import__
def guarded(name,*a,**k):
    assert name.split('.')[0] not in {'torch','pickle','gymnasium','stable_worldmodel','transformers'}, name
    assert 'trusted_cuda_validation_20261028_v3' not in name
    return original(name,*a,**k)
import numpy
from experiments.spectral_reachability import agent_hilewm_real_high_cem_20261028_v1
"""
    # NumPy's own optional import of pickle is outside the adapter; install
    # import guard after NumPy to test adapter's new imports precisely.
    code=code.replace('import builtins','import numpy\nimport builtins',1).replace('\nimport numpy\nfrom experiments','\nfrom experiments')
    subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,check=True,timeout=30)


def test_store_actual32mib_and_exclusive(tmp_path,monkeypatch):
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    s=m.Store(tmp_path/'results'/(m.TAG+'_RAM'),m.time.monotonic())
    s.array('array.npy',np.zeros((2,8),np.float32))
    assert s.used==sum(f.stat().st_size for f in s.path.iterdir())
    with pytest.raises(ValueError):s.array('array.npy',np.zeros(1,np.float32))
    with pytest.raises(ValueError):s.check(m.CAP)
    with pytest.raises(ValueError):s.put('../out',b'no')


def test_owned_root_direct_binding_not_analytic_substitution():
    source=Path(m.__file__).read_text()
    assert 'ObservedHigh.bind_high_model(info,root,' in source and 'super().get_cost(expanded,candidates)' in source
    assert 'Analytic()' not in source and 'root.get_cost=' not in source and 'torch.load(' not in source
    assert 'main_v3_allfour_approved is True' in source


@pytest.mark.parametrize('fault',[None,'payload','census','model','resources','partial'])
def test_saved_consume_full_ram_artifact(tmp_path,monkeypatch,ram,fault):
    # Only literal buffers on temporary disk; never construct/load a model or
    # production prepare. Exercise actual NPY/report/census consumer end to end.
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    store=m.Store(tmp_path/'results'/(m.TAG+'_RAM_saved'),m.time.monotonic())
    cases,arrays=deepcopy(ram)
    for name,a in arrays.items():store.array(name,a)
    authority={k:k for k in ('export_report','export_report_sha256','export_receipt','export_receipt_sha256')}
    manifest=dict(sources={'RAM':'literal'},authority=dict(export=authority))
    monkeypatch.setattr(m,'verify',lambda *a:deepcopy(manifest))
    weights={'RAM_tensor':'RAM_value_hash'}
    monkeypatch.setattr(m.diag.loader,'authorities',lambda *a:(None,dict(entries=[dict(key='RAM_tensor')]),
        None,dict(rows=[dict(array=dict(value_sha256='RAM_value_hash'))])))
    construction=dict(tensor_value_hashes=weights,all_parameters_frozen=True,
        tensor_copy_attempts=425,tensor_copy_returns=425,export_report_sha256=authority['export_report_sha256'],
        export_receipt_sha256=authority['export_receipt_sha256'],ir_sha256=m.diag.loader.IR_SHA,device='cuda:0',torch='RAM')
    guard=dict(deterministic=True,tf32_matmul=False,tf32_cudnn=False,autocast=False,grad=False)
    report=dict(schema=m.TAG+'.report',manifest_path=str(tmp_path/'manifest.json'),manifest_sha256='RAM_sha',
        protocol=m.protocol(),sources=manifest['sources'],completed=True,errors=[],main_v3_allfour_approved=True,
        elapsed_seconds=1.,cases=cases,model_before=deepcopy(weights),model_after=deepcopy(weights),
        construction=construction,runtime=dict(device='cuda:0',torch='RAM',deterministic=True,
            tf32_matmul=False,tf32_cudnn=False,autocast=False),precision_before=guard,precision_after=guard,
        arrays=store.arrays,resources=m.resources(cases),baseline_runnable=False,official_parity=False,scientific_admission=False)
    if fault=='payload':
        # SHA changed while claimed array metadata did not.
        with (store.path/next(iter(arrays))).open('ab') as f:f.write(b'trailing')
    elif fault=='census':store.put('unreferenced.npy',b'not an array')
    elif fault=='model':report['model_after']['RAM_tensor']='changed'
    elif fault=='resources':report['resources']['bridge_candidate_rows']-=1
    elif fault=='partial':report['completed']=False
    body=m.encode(report);store.put('report.json',body,final=True)
    args=(tmp_path/'manifest.json','RAM_sha',store.path/'report.json',m.digest(body))
    if fault:
        with pytest.raises(ValueError):m.consume(*args)
    else:
        result=m.consume(*args)
        assert result['resources']['cost']==dict(attempted=12,returned=12)
        assert not result['scientific_admission'] and not result['neural_parity_tolerance_used']
