"""Exception/storage/metadata/mock tests only. No CUDA/neural/physics execution."""
from copy import deepcopy
from types import SimpleNamespace as NS
from pathlib import Path
from contextlib import nullcontext
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import p2_composition_predict_frontier_cuda_20261027_v2 as m
from test_p2_composition_predict_frontier_20261027_v1 import base,legacy_inputs
from test_p2_composition_predict_frontier_cuda_20261027_v1 import fake_model,tensor,dispatch,operation_fixture


class RestoreFailure(NS):
    def __setattr__(self,name,value):
        if name in vars(self).get('restore_targets',{}) and value is self.restore_targets[name]:
            raise OSError('restore '+name)
        super().__setattr__(name,value)


def restore_model():
    model,replay,calls=fake_model()
    model=RestoreFailure(**vars(model));encoder=RestoreFailure(**vars(model.action_encoder));model.action_encoder=encoder
    model.restore_targets={'predict':model.predict};encoder.restore_targets={'forward':encoder.forward}
    return model,replay,calls


def test_kernel_primary_survives_terminal_journal_and_all_restore_errors():
    model,replay,_=restore_model();kernel=RuntimeError('kernel original');faults=m.Faults();events=[]
    def fail(x):raise kernel
    model.action_encoder.forward=fail;model.action_encoder.restore_targets={'forward':fail}
    replay._p2_scope=m.expected_scopes()[0]
    def journal(event):
        events.append(event)
        if event['kind']=='entry_terminal':raise OSError('journal terminal')
    with pytest.raises(RuntimeError) as caught:
        with m.guarded_entries(model,replay,journal,faults):model.action_encoder.forward(tensor('action_encoder'))
    assert caught.value is kernel and faults.primary is kernel
    assert [r['error'] for r in faults.receipt()['secondary']]==['journal terminal','restore forward','restore predict']
    assert events[0]['kind']=='entry_attempt' and not events[1]['entry']['completed']


def test_restore_without_work_error_raises_first_restore_after_attempting_all():
    model,replay,_=restore_model();faults=m.Faults()
    with pytest.raises(OSError,match='restore forward') as caught:
        with m.guarded_entries(model,replay,lambda e:None,faults):pass
    assert caught.value is faults.primary
    assert [e['error'] for e in faults.records]==['restore forward','restore predict']


def test_guard_body_exception_not_replaced_by_restore():
    model,replay,_=restore_model();original=KeyboardInterrupt('cancel');faults=m.Faults()
    with pytest.raises(KeyboardInterrupt) as caught:
        with m.guarded_entries(model,replay,lambda e:None,faults):raise original
    assert caught.value is original and faults.primary is original and len(faults.records)==3


def test_attempt_journal_failure_stops_before_dispatch_and_poisons():
    model,replay,calls=fake_model();faults=m.Faults();replay._p2_scope=m.expected_scopes()[0]
    original=OSError('cannot fsync attempt')
    def journal(e):
        if e['kind']=='entry_attempt':raise original
    with m.guarded_entries(model,replay,journal,faults) as entries:
        with pytest.raises(OSError) as caught:model.action_encoder.forward(tensor('action_encoder'))
        assert caught.value is original
        with pytest.raises(ValueError):model.action_encoder.forward(tensor('action_encoder'))
    assert not calls and len(entries)==1 and not entries[0]['completed'] and faults.primary is original


def test_metadata_error_survives_blocked_journal_failure():
    model,replay,calls=fake_model();faults=m.Faults()
    def journal(e):raise OSError('blocked log failure')
    with pytest.raises(ValueError) as caught:
        with m.guarded_entries(model,replay,journal,faults):model.predict(tensor('predict'))
    assert caught.value is faults.primary and not calls
    assert faults.records[1]['error']=='blocked log failure'


def test_full_dual_guard_still332_and_same_mathematics():
    model,replay,calls=fake_model();faults=m.Faults()
    with m.guarded_entries(model,replay,lambda e:None,faults) as entries:
        for spec,scope in zip(m.expected_entries(80),m.expected_scopes()):
            replay._p2_scope=scope;dispatch(model,spec['kind'],tensor(spec['kind']))
    assert m.validate_actual(entries)['standalone_predict']==[168,168] and len(calls)==664
    assert faults.receipt()==dict(primary=None,secondary=[])


@pytest.mark.parametrize('arm',m.ARMS)
def test_actual_new_run_arm_mock_count_unchanged(arm):
    _,args,calls,_=legacy_inputs();args['bank']=m.candidate_bank(base(),m.ROOTS[0])
    r=m.run_arm(arm=arm,**args)
    assert r['complete'] and len(calls)==164 and r['faults']==dict(primary=None,secondary=[])
    assert len(r['attempts'])==(80 if arm==m.ARMS[0] else 40)


def test_run_arm_primary_survives_failure_journal():
    _,args,_,_=legacy_inputs();args['bank']=m.candidate_bank(base(),m.ROOTS[0]);original=RuntimeError('metric')
    def metric(c):raise original
    def journal(e):
        if e.get('kind')=='arm_failure':raise OSError('failed failure journal')
    args['target_metric']=metric;faults=m.Faults();r=m.run_arm(arm=m.ARMS[0],journal=journal,faults=faults,**args)
    assert not r['complete'] and faults.primary is original
    assert r['faults']['secondary'][0]['error']=='failed failure journal'


def derivative_fixture(tmp_path,*,journal=None,solve=None,nominal=None,jacobian=None,convert=None):
    (tmp_path/'arrays').mkdir();faults=m.Faults();receipts=[];stages=[];order=[];events=[]
    def sink(e):
        events.append(deepcopy(e))
        if journal:journal(e)
    emit=faults.journal(sink)
    def save(stage,a):
        order.append('save_'+stage)
        return m.persist_array(tmp_path,'root_0_shared_'+stage,a,dict(root_sha256=m.ROOTS[0]),receipts,emit)
    def n():order.append('nominal');return np.zeros(192)
    def j():
        assert (tmp_path/'arrays/root_0_shared_nominal.json').exists()
        order.append('jacobian');return np.eye(192,20)
    def s(j,n):
        assert (tmp_path/'arrays/root_0_shared_jacobian.json').exists()
        order.append('solve');return 'solved'
    kw=dict(nominal=nominal or n,jacobian=jacobian or j,convert=convert or np.asarray,persist=save,
        solve=solve or s,journal=emit,faults=faults,stages=stages,kind='jacobian')
    return kw,faults,receipts,stages,order,events


def test_arrays_fsynced_before_Jacobian_and_solve(tmp_path):
    kw,faults,receipts,stages,order,events=derivative_fixture(tmp_path)
    assert m.derivative_stages(**kw)=='solved'
    assert order==['nominal','save_nominal','jacobian','save_jacobian','solve']
    assert all(s['persisted'] for s in stages) and len(receipts)==2 and faults.primary is None


@pytest.mark.parametrize('failure',['jacobian','solve','convert_nominal','convert_jacobian','save_nominal','save_jacobian'])
def test_partial_array_durability_no_retry(tmp_path,monkeypatch,failure):
    original=RuntimeError(failure)
    def fail(*a):raise original
    kw,faults,receipts,stages,order,_=derivative_fixture(tmp_path)
    if failure in ('jacobian','solve'):kw[failure]=fail
    elif failure.startswith('convert'):
        def convert(a):
            if (a.ndim==1)==(failure=='convert_nominal'):raise original
            return a
        kw['convert']=convert
    else:
        write=m.write
        def broken(path,value):
            if Path(path).stem.endswith(failure.removeprefix('save_')):raise original
            return write(path,value)
        monkeypatch.setattr(m,'write',broken)
    with pytest.raises(RuntimeError) as caught:m.derivative_stages(**kw)
    assert caught.value is original and 'solve' not in order
    expected=failure in ('jacobian','solve','convert_jacobian','save_jacobian')
    assert (tmp_path/'arrays/root_0_shared_nominal.json').exists()==expected
    assert (tmp_path/'arrays/root_0_shared_jacobian.json').exists()==(failure=='solve')


def test_Jacobian_persisted_even_when_completion_journal_fails(tmp_path):
    original=OSError('Jacobian completion log')
    def journal(e):
        if e.get('kind')=='jacobian' and e['completed']:raise original
    kw,faults,receipts,_,order,_=derivative_fixture(tmp_path,journal=journal)
    with pytest.raises(OSError) as caught:m.derivative_stages(**kw)
    assert caught.value is original and len(receipts)==2 and 'solve' not in order
    assert m.read(tmp_path/'arrays/root_0_shared_jacobian.json')['shape']==[192,20]


def test_array_remains_if_its_receipt_journal_fails(tmp_path):
    def journal(e):
        if e.get('kind')=='array_saved':raise OSError('array receipt')
    kw,_,receipts,_,order,_=derivative_fixture(tmp_path,journal=journal)
    with pytest.raises(OSError):m.derivative_stages(**kw)
    assert len(receipts)==1 and m.read(tmp_path/'arrays/root_0_shared_nominal.json')['shape']==[192]
    assert 'jacobian' not in order


@pytest.mark.parametrize('mode',['ridge','mode_svd'])
def test_old_mathematical_failure_and_finally_log_failure_keep_primary(tmp_path,monkeypatch,mode):
    old=m.legacy();original=RuntimeError(mode);secondary=OSError('SVD finally log')
    def journal(e):
        if e.get('kind')==('bounded_ridge_svd' if mode=='ridge' else 'mode_svd') and not e['completed'] and 'wall_time_s' in e:raise secondary
    if mode=='ridge':
        def fail(*a,**kw):raise original
        monkeypatch.setattr(old.compiler,'bounded_ridge',fail)
    else:
        svd=np.linalg.svd;calls=[]
        def fail_mode(*a,**kw):
            calls.append(1)
            if len(calls)==2:raise original
            return svd(*a,**kw)
        monkeypatch.setattr(np.linalg,'svd',fail_mode)
    kw,faults,receipts,_,_,_=derivative_fixture(tmp_path,journal=journal)
    kw['solve']=lambda j,n:old.sra.spectral_bank(j,n,np.zeros(192),np.eye(192),journal=kw['journal'])
    with pytest.raises(RuntimeError) as caught:m.derivative_stages(**kw)
    assert caught.value is original and faults.primary is original and len(receipts)==2
    assert faults.exceptions[1] is secondary


def mock_row(index=0):
    return dict(root_index=index,root_sha256=m.ROOTS[index],complete=True,results=[],
        completed_operations=dict(jacobian_calls=0,scalar_output_vjp_directions=0,svd_calls=0,bounded_solve_calls=0),
        resource_receipt=dict(unique=dict(predict_calls=0,action_encoder_calls=0)))


def healthy_work(state,faults):
    state['row_cache']=[mock_row(i) for i in range(4)]
    state['rows']=[dict(root_index=i,complete=True) for i in range(4)]


def source_ok(state):return dict(source_unchanged=True,protocol_sha256='p')
def model_ok(state):return dict(model_unchanged=True,frozen_eval=True)


@pytest.mark.parametrize('stage',['model.load','model.initial_hash','root.synchronize','row.hash','source.compilation_read'])
def test_execution_primary_report_with_both_finalizations(tmp_path,stage):
    original=RuntimeError(stage);calls=[]
    def work(state,faults):state['execution_stage']=stage;raise original
    def source(state):calls.append('source');return source_ok(state)
    def model(state):calls.append('model');return model_ok(state)
    with pytest.raises(RuntimeError) as caught:m.finalized_run(tmp_path,{},work,source,model)
    report=m.read(tmp_path/'report.json')
    assert caught.value is original and report['faults']['primary']['stage']==stage
    assert report['status']=='incomplete_or_failed' and calls==['source','model']
    assert (tmp_path/'source_finalization.json').exists() and (tmp_path/'model_finalization.json').exists()


def test_source_model_finalize_errors_are_independent_and_secondary(tmp_path):
    original=RuntimeError('kernel');source_failure=ValueError('source drift');model_failure=OSError('model hash')
    def work(state,faults):raise original
    def source(state):raise source_failure
    def model(state):raise model_failure
    with pytest.raises(RuntimeError) as caught:m.finalized_run(tmp_path,{},work,source,model)
    report=m.read(tmp_path/'report.json');assert caught.value is original
    assert [e['error'] for e in report['faults']['secondary']]==['source drift','model hash']
    assert report['finalization']['source']['status']==report['finalization']['model']['status']=='failed'


@pytest.mark.parametrize('filename',['execution.json','source_finalization.json','model_finalization.json','report.json','completion.json'])
def test_finalization_write_failures_never_skip_other_checks(tmp_path,filename):
    original=OSError(filename);calls=[]
    def writer(path,value):
        if Path(path).name==filename:raise original
        return m.write(path,value)
    def source(state):calls.append('source');return source_ok(state)
    def model(state):calls.append('model');return model_ok(state)
    with pytest.raises(OSError) as caught:m.finalized_run(tmp_path,{},healthy_work,source,model,writer=writer)
    assert caught.value is original and calls==['source','model']
    emergency=m.read(tmp_path/'finalization_failure.json');assert emergency['faults']['primary']['error']==filename
    assert not (tmp_path/'completion.json').exists()


def test_work_primary_survives_final_report_write_failure(tmp_path):
    original=RuntimeError('load original')
    def work(state,faults):raise original
    def writer(path,value):
        if Path(path).name=='report.json':raise OSError('report disk')
        return m.write(path,value)
    with pytest.raises(RuntimeError) as caught:m.finalized_run(tmp_path,{},work,source_ok,model_ok,writer)
    assert caught.value is original
    emergency=m.read(tmp_path/'finalization_failure.json')
    assert emergency['faults']['primary']['error']=='load original' and emergency['faults']['secondary'][0]['error']=='report disk'


def test_emergency_write_failure_cannot_replace_primary(tmp_path):
    original=RuntimeError('kernel')
    def work(state,faults):raise original
    def writer(path,value):
        if Path(path).name in ('report.json','finalization_failure.json'):raise OSError('storage unavailable')
        return m.write(path,value)
    with pytest.raises(RuntimeError) as caught:m.finalized_run(tmp_path,{},work,source_ok,model_ok,writer)
    assert caught.value is original


@pytest.mark.parametrize('check',['source','model'])
def test_finalization_false_never_accepted(tmp_path,check):
    with pytest.raises(ValueError):m.finalized_run(tmp_path,{},healthy_work,
        (lambda s:dict(source_unchanged=False)) if check=='source' else source_ok,
        (lambda s:dict(model_unchanged=True,frozen_eval=False)) if check=='model' else model_ok)
    assert m.read(tmp_path/'report.json')['status']=='incomplete_or_failed'


def test_report_commit_only_after_clean_finalization(tmp_path):
    report=m.finalized_run(tmp_path,dict(protocol_sha256='p'),healthy_work,source_ok,model_ok)
    assert report['status']=='cuda_completed' and report['faults']==dict(primary=None,secondary=[])
    assert m.read(tmp_path/'completion.json')['report_sha256']==m.file_sha(tmp_path/'report.json')


@pytest.mark.parametrize('failure',['write','hash','file_hash'])
def test_real_row_commit_failure_final_report_still_written(tmp_path,monkeypatch,failure):
    original=OSError('row '+failure);original_write=m.write;original_digest=m.digest;original_sha=m.file_sha
    def write(path,value):
        if Path(path).name=='root_0.json' and failure=='write':raise original
        return original_write(path,value)
    def digest(value):
        if failure=='hash' and isinstance(value,dict) and 'resource_receipt' in value:raise original
        return original_digest(value)
    def sha(path):
        if failure=='file_hash' and Path(path).name=='root_0.json':raise original
        return original_sha(path)
    monkeypatch.setattr(m,'write',write);monkeypatch.setattr(m,'digest',digest);monkeypatch.setattr(m,'file_sha',sha)
    def work(state,faults):m.commit_row(tmp_path,mock_row(),state,faults)
    with pytest.raises(OSError) as caught:m.finalized_run(tmp_path,{},work,source_ok,model_ok)
    assert caught.value is original and m.read(tmp_path/'report.json')['faults']['primary']['error']=='row '+failure


def test_actual_root_sync_fault_preserves_partial_arrays_and_row(tmp_path,monkeypatch):
    old=m.legacy();original=RuntimeError('synchronize');(tmp_path/'arrays').mkdir()
    _,args,_,_=legacy_inputs();ready=dict(root_index=0,root_sha256=m.ROOTS[0],root=args['root'])
    compiled=dict(goal_vectors=[np.zeros(192).tolist()],scoring=dict(metric=dict(projection=np.eye(192).tolist())))
    class Replay:
        def __init__(self,*a):self._p2_scope=None
    monkeypatch.setattr(old.full20,'RootAnchoredCudaReplay',Replay)
    def compile(model,device,ready,compiled,replay,journal,save,faults,stages):
        save('nominal',np.zeros(192));raise RuntimeError('Jacobian earlier')
    monkeypatch.setattr(m,'compile_cuda',compile)
    def synchronize():raise original
    model,_,_=fake_model()
    faults=m.Faults();row=m.run_root(tmp_path,{},dict(source_seals=[]),'a'*64,ready,compiled,model,None,synchronize,faults)
    assert row['error']['error']=='Jacobian earlier'
    assert 'synchronize' in [e['error'] for e in row['faults']['secondary']]
    assert not row['complete'] and len(row['array_receipts'])==1


def test_prepare_v2_only_and_old_v1_byte_invariance(tmp_path):
    old_files=[m.PROJECT/'experiments/spectral_reachability/p2_composition_predict_frontier_cuda_20261027_v1.py',
        m.PROJECT/'tests/test_p2_composition_predict_frontier_cuda_20261027_v1.py',m.PROJECT/'innerdocs/p2_composition_predict_frontier_cuda_20261027_v1.md']
    expected=[m.file_sha(p) for p in old_files]
    output=tmp_path/(m.NAME+'_prepared');result=m.prepare(m.PROJECT/'results/p2_composition_paired_20261027_v2_prepared01',output)
    _,p,s=m.load_prepared(output,result['prepared_sha256'])
    assert p['contract']['derivative_durability'].startswith('R2') and s['new_neural_calls']==0
    assert [m.file_sha(p) for p in old_files]==expected
    with pytest.raises(ValueError):m.load_prepared(output,'f'*64)


class MockTensor(np.ndarray):
    """NumPy storage carrying fake CUDA metadata; NO torch/neural execution."""
    def __new__(cls,value):return np.asarray(value,dtype=np.float32).view(cls)
    @property
    def device(self):return NS(type='cuda')
    def new_zeros(self,n):return MockTensor(np.zeros(n))
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return np.asarray(self)
    def requires_grad_(self,value):return self


@pytest.mark.parametrize('which',['shared','suffix'])
def test_actual_v2_primitives_mock_torch_equal_sealed_old_math(tmp_path,monkeypatch,which):
    old,args,_,_=legacy_inputs();jac=np.eye(192,20 if which=='shared' else 10,dtype=np.float32)
    dispatches=[]
    def transition(emb,history,raw):
        model.action_encoder.forward(MockTensor(np.zeros((1,3,10))))
        model.predict(MockTensor(np.zeros((1,3,192))))
        z=np.asarray(emb[0,-1]).copy();z[:10]+=np.asarray(raw).reshape(10)
        state=np.concatenate((emb[:,1:],z[None,None]),axis=1)
        between=np.concatenate((history[:,1:],np.asarray(raw).reshape(1,1,10)),axis=1)
        return MockTensor(state),MockTensor(between)
    model=NS(predict=lambda *a:dispatches.append('predict'),action_encoder=NS(forward=lambda *a:dispatches.append('actenc')))
    def autograd(fn,zero,vectorize):
        assert vectorize is True;fn(zero);return MockTensor(jac)
    fake_torch=NS(float32=np.float32,tensor=lambda x,**kw:MockTensor(x),no_grad=nullcontext,
        tanh=lambda x:MockTensor(np.tanh(x)),autograd=NS(functional=NS(jacobian=autograd)))
    monkeypatch.setitem(sys.modules,'torch',fake_torch)
    monkeypatch.setattr(old.compiler,'make_transition',lambda *a:transition)
    compiled=dict(goal_vectors=[np.ones(192).tolist()],scoring=dict(metric=dict(projection=np.eye(192).tolist())))
    ready=dict(root=args['root']);device=NS(type='cuda');replay=NS(_p2_scope=None)
    if which=='shared':expected,_=old.sra.compile_cuda(model,device,ready,compiled,replay,lambda e:None)
    else:expected=old.suffix_cuda(model,device,replay,args['root']['context'],np.zeros(10),compiled,0,lambda e:None)
    dispatches.clear();faults=m.Faults();stages=[];receipts=[];(tmp_path/'arrays').mkdir()
    def save(stage,value):return m.persist_array(tmp_path,'mock_'+stage,value,{},receipts,lambda e:None)
    if which=='shared':actual=m.compile_cuda(model,device,ready,compiled,replay,lambda e:None,save,faults,stages)
    else:actual=m.suffix_cuda(model,device,replay,args['root']['context'],np.zeros(10),compiled,0,lambda e:None,save,faults,stages)
    assert actual==expected and replay._p2_scope is None and faults.primary is None
    assert dispatches.count('predict')==dispatches.count('actenc')==(4 if which=='shared' else 2)
    assert len(receipts)==len(stages)==2 and all(e['persisted'] for e in stages)


@pytest.mark.parametrize('damage',[None,'value','goal','context','index','receipt','stage','missing','name'])
def test_82_source_bound_array_export_validation(tmp_path,damage):
    old,args,_,_=legacy_inputs();(tmp_path/'arrays').mkdir()
    ready=dict(root_sha256=m.ROOTS[0],root_index=0,root=args['root'])
    p={};prepared_sha='a'*64;compiled=dict(goal_vectors=[np.zeros(192).tolist()],scoring=dict(metric=dict(projection=np.eye(192).tolist())))
    bank=m.candidate_bank(base(),m.ROOTS[0]);saved=dict(base=dict(nominal=np.zeros(192).tolist(),jacobian=np.zeros((192,20)).tolist()),expanded=bank)
    binding=m.array_binding(p,ready,compiled,prepared_sha);receipts=[];stages=[];attempts=[]
    for stage in ('nominal','jacobian'):
        m.persist_array(tmp_path,'root_0_shared_'+stage,saved['base'][stage],binding,receipts,lambda e:None)
        stages.append(dict(kind='jacobian',index=None,stage=stage,returned=True,converted=True,persisted=True))
    for i in range(40):
        context=args['root']['context'];local=dict(nominal=np.zeros(192).tolist(),jacobian=np.zeros((192,10)).tolist())
        attempts.append(dict(suffix_compilation=local,witness=dict(edges=[dict(output_context=context)])))
        b=dict(binding,input_context_sha256=old.digest(context),prefix_logits_sha256=old.digest(bank['logits'][i][:10]),index=i)
        for stage in ('nominal','jacobian'):
            m.persist_array(tmp_path,f'root_0_suffix_{i}_{stage}',local[stage],b,receipts,lambda e:None)
            stages.append(dict(kind='suffix_jacobian',index=i,stage=stage,returned=True,converted=True,persisted=True))
    row=dict(results=[dict(arm=m.ARMS[0]),dict(arm=m.ARMS[1],attempts=attempts)],array_receipts=receipts,derivative_stages=stages)
    if damage=='value':saved['base']['nominal'][0]=1
    elif damage=='goal':compiled['goal_vectors'][0][0]=1
    elif damage=='context':attempts[-1]['witness']['edges'][0]['output_context']=dict(args['root']['context'],action_history=np.ones((2,10)).tolist())
    elif damage=='index':ready['root_index']=1
    elif damage=='receipt':receipts[-1]['sha256']='f'*64
    elif damage=='stage':stages[-1]['persisted']=False
    elif damage=='missing':receipts.pop()
    elif damage=='name':receipts[-1]['name']='root_1_suffix_39_jacobian'
    if damage:
        with pytest.raises(ValueError):m.validate_array_artifacts(tmp_path,p,prepared_sha,ready,compiled,saved,row)
    else:m.validate_array_artifacts(tmp_path,p,prepared_sha,ready,compiled,saved,row)


def test_immediate_array_progress_precedes_AD_and_solve():
    progress=m.expected_array_progress()
    assert len(progress)==82 and progress[:4]==[(4,0),(8,1),(340,6),(342,7)]
    assert progress[-2:]==[(652,162),(654,163)]


def test_scope_cleanup_failure_does_not_replace_work_primary():
    class Replay:
        _p2_scope=None
        def __setattr__(self,name,value):
            if value is None:raise OSError('scope restore')
            object.__setattr__(self,name,value)
    faults=m.Faults();original=RuntimeError('AD primary')
    with pytest.raises(RuntimeError) as caught:
        with m.scoped(Replay(),dict(phase='AD'),faults):raise original
    assert caught.value is faults.primary is original
    assert faults.records[1]['error']=='scope restore'


def test_no_silent_success_on_missing_roots(tmp_path):
    with pytest.raises(ValueError,match='four complete root'):m.finalized_run(tmp_path,{},lambda *a:None,source_ok,model_ok)
    assert m.read(tmp_path/'report.json')['status']=='incomplete_or_failed'


def test_file_write_primary_survives_file_close_failure(tmp_path,monkeypatch):
    original=OSError('write original');secondary=OSError('close secondary')
    class Stream:
        def write(self,s):raise original
        def close(self):raise secondary
    monkeypatch.setattr(Path,'open',lambda *a,**kw:Stream())
    faults=m.Faults()
    with pytest.raises(OSError) as caught:m.write(tmp_path/'new.json',{})
    assert caught.value is original
    faults.capture('outer',caught.value)
    assert faults.exceptions==[original,secondary]
    assert faults.records[1]['stage']=='write.close_file'


def test_parent_fsync_primary_survives_parent_close_failure(tmp_path,monkeypatch):
    original=OSError('directory fsync');secondary=OSError('directory close');fsync=m.os.fsync
    def sync(fd):
        if fd==998877:raise original
        return fsync(fd)
    def close(fd):raise secondary
    with monkeypatch.context() as patch:
        patch.setattr(m.os,'open',lambda *a,**kw:998877)
        patch.setattr(m.os,'fsync',sync);patch.setattr(m.os,'close',close)
        with pytest.raises(OSError) as caught:m.write(tmp_path/'new.json',{})
    assert caught.value is original
    faults=m.Faults();faults.capture('outer',caught.value)
    assert faults.exceptions==[original,secondary] and (tmp_path/'new.json').exists()


def test_report_builder_failure_leaves_fallback_with_unknown_not_zero_budget(tmp_path,monkeypatch):
    original=RuntimeError('aggregation')
    def build(*a):raise original
    monkeypatch.setattr(m,'build_report',build)
    with pytest.raises(RuntimeError) as caught:m.finalized_run(tmp_path,{},healthy_work,source_ok,model_ok)
    report=m.read(tmp_path/'report.json')
    assert caught.value is original and report['resource_totals'] is None
    assert report['status']=='incomplete_or_failed' and report['faults']['primary']['stage']=='finalization.report_build'
