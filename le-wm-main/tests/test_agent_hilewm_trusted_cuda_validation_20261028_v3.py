"""RAM-only shape/layout callbacks and full saved mutation checks; no torch."""
from contextlib import nullcontext
import json
import io
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments.spectral_reachability import agent_hilewm_trusted_cuda_validation_20261028_v3 as m
import test_agent_hilewm_trusted_cuda_validation_20261028_v2 as v2test


class View(v2test.View):
    @property
    def ndim(self):return self.a.ndim
    def __getitem__(self,k):return View(self.a[k])
    def reshape(self,*s):return View(self.a.reshape(*s))
    def expand(self,*s):return View(np.broadcast_to(self.a,tuple(d if n==-1 else n for d,n in zip(self.shape,s))))
    def __sub__(self,b):return View(self.a-b.a)
    def square(self):return View(self.a*self.a)
    def sum(self,dim):return View(self.a.sum(axis=dim))


OPS=NS(stack=lambda xs,dim:View(np.stack([x.a for x in xs],axis=dim)),
    cat=lambda xs,dim:View(np.concatenate([x.a for x in xs],axis=dim)))


@pytest.mark.parametrize('steps',[4,5])
def test_correct_reference_own_append_stack_layout_no_neural(steps):
    z=View(m.old.array_input([2,192],3));a=View(m.old.array_input([2,2,steps,32],10));calls=[]
    def high(z,a):
        calls.append((m.prior.layout(z),m.prior.layout(a),a.a.copy()));return View(np.zeros(z.shape,dtype='<f4'))
    out=m.reference_rollout(OPS,{'high_predictor.pos_embedding':NS(shape=(1,4,192))},{},z,a,predict=high)
    assert out.shape==(2,2,steps,192)
    for k,(zl,al,values) in enumerate(calls):
        c=min(4,k+1);expected=np.stack([a.a[:,:,j].reshape(4,32) for j in range(k+1)],axis=1)[:,-c:]
        assert values.tobytes()==expected.tobytes() and al==m.tail_layout(4,k+1,32,c,'cuda:0')
        assert zl==m.tail_layout(4,k+1,192,c,'cuda:0')
    assert calls[0][1]['stride']==[32,32,1]
    if steps==5:assert calls[-1][1]['storage_offset']==32


def test_reference_cost_original_sum_not_mean():
    z=View(np.zeros((2,192),'<f4'));goal=View(np.ones((2,192),'<f4'));a=View(np.zeros((2,2,2,64),'<f4'));calls=[];buffers=[]
    def high(z,a):calls.append(1);return View(np.zeros(z.shape,'<f4'))
    q=m.reference_cost(OPS,{'high_predictor.pos_embedding':NS(shape=(1,4,192))},{},z,goal,a,predict=high,record=buffers.append)
    assert len(calls)==4 and np.array_equal(q.a,np.full((2,2),192.,'<f4'))
    assert buffers[0]['square'].shape==(2,2,192)


def authority_stub():
    # Existing metadata, NOT a reconstructed production receipt in this fixture.
    parent=m.loader.json_read(m.prior.PREPARE,m.prior.PREPARE_SHA);accepted=m.loader.json_read(m.RECEIPT,m.RECEIPT_SHA)
    return dict(manifest_path=str(m.MANIFEST),manifest_sha256=m.MANIFEST_SHA,report_path=str(m.REPORT),report_sha256=m.REPORT_SHA,
        receipt_path=str(m.RECEIPT),receipt_sha256=m.RECEIPT_SHA,full_strict_rebuilt_here=True,
        strict_receipt=accepted,source_evidence=m.prior.source_evidence()),parent


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'upstream',authority_stub)  # New v3 boundary ONLY.
    monkeypatch.setattr(m,'scoped',lambda p:Path(p))
    return m.prepare(tmp_path/'prepare',main_authorized=True)


def test_saved_prepare_authority_no_torch(prepared):
    p=m.verify(prepared['path'],prepared['sha256']);assert p['upstream']['receipt_sha256']==m.RECEIPT_SHA
    assert p['policy']['cases']==m.old.policy()['cases'] and p['policy']['high_predict_calls_total']==22
    assert not p['legacy_v1_released'] and 'torch' not in sys.modules


@pytest.mark.parametrize('key',['policy','own_sha256','dependency_sha256','upstream','export_receipt_sha256','legacy_v1_released'])
def test_manifest_mutations_failclosed(prepared,tmp_path,key):
    p=json.loads(Path(prepared['path']).read_text());p[key]={};path=tmp_path/'mutated.json';payload=m.old.encoded(p);path.write_bytes(payload)
    with pytest.raises(ValueError):m.verify(path,m.loader.sha(payload))


def test_authorization_env_before_io(monkeypatch):
    monkeypatch.setattr(m,'build',lambda:pytest.fail('unauthorized read'))
    with pytest.raises(ValueError,match='MAIN'):m.prepare('unused')
    with pytest.raises(ValueError,match='MAIN'):m.run_cuda(None,None,None)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    with pytest.raises(ValueError,match='CUBLAS'):m.run_cuda(None,None,None,main_authorized=True)


def expected_construction(p):
    _,spec,_,export=m.loader.authorities(p['export_report'],p['export_report_sha256'],p['export_receipt'],p['export_receipt_sha256'])
    expected={e['key']:row['array']['value_sha256'] for row,e in zip(export['rows'],spec['entries']) if row['array'] is not None}
    return dict(schema=m.loader.TAG+'.construction',status='owned_tensor_graph_constructed_not_native_forward_verified',
        own_sha256={str(m.ROOT/k):v for k,v in m.old.PINS.items() if 'trusted_loader' in k},ir_sha256=m.loader.IR_SHA,
        tensor_value_hashes=expected,tensor_copy_attempts=425,tensor_copy_returns=425,state_key_count=425,module_count=531,entry_count=491,
        all_parameters_frozen=True,forward_calls=0,baseline_runnable=False,export_report_sha256=m.loader.EXPORT_SHA,
        export_receipt_sha256=p['export_receipt_sha256'],device='cuda:0',torch='RAM_NOT_MODEL_EXECUTION')


class Hook:
    def __init__(self,root,name):self.root,self.name=root,name;self.pre=[];self.post=[]
    def register_forward_pre_hook(self,fn):
        self.pre.append(fn);return NS(remove=lambda:self.root.removed.append(self.name+'_pre'))
    def register_forward_hook(self,fn):
        self.post.append(fn);return NS(remove=lambda:self.root.removed.append(self.name+'_post'))


class Root:
    """No model/weights/neural operators. Zero tensors and actual hook protocol."""
    def __init__(self,fail):
        self.removed=[];self.fail=fail
        self.macro_to_condition=Hook(self,'macro');self.high_predictor=Hook(self,'high');self.high_pred_proj=Hook(self,'proj')
    def parameters(self):return iter([NS(device='cuda:0')])
    def state_dict(self):return {'high_predictor.pos_embedding':NS(shape=(1,4,192))}
    def predict_high(self,z,a):
        for fn in self.macro_to_condition.pre:fn(None,(a,))
        condition=View(np.zeros(z.shape,'<f4'))
        for fn in self.high_predictor.pre:fn(None,(z,condition))
        if self.fail:raise ValueError('RAM injected first high failure')
        y=View(np.zeros(z.shape,'<f4'))
        for fn in self.high_predictor.post:fn(None,(z,condition),y)
        flat=y.reshape(-1,192);out=View(flat.a.copy())
        for fn in self.high_pred_proj.post:fn(None,(flat,),out)
        return out.reshape(*z.shape)
    def rollout_high(self,z,a):
        # Independent synthetic native assembly; NOT production ref reuse.
        b,s,n,d=a.shape;history=np.broadcast_to(z.a[:,None] if z.ndim==2 else z.a,(b,s,192)).reshape(b*s,1,192)
        packets=[];outputs=[]
        for k in range(n):
            packets.append(a.a[:,:,k].reshape(b*s,d));seq=np.stack(packets,axis=1);c=min(4,k+1)
            y=self.predict_high(View(history[:,-c:]),View(seq[:,-c:]));nxt=y.a[:,-1:]
            history=np.concatenate((history,nxt),axis=1);outputs.append(nxt[:,0])
        return View(np.stack(outputs,axis=1).reshape(b,s,n,192))
    def get_cost(self,info,a):
        b,s,h,w=a.shape;y=self.rollout_high(info['z_init'],a.reshape(b,s,h*(w//32),32))
        return (y[:,:,-1]-info['z_goal'][:,None].expand(-1,s,-1)).square().sum(dim=-1)


@pytest.fixture
def bundle(prepared,tmp_path,monkeypatch):
    p=m.verify(prepared['path'],prepared['sha256']);construction=expected_construction(p);root=Root(False)
    fake=NS(__version__='RAM_NOT_MODEL_EXECUTION',version=NS(cuda='RAM'),cuda=NS(is_available=lambda:True,synchronize=lambda:None,get_device_name=lambda d:'RAM'),
        use_deterministic_algorithms=lambda x:None,are_deterministic_algorithms_enabled=lambda:True,is_autocast_enabled=lambda device:False,
        backends=NS(cuda=NS(matmul=NS(allow_tf32=None)),cudnn=NS(allow_tf32=None)),from_numpy=View,inference_mode=nullcontext,
        stack=OPS.stack,cat=OPS.cat)
    # Proxy ONLY new-module bindings; no monkeypatch of old module globals.
    proxy=NS(**vars(m.old));proxy.weights=lambda r:construction['tensor_value_hashes']
    def ref(t,state,graph,z,a,meter):
        meter.extend([dict(phase=v,batch=z.shape[0],context=z.shape[1]) for v in ('attempt','return')]);return View(np.zeros(z.shape,'<f4'))
    proxy.reference_high=ref;monkeypatch.setattr(m,'old',proxy)
    loadproxy=NS(**vars(m.loader));loadproxy.construct_cuda=lambda *a,**k:(root,construction);monkeypatch.setattr(m,'loader',loadproxy)
    monkeypatch.setitem(sys.modules,'torch',fake);monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    output=tmp_path/'run';r=m.run_cuda(prepared['path'],prepared['sha256'],output,main_authorized=True)
    assert len(root.removed)==4
    return prepared,output,r,root


def test_meaningful_real_driver_dispatch_callbacks_and_full_consume(bundle):
    p,d,r,root=bundle;receipt=m.consume(p['path'],p['sha256'],r['path'],r['sha256'])
    assert receipt['rows_verified']==4 and receipt['high_predict_calls_total']==22
    report=json.loads((d/'report.json').read_text())
    assert len(report['ledger'])==16 and len(report['journal_events'])==60
    assert len(report['high_predictor_meters']['owned'])==22
    assert not receipt['official_parity'] and not receipt['baseline_runnable']


@pytest.mark.parametrize('bad',['case','partial','input','array','stride','offset','history','cost','cost_goal','weights','meter','ledger','journal','hook','precision','claim','inventory'])
def test_full_saved_consumer_mutation_rejection(bundle,bad):
    p,d,r,_=bundle;report=json.loads((d/'report.json').read_text())
    if bad=='case':report['rows'].reverse()
    elif bad=='partial':report['rows'][2]['status']='failed_partial'
    elif bad=='input':report['rows'][0]['inputs']['z']['array']['shape']=[1,192]
    elif bad=='array':
        descriptor=report['rows'][0]['owned']['array'];(d/descriptor['file']).write_bytes(b'corrupt')
    elif bad=='stride':report['rows'][2]['traces']['reference'][0]['actions']['layout']['stride'][0]=160
    elif bad=='offset':report['rows'][2]['traces']['reference'][-1]['actions']['layout']['storage_offset']=0
    elif bad=='history':report['rows'][2]['traces']['reference'][1]['z']['layout']['stride'][0]=192
    elif bad=='cost':report['rows'][3]['cost_buffers']['value']['array']['value_sha256']='0'*64
    elif bad=='cost_goal':report['rows'][3]['cost_buffers']['goal']['array']['value_sha256']='0'*64
    elif bad=='weights':report['model_after']={}
    elif bad=='meter':report['high_predictor_meters']['reference'].pop()
    elif bad=='ledger':report['ledger'].pop()
    elif bad=='journal':report['journal_events'][1],report['journal_events'][2]=report['journal_events'][2],report['journal_events'][1]
    elif bad=='hook':report['hooks_removed']=2
    elif bad=='precision':report['runtime']['tf32_matmul']=True
    elif bad=='claim':report['legacy_v1_released']=True
    else:(d/'unaccounted').write_text('extra')
    payload=m.old.encoded(report);(d/'report.json').write_bytes(payload)
    with pytest.raises((ValueError,KeyError)):m.consume(p['path'],p['sha256'],d/'report.json',m.loader.sha(payload))


def test_partial_first_cause_stops_keeps_all4_denominators(bundle,monkeypatch,tmp_path):
    p,_,_,_=bundle;root=Root(True);construction=expected_construction(m.verify(p['path'],p['sha256']))
    proxy=NS(**vars(m.loader));proxy.construct_cuda=lambda *a,**k:(root,construction);monkeypatch.setattr(m,'loader',proxy)
    d=tmp_path/'failure'
    with pytest.raises(ValueError,match='partial preserved'):m.run_cuda(p['path'],p['sha256'],d,main_authorized=True)
    report=json.loads((d/'report.json').read_text())
    assert report['errors'][0]['message']=='RAM injected first high failure'
    assert len(report['ledger'])==1 and len(report['rows'])==4
    assert [r['status'] for r in report['rows']]==['failed_partial','not_started','not_started','not_started']
    assert report['pending_owned'] is not None and len(root.removed)==4
    with pytest.raises(ValueError):m.consume(p['path'],p['sha256'],d/'report.json',m.loader.sha((d/'report.json').read_bytes()))


def test_exact_signed_zero_still_not_tolerance():
    a=np.array([0.],'<f4');b=np.array([-0.],'<f4');c=m.prior.comparison(a,b)
    assert c['max_abs_error']==0 and not c['exact_equal']


def test_independent_cost_reduction_bound_normal_zero_subnormal_and_corruption():
    for square in (np.zeros((2,2,192),'<f4'),np.full((2,2,192),2.**-149,'<f4'),
                   np.arange(768,dtype='<f4').reshape(2,2,192)/128):
        value=square.sum(axis=-1);m.validate_cost_reduction(square,value)
        with pytest.raises(ValueError,match='reduction bound'):
            m.validate_cost_reduction(square,value+np.float32(1))


def test_coherent_cost_output_corruption_still_rejected_by_arithmetic(bundle):
    p,d,r,_=bundle;report=json.loads((d/'report.json').read_text());row=report['rows'][3]
    value=m.old.read_array(d,row['owned']['array'])+np.float32(1)
    # All three output files/descriptors and the exact route comparison agree;
    # untouched terminal/residual/square still prove this is not their sum.
    for ref in (row['owned'],row['reference'],row['cost_buffers']['value']):
        stream=io.BytesIO();np.save(stream,value,allow_pickle=False);raw=stream.getvalue()
        desc=ref['array'];(d/desc['file']).write_bytes(raw)
        desc.update(file_sha256=m.loader.sha(raw),file_bytes=len(raw),value_sha256=m.loader.sha(value.tobytes()))
    row['comparison']=m.prior.comparison(value,value)
    raw=m.old.encoded(report);(d/'report.json').write_bytes(raw)
    with pytest.raises(ValueError,match='independent source cost float32 reduction bound'):
        m.consume(p['path'],p['sha256'],d/'report.json',m.loader.sha(raw))


def test_real_saved_v2_authority_rebuild_no_neural_no_production_prepare():
    authority,parent=m.upstream()
    assert authority['full_strict_rebuilt_here'] is True
    assert authority['receipt_sha256']==m.RECEIPT_SHA and authority['report_sha256']==m.REPORT_SHA
    assert authority['strict_receipt']['status']=='all_saved_diagnostic_rows_verified_not_release'
    assert authority['source_evidence']['legacy_reference_is_layout_faithful'] is False
    assert parent['export_report_sha256']==m.loader.EXPORT_SHA and 'torch' not in sys.modules


def test_actual_receipt_hash_tamper_rejected_new_boundary_only(tmp_path,monkeypatch):
    receipt=tmp_path/'receipt.json';receipt.write_bytes(m.loader.bytes_read(m.RECEIPT,m.loader.JSON_CAP)+b' ')
    monkeypatch.setattr(m,'RECEIPT',receipt)
    with pytest.raises(ValueError):m.upstream()


def test_no_flat_results_output_and_exclusive_writer(tmp_path):
    with pytest.raises(ValueError):m.scoped(m.ROOT/'results')
    f=tmp_path/'receipt.json';m.old.write_new(f,b'{}')
    with pytest.raises(FileExistsError):m.old.write_new(f,b'{}')
