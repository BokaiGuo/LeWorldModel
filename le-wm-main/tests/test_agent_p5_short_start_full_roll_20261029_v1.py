"""RAM and temporary saved fixtures, no model/torch/physics production.

Full656 descriptor schedule is tested. File integration reduces only this NEW
module's N to1 for lightweight native-shaped fixtures; CLI has no such switch.
"""
from copy import deepcopy
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_short_start_full_roll_20261029_v1 as m
from test_agent_p5_short_start_native_abi_20261029_v1 import fixture as original_fixture
from test_agent_p5_short_start_cuda_probe_20261029_v1 import FixtureBackend,Tensor


def array(path,value):
    b=io.BytesIO();np.save(b,value,allow_pickle=False);path.write_bytes(b.getvalue())
    return dict(file=path.name,file_bytes=path.stat().st_size,sha256=m.sha(path),dtype='float32',shape=list(value.shape),nbytes=value.nbytes)


@pytest.fixture
def case(tmp_path,monkeypatch):
    root=tmp_path/'inner';(root/'results').mkdir(parents=True);source=root/'original';source.mkdir()
    r,p,c=original_fixture(source);record=m.abi.resolve_actual0(r,p,c,source,0)
    labelref=dict(directory=str(source),array=array(source/'labels.npy',np.full((2,256,384),2,np.float32)),report_json_pointer='/fixture/labels')
    monkeypatch.setattr(m,'N',1);monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'pair_schedule',lambda:[])
    a=dict(schema=m.NAME+'.authority',rows=[dict(index=0,record=record,labels=[record['refs']['encoder_patch'],labelref])],
        bindings={str(q):m.sha(q) for q in source.iterdir()},protocol=m.protocol(),bridge_prepared='fixture_only',bridge_prepare_sha256='a'*64)
    monkeypatch.setattr(m,'authority',lambda:deepcopy(a))
    prepared=root/'results'/(m.NAME+'_prepare_fixture.json');meta=m.prepare(prepared)
    manifest=m.verify(prepared,meta['sha256'])
    return NS(root=root,source=source,prepared=prepared,expected=meta['sha256'],m=manifest,
        out=root/'results'/(m.NAME+'_fixture'),authority=a)


def perform(case,backend=FixtureBackend):
    result=m.execute(case.m,case.prepared,case.expected,case.out,backend_factory=backend)
    return result,m.read(case.out/'report.json',result['sha256'])


def test_164_656_1476_exact_native_descriptor_schedule():
    assert m.N==164 and len(m.pair_schedule())==80 and m.units()==dict(api=656,batch_samples=656,frame_samples=1476)
    assert m.protocol()['native_IO_bytes']==1209198240 and m.protocol()['bank_tensor_bytes']==67207200
    assert m.protocol()['label_tensor_bytes']==322437120
    guard=m.Meter()
    for i in range(656):
        t=(1,2,3,3)[i%4]
        w=guard.before('predictor',[(1,t,1,16,16,384),(1,t,10),(1,t,256,16)],['cuda:0']*3,['float32']*3,[False]*3,True,True,True)
        assert w==dict(stage=t,units=dict(api=1,batch_samples=1,frame_samples=t))
        guard.after([(1,t,256,384),None,(1,t,256,16)],['cuda:0']*2,['float32']*2)
    assert guard.attempted==guard.returned==m.units()
    with pytest.raises(ValueError):guard.before('predictor',[],[],[],[],True,True,True)


def test_full164_missing_656th_prediction_rejects_before_any_label_or_seal_read(monkeypatch):
    rows=[dict(stages=[dict(prediction={'fixture':'not_a_tensor'}) for j in range(4)]) for i in range(164)]
    rows[-1]['stages'][-1]['prediction']=None
    def forbidden(*args,**kwargs):raise AssertionError('must not read even the seal before completeness guard')
    monkeypatch.setattr(m,'read',forbidden)
    with pytest.raises(ValueError):list(m.labels_after_seal({},rows,NS(path=Path('/unused')),{},'0'*64))


def test_scalar_MSE_selects_only_each_new_last_token_with_common_labels():
    labels=[np.zeros((3,256,384),np.float32),np.zeros((2,256,384),np.float32)]
    labels[0][0]=999  # actual0 in the original batch is not a successor label.
    predictions=[]
    for j,t in enumerate((1,2,3,3)):
        visual=np.full((1,t,256,384),100,np.float32);visual[:,-1]=j+1
        predictions.append((visual,None,np.zeros((1,t,256,16),np.float32)))
    out=m.scores(predictions,labels,np.zeros((1,1,1,16,16,384),np.float32))
    assert out==dict(MSE=[1.,4.,9.,16.],actual0_persistence_MSE=[0.]*4)


def test_native_shaped_prepare_verify_four_rolls_store_and_consume(case):
    result,r=perform(case)
    assert r['status']=='all164_short_start_causal_roll_recorded' and r['tensor_bytes']==9339240
    assert len(r['rows'])==1 and len(r['calls'])==4 and len(list(case.out.glob('tensor_*.npy')))==22
    assert [c['T'] for c in r['calls']]==[1,2,3,3]
    assert r['counts']==dict(attempted=m.units(),returned=m.units(),validated=m.units())
    assert r['prefix_checks']==[dict(index=0,reference_index=0,actual0_equal=[True]*3,warmup_packets_equal=True,
        pred5_10_equal=[dict(visual=True,proprio=True)]*2)]
    assert m.consume(case.out,result['sha256'])['status']=='all164_short_start_causal_roll_saved_verified'
    events=list(m.codec.journal(case.out/'journal.jsonl'));kinds=[e['kind'] for e in events]
    assert kinds.index('all_predictions_sealed')>max(i for i,k in enumerate(kinds) if k=='outputs_saved')
    assert kinds.index('label_read_intent')>kinds.index('all_predictions_sealed')
    assert not r['P1_admission'] and not r['P5_admission']


def test_four_stages_causal_last_token_only_and_action_sliding(case):
    b=m.bank_read(case.m);args,acts=m.bank_inputs(case.m,b,0);initial=tuple(x.copy() for x in args)
    histories=[]
    for j,t in enumerate((1,2,3,3)):
        assert args[0].shape==(1,t,1,16,16,384) and m.exact(args[1],acts[:,m.PACKETS[j]])
        histories.append(tuple(x.copy() for x in args))
        out=(np.full((1,t,256,384),10+j,np.float32),None,np.full((1,t,256,16),20+j,np.float32))
        if j<3:args=m.next_inputs(j,args,out,acts)
    assert m.exact(histories[1][0][:,:1],initial[0]) and m.exact(histories[2][2][:,:1],initial[2])
    assert [float(histories[3][0][0,k,0,0,0,0]) for k in range(3)]==[10,11,12]
    assert [float(histories[3][2][0,k,0,0]) for k in range(3)]==[20,21,22]
    assert not m.exact(histories[3][0][:,:1],initial[0])
    with pytest.raises(ValueError):m.next_inputs(3,args,out,acts)


@pytest.mark.parametrize('mode',('dispatch','return'))
def test_partial_fourth_stage_retains_denominators_and_forbids_labels(case,mode):
    class Broken(FixtureBackend):
        fail_stage=4 if mode=='dispatch' else None
        bad_stage=4 if mode=='return' else None
    result,r=perform(case,Broken)
    assert r['status']=='partial_or_failed' and len(r['rows'][0]['stages'])==4 and r['rows'][0]['stages'][3]['prediction'] is None
    assert r['rows'][0]['labels'] is None and r['rows'][0]['score'] is None
    assert r['counts']['attempted']['api']==4 and r['counts']['validated']['api']==3
    assert r['counts']['returned']['api']==(3 if mode=='dispatch' else 4)
    assert 'label_read_intent' not in [e['kind'] for e in m.codec.journal(case.out/'journal.jsonl')]
    assert m.consume(case.out,result['sha256'])['status']=='blocked_partial_roll'


def test_label_guard_rejects_unsealed_or_missing_branch_without_loading(case):
    rows=m.empty_rows(case.m);s=m.Store(case.out)
    with pytest.raises(ValueError):list(m.labels_after_seal(case.m,rows,s,dict(file='absent',sha256='0'*64),case.expected))
    assert not (case.out/'journal.jsonl').exists()


@pytest.mark.parametrize('bad',('CPU','encoder','full3','dtype','grad','frozen','device_return'))
def test_new_native_hooks_fail_closed(case,bad):
    store=m.Store(case.out);trace=m.Trace(store);backend=FixtureBackend(case.m,trace)
    args,_=m.bank_inputs(case.m,m.bank_read(case.m),0);xs=[Tensor(x) for x in args];model=backend.models['predictor']
    if bad=='CPU':xs[0].device='cpu'
    elif bad=='encoder':model=backend.models['proprio_encoder']
    elif bad=='full3':xs[0].shape=(1,3,1,16,16,384)
    elif bad=='dtype':xs[0].dtype='float64'
    elif bad=='grad':xs[0].requires_grad=True
    elif bad=='frozen':xs[0].requires_grad=True
    else:model.bad=True
    with pytest.raises(ValueError):model(*xs)
    assert trace.guard.attempted['api']==int(bad=='device_return')
    assert trace.observed_returns['api']==int(bad=='device_return') and trace.guard.returned['api']==0
    assert not trace.close()


def test_real_store_two_events_exactly_one_LF_hashchain_and_caps(case):
    s=m.Store(case.out);s.emit('first',n=1);s.emit('second',n=2)
    body=(case.out/'journal.jsonl').read_bytes()
    assert body.count(b'\n')==2 and not body.endswith(b'\n\n')
    assert [e['kind'] for e in m.codec.journal(case.out/'journal.jsonl')]==['first','second']
    assert s.artifacts['journal.jsonl']['sha256']==m.sha(case.out/'journal.jsonl')
    with pytest.raises(ValueError):s.put('../escape',b'bad')
    with pytest.raises(ValueError):s.array(np.zeros(1,np.float64))
    with pytest.raises(ValueError):s.array(np.full(1,np.nan,np.float32))
    with pytest.raises(ValueError):s.check(m.CAP)
    with pytest.raises(ValueError):m.Store(case.out)
    (case.out/'journal.jsonl').write_bytes(body+b'\n')
    with pytest.raises(ValueError):list(m.codec.journal(case.out/'journal.jsonl'))


@pytest.mark.parametrize('fault',('bank','actions','source','trailer','extra'))
def test_input_bank_verify_refuses_wrong_identity_or_bytes(case,fault):
    b=m.bank_read(case.m);d=Path(case.m['bank']['directory'])
    if fault in ('bank','actions'):
        rec=b['rows'][0]['inputs'][0] if fault=='bank' else b['rows'][0]['packets'];q=d/rec['file'];q.write_bytes(q.read_bytes()+b'\n')
    elif fault=='source':
        ref=case.authority['rows'][0]['record']['refs']['visual'];q=Path(ref['directory'])/ref['array']['file'];q.write_bytes(b'changed')
    elif fault=='extra':(d/'extra.npy').write_bytes(b'unknown')
    else:
        ref=deepcopy(case.authority['rows'][0]['record']['refs']['visual']);q=Path(ref['directory'])/ref['array']['file'];q.write_bytes(q.read_bytes()+b'\n')
        ref['array'].update(sha256=m.sha(q),file_bytes=q.stat().st_size)
        with pytest.raises(ValueError):m.first_slice(ref)
        return
    with pytest.raises(ValueError):m.verify(case.prepared,case.expected)


@pytest.mark.parametrize('fault',('artifact','extra','model','counts','causal','time','metadata','units','label','score','seal','fused','source','pair','prefix'))
def test_complete_saved_consumer_refuses_tamper(case,fault):
    result,r=perform(case)
    if fault=='artifact':
        q=case.out/'tensor_00000.npy';q.write_bytes(q.read_bytes()+b'\n')
    elif fault=='extra':(case.out/'extra').write_bytes(b'wrong')
    elif fault=='model':r['model_after']={}
    elif fault=='counts':r['counts']['attempted']['api']=5
    elif fault=='causal':r['calls'][3]['inputs']=r['calls'][2]['inputs']
    elif fault=='time':r['rows'][0]['stages'][3]['frames']=[0,5,10]
    elif fault=='metadata':r['calls'][3]['native_outputs'][0]['device']='cpu'
    elif fault=='units':r['calls'][0]['units']['frame_samples']=3
    elif fault=='label':r['rows'][0]['labels'].reverse()
    elif fault=='score':r['rows'][0]['score']['MSE'][3]=0.
    elif fault=='seal':r['prediction_seal']['sha256']='0'*64
    elif fault=='fused':r['attention_ops']={'aten._scaled_dot_product_flash_attention.default':1}
    elif fault=='source':r['imported_sources']={'wrong':'/nonexistent'}
    elif fault=='prefix':r['prefix_checks'][0]['warmup_packets_equal']=False
    else:r['pairs']=[{}]
    (case.out/'report.json').write_bytes(m.raw(r))
    with pytest.raises((ValueError,FileNotFoundError)):m.consume(case.out,m.sha(case.out/'report.json'))


def test_pair_zero_null_negative_and_central_difference_arithmetic(monkeypatch):
    spec=dict(slot=0,root_slot=0,condition='off',h=.01,coord=0,minus_index=0,plus_index=1)
    monkeypatch.setattr(m,'pair_schedule',lambda:[spec]);arrays={};rows=[]
    def save(a):
        k=str(len(arrays));arrays[k]=a;return k
    for i in range(2):
        rows.append(dict(index=i,score=dict(MSE=[0.]*4),labels=[save(np.zeros((3,1,1))),save(np.zeros((2,1,1)))],
            stages=[dict(prediction={'$tuple':[{'$array':save(np.zeros((1,1,1,1)))},None,None]}) for j in range(4)]))
    result=m.pairs(rows,arrays.__getitem__)[0]
    assert len(result['times'])==4 and all(t['cosine'] is None for t in result['times'])
    arrays[rows[1]['stages'][3]['prediction']['$tuple'][0]['$array']][:]=1
    arrays[rows[1]['labels'][1]][1]=-1
    t=m.pairs(rows,arrays.__getitem__)[0]['times'][3]
    assert t['cosine']==-1 and t['dot']==-1 and t['predicted_central_difference_l2']==50 and t['observed_central_difference_l2']==50
    rows[1]['score']=None;assert m.pairs(rows,arrays.__getitem__)[0]['missing']==[1]


def test_run_clock_covers_verify_and_requires_MAIN(case,monkeypatch):
    monkeypatch.delenv('PYTHONPATH',raising=False);calls=[]
    monkeypatch.setattr(m.time,'monotonic',lambda:42.)
    monkeypatch.setattr(m,'verify',lambda *args:calls.append('verify') or case.m)
    monkeypatch.setattr(m,'execute',lambda *args,**kw:calls.append(kw['started']) or {'status':'fixture'})
    m.run('unused','0'*64,'unused',m.AUTH);assert calls==['verify',42.]
    with pytest.raises(ValueError):m.run('unused','0'*64,'unused','wrong')


def test_no_neural_or_physics_import_and_complete_CLI():
    code='import sys; from experiments.spectral_reachability import '+m.NAME+'; assert "torch" not in sys.modules; assert "pymunk" not in sys.modules'
    subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,capture_output=True,check=True)
    r=subprocess.run([sys.executable,'-m','experiments.spectral_reachability.'+m.NAME,'--help'],cwd=m.ROOT,capture_output=True,text=True,check=True)
    assert '{prepare,verify,run,consume}' in r.stdout


def test_protocol_and_probe_bindings_are_exact():
    assert m.PROBE_SHAS['strict']=='1e376b1f06ca3b04ad71822b76178308dc4ad3d700641c97726b1c7f94c5d346'
    assert m.PROBE_SHAS['prepared']=='1fe5f72e9234cfa2360d5a837a68dfbc56b1a94b37fb82775f0be90f63fd8e2c'
    assert m.sha(m.PROTOCOL_PATH)==m.PROTOCOL_SHA and m.sha(m.probe.__file__)==m.PROBE_SOURCE
    assert m.protocol()['new_encoder']==m.protocol()['new_physics']==0 and not m.protocol()['P1_admission']
