"""No torch: native-shaped fixtures, actual exclusive Store, full saved reader."""
import copy
import io
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.spectral_reachability import agent_lewm_finalmean_paid_evaluator_20261029_v1 as m


@pytest.fixture
def saved(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:type('Free',(),{'free':10*1024**3})())
    source=tmp_path/'source';source.mkdir();source_inventory={}
    def original(a):
        buf=io.BytesIO();np.save(buf,a,allow_pickle=False);name=f'input_{len(source_inventory)}.npy'
        (source/name).write_bytes(buf.getvalue());rec=dict(file=name,sha256=m.sha(source/name),file_bytes=len(buf.getvalue()),
            nbytes=a.nbytes,shape=list(a.shape),dtype=a.dtype.str)
        source_inventory[name]={k:rec[k] for k in ('sha256','file_bytes')};return rec
    store=m.Store(tmp_path/'results'/(m.NAME+'_fixture'));rows=[];slots=[]
    for root in (0,1):
        base=dict(emb=np.full((1,3,192),root,np.float32),pixels=np.zeros((1,3,3,224,224),np.float32),
            goal=np.zeros((1,1,3,224,224),np.float32),goal_emb=np.full((1,1,192),.5,np.float32),
            warm=np.arange(20,dtype=np.float32).reshape(1,1,2,10),actions=np.arange(20,dtype=np.float32).reshape(1,2,10)/4)
        refs={k:original(a) for k,a in base.items()};slots.append(dict(root=root,refs=refs,inventory=copy.deepcopy(source_inventory)))
        row=dict(root=root,status='paid_finalmean_evaluated',modules=[],layouts={},scores={},inputs={k:store.array(a) for k,a in base.items()})
        actions=np.concatenate((base['warm'],base['actions'][:,None]),axis=2)
        def module(kind,layout,args,out):
            row['modules'].append(dict(index=len(row['modules']),kind=kind,layout=layout,returned=True,
                inputs=[store.array(a) for a in args],output=store.array(out)))
        for layout in m.LAYOUTS:
            pred=np.zeros((1,1,5,192),np.float32);pred[:,0,:3]=base['emb']
            pred[:,:,3]=np.float32(root+.125);pred[:,:,4]=np.float32(root+.25)
            if layout=='segmented':pred[:,:,3:]+=np.float32(1/4096)
            e=dict(attempted=True,returned=True)
            if layout=='whole':
                ae=np.zeros((1,4,192),np.float32);module('action_encoder',layout,[actions[:,0]],ae)
                e['action_prefix']=store.array(actions[:,:,:3])
                for step in (0,1):
                    po=pred[:,0,step:step+3].copy();po[:,-1]=pred[:,0,3+step]
                    module('predictor',layout,[pred[:,0,step:step+3],ae[:,step:step+3]],po)
                    module('pred_proj',layout,[po.reshape(3,192)],po.reshape(3,192).copy())
            else:
                e['full3']=[]
                for step in (0,1):
                    ae=np.zeros((1,3,192),np.float32);module('action_encoder',layout,[actions[:,0,step:step+3]],ae)
                    po=pred[:,0,step:step+3].copy();po[:,-1]=pred[:,0,3+step]
                    module('predictor',layout,[pred[:,0,step:step+3],ae],po)
                    module('pred_proj',layout,[po.reshape(3,192)],po.reshape(3,192).copy());e['full3'].append(store.array(po))
            e['predicted_emb']=store.array(pred);row['layouts'][layout]=e
            cost=((pred[:,:,-1]-base['goal_emb'][:,0,None])**2).sum(axis=-1,dtype=np.float32)
            row['scores'][layout]=dict(attempted=True,returned=True,cost=store.array(cost))
        rows.append(row)
    prepared=dict(slots=slots,upstream=dict(report=str(source/'report.json')),inherited={'costs':'fixture'},expected_model_hash='c'*64)
    status='known2_paid_finalmean_evaluated'
    for event in m.expected_events('a'*64,status):store.event(**event)
    journal=store.path/'journal.jsonl';store.inventory['journal.jsonl']=dict(sha256=m.sha(journal),file_bytes=journal.stat().st_size)
    report=dict(schema=m.NAME+'.report',status=status,prepared='fixture',prepare_sha256='a'*64,protocol=m.protocol(),rows=rows,
        all_predictions_sealed=True,source_verified=True,errors=[],model_before='c'*64,model_after='c'*64,
        inherited=prepared['inherited'],upstream=prepared['upstream'],physical_release=False,scientific_admission=False,
        elapsed_seconds=1.,cuda_peak_allocated=1,cuda_peak_reserved=1,journal_tail=store.tail,artifacts=copy.deepcopy(store.inventory),
        numerics=dict(requested_SDP='MATH',kernel_certificate=False,observed_inference=True,observed_autocast=False,observed_TF32=False))
    store.put('report.json',m.raw(report),final=True);monkeypatch.setattr(m,'verify',lambda *a:prepared)
    return store,report,prepared,source


def test_real_store_full_saved_consumer_nonexact_allowed(saved):
    store,r,_,_=saved
    receipt=m.consume(store.path,m.sha(store.path/'report.json'))
    assert receipt['status']=='known2_paid_finalmean_saved_verified'
    assert receipt['old_slice_mean_evaluated'] is False and receipt['parity_upgraded'] is False
    assert len(receipt['rows'])==2
    for row in receipt['rows']:
        assert row['comparisons']['all_future']['bitwise'] is False
        assert row['comparisons']['final20']['max_ULP']>0
        assert row['comparisons']['all_future']['maxabs']==1/4096
        assert row['returned_mean_evaluated_in_new_job'] is True
    assert receipt['resources']['AE_tokens']==20 and receipt['resources']['predictor']==8
    lines=(store.path/'journal.jsonl').read_bytes().splitlines(keepends=True)
    assert len(lines)==67 and all(line.endswith(b'\n') and not line.endswith(b'\n\n') for line in lines)


@pytest.mark.parametrize('mutation',['delivery','mean','warm','causal','layout','AE','projection','full3','score','missing','count','dtype'])
def test_cellwise_mutations(saved,mutation):
    store,r,p,source=saved;row=r['rows'][0]
    def change(rec,fn):
        a=m.old.load_array(store.path,rec,r['artifacts']);new=store.array(fn(a.copy()))
        r['artifacts'][new['file']]={k:new[k] for k in ('sha256','file_bytes')};return new
    if mutation=='delivery':row['inputs']['emb']=change(row['inputs']['emb'],lambda a:a+1)
    elif mutation=='mean':row['inputs']['actions']=change(row['inputs']['actions'],lambda a:a+1)
    elif mutation=='warm':row['inputs']['warm']=change(row['inputs']['warm'],lambda a:a+1)
    elif mutation=='causal':row['modules'][9]['inputs'][0]=change(row['modules'][9]['inputs'][0],lambda a:a+1)
    elif mutation=='layout':row['modules'][5]['layout']='whole'
    elif mutation=='AE':row['modules'][8]['inputs'][0]=change(row['modules'][8]['inputs'][0],lambda a:a+1)
    elif mutation=='projection':row['modules'][10]['inputs'][0]=change(row['modules'][10]['inputs'][0],lambda a:a+1)
    elif mutation=='full3':row['layouts']['segmented']['full3'][0]=change(row['layouts']['segmented']['full3'][0],lambda a:a+1)
    elif mutation=='score':row['scores']['whole']['cost']=change(row['scores']['whole']['cost'],lambda a:a+1)
    elif mutation=='missing':del row['layouts']['segmented']
    elif mutation=='count':row['modules'].pop()
    else:row['modules'][1]['output']=change(row['modules'][1]['output'],lambda a:a.astype(np.int64))
    with pytest.raises(ValueError):m.validate_row(p['slots'][0],row,
        lambda rec:m.old.load_array(store.path,rec,r['artifacts']),lambda rec:m.old.load_array(source,rec,p['slots'][0]['inventory']))


@pytest.mark.parametrize('mutation',['partial','source','model','seal','memory','inherited','upstream','untracked','blank','hash','sealorder'])
def test_complete_saved_mutations(saved,mutation):
    store,r,_,_=saved
    if mutation=='partial':r['status']='partial_or_failed'
    elif mutation=='source':r['source_verified']=False
    elif mutation=='model':r['model_after']='d'*64
    elif mutation=='seal':r['all_predictions_sealed']=False
    elif mutation=='memory':r['cuda_peak_reserved']=5*1024**3
    elif mutation=='inherited':r['inherited']={}
    elif mutation=='upstream':r['upstream']={}
    elif mutation=='untracked':(store.path/'extra').write_bytes(b'x')
    elif mutation=='hash':r['journal_tail']='e'*64
    else:
        path=store.path/'journal.jsonl';data=path.read_bytes()
        if mutation=='blank':data+=b'\n'
        else:data=data.replace(b'all_predictions_sealed',b'premature_score_before_seal')
        path.write_bytes(data);r['artifacts']['journal.jsonl']=dict(sha256=m.sha(path),file_bytes=path.stat().st_size)
    (store.path/'report.json').write_bytes(m.raw(r))
    with pytest.raises((ValueError,json.JSONDecodeError)):m.consume(store.path,m.sha(store.path/'report.json'))


def test_bitwise_and_ulp_not_tolerance():
    a=np.array([-1.,-0.,0.,1.],np.float32);b=np.nextafter(a,np.float32(np.inf))
    c=m.compare(a,b);assert c['max_ULP']==1 and c['bitwise_different_elements']==4 and c['bitwise'] is False
    z=m.compare(np.array([-0.],np.float32),np.array([0.],np.float32))
    assert z['max_ULP']==0 and z['signed_zero_bit_differences']==1 and z['bitwise'] is False
    assert m.compare(a,a.copy())['bitwise'] is True
    with pytest.raises(ValueError):m.compare(a.astype(np.float64),b)
    with pytest.raises(ValueError):m.compare(np.array([np.nan],np.float32),np.array([0.],np.float32))


def test_budget_schedule_and_seal_order():
    schedule=m.schedule();twice=schedule*2;counts={k:sum(x[0]==k for x in twice) for k in m.KINDS}
    assert counts==dict(action_encoder=6,predictor=8,pred_proj=8)
    assert sum(x[1][0][1] for x in twice if x[0]=='action_encoder')==20
    events=m.expected_events('a','complete');seal=next(i for i,e in enumerate(events) if e['kind']=='all_predictions_sealed')
    assert all(i>seal for i,e in enumerate(events) if e['kind'].startswith('criterion_'))
    assert m.protocol()['output_cap']==512*1024**2 and m.protocol()['final_reserve']==64*1024**2
    assert m.sha(m.old.__file__)==m.OLD_SHA


@pytest.mark.parametrize('kind',['wall','cap','floor'])
def test_store_resource_guards(saved,monkeypatch,kind):
    store=saved[0]
    if kind=='wall':store.started-=m.WALL+1
    elif kind=='cap':store.total=m.CAP-m.RESERVE
    else:monkeypatch.setattr(m.shutil,'disk_usage',lambda p:type('Free',(),{'free':m.FLOOR})())
    with pytest.raises(ValueError):store.put('not_written',b'x')


def test_authority_and_no_tensor_import(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'RECEIPT',tmp_path/'missing.json')
    # Reject missing authority before production verify/model load.
    monkeypatch.setattr(m,'REPORT',tmp_path/'missing_report.json')
    with pytest.raises(ValueError):m.build()
    with pytest.raises(ValueError):m.run('missing','no','missing','wrong')
    result=subprocess.run([sys.executable,'-c',
        'import sys;from experiments.spectral_reachability import agent_lewm_finalmean_paid_evaluator_20261029_v1 as m;'
        'assert "torch" not in sys.modules;m.main()','--help'],capture_output=True,text=True)
    assert result.returncode==0 and 'consume' in result.stdout


class FakeModule:
    def __init__(self):self.pre=[];self.post=[]
    def register_forward_pre_hook(self,fn):
        self.pre.append(fn);return SimpleNamespace(remove=lambda:self.pre.remove(fn))
    def register_forward_hook(self,fn):
        self.post.append(fn);return SimpleNamespace(remove=lambda:self.post.remove(fn))


class FakeTensor:
    def __init__(self,shape):
        self.a=np.zeros(shape,np.float32);self.dtype=np.float32;self.device=SimpleNamespace(type='cuda');self.requires_grad=False;self.shape=shape
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.a


def test_trace_native_shapes_and_cleanup(saved,monkeypatch):
    fake=SimpleNamespace(float32=np.float32,is_inference_mode_enabled=lambda:True,is_grad_enabled=lambda:False,
        is_autocast_enabled=lambda device:False,backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
        cudnn=SimpleNamespace(allow_tf32=False)))
    monkeypatch.setitem(sys.modules,'torch',fake)
    model=SimpleNamespace(**{k:FakeModule() for k in (*m.KINDS,'encoder','projector')})
    row=dict(root=0,modules=[],active_layout='whole');store=saved[0]
    with m.Trace(model,store,row):
        with pytest.raises(ValueError):model.encoder.pre[0](model.encoder,())
        with pytest.raises(ValueError):model.predictor.pre[0](model.predictor,(FakeTensor((1,3,192)),)*2)
        for i,(kind,shapes,outshape) in enumerate(m.schedule()):
            row['active_layout']='whole' if i<5 else 'segmented'
            module=getattr(model,kind);args=tuple(FakeTensor(s) for s in shapes)
            module.pre[0](module,args);module.post[0](module,args,FakeTensor(outshape))
    assert len(row['modules'])==11 and all(e['returned'] for e in row['modules'])
    assert all(not getattr(model,k).pre and not getattr(model,k).post for k in (*m.KINDS,'encoder','projector'))
    with pytest.raises(RuntimeError):
        with m.Trace(model,store,dict(root=0,modules=[],active_layout='whole')):raise RuntimeError('fixture interrupt')
    assert all(not getattr(model,k).pre and not getattr(model,k).post for k in (*m.KINDS,'encoder','projector'))
    with m.NoNeural(model):
        for k in (*m.KINDS,'encoder','projector'):
            with pytest.raises(ValueError):getattr(model,k).pre[0](getattr(model,k),())
    assert all(not getattr(model,k).pre for k in (*m.KINDS,'encoder','projector'))


def test_prepare_extracts_only_real_delivered_mean(saved,monkeypatch,tmp_path):
    _,newreport,p,source=saved
    upstream_rows=[];tasks=[]
    for slot,row in zip(p['slots'],newreport['rows']):
        upstream_rows.append(dict(root=slot['root'],status='normalized_mean_sealed',base_inputs={k:v for k,v in slot['refs'].items() if k not in ('actions','warm')},
            warm=slot['refs']['warm'],actions=slot['refs']['actions'],mean=slot['refs']['actions'],callbacks=[{'mean':slot['refs']['actions']}]))
        original=lambda key:m.old.load_array(source,slot['refs'][key],slot['inventory'])
        tasks.append(dict(source_slot=slot['root'],emb=original('emb')[0].tolist(),warm=original('warm')[0,0].tolist(),goal_emb=original('goal_emb')[0,0].tolist()))
    original_report=dict(schema=m.old.NAME+'.report',status='all_known2_native_normalized_CEM_recorded',prepare_sha256=m.PREPARE_SHA,
        errors=[],source_verified=True,protocol=m.old.protocol(),prepared='fake_original_prepare',model_before='c'*64,model_after='c'*64,
        rows=upstream_rows,artifacts={k:v for slot in p['slots'] for k,v in slot['inventory'].items()})
    receipt=dict(schema=m.old.NAME+'.consume',status='all_known2_native_normalized_CEM_saved_verified',report_sha256=m.REPORT_SHA,
        inherited={'encoding':'old'},per_root_resources=m.old.protocol()['per_root'],total_resources={k:2*v for k,v in m.old.protocol()['per_root'].items()},
        rows=[dict(root=i,iterations=30,candidate_count=9000,returned_mean_evaluated=False,raw_actions=None,physical_release=False) for i in (0,1)])
    original_prepare=dict(inherited=receipt['inherited'],expected_model_hash='c'*64,slots=tasks,sources={},policy='original_policy')
    monkeypatch.setattr(m,'REPORT',source/'report.json');monkeypatch.setattr(m,'RECEIPT',tmp_path/'authority.json')
    monkeypatch.setattr(m,'read',lambda path,h:original_report if path==m.REPORT else receipt)
    monkeypatch.setattr(m,'verify_upstream',lambda *args:original_prepare)
    # New source/doc hashes are synthetic here, while actual source NPYs are still read and SHA-checked by old.load_array.
    # build final closure requires all bound SHA; fixture reads retain genuine hashing via old.sha.
    refs=[rec for slot in p['slots'] for rec in slot['refs'].values()]
    known={str(source/rec['file']):rec['sha256'] for rec in refs}
    known.update({str(m.REPORT):m.REPORT_SHA,str(m.RECEIPT):m.RECEIPT_SHA,'fake_original_prepare':m.PREPARE_SHA})
    monkeypatch.setattr(m,'sha',lambda path:m.OLD_SHA if str(path)==str(m.old.__file__) else known.get(str(path),'f'*64))
    result=m.build();assert [s['root'] for s in result['slots']]==[0,1]
    assert result['inherited']['CEM']['predictor']==120 and result['protocol']['old_slice_mean_evaluated'] is False
    receipt['rows'][1]['returned_mean_evaluated']=True
    with pytest.raises(ValueError):m.build()
