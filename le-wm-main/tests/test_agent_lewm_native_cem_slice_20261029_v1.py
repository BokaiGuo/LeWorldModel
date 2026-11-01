"""RAM/filesystem fixtures only: never import torch or execute native solver."""
import ast
import copy
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from experiments.spectral_reachability import agent_lewm_native_cem_slice_20261029_v1 as m


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    monkeypatch.setattr(m,'SAMPLES',4);monkeypatch.setattr(m,'STEPS',2);monkeypatch.setattr(m,'ELITE',2)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p: type('Space',(),{'free':10*1024**3})())
    rgb=np.arange(224*224*3,dtype=np.uint8).reshape(224,224,3)
    packet=dict(rgb_history=np.stack([rgb]*3),goal_rgb=rgb)
    monkeypatch.setattr(m.inputs.moving,'load_blind',lambda *a:packet)
    folder=tmp_path/'results'/(m.NAME+'_fixture');store=m.Store(folder)
    slots=[];rows=[];store.event('started',prepare_sha256='a'*64)
    for root in (0,1):
        slot=dict(source_slot=root,emb=np.zeros((3,192),np.float32).tolist(),goal_emb=np.zeros(192,np.float32).tolist(),
                  warm=np.zeros((2,10),np.float32).tolist(),blind_npz='fixture',blind_sha256='b'*64)
        slots.append(slot)
        row=dict(root=root,status='normalized_mean_sealed',cost_attempted=2,cost_returned=2,costs=[],callbacks=[],modules=[])
        base=dict(pixels=np.stack([m.transformed_rgb(rgb)]*3)[None],goal=m.transformed_rgb(rgb)[None,None],
                  emb=np.zeros((1,3,192),np.float32),goal_emb=np.zeros((1,1,192),np.float32))
        row['base_inputs']={k:store.array(v) for k,v in base.items()};row['warm']=store.array(np.zeros((1,1,2,10),np.float32))
        mean=np.zeros((1,2,10),np.float32);std=np.ones_like(mean)
        for j in range(2):
            store.event('cost_intent',root=root,iteration=j)
            candidate=np.arange(80,dtype=np.float32).reshape(1,4,2,10)/np.float32(100)
            candidate[:,0]=mean
            actions=np.concatenate([np.zeros_like(candidate),candidate],axis=2)
            pred=np.zeros((1,4,5,192),np.float32)
            pred[:,:,3]=candidate.sum(axis=(2,3))[...,None]
            pred[:,:,4]=pred[:,:,3]*np.float32(2)
            emb=np.zeros((4,4,192),np.float32)
            records=[('action_encoder',[actions.reshape(4,4,10)],emb)]
            for step in (0,1):
                po=pred[:,:,step:step+3].reshape(4,3,192).copy();po[:,-1]=pred[:,:,3+step].reshape(4,192)
                records.extend([('predictor',[pred[:,:,step:step+3].reshape(4,3,192),emb[:,step:step+3]],po),
                                ('pred_proj',[po.reshape(12,192)],po.reshape(12,192).copy())])
            for kind,args,out in records:
                index=len(row['modules']);store.event('module_intent',root=root,index=index,kind_name=kind)
                row['modules'].append(dict(index=index,kind=kind,iteration=j,returned=True,
                    inputs=[store.array(v) for v in args],output=store.array(out)))
                store.event('module_return',root=root,index=index,kind_name=kind)
            cost=(pred[:,:,-1]**2).sum(axis=-1,dtype=np.float32)
            row['costs'].append(dict(iteration=j,returned=True,candidates=store.array(candidate),actions=store.array(actions),
                predicted_emb=store.array(pred),action_prefix=store.array(actions[:,:,:3]),cost=store.array(cost)))
            store.event('cost_return',root=root,iteration=j)
            inds=np.argsort(cost,axis=1)[:,:2];elite=candidate[np.arange(1)[:,None],inds]
            newmean=elite.mean(axis=1);newstd=elite.std(axis=1,ddof=1)
            cb=dict(candidates=candidate,costs=cost,topk_vals=np.take_along_axis(cost,inds,axis=1),topk_inds=inds,
                    topk_candidates=elite,mean=newmean,var=newstd,prev_mean=mean,prev_var=std)
            row['callbacks'].append({k:store.array(v) for k,v in cb.items()});store.event('cem_update',root=root,iteration=j)
            mean,std=newmean,newstd
        row.update(actions=store.array(mean),mean=store.array(mean),std=store.array(std),elite_average_cost=cb['topk_vals'].mean(axis=1).tolist())
        rows.append(row);store.event('root_sealed',root=root)
    status='all_known2_native_normalized_CEM_recorded';store.event('finished',status=status)
    journal=folder/'journal.jsonl';store.inventory['journal.jsonl']=dict(sha256=m.sha(journal),file_bytes=journal.stat().st_size)
    prepared=dict(slots=slots,expected_model_hash='c'*64,inherited={'encoder':8,'projector':8})
    report=dict(schema=m.NAME+'.report',protocol=m.protocol(),status=status,prepared='fixture',prepare_sha256='a'*64,
        rows=rows,errors=[],source_verified=True,model_before='c'*64,model_after='c'*64,physical_release=False,
        scientific_admission=False,artifacts=copy.deepcopy(store.inventory),elapsed_seconds=1.,journal_tail=store.tail,
        inherited=prepared['inherited'],cuda_peak_allocated=1,cuda_peak_reserved=1,
        numerics=dict(requested_SDP='MATH',kernel_certificate=False,observed_inference=True,observed_TF32=False,observed_autocast=False))
    monkeypatch.setattr(m,'verify',lambda *a:prepared)
    store.put('report.json',m.raw(report),final=True)
    return folder,report,prepared,store


def test_real_store_complete_reader_and_hashchain(fixture):
    folder,report,_,_=fixture
    lines=(folder/'journal.jsonl').read_bytes().splitlines(keepends=True)
    assert len(lines)==56 and all(line.endswith(b'\n') and not line.endswith(b'\n\n') for line in lines)
    receipt=m.consume(folder,m.sha(folder/'report.json'))
    assert receipt['status']=='all_known2_native_normalized_CEM_saved_verified'
    assert len(receipt['rows'])==2 and receipt['new_encoder']==0 and receipt['physical_release'] is False
    assert all(r['candidate_count']==8 and r['returned_mean_evaluated'] is False for r in receipt['rows'])


@pytest.mark.parametrize('mutation', ['candidate','history','output','mean','elite','count','module_shape','pixels','unreferenced'])
def test_saved_mutations_reject(fixture,mutation):
    folder,r,_,store=fixture
    row=r['rows'][0]
    def replaced(rec,fn):
        a=m.load_array(folder,rec,r['artifacts']).copy();a=fn(a);new=store.array(a)
        del r['artifacts'][rec['file']]
        # Keep old file: row-only test deliberately avoids census, tests algebra.
        r['artifacts'][new['file']]={k:new[k] for k in ('sha256','file_bytes')}
        return new
    if mutation=='candidate':row['costs'][0]['candidates']=replaced(row['costs'][0]['candidates'],lambda a:a+1)
    elif mutation=='history':row['base_inputs']['emb']=replaced(row['base_inputs']['emb'],lambda a:a+1)
    elif mutation=='output':row['costs'][0]['predicted_emb']=replaced(row['costs'][0]['predicted_emb'],lambda a:a+1)
    elif mutation=='mean':row['mean']=replaced(row['mean'],lambda a:a+1)
    elif mutation=='elite':row['callbacks'][0]['topk_inds']=replaced(row['callbacks'][0]['topk_inds'],lambda a:np.zeros_like(a))
    elif mutation=='count':row['cost_returned']=1
    elif mutation=='module_shape':row['modules'][1]['output']=replaced(row['modules'][1]['output'],lambda a:a.reshape(12,192))
    elif mutation=='pixels':row['base_inputs']['pixels']=replaced(row['base_inputs']['pixels'],lambda a:a+1)
    else:row['modules'].append(copy.deepcopy(row['modules'][0]))
    with pytest.raises(ValueError):m.validate_row(fixture[2]['slots'][0],row,lambda rec:m.load_array(folder,rec,r['artifacts']))


@pytest.mark.parametrize('change',['blank_line','tail','extra_file','partial','inherited','memory','SHA'])
def test_whole_receipt_rejects(fixture,change):
    folder,r,_,_=fixture
    if change=='blank_line':
        p=folder/'journal.jsonl';p.write_bytes(p.read_bytes()+b'\n')
        r['artifacts']['journal.jsonl']=dict(sha256=m.sha(p),file_bytes=p.stat().st_size)
    elif change=='tail':r['journal_tail']='d'*64
    elif change=='extra_file':(folder/'untracked').write_bytes(b'x')
    elif change=='partial':r['status']='partial_or_failed'
    elif change=='inherited':r['inherited']={'encoder':0}
    elif change=='memory':r['cuda_peak_allocated']=9*1024**3
    else:r['artifacts'][next(iter(r['artifacts']))]['sha256']='0'*64
    (folder/'report.json').write_bytes(m.raw(r))
    with pytest.raises((ValueError,json.JSONDecodeError)):m.consume(folder,m.sha(folder/'report.json'))


def test_store_exclusive_and_two_event_codec(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:type('Space',(),{'free':10*1024**3})())
    store=m.Store(tmp_path/'results'/(m.NAME+'_two'))
    store.event('one');store.event('two')
    tail=None
    for j,line in enumerate((store.path/'journal.jsonl').read_bytes().splitlines(keepends=True)):
        e=json.loads(line);h=e.pop('event_sha256')
        assert e['sequence']==j and e['previous_sha256']==tail and hashlib.sha256(m.raw(e)).hexdigest()==h
        tail=h
    assert tail==store.tail and len(store.events)==2
    with pytest.raises(ValueError):m.Store(store.path)
    with pytest.raises(ValueError):store.put('../escape',b'no')
    with pytest.raises(ValueError):store.array(np.zeros(1,np.float64))


def test_npy_header_rejected_before_allocation(fixture):
    folder,report,_,store=fixture
    buf=io.BytesIO();np.lib.format.write_array_header_1_0(buf,dict(descr='<f4',fortran_order=False,shape=(10**15,)))
    rec=store.put('bad.npy',buf.getvalue());report['artifacts']['bad.npy']={k:rec[k] for k in ('sha256','file_bytes')}
    rec.update(shape=[10**15],dtype='<f4',nbytes=4*10**15)
    with pytest.raises(ValueError):m.load_array(folder,rec,report['artifacts'])


def test_fixed_payload_bound_and_no_budget_override(monkeypatch):
    # Exact tensor element census from native shapes, independent of producer budget.
    samples,elite=300,30
    modules=4*(samples*4*10+samples*4*192+2*(3*samples*3*192)+2*(2*samples*3*192))
    calls=4*(samples*2*10+samples*4*10+samples*5*192+samples*3*10+samples)
    callbacks=4*(samples*2*10+samples+elite+elite*2*10+4*2*10)+8*elite
    assert modules+calls+callbacks==9171080
    assert 60*(modules+calls+callbacks)<600*1024**2
    monkeypatch.setattr(m,'SAMPLES',24)
    with pytest.raises(ValueError):m.fixed_defaults()


@pytest.mark.parametrize('change',['wall','cap','floor'])
def test_store_budgets(fixture,monkeypatch,change):
    store=fixture[3]
    if change=='wall':store.started-=m.WALL+1
    elif change=='cap':store.total=m.CAP-m.RESERVE
    else:monkeypatch.setattr(m.shutil,'disk_usage',lambda p:type('Space',(),{'free':m.FLOOR})())
    with pytest.raises(ValueError):store.put('budget.bin',b'x')


def test_installed_source_defaults_without_import():
    node=ast.parse(m.CEM.read_text())
    init=next(n for n in ast.walk(node) if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    defaults=dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):],init.args.defaults))
    assert {k:ast.literal_eval(defaults[k]) for k in ('batch_size','num_samples','n_steps','topk','var_scale','seed')}==dict(
        batch_size=1,num_samples=300,n_steps=30,topk=30,var_scale=1,seed=1234)
    assert all(m.sha(p)==h for p,h in m.PINS.items())
    m.fixed_defaults()
    p=m.protocol();assert p['per_root']['predictor']==60 and p['per_root']['predictor_samples']==18000
    assert p['final_mean_evaluated'] is False and p['physical_release'] is False


def test_no_tensor_import_cli_and_unauthorized():
    result=subprocess.run([sys.executable,'-c',
        'import sys; from experiments.spectral_reachability import agent_lewm_native_cem_slice_20261029_v1 as m; '
        'assert "torch" not in sys.modules; m.main()', '--help'],capture_output=True,text=True)
    assert result.returncode==0 and 'consume' in result.stdout
    with pytest.raises(ValueError):m.run('nonexistent','no','nonexistent','wrong')


@pytest.mark.parametrize('actual,reference',[(np.array([np.nan]),np.array([0.])),(np.zeros(2),np.zeros(1)),(np.array([2.]),np.array([0.]))])
def test_reduction_rejects(actual,reference):
    with pytest.raises(ValueError):m.bounded_reduction(actual,reference,192)


def test_transformed_rgb():
    a=m.transformed_rgb(np.zeros((224,224,3),np.uint8))
    assert a.dtype==np.float32 and a.shape==(3,224,224)
    assert np.array_equal(a[:,0,0],-np.array([.485,.456,.406],np.float32)/np.array([.229,.224,.225],np.float32))
    with pytest.raises(ValueError):m.transformed_rgb(np.zeros((223,224,3),np.uint8))
