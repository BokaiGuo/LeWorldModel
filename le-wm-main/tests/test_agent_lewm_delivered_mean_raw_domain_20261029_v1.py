"""Pure NumPy and temporary saved files only, no real cohort analyze/fullstrict."""
import copy
import json
import subprocess
import sys

import numpy as np
import pytest
from experiments.spectral_reachability import agent_lewm_delivered_mean_raw_domain_20261029_v1 as m


def task(mean=(0.,0.),std=(1.,1.)):
    return dict(action_mean=list(mean),action_std=list(std))


def evaluate(tokens,mean=(0.,0.),std=(1.,1.),root=0):
    return m.evaluate(root,np.asarray(tokens,np.float32).reshape(1,2,10),np.asarray(mean,np.float32),np.asarray(std,np.float32),task(mean,std))


def test_identity_preserves_every_raw_and_bit():
    a=np.arange(20,dtype=np.float32)/20-.5;result=evaluate(a)
    assert result['domain']['legal'] is True and result['domain']['outside_count']==0
    assert result['status']=='saved_domain_and_roundtrip_exact_only'
    assert result['roundtrip_comparison']['bitwise'] is True
    assert result['old_mean_cost_certifies_roundtrip_tokens'] is True
    assert result['deployment_token_replay_required'] is False
    for key in ('delivered_tokens','inverse_product','raw_packets','raw10','roundtrip_subtracted','roundtrip_tokens'):
        entry=result[key];b=bytes.fromhex(entry['raw_bytes_hex'])
        assert len(b)==entry['nbytes']==80 and m.hashlib.sha256(b).hexdigest()==entry['raw_bytes_sha256']
        decoded=np.frombuffer(b,dtype=np.float32).reshape(entry['shape'])
        assert decoded.view(np.uint32).tolist()==entry['uint32_bits']
    assert np.array_equal(np.array(result['raw10']['values'],np.float32),a.reshape(10,2))
    assert result['physical_release'] is result['scientific_admission'] is False


def test_boundary_is_exact_not_tolerance():
    a=np.zeros(20,np.float32);a[0]=-1.;a[1]=1.
    assert evaluate(a)['domain']['legal']
    a[1]=np.nextafter(np.float32(1),np.float32(np.inf));result=evaluate(a)
    assert result['status']=='blocked_raw_domain' and result['domain']['outside_count']==1
    assert result['domain']['violations']==[dict(raw_index=0,packet=0,offset=0,axis=1,value=float(a[1]),
        uint32_bits=int(a.view(np.uint32)[1]),side='above_upper')]
    assert result['roundtrip_is_legal_normalizer_input'] is False
    assert result['old_mean_cost_certifies_roundtrip_tokens'] is False
    assert result['raw10']['uint32_bits'][0][1]==int(a.view(np.uint32)[1])  # never clip


def test_all_packet_slots_and_axes_preserved():
    a=np.zeros(20,np.float32);a[0]=-2.;a[11]=1.5;a[18]=2.
    result=evaluate(a)
    assert result['domain']['packet_accepted']==[False,False]
    assert [(v['packet'],v['offset'],v['axis']) for v in result['domain']['violations']]==[(0,0,0),(1,0,1),(1,4,0)]
    assert sum(not x for row in result['domain']['per_element_in_box'] for x in row)==3
    assert result['roundtrip_comparison']['bitwise'] is True  # domain remains blocked independently


def test_nonexact_roundtrip_requires_new_paid_replay():
    # Synthetic precision fixture, not MAIN's actual root actions or outcomes.
    a=np.full(20,np.float32(.00001));result=evaluate(a,mean=(-.0087,.0068),std=(.2019,.2002))
    assert result['domain']['legal'] is True
    assert result['roundtrip_comparison']['bitwise'] is False
    assert result['status']=='requires_paid_deployment_token_replay'
    assert result['deployment_token_replay_required'] is True
    assert result['old_mean_cost_certifies_roundtrip_tokens'] is False
    assert result['old_mean_cost_must_not_be_reused_for_changed_tokens'] is True
    diff=result['roundtrip_comparison']
    assert np.asarray(diff['per_element_ULP']).shape==(1,2,10)
    assert np.asarray(diff['per_element_bit_equal']).shape==(1,2,10)
    assert diff['max_ULP']==max(x for packet in diff['per_element_ULP'][0] for x in packet)


def test_signed_zero_is_not_bitexact():
    a=np.zeros(20,np.float32);a[0]=-0.
    result=evaluate(a)
    assert result['roundtrip_comparison']['bitwise'] is False
    assert result['roundtrip_comparison']['signed_zero_bit_differences']==1
    assert result['roundtrip_comparison']['max_ULP']==0
    assert result['deployment_token_replay_required'] is True


@pytest.mark.parametrize('kind',['wrongroot','nan','shape','dtype','stats','zerostd'])
def test_bad_input_rejected(kind):
    a=np.zeros((1,2,10),np.float32);mean=np.zeros(2,np.float32);std=np.ones(2,np.float32);t=task();root=0
    if kind=='wrongroot':root=2
    elif kind=='nan':a[0,0,0]=np.nan
    elif kind=='shape':a=a.reshape(2,10)
    elif kind=='dtype':a=a.astype(np.float64)
    elif kind=='stats':t['action_mean']=[1.,0.]
    else:std[0]=0.
    with pytest.raises(ValueError):m.evaluate(root,a,mean,std,t)


def test_original_p2_source_constants_only(tmp_path):
    mean,std=m.constant_stats(m.ROOT/'experiments/run_pusht_counterfactual.py')
    assert mean.view(np.uint32).tolist()==np.array([-.0087,.0068],np.float32).view(np.uint32).tolist()
    assert std.view(np.uint32).tolist()==np.array([.2019,.2002],np.float32).view(np.uint32).tolist()
    for path,h in m.PINS.items():assert m.sha(path)==h
    source=tmp_path/'bad.py';source.write_text('PUBLIC_ACTION_MEAN=np.asarray([0,0], dtype=np.float64)\nPUBLIC_ACTION_STD=np.asarray([1,1], dtype=np.float32)\n')
    with pytest.raises(ValueError):m.constant_stats(source)


@pytest.fixture
def saved_data(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    monkeypatch.setattr(m.shutil,'disk_usage',lambda path:type('Space',(),{'free':1024**3})())
    data=dict(rows=[dict(root=i,tokens=np.zeros((1,2,10),np.float32),mean=np.zeros(2,np.float32),std=np.ones(2,np.float32),
        task=task(),identity={'fixture_root':i}) for i in (0,1)],sources={},authority={'fixture':'not production'},inherited={})
    # Keep both, including one domain failure; no reduced ready cohort.
    data['rows'][0]['tokens'][0,1,-1]=2.
    monkeypatch.setattr(m,'collect',lambda:data)
    return tmp_path/'results'/(m.NAME+'_fixture.json'),data


def test_actual_exclusive_writer_and_whole_verify(saved_data):
    path,data=saved_data
    result=m.analyze(path);assert result['status']=='known2_raw_domain_and_roundtrip_analyzed'
    assert m.verify(path,result['sha256'])['status']=='known2_raw_domain_roundtrip_saved_verified'
    r=json.loads(path.read_bytes());assert [row['root'] for row in r['rows']]==[0,1]
    assert r['rows'][0]['status']=='blocked_raw_domain' and r['rows'][1]['domain']['legal'] is True
    assert r['all_roots_in_box'] is False and r['all_roundtrips_bitexact'] is True
    assert r['physical_release'] is False
    with pytest.raises(ValueError):m.analyze(path)
    data['rows'][1]['tokens'][0,0,0]=.5
    with pytest.raises(ValueError):m.verify(path,result['sha256'])


@pytest.mark.parametrize('mutation',['drop_root','domain','bits','ULP','raw','upstream','release'])
def test_saved_mutants_rejected(saved_data,mutation):
    path,_=saved_data;m.analyze(path);r=json.loads(path.read_bytes())
    if mutation=='drop_root':r['rows'].pop()
    elif mutation=='domain':r['rows'][0]['domain']['legal']=True
    elif mutation=='bits':r['rows'][0]['raw10']['uint32_bits'][0][0]=7
    elif mutation=='ULP':r['rows'][0]['roundtrip_comparison']['per_element_ULP'][0][0][0]=7
    elif mutation=='raw':r['rows'][0]['raw10']['values'][0][0]=.1
    elif mutation=='upstream':r['authority']={}
    else:r['physical_release']=True
    path.write_bytes(m.raw(r))
    with pytest.raises(ValueError):m.verify(path,m.sha(path))


def test_signed_zero_mutation_rejected_in_values(saved_data):
    path,_=saved_data;m.analyze(path);r=json.loads(path.read_bytes())
    r['rows'][1]['raw10']['values'][0][0]=-0.
    path.write_bytes(m.raw(r))
    with pytest.raises(ValueError):m.verify(path,m.sha(path))


def test_bounded_output_and_paths(saved_data,monkeypatch):
    path,_=saved_data
    with pytest.raises(ValueError):m.analyze(path.parent/'not_own.json')
    monkeypatch.setattr(m,'CAP',10)
    with pytest.raises(ValueError):m.analyze(path)
    assert not path.exists()


def test_cold_cli_no_torch_or_physics():
    result=subprocess.run([sys.executable,'-c',
        'import sys;from experiments.spectral_reachability import agent_lewm_delivered_mean_raw_domain_20261029_v1 as m;'
        'assert "torch" not in sys.modules;assert "pymunk" not in sys.modules;m.main()','--help'],capture_output=True,text=True)
    assert result.returncode==0 and 'analyze' in result.stdout and 'verify' in result.stdout


@pytest.fixture
def authority(monkeypatch):
    """Synthetic complete metadata graph. Never invoke real collect/owner strict."""
    mean=np.array([-.0087,.0068],np.float32);std=np.array([.2019,.2002],np.float32)
    old=m.paid.old;docs={};arrays={};hashes={str(p):h for p,h in m.PINS.items()}
    upstream=dict(report='/fixture/native/report.json',report_sha256=m.paid.REPORT_SHA,
        receipt='/fixture/native/receipt.json',receipt_sha256=m.paid.RECEIPT_SHA,
        prepared='/fixture/native/prepare.json',prepare_sha256=m.paid.PREPARE_SHA)
    tasks={};pslots=[];tslots=[];rows=[];native_rows=[];receiptrows=[];native_receiptrows=[];sources={}
    def ref(root,key,a):
        name=f'r{root}_{key}.npy';rec=dict(file=name,sha256='a'*64);arrays[name]=a
        hashes['/fixture/native/'+name]='a'*64;hashes[str(m.REPORT.parent/name)]='a'*64
        return rec
    for root in (0,1):
        inputs={k:ref(root,k,a) for k,a in dict(actions=np.zeros((1,2,10),np.float32),
            emb=np.zeros((1,3,192),np.float32),warm=np.zeros((1,1,2,10),np.float32),goal_emb=np.zeros((1,1,192),np.float32)).items()}
        task_path=f'/fixture/task{root}.json.gz';t=dict(action_mean=mean.tolist(),action_std=std.tolist(),goal=[0.]*192,
            root_context=dict(latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist()))
        tasks[task_path]=t;hashes[task_path]='e'*64;sources[task_path]='e'*64
        pslots.append(dict(root=root,refs={'actions':inputs['actions']}));tslots.append(dict(source_slot=root,task_export=task_path,task_sha256='e'*64))
        rows.append(dict(root=root,status='paid_finalmean_evaluated',inputs=inputs))
        native_rows.append(dict(root=root,status='normalized_mean_sealed',actions=inputs['actions'],mean=inputs['actions'],callbacks=[{'mean':inputs['actions']}]))
        receiptrows.append(dict(root=root,returned_mean_evaluated_in_new_job=True,old_slice_mean_evaluated=False))
        native_receiptrows.append(dict(root=root,returned_mean_evaluated=False))
    docs[str(m.REPORT)]=dict(schema=m.paid.NAME+'.report',status='known2_paid_finalmean_evaluated',errors=[],all_predictions_sealed=True,
        source_verified=True,protocol=m.paid.protocol(),prepare_sha256=m.PREPARE_SHA,prepared='/fixture/paid_prepare.json',
        model_before='c'*64,model_after='c'*64,upstream=upstream,inherited={},rows=rows,artifacts={})
    docs['/fixture/paid_prepare.json']=dict(schema=m.paid.NAME+'.prepare',protocol=m.paid.protocol(),slots=pslots,
        expected_model_hash='c'*64,upstream=upstream,inherited={},sources=sources)
    docs[str(m.RECEIPT)]=dict(schema=m.paid.NAME+'.consume',status='known2_paid_finalmean_saved_verified',report_sha256=m.REPORT_SHA,
        upstream=upstream,resources=m.paid.protocol()['resources'],inherited={},rows=receiptrows,old_slice_mean_evaluated=False,physical_release=False,scientific_admission=False)
    docs[upstream['report']]=dict(schema=old.NAME+'.report',status='all_known2_native_normalized_CEM_recorded',errors=[],
        prepare_sha256=m.paid.PREPARE_SHA,prepared=upstream['prepared'],protocol=old.protocol(),model_before='c'*64,model_after='c'*64,rows=native_rows,artifacts={})
    docs[upstream['receipt']]=dict(schema=old.NAME+'.consume',status='all_known2_native_normalized_CEM_saved_verified',report_sha256=m.paid.REPORT_SHA,rows=native_receiptrows)
    docs[upstream['prepared']]=dict(schema=old.NAME+'.prepare',protocol=old.protocol(),expected_model_hash='c'*64,slots=tslots)
    hashes.update({str(m.REPORT):m.REPORT_SHA,str(m.RECEIPT):m.RECEIPT_SHA,'/fixture/paid_prepare.json':m.PREPARE_SHA,
        upstream['report']:m.paid.REPORT_SHA,upstream['receipt']:m.paid.RECEIPT_SHA,upstream['prepared']:m.paid.PREPARE_SHA})
    def read(path,h):
        m.need(hashes[str(path)]==h,'fixture external SHA');return docs[str(path)]
    monkeypatch.setattr(m,'read',read);monkeypatch.setattr(m,'sha',lambda path:hashes.get(str(path),'f'*64))
    monkeypatch.setattr(m,'load_array',lambda folder,rec,inventory:arrays[rec['file']].copy())
    monkeypatch.setattr(m,'load_task',lambda path,h:{'gn_task':tasks[path]})
    monkeypatch.setattr(m,'task_identity',lambda t:'fixture_task')
    monkeypatch.setattr(m,'constant_stats',lambda path:(mean,std))
    return docs,arrays,tasks,hashes


def test_complete_authority_graph_retains_both(authority):
    result=m.collect()
    assert [r['root'] for r in result['rows']]==[0,1]
    assert result['authority']['Hi_original_scaler'] is False
    assert result['authority']['paid_report_sha256']==m.REPORT_SHA
    assert '/fixture/native/receipt.json' in result['sources']
    assert all(row['tokens'].shape==(1,2,10) for row in result['rows'])


@pytest.mark.parametrize('mutation',['paid_receipt','native_receipt','order','mean','context','scaler','source','model','subset'])
def test_authority_misbindings_rejected(authority,mutation):
    docs,arrays,tasks,hashes=authority
    if mutation=='paid_receipt':docs[str(m.RECEIPT)]['report_sha256']='0'*64
    elif mutation=='native_receipt':docs['/fixture/native/receipt.json']['status']='partial'
    elif mutation=='order':docs[str(m.REPORT)]['rows'][1]['root']=0
    elif mutation=='mean':
        arrays['wrong.npy']=np.ones((1,2,10),np.float32)
        docs['/fixture/native/report.json']['rows'][0]['callbacks'][-1]['mean']={'file':'wrong.npy','sha256':'a'*64}
    elif mutation=='context':arrays['r1_emb.npy'][0,0,0]=1.
    elif mutation=='scaler':tasks['/fixture/task1.json.gz']['action_std']=[1.,1.]
    elif mutation=='source':hashes['/fixture/task1.json.gz']='0'*64
    elif mutation=='model':docs[str(m.REPORT)]['model_after']='0'*64
    else:docs['/fixture/paid_prepare.json']['slots'].pop()
    with pytest.raises(ValueError):m.collect()
