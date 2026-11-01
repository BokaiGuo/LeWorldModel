"""Saved-fixture only: no model, encoder, simulator or native runtime."""
from copy import deepcopy
import io
import json
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p5_jepa_callback_zero_sets_20261029_v1 as m


def save(path,v):
    path.write_bytes(m.canonical(v)+b'\n');return m.sha(path)


def fixture(tmp_path):
    jdir=tmp_path/'jepa';cdir=tmp_path/'callback';jdir.mkdir();cdir.mkdir()
    image=tmp_path/'original.png';image.write_bytes(b'RAM-image-marker-not-a-neural-or-PNG-test');ish=m.sha(image)
    surface=cdir/'surface.png';surface.write_bytes(b'RAM-surface-marker');ssh=m.sha(surface)
    target=cdir/'target.png';target.write_bytes(image.read_bytes())
    arr=[];inv={}
    for index,value in enumerate((0.,1.)):
        a=np.full((1,256,384),value,np.float32);stream=io.BytesIO();np.save(stream,a,allow_pickle=False)
        name=f'patch{index}.npy';(jdir/name).write_bytes(stream.getvalue())
        d=dict(file=name,sha256=m.sha(jdir/name),shape=list(a.shape),dtype='float32',nbytes=a.nbytes,file_bytes=len(stream.getvalue()))
        arr.append(d);inv[name]=dict(sha256=d['sha256'],file_bytes=d['file_bytes'])
    requests=[];jrows=[];crows=[];jobs=[];captures=[]
    for i in range(164):
        slot,local=divmod(i,41);root,cond=m.ORDER[slot];overlay=1 if cond=='on' else 0
        base=dict(index=i,root_slot=root,condition=cond,task_id=f'task{slot}',selection_digest=f'selection{i}')
        label=dict(path=str(image),file_sha256=ish,pixel_sha256='a'*64,phase='plan',step=10,condition=cond,overlay_index=overlay)
        commands=np.zeros((10,2),np.float32).tolist()
        if local:
            hi,offset=divmod(local-1,20);coord,si=divmod(offset,2);h=(.01,.001)[hi];sign=(-1,1)[si]
            pid=f's{slot}_h{hi}_c{coord}_{"minus" if sign<0 else "plus"}'
            commands[5+coord//2][coord%2]=float(np.float32(sign*h))
        else:coord=h=None;sign=0;pid=f's{slot}_baseline'
        job=dict(index=i,slot=slot,root_slot=root,condition=cond,h=h,coord=coord,sign=sign,id=pid,
            job_sha256=f'{i+1:064x}',full10=deepcopy(commands))
        jobs.append(job)
        requests.append(dict(base,commands=[[0.,0.]]*5+deepcopy(commands),context_global_frames=[5,10,15],successor_global_frame=20,
            label=label,images=[label]*3))
        # All observations initially equal; a test changes exact membership.
        jrows.append(dict(base,label_patch=arr[0]))
        plan=dict(base,probe_id=pid,producer_job_sha256=job['job_sha256'],commands=deepcopy(commands))
        crows.append(dict(base,plan=plan))
        captures.append(dict(branch=i,original_observation={k:v for k,v in label.items() if k!='path'},
            rgb224=dict(file='target.png',sha256=ish,pixel_sha256=label['pixel_sha256']),
            surface512=dict(file='surface.png',sha256=ssh)))
    counts=dict(RAM_only=164)
    jr=dict(schema=m.JEPA+'.report',rows=jrows,pairs=[dict(s,missing=[],observed_response_l2=0.) for s in m.schedule()],
        artifacts=inv,counts=counts,status='fixed164_actual15_conditioned_factual_H1_recorded',errors=[],source_errors=[],model_unchanged=True)
    ca=dict(schema=m.CALLBACK+'.consume',status='all164_native_callback_capture_verified',fixed_jobs=164,fixed_pairs=80,
        counts=counts,full_replay_exact=True,historical_callback_membership=None,sufficient_original_cause_certified=False,
        P1_admission=False,pairs=[dict(s,missing=[],callbacks_equal=False,integer_primitives_equal=True,
            surface512_equal=True,rgb224_equal=True,first_loss_stage='RAM') for s in m.schedule()])
    cr=dict(schema=m.CALLBACK+'.report',rows=crows,captures=captures,artifacts={'target.png':ish,'surface.png':ssh},
        full_replay_exact=True,historical_callback_membership=None,counts=counts,errors=[],contract=dict(pairs_schedule=m.schedule()))
    original=tmp_path/'original_prepare.json';opsha=save(original,dict(jobs=jobs))
    history=tmp_path/'original_report.json';history.write_bytes(b'RAM sealed physical report');orsha=m.sha(history)
    jp=dict(schema=m.JEPA+'.prepare',bridge=dict(requests=requests),protocol=dict(pairs=m.schedule()),
        bindings={str(image):ish,str(original):opsha,str(history):orsha})
    cp=dict(schema=m.CALLBACK+'.prepare',original_prepare=str(original),original_prepare_sha256=opsha,
        original_report=str(history),original_report_sha256=orsha,bindings={str(image):ish})
    jpfile=tmp_path/'jprepare.json';cpfile=tmp_path/'cprepare.json'
    jr['execution_binding']=dict(prepared=str(jpfile),prepare_sha256=save(jpfile,jp))
    cr['execution_binding']=dict(prepared=str(cpfile),prepare_sha256=save(cpfile,cp),original_report_sha256=orsha)
    ja=dict(schema=m.JEPA+'.consume',status='all164_factual_H1_and80_pairs_verified',requests=164,pairs=80,
        counts=counts,P1_admission=False,P5_admission=False)
    return dict(jdir=jdir,cdir=cdir,jr=jr,cr=cr,ja=ja,ca=ca,jp=jp,cp=cp,jobs=jobs,arr=arr)


def persist(f,tmp_path):
    jsha=save(f['jdir']/'report.json',f['jr']);csha=save(f['cdir']/'report.json',f['cr'])
    f['ja']['report_sha256']=jsha;f['ca']['report_sha256']=csha
    jreceipt=tmp_path/'jreceipt.json';creceipt=tmp_path/'creceipt.json'
    return (f['jdir'],jsha,jreceipt,save(jreceipt,f['ja']),f['cdir'],csha,creceipt,save(creceipt,f['ca']))


def connect(f):
    return m.connect(f['jr'],f['cr'],f['ca'],f['jp'],f['cp'],dict(jobs=f['jobs']),f['jdir'],f['cdir'],{})


def test_full80_joins_and_set_membership_from_actual_saved_arrays(tmp_path):
    f=fixture(tmp_path);r=connect(f)
    assert r['status']=='all80_exact_identity_joined' and len(r['rows'])==80
    assert all(x['observed_response_zero'] is True for x in r['rows'])
    assert all(v['full_sets_equal'] is True for v in r['sets']['layers'].values())
    assert all(x['historical_callback_membership'] is None for x in r['rows'])


def test_equal24_cardinality_does_not_mean_equal_sets(tmp_path):
    f=fixture(tmp_path)
    for n,s in enumerate(m.schedule()):
        zero=n<24
        if not zero:f['jr']['rows'][s['plus_index']]['label_patch']=f['arr'][1]
        f['jr']['pairs'][n]['observed_response_l2']=0. if zero else float(np.sqrt(256*384))
        # Same cardinality, but membership shifted by one.
        for k in m.LAYERS:f['ca']['pairs'][n][k]=1<=n<=24
    r=connect(f);summary=r['sets']['layers']['rgb224_equal']
    assert len(summary['equal_members'])==len(summary['zero_members'])==24
    assert summary['full_sets_equal'] is False and len(summary['intersection'])==23
    assert summary['equal_only_known']==[r['rows'][24]['pair_id']]
    assert summary['zero_only_known']==[r['rows'][0]['pair_id']]


@pytest.mark.parametrize('field',['minus_index','plus_index','root_slot','condition','h','coord','missing_identity','duplicate'])
def test_exact_identity_not_positional_fallback(tmp_path,field):
    f=fixture(tmp_path);v=f['ca']['pairs'][0]
    if field=='missing_identity':del v['coord']
    elif field=='duplicate':f['ca']['pairs'].append(deepcopy(v))
    elif field=='condition':v[field]='on'
    elif field=='h':v[field]=.1
    else:v[field]+=1
    out=connect(f)
    assert len(out['rows'])==80 and out['status']=='blocked_missing_pair_mapping'
    assert out['rows'][0]['missing'] and out['rows'][0]['rgb224_equal'] is None
    assert out['sets']['layers']['rgb224_equal']['full_sets_equal'] is None


def test_pair_order_shuffled_does_not_break_exact_key_join(tmp_path):
    f=fixture(tmp_path);f['ca']['pairs'].reverse();f['jr']['pairs'].reverse()
    assert connect(f)['status']=='all80_exact_identity_joined'


def test_extra_unmappable_record_is_not_silently_filtered(tmp_path):
    f=fixture(tmp_path);f['ca']['pairs'].append(dict(note='no identity'))
    r=connect(f)
    assert r['input_pair_counts']['callback']==81 and r['invalid_identity_records']['callback']==[80]
    assert r['status']=='blocked_missing_pair_mapping'


@pytest.mark.parametrize('field',['sign','job_id','task','selection','raw','capture','PNG','source_binding','time'])
def test_unmappable_branch_provenance_is_missing_never_guessed(tmp_path,field):
    f=fixture(tmp_path)
    if field=='sign':f['jobs'][1]['sign']=1
    elif field=='job_id':f['jobs'][1]['id']='wrong'
    elif field=='task':f['cr']['rows'][1]['task_id']='wrong'
    elif field=='selection':f['cr']['rows'][1]['selection_digest']='wrong'
    elif field=='raw':f['cr']['rows'][1]['plan']['commands'][0][0]=.5
    elif field=='capture':f['cr']['captures']=[x for x in f['cr']['captures'] if x['branch']!=1]
    elif field=='PNG':f['cr']['captures'][1]['rgb224']['pixel_sha256']='b'*64
    elif field=='source_binding':f['cp']['bindings'].clear()
    else:f['jp']['bridge']['requests'][1]['context_global_frames']=[0,5,10]
    r=connect(f)
    assert r['rows'][0]['missing'] and r['rows'][0]['observed_response_zero'] is None


def test_exact_zero_without_epsilon_and_reject_lying_report_norm(tmp_path):
    f=fixture(tmp_path);f['jr']['pairs'][0]['observed_response_l2']=1e-20
    with pytest.raises(ValueError,match='no epsilon'):connect(f)


def test_real_saved_array_shape_and_byte_mutation_rejection(tmp_path):
    f=fixture(tmp_path);(f['jdir']/'patch0.npy').write_bytes(b'bad')
    with pytest.raises(ValueError,match='SHA256'):connect(f)


def test_legacy_unexplained_layer_disagreement_preserved(tmp_path):
    f=fixture(tmp_path);f['ca']['pairs'][0]['surface512_equal']=True;f['ca']['pairs'][0]['rgb224_equal']=False
    f['ca']['pairs'][0]['first_loss_stage']='unexplained_same512_changed224'
    r=connect(f)
    assert r['rows'][0]['first_loss_stage']=='unexplained_same512_changed224'
    assert r['sets']['layers']['rgb224_equal']['full_sets_equal'] is False


def test_complete_disk_entry_external_receipts_and_exclusive_writer(tmp_path):
    f=fixture(tmp_path);args=persist(f,tmp_path);r=m.analyze(*args)
    assert r['fixed_pairs']==80 and r['full_upstream_consumers_rebuilt_here'] is False
    assert r['historical_callback_membership'] is None and not r['sufficient_original_cause_certified']
    out=tmp_path/(m.NAME+'_RAM.json');w=m.write_new(r,out)
    assert m.read(out,w['sha256'],{})==r
    with pytest.raises(ValueError):m.write_new(r,out)


@pytest.mark.parametrize('field',['jepa_sha','callback_sha','receipt_sha','H2','history_claim'])
def test_authority_rejection(tmp_path,field):
    f=fixture(tmp_path)
    if field=='H2':f['jr']['schema']='H2.report'
    if field=='history_claim':f['ca']['historical_callback_membership']=True
    args=list(persist(f,tmp_path))
    if field=='jepa_sha':args[1]='0'*64
    elif field=='callback_sha':args[5]='0'*64
    elif field=='receipt_sha':args[7]='0'*64
    with pytest.raises(ValueError):m.analyze(*args)
