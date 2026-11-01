"""RAM/temporary saved fixtures only; no production factor analysis or models."""
from copy import deepcopy
import io
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_nonzero_factors_20261029_v1 as m


def test_response_common_error_does_not_create_direction_error():
    r=m.response([100.,100.],[102.,100.],[0.,0.],[2.,0.])
    assert r['cosine']==1 and r['differential_error_MSE']==0 and r['common_error_MSE']==10000
    assert r['response_error_L2']==r['response_error_identity_residual_L2']==0
    assert r['endpoint_MSE_decomposition_residual']==0 and r['parallel_gain']==1


def test_reversal_and_low_signal_remain_distinct():
    r=m.response([0.,0.],[-1.,0.],[0.,0.],[1e-8,0.])
    assert r['cosine']==-1 and r['observed_L2']>0 and not r['observed_exact_zero']
    assert r['parallel_gain']<0 and r['response_error_over_signal']>1e7
    assert r['response_over_ULP_envelope'] is not None


def test_exact_zero_and_orthogonal_no_epsilon_or_fabricated_cosine():
    r=m.response([0,0],[0,1],[0,0],[1,0])
    assert r['cosine']==0 and r['parallel_gain']==0 and r['orthogonal_predicted_L2']==1
    r=m.response([0,0],[0,1],[0,0],[0,0])
    assert r['cosine'] is None and r['response_error_over_signal'] is None and r['observed_exact_zero']


@pytest.mark.parametrize('mutation',('shape','nan','empty'))
def test_nonfinite_support_rejected(mutation):
    values=[[0,1],[0,2],[0,1],[0,2]]
    if mutation=='shape':values[0]=[0]
    if mutation=='nan':values[1]=[float('nan'),0]
    if mutation=='empty':values[0]=[]
    with pytest.raises(ValueError):m.response(*values)


def test_nonuniform_steps_linear_and_curved_response():
    dm,dp=.1,.2
    r=m.local([-dm],[dp],[0],[-dm],[dp],[0],dm,dp)
    assert r['observed']['second_secant_L2']==0 and r['observed']['one_sided_cosine']==1
    r=m.local([dm*dm],[dp*dp],[0],[dm*dm],[dp*dp],[0],dm,dp)
    assert r['observed']['second_secant_L2']==pytest.approx(2)
    assert r['observed']['one_sided_cosine']==-1
    with pytest.raises(ValueError):m.local([0],[1],[0],[0],[1],[0],0,dp)


def test_scale_comparison_preserves_zero_and_opposite():
    assert m.scale_compare([1,0],[0,0])['cosine'] is None
    assert m.scale_compare([1,0],[-1,0])['cosine']==-1
    assert m.scale_compare([1,0],[1,0])['relative_difference_to_large']==0


def test_pixel_and_full_native_axes_no_object_only_failure_claim():
    a=np.zeros((224,224,3),np.uint8);b=a.copy();b[0,0,2]=1
    p=m.pixel_response(a,b);assert p['changed_channels']==p['changed_pixels']==p['max_integer_change']==1
    n=m.native_response([0]*7,[1,2,3,4,7,6,8])
    assert n['agent_delta_velocity']==[6,8] and n['object_delta_xytheta'][:2]==[3,4]
    assert len(n['delta_native7'])==7 and 'success' not in n
    with pytest.raises(ValueError):m.native_response([0]*5,[0]*5)


def test_real_npy_hash_shape_inventory_and_trailing_bytes(tmp_path):
    a=np.zeros((1,256,384),np.float32);b=io.BytesIO();np.save(b,a,allow_pickle=False)
    p=tmp_path/'array.npy';p.write_bytes(b.getvalue())
    rec=dict(file=p.name,shape=list(a.shape),dtype='float32',nbytes=a.nbytes,file_bytes=p.stat().st_size,sha256=m.sha(p))
    inv={p.name:{k:rec[k] for k in ('sha256','file_bytes')}}
    assert np.array_equal(m.load_array(tmp_path,rec,a.shape,inv,{}),a)
    bad=deepcopy(rec);bad['dtype']='float64'
    with pytest.raises(ValueError):m.load_array(tmp_path,bad,a.shape,inv,{})
    p.write_bytes(p.read_bytes()+b'trailing');rec.update(file_bytes=p.stat().st_size,sha256=m.sha(p))
    inv[p.name]={k:rec[k] for k in ('sha256','file_bytes')}
    with pytest.raises(ValueError):m.load_array(tmp_path,rec,a.shape,inv,{})


def identity_fixture():
    job=dict(root_slot=0,condition='off',full10=[[0.,0.]]*10,job_sha256='job')
    common=dict(index=0,root_slot=0,condition='off',task_id='task',selection_digest='selection')
    label=dict(phase='plan',step=10,condition='off',overlay_index=0)
    a=dict(common,commands=[[0.,0.]]*15,context_global_frames=[5,10,15],successor_global_frame=20,label=label)
    b=dict(common,commands=[[0.,0.]]*20,context_global_frames=[0,5,10],successor_global_frames=[15,20],labels=[{},label])
    native=dict(rows=[dict(common,plan=dict(commands=job['full10'],producer_job_sha256='job'))])
    return job,dict(rows=[common]),dict(rows=[common]),dict(bridge=dict(requests=[a])),dict(bridge=dict(requests=[b])),native


def test_identity_join_accepts_only_same_root_task_actions_and_time():
    args=identity_fixture();m.branch_identity(0,*args)
    for mutate in ('root','selection','raw','time','label'):
        a=deepcopy(args)
        if mutate=='root':a[4]['bridge']['requests'][0]['root_slot']=1
        if mutate=='selection':a[4]['bridge']['requests'][0]['selection_digest']='other'
        if mutate=='raw':a[4]['bridge']['requests'][0]['commands'][-1]=[.5,0]
        if mutate=='time':a[4]['bridge']['requests'][0]['context_global_frames']=[5,10,15]
        if mutate=='label':a[4]['bridge']['requests'][0]['labels'][1]['step']=5
        with pytest.raises(ValueError):m.branch_identity(0,*a)


def full_fixture(tmp_path,monkeypatch):
    """Small vectors/metadata doubles, not production tensors or empirical results."""
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'SOURCE_PINS',{})
    zero=np.zeros((224,224,3),np.uint8);one=zero.copy();one[0,0,0]=1
    schedule=m.saved.schedule();jobs=[];groups=[];zrows=[]
    for slot in range(4):
        groups.append([dict(phases={p:(np.zeros(2,np.float32),np.zeros(2,np.float32)) for p in m.PHASES},rgb=zero,
            state=[0]*7,label20_H1_H2=dict(bitwise_equal=True,maxabs=0),first_H2_inputs={'$tuple':[]},
            first_H2_frame15={'pixel_sha256':'fixed-prefix'}) for _ in range(41)])
        jobs.extend(dict(raw_suffix5=np.zeros((5,2),np.float32).tolist()) for _ in range(41))
    for n,s in enumerate(schedule):
        observed=n>=24;slot=s['slot'];base=41*slot;coord=s['coord']
        distances=[]
        for label,sgn in (('minus',-1),('plus',1)):
            i=s[label+'_index'];a=np.zeros(10,np.float32);a[coord]=np.float32(sgn*s['h']);distances.append(abs(float(a[coord])))
            jobs[i]['raw_suffix5']=a.reshape(5,2).tolist();g=groups[slot][i-base]
            real=np.array([float(a[coord]),0],np.float32) if observed else np.zeros(2,np.float32)
            pred=real.copy() if coord%2==0 else -real
            g['phases']=dict(H1_20=(pred,real),H2_15=(np.zeros(2,np.float32),np.zeros(2,np.float32)),H2_20=(pred,real))
            if observed and sgn==1:g['rgb']=one
            g['state']=[0.,0.,float(a[coord]),0.,0.,0.,0.]
        span=sum(distances)
        for key in ('minus_index','plus_index'):jobs[s[key]]['actual_pair_spacing']=span
        zrows.append(dict(s,pair_id=f'p{n}',missing=[],observed_response_zero=not observed,
            replay_to_original224_exact=True,integer_primitives_equal=not observed,surface512_equal=not observed,
            rgb224_equal=not observed,historical_callback_membership=None))
    fake=dict(model_before={'fixture_only':True})
    monkeypatch.setattr(m,'bundle',lambda *args:(fake,{}))
    monkeypatch.setattr(m,'collect_group',lambda slot,*args:groups[slot])
    z=dict(schema=m.saved.NAME+'.analysis',status='all80_exact_identity_joined',rows=zrows,
        authority=dict(jepa_report_sha256='h1-report',jepa_receipt_sha256='h1-receipt'),
        P1_admission=False,P5_admission=False,sufficient_original_cause_certified=False,historical_callback_membership=None,bindings={})
    def persist(name,v):
        p=tmp_path/'results'/name;p.write_bytes(m.saved.canonical(v));return name,m.sha(p)
    fp=persist('fd.json',dict(jobs=jobs));pr=persist('physical.json',dict(rows=[{}]*164,errors=[]))
    monkeypatch.setattr(m,'PINS',dict(fd_prepare=fp,physical_report=pr))
    zname,zh=persist('zero.json',z)
    args=dict(h1_output=str(tmp_path/'h1'),h1_report_sha='h1-report',h1_receipt='unused',h1_receipt_sha='h1-receipt',
        h2_output=str(tmp_path/'h2'),h2_report_sha='h2-report',h2_receipt='unused',h2_receipt_sha='h2-receipt',
        zero_analysis=str(tmp_path/'results'/zname),zero_sha=zh)
    for p in (tmp_path/'tests',tmp_path/'innerdocs'):p.mkdir()
    for p in (tmp_path/'tests'/('test_'+m.NAME+'.py'),tmp_path/'innerdocs'/(m.NAME+'.md')):p.write_bytes(b'fixture-not-production')
    return args,zrows,groups


def test_all80_fixed56_40_scales_groups_and_roundtrip_saved_writer(tmp_path,monkeypatch):
    args,zrows,groups=full_fixture(tmp_path,monkeypatch);r=m.analyze(**args)
    assert len(r['rows'])==80 and len(r['cross_scale'])==40 and sum(x['primary_H1_nonzero'] for x in r['rows'])==56
    assert r['arguments']==args and r['P1_admission'] is r['P5_admission'] is False
    assert sum(r['paired_direction_signs']['all80'].values())==80
    assert sum(r['paired_direction_signs']['fixed56'].values())==56
    assert all(v for v in r['rows'][0]['H2_before_perturbation_control'].values())
    assert [x['level']['global_raw_step'] for x in r['factor_tables'] if x['factor']=='raw_step']==[16,17,18,19,20]
    out=tmp_path/'results'/(m.NAME+'_RAM.json');receipt=m.write_new(r,out)
    loaded=m.saved.read(out,receipt['sha256'],{})
    assert loaded==r==m.analyze(**loaded['arguments'])
    with pytest.raises(ValueError):m.write_new(r,out)
    # A real schema mismatch cannot silently move membership or redefine56.
    zrows[0]['observed_response_zero']=False
    zfile=tmp_path/'results'/'zero.json';z=json.loads(zfile.read_bytes());z['rows']=zrows;zfile.write_bytes(m.saved.canonical(z));args['zero_sha']=m.sha(zfile)
    with pytest.raises(ValueError):m.analyze(**args)


def test_pixel_change_cannot_be_silently_called_quantized_zero(tmp_path,monkeypatch):
    args,zrows,groups=full_fixture(tmp_path,monkeypatch)
    groups[0][1]['rgb']=np.ones((224,224,3),np.uint8)
    with pytest.raises(ValueError,match='224 equality'):m.analyze(**args)


def test_missing_source_or_noncanonical_receipt_rejected(tmp_path):
    with pytest.raises(ValueError):m.saved.read(tmp_path/'missing','0'*64,{})
    p=tmp_path/'duplicate.json';p.write_bytes(b'{"x":1,"x":2}')
    with pytest.raises(ValueError):m.saved.read(p,m.sha(p),{})


def test_output_floor_and_foreign_prefix(tmp_path,monkeypatch):
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    with pytest.raises(ValueError):m.write_new(dict(status='fixture'),tmp_path/'results'/'foreign.json')
    import shutil
    monkeypatch.setattr(shutil,'disk_usage',lambda *a:NS(free=1))
    with pytest.raises(ValueError):m.write_new(dict(status='fixture'),tmp_path/'results'/(m.NAME+'_RAM.json'))


def test_bundle_real_metadata_sha_schema_and_counter_guards(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    source=tmp_path/'experiments/spectral_reachability'/(m.H1+'.py');source.parent.mkdir(parents=True);source.write_bytes(b'unit fixture source')
    monkeypatch.setitem(m.SOURCE_PINS,m.H1,m.sha(source))
    output=tmp_path/'run';output.mkdir()
    counts=dict(attempted=dict(fixture=1),returned=dict(fixture=1))
    p=dict(schema=m.H1+'.prepare',protocol=dict(pairs=m.saved.schedule(),caps=counts['attempted']),bindings={str(source):m.sha(source)})
    prep=tmp_path/'prepare.json';prep.write_bytes(m.saved.canonical(p))
    r=dict(schema=m.H1+'.report',status='fixed164_actual15_conditioned_factual_H1_recorded',counts=counts,
        rows=[{}]*164,pairs=[dict(s,missing=[]) for s in m.saved.schedule()],errors=[],source_errors=[],model_unchanged=True,
        model_before={},model_after_predictions={},model_after={},P1_admission=False,P5_admission=False,protocol=p['protocol'],
        execution_binding=dict(prepared=str(prep),prepare_sha256=m.sha(prep)))
    a=dict(schema=m.H1+'.consume',status='all164_factual_H1_and80_pairs_verified',requests=164,pairs=80,
        counts=deepcopy(counts),P1_admission=False,P5_admission=False)
    report=output/'report.json';receipt=tmp_path/'receipt.json'
    def call():
        report.write_bytes(m.saved.canonical(r));a['report_sha256']=m.sha(report);receipt.write_bytes(m.saved.canonical(a))
        return m.bundle(output,m.sha(report),receipt,m.sha(receipt),'H1',{})
    assert call()[0]==r
    r['model_unchanged']=False
    with pytest.raises(ValueError):call()
    r['model_unchanged']=True;r['pairs'][0]['condition']='other'
    with pytest.raises(ValueError):call()


def test_collect_real_array_shapes_and_saved_scores_without_model(tmp_path,monkeypatch):
    # Full native patch dimensions, but deterministic RAM protocol arrays, no empirical result.
    pred1=np.broadcast_to(np.float32(2),(1,3,256,384));pred15=np.zeros_like(pred1);pred20=pred1+1
    label1=np.broadcast_to(np.float32(1),(1,256,384));label2=np.concatenate([label1*0,label1])
    arrays=dict(h1pred=pred1,h2pred15=pred15,h2pred=pred20,label1=label1,label2=label2)
    def desc(key):return {'fixture_array':key}
    rows1=[dict(prediction=desc('h1pred'),label_patch=desc('label1'),score=dict(successor_MSE=1)) for _ in range(41)]
    rows2=[dict(prediction=desc('h2pred'),prediction15=desc('h2pred15'),label_patch=desc('label2'),
        score=dict(successor_MSE=4,prediction15_MSE=0)) for _ in range(41)]
    h1=dict(rows=rows1,artifacts={});h2=dict(rows=rows2,artifacts={},calls=[dict(inputs={'fixture':True})]*164)
    monkeypatch.setattr(m,'branch_identity',lambda *a:({'label':{}},{'labels':[{},{}]}))
    monkeypatch.setattr(m,'load_array',lambda directory,rec,*args:arrays[rec['fixture_array']])
    monkeypatch.setattr(m,'load_rgb',lambda *a:np.zeros((224,224,3),np.uint8))
    group=m.collect_group(0,h1,h2,{}, {},[{}]*41,dict(rows=[dict(final_state=[0]*7)]*41),tmp_path/'h1',tmp_path/'h2',{})
    assert len(group)==41 and group[0]['label20_H1_H2']['bitwise_equal']
    assert m.response(*[group[k]['phases']['H2_20'][axis] for k,axis in ((0,0),(1,0),(0,1),(1,1))])['observed_exact_zero']
    rows2[0]['score']['successor_MSE']=3
    with pytest.raises(ValueError,match='endpoint MSE'):m.collect_group(0,h1,h2,{}, {},[{}]*41,
        dict(rows=[dict(final_state=[0]*7)]*41),tmp_path/'h1',tmp_path/'h2',{})
