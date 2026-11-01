"""Model-free RAM and temporary saved-byte fixtures; never scientific evidence."""
import ast
from copy import deepcopy
import gzip
import io
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_common_resource_rgb_scalar_summary_20261029_v1 as m
import test_agent_p2_common_resource_rgb_20261029_v1 as rgb


def fixture(monkeypatch,stops=(15,15,15,15,15,14),missing=()):
    p,r,store=rgb.execute(monkeypatch,stops=stops,missing=missing)
    pp,ps,mp=rgb.fixture(stops,missing)
    source=dict(physical_prepare=pp,physical_strict=ps,model_prepare=mp,physical_output='/RAM/physical')
    rows=[rgb.m.analyze(spec,row,store.load) for spec,row in zip(p['rows'],r['rows'],strict=True)]
    strict=dict(schema=m.RGB+'.consume',status='all18_common_resource_RGB_slots_verified_NOT_science',rows=rows,
        groups=rgb.m.grouped(rows),pairs=rgb.m.paired(rows),fixed_denominator=18,plan_denominator=6,
        group_denominator=6,pair_denominator=18,counts=deepcopy(r['counts']),
        missing_slots=[x['index'] for x in rows if x['latent_missing']],
        real_pairs_available=sum(not x['real_pair_missing'] for x in rgb.m.paired(rows)),
        P1_admission=False,P2_admission=False,P2_mechanism=False,physical_release=False)
    return p,r,strict,source,store


def test_hand_calculable_independent_mse_and_error_identity():
    d=m.decomposition(np.full(192,3.),np.full(192,2.),np.ones(192))
    assert (d['model_Q'],d['real_Q'],d['prediction_error_MSE'])==(4.,1.,1.)
    assert d['actual_goal_error_cross_prediction_error']==2. and d['model_minus_real_Q']==3.
    p=np.zeros(192);p[0]=12.;assert m.mse(p,np.zeros(192))==144/192
    assert m.decomposition(p,p,p)['identity_rounding_residual']==0
    assert m.decomposition(np.zeros(192),np.zeros(192),np.zeros(192))['identity_rounding_bound']==0


@pytest.mark.parametrize('scale',[1e-20,1.,1e20])
def test_roundoff_bound_does_not_change_sign_threshold(scale):
    g=np.linspace(-1,1,192)*scale;a=g+scale/17;p=a+np.linspace(-.3,.3,192)*scale
    d=m.decomposition(p,a,g)
    assert abs(d['identity_rounding_residual'])<=d['identity_rounding_bound']
    assert d['model_minus_real_Q']==m.mse(p,g)-m.mse(a,g)
    assert m.sign(1e-300)=='positive' and m.sign(-1e-300)=='negative' and m.sign(-0.)=='zero'


def test_native_velocity_raw_angle_and_wrapped_axis_are_distinct():
    goal=[0.,0.,0.,0.,0.,3.,4.];state=[3.,4.,0.,12.,4*np.pi+2,0.,0.]
    native,axes=m.native_scalars(goal,state)
    assert axes['agent_xy_squared']==25 and axes['object_xy_squared']==144 and axes['velocity_squared']==25
    assert axes['joint_xy_error']==13 and axes['velocity_error']==5
    assert axes['negative_native_angle'] and axes['original_success'] and not axes['wrapped_success_diagnostic']
    assert axes['native7_distance']>axes['pose5_distance'] and native['native7_reward']==-np.linalg.norm(np.asarray(state)-goal)


@pytest.mark.parametrize('stops,missing',[
    ((15,15,15,15,15,14),()),((15,)*6,()),((0,)*6,()),((4,9,14,14,0,15),()),
    ((15,)*6,(0,4)),((15,)*6,tuple(range(6)))])
def test_all18_counterbalanced_slots_and_pairs_with_nulls(monkeypatch,stops,missing):
    p,r,s,source,store=fixture(monkeypatch,stops,missing)
    rows=m.eighteen_rows(p,r,s,source,store.load);groups,pairs=m.paired(rows,s)
    assert [(x['root_slot'],x['arm'],x['checkpoint']) for x in rows]==list(m.ORDER)
    assert len(rows)==len(pairs)==18 and len(groups)==6
    assert [x['index'] for x in rows if x['missing']]==s['missing_slots']
    for pair in pairs:
        a,b=rows[pair['left_index']],rows[pair['right_index']]
        assert (a['arm'],b['arm']) in m.DIRECTIONS
        assert pair['real_pair_missing']==(a['missing'] or b['missing'])
        if not pair['real_pair_missing']:
            d=pair['scalar_error_identity'];assert abs(d['identity_rounding_residual'])<=d['identity_rounding_bound']
            assert pair['latent']['real_Q']['delta']==a['real_Q']-b['real_Q']
    if stops==(15,15,15,15,15,14) and not missing:
        assert rows[17]['missing'] and rows[17]['model_Q'] is not None and rows[17]['real_Q'] is None
        assert sum(x['real_pair_missing'] for x in pairs)==2
    for i in missing:
        assert all(x['model_Q'] is None for x in rows[3*i:3*i+3])


@pytest.mark.parametrize('a,b,want',[(None,1,'missing'),(0,0,'both_tie'),(0,1,'left_tie'),(1,0,'right_tie'),
    (-1,-2,'strict_same'),(1,-1,'strict_reversal')])
def test_ties_missing_and_strict_reversal(a,b,want):assert m.relation(a,b)==want


@pytest.mark.parametrize('fault',['drop','order','goal','prediction','realQ','gap','error','nativeGoal','velocity','wrapped',
    'reward','success','checkpoint','witness','raw','history','task','counts','missing','float32','projector',
    'pairDelta','pairOrder','pairNative','group','Qcodec'])
def test_exact_source_math_and_pair_mutations(monkeypatch,fault):
    p,r,s,source,store=fixture(monkeypatch)
    if fault=='drop':s['rows'].pop()
    elif fault=='order':s['rows'].reverse()
    elif fault=='goal':p['rows'][0]['goal192'][0]+=.1
    elif fault=='prediction':p['rows'][0]['predicted192'][0]+=.1
    elif fault=='realQ':s['rows'][0]['real_Q']+=.1
    elif fault=='gap':s['rows'][0]['model_minus_real_Q']+=.1
    elif fault=='error':s['rows'][0]['prediction_vs_encoded_MSE']+=.1
    elif fault=='nativeGoal':p['rows'][0]['native_goal'][6]+=.1
    elif fault=='velocity':
        for v in (p['rows'][0],r['rows'][0],s['rows'][0]):v['native_metrics']['velocity_error']+=1
    elif fault=='wrapped':s['rows'][0]['native_metrics']['angle_wrapped_error']+=.1
    elif fault=='reward':s['rows'][0]['response']['reward']+=1
    elif fault=='success':s['rows'][0]['response']['terminated']=not s['rows'][0]['response']['terminated']
    elif fault=='checkpoint':source['physical_strict']['rows'][0]['checkpoints'][1]['state'][0]+=1
    elif fault=='witness':source['physical_prepare']['model_witnesses'][0]['contexts'][0]['latent_history'][-1][0]+=.1
    elif fault=='raw':r['rows'][0]['raw_float32_bits']='00'
    elif fault=='history':r['rows'][0]['causal_raw10_bits']='00'
    elif fault=='task':s['rows'][0]['task_sha256']='f'*64
    elif fault=='counts':s['counts']['encoder']['completed']-=1
    elif fault=='missing':s['rows'][0]['latent_missing']=True
    elif fault=='float32':
        p['rows'][0]['predicted192'][0]+=1e-12;r['rows'][0]['predicted192'][0]+=1e-12
    elif fault=='projector':r['rows'][0]['modules'][-1]['output']=store.array('bad_projector',np.ones((1,192),np.float32))
    elif fault=='pairDelta':s['pairs'][0]['model_Q']['delta']+=1
    elif fault=='pairOrder':s['pairs'].reverse()
    elif fault=='pairNative':s['pairs'][0]['physical_metrics']['delta']['native7_distance']+=1
    elif fault=='group':s['groups'][0]['indices']['grasp']=0
    else:p['rows'][0]['Q_sha256']=m.bhash(m.raw(source['model_prepare']['slots'][0]['q']))
    with pytest.raises((ValueError,KeyError)):
        rows=m.eighteen_rows(p,r,s,source,store.load);m.paired(rows,s)


def npy(directory,a,tail=b''):
    stream=io.BytesIO();np.save(stream,a,allow_pickle=False);body=gzip.compress(stream.getvalue()+tail,mtime=0)
    path=directory/'real.npy.gz';path.write_bytes(body)
    rec=dict(file=path.name,sha256=m.bhash(body),array_sha256=m.bhash(a.tobytes()),shape=list(a.shape),dtype=str(a.dtype))
    return rec,{path.name:m.bhash(body)}


def test_original_npy_signed_zero(tmp_path):
    a=np.zeros(192,np.float32);a[0]=-0.;rec,hashes=npy(tmp_path,a)
    assert m.load_npy(tmp_path,rec,hashes).tobytes()==a.tobytes()


@pytest.mark.parametrize('bad',['fileSHA','arraySHA','shape','dtype','trailing','float64','path','bomb','concat'])
def test_npy_mutations(tmp_path,bad):
    a=np.zeros(192,np.float64 if bad=='float64' else np.float32)
    rec,hashes=npy(tmp_path,a,tail=b'trailing' if bad=='trailing' else b'')
    if bad=='fileSHA':rec['sha256']='b'*64
    elif bad=='arraySHA':rec['array_sha256']='b'*64
    elif bad=='shape':rec['shape']=[1,192]
    elif bad=='dtype':rec['dtype']='uint8'
    elif bad=='path':rec['file']='../real.npy.gz'
    elif bad in ('bomb','concat'):
        body=gzip.compress(b'X'*(m.ARRAY_CAP+1)) if bad=='bomb' else (tmp_path/'real.npy.gz').read_bytes()+gzip.compress(b'')
        (tmp_path/'real.npy.gz').write_bytes(body);rec['sha256']=hashes['real.npy.gz']=m.bhash(body)
    with pytest.raises((ValueError,KeyError)):m.load_npy(tmp_path,rec,hashes)


def test_no_producer_math_or_dynamic_imports():
    tree=ast.parse(Path(m.__file__).read_text());modules=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Import):modules.extend(a.name for a in n.names)
        elif isinstance(n,ast.ImportFrom):modules.append(n.module)
    assert all(x=='numpy' or x.split('.')[0] in {'__future__','argparse','copy','hashlib','io','json','os','pathlib','zlib'} for x in modules)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'eval','exec','__import__'} for n in ast.walk(tree))


def test_exclusive_writer_and_strict_json(tmp_path):
    path=tmp_path/(m.NAME+'_RAM.json');m.write(path,dict(RAM=True))
    assert m.read(path)==dict(RAM=True)
    with pytest.raises(ValueError):m.write(path,{})
    path=tmp_path/'bad.json';path.write_bytes(b'{"x":1,"x":2}')
    with pytest.raises(ValueError):m.read(path)
    path.write_bytes(b'{"x":NaN}')
    with pytest.raises(ValueError):m.read(path)


def saved_corpus(tmp_path,monkeypatch,stops=(15,15,15,15,15,14),missing=()):
    """Actual schema, synthetic bytes only; no claim of native strict/PNG evidence."""
    p,r,s,source,store=fixture(monkeypatch,stops,missing)
    modeldir=tmp_path/'model';physicaldir=tmp_path/'physical';rgbdir=tmp_path/'RGB'
    for d in (modeldir,physicaldir,rgbdir):d.mkdir()
    def dump(path,obj):path.write_bytes(m.raw(obj));return m.sha(path)
    pp,ps,mp=(source[k] for k in ('physical_prepare','physical_strict','model_prepare'))
    bindings={}
    for i,spec in enumerate(p['rows']):
        if spec['source'] is None:continue
        cp=spec['source'];path=physicaldir/cp['png'];path.write_bytes(b'RAM-only placeholder PNG bytes')
        cp['path']=str(path);cp['file_sha256']=m.sha(path);bindings[str(path)]=m.sha(path)
        r['rows'][i]['source']=deepcopy(cp);s['rows'][i]['source']=deepcopy(cp)
        for c in ps['rows'][spec['plan_index']]['checkpoints']:
            if c['step']==spec['checkpoint']:c['file_sha256']=cp['file_sha256']
    modelprep=modeldir/'prepare.json';modelprep_sha=dump(modelprep,mp)
    mr=dict(schema=m.MODEL+'.report',status='records_sealed',prepared=str(modelprep),prepare_sha256=modelprep_sha)
    modelreport=modeldir/'report.json';model_sha=dump(modelreport,mr)
    ms=dict(schema=m.MODEL+'.consume',status='all6_common_cap_records_verified_not_admitted',report_sha256=model_sha,
        plans=[x['planner_plan'] if x else None for x in pp['plans']])
    modelstrict=modeldir/'strict.json';model_strict_sha=dump(modelstrict,ms)
    pp['arguments'].update(planner_output=str(modeldir),planner_report_sha=model_sha,
        planner_receipt=str(modelstrict),planner_receipt_sha=model_strict_sha)
    pp['bindings']={str(path):m.sha(path) for path in (modelprep,modelreport,modelstrict)}
    physprep=physicaldir/'prepare.json';physprep_sha=dump(physprep,pp)
    physicalreport=physicaldir/'report.json'
    physical_sha=dump(physicalreport,dict(schema=m.PHYSICAL+'.report',status='common_resource6_native_recorded_not_admitted',
        rows=ps['rows'],execution_binding=dict(prepared=str(physprep),prepare_sha256=physprep_sha)))
    ps['physical_report_sha256']=physical_sha;physicalstrict=physicaldir/'strict.json';physical_strict_sha=dump(physicalstrict,ps)
    for path in (modelprep,modelreport,modelstrict,physprep,physicalreport,physicalstrict):bindings[str(path)]=m.sha(path)
    p.update(schema=m.RGB+'.prepare',expected_model_hash=mp['expected_model_hash'],bindings=bindings,
        physical_report_sha256=physical_sha,physical_receipt_sha256=physical_strict_sha,
        runtime_report_sha256=model_sha,runtime_receipt_sha256=model_strict_sha,runtime_prepare_sha256=modelprep_sha,
        arguments=dict(physical_output=str(physicaldir),physical_report_sha=physical_sha,
            physical_receipt=str(physicalstrict),physical_receipt_sha=physical_strict_sha,physical_source_sha='f'*64))
    prep=tmp_path/'rgb_prepare.json';prep_sha=dump(prep,p)
    for name,body in store.files.items():(rgbdir/name).write_bytes(body)
    dump(rgbdir/'binding.json',dict(prepared=str(prep),prepare_sha256=prep_sha));dump(rgbdir/'initial_roster.json',p['rows'])
    for i,event in enumerate(store.events):dump(rgbdir/f'event_{i:04}.json',event)
    seal_sha=dump(rgbdir/'native_seal.json',{k:r[k] for k in ('rows','modules','guards','blocked','errors')})
    r.update(schema=m.RGB+'.report',status='common_resource_RGB_encoded_NOT_science',prepared=str(prep),prepare_sha256=prep_sha,
        source_verified=True,model_unchanged=True,model_hash_before=mp['expected_model_hash'],model_hash_after=mp['expected_model_hash'],
        native_seal_sha256=seal_sha,P1_admission=False,P2_admission=False,P2_mechanism=False,physical_release=False,
        artifact_sha256={f.name:m.sha(f) for f in rgbdir.iterdir()})
    report=rgbdir/'report.json';report_sha=dump(report,r)
    s.update(report_sha256=report_sha,physical_report_sha256=physical_sha,physical_receipt_sha256=physical_strict_sha,
        runtime_report_sha256=model_sha,runtime_receipt_sha256=model_strict_sha,runtime_prepare_sha256=modelprep_sha)
    strict=tmp_path/'rgb_strict.json';strict_sha=dump(strict,s)
    return dict(prepared=str(prep),prepare_sha=prep_sha,report=str(report),report_sha=report_sha,strict=str(strict),strict_sha=strict_sha)


@pytest.mark.parametrize('stops,missing',[
    ((15,15,15,15,15,14),()),((0,)*6,()),((15,)*6,(0,4))])
def test_full_saved_byte_chain_and_original_npy_analysis(tmp_path,monkeypatch,stops,missing):
    args=saved_corpus(tmp_path,monkeypatch,stops,missing);out=m.analyze(**args)
    assert out['fixed_denominator']==out['pair_denominator']==18 and len(out['groups'])==6
    assert sum(out['model_real_sign_census'].values())==18
    if not missing and stops[-1]==14:
        assert out['missing_slots']==[17] and out['real_pairs_available']==16
    for k in ('prepared','report','strict'):
        assert out['bindings'][args[k]]==args[k.replace('prepared','prepare')+'_sha']
    path=tmp_path/(m.NAME+'_RAM_summary.json');m.write(path,out)
    m.same(m.read(path),m.analyze(**args),'complete repeat saved scalar reconstruction')


@pytest.mark.parametrize('fault',['prepare_sha','report_sha','strict_sha','source','array','extra','runtimeReceipt'])
def test_saved_binding_mutations(tmp_path,monkeypatch,fault):
    args=saved_corpus(tmp_path,monkeypatch)
    if fault.endswith('_sha'):args[fault]='e'*64
    elif fault=='source':(tmp_path/'model'/'prepare.json').write_bytes(b'{}')
    elif fault=='array':next((Path(args['report']).parent).glob('*latent.npy.gz')).write_bytes(b'bad')
    elif fault=='runtimeReceipt':(tmp_path/'model'/'strict.json').write_bytes(b'{}')
    else:(Path(args['report']).parent/'extra.json').write_bytes(b'{}')
    with pytest.raises(ValueError):m.analyze(**args)
