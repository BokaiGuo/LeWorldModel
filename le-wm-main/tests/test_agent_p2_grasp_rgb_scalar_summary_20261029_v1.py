"""Synthetic RAM and temporary saved-byte tests; not production scalar evidence."""
import ast
from copy import deepcopy
import gzip
import io
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_grasp_rgb_scalar_summary_20261029_v1 as m
import test_agent_p2_history_grasp_rgb_20261029_v1 as rgb


def fixture(monkeypatch,stops=(15,15)):
    p,r,store=rgb.execute(monkeypatch,stops=stops)
    pp,ps,mp,results=rgb.fixture(stops)
    source=dict(physical_prepare=pp,physical_strict=ps,model_prepare=mp,root_results=results,physical_output='/RAM/physical')
    rows=[]
    for spec,row in zip(p['rows'],r['rows'],strict=True):
        present=spec['source'] is not None;z=store.load(row['latent']) if present else None
        fields=('index','plan_index','root_slot','condition','arm','checkpoint','task_id','task_sha256',
            'selection_digest','model_selection_digest','source','native_metrics','response','causal_raw10_bits','physical_status','actual_returned')
        out={k:deepcopy(spec[k]) for k in fields}
        out.update(model_Q=spec['model_Q'],real_Q=None if z is None else m.mse(z,spec['goal192']),
            model_minus_real_Q=None if z is None else spec['model_Q']-m.mse(z,spec['goal192']),
            prediction_vs_encoded_MSE=None if z is None else m.mse(z,spec['predicted192']),latent_missing=not present)
        rows.append(out)
    strict=dict(schema=m.RGB+'.consume',status='all6_grasp_RGB_slots_verified_NOT_science',rows=rows,
        fixed_denominator=6,model_real_comparison_denominator=6,plan_denominator=2,counts=deepcopy(r['counts']),
        missing_slots=[x['index'] for x in rows if x['latent_missing']],real_comparisons_available=sum(not x['latent_missing'] for x in rows),
        P1_admission=False,P2_admission=False,P2_mechanism=False,physical_release=False)
    return p,r,strict,source,store


def test_hand_computable_mean_and_quadratic_identity():
    p=np.full(192,3.);a=np.full(192,2.);g=np.ones(192)
    x=m.decomposition(p,a,g)
    assert (x['model_Q'],x['real_Q'],x['prediction_error_MSE'])==(4.,1.,1.)
    assert x['actual_goal_error_cross_prediction_error']==2. and x['model_minus_real_Q']==3.
    assert x['identity_rounding_residual']==0 and not x['scientific_tolerance_changed']
    p=np.zeros(192);p[0]=12.
    assert m.mse(p,np.zeros(192))==144/192
    assert m.decomposition(p,p,p)['identity_rounding_residual']==0
    assert m.decomposition(np.zeros(192),np.zeros(192),np.zeros(192))['identity_rounding_bound']==0


@pytest.mark.parametrize('scale',[1e-20,1.,1e20])
def test_cancellation_rounding_bound_is_not_zero_gap_threshold(scale):
    g=np.linspace(-1,1,192)*scale;a=g+scale/17;p=a+np.linspace(-.3,.3,192)*scale
    x=m.decomposition(p,a,g)
    assert abs(x['identity_rounding_residual'])<=x['identity_rounding_bound']
    assert x['model_minus_real_Q']==m.mse(p,g)-m.mse(a,g)


def test_native_axes_velocity_and_angle_quirk_not_replaced():
    goal=[0.,0.,0.,0.,0.,3.,4.];state=[3.,4.,0.,12.,4*np.pi+2,0.,0.]
    original,axes=m.native_scalars(goal,state)
    assert axes['agent_xy_squared']==25 and axes['object_xy_squared']==144 and axes['velocity_squared']==25
    assert axes['joint_xy_error']==13 and axes['velocity_error']==5
    assert axes['native7_squared']==sum(axes['squared_coordinate_contributions'])
    assert axes['native7_distance']>axes['pose5_distance']
    assert axes['negative_native_angle'] and axes['original_success'] and not axes['wrapped_success_diagnostic']
    assert original['native7_reward']==-np.linalg.norm(np.asarray(state)-goal)
    assert axes['object_only_success'] is None


@pytest.mark.parametrize('stops',[(15,15),(15,14),(0,0),(4,9)])
def test_all_six_rows_keep_time_success_and_missing(monkeypatch,stops):
    p,r,s,source,store=fixture(monkeypatch,stops)
    out=m.six_rows(p,r,s,source,store.load)
    assert [(x['root_slot'],x['checkpoint']) for x in out]==list(m.ORDER)
    assert [x['index'] for x in out if x['missing']]==s['missing_slots']
    assert all(x['decomposition'] is x['native_axes'] is None for x in out if x['missing'])
    for x in out:
        if not x['missing']:assert x['decomposition']['real_Q']==s['rows'][x['index']]['real_Q']


@pytest.mark.parametrize('fault',['drop','order','goal','prediction','strictMSE','gap','error','nativeGoal','velocity','wrapped',
    'reward','success','checkpoint','result','raw','history','task','counts','missing','float32','projected'])
def test_independent_scalar_and_source_mutations_rejected(monkeypatch,fault):
    p,r,s,source,store=fixture(monkeypatch)
    if fault=='drop':s['rows'].pop()
    elif fault=='order':s['rows'].reverse()
    elif fault=='goal':p['rows'][0]['goal192'][0]+=.1
    elif fault=='prediction':p['rows'][0]['predicted192'][0]+=.1
    elif fault=='strictMSE':s['rows'][0]['real_Q']+=.1
    elif fault=='gap':s['rows'][0]['model_minus_real_Q']+=.1
    elif fault=='error':s['rows'][0]['prediction_vs_encoded_MSE']+=.1
    elif fault=='nativeGoal':p['rows'][0]['native_goal'][6]+=.1
    elif fault=='velocity':p['rows'][0]['native_metrics']['velocity_error']+=.1
    elif fault=='wrapped':s['rows'][0]['native_metrics']['angle_wrapped_error']+=.1
    elif fault=='reward':s['rows'][0]['response']['reward']+=.1
    elif fault=='success':s['rows'][0]['response']['terminated']=not s['rows'][0]['response']['terminated']
    elif fault=='checkpoint':source['physical_strict']['rows'][0]['checkpoints'][1]['state'][0]+=.1
    elif fault=='result':source['root_results'][0]['whole'][0]['latent_history'][-1][0]+=.1
    elif fault=='raw':r['rows'][0]['raw_float32_bits']='00'
    elif fault=='history':r['rows'][0]['causal_raw10_bits']='00'
    elif fault=='task':s['rows'][0]['task_sha256']='f'*64
    elif fault=='counts':s['counts']['encoder']['completed']-=1
    elif fault=='missing':s['rows'][0]['latent_missing']=True
    elif fault=='float32':
        p['rows'][0]['predicted192'][0]+=1e-12;r['rows'][0]['predicted192'][0]+=1e-12
    else:
        r['rows'][0]['modules'][-1]['output']=store.array('bad_projected',np.ones((1,192),np.float32))
    with pytest.raises((ValueError,KeyError)):m.six_rows(p,r,s,source,store.load)


def npy(directory,a,tail=b'',dtype=None):
    stream=io.BytesIO();np.save(stream,a,allow_pickle=False);body=gzip.compress(stream.getvalue()+tail,mtime=0)
    path=directory/'real.npy.gz';path.write_bytes(body)
    rec=dict(file=path.name,sha256=m.bhash(body),array_sha256=m.bhash(a.tobytes()),shape=list(a.shape),dtype=str(a.dtype) if dtype is None else dtype)
    return rec,{path.name:m.bhash(body)}


def test_independent_original_gzip_npy_reader_and_signed_zero(tmp_path):
    a=np.zeros(192,np.float32);a[0]=-0.
    rec,hashes=npy(tmp_path,a);out=m.load_npy(tmp_path,rec,hashes)
    assert out.tobytes()==a.tobytes() and np.signbit(out[0])


@pytest.mark.parametrize('bad',['fileSHA','arraySHA','shape','dtype','trailing','float64','object','symlink','path','bomb','concat'])
def test_npy_mutations_and_bounds(tmp_path,bad):
    a=np.zeros(192,np.float32)
    if bad=='float64':a=a.astype(np.float64)
    elif bad=='object':
        stream=io.BytesIO();np.save(stream,np.array([object()],object),allow_pickle=True)
        body=gzip.compress(stream.getvalue());(tmp_path/'real.npy.gz').write_bytes(body)
        rec=dict(file='real.npy.gz',sha256=m.bhash(body),array_sha256='b'*64,shape=[1],dtype='object')
        with pytest.raises(ValueError):m.load_npy(tmp_path,rec,{'real.npy.gz':m.bhash(body)})
        return
    rec,hashes=npy(tmp_path,a,tail=b'trailing' if bad=='trailing' else b'')
    if bad=='fileSHA':rec['sha256']='b'*64
    elif bad=='arraySHA':rec['array_sha256']='b'*64
    elif bad=='shape':rec['shape']=[1,192]
    elif bad=='dtype':rec['dtype']='uint8'
    elif bad=='path':rec['file']='../real.npy.gz'
    elif bad=='symlink':
        (tmp_path/'link.npy.gz').symlink_to(tmp_path/'real.npy.gz');rec['file']='link.npy.gz';hashes['link.npy.gz']=rec['sha256']
    elif bad in ('bomb','concat'):
        body=gzip.compress(b'X'*(m.ARRAY_CAP+1)) if bad=='bomb' else (tmp_path/'real.npy.gz').read_bytes()+gzip.compress(b'')
        (tmp_path/'real.npy.gz').write_bytes(body);rec['sha256']=hashes['real.npy.gz']=m.bhash(body)
    with pytest.raises((ValueError,KeyError)):m.load_npy(tmp_path,rec,hashes)


def test_source_has_no_producer_math_or_dynamic_imports():
    tree=ast.parse(Path(m.__file__).read_text())
    modules=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Import):modules.extend(a.name for a in n.names)
        elif isinstance(n,ast.ImportFrom):modules.append(n.module)
    assert all(x=='numpy' or x.split('.')[0] in {'__future__','argparse','copy','hashlib','io','json','os','pathlib','zlib'} for x in modules)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'eval','exec','__import__'} for n in ast.walk(tree))


def test_strict_json_and_exclusive_summary_writer(tmp_path):
    p=tmp_path/(m.NAME+'_RAM.json');m.write(p,dict(RAM=True))
    assert m.read(p)==dict(RAM=True)
    with pytest.raises(ValueError):m.write(p,{})
    p=tmp_path/'bad.json';p.write_bytes(b'{"x":1,"x":2}')
    with pytest.raises(ValueError):m.read(p)
    p.write_bytes(b'{"x":NaN}')
    with pytest.raises(ValueError):m.read(p)


@pytest.mark.parametrize('fault',['wrapperVelocity','originalAngle','reward','twoNPYs'])
def test_independent_math_rejects_consistently_copied_bad_scalars(monkeypatch,fault):
    p,r,s,source,store=fixture(monkeypatch)
    if fault=='wrapperVelocity':
        for v in (p['rows'][0],r['rows'][0],s['rows'][0]):v['native_metrics']['velocity_error']+=1
    elif fault=='originalAngle':
        for v in (p['rows'][0],r['rows'][0],s['rows'][0]):
            v['native_metrics']['native']['native_angle_term']+=1
            v['response']['native_metrics']['native_angle_term']+=1
        source['physical_strict']['rows'][0]['responses'][0]['native_metrics']['native_angle_term']+=1
    elif fault=='reward':
        for v in (p['rows'][0],r['rows'][0],s['rows'][0]):v['response']['reward']+=1
        source['physical_strict']['rows'][0]['responses'][0]['reward']+=1
    else:
        z=np.full(192,123,np.float32)
        r['rows'][0]['latent']=store.array('bad_real_consistent',z)
        r['rows'][0]['modules'][-1]['output']=store.array('bad_projector_consistent',z[None])
    with pytest.raises(ValueError):m.six_rows(p,r,s,source,store.load)


def saved_corpus(tmp_path,monkeypatch,stops=(15,15)):
    """Temporary synthetic bytes exercising real reader; not genuine producer strict evidence."""
    p,r,s,source,store=fixture(monkeypatch,stops)
    modeldir=tmp_path/'model';physicaldir=tmp_path/'physical';rgbdir=tmp_path/'RGB'
    for directory in (modeldir,physicaldir,rgbdir):directory.mkdir()
    def dump(path,obj):path.write_bytes(m.raw(obj));return m.sha(path)
    pp=source['physical_prepare'];ps=source['physical_strict'];mp=source['model_prepare']
    bindings={}
    for i,spec in enumerate(p['rows']):
        if spec['source'] is None:continue
        cp=spec['source'];path=physicaldir/cp['png'];path.write_bytes(b'RAM-only placeholder PNG bytes')
        cp['path']=str(path);cp['file_sha256']=m.sha(path);bindings[str(path)]=m.sha(path)
        r['rows'][i]['source']=deepcopy(cp);s['rows'][i]['source']=deepcopy(cp)
        actual=ps['rows'][spec['root_slot']]
        for c in actual['checkpoints']:
            if c['step']==spec['checkpoint']:c['file_sha256']=cp['file_sha256']
        for c in actual['trajectory']:
            if c['step']==spec['checkpoint']:c['file_sha256']=cp['file_sha256']
    modelprep=modeldir/'prepare.json';modelprep_sha=dump(modelprep,mp)
    artifacts={f'root{i}.json':dump(modeldir/f'root{i}.json',result) for i,result in enumerate(source['root_results'])}
    mr=dict(schema=m.MODEL+'.report',status='normalized_model_variant_complete_not_admitted',prepared=str(modelprep),
        prepare_sha256=modelprep_sha,artifact_sha256=artifacts,
        rows=[dict(root_slot=i,result_file=f'root{i}.json') for i in (0,1)])
    modelreport=modeldir/'report.json';modelreport_sha=dump(modelreport,mr)
    pp['arguments'].update(planner_output=str(modeldir),planner_report_sha=modelreport_sha)
    physprep=physicaldir/'prepare.json';physprep_sha=dump(physprep,pp)
    physicalreport=physicaldir/'report.json'
    physical_sha=dump(physicalreport,dict(rows=ps['rows'],execution_binding=dict(prepared=str(physprep),prepare_sha256=physprep_sha)))
    ps['physical_report_sha256']=physical_sha;physical_receipt=physicaldir/'strict.json';physical_receipt_sha=dump(physical_receipt,ps)
    for path in (modelprep,modelreport,physprep,physicalreport,physical_receipt,*[modeldir/n for n in artifacts]):bindings[str(path)]=m.sha(path)
    p.update(schema=m.RGB+'.prepare',expected_model_hash=mp['expected_model_hash'],bindings=bindings,
        physical_report_sha256=physical_sha,physical_receipt_sha256=physical_receipt_sha,
        arguments=dict(physical_output=str(physicaldir),physical_report_sha=physical_sha,
            physical_receipt=str(physical_receipt),physical_receipt_sha=physical_receipt_sha,physical_source_sha='f'*64))
    prep=tmp_path/'rgb_prepare.json';prep_sha=dump(prep,p)
    for name,body in store.files.items():(rgbdir/name).write_bytes(body)
    dump(rgbdir/'binding.json',dict(prepared=str(prep),prepare_sha256=prep_sha))
    dump(rgbdir/'initial_roster.json',p['rows'])
    for i,event in enumerate(store.events):dump(rgbdir/f'event_{i:04}.json',event)
    seal_sha=dump(rgbdir/'native_seal.json',{k:r[k] for k in ('rows','modules','guards','blocked','errors')})
    r.update(schema=m.RGB+'.report',status='grasp_RGB_encoded_NOT_science',prepared=str(prep),prepare_sha256=prep_sha,
        source_verified=True,model_unchanged=True,model_hash_before=mp['expected_model_hash'],model_hash_after=mp['expected_model_hash'],
        native_seal_sha256=seal_sha,P1_admission=False,P2_admission=False,P2_mechanism=False,physical_release=False,
        artifact_sha256={f.name:m.sha(f) for f in rgbdir.iterdir()})
    report=rgbdir/'report.json';report_sha=dump(report,r)
    s.update(report_sha256=report_sha,physical_report_sha256=physical_sha,physical_receipt_sha256=physical_receipt_sha)
    strict=tmp_path/'rgb_strict.json';strict_sha=dump(strict,s)
    return dict(prepared=str(prep),prepare_sha=prep_sha,report=str(report),report_sha=report_sha,strict=str(strict),strict_sha=strict_sha)


@pytest.mark.parametrize('stops',[(15,15),(15,14),(0,0)])
def test_real_saved_provenance_and_npy_full_analysis(tmp_path,monkeypatch,stops):
    args=saved_corpus(tmp_path,monkeypatch,stops)
    out=m.analyze(**args)
    assert out['fixed_denominator']==6 and len(out['rows'])==6 and not out['producer_math_imported']
    expected=[i for i,(root,step) in enumerate(m.ORDER) if step>stops[root]]
    assert out['missing_slots']==expected
    for key in ('prepared','report','strict'):assert out['bindings'][args[key]]==args[key.replace('prepared','prepare')+'_sha']
    path=tmp_path/(m.NAME+'_RAM_summary.json');m.write(path,out)
    m.same(m.read(path),m.analyze(**args),'same complete saved-only reconstruction')


@pytest.mark.parametrize('field',['prepare_sha','report_sha','strict_sha','source','array','extra'])
def test_real_saved_binding_mutations_fail_closed(tmp_path,monkeypatch,field):
    args=saved_corpus(tmp_path,monkeypatch)
    if field.endswith('_sha'):args[field]='e'*64
    elif field=='source':(tmp_path/'model'/'root0.json').write_bytes(b'{}')
    elif field=='array':
        path=next((Path(args['report']).parent).glob('*latent.npy.gz'));path.write_bytes(b'bad')
    else:(Path(args['report']).parent/'unknown.json').write_bytes(b'{}')
    with pytest.raises(ValueError):m.analyze(**args)
