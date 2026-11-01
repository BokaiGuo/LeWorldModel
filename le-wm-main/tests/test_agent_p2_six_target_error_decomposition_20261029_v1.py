"""RAM fixtures only: never execute the actual saved experiment or fullstrict."""
import ast
import copy
import gzip
import importlib.util
import inspect
import io
import itertools
from pathlib import Path

import numpy as np
import pytest

SOURCE=Path(__file__).resolve().parents[1]/'experiments/spectral_reachability/agent_p2_six_target_error_decomposition_20261029_v1.py'
spec=importlib.util.spec_from_file_location('local_decomposition_under_test',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def vec(x):return np.full(192,x,np.float32)


def fixture_panel():
    """Dyadic, synthetic values. Not a claim about actual experiment outcomes."""
    p={'rows':[]};r={'rows':[]};zs=[]
    for slot in range(6):
        for j,cp in enumerate(m.CP):
            i=slot*4+j;pred=vec(slot//3+1 if j==0 else slot+1+j/4)
            z=pred.copy() if j==0 else pred+np.float32((slot%3-1)/8)
            state=[float(slot+j),2.,3.,4.,.25,5.,6.]
            frame=dict(step=cp,state=state,snapshot={'state':state})
            s=dict(index=i,slot=slot,root=slot//3,arm=m.ARMS[slot%3],condition='off',cp_public_step=cp,
                predicted192=pred.tolist(),target192=vec(slot/2).tolist(),goal192=vec(slot//3).tolist(),
                predicted_sha256=m.digest(pred.tobytes()),prediction_origin='observed_history' if j==0 else 'paid_actual_deployment_tokens',
                prediction_pointer={'index':[0,2] if j==0 else [0,0,j-1]},
                pixel_source={'path':f'fixture_{i}.png'},frame=frame,native_goal=[0.]*7,
                response=None if j==0 else dict(step=cp,state=state),task_id=f'task{slot//3}',goal_source={'root':slot//3},
                raw15_sha256='raw',source_time=None,target_origin={'known':False},original_z_sha256=f'z{slot//3}',
                original_a_sha256=f'a{slot//3}',reason=None,status='pending',latent=None,pixels=None,preprocessed=None,modules=[])
            row=copy.deepcopy(s);row.update(status='encoded',latent={'fixture':i})
            p['rows'].append(s);r['rows'].append(row);zs.append(z)
    rows,new=m.original_rows(p,r,zs)
    old,_,_=m.pairs(rows,new)
    return p,r,{'rows':rows,'pairs':old},zs


@pytest.mark.parametrize('a,b,kind,zero',[
    (1.,-1.,'strict_reversal',None),(-1.,1.,'strict_reversal',None),
    (5e-324,-5e-324,'strict_reversal',None),(2.,1.,'same_nonzero_sign',None),
    (-2.,-1.,'same_nonzero_sign',None),(0.,-0.,'tie','both_zero'),
    (-0.,3.,'tie','first_zero'),(-3.,0.,'tie','second_zero'),
    (None,0.,'null',None),(1.,None,'null',None)])
def test_sign_without_product(a,b,kind,zero):
    assert m.classify(a,b)==dict(kind=kind,zero_kind=zero)
    assert not any(isinstance(n,ast.Mult) for n in ast.walk(ast.parse(inspect.getsource(m.classify))))


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),True,'1'])
def test_invalid_sign(bad):
    with pytest.raises(ValueError):m.classify(bad,1.)


@pytest.mark.parametrize('p,z,t,g',[(3.,1.,2.,0.),(-2.,1.,-1.,0.),(.25,.5,-.25,.125)])
def test_algebra_independent_dyadic(p,z,t,g):
    a=m.decompose(vec(p),vec(z),vec(t),vec(g))
    for name,ref in [('task_goal',g),('tracking_target',t)]:
        v=a['residual'][name]
        assert v['model_goal_MSE']==(p-ref)**2
        assert v['prediction_residual_energy']==(p-z)**2
        assert v['signed_residual_cross_term']==2*(p-z)*(z-ref)
        assert v['identity_roundoff_residual']==0
    for route,x in [('model',p),('actual',z)]:
        v=a['tracking'][route]
        assert v['signed_tracking_goal_cross_term']==2*(x-t)*(t-g)
        assert v['original_task_goal_MSE']==(x-g)**2
        assert v['identity_roundoff_residual']==0


def test_missing_not_filled():
    a=m.decompose(vec(1),None,vec(2),vec(0))
    assert a['tracking']['actual'] is None
    assert a['residual']['task_goal']['actual_goal_MSE'] is None
    assert a['residual']['task_goal']['model_goal_MSE']==1


def test_native_units_angle_velocity_separated():
    x=m.native_scalars([3.,4.,0.,0.,7.,12.,0.],[0.]*7)
    assert x['joint_xy_squared']==25 and x['agent_velocity_squared']==144
    assert x['raw_angle_squared']==49 and x['native7_squared']==218
    assert x['native_angle_term']<0 and x['wrapped_angle_error']>0
    assert x['native_success'] is True  # inherited raw native rule, not wrapped replacement
    assert m.native_decomposition(x)['identity_roundoff_residual']==0


def test_full_denominator_and_temporal():
    p,r,c,z=fixture_panel();out=m.analyze(p,r,c,z)
    assert out['denominator']==dict(slots=6,positions=24,pairs=24,cp0_differences=24,adjacent_differences=18)
    assert len(out['temporal']['cp0'])==24 and len(out['temporal']['adjacent'])==18
    assert out['winner'] is None and not out['scientific_admission']
    for s in ('A','B','C'):
        count=out['breakpoint_counts'][s]
        assert sum(count[k] for k in ('same_nonzero_sign','strict_reversal','tie','null'))==24
    for row in out['temporal']['cp0'][::4]:
        assert row['common_goal']['model_goal_MSE']==0
        assert row['common_goal']['actual_goal_MSE']==0
    # Explicit right-minus-left direction and independent scalar difference.
    a=out['pairs'][1]
    assert a['common_goal']['model_goal_MSE']==c['rows'][5]['model_task_Q']-c['rows'][1]['model_task_Q']


@pytest.mark.parametrize('mutation',['scalar','pair','cp','order','prediction','goal','history','cp0','target'])
def test_saved_semantic_mutations(mutation):
    p,r,c,z=fixture_panel()
    if mutation=='scalar':c['rows'][1]['actual_task_Q']+=1
    elif mutation=='pair':c['pairs'][1]['right_minus_left_actual_task_Q']+=1
    elif mutation=='cp':p['rows'][1]['cp_public_step']=4
    elif mutation=='order':r['rows'][0],r['rows'][1]=r['rows'][1],r['rows'][0]
    elif mutation=='prediction':p['rows'][1]['prediction_origin']='old_normalized'
    elif mutation=='goal':
        p['rows'][4]['goal192']=vec(1).tolist();r['rows'][4]['goal192']=vec(1).tolist()
    elif mutation=='history':
        p['rows'][4]['original_z_sha256']='foreign';r['rows'][4]['original_z_sha256']='foreign'
    elif mutation=='cp0':z[0]=vec(9)
    else:
        p['rows'][1]['target192']=vec(99).tolist();r['rows'][1]['target192']=vec(99).tolist()
    with pytest.raises(ValueError):m.analyze(p,r,c,z)


def test_null_counts_no_padding():
    p,r,c,z=fixture_panel();i=1
    for d in (p['rows'][i],r['rows'][i]):
        d.update(pixel_source=None,frame=None,response=None,latent=None,reason='native_missing')
    r['rows'][i]['status']='missing';z[i]=None
    rows,new=m.original_rows(p,r,z);old,_,_=m.pairs(rows,new)
    out=m.analyze(p,r,dict(rows=rows,pairs=old),z)
    for stage in ('A','B','C'):assert out['breakpoint_counts'][stage]['null']==2
    assert out['positions'][1]['arithmetic']['residual']['task_goal']['actual_goal_MSE'] is None
    assert out['temporal']['adjacent'][0]['common_goal']['actual_goal_MSE'] is None


def latent_file(folder,a,tail=b''):
    buf=io.BytesIO();np.save(buf,a,allow_pickle=False)
    data=gzip.compress(buf.getvalue()+tail,mtime=0);path=folder/'latent.npy.gz';path.write_bytes(data)
    h=m.digest(data)
    return dict(file=path.name,sha256=h,array_sha256=m.digest(a.tobytes()),shape=[192],dtype='float32'),{path.name:h}


@pytest.mark.parametrize('mutation',['ok','tail','datahash','inventory','shape','traversal','symlink','filehash'])
def test_bounded_array_loader(tmp_path,mutation):
    a=vec(1);desc,inventory=latent_file(tmp_path,a,b'tail' if mutation=='tail' else b'')
    if mutation=='datahash':desc['array_sha256']='0'*64
    elif mutation=='inventory':inventory.clear()
    elif mutation=='shape':desc['shape']=[1,192]
    elif mutation=='traversal':desc['file']='../latent.npy.gz'
    elif mutation=='filehash':(tmp_path/desc['file']).write_bytes(b'bad')
    elif mutation=='symlink':
        (tmp_path/'other').write_bytes((tmp_path/desc['file']).read_bytes())
        (tmp_path/desc['file']).unlink();(tmp_path/desc['file']).symlink_to(tmp_path/'other')
    if mutation=='ok':assert m.load_latent(tmp_path,desc,inventory).tobytes()==a.tobytes()
    else:
        with pytest.raises((ValueError,OSError)):m.load_latent(tmp_path,desc,inventory)


@pytest.mark.parametrize('raw',[b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}'])
def test_json_reject(raw):
    with pytest.raises(ValueError):m.decode(raw)


def test_float32_strict_signedzero():
    assert m.f32([-0.]*192).tobytes()==vec(-0.).tobytes()
    with pytest.raises(ValueError):m.f32([.1]*192)
    with pytest.raises(ValueError):m.f32([True]*192)
    assert not m.equal(0.,-0.)


def test_no_upstream_or_neural_imports():
    tree=ast.parse(SOURCE.read_text())
    imports=[n.module if isinstance(n,ast.ImportFrom) else a.name for n in ast.walk(tree)
             if isinstance(n,(ast.Import,ast.ImportFrom)) for a in (n.names if isinstance(n,ast.Import) else [None])]
    assert not any(s and (s.startswith(('torch','experiments')) or 'pickle' in s) for s in imports)
    assert 'consume(' not in SOURCE.read_text()


def test_exclusive_writer_and_hash(tmp_path,monkeypatch):
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    out=tmp_path/'results'/(m.NAME+'_fixture.json');receipt=m.write(out,{'fixture':True})
    assert m.read(out,receipt['sha256'])=={'fixture':True}
    with pytest.raises(FileExistsError):m.write(out,{'fixture':False})
    with pytest.raises(ValueError):m.write(tmp_path/'foreign.json',{})


def fixture_documents(folder):
    p,r,c,z=fixture_panel();folder.mkdir()
    p.update(schema=m.RGB+'.prepare',contract={'wall_boundary':'fixture-inclusive'},expected_model_hash='model',bindings={},
        imported_paid_receipt={'report_sha256':'c3733a752d1e9b80929edf9726bc0eba2ad1c2ae39b6f05550652ff7b3b74dbd'},
        imported_controller_receipt={'report_sha256':'b81f12ee3602835bc14fd2e22d7e3644b8376f7ba8b0b265948bfb18429b6934'},
        imported_native_receipt={'physical_report_sha256':'ddbb339f87e0f49b1d99d32ae4f41340d72e6c19f8199fe69e6d0261f78bca2b',
                                 'counts':{'fixture':True},'rows':[]},goal_sources=[{'root':0},{'root':1}])
    for slot in range(6):
        raw=np.zeros((15,2),np.float32);s=p['rows'][slot*4]
        plan=dict(task_id=s['task_id'],task_sha256='taskhash',blind_sha256='blindhash',selection_digest='selection',
            raw_float32_bits=raw.tobytes().hex(),raw_float32_sha256=m.digest(raw.tobytes()),commands=raw.tolist(),
            target_origin=s['target_origin'],payload_sha256={'z':s['original_z_sha256'],'a':s['original_a_sha256']},
            controller_payload={'target':[s['target192']]})
        native=dict(plan=plan,goal=[0.]*7,checkpoints=[],responses=[None]*15)
        for j in range(4):
            i=slot*4+j;spec=p['rows'][i];cp=m.CP[j];frame=spec['frame']
            frame.update(png=f'r{i}.png',file_sha256=f'pnghash{i}',pixel_sha256=f'pixelhash{i}')
            src=dict(path=str(folder/frame['png']),file_sha256=frame['file_sha256'],pixel_sha256=frame['pixel_sha256'])
            p['bindings'][src['path']]=src['file_sha256']
            spec.update(task_sha256='taskhash',blind_sha256='blindhash',selection_digest='selection',
                full_raw_bits=raw.tobytes().hex(),raw_prefix_bits=raw[:cp].tobytes().hex(),raw15_sha256=m.digest(raw.tobytes()),pixel_source=src)
            native['checkpoints'].append(copy.deepcopy(frame))
            if cp:native['responses'][cp-1]=copy.deepcopy(spec['response'])
            row=copy.deepcopy(spec);row['status']='encoded'
            buf=io.BytesIO();np.save(buf,z[i],allow_pickle=False);b=gzip.compress(buf.getvalue(),mtime=0)
            name=f'r{i}_latent.npy.gz';(folder/name).write_bytes(b)
            row['latent']=dict(file=name,sha256=m.digest(b),array_sha256=m.digest(z[i].tobytes()),shape=[192],dtype='float32')
            r['rows'][i]=row
        p['imported_native_receipt']['rows'].append(native)
    original,new=m.original_rows(p,r,z);old,_,_=m.pairs(original,new)
    c.update(rows=original,pairs=old)
    r.update(schema=m.RGB+'.report',contract=p['contract'],completed=True,errors=[],modules=[],counts={'encoder':24},
        resources={'new_predict':0},model_before='model',model_after='model',wall_before_report=1.,artifact_sha256={})
    for row in r['rows']:r['artifact_sha256'][row['latent']['file']]=row['latent']['sha256']
    seal={k:r[k] for k in ('rows','modules','counts','errors')}
    (folder/'native_seal.json').write_bytes(m.encode(seal));r['native_seal_sha256']=m.sha(folder/'native_seal.json')
    r['artifact_sha256']['native_seal.json']=r['native_seal_sha256']
    pp=folder/'prepare.json';pp.write_bytes(m.encode(p));ph=m.sha(pp)
    r.update(prepared=str(pp),prepare_sha256=ph)
    rp=folder/'report.json';rp.write_bytes(m.encode(r));rh=m.sha(rp)
    d=dict(schema=m.RGB+'.completion',report_sha256=rh,complete=True,wall_inclusive=2.,boundary='fixture-inclusive')
    dp=folder/'completion.json';dp.write_bytes(m.encode(d));dh=m.sha(dp)
    c.update(schema=m.RGB+'.consume',status='all24_saved_RGB_positions_verified_science_blocked',
        prepare_sha256=ph,report_sha256=rh,completion_sha256=dh,whole_wall_through_report_fsync_hash=2.,
        denominator=dict(slots=6,rows=24,pairs=24),scientific_admission=False,known_roots=2,fresh=False,winner=None,
        new_predict=0,new_physics=0,new_AD=0,inherited_paid_receipt=p['imported_paid_receipt'],
        original_controller_receipt=p['imported_controller_receipt'],inherited_native_counts=p['imported_native_receipt']['counts'],
        bindings=p['bindings'],counts=r['counts'],resources=r['resources'])
    cp=folder/'receipt.json';cp.write_bytes(m.encode(c))
    refs={k:dict(path=str(path),sha256=m.sha(path)) for k,path in dict(prepare=pp,report=rp,completion=dp,receipt=cp).items()}
    return p,r,d,c,refs


def test_fixture_four_files_build_write_verify(tmp_path,monkeypatch):
    p,r,d,c,refs=fixture_documents(tmp_path/'inputs')
    monkeypatch.setattr(m,'actual_refs',lambda:refs)
    monkeypatch.setattr(m,'closure',lambda:{str(SOURCE):m.sha(SOURCE)})
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    result=m.build(refs)
    assert result['upstream_authentication_imported'] and not result['independent_fullstrict']
    assert result['new_calls']==dict(encoder=0,predict=0,AD=0,physics=0)
    out=tmp_path/'results'/(m.NAME+'_fixture.json');receipt=m.write(out,result)
    assert m.verify(out,receipt['sha256'])['verified']
    # On-disk mutation is caught, not replaced by imported success status.
    (tmp_path/'inputs'/'r0_latent.npy.gz').write_bytes(b'changed')
    with pytest.raises(ValueError):m.verify(out,receipt['sha256'])


@pytest.mark.parametrize('mutation',['reporthash','completion','partial','model','denominator','imports','resources','wall','bindings'])
def test_four_authority_join_mutations(tmp_path,mutation):
    p,r,d,c,refs=fixture_documents(tmp_path/'inputs')
    if mutation=='reporthash':c['report_sha256']='0'*64
    elif mutation=='completion':d['complete']=False
    elif mutation=='partial':r['errors']=['partial']
    elif mutation=='model':r['model_after']='other'
    elif mutation=='denominator':c['denominator']['rows']=23
    elif mutation=='imports':c['inherited_paid_receipt']={'report_sha256':'other'}
    elif mutation=='resources':c['resources']={'other':1}
    elif mutation=='wall':d['wall_inclusive']=901
    else:c['bindings']={'other':'hash'}
    with pytest.raises(ValueError):m.documents(p,r,d,c,refs)


@pytest.mark.parametrize('mutation',['raw','prefix','target','history','goal','png','response','native_state'])
def test_source_join_mutations(tmp_path,mutation):
    p,r,d,c,refs=fixture_documents(tmp_path/'inputs');s=p['rows'][1]
    if mutation=='raw':s['full_raw_bits']='00'
    elif mutation=='prefix':s['raw_prefix_bits']='bad'
    elif mutation=='target':s['target192']=vec(9).tolist()
    elif mutation=='history':s['original_a_sha256']='other'
    elif mutation=='goal':s['goal_source']={'root':99}
    elif mutation=='png':s['pixel_source']['file_sha256']='other'
    elif mutation=='response':s['response']={'step':4}
    else:s['frame']['state'][0]=999
    with pytest.raises(ValueError):m.source_join(p,r,lambda:None)


@pytest.mark.parametrize('shape,dtype',[((10**12,),'<f4'),((192,),'<f8'),((192,),'|O'),((1,192),'<f4')])
def test_npy_header_rejected_before_allocation(tmp_path,shape,dtype):
    buf=io.BytesIO()
    np.lib.format.write_array_header_1_0(buf,dict(shape=shape,fortran_order=False,descr=dtype))
    raw=gzip.compress(buf.getvalue(),mtime=0);path=tmp_path/'header.npy.gz';path.write_bytes(raw)
    desc=dict(file=path.name,sha256=m.digest(raw),array_sha256='unused',shape=[192],dtype='float32')
    with pytest.raises(ValueError,match='before allocation'):m.load_latent(tmp_path,desc,{path.name:desc['sha256']})


def test_fixed_authority_and_wall_failclosed(tmp_path,monkeypatch):
    _,_,_,_,refs=fixture_documents(tmp_path/'inputs')
    monkeypatch.setattr(m,'actual_refs',lambda:refs)
    other=copy.deepcopy(refs);other['report']['sha256']='0'*64
    with pytest.raises(ValueError,match='fixed actual'):m.build(other)
    tick=iter((0.,121.))
    with pytest.raises(ValueError,match='whole120'):m.build(refs,clock=lambda:next(tick))


def test_roundoff_is_retained_without_threshold():
    p=np.linspace(-7,3,192,dtype=np.float32)
    z=np.linspace(11,-5,192,dtype=np.float32)
    t=vec(.3);g=vec(-.7)
    d=m.decompose(p,z,t,g)
    for route in ('task_goal','tracking_target'):
        a=d['residual'][route]
        assert a['identity_roundoff_residual']==a['model_goal_MSE']-a['reconstructed_model_MSE']
    assert set(d['residual']['task_goal'])=={'model_goal_MSE','actual_goal_MSE','prediction_residual_energy',
        'signed_residual_cross_term','model_minus_actual_goal_MSE','reconstructed_model_MSE','identity_roundoff_residual'}
