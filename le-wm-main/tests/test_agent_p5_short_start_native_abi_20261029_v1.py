"""Synthetic native-shaped saved-IO fixtures and descriptor guard tests only."""
from copy import deepcopy
import io
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_short_start_native_abi_20261029_v1 as m


def fixture(tmp_path):
    artifacts={}
    def array(name,a):
        buf=io.BytesIO();np.save(buf,a,allow_pickle=False);p=tmp_path/(name+'.npy');p.write_bytes(buf.getvalue())
        rec=dict(file=p.name,file_bytes=p.stat().st_size,sha256=m.sha(p),dtype='float32',shape=list(a.shape),nbytes=a.nbytes)
        artifacts[p.name]={k:rec[k] for k in ('file_bytes','sha256')};return {'$array':rec}
    stats=dict(ACTION_MEAN=[-.0087,.0068],ACTION_STD=[.2019,.2002],PROPRIO_MEAN=[236.6155,264.5674,-2.93032027,2.54307914],PROPRIO_STD=[101.1202,87.0112,74.84556075,74.14009094])
    prop=np.arange(12,dtype=np.float32).reshape(3,4);commands=np.zeros((20,2),np.float64)
    p,u=m.pre.normalized(prop,commands,stats)
    patch=np.arange(3*256*384,dtype=np.float32).reshape(3,256,384)/100
    features=np.arange(48,dtype=np.float32).reshape(1,3,1,16)
    calls=[None]*820
    def call(index,key,inputs,outputs,units):
        return dict(index=index,branch=0,phase='prediction',module=key,status='returned',inputs=inputs,outputs=outputs,units=units)
    enc=array('encoder',patch)
    calls[0]=call(0,'official_encoder',{}, {'$dict':{'x_norm_patchtokens':enc}},dict(api=1,batch_samples=3,frame_samples=3))
    calls[1]=call(1,'proprio_encoder',{'$tuple':[{'$tuple':[array('rawprop',p[None])]}, {'$dict':{}}]},array('propenc',features),dict(api=1,batch_samples=1,frame_samples=3))
    inputs={'$tuple':[{'$tuple':[array('visual',patch.reshape(m.pre.SHAPES[0])),array('action',u[None,:3]),array('proprio',np.broadcast_to(features,m.pre.SHAPES[2]).copy())]}, {'$dict':{}}]}
    calls[2]=call(2,'predictor',inputs,{},dict(api=1,batch_samples=1,frame_samples=3))
    req=dict(index=0,root_slot=0,condition='off',task_id='literal_task',selection_digest='literal_plan',context_global_frames=[0,5,10],successor_global_frames=[15,20],
        images=[dict(phase='warmup',step=k,condition='off',overlay_index=0,path='/fixture/'+str(k)+'.png') for k in (0,5,10)],raw_proprio=prop.tolist(),
        normalized_proprio=p.tolist(),commands=commands.tolist(),packets=u.tolist())
    row={k:req[k] for k in ('index','root_slot','condition','task_id','selection_digest')};row['context_patch']=enc['$array']
    before={'fixture_model':'not_actual_loaded_weights'}
    report=dict(schema=m.H2+'.report',status='fixed164_root10_open_loop_H2_recorded',rows=[row]+[None]*163,calls=calls,artifacts=artifacts,errors=[],source_errors=[],
        execution_binding=dict(prepare_sha256=m.PREPARE_SHA),model_before=before,model_after_predictions=before,model_after=before,strict_loads={})
    manifest=dict(schema=m.H2+'.prepare',bridge=dict(requests=[req]+[None]*163,stats=stats))
    strict=dict(schema=m.H2+'.consume',status='all164_open_loop_H2_and80_pairs_verified',report_sha256=m.REPORT_SHA,requests=164,pairs=80)
    return report,manifest,strict


def test_actual0_parent_receipt_and_real_NPY_crosslinks(tmp_path):
    r,p,c=fixture(tmp_path);item=m.resolve_actual0(r,p,c,tmp_path,0)
    assert item['native_parent_call']==2 and item['refs']['visual']['report_json_pointer']=='/calls/2/inputs/$tuple/0/$tuple/0/$array'
    args,receipt=m.materialize_actual0(item)
    assert [a.shape for a in args]==[(1,1,1,16,16,384),(1,1,10),(1,1,256,16)]
    assert receipt['frames']==[0] and not receipt['short_T_execution_certified'] and not receipt['full_strict_rebuilt_here']
    source=m.load_array(item['refs']['visual']);assert m.exact(args[0],source[:,0:1])
    assert m.exact(args[2],m.load_array(item['refs']['proprio'])[:,0:1])


@pytest.mark.parametrize('bad',('phase','index','root','task','time','overlay','kwargs','module','units','shape','inventory','receipt','model','context'))
def test_metadata_and_parent_misrouting_fail_closed(tmp_path,bad):
    r,p,c=fixture(tmp_path);q=p['bridge']['requests'][0]
    if bad=='phase':r['calls'][2]['phase']='label'
    elif bad=='index':r['calls'][2]['index']=3
    elif bad=='root':q['root_slot']=1
    elif bad=='task':q['task_id']='wrong'
    elif bad=='time':q['context_global_frames']=[5,10,15]
    elif bad=='overlay':q['images'][0]['overlay_index']=1
    elif bad=='kwargs':r['calls'][2]['inputs']['$tuple'][1]={'$dict':{'wrong':1}}
    elif bad=='module':r['calls'][2]['module']='proprio_encoder'
    elif bad=='units':r['calls'][2]['units']['frame_samples']=1
    elif bad=='shape':r['calls'][2]['inputs']['$tuple'][0]['$tuple'][0]['$array']['shape']=[1,1,1,16,16,384]
    elif bad=='inventory':r['artifacts'].clear()
    elif bad=='receipt':c['report_sha256']='0'*64
    elif bad=='model':r['model_after']={}
    else:r['rows'][0]['context_patch']={}
    with pytest.raises(ValueError):m.resolve_actual0(r,p,c,tmp_path,0)


def test_original_array_SHA_and_single_NPY_required(tmp_path):
    r,p,c=fixture(tmp_path);item=m.resolve_actual0(r,p,c,tmp_path,0)
    rec=item['refs']['visual']['array'];path=tmp_path/rec['file'];path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError):m.materialize_actual0(item)
    rec['file_bytes']=path.stat().st_size;rec['sha256']=m.sha(path)
    with pytest.raises(ValueError):m.materialize_actual0(item)


def before(g,t,**kw):
    values=dict(module='predictor',shapes=[(1,t,1,16,16,384),(1,t,10),(1,t,256,16)],devices=['cuda:0']*3,dtypes=['float32']*3,
        requires_grad=[False]*3,inference=True,TF32off=True,MATH=True)
    values.update(kw);return g.before(**values)


def after(g,t,**kw):
    values=dict(shapes=[(1,t,256,384),None,(1,t,256,16)],devices=['cuda:0']*2,dtypes=['float32']*2)
    values.update(kw);g.after(**values)


def test_ordered_T1_T2_paid_counts_no_extra_replay():
    g=m.ShapeMeter();before(g,1);after(g,1);before(g,2);after(g,2)
    assert g.attempted==g.returned==dict(api=2,batch_samples=2,frame_samples=3)
    with pytest.raises(ValueError):before(g,1)


@pytest.mark.parametrize('kw',({'module':'official_encoder'},{'devices':['cpu']*3},{'dtypes':['float64']*3},
    {'requires_grad':[True]*3},{'inference':False},{'TF32off':False},{'MATH':False}))
def test_guard_rejects_unsupported_dispatch(kw):
    g=m.ShapeMeter()
    with pytest.raises(ValueError):before(g,1,**kw)
    assert g.attempted['api']==0


def test_original_full3_shape_guard_not_relaxed_in_new_ABI():
    g=m.ShapeMeter()
    with pytest.raises(ValueError):before(g,3)
    assert g.attempted['api']==0


def test_wrong_return_keeps_attempt_and_prevents_retry():
    g=m.ShapeMeter();before(g,1)
    with pytest.raises(ValueError):after(g,1,devices=['cuda:1']*2)
    assert g.attempted['api']==1 and g.returned['api']==0 and g.pending==1
    with pytest.raises(ValueError):before(g,1)


def test_design_no_full_experiment_and_no_labels():
    d=m.design();assert d['fixed_branch']==0 and d['caps']['predictor']==dict(api=2,batch_samples=2,frame_samples=3)
    assert d['tensor_IO_effective_bytes']==819240*3 and not d['production_ready'] and not d['HOSTCUDA_executed']


def test_import_and_help_never_loads_models_or_consumers():
    code='import sys; from experiments.spectral_reachability import '+m.NAME+'; assert "torch" not in sys.modules; assert "pymunk" not in sys.modules; assert not any("raw164_jepa_h2_cuda" in x for x in sys.modules)'
    subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,capture_output=True,check=True)
    r=subprocess.run([sys.executable,'-m','experiments.spectral_reachability.'+m.NAME,'--help'],cwd=m.ROOT,capture_output=True,text=True,check=True)
    assert '--report-sha' in r.stdout
