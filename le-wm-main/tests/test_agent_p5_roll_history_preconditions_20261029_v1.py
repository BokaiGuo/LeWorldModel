"""Lightweight fixtures only; no real manifest inspection or full strict run."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_roll_history_preconditions_20261029_v1 as m

STATS=dict(ACTION_MEAN=[-.0087,.0068],ACTION_STD=[.2019,.2002],PROPRIO_MEAN=[236.6155,264.5674,-2.93032027,2.54307914],
    PROPRIO_STD=[101.1202,87.0112,74.84556075,74.14009094])


def image(phase,step,condition,pixel):
    return dict(phase=phase,step=step,condition=condition,overlay_index=int(condition=='on'),
        file_sha256='a'*64,pixel_sha256=pixel*64,path='/fixture/'+phase+str(step)+'.png')


def manifests():
    a=[];b=[]
    for i in range(164):
        root,cond=((0,'off'),(0,'on'),(1,'off'),(1,'on'))[i//41]
        ident=dict(index=i,root_slot=root,condition=cond,task_id='fixture',selection_digest='literal')
        commands=np.tile(np.array([.1,0.],np.float32),(20,1)).astype(np.float64)
        pb=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]],np.float32)
        pa=np.concatenate([pb[1:],np.array([[13,14,15,16]],np.float32)])
        na,ua=m.normalized(pa,commands[5:],STATS);nb,ub=m.normalized(pb,commands,STATS)
        label=image('plan',10,cond,'f')
        a.append(dict(ident,context_global_frames=[5,10,15],successor_global_frame=20,
            images=[image('warmup',5,cond,'b'),image('plan',0,cond,'c'),image('plan',5,cond,'d')],
            raw_proprio=pa.tolist(),normalized_proprio=na.tolist(),commands=commands[5:].tolist(),packets=ua.tolist(),label=label))
        b.append(dict(ident,context_global_frames=[0,5,10],successor_global_frames=[15,20],
            images=[image('warmup',0,cond,'a'),image('warmup',5,cond,'b'),image('warmup',10,cond,'c')],
            raw_proprio=pb.tolist(),normalized_proprio=nb.tolist(),commands=commands.tolist(),packets=ub.tolist(),
            labels=[image('plan',5,cond,'d'),label],history_semantics='original pre-constructed-goal-bind warmup0/5/10; native default goal_pose overlay, never redrawn'))
    out=[]
    for key,reqs in (('H1',a),('H2',b)):
        name,_,h=m.PINS[key]
        out.append(dict(schema=name+'.prepare',bridge_prepared='/fixture/bridge.json',bridge_prepare_sha256='a'*64,
            bindings={str(m.ROOT/'experiments/spectral_reachability'/(name+'.py')):h},
            bridge=dict(requests=reqs,stats=STATS),protocol=dict(pairs=m.pair_schedule())))
    return out


def test_complete164_80_still_blocked_no_fake_prediction():
    rows=m.validate_manifests(*manifests())
    assert len(rows)==164 and len(m.pair_schedule())==80
    assert all(r['missing']['missing_saved_rolled_predicted_frames']==[5,10] for r in rows)
    assert all(r['missing']['full3_missing_actual_start_frames']==[-10,-5] for r in rows)
    assert all(r['missing']['full3_missing_raw_action_steps']==list(range(-9,1)) for r in rows)
    assert all(not r['missing']['short_start_missing_actual_frames'] for r in rows)
    assert not any(r['missing']['full3_bootstrap_constructible'] for r in rows)
    assert rows[-1]['index']==163


@pytest.mark.parametrize('bad',('omit','reorder','boolindex','root','condition','time','task','packet','raw','label','overlay','bridge','pair','source','pixel','prop','semantics'))
def test_metadata_mutations_refuse(bad):
    a,b=manifests();r=b['bridge']['requests'][0]
    if bad=='omit':b['bridge']['requests'].pop()
    elif bad=='reorder':b['bridge']['requests'][0],b['bridge']['requests'][1]=b['bridge']['requests'][1],b['bridge']['requests'][0]
    elif bad=='boolindex':r['index']=False
    elif bad=='root':r['root_slot']=1
    elif bad=='condition':r['condition']='on'
    elif bad=='time':r['context_global_frames']=[-10,-5,0]
    elif bad=='task':r['task_id']='different'
    elif bad=='packet':r['packets'][0][0]+=1
    elif bad=='raw':r['commands'][5][0]+=1
    elif bad=='label':r['labels'][1]['step']=9
    elif bad=='overlay':r['images'][0]['overlay_index']=1
    elif bad=='bridge':b['bridge_prepare_sha256']='b'*64
    elif bad=='pair':b['protocol']['pairs'].pop()
    elif bad=='source':b['bindings']={}
    elif bad=='pixel':r['images'][1]['pixel_sha256']='f'*64
    elif bad=='prop':r['raw_proprio'][1][0]+=1
    else:r['history_semantics']='post-goal redraw guessed'
    with pytest.raises(ValueError):m.validate_manifests(a,b)


def test_four_calls_true_early_predictions_and_action_alignment():
    calls=m.abi()['calls']
    assert [r['predicts_global_frame'] for r in calls]==[5,10,15,20]
    assert calls[0]['context_global_frames']==[0]
    assert calls[0]['action_raw_inclusive']==[[1,5]]
    assert calls[3]['context_global_frames']==[5,10,15]
    assert calls[3]['context_origin']==['sealed_prediction']*3
    assert calls[3]['action_raw_inclusive']==[[6,10],[11,15],[16,20]]
    raw=np.arange(40,dtype=np.float64).reshape(20,2)/100
    v=m.bootstrap_packets([0],list(range(1,21)),np.zeros((1,4)),raw,STATS)
    assert v['packet_bank'].shape==(4,10) and len(v['call_packets'])==4
    for ids,w in zip(([0],[0,1],[0,1,2],[1,2,3]),v['call_packets'],strict=True):np.testing.assert_array_equal(w,v['packet_bank'][ids])


def test_no_fake_bootstrap():
    with pytest.raises(ValueError):m.bootstrap_packets([0,0,0],list(range(1,21)),np.zeros((3,4)),np.zeros((20,2)),STATS)


def test_no_missing_actions_filled_or_zero_padded():
    with pytest.raises(ValueError):m.bootstrap_packets([0],list(range(6,21)),np.zeros((1,4)),np.zeros((15,2)),STATS)
    v=m.missing_history([-10,-5,0],list(range(-9,21)),[])
    assert v['full3_bootstrap_constructible'] and not v['sufficient_for_new_contrast']


def arrays():return tuple(np.zeros(s,np.float32) for s in m.SHAPES)


def identity():
    return dict(branch=0,root_slot=0,condition='off',task_id='fixed',selection_digest='fixed',aligned_frames=[5,10,15],
        model_state_sha256='a'*64,normalizer_source_sha256='b'*64,common_label_identity='literal_source',goal_overlay_history_identity='original_schedule')


def test_same_inputs_or_latest_only_cannot_be_new_early_contrast():
    a=arrays();b=deepcopy(a)
    with pytest.raises(ValueError):m.contrast_preflight(a,b,identity(),identity())
    b[0][:,2]=1;b[2][:,2]=2
    with pytest.raises(ValueError):m.contrast_preflight(a,b,identity(),identity())
    b[0][:,0]=3
    assert m.contrast_preflight(a,b,identity(),identity())['changed_by_slot']['visual']==[True,False,True]


def test_signed_zero_only_change_is_not_numeric_contrast():
    a=arrays();b=deepcopy(a);b[0][:,0]=-0.
    with pytest.raises(ValueError):m.contrast_preflight(a,b,identity(),identity())


@pytest.mark.parametrize('bad',('action','label','model','time','overlay','shape','dtype','nan'))
def test_native_early_contrast_fail_closed(bad):
    a=arrays();b=list(deepcopy(a));b[0][:,0]=1;ident=identity()
    if bad=='action':b[1][:,0]=1
    elif bad=='label':ident['common_label_identity']='other'
    elif bad=='model':ident['model_state_sha256']='f'*64
    elif bad=='time':ident['aligned_frames']=[0,5,10]
    elif bad=='overlay':ident['goal_overlay_history_identity']='redrawn'
    elif bad=='shape':b[0]=b[0][:,:2]
    elif bad=='dtype':b[0]=b[0].astype(np.float64)
    else:b[0][0,0,0,0,0,0]=np.nan
    with pytest.raises(ValueError):m.contrast_preflight(a,b,identity(),ident)


def test_rollover_uses_full_native_prediction_last_token_both_modalities():
    a=arrays();v=np.full((1,1,256,384),7,np.float32);p=np.full((1,1,256,16),9,np.float32)
    result=m.advance_history([0],a[0][:,:1],a[2][:,:1],(v,None,p))
    assert result['frames']==[0,5]
    assert np.all(result['visual'][:,:1]==0) and np.all(result['visual'][:,1]==7)
    assert np.all(result['proprio'][:,1]==9)
    with pytest.raises(ValueError):m.advance_history([0],a[0][:,:1],a[2][:,:1],(v,None,None))


def test_metadata_codec_sha_duplicate_nonfinite_no_closure_execution(tmp_path):
    p=tmp_path/'m.json';p.write_bytes(m.raw({'schema':'fixture'}))
    assert m.read(p,m.sha(p))=={'schema':'fixture'}
    with pytest.raises(ValueError):m.read(p,'0'*64)
    for body in (b'{"x":1,"x":2}',b'{"x":1e999}',b'{"x":NaN}'):
        p.write_bytes(body)
        with pytest.raises(ValueError):m.read(p,m.sha(p))


def test_exclusive_real_writer_and_metadata_verify(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    p=tmp_path/'results'/(m.NAME+'_fixture.json')
    value=dict(schema=m.NAME+'.inventory',status='blocked_fixture',arguments={})
    receipt=m.write_new(value,p)
    with pytest.raises(ValueError):m.write_new(value,p)
    monkeypatch.setattr(m,'inspect',lambda **k:value)
    assert m.verify(p,receipt['sha256'])['production_ready'] is False


def test_cli_and_import_no_neural_physics_or_consumers():
    code='import sys; from experiments.spectral_reachability import '+m.NAME+'; assert "torch" not in sys.modules; assert "pymunk" not in sys.modules; assert not any("raw164_jepa_cuda" in x for x in sys.modules)'
    subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,capture_output=True,check=True)
    r=subprocess.run([sys.executable,'-m','experiments.spectral_reachability.'+m.NAME,'inspect','--help'],cwd=m.ROOT,capture_output=True,text=True,check=True)
    assert '--h1-sha' in r.stdout and '--h2-manifest' in r.stdout


def test_full_metadata_inventory_fixture_never_execution_ready(monkeypatch,tmp_path):
    a,b=manifests()
    for manifest in (a,b):manifest['bindings'].update({str(m.ROOT/p):h for p,h in m.NATIVE_SOURCES.items()})
    # These are schema fixtures; no real historical manifest/output is consumed.
    monkeypatch.setattr(m,'read',lambda path,expected:a if str(path).endswith('h1.json') else b)
    result=m.inspect(tmp_path/'h1.json',m.PINS['H1'][1],tmp_path/'h2.json',m.PINS['H2'][1])
    assert result['branch_count']==164 and result['pair_count']==80 and len(result['branches'])==164
    assert result['full_strict_consumed'] is result['entire_panel_execution_ready'] is result['production_prepare'] is False
    assert result['global_missing']['short_start_missing_RGB_or_raw_actions']==[]
    assert result['global_missing']['full3_only_missing_actual_RGB_and_proprio_frames']==[-10,-5]
    assert result['roll_abi']['context_lengths']==[1,2,3,3]
    assert len(m.raw(result))<m.CAP


def test_short_start_full4_rollovers_keep_predicted_both_modalities():
    frames=[0];visual=np.zeros((1,1,1,16,16,384),np.float32);prop=np.zeros((1,1,256,16),np.float32)
    for target,expected in ((5,[0,5]),(10,[0,5,10]),(15,[5,10,15]),(20,[10,15,20])):
        t=len(frames)
        predictions=(np.full((1,t,256,384),target,np.float32),None,np.full((1,t,256,16),-target,np.float32))
        r=m.advance_history(frames,visual,prop,predictions)
        assert r['frames']==expected and np.all(r['visual'][:,-1]==target) and np.all(r['proprio'][:,-1]==-target)
        frames,visual,prop=r['frames'],r['visual'],r['proprio']
    assert np.all(visual[:,0]==10) and np.all(prop[:,0]==-10)


def test_bool_timestamps_cannot_stand_in_for_actual_frame0():
    with pytest.raises(ValueError):m.bootstrap_packets([False],list(range(1,21)),np.zeros((1,4)),np.zeros((20,2)),STATS)
