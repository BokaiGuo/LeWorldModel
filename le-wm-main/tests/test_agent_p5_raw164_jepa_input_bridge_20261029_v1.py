"""RAM and tiny saved-file fixtures; no torch/env/model import or production."""
from copy import deepcopy
import hashlib
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_raw164_jepa_input_bridge_20261029_v1 as m


def state_frame(phase,step,condition='off'):
    s=np.array([20.,30.,40.,50.,.1,6.,-7.],np.float64)
    return dict(phase=phase,step=step,condition=condition,overlay_index=0 if condition=='off' else 1,
        png=f'{phase}_{step}.png',file_sha256='fixture_file',pixel_sha256='fixture_pixel',state=s.tolist(),
        snapshot=dict(state=s.tolist(),state_float64_bits=s.tobytes().hex(),
            bodies=dict(agent=dict(position=s[:2].tolist(),velocity=s[5:].tolist()))))


def fixtures(condition='off'):
    plan=dict(index=0,condition=condition,root_slot=0,selection_digest='selection',task_id='task',
        commands=np.linspace(-.2,.3,20,dtype=np.float32).reshape(10,2).tolist())
    row=dict(index=0,plan=deepcopy(plan),condition=condition,root_slot=0,selection_digest='selection',
        status='horizon_completed',own_returned=10,warmup_returned=10,prefix51=dict(complete=True),
        trajectory=[state_frame('warmup',5,condition)]+[state_frame('plan',s,condition) for s in (0,5,10)])
    return row,plan


def stats():return dict(ACTION_MEAN=[-.0087,.0068],ACTION_STD=[.2019,.2002],PROPRIO_MEAN=[1,2,3,4],PROPRIO_STD=[2,3,4,5])


def test_import_and_full_fixed_denominator():
    assert not any(k in sys.modules for k in ('torch','pygame','pymunk','cv2','stable_worldmodel'))
    p=m.protocol();assert p['requests']==164 and len(p['pairs'])==80
    assert p['conditions']==['off','on'] and not p['fresh'] and not p['P5_admission']


def test_actual_original_stats_ast_no_torch():
    value=m.stats_from_source(m.STATS)
    assert value['ACTION_MEAN']==[-.0087,.0068]
    assert len(value['PROPRIO_MEAN'])==4 and 'torch' not in sys.modules


def test_saved_proprio_velocity_not_estimated():
    frame=state_frame('plan',5)
    assert m.proprio(frame).tolist()==[20.,30.,6.,-7.]


@pytest.mark.parametrize('mutation',('missing_velocity','wrong_velocity','missing_state','shape5','bits','snapshot'))
def test_invalid_proprio_rejected(mutation):
    frame=state_frame('plan',5)
    if mutation=='missing_velocity':del frame['snapshot']['bodies']['agent']['velocity']
    if mutation=='wrong_velocity':frame['snapshot']['bodies']['agent']['velocity'][0]=0.
    if mutation=='missing_state':del frame['state']
    if mutation=='shape5':frame['state']=frame['state'][:5]
    if mutation=='bits':frame['snapshot']['state_float64_bits']='00'
    if mutation=='snapshot':frame['snapshot']['state'][0]+=1
    with pytest.raises((ValueError,KeyError)):m.proprio(frame)


@pytest.mark.parametrize('condition',('off','on'))
def test_request_actual_time_actions_and_condition(monkeypatch,tmp_path,condition):
    monkeypatch.setattr(m,'image_record',lambda frame,d:dict(phase=frame['phase'],step=frame['step'],condition=frame['condition']))
    row,plan=fixtures(condition);req=m.request(row,plan,tmp_path,stats())
    assert req['context_global_frames']==[5,10,15] and req['successor_global_frame']==20
    assert [(r['phase'],r['step']) for r in req['images']]==[('warmup',5),('plan',0),('plan',5)]
    assert req['label']==dict(phase='plan',step=10,condition=condition)
    assert np.array_equal(np.asarray(req['commands'])[5:],np.asarray(plan['commands'],np.float32).astype(np.float64))


@pytest.mark.parametrize('mutation',('missing','duplicate','wrong_condition','partial','wrong_selection'))
def test_no_fill_or_subset_request(monkeypatch,tmp_path,mutation):
    monkeypatch.setattr(m,'image_record',lambda f,d:{})
    row,plan=fixtures()
    if mutation=='missing':row['trajectory'].pop()
    if mutation=='duplicate':row['trajectory'].append(deepcopy(row['trajectory'][0]))
    if mutation=='wrong_condition':row['trajectory'][0]['condition']='on'
    if mutation=='partial':row['own_returned']=9
    if mutation=='wrong_selection':row['selection_digest']='other'
    with pytest.raises(ValueError):m.request(row,plan,tmp_path,stats())


def test_normalization_exact_float32_projection_history_preserved():
    u=np.linspace(-.24,.7,30).reshape(15,2);p=np.arange(12).reshape(3,4)
    prop,packets=m.normalized(p,u,stats())
    expected=((100.*u).astype(np.float32)/np.float32(100.)-np.asarray(stats()['ACTION_MEAN'],np.float32))/np.asarray(stats()['ACTION_STD'],np.float32)
    assert packets.dtype==prop.dtype==np.float32 and np.array_equal(packets,expected.reshape(3,10))
    assert not np.array_equal(packets,((u/100.-np.asarray(stats()['ACTION_MEAN']))/np.asarray(stats()['ACTION_STD'])).reshape(3,10))


def test_normalizer_never_accepts_zero_std():
    value=stats();value['ACTION_STD'][0]=0
    with pytest.raises(ValueError):m.normalized(np.zeros((3,4)),np.zeros((15,2)),value)


def test_load_authority_before_verify_or_torch(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('must reject before source work'))
    with pytest.raises(ValueError):m.load_host_cuda('none','none','bad')
    assert 'torch' not in sys.modules


def test_host_inputs_no_future_image_read(monkeypatch):
    image={'role':'history'};label={'role':'future'}
    req=dict(index=0,root_slot=0,condition='off',images=[image]*3,label=label,
        raw_proprio=np.zeros((3,4)).tolist(),commands=np.zeros((15,2)).tolist())
    p,a=m.normalized(req['raw_proprio'],req['commands'],stats())
    req.update(normalized_proprio=p.tolist(),packets=a.tolist())
    rows=[dict(deepcopy(req),index=i) for i in range(164)]
    def pixels(rec):
        assert rec['role']=='history'
        return np.zeros((224,224,3),np.uint8)
    monkeypatch.setattr(m,'pixels',pixels)
    values=list(m.host_inputs(dict(protocol=m.protocol(),requests=rows,stats=stats())))
    assert len(values)==164 and all('label' not in r for r in values)


def test_label_decode_waits_all164_actual_arrays(tmp_path,monkeypatch):
    q=dict(requests=[dict(label={}) for _ in range(164)])
    seal=dict(schema=m.NAME+'.prediction_seal',prepared_semantic_sha256=m.digest(q),predictions=[])
    monkeypatch.setattr(m,'read',lambda *a:seal)
    monkeypatch.setattr(m,'pixels',lambda *a:pytest.fail('no labels before full164'))
    with pytest.raises(ValueError):m.labels_after_seal(q,'seal','sha')
    path=tmp_path/'pred.npy';np.save(path,np.zeros((1,3,256,384),np.float32))
    seal['predictions']=[dict(index=i,path=str(path),sha256=m.sha(path)) for i in range(164)]
    seal['predictions'][-1]['sha256']='wrong'
    with pytest.raises(ValueError):m.labels_after_seal(q,'seal','sha')


def test_real_metadata_writer_reader(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    monkeypatch.setattr(m.shutil,'disk_usage',lambda *a:NS(free=2**30))
    path=tmp_path/'results'/(m.NAME+'_test.json');q=dict(status='fixture',newline='inside\nstring')
    result=m.write_new(path,q)
    assert path.read_bytes()==m.raw(q) and path.read_bytes().count(b'\n')==1
    assert m.read(path,result['sha256'])==q
    with pytest.raises(ValueError):m.write_new(path,q)


@pytest.mark.parametrize('body',(b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":1e999}'))
def test_invalid_json_rejected(tmp_path,body):
    path=tmp_path/'bad.json';path.write_bytes(body)
    with pytest.raises(ValueError):m.read(path)


def test_real_png_bytes_pixel_identity(tmp_path):
    from PIL import Image
    a=np.zeros((224,224,3),np.uint8);path=tmp_path/'rgb.png';Image.fromarray(a).save(path)
    rec=dict(path=str(path),file_sha256=m.sha(path),pixel_sha256=hashlib.sha256(a.tobytes()).hexdigest())
    assert np.array_equal(m.pixels(rec),a)
    rec['pixel_sha256']='bad'
    with pytest.raises(ValueError):m.pixels(rec)
