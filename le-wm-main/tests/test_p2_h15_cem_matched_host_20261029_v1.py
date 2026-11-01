import runpy
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import p2_h15_cem_matched_host_20261029_v1 as m


def test_ball_mapping_deterministic():
    a=m.draw(np.zeros(30), np.ones(30), 123)
    b=m.draw(np.zeros(30), np.ones(30), 123)
    assert np.array_equal(a,b)
    assert a.shape==(7,30)
    assert np.max(np.linalg.norm(a,axis=1))<=m.RADIUS+1e-12
    assert np.max(np.abs(m.raw(a)))<=1


def test_selection_and_update_stable():
    samples=np.arange(210,dtype=float).reshape(7,30)/1000
    mean,std=m.update(samples,[1,0,0,3,4,5,6])
    assert np.array_equal(mean,(samples[1]+samples[2])/2)
    assert np.all(std>=m.STD_FLOOR)


def test_out_of_ball_and_nonfinite_rejected():
    with pytest.raises(ValueError): m.raw(np.ones((1,30))*100)
    with pytest.raises(ValueError): m.update(np.zeros((7,30)),[0,0,float('nan'),0,0,0,0])


def test_native_fixture_complete_and_audit(tmp_path):
    fixture=runpy.run_path(str(Path(__file__).parent/'test_agent_p3_parent_native_cuda_20261029_v1.py'))
    task=m.bridge.build_bridge()['roots'][0]['task']
    path=tmp_path/'slot'
    report,sha=m.run_slot(path,task,lambda sink:fixture['NumpyNative'](task,sink),lambda:None,'RAM_FIXTURE',0)
    assert report['status']=='complete'
    assert report['resources']['attempted']['predict_samples']==132
    assert m.audit_slot(path,sha,task,'RAM_FIXTURE',0)['status']=='complete'


def test_failure_is_saved_and_not_certified(tmp_path):
    fixture=runpy.run_path(str(Path(__file__).parent/'test_agent_p3_parent_native_cuda_20261029_v1.py'))
    task=m.bridge.build_bridge()['roots'][0]['task']; path=tmp_path/'slot'
    report,sha=m.run_slot(path,task,lambda sink:fixture['NumpyNative'](task,sink,fail=True),lambda:None,'RAM_FIXTURE',0)
    assert report['status']=='failed' and report['result'] is None
    with pytest.raises(ValueError): m.audit_slot(path,sha,task,'RAM_FIXTURE',0)


def test_actual_prepare_rebuild_and_no_authorization(tmp_path):
    prepared=tmp_path/'prepare.json'
    m.life.single._put(prepared,m.prepare_value()); digest=m.bridge.sha(prepared)
    value=m.verify_prepare(prepared,digest)
    assert value['budget_per_root']['predict_samples']==value['h15_budget_per_root']['predict_samples']==132
    assert value['h15_budget_per_root']['VJP_directions']==192
    with pytest.raises(ValueError,match='authorization'):
        m.run(prepared,digest,tmp_path/'output','not-authorized')
    assert not (tmp_path/'output').exists()


def test_success_slot_tamper_is_rejected(tmp_path):
    fixture=runpy.run_path(str(Path(__file__).parent/'test_agent_p3_parent_native_cuda_20261029_v1.py'))
    task=m.bridge.build_bridge()['roots'][1]['task'];path=tmp_path/'slot'
    report,sha=m.run_slot(path,task,lambda sink:fixture['NumpyNative'](task,sink),lambda:None,'RAM_FIXTURE',1)
    assert report['status']=='complete'
    with (path/'events.jsonl').open('ab') as stream:stream.write(b'{}\n')
    with pytest.raises(ValueError):m.audit_slot(path,sha,task,'RAM_FIXTURE',1)
