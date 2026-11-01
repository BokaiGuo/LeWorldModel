"""Real loader control flow with invented bytes and a non-neural fake model.

No checkpoint, CUDA, physics, or neural forward is executed. Fixture evidence
must never be promoted to the actual authority's model identity.
"""
import hashlib
import json
import time
from pathlib import Path

import pytest

from experiments.spectral_reachability import agent_e3_continuous_host_launcher_20261029_v1 as launch
from experiments.spectral_reachability import agent_e3_continuous_saved_consumer_20261029_v1 as saved
from test_agent_e3_launcher_authority_fixture_20261029_v1 import actual_authority
from test_agent_e3_continuous_saved_consumer_20261029_v1 import exercise_fake_episode
from test_agent_e3_gd_host_cuda_20261029_v1 import Model as AlgebraModel,DEVICE


class FakeParameter:
    def __init__(self):
        import numpy as np
        self.a=np.asarray([0.125],dtype=np.float32);self.device=DEVICE;self.requires_grad=False
        self.dtype=self.a.dtype;self.shape=self.a.shape
    def detach(self):return self
    def cpu(self):return self
    def contiguous(self):return self
    def numpy(self):return self.a.copy()


class FakeModel(AlgebraModel):
    def __init__(self):
        super().__init__();self.training=False;self.loaded=False;self.weight=FakeParameter();self.assigned_device=None
        self.encode_calls=[];self.predict_calls=0
    def load_state_dict(self,state,strict=False):
        assert state=={'fixture.weight':[0.125]} and strict is False
        self.loaded=True
        return [],[]
    def to(self,device): assert device=='cuda:0';self.assigned_device=device;return self
    def eval(self): return self
    def requires_grad_(self,value): assert value is False;return self
    def modules(self): return [self]
    def parameters(self): return [self.weight]
    def buffers(self): return []
    def state_dict(self): return {'fixture.weight':self.weight}
    def named_modules(self): return [('',self)]
    def fixture_encode(self,boundary):self.encode_calls.append(boundary)
    def predict(self,z,e):
        self.predict_calls+=1
        return super().predict(z,e)


class FakeTorch:
    @staticmethod
    def load(stream,map_location,weights_only):
        assert stream.read()==b'fixture weights only' and map_location=='cpu' and weights_only is True
        return {'fixture.weight':[0.125]}


def fixture_prepare(tmp_path):
    policy=tmp_path/'fixture_policy';policy.mkdir()
    files={'config.json':json.dumps({'_target_':__name__+'.FakeModel'}).encode(),
           'weights.pt':b'fixture weights only'}
    hashes={}
    for name,data in files.items():
        path=policy/name;path.write_bytes(data);hashes[str(path)]=hashlib.sha256(data).hexdigest()
    expected=launch.host.primitives.native.loader.enc.tensor_hash(FakeModel())
    return dict(policy=str(policy),bindings_sha256=hashes,expected_model_hash=expected)


@pytest.mark.parametrize('stop,count,boundaries',[(None,20,[10,15]),(22,12,[10])])
def test_real_loader_control_flow_and_fake_episode(tmp_path,stop,count,boundaries):
    authority,authority_sha,_=actual_authority(tmp_path);prepared=fixture_prepare(tmp_path)
    fixture_prepare_sha='a'*64;artifacts={}
    def perform(binding,store):
        launch._record(store,'fixture_only',{'synthetic_checkpoint':True,'fake_model':True,'no_cuda_or_physics':True})
        loader=launch.host.primitives.native.loader
        model,model_sha=loader.loaded_model(prepared,FakeTorch,'cuda:0',
                                             launch.loader_bridge.LoaderBridge(store),fixture_prepare_sha)
        assert model.loaded and model_sha==prepared['expected_model_hash']
        from types import SimpleNamespace
        from experiments.spectral_reachability import agent_e3_model_source_guard_20261029_v1 as guard
        cuda=SimpleNamespace(synchronize=lambda device:None,max_memory_allocated=lambda device:1,
            max_memory_reserved=lambda device:2,matmul=SimpleNamespace(allow_tf32=False),
            math_sdp_enabled=lambda:True,flash_sdp_enabled=lambda:False,mem_efficient_sdp_enabled=lambda:False)
        fake_torch=SimpleNamespace(cuda=cuda,backends=SimpleNamespace(cuda=cuda,cudnn=SimpleNamespace(allow_tf32=False)),
                                   is_autocast_enabled=lambda device:False)
        source=str(Path(__file__).resolve());sources={source:hashlib.sha256(Path(source).read_bytes()).hexdigest()}
        def snap():return guard.snapshot(model,fake_torch,SimpleNamespace(type='cuda'),sources,[source])
        before=snap();identity=[];parameter=model.weight
        def trace(kind,actual,device):
            assert actual is model and actual.weight is parameter and actual.assigned_device=='cuda:0'
            assert device is DEVICE and parameter.device is DEVICE and not parameter.requires_grad
            identity.append(kind)
        row,kw,body,first,_=exercise_fake_episode(store,stop,model=model,identity_hook=trace)
        after=snap()
        assert before==after and before['tensor_sha256']==prepared['expected_model_hash']
        assert identity.count('encoding')==identity.count('node')==len(boundaries)
        assert model.encode_calls==boundaries and model.predict_calls>0 and identity[-1]=='cleanup'
        launch._record(store,'fixture_fake_guard_pair',{'before':before,'after':after,'fixture_only':True})
        artifacts.update(row=row,kw=kw,body=body,first=first)
        return row
    result=launch._run_lifecycle(tmp_path/'job',time.monotonic(),
        lambda:launch.read_authority(str(authority),authority_sha),perform)
    assert result['complete']
    seal=result['publication'];out=tmp_path/'job'
    events=[launch.pub.base.decode(x) for x in (out/'events.jsonl').read_bytes().splitlines()]
    launch.loader_bridge.verify_events(events,prepared,fixture_prepare_sha,launch.host.primitives.native.loader)
    a=artifacts
    feedback=saved.validate_feedback(events,out,seal['report_sha256'],seal['completion_sha256'],
        a['body'],a['kw']['prefix5'],a['first'],a['row']['episode_id'],count)
    assert [node['boundary'] for node in feedback['nodes']]==boundaries
    # Fixture checkpoint identity is deliberately different from actual authority.
    binding=launch.read_authority(str(authority),authority_sha)
    assert prepared['expected_model_hash']!=binding['prepared']['expected_model_hash']
    with pytest.raises(ValueError):
        launch.loader_bridge.verify_events(events,binding['prepared'],binding['prepare_sha'],
                                           launch.host.primitives.native.loader)
    with pytest.raises(ValueError):
        saved.consume(out,seal['report_sha256'],seal['completion_sha256'],str(authority),authority_sha)


def test_real_loader_rejects_changed_fixture_weight_before_deserialization(tmp_path):
    prepared=fixture_prepare(tmp_path);path=Path(prepared['policy'])/'weights.pt'
    path.write_bytes(b'changed')
    store=launch.pub.Store(tmp_path/'job')
    with pytest.raises(ValueError,match='exact consumed checkpoint bytes'):
        launch.host.primitives.native.loader.loaded_model(prepared,FakeTorch,'cuda:0',
            launch.loader_bridge.LoaderBridge(store),'a'*64)


def test_same_loaded_object_encoding_failure_and_metered_cleanup(tmp_path):
    from experiments.spectral_reachability import agent_e3_native_meter_20261029_v1 as meter
    authority,authority_sha,_=actual_authority(tmp_path);prepared=fixture_prepare(tmp_path);trace=[]
    def perform(binding,store):
        model,_=launch.host.primitives.native.loader.loaded_model(prepared,FakeTorch,'cuda:0',
                      launch.loader_bridge.LoaderBridge(store),'a'*64)
        weight=model.weight
        def identity(kind,actual,device):
            assert actual is model and actual.weight is weight and device is DEVICE
            assert actual.assigned_device=='cuda:0' and weight.device is DEVICE
            trace.append(kind)
        try:
            exercise_fake_episode(store,None,model=model,identity_hook=identity,fail_encoding=True)
        finally:
            loader=launch.host.primitives.native.loader
            loader.enc.frozen(model)
            assert loader.enc.tensor_hash(model)==prepared['expected_model_hash']
            trace.append('post_cleanup_hash')
    result=launch._run_lifecycle(tmp_path/'job',time.monotonic(),
                 lambda:launch.read_authority(str(authority),authority_sha),perform)
    assert not result['complete'] and trace==['encoding','cleanup','post_cleanup_hash']
    seal=result['publication'];out=tmp_path/'job'
    ledger=meter.consume(out,seal['report_sha256'],seal['completion_sha256'],'fixture')
    assert ledger['counts']['close']==dict(attempted=1,completed=1,failed=0)
    assert not ledger['ledger_complete'] and not ledger['full_native_strict']
