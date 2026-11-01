import hashlib
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest

from experiments.spectral_reachability import agent_e3_encoder_identity_20261029_v1 as contract
from test_agent_e3_gd_host_cuda_20261029_v1 import Module


class Device(str):
    @property
    def type(self):return 'cuda'


DEVICE=Device('cuda:0')


class Param:
    def __init__(self):
        self.device=DEVICE;self.requires_grad=False;self.data=np.asarray([.25],np.float32)
        self.shape=self.data.shape;self.dtype=self.data.dtype
    def detach(self):return self
    def cpu(self):return self
    def contiguous(self):return self
    def numpy(self):return self.data.copy()


class Model:
    def __init__(self):
        self.training=False;self.encoder=Module(lambda x:x);self.projector=Module(lambda x:x);self.p=Param()
    def named_parameters(self):return [('p',self.p)]
    def parameters(self):return [self.p]
    def buffers(self):return []
    def modules(self):return [self]
    def state_dict(self):return {'p':self.p}
    def load_state_dict(self,state,strict=False):assert state=={'p':[.25]} and strict is False;return [],[]
    def to(self,device):assert device is DEVICE;return self
    def eval(self):return self
    def requires_grad_(self,value):assert value is False;return self


class FakeTorch:
    @staticmethod
    def load(stream,map_location,weights_only):
        assert stream.read()==b'fixture' and map_location=='cpu' and weights_only is True
        return {'p':[.25]}


def prepared(tmp_path):
    policy=tmp_path/'policy';policy.mkdir()
    data={'config.json':json.dumps({'_target_':__name__+'.Model'}).encode(),'weights.pt':b'fixture'}
    paths={}
    for name,content in data.items():
        path=policy/name;path.write_bytes(content);paths[str(path)]=hashlib.sha256(content).hexdigest()
    return dict(policy=str(policy),bindings_sha256=paths,
                expected_model_hash=contract.launch.host.primitives.native.loader.enc.tensor_hash(Model()))


def read_events(store):
    return [contract.launch.pub.base.decode(line) for line in (store.output/'events.jsonl').read_bytes().splitlines()]


def implementation(model,torch,device,store,frame,boundary,check):
    for role in ('encoder','projector'):
        module=getattr(model,role);token=[]
        def pre(module,args):
            t=store.index;token.append(t);store.event('encoding','intent',role,t,{'boundary':boundary})
        handle=module.register_forward_pre_hook(pre)
        try:module(None)
        finally:handle.remove()
        store.event('encoding','return',role,token[0],{'boundary':boundary})
    return 'fixture result'


def test_loader_bound_object_and_saved_reverse_corruption(tmp_path):
    p=prepared(tmp_path);store=contract.launch.pub.Store(tmp_path/'job');sha='a'*64
    binding=contract.load_bound(p,FakeTorch,DEVICE,store,sha)
    assert binding._encode(FakeTorch,None,10,lambda:None,implementation)=='fixture result'
    seal=store.publish('completed');assert seal['complete']
    events=read_events(store);result=contract.verify(events,p,sha)
    sources={str(Path(m.__file__).resolve()):contract.launch.host.sha(m.__file__) for m in
             (contract,contract.launch,contract.launch.loader_bridge,contract.launch.host.primitives.native.loader,contract.encoder)}
    disk=contract.consume(store.output,seal['report_sha256'],seal['completion_sha256'],p,sha,sources)
    assert disk['transport_complete'] and disk['encodes']==result['encodes']
    assert result['encodes'][0]['callbacks']==['encoder','projector'] and not result['full_saved_strict']
    for mutation in ('object','tag','loader','chronology','parameter'):
        bad=deepcopy(events)
        if mutation=='object':next(e for e in bad if e['resource']=='encoder_identity_callback')['detail']['module_instance']='0x0'
        if mutation=='tag':next(e for e in bad if e['resource']=='encoder_identity_return')['detail']['tag']='0'*64
        if mutation=='loader':next(e for e in bad if e['resource']=='loader_receipt')['detail']['receipt']['tensor_hash_before']='0'*64
        if mutation=='chronology':next(e for e in bad if e['resource']=='encoder_identity_callback')['index']=100000
        if mutation=='parameter':next(e for e in bad if e['resource']=='encoder_identity_bind')['detail']['parameters'][0]['device']='cpu'
        with pytest.raises(ValueError):contract.verify(bad,p,sha)


@pytest.mark.parametrize('kind',['module','parameter','callback_failure'])
def test_live_identity_mutation_and_failure_preserved(tmp_path,kind):
    p=prepared(tmp_path);store=contract.launch.pub.Store(tmp_path/'job');binding=contract.load_bound(p,FakeTorch,DEVICE,store,'a'*64)
    if kind=='module':binding.model.encoder=Module(lambda x:x)
    if kind=='parameter':binding.model.p=Param()
    if kind in ('module','parameter'):
        with pytest.raises(ValueError):binding._encode(FakeTorch,None,10,lambda:None,implementation)
    else:
        def fail(model,torch,device,store,frame,boundary,check):
            model.encoder(None);raise RuntimeError('fixture failure')
        with pytest.raises(RuntimeError,match='fixture failure'):binding._encode(FakeTorch,None,10,lambda:None,fail)
        result=contract.verify(read_events(store),p,'a'*64)
        assert result['encodes'][0]['outcome']=='failure' and result['encodes'][0]['callbacks']==['encoder']
    assert not binding.modules['encoder'].pre and not binding.modules['projector'].pre
