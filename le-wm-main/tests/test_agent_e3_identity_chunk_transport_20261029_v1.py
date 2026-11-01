from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_e3_identity_chunk_transport_20261029_v1 as c
from experiments.spectral_reachability import agent_e3_preflight_array_transport_20261029_v1 as transport
from test_agent_e3_encoder_identity_20261029_v1 import prepared, FakeTorch, DEVICE, implementation, read_events, Model, Param


def test_actual_loader_same_object_two_nodes_and_saved(tmp_path):
    p=prepared(tmp_path); store=transport.Store(tmp_path/'job')
    binding=c.load_bound(p,FakeTorch,DEVICE,store,'a'*64)
    for boundary in (10,15):
        assert binding._encode(FakeTorch,None,boundary,lambda:None,implementation)=='fixture result'
    seal=store.publish('completed')
    assert c.old.launch.pub.consume(store.output,seal['report_sha256'],seal['completion_sha256'])['complete']
    result=c.verify(read_events(store),p,'a'*64)
    assert [r['boundary'] for r in result['encodes']]==[10,15]
    assert all(r['outcome']=='return' for r in result['encodes'])
    assert result['binding']['model_instance']==hex(id(binding.model))
    assert result['binding']['parameters']==binding.params


def test_large_actual_identity_chunks_and_device_mutation(tmp_path):
    class Large(Model):
        def __init__(self):
            super().__init__();self.ps=[Param() for _ in range(500)]
        def named_parameters(self):return [(f'encoder.layer.{i}.weight',p) for i,p in enumerate(self.ps)]
    model=Large();store=transport.Store(tmp_path/'job'); binding=c.Binding(model,DEVICE,'b'*64,'a'*64,store)
    events=read_events(store); assert len([e for e in events if e['resource']=='preflight_binding_chunk'])>2
    reconstructed=c.reconstruct(events)
    actual=next(e for e in reconstructed if e['resource']=='encoder_identity_bind')['detail']
    assert actual==binding.descriptor and len(actual['parameters'])==500
    assert max(map(len,(store.output/'events.jsonl').read_bytes().splitlines()))<=16384
    model.ps[0].device='cpu'
    with pytest.raises(ValueError):binding.check()


@pytest.mark.parametrize('mutation',['data','order','seal','drop','extra','legacy'])
def test_chunk_corruption_fails_closed(tmp_path,mutation):
    store=transport.Store(tmp_path/'job'); c.Binding(Model(),DEVICE,'b'*64,'a'*64,store)
    events=deepcopy(read_events(store)); chunks=[e for e in events if e['resource']=='preflight_binding_chunk']
    seal=next(e for e in events if e['resource']=='preflight_binding_seal')
    if mutation=='data':chunks[0]['detail']['data']='AAAA'
    if mutation=='order':chunks[0]['detail']['index']=1
    if mutation=='seal':seal['detail']['sha256']='0'*64
    if mutation=='drop':events.remove(chunks[0])
    if mutation=='extra':events.insert(0,deepcopy(chunks[0]))
    if mutation=='legacy':events.append(dict(seal,resource='encoder_identity_bind'))
    with pytest.raises((ValueError, TypeError)):c.reconstruct(events)


def test_partial_unsealed_and_encoder_failure(tmp_path):
    p=prepared(tmp_path);store=transport.Store(tmp_path/'job');b=c.load_bound(p,FakeTorch,DEVICE,store,'a'*64)
    def fail(model,torch,device,store,frame,boundary,check):
        model.encoder(None);raise RuntimeError('encoder failure')
    with pytest.raises(RuntimeError):b._encode(FakeTorch,None,10,lambda:None,fail)
    events=read_events(store);result=c.verify(events,p,'a'*64)
    assert result['encodes'][0]['outcome']=='failure' and result['encodes'][0]['callbacks']==['encoder']
    assert not b.modules['encoder'].pre and not b.modules['projector'].pre
    incomplete=[e for e in events if e['resource']!='preflight_binding_seal']
    assert c.reconstruct(incomplete,allow_partial=True) is None
    with pytest.raises(ValueError):c.reconstruct(incomplete)


def test_mid_chunk_failure_preserves_partial_journal(tmp_path):
    class Broken(transport.Store):
        def event(self, origin, kind, resource, token, detail):
            if resource=='preflight_binding_seal': raise OSError('injected seal failure')
            return super().event(origin,kind,resource,token,detail)
    s=Broken(tmp_path/'job')
    with pytest.raises(OSError):c.Binding(Model(),DEVICE,'b'*64,'a'*64,s)
    seal=s.publish('partial')
    assert not c.old.launch.pub.consume(s.output,seal['report_sha256'],seal['completion_sha256'])['complete']
    assert c.reconstruct(read_events(s),allow_partial=True) is None


def test_real_source_to_v3_authority(tmp_path):
    from test_agent_e3_identity_authority_prepare_20261029_v1 import refs,RESOLVER
    from experiments.spectral_reachability import agent_e3_identity_authority_prepare_20261029_v1 as identity_prepare
    from experiments.spectral_reachability import agent_e3_preflight_v3_authority_prepare_20261029_v1 as authority
    source=identity_prepare.prepare(refs(),RESOLVER,0,str(tmp_path/'source.json'))
    result=authority.prepare(source['path'],source['sha256'],str(tmp_path/'v3.json'))
    assert authority.verify(source['path'],source['sha256'],result['output'],result['sha256'])['verified']


def test_v3_saved_partial_after_array_and_bound_chunks(tmp_path,monkeypatch):
    import numpy as np
    from experiments.spectral_reachability import agent_e3_encoder_cuda_preflight_20261029_v3 as v3
    s=transport.Store(tmp_path/'job')
    v3.record(s,'preflight_started',dict(schema=v3.NAME,scientific_admission=False,physics_calls=0))
    c.Binding(Model(),DEVICE,'b'*64,'a'*64,s)
    s.array('encoding','pixels',np.zeros((224,224,3),np.uint8))
    v3.record(s,'preflight_failure',dict(type='RuntimeError',message='fixture',traceback='fixture'))
    seal=s.publish('partial')
    monkeypatch.setattr(v3.host,'authority',lambda *a:{})
    monkeypatch.setattr(v3,'sources',lambda binding:({},[]))
    result=v3.consume(str(s.output),seal['report_sha256'],seal['completion_sha256'],'/fixture','x')
    assert result['status']=='partial' and result['binding_chunks_complete']
    assert not result['encoder_calls_certified'] and result['array_transport']['returned']==1
