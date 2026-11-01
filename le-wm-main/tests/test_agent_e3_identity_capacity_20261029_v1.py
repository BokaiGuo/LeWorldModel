import pytest
from experiments.spectral_reachability import agent_e3_identity_capacity_20261029_v1 as cap
from test_agent_e3_encoder_identity_20261029_v1 import prepared,FakeTorch,DEVICE,Model,Param,read_events


def test_real_store_actual_fixture_descriptor_byte_count(tmp_path):
    store=cap.identity.launch.pub.Store(tmp_path/'job');p=prepared(tmp_path)
    binding,proof=cap.load_bound(p,FakeTorch,DEVICE,store,'a'*64)
    events=read_events(store);assert events[-1]['resource']=='encoder_identity_bind'
    line=(store.output/'events.jsonl').read_bytes().splitlines(keepends=True)[-1]
    assert len(line)==proof['binding_event_bytes'] and proof['parameter_count']==1
    assert binding.model is not None and not proof['later_capacity_reserved']
    assert proof['max_identity_bytes']==147465


def test_oversized_actual_inventory_rejects_before_binding_write(tmp_path):
    store=cap.identity.launch.pub.Store(tmp_path/'job');model=Model()
    params=[('parameter_'+str(i)+'x'*80,Param()) for i in range(300)]
    model.named_parameters=lambda:params
    with pytest.raises(ValueError,match='exceeds per-event cap'):
        cap.preflight(model,DEVICE,'a'*64,'b'*64,store)
    assert store.index==0 and not (store.output/'events.jsonl').exists()


def test_wrong_device_is_rejected_before_binding(tmp_path):
    store=cap.identity.launch.pub.Store(tmp_path/'job');model=Model();model.p.device='cpu'
    with pytest.raises(ValueError):cap.preflight(model,DEVICE,'a'*64,'b'*64,store)
    assert store.index==0
