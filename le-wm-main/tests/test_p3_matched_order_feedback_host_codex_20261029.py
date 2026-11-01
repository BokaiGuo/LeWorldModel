"""CPU-only boundary tests; no neural inference or physics execution."""
from copy import deepcopy
import hashlib
import numpy as np
import pytest
import agent_p3_fullcommit_live_20261028_v1_tests as fixture
from experiments.spectral_reachability import p3_matched_order_feedback_host_codex_20261029 as host
from experiments.spectral_reachability import p3_matched_order_feedback_host_saved_codex_20261029 as saved


def native_events():
    events=[]
    def pair(name,region):
        token=len(events); detail=dict(episode_id='matched_slot_00',region=region,units=1)
        for kind in ('intent','return'):
            events.append(dict(resource='native_meter/'+name,kind=kind,token=token,detail=detail))
    pair('constructor','setup')
    pair('reset','setup');pair('reset','setup');pair('goal_bind','goal_bind')
    for i in range(30):pair('public_step','warmup' if i<10 else 'prefix')
    pair('close','cleanup')
    return events


def test_saved_native_lifecycle_refuses_missing_close_and_late_dispatch():
    events=native_events()
    assert saved._native_lifecycle(events,'matched_slot_00')['public_step']==30
    with pytest.raises(Exception):saved._native_lifecycle(events[:-2],'matched_slot_00')
    changed=deepcopy(events);changed[-1]['token']+=1
    with pytest.raises(Exception):saved._native_lifecycle(changed,'matched_slot_00')
    changed=deepcopy(events);changed.insert(-2,dict(resource='native_control_return',kind='record',token=0,
        detail=dict(terminated=True,truncated=False)))
    with pytest.raises(Exception):saved._native_lifecycle(changed,'matched_slot_00')


def test_authorization_rejected_before_any_output_or_cuda(tmp_path):
    with pytest.raises(Exception):host.run('unused','unused',tmp_path/'run','bad','not-coordinated')
    assert not (tmp_path/'run').exists()


def test_numeric_missing_chunk_is_rejected(tmp_path):
    store=host.storage.Store(tmp_path/'run')
    saved.bridge.Bridge(store,0).event(dict(kind='arm_event',slot=0,payload={'x':'z'*12000}))
    events=host.storage.inspect_journal(store.output/'events.jsonl',host.source.host.sha(store.output/'events.jsonl'))['events']
    assert len(events)>1
    with pytest.raises(Exception):saved._events_with_positions(events[:-1],0)


@pytest.mark.parametrize('policy',['fixed_order','signal_order'])
def test_saved_block_math_rebuilds_two_stages_from_recorded_fixture(policy):
    root,_=fixture.request();warm=fixture.m.old.kernel.moving.warmup_commands()
    root['root_context']['action_history']=fixture.m.moving.normalized(root,warm)
    root['binding']['context_sha256']=host.life.seam.core.digest(root['root_context'])
    calls=[];events=fixture.Events()
    class Session:
        episode_id='fixture'
        def packet(self,start,raw):
            return dict(raw=raw.copy(),rgb_bytes=str(start+5).encode(),end=start+5,stopped=False,episode_id=self.episode_id)
        def encode(self,end,rgb):
            return dict(latent=np.full(192,end/100,np.float32),input_sha256=hashlib.sha256(rgb).hexdigest(),
                        end=end,episode_id=self.episode_id)
    class Backend:
        kind='RAM_FIXTURE'
        def __init__(self,task,sink):self.task=deepcopy(task);self.backend=fixture.RamBackend(task,sink)
        def signal(self,request,ledger):
            out=self.backend.signal(request,ledger)
            calls.append([dict(method='signal',task=deepcopy(self.task),request=deepcopy(request)),dict(result=deepcopy(out))]);return out
        def replay(self,context,actions,segmented,ledger):
            out=self.backend.replay(context,actions,segmented,ledger)
            calls.append([dict(method='replay',task=deepcopy(self.task),request=dict(parent_context=deepcopy(context),
                actions=deepcopy(actions),segmented=segmented)),dict(result=deepcopy(out))]);return out
    result=host.life.seam.execute_slot(root,policy,warm[-5:].copy(),Session(),Backend,events,permit_ram=True)
    assert result['status']=='callback_feedback_complete_not_certified',result['errors']
    observations={'warm':warm}
    observations.update({end:dict(latent=np.full(192,end/100,np.float32)) for end in (5,10,15,20)})
    saved._block_math(root,result,observations,calls)
    altered=deepcopy(result);altered['blocks'][1]['context_sha256']='0'*64
    with pytest.raises(Exception):saved._block_math(root,altered,observations,calls)
