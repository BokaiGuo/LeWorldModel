import hashlib
from pathlib import Path
import time
import pytest

from experiments.spectral_reachability import agent_e3_integrated_host_saved_20261029_v1 as integrated
from test_agent_e3_launcher_authority_fixture_20261029_v1 import actual_authority,synthetic_loader


def pinned_authority(tmp_path):
    path,sha,_=actual_authority(tmp_path)
    manifest=integrated.launch.host.read(str(path),sha)
    for module in (integrated,integrated.saved,integrated.partial_guard):
        source=str(Path(module.__file__).resolve())
        manifest['additional_sources'][source]=integrated.launch.host.sha(source)
    data=integrated.launch.host.raw(manifest);path.write_bytes(data)
    return path,hashlib.sha256(data).hexdigest()


def test_partial_loader_and_guard_are_joined_and_never_strict(tmp_path):
    path,sha=pinned_authority(tmp_path)
    def fake_perform(binding,store):
        synthetic_loader(binding,store)
        raise RuntimeError('fixture failure; no model or environment')
    output=tmp_path/'job'
    result=integrated.launch._run_lifecycle(output,time.monotonic(),lambda:integrated._authority(str(path),sha),fake_perform)
    seal=result['publication'];args=(output,seal['report_sha256'],seal['completion_sha256'],str(path),sha)
    evidence=integrated.inspect(*args)
    assert evidence['status']=='partial' and evidence['cell']==0
    assert evidence['partial_guard']['partial']['loader_receipt_verified']
    assert not evidence['partial_guard']['guard_pair_verified'] and not evidence['full_saved_strict']
    with pytest.raises(ValueError,match='full saved strict unavailable'):integrated.require_strict(*args)


def test_integration_source_must_be_pinned(tmp_path):
    path,sha,_=actual_authority(tmp_path)
    with pytest.raises(ValueError):integrated._authority(str(path),sha)


def test_unauthorized_never_creates_output(tmp_path):
    out=tmp_path/'job'
    with pytest.raises(ValueError,match='authorization'):
        integrated.run('/missing','0'*64,out,'fixture')
    assert not out.exists()


@pytest.mark.parametrize('mode',['valid','bad_snapshot','bad_terminal'])
def test_same_lifecycle_partial_snapshot_and_failure_row(tmp_path,mode):
    import numpy as np
    from experiments.spectral_reachability import agent_e3_native_capture_transport_20261029_v1 as capture
    from experiments.spectral_reachability import agent_e3_terminal_row_20261029_v1 as terminal
    path,sha=pinned_authority(tmp_path);out=tmp_path/'job'
    def perform(binding,store):
        synthetic_loader(binding,store);bridge=capture.CaptureStore(store)
        pixels=np.zeros((224,224,3),np.uint8);pixels[0,0]=[65,105,225]
        frame_sha=bridge.frame('frame.png',pixels);state=np.zeros(7,np.float64)
        def body():
            return dict(position=[0.,0.],velocity=[0.,0.],unwrapped_angle=0.,angular_velocity=0.,
                        force=[0.,0.],torque=0.,is_sleeping=False,float64_bits=np.zeros(9,dtype='<f8').tobytes().hex())
        snapshot=dict(state=state.tolist(),state_float64_bits=state.astype('<f8').tobytes().hex(),
                      bodies=dict(agent=body(),block=body()))
        if mode=='bad_snapshot':snapshot['bodies']['agent']['position'][0]=1.
        receipt=dict(path=str(out/'frame.png'),file_sha256=frame_sha,pixel_sha256=bridge.frames['frame.png']['pixel_sha256'],
                     phase='warmup',step=0,observed_event_index=bridge.index,state=state.tolist(),snapshot=snapshot)
        bridge.emit('fresh_observation',episode_id='cell00',receipt=receipt)
        error=dict(type='RuntimeError',message='fixture episode failed',traceback='fixture only')
        bridge.emit('episode_failure',error=error)
        row=dict(episode_id='cell00',status='failed',error=error if mode!='bad_terminal' else dict(error,message='altered'))
        terminal.write(store,row)
        raise RuntimeError('fixture episode failed')
    result=integrated.launch._run_lifecycle(out,time.monotonic(),lambda:integrated._authority(str(path),sha),perform)
    seal=result['publication'];args=(out,seal['report_sha256'],seal['completion_sha256'],str(path),sha)
    if mode=='valid':
        audit=integrated.inspect(*args)
        assert audit['partial_native']==dict(observations=1,unmatched_frame_count=0,episode_failure_observed=True,
                                              terminal_row_verified=True,physics_replayed=False,full_failure_semantics=False)
        assert not audit['full_saved_strict']
    else:
        with pytest.raises(ValueError):integrated.inspect(*args)


@pytest.mark.parametrize('bad',['none','loader_late','native_after_cleanup','failure'])
def test_success_lifecycle_order_rejects_splice(bad):
    resources=['loader_receipt','guard_before','native_frame','native_cleanup_intent',
               'native_meter/close','native_cleanup_return','guard_after','launcher_guarded_return']
    if bad=='loader_late':resources[0],resources[1]=resources[1],resources[0]
    if bad=='native_after_cleanup':resources.insert(5,'native_control_return')
    if bad=='failure':resources.insert(5,'native_episode_failure')
    events=[dict(origin='physical',kind='record',resource=resource,index=i) for i,resource in enumerate(resources)]
    if bad=='none':
        assert integrated.successful_lifecycle_chronology(events)['guard_after_cleanup']
    else:
        with pytest.raises(ValueError):integrated.successful_lifecycle_chronology(events)


@pytest.mark.parametrize('damage',['none','wrong_token','wrong_order','bad_boundary','missing_return'])
def test_encoder_ledger_exact_attempts_and_corruption(damage):
    events=[dict(origin='physical',resource='native_e3_encode_intent',kind='record',index=0,token=0,detail={'boundary':10})]
    for i,(resource,kind,token) in enumerate([('encoder','intent',1),('encoder','return',1),
            ('projector','intent',3),('projector','return',3),('encoding_complete','record',5)],1):
        events.append(dict(origin='encoding',resource=resource,kind=kind,index=i,token=token,detail={'boundary':10}))
    events.append(dict(origin='physical',resource='native_e3_encode_return',kind='record',index=6,token=6,detail={'boundary':10}))
    if damage=='wrong_token':events[2]['token']=2
    if damage=='wrong_order':events[2]['resource']='projector'
    if damage=='bad_boundary':events[3]['detail']['boundary']=True
    if damage=='missing_return':events.pop(2)
    if damage!='none':
        with pytest.raises(ValueError):integrated.encoder_ledger(events,True)
    else:
        result=integrated.encoder_ledger(events,True)
        assert result['boundaries'][0]['counts']=={'encoder':{'attempted':1,'returned':1},
                                                  'projector':{'attempted':1,'returned':1}}
        assert not result['model_encoder_instance_identity_verified']
        partial=integrated.encoder_ledger(events[:2],False)
        assert partial['boundaries'][0]['pending']=='encoder'
        assert partial['boundaries'][0]['counts']['encoder']=={'attempted':1,'returned':0}
