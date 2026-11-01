import numpy as np
import pytest
from copy import deepcopy
from experiments.spectral_reachability import agent_e3_native_capture_transport_20261029_v1 as cap
from experiments.spectral_reachability import agent_e3_continuous_saved_consumer_20261029_v1 as saved


def test_warm_stop_real_journal_and_pixel_tamper(tmp_path):
    store=cap.pub.Store(tmp_path/'job');bridge=cap.CaptureStore(store)
    image=np.zeros((224,224,3),np.uint8);image[0,0]=[65,105,225];state=np.zeros(7,np.float64)
    def body():
        x=np.zeros(9,dtype='<f8')
        return dict(position=[0.,0.],velocity=[0.,0.],unwrapped_angle=0.,angular_velocity=0.,
                    force=[0.,0.],torque=0.,is_sleeping=False,float64_bits=x.tobytes().hex())
    snapshot=dict(state=state.tolist(),state_float64_bits=state.astype('<f8').tobytes().hex(),
                  bodies=dict(agent=body(),block=body()))
    for i in (0,1):
        name=f'warm_{i}.png';h=bridge.frame(name,image)
        rec=dict(phase='warmup',step=i,path=str(store.output/name),file_sha256=h,
                 pixel_sha256=bridge.frames[name]['pixel_sha256'],state=state.tolist(),snapshot=snapshot,
                 observed_event_index=bridge.index)
        bridge.emit('fresh_observation',episode_id='x',receipt=rec)
        bridge.emit('guard_passed',episode_id='x',phase='warmup',step=i,canonical_agent_pixels=1)
        if i:bridge.emit('warmup_return',episode_id='x',step=1,command=[0.,0.],terminated=True,truncated=False)
    seal=store.publish('completed')
    assert cap.pub.consume(store.output,seal['report_sha256'],seal['completion_sha256'])['complete']
    events=[cap.pub.base.decode(x) for x in (store.output/'events.jsonl').read_bytes().splitlines()]
    setup=dict(ref=dict(raw_history=np.zeros((10,2),np.float32),states=np.zeros((3,7)),
                        dynamics=[snapshot]*3,history=np.stack([image]*3)),
               prefix5=np.zeros((5,2),np.float32),new_goal=state,expected_goal_pixel='a'*64)
    row=dict(episode_id='x',PNG=2,warmup_returned=1,control_returned=0,executed_raw=[],status='native_stopped_warmup')
    result=saved.validate_physical(events,store.output,row,setup,np.zeros((5,2),np.float32))
    assert result['warm_steps']==1 and result['control_steps']==0
    shifted=deepcopy(events)
    guards=[e for e in shifted if e['resource']=='native_guard_passed']
    frames=[e for e in shifted if e['resource']=='native_frame']
    guards[0]['index']=frames[1]['index']+1
    with pytest.raises(ValueError,match='capture chronology'):
        saved.validate_physical(shifted,store.output,row,setup,np.zeros((5,2),np.float32))
    setup['ref']['history'][0,0,0,0]=1
    with pytest.raises(ValueError,match='RGB bits'):
        saved.validate_physical(events,store.output,row,setup,np.zeros((5,2),np.float32))
    bad=deepcopy(snapshot);bad['bodies']['agent']['position'][0]=1.
    with pytest.raises(ValueError,match='body dynamics bits'):
        saved.validate_snapshot(bad,state)


def test_no_node_before_control10_and_forbid_stray():
    from test_agent_e3_gd_host_cuda_20261029_v1 import body
    zeros=np.zeros((5,2),np.float32)
    assert saved.validate_feedback([], '/tmp/none', 'x','y',body(),zeros,zeros,'e',5)=={'nodes':[]}
    with pytest.raises(ValueError,match='uncharged'):
        saved.validate_feedback([dict(origin='encoding')],'/tmp/none','x','y',body(),zeros,zeros,'e',5)


def exercise_fake_episode(store,stop,model=None,identity_hook=None,fail_encoding=False):
    from test_agent_e3_gd_host_cuda_20261029_v1 import body,Model,DEVICE,T
    from test_agent_e3_feedback_setup_20261029_v2 import case
    from experiments.spectral_reachability import agent_e3_feedback_setup_20261029_v2 as setup
    from experiments.spectral_reachability import agent_e3_native_region_adapter_20261029_v1 as region
    from experiments.spectral_reachability import agent_e3_continuous_dispatch_20261029_v1 as driver
    from experiments.spectral_reachability import agent_e3_published_nodes_saved_20261029_v1 as numeric
    from experiments.spectral_reachability import agent_e3_encoder_io_20261029_v1 as enc
    from experiments.spectral_reachability import agent_e3_reset_identity_20261029_v1 as reset
    from test_agent_e3_reset_identity_20261029_v1 import Polygon
    from pathlib import Path
    raw,kw,_,row=case(stop=stop);bridge=cap.CaptureStore(store)
    env=reset.TracedEnv(raw,store,row['episode_id'])
    path=Path(__file__).resolve().parents[1]/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py'
    raw.block.shapes=[Polygon(p) for p in reset.official_polygons(path,reset.ENV_SHA)]
    kw['source_plan']=dict(recipe=dict(position_jitter=1.,angle_offset_range=[-.2,.2],
                                       edge_fraction_range=[.2,.8],pusher_surface_offset=2.))
    kw['ops'].source.sample_state=reset.old.native.source.sample_state
    old_capture=kw['ops'].capture
    def capture(env,obs,row,st,phase,step):
        image,rec=old_capture(env.env,obs,row,st,phase,step);name=f'{phase}_{step}.png';h=st.frame(name,image)
        rec.update(path=str(st.path/name),file_sha256=h)
        st.emit('fresh_observation',episode_id=row['episode_id'],receipt=deepcopy(rec))
        return image,rec
    kw['ops'].capture=capture;kw['ops']=reset.Ops(kw['ops'],store,row['episode_id'])
    kw.update(store=bridge,packet_ops=region)
    b=body();b['root_context']['action_history'][-1]=((kw['prefix5']-np.asarray(b['action_mean'],np.float32))
        /np.asarray(b['action_std'],np.float32)).reshape(10).tolist()
    model=Model() if model is None else model
    def encode(actual,ob,boundary):
        if identity_hook is not None:
            identity_hook('encoding',model,DEVICE);model.fixture_encode(boundary)
        if fail_encoding:raise RuntimeError('fixture encoding failure')
        frame=bridge.descriptor(ob);rgb=(store.output/frame['file']).read_bytes()
        pixels=np.asarray(__import__('PIL.Image',fromlist=['Image']).open(store.output/frame['file'])).copy()
        prep=enc.pixels.preprocess_pixels(pixels);hidden=np.zeros((1,257,192),np.float32)
        latent=np.full(192,boundary,np.float32)
        arrays={k:store.array('encoding',k,v) for k,v in dict(pixels=pixels,preprocessed=prep,
             encoder_input=prep.reshape(1,3,224,224),encoder_hidden=hidden,projector_input=hidden[:,0,:].copy(),
             projector_output=latent.reshape(1,192),latent=latent).items()}
        for name in ('encoder','projector'):
            token=store.index;store.event('encoding','intent',name,token,dict(boundary=boundary))
            store.event('encoding','return',name,token,dict(boundary=boundary))
        store.event('encoding','record','encoding_complete',store.index,dict(boundary=boundary,frame=frame,arrays=arrays))
        return dict(latent=latent,rgb_bytes=rgb,input_sha=frame['sha256'],episode_id=row['episode_id'],boundary=boundary)
    def plan(task):
        if identity_hook is not None:identity_hook('node',model,DEVICE)
        adapter=numeric.m.NodeBridge(store,task['feedback_boundary']);record=dict(forwards=[],AD=[],result=None)
        backend=numeric.m.bridge.host.HostBackend(model,DEVICE,T,adapter,task,record)
        with backend.observe():record['result']=numeric.m.node.solve(task,backend,adapter.event)
        adapter.finish(record);return record['result']
    first=np.zeros((5,2),np.float32)
    try:
        s=setup.setup(env,**kw)
        driver.dispatch(env,b,s,first,encode,plan,bridge,kw['meter'],row,
                        packet_fn=lambda env,action,**args:region.packet(env,action,ops=kw['ops'],**args))
    finally:
        if identity_hook is not None:
            from experiments.spectral_reachability import agent_e3_native_meter_20261029_v1 as native_meter
            meter=native_meter.Meter(store,row['episode_id'])
            bridge.emit('cleanup_intent',episode_id=row['episode_id'])
            meter.cleanup(lambda:identity_hook('cleanup',model,DEVICE))
            bridge.emit('cleanup_return',episode_id=row['episode_id'])
    return row,kw,b,first,path


@pytest.mark.parametrize('stop,count,boundaries',[(None,20,[10,15]),(22,12,[10])])
def test_two_node_end_to_end_saved_replay(tmp_path,stop,count,boundaries):
    from experiments.spectral_reachability import agent_e3_reset_identity_20261029_v1 as reset
    store=cap.pub.Store(tmp_path/'job')
    row,kw,b,first,path=exercise_fake_episode(store,stop)
    seal=store.publish('completed')
    events=[cap.pub.base.decode(x) for x in (store.output/'events.jsonl').read_bytes().splitlines()]
    assert reset.verify_events(events,row['episode_id'],dict(ref=kw['ref'],source_plan=kw['source_plan']))['reset_completed']==2
    assert reset.verify_geometry(events,{str(path):reset.ENV_SHA})['polygon_count']==2
    result=saved.validate_feedback(events,store.output,seal['report_sha256'],seal['completion_sha256'],
                                   b,kw['prefix5'],first,row['episode_id'],count)
    assert [v['boundary'] for v in result['nodes']]==boundaries
    with pytest.raises(ValueError):
        saved.validate_feedback(events,store.output,seal['report_sha256'],seal['completion_sha256'],
                                b,kw['prefix5'],first,row['episode_id'],11 if count==20 else 20)


@pytest.mark.parametrize('stop,count,boundaries',[(None,20,[10,15]),(22,12,[10])])
def test_authority_loader_episode_one_lifecycle(tmp_path,stop,count,boundaries):
    import time
    from test_agent_e3_launcher_authority_fixture_20261029_v1 import actual_authority,synthetic_loader
    from experiments.spectral_reachability import agent_e3_continuous_host_launcher_20261029_v1 as launch
    from experiments.spectral_reachability import agent_e3_reset_identity_20261029_v1 as reset
    path,sha,_=actual_authority(tmp_path);saved_fixture={}
    def perform(binding,store):
        synthetic_loader(binding,store)
        row,kw,b,first,source=exercise_fake_episode(store,stop)
        saved_fixture.update(row=row,kw=kw,body=b,first=first,source=source)
        return row
    result=launch._run_lifecycle(tmp_path/'job',time.monotonic(),lambda:launch.read_authority(str(path),sha),perform)
    assert result['complete']
    seal=result['publication'];out=tmp_path/'job'
    events=[cap.pub.base.decode(x) for x in (out/'events.jsonl').read_bytes().splitlines()]
    binding=launch.read_authority(str(path),sha)
    launch.loader_bridge.verify_events(events,binding['prepared'],binding['prepare_sha'],launch.host.primitives.native.loader)
    f=saved_fixture;kw=f['kw'];row=f['row']
    assert reset.verify_events(events,row['episode_id'],dict(ref=kw['ref'],source_plan=kw['source_plan']))['reset_completed']==2
    assert reset.verify_geometry(events,{str(f['source']):reset.ENV_SHA})['polygon_count']==2
    result=saved.validate_feedback(events,out,seal['report_sha256'],seal['completion_sha256'],f['body'],
                                   kw['prefix5'],f['first'],row['episode_id'],count)
    assert [node['boundary'] for node in result['nodes']]==boundaries
    assert events[0]['resource']=='launcher_started' and events[-1]['resource']=='launcher_guarded_return'
