"""Non-neural/non-physics fixture; does not certify production or native semantics."""
import numpy as np
import pytest
from experiments.spectral_reachability import agent_e3_authorized_episode_20261029_v1 as guarded
from experiments.spectral_reachability import agent_e3_continuous_dispatch_20261029_v1 as driver
from experiments.spectral_reachability import agent_e3_feedback_setup_20261029_v2 as setup
from experiments.spectral_reachability import agent_e3_native_capture_transport_20261029_v1 as capture
from experiments.spectral_reachability import agent_e3_native_region_adapter_20261029_v1 as region
from experiments.spectral_reachability import agent_e3_published_nodes_saved_20261029_v1 as saved
from experiments.spectral_reachability import agent_e3_terminal_row_20261029_v1 as terminal
from experiments.spectral_reachability import agent_e3_feedback_saved_join_20261029_v1 as feedback
from test_agent_e3_feedback_setup_20261029_v2 import case
from test_agent_e3_gd_host_cuda_20261029_v1 import body,Model,DEVICE,T


def test_same_fixture_env_two_nodes_published_and_replayed(tmp_path):
    env,kw,_,row=case();unified=capture.pub.Store(tmp_path/'job');bridge=capture.CaptureStore(unified)
    original=kw['ops'].capture
    def capture_frame(env,obs,row,store,phase,step):
        image,rec=original(env,obs,row,store,phase,step)
        name=f'fixture_{phase}_{step:02}.png';h=store.frame(name,image)
        rec.update(path=str(store.path/name),file_sha256=h)
        store.emit('fresh_observation',episode_id=row['episode_id'],receipt=rec.copy())
        return image,rec
    kw['ops'].capture=capture_frame
    kw.update(store=bridge,packet_ops=region)
    b=body();b['root_context']['action_history'][-1]=((kw['prefix5']-np.asarray(b['action_mean'],np.float32))
        /np.asarray(b['action_std'],np.float32)).reshape(10).tolist()
    model=Model();bodies=[];closed=[]
    env.close=lambda:closed.append(1)
    def encode(instance,obs,boundary):
        assert instance is env
        d=bridge.descriptor(obs);rgb=(unified.output/d['file']).read_bytes()
        # Explicit fake encoder output: no neural inference or parity claim.
        latent=np.full(192,boundary,np.float32)
        pixels=np.asarray(__import__('PIL.Image',fromlist=['Image']).open(unified.output/d['file'])).copy()
        prep=feedback.encoder.pixels.preprocess_pixels(pixels)
        hidden=np.zeros((1,257,192),np.float32)
        values=dict(pixels=pixels,preprocessed=prep,encoder_input=prep.reshape(1,3,224,224),
                    encoder_hidden=hidden,projector_input=hidden[:,0,:].copy(),
                    projector_output=latent.reshape(1,192),latent=latent)
        arrays={k:unified.array('encoding',k,v) for k,v in values.items()}
        for resource in ('encoder','projector'):
            token=unified.index;unified.event('encoding','intent',resource,token,dict(boundary=boundary))
            unified.event('encoding','return',resource,token,dict(boundary=boundary))
        unified.event('encoding','record','encoding_complete',unified.index,dict(boundary=boundary,frame=d,arrays=arrays))
        return dict(latent=latent,rgb_bytes=rgb,input_sha=d['sha256'],episode_id=row['episode_id'],boundary=boundary)
    def plan(task):
        bodies.append(task);a=saved.m.NodeBridge(unified,task['feedback_boundary'])
        r=dict(forwards=[],AD=[],result=None);backend=saved.m.bridge.host.HostBackend(model,DEVICE,T,a,task,r)
        with backend.observe():r['result']=saved.m.node.solve(task,backend,a.event)
        a.finish(r);return r['result']
    def execute(instance,binding):
        assert instance is env
        try:
            s=setup.setup(instance,**kw)
            driver.dispatch(instance,b,s,np.zeros((5,2),np.float32),encode,plan,bridge,kw['meter'],row,
                            packet_fn=lambda env,action,**args:region.packet(env,action,ops=kw['ops'],**args))
        finally:instance.close()
        terminal.write(unified,row)
        return row
    value=guarded._sequence(lambda:{'fixture':True},lambda:{'fixture':True},lambda:env,execute,
                            lambda a,b:None,lambda k,v:unified.event('physical','record',k,unified.index,v))
    assert value['status']=='completed20' and closed==[1] and env.i==30
    seal=unified.publish('completed')
    out=saved.consume(bodies,unified.output,seal['report_sha256'],seal['completion_sha256'])
    assert [x['boundary'] for x in out['nodes']]==[10,15]
    events=[capture.pub.base.decode(line) for line in (unified.output/'events.jsonl').read_bytes().splitlines()]
    assert terminal.read(events)==row
    assert not out['source_model_rgb_authenticated'] and not out['scientific_admission']
    joined=feedback.consume(unified.output,seal['report_sha256'],seal['completion_sha256'],b,kw['prefix5'],
                            np.zeros((5,2),np.float32),row['episode_id'])
    assert joined['completed20_feedback_io_joined'] and not joined['full_native_strict']
    wrong=kw['prefix5'].copy();wrong[0,0]+=.1
    with pytest.raises(ValueError):
        feedback.consume(unified.output,seal['report_sha256'],seal['completion_sha256'],b,wrong,
                         np.zeros((5,2),np.float32),row['episode_id'])
    with pytest.raises(ValueError):
        feedback.consume(unified.output,seal['report_sha256'],seal['completion_sha256'],b,kw['prefix5'],
                         np.ones((5,2),np.float32),row['episode_id'])
