"""Same-env source setup through feedback core; fake model/physical callbacks only."""
import hashlib
import numpy as np
import pytest
from experiments.spectral_reachability import agent_e3_feedback_setup_20261029_v2 as setup
from experiments.spectral_reachability import agent_e3_continuous_dispatch_20261029_v1 as driver
from test_agent_e3_feedback_setup_20261029_v2 import case
from test_agent_e3_gd_host_cuda_20261029_v1 import body


@pytest.mark.parametrize('stop',[None,17,23,29])
def test_setup_feedback_and_cleanup_same_env(stop):
    env,kw,store,row=case(stop=stop);original=kw['ops'].capture
    def capture(env,obs,row,store,phase,step):
        image,receipt=original(env,obs,row,store,phase,step)
        receipt['file_sha256']=hashlib.sha256(image.tobytes()).hexdigest()
        return image,receipt
    kw['ops'].capture=capture
    b=body();b['root_context']['action_history'][-1]=((kw['prefix5']-np.asarray(b['action_mean'],np.float32))/np.asarray(b['action_std'],np.float32)).reshape(10).tolist()
    calls=[];closed=[];env.close=lambda:closed.append(env)
    def pack(instance,action,**args):
        assert instance is env;args['ops']=kw['ops']
        return setup.draft.packet.packet(instance,action,**args)
    def encode(instance,ob,boundary):
        assert instance is env;calls.append(('encode',boundary))
        rgb=np.full((224,224,3),env.i,np.uint8).tobytes()
        return dict(latent=np.full(192,boundary,np.float32),rgb_bytes=rgb,input_sha=hashlib.sha256(rgb).hexdigest(),
                    episode_id=row['episode_id'],boundary=boundary)
    def plan(task):calls.append(('plan',task['feedback_boundary']));return dict(first5=np.zeros((5,2),np.float32).tolist())
    try:
        s=setup.setup(env,**kw);assert s is not None
        driver.dispatch(env,b,s,np.zeros((5,2),np.float32),encode,plan,store,kw['meter'],row,packet_fn=pack)
    finally:env.close()
    assert closed==[env]
    assert row['status']==('completed20' if stop is None else 'native_stopped')
    expected=([] if stop==17 else [('encode',10),('plan',10)] if stop==23 else
              [('encode',10),('plan',10),('encode',15),('plan',15)])
    assert calls==expected
