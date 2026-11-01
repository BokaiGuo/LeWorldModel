from copy import deepcopy
import hashlib
import numpy as np
import pytest
from experiments.spectral_reachability import agent_e3_setup_authority_20261029_v1 as m
from test_agent_e3_gd_host_cuda_20261029_v1 import body


def fixture():
    b=body();r=m.host.numerical.resolver;rgb=np.zeros((224,224,3),np.uint8)
    pixel=hashlib.sha256(rgb.tobytes()).hexdigest();goal=[0.0]*7
    c0=dict(step=0,state=goal,snapshot={'state':goal},pixel_sha256=pixel)
    c5=dict(step=5,state=goal,snapshot={'state':goal},pixel_sha256=pixel)
    branch=dict(c0=c0,c5=c5);prov=dict(original_task={'binding':{'root_sha256':b['root_sha256']}},
        root_native_branches=[deepcopy(branch),deepcopy(branch)],prefix5=np.zeros((5,2),np.float32).tolist())
    ge=dict(native_goal=goal,goal_pixel_sha256=pixel,latent=b['goal'])
    cell=dict(r.roster()[0],status='input_ready_NOT_run_authorized',task={'body':b},goal_evidence=ge,
              evaluator={'native_goal':goal,'goal_pixel_sha256':pixel})
    prep=dict(schema=r.NAME+'.prepare',ready=16,cells=[cell]*16,root_provenance=[dict(root=0,
              status='qualified_saved_prefix',provenance=prov)],bindings={})
    root=dict(root_sha256=b['root_sha256'],slot={},goal=np.zeros(7),
              packets={'off':dict(rgb_history=np.zeros((3,224,224,3),np.uint8),
                   raw_action_history=np.zeros((10,2),np.float32),goal_rgb=rgb)},
              warm={i:dict(state=goal,snapshot={'state':goal}) for i in (0,5,10)})
    return prep,dict(roots=[root],source_plan={}),b


def test_setup_source_join():
    p,d,b=fixture();result=m.assemble(p,d,0,b)
    assert result['ref']['history'].shape==(3,224,224,3) and result['expected_cp5']['step']==5


@pytest.mark.parametrize('mutation',['body','branch','source','goal','warm'])
def test_reject_changed_setup_authority(mutation):
    p,d,b=fixture()
    if mutation=='body':b=deepcopy(b);b['identity_sha256']='1'*64
    if mutation=='branch':p['root_provenance'][0]['provenance']['root_native_branches'][1]['c5']['pixel_sha256']='0'*64
    if mutation=='source':d['roots'][0]['root_sha256']='1'*64
    if mutation=='goal':p['cells'][0]['evaluator']['native_goal']=[1.0]*7
    if mutation=='warm':d['roots'][0]['warm'][10]['state']=[1.0]*7
    with pytest.raises(ValueError):m.assemble(p,d,0,b)
