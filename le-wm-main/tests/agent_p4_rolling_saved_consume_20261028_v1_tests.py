"""RAM arrays/PNG bytes and mutation tests; no producer, torch, native physics."""
from copy import deepcopy
import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from experiments.spectral_reachability import agent_p4_rolling_saved_consume_20261028_v1 as m


def fixture(mode='R', n=20):
    warm = np.arange(20, dtype=np.float32).reshape(10, 2)/100
    executed = (np.arange(n*2, dtype=np.float32).reshape(n, 2)-20)/100
    task = dict(action_mean=[0., 0.], action_std=[1., 1.], goal=[0.]*192,
                root_context=dict(layout='RAM',latent_history=[[float(i)]*192 for i in range(3)],
                                  action_history=warm.reshape(2,10).tolist()))
    layer = dict(commands=np.concatenate((executed[:5],warm[:5])).tolist(),imagined_prefix=[7.]*192)
    row = dict(mode=mode,executed_raw=executed.tolist(),control_returned=n)
    requests = {}; responses = {}; latents = {}; observed=deepcopy(task['root_context']); imagined=None
    for b in (b for b in (5,10,15) if b < n):
        raw10=np.concatenate((warm,executed[:b]))[-10:]
        z=np.full(192,100+b,np.float32);latents[b]=z
        observed=m.rolled(observed,z,raw10,task)
        if imagined is None:imagined=m.rolled(task['root_context'],layer['imagined_prefix'],raw10,task)
        context=deepcopy(observed if mode=='R' else imagined)
        requests[b]=dict(actualrawhistory10=raw10.tolist(),context3=context,
                         diagnosticraw5=layer['commands'][5:],goal=task['goal'])
        commands=np.full((5,2),.25,np.float32)
        commands[:min(5,n-b)]=executed[b:b+5]
        imagined=m.rolled(context,np.full(192,-b,np.float32),
                          np.concatenate((raw10[5:],commands)),task)
        responses[b]=dict(dispatch_raw5=commands.tolist(),predicted_successor=deepcopy(imagined))
    return row,layer,task,warm,requests,responses,latents


@pytest.mark.parametrize('mode',['I','R'])
@pytest.mark.parametrize('n',[1,5,6,10,11,15,16,20])
def test_full_and_native_early_stop_preserves_reached_boundary_denominator(mode,n):
    args=fixture(mode,n)
    out=m.check_episode(*args)
    assert [x['boundary'] for x in out]==[b for b in (5,10,15) if b<n]
    assert sum(x['executed_in_packet'] for x in out)==max(0,n-5)


@pytest.mark.parametrize('mode',['I','R'])
@pytest.mark.parametrize('mutation',['history','stale_raw','latent','action_tokens','goal','diagnostic',
                                    'drop_boundary','drop_latent','foreign_successor','actions','normalizer'])
def test_independent_rolling_mutations_rejected(mode,mutation):
    args=fixture(mode);row,layer,task,warm,req,res,z=args
    if mutation=='history':req[10]['actualrawhistory10'][0][0]+=.125
    elif mutation=='stale_raw':req[15]['actualrawhistory10']=deepcopy(req[10]['actualrawhistory10'])
    elif mutation=='latent':req[10]['context3']['latent_history'][0][0]+=.125
    elif mutation=='action_tokens':req[10]['context3']['action_history'][0][0]+=.125
    elif mutation=='goal':req[5]['goal']=[1.]*192
    elif mutation=='diagnostic':req[10]['diagnosticraw5']=[[0.,0.]]*5
    elif mutation=='drop_boundary':del req[10]
    elif mutation=='drop_latent':del z[10]
    elif mutation=='foreign_successor':
        # I next input must be the EXACT own preceding saved successor.
        # R must use own encoded latent instead, not that successor.
        if mode=='I':res[5]['predicted_successor']['latent_history'][0][0]+=.125
        else:z[5][0]+=.125
    elif mutation=='actions':row['executed_raw'][10][0]+=.125
    elif mutation=='normalizer':task['action_std']=[2.,1.]
    with pytest.raises(ValueError):m.check_episode(*args)


def test_signed_zero_and_float32_exactness():
    with pytest.raises(ValueError):m.exact([[-0.,0.]],[[0.,0.]],'bit mismatch')
    with pytest.raises(ValueError):m.array([.1],(1,))
    with pytest.raises(ValueError):m.array([float('nan')],(1,))


def test_original_two_json_hash_conventions_are_not_interchangeable():
    x={'a':[1.,2.]}
    assert m.object_sha(x)==m.digest(b'{"a": [1.0, 2.0]}')
    assert m.packet_sha(x)==m.digest(b'{"a":[1.0,2.0]}\n')
    assert m.object_sha(x)!=m.packet_sha(x)


@pytest.mark.parametrize('payload',[b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":Infinity}'])
def test_json_fail_closed(payload):
    with pytest.raises(ValueError):m.decode(payload)


class RAM:
    def __init__(self):self.files={}
    def blob(self,path,expected=None,cap=64*1024**2):
        b=self.files[str(path)]
        m.require(len(b)<=cap and (expected is None or m.digest(b)==expected),'RAM SHA/bound')
        return b
    def json(self,path,expected=None):return m.decode(self.blob(path,expected))
    def put(self,path,value):self.files[str(path)]=m.canonical(value).encode()


def encoding_fixture():
    r=RAM();out=Path('/RAM');eid='e00_R';ns=eid+'_b10';step=10
    image=np.zeros((224,224,3),np.uint8);image[0,0]=[1,2,3]
    stream=io.BytesIO();Image.fromarray(image).save(stream,format='PNG');png=stream.getvalue()
    original=dict(path=str(out/(eid+'_plan_10.png')),file_sha256=m.digest(png),pixel_sha256=m.digest(image.tobytes()))
    mapping=dict(schema=m.PRODUCER+'.visual_bridge',episode_id=eid,actual_step=10,original_png=original,
                 legacy_episode_id=ns,legacy_step=5,alias_png=str(out/(ns+'_plan_05.png')))
    r.files[original['path']]=png;r.files[mapping['alias_png']]=png
    request=dict(schema='p1_moving_live_feedback_physical_20261027_v1.encode_request',episode_id=ns,observed_step=5,
                 png=dict(original,path=mapping['alias_png']),task_export='/task.gz',model_binding_sha256=m.object_sha({}))
    request['request_sha256']=m.object_sha(request)
    z=np.arange(192,dtype=np.float32);stream=io.BytesIO();np.save(stream,z,allow_pickle=False);payload=stream.getvalue()
    latent=dict(path=str(out/(ns+'_encode/native192.npy')),file_sha256=m.digest(payload),array_sha256=m.digest(z.tobytes()),
                dtype='float32',shape=[192]);r.files[latent['path']]=payload
    response=dict(schema='p1_moving_live_feedback_physical_20261027_v1.encode_response',episode_id=ns,observed_step=5,
                  request_sha256=request['request_sha256'],latent=latent,device_type='cuda',source_verified=True,
                  model_unchanged=True,model_hash_before='tensor',model_hash_after='tensor')
    receipt=dict(actual_step=10,mapping=mapping,request=request,response=response)
    r.put(out/(ns+'_encoding.json'),receipt);r.put(out/(ns+'_visual_bridge.json'),mapping)
    r.put(out/(ns+'_encode/receipt.json'),response)
    row=dict(episode_id=eid,encodings=[dict(step=10,receipt_sha256=m.object_sha(receipt),latent=latent)])
    events=[dict(kind='fresh_observation',episode_id=eid,index=12,
                 receipt=dict(original,phase='plan',step=10,observed_event_index=12)),
            dict(kind='boundary_encoded',episode_id=eid,index=15,actual_step=10,
                 encoding_sha256=m.object_sha(receipt),mapping_sha256=m.object_sha(mapping))]
    return r,out,row,step,dict(task_export='/task.gz'),{},'tensor',events


def test_actual_png_alias_bytes_to_encoding_output():
    args=encoding_fixture();z,obs=m.encoding(*args)
    assert np.array_equal(z,np.arange(192,dtype=np.float32))
    assert (obs['observed_index'],obs['encoded_index'],obs['boundary'])==(12,15,10)
    assert obs['observed_latent_sha256']==m.packet_sha(z.tolist())


@pytest.mark.parametrize('mutation',['alias','original','output','receipt','mapping','row','event','duplicate_event',
                                    'wrong_observation','causality','tensor'])
def test_visual_join_mutations_rejected(mutation):
    args=list(encoding_fixture());r,out,row,step,layer,models,tensor,events=args;ns='e00_R_b10'
    if mutation=='alias':r.files[str(out/(ns+'_plan_05.png'))]+=b'changed'
    elif mutation=='original':r.files[str(out/'e00_R_plan_10.png')]+=b'changed'
    elif mutation=='output':r.files[str(out/(ns+'_encode/native192.npy'))]+=b'changed'
    elif mutation=='receipt':r.put(out/(ns+'_encode/receipt.json'),{})
    elif mutation=='mapping':r.put(out/(ns+'_visual_bridge.json'),{})
    elif mutation=='row':row['encodings'][0]['receipt_sha256']='wrong'
    elif mutation=='event':events[-1]['encoding_sha256']='wrong'
    elif mutation=='duplicate_event':events.append(deepcopy(events[-1]))
    elif mutation=='wrong_observation':events[0]['episode_id']='e00_I'
    elif mutation=='causality':events[0]['index']=16;events[0]['receipt']['observed_event_index']=16
    elif mutation=='tensor':args[6]='foreign tensor'
    with pytest.raises(ValueError):m.encoding(*args)
