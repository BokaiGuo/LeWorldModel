import base64
import copy
import hashlib
import json
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_c3_k17_replay_guard_bridge_20261029_v1 as m

def blob(x):
    a=np.asarray(x,dtype='<f4'); raw=a.tobytes()
    return dict(shape=list(a.shape),dtype=a.dtype.str,sha256=hashlib.sha256(raw).hexdigest(),data=base64.b64encode(raw).decode())

def unblob(b):
    raw=base64.b64decode(b['data'],validate=True)
    if hashlib.sha256(raw).hexdigest()!=b['sha256']: raise ValueError('blob hash')
    return np.frombuffer(raw,dtype=b['dtype']).reshape(b['shape'])

def example():
    task=dict(binding=dict(task_id='fixture',root_sha256='a'),action_mean=[0.,0.],action_std=[1.,1.],
      root_context=dict(action_history=[[0.]*10]*2,latent_history=[[0.]*192]*3))
    actions=[[0.,0.]]*5; path=[{'fixture':True}]; plan=dict(actions=actions,path=path)
    scope=lambda name:dict(mode='fixedH1',K=17,phase='verification',replay_mode=name)
    records=[]; events=[]
    for n,name in enumerate(('whole','segmented')):
        packet=dict(kind='program_packet',phase=name,step=0,arrays={k:blob(v) for k,v in dict(
          token=np.zeros(10),raw=np.zeros((5,2)),history=np.zeros((1,2,10)),latent=np.zeros((1,3,192)),
          mean=np.zeros(2),std=np.ones(2),outgoing=np.zeros((1,3,10))).items()})
        ae=dict(kind='io_input',phase=name,step=0,arrays={'action_encoder:0':blob(np.zeros((1,3,10)))})
        call=[dict(kind='native_call_attempt',method='replay',task=copy.deepcopy(task),request=dict(
          parent_context=copy.deepcopy(task['root_context']),actions=copy.deepcopy(actions),segmented=name=='segmented')),
          packet,ae,dict(kind='native_call_return',result=copy.deepcopy(path))]
        records.append(dict(call_id=n,outer_index=n,transcript=call))
        events.append(dict(index=n,resource='replay',status='returned',scope=scope(name)))
    tree=dict(mode='fixedH1',K=17,backend_kind='HOST_CUDA',status='tree_complete_not_admitted',
      root_task_sha256=m.digest(task),model_plan=plan,replays=[dict(mode=x,status='returned',path=copy.deepcopy(path))
      for x in ('whole','segmented')],resource_receipt=dict(events=events),native_calls=records)
    return task,tree,dict(slot=3,root=0,mode='fixedH1',K=17)

def extract(t,x,i):
    return m.extract(t,x,i,lambda slot,cid,call:dict(sha256=hashlib.sha256(m.canon(call)).hexdigest()),lambda c:True,unblob)

def test_both_paid_replay_ae_tokens_and_guard():
    t,x,i=example(); v=extract(t,x,i)
    assert len(v['replays'])==2 and v['replays'][0]['token_f32_hex']==v['replays'][1]['token_f32_hex']
    assert not v['physical_dispatch_ready'] and v['replays'][0]['action_guard']['actual_replay_tokens_exact']

@pytest.mark.parametrize('change',['scope','outer','request','path','root','raw','mean','history','latent','token','ae','phase','blob','duplicate','capsule','audit'])
def test_tamper_fails(change):
    t,x,i=example(); call=x['native_calls'][0]['transcript']; p=call[1]['arrays']; ae=call[2]['arrays']
    if change=='scope': x['resource_receipt']['events'][0]['scope']['phase']='program'
    if change=='outer': x['native_calls'][0]['outer_index']=9
    if change=='request': call[0]['request']['segmented']=True
    if change=='path': call[-1]['result']=[]
    if change=='root': x['root_task_sha256']='bad'
    if change=='raw': p['raw']=blob(np.ones((5,2)))
    if change=='mean': p['mean']=blob(np.ones(2))
    if change=='history': p['history']=blob(np.ones((1,2,10)))
    if change=='latent': p['latent']=blob(np.ones((1,3,192)))
    if change=='token': p['token']=blob(np.ones(10))
    if change=='ae': ae['action_encoder:0']=blob(np.ones((1,3,10)))
    if change=='phase': call[1]['phase']='segmented'
    if change=='blob': p['token']['sha256']='0'*64
    if change=='duplicate': call.insert(1,copy.deepcopy(call[1]))
    cap=(lambda *a:None) if change=='capsule' else lambda *a:dict(sha256='f'*64)
    audit=(lambda *a:False) if change=='audit' else lambda *a:True
    with pytest.raises((ValueError,KeyError,TypeError)): m.extract(t,x,i,cap,audit,unblob)

def test_no_paid_token_and_no_dispatch(monkeypatch):
    t,x,i=example(); v=extract(t,x,i)
    monkeypatch.setattr(m.guard,'history_guard',lambda **kw:dict(accepted=True))
    live={'goal_frame':{'receipt':{'png':'b0.png'}}}
    result=m.check_live_only(v,t,{},live); assert not result['physical_dispatch_ready']
    v['replays'][1]['token_f32_hex']='00'
    with pytest.raises(ValueError): m.check_live_only(v,t,{},live)

def test_project_filename_only():
    x={'receipt':{'png':'branch.png','file_sha256':'a','pixel_sha256':'b'},'pre':{},'post':{}}
    y=m.frame_projection(x); assert y['receipt']=={'file_sha256':'a','pixel_sha256':'b'}
    assert x['receipt']['png']=='branch.png'
    x['receipt']['png']='../x.png'
    with pytest.raises(ValueError): m.frame_projection(x)

def test_source_reference_four_rows_and_png(tmp_path,monkeypatch):
    monkeypatch.setattr(m.handoff,'PHYSICAL',tmp_path/'report.json')
    monkeypatch.setattr(m.handoff,'history_export',lambda *args:None)
    data=b'fixture PNG bytes'; digest=hashlib.sha256(data).hexdigest()
    inv=dict(goal_state=[1]*7,goal_pose=[2],renderer={'fixed':True},snapshot={'state':[3]*7})
    rows=[]
    for root in (0,1):
        for arm in (0,1):
            name=f'{root}_{arm}.png'; (tmp_path/name).write_bytes(data)
            receipt=dict(phase='plan',step=0,overlay_index=0,png=name,file_sha256=digest,
              pixel_sha256='pixel',state=[3]*7,snapshot=inv['snapshot'])
            frame=dict(pre=copy.deepcopy(inv),post=copy.deepcopy(inv),receipt=receipt)
            rows.append(dict(root_slot=root,task_id='t',root_sha256='r',root_state=[3]*7,
              goal=[1]*7,goal_pose=[2],initial_exact=True,prefix51={'frames':[frame]}))
    phys=dict(source_roots=[dict(root_slot=i,root_sha256='r',root_state=[3]*7,goal=[1]*7,goal_pose=[2],renderer={'fixed':True}) for i in (0,1)],
      source_slots=[{'index':0},{'index':1}],warm_traces=[[{'receipt':{'step':s,'overlay_index':k}} for s,k in m.guard.REF_ORDER]]*2)
    rec=dict(physical_report_sha256=m.handoff.PHYSICAL_SHA,status='h3_witness4_control20_native_recorded_not_admitted',
      saved_PNG_verified=384,rows=copy.deepcopy(rows))
    task=dict(binding={'task_id':'t','root_sha256':'r'}); export={'task':{'root_raw_action_history':[[0.,0.]]*10}}
    ref=m.source_reference(0,task,phys,{'rows':rows},rec,export,'sourcehash')
    assert ref['reset_identity']=={'index':0} and 'png' not in ref['goal_frame']['receipt']
    (tmp_path/'0_1.png').write_bytes(b'tampered')
    with pytest.raises(ValueError):m.source_reference(0,task,phys,{'rows':rows},rec,export,'sourcehash')
    (tmp_path/'0_1.png').write_bytes(data); rec['rows'][1]['goal']=[9]*7
    with pytest.raises(ValueError):m.source_reference(0,task,phys,{'rows':rows},rec,export,'sourcehash')

def test_tempdisk_write_verify_tamper(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path); (tmp_path/'results').mkdir()
    monkeypatch.setattr(m,'build',lambda:dict(denominator=6,execution_authorized=False))
    path=tmp_path/'results'/(m.NAME+'_fixture.json')
    monkeypatch.setattr(sys,'argv',['x','analyze','--output',str(path)]);m.main(); h=m.sha(path)
    monkeypatch.setattr(sys,'argv',['x','verify','--output',str(path),'--sha256',h]);m.main()
    path.write_text(json.dumps({'denominator':0}))
    with pytest.raises(ValueError): m.main()
