"""RAM saved arithmetic only: never import torch or execute a model/AD/physics."""
from copy import deepcopy
import gzip
from pathlib import Path
import subprocess
import sys
import zlib
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_history_joint30_grasp_cuda_20261029_v1 as m


@pytest.fixture(scope='module')
def task():
    p=m.inputs.parse(m.inputs.read_bound(m.inputs.UPSTREAM,m.inputs.UPSTREAM_SHA))
    s=p['slots'][0]
    e=m.inputs.parse(m.inputs.read_bound(s['task_file'],s['task_sha256'],compressed=True))
    return e['task']


def chain(payloads):
    tail='0'*64;out=[]
    for i,p in enumerate(payloads):
        e=dict(index=i,previous=tail,payload=p);tail=m.digest(e);out.append(m.raw(dict(e,sha256=tail)).rstrip(b'\n'))
    return out


def native_events(task,label='fixture/0'):
    z=np.asarray(task['root_context']['latent_history'],np.float32)[None]
    h=np.asarray(task['root_context']['action_history'],np.float32)[None]
    a=np.linspace(-.1,.1,10,dtype=np.float32).reshape(5,2)
    mean,std=(np.tile(task[k],5).astype(np.float32) for k in ('action_mean','action_std'))
    outgoing=np.concatenate((h,((a.reshape(10)-mean)/std)[None,None]),1)
    ae=np.full((1,3,192),.25,np.float32)
    predictor=np.full((1,3,192),-.5,np.float32)
    proj=np.linspace(-.3,.8,3*192,dtype=np.float32).reshape(3,192)
    payloads=[dict(kind='forward_attempt',index=0,label=label,z=m.pack(z),h=m.pack(h),raw=m.pack(a))]
    for k,(name,ins,out) in enumerate((('action_encoder',[outgoing],ae),('predictor',[z,ae],predictor),('pred_proj',[predictor.reshape(3,192)],proj))):
        if k==1:payloads.append(dict(kind='predict_attempt',index=0))
        payloads.extend([dict(kind='module_attempt',index=0,resource=name,inputs=[m.pack(x) for x in ins]),
            dict(kind='module_return',index=0,resource=name,output=m.pack(out))])
    zz=np.concatenate((z[:,1:],proj[-1:][None]),1);hh=outgoing[:,1:]
    payloads.extend([dict(kind='predict_return',index=0,output=m.pack(proj.reshape(1,3,192))),
        dict(kind='forward_return',index=0,z=m.pack(zz),h=m.pack(hh))])
    return payloads,(z,h,a),(zz,hh)


def test_fixed_all900_schedule_and_no_equal_cap_claim():
    p=list(m.phases());assert len(p)==900
    assert sum(k=='joint' for k,_,_ in p)==750
    assert [(k,j) for t,k,j in p if t=='sync' and j==0]==[(k,0) for k in range(49,750,50)]
    assert p[-1]==('sync',749,9)
    b=m.budget();assert b['predict_API']==2250+450+6==2706
    assert b['scalar_backward']==b['Adam_step']==900
    assert b['virtual_noise']==750 and b['pred_proj_rows']==8118
    assert m.contract()['total']['predict_API']==5412
    assert m.contract()['equal_budget_to_1902'] is False and m.contract()['official_reproduction'] is False
    assert all(m.contract()[k] is False for k in ('physical_release','P1_admission','P2_admission','fresh'))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape',[(),(30,),(1,2,192),(3,192)])
def test_native_array_roundtrip_scalar_shape_precision(dtype,shape):
    a=np.full(shape,.123456,dtype);b=m.pack(a);r=m.unpack(b,shape,np.dtype(dtype))
    assert r.shape==shape and r.dtype==dtype and r.tobytes()==a.tobytes()


@pytest.mark.parametrize('change',[
    lambda b:b.update(dtype='int64'),lambda b:b.update(shape=[-1]),lambda b:b.update(shape=[193]),
    lambda b:b.update(sha256='f'*64),lambda b:b.update(hex=b['hex']+'00'),lambda b:b.update(extra=0),
])
def test_array_mutation_refused(change):
    b=m.pack(np.ones(30,np.float64));change(b)
    with pytest.raises((ValueError,KeyError)):m.unpack(b)


def test_global30_projection_not_three_packet_balls():
    x=np.zeros(30);x[0]=m.BALL
    assert np.array_equal(m.project(x),x)
    assert np.linalg.norm(x[:10])>.75*np.sqrt(5)
    out=m.project(np.ones(30)*100)
    assert np.linalg.norm(out)<=m.BALL and out.shape==(30,)
    assert np.array_equal(m.project(out),out)
    a=m.chart(out);assert a.dtype==np.float32 and a.shape==(3,5,2)
    assert a.tobytes()==np.tanh(out).astype(np.float32).tobytes()


def test_original_H3_action_alignment_all_three_edges(task):
    root=task['root_context'];v=np.stack((np.full(192,1,np.float32),np.full(192,2,np.float32)))
    x=np.linspace(-.3,.3,30);a=m.chart(x)
    tokens=np.concatenate([m.inputs.core.normalized_tokens(task,aa) for aa in a])
    c0=m.lifted_context(root,v,0,tokens);c1=m.lifted_context(root,v,1,tokens);c2=m.lifted_context(root,v,2,tokens)
    assert c0==root
    assert c1['latent_history']==root['latent_history'][1:]+[v[0].tolist()]
    assert c2['latent_history']==root['latent_history'][2:]+v.tolist()
    assert c1['action_history']==root['action_history'][1:]+[tokens[0].tolist()]
    assert c2['action_history']==tokens[:2].tolist()
    assert np.asarray(c2['latent_history']).shape==(3,192)
    assert not np.array_equal(tokens[0],np.zeros(10))


def test_full_saved_native_bridge_without_neural_execution(task):
    events,args,want=native_events(task);c=m.Cursor(chain(events));b=m.Saved(task,c)
    got=b.step(*args,'fixture/0');c.done()
    assert all(np.array_equal(x,y) for x,y in zip(got,want,strict=True)) and b.n==1


@pytest.mark.parametrize('change',[
    lambda e:e[0].update(label='other'),
    lambda e:e[0].update(index=1),
    lambda e:e[1].update(resource='encoder'),
    lambda e:e[1]['inputs'].__setitem__(0,m.pack(np.zeros((1,3,10),np.float32))),
    lambda e:e[3].update(kind='unknown_predict'),
    lambda e:e[4]['inputs'].__setitem__(0,m.pack(np.zeros((1,1,192),np.float32))),
    lambda e:e[6]['inputs'].__setitem__(0,m.pack(np.ones((3,192),np.float32))),
    lambda e:e[-2].update(output=m.pack(np.ones((1,3,192),np.float32))),
    lambda e:e[-1].update(z=m.pack(np.zeros((1,3,192),np.float32))),
    lambda e:e.insert(0,dict(kind='extra')),
])
def test_mutated_saved_wiring_even_rehashed_refused(task,change):
    events,args,_=native_events(task);change(events)
    b=m.Saved(task,m.Cursor(chain(events)))
    with pytest.raises((ValueError,KeyError)):b.step(*args,'fixture/0')


def test_coherent_wrong_cost_not_trusted_flags():
    goal=np.zeros((1,192),np.float32);preds=[np.full((1,192),k,np.float32) for k in (1,2,3)]
    v=np.zeros((1,2,192),np.float32)
    e=dict(loss=m.pack(np.array(28,np.float32)),dyn=m.pack(np.array(14,np.float32)),dense=m.pack(np.array(14,np.float32)))
    m.loss_check(e,preds,v,goal,True)
    # All three persisted scalar fields changed coherently, valid SHA/shape.
    wrong={k:m.pack(m.unpack(a)*np.float32(2)) for k,a in e.items()}
    with pytest.raises(ValueError):m.loss_check(wrong,preds,v,goal,True)
    s=dict(loss=m.pack(np.array(9,np.float32)),dyn=None,dense=None);m.loss_check(s,[preds[-1]],v,goal,False)
    s['loss']=m.pack(np.array(8,np.float32))
    with pytest.raises(ValueError):m.loss_check(s,[preds[-1]],v,goal,False)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_saved_adam_first_step_and_moment_mutations(dtype):
    param=np.full(30,.1,dtype);grad=np.linspace(-.01,.01,30,dtype=dtype)
    before=m.blank((30,),dtype);am=(.1*grad).astype(dtype);aq=(.001*grad*grad).astype(dtype)
    after=dict(step=1,m=m.pack(am),q=m.pack(aq));lr=.005
    proposal=param-lr/(1-.9)*am/(np.sqrt(aq)/np.sqrt(1-.999)+1e-8)
    m.adam_check(before,after,param,grad,proposal.astype(dtype),lr)
    with pytest.raises(ValueError):m.adam_check(before,after,param,grad,proposal.astype(dtype)+dtype(.001),lr)
    bad=deepcopy(after);bad['step']=2
    with pytest.raises(ValueError):m.adam_check(before,bad,param,grad,proposal,lr)
    bad=deepcopy(after);bad['m']=m.pack(am*2)
    with pytest.raises(ValueError):m.adam_check(before,bad,param,grad,proposal,lr)


def test_cursor_unknown_extra_reordered_root_denominator():
    c=m.Cursor(chain([dict(root_slot=1,event=dict(kind='start'))]));c.root=0
    with pytest.raises(ValueError):c.get('start')
    c=m.Cursor(chain([dict(kind='unknown')]))
    with pytest.raises(ValueError):c.get('start')
    c=m.Cursor(chain([dict(kind='start'),dict(kind='extra')]))
    c.get('start')
    with pytest.raises(ValueError):c.done()


def test_bounded_journal_single_stream_truncation_and_hash(tmp_path):
    lines=chain([dict(kind='RAM')]);valid=gzip.compress(b'\n'.join(lines)+b'\n',mtime=0)
    p=tmp_path/'trace.gz';p.write_bytes(valid)
    c=m.Cursor(m.journal_lines(p));c.get('RAM');c.done()
    for bad in (valid[:-3],valid+b'extra',valid+valid,gzip.compress(lines[0],mtime=0)):
        p.write_bytes(bad)
        with pytest.raises(ValueError):list(m.journal_lines(p))
    e=m.inputs.parse(lines[0]);e['payload']['kind']='changed'
    with pytest.raises(ValueError):m.Cursor([m.raw(e).rstrip(b'\n')]).get('changed')


def test_exclusive_store_and_attempt_return_partial_census(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'CAP',128*1024);monkeypatch.setattr(m,'FINAL',4096);monkeypatch.setattr(m,'FLOOR',0)
    p=tmp_path/(m.NAME+'_RAM');s=m.Store(p,m.time.monotonic())
    s.event(dict(root_slot=0,event=dict(kind='predict_attempt',index=0)))
    assert s.counts=={'0:predict_attempt':1} # No manufactured return on failure.
    s.close();assert list(m.journal_lines(p/'events.jsonl.gz'))
    with pytest.raises(ValueError):s.event(dict(kind='after'))
    with pytest.raises(ValueError):m.Store(p,m.time.monotonic())
    with pytest.raises(ValueError):s.write('../escape',b'x')


def test_complete_census_is_per_root_and_all_axes():
    c=m.expected_census()
    for s in ('0','1'):
        assert c[s+':predict_attempt']==c[s+':predict_return']==2706
        assert c[s+':backward_attempt']==c[s+':backward_return']==900
        assert c[s+':chart_backward']==900 and c[s+':joint_optimizer_reset']==15
        for name in ('action_encoder','predictor','pred_proj'):assert c[s+':module_attempt:'+name]==2706
    assert c['load_intent']==c['consumed_buffer_frozen']==2


def test_full15_serial_and_segmented_align_each_raw_packet(task,monkeypatch):
    x=np.linspace(-.2,.2,30);commands=m.chart(x);observed=[]
    def fixture(z,h,a,label):
        # Symbolic saved-boundary fixture only, NOT a dynamics experiment.
        k=len(observed);observed.append((z.copy(),h.copy(),a.copy(),label))
        token=m.inputs.core.normalized_tokens(task,a)[0]
        zz=np.concatenate((z[:,1:],np.full((1,1,192),k+1,np.float32)),1)
        hh=np.concatenate((h[:,1:],token[None,None]),1)
        return zz,hh
    b=m.Saved(task,None);monkeypatch.setattr(b,'step',fixture)
    _,path=b.serial(x,'whole')
    assert len(observed)==len(path)==3
    assert np.array_equal(observed[0][0][0],task['root_context']['latent_history'])
    for k in range(3):
        assert observed[k][3]=='whole/'+str(k) and np.array_equal(observed[k][2],commands[k])
        if k:assert observed[k][0][0].tolist()==path[k-1]['latent_history']
    observed.clear();_,seg=b.serial(x,'segment',True)
    assert seg==path and [e[-1] for e in observed]==['segment/0','segment/1','segment/2']


def test_source_closure_has_real_installed_classes_and_adam_without_importing_torch():
    files={str(p) for p in m.source_files()}
    assert str(Path(m.__file__).resolve()) in files
    assert str(m.ROOT/'tests'/('test_'+m.NAME+'.py')) in files
    assert str(m.ROOT/'innerdocs'/(m.NAME+'.md')) in files
    assert any(p.endswith('/torch/optim/adam.py') for p in files)
    for entry in m.installed.installed_class_sources().values():assert entry['path'] in files


def test_auth_refused_before_prepare_I_O_or_torch(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('must not touch prepare before authority'))
    with pytest.raises(ValueError,match='authorization'):m.run('missing','f'*64,'missing','OTHER')


def test_cold_no_neural_foreign_network_import_or_invocation():
    code='''import builtins,sys
orig=builtins.__import__
def guard(name,*a,**k):
 if name.split('.')[0] in ('torch','tensorflow','transformers','gymnasium','grasp_solver','sklearn'):
  raise AssertionError('forbidden '+name)
 return orig(name,*a,**k)
builtins.__import__=guard
def audit(event,args):
 if event in ('socket.__new__','socket.connect','socket.getaddrinfo'):raise AssertionError('network')
sys.addaudithook(audit)
from experiments.spectral_reachability import agent_p2_history_joint30_grasp_cuda_20261029_v1 as m
assert m.budget()['predict_API']==2706
try:m.run('none','f'*64,'none','wrong')
except ValueError:pass
else:raise AssertionError('authorization')
assert 'torch' not in sys.modules
'''
    r=subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,capture_output=True,text=True,timeout=30)
    assert r.returncode==0,r.stderr


def test_new_source_no_outcome_selector_training_or_globals_patch():
    src=Path(m.__file__).read_text()
    assert 'model.predict=' not in src and '.forward =' not in src
    assert 'torch.load(' not in src and 'weights_only=False' not in src
    assert 'min(results' not in src and 'loss.backward()' in src
    assert 'normalized_model_variant_complete_not_admitted' in src
