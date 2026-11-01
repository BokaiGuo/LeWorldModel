"""RAM arithmetic/mutations and small saved ABI only; never production strict."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_six_target_prediction_comparison_20261029_v1 as m


def data(i):
    return dict(z=np.arange(576,dtype=np.float32).reshape(1,3,192)/np.float32(1024)+np.float32(i//3),
        a=np.arange(20,dtype=np.float32).reshape(1,2,10)/np.float32(1024),
        target=np.full((1,192),.125+i*.01,np.float32))


def manifest():
    slots=[]
    for i in range(6):
        v=data(i)
        source=dict(row=dict(origin=dict(physical_time=None if i%3==0 else dict(cp=5)))) if i%3!=2 else dict(plan=dict(source_time=None))
        slots.append(dict(slot=i,root=i//3,condition='off',arm=m.runtime.ARMS[i%3],source=source,
            controller_payload={k:v.tolist() for k,v in v.items()},input_sha256={k:m.runtime.core.digest(v.tobytes()) for k,v in v.items()}))
    return dict(schema=m.runtime.TAG+'.prepare',slots=slots)


def trace(i=0):
    core=m.runtime.core
    def check(x):assert x.dtype==np.float32 and np.isfinite(x).all()
    kernels=core.low.Kernels(lambda xs,dim:np.concatenate(xs,axis=dim),lambda xs,dim:np.stack(xs,axis=dim),check,
        lambda a:np.repeat(a.sum(-1,keepdims=True,dtype=np.float32),192,axis=-1),
        lambda z,a:z+a/np.float32(16),lambda x:x.copy(),'ram_fixture')
    v=data(i)
    return core.track_fixture(v['z'],v['a'],v['target'],spec=m.runtime.tracker.spec(),kernels=kernels)


def goals():
    return [dict(root=r,value=np.full(192,r+.75,np.float32),source=dict(goal_sha256=m.runtime.core.digest(np.full(192,r+.75,np.float32).tobytes()))) for r in range(2)]


def test_scalar_independent_f64():
    z=np.arange(192,dtype=np.float32);g=np.zeros(192,np.float32)
    assert m.mse64(z,g)==sum(int(x)**2 for x in z)/192
    assert m.mse64(g,-g)==0
    assert m.mse64(np.full(192,np.float32(.1)),g)==float(np.float32(.1))**2


@pytest.mark.parametrize('fault',['shape','dtype','nan','inf','list'])
def test_scalar_reject(fault):
    z=np.zeros(192,np.float32);g=z.copy()
    if fault=='shape':z=z[None]
    elif fault=='dtype':z=z.astype(np.float64)
    elif fault=='nan':z[0]=np.nan
    elif fault=='inf':g[0]=np.inf
    else:z=z.tolist()
    with pytest.raises(ValueError):m.mse64(z,g)


def test_all_six_cp_pairs_no_cross_target_ranking():
    cache={i:trace(i) for i in range(6)}
    out=m.panel(manifest(),dict(rows=[dict(status='returned') for _ in range(6)]),goals(),cache.__getitem__)
    assert out['denominator']==dict(roots=2,slots=6,checkpoints_per_slot=4,total_rows=24,pairs=24)
    for i in range(6):
        rows=out['rows'][4*i:4*i+4]
        assert [r['cp_public_step'] for r in rows]==[0,5,10,15]
        tr,a=cache[i];pred=a[tr['replays'][0]['rollout']['prediction']]
        for j,r in enumerate(rows):
            z=data(i)['z'][0,-1] if j==0 else pred[0,0,j-1]
            assert r['task_goal_mse64']==m.mse64(z,goals()[i//3]['value'])
            assert r['tracking_target_mse64']==m.mse64(z,data(i)['target'][0])
            assert r['task_goal_reduction_from_cp0']==rows[0]['task_goal_mse64']-r['task_goal_mse64']
            assert r['native_metrics'] is None and r['scientific_admission'] is False
        assert rows[0]['task_goal_reduction_from_cp0']==0
    for p in out['same_task_goal_pairs']:
        j=m.CP.index(p['cp_public_step'])
        assert p['right_minus_left_task_goal_mse64']==out['rows'][4*p['right_slot']+j]['task_goal_mse64']-out['rows'][4*p['left_slot']+j]['task_goal_mse64']
        assert p['cross_target_tracking_loss_difference'] is None


@pytest.mark.parametrize('fault',['partial','error','replay_count','replay_exact','phase','returned','input','mean','future','prediction','checkpoint','final','cost','signedzero'])
def test_trace_mutations_fail_closed(fault):
    tr,a=trace();v=data(0);w,s=tr['replays']
    if fault=='partial':tr['completed']=False
    elif fault=='error':tr['errors']=['failure']
    elif fault=='replay_count':tr['replays'].pop()
    elif fault=='replay_exact':tr['selected_mean_replay_exact']=False
    elif fault=='phase':w['phase']='elite'
    elif fault=='returned':s['returned']=False
    elif fault=='input':v['target']=v['target']+np.float32(1)
    elif fault=='mean':a[tr['selected']['mean']]=np.zeros((1,2,10),np.float32)
    elif fault=='future':a[s['future']]=a[s['future']].copy()+np.float32(1)
    elif fault=='prediction':a[s['rollout']['prediction']]=a[s['rollout']['prediction']].copy()+np.float32(1)
    elif fault=='checkpoint':a[w['rollout']['checkpoints'][0]['z']]=np.zeros((1,4,192),np.float32)
    elif fault=='final':a[s['rollout']['final']['a']]=np.zeros((1,5,10),np.float32)
    elif fault=='cost':a[s['cost']]=a[s['cost']].copy()+np.float32(1)
    else:
        a[tr['inputs']['a']]=a[tr['inputs']['a']].copy();a[tr['inputs']['a']][0,0,0]=np.float32(-0.)
    with pytest.raises(ValueError):m.checkpoint_vectors(tr,a,v)


def actual_metadata():
    rr=m.actual_refs();return rr,[m.read(rr[k]['path'],rr[k]['sha256']) for k in ('manifest','report','receipt')]


def test_actual_small_metadata_only():
    refs,(mm,r,c)=actual_metadata();m.check_complete(mm,r,c,refs)
    assert c['predict_API']==72 and c['predictor_samples']==324
    for p,h in m.PINS.items():assert m.sha(p)==h


@pytest.mark.parametrize('fault',['partial','reportsha','manifestsha','receiptstatus','denominator','local','model','source','slot','missing'])
def test_complete_authority_metadata_mutations(fault):
    refs,(mm,r,c)=actual_metadata()
    if fault=='partial':r['completed']=False
    elif fault=='reportsha':c['report_sha256']='0'*64
    elif fault=='manifestsha':r['manifest_sha256']='0'*64
    elif fault=='receiptstatus':c['status']='ok'
    elif fault=='denominator':c['denominator']['total']=5
    elif fault=='local':c['local_checks'][0]['local_check']={}
    elif fault=='model':r['model_after']={}
    elif fault=='source':r['sources_after']={}
    elif fault=='slot':r['rows'][0]['root']=1
    else:r['rows'][0]['status']='missing'
    with pytest.raises(ValueError):m.check_complete(mm,r,c,refs)


def goal_abi():
    refs,(mm,_,_)=actual_metadata();apref=mm['authorities']['arbitrary_prepare'];ap=m.read(apref['path'],apref['sha256'])
    prref=ap['slots'][0]['task']['coordinate_refs']['task_goal'];pr=m.read(prref['path'],prref['sha256']);folder=Path(prref['path']).parent
    def get(name):
        d=pr['arrays'][name];p=folder/name
        assert mm['sources'][str(p)]==ap['bindings'][str(p)]==d['file_sha256']
        return m.runtime.tracker.parity.load_array(folder,name,d)
    return mm,ap,pr,get


def test_actual_goal_small_array_abi_only():
    g=m.resolve_goals(*goal_abi())
    assert [x['root'] for x in g]==[0,1] and all(x['value'].shape==(192,) for x in g)


@pytest.mark.parametrize('fault',['pointer','task','goal','history','source','inputhash','identity'])
def test_goal_source_mutations(fault):
    mm,ap,pr,get=goal_abi();s=ap['slots'][0]
    if fault=='pointer':s['goal_pointer']='/rows/0/pixels/0/paths/hi/output'
    elif fault=='task':s['task']['goal'][0]+=1
    elif fault=='goal':s['goal'][0]+=1
    elif fault=='history':s['z'][0][0][0]+=1
    elif fault=='source':s['source_arrays']['goal']=s['source_arrays']['z']
    elif fault=='inputhash':s['input_sha256']['goal']='0'*64
    else:s['task_sha256']='0'*64
    with pytest.raises(ValueError):m.resolve_goals(mm,ap,pr,get)


def test_wrong_authority_stops_before_upstream(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('production upstream forbidden')
    monkeypatch.setattr(m.runtime,'build',forbidden)
    rr=m.actual_refs();rr['receipt']['sha256']='0'*64
    with pytest.raises(ValueError):m.build(rr)


@pytest.mark.parametrize('tamper',[False,True])
def test_build_entire_receipt_reconstruction_ram(monkeypatch,tmp_path,tamper):
    """Exercise real build control flow with explicit inert authentication callbacks."""
    refs={k:dict(path=str(tmp_path/k),sha256=k+'hash') for k in ('manifest','report','receipt')}
    apref=dict(path=str(tmp_path/'ap'),sha256='apsha');prref=dict(path=str(tmp_path/'pr'),sha256='prsha')
    mm=dict(upstream_authentication={},sources={apref['path']:'apsha',prref['path']:'prsha'},
        full_receipts=dict(arbitrary={}),authorities=dict(arbitrary_prepare=apref,arbitrary_receipt={}))
    rr=dict(whole_wall_seconds_before_report=1)
    cc=dict(completion_sha256='done',saved_consume_authentication={})
    reconstructed=dict(report_sha256='reporthash',manifest_sha256='manifesthash',completion_sha256='done',
        saved_consume_authentication={},upstream_arbitrary_receipt={})
    cc.update(reconstructed)
    if tamper:cc['unauthenticated_extra']='must reject'
    cc['accounting27']={}
    # accounting belongs to the reconstructed strict body, not an appended claim.
    done=dict(schema=m.runtime.TAG+'.completion',complete=True,report_sha256='reporthash',
        whole_wall_through_report_fsync_hash=2,boundary='Store start before verify through report fsync/hash; completion acknowledgement excluded')
    ap=dict(slots=[dict(task=dict(coordinate_refs=dict(task_goal=prref))) for _ in range(2)])
    docs={refs['manifest']['path']:mm,refs['report']['path']:rr,refs['receipt']['path']:cc,
        str(tmp_path/'completion.json'):done,apref['path']:ap,prref['path']:dict(arrays={})}
    monkeypatch.setattr(m,'actual_refs',lambda:refs)
    monkeypatch.setattr(m,'read',lambda p,h:deepcopy(docs[str(p)]))
    monkeypatch.setattr(m,'check_complete',lambda *a:None)
    monkeypatch.setattr(m.runtime,'build',lambda a:deepcopy(mm))
    monkeypatch.setattr(m.runtime,'check_authentication',lambda *a:None)
    monkeypatch.setattr(m.runtime,'analyze_saved',lambda *a:dict(accounting27={}))
    monkeypatch.setattr(m,'closure',lambda:{})
    monkeypatch.setattr(m,'resolve_goals',lambda *a:[dict(source={}) for _ in range(2)])
    monkeypatch.setattr(m,'panel',lambda *a:dict(rows=[],same_task_goal_pairs=[]))
    hashes={v['path']:v['sha256'] for v in refs.values()};hashes.update({apref['path']:'apsha',prref['path']:'prsha',str(tmp_path/'completion.json'):'done'})
    monkeypatch.setattr(m,'sha',lambda p:hashes[str(p)])
    if tamper:
        with pytest.raises(ValueError,match='entire six strict receipt'):m.build(refs)
    else:
        out=m.build(refs)
        assert out['full_strict_receipt']==cc and out['winner'] is None
        assert out['new_calls']==dict(encoder=0,predict=0,AD=0,physics=0)


def test_exclusive_write_verify_fixture(tmp_path,monkeypatch):
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    payload=dict(authorities={'fixture':True},rows=[],schema=m.TAG+'.comparison')
    path=tmp_path/'results'/(m.TAG+'_fixture.json');ref=m.write(path,payload)
    monkeypatch.setattr(m,'build',lambda refs:deepcopy(payload))
    assert m.verify(path,ref['sha256'])==dict(verified=True,new_neural=0,physics=0)
    with pytest.raises(FileExistsError):m.write(path,payload)
    with pytest.raises(ValueError):m.write(tmp_path/'wrong.json',payload)
    payload['rows']=[1]
    with pytest.raises(ValueError):m.verify(path,ref['sha256'])


def test_cold_import_no_neural():
    code='import sys; from experiments.spectral_reachability import '+m.TAG+'; assert "torch" not in sys.modules'
    subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,check=True)


def test_cli_dispatch_fixture(monkeypatch,capsys,tmp_path):
    called=[]
    monkeypatch.setattr(m,'build',lambda refs:called.append(refs) or {'fixture':True})
    monkeypatch.setattr(m,'write',lambda p,v:dict(fixture=v))
    args=['build','--output',str(tmp_path/'out')]
    for k,v in m.actual_refs().items():args+=['--'+k,v['path'],'--'+k+'-sha256',v['sha256']]
    m.main(args)
    assert called==[m.actual_refs()] and json.loads(capsys.readouterr().out)=={'fixture':{'fixture':True}}
