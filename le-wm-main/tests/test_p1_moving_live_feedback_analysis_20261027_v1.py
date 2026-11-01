"""In-memory saved contracts only. No neural, physics, files, or actual report."""
from copy import deepcopy
import gzip
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import pytest
from experiments.spectral_reachability import p1_moving_live_feedback_analysis_20261027_v1 as m


def fixture():
    physical = dict(constructor=12, reset=36, set_state=72, goal_render=36, goal_setter=48,
                    goal_bind=12, public_step=240, physics_step=2688, explicit_render=288)
    c = dict(layers=[list(x) for x in m.LAYERS], modes=list(m.MODES), episodes=12, feedback_step=5,
             horizon=10, warmup=10, with_target=False, neural_caps=deepcopy(m.NEURAL), physical_caps=physical)
    layers = [dict(source_slot=s, arm=a, task_id=f'task{s}', task_export=f'/RAM/task{s}.json.gz',
                   task_sha256=m.TASK_SHAS[s], selection_sha256='c'*64, commands=np.full((10,2),np.float32(.1)).astype(float).tolist())
              for s,a in m.LAYERS]
    p = dict(schema=m.DRIVER+'.prepare', contract=c, layers=layers,
             bindings={f'/RAM/task{s}.json.gz':m.TASK_SHAS[s] for s in (0,1)})
    p['bindings'][str(Path(m.__file__).with_name(m.DRIVER+'.py').resolve())]=m.DRIVER_SHA
    rows = []; latent = {}; events = [dict(kind='run_started', prepare=deepcopy(p))]; op = 0
    counts = {k:dict(attempted=0,completed=0) for k in dict(physical, **m.NEURAL)}
    for i,(s,a) in enumerate(m.LAYERS):
        for mode in m.MODES:
            ep = f'e{i:02}_{mode}'; cp = []; raw = np.asarray(layers[i]['commands'][:5],np.float32)
            for step in (0,5,10):
                cp.append(dict(phase='plan', step=step, state=[0.,0.,float(step),0.,0.], goal=[0.]*5,
                               snapshot=dict(state=[0.,0.,float(step),0.,0.]), pixel_sha256=str(step)*32 if step==10 else 'd'*64))
            r = dict(episode_id=ep,source_slot=s,prefix_arm=a,mode=mode,status='completed10',initial_exact=True,
                     full_horizon=True,native_success=True,terminated=True,truncated=False,
                     first_error=None,secondary_errors=[],checkpoints=cp,final_state=cp[-1]['state'],
                     final_metrics=m.native_metrics(cp[-1]['state'],cp[-1]['goal']))
            r['prefix_pairing']=dict(pixel_sha256=cp[1]['pixel_sha256'],state=cp[1]['state'],snapshot=cp[1]['snapshot'],
                actual_raw_sha256=m.sha_bytes(raw.tobytes()),goal_sha256=m.digest(cp[1]['goal']))
            for step in (5,10):
                z=np.full(192,1. if step==5 else {'O':3.,'I':2.,'R':1.}[mode],np.float32)
                path=f'/RAM/run/{ep}_encode{step}/native192.npy';latent[path]=z
                r[f'encoding{step}']=dict(schema=m.DRIVER+'.encode_response',episode_id=ep,observed_step=step,
                    device_type='cuda',source_verified=True,model_unchanged=True,model_hash_before='e'*64,model_hash_after='e'*64,
                    resources={k:dict(attempted=1,completed=1) for k in ('encoder','projector')},
                    latent=dict(path=path,dtype='float32',shape=[192],array_sha256=m.sha_bytes(z.tobytes()),file_sha256='f'*64))
                r[f'actual_Q{step}']=m.observed_q(z,np.zeros(192,np.float32))
            rows.append(r)
            operations = dict(encoder=2,projector=2)
            if mode!='O':operations.update(m.SUFFIX)
            for resource, units in operations.items():
                region='encoding' if resource in ('encoder','projector') else 'compiling'
                for kind in ('operation_attempted','operation_completed'):
                    events.append(dict(kind=kind,resource=resource,operation_id=op,episode_id=ep,units=units,region=region))
                    counts[resource]['attempted' if kind.endswith('attempted') else 'completed']+=units
                op+=1
    report=dict(schema=m.DRIVER+'.report',status='live_feedback_calibration_recorded',contract=deepcopy(c),rows=rows,
        fixed_denominator=12,independent_tasks=2,P1_admission=False,P4_admission=False,training=False,
        independent_confirmation=False,errors=[],counts=counts)
    events.append(dict(kind='run_finished',status=report['status'],counts=counts,errors=[]))
    return report,p,events,{s:np.zeros(192,np.float32) for s in (0,1)},latent


def run(data):
    r,p,e,g,z=data
    return m.analyze(r,p,e,g,lambda receipt:z.get(receipt['path']))


def test_full12_paired_direction_budget_and_no_mutation():
    data=fixture();before=deepcopy(data[:3]);out=run(data)
    assert data[:3]==before and len(out['rows'])==12 and len(out['contrasts'])==12
    assert out['fixed_episode_denominator']==12 and out['independent_tasks']==2 and out['observed_endpoint_count']==24
    assert [x['endpoints']['10']['Q'] for x in out['contrasts'][:3]]==[-5.,-3.,-8.]
    assert [x['suffix_budget_contrast']['predict']['completed'] for x in out['contrasts'][:3]]==[5.,0.,5.]
    assert all(x['endpoints']['5']['Q']==0 for x in out['contrasts'])
    assert all(x['suffix_budget_matches_full_protocol'] for x in out['rows'])
    assert out['census']['total']['predict']['completed']==40
    assert out['census']['total']['predictor']['completed']==40
    assert out['census']['total']['vjp_directions']['completed']==1536
    assert out['significance'] is None and out['P1_admission'] is False and not out['independent_samples']


@pytest.mark.parametrize('missing',['row','checkpoint','encoding','latent','journal'])
def test_missing_null_preserves_full_denominator(missing):
    data=list(fixture());r,p,e,g,z=data
    if missing=='row':r['rows'].pop(2)
    elif missing=='checkpoint':
        r['rows'][2]['checkpoints']=[c for c in r['rows'][2]['checkpoints'] if c['step']!=10]
    elif missing=='encoding':r['rows'][2].pop('encoding10')
    elif missing=='latent':z.pop(r['rows'][2]['encoding10']['latent']['path'])
    else:data[2]=None
    out=run(data);assert len(out['rows'])==12 and len(out['contrasts'])==12
    if missing=='journal':assert all(x['suffix_budget_actual'] is None for x in out['rows'])
    else:
        assert out['rows'][2]['endpoints']['10']['observed_Q'] is None
        assert out['contrasts'][1]['endpoints']['10']['Q'] is None
        assert out['rows'][2]['issues']


def test_partial_prefix_encoding_still_measured_without_driver_final_q():
    data=fixture();row=data[0]['rows'][2];row.pop('actual_Q5');row.pop('actual_Q10')
    row.pop('encoding10');row['checkpoints']=[c for c in row['checkpoints'] if c['step']!=10]
    row.update(status='failed',full_horizon=False,first_error=dict(stage='suffix'))
    out=run(data);assert out['rows'][2]['endpoints']['5']['observed_Q']==1
    assert out['rows'][2]['endpoints']['10']['observed_Q'] is None
    assert out['rows'][2]['first_error']==dict(stage='suffix')


def test_prefix_mismatch_not_silently_a_matched_contrast():
    data=fixture();r=data[0]['rows'][2];r['initial_exact']=False
    out=run(data)
    assert out['contrasts'][1]['prefix_pairing_exact'] is False
    assert all(x is None for x in out['contrasts'][1]['endpoints']['10'].values())
    assert out['contrasts'][0]['endpoints']['10']['Q']==-5


@pytest.mark.parametrize('bad',['unknownmode','boolslot','duplicate','schema','budget','subsetlayers','cpu','q','pairingspoof','changedgoal','native','unmatchedreturn','cap','count'])
def test_failclosed_wrong_identity_and_present_corruption(bad):
    data=fixture();r,p,e,g,z=data;row=r['rows'][0]
    if bad=='unknownmode':row['mode']='S'
    elif bad=='boolslot':row['source_slot']=False
    elif bad=='duplicate':r['rows'].append(deepcopy(row))
    elif bad=='schema':r['schema']='other.report'
    elif bad=='budget':p['contract']['neural_caps']['predict']=41
    elif bad=='subsetlayers':p['layers'].pop()
    elif bad=='cpu':row['encoding5']['device_type']='cpu'
    elif bad=='q':row['actual_Q5']=100.
    elif bad=='pairingspoof':row['prefix_pairing']['state']=[1.]*5
    elif bad=='changedgoal':row['checkpoints'][-1]['goal'][0]=1
    elif bad=='native':row['final_metrics']['native_distance']=999.
    elif bad=='unmatchedreturn':e[1]['kind']='operation_completed'
    elif bad=='cap':p['contract']['neural_caps']['encoder']=1
    else:r['counts']['predict']['completed']-=1
    with pytest.raises(ValueError):run(data)


def test_native_nonmod_bug_preserved_with_separate_wrapped_diagnostic():
    v=m.native_metrics([0.,0.,0.,0.,7.],[0.]*5)
    assert v['negative_native_angle'] and v['native_success'] and not v['wrapped_allxy_success']
    assert v['native_angle_term']<0 and v['object_success'] is None


def test_array_and_scalar_guards_no_implicit_cpu_neural():
    for z in (np.ones(191,np.float32),np.ones(192,np.float64),np.full(192,np.nan,np.float32)):
        with pytest.raises(ValueError):m.observed_q(z,np.zeros(192))
    for v in (True,float('inf'),float('nan')):
        with pytest.raises(ValueError):m.finite(v)
    assert not any(k in sys.modules for k in ('torch','gymnasium','pymunk','stable_worldmodel'))


def chained(events):
    previous=None;out=[]
    for i,e in enumerate(deepcopy(events)):
        e.update(index=i,previous_sha256=previous);e['event_sha256']=m.digest(e);previous=e['event_sha256'];out.append(e)
    return b''.join(m.canonical(e) for e in out),previous


def test_chain_duplicate_json_key_and_nonfinite_rejection():
    value,tail=chained([dict(kind='one'),dict(kind='two')]);assert len(m.journal(value,tail))==2
    with pytest.raises(ValueError):m.journal(value,'0'*64)
    for raw in (b'{"x":1,"x":2}',b'{"x":NaN}'):
        with pytest.raises(ValueError):m.parse(raw)


def test_build_ram_reads_only_explicit_saved_files_and_rejects_changed_external_prepare(monkeypatch):
    r,p,e,g,z=fixture();h='a'*64
    e[0]['prepare']=deepcopy(p);journal,tail=chained(e);r.update(journal_sha256=m.sha_bytes(journal),journal_chain_tail=tail)
    saved={'/RAM/run/report.json':m.canonical(r),'/RAM/prepare.json':m.canonical(p),
        '/RAM/run/execution_binding.json':m.canonical(dict(prepared='/RAM/prepare.json',prepare_sha256=h)),
        '/RAM/run/journal.jsonl':journal}
    saved[str(Path(m.__file__).with_name(m.DRIVER+'.py').resolve())]=b'RAM driver authority'
    for path,a in z.items():
        b=BytesIO();np.save(b,a,allow_pickle=False);saved[path]=b.getvalue()
    for s in (0,1):
        task=dict(schema=m.ENCODER+'.task_export',source_slot=s,task_id=f'task{s}',with_target=False,P1_admission=False,
            task_contract=dict(metric='identity_native192_no_fit'),task=dict(goal=g[s].tolist(),projection=np.eye(192).tolist()))
        saved[f'/RAM/task{s}.json.gz']=gzip.compress(m.canonical(task),mtime=0)
    calls=[]
    def read(path,expected,cap=16*1024**2):
        calls.append(str(path));assert 'sourcepack' not in str(path) and 'weights' not in str(path)
        return saved[str(path)]
    monkeypatch.setattr(m,'read_bytes',read)
    monkeypatch.setattr(Path,'read_bytes',lambda p:saved.get(str(p),b'RAM analysis source'))
    monkeypatch.setattr(Path,'exists',lambda p:str(p) in saved)
    monkeypatch.setattr(Path,'is_symlink',lambda p:False)
    out=m.build('/RAM/run/report.json',h,'/RAM/prepare.json',h)
    assert len(out['rows'])==12 and out['observed_endpoint_count']==24 and len(out['analysis_source_sha256'])==3
    assert len([p for p in calls if p.endswith('native192.npy')])==48  # input + final seal recheck
    with pytest.raises(ValueError,match='execution binding'):m.build('/RAM/run/report.json',h,'/RAM/prepare.json','b'*64)
    r['rows'][0]['encoding5']['latent']['path']=r['rows'][1]['encoding5']['latent']['path']
    saved['/RAM/run/report.json']=m.canonical(r)
    with pytest.raises(ValueError,match='namespace'):m.build('/RAM/run/report.json',h,'/RAM/prepare.json',h)


def test_local_paths_caps_floor_and_sealed_read_hash_in_ram():
    with pytest.raises(ValueError):m.local('/RAM/run','../other.npy')
    with pytest.raises(ValueError):m.local('/RAM/run','/other.npy')
    with patch.object(Path,'is_file',return_value=True),patch.object(Path,'is_symlink',return_value=False),\
         patch.object(Path,'stat',return_value=NS(st_size=3)),patch.object(Path,'read_bytes',return_value=b'abc'):
        assert m.read_bytes('/RAM/file',m.sha_bytes(b'abc'))==b'abc'
        with pytest.raises(ValueError):m.read_bytes('/RAM/file','a'*64)
    with patch.object(Path,'exists',return_value=False),patch.object(m.shutil,'disk_usage',return_value=NS(free=m.FLOOR-1)):
        with pytest.raises(ValueError):m.save('/RAM/not_created.json',{})
