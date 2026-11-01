from contextlib import nullcontext
import pytest
import test_agent_experiment1_common_controller_20261028_v1 as fixture
from experiments.spectral_reachability import agent_e1_forced_candidate_replay_cuda_20261029_v1 as m


def test_real_contract_identity_saved_only():
    value=m.forced()
    assert [(x['root'],x['method']) for x in value['slots']]==list(m.SLOTS)
    assert value['protocol']['model_replay_axis_total']==24
    for row in value['slots']:
        assert m.phase.beam.commands(row['raw10']).shape == (10, 2)
        assert m.old.sha(m.old.canonical(row['raw10'])) == row['raw_sha256']


def test_complete_six_replays_temp_store_and_reconstruction(tmp_path):
    tasks=[fixture.task(0),fixture.task(1)]
    slots=[]
    for i,(root,method) in enumerate(m.SLOTS):
        raw=[[float(i)/8,0.] for _ in range(10)]
        slots.append(dict(root=root,method=method,raw10=raw,raw_sha256=m.old.sha(m.old.canonical(raw))))
    fixed=dict(slots=slots);store=m.old.Store(tmp_path/'case');store.put('inputs.z',slots)
    store.event(dict(kind='inputs_sealed_before_model',sha256=m.old.sha(m.old.canonical(slots))))
    policy=tmp_path/'policy'
    previous=dict(policy=str(policy),bindings_sha256={str(policy/'config.json'):'1'*64,str(policy/'weights.pt'):'2'*64})
    loaded=dict(config_sha256='1'*64,weights_sha256='2'*64,strict_key_coverage=True)
    store.event(dict(kind='model_loaded',loaded=loaded,model_hash='3'*64,cuda=True,tf32=False,autocast=False,math_requested=True))
    rows=[dict(root=r,method=k,status='not_started',resources=None,predictor_samples=None) for r,k in m.SLOTS]
    result=m.execute(store,fixed,tasks,lambda _:fixture.Backend(),lambda i,meter:nullcontext(),lambda:None,store.check,rows)
    report=dict(schema=m.NAME+'.report',status='all_six_sealed',rows=rows,errors=[],loaded=loaded,
                model_hash_before='3'*64,model_hash_after='3'*64,scientific_admission=False,training=False,
                new_physics=0,new_encoder=0,new_ad=0,storage=m.old.storage_bound(),prepared='/fixture',
                prepare_sha256='4'*64,host=dict(synthetic=True),
                gpu=dict(cap=m.host.GPU_CAP,peak_allocated=0,peak_reserved=0,tf32_matmul=False,
                         tf32_cudnn=False,autocast=False,requested_math=True,observed_math=None))
    terminal=m.host.stage.finish(store,report);report=terminal['report']
    payload,events=m.old.read_store(store.path,report)
    verified=m.verify_saved(fixed,tasks,report,payload,events)
    assert verified['slots']==6 and verified['predict_per_axis']==24 and len(result)==6
    completion=m.old.parse_canonical((store.path/'completion.json').read_bytes())
    timing=m.old.parse_canonical((store.path/'timing.json').read_bytes())
    checked=m.consume_checked(store.path,terminal['report_sha256'],terminal['completion_sha256'],terminal['timing_sha256'],
                              report,completion,timing,dict(expected_model_hash='3'*64),fixed,previous,tasks)
    assert checked['status']=='six_forced_saved_replays_verified'
    events[-1]['sha256']='0'*64
    with pytest.raises(ValueError):m.verify_saved(fixed,tasks,report,payload,events)


def test_failed_attempt_keeps_four_packet_partial_count(tmp_path):
    tasks=[fixture.task(0),fixture.task(1)];raw=[[0.,0.]]*10
    slots=[dict(root=r,method=k,raw10=raw,raw_sha256=m.old.sha(m.old.canonical(raw))) for r,k in m.SLOTS]
    rows=[dict(root=r,method=k,status='not_started',resources=None,predictor_samples=None) for r,k in m.SLOTS]
    store=m.old.Store(tmp_path/'failure')
    with pytest.raises(RuntimeError):
        m.execute(store,dict(slots=slots),tasks,lambda _:fixture.Backend(fail=1),lambda i,meter:nullcontext(),
                  lambda:None,store.check,rows)
    assert rows[0]['resources']['predict_API']==dict(attempted=1,completed=0)
    assert all(x['status']=='not_started' for x in rows[1:])
    store.close()
