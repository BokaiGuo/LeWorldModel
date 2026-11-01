"""Saved artifact and RAM checks only; no native/NN execution."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
import numpy as np
from experiments.spectral_reachability import agent_e1_six_slot_physical_20261029_v1 as m
from test_p1_constructive_moving_history_physical_20261027_v1 import ArrayEnv,fixture as moving_fixture,REQUEST
from test_p1_constructive_plan_physical_20261027_v1 import MemoryStore


@pytest.fixture(scope='module')
def actual():
    return m.build()


def test_actual_six_fixed_plans_and_resource_caps(actual):
    assert [(p['root_slot'],p['arm']) for p in actual['plans']]==list(m.producer.core.SLOTS)
    assert actual['status']=='saved_inputs_verified_not_physical_execution'
    caps=actual['contract']['caps']
    assert caps['public_step']==120 and caps['physics_step']==1224 and caps['explicit_render']==132
    assert caps['reset']==12 and caps['set_state']==24 and caps['goal_render']==12
    assert actual['contract']['physical_outcome_selection'] is False


def test_identity_seal_and_no_missing_plan(actual):
    plans=actual['plans']
    assert all(p['selection_digest']==m.old.source.digest({k:v for k,v in p.items() if k!='selection_digest'}) for p in plans)
    assert len(set(p['compiler_selection_sha256'] for p in plans))==6
    changed=deepcopy(plans[1]);changed['commands'][0][0]+=0.01
    assert changed['selection_digest']!=m.old.source.digest({k:v for k,v in changed.items() if k!='selection_digest'})


def fake_inputs():
    _,data=moving_fixture();plans=[]
    for root,arm in m.producer.core.SLOTS:
        plan=dict(root_slot=root,root_sha256=data['references'][root]['root_sha256'],arm=arm,
                  commands=np.full((10,2),.25,np.float32).tolist(),target=np.full(192,.5,np.float32).tolist(),
                  predicted_contexts=[{},{}],model_terminal_cost=2.,prefix_target_cost=1.,tracking_cost=1.5)
        plan['selection_digest']=m.old.source.digest(plan);plans.append(plan)
    return dict(plans=plans,contract=m.contract(),bindings={},source_prepare_id='b'*64),data


@pytest.mark.parametrize('fault',[None,'throw','guard','terminal'])
def test_six_native_branches_ram(fault):
    p,data=fake_inputs();store=MemoryStore();envs=[]
    def factory():
        env=ArrayEnv(fault=fault);envs.append(env);return env
    def digest(path):
        assert Path(path).name=='journal.jsonl';return m.old.source.digest(store.events)
    with patch.object(m.old.kernel.moving,'validate_env',lambda _:None),\
         patch.object(m.old.source,'sample_state',lambda *a:REQUEST.copy()),patch.object(m.old,'sha',digest):
        result=m.execute(p,data,store,factory,authorization=m.AUTH,
                         execution_binding=dict(prepared='/RAM/prepare',prepare_sha256='a'*64))
    assert len(result['rows'])==result['fixed_branch_denominator']==6
    assert len(envs)==6 and all(e.closed for e in envs)
    operations=m.validate_operation_ledger(store.events,p,result)
    assert len(operations)==sum(x['attempted'] for x in result['counts'].values())
    damaged=deepcopy(store.events)
    next(e for e in damaged if e['kind']=='operation_dispatch_intent')['receipt']['resource']='unknown'
    with pytest.raises(ValueError):m.validate_operation_ledger(damaged,p,result)
    if fault is None:
        assert result['status']=='six_slot_native_recorded',result['errors']
        assert result['counts']['public_step']==dict(attempted=120,completed=120)
        assert result['counts']['physics_step']==dict(attempted=1224,completed=1224)
        assert sum(row['PNG'] for row in result['rows'])==132
    else:
        assert all(not row['full_horizon'] for row in result['rows'])
        assert result['status']==('six_slot_native_recorded' if fault=='terminal' else 'partial_or_failed')


def test_no_wrong_auth_or_subset():
    p,data=fake_inputs();p['plans'].pop()
    with pytest.raises(ValueError):m.execute(p,data,MemoryStore(),ArrayEnv,authorization=m.AUTH)
    p,data=fake_inputs()
    with pytest.raises(ValueError):m.execute(p,data,MemoryStore(),ArrayEnv,authorization='wrong')


def test_six_real_tempdisk_store_and_saved_validation(tmp_path):
    p,data=fake_inputs();out=tmp_path/'physical';store=m.Store(str(out),600)
    binding=dict(prepared='/RAM/prepare',prepare_sha256='a'*64)
    store.put('execution_binding.json',binding)
    with patch.object(m.old.kernel.moving,'validate_env',lambda _:None),\
         patch.object(m.old.source,'sample_state',lambda *a:REQUEST.copy()):
        result=m.execute(p,data,store,ArrayEnv,authorization=m.AUTH,execution_binding=binding)
    assert result['status']=='six_slot_native_recorded',result['errors']
    report=m.old.read(out/'report.json',m.old.CAP)
    events=m.verify_journal(out/'journal.jsonl',report['journal_sha256'],report['journal_chain_tail'])
    checkpoints=m.validate_saved(p,data,report,events,out)
    assert len(checkpoints)==18 and len(list(out.glob('*.png')))==132
    bad=deepcopy(report);bad['counts']['physics_step']['completed']-=1
    with pytest.raises(ValueError):m.validate_saved(p,data,bad,events,out)
    bad=deepcopy(events);next(e for e in bad if e['kind']=='returned_transition' and e['phase']=='plan')['returned_goal_rgb_sha256']='0'*64
    with pytest.raises(ValueError,match='plan action and goal'):m.validate_saved(p,data,report,bad,out)
    bad=deepcopy(events);next(e for e in bad if e['kind']=='returned_transition' and e['phase']=='plan')['state'][0]+=1
    with pytest.raises(ValueError,match='every frame matches returned state'):m.validate_saved(p,data,report,bad,out)
    bad=deepcopy(report);bad['rows'][0]['checkpoints'][0]['state'][0]+=1
    with pytest.raises(ValueError):m.validate_saved(p,data,bad,events,out)
    path=next(out.glob('*.png'));path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='PNG file bytes'):m.validate_saved(p,data,report,events,out)


@pytest.mark.parametrize('fault',['throw','guard','terminal'])
def test_partial_and_early_stop_tempdisk(tmp_path,fault):
    p,data=fake_inputs();out=tmp_path/'physical';store=m.Store(str(out),600)
    binding=dict(prepared='/RAM/prepare',prepare_sha256='a'*64)
    store.put('execution_binding.json',binding)
    with patch.object(m.old.kernel.moving,'validate_env',lambda _:None),\
         patch.object(m.old.source,'sample_state',lambda *a:REQUEST.copy()):
        result=m.execute(p,data,store,lambda:ArrayEnv(fault=fault),authorization=m.AUTH,execution_binding=binding)
    report=m.old.read(out/'report.json',m.old.CAP)
    events=m.verify_journal(out/'journal.jsonl',report['journal_sha256'],report['journal_chain_tail'])
    m.validate_saved(p,data,report,events,out)
    assert len(result['rows'])==6 and result['status']==('six_slot_native_recorded' if fault=='terminal' else 'partial_or_failed')


def test_storage_census_real_shape_and_failure_reserve(tmp_path):
    c=m.storage_census()
    assert c['total']==132*160*1024+6*1024**2+256*1024+4*1024**2+32*1024
    assert c['total']<=32*1024**2 and c['slack']==32*1024**2-c['total']
    out=tmp_path/'storage';s=m.Store(str(out),600)
    s.put('execution_binding.json',{'source':'fixture'})
    image=np.random.default_rng(1).integers(0,256,size=(224,224,3),dtype=np.uint8)
    s.frame('actual_shape.png',image)
    assert (out/'actual_shape.png').stat().st_size<=m.FRAME_LIMIT
    original=(out/'actual_shape.png').read_bytes()
    s.emit('fixture')
    before=(out/'journal.jsonl').read_bytes()
    with pytest.raises(ValueError,match='journal allocation exceeded'):
        s.write('journal.jsonl',b'x'*(m.JOURNAL_LIMIT+1),append=True)
    assert (out/'journal.jsonl').read_bytes()==before
    with pytest.raises(ValueError,match='frame allocation exceeded'):
        s.write('oversize.png',b'x'*(m.FRAME_COUNT*m.FRAME_LIMIT+1))
    assert (out/'actual_shape.png').read_bytes()==original and not (out/'oversize.png').exists()
    with pytest.raises(ValueError,match='report allocation exceeded'):
        s.write('report.json',b'x'*(m.REPORT_LIMIT+1),final=True)
    s.emergency('fixture_over_limit')
    marker=m.old.read(out/'emergency.json',m.EMERGENCY_LIMIT)
    assert marker['status']=='partial_unverified' and marker['fixed_branch_denominator']==6
    assert marker['journal_sha256']==m.old.sha(out/'journal.jsonl') and marker['report_present'] is False


def test_all_allocations_exact_disk_ceiling(tmp_path):
    # Transport ceiling test, not PNG or journal semantic evidence.
    out=tmp_path/'ceiling';s=m.Store(str(out),600)
    s.write('execution_binding.json',b'B'*m.BINDING_LIMIT)
    for i in range(m.FRAME_COUNT):s.write(f'{i:03}.png',b'F'*m.FRAME_LIMIT)
    s.write('journal.jsonl',b'J'*m.JOURNAL_LIMIT,append=True)
    s.write('report.json',b'R'*m.REPORT_LIMIT,final=True)
    s.write('emergency.json',b'E'*m.EMERGENCY_LIMIT,final=True)
    assert s.used()==m.storage_census()['total']<=m.old.CAP
    with pytest.raises(ValueError):s.write('extra.png',b'x')


def test_report_failure_durable_marker(tmp_path):
    class ReportFail(m.Store):
        def put(self,name,value,final=False):
            if name=='report.json':raise ValueError('simulated report allocation failure')
            return super().put(name,value,final)
    p,data=fake_inputs();out=tmp_path/'partial';s=ReportFail(str(out),600)
    binding=dict(prepared='/RAM/prepare',prepare_sha256='a'*64)
    s.put('execution_binding.json',binding)
    with patch.object(m.old.kernel.moving,'validate_env',lambda _:None),\
         patch.object(m.old.source,'sample_state',lambda *a:REQUEST.copy()):
        result=m.execute(p,data,s,lambda:ArrayEnv(fault='throw'),authorization=m.AUTH,execution_binding=binding)
    marker=m.old.read(out/'emergency.json',m.EMERGENCY_LIMIT)
    assert result['status']=='partial_or_failed' and len(result['rows'])==6
    assert marker['stage']=='report_publication_failed' and marker['journal_sha256']==m.old.sha(out/'journal.jsonl')
    assert len(list(out.glob('*.png')))>0 and not (out/'report.json').exists()


def test_journal_finalization_failure_is_not_strict_success(tmp_path):
    class FinalJournalFail(m.Store):
        def emit(self,kind,**kwargs):
            if kind=='run_finished':raise ValueError('simulated journal allocation failure')
            return super().emit(kind,**kwargs)
    p,data=fake_inputs();out=tmp_path/'partial';s=FinalJournalFail(str(out),600)
    binding=dict(prepared='/RAM/prepare',prepare_sha256='a'*64)
    s.put('execution_binding.json',binding)
    with patch.object(m.old.kernel.moving,'validate_env',lambda _:None),\
         patch.object(m.old.source,'sample_state',lambda *a:REQUEST.copy()):
        result=m.execute(p,data,s,lambda:ArrayEnv(fault='throw'),authorization=m.AUTH,execution_binding=binding)
    marker=m.old.read(out/'emergency.json',m.EMERGENCY_LIMIT)
    assert result['status']=='partial_or_failed' and marker['stage']=='journal_publication_failed'
    assert marker['report_present'] is True and marker['journal_sha256']==m.old.sha(out/'journal.jsonl')
    assert len(result['rows'])==6 and not ('journal_sha256' in result)


def test_run_and_consume_real_disk_fixture(tmp_path):
    p,data=fake_inputs();out=tmp_path/'run';prepared=str(tmp_path/'fixture_prepare.json')
    with patch.object(m,'verify',return_value=p),patch.object(m,'source_data',return_value=data),\
         patch.object(m.old.kernel.moving,'validate_env',lambda _:None),\
         patch.object(m.old.source,'sample_state',lambda *a:REQUEST.copy()):
        report=m.run(prepared,'a'*64,str(out),ArrayEnv,authorization=m.AUTH)
        receipt=m.consume(str(out),m.old.sha(out/'report.json'))
    assert report['status']=='six_slot_native_recorded' and receipt['fixed_branch_denominator']==6
    assert receipt['status']=='six_slot_native_recorded' and len(receipt['checkpoints'])==18
