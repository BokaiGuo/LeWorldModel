from copy import deepcopy
from unittest.mock import patch
import numpy as np
import pytest
from experiments.spectral_reachability import agent_e1_forced_candidate_physical_20261029_v1 as m
from test_p1_constructive_moving_history_physical_20261027_v1 import ArrayEnv,fixture,REQUEST


def fake_inputs():
    _,data=fixture();plans=[]
    for root,arm in m.SLOTS:
        plan=dict(root_slot=root,root_sha256=data['references'][root]['root_sha256'],arm=arm,
                  commands=np.full((10,2),.25,np.float32).tolist(),target=np.full(192,.5,np.float32).tolist(),
                  predicted_contexts=[{},{}],model_terminal_cost=2.,prefix_target_cost=1.,tracking_cost=1.5)
        plan['selection_digest']=m.old.source.digest(plan);plans.append(plan)
    return dict(plans=plans,contract=m.contract(),bindings={},source_prepare_id='b'*64),data


def test_real_saved_abi_and_all_forced_plans():
    report=m.producer.parse_canonical(m.producer.read_bound(m.OUTPUT/'report.json',m.REPORT_SHA,m.producer.RESERVE))
    payload,_=m.producer.read_store(m.OUTPUT,report)
    receipt=m.old.read(m.RECEIPT,m.old.CAP)
    assert m.old.sha(m.RECEIPT)==m.RECEIPT_SHA
    data=m.source_data();fixed=m.replay.forced();authority,previous=m.replay.host.authorities()
    plans=m.checked_plans(receipt,payload,data,fixed,authority,previous)
    assert [(x['root_slot'],x['arm']) for x in plans]==list(m.SLOTS)
    for i,p in enumerate(plans):
        assert p['commands']==fixed['slots'][i]['raw10']
        assert p['predicted_contexts']==payload[f'slot_{i}.replay.z']['whole']
    changed=deepcopy(payload);changed['slot_0.replay.z']['raw'][0][0]+=.1
    with pytest.raises(ValueError):m.checked_plans(receipt,changed,data,fixed,authority,previous)


@pytest.mark.parametrize('fault',[None,'throw','guard','terminal'])
def test_six_actual_store_saved_validation(tmp_path,fault):
    p,data=fake_inputs();out=tmp_path/'physical';store=m.Store(str(out),600)
    binding=dict(prepared='/RAM/prepare',prepare_sha256='a'*64)
    store.put('execution_binding.json',binding)
    with patch.object(m.old.kernel.moving,'validate_env',lambda _:None),\
         patch.object(m.old.source,'sample_state',lambda *a:REQUEST.copy()):
        result=m.execute(p,data,store,lambda:ArrayEnv(fault=fault),authorization=m.AUTH,execution_binding=binding)
    report=m.old.read(out/'report.json',m.old.CAP)
    events=m.verify_journal(out/'journal.jsonl',report['journal_sha256'],report['journal_chain_tail'])
    checkpoints=m.validate_saved(p,data,report,events,out)
    assert len(result['rows'])==6
    if fault is None:
        assert result['status']=='six_slot_native_recorded' and len(checkpoints)==18
        assert result['counts']['public_step']['completed']==120 and result['counts']['physics_step']['completed']==1224
        changed=deepcopy(events);next(e for e in changed if e['kind']=='returned_transition' and e['phase']=='plan')['state'][0]+=1
        with pytest.raises(ValueError):m.validate_saved(p,data,report,changed,out)
    else:
        assert all(not row['full_horizon'] for row in report['rows'])
        assert result['status']==('six_slot_native_recorded' if fault=='terminal' else 'partial_or_failed')


def test_reject_wrong_auth_or_missing_slot():
    p,data=fake_inputs();p['plans'].pop()
    with pytest.raises(ValueError):m.execute(p,data,None,ArrayEnv,authorization=m.AUTH)
    p,data=fake_inputs()
    with pytest.raises(ValueError):m.execute(p,data,None,ArrayEnv,authorization='wrong')


def test_run_consume_full_temporary_fixture(tmp_path):
    p,data=fake_inputs();output=tmp_path/'run';prepared=str(tmp_path/'prepare.json')
    with patch.object(m,'verify',return_value=p),patch.object(m,'source_data',return_value=data),\
         patch.object(m.old.kernel.moving,'validate_env',lambda _:None),\
         patch.object(m.old.source,'sample_state',lambda *a:REQUEST.copy()):
        result=m.run(prepared,'a'*64,str(output),ArrayEnv,authorization=m.AUTH)
        receipt=m.consume(str(output),m.old.sha(output/'report.json'))
    assert result['status']=='six_slot_native_recorded'
    assert receipt['fixed_branch_denominator']==6 and len(receipt['checkpoints'])==18
