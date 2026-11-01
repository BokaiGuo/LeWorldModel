from contextlib import nullcontext
import time
import pytest
import test_agent_experiment1_common_controller_20261028_v1 as fixture
from experiments.spectral_reachability import agent_e1_same_target_durable_staging_20261029_v1 as m


def test_full_four_slot_store_and_saved_consumer(tmp_path):
    tasks=[fixture.task(0),fixture.task(1)]
    targets=[[0.]*192 for _ in range(2)]
    prefixes=[[[0.,0.] for _ in range(5)] for _ in range(2)]
    store=m.old.Store(tmp_path/'full')
    report=m.execute(store,tasks,targets,prefixes,lambda _:fixture.Backend(),
                     lambda i,meter:nullcontext(),lambda:None,store.check)
    assert report['status']=='all_four_sealed'
    report=m.finish(store,report)['report']
    payload, events=m.old.read_store(store.path,report)
    assert m.verify_payload(tasks,targets,prefixes,report,payload,events)['verified']
    events[-1]['value'][0]='0'*64
    with pytest.raises(ValueError): m.verify_payload(tasks,targets,prefixes,report,payload,events)


def test_durable_failed_attempt_and_partial_denominator(tmp_path):
    store=m.old.Store(tmp_path/'partial')
    report=m.execute(store,[fixture.task(0),fixture.task(1)],[[0.]*192]*2,[[[0.,0.]]*5]*2,
                     lambda _:fixture.Backend(fail=1),lambda i,meter:nullcontext(),lambda:None,store.check)
    assert report['status']=='partial_or_failed'
    assert report['rows'][0]['controller_resources']['predict_API']==dict(attempted=1,completed=0)
    assert len(report['rows'])==4 and m.inspect_partial(report)['status']=='partial_not_consumable'
    saved=m.finish(store,report)
    assert m.old.parse_canonical((store.path/'report.json').read_bytes()) == saved['report']
    with pytest.raises(ValueError): m.verify_payload([],[],[],report,{},[])
