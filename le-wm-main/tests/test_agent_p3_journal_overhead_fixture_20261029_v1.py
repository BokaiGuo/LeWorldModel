import json
import pytest
from experiments.spectral_reachability import agent_p3_journal_overhead_fixture_20261029_v1 as m


@pytest.mark.parametrize('preexisting',(False,True))
@pytest.mark.parametrize('raise_at',(None,1))
def test_actual_observe_restore_shared_fake_model(preexisting,raise_at):
    rows=m.observe_fixture(nodes=6,raise_at=raise_at,preexisting=preexisting)
    for row in rows:
        assert set(row['before'].values())==set(row['after'].values())=={0}
        assert set(row['inside'].values())=={1}
        assert row['exact_attribute_restore']
        assert row['calls']==['action_encoder','predictor','pred_proj']


def test_actual_store_journal_ledger_io_growth(tmp_path):
    v=m.io_fixture(tmp_path);rows=v['rows']
    assert v['neural_calls']==v['physics_calls']==0 and v['production_attribution'] is False
    # Three event writes: attempt/body/return; each checks job and size.
    assert [r['checks_and_IO_counts']['scans'] for r in rows]==[6,6,6]
    entries=[r['checks_and_IO_counts']['entries'] for r in rows]
    stats=[r['checks_and_IO_counts']['explicit_stat'] for r in rows]
    assert entries[0]<entries[1]<entries[2] and stats[0]<stats[1]<stats[2]
    for row in rows:
        assert row['observed_call_wall_ns']>=row['measured_body_wall_ns']
        assert row['checks_and_IO_counts']['check_wall_ns']>0
        assert row['receipt']['attempted']==row['receipt']['completed']
        m.p.core.audit_ledger(row['receipt'])
    for i in range(3):
        path=tmp_path/'output'/f'n{i}'/'events.jsonl';payload=path.read_bytes()
        # Original real writer: no indiscriminate empty-line filter.
        assert b'\n\n' not in payload and payload.endswith(b'\n')
        records=[json.loads(line) for line in payload.splitlines()]
        assert [e['payload']['kind'] for e in records]==['resource_attempt','fixture_body','resource_return']
        previous='0'*64
        for index,e in enumerate(records):
            assert e['index']==index and e['previous']==previous
            got=e.pop('sha256');assert got==m.p.digest(e);previous=got


def test_same_node_clock_and_sink_timing_exact_arithmetic():
    clock=[1000];events=[];caps={k:0 for k in m.p.core.AXES};caps['schedule_decisions']=1
    def sink(e):events.append(e);clock[0]+=100 if e['kind']=='resource_attempt' else 10000
    for node in range(3):
        ledger=m.p.core.Ledger(caps,clock=lambda:clock[0],sink=sink);ledger.scope=dict(node=node)
        def body():clock[0]+=7;return 1
        assert ledger.call('schedule',body)==1
        e=ledger.events[0]
        assert e['started_ns']==0 and e['ended_ns']==107
        # Returned event durability is outside this op's elapsed interval.
        assert ledger.receipt()['inclusive_operation_wall_ns']['schedule']==107
        m.p.core.audit_ledger(ledger.receipt())
