from contextlib import nullcontext
import pytest
import test_agent_experiment1_common_controller_20261028_v1 as fixture
from experiments.spectral_reachability import agent_e1_same_target_four_slot_execution_20261029_v1 as m


def test_all_four_slots_real_dispatch_saved_reconstruction(tmp_path):
    tasks = [fixture.task(0), fixture.task(1)]
    targets = [[0.0]*192 for _ in range(2)]
    prefixes = [[[0.0,0.0] for _ in range(5)] for _ in range(2)]
    writers = {(i,p):m.old.Journal(tmp_path/f'{i}-{p}') for i in range(4) for p in ('controller','extra','replay')}
    barriers = []; order = []; timeline = []
    def sink_for(i,p):
        def sink(event):
            order.append(('event',i,p)); timeline.append(dict(type='event',slot=i,phase=p,event=event)); writers[i,p].event(event)
        return sink
    def seal(i,p,value):
        raw = m.old.canonical(value); name=f'{i}-{p}.json'
        m.old.put_new(tmp_path,name,raw)
        return m.old.parse_canonical(m.old.read_bound(tmp_path/name,m.old.sha(raw),16*1024*1024))
    def barrier(name,value):
        order.append(('barrier',name)); barriers.append((name,value)); timeline.append(dict(type='barrier',name=name,value=value))
    rows = [dict(root=r,arm=a,status='not_started') for r,a in m.SLOTS]
    result = m.execute(tasks,targets,prefixes,lambda _:fixture.Backend(),lambda i,meter:nullcontext(),
                       sink_for,seal,barrier,lambda:None,lambda:None,rows)
    events = []
    for i in range(4):
        item={}
        for p in ('controller','extra','replay'):
            writers[i,p].close(); item[p]=m.old.read_journal(writers[i,p].path)[0]
        events.append(item)
    global_barriers = [x for x in barriers if x[0] != 'slot_payload_sealed']
    assert m.verify_saved(tasks,targets,prefixes,result,events,global_barriers)['verified']
    assert m.verify_timeline(tasks,targets,prefixes,result,timeline)['verified']
    assert all(row['status']=='sealed' for row in rows)
    b0=order.index(('barrier','all_banks_sealed')); b1=order.index(('barrier','all_choices_sealed'))
    assert all(x[2]=='controller' for x in order[:b0] if x[0]=='event')
    assert all(x[2]=='extra' for x in order[b0+1:b1] if x[0]=='event')
    events[3]['extra'][0]['kind']='tampered'
    with pytest.raises(ValueError): m.verify_saved(tasks,targets,prefixes,result,events,global_barriers)
    with pytest.raises(ValueError): m.verify_timeline(tasks,targets,prefixes,result,timeline[:-1])


def test_failure_preserves_unfinished_four_slot_rows():
    rows=[dict(root=r,arm=a,status='not_started') for r,a in m.SLOTS]
    with pytest.raises(RuntimeError):
        m.execute([fixture.task(0),fixture.task(1)],[[0.]*192]*2,[[[0.,0.]]*5]*2,
                  lambda _:fixture.Backend(fail=1),lambda i,meter:nullcontext(),lambda i,p:lambda e:None,
                  lambda i,p,v:v,lambda n,v:None,lambda:None,lambda:None,rows)
    assert rows[0]['controller_resources']['predict_API']=={'attempted':1,'completed':0}
    assert [r['status'] for r in rows[1:]]==['not_started']*3
