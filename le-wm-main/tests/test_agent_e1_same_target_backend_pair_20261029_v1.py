from contextlib import nullcontext
import pytest
import test_agent_experiment1_common_controller_20261028_v1 as fixture
from experiments.spectral_reachability import agent_e1_same_target_backend_pair_20261029_v1 as m


def test_two_full_controllers_extra_and_replay_saved(tmp_path):
    task = fixture.task(0); target = [0.0]*192
    writers = {(i,p): m.old.Journal(tmp_path / f'{i}-{p}') for i in range(2) for p in ('controller','extra','replay')}
    saved = {}
    def seal(i, phase, value):
        path = tmp_path / f'{i}-{phase}.json'
        data = m.old.canonical(value)
        m.old.put_new(tmp_path, path.name, data)
        saved[i,phase] = m.old.parse_canonical(m.old.read_bound(path, m.old.sha(data), 16*1024*1024))
        return saved[i,phase]
    rows = [dict(status='not_started') for _ in range(2)]
    result = m.run_pair(task, 0, target, [[0.,0.]]*5, lambda _:fixture.Backend(),
                        lambda i,meter:nullcontext(), lambda i,p:writers[i,p].event,
                        seal, lambda:None, lambda:None, rows)
    assert result['bank']['exact'] and all(r['status'] == 'sealed' for r in rows)
    events = []
    for i in range(2):
        item = {}
        for phase in ('controller','extra','replay'):
            writers[i,phase].close()
            item[phase] = m.old.read_journal(writers[i,phase].path)[0]
        events.append(item)
    args = (task, 0, target, [[0.,0.]]*5,
            [saved[i,'controller'] for i in range(2)],
            [saved[i,'choice'] for i in range(2)],
            [saved[i,'replay'] for i in range(2)], events)
    assert m.verify_saved(*args)['verified']
    events[1]['extra'][0]['kind'] = 'tampered'
    with pytest.raises(ValueError): m.verify_saved(*args)


def test_failure_retains_attempted_module_counts():
    rows = [dict(status='not_started') for _ in range(2)]
    events = []
    with pytest.raises(RuntimeError, match='RAM fault'):
        m.run_pair(fixture.task(0), 0, [0.0]*192, [[0.,0.]]*5,
                   lambda _: fixture.Backend(fail=1), lambda i,meter:nullcontext(),
                   lambda i,p:events.append, lambda i,p,v:v,
                   lambda:None, lambda:None, rows)
    assert rows[0]['status'] == 'controller_attempted'
    assert rows[0]['controller_resources']['predict_API'] == {'attempted':1, 'completed':0}
    assert rows[0]['controller_resources']['action_encoder'] == {'attempted':1, 'completed':1}
    assert rows[1]['status'] == 'not_started'
