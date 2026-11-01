from contextlib import nullcontext
from types import SimpleNamespace
import pytest
from experiments.spectral_reachability import agent_p3_c3_k17_native_outer_20261029_v1 as m
from experiments.spectral_reachability import agent_p3_c3_k17_native_store_20261029_v1 as native_store
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as journal


def setup(monkeypatch, tmp_path):
    store = native_store.NativeStore(tmp_path / 'run')
    monkeypatch.setattr(m.h3, 'validate_env', lambda body: None)
    monkeypatch.setattr(m.h3, 'native_hooks', lambda *args: nullcontext())
    def factory():
        body = SimpleNamespace(render_action=False, render_size=224, render_mode='rgb_array')
        return SimpleNamespace(unwrapped=body, close=lambda: None)
    records = [({'slot': s, 'root': r, 'mode': mode, 'K': 17}, {}, {}) for s, r, mode in m.ORDER]
    return store, factory, records


def test_six_rows_and_real_tempdisk_ledger(monkeypatch, tmp_path):
    store, factory, records = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(m.episode, 'episode', lambda *args: {'status': 'horizon_completed', 'returned': 5, 'missing_steps': []})
    result = m.execute_material({'roots': [{}, {}], 'source_plan': {}}, records, store, factory)
    assert len(result['rows']) == result['fixed_denominator'] == 6
    assert all(x['status'] == 'horizon_completed' for x in result['rows'])
    assert result['counts']['constructor']['completed'] == result['counts']['close']['completed'] == 6
    assert sum(e['kind'] == 'branch_closed' for e in journal.read_journal(store.path / 'journal.jsonl')) == 6
    assert result['production_authorized'] is False


def test_early_stop_continues_without_padding(monkeypatch, tmp_path):
    store, factory, records = setup(monkeypatch, tmp_path)
    def stop(*args): raise m.h3.native.allmode.NativeStop('warm early')
    monkeypatch.setattr(m.episode, 'episode', stop)
    result = m.execute_material({'roots': [{}, {}], 'source_plan': {}}, records, store, factory)
    assert all(x['status'] == 'native_stopped' and x['result'] is None for x in result['rows'])
    assert result['counts']['public_step']['attempted'] == 0


def test_failure_retains_remaining_slots(monkeypatch, tmp_path):
    store, factory, records = setup(monkeypatch, tmp_path)
    def fail(*args): raise ValueError('fixture')
    monkeypatch.setattr(m.episode, 'episode', fail)
    result = m.execute_material({'roots': [{}, {}], 'source_plan': {}}, records, store, factory)
    assert result['rows'][0]['status'] == 'engineering_failed'
    assert [r['status'] for r in result['rows'][1:]] == ['not_started'] * 5
    assert result['counts']['close']['completed'] == 1


def test_wrong_slot_rejected(monkeypatch, tmp_path):
    store, factory, records = setup(monkeypatch, tmp_path)
    records[0][0]['slot'] = 4
    with pytest.raises(ValueError, match='source row order'):
        m.execute_material({'roots': [{}, {}], 'source_plan': {}}, records, store, factory)


def test_material_builds_single_snapshot_and_checks_every_identity(monkeypatch):
    calls = []
    roots = [dict(slot=i, warm_trace=[i], root_sha256=str(i)) for i in range(2)]
    monkeypatch.setattr(m.h3.native.allmode, 'source_material', lambda: {'roots': roots})
    snapshot = [dict(slot=s, root=r, mode=mode, K=17) for s, r, mode in m.ORDER]
    def build():
        calls.append('build')
        return {'rows': snapshot}
    monkeypatch.setattr(m.gate.bridge, 'build', build)
    def authorities(slot, replay_rows):
        assert replay_rows is snapshot
        item = next(x for x in replay_rows if x['slot'] == slot)
        root = roots[item['root']]
        return item, {}, dict(reset_identity=root['slot'], warm_trace=root['warm_trace'], root_sha256=root['root_sha256'])
    monkeypatch.setattr(m.gate, '_authorities_from_snapshot', authorities)
    _, rows = m.material()
    assert calls == ['build']
    assert len(rows) == 6
    roots[1]['root_sha256'] = 'changed'
    def mismatch(slot, replay_rows):
        item, task, ref = authorities(slot, replay_rows)
        if item['root'] == 1:
            ref['root_sha256'] = 'old'
        return item, task, ref
    monkeypatch.setattr(m.gate, '_authorities_from_snapshot', mismatch)
    with pytest.raises(ValueError, match='native reset/history/root source mismatch'):
        m.material()
