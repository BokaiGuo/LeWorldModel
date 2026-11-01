import hashlib
import pytest
from experiments.spectral_reachability import agent_p3_c3_k17_physical_gate_20261029_v1 as g
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as t


def live_files(path):
    frames = []
    for i in range(34):
        name = f'{i}.png'
        data = f'fixture{i}'.encode()
        (path / name).write_bytes(data)
        frames.append({'receipt': {'png': name, 'file_sha256': hashlib.sha256(data).hexdigest()}})
    return {'warm_frames': frames[:33], 'goal_frame': frames[33]}


def test_disk_journal_guard_and_tamper(tmp_path, monkeypatch):
    live = live_files(tmp_path)
    monkeypatch.setattr(g.bridge, 'check_live_only', lambda *args: {'physical_dispatch_ready': False})
    journal = t.Journal(tmp_path / 'journal', 10000)
    g.record_preflight(journal, {'slot': 3}, {}, {}, live, tmp_path)
    journal.close()
    assert g.consume_preflight(tmp_path / 'journal')['physical_dispatch_authorized'] is False
    (tmp_path / '0.png').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='live PNG bytes'):
        g.capture_check({'slot': 3}, {}, {}, live, tmp_path)


def test_rejection_is_durable(tmp_path):
    live = live_files(tmp_path)
    live['goal_frame']['receipt']['png'] = '../bad.png'
    journal = t.Journal(tmp_path / 'journal', 10000)
    with pytest.raises(ValueError, match='unique relative PNG'):
        g.record_preflight(journal, {'slot': 3}, {}, {}, live, tmp_path)
    journal.close()
    assert t.read_journal(tmp_path / 'journal')[0]['kind'] == 'guard_rejected'


def test_no_fake_authorization(tmp_path):
    j = t.Journal(tmp_path / 'journal', 10000)
    j.emit('guard_checked', {'slot': 3, 'capacity_proved': True, 'dispatch_authorized': True,
                             'result': {'physical_dispatch_ready': True}})
    j.close()
    with pytest.raises(ValueError, match='no dispatch'):
        g.consume_preflight(tmp_path / 'journal')


def test_snapshot_rejects_missing_and_duplicate_slot_before_reading_receipts():
    with pytest.raises(ValueError, match='unknown K17 slot'):
        g._authorities_from_snapshot(3, [{'slot': 4}])
    with pytest.raises(ValueError, match='unknown K17 slot'):
        g._authorities_from_snapshot(3, [{'slot': 3}, {'slot': 3}])
