"""CPU-only synthetic four-slot consumer faults; no real model/physics evidence.

Semantic attacks are injected *after* authentic transport/inventory validation to
test the independent consumer even if an internally consistent producer emitted
bad events. They do not purport to test cryptographic resealing.
"""
from copy import deepcopy

import pytest

import test_p3_matched_order_feedback_joint_v2_codex_20261029 as joint
from experiments.spectral_reachability import p3_matched_order_feedback_host_v2_codex_20261029 as host
from experiments.spectral_reachability import p3_matched_order_feedback_host_saved_v2_codex_20261029 as saved


def test_joint_four_slots_reject_independent_feedback_faults(tmp_path, monkeypatch):
    output, report_sha, completion_sha, prepared, original, data = joint.make_fixture(tmp_path)
    monkeypatch.setattr(host, 'verify', lambda *args: prepared)
    monkeypatch.setattr(host.source.host, 'verify_prepare', lambda *args: original)
    monkeypatch.setattr(host.source.physical, 'source_material', lambda: data)
    baseline = saved.consume(output, report_sha, completion_sha)
    assert len(baseline['slots']) == 4 and baseline['full_saved_strict']
    assert all(x['physical_steps'] == 20 and x['encoding_boundaries'] == 4 for x in baseline['slots'])
    assert baseline['scientific_pass'] is False

    original_inspect = host.storage.inspect_journal
    journal = original_inspect(output / 'events.jsonl', host.source.host.sha(output / 'events.jsonl'))
    events = journal['events']
    report = host.source.host.design.read_pinned(output / 'report.json', report_sha)
    inventory = saved._artifacts(output, events, report)
    # Inventory was validated above on the unmodified saved dataset.  The
    # following exercise _slot's semantic checks, not journal or file hashes.
    monkeypatch.setattr(saved, '_artifacts', lambda *args, **kwargs: inventory)
    def selected(slot, resource):
        start = next(e['index'] for e in events if e['resource'] == 'matched_slot_attempt' and e['detail']['slot'] == slot)
        stop = next(e['index'] for e in events if e['resource'] == 'matched_slot_return' and e['detail']['slot'] == slot)
        return next(i for i, e in enumerate(events) if start < e['index'] < stop and e['resource'] == resource)

    cases = [
        (0, 'native_authorized_dispatch', lambda e: e['detail'].__setitem__('raw_bits', 'f' * 16), 'public dispatch'),
        (1, 'native_matched_root_accepted', lambda e: e['detail'].__setitem__('root_sha256', '0' * 64), 'root guard'),
        (2, 'native_control_return', lambda e: e['detail'].__setitem__('terminated', True), 'no native dispatch after stop'),
        (3, 'encoding_complete', lambda e: e['detail']['frame'].__setitem__('pixel_sha256', '0' * 64), 'same fresh frame'),
    ]
    for slot, resource, mutate, message in cases:
        altered = list(events)
        index = selected(slot, resource)
        altered[index] = deepcopy(altered[index])
        mutate(altered[index])
        simulated = dict(journal, events=altered)
        monkeypatch.setattr(host.storage, 'inspect_journal', lambda *args, _j=simulated: _j)
        with pytest.raises(ValueError, match=message):
            saved.consume(output, report_sha, completion_sha)

    monkeypatch.setattr(host.storage, 'inspect_journal', original_inspect)
    read_pinned = host.source.host.design.read_pinned
    def read_bad_report(path, expected):
        value = read_pinned(path, expected)
        if path.name == 'report.json':
            value = deepcopy(value)
            value['peak_bytes'] = [prepared['cuda_allocated_bytes'] + 1, 0]
        return value
    monkeypatch.setattr(host.source.host.design, 'read_pinned', read_bad_report)
    with pytest.raises(ValueError, match='GPU peak bound'):
        saved.consume(output, report_sha, completion_sha)
