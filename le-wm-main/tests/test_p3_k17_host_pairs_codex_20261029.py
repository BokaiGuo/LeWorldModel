"""CPU-only tests; actual saved bundle is read, never executed."""
import copy
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'experiments/spectral_reachability'))
import p3_k17_host_pairs_codex_20261029 as pairs

BUNDLE = BASE / 'results/p3_k17_host_bundle_20261029_MAIN01'
RECEIPT = BASE / 'results/p3_k17_host_bundle_20261029_consume_MAIN01.json'


def test_real_result_and_resource_vectors():
    result = pairs.analyze(BUNDLE, RECEIPT)
    assert len(result['pairs']) == 6
    assert [p['action_bit_equal_prefix'] for p in result['pairs']] == [0, 0, 0, 0, 0, 5]
    assert all(b['resources'] == result['branches'][0]['resources'] for b in result['branches'])
    assert result['resource_totals']['physics_step'] == 924


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'bits', 'goal', 'nan'])
def test_reject_corrupted_events(monkeypatch, mutation):
    events = pairs.read_journal(BUNDLE / 'native/journal.jsonl')
    events = copy.deepcopy(events)
    plan = next(e for e in events if e['kind'] == 'returned_transition' and e['payload']['phase'] == 'plan')
    if mutation == 'missing': events.remove(plan)
    if mutation == 'duplicate': events.append(copy.deepcopy(plan))
    if mutation == 'bits': plan['payload']['actual_bits'] = '0000000000000000'
    if mutation == 'goal': plan['payload']['goal_state'][0] += 1
    if mutation == 'nan': plan['payload']['reward'] = float('nan')
    monkeypatch.setattr(pairs, 'read_journal', lambda _: events)
    with pytest.raises(ValueError): pairs.analyze(BUNDLE, RECEIPT)


def test_external_anchor_rejected():
    with pytest.raises(ValueError, match='completion anchor'):
        pairs.analyze(BUNDLE, RECEIPT, completion_sha='0' * 64)
    with pytest.raises(ValueError, match='receipt anchor'):
        pairs.analyze(BUNDLE, RECEIPT, receipt_sha='0' * 64)


def test_same_root_goal_mismatch(monkeypatch):
    events = copy.deepcopy(pairs.read_journal(BUNDLE / 'native/journal.jsonl'))
    for e in events:
        p = e['payload']
        if e['kind'] == 'goal_attached' and p['branch'] == 1: p['goal_pose'][0] += 1
    monkeypatch.setattr(pairs, 'read_journal', lambda _: events)
    with pytest.raises(ValueError, match='same-root goal mismatch'): pairs.analyze(BUNDLE, RECEIPT)


def test_manifest_file_tamper_detected(monkeypatch):
    original = pairs.sha
    monkeypatch.setattr(pairs, 'sha', lambda path: '0' * 64 if path.name == 'report.json' else original(path))
    with pytest.raises(ValueError, match='manifest file'): pairs.analyze(BUNDLE, RECEIPT)


def test_broken_journal_chain(tmp_path):
    path = tmp_path / 'journal.jsonl'
    first = (BUNDLE / 'native/journal.jsonl').read_bytes().splitlines()[0]
    path.write_bytes(first.replace(b'run_started', b'run_xxxxxx') + b'\n' + first + b'\n')
    with pytest.raises(ValueError, match='journal chain mismatch'): pairs.read_journal(path)
