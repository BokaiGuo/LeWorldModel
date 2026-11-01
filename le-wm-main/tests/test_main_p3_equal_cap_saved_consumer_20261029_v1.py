"""Read-only compatibility ABI and corruption tests; no neural code."""
from copy import deepcopy
import json
import pytest
from experiments.spectral_reachability import main_p3_equal_cap_saved_consumer_20261029_v1 as m


def record(index, previous, payload):
    value = dict(index=index, previous=previous, payload=payload)
    return dict(value, sha256=m.digest(value))


def saved(tmp_path, payload=None):
    a = record(0, '0'*64, dict(kind='first', text='x'*128) if payload is None else payload)
    b = record(1, a['sha256'], dict(kind='last'))
    (tmp_path/'events.jsonl').write_bytes(m.producer.canonical(a)+b'\n'+m.producer.canonical(b)+b'\n')
    return [a, b], dict(journal_tail=b['sha256'])


def test_full_chain_and_bounded_abi(tmp_path):
    rows, report = saved(tmp_path)
    assert m.load_events(tmp_path, report, total_cap=4096, record_cap=2048) == [r['payload'] for r in rows]
    with pytest.raises(ValueError, match='bounded complete'):
        m.load_events(tmp_path, report, total_cap=4096, record_cap=128)


@pytest.mark.parametrize('fault', ['tail', 'index', 'previous', 'payload', 'truncated', 'duplicate', 'nonfinite'])
def test_corruption_never_accepted(tmp_path, fault):
    rows, report = saved(tmp_path)
    if fault == 'tail': report['journal_tail'] = 'f'*64
    elif fault in ('index', 'previous', 'payload'): rows[1][fault] = {'bad': 1} if fault == 'payload' else 99
    text = b'\n'.join(m.producer.canonical(r) for r in rows)+b'\n'
    if fault == 'truncated': text = text[:-1]
    elif fault == 'duplicate': text = text.replace(b'"index":0', b'"index":0,"index":0')
    elif fault == 'nonfinite': text = text.replace(b'"kind":"first"', b'"kind":NaN')
    (tmp_path/'events.jsonl').write_bytes(text)
    with pytest.raises((ValueError, TypeError)):
        m.load_events(tmp_path, report, total_cap=4096, record_cap=2048)


def test_total_and_per_record_caps_are_independent(tmp_path):
    saved(tmp_path)
    with pytest.raises(ValueError):
        m.load_events(tmp_path, dict(journal_tail='0'*64), total_cap=64, record_cap=32)
    with pytest.raises(ValueError):
        m.load_events(tmp_path, {}, total_cap=64, record_cap=65)


def test_actual_producer_and_original_budget_contract():
    assert m.sha(m.producer.__file__) == m.PRODUCER_SHA
    assert m.producer.profile()['final_reserve_bytes'] == 256*1024**2
    assert m.producer.profile()['output_bytes'] == 1024**3
    assert m.producer.FLAGS['P3_admission'] is False


def test_real_over_64MiB_record_without_any_math_bypass(tmp_path):
    rows, report = saved(tmp_path, dict(kind='fixture', text='x'*(64*1024**2)))
    assert (tmp_path/'events.jsonl').stat().st_size > 64*1024**2
    assert m.load_events(tmp_path, report, total_cap=128*1024**2, record_cap=96*1024**2) == [r['payload'] for r in rows]
    with pytest.raises(ValueError, match='bounded complete'):
        m.load_events(tmp_path, report, total_cap=128*1024**2, record_cap=64*1024**2)
