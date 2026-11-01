import copy
import pytest

from experiments.spectral_reachability import p2_h15_source_lifecycle_20261029_v1 as m


def test_real_mapping_and_checkpoint():
    x = m.build()
    assert [r['root_slot'] for r in x['roots']] == [0, 1]
    assert x['roots'][1]['task_file'].endswith('task_02.json.gz')
    assert x['checkpoint_file_sha256'] != x['expected_state_dict_sha256']
    assert not x['execution_authorized'] and not x['runtime_precision_verified']


def test_lifecycle_fail_closed_and_tamper(tmp_path):
    manifest = m.build()
    p = tmp_path / 'p2_h15_source_lifecycle_20261029_v1_test.json'
    digest = m.write_new(p, manifest)
    assert m.consume(p, digest) == manifest
    with pytest.raises(ValueError):
        m.consume(p, '0' * 64)
    altered = copy.deepcopy(manifest)
    altered['roots'][0]['task_id'] = '0' * 64
    assert altered != m.build()
    with pytest.raises(ValueError):
        m.write_new(p, altered)


def test_lifecycle_transitions():
    state = m.lifecycle([])
    assert state == ['not_started', 'not_started']
    assert m.lifecycle([('start', 0), ('fail', 0)]) == ['failed', 'not_started']
    assert m.lifecycle([('start', 0), ('seal', 0), ('start', 1), ('fail', 1)]) == ['sealed', 'failed']
    for events in [[('seal', 0)], [('start', 1)], [('start', 0), ('seal', 0), ('start', 0)]]:
        with pytest.raises(ValueError):
            m.lifecycle(events)
