from pathlib import Path
import runpy
import pytest
from experiments.spectral_reachability import p2_h15_bridge_lifecycle_20261029_v1 as m


@pytest.fixture(scope='module')
def native():
    return runpy.run_path(str(Path(__file__).with_name('test_agent_p3_parent_native_cuda_20261029_v1.py')))['NumpyNative']


def test_real_bridge():
    bridge = m.build_bridge()
    assert len(bridge['roots']) == 2
    for root in bridge['roots']:
        assert root['physical_root_sha256'] != root['task']['binding']['root_sha256']
        assert root['checkpoint_file_sha256'] != root['task']['binding']['model_sha256']
    assert bridge['host_authorized'] is False


def test_two_root_fixture_saved_consumer(tmp_path, native):
    path = tmp_path/'run'
    receipt = m.run_fixture(path, lambda t,s: native(t,s))
    checked = m.consume(path, receipt['report_sha256'])
    assert checked['root_status'] == ['sealed', 'sealed']
    assert checked['fixture_only'] and not checked['host_authorized']
    with (path/'root0'/'events.jsonl').open('ab') as f: f.write(b'{}\n')
    with pytest.raises(ValueError): m.consume(path, receipt['report_sha256'])


def test_failure_retained_and_next_root_not_started(tmp_path, native):
    path = tmp_path/'run'
    result = m.run_fixture(path, lambda t,s: native(t,s,fail=True))
    checked = m.consume(path, result['report_sha256'])
    assert checked['root_status'] == ['failed','not_started']
    assert checked['fixture_only'] and not (path/'root1').exists()
    with pytest.raises(ValueError): m.consume(path, '0'*64)


def test_host_is_never_enabled():
    with pytest.raises(ValueError): m.run_host(None)


def test_second_root_failure_keeps_first_sealed(tmp_path, native):
    count = [0]
    def factory(t, s):
        count[0] += 1
        return native(t, s, fail=count[0] == 2)
    result = m.run_fixture(tmp_path/'run', factory)
    assert m.consume(tmp_path/'run', result['report_sha256'])['root_status'] == ['sealed', 'failed']


def test_bridge_exclusive_and_external_hash(tmp_path):
    path = tmp_path/'bridge.json'
    digest = m.save_bridge(path)
    assert len(m.verify_bridge(path, digest)['roots']) == 2
    with pytest.raises(ValueError): m.verify_bridge(path, '0'*64)
    with pytest.raises(FileExistsError): m.save_bridge(path)


def test_outer_exception_is_persisted_but_not_native_audited(tmp_path, monkeypatch):
    def broken(*args): raise OSError('slot writer unavailable')
    monkeypatch.setattr(m.single, 'run_fixture', broken)
    result = m.run_fixture(tmp_path/'run', lambda t,s: None)
    report = m.consume(tmp_path/'run', result['report_sha256'])
    assert report['root_status'] == ['failed','not_started']
    assert report['slots'][0]['status'] == 'orchestration_exception'
