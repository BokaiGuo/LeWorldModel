import json
import pytest

from experiments.spectral_reachability import p2_h15_two_root_saved_host_20261029_v1 as m

MANIFEST = m.source.prep.ROOT / 'results/p2_h15_source_lifecycle_20261029_v1_MAIN01.json'
SHA = '16b781186b86c620d21a263bfa559bfc015066eff2a1702bd586b0e43c2c93ec'


def test_real_manifest_host_refusal_saved_and_rebuilt(tmp_path):
    result = m.run_host(tmp_path / 'run', MANIFEST, SHA)
    checked = m.consume(tmp_path / 'run', result['report_sha256'], MANIFEST, SHA)
    assert checked['root_status'] == ['not_started', 'not_started']
    assert checked['fixed_denominator'] == 2
    assert checked['attempted_predict_samples'] == 0
    assert checked['host_certified'] is False and checked['physical_release'] is False
    with pytest.raises(FileExistsError):
        m.run_host(tmp_path / 'run', MANIFEST, SHA)


def test_journal_and_external_identity_fail_closed(tmp_path):
    result = m.run_host(tmp_path / 'run', MANIFEST, SHA)
    with pytest.raises(ValueError):
        m.consume(tmp_path / 'run', '0'*64, MANIFEST, SHA)
    with pytest.raises(ValueError):
        m.consume(tmp_path / 'run', result['report_sha256'], MANIFEST, '0'*64)
    with (tmp_path / 'run' / 'events.jsonl').open('ab') as f:
        f.write(b'{}\n')
    with pytest.raises(ValueError):
        m.consume(tmp_path / 'run', result['report_sha256'], MANIFEST, SHA)


def test_missing_final_is_not_complete(tmp_path):
    m.run_host(tmp_path / 'run', MANIFEST, SHA)
    (tmp_path / 'run' / 'report.json').unlink()
    with pytest.raises((ValueError, OSError)):
        m.consume(tmp_path / 'run', '0'*64, MANIFEST, SHA)
