import hashlib
import json
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_p3_capability_prospective_binding_20261029_v1 as b
from experiments.spectral_reachability import agent_p3_future_capability_abi_20261029_v1 as m


def digest(data): return hashlib.sha256(data).hexdigest()


def put(path, data):
    path.write_bytes(data)
    return {'path': str(path), 'sha256': digest(data)}


@pytest.fixture
def handoff(tmp_path):
    refs = [put(tmp_path/name, name.encode()) for name in ('cohort', 'plan', 'executor', 'bridge')]
    c, p, e, bridge = refs
    proposal = b.proposal('future-test', c['path'], c['sha256'], p['path'], p['sha256'], e['path'], e['sha256'])
    pr = put(tmp_path/'proposal', b.encoded(proposal))
    dec = dict(schema=m.NAME+'.declaration', run_id='future-test', cohort=c, static_prepare=p,
               executor=e, bridge_source=bridge, endpoint_order=m.ORDER, signal_count=7,
               endpoint_count=5, executor_input_schema='unreviewed.fixture')
    return tmp_path, pr, dec


def invoke(handoff, dec=None):
    path, pr, original = handoff
    declaration = put(path/'declaration', json.dumps(original if dec is None else dec).encode())
    return m.build(pr['path'], pr['sha256'], declaration['path'], declaration['sha256'])


def test_fixture_remains_blocked(handoff):
    result = invoke(handoff)
    assert result['descriptor_bytes_verified'] is True
    for field in ('source_authority_verified', 'executor_abi_verified', 'external_order_verified',
                  'trusted_timestamp_proven', 'fresh_roots_certified', 'execution_authorized', 'scientific_admission'):
        assert result[field] is False
    assert result['contract']['VJP'] == 1344
    assert result['contract']['stage_C_branches'] == 10


@pytest.mark.parametrize('field,value', [('signal_count', True), ('signal_count', 8), ('endpoint_count', 4),
    ('endpoint_order', [[0, 'C']]), ('run_id', 'different'), ('executor_input_schema', '')])
def test_reject_invalid_declaration(handoff, field, value):
    d = dict(handoff[2]); d[field] = value
    with pytest.raises(ValueError): invoke(handoff, d)


def test_changed_precommitted_bytes_fail(handoff):
    path, _, d = handoff
    (path/'cohort').write_bytes(b'tampered')
    with pytest.raises(ValueError): invoke(handoff, d)


def test_descriptor_swap_and_extra_fields_fail(handoff):
    d = dict(handoff[2]); d['executor'] = d['bridge_source']
    with pytest.raises(ValueError): invoke(handoff, d)
    d = dict(handoff[2]); d['authority_verified'] = True
    with pytest.raises(ValueError): invoke(handoff, d)


def test_duplicate_json_rejected(handoff):
    path, pr, _ = handoff
    d = put(path/'dupe', b'{"schema":1,"schema":2}')
    with pytest.raises(ValueError): m.build(pr['path'], pr['sha256'], d['path'], d['sha256'])


def test_real_disk_build_verify_and_tamper(handoff):
    path, pr, d = handoff
    dec = put(path/'declaration', json.dumps(d).encode())
    out = path/(m.NAME+'_output.json')
    args = [sys.executable, '-B', '-m', 'experiments.spectral_reachability.'+m.NAME,
            'build', '--proposal', pr['path'], '--proposal-sha', pr['sha256'],
            '--declaration', dec['path'], '--declaration-sha', dec['sha256'], '--output', str(out)]
    subprocess.run(args, check=True, capture_output=True)
    sha = digest(out.read_bytes())
    args[4] = 'verify'
    subprocess.run(args+['--output-sha', sha], check=True, capture_output=True)
    out.write_bytes(out.read_bytes()+b' ')
    assert subprocess.run(args+['--output-sha', sha], capture_output=True).returncode != 0


def test_existing_output_not_overwritten(handoff):
    path, pr, d = handoff
    dec = put(path/'declaration', json.dumps(d).encode()); out = path/(m.NAME+'_output.json')
    out.write_bytes(b'preserve')
    args = [sys.executable, '-B', '-m', 'experiments.spectral_reachability.'+m.NAME, 'build',
            '--proposal', pr['path'], '--proposal-sha', pr['sha256'], '--declaration', dec['path'],
            '--declaration-sha', dec['sha256'], '--output', str(out)]
    assert subprocess.run(args, capture_output=True).returncode != 0
    assert out.read_bytes() == b'preserve'
