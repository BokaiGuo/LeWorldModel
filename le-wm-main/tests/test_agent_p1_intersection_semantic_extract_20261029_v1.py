import copy
import json
from pathlib import Path

import pytest
from experiments.spectral_reachability import agent_p1_intersection_semantic_extract_20261029_v1 as m

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
N = RESULTS / 'agent_p1_intersection_goal_receipt_adapter_20261029_v1_MAIN01.json'
S = RESULTS / 'agent_p1_intersection_goal_adapter_20261029_v2_spec_MAIN01.json'
P = RESULTS / 'agent_p1_prospective_goal_semantics_20261029_v1_prepare_MAIN01.json'
NS = 'fae9926a3f0c7f663224db15f27aef8e2f20de067e16d07081867a988b2bd734'
SS = '8fd89e22374c61df38de33362338b3634bb73f1c07b58c1f6a8befa217e6ce3e'
PS = 'f77f4323d5f1c47a6ece122924df4cac060456a9250826f8303072b8e69fdc7c'


@pytest.fixture(scope='module')
def built():
    return m.build(str(N), NS, str(S), SS, str(P), PS)


def test_real_schema_and_denominator(built):
    r, a = built
    assert len(r['rows']) == len(a['semantic_analysis']['slots']) == 6
    assert sum(len(x['points']) for x in r['rows']) == 24
    assert sum(p['state7'] is None for x in r['rows'] for p in x['points']) == 1
    assert all(x['native_checkpoint_provenance'][0]['source'] == 'derived_from_bound_cp0_not_native_return' for x in a['checkpoint_provenance'])
    assert not a['external_trusted_prospective_verified'] and not a['P1_admission']


@pytest.mark.parametrize('i', [1, 3, 5])
def test_wrong_external_hash_rejected(i):
    args = [str(N), NS, str(S), SS, str(P), PS]
    args[i] = '0' * 64
    with pytest.raises(ValueError): m.build(*args)


def test_forged_normalized_rejected_even_with_new_hash(tmp_path):
    n = json.loads(N.read_bytes())
    n['details'][0]['status'] = 'engineering_failed'
    data = m.semantic.encoded(n)
    path = tmp_path / 'forged.json'
    path.write_bytes(data)
    with pytest.raises(ValueError, match='whole normalized'):
        m.build(str(path), m.semantic.sha(data), str(S), SS, str(P), PS)


@pytest.mark.parametrize('index', [0, 1])
def test_disk_whole_verify_and_no_overwrite(tmp_path, built, index):
    path = tmp_path / (m.NAME+'_result.json')
    data = m.semantic.encoded(built[index])
    assert m.output('extract', path, data) == m.semantic.sha(data)
    assert m.output('verify-extract', path, data) == m.semantic.sha(data)
    with pytest.raises(FileExistsError): m.output('extract', path, data)
    path.write_bytes(data + b' ')
    with pytest.raises(ValueError, match='whole output'): m.output('verify-extract', path, data)


def test_output_path_boundary(tmp_path, built):
    data = m.semantic.encoded(built[0])
    with pytest.raises(ValueError): m.output('extract', tmp_path/'other.json', data)
    target = tmp_path/'target'; target.write_bytes(data)
    link = tmp_path/(m.NAME+'_link'); link.symlink_to(target)
    with pytest.raises(ValueError): m.output('verify-extract', link, data)
