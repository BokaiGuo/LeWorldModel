"""Non-neural fixtures only; importing the audit must not import torch."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.spectral_reachability import p5_local_asset_binding_20261027_v1 as p5


def fixtures():
    pre = dict(schema='tworoom_d2p_p1_reconfiguration_preflight.v2', status='pass', gates={'integrity': True},
               protocol=dict(history_len=3, action_block=5, prefix_horizon=1),
               rows=[dict(seed=s) for s in p5.SEEDS], training_packet_support=dict(dataset_sha256='a'*64))
    cal = dict(schema='tworoom_d2p_p1_physical_calibration.v2', status='fail', upstream_p1=str(p5.PREFLIGHT.parent),
               protocol=dict(cohort='calibration', outcome_roots_opened=False, root_seeds=list(p5.SEEDS)),
               rows=[dict(seed=s) for s in p5.SEEDS])
    stats = dict(schema='tworoom_d2p_training_action_stats.v1', status='training_semantics_validated',
                 source=dict(dataset=str(p5.DATASET), dataset_sha256='a'*64),
                 reports=[dict(all_packet_slots_share_physical_statistics=True,
                               calibration_action_mean=[0., 0.]*5, calibration_action_std=[1., 1.]*5)])
    return pre, cal, stats


def test_failed_physical_calibration_is_not_promoted():
    pre, cal, stats = fixtures()
    assert cal['status'] == 'fail'
    assert p5.validate_calibration(pre, cal, stats)


@pytest.mark.parametrize('change', ['new_seed', 'duplicate', 'reorder', 'bool_seed', 'outcome', 'integer_false',
                                    'integer_true', 'foreign_upstream', 'bad_std', 'slot_stats', 'dataset', 'source_sha'])
def test_fail_closed(change):
    pre, cal, stats = fixtures()
    if change == 'new_seed': pre['rows'][0]['seed'] += 99
    elif change == 'duplicate': cal['protocol']['root_seeds'][1] = p5.SEEDS[0]
    elif change == 'reorder': pre['rows'].reverse()
    elif change == 'bool_seed': cal['rows'][0]['seed'] = True
    elif change == 'outcome': cal['protocol']['cohort'] = 'outcome'
    elif change == 'integer_false': cal['protocol']['outcome_roots_opened'] = 0
    elif change == 'integer_true': pre['gates']['integrity'] = 1
    elif change == 'foreign_upstream': cal['upstream_p1'] = '/tmp/foreign'
    elif change == 'bad_std': stats['reports'][0]['calibration_action_std'][0] = -1
    elif change == 'slot_stats': stats['reports'][0]['calibration_action_mean'][2] = 3
    elif change == 'dataset': stats['source']['dataset'] = '/tmp/outcome.h5'
    elif change == 'source_sha': stats['source']['dataset_sha256'] = 'b'*64
    with pytest.raises(ValueError): p5.validate_calibration(pre, cal, stats)


@pytest.mark.parametrize('value', ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'])
def test_strict_json(tmp_path, value):
    path = tmp_path/'bad.json'
    path.write_text(value)
    with pytest.raises(ValueError): p5.strict_json(path)


def test_asset_hash_and_missing(tmp_path):
    path = tmp_path/'asset'
    assert p5.asset(path) == dict(path=str(path), exists=False)
    path.write_bytes(b'abc')
    assert p5.asset(path)['sha256'] == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    assert p5.asset(path, hash_bytes=False)['sha256'] is None
    link = tmp_path/'link'
    link.symlink_to(path)
    with pytest.raises(ValueError): p5.asset(link)


def test_new_output_exclusive_and_confined(tmp_path, monkeypatch):
    monkeypatch.setattr(p5, 'ROOT', tmp_path)
    (tmp_path/'results').mkdir()
    out = tmp_path/'results'/p5.PREFIX
    p5.new_output(out)
    with pytest.raises(FileExistsError): p5.new_output(out)
    for bad in (tmp_path/'other', tmp_path/'results/foreign', out/'nested'):
        with pytest.raises(ValueError): p5.new_output(bad)


def test_context_checks_every_field():
    expected = dict(history_rgb_sha256=['a','b','c'], root_rgb_sha256='r', goal_rgb_sha256='g', warmup_actions_sha256='w')
    p5.check_context(expected, expected)
    for key in expected:
        changed = copy.deepcopy(expected)
        changed[key] = 'wrong'
        with pytest.raises(ValueError): p5.check_context(changed, expected)


def test_import_no_neural_runtime():
    code = ('import sys; from experiments.spectral_reachability import p5_local_asset_binding_20261027_v1; '
            'assert "torch" not in sys.modules; assert "gymnasium" not in sys.modules')
    subprocess.run([sys.executable, '-B', '-c', code], cwd=p5.ROOT, check=True)


def test_real_known_calibration_metadata_only():
    # Only the three explicitly allowed existing calibration/normalizer files.
    p5.validate_calibration(*map(p5.strict_json, (p5.PREFLIGHT, p5.CALIBRATION, p5.STATS)))
    assert len(p5.SEEDS) == 8
    assert not any('outcome' in str(p).lower() for p in p5.input_paths())


def test_manifest_cannot_supply_paths():
    manifest = dict(schema=p5.PREFIX+'.v1', P1_admission=False, P5_admission=False,
                    calibration_roots=list(p5.SEEDS), tworoom_replay_assets_ready=True,
                    files=[dict(path='/tmp/arbitrary')])
    with pytest.raises(ValueError, match='allowlist'): p5.verify_manifest(manifest)
    manifest['P5_admission'] = True
    with pytest.raises(ValueError, match='binding manifest'): p5.verify_manifest(manifest)


def test_config_contract_not_runtime_evidence():
    config = dict(_target_='stable_worldmodel.wm.lewm.LeWM', predictor=dict(num_frames=3, output_dim=192),
                  action_encoder=dict(input_dim=10))
    assert p5.config_contract(config)['static_contract_compatible']
    assert not p5.config_contract(config)['runtime_load_verified']
    config['action_encoder']['input_dim'] = 2
    assert not p5.config_contract(config)['static_contract_compatible']


def test_exclusive_json_write(tmp_path):
    path = tmp_path/'report.json'
    p5.write_new(path, {'P5_admission': False})
    assert json.loads(path.read_text())['P5_admission'] is False
    with pytest.raises(FileExistsError): p5.write_new(path, {})


def test_training_action_columns_only(tmp_path, monkeypatch):
    import hashlib
    import h5py
    import numpy as np
    dataset = tmp_path/'training.h5'
    actions = np.arange(24, dtype=np.float32).reshape(12, 2)
    with h5py.File(dataset, 'w') as handle:
        handle['action'] = actions
        handle['ep_len'] = [6, 6]
        handle['ep_offset'] = [0, 6]
        # No image, state, goal, success, or outcome columns exist.
    monkeypatch.setattr(p5, 'DATASET', dataset)
    expected = np.stack([actions[0:5], actions[1:6], actions[6:11], actions[7:12]])
    metadata = dict(packet_count=4, pool_sha256=hashlib.sha256(expected.tobytes()).hexdigest())
    assert np.array_equal(p5.packet_pool_from_columns(metadata), expected)
    metadata['pool_sha256'] = '0'*64
    with pytest.raises(ValueError): p5.packet_pool_from_columns(metadata)


def test_json_size_bound(tmp_path):
    path = tmp_path/'large.json'
    with path.open('wb') as stream: stream.truncate(8*1024*1024+1)
    with pytest.raises(ValueError, match='8 MiB'): p5.strict_json(path)
