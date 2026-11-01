"""Array/protocol checks only, no model or simulator."""
import numpy as np
import pytest
import json

from experiments.spectral_reachability import p5_jepa_fresh_fidelity_main_20261027_v10 as fresh


def test_env_arguments_not_divided_by100():
    rgb = np.zeros((4, 224, 224, 3), np.uint8)
    prop = np.ones((4, 4), np.float32)
    commands = np.full((15, 2), .2, np.float64)
    stats = dict(ACTION_MEAN=[.1, -.1], ACTION_STD=[.2, .1], PROPRIO_MEAN=[0]*4, PROPRIO_STD=[2]*4)
    p, packets = fresh.normalized_inputs(rgb, prop, commands, stats)
    np.testing.assert_array_equal(p, .5)
    np.testing.assert_allclose(packets.reshape(15, 2), np.tile([.5, 3.], (15, 1)), atol=1e-6)


def test_last_packet_only_and_axis_order():
    packet = np.arange(30, dtype=np.float32).reshape(3, 10)
    bank = fresh.controls(packet)
    for arm in fresh.ARMS:
        np.testing.assert_array_equal(bank[arm][:2], packet[:2])
    np.testing.assert_array_equal(bank['reverse_steps'][-1].reshape(5, 2), packet[-1].reshape(5, 2)[::-1])
    np.testing.assert_array_equal(bank['negative_normalized'][-1], -packet[-1])
    assert np.all(bank['normalized_zero'][-1] == 0)
    np.testing.assert_array_equal(packet, np.arange(30, dtype=np.float32).reshape(3, 10))


def test_zero_or_constant_last_packet_not_informative():
    with pytest.raises(ValueError, match='nontrivial'):
        fresh.controls(np.zeros((3, 10)))
    with pytest.raises(ValueError, match='nontrivial'):
        fresh.controls(np.ones((3, 10)))


def test_malformed_inputs():
    with pytest.raises(ValueError):
        fresh.controls(np.zeros((30,)))
    with pytest.raises(ValueError):
        fresh.numeric([float('nan')], (1,))


def test_contract_is_not_efficacy():
    p = fresh.protocol()
    assert not p['P5_admission'] and not p['target_optimized'] and not p['training']
    assert not p['cpu_neural_fallback'] and not p['prospective_control']
    assert p['counts'] == dict(official_encoder=1, proprio_encoder=1, predictor=4)
    assert p['frames'] == [0, 5, 10, 15]


def saved_factual(tmp_path):
    """Serialization fixture, never a simulator or efficacy sample."""
    directory = tmp_path / fresh.FACTUAL / 'fixture'
    directory.mkdir(parents=True)
    source = directory.parent / 'dummy_source.txt'
    source.write_text('test serialization fixture, not physics')
    schedule = np.arange(30).reshape(3, 5, 2) / 100.
    protocol = dict(seed=2026102701, frames=[0, 5, 10, 15], schedule=schedule.tolist())
    manifest = dict(schema=fresh.FACTUAL, protocol=protocol, seals={str(source): fresh.parent.sha(source)})
    frozen = directory.parent / 'manifest_frozen.json'
    frozen.write_text(json.dumps(manifest))
    payloads = []
    for role in ('a', 'b'):
        for frame in range(16):
            dest = directory / role / f'frame_{frame:02d}'
            dest.mkdir(parents=True)
            for name, array in dict(rgb=np.zeros((224, 224, 3), np.uint8),
                state=np.zeros(7, np.float32), proprio=np.zeros(4, np.float32)).items():
                np.save(dest / (name + '.npy'), array, allow_pickle=False)
            for name in ('full_public_state.json', 'transition.json'):
                (dest / name).write_text('{}')
            files = {p.name: dict(sha256=fresh.parent.sha(p)) for p in dest.iterdir()}
            payloads.append(dict(phase='assets_durable', role=role, record=dict(frame=frame, files=files)))
        for frame in range(1, 16):
            (directory / role / f'command_{frame:02d}.json').write_text(json.dumps(dict(
                frame=frame, raw_env_argument=schedule.reshape(15, 2)[frame-1].tolist())))
    report = dict(schema=fresh.FACTUAL, status='fresh_native_nonzero_factual_recorded', error=None,
        source_hashes_unchanged=True, manifest_file_sha256=fresh.parent.sha(frozen), protocol=protocol,
        trajectory_frames={'a': 16, 'b': 16}, rows=[dict(frame=i, all_exact=True, exact={'fixture': True}) for i in range(16)],
        native_call_receipts=dict(python_call_entries=dict(env_construct=2, seed=4, reset=2,
            control_step=30, space_step=302, native_render=32, native_get_obs=32, close=2)))
    payloads.append(dict(phase='terminal', summary_sha256=fresh.digest(report)))
    entries, previous = [], None
    for i, event in enumerate(payloads):
        entry = dict(sequence=i, previous_sha256=previous, event=event)
        entry['entry_sha256'] = fresh.digest(entry)
        previous = entry['entry_sha256']
        entries.append(json.dumps(entry))
    journal = directory / 'journal.jsonl'
    journal.write_text('\n'.join(entries) + '\n')
    report['journal_sha256'] = fresh.parent.sha(journal)
    (directory / 'report.json').write_text(json.dumps(report))
    return directory


def test_saved_factual_identity_and_units(tmp_path):
    directory = saved_factual(tmp_path)
    rgb, prop, commands, bindings = fresh.factual_inputs(directory)
    assert rgb.shape == (4, 224, 224, 3) and prop.shape == (4, 4)
    np.testing.assert_array_equal(commands, np.arange(30).reshape(15, 2) / 100.)
    assert len(bindings) > 190


@pytest.mark.parametrize('tamper', ['asset', 'command', 'journal'])
def test_saved_factual_tamper_rejected(tmp_path, tamper):
    directory = saved_factual(tmp_path)
    if tamper == 'asset':
        np.save(directory / 'a/frame_02/state.npy', np.ones(7, np.float32), allow_pickle=False)
    elif tamper == 'command':
        path = directory / 'a/command_01.json'
        value = json.loads(path.read_text())
        value['raw_env_argument'] = [99, 99]
        path.write_text(json.dumps(value))
    else:
        with (directory / 'journal.jsonl').open('a') as stream:
            stream.write('{}\n')
    with pytest.raises(ValueError):
        fresh.factual_inputs(directory)
