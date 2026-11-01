from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import agent_e3_launcher_loader_bridge_20261029_v1 as bridge
from experiments.spectral_reachability import agent_e3_transport_publication_20261029_v1 as pub
from experiments.spectral_reachability import agent_e3_gd_host_cuda_20261029_v1 as host


def test_real_disk_loader_lifecycle_and_mutation(tmp_path):
    store = pub.Store(tmp_path / 'output')
    adapter = bridge.LoaderBridge(store)
    policy = str(tmp_path / 'policy')
    files = [dict(path=str(Path(policy) / n), sha256=str(i) * 64)
             for i, n in enumerate(('config.json', 'weights.pt'), 1)]
    for f in files:
        adapter.event(dict(kind='load_intent', path=f['path'], expected_sha256=f['sha256']))
        adapter.event(dict(kind='consumed_buffer_frozen', receipt=f))
    receipt = dict(files=files)
    adapter.put('loader.json', receipt)
    with pytest.raises(ValueError):
        adapter.put('loader.json', receipt)
    seal = store.publish('completed')
    assert pub.consume(store.output, seal['report_sha256'], seal['completion_sha256'])['complete']
    events = [pub.base.decode(line) for line in (store.output / 'events.jsonl').read_bytes().splitlines()]
    def validate(value, prepared, sha):
        assert value == receipt and sha == 'p' and prepared['policy'] == policy
    loader = SimpleNamespace(validate_loader=validate)
    assert bridge.verify_events(events, {'policy': policy}, 'p', loader) == receipt
    events[2]['detail']['event']['path'] = '/wrong/weights.pt'
    with pytest.raises(ValueError):
        bridge.verify_events(events, {'policy': policy}, 'p', loader)


def test_actual_original_loader_validator_receipt(tmp_path):
    loader = host.primitives.native.loader
    policy = str(tmp_path / 'policy')
    files = []
    for i, name in enumerate(('config.json', 'weights.pt'), 1):
        h = str(i) * 64
        files.append(dict(path=str(Path(policy) / name), sha256=h, bytes=123,
                          hash_buffer_sha256=h, deserialize_buffer_sha256=h, same_buffer=True))
    prepared = dict(policy=policy, expected_model_hash='3' * 64,
                    bindings_sha256={f['path']: f['sha256'] for f in files})
    receipt = dict(schema=loader.NAME+'.loader_receipt', prepare_sha256='p',
                   adapter_source_sha256=loader.sha(loader.__file__),
                   legacy_loader_source_sha256=loader.native.SOURCES['experiments/checkpoint_compat.py'],
                   files=files, device='cuda:0', remapped_keys=[], missing=[], unexpected=[],
                   tensor_hash_before=prepared['expected_model_hash'], frozen_eval=True)
    store = pub.Store(tmp_path / 'out'); adapter = bridge.LoaderBridge(store)
    for f in files:
        adapter.event(dict(kind='load_intent',path=f['path'],expected_sha256=f['sha256']))
        adapter.event(dict(kind='consumed_buffer_frozen',receipt=f))
    adapter.put('loader.json', receipt)
    seal = store.publish('completed')
    assert pub.consume(store.output, seal['report_sha256'], seal['completion_sha256'])['complete']
    events = [pub.base.decode(line) for line in (store.output/'events.jsonl').read_bytes().splitlines()]
    assert bridge.verify_events(events, prepared, 'p', loader) == receipt
    events[-1]['detail']['receipt']['files'][0]['same_buffer'] = False
    with pytest.raises(ValueError):
        bridge.verify_events(events, prepared, 'p', loader)
