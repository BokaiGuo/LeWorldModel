import hashlib
from pathlib import Path
import pytest
from experiments.spectral_reachability import agent_e3_preflight_v2_authority_prepare_20261029_v1 as p
from test_agent_e3_identity_host_20261029_v1 import authority


def sha(data):
    return hashlib.sha256(data).hexdigest()


def test_real_saved_authority_build_disk_verify_and_mutations(tmp_path):
    original, source_sha = authority(tmp_path)
    before = original.read_bytes()
    out = tmp_path / 'new-authority.json'
    result = p.prepare(str(original), source_sha, str(out))
    assert original.read_bytes() == before
    assert result['sha256'] == sha(out.read_bytes())
    assert p.verify(str(original), source_sha, str(out), result['sha256'])['verified']
    old = p.host.launch.host.read(str(original), source_sha)
    new = p.host.launch.host.read(str(out), result['sha256'])
    assert all(new[k] == old[k] for k in ('schema', 'cell', 'control5', 'resolver'))
    for m in (p.preflight, p.preflight.array_transport, p.preflight.capture, p.host.identity.encoder):
        path = str(Path(m.__file__).resolve())
        assert new['additional_sources'][path] == p.host.launch.host.sha(path)
    assert new['additional_sources'][str(Path(p.__file__).resolve())] == p.host.launch.host.sha(p.__file__)
    with pytest.raises((ValueError, FileExistsError)):
        p.prepare(str(original), source_sha, str(out))
    altered = dict(new, cell=1)
    data = p.host.launch.host.raw(altered)
    bad = tmp_path / 'bad.json'; bad.write_bytes(data)
    with pytest.raises(ValueError, match='reconstruction'):
        p.verify(str(original), source_sha, str(bad), sha(data))
    altered = dict(new, additional_sources=dict(new['additional_sources']))
    altered['additional_sources'][str(Path(p.preflight.__file__).resolve())] = '0' * 64
    data = p.host.launch.host.raw(altered); bad.write_bytes(data)
    with pytest.raises(ValueError, match='reconstruction'):
        p.verify(str(original), source_sha, str(bad), sha(data))
    with pytest.raises(ValueError):
        p.verify(str(original), '0' * 64, str(out), result['sha256'])


def test_pin_conflict_rejected(tmp_path):
    original, source_sha = authority(tmp_path)
    value = p.host.launch.host.read(str(original), source_sha)
    value['additional_sources'][str(Path(p.preflight.__file__).resolve())] = '0' * 64
    data = p.host.launch.host.raw(value)
    path = tmp_path / 'conflict.json'; path.write_bytes(data)
    with pytest.raises(ValueError):
        p.build(str(path), sha(data))
