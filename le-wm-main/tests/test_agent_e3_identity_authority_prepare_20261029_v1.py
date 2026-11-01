import hashlib
import json
import subprocess
import sys
from pathlib import Path
import pytest
from experiments.spectral_reachability import agent_e3_identity_authority_prepare_20261029_v1 as mod
from experiments.spectral_reachability import agent_e3_preflight_v2_authority_prepare_20261029_v1 as v2

ROOT = mod.host.ROOT
R = ROOT / 'results'
REFS = {
    'prepare': ('agent_e3_gd_host_cuda_20261029_v1_prepare_MAIN01.json', mod.launch.CONTROL5_PREPARE_SHA),
    'report': ('agent_e3_gd_host_cuda_20261029_v1_cell00_MAIN01/report.json', 'e6d465652eb993e31d19ce11e5e0f494cb18bf54e65d4d9f0cac983a733e7385'),
    'completion': ('agent_e3_gd_host_cuda_20261029_v1_cell00_MAIN01/completion.json', 'ca309a42a5bf7ab8344b8c1672b66829d3b8bc3b4a82fac3650f04ce5d360b1b'),
    'receipt': ('agent_e3_gd_host_cuda_20261029_v1_consume_cell00_MAIN01.json', '9b5b8751e38138b6f127d05de331e54bd355075501cdaf6914fb1fb3e63b945a'),
}
RESOLVER = dict(path=str(R/'agent_e3_frozen_model_task_resolver_20261029_v1_prepare_MAIN01.json'), sha256=mod.launch.RESOLVER_SHA)


def refs():
    result = {k: dict(path=str(R/path), sha256=sha) for k, (path, sha) in REFS.items()}
    if not all(Path(x['path']).is_file() for x in [*result.values(), RESOLVER]):
        pytest.skip('actual saved inputs unavailable')
    return result


def test_actual_disk_lifecycle_v2_join_and_tamper(tmp_path):
    inp = refs(); out = tmp_path/'identity.json'
    result = mod.prepare(inp, RESOLVER, 0, str(out))
    assert mod.verify(inp, RESOLVER, 0, str(out), result['sha256'])['verified']
    downstream = v2.prepare(str(out), result['sha256'], str(tmp_path/'preflight.json'))
    assert v2.verify(str(out), result['sha256'], downstream['output'], downstream['sha256'])['verified']
    with pytest.raises((ValueError, FileExistsError)):
        mod.prepare(inp, RESOLVER, 0, str(out))
    bad = json.loads(out.read_bytes()); bad['cell'] = 1
    data = mod.host.raw(bad); tampered = tmp_path/'bad.json'; tampered.write_bytes(data)
    with pytest.raises(ValueError, match='reconstruction'):
        mod.verify(inp, RESOLVER, 0, str(tampered), hashlib.sha256(data).hexdigest())
    wrong = dict(inp, receipt=dict(inp['receipt'], sha256='0'*64))
    with pytest.raises(ValueError): mod.build(wrong, RESOLVER, 0)
    with pytest.raises(ValueError): mod.build(inp, dict(RESOLVER, sha256='0'*64), 0)
    with pytest.raises(ValueError): mod.build(inp, RESOLVER, True)


def test_real_cli_prepare_verify(tmp_path):
    inp = refs(); out = tmp_path/'cli.json'
    common = []
    for name, ref in [*inp.items(), ('resolver', RESOLVER)]:
        common += ['--'+name, ref['path'], '--'+name+'-sha256', ref['sha256']]
    common += ['--cell', '0', '--output', str(out)]
    cmd = [sys.executable, '-m', mod.__name__]
    prepared = json.loads(subprocess.check_output(cmd+['prepare']+common, cwd=ROOT))
    verified = json.loads(subprocess.check_output(cmd+['verify']+common+['--output-sha256', prepared['sha256']], cwd=ROOT))
    assert verified['verified'] is True and not verified['full_upstream_strict_rebuilt']
