from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest
import test_agent_p1_two_packet_uphill_generator_20261029_v1 as fixture

from experiments.spectral_reachability import agent_p1_joint_vreach_authority_prepare_20261029_v1 as m

inputs=fixture.inputs
complete=fixture.complete
requests=fixture.requests


@pytest.fixture
def prepared_fixture(requests,tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();(tmp_path/'results').mkdir()
    metadata=dict(schema=m.authority.NAME+'.material',source_report_sha256='a'*64,
        source_prepare_sha256='b'*64,whole_producer_strict_reconstructed=True,
        rows=[dict(request=deepcopy(q)) for q in requests],bindings={},policy='fixture',
        expected_model_hash='c'*64,P1_admission=False,physical_release=False)
    monkeypatch.setattr(m.authority,'material',lambda p,h:deepcopy(metadata))
    monkeypatch.setattr(m.F,'installed_class_sources',lambda:{})
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m,'own_files',lambda:[])
    # Source/test/doc hashes remain checked against actual source files; only
    # the producer authority is substituted by a fixture in this test.
    monkeypatch.setattr(m,'PINS',())
    monkeypatch.setattr(m.component,'check_sources',lambda:None)
    return source,metadata


def test_build_and_full_verify_rebuild(prepared_fixture,tmp_path):
    source,_=prepared_fixture
    p=m.build(str(source),'a'*64)
    assert p['production_ready'] is False and p['contract']['candidates']==160
    assert [r['request']['slot'] for r in p['authority']['rows']]==list(range(4))
    path=tmp_path/'results'/(m.NAME+'_fixture.json')
    written=m.write(path,p)
    assert m.verify(path,written['sha256'])==p
    with pytest.raises(ValueError):m.write(path,p)


@pytest.mark.parametrize('field',('strict','slot','bits','report','admission'))
def test_wrong_authority_rejected(prepared_fixture,field):
    source,authority=prepared_fixture
    if field=='strict':authority['whole_producer_strict_reconstructed']=False
    elif field=='slot':authority['rows'][1]['request']['slot']=0
    elif field=='bits':authority['rows'][0]['request']['winner_raw_bits']='00'
    elif field=='report':authority['source_report_sha256']='d'*64
    else:authority['P1_admission']=True
    with pytest.raises(ValueError):m.build(str(source),'a'*64)


def test_prepare_tamper_and_collision(prepared_fixture,tmp_path):
    source,_=prepared_fixture;p=m.build(str(source),'a'*64)
    path=tmp_path/'results'/(m.NAME+'_tamper.json');written=m.write(path,p)
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):m.verify(path,written['sha256'])
    x={'/tmp/a':'a'*64}
    with pytest.raises(ValueError):m.add(x,'/tmp/a','b'*64)


def test_actual_pins_and_cli_help():
    assert all(m.sha(module.__file__)==h for module,h in m.PINS)
    for command in ('prepare','verify'):
        p=subprocess.run([sys.executable,'-m',m.__name__,command,'--help'],capture_output=True,text=True,check=True)
        assert '--' in p.stdout
