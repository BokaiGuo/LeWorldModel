import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest
from experiments.spectral_reachability import agent_p3_future_cp10_source_seal_20261029_v1 as m

old=runpy.run_path(str(Path(__file__).parent/'test_agent_p3_future_cp10_receipt_join_20261029_v1.py'))
js=old['js']


@pytest.fixture
def sealed_input(tmp_path):
    refs=old['bundle'].__wrapped__(tmp_path)
    manifest=js(tmp_path/'source_manifest.json',dict(schema=m.NAME+'.producer_source_manifest',
                  producer_source=refs['source'],files=[refs['source']]))
    pp=Path(refs['prepare']['path']);p=json.loads(pp.read_bytes());p['producer_source_manifest']=manifest;refs['prepare']=js(pp,p)
    rp=Path(refs['report']['path']);r=json.loads(rp.read_bytes());r['prepare_sha256']=refs['prepare']['sha256'];refs['report']=js(rp,r)
    cp=Path(refs['completion']['path']);c=json.loads(cp.read_bytes());c['prepare_sha256']=refs['prepare']['sha256'];c['report_sha256']=refs['report']['sha256'];refs['completion']=js(cp,c)
    sp=Path(refs['receipt']['path']);s=json.loads(sp.read_bytes());s['report_sha256']=refs['report']['sha256'];s['completion_sha256']=refs['completion']['sha256'];s['completion']=c
    for d in (refs['prepare'],refs['report'],refs['completion'],manifest):s['bindings'][d['path']]=d['sha256']
    refs['receipt']=js(sp,s)
    return refs,manifest,tmp_path


def test_fixture_join_reconstructed_without_trust_claim(sealed_input):
    refs,manifest,_=sealed_input;r=m.build(refs,manifest)
    assert len(r['task_projection']['signals'])==7
    for key in ('source_authority_verified','source_closure_completeness_independently_verified',
                'execution_authorized','fresh_roots_certified','trusted_timestamp_proven'):
        assert r[key] is False


def test_manifest_mutation_and_source_tamper(sealed_input):
    refs,manifest,_=sealed_input
    Path(refs['source']['path']).write_bytes(b'mutated')
    with pytest.raises(ValueError):m.build(refs,manifest)


def test_manifest_not_bound_in_prepare(sealed_input):
    refs,manifest,tmp=sealed_input
    duplicate=js(tmp/'other_manifest.json',json.loads(Path(manifest['path']).read_bytes()))
    with pytest.raises(ValueError):m.build(refs,duplicate)


def test_duplicate_source_and_unbound_receipt_rejected(sealed_input):
    refs,manifest,tmp=sealed_input
    x=json.loads(Path(manifest['path']).read_bytes());x['files'].append(x['files'][0])
    duplicate=js(tmp/'duplicate.json',x)
    with pytest.raises(ValueError):m.build(refs,duplicate)
    rp=Path(refs['receipt']['path']);receipt=json.loads(rp.read_bytes())
    del receipt['bindings'][refs['source']['path']];refs['receipt']=js(rp,receipt)
    with pytest.raises((ValueError,KeyError)):m.build(refs,manifest)


def test_tempdisk_seal_full_verify_and_mutation(sealed_input):
    refs,manifest,tmp=sealed_input;out=tmp/(m.NAME+'_test.json')
    args=[sys.executable,'-B','-m','experiments.spectral_reachability.'+m.NAME,'seal']
    for k in ('prepare','report','completion','receipt','source'):
        args+=['--'+k,refs[k]['path'],'--'+k+'-sha',refs[k]['sha256']]
    args+=['--manifest',manifest['path'],'--manifest-sha',manifest['sha256'],'--output',str(out)]
    subprocess.run(args,check=True,capture_output=True);h=hashlib.sha256(out.read_bytes()).hexdigest()
    args[4]='verify'; subprocess.run(args+['--output-sha',h],check=True,capture_output=True)
    out.write_bytes(out.read_bytes()+b' ')
    assert subprocess.run(args+['--output-sha',h],capture_output=True).returncode != 0
