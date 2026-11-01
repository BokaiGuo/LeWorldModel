"""Synthetic metadata only: no candidate outcomes, producer, NN or physics."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_p5_fresh_confirmation_preconditions_20261029_v1 as m


@pytest.fixture
def fixture(monkeypatch):
    a=m.reuse();values={}
    values['authority_source']=m.AUTHORITY.read_bytes();values['protocol']=b'RAM fixed contract'
    values['preregistration']=dict(schema='chat_freshroots_20261027.preregistration.v1',
        protocol_sha256=hashlib.sha256(values['protocol']).hexdigest(),candidate_order=deepcopy(m.SEEDS),
        sampling_denominator=4,attempt_per_slot=1,proposal_per_slot=1,replacement_or_retry=False,new_outcomes_opened=False)
    values['fixed_slot_consumer']=dict(schema='chat_freshroots_20261027.consumer_manifest.v2',sampling_denominator=4,
        candidates=[dict(s,actual_root_identity=None,freshness_certified=False) for s in m.SEEDS],
        coverage_certified=False,freshness_certified=False,source_run_authorized=False,certified_fresh_roots=[],
        unresolved_dependencies=['RAM missing reset ledger'])
    values['exclusions']=dict(schema='moving_fresh_source_exclusions.v1',source_run_preconditions_met=False,
        freshness_certified=False,approval=dict(approved=False),unresolved_sources=[dict(reason='RAM unknown')],
        protocol=dict(path=str(m.PINS['protocol'][0]),sha256=values['preregistration']['protocol_sha256']),coverage_content_sha256='a'*64)
    sources=[];dispatches=[]
    for sid in 'ABCDEFGH':
        sources.append(dict(id=sid,path='/RAM/'+sid+'/report.json',sha256=hashlib.sha256(sid.encode()).hexdigest(),
            references=[dict(parent=p) for p in ('reachable_prepare','moving_prepare')]))
        for i in range(3 if sid=='B' else 1):
            dispatches.append(dict(source_id=sid,call_id=sid+str(i),command='python original.py',
                producer='original.py',log='/RAM/original.jsonl',line=i+1))
    values['reset_catalog']=dict(sources=sources,recovery=dict(dispatches=dispatches))
    pins=deepcopy(m.PINS)
    def reader():
        blobs={}
        for role,value in values.items():
            b=value if type(value) is bytes else m.raw(value)
            path=pins[role][0];pins[role]=(path,hashlib.sha256(b).hexdigest());blobs[str(path)]=b
        monkeypatch.setattr(m,'PINS',deepcopy(pins))
        return a.Reader(provider=blobs.__getitem__)
    return values,reader


def test_bounded_resolution_reuses_authority_and_preserves_fixed_denominators(fixture):
    values,reader=fixture;r=reader();v=m.resolve(r)
    assert len(r.cache)==6
    assert v['status']=='blocked_no_certified_fresh_cohort'
    assert v['certified_fresh_count']==0 and v['certified_fresh_roots']==[]
    assert v['old_design_slots']==m.SEEDS and v['old_design_denominator']==4 and v['P5_confirmation_denominator'] is None
    sources=v['upstream_requests'][:8]
    assert [x['source_id'] for x in sources]==list('ABCDEFGH')
    assert sum(len(x['inherited_refs']) for x in sources)==16
    assert sum(len(x['dispatches']) for x in sources)==10
    assert len(sources[1]['dispatches'])==3
    assert all(x['status']=='blocked' for x in v['upstream_requests'])
    assert not any(v[k] for k in ('source_run_authorized','reserved_outcomes_opened','P1_admission','P5_admission','neural_calls','physics_calls'))
    assert v==m.resolve(reader())


@pytest.mark.parametrize('fault',('seed','order','denominator','retry','outcome','actual_identity','claim_fresh','claim_coverage','approval','no_unknown','protocol','catalog_calls'))
def test_no_implicit_ready_or_relabelled_root(fixture,fault):
    v,reader=fixture
    if fault=='seed':v['preregistration']['candidate_order'][0]['root_seed']+=1
    elif fault=='order':v['preregistration']['candidate_order'].reverse()
    elif fault=='denominator':v['fixed_slot_consumer']['sampling_denominator']=164
    elif fault=='retry':v['preregistration']['replacement_or_retry']=True
    elif fault=='outcome':v['preregistration']['new_outcomes_opened']=True
    elif fault=='actual_identity':v['fixed_slot_consumer']['candidates'][0]['actual_root_identity']={'pretend':'new'}
    elif fault=='claim_fresh':v['fixed_slot_consumer']['certified_fresh_roots']=['known-root']
    elif fault=='claim_coverage':v['fixed_slot_consumer']['coverage_certified']=True
    elif fault=='approval':v['exclusions']['approval']['approved']=True
    elif fault=='no_unknown':v['exclusions']['unresolved_sources']=[]
    elif fault=='protocol':v['exclusions']['protocol']['sha256']='0'*64
    else:v['reset_catalog']['recovery']['dispatches'].pop()
    with pytest.raises(ValueError):m.resolve(reader())


def test_SHA_mismatch_not_treated_as_missing_ready(fixture):
    _,reader=fixture;r=reader();p,h=m.PINS['reset_catalog'];r.provider=lambda _:b'tampered'
    with pytest.raises(ValueError):m.resolve(r)


def test_reserved_outcomes_are_not_followed(fixture):
    values,reader=fixture
    values['fixed_slot_consumer']['reserved_outcome_path']='/FORBIDDEN/reserved.json'
    values['exclusions']['outcome_payload']={'ignored':123}
    r=reader();v=m.resolve(r)
    assert '/FORBIDDEN/reserved.json' not in r.cache and v['reserved_outcomes_opened'] is False


def test_writer_real_exclusive_roundtrip_and_verify(tmp_path,monkeypatch,fixture):
    _,reader=fixture;v=m.resolve(reader());root=tmp_path/'inner';(root/'results').mkdir(parents=True)
    monkeypatch.setattr(m,'ROOT',root);p=root/'results'/(m.NAME+'_fixture.json')
    result=m.write_new(p,v);a=m.reuse();monkeypatch.setattr(m,'resolve',lambda:v)
    assert m.verify(p,result['sha256'])['status']=='blocked_preconditions_reproduced'
    with pytest.raises(ValueError):m.write_new(p,v)
    with pytest.raises(ValueError):m.verify(p,'0'*64)
    with pytest.raises(ValueError):m.write_new(root/'wrong.json',v)
    assert a.parse(p.read_bytes())==v


def test_reuse_rejects_mutated_source_before_import(monkeypatch):
    pins=deepcopy(m.PINS);pins['authority_source']=(m.AUTHORITY,'0'*64);monkeypatch.setattr(m,'PINS',pins)
    with pytest.raises(ValueError):m.reuse()


def test_cli_and_no_discovery_or_neural_import():
    tree=ast.parse(Path(m.__file__).read_text());modules=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):modules.extend(x.name for x in node.names)
        elif isinstance(node,ast.ImportFrom):modules.append(node.module or '')
        if isinstance(node,ast.Attribute):assert node.attr not in ('glob','rglob','walk','iterdir')
    assert not any(any(k in x for k in ('torch','pymunk','producer','numpy')) for x in modules)
    p=subprocess.run([sys.executable,'-m',m.__name__,'--help'],capture_output=True,text=True)
    assert p.returncode==0 and 'resolve' in p.stdout and 'verify' in p.stdout
